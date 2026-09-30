"""确定的数值工具（DEV-0009，D24）：三角函数与求平均，结果与平台 libm、numpy 的 SIMD 实现、编译器都无关。

为什么
- IEEE 754 只保证 +、-、×、÷、sqrt 正确舍入。math.cos/sin/atan2 走平台 libm，np.cos/np.sin 在 NEON 与 AVX 上
  是不同的实现，末位可能不同；再经过量化或取整，就会放大成两台机器上不同的样本。
- numpy 的浮点求和（np.sum、np.mean）的求和顺序是实现细节，将来的版本或 SIMD 分派可能改变它。

三角函数
- 全部用 Python 的 decimal 计算（libmpdec；按 General Decimal Arithmetic 规范逐次正确舍入，结果与平台无关），
  用本模块的 Context CTX（DEC_PREC 位有效数字、ROUND_HALF_EVEN），不读也不改全局上下文。
- 角度一律以"度"为单位进入，先用 as_integer_ratio() 变成精确的有理数 n/d（float、int、Decimal、Fraction 都可以），
  在整数上做精确的参数约简：模 360，按 45° 分成八个区间，映射到 [0°, 45°]。所以 90° 的整数倍给出精确的 0 与 ±1，
  奇偶性、互余关系也是精确的。约简后的角度换成弧度（π 取 50 位），在 [0, π/4] 上用固定 SERIES_TERMS 项的泰勒级数。
- atan：用 atan(t) = 2·atan(t / (1 + sqrt(1 + t²))) 约简三次到 t ≤ tan(π/32) 再求级数；atan2 按象限拼接。
  asin(x) = atan2(x, sqrt(1 - x²))。
- 精度：级数截断误差 < 1e-45；DEC_PREC = 40 位下，结果与真值之差在 1e-38 量级（相对 1）。
  调用方再把结果量化到自己的步长（如 TRIG_QUANTUM = 2^-40 ≈ 9.1e-13），或用 float(Decimal) 转成最近的 float
  （CPython 自带的正确舍入转换，与平台无关）。

求平均
- fixed_point_mean：逐项四舍五入到 quantum 的整数倍，精确求和（全是整数、部分和 < 2^53，与求和顺序无关、没有舍入），
  再做一次正确舍入的除法。用于会影响分档、拒绝或写进元数据的平均值（e 的平均位移）。
- math.fsum 也是确定的，但它逐个处理 Python float：2.6 万项约 460 µs，这里约 20 µs（开发机实测）。
  e 的二分每次尝试要算几十次这样的平均，用 fsum 会让 e 慢一个数量级。
"""

from __future__ import annotations

import decimal
import functools
from decimal import Decimal
from fractions import Fraction

import numpy as np

from src.synth.constants import DEC_PREC, TRIG_QUANTUM

# 本模块与 shapes.py（单位形状）共用的上下文：只通过 decimal.localcontext(CTX) 使用，不修改全局上下文。
CTX = decimal.Context(
    prec=DEC_PREC,
    rounding=decimal.ROUND_HALF_EVEN,
    Emax=decimal.MAX_EMAX,
    Emin=decimal.MIN_EMIN,
    traps=[decimal.InvalidOperation, decimal.DivisionByZero, decimal.Overflow],
)

# π，51 位有效数字（tests/test_synth_dmath.py 用 Machin 公式在 60 位精度下独立复算核对）。
PI = Decimal("3.14159265358979323846264338327950288419716939937510")

# sin/cos 级数的项数：|x| ≤ π/4 时第 SERIES_TERMS 项 < x^40/40! < 1e-51。
SERIES_TERMS = 20
# atan 级数的项数：约简后 t ≤ 0.0985，t^(2k+1)/(2k+1) 在 k = 22 时 < 1e-45。
ATAN_TERMS = 23


def _ratio(x) -> tuple[int, int]:
    """int / float / numpy 标量 / Decimal / Fraction → 精确的 (分子, 分母)，分母 > 0。NaN、inf 抛 ValueError。"""
    if isinstance(x, (bool, np.bool_)):
        raise TypeError("角度不能是 bool")
    if isinstance(x, (int, np.integer)):
        return int(x), 1
    if isinstance(x, np.floating):
        x = float(x)
    try:
        n, d = x.as_integer_ratio()
    except (OverflowError, ValueError) as e:
        raise ValueError(f"角度必须是有限数，收到 {x!r}") from e
    return int(n), int(d)


def _sin_cos_reduced(x: Decimal) -> tuple[Decimal, Decimal]:
    """0 ≤ x ≤ π/4（弧度）的 (sin x, cos x)，固定项数的泰勒级数。调用方已进入 CTX。"""
    x2 = x * x
    s = term_s = x
    c = term_c = Decimal(1)
    for k in range(1, SERIES_TERMS):
        term_s = -term_s * x2 / ((2 * k) * (2 * k + 1))
        term_c = -term_c * x2 / ((2 * k - 1) * (2 * k))
        s += term_s
        c += term_c
    return s, c


def cos_sin_deg(angle) -> tuple[Decimal, Decimal]:
    """(cos θ, sin θ)，θ 以度为单位（精确的有理数）。参数约简在整数上精确完成，见模块说明。"""
    n, d = _ratio(angle)
    n %= 360 * d  # θ ∈ [0°, 360°)
    q, r = divmod(n, 45 * d)  # θ = 45q + r/d，0 ≤ q < 8，0 ≤ r < 45d
    odd = q % 2 == 1
    if odd:  # θ = 45(q+1) - r'/d，r' = 45d - r ∈ (0, 45d]
        r = 45 * d - r
    k = ((q + 1) // 2) % 4  # θ = 90k + φ，φ = +r/d（q 偶）或 -r'/d（q 奇）
    with decimal.localcontext(CTX):
        x = Decimal(r) * PI / Decimal(180 * d)
        s, c = _sin_cos_reduced(x)
        if odd:
            s = -s
        if k == 0:
            return +c, +s
        if k == 1:
            return -s, +c
        if k == 2:
            return -c, -s
        return +s, -c


def sin_deg(angle) -> Decimal:
    return cos_sin_deg(angle)[1]


def _atan_unit(t: Decimal) -> Decimal:
    """0 ≤ t ≤ 1 的 atan(t)（弧度）。调用方已进入 CTX。"""
    for _ in range(3):
        t = t / (1 + (1 + t * t).sqrt())
    t2 = t * t
    s = p = t
    for k in range(1, ATAN_TERMS):
        p = -p * t2
        s += p / (2 * k + 1)
    return s * 8


def atan2_deg(y, x) -> Decimal:
    """atan2(y, x)，单位度，范围 (-180, 180]。y、x 可以是 int / float / Decimal（按精确值使用）。"""
    with decimal.localcontext(CTX):
        yd = Decimal(y) if not isinstance(y, Decimal) else y
        xd = Decimal(x) if not isinstance(x, Decimal) else x
        if not (yd.is_finite() and xd.is_finite()):
            raise ValueError(f"atan2 的参数必须是有限数，收到 ({y!r}, {x!r})")
        if yd.is_zero() and xd.is_zero():
            raise ValueError("atan2(0, 0) 没有定义")
        ay, ax = yd.copy_abs(), xd.copy_abs()
        to_deg = Decimal(180) / PI
        if ay <= ax:
            base = _atan_unit(ay / ax) * to_deg
        else:
            base = 90 - _atan_unit(ax / ay) * to_deg
        if xd.is_signed():  # 第二、三象限（x < 0）
            base = 180 - base
        if yd.is_signed() and not yd.is_zero():
            return -base
        return +base


def asin_deg(x) -> Decimal:
    """asin(x)，单位度，|x| ≤ 1。"""
    with decimal.localcontext(CTX):
        xd = Decimal(x) if not isinstance(x, Decimal) else x
        one_minus = 1 - xd * xd
        if one_minus < 0:
            raise ValueError(f"asin 的参数必须在 [-1, 1] 内，收到 {x!r}")
        return atan2_deg(xd, one_minus.sqrt())


def quantize_dec(v: Decimal, step: float) -> float:
    """Decimal → 四舍五入（取偶）到 step 的整数倍的 float。step 必须是 2 的幂（于是结果是精确的 float）。"""
    num, den = Fraction(step).as_integer_ratio()
    if num != 1 or den & (den - 1):
        raise ValueError(f"量化步长必须是 2 的负整数次幂，收到 {step!r}")
    with decimal.localcontext(CTX):
        k = int((v * den).to_integral_value(rounding=decimal.ROUND_HALF_EVEN))
    return k / den  # int / int：正确舍入；|k| < 2^53 时精确。k 是 Python int，没有 -0.0


@functools.lru_cache(maxsize=4096)
def _rot_cs_cached(angle: float) -> tuple[float, float]:
    c, s = cos_sin_deg(angle)
    return quantize_dec(c, TRIG_QUANTUM), quantize_dec(s, TRIG_QUANTUM)


def rot_cs(angle_deg) -> tuple[float, float]:
    """(cos θ, sin θ)，θ 以度为单位，各自量化到 TRIG_QUANTUM；90° 的整数倍给出精确的 0 与 ±1。结果按角度缓存。"""
    return _rot_cs_cached(float(angle_deg))


def fixed_point_mean(values: np.ndarray, quantum: float) -> float:
    """一维非空浮点数组的确定均值：逐项四舍五入（取偶）到 quantum 的整数倍，精确求和，再做一次正确舍入的除法。

    - quantum 必须是 2 的负整数次幂：乘 1/quantum 没有舍入，np.rint 是精确的取偶舍入。
    - 舍入后每一项都是整数（以 float64 表示）。只要 项数 × max|项| < 2^53，任何部分和都是可精确表示的整数，
      每一次加法都没有舍入，所以不论 numpy 用什么顺序、什么 SIMD 宽度求和，结果都是同一个精确整数。
      超出这个范围直接抛 ValueError，不静默降级。
    - 均值 = 精确和 / (项数 × 1/quantum)：一次 IEEE 除法（分母 < 2^53，精确转成 float），正确舍入。
    """
    a = np.asarray(values, dtype=np.float64)
    if a.ndim != 1 or a.size == 0:
        raise ValueError(f"fixed_point_mean 需要一维非空数组，收到形状 {a.shape}")
    num, den = Fraction(quantum).as_integer_ratio()
    if num != 1 or den & (den - 1):
        raise ValueError(f"quantum 必须是 2 的负整数次幂，收到 {quantum!r}")
    scaled = np.rint(a * float(den))
    peak = max(float(scaled.max()), -float(scaled.min()))
    if not peak * a.size < 2.0**53 or a.size * den >= 2**53:  # NaN 也走这里
        raise ValueError("fixed_point_mean：数值超出精确整数求和的范围（或含非有限值）")
    return float(scaled.sum()) / (a.size * den)
