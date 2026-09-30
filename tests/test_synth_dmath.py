"""确定的三角函数与求平均（src/synth/dmath.py，DEV-0009）。

- π 常量用 Machin 公式独立复算；sin/cos/atan2/asin 与 libm 的差在 1–2 ULP 内；
- 精确的参数约简：90° 的整数倍给出精确值，奇偶、互余、周期关系逐位成立；
- 结果与全局 decimal 上下文无关；rot_cs 的量化；
- fixed_point_mean 精确、与求和顺序无关；
- 钉住一张 rot_cs 表的摘要：在服务器上跑一遍即是跨平台核对。
"""

import decimal
import hashlib
import math
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest

from src.synth import dmath
from src.synth.constants import MEAN_TERM_QUANTUM, TRIG_QUANTUM


def _machin_pi(prec: int) -> Decimal:
    """π = 16·atan(1/5) - 4·atan(1/239)，在 prec 位精度下用级数独立计算。"""
    with decimal.localcontext(decimal.Context(prec=prec)):

        def atan_inv(n: int) -> Decimal:
            x = Decimal(1) / n
            x2 = x * x
            s = term = x
            k = 1
            while True:
                term = -term * x2
                t = term / (2 * k + 1)
                if abs(t) < Decimal(10) ** (-prec - 2):
                    return s
                s += t
                k += 1

        return 16 * atan_inv(5) - 4 * atan_inv(239)


def test_pi_constant_is_correct():
    with decimal.localcontext(decimal.Context(prec=70)):
        assert abs(dmath.PI - _machin_pi(66)) < Decimal("1e-50")


def test_cos_sin_agree_with_libm_within_two_ulp():
    worst = 0.0
    for k in range(0, 360 * 256, 13):
        a = k / 256 - 180.0
        c, s = dmath.cos_sin_deg(a)
        t = math.radians(a)
        worst = max(worst, abs(float(c) - math.cos(t)), abs(float(s) - math.sin(t)))
    assert worst <= 4.5e-16


def test_exact_argument_reduction():
    for k in range(-8, 9):
        c, s = dmath.cos_sin_deg(90 * k)
        assert (c, s) == [(1, 0), (0, 1), (-1, 0), (0, -1)][k % 4]
        assert dmath.rot_cs(90 * k) == [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)][k % 4]
    rng = np.random.default_rng(0)
    for a in list(rng.uniform(-720, 720, 200)) + [Fraction(360, 7), Fraction(-45, 2), 17.77, 1 / 256]:
        c, s = dmath.cos_sin_deg(a)
        f = Fraction(a)
        assert dmath.cos_sin_deg(f + 360) == (c, s) and dmath.cos_sin_deg(f - 720) == (c, s)  # 周期
        # 比较时用 copy_negate：一元负号会按全局上下文（28 位）舍入
        assert dmath.cos_sin_deg(-f) == (c, s.copy_negate())  # 奇偶
        assert dmath.cos_sin_deg(90 - f) == (s, c)  # 互余
        assert dmath.cos_sin_deg(180 - f) == (c.copy_negate(), s)
        with decimal.localcontext(dmath.CTX):
            assert abs(c * c + s * s - 1) < Decimal("1e-38")


def test_independent_of_the_global_decimal_context():
    ref = [dmath.cos_sin_deg(a) for a in (0.1, 33.0, 1234.5678)] + [dmath.atan2_deg(3, -4), dmath.asin_deg(0.3)]
    with decimal.localcontext(decimal.Context(prec=5, rounding=decimal.ROUND_DOWN)):
        again = [dmath.cos_sin_deg(a) for a in (0.1, 33.0, 1234.5678)] + [dmath.atan2_deg(3, -4), dmath.asin_deg(0.3)]
    assert again == ref


def test_atan2_and_asin():
    worst = 0.0
    for y in np.linspace(-5, 5, 41):
        for x in np.linspace(-5, 5, 43):
            if x == 0 and y == 0:
                continue
            worst = max(worst, abs(float(dmath.atan2_deg(y, x)) - math.degrees(math.atan2(y, x))))
    assert worst < 1e-13
    assert dmath.atan2_deg(0, 1) == 0 and dmath.atan2_deg(1, 0) == 90 and dmath.atan2_deg(0, -1) == 180
    assert dmath.atan2_deg(-1, 0) == -90
    with decimal.localcontext(dmath.CTX):
        assert abs(dmath.atan2_deg(1, 1) - 45) < Decimal("1e-37")
        assert abs(dmath.asin_deg(0.5) - 30) < Decimal("1e-36")
        assert abs(dmath.asin_deg(1) - 90) < Decimal("1e-36")
    assert dmath.asin_deg(-0.3) == dmath.asin_deg(0.3).copy_negate()
    for x in np.linspace(-0.99, 0.99, 37):
        assert abs(float(dmath.asin_deg(x)) - math.degrees(math.asin(x))) < 1e-12
    with pytest.raises(ValueError):
        dmath.atan2_deg(0, 0)
    with pytest.raises(ValueError):
        dmath.asin_deg(1.5)
    with pytest.raises(ValueError):
        dmath.cos_sin_deg(float("nan"))


def test_rot_cs_is_quantized_and_close():
    for a in np.linspace(-400, 400, 997):
        c, s = dmath.rot_cs(a)
        assert c / TRIG_QUANTUM == round(c / TRIG_QUANTUM) and s / TRIG_QUANTUM == round(s / TRIG_QUANTUM)
        t = math.radians(a)
        assert abs(c - math.cos(t)) <= TRIG_QUANTUM / 2 + 1e-16 and abs(s - math.sin(t)) <= TRIG_QUANTUM / 2 + 1e-16
    assert dmath.rot_cs(np.float64(30.0)) == dmath.rot_cs(30) == dmath.rot_cs(30.0)


def test_quantize_dec():
    assert dmath.quantize_dec(Decimal("0.3"), 0.25) == 0.25
    assert dmath.quantize_dec(Decimal("0.375"), 0.25) == 0.5  # 取偶：1.5 → 2
    assert dmath.quantize_dec(Decimal("0.125"), 0.25) == 0.0  # 取偶：0.5 → 0
    assert dmath.quantize_dec(Decimal("-0.125"), 0.25) == 0.0 and math.copysign(1, dmath.quantize_dec(Decimal("-0.1"), 0.25)) == 1
    with pytest.raises(ValueError):
        dmath.quantize_dec(Decimal(1), 0.3)


def test_fixed_point_mean_is_exact_and_order_independent():
    assert dmath.fixed_point_mean(np.array([1.0, 2.0]), MEAN_TERM_QUANTUM) == 1.5
    rng = np.random.default_rng(1)
    a = rng.uniform(0, 1500, 30001)
    m = dmath.fixed_point_mean(a, MEAN_TERM_QUANTUM)
    for _ in range(3):
        assert dmath.fixed_point_mean(rng.permutation(a), MEAN_TERM_QUANTUM) == m
    assert dmath.fixed_point_mean(a[::-1].copy(), MEAN_TERM_QUANTUM) == m
    # 等于"逐项取偶舍入后的精确有理数均值"的正确舍入
    k = [int(v) for v in np.rint(a / MEAN_TERM_QUANTUM)]
    assert m == float(Fraction(sum(k), len(k)) * Fraction(MEAN_TERM_QUANTUM))
    assert abs(m - math.fsum(a) / len(a)) <= MEAN_TERM_QUANTUM
    for bad in (np.array([]), np.array([1.0, np.nan]), np.array([2.0**40] * 10)):
        with pytest.raises(ValueError):
            dmath.fixed_point_mean(bad, MEAN_TERM_QUANTUM)
    with pytest.raises(ValueError):
        dmath.fixed_point_mean(a, 0.1)


def test_pinned_trig_table_digest():
    """rot_cs 在 c、d 用的 1/256° 网格上抽样、e 用的整数度，以及几个 atan2/asin 值的摘要。

    在开发机上红 = 三角函数实现变了（生成器版本要加一）；只在服务器上红 = 跨平台不能逐位复现，上报管理员。
    """
    h = hashlib.sha256()
    for k in range(0, 360 * 256, 97):
        h.update(repr(dmath.rot_cs(k / 256)).encode())
    for k in range(-360, 361):
        h.update(repr(dmath.rot_cs(k)).encode())
    for y, x in ((1, 3), (-2, 5), (0.3, -0.7), (Decimal("0.123456789"), 1)):
        h.update(str(dmath.atan2_deg(y, x)).encode())
    h.update(str(dmath.asin_deg(0.4)).encode())
    assert h.hexdigest() == PINNED_TRIG_DIGEST


PINNED_TRIG_DIGEST = "afc432935f1d9f81c19936fd48a2e43c15e6540c959e97ba653a7cea18eed78f"  # 开发机 arm64 macOS，DEV-0009
