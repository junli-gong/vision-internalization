"""a 出平面三维旋转、b 视角变化的"目标相对深度（起伏）网格"、起伏误差与凸起先验基线（DEV-0018，D28-13；DEV-0028，D59-4）。
无学习组件。

定义（唯一定义在这里；编码与解析在 src/grid/codec.py：RELIEF_NUM_LEVELS、RELIEF_SYMBOLS、parse_relief_grid_text）
- 格子深度：mask 的 N×N 网格（mask_to_grid）为 1 的格子里，mask 像素深度的平均（DEPTH_QUANTUM_M 的整数单位精确求和，
  四舍五入取偶到整数单位；cell_depth_units，与 render3d.depth_grid 同一套数）。
- 物体平均深度 μ：前景格子深度的平均（每格等权，精确有理数，不舍入）。
- 起伏 = 格子深度 − μ。在物体外接半径 r 的范围 [−r, r) 里等分成 L = RELIEF_NUM_LEVELS = 32 档，档宽 2r / L，越界的截到
  首档或末档：档 k = clip(floor((d − μ) / (2r / L)) + L/2, 0, L − 1)。k = 0 最近、k = L − 1 最远；d = μ 落在档 L/2 的下沿。
  全程整数：k = floor((L/2)·(K·d − S) / (K·R)) + L/2，其中 K = 前景格数、S = 前景格子深度之和、R = r 的深度单位数（r 是
  SCENE_QUANTUM_M 的整数倍，R 精确）。
- 起伏网格（代码）：背景 0，档 k 记 k + 1（uint8，33 个取值），即 src/grid/codec.py 的代码约定。
- 起伏误差（relief_error）：在目标前景与预测前景都为 1 的格子上，预测档与目标档之差的均方根（RMSE），除以目标起伏的标准差
  （目标全部前景格的档的总体标准差）。1.0 大约相当于"预测一个平面"：在任何覆盖目标的 mask 上预测常数档 = 目标的平均档，
  误差恰为 1（只差在常数档取整）。mask 的对错由 mask IoU 另计，这里不罚漏掉或多出的格子。
  没有定义时返回 None（空集合报 null）：目标没有前景、目标起伏的标准差为 0（全部前景格同一档）、或两者没有公共前景格。

- 凸起先验（convex_prior_codes，D59-4；项目负责人定义的"不需要想象"的基线：离 mask 边界越远的格子越近，一个幅度参数）：
  对一张 0/1 网格 g（N×N），前景格 c 的
  * 边界距离 d(c) = c 的格心到最近的背景格格心的欧氏距离（格为单位；网格之外的格子都算背景，所以贴着网格边的前景格
    d = 1）。d ≥ 1，贴着背景的前景格 d = 1。d² 是精确整数（boundary_distance_sq）。
  * 归一化 f(c) = (d(c) − 1) / (d_max − 1) ∈ [0, 1]（d_max = 前景格里 d 的最大值；d_max = 1 时 f ≡ 0），单调递增：
    贴边的格 f = 0，最里面的格 f = 1。d 取定点：F = ⌊√(d²) · 2^16⌋ − 2^16（整数开方，精确），f = F / F_max。
  * 档 = FLAT_LEVEL − round(A · f(c))（任务书与项目负责人的字面定义"平均档减去 A × f(d)"；平均档 = 起伏 0 所在的档
    FLAT_LEVEL，即"平面" copy_flat 的档）；round 为四舍五入取偶，结果截到 [0, L − 1]。贴边的格子取平面档，越往里越近，
    最里面的格子比平面近 A 档。A = 0 时与 copy_flat 逐位相同。不减去 f 的平均：照字面（DEV-0028 在 val 上也看过减去平均的
    版本，误差更大，见完成总结）。全程整数：round(A · F / F_max)。
  * 唯一的参数 A（档数，整数）按任务在 val 划分上网格搜索选定后冻结在 CONVEX_PRIOR_AMPLITUDE（选法见 src/e5/convex_select.py）。
  * 基线 copy_convex：mask 与网格取原样复制（copy），起伏 = 对 copy 的网格做凸起先验。它是 copy 网格与冻结常数 A 的纯函数，
    由 derived_baselines 在评分时现算，不存进样本（于是不进样本摘要：a 的摘要与任务版本都不变）。

确定性（D24：a、b 在任何平台上逐位相同）：深度、平均、档全是整数运算；误差 = sqrt(整数之比)，在 decimal（dmath.CTX）里开方，
再量化到 RELIEF_ERROR_QUANTUM（2 的幂）。不用 np.mean、平台超越函数或浮点求和。凸起先验只用整数（开方用 np.sqrt 取初值，
再用整数平方核对、修正到精确的整数平方根）。
"""

from __future__ import annotations

import decimal
from dataclasses import dataclass
from decimal import Decimal

import numpy as np

from src.grid.codec import RELIEF_NUM_LEVELS, as_relief_codes, cell_edges, mask_to_grid
from src.synth.constants import DEPTH_QUANTUM_M
from src.synth.dmath import CTX, quantize_dec

HALF_LEVELS = RELIEF_NUM_LEVELS // 2
if 2 * HALF_LEVELS != RELIEF_NUM_LEVELS:  # pragma: no cover - 常量写错
    raise ImportError("RELIEF_NUM_LEVELS 必须是偶数（起伏范围 [−r, r) 关于 0 对称地分档）")
# 写进元数据的起伏误差、标准差量化到这个步长（2 的幂，于是是精确的 float）
RELIEF_ERROR_QUANTUM = 2.0**-30


def radius_depth_units(radius_m: float) -> int:
    """外接半径（米）→ 深度单位数（整数，精确）。radius 必须是 DEPTH_QUANTUM_M 的整数倍（场景规格里的半径是 SCENE_QUANTUM_M 的倍数）。"""
    r = float(radius_m) / DEPTH_QUANTUM_M  # 除以 2 的幂：没有舍入
    if not (r > 0 and r == int(r)):
        raise ValueError(f"外接半径 {radius_m!r} 不是 DEPTH_QUANTUM_M 的正整数倍")
    return int(r)


def cell_depth_units(depth_units: np.ndarray, mask: np.ndarray, n: int,
                     grid: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(网格 (n, n) uint8 = mask_to_grid(mask, n)，格子深度 (n, n) int64)：网格为 1 的格子里 mask 像素深度的平均
    （整数单位精确求和，四舍五入取偶到整数单位）；其余格子为 0。

    grid：调用方已算好的 mask_to_grid(mask, n)（省掉重算；DEV-0028 提速）。求和只在覆盖 mask 包围盒的那几行、几列格子上做：
    其余格子没有前景像素，和与计数都是 0，结果逐位相同。
    """
    m = np.asarray(mask, dtype=bool)
    g = mask_to_grid(m, n) if grid is None else np.asarray(grid, dtype=np.uint8)
    h, w = m.shape
    if g.shape != (n, n):
        raise ValueError(f"grid 的形状 {g.shape} 不是 ({n}, {n})")
    out = np.zeros((n, n), dtype=np.int64)
    rows = np.nonzero(m.any(axis=1))[0]
    if rows.size == 0:
        return g, out
    cols = np.nonzero(m.any(axis=0))[0]
    re_, ce_ = cell_edges(h, n), cell_edges(w, n)
    # 包围盒所在的格子范围 [k0, k1]：edges[k] ≤ 首行 < edges[k + 1]
    k0, k1 = int(np.searchsorted(re_, rows[0], "right")) - 1, int(np.searchsorted(re_, rows[-1], "right")) - 1
    l0, l1 = int(np.searchsorted(ce_, cols[0], "right")) - 1, int(np.searchsorted(ce_, cols[-1], "right")) - 1
    ri, ci = re_[k0 : k1 + 1] - re_[k0], ce_[l0 : l1 + 1] - ce_[l0]
    win = (slice(int(re_[k0]), int(re_[k1 + 1])), slice(int(ce_[l0]), int(ce_[l1 + 1])))
    mw = m[win]
    du = np.where(mw, np.asarray(depth_units)[win].astype(np.int64), 0)
    s = np.add.reduceat(np.add.reduceat(du, ri, axis=0), ci, axis=1)
    c = np.add.reduceat(np.add.reduceat(mw.astype(np.int64), ri, axis=0), ci, axis=1)
    cs = np.maximum(c, 1)
    q, r = np.divmod(s, cs)
    up = (2 * r > cs) | ((2 * r == cs) & (q % 2 == 1))
    gw = g[k0 : k1 + 1, l0 : l1 + 1]
    out[k0 : k1 + 1, l0 : l1 + 1] = np.where(gw == 1, q + up.astype(np.int64), 0)
    return g, out


@dataclass(frozen=True)
class ReliefGrid:
    codes: np.ndarray  # (n, n) uint8：0 背景，k + 1 = 档 k
    n_cells: int  # 前景格数 K
    mean_units_num: int  # 平均深度 μ = mean_units_num / n_cells（深度单位，精确）
    radius_units: int  # R
    n_clipped: int  # 起伏超出 [−r, r)、被截到首末档的格数

    def mean_depth_m(self) -> float | None:
        """μ（米），量化到 DEPTH_QUANTUM_M / 2^20；没有前景时 None。"""
        if self.n_cells == 0:
            return None
        with decimal.localcontext(CTX):
            v = Decimal(self.mean_units_num) / Decimal(self.n_cells) * Decimal(repr(DEPTH_QUANTUM_M))
        return quantize_dec(v, DEPTH_QUANTUM_M * 2.0**-20)


def relief_grid(depth_units: np.ndarray, mask: np.ndarray, n: int, radius_m: float,
                grid: np.ndarray | None = None) -> ReliefGrid:
    """mask 与深度图（DEPTH_QUANTUM_M 的整数单位）→ 起伏网格。定义见模块说明。grid：可选的 mask_to_grid(mask, n)。"""
    g, u = cell_depth_units(depth_units, mask, n, grid)
    fg = g == 1
    k = int(np.count_nonzero(fg))
    big_r = radius_depth_units(radius_m)
    if k == 0:
        return ReliefGrid(np.zeros((n, n), dtype=np.uint8), 0, 0, big_r, 0)
    s = int(u[fg].sum())  # int64 精确：K ≤ n²、单格 < 2^16
    num = HALF_LEVELS * (k * u - s)
    lev = num // (k * big_r) + HALF_LEVELS  # floor 除法，精确
    clipped = int(np.count_nonzero(fg & ((lev < 0) | (lev > RELIEF_NUM_LEVELS - 1))))
    lev = np.clip(lev, 0, RELIEF_NUM_LEVELS - 1)
    codes = np.where(fg, lev + 1, 0).astype(np.uint8)
    return ReliefGrid(codes, k, s, big_r, clipped)


def relief_levels(codes: np.ndarray) -> np.ndarray:
    """代码 → 档（int64）；背景记 −1。"""
    c = as_relief_codes(codes).astype(np.int64)
    return c - 1


@dataclass(frozen=True)
class ReliefScore:
    error: float | None  # RMSE / 目标标准差（量化到 RELIEF_ERROR_QUANTUM）；没有定义时 None
    n_eval_cells: int  # 公共前景格数
    n_target_cells: int
    target_std_levels: float | None  # 目标档的总体标准差（档为单位，量化）；没有前景时 None
    reason: str | None  # error 为 None 的原因："no_target" / "flat_target" / "no_overlap"


def _qsqrt(num: int, den: int) -> float:
    """sqrt(num / den)（num ≥ 0、den > 0 的整数），decimal 开方后量化到 RELIEF_ERROR_QUANTUM。"""
    with decimal.localcontext(CTX):
        v = (Decimal(num) / Decimal(den)).sqrt()
    return quantize_dec(v, RELIEF_ERROR_QUANTUM)


def relief_score(pred_codes: np.ndarray, target_codes: np.ndarray) -> ReliefScore:
    """起伏误差及其组成（定义见模块说明）。两个代码数组形状必须相同。"""
    p, t = as_relief_codes(pred_codes, "pred"), as_relief_codes(target_codes, "target")
    if p.shape != t.shape:
        raise ValueError(f"预测 {p.shape} 与目标 {t.shape} 的形状不同")
    tf = t > 0
    k = int(np.count_nonzero(tf))
    if k == 0:
        return ReliefScore(None, 0, 0, None, "no_target")
    tl = t[tf].astype(np.int64) - 1
    st, stt = int(tl.sum()), int((tl * tl).sum())
    var_num = k * stt - st * st  # = K² · 方差，精确整数
    std = _qsqrt(var_num, k * k)
    both = tf & (p > 0)
    m = int(np.count_nonzero(both))
    if var_num == 0:
        return ReliefScore(None, m, k, std, "flat_target")
    if m == 0:
        return ReliefScore(None, 0, k, std, "no_overlap")
    d = p[both].astype(np.int64) - t[both].astype(np.int64)
    sdd = int((d * d).sum())
    # (RMSE / std)² = (Σd² / m) / (var_num / K²) = Σd² · K² / (m · var_num)
    return ReliefScore(_qsqrt(sdd * k * k, m * var_num), m, k, std, None)


def relief_error(pred_codes: np.ndarray, target_codes: np.ndarray) -> float | None:
    """起伏误差（RMSE / 目标起伏的标准差），没有定义时 None。"""
    return relief_score(pred_codes, target_codes).error


# ---------------------------------------------------------------------------
# a、b 的样本里的名字（rotation3d、viewpoint 写，E5 评分读；只在这里定义）
# ---------------------------------------------------------------------------

# 样本带起伏（extras[RELIEF_TARGET_KEY] 与各基线的起伏代码）的任务：a 自任务版本 4（DEV-0018），b 自任务版本 5（DEV-0028）
RELIEF_TASKS = ("rotation3d", "viewpoint")

RELIEF_TARGET_KEY = "relief_grid"  # Sample.extras 里目标的起伏代码 (N, N) uint8
RELIEF_BASELINE_PREFIX = "baseline_relief_"  # Sample.extras 里各基线的起伏代码：RELIEF_BASELINE_PREFIX + 基线名
# "copy_flat"：mask 取原样复制，起伏取常数档 FLAT_LEVEL（起伏 0 所在的档）——"预测一个平面"的非真值基线（DEV-0018 加，
# 用来检验 D28-13 的判定条件：见 E5 的门槛说明）
FLAT_LEVEL = HALF_LEVELS


def baseline_relief_key(name: str) -> str:
    return RELIEF_BASELINE_PREFIX + name


def flat_codes(grid: np.ndarray) -> np.ndarray:
    """0/1 网格 → 起伏代码：前景格一律取档 FLAT_LEVEL（代码 FLAT_LEVEL + 1），背景 0。"""
    g = np.asarray(grid)
    return np.where(g > 0, FLAT_LEVEL + 1, 0).astype(np.uint8)


def relief_meta(definition: str, radius_m: float, target: ReliefGrid, input_: ReliefGrid,
                base_codes: dict[str, np.ndarray]) -> dict:
    """样本元数据 meta["relief"]：定义出处、档数、半径、档宽、平面档、目标与输入视角的格数 / 平均深度 / 标准差 / 截断格数，
    以及各基线的起伏误差（relief_score）。a、b 共用（a 的取值与 DEV-0018 逐位相同）。"""
    t_score = relief_score(target.codes, target.codes)
    meta = {
        "definition": definition,
        "num_levels": RELIEF_NUM_LEVELS,
        "radius_m": radius_m,
        "level_width_m": 2.0 * radius_m / RELIEF_NUM_LEVELS,
        "flat_level": FLAT_LEVEL,
        "target": {"n_cells": target.n_cells, "mean_depth_m": target.mean_depth_m(),
                   "std_levels": t_score.target_std_levels, "n_clipped": target.n_clipped},
        "input": {"n_cells": input_.n_cells, "mean_depth_m": input_.mean_depth_m(), "n_clipped": input_.n_clipped},
        "baseline_error": {},
    }
    for name in sorted(base_codes):
        sc = relief_score(base_codes[name], target.codes)
        meta["baseline_error"][name] = {"error": sc.error, "n_eval_cells": sc.n_eval_cells, "reason": sc.reason}
    return meta


# ---------------------------------------------------------------------------
# 凸起先验（D59-4；定义见模块说明）
# ---------------------------------------------------------------------------

CONVEX_BASELINE = "copy_convex"
CONVEX_SOURCE_BASELINE = "copy"  # 凸起先验的 mask 取这个基线（原样复制）
CONVEX_FIXED_POINT_BITS = 16  # d 的定点精度：F = ⌊d · 2^16⌋ − 2^16
# 幅度 A（档数，整数），按任务在 val 划分上选定后冻结（DEV-0028；选法见 src/e5/convex_select.py）。只在这里定义；
# E5 的运行配置（config.yaml）与 README 都记下它。改它不改变任何样本（不进摘要），但会改变 copy_convex 基线与门槛：
# 改动要先有新的 D 条目。
# 选定记录（val 划分、序号规则同 E5 评测集、每档 200 条，标题口径上 copy_convex 起伏误差的严格均值；
# `python -m src.e5.convex_select select`；tests/test_synth_convex_relief.py 在同一个 val 集合上复算 a 的 A = 1、2、3）：
#   a（30° 以上三档，600 条）：A = 0 1.0025 | 1 0.9548 | 2 0.9390 | 3 0.9642 | 4 1.0118 | 5 1.0865 | 6 1.1822 → 选 2
#   b（环绕移动，419 条）：    A = 0 1.1458 | 1 1.1371 | 2 1.1388 | 3 1.1558 | 4 1.1818 | 5 1.2214 | 6 1.2763 → 选 1
CONVEX_PRIOR_AMPLITUDE: dict[str, int] = {"rotation3d": 2, "viewpoint": 1}
CONVEX_AMPLITUDE_SEARCH = tuple(range(0, 13))  # 网格搜索的取值（档）：0–12
# 派生基线（不存进样本、评分时现算）的角色；与样本 meta["baseline_roles"] 同一口径
DERIVED_BASELINE_ROLES = {CONVEX_BASELINE: "trivial"}


def boundary_distance_sq(grid: np.ndarray) -> np.ndarray:
    """(N, N) int64：前景格的格心到最近的背景格格心的欧氏距离的平方（网格之外的格子都算背景）；背景格为 0。精确整数。"""
    g = np.asarray(grid) > 0
    if g.ndim != 2:
        raise ValueError(f"网格必须是二维，收到形状 {g.shape}")
    n0, n1 = g.shape
    out = np.zeros((n0, n1), dtype=np.int64)
    fi, fj = np.nonzero(g)
    if fi.size == 0:
        return out
    fi, fj = fi.astype(np.int64), fj.astype(np.int64)
    # 网格之外最近的格子：沿行或列直走出去（斜着出去更远）
    d2 = np.minimum(np.minimum(fi + 1, n0 - fi), np.minimum(fj + 1, n1 - fj)) ** 2
    bi, bj = np.nonzero(~g)
    if bi.size:
        bi, bj = bi.astype(np.int64), bj.astype(np.int64)
        inner = ((fi[:, None] - bi[None, :]) ** 2 + (fj[:, None] - bj[None, :]) ** 2).min(axis=1)
        d2 = np.minimum(d2, inner)
    out[fi, fj] = d2
    return out


def _isqrt(x: np.ndarray) -> np.ndarray:
    """非负 int64 数组的整数平方根 ⌊√x⌋（精确）：np.sqrt 取初值（x < 2^52 时误差 < 1），再用整数平方核对、修正。"""
    x = np.asarray(x, dtype=np.int64)
    if x.size and (int(x.min()) < 0 or int(x.max()) >= 1 << 52):
        raise ValueError("_isqrt 只接受 [0, 2^52) 的整数")
    r = np.floor(np.sqrt(x.astype(np.float64))).astype(np.int64)
    r = np.where(r * r > x, r - 1, r)
    return np.where((r + 1) * (r + 1) <= x, r + 1, r)


def _div_round_half_even(num: np.ndarray, den: int) -> np.ndarray:
    """整数 num / den（den > 0）四舍五入取偶到整数（与 cell_depth_units 同一规则）。"""
    q, r = np.divmod(np.asarray(num, dtype=np.int64), int(den))
    up = (2 * r > den) | ((2 * r == den) & (q % 2 == 1))
    return q + up.astype(np.int64)


def convex_prior_codes(grid: np.ndarray, amplitude: int) -> np.ndarray:
    """0/1 网格 → 凸起先验的起伏代码（uint8，0 背景，k + 1 = 档 k）。定义见模块说明；amplitude = A（非负整数档数）。"""
    if isinstance(amplitude, bool) or int(amplitude) != amplitude or amplitude < 0:
        raise ValueError(f"凸起先验的幅度必须是非负整数，收到 {amplitude!r}")
    a = int(amplitude)
    g = np.asarray(grid) > 0
    codes = np.zeros(g.shape, dtype=np.uint8)
    k = int(np.count_nonzero(g))
    if k == 0:
        return codes
    d2 = boundary_distance_sq(g)[g]
    one = 1 << CONVEX_FIXED_POINT_BITS
    f = _isqrt(d2 << (2 * CONVEX_FIXED_POINT_BITS)) - one  # ⌊d · 2^16⌋ − 2^16 ≥ 0
    f_max = int(f.max())
    if f_max == 0 or a == 0:
        lev = np.full(k, FLAT_LEVEL, dtype=np.int64)
    else:
        lev = FLAT_LEVEL - _div_round_half_even(a * f, f_max)
    codes[g] = (np.clip(lev, 0, RELIEF_NUM_LEVELS - 1) + 1).astype(np.uint8)
    return codes


def convex_amplitude(task: str) -> int:
    """该任务冻结的凸起先验幅度 A。"""
    from src.synth.seeding import canonical_task

    t = canonical_task(task)
    if t not in CONVEX_PRIOR_AMPLITUDE:
        raise KeyError(f"{t} 没有起伏，没有凸起先验")
    return CONVEX_PRIOR_AMPLITUDE[t]


def derived_baseline_names(task: str) -> tuple[str, ...]:
    """该任务的派生基线名（不算起伏代码）。"""
    from src.synth.seeding import canonical_task

    return (CONVEX_BASELINE,) if canonical_task(task) in RELIEF_TASKS else ()


def derived_baselines(sample, amplitude: int | None = None) -> dict:
    """样本的派生基线（不存进样本、由已存的基线与冻结常数现算）：名字 → (Prediction, 起伏代码)。

    带起伏的任务（RELIEF_TASKS）有 copy_convex：mask 与网格取 copy，起伏 = convex_prior_codes(copy 的网格, A)；
    A 默认取该任务冻结的 CONVEX_PRIOR_AMPLITUDE（amplitude 只给选 A 的网格搜索用）。其他任务没有派生基线。
    """
    from src.synth.seeding import canonical_task

    t = canonical_task(sample.task)
    if t not in RELIEF_TASKS:
        return {}
    src = sample.baselines[CONVEX_SOURCE_BASELINE]
    a = convex_amplitude(t) if amplitude is None else amplitude
    return {CONVEX_BASELINE: (src, convex_prior_codes(src.grid, a))}
