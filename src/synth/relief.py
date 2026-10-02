"""a 出平面三维旋转的"目标相对深度（起伏）网格"与起伏误差（DEV-0018，D28-13）。无学习组件。

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

确定性（D24：a 在任何平台上逐位相同）：深度、平均、档全是整数运算；误差 = sqrt(整数之比)，在 decimal（dmath.CTX）里开方，
再量化到 RELIEF_ERROR_QUANTUM（2 的幂）。不用 np.mean、平台超越函数或浮点求和。
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


def cell_depth_units(depth_units: np.ndarray, mask: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """(网格 (n, n) uint8 = mask_to_grid(mask, n)，格子深度 (n, n) int64)：网格为 1 的格子里 mask 像素深度的平均
    （整数单位精确求和，四舍五入取偶到整数单位）；其余格子为 0。"""
    m = np.asarray(mask, dtype=bool)
    g = mask_to_grid(m, n)
    h, w = m.shape
    re_, ce_ = cell_edges(h, n), cell_edges(w, n)
    du = np.where(m, np.asarray(depth_units, dtype=np.int64), 0)
    s = np.add.reduceat(np.add.reduceat(du, re_[:-1], axis=0), ce_[:-1], axis=1)
    c = np.add.reduceat(np.add.reduceat(m.astype(np.int64), re_[:-1], axis=0), ce_[:-1], axis=1)
    cs = np.maximum(c, 1)
    q, r = np.divmod(s, cs)
    up = (2 * r > cs) | ((2 * r == cs) & (q % 2 == 1))
    return g, np.where(g == 1, q + up.astype(np.int64), 0)


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


def relief_grid(depth_units: np.ndarray, mask: np.ndarray, n: int, radius_m: float) -> ReliefGrid:
    """mask 与深度图（DEPTH_QUANTUM_M 的整数单位）→ 起伏网格。定义见模块说明。"""
    g, u = cell_depth_units(depth_units, mask, n)
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
# a 的样本里的名字（rotation3d 写，E5 评分读；只在这里定义）
# ---------------------------------------------------------------------------

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
