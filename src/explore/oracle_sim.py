"""真值模拟的"理想 LLM"（DEV-0068；探索 X-1）：从真值 mask 按确定性规则生成各通道的输入，可选噪声档模拟不完美的 LLM。

只用于上限测量（scripts/explore_ceiling.py），不跑任何模型；规则与 src/explore/guidance.py 的工具分开：工具不看真值，这里只看真值。

规则：
- **坐标**：模拟的 LLM 输出都是 0–1000 网格的整数，x 在前（E1 的约定）。所以每个点、控制点、修正框都先换成网格整数
  （src/e3/c3.norm1000：floor(x / W · 1000 + 0.5)，夹到 [0, 1000]），再按 src/executors/geometry.norm1000_to_abs 换回原图像素
  交给工具。真值框也同样取整（理想的 LLM 也只能写整数）。
- **点**（oracle_points）：
  - 前景点：区域 = 真值 ∩ 框（为空时用整个真值）；距离变换（scipy distance_transform_edt）的极大值依次取，取到一点后把以它为圆心、
    半径 max(suppress_factor × 该点的距离值, min_suppress_px) 的圆盘里的距离值清零，再取下一个；距离值全为 0 就停。
    点 = 该像素的中心 (列 + 0.5, 行 + 0.5)。并列取行优先的第一个（np.argmax）。
  - 背景点：区域 = 框内、真值外，同法。
  - 噪声：每个点先以 point_wrong_rate 的概率变成"错点"（换成对方区域里均匀抽的一个像素中心；对方区域为空则不变），再在 x、y 上各加
    N(0, σ²)，σ = point_jitter_px + point_jitter_frac × 真值外接框的短边；夹到图内。
- **B 样条**（oracle_spline）：真值按 guidance.fit_mask_bspline 拟合（parts、holes 同工具）；噪声为每个控制点的 x、y 各加 N(0, σ²)，
  σ = ctrl_jitter_px + ctrl_jitter_frac × 真值外接框的短边。
- **局部修正**（oracle_corrections，每轮按当前 mask 重算）：
  - 漏掉的 = 真值 ∖ 当前 mask，记 "+"；多出的 = 当前 mask ∖ 真值，记 "-"。给了约束框时两者先与框求交（框外的改不了）。
  - 各自的 8 连通块合在一起按面积从大到小（并列：先 "+"、再按块里第一个像素的行优先序号）取前 m 个，框 = 外接框
    [列₀, 行₀, 列₁ + 1, 行₁ + 1]。
  - 噪声：每个框以 corr_sign_flip_rate 的概率翻转符号；x 坐标各加 N(0, (corr_box_jitter_frac × 框宽)²)，y 同理（框高）；
    排好序、夹到图内。
- **随机性**：每个格点一个独立的生成器 rng_for(salt, 样本 id, 框的种类, 格点名)（sha256 → 64 位种子），与进程、顺序无关。
- **token 估计**：每个通道的输出按下面的文本格式写出（坐标是 0–1000 的整数，与 E1 相同），用适配器的 tokenizer 数
  （count_tokens：tokenizer(text, add_special_tokens=False) 的长度；不含结束符）。只数这个通道额外的输出，框本身（提出一步）不计。
  - 点：{"foreground": [[x, y], …], "background": [[x, y], …]}（没有背景点时不写 background）；
  - B 样条：一条外环写成 [[x, y], …]（E1 的统一多边形格式，src/e3/c3.polygon_text 的写法）；多条写成 [[[x, y], …], …]；
    有洞时 {"outer": […], "holes": […]}；
  - 局部修正：每轮 [{"sign": "+", "box": [x0, y0, x1, y1]}, …]（没有框时 []），多轮的 token 相加；
  - 生成器库：挑出的编号（十进制整数）。
  json.dumps 的分隔符都是 (", ", ": ")，同 E1 的统一格式。
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, fields
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import ndimage

from src.e3.c3 import norm1000
from src.executors.geometry import norm1000_to_abs
from src.executors.raster import box_to_mask
from src.explore import guidance as G

TEXT_FORMAT_VERSION = "x1-text-v1"
JSON_SEPARATORS = (", ", ": ")


@dataclass(frozen=True)
class NoiseLevel:
    """一个噪声档（configs/explore_ceiling.json 的 noise_levels；缺的键取 0）。"""

    name: str
    point_jitter_px: float = 0.0
    point_jitter_frac: float = 0.0
    point_wrong_rate: float = 0.0
    ctrl_jitter_px: float = 0.0
    ctrl_jitter_frac: float = 0.0
    corr_box_jitter_frac: float = 0.0
    corr_sign_flip_rate: float = 0.0

    @classmethod
    def from_dict(cls, name: str, d: Mapping[str, Any]) -> "NoiseLevel":
        known = {f.name for f in fields(cls)} - {"name"}
        extra = set(d) - known
        if extra:
            raise ValueError(f"噪声档 {name}：未知的键 {sorted(extra)}；可选 {sorted(known)}")
        vals = {}
        for k, v in d.items():
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
                raise ValueError(f"噪声档 {name}：{k} 必须是非负的有限数，收到 {v!r}")
            if k.endswith("_rate") and v > 1:
                raise ValueError(f"噪声档 {name}：{k} 是概率，必须 ≤ 1，收到 {v}")
            vals[k] = float(v)
        return cls(name=name, **vals)

    @property
    def is_zero(self) -> bool:
        return all(getattr(self, f.name) == 0.0 for f in fields(self) if f.name != "name")


NO_NOISE = NoiseLevel("none")


def rng_for(*parts: Any) -> np.random.Generator:
    """确定的随机数生成器：种子 = sha256("|".join(parts)) 的前 16 位十六进制。"""
    seed = int(hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()[:16], 16)
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# 0–1000 网格
# ---------------------------------------------------------------------------


def quantize_xy(pts: Any, width: float, height: float) -> list[list[int]]:
    """(n, 2) 的原图像素 (x, y) → 0–1000 网格的整数 [[x, y], …]（src/e3/c3.norm1000）。"""
    a = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    return [[norm1000(x, width), norm1000(y, height)] for x, y in a]


def dequantize_xy(q: Any, width: float, height: float) -> np.ndarray:
    a = np.asarray(q, dtype=np.float64).reshape(-1, 2)
    return norm1000_to_abs(a, width=width, height=height) if len(a) else a


def quantize_box(box: Sequence[float], width: float, height: float) -> list[int]:
    x0, y0, x1, y1 = (float(v) for v in box)
    return [norm1000(x0, width), norm1000(y0, height), norm1000(x1, width), norm1000(y1, height)]


def dequantize_box(q: Sequence[int], width: float, height: float) -> list[float]:
    return [float(v) for v in norm1000_to_abs(np.asarray(q, dtype=np.float64).reshape(2, 2), width=width, height=height).ravel()]


def mask_bbox(mask: np.ndarray) -> tuple[float, float, float, float] | None:
    """mask 的外接框（连续坐标 [列₀, 行₀, 列₁ + 1, 行₁ + 1]）；空 mask 给 None。"""
    rows, cols = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
    if not len(rows):
        return None
    return float(cols[0]), float(rows[0]), float(cols[-1] + 1), float(rows[-1] + 1)


def _gt_scale(gt: np.ndarray) -> float:
    b = mask_bbox(gt)
    return 0.0 if b is None else min(b[2] - b[0], b[3] - b[1])


# ---------------------------------------------------------------------------
# 点
# ---------------------------------------------------------------------------


def dt_peaks(region: np.ndarray, k: int, *, suppress_factor: float = 1.0, min_suppress_px: float = 2.0) -> list[tuple[float, float]]:
    """region 的距离变换的极大值依次取 k 个（规则见模块说明）。返回像素中心 (x, y) 的列表（可能少于 k 个）。"""
    reg = np.asarray(region, dtype=bool)
    if not reg.any() or k <= 0:
        return []
    dt = ndimage.distance_transform_edt(reg)
    h, w = reg.shape
    cy = np.arange(h)[:, None] + 0.5
    cx = np.arange(w)[None, :] + 0.5
    out = []
    for _ in range(int(k)):
        idx = int(np.argmax(dt))
        r, c = divmod(idx, w)
        v = float(dt[r, c])
        if v <= 0:
            break
        x, y = c + 0.5, r + 0.5
        out.append((x, y))
        rad = max(suppress_factor * v, min_suppress_px)
        dt[(cx - x) ** 2 + (cy - y) ** 2 <= rad * rad] = 0.0
    return out


def _random_pixel(region: np.ndarray, rng: np.random.Generator) -> tuple[float, float] | None:
    idx = np.flatnonzero(region)
    if not len(idx):
        return None
    r, c = divmod(int(idx[int(rng.integers(len(idx)))]), region.shape[1])
    return c + 0.5, r + 0.5


def _perturb_points(pts: list[tuple[float, float]], other: np.ndarray, sigma: float, wrong: float, rng: np.random.Generator,
                    h: int, w: int) -> tuple[list[tuple[float, float]], int]:
    out, n_wrong = [], 0
    for x, y in pts:
        if wrong > 0 and rng.random() < wrong:
            p = _random_pixel(other, rng)
            if p is not None:
                x, y = p
                n_wrong += 1
        if sigma > 0:
            x, y = x + float(rng.normal(0.0, sigma)), y + float(rng.normal(0.0, sigma))
        out.append((min(max(x, 0.0), float(w)), min(max(y, 0.0), float(h))))
    return out, n_wrong


def oracle_points(gt: np.ndarray, box: Sequence[float], k: int, *, with_bg: bool, noise: NoiseLevel = NO_NOISE,
                  rng: np.random.Generator | None = None, suppress_factor: float = 1.0, min_suppress_px: float = 2.0) -> dict:
    """理想 LLM 的前景点、背景点（规则见模块说明）。返回 {"fg", "bg"（原图像素，已经过 0–1000 网格）, "fg_q", "bg_q", "n_wrong"}。"""
    g = np.asarray(gt, dtype=bool)
    h, w = g.shape
    inside = box_to_mask(box, h, w)
    fg_reg = g & inside
    if not fg_reg.any():
        fg_reg = g
    bg_reg = inside & ~g
    fg = dt_peaks(fg_reg, k, suppress_factor=suppress_factor, min_suppress_px=min_suppress_px)
    bg = dt_peaks(bg_reg, k, suppress_factor=suppress_factor, min_suppress_px=min_suppress_px) if with_bg else []
    n_wrong = 0
    if not noise.is_zero:
        if rng is None:
            raise ValueError("有噪声时要给 rng")
        sigma = noise.point_jitter_px + noise.point_jitter_frac * _gt_scale(g)
        fg, a = _perturb_points(fg, bg_reg, sigma, noise.point_wrong_rate, rng, h, w)
        bg, b = _perturb_points(bg, fg_reg, sigma, noise.point_wrong_rate, rng, h, w)
        n_wrong = a + b
    fq, bq = quantize_xy(fg, w, h), quantize_xy(bg, w, h)
    return {"fg": dequantize_xy(fq, w, h), "bg": dequantize_xy(bq, w, h), "fg_q": fq, "bg_q": bq, "n_wrong": n_wrong}


def points_text(fg_q: Sequence[Sequence[int]], bg_q: Sequence[Sequence[int]]) -> str:
    doc: dict[str, Any] = {"foreground": [list(map(int, p)) for p in fg_q]}
    if len(bg_q):
        doc["background"] = [list(map(int, p)) for p in bg_q]
    return json.dumps(doc, separators=JSON_SEPARATORS)


# ---------------------------------------------------------------------------
# B 样条
# ---------------------------------------------------------------------------


def oracle_spline(gt: np.ndarray, n: int, *, noise: NoiseLevel = NO_NOISE, rng: np.random.Generator | None = None, parts: int = 1,
                  holes: bool = False) -> dict:
    """真值轮廓的 n 点 B 样条（规则见模块说明）。返回 {"loops"（控制点为原图像素，已经过 0–1000 网格）, "loops_q"}。"""
    g = np.asarray(gt, dtype=bool)
    h, w = g.shape
    loops = G.fit_mask_bspline(g, n, parts=parts, holes=holes)
    sigma = 0.0
    if not noise.is_zero:
        if rng is None:
            raise ValueError("有噪声时要给 rng")
        sigma = noise.ctrl_jitter_px + noise.ctrl_jitter_frac * _gt_scale(g)
    out, out_q = [], []
    for lp in loops:
        c = np.asarray(lp["ctrl"], dtype=np.float64)
        if sigma > 0:
            c = c + rng.normal(0.0, sigma, size=c.shape)
        c = np.stack([np.clip(c[:, 0], 0.0, float(w)), np.clip(c[:, 1], 0.0, float(h))], axis=1)
        q = quantize_xy(c, w, h)
        out.append({"role": lp["role"], "ctrl": dequantize_xy(q, w, h)})
        out_q.append({"role": lp["role"], "ctrl": q})
    return {"loops": out, "loops_q": out_q}


def spline_text(loops_q: Sequence[Mapping[str, Any]]) -> str:
    outer = [[list(map(int, p)) for p in lp["ctrl"]] for lp in loops_q if lp["role"] == "outer"]
    holes = [[list(map(int, p)) for p in lp["ctrl"]] for lp in loops_q if lp["role"] == "hole"]
    body: Any = outer[0] if len(outer) == 1 else outer
    if holes:
        body = {"outer": body, "holes": holes}
    return json.dumps(body, separators=JSON_SEPARATORS)


# ---------------------------------------------------------------------------
# 局部修正
# ---------------------------------------------------------------------------


def diff_components(mask: np.ndarray, gt: np.ndarray, *, confine_box: Sequence[float] | None = None) -> list[dict]:
    """当前 mask 与真值之差的 8 连通块（规则见模块说明），已排好序：[{"sign", "area", "box"}]。"""
    m, g = np.asarray(mask, dtype=bool), np.asarray(gt, dtype=bool)
    h, w = g.shape
    lim = box_to_mask(confine_box, h, w) if confine_box is not None else np.ones((h, w), bool)
    out = []
    for order, (sign, region) in enumerate(((G.SIGN_ADD, g & ~m & lim), (G.SIGN_REMOVE, m & ~g & lim))):
        lab, n = ndimage.label(region, structure=np.ones((3, 3), bool))
        if not n:
            continue
        area = np.bincount(lab.ravel(), minlength=n + 1)
        first = np.full(n + 1, np.iinfo(np.int64).max, np.int64)
        flat = lab.ravel()
        nz = np.flatnonzero(flat)
        np.minimum.at(first, flat[nz], nz)
        sl = ndimage.find_objects(lab)
        for i in range(1, n + 1):
            rs, cs = sl[i - 1]
            out.append({"sign": sign, "area": int(area[i]), "box": [float(cs.start), float(rs.start), float(cs.stop), float(rs.stop)],
                        "_key": (-int(area[i]), order, int(first[i]))})
    out.sort(key=lambda d: d["_key"])
    for d in out:
        del d["_key"]
    return out


def oracle_corrections(mask: np.ndarray, gt: np.ndarray, m: int, *, confine_box: Sequence[float] | None = None,
                       noise: NoiseLevel = NO_NOISE, rng: np.random.Generator | None = None) -> dict:
    """一轮的修正框（规则见模块说明）。返回 {"proposals": [(框（原图像素，已经过 0–1000 网格）, 符号)], "proposals_q", "n_flipped"}。"""
    g = np.asarray(gt, dtype=bool)
    h, w = g.shape
    comps = diff_components(mask, g, confine_box=confine_box)[: max(int(m), 0)]
    props, props_q, n_flip = [], [], 0
    for d in comps:
        sign = d["sign"]
        x0, y0, x1, y1 = d["box"]
        if not noise.is_zero:
            if rng is None:
                raise ValueError("有噪声时要给 rng")
            if noise.corr_sign_flip_rate > 0 and rng.random() < noise.corr_sign_flip_rate:
                sign = G.SIGN_REMOVE if sign == G.SIGN_ADD else G.SIGN_ADD
                n_flip += 1
            sx, sy = noise.corr_box_jitter_frac * (x1 - x0), noise.corr_box_jitter_frac * (y1 - y0)
            if sx > 0 or sy > 0:
                x0, x1 = x0 + float(rng.normal(0.0, sx)), x1 + float(rng.normal(0.0, sx))
                y0, y1 = y0 + float(rng.normal(0.0, sy)), y1 + float(rng.normal(0.0, sy))
            x0, x1 = sorted((min(max(x0, 0.0), float(w)), min(max(x1, 0.0), float(w))))
            y0, y1 = sorted((min(max(y0, 0.0), float(h)), min(max(y1, 0.0), float(h))))
        q = quantize_box([x0, y0, x1, y1], w, h)
        props_q.append({"sign": sign, "box": q})
        props.append((dequantize_box(q, w, h), sign))
    return {"proposals": props, "proposals_q": props_q, "n_flipped": n_flip}


def corrections_text(props_q: Sequence[Mapping[str, Any]]) -> str:
    return json.dumps([{"sign": p["sign"], "box": [int(v) for v in p["box"]]} for p in props_q], separators=JSON_SEPARATORS)


def pick_text(index: int) -> str:
    return str(int(index))


# ---------------------------------------------------------------------------
# tokenizer
# ---------------------------------------------------------------------------


def load_tokenizer(model_key: str) -> tuple[Any, dict]:
    """注册表里 model_key 的 tokenizer（固定 revision；AutoTokenizer，不加载权重）。返回 (tokenizer, 记录)。transformers 延迟 import。"""
    from transformers import AutoTokenizer

    from src.adapters import registry

    e = registry.get_model(model_key)
    tok = AutoTokenizer.from_pretrained(e.repo, revision=e.revision)
    return tok, {"model": model_key, "repo": e.repo, "revision": e.revision, "class": type(tok).__name__}


def count_tokens(tokenizer: Any, texts: Sequence[str]) -> list[int]:
    if not texts:
        return []
    ids = tokenizer(list(texts), add_special_tokens=False)["input_ids"]
    return [len(x) for x in ids]


__all__ = [
    "JSON_SEPARATORS", "NO_NOISE", "NoiseLevel", "TEXT_FORMAT_VERSION", "corrections_text", "count_tokens", "dequantize_box",
    "dequantize_xy", "diff_components", "dt_peaks", "load_tokenizer", "mask_bbox", "oracle_corrections", "oracle_points",
    "oracle_spline", "pick_text", "points_text", "quantize_box", "quantize_xy", "rng_for", "spline_text",
]
