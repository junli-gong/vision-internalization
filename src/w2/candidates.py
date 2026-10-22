"""W2 的候选生成器（DEV-0058；D113；接口约定见 docs/w2_plan.md §1，算法、参数与耗时见 docs/w2_tools.md）。

W2-a 的第二步：LLM 给出框（可选再给几个点）之后，工具在框和点上生成固定数量 K 个候选 mask，编号后交给 LLM 挑一个。
本模块只管"生成"：generate_candidates(image_rgb, box_abs, fg_points_abs=, bg_points_abs=, spec=) → K 个 Candidate。

约定：
- 坐标：框与点都是原图的绝对像素（框 xyxy，点 (x, y)），与 src/w1/baselines 交给 W1-F 的框相同。
- 生成器清单 CandidateSpec.generators：按顺序产出，清单长度就是 K。每项 {"type": …, 参数…}，另有三个通用键：
  name（候选名，缺省按类型与参数自动生成）、post（后处理，见 POST_OPS）。第一项必须恰为 {"type": "grabcut_default"}。
- 1 号候选 grabcut_default 逐位等于 W1-F：与 src/w1/baselines.w1f_episode 同一条调用
  src.w1.workbench.grabcut_mask(image_rgb, box_abs=[float(v) for v in box], iterations=GRABCUT_ITERATIONS)（测试钉住）。
- 失败（生成器跑不起来：GrabCut 的某一类没有像素、标记为空、裁剪区域太小、要点却没有点、OpenCV 报错）：
  mask 用框本身（box_mask，与"自己的框"同一个光栅化），ok = False，原因记进 info["error"]。编号与 K 不变。
  生成器正常跑完、但结果为空 mask 不算失败（照实给出，info["area"] = 0）；这样 1 号候选只要 GrabCut 没报错就与 W1-F 逐位相同。
- 不去重：完全相同的候选照常编号（duplicate_rate 统计重复率）。
- 确定性：同样的输入逐位得到同样的 mask。GrabCut 每次调用前由 grabcut_mask 把 OpenCV 的随机数种子设成固定值；
  SLIC、分水岭、连通域、形态学都是确定的算法。耗时（info["time_s"]）不在确定性之内。
- 同一次调用里，类型与参数相同、只差 post 或 name 的生成器共用一次基础计算（info["reused"] = True，耗时只计后处理）。

禁止项（D113 §二.2）：任何学习型视觉模型或预训练权重，OpenCV 的 DNN 模块。这里只用 numpy、scipy、OpenCV 与
scikit-image 里不含学习的函数（复用 src/w1/workbench.py 的同一批实现）。
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

import cv2
import numpy as np
from skimage.segmentation import slic
from skimage.segmentation import watershed as sk_watershed

from src.executors.geometry import check_box
from src.executors.raster import box_to_mask
from src.w1 import workbench as WB

# ---------------------------------------------------------------------------
# 数据类型（docs/w2_plan.md §1）
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class Candidate:
    name: str  # 生成器名，例如 "gc_default"、"gc_box_plus10"、"slic_s30_t50"、"ws_pts"、"box"
    mask: np.ndarray  # (H, W) bool，原图分辨率
    ok: bool  # 生成器是否正常产出；失败时 mask 为框本身且 ok = False
    info: dict  # 参数、耗时、失败原因


# ---------------------------------------------------------------------------
# 固定参数（docs/w2_tools.md 引用这里的名字；测试核对）
# ---------------------------------------------------------------------------

GRABCUT_DEFAULT = "grabcut_default"
BOX_PCT_MIN = -45.0  # 框内缩至多每边 45%（再多框就没了）
BOX_PCT_MAX = 100.0
SEED_RADIUS_MIN_PX = 2.0  # 点标记圆盘的最小半径（px）
NAME_MAX_LEN = 64
COMMON_KEYS = ("type", "name", "post")
POST_OPS = ("largest_cc", "fill_holes", "open", "close", "dilate", "erode")  # 后四种写成 "op:r"，r 是圆盘半径（px）
_POST_SHORT = {"largest_cc": "lcc", "fill_holes": "fill"}


def _p(kind: str, default: Any, doc: str, **kw: Any) -> dict:
    return {"type": kind, "default": default, "doc": doc, **kw}


_ITER = _p("integer", WB.GRABCUT_ITERATIONS, "GrabCut iterations.", minimum=1, maximum=WB.GRABCUT_MAX_ITERATIONS)
_BOX_PCT = _p(
    "number", 0.0,
    "Each side of the box moves outward by this percent of the box width (left, right) or height (top, bottom); "
    "negative moves inward.", minimum=BOX_PCT_MIN, maximum=BOX_PCT_MAX,
)
_INIT = _p(
    "string", "rect",
    "rect: inside the (adjusted) box = probable foreground, outside = definite background (as W1-F). "
    "mask: the (adjusted) box is GrabCut's init mask, so a band of GRABCUT_MASK_MARGIN around it is only probable background.",
    enum=["rect", "mask"],
)
_SEED_R = _p(
    "number", 4.0, "Seed disk radius around each point, in percent of the box's shorter side (at least SEED_RADIUS_MIN_PX px).",
    minimum=0.0, maximum=50.0,
)

# 全部生成器类型与参数的说明（JSON 可写；docs/w2_tools.md 逐项照录，测试核对）
GENERATOR_TYPES: Mapping[str, Mapping[str, Any]] = MappingProxyType({
    "grabcut_default": {
        "doc": "W1-F: GrabCut initialized with the box (src.w1.workbench.grabcut_mask, GRABCUT_ITERATIONS iterations). No parameters.",
        "uses_points": False,
        "params": {},
    },
    "grabcut": {
        "doc": "GrabCut with a chosen iteration count, box expanded or shrunk by box_pct, initialized as a hard box (rect) or a soft mask.",
        "uses_points": False,
        "params": {"iterations": _ITER, "box_pct": _BOX_PCT, "init": _INIT},
    },
    "grabcut_points": {
        "doc": "As grabcut, plus disks around the foreground / background points as definite foreground / background seeds. "
               "Fails without any point inside the image.",
        "uses_points": True,
        "params": {"iterations": _ITER, "box_pct": _BOX_PCT, "init": _INIT, "seed_radius_pct": _SEED_R},
    },
    "slic": {
        "doc": "SLIC superpixels on a crop around the box; the union of superpixels with at least threshold_pct of their area inside the box.",
        "uses_points": False,
        "params": {
            "n_segments": _p("integer", 50, "Approximate number of superpixels in the crop (skimage slic n_segments).", minimum=4, maximum=2000),
            "threshold_pct": _p("number", 50.0, "A superpixel is kept if at least this percent of its pixels lie inside the box.",
                                minimum=1.0, maximum=100.0),
            "compactness": _p("number", WB.SLIC_COMPACTNESS, "SLIC compactness (CIELAB color vs. space).", minimum=0.01, maximum=100.0),
            "context_pct": _p("number", 25.0, "The crop is the box expanded by this percent per side (as box_pct), clipped to the image.",
                              minimum=0.0, maximum=200.0),
        },
    },
    "watershed": {
        "doc": "Marker-based watershed on the color gradient of the whole image. Foreground marker: disks around the foreground points "
               "(use_points and points given), else the box shrunk by fg_shrink_pct per side. Background marker: everything outside the "
               "box expanded by bg_margin_pct per side, plus disks around the background points (use_points).",
        "uses_points": True,
        "params": {
            "use_points": _p("boolean", True, "Use the points as markers when they are given."),
            "fg_shrink_pct": _p("number", 30.0, "Without foreground points, the foreground marker is the box shrunk by this percent per side.",
                                minimum=0.0, maximum=49.0),
            "bg_margin_pct": _p("number", 0.0, "The background marker starts this percent (per side, as box_pct) outside the box.",
                                minimum=0.0, maximum=100.0),
            "seed_radius_pct": _SEED_R,
        },
    },
    "box": {
        "doc": "The box itself, rasterized like the own-box control (pixel centers in (lo, hi]; equals the COCO polygon raster of its corners).",
        "uses_points": False,
        "params": {},
    },
})

# ---------------------------------------------------------------------------
# 生成器清单
# ---------------------------------------------------------------------------


def _fmt(v: float) -> str:
    return f"{abs(float(v)):g}"


def _gc_suffix(r: Mapping[str, Any]) -> str:
    s = ""
    if r["box_pct"] != 0:
        s += f"_box_{'plus' if r['box_pct'] > 0 else 'minus'}{_fmt(r['box_pct'])}"
    if r["iterations"] != WB.GRABCUT_ITERATIONS:
        s += f"_iter{r['iterations']}"
    return s


def _auto_name(t: str, r: Mapping[str, Any]) -> str:
    """按类型与参数自动生成的候选名（与缺省值相同的参数不写进名字）。"""
    d = {k: v["default"] for k, v in GENERATOR_TYPES[t]["params"].items()}
    if t == GRABCUT_DEFAULT:
        return "gc_default"
    if t == "grabcut":
        base = "gc" if r["init"] == "rect" else "gc_soft"
        suf = _gc_suffix(r)
        return base + (suf or ("_rect" if r["init"] == "rect" else ""))
    if t == "grabcut_points":
        base = "gc_pts" + ("_soft" if r["init"] == "mask" else "")
        return base + _gc_suffix(r) + (f"_r{_fmt(r['seed_radius_pct'])}" if r["seed_radius_pct"] != d["seed_radius_pct"] else "")
    if t == "slic":
        s = f"slic_s{r['n_segments']}_t{_fmt(r['threshold_pct'])}"
        if r["compactness"] != d["compactness"]:
            s += f"_c{_fmt(r['compactness'])}"
        if r["context_pct"] != d["context_pct"]:
            s += f"_ctx{_fmt(r['context_pct'])}"
        return s
    if t == "watershed":
        s = "ws_pts" if r["use_points"] else "ws_center"
        if r["fg_shrink_pct"] != d["fg_shrink_pct"]:
            s += f"_shrink{_fmt(r['fg_shrink_pct'])}"
        if r["bg_margin_pct"] != d["bg_margin_pct"]:
            s += f"_margin{_fmt(r['bg_margin_pct'])}"
        if r["use_points"] and r["seed_radius_pct"] != d["seed_radius_pct"]:
            s += f"_r{_fmt(r['seed_radius_pct'])}"
        return s
    if t == "box":
        return "box"
    raise AssertionError(t)


def _check_value(where: str, pname: str, schema: Mapping[str, Any], v: Any) -> Any:
    kind = schema["type"]
    if kind == "boolean":
        if not isinstance(v, bool):
            raise ValueError(f"{where}：参数 {pname} 必须是 true / false，收到 {v!r}")
        return v
    if kind == "string":
        if v not in schema["enum"]:
            raise ValueError(f"{where}：参数 {pname} 必须是 {schema['enum']} 之一，收到 {v!r}")
        return v
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise ValueError(f"{where}：参数 {pname} 必须是有限的数，收到 {v!r}")
    if kind == "integer":
        if int(v) != v:
            raise ValueError(f"{where}：参数 {pname} 必须是整数，收到 {v!r}")
        v = int(v)
    else:
        v = float(v)
    if not schema["minimum"] <= v <= schema["maximum"]:
        raise ValueError(f"{where}：参数 {pname} = {v:g} 不在 [{schema['minimum']:g}, {schema['maximum']:g}] 内")
    return v


def parse_post(where: str, ops: Any) -> tuple[tuple[str, int | None], ...]:
    """post 的写法：["largest_cc", "fill_holes", "open:3", …]，按顺序执行。返回 ((op, 半径或 None), …)。"""
    if not isinstance(ops, (list, tuple)):
        raise ValueError(f"{where}：post 必须是列表，收到 {ops!r}")
    out = []
    for op in ops:
        if not isinstance(op, str):
            raise ValueError(f"{where}：post 的每项必须是字符串，收到 {op!r}")
        name, _, arg = op.partition(":")
        if name in ("largest_cc", "fill_holes"):
            if arg:
                raise ValueError(f"{where}：后处理 {name} 不带参数，收到 {op!r}")
            out.append((name, None))
        elif name in POST_OPS:
            try:
                r = int(arg)
            except ValueError:
                raise ValueError(f"{where}：后处理 {name} 要写成 '{name}:半径'（整数 px），收到 {op!r}") from None
            if not 1 <= r <= WB.MORPH_RADIUS_MAX or str(r) != arg:
                raise ValueError(f"{where}：后处理 {op!r} 的半径必须是 1..{WB.MORPH_RADIUS_MAX} 的整数")
            out.append((name, r))
        else:
            raise ValueError(f"{where}：未知的后处理 {op!r}；可选 {POST_OPS}")
    return tuple(out)


def _post_suffix(post: Sequence[tuple[str, int | None]]) -> str:
    return "".join(f"+{_POST_SHORT.get(op, op)}{'' if r is None else r}" for op, r in post)


def resolve_generator(entry: Mapping[str, Any], index: int = 0) -> dict:
    """校验一项生成器并补全缺省参数。返回 {"type", 全部参数…, "post": [...], "name"}（JSON 可写）。"""
    where = f"生成器 {index + 1}"
    if not isinstance(entry, Mapping):
        raise ValueError(f"{where}：必须是对象，收到 {entry!r}")
    t = entry.get("type")
    if t not in GENERATOR_TYPES:
        raise ValueError(f"{where}：未知的类型 {t!r}；可选 {sorted(GENERATOR_TYPES)}")
    params = GENERATOR_TYPES[t]["params"]
    extra = set(entry) - set(params) - set(COMMON_KEYS)
    if extra:
        raise ValueError(f"{where}（{t}）：未知的参数 {sorted(extra)}；可选 {sorted(params)} 与 {list(COMMON_KEYS[1:])}")
    r: dict[str, Any] = {"type": t}
    for pname, schema in params.items():
        r[pname] = _check_value(f"{where}（{t}）", pname, schema, entry.get(pname, schema["default"]))
    post = parse_post(f"{where}（{t}）", entry.get("post", []))
    r["post"] = [op if rad is None else f"{op}:{rad}" for op, rad in post]
    name = entry.get("name")
    if name is None:
        name = _auto_name(t, r) + _post_suffix(post)
    if not isinstance(name, str) or not name or len(name) > NAME_MAX_LEN or not name.isprintable() or name != name.strip():
        raise ValueError(f"{where}：name 必须是 1..{NAME_MAX_LEN} 个可打印字符、首尾无空白，收到 {name!r}")
    r["name"] = name
    return r


@dataclass(frozen=True)
class CandidateSpec:
    """configs/w2.json 的 candidates 一节；生成器按清单顺序产出，清单长度就是 K。第一项必须是 {"type": "grabcut_default"}。

    构造时校验每一项（类型、参数名、取值范围、post 的写法、候选名唯一），出错抛 ValueError。generators 保存原样（深拷贝），
    resolved 是补全缺省值之后的清单。"""

    generators: tuple[dict, ...]

    def __post_init__(self) -> None:
        gens = self.generators
        if isinstance(gens, Mapping) or not isinstance(gens, (list, tuple)):
            raise ValueError(f"generators 必须是列表，收到 {type(gens).__name__}")
        try:
            gens = tuple(json.loads(json.dumps(list(gens), allow_nan=False)))
        except (TypeError, ValueError) as e:
            raise ValueError(f"generators 必须能写成 JSON：{e}") from None
        if not gens:
            raise ValueError("generators 不能为空")
        if gens[0] != {"type": GRABCUT_DEFAULT}:
            raise ValueError(f"第一项必须恰为 {{'type': '{GRABCUT_DEFAULT}'}}（1 号候选 = W1-F），收到 {gens[0]!r}")
        resolved = tuple(resolve_generator(g, i) for i, g in enumerate(gens))
        names = [r["name"] for r in resolved]
        dup = sorted({n for n in names if names.count(n) > 1})
        if dup:
            raise ValueError(f"候选名重复：{dup}（相同的生成器写两遍没有意义；确需如此请给不同的 name）")
        object.__setattr__(self, "generators", gens)
        object.__setattr__(self, "_resolved", resolved)

    @property
    def resolved(self) -> tuple[dict, ...]:
        return tuple(dict(r, post=list(r["post"])) for r in self._resolved)  # type: ignore[attr-defined]

    @property
    def k(self) -> int:
        return len(self.generators)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(r["name"] for r in self._resolved)  # type: ignore[attr-defined]

    @property
    def uses_points(self) -> bool:
        return any(GENERATOR_TYPES[r["type"]]["uses_points"] for r in self._resolved)  # type: ignore[attr-defined]

    def as_dict(self) -> dict:
        """configs/w2.json 的写法：{"generators": [...]}（原样）。"""
        return {"generators": [dict(g) for g in self.generators]}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "CandidateSpec":
        if not isinstance(d, Mapping) or set(d) != {"generators"}:
            raise ValueError(f"candidates 一节必须恰有 generators 一个键，收到 {sorted(d) if isinstance(d, Mapping) else d!r}")
        return cls(generators=tuple(d["generators"]))

    def __hash__(self) -> int:
        return hash(self.digest())

    def digest(self) -> str:
        """补全缺省值之后的清单的 sha256（候选缓存与运行身份用；写法不同、含义相同的清单得到同一个值）。"""
        return hashlib.sha256(json.dumps(list(self._resolved), sort_keys=True, ensure_ascii=False,  # type: ignore[attr-defined]
                                         separators=(",", ":")).encode("utf-8")).hexdigest()


# 默认清单（调参的起点；理由见 docs/w2_tools.md §4）。K = 8；网格布局加原图参照正好 3×3。
DEFAULT_SPEC = CandidateSpec(generators=(
    {"type": "grabcut_default"},
    {"type": "grabcut", "iterations": 5, "box_pct": 10.0, "init": "rect"},
    {"type": "grabcut", "iterations": 5, "box_pct": -10.0, "init": "mask"},
    {"type": "grabcut_default", "post": ["largest_cc", "fill_holes"]},
    {"type": "slic", "n_segments": 30, "threshold_pct": 50.0, "compactness": 10.0, "context_pct": 25.0},
    {"type": "slic", "n_segments": 100, "threshold_pct": 50.0, "compactness": 10.0, "context_pct": 25.0},
    {"type": "watershed", "use_points": True, "fg_shrink_pct": 30.0, "bg_margin_pct": 0.0, "seed_radius_pct": 4.0},
    {"type": "box"},
))

# ---------------------------------------------------------------------------
# 生成
# ---------------------------------------------------------------------------


class _Ctx:
    """一次 generate_candidates 的共用输入与缓存（框、点、彩色梯度）。"""

    def __init__(self, img: np.ndarray, box: tuple[float, float, float, float], fg: np.ndarray, bg: np.ndarray):
        self.img, self.box, self.fg, self.bg = img, box, fg, bg
        self.h, self.w = img.shape[:2]
        self.box_mask = box_to_mask(box, self.h, self.w)
        self._gradient: np.ndarray | None = None

    def gradient(self) -> np.ndarray:
        if self._gradient is None:
            self._gradient = WB.color_gradient(self.img)
        return self._gradient

    def adjusted(self, pct: float) -> tuple[float, float, float, float]:
        """框每边外扩 pct%（左右按框宽、上下按框高；负数内缩）。不夹到图内：光栅化时图外的部分自然忽略。"""
        x0, y0, x1, y1 = self.box
        dx, dy = pct / 100.0 * (x1 - x0), pct / 100.0 * (y1 - y0)
        return x0 - dx, y0 - dy, x1 + dx, y1 + dy

    def seed_radius(self, pct: float) -> float:
        x0, y0, x1, y1 = self.box
        return max(SEED_RADIUS_MIN_PX, pct / 100.0 * min(x1 - x0, y1 - y0))

    def disks(self, pts: np.ndarray, pct: float) -> np.ndarray | None:
        return WB.point_disks((self.h, self.w), pts, self.seed_radius(pct)) if len(pts) else None


def _gen_grabcut_default(c: _Ctx, r: Mapping[str, Any]) -> tuple[np.ndarray, dict]:
    # 与 src/w1/baselines.w1f_episode 同一条调用（W1-F）：框原样、迭代次数取工作台的常量
    return WB.grabcut_mask(c.img, box_abs=[float(v) for v in c.box], iterations=WB.GRABCUT_ITERATIONS), {}


def _gc_kwargs(c: _Ctx, r: Mapping[str, Any]) -> tuple[dict, dict]:
    adj = c.adjusted(r["box_pct"])
    kw: dict[str, Any] = {"iterations": r["iterations"]}
    if r["init"] == "rect":
        kw["box_abs"] = list(adj)
    else:
        kw["init_mask"] = box_to_mask(adj, c.h, c.w)
    return kw, {"box_used": [round(v, 3) for v in adj]}


def _gen_grabcut(c: _Ctx, r: Mapping[str, Any]) -> tuple[np.ndarray, dict]:
    kw, info = _gc_kwargs(c, r)
    return WB.grabcut_mask(c.img, **kw), info


def _gen_grabcut_points(c: _Ctx, r: Mapping[str, Any]) -> tuple[np.ndarray, dict]:
    if not len(c.fg) and not len(c.bg):
        raise ValueError("没有落在图内的点（grabcut_points 要点）")
    kw, info = _gc_kwargs(c, r)
    fg, bg = c.disks(c.fg, r["seed_radius_pct"]), c.disks(c.bg, r["seed_radius_pct"])
    if fg is not None:
        kw["fg"] = fg
    if bg is not None:
        kw["bg"] = bg
    info.update(seed_radius_px=round(c.seed_radius(r["seed_radius_pct"]), 3), n_fg_points=len(c.fg), n_bg_points=len(c.bg))
    return WB.grabcut_mask(c.img, **kw), info


def _gen_slic(c: _Ctx, r: Mapping[str, Any]) -> tuple[np.ndarray, dict]:
    if not c.box_mask.any():
        raise ValueError("框里没有像素")
    x0, y0, x1, y1 = c.adjusted(r["context_pct"])
    c0, c1 = max(int(math.floor(x0)), 0), min(int(math.ceil(x1)), c.w)
    r0, r1 = max(int(math.floor(y0)), 0), min(int(math.ceil(y1)), c.h)
    if c1 - c0 < 2 or r1 - r0 < 2:
        raise ValueError(f"裁剪区域 {c1 - c0}×{r1 - r0} px 太小")
    lab = slic(np.ascontiguousarray(c.img[r0:r1, c0:c1]), n_segments=r["n_segments"], compactness=r["compactness"],
               sigma=WB.SLIC_SIGMA, start_label=1, channel_axis=-1)
    n = int(lab.max())
    total = np.bincount(lab.ravel(), minlength=n + 1)
    inside = np.bincount(lab[c.box_mask[r0:r1, c0:c1]], minlength=n + 1)
    keep = inside * 100.0 >= r["threshold_pct"] * total
    keep[0] = False
    out = np.zeros((c.h, c.w), bool)
    out[r0:r1, c0:c1] = keep[lab]
    return out, {"crop": [c0, r0, c1, r1], "n_superpixels": n, "n_kept": int(keep.sum())}


def _gen_watershed(c: _Ctx, r: Mapping[str, Any]) -> tuple[np.ndarray, dict]:
    pts = r["use_points"]
    fg = c.disks(c.fg, r["seed_radius_pct"]) if pts else None
    how = "points"
    if fg is None:
        fg, how = box_to_mask(c.adjusted(-r["fg_shrink_pct"]), c.h, c.w), "center"
    bg = ~box_to_mask(c.adjusted(r["bg_margin_pct"]), c.h, c.w)
    if pts and len(c.bg):
        bg |= c.disks(c.bg, r["seed_radius_pct"])
    if not fg.any():
        raise ValueError("前景标记为空（框太小）")
    markers = np.zeros((c.h, c.w), np.int32)
    markers[bg] = 1
    markers[fg] = 2  # 前景与背景标记重叠时前景优先（例如前景点落在框外）
    if not (markers == 1).any():
        raise ValueError("背景标记为空（框盖满了整张图）")
    return sk_watershed(c.gradient(), markers) == 2, {"fg_marker": how, "fg_marker_px": int(fg.sum())}


def _gen_box(c: _Ctx, r: Mapping[str, Any]) -> tuple[np.ndarray, dict]:
    return c.box_mask.copy(), {}


_GENERATORS: Mapping[str, Callable[[_Ctx, Mapping[str, Any]], tuple[np.ndarray, dict]]] = MappingProxyType({
    "grabcut_default": _gen_grabcut_default,
    "grabcut": _gen_grabcut,
    "grabcut_points": _gen_grabcut_points,
    "slic": _gen_slic,
    "watershed": _gen_watershed,
    "box": _gen_box,
})
assert set(_GENERATORS) == set(GENERATOR_TYPES)


def apply_post(mask: np.ndarray, post: Sequence[str]) -> np.ndarray:
    """按顺序执行后处理：largest_cc（只留面积最大的 8 连通域，同面积取扫描顺序在前的）、fill_holes、open/close/dilate/erode:r
    （src.w1.workbench.morph_mask，半径 r 的圆盘）。总是返回新数组（不与输入共用内存）；空 mask 照样是空的。"""
    out = np.array(mask, dtype=bool, copy=True)
    for op, rad in parse_post("post", list(post)):
        if op == "largest_cc":
            lab, n, order = WB.components_by_area(out)
            if n:
                out = lab == order[0]
        else:
            out = np.asarray(WB.morph_mask(out, op, rad if rad is not None else WB.MORPH_RADIUS_DEFAULT), dtype=bool)
    return out


def _points(pts: Any, h: int, w: int, what: str) -> tuple[np.ndarray, int]:
    """点 → (落在图内 [0, W] × [0, H] 的点 (n, 2), 丢掉的个数)。非有限值报错。"""
    if pts is None:
        return np.zeros((0, 2)), 0
    a = np.asarray(pts, dtype=np.float64)
    if a.size == 0:
        return np.zeros((0, 2)), 0
    if a.ndim != 2 or a.shape[1] != 2:
        raise ValueError(f"{what} 必须是 (n, 2) 的 (x, y) 绝对像素，收到形状 {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{what} 含非有限值")
    keep = (a[:, 0] >= 0) & (a[:, 0] <= w) & (a[:, 1] >= 0) & (a[:, 1] <= h)
    return a[keep], int((~keep).sum())


def generate_candidates(image_rgb: np.ndarray, box_abs: Sequence[float], *, fg_points_abs: Any = None, bg_points_abs: Any = None,
                        spec: CandidateSpec) -> list[Candidate]:
    """按 spec 在框（与可选的点）上生成 K 个候选（模块说明）。

    image_rgb：(H, W, 3) uint8，即 np.asarray(PIL 原图.convert("RGB"))。box_abs：原图绝对像素 xyxy，必须有限且 x1 ≥ x0、y1 ≥ y0
    （解析器保证；否则抛 ValueError——这是调用方的错，不是生成器失败）。fg_points_abs / bg_points_abs：(n, 2) 的 (x, y) 绝对像素，
    落在图外的点丢掉（info["dropped_points"] 记个数）；落在框外、图内的点照用。
    """
    if not isinstance(spec, CandidateSpec):
        raise TypeError(f"spec 必须是 CandidateSpec，收到 {type(spec).__name__}")
    img = np.asarray(image_rgb)
    if img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"image_rgb 必须是 (H, W, 3) uint8，收到 dtype {img.dtype}、形状 {img.shape}")
    h, w = img.shape[:2]
    box = check_box(box_abs)
    fg, n_drop_fg = _points(fg_points_abs, h, w, "fg_points_abs")
    bg, n_drop_bg = _points(bg_points_abs, h, w, "bg_points_abs")
    c = _Ctx(img, box, fg, bg)
    base_cache: dict[str, tuple[np.ndarray | None, dict, str | None]] = {}
    out: list[Candidate] = []
    for i, r in enumerate(spec.resolved, start=1):
        t = r["type"]
        params = {k: v for k, v in r.items() if k not in COMMON_KEYS}
        key = json.dumps([t, params], sort_keys=True)
        t0 = time.perf_counter()
        reused = key in base_cache
        if not reused:
            try:
                m, extra = _GENERATORS[t](c, r)
                m = np.asarray(m, dtype=bool)
                if m.shape != (h, w):
                    raise RuntimeError(f"生成器 {t} 返回形状 {m.shape}，应为 {(h, w)}")
                base_cache[key] = (m, extra, None)
            except (ValueError, cv2.error) as e:
                base_cache[key] = (None, {}, f"{type(e).__name__}: {str(e).strip()[:300]}")
        base, extra, err = base_cache[key]
        mask = apply_post(base, r["post"]) if base is not None else c.box_mask.copy()
        ok = base is not None
        info = {"index": i, "type": t, "params": params, "post": list(r["post"]), "ok": ok, "error": err, "reused": reused,
                "time_s": time.perf_counter() - t0, "area": int(np.count_nonzero(mask)), **extra}
        if GENERATOR_TYPES[t]["uses_points"]:
            info["dropped_points"] = {"fg": n_drop_fg, "bg": n_drop_bg}
        out.append(Candidate(name=r["name"], mask=mask, ok=ok, info=info))
    return out


def duplicate_rate(candidates: Sequence[Candidate]) -> float:
    """重复率：K 个候选里，与编号更小的某个候选逐位相同的候选所占的比例（0 = 两两不同）。"""
    seen: set[bytes] = set()
    dup = 0
    for cand in candidates:
        d = hashlib.sha256(np.packbits(cand.mask).tobytes() + repr(cand.mask.shape).encode()).digest()
        dup += d in seen
        seen.add(d)
    return dup / len(candidates) if candidates else 0.0


# ---------------------------------------------------------------------------
# 耗时（docs/w2_tools.md §3；W2-c 估算训练集规模用）：python -m src.w2.candidates --bench <图目录>
# ---------------------------------------------------------------------------

# 每种生成器类型至少一项（缺省参数或 DEFAULT_SPEC 的参数），外加 DEFAULT_SPEC 没有的几种常见变体
BENCH_SPEC = CandidateSpec(generators=(
    {"type": "grabcut_default"},
    {"type": "grabcut", "box_pct": 10.0},
    {"type": "grabcut", "init": "mask", "box_pct": -10.0},
    {"type": "grabcut", "iterations": 10},
    {"type": "grabcut", "iterations": 1},
    {"type": "grabcut_points"},
    {"type": "grabcut_default", "post": ["largest_cc", "fill_holes"]},
    {"type": "slic", "n_segments": 30},
    {"type": "slic", "n_segments": 100},
    {"type": "watershed", "use_points": False},
    {"type": "watershed"},
    {"type": "box"},
))
BENCH_BOXES = ((0.25, 0.25, 0.75, 0.75), (0.40, 0.35, 0.65, 0.70), (0.05, 0.10, 0.90, 0.95))  # 相对原图宽高的 xyxy：中、小、大


def _machine() -> dict[str, Any]:
    import os
    import platform
    import subprocess

    import skimage

    cpu = platform.processor() or platform.machine()
    try:
        if platform.system() == "Darwin":
            cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=True).stdout.strip()
        elif os.path.exists("/proc/cpuinfo"):
            with open("/proc/cpuinfo", encoding="utf-8") as f:
                cpu = next((ln.split(":", 1)[1].strip() for ln in f if ln.startswith("model name")), cpu)
    except (OSError, subprocess.CalledProcessError, StopIteration):
        pass
    return {"platform": platform.platform(), "cpu": cpu, "cpu_count": os.cpu_count(), "python": platform.python_version(),
            "opencv": cv2.__version__, "opencv_threads": cv2.getNumThreads(), "skimage": skimage.__version__, "numpy": np.__version__}


def bench(paths: Sequence[str], repeats: int = 1, progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    """每张图 × BENCH_BOXES 的每个框：BENCH_SPEC 里每个生成器的单次耗时（info["time_s"]，共用的基础计算只计一次），
    以及 DEFAULT_SPEC 整份清单的墙钟。grabcut_points / watershed 用两个点：框中心（前景）、框左上角外侧 5% 处（背景）。"""
    from PIL import Image

    per: dict[str, list[float]] = {}
    totals: list[float] = []
    sizes = set()
    for k, p in enumerate(paths, start=1):
        if progress:
            progress(f"[{k}/{len(paths)}] {p}（DEFAULT_SPEC 到目前的中位数 {np.median(totals) if totals else float('nan'):.2f} s）")
        img = np.asarray(Image.open(p).convert("RGB"))
        h, w = img.shape[:2]
        sizes.add(f"{w}x{h}")
        for fx0, fy0, fx1, fy1 in BENCH_BOXES:
            box = [fx0 * w, fy0 * h, fx1 * w, fy1 * h]
            fg = [[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]]
            bg = [[max(box[0] - 0.05 * w, 0.0), max(box[1] - 0.05 * h, 0.0)]]
            for _ in range(repeats):
                for cand in generate_candidates(img, box, fg_points_abs=fg, bg_points_abs=bg, spec=BENCH_SPEC):
                    per.setdefault(cand.name, []).append(cand.info["time_s"])
                t0 = time.perf_counter()
                generate_candidates(img, box, spec=DEFAULT_SPEC)
                totals.append(time.perf_counter() - t0)

    def stats(v: Sequence[float]) -> dict[str, float]:
        a = np.asarray(v)
        return {"n": int(a.size), "median_s": float(np.median(a)), "mean_s": float(a.mean()), "max_s": float(a.max())}

    return {"machine": _machine(), "images": len(paths), "image_paths": list(paths), "sizes": sorted(sizes),
            "boxes": [list(b) for b in BENCH_BOXES],
            "repeats": repeats, "per_generator": {k: stats(v) for k, v in per.items()}, "default_spec_total": stats(totals),
            "default_spec_names": list(DEFAULT_SPEC.names)}


def _main(argv: Sequence[str] | None = None) -> int:
    import argparse
    import glob
    import os

    ap = argparse.ArgumentParser(description="W2 候选生成器的耗时（docs/w2_tools.md §3）")
    ap.add_argument("--bench", required=True, help="图目录（递归找 *.jpg、*.png）")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--size", default="640x480", help="只用这个尺寸的图（宽x高；'any' = 不限）")
    args = ap.parse_args(argv)
    from PIL import Image

    paths = sorted(p for ext in ("jpg", "jpeg", "png") for p in glob.glob(os.path.join(args.bench, "**", f"*.{ext}"), recursive=True))
    paths = [p for p in paths if Image.open(p).mode in ("RGB", "RGBA", "P", "L")]  # 跳过 16 位深度图之类
    if args.size != "any":
        paths = [p for p in paths if "x".join(map(str, Image.open(p).size)) == args.size]
    if not paths:
        print(f"{args.bench} 里没有 {args.size} 的图")
        return 2
    # 预热一次（首次调用含库的初始化）
    generate_candidates(np.asarray(Image.open(paths[0]).convert("RGB")), [10.0, 10.0, 100.0, 100.0], spec=DEFAULT_SPEC)
    import sys

    res = bench(paths, args.repeats, progress=lambda s: print(s, file=sys.stderr, flush=True))
    print(json.dumps(res, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = [
    "BENCH_SPEC",
    "BOX_PCT_MAX",
    "BOX_PCT_MIN",
    "COMMON_KEYS",
    "Candidate",
    "CandidateSpec",
    "DEFAULT_SPEC",
    "GENERATOR_TYPES",
    "GRABCUT_DEFAULT",
    "POST_OPS",
    "SEED_RADIUS_MIN_PX",
    "apply_post",
    "duplicate_rate",
    "generate_candidates",
    "parse_post",
    "resolve_generator",
]
