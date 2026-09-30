"""场景规格（dataclass，可序列化成 JSON）与渲染（numpy；无学习）。

- SceneSpec：画布 (H, W)、背景、可选的地面（d 的侧视场景）、物体列表。物体按 obj_id 排列，
  z 越大越靠前；渲染按 z 升序画（同 z 按 obj_id），后画的盖住先画的。
- 每个物体的非模态 mask = 它的多边形的光栅化（raster.rasterize，精确整数规则）；可见 mask = 非模态 mask 减去所有
  更靠前物体的非模态 mask 之并。图像 = 背景 → 地面带 → 各物体按 z 序涂色，所以任一物体可见 mask 里的
  像素颜色必然是该物体外观的颜色之一（纯色即等于该颜色；条纹/棋盘格为主色或黑色），无抗锯齿、无容差。
- 条纹与棋盘格在物体自身坐标系里计算（随物体旋转），格宽 = 外接半径 × PATTERN_CELL_FRAC。
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field

import numpy as np

from src.synth.constants import PATTERN_CELL_FRAC, PATTERN_SECOND_COLOR
from src.synth.shapes import Identity, appearance, describe, identity_key, pose_polygon, qcoord, rasterize, rot_cs


@dataclass(frozen=True)
class ObjectSpec:
    obj_id: int
    family: str
    variant: int
    appearance: int
    cx: float  # 质心 x（像素）
    cy: float  # 质心 y（像素）
    radius: float  # 外接半径（像素）
    angle: float  # 度，屏幕上顺时针为正
    z: int = 0

    @property
    def identity(self) -> Identity:
        return (self.family, int(self.variant), int(self.appearance))

    @property
    def key(self) -> str:
        return identity_key(self.identity)

    @property
    def description(self) -> str:
        return describe(self.family, self.variant, self.appearance)

    def polygon(self) -> np.ndarray:
        return pose_polygon(self.family, self.variant, self.cx, self.cy, self.radius, self.angle)

    def moved(self, **pose) -> "ObjectSpec":
        """换位姿（cx、cy、radius、angle 中的任意几个），数值量化规则不变。"""
        for k in ("cx", "cy", "radius"):
            if k in pose:
                pose[k] = qcoord(pose[k])
        return dataclasses.replace(self, **pose)


@dataclass(frozen=True)
class Background:
    kind: str  # "solid" | "vgradient"
    top: tuple[int, int, int]
    bottom: tuple[int, int, int]


@dataclass(frozen=True)
class Ground:
    y: float  # 地面上表面（像素）；中心在其下方的像素行画成地面色
    color: tuple[int, int, int]


@dataclass(frozen=True)
class SceneSpec:
    height: int
    width: int
    background: Background
    objects: tuple[ObjectSpec, ...] = field(default_factory=tuple)
    ground: Ground | None = None

    # ---- 序列化 ----
    def to_dict(self) -> dict:
        return json.loads(json.dumps(dataclasses.asdict(self)))

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, d: dict) -> "SceneSpec":
        bg = d["background"]
        g = d.get("ground")
        return cls(
            height=int(d["height"]),
            width=int(d["width"]),
            background=Background(bg["kind"], tuple(bg["top"]), tuple(bg["bottom"])),
            objects=tuple(ObjectSpec(**o) for o in d["objects"]),
            ground=None if g is None else Ground(float(g["y"]), tuple(g["color"])),
        )

    @classmethod
    def from_json(cls, s: str) -> "SceneSpec":
        return cls.from_dict(json.loads(s))

    def object(self, obj_id: int) -> ObjectSpec:
        for o in self.objects:
            if o.obj_id == obj_id:
                return o
        raise KeyError(obj_id)


# ---------------------------------------------------------------------------
# 渲染
# ---------------------------------------------------------------------------


def render_background(bg: Background, height: int, width: int) -> np.ndarray:
    """纯色，或从上到下的线性渐变（整数运算：第 i 行 = (top·(H-1-i) + bottom·i + (H-1)//2) // (H-1)）。"""
    top = np.asarray(bg.top, dtype=np.int64)
    if bg.kind == "solid":
        row = np.broadcast_to(top, (height, 3))
    elif bg.kind == "vgradient":
        bot = np.asarray(bg.bottom, dtype=np.int64)
        i = np.arange(height, dtype=np.int64)[:, None]
        den = max(height - 1, 1)
        row = (top * (den - i) + bot * i + den // 2) // den
    else:
        raise ValueError(f"未知背景类型 {bg.kind!r}")
    return np.ascontiguousarray(np.broadcast_to(row[:, None, :], (height, width, 3)).astype(np.uint8))


def object_masks(scene: SceneSpec) -> np.ndarray:
    """(K, H, W) bool：每个物体的非模态 mask，顺序与 scene.objects 相同。"""
    h, w = scene.height, scene.width
    if not scene.objects:
        return np.zeros((0, h, w), dtype=bool)
    return np.stack([rasterize(o.polygon(), h, w) for o in scene.objects])


def paint_order(scene: SceneSpec) -> list[int]:
    """渲染顺序（scene.objects 的下标）：z 升序，同 z 按 obj_id。"""
    return sorted(range(len(scene.objects)), key=lambda k: (scene.objects[k].z, scene.objects[k].obj_id))


def visible_masks(scene: SceneSpec, amodal: np.ndarray) -> np.ndarray:
    """(K, H, W) bool：每个物体的可见部分 = 非模态 mask 减去所有更靠前物体的非模态 mask。"""
    vis = np.zeros_like(amodal)
    cover = np.zeros(amodal.shape[1:], dtype=bool)
    for k in reversed(paint_order(scene)):
        vis[k] = amodal[k] & ~cover
        cover |= amodal[k]
    return vis


def _pattern_select(obj: ObjectSpec, rows: np.ndarray, cols: np.ndarray, pattern: str) -> np.ndarray:
    """条纹/棋盘格：像素中心在物体坐标系里的 (u, v)，按格宽取整后的奇偶。True = 第二颜色（黑）。"""
    c, s = rot_cs(obj.angle)
    dx = cols.astype(np.float64) + 0.5 - obj.cx
    dy = rows.astype(np.float64) + 0.5 - obj.cy
    u = dx * c + dy * s  # R(angle) 的逆作用在 (dx, dy) 上
    v = dy * c - dx * s
    cell = qcoord(obj.radius * PATTERN_CELL_FRAC)
    iv = np.floor(v / cell).astype(np.int64)
    if pattern == "striped":
        return (iv % 2) == 1
    iu = np.floor(u / cell).astype(np.int64)
    return ((iu + iv) % 2) == 1


def paint_object(img: np.ndarray, mask: np.ndarray, obj: ObjectSpec) -> None:
    app = appearance(obj.appearance)
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return
    if app.pattern == "solid":
        img[rows, cols] = app.rgb
        return
    sel = _pattern_select(obj, rows, cols, app.pattern)
    colors = np.array([app.rgb, PATTERN_SECOND_COLOR[1]], dtype=np.uint8)
    img[rows, cols] = colors[sel.astype(np.int64)]


def render_image(scene: SceneSpec, amodal: np.ndarray | None = None) -> np.ndarray:
    """(H, W, 3) uint8：背景 → 地面带 → 各物体按 z 序涂色。amodal 可以传入已算好的 object_masks(scene)。"""
    h, w = scene.height, scene.width
    img = render_background(scene.background, h, w)
    if scene.ground is not None:
        rows = np.arange(h, dtype=np.float64) + 0.5 > scene.ground.y
        img[rows] = scene.ground.color
    if amodal is None:
        amodal = object_masks(scene)
    for k in paint_order(scene):
        paint_object(img, amodal[k], scene.objects[k])
    return img
