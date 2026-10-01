"""图像口径（每张图送进视觉编码器之前怎么缩放 / 切 tile，决定视觉 token 数）的唯一登记处（D13-5、D28-3；DEV-0011）。

每个家族一套策略类型，构造适配器时给定，适配器每次调用图像处理器都把参数显式传入（不依赖处理器默认值，D13-5）：

- Qwen3.5：二选一。
  - `FixedSize(h, w)`：先用 PIL 把图缩放到 (h, w)（都是 32 的倍数），处理器不再缩放；全部图共用一个网格 (h/32, w/32)。
    DEV-0003 起的原有模式（冒烟、健康检查沿用它）。
  - `PixelBudget(n_tokens)`：不预先缩放，处理器按 smart resize 保持长宽比，`min_pixels = max_pixels = n_tokens·32²`；
    每张图有自己的网格。网格的 token 数只是约等于 n_tokens：放大时向上取整、缩小时向下取整到 32 的倍数
    （例如 540×360 在 300 档是 22×15 = 330 个；640×480 恰为 20×15 = 300 个，不缩放）。
- Gemma 4：`SoftTokenBudget(max_soft_tokens)`，只能取 70/140/280/560/1120（适配器构造时校验）。
- InternVL3.5：`TilePolicy(crop_to_patches, max_patches)`。单 tile 是 `TilePolicy(False, 1)`；原生动态 tile 是
  `TilePolicy(True, 12)`（preprocessor_config 的 12；640×480 切成 12 个 tile 加 1 张缩略图 = 13 个、3328 个 token）。

E1 冻结的口径（D28-3，E3 的 T 通道评测必须沿用）见 E1_IMAGE_POLICIES；factory.make_adapter 不给 image_policy 时取它。
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Union

from src.adapters import registry

# Qwen 每个 LLM 视觉 token 对应缩放后图像的边长（patch 16 × merge 2；model_notes §5.3）。qwen35.PIXELS_PER_LLM_TOKEN
# 引用这里，适配器 attach 时与 config 比对。
QWEN_PIXELS_PER_TOKEN = 32


def _positive_int(name: str, v: Any) -> int:
    if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
        raise ValueError(f"{name} 必须是正整数，收到 {v!r}")
    return int(v)


@dataclass(frozen=True)
class FixedSize:
    """Qwen：所有图先缩放到 (h, w)，h、w 都是 32 的倍数（不保持长宽比）。"""

    h: int
    w: int

    def __post_init__(self) -> None:
        for name in ("h", "w"):
            v = _positive_int(name, getattr(self, name))
            if v % QWEN_PIXELS_PER_TOKEN:
                raise ValueError(f"FixedSize 的 {name}={v} 不是 {QWEN_PIXELS_PER_TOKEN} 的倍数")

    def as_dict(self) -> dict[str, Any]:
        return {"image_size": [self.h, self.w]}


@dataclass(frozen=True)
class PixelBudget:
    """Qwen：保持长宽比，min_pixels = max_pixels = n_tokens·32²，由处理器的 smart resize 决定每张图的网格。"""

    n_tokens: int

    def __post_init__(self) -> None:
        _positive_int("n_tokens", self.n_tokens)

    @property
    def pixels(self) -> int:
        """传给处理器的 min_pixels = max_pixels。"""
        return self.n_tokens * QWEN_PIXELS_PER_TOKEN * QWEN_PIXELS_PER_TOKEN

    def as_dict(self) -> dict[str, Any]:
        return {"pixel_budget_tokens": self.n_tokens, "min_pixels": self.pixels, "max_pixels": self.pixels}


@dataclass(frozen=True)
class SoftTokenBudget:
    """Gemma 4：每张图的 max_soft_tokens（按长宽比缩放，网格 h·w ≤ max_soft_tokens）。"""

    max_soft_tokens: int

    def __post_init__(self) -> None:
        _positive_int("max_soft_tokens", self.max_soft_tokens)

    def as_dict(self) -> dict[str, Any]:
        return {"max_soft_tokens": self.max_soft_tokens}


@dataclass(frozen=True)
class TilePolicy:
    """InternVL3.5：是否动态切 tile、tile 数上限（min_patches 固定为 1，tile 边长 448，见 internvl35）。"""

    crop_to_patches: bool
    max_patches: int

    def __post_init__(self) -> None:
        if not isinstance(self.crop_to_patches, bool):
            raise TypeError(f"crop_to_patches 必须是 bool，收到 {self.crop_to_patches!r}")
        _positive_int("max_patches", self.max_patches)

    def as_dict(self) -> dict[str, Any]:
        return {"crop_to_patches": self.crop_to_patches, "max_patches": self.max_patches}


ImagePolicy = Union[FixedSize, PixelBudget, SoftTokenBudget, TilePolicy]

# 各家族接受的策略类型
FAMILY_POLICY_TYPES: Mapping[str, tuple[type, ...]] = MappingProxyType({
    registry.FAMILY_QWEN35: (FixedSize, PixelBudget),
    registry.FAMILY_GEMMA4: (SoftTokenBudget,),
    registry.FAMILY_INTERNVL35: (TilePolicy,),
})

# E1 的口径（D28-3，冻结；E3 的 T 通道评测沿用）
E1_PIXEL_BUDGET_TOKENS = 300  # Qwen：640×480 恰为 20×15 = 300 个 token，不放大也不缩小
E1_SUBSET_PIXEL_BUDGET_TOKENS = 1024  # Qwen：约 500 条子集上的约 1K token 档（D28-3）
E1_IMAGE_POLICIES: Mapping[str, ImagePolicy] = MappingProxyType({
    registry.FAMILY_QWEN35: PixelBudget(E1_PIXEL_BUDGET_TOKENS),
    registry.FAMILY_GEMMA4: SoftTokenBudget(280),
    registry.FAMILY_INTERNVL35: TilePolicy(crop_to_patches=False, max_patches=1),
})
# InternVL 的原生动态 tile（preprocessor_config 的 max_patches=12）：只用于子集对照，量出单 tile 的代价（D28-3）
INTERNVL_NATIVE_TILES = TilePolicy(crop_to_patches=True, max_patches=12)


def check_policy(family: str, policy: Any) -> ImagePolicy:
    """policy 是否是 family 接受的策略类型；不是就报 TypeError（不静默换算）。"""
    if family not in FAMILY_POLICY_TYPES:
        raise ValueError(f"未知家族 {family!r}")
    allowed = FAMILY_POLICY_TYPES[family]
    if not isinstance(policy, allowed):
        names = " / ".join(t.__name__ for t in allowed)
        raise TypeError(f"{family} 的图像口径必须是 {names}，收到 {policy!r}")
    return policy


def parse_policy(spec: str) -> ImagePolicy:
    """命令行写法 → 策略对象（validate_batching.py 等脚本用）：
    `fixed:480x640`（H×W）、`pixel_budget:300`、`soft_tokens:280`、`tiles:single`（= TilePolicy(False, 1)）、
    `tiles:native`（= INTERNVL_NATIVE_TILES）、`tiles:<crop true|false>:<max_patches>`。"""
    kind, _, rest = spec.partition(":")
    try:
        if kind == "fixed":
            h, w = rest.lower().split("x")
            return FixedSize(int(h), int(w))
        if kind == "pixel_budget":
            return PixelBudget(int(rest))
        if kind == "soft_tokens":
            return SoftTokenBudget(int(rest))
        if kind == "tiles":
            if rest == "single":
                return TilePolicy(False, 1)
            if rest == "native":
                return INTERNVL_NATIVE_TILES
            crop, n = rest.split(":")
            if crop not in ("true", "false"):
                raise ValueError(crop)
            return TilePolicy(crop == "true", int(n))
    except ValueError as e:
        raise ValueError(f"认不出图像口径 {spec!r}：{e}") from None
    raise ValueError(f"认不出图像口径 {spec!r}（fixed:HxW / pixel_budget:N / soft_tokens:N / tiles:single|native|crop:N）")


__all__ = [
    "E1_IMAGE_POLICIES",
    "E1_PIXEL_BUDGET_TOKENS",
    "E1_SUBSET_PIXEL_BUDGET_TOKENS",
    "FAMILY_POLICY_TYPES",
    "FixedSize",
    "INTERNVL_NATIVE_TILES",
    "ImagePolicy",
    "PixelBudget",
    "QWEN_PIXELS_PER_TOKEN",
    "SoftTokenBudget",
    "TilePolicy",
    "check_policy",
    "parse_policy",
]
