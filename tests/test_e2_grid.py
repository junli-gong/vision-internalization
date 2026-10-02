"""DEV-0023：token 序列 ↔ [H_t, W_t] 网格的映射与适配器的图像 token 信息一致（三个家族、E1 冻结口径）。

- 形状：Qwen 的 merge 后网格（= smart resize 的 patch 网格 / merge）、Gemma 的 soft token 网格（= image_position_ids 推出的网格）、
  InternVL 的单 tile（16×16）；横、竖两种朝向都查（h、w 写反会在这里露出来）。
- 顺序：E2 的 [h, w, D] 第 (r, c) 格恰为序列里第 r·w + c 个视觉 token（行主序）。
- 语义：只在一个 token 格的中心画一个白块，vision_features 变化最大的格恰好是那一格（小型随机模型上峰值是中位数的几十到上万倍）。
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from src.adapters import gemma4
from src.adapters.base import ImageTokenInfo, ModelInputs, Span
from src.adapters.factory import make_adapter
from src.e2.config import load_config
from src.e2.features import FeatureExtractor
from src.e2.grid import grid_to_chw, single_image, tokens_to_grid

FAMILIES = ("qwen3.5-0.8b", "gemma-4-e2b-it", "internvl3.5-2b-hf")
# E1 冻结口径（D28-3）下 640×480（宽×高）的网格 (h, w)；竖图 480×640 反过来。完成总结里的数字以这里为准。
E1_GRID_640x480 = {"qwen3.5-0.8b": (15, 20), "gemma-4-e2b-it": (14, 19), "internvl3.5-2b-hf": (16, 16)}


@pytest.fixture(scope="module", params=FAMILIES)
def adapter(request):
    return make_adapter(request.param, device="cpu", dtype=torch.float32, tiny=True, init_std=0.05)


def _img(w: int, h: int, seed: int = 0) -> Image.Image:
    return Image.fromarray((np.random.default_rng(seed).uniform(0, 255, size=(h, w, 3))).astype(np.uint8))


def _family_grid(ad, inputs, img: Image.Image) -> tuple[int, int]:
    """各家族自己的网格公式（不经过 ImageTokenInfo）。"""
    key = ad.model_key
    if "qwen" in key:
        merge = int(ad._ip.merge_size)
        gh, gw = ad.expected_grid((img.height, img.width))
        return gh // merge, gw // merge
    if "gemma" in key:
        return gemma4.grid_from_position_ids(inputs.tensors["image_position_ids"][0], ad._pool)
    return (16, 16)


@pytest.mark.parametrize("wh", [(640, 480), (480, 640), (683, 512), (300, 170)])
def test_grid_shape_matches_family_formula(adapter, wh):
    img = _img(*wh)
    inp = adapter.build_inputs([img], "Describe this image.", "image_first")
    info = single_image(inp)
    assert info.grid_hw == _family_grid(adapter, inp, img)
    assert len(info.span) == info.grid_hw[0] * info.grid_hw[1]
    if wh == (640, 480):
        assert info.grid_hw == E1_GRID_640x480[adapter.model_key.removeprefix("tiny(").removesuffix(")")]
    if wh == (480, 640) and "internvl" not in adapter.model_key:
        h, w = E1_GRID_640x480[adapter.model_key.removeprefix("tiny(").removesuffix(")")]
        assert info.grid_hw == (w, h)


def test_extractor_grid_is_row_major_view_of_hidden_states(adapter, monkeypatch):
    import src.e2.features as feats

    cfg = load_config()
    layers = list(range(adapter.num_layers() + 1))
    ex = FeatureExtractor(adapter, cfg, "depth", layers, [])
    img = _img(683, 512, seed=1)

    class S:  # 最小的样本替身：FeatureExtractor 只读 images 与 id
        id = "x"
        images = ("unused",)
        fields: dict = {}

    monkeypatch.setattr(feats, "load_image", lambda s: img)
    raw, inp = ex.raw(S(), phase="test")
    with torch.no_grad():
        hs = adapter.hidden_states(inp, layers)
    h, w = inp.images[0].grid_hw
    for l in layers:
        g = raw[l]
        assert tuple(g.shape) == (h, w, adapter.hidden_size())
        for r, c in [(0, 0), (0, w - 1), (h - 1, 0), (h // 2, w // 3), (h - 1, w - 1)]:
            assert torch.equal(g[r, c], hs.image[l][r * w + c])
        assert torch.equal(grid_to_chw(g)[:, 1, 2], g[1, 2])


@pytest.mark.parametrize("wh", [(640, 480), (480, 640)])
def test_token_cell_sees_its_own_image_region(adapter, wh):
    W, H = wh
    base = np.full((H, W, 3), 128, np.uint8)
    inp0 = adapter.build_inputs([Image.fromarray(base)], "Describe this image.", "image_first")
    h, w = single_image(inp0).grid_hw
    with torch.no_grad():
        v0 = tokens_to_grid(adapter.vision_features(inp0), (h, w))
    for r, c in [(0, 0), (h - 1, w - 1), (1, w - 2), (h - 2, 1), (h // 2, w // 2)]:
        img = base.copy()
        y0, y1 = r * H / h, (r + 1) * H / h  # 整图缩放进视觉编码器：第 (r, c) 格对应原图的这一块
        x0, x1 = c * W / w, (c + 1) * W / w
        cy, cx, ry, rx = (y0 + y1) / 2, (x0 + x1) / 2, (y1 - y0) / 4, (x1 - x0) / 4
        img[int(cy - ry):int(cy + ry), int(cx - rx):int(cx + rx)] = 255
        inp = adapter.build_inputs([Image.fromarray(img)], "Describe this image.", "image_first")
        with torch.no_grad():
            v = tokens_to_grid(adapter.vision_features(inp), (h, w))
        d = (v - v0).norm(dim=-1)
        assert divmod(int(d.argmax()), w) == (r, c), f"{adapter.model_key} {wh}：白块在第 ({r}, {c}) 格"


def test_single_image_guards():
    t = torch.zeros(1, 4)
    one = ImageTokenInfo(Span(1, 7), (2, 3), n_pixels=6)
    ok = ModelInputs(tensors={"input_ids": t}, order="image_first", images=(one,), instruction_span=Span(7, 8))
    assert single_image(ok) is one
    two = ModelInputs(tensors={"input_ids": t}, order="image_first", images=(one, one), instruction_span=Span(7, 8))
    with pytest.raises(ValueError, match="恰好一张图"):
        single_image(two)
    sep = ModelInputs(tensors={"input_ids": t}, order="image_first",
                      images=(ImageTokenInfo(Span(1, 8), (2, 3), n_pixels=6),), instruction_span=Span(8, 9))
    with pytest.raises(ValueError, match="非网格 token"):
        single_image(sep)
    x = torch.arange(12.0).reshape(6, 2)
    g = tokens_to_grid(x, (2, 3))
    assert g[1, 0].tolist() == x[3].tolist() and g[0, 2].tolist() == x[2].tolist()
    with pytest.raises(ValueError):
        tokens_to_grid(x, (3, 3))
