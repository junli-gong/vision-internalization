"""DEV-0027：pre_projection_features（D59-8，E2 的 pre_proj 参考线）。三家族，CPU 小模型（fp32）。

- 顺序用元数据核对（参照 docs/reviews/2026-10-02-e2-scripts/order_source.py）：
  - Qwen：处理器的 patch 展平顺序（用坐标编码的图）与视觉 RoPE 的 (行, 列)；
  - Gemma：image_position_ids 的 (x, y)；
  - InternVL：pixel_shuffle 的下标搬运（下面的"投影之后"核对里就用它）。
- 与"投影之后"的关系：对它做本家族的合并 / 投影，得到 vision_features。Qwen（merger）与 InternVL（pixel_shuffle +
  multi_modal_projector）逐位相同；Gemma 的 pooler 用 one-hot 矩阵乘求平均，与 reshape 求平均的求和顺序不同，按 fp32 容差比。
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from src.adapters import qwen35
from src.adapters.factory import make_adapter
from src.adapters.image_policy import PixelBudget, SoftTokenBudget, TilePolicy

INIT_STD = 0.05


def _img(seed, h, w):
    return Image.fromarray(np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8))


@pytest.fixture(scope="module")
def qwen():
    return make_adapter("qwen3.5-0.8b", device="cpu", dtype=torch.float32, image_policy=PixelBudget(64), tiny=True,
                        init_std=INIT_STD)


@pytest.fixture(scope="module")
def gemma():
    return make_adapter("gemma-4-e2b-it", device="cpu", dtype=torch.float32, image_policy=SoftTokenBudget(70), tiny=True,
                        init_std=INIT_STD)


@pytest.fixture(scope="module")
def internvl():
    return make_adapter("internvl3.5-2b-hf", device="cpu", dtype=torch.float32, image_policy=TilePolicy(False, 1), tiny=True,
                        init_std=INIT_STD)


SIZES = [(100, 150), (150, 100), (120, 120), (64, 256)]


# ---------------------------------------------------------------------------
# Qwen
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("hw", SIZES)
def test_qwen_shape_and_merger_gives_vision_features(qwen, hw):
    inp = qwen.build_inputs([_img(1, *hw)], "Describe.", "image_first")
    with torch.no_grad():
        pre, (gh, gw) = qwen.pre_projection_features(inp)
        vf = qwen.vision_features(inp)
        visual = qwen.model.model.visual
        m = int(visual.spatial_merge_size)
        assert (gh // m, gw // m) == inp.images[0].grid_hw and (gh, gw) == tuple(int(v) for v in inp.tensors["image_grid_thw"][0, 1:])
        assert tuple(pre.shape) == (gh * gw, visual.config.hidden_size) and pre.device == qwen.device
        order = qwen35.vision_patch_row_major_order(inp.tensors["image_grid_thw"], m)
        seq = torch.empty_like(pre)
        seq[order] = pre  # 逆排回视觉塔里的窗口序
        merged = visual.merger(seq)
    assert torch.equal(merged, vf)


@pytest.mark.parametrize("hw", [(256, 256), (192, 320)])
def test_qwen_order_from_processor_patches_and_vision_rope(qwen, hw):
    """坐标编码的图（每个 16 px 的 patch 填它的 (行, 列)）：处理器给出的 patch 序列里第 k 个 patch 的坐标，按
    vision_patch_row_major_order 重排之后恰是行主序；视觉 RoPE 的 (行, 列) 也同样重排成行主序。图的尺寸取像素预算下不缩放的尺寸。"""
    H, W = hw
    yy, xx = np.mgrid[0:H, 0:W]
    code = np.stack([(yy // 16) % 256, (xx // 16) % 256, np.zeros_like(yy)], -1).astype(np.uint8)
    px = H * W
    enc = qwen._ip(images=[Image.fromarray(code)], min_pixels=px, max_pixels=px, return_tensors="pt", do_normalize=False,
                   do_rescale=False)
    grid = enc["image_grid_thw"]
    gh, gw = int(grid[0, 1]), int(grid[0, 2])
    assert (gh, gw) == (H // 16, W // 16), "处理器缩放了图，坐标编码失效"
    pv = enc["pixel_values"].reshape(gh * gw, 3, -1)
    py, pxx = pv[:, 0, 0].round().long(), pv[:, 1, 0].round().long()
    order = qwen35.vision_patch_row_major_order(grid, int(qwen.image_processor.merge_size))
    want_r = torch.arange(gh).repeat_interleave(gw)
    want_c = torch.arange(gw).repeat(gh)
    assert torch.equal(py[order], want_r) and torch.equal(pxx[order], want_c)
    assert not torch.equal(order, torch.arange(gh * gw)), "窗口序与行主序相同，核对没有意义"
    from transformers.vision_utils import get_vision_position_ids

    pos = get_vision_position_ids(grid, int(qwen.image_processor.merge_size))
    assert torch.equal(pos[order, 0], want_r) and torch.equal(pos[order, 1], want_c)


def test_qwen_locality_white_block(qwen):
    """把 patch (5, 9) 的像素改成白块，pre_proj 网格上变化最大的恰是这一格（1 层视觉塔，残差保留局部）。"""
    base = np.asarray(_img(3, 256, 256)).copy()
    alt = base.copy()
    alt[5 * 16: 6 * 16, 9 * 16: 10 * 16] = 255
    with torch.no_grad():
        a, (gh, gw) = qwen.pre_projection_features(qwen.build_inputs([Image.fromarray(base)], "x", "image_first"))
        b, _ = qwen.pre_projection_features(qwen.build_inputs([Image.fromarray(alt)], "x", "image_first"))
    d = (a - b).norm(dim=-1).reshape(gh, gw)
    assert (gh, gw) == (16, 16) and divmod(int(d.argmax()), gw) == (5, 9)


def test_qwen_rejects_not_single_image(qwen):
    two = qwen.build_inputs([_img(1, 100, 150), _img(2, 120, 120)], "x", "image_first")
    none = qwen.build_inputs([], "x", "image_first")
    for inp in (two, none):
        with pytest.raises(ValueError, match="单图"):
            qwen.pre_projection_features(inp)


# ---------------------------------------------------------------------------
# Gemma
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("hw", SIZES)
def test_gemma_shape_order_and_pooling_gives_vision_features(gemma, hw):
    inp = gemma.build_inputs([_img(1, *hw)], "Describe.", "image_first")
    h, w = inp.images[0].grid_hw
    k = gemma._pool
    with torch.no_grad():
        pre, (ny, nx) = gemma.pre_projection_features(inp)
        vf = gemma.vision_features(inp)
        vt = gemma.model.model.vision_tower
        d_v = vt.config.hidden_size
        pos = inp.tensors["image_position_ids"][0]
        n_valid = int((~(pos == -1).all(-1)).sum())
        assert (ny, nx) == (k * h, k * w) and tuple(pre.shape) == (ny * nx, d_v) == (n_valid, d_v)
        assert n_valid < pos.shape[0], "这张图没有填充 patch，去填充的核对没有意义"
        # 元数据：有效 patch 的 (x, y) 按行主序排（x 变化最快）恰是 ny × nx 网格
        xy = pos[~(pos == -1).all(-1)]
        key = xy[:, 1] * nx + xy[:, 0]
        assert torch.equal(torch.sort(key).values, torch.arange(ny * nx))
        # 投影之后：k × k 平均 → × sqrt(D_v) → 标准化 → embed_vision
        x = pre.reshape(h, k, w, k, d_v).float().mean(dim=(1, 3)).reshape(h * w, d_v) * d_v ** 0.5
        if vt.config.standardize:
            x = (x - vt.std_bias.float()) * vt.std_scale.float()
        out = gemma.model.model.embed_vision(inputs_embeds=x.to(pre.dtype))
    assert out.shape == vf.shape and torch.allclose(out, vf, rtol=1e-4, atol=1e-5)
    # 反例：把两个属于不同池化块的 patch 对调，平均出来就对不上（说明上面的核对能查出顺序错误）
    bad = pre.clone()
    bad[[0, k]] = pre[[k, 0]]
    xb = bad.reshape(h, k, w, k, d_v).float().mean(dim=(1, 3)).reshape(h * w, d_v) * d_v ** 0.5
    assert not torch.allclose(xb, pre.reshape(h, k, w, k, d_v).float().mean(dim=(1, 3)).reshape(h * w, d_v) * d_v ** 0.5)


def test_gemma_rejects_not_single_image(gemma):
    two = gemma.build_inputs([_img(1, 100, 150), _img(2, 120, 120)], "x", "image_first")
    with pytest.raises(ValueError, match="单图"):
        gemma.pre_projection_features(two)


# ---------------------------------------------------------------------------
# InternVL
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("hw", SIZES[:2])
def test_internvl_shape_and_pixel_shuffle_projector_gives_vision_features(internvl, hw):
    inp = internvl.build_inputs([_img(1, *hw)], "Describe.", "image_first")
    inner = internvl.model.model
    with torch.no_grad():
        pre, (n, n2) = internvl.pre_projection_features(inp)
        vf = internvl.vision_features(inp)
        d_v = inner.vision_tower.config.hidden_size
        assert n == n2 == 32 and tuple(pre.shape) == (n * n, d_v) and inp.images[0].grid_hw == (16, 16)
        x = inner.pixel_shuffle(pre.reshape(1, n, n, d_v), scale_factor=internvl.model.config.downsample_ratio)
        out = inner.multi_modal_projector(x.reshape(1, -1, x.shape[-1]))[0]
    assert torch.equal(out, vf)


def test_internvl_rejects_multi_tile_and_multi_image():
    ad = make_adapter("internvl3.5-2b-hf", device="cpu", dtype=torch.float32, image_policy=TilePolicy(True, 4), tiny=True,
                      init_std=INIT_STD)
    multi = ad.build_inputs([_img(1, 100, 300)], "x", "image_first")
    assert multi.images[0].grid_hw != (16, 16)
    with pytest.raises(ValueError, match="tile"):
        ad.pre_projection_features(multi)


def test_internvl_rejects_two_images(internvl):
    two = internvl.build_inputs([_img(1, 100, 150), _img(2, 120, 120)], "x", "image_first")
    with pytest.raises(ValueError, match="单图"):
        internvl.pre_projection_features(two)
