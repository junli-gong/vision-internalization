"""DEV-0023：token 序列 ↔ [H_t, W_t] 网格的映射与适配器的图像 token 信息一致（三个家族、E1 冻结口径）。

- 形状：Qwen 的 merge 后网格（= smart resize 的 patch 网格 / merge）、Gemma 的 soft token 网格（= image_position_ids 推出的网格）、
  InternVL 的单 tile（16×16）；横、竖两种朝向都查（h、w 写反会在这里露出来）。
- 顺序：E2 的 [h, w, D] 第 (r, c) 格恰为序列里第 r·w + c 个视觉 token（行主序）。
- 语义：只在一个 token 格的中心画一个白块，vision_features 变化最大的格恰好是那一格（小型随机模型上峰值是中位数的几十到上万倍）。

DEV-0024（D53 审查 §3 的建议 a、b、c；审查脚本 docs/reviews/2026-10-02-e2-scripts/order_source.py、sweep_grid.py、
gemma_noenc_sweep.py 的写法）：
- (a) 按模型自己的元数据确定性地核对顺序（不靠 argmax），640×480、480×640、683×512：Qwen 的 M-RoPE（模型的 get_rope_index）
  与处理器的 patch 展平顺序；Gemma 由 image_position_ids 经模型自己的 pooler 算出的核归属；InternVL 的 pixel_shuffle 下标搬运。
- (b) Qwen 与 InternVL：全格 × 每个子 patch 的白块扫描（vision_features 上 argmax），两种朝向（缩放是恒等或整数倍）。
- (c) Gemma：旁路视觉编码器（只留 patch 嵌入 → pooler，没有跨 token 的运算），在 pooler 输出上全格 × 每个子 patch 扫描：
  恰好只有那一格变。不在 vision_features 上用 argmax（小模型随机编码器的"枢纽" token 经 embed_vision 的 RMSNorm 放大，
  有 82–180/2394 个失配，不是排布错误；审查报告 §3）。
单个测试在开发机上都不到 30 s（用时见 DEV-0024 完成总结），不标 slow、默认全跑。
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


# ---------------------------------------------------------------------------
# DEV-0024 (a)：按模型自己的元数据核对行主序
# ---------------------------------------------------------------------------

_ADS: dict[str, object] = {}
ORDER_SIZES = [(640, 480), (480, 640), (683, 512)]


def _ad(key: str):
    if key not in _ADS:
        _ADS[key] = make_adapter(key, device="cpu", dtype=torch.float32, tiny=True, init_std=0.05)
    return _ADS[key]


@pytest.mark.parametrize("wh", ORDER_SIZES)
def test_qwen_order_from_mrope_and_processor(wh):
    ad = _ad("qwen3.5-0.8b")
    inp = ad.build_inputs([_img(*wh)], "Describe this image.", "image_first")
    info = single_image(inp)
    h, w = info.grid_hw
    t = inp.tensors
    ref, _ = ad.model.model.get_rope_index(t["input_ids"], mm_token_type_ids=t["mm_token_type_ids"],
                                           image_grid_thw=t["image_grid_thw"], attention_mask=t["attention_mask"])
    seg = ref[:, 0, info.span.start:info.span.end]  # (3, n)：T、H、W
    s = seg[0, 0]
    k = torch.arange(h * w)
    assert bool((seg[0] == s).all()), "图像 token 的 T 位置应当相同"
    assert torch.equal(seg[1] - s, k // w) and torch.equal(seg[2] - s, k % w), f"{wh}：第 k 个图像 token 的 M-RoPE (H, W) ≠ (k//w, k%w)"
    # 处理器的 patch 展平顺序：像素值编码 patch 坐标的图（缩放为恒等），第 k 个 LLM token 的 m² 个连续 patch 恰好来自格 (k//w, k%w)
    m, P = int(ad._ip.merge_size), int(ad._ip.patch_size)
    Hc, Wc = h * m * P, w * m * P
    yy, xx = np.mgrid[0:Hc, 0:Wc]
    code = np.stack([(yy // P) % 256, (xx // P) % 256, ((yy // P) // 256) * 16 + (xx // P) // 256], -1).astype(np.uint8)
    enc = ad._ip(images=[Image.fromarray(code)], min_pixels=Hc * Wc, max_pixels=Hc * Wc, return_tensors="pt",
                 do_normalize=False, do_rescale=False)
    assert enc["image_grid_thw"][0].tolist() == [1, h * m, w * m]
    pv = enc["pixel_values"].reshape(-1, 3, int(ad._ip.temporal_patch_size), P, P)
    py = pv[:, 0, 0, 0, 0].round().long() + 256 * (pv[:, 2, 0, 0, 0].round().long() // 16)
    px = pv[:, 1, 0, 0, 0].round().long() + 256 * (pv[:, 2, 0, 0, 0].round().long() % 16)
    tok = torch.arange(len(py)) // (m * m)
    assert len(py) == h * w * m * m
    assert torch.equal(py // m, tok // w) and torch.equal(px // m, tok % w), f"{wh}：处理器的 patch 顺序不是每 m² 个连续 patch 一格"
    # 一格之内 m² 个 patch 各不相同（恰是 m×m 窗口）
    win = set(zip((py % m)[: m * m].tolist(), (px % m)[: m * m].tolist()))
    assert win == {(i, j) for i in range(m) for j in range(m)}


@pytest.mark.parametrize("wh", ORDER_SIZES)
def test_gemma_pooler_kernel_assignment_is_row_major(wh):
    """由 image_position_ids 经模型自己的 pooler（Gemma4VisionPooler.forward）算出每个 patch 归哪个输出 token：
    输入是单位矩阵，输出的第 j 行就是第 j 个 token 的 patch 权重（每个 1/k²）。"""
    g = _ad("gemma-4-e2b-it")
    inp = g.build_inputs([_img(*wh)], "Describe this image.", "image_first")
    h, w = single_image(inp).grid_hw
    pos = inp.tensors["image_position_ids"][0].long()  # (n_in, 2)：(x, y)，填充行为 (-1, -1)
    k = int(g._pool)
    n_in = pos.shape[0]
    pad = (pos == -1).all(dim=-1)
    pooler = g.model.model.vision_tower.pooler
    with torch.no_grad():
        out, mask = pooler(torch.eye(n_in)[None], pos[None], pad[None], output_length=n_in // (k * k))
    rows = (out[0].float() / pooler.root_hidden_size)[mask[0]]  # 去掉填充 token 之后的 [h·w, n_in]
    valid = ~pad
    assert rows.shape == (h * w, n_in) and int(valid.sum()) == h * w * k * k
    assert bool((rows[:, pad] == 0).all()) and bool(((rows > 0).sum(dim=1) == k * k).all())
    assert torch.allclose(rows.sum(dim=1), torch.ones(h * w))
    x, y = pos[valid, 0], pos[valid, 1]
    owner = rows[:, valid].argmax(dim=0)
    assert torch.equal(owner, (y // k) * w + x // k), f"{wh}：pooler 的核下标不是行主序 (y//k)·w + x//k"


@pytest.mark.parametrize("wh", ORDER_SIZES)
def test_internvl_pixel_shuffle_maps_2x2_windows_row_major(wh):
    iv = _ad("internvl3.5-2b-hf")
    inp = iv.build_inputs([_img(*wh)], "Describe this image.", "image_first")
    assert single_image(inp).grid_hw == (16, 16)
    ps = iv.model.config.vision_config.patch_size  # [ph, pw]
    ph, pw = (int(ps[0]), int(ps[1])) if isinstance(ps, (list, tuple)) else (int(ps), int(ps))
    pv = inp.tensors["pixel_values"]
    n = int(pv.shape[-1]) // pw  # 视觉塔的 patch 网格（单 tile：448 / 14 = 32）
    assert int(pv.shape[-2]) // ph == n
    ratio = float(iv.model.config.downsample_ratio)
    s = int(round(1 / ratio))
    yy, xx = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
    f = torch.stack([yy, xx], -1).float().reshape(1, n, n, 2)  # 与 get_image_features 相同：行主序的 patch 序列 reshape 成网格
    out = iv.model.model.pixel_shuffle(f, scale_factor=ratio)
    out = out.reshape(1, -1, out.shape[-1])[0]  # [n_tok, s²·2]，每个 patch 两个通道 (y, x)
    n_tok = (n // s) ** 2
    assert out.shape == (n_tok, s * s * 2) and n_tok == 256
    for kk in range(n_tok):
        r, c = divmod(kk, n // s)
        got = {(int(a), int(b)) for a, b in zip(out[kk, 0::2].tolist(), out[kk, 1::2].tolist())}
        assert got == {(s * r + i, s * c + j) for i in range(s) for j in range(s)}, f"第 {kk} 个 token 的 patch 不是格 ({r}, {c}) 的窗口"


# ---------------------------------------------------------------------------
# DEV-0024 (b)：Qwen 与 InternVL 的全格 × 子 patch 白块扫描（vision_features 上 argmax）
# ---------------------------------------------------------------------------

SWEEPS = [("qwen3.5-0.8b", (640, 480)), ("qwen3.5-0.8b", (480, 640)),  # 640×480 = 300 个 token 的像素预算：缩放恒等
          ("internvl3.5-2b-hf", (448, 448)), ("internvl3.5-2b-hf", (896, 448)), ("internvl3.5-2b-hf", (448, 896))]


def _sub(ad) -> int:
    """每格每边几个子 patch：Qwen 的 merge_size；InternVL 的 1 / downsample_ratio。"""
    if "qwen" in ad.model_key:
        return int(ad._ip.merge_size)
    return int(round(1 / float(ad.model.config.downsample_ratio)))


@pytest.mark.parametrize("key,wh", SWEEPS)
def test_full_grid_subpatch_sweep(key, wh):
    ad = _ad(key)
    W, H = wh
    base = np.full((H, W, 3), 128, np.uint8)
    inp0 = ad.build_inputs([Image.fromarray(base)], "Describe this image.", "image_first")
    h, w = single_image(inp0).grid_hw
    m = _sub(ad)
    assert H % (h * m) == 0 and W % (w * m) == 0, "子 patch 要落在整数像素上"
    ph, pw = H // (h * m), W // (w * m)
    with torch.no_grad():
        v0 = tokens_to_grid(ad.vision_features(inp0), (h, w))
    bad = []
    for r in range(h):
        for c in range(w):
            for i in range(m):
                for j in range(m):
                    img = base.copy()
                    y0, x0 = (r * m + i) * ph, (c * m + j) * pw
                    img[y0:y0 + ph, x0:x0 + pw] = 255
                    inp = ad.build_inputs([Image.fromarray(img)], "Describe this image.", "image_first")
                    with torch.no_grad():
                        v = tokens_to_grid(ad.vision_features(inp), (h, w))
                    got = divmod(int((v - v0).norm(dim=-1).argmax()), w)
                    if got != (r, c):
                        bad.append(((r, c), (i, j), got))
    assert not bad, f"{key} {wh}：{len(bad)}/{h * w * m * m} 个子 patch 的白块没有落在自己的格：{bad[:5]}"


# ---------------------------------------------------------------------------
# DEV-0024 (c)：Gemma 旁路编码器，在 pooler 输出上扫描
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("wh", [(912, 672), (672, 912)])
def test_gemma_pooler_sweep_with_encoder_bypassed(wh):
    from transformers.modeling_outputs import BaseModelOutput

    g = _ad("gemma-4-e2b-it")
    vt = g.model.model.vision_tower
    k = int(g._pool)
    W, H = wh
    base = np.full((H, W, 3), 128, np.uint8)
    vt.encoder.forward = lambda inputs_embeds=None, **kw: BaseModelOutput(last_hidden_state=inputs_embeds)
    try:
        def pooled(arr):
            inp = g.build_inputs([Image.fromarray(arr)], "Describe this image.", "image_first")
            t = inp.tensors
            with torch.no_grad():
                out = vt(pixel_values=t["pixel_values"], pixel_position_ids=t["image_position_ids"]).last_hidden_state
            return out.reshape(-1, out.shape[-1]), single_image(inp).grid_hw

        p0, (h, w) = pooled(base)
        P = 16  # Gemma 的 patch（48 = 16 × 3 的格；912×672 正好 19×14 格，缩放恒等）
        assert h * k * P == H and w * k * P == W and p0.shape[0] == h * w
        bad = []
        for r in range(h):
            for c in range(w):
                for i in range(k):
                    for j in range(k):
                        img = base.copy()
                        y0, x0 = (r * k + i) * P, (c * k + j) * P
                        img[y0:y0 + P, x0:x0 + P] = 255
                        p, _ = pooled(img)
                        changed = torch.nonzero((p - p0).norm(dim=-1)).flatten().tolist()
                        if changed != [r * w + c]:
                            bad.append(((r, c), (i, j), changed[:3]))
        assert not bad, f"gemma {wh}：{len(bad)}/{h * w * k * k} 个子 patch 改动的不恰是自己那一格：{bad[:5]}"
    finally:
        del vt.encoder.forward


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
