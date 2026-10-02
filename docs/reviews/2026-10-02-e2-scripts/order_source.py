"""按模型自己的元数据核对行主序：Qwen 的 M-RoPE (H, W)；Gemma pooler 的核下标；InternVL pixel_shuffle 的下标搬运；
Qwen 处理器的 patch 展平顺序（每 merge² 个连续 patch 是同一个 2×2 窗口）。"""
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, ".")
from src.adapters.factory import make_adapter  # noqa: E402
from src.adapters import gemma4  # noqa: E402

# --- Qwen：M-RoPE 位置 ---
ad = make_adapter("qwen3.5-0.8b", device="cpu", dtype=torch.float32, tiny=True, init_std=0.05)
for wh in [(640, 480), (480, 640), (683, 512)]:
    im = Image.fromarray(np.random.default_rng(0).uniform(0, 255, size=(wh[1], wh[0], 3)).astype(np.uint8))
    inp = ad.build_inputs([im], "Describe this image.", "image_first")
    info = inp.images[0]
    h, w = info.grid_hw
    pos = ad.position_ids(inp)[1:, 0]  # (3, L): T, H, W
    s = info.span.start
    seg = pos[:, info.span.start:info.span.end]
    t0 = int(seg[0, 0])
    rr = (seg[1] - t0).reshape(h, w)
    cc = (seg[2] - t0).reshape(h, w)
    ok = torch.equal(rr, torch.arange(h)[:, None].expand(h, w)) and torch.equal(cc, torch.arange(w)[None].expand(h, w))
    print(f"qwen {wh}: grid={h}x{w}; M-RoPE (H-s, W-s) of token k == (k//w, k%w): {ok}; T const={bool((seg[0] == t0).all())}")
    # 处理器 patch 展平顺序：用像素坐标编码图像，检查第 k 个 LLM token 的 merge² 个 patch 恰好来自格 (k//w, k%w)
    H, W = h * 32, w * 32
    yy, xx = np.mgrid[0:H, 0:W]
    code = np.stack([(yy // 16) % 256, (xx // 16) % 256, ((yy // 16) // 256) * 16 + (xx // 16) // 256], -1).astype(np.uint8)
    enc = ad._ip(images=[Image.fromarray(code)], min_pixels=H * W, max_pixels=H * W, return_tensors="pt", do_normalize=False,
                 do_rescale=False)
    pv = enc["pixel_values"]  # [n_patches, C*T*16*16]
    gt = enc["image_grid_thw"][0].tolist()
    C, T, P = 3, int(ad._ip.temporal_patch_size), 16
    pv = pv.reshape(-1, C, T, P, P)
    py = pv[:, 0, 0, 0, 0].round().long() + 256 * (pv[:, 2, 0, 0, 0].round().long() // 16)
    px = pv[:, 1, 0, 0, 0].round().long() + 256 * (pv[:, 2, 0, 0, 0].round().long() % 16)
    m = int(ad._ip.merge_size)
    tok = torch.arange(len(py)) // (m * m)
    ok2 = torch.equal(py // m, tok // w) and torch.equal(px // m, tok % w)
    print(f"   processor grid_thw={gt}; 每 {m * m} 个连续 patch 属于同一格 (k//w, k%w): {ok2}; "
          f"窗口内顺序 (dy,dx)={list(zip((py % m)[:4].tolist(), (px % m)[:4].tolist()))}")

# --- Gemma：pooler 的核下标 ---
g = make_adapter("gemma-4-e2b-it", device="cpu", dtype=torch.float32, tiny=True, init_std=0.05)
for wh in [(640, 480), (480, 640), (683, 512)]:
    im = Image.fromarray(np.random.default_rng(0).uniform(0, 255, size=(wh[1], wh[0], 3)).astype(np.uint8))
    inp = g.build_inputs([im], "Describe this image.", "image_first")
    h, w = inp.images[0].grid_hw
    pos = inp.tensors["image_position_ids"][0]
    k = g._pool
    clamped = pos.clamp(min=0)
    max_x = clamped[:, 0].max() + 1
    kidx = torch.div(clamped, k, rounding_mode="floor")
    kidx = kidx[:, 0] + (max_x // k) * kidx[:, 1]
    valid = (pos >= 0).all(1)
    # 输出 token j 的来源 patch 的 (y//k, x//k) 必须恰是 (j//w, j%w)
    ys, xs = pos[valid, 1] // k, pos[valid, 0] // k
    ok = torch.equal(kidx[valid], ys * w + xs) and int(kidx[valid].max()) == h * w - 1
    print(f"gemma {wh}: grid={h}x{w}; pooler 核下标 = (y//3)*w + x//3（行主序）: {ok}; 有效 patch {int(valid.sum())} = {h * w * 9}")

# --- InternVL：pixel_shuffle 的下标搬运 ---
iv = make_adapter("internvl3.5-2b-hf", device="cpu", dtype=torch.float32, tiny=True, init_std=0.05)
n = 32
yy, xx = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
feat = torch.stack([yy, xx], -1).float().reshape(1, n * n, 2)  # 行主序的 patch 序列，通道 = (y, x)
f = feat.reshape(1, n, n, 2)
out = iv.model.model.pixel_shuffle(f, scale_factor=0.5).reshape(1, -1, 8)[0]  # [256, 4*2]
ys, xs = out[:, 0::2], out[:, 1::2]
k = torch.arange(256)
ok = torch.equal(ys.min(1).values.long() // 2, k // 16) and torch.equal(xs.min(1).values.long() // 2, k % 16) \
    and bool(((ys.max(1).values - ys.min(1).values) == 1).all()) and bool(((xs.max(1).values - xs.min(1).values) == 1).all())
print(f"internvl pixel_shuffle: 输出 token k 的 4 个 patch 恰是格 (k//16, k%16) 的 2×2 窗口: {ok}; "
      f"token0 的 (y,x)={list(zip(ys[0].long().tolist(), xs[0].long().tolist()))}")
