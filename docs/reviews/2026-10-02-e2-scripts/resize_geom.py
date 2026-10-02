"""处理器的几何：把 patch 拼回画布，与"整图直接缩放到画布尺寸"比较（不裁、不补边）。"""
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, ".")
from src.adapters.factory import make_adapter  # noqa: E402

rng = np.random.default_rng(0)


def smooth(w, h):
    yy, xx = np.mgrid[0:h, 0:w]
    a = np.stack([127 + 120 * np.sin(xx / 37.0), 127 + 120 * np.cos(yy / 23.0), 127 + 120 * np.sin((xx + yy) / 51.0)], -1)
    return Image.fromarray(a.astype(np.uint8))


# Gemma
g = make_adapter("gemma-4-e2b-it", device="cpu", dtype=torch.float32, tiny=True, init_std=0.05)
for wh in [(640, 480), (480, 640), (683, 512), (500, 375)]:
    im = smooth(*wh)
    enc = g._ip(images=[im], max_soft_tokens=280, return_tensors="pt", do_rescale=False, do_normalize=False)
    pv, pos = enc["pixel_values"][0], enc["image_position_ids"][0]
    valid = (pos >= 0).all(1)
    nx, ny = int(pos[valid, 0].max()) + 1, int(pos[valid, 1].max()) + 1
    P = 16
    canvas = np.zeros((ny * P, nx * P, 3), np.float32)
    for k in torch.nonzero(valid).flatten().tolist():
        x, y = pos[k].tolist()
        canvas[y * P:(y + 1) * P, x * P:(x + 1) * P] = pv[k].reshape(P, P, 3).numpy()  # 每个 patch 的布局：(p, p, c)?
    ref = np.asarray(im.resize((nx * P, ny * P), resample=Image.BICUBIC), np.float32)
    err = np.abs(canvas - ref)
    # 也试 (c, p, p) 布局
    canvas2 = np.zeros_like(canvas)
    for k in torch.nonzero(valid).flatten().tolist():
        x, y = pos[k].tolist()
        canvas2[y * P:(y + 1) * P, x * P:(x + 1) * P] = pv[k].reshape(3, P, P).permute(1, 2, 0).numpy()
    err2 = np.abs(canvas2 - ref)
    e = min(err.mean(), err2.mean())
    print(f"gemma {wh} → canvas {nx * P}x{ny * P} (grid {ny // 3}x{nx // 3}); mean|canvas - resize(full)| = {e:.2f} "
          f"(layout {'p,p,c' if err.mean() < err2.mean() else 'c,p,p'}; max {min(err.max(), err2.max()):.1f}); "
          f"edge cols mean err L/R = {min(err, err2, key=np.mean)[:, :P].mean():.2f}/{min(err, err2, key=np.mean)[:, -P:].mean():.2f}")

# InternVL
iv = make_adapter("internvl3.5-2b-hf", device="cpu", dtype=torch.float32, tiny=True, init_std=0.05)
for wh in [(640, 480), (683, 512)]:
    im = smooth(*wh)
    enc = iv._ip(images=[im], return_tensors="pt", do_rescale=False, do_normalize=False, **iv.processor_kwargs())
    pv = enc["pixel_values"][0].permute(1, 2, 0).numpy()
    ref = np.asarray(im.resize((448, 448), resample=Image.BICUBIC), np.float32)
    print(f"internvl {wh} → {pv.shape}; mean|pv - resize(full,448²)| = {np.abs(pv - ref).mean():.2f}, max {np.abs(pv - ref).max():.1f}")

# Qwen（非恒等缩放）
q = make_adapter("qwen3.5-0.8b", device="cpu", dtype=torch.float32, tiny=True, init_std=0.05)
for wh in [(683, 512), (500, 375), (640, 427)]:
    im = smooth(*wh)
    gh, gw = q.expected_grid((im.height, im.width))
    px = q.processor_pixels()
    enc = q._ip(images=[im], min_pixels=px, max_pixels=px, return_tensors="pt", do_rescale=False, do_normalize=False)
    pv = enc["pixel_values"]  # [gh*gw, C*T*P*P]，按 (gh/m, gw/m, m, m) 排
    m, T, P = int(q._ip.merge_size), int(q._ip.temporal_patch_size), 16
    x = pv.reshape(gh // m, gw // m, m, m, 3, T, P, P)[:, :, :, :, :, 0]  # 取第一帧
    canvas = x.permute(0, 2, 5, 1, 3, 6, 4).reshape(gh * P, gw * P, 3).numpy()
    ref = np.asarray(im.resize((gw * P, gh * P), resample=Image.BICUBIC), np.float32)
    print(f"qwen {wh} → canvas {gw * P}x{gh * P} (llm grid {gh // m}x{gw // m}); mean|canvas - resize(full)| = "
          f"{np.abs(canvas - ref).mean():.2f}, max {np.abs(canvas - ref).max():.1f}")
