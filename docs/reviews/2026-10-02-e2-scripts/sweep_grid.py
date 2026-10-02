"""独立核对 3：全格扫描 + 子 patch 扫描。每个 token 格 (r, c) 的每个子 patch (i, j) 单独涂白（恰好一个 patch 大小），
vision_features 变化最大的格必须是 (r, c)。原图尺寸取各家族缩放后的尺寸，使缩放是恒等变换。"""
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, ".")
from src.adapters.factory import make_adapter  # noqa: E402
from src.e2.grid import single_image, tokens_to_grid  # noqa: E402

# (key, 原图 W×H, 每格像素, patch 像素)
CASES = [
    ("qwen3.5-0.8b", (640, 480), 32, 16),
    ("qwen3.5-0.8b", (480, 640), 32, 16),
    ("gemma-4-e2b-it", (912, 672), 48, 16),
    ("gemma-4-e2b-it", (672, 912), 48, 16),
    ("internvl3.5-2b-hf", (448, 448), 28, 14),
]

ads = {}
for key, (W, H), cell, patch in CASES:
    if key not in ads:
        ads[key] = make_adapter(key, device="cpu", dtype=torch.float32, tiny=True, init_std=0.05)
    ad = ads[key]
    base = np.full((H, W, 3), 128, np.uint8)
    inp0 = ad.build_inputs([Image.fromarray(base)], "Describe this image.", "image_first")
    info = single_image(inp0)
    h, w = info.grid_hw
    extra = ""
    if "image_position_ids" in inp0.tensors:
        pos = inp0.tensors["image_position_ids"][0]
        valid = (pos >= 0).all(1)
        extra = f" gemma patch grid n_x={int(pos[valid, 0].max()) + 1} n_y={int(pos[valid, 1].max()) + 1}"
    if "image_grid_thw" in inp0.tensors:
        extra = f" qwen image_grid_thw={inp0.tensors['image_grid_thw'].tolist()}"
    assert h * cell == H and w * cell == W, (key, (h, w), (H, W), extra)
    with torch.no_grad():
        v0 = tokens_to_grid(ad.vision_features(inp0), (h, w))
    k = cell // patch
    bad, margins, t0 = [], [], time.time()
    for r in range(h):
        for c in range(w):
            for i in range(k):
                for j in range(k):
                    im = base.copy()
                    y0, x0 = r * cell + i * patch, c * cell + j * patch
                    im[y0:y0 + patch, x0:x0 + patch] = 255
                    inp = ad.build_inputs([Image.fromarray(im)], "Describe this image.", "image_first")
                    with torch.no_grad():
                        v = tokens_to_grid(ad.vision_features(inp), (h, w))
                    d = (v - v0).norm(dim=-1).flatten()
                    top2 = torch.topk(d, 2).values
                    margins.append(float(top2[0] / top2[1]))
                    got = divmod(int(d.argmax()), w)
                    if got != (r, c):
                        bad.append(((r, c), (i, j), got))
    print(f"{ad.model_key} {W}x{H} grid={h}x{w} sub={k}x{k}{extra}: {h * w * k * k} blocks, "
          f"mismatches={len(bad)} {bad[:5]}; min top1/top2 ratio={min(margins):.2f}; {time.time() - t0:.0f}s", flush=True)
