"""Gemma 子 patch 全扫描，取 pooler 之后、embed_vision（逐 token 的 RMSNorm + 线性）之前的输出。
embed_vision 是逐 token 的，不能在位置之间搬信息；它的 RMSNorm 会按各 token 自己的范数放大差值，使 ||Δ|| 的 argmax 失真。"""
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, ".")
from src.adapters.factory import make_adapter  # noqa: E402
from src.e2.grid import single_image  # noqa: E402

g = make_adapter("gemma-4-e2b-it", device="cpu", dtype=torch.float32, tiny=True, init_std=0.05)
vt = g.model.model.vision_tower
ev = g.model.model.embed_vision
print("embed_vision is per-token:", type(ev).__name__, [type(m).__name__ for m in ev.children()])

for (W, H) in [(912, 672), (672, 912)]:
    for base_kind in ("gray", "noise"):
        base = (np.full((H, W, 3), 128, np.uint8) if base_kind == "gray"
                else np.random.default_rng(7).uniform(0, 255, (H, W, 3)).astype(np.uint8))

        def pooled(arr):
            inp = g.build_inputs([Image.fromarray(arr)], "Describe this image.", "image_first")
            t = inp.tensors
            with torch.no_grad():
                out = vt(pixel_values=t["pixel_values"], pixel_position_ids=t["image_position_ids"]).last_hidden_state
                full = g.vision_features(inp)
                again = ev(out.unsqueeze(0))[0] if out.dim() == 2 else ev(out)
            return out.reshape(-1, out.shape[-1]), single_image(inp).grid_hw, bool(torch.equal(again.reshape(full.shape), full))

        p0, (h, w), consistent = pooled(base)
        bad, rel_bad, t0 = [], [], time.time()
        for r in range(h):
            for c in range(w):
                for i in range(3):
                    for j in range(3):
                        im = base.copy()
                        y0, x0 = r * 48 + i * 16, c * 48 + j * 16
                        im[y0:y0 + 16, x0:x0 + 16] = 255
                        p, _, _ = pooled(im)
                        d = (p - p0).norm(dim=-1)
                        if divmod(int(d.argmax()), w) != (r, c):
                            bad.append(((r, c), (i, j), divmod(int(d.argmax()), w)))
        print(f"gemma {W}x{H} {base_kind}: embed_vision(pooled) == vision_features: {consistent}; pooled-level mismatches "
              f"{len(bad)}/{h * w * 9} {bad[:4]}; {time.time() - t0:.0f}s", flush=True)
