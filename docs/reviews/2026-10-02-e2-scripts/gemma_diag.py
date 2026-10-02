"""Gemma 子 patch 扫描失配的诊断：是映射错，还是小型随机模型的数值现象。"""
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, ".")
from src.adapters.factory import make_adapter  # noqa: E402
from src.e2.grid import single_image, tokens_to_grid  # noqa: E402

g = make_adapter("gemma-4-e2b-it", device="cpu", dtype=torch.float32, tiny=True, init_std=0.05)
vm = g.model.model.vision_tower
print("vision config standardize:", vm.config.standardize, "layers:", vm.config.num_hidden_layers)
if vm.config.standardize:
    print("std_scale stats: min %.3e max %.3e  | std_bias min %.3e max %.3e" % (
        vm.std_scale.min(), vm.std_scale.max(), vm.std_bias.min(), vm.std_bias.max()))

W, H, cell, patch = 912, 672, 48, 16


def feats(arr):
    inp = g.build_inputs([Image.fromarray(arr)], "Describe this image.", "image_first")
    h, w = single_image(inp).grid_hw
    with torch.no_grad():
        return tokens_to_grid(g.vision_features(inp), (h, w)), inp


def patch_level(arr):
    """embed_vision 之前、pooler 之前：patch embedder + encoder 的输出（逐 patch）。"""
    inp = g.build_inputs([Image.fromarray(arr)], "Describe this image.", "image_first")
    pv, pos = inp.tensors["pixel_values"], inp.tensors["image_position_ids"]
    pad = (pos == -1).all(-1)
    with torch.no_grad():
        e = vm.patch_embedder(pv, pos, pad)
        o = vm.encoder(inputs_embeds=e, attention_mask=~pad, pixel_position_ids=pos).last_hidden_state
    return e[0], o[0], pos[0]


rng = np.random.default_rng(0)
for base_kind in ("gray", "noise"):
    base = np.full((H, W, 3), 128, np.uint8) if base_kind == "gray" else rng.uniform(0, 255, (H, W, 3)).astype(np.uint8)
    v0, _ = feats(base)
    e0, o0, pos = patch_level(base)
    h, w = v0.shape[:2]
    bad, n = 0, 0
    bad_patch_emb, bad_patch_enc = 0, 0
    for (r, c) in [(0, 5), (0, 6), (0, 9), (0, 10), (0, 12), (3, 7), (7, 9), (13, 18), (5, 0), (10, 15)]:
        for i in range(3):
            for j in range(3):
                im = base.copy()
                y0, x0 = r * cell + i * patch, c * cell + j * patch
                im[y0:y0 + patch, x0:x0 + patch] = 255
                v, _ = feats(im)
                d = (v - v0).norm(dim=-1).flatten()
                got = divmod(int(d.argmax()), w)
                n += 1
                bad += got != (r, c)
                e1, o1, _ = patch_level(im)
                want_patch = (r * 3 + i, c * 3 + j)  # (y, x) in patch units
                de = (e1 - e0).norm(dim=-1)
                do = (o1 - o0).norm(dim=-1)
                pe = pos[int(de.argmax())].tolist()  # (x, y)
                po = pos[int(do.argmax())].tolist()
                bad_patch_emb += (pe[1], pe[0]) != want_patch
                bad_patch_enc += (po[1], po[0]) != want_patch
                if base_kind == "gray" and got != (r, c) and bad <= 3:
                    tgt = float(d.reshape(h, w)[r, c])
                    print(f"  gray miss: cell {(r, c)} sub {(i, j)} → argmax {got}; d[target]={tgt:.3e} d[max]={float(d.max()):.3e} "
                          f"median={float(d.median()):.3e}; patch-emb argmax ok={(pe[1], pe[0]) == want_patch}; "
                          f"encoder-out argmax ok={(po[1], po[0]) == want_patch}; encoder d[target patch]/max="
                          f"{float(do.max()):.3e}")
    print(f"{base_kind}: soft-token argmax mismatches {bad}/{n}; patch-embedder argmax mismatches {bad_patch_emb}/{n}; "
          f"encoder-output argmax mismatches {bad_patch_enc}/{n}")
