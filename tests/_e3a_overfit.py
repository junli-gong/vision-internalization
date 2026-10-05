"""E3a 测试的过拟合小数据与训练流程（DEV-0034 §5；tests/test_e3a_train.py 与 test_e3a_eval.py 共用）。

数据（数值全部由固定种子生成，块对齐到 16 × 16 的查询网格，128 × 128 的输出能逐块表示）：
- refseg：图 A（96 × 128）上三个不同颜色的矩形，三句指代句各指一个（去掉指令后读出分不清指的是哪个：泄漏检查）；图 B 上一个矩形；
- semseg：两张图（96 × 128），三类（ADE 标签 1、5、13，另有 0 = 忽略）按块排列，颜色随类；
- depth：两张图（480 × 640，Eigen crop 有定义），两档深度按块排列；训练用 16 位 PNG（gt.depth），评测用 depth_npz；
- normals：两张图（480 × 640），两个方向的法向按块排列，PNG 编码同 Bae 2021（(n + 1)/2 · 255）。
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from src.adapters import registry
from src.adapters.factory import make_adapter
from src.adapters.image_policy import parse_policy
from src.data.records import Sample
from src.executors.raster import mask_to_rle
from src.readout import data as D
from src.readout.config import load_config, resolve_run
from src.readout.train import Trainer

CFG = load_config()
GRID = 16


def tiny_adapter(key: str) -> Any:
    fam = registry.get_model(key).family
    t = CFG.doc["tiny"]
    return make_adapter(key, device="cpu", dtype=torch.float32, image_policy=parse_policy(t["image_policies"][fam]), tiny=True,
                        init_std=float(t["init_std"]))


def _png(path: Path, a: np.ndarray) -> str:
    """uint8 (H, W) → 8 位灰度、(H, W, 3) → RGB；uint16 (H, W) → 16 位 PNG（BTS 的深度格式）。"""
    Image.fromarray(a).save(path)
    return str(path)


def _blocks(h: int, w: int, cells: list[tuple[int, int, int, int]]) -> np.ndarray:
    """网格格子 (r0, c0, r1, c1)（16 × 16 网格上的半开区间）→ 原图上的 bool mask。"""
    m = np.zeros((h, w), dtype=bool)
    for r0, c0, r1, c1 in cells:
        m[r0 * h // GRID: r1 * h // GRID, c0 * w // GRID: c1 * w // GRID] = True
    return m


def overfit_samples(root: Path) -> dict[str, list[Sample]]:
    root.mkdir(parents=True, exist_ok=True)
    out: dict[str, list[Sample]] = {"refseg": [], "semseg": [], "depth": [], "normals": []}
    h, w = 96, 128
    # refseg
    objs = {"A": [((2, 2, 7, 7), (220, 40, 40), "red"), ((9, 3, 14, 8), (40, 200, 60), "green"),
                  ((3, 10, 8, 15), (50, 70, 230), "blue")],
            "B": [((5, 5, 12, 12), (230, 200, 30), "yellow")]}
    for img_name, items in objs.items():
        img = np.full((h, w, 3), 90, dtype=np.uint8)
        for cells, color, _ in items:
            img[_blocks(h, w, [cells])] = color
        path = _png(root / f"ref_{img_name}.png", img)
        for k, (cells, _, word) in enumerate(items):
            m = _blocks(h, w, [cells])
            sid = f"overfit/refseg/{img_name}{k}"
            out["refseg"].append(Sample(id=sid, task="refexp", images=(path,), image_keys=(f"overfit:{img_name}",),
                                        fields={"expression": f"the {word} block", "category": "person" if k == 0 else "dog",
                                                "image_id": img_name, "ann_id": k},
                                        gt={"mask": {"rle": mask_to_rle(m), "height": h, "width": w}}, source="overfit",
                                        split="val"))
    # semseg：ADE 原始标签 1、5、13（reduce 之后 0、4、12）
    layouts = [[(1, (0, 0, 8, 16)), (5, (8, 0, 16, 8)), (13, (8, 8, 16, 16))],
               [(13, (0, 0, 16, 8)), (1, (0, 8, 8, 16)), (5, (8, 8, 16, 16))]]
    colors = {1: (200, 200, 200), 5: (30, 160, 30), 13: (160, 60, 160)}
    for j, lay in enumerate(layouts):
        lab = np.zeros((h, w), dtype=np.uint8)
        img = np.zeros((h, w, 3), dtype=np.uint8)
        for c, cells in lay:
            m = _blocks(h, w, [cells])
            lab[m] = c
            img[m] = colors[c]
        lab[:6, :8] = 0  # 一小块忽略值
        ip = _png(root / f"sem_img{j}.png", img)
        lp = _png(root / f"sem_lab{j}.png", lab)
        out["semseg"].append(Sample(id=f"overfit/semseg/{j}", task="semseg", images=(ip,), image_keys=(f"overfit:sem{j}",),
                                    fields={"stem": f"s{j}"}, gt={"label": lp}, source="overfit", split="val"))
    # depth / normals（480 × 640）
    H, W = 480, 640
    for j in range(2):
        near = _blocks(H, W, [(0, 0, 16, 8)] if j == 0 else [(0, 8, 16, 16)])
        depth = np.where(near, 1.5, 4.0).astype(np.float32)
        img = np.where(near[..., None], np.array([200, 120, 60], np.uint8), np.array([40, 60, 160], np.uint8)).astype(np.uint8)
        ip = _png(root / f"dep_img{j}.png", img)
        dp = _png(root / f"dep_png{j}.png", np.round(depth * 1000.0).astype(np.uint16))
        np.savez(root / f"dep{j}.npz", depth_raw=depth)
        out["depth"].append(Sample(id=f"overfit/depth/{j}", task="depth", images=(ip,), image_keys=(f"overfit:dep{j}",),
                                   fields={"idx": j}, gt={"depth": dp, "depth_npz": str(root / f"dep{j}.npz"), "key": "depth_raw"},
                                   source="overfit", split="test"))
        up = _blocks(H, W, [(0, 0, 8, 16)] if j == 0 else [(8, 0, 16, 16)])
        n = np.where(up[..., None], np.array([0.0, -1.0, 0.0]), np.array([0.6, 0.0, 0.8])).astype(np.float64)
        enc = np.round((n + 1.0) / 2.0 * 255.0).astype(np.uint8)
        img = np.where(up[..., None], np.array([220, 220, 220], np.uint8), np.array([70, 30, 30], np.uint8)).astype(np.uint8)
        ip = _png(root / f"nrm_img{j}.png", img)
        npth = _png(root / f"nrm_gt{j}.png", enc)
        out["normals"].append(Sample(id=f"overfit/normals/{j}", task="normal", images=(ip,), image_keys=(f"overfit:nrm{j}",),
                                     fields={"idx": j}, gt={"normal": npth}, source="overfit", split="test"))
    return out


def overfit_examples(samples: dict[str, list[Sample]], out_size: int) -> list[D.Example]:
    b = D.ExampleBuilder(CFG, samples, out_size=out_size)
    ex = []
    for task, ss in samples.items():
        for i, s in enumerate(ss):
            plan = D.SlotPlan(k=len(ex), step=0, task=task, j=i, index=i, epoch=0, sample_id=s.id)
            ex.append(b.build(plan))
    return ex


def overfit_trainer(key: str, variant: str, *, steps: int, lr_scale: float = 1.0, seed: int = 0) -> Trainer:
    ad = tiny_adapter(key)
    rc = resolve_run(CFG, variant=variant, tiny=True)
    rc = copy.deepcopy(rc)
    rc["steps"], rc["warmup_steps"], rc["micro_batch_size"] = steps, 0, 8
    rc["optim"]["min_lr_ratio"] = 1.0
    for g, v in rc["optim"]["groups"].items():
        v["lr"] = {"lora": 2e-3, "query": 1e-2, "mix": 1e-2, "readout": 3e-3}[g] * lr_scale
        v["weight_decay"] = 0.0
    if rc["lora"]:
        rc["lora"] = {**rc["lora"], "dropout": 0.0}
    tr = Trainer(ad, CFG, rc, variant=variant, seed=seed, tiny=True)
    tr.setup_fresh()
    return tr
