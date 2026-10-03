#!/usr/bin/env python
"""管理员用：E1 主评测（RUN-0027）8 个模型的主表（严格读数），供 G1 门报告。只读 metrics.json。"""
import json, sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else "experiments/E1/0027_e1_main")
order = ["qwen3.5-0.8b", "qwen3.5-2b", "qwen3.5-4b", "qwen3.5-9b", "internvl3.5-2b-hf", "internvl3.5-4b-hf", "gemma-4-e2b-it", "gemma-4-e4b-it"]

def m(model, leaf):
    f = root / model / leaf / "metrics.json"
    return json.loads(f.read_text()) if f.exists() else None

def col(d, name, scope="all", key="strict"):
    if not d: return None
    for c in d.get("three_columns", []):
        if c["name"] == name and c.get("scope", "all") == scope:
            return c[key]["value"] if key in ("strict", "conditional") and c.get(key) else c.get(key)
    return None

def split(d, sp, metric="acc@0.5"):
    try: return d["breakdown"]["split"][sp][metric]["value"]
    except Exception: return None

def f(x, pct=True):
    return "—" if x is None else (f"{100*x:.1f}" if pct else f"{x:.3f}")

hdr = ["模型", "定位 全划分", "定位 RefCOCO val", "定位 有效率", "多边形 cIoU", "网格 gIoU", "网格 解析失败", "计数",
       "检测 COCO F1", "检测 ODinW F1", "NYU 点对", "BLINK", "CV-Bench", "EmbSpatial"]
print("| " + " | ".join(hdr) + " |"); print("|" + "---|" * len(hdr))
for mo in order:
    box = m(mo, "refexp_box"); poly = m(mo, "refexp_poly"); grid = m(mo, "grid16"); cnt = m(mo, "count"); ovd = m(mo, "ovd")
    row = [mo, f(col(box, "acc@0.5")), f(split(box, "refcoco_unc/val")), f(col(box, "acc@0.5", key="valid_rate")),
           f(col(poly, "cIoU"), False), f(col(grid, "gIoU"), False),
           f((grid or {}).get("parse", {}).get("failure_rate")), f(col(cnt, "accuracy")),
           f(col(ovd, "F1@0.5", "coco_val2017"), False), f(col(ovd, "F1@0.5", "odinw13_mean_over_datasets"), False),
           f(col(m(mo, "rel_depth_nyu"), "accuracy")), f(col(m(mo, "rel_depth_blink"), "accuracy")),
           f(col(m(mo, "spatial_cvbench"), "accuracy")), f(col(m(mo, "spatial_embspatial"), "accuracy"))]
    print("| " + " | ".join(row) + " |")
print()
print("| 模型 | 附加对照 | 定位（对照 / 主口径同子集） | 检测 COCO（对照） |"); print("|---|---|---|---|")
for mo in order:
    for leaf in sorted((root / mo).glob("refexp_box__extra__*")):
        pol = leaf.name.split("__extra__")[1]
        e = m(mo, leaf.name); o = m(mo, f"ovd__extra__{pol}")
        print(f"| {mo} | {pol} | {f(col(e, 'acc@0.5'))} | {f(col(o, 'F1@0.5', 'coco_val2017'), False)} |")
