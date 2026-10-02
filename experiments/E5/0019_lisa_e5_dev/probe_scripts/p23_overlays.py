#!/usr/bin/env python
"""RUN-0019 §3.2：每个任务取外部预测目录里前 2 条成功（没有 failure）的样本，画"原图 | mask 叠加图"并在下方写 text，存 PNG。
mask 的解码按 src/e5/external.py 的格式说明（未压缩 RLE，COCO 列优先）在这里另写一份（只读预测目录，不调用评分）；叠加与 LISA
chat.py 相同：mask 内 0.5·原图 + 0.5·红。只用于目检。usage: python p23_overlays.py <pred_dir> <inputs_dir> <out_dir> [k=2]
"""
import json, os, sys
import numpy as np
from PIL import Image, ImageDraw

PRED, INP, OUT = sys.argv[1:4]
K = int(sys.argv[4]) if len(sys.argv) > 4 else 2
os.makedirs(OUT, exist_ok=True)


def rle_decode(rle):
    h, w = rle["size"]; flat = np.zeros(h * w, dtype=bool); pos, val = 0, False
    for c in rle["counts"]:
        flat[pos:pos + c] = val; pos += c; val = not val
    return flat.reshape((w, h)).T  # 列优先


index = []
for task in ("rotation3d", "viewpoint", "occlusion", "physics"):
    p = os.path.join(PRED, f"{task}.jsonl")
    if not os.path.exists(p):
        continue
    inputs = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(INP, task, "inputs.jsonl"))}
    ok = [json.loads(l) for l in open(p) if "failure" not in json.loads(l)][:K]
    for r in ok:
        inp = inputs[r["id"]]
        img = np.array(Image.open(os.path.join(INP, task, inp["image"])).convert("RGB"))
        m = rle_decode(r["mask_rle"])
        ov = img.copy(); ov[m] = (img[m] * 0.5 + np.array([255, 0, 0]) * 0.5).astype(np.uint8)
        h, w = img.shape[:2]; cap = 70
        canvas = Image.new("RGB", (2 * w + 10, h + cap), "white")
        canvas.paste(Image.fromarray(img), (0, 0)); canvas.paste(Image.fromarray(ov), (w + 10, 0))
        d = ImageDraw.Draw(canvas)
        d.text((4, h + 4), f"{r['id']}  n_mask_pixels={r['n_mask_pixels']}  n_seg={r['n_seg']}", fill="black")
        d.text((4, h + 22), "text: " + r["text"][:120], fill="black")
        d.text((4, h + 40), "instr: " + inp["instruction"][:150], fill="black")
        name = r["id"].replace("/", "_") + ".png"
        canvas.save(os.path.join(OUT, name))
        index.append({"file": name, "id": r["id"], "task": task, "instruction": inp["instruction"], "target_description": inp.get("target_description"),
                      "text": r["text"], "n_mask_pixels": r["n_mask_pixels"], "n_seg": r["n_seg"]})
json.dump(index, open(os.path.join(OUT, "index.json"), "w"), indent=1, ensure_ascii=False)
print(len(index), "overlays ->", OUT)
