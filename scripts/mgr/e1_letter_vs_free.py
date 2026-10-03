#!/usr/bin/env python
"""管理员用：E1 主评测里，受限字母读数与 free 自由作答在同一批样本（思考子集 id）上的配对对照，以及字母读数的位置偏好。"""
import json, gzip, collections
from pathlib import Path
root = Path("experiments/E1/0027_e1_main")
order = ["qwen3.5-0.8b","qwen3.5-2b","qwen3.5-4b","qwen3.5-9b","internvl3.5-2b-hf","internvl3.5-4b-hf","gemma-4-e2b-it","gemma-4-e4b-it"]
def load(mo, leaf):
    f = root / mo / leaf / "predictions.jsonl"
    return [json.loads(l) for l in (f.open() if f.exists() else gzip.open(str(f) + ".gz", "rt"))]
print("| 模型 | 任务 | n | 字母读数 | 自由作答（严格） | 自由作答有效率 | 只有字母对 | 只有自由对 | 字母读数最常见的字母占比 |")
print("|---|---|---|---|---|---|---|---|---|")
for mo in order:
    for t in ("rel_depth_nyu","rel_depth_blink","spatial_cvbench","spatial_embspatial"):
        ev = {r["id"]: r for r in load(mo, t)}; fr = {r["id"]: r for r in load(mo, t + "__free")}
        ids = [i for i in fr if i in ev]
        a = [bool(ev[i]["score"]["correct"]) for i in ids]; b = [bool((fr[i].get("score") or {}).get("correct")) for i in ids]
        valid = sum(fr[i]["failure"] is None for i in ids) / len(ids)
        c = collections.Counter(r["parsed"]["letter"] for r in ev.values()); top, n = c.most_common(1)[0]
        print(f"| {mo} | {t} | {len(ids)} | {sum(a)/len(ids):.3f} | {sum(b)/len(ids):.3f} | {valid:.2f} | "
              f"{sum(x and not y for x,y in zip(a,b))} | {sum(y and not x for x,y in zip(a,b))} | {top} {n/len(ev):.2f} |")
