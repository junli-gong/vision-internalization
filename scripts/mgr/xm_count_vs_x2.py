"""X-M 的计数（CountBench 442 题）对 X-2 三个种子的逐题配对差，以及错误的方向（多数、少数）。管理员的分析脚本，只读 T 文本预测，不跑模型。
都在 06 上：X-M（RUN-0114）、X-2 seed 0（RUN-0091）、seed 1、2（RUN-0096），基座 RUN-0069。配对差的算法同 scripts/mgr/lang_cost_table.ttext_pair（10,000 次）。
只记现象，不作解释（D154、D162）。用法：python scripts/mgr/xm_count_vs_x2.py
"""
import importlib.util, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("lct", ROOT / "scripts/mgr/lang_cost_table.py"); LCT = importlib.util.module_from_spec(spec); spec.loader.exec_module(LCT)
XM = ROOT / "experiments/XM/0114_xm_ttext_hpcgpu06/t_text/xm/seed0/qwen3.5-9b/count"
BASE = ROOT / "experiments/W2/0069_w2c_hpcgpu06/0069_ttext/qwen3.5-9b/count"
X2 = {"seed0": "experiments/X/0091_x_ttext_hpcgpu06/t_text/x2/seed0/qwen3.5-9b/count",
      "seed1": "experiments/X/0096_lang_seeds_hpcgpu06/t_text/x2s1/seed1/qwen3.5-9b/count",
      "seed2": "experiments/X/0096_lang_seeds_hpcgpu06/t_text/x2s2/seed2/qwen3.5-9b/count"}
for k, p in X2.items():
    r = LCT.ttext_pair(XM, ROOT / p)
    print(f"X-M − X-2 {k}: {r['sys']:.1f} vs {r['base']:.1f}: {r['delta_pp']:+.1f} [{r['ci95_pp'][0]:+.1f}, {r['ci95_pp'][1]:+.1f}]")
def load(p):
    return {r["id"]: r for r in map(json.loads, (Path(p) / "predictions.jsonl").read_text().splitlines())}
def direction(rows):
    over = under = fail = ok = 0
    for r in rows.values():
        c = (r.get("parsed") or {}).get("count"); g = r["score"]["gt"]
        if c is None: fail += 1
        elif c > g: over += 1
        elif c < g: under += 1
        else: ok += 1
    n = len(rows); return f"n={n} correct {ok/n:.1%} over {over/n:.1%} under {under/n:.1%} fail {fail/n:.1%}"
for name, p in [("base@0069", BASE), ("X-2 seed0", ROOT / X2["seed0"]), ("X-2 seed1", ROOT / X2["seed1"]), ("X-2 seed2", ROOT / X2["seed2"]), ("X-M", XM)]:
    print(name, direction(load(p)))
xm, b = load(XM), load(BASE)
flip = [i for i in xm if b[i]["score"]["correct"] and not xm[i]["score"]["correct"]]
o = sum(1 for i in flip if (xm[i].get("parsed") or {}).get("count") is not None and xm[i]["parsed"]["count"] > xm[i]["score"]["gt"])
print(f"base correct -> X-M wrong: {len(flip)}; over {o}, under {len(flip)-o}")
