"""RUN-0045 report helper: per run and per chunk (timing_run_NNN.jsonl = one chunk of up to 500 steps), the median step_s,
build_inputs_s, fwd_s, bwd_s, steps, and the ratio of the step median to the RUN-0044 throughput median (task book §3: stop if > 1.15).
usage: python p45_segments.py <experiments/E3/0045_hpc_train_<node>>"""
import json, statistics, sys
from pathlib import Path
REF = {("qwen3.5-4b", "c3"): 5.562, ("qwen3.5-4b", "e3b_c3"): 8.636, ("qwen3.5-0.8b", "c3"): 2.102, ("qwen3.5-0.8b", "e3b_c3"): 2.986}
root = Path(sys.argv[1])
print("| 运行 | 段 | 步数 | 每步中位 s | 输入构建中位 s | 前向中位 s | 反向中位 s | 对 RUN-0044 中位的比 |")
print("|---|---|---|---|---|---|---|---|")
for acc in sorted(root.glob("*/*/seed*/accounting.json")):
    d = acc.parent; variant, model, seed = d.parts[-3], d.parts[-2], d.parts[-1]
    for f in sorted((d / "logs").glob("timing_run_*.jsonl")):
        t = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
        if not t:
            continue
        med = lambda k: statistics.median(r[k] for r in t if k in r)
        ref = REF.get((model, variant))
        print(f"| {variant}/{model}/{seed} | {f.stem.replace('timing_run_', '')} | {len(t)} | {med('step_s'):.3f} | {med('build_inputs_s'):.3f} | "
              f"{med('fwd_s'):.3f} | {med('bwd_s'):.3f} | {med('step_s') / ref:.3f} |" if ref else "")
