"""RUN-0051 report helper: every pre-eval merge check under trained/ that did not pass (evaluation refused).
Prints run, written time, failures, restricted thinking_off decode_vs_generate_batch exact_rows, divergent rows / rows, gap median and max,
merge pass, teacher-forcing agreement, timing, and the file path. Facts only, no scores.
usage: python p51_refusals.py <experiments/E5/0051_e5_main_hpcgpu0X> [...]"""
import json, sys
from pathlib import Path
print("| 运行 | 核对写于 | restricted exact_rows（不同行 / 行） | gap 中位 / 最大 | merge pass | restricted 教师强制 agreement | timing_s.total_s | failures | 核对文件 |")
print("|---|---|---|---|---|---|---|---|---|")
for root in sys.argv[1:]:
    root = Path(root)
    for f in sorted(root.glob("trained/*/*/n*/seed*/merge_check.json")):
        d = json.loads(f.read_text())
        if d.get("pass"):
            continue
        r = d["checks"]["restricted"]["by_thinking"]["thinking_off"]; x = r.get("decode_vs_generate_batch", {}); g = x.get("gap", {})
        tf = r.get("teacher_forcing", {}).get("agreement")
        run = "/".join(f.parent.parts[-4:])
        print(f"| {run} | {d.get('written_utc')} | {x.get('exact_rows')}（{x.get('n_divergent_rows')} / {x.get('n_rows')}） | "
              f"{g.get('median', 0):.4f} / {g.get('max', 0):.4f} | {d['checks'].get('merge', {}).get('pass')} | {tf} | "
              f"{(d.get('timing_s') or {}).get('total_s')} | {json.dumps(d.get('failures'), ensure_ascii=False)} | `{f.relative_to(root.parent)}` |")
