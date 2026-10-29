"""RUN-0092 报告用：每个候选、每个 test 划分、每个设置的 cIoU / gIoU [95% CI]、n、complete、failures、spline_failure、fallback。
用法：python p92_split.py <0092_<名>_test_hpcgpu06 目录> [划分 ...]（默认列出已有的划分）"""
import json, sys
from pathlib import Path

root = Path(sys.argv[1]) / "qwen3.5-9b" / "refseg"
SETTINGS = ("x2_spline_trimap", "x2_ellipse_trimap")
splits = sys.argv[2:] or sorted(p.name for p in (root / SETTINGS[0]).iterdir() if p.is_dir())
print(f"## {Path(sys.argv[1]).name}\n")
print("| 划分 | 设置 | n | cIoU [95% CI] | gIoU [95% CI] | complete | failures | spline_failure | fallback |")
print("|---|---|---|---|---|---|---|---|---|")
for sp in splits:
    for st in SETTINGS:
        p = root / st / sp / "metrics.json"
        if not p.exists():
            print(f"| {sp} | {st} | — | 未完成 | | | | | |"); continue
        m = json.loads(p.read_text())
        h = {x["name"]: x for x in m["headline"]}
        f = lambda k: f"{h[k]['value']:.4f} [{h[k]['ci']['low']:.4f}, {h[k]['ci']['high']:.4f}]"
        j = lambda k: json.dumps(m.get(k), ensure_ascii=False)
        print(f"| {sp} | {st} | {m['n']} | {f('cIoU')} | {f('gIoU')} | {m.get('complete')} | {j('failures')} | {j('spline_failure')} | {j('fallback')} |")
