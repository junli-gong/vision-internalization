"""RUN-0080 报告用：x2_summary/<划分>.json 里各设置的 cIoU [CI]、gIoU、是否多轮 / 诊断 / 消融、每条调用数、fallback 与解析失败相关字段。
用法：python p80_summary.py <x2_summary 目录> [划分 ...]"""
import json, sys
from pathlib import Path

D = Path(sys.argv[1])
files = [D / f"{s}.json" for s in sys.argv[2:]] if len(sys.argv) > 2 else sorted(D.glob("*.json"))
for p in files:
    d = json.loads(p.read_text())
    print(f"### {d['split']}（n {d['n']}，checkpoint {d.get('checkpoint', '')[:8]}…）\n")
    extra_keys = sorted({k for v in d["settings"].values() for k in v} - {"cIoU", "cIoU_ci", "gIoU", "multi_round", "diagnostic", "ablation", "per_sample", "fallback"})
    print("| 设置 | cIoU [95% CI] | gIoU | 多轮 | 诊断 | 消融 | 每条调用 | 每条生成 token | fallback |" + "".join(f" {k} |" for k in extra_keys))
    print("|---|---|---|---|---|---|---|---|---|" + "---|" * len(extra_keys))
    for k, v in d["settings"].items():
        ci = v.get("cIoU_ci") or {}
        ps = v.get("per_sample") or {}
        cells = [k, f"{v['cIoU']:.4f} [{ci.get('low', float('nan')):.4f}, {ci.get('high', float('nan')):.4f}]", f"{v.get('gIoU', float('nan')):.4f}",
                 str(v.get("multi_round")), str(v.get("diagnostic")), str(v.get("ablation")),
                 f"{ps.get('n_calls', float('nan')):.2f}", f"{ps.get('generated_tokens', float('nan')):.1f}", json.dumps(v.get("fallback"), ensure_ascii=False)]
        cells += [json.dumps(v.get(x), ensure_ascii=False)[:120] for x in extra_keys]
        print("| " + " | ".join(cells) + " |")
    other = {k: v for k, v in d.items() if k not in ("experiment", "split", "model", "checkpoint", "n", "settings")}
    if other:
        print("\n- 其他字段：" + json.dumps(other, ensure_ascii=False)[:1500])
    print()
