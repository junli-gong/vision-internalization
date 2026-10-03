#!/usr/bin/env python
"""管理员用：汇总 E1 主评测（RUN-0027）各模型、各任务的三列读数（只读 metrics.json，不改任何结果）。"""
import json, sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else "experiments/E1/0027_e1_main")
for mdir in sorted(p for p in root.iterdir() if p.is_dir() and (p / "refexp_box").exists() or (p / "count").exists()):
    print(f"== {mdir.name}")
    for t in sorted(mdir.iterdir()):
        f = t / "metrics.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        for c in d.get("three_columns", []):
            st = c["strict"]; ci = st.get("ci") or {}
            cond = c.get("conditional") or {}
            cv = cond.get("value")
            print(f"  {t.name:32s} {c['name']:10s} {c.get('scope',''):8s} strict={st['value']:.4f} "
                  f"[{ci.get('low', float('nan')):.4f},{ci.get('high', float('nan')):.4f}] n={c['n']} "
                  f"valid={c.get('valid_rate', float('nan')):.3f} cond={'' if cv is None else f'{cv:.4f}'}")
        tr = d.get("truncation")
        if isinstance(tr, dict) and tr.get("rate"):
            print(f"  {'':32s} truncation={tr.get('rate'):.3f}")
