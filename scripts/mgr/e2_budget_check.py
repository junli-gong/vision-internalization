"""E2 运行包的预算判定（E2 预注册 v1 §5 第 2 步、N23；管理员脚本，只读 run_e2 --estimate 的 estimate.json）。

总计 = 各模型估算之和（正式运行）+ 噪声底（NOISE_MODEL 的估算再算一遍）+ 准备开销 OVERHEAD_H（模型加载、bootstrap、
预跑判定与重测估算本身）。总计 ≤ 上限就 GO，否则 STOP。

用法：python scripts/mgr/e2_budget_check.py <估算目录：<out>/<模型>/estimate/estimate.json 的 <out>> --cap 32 \
          --models qwen3.5-0.8b qwen3.5-2b qwen3.5-4b qwen3.5-9b [--out budget_check.json]
退出码：0 = GO；3 = STOP（超上限）；2 = 缺估算或估算不全（没含法向、没含 pre_proj）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

NOISE_MODEL = "qwen3.5-0.8b"
OVERHEAD_H = 1.5
REQUIRED_TASKS = {"depth", "semseg", "normals", "refseg"}
REQUIRED_REFS = {"post_proj", "pre_proj"}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("estimate_root")
    p.add_argument("--cap", type=float, required=True, help="批准的 GPU 小时上限")
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--out")
    a = p.parse_args(argv)
    per, problems = {}, []
    for m in a.models:
        f = Path(a.estimate_root) / m / "estimate" / "estimate.json"
        if not f.exists():
            problems.append(f"缺 {f}")
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        h = d["header"]
        tasks = {r["task"] for r in d["tasks"]}
        if not REQUIRED_TASKS <= tasks:
            problems.append(f"{m} 的估算缺任务 {sorted(REQUIRED_TASKS - tasks)}")
        if not REQUIRED_REFS <= set(h.get("references") or []):
            problems.append(f"{m} 的估算缺参考线 {sorted(REQUIRED_REFS - set(h.get('references') or []))}")
        per[m] = {"gpu_hours": float(d["gpu_hours_total"]),
                  "by_task": {r["task"]: float(r["gpu_hours"]) for r in d["tasks"]},
                  "upsample_impls": h.get("upsample_impls"), "upsample_audit": h.get("upsample_audit")}
    main_h = sum(v["gpu_hours"] for v in per.values())
    noise_h = per.get(NOISE_MODEL, {}).get("gpu_hours", 0.0)
    total = main_h + noise_h + OVERHEAD_H
    rc = 2 if problems else (0 if total <= a.cap else 3)
    doc = {"cap_gpu_hours": a.cap, "per_model": per, "main_gpu_hours": main_h, "noise_floor_gpu_hours": noise_h,
           "overhead_gpu_hours": OVERHEAD_H, "total_gpu_hours": total,
           "decision": {0: "GO", 3: "STOP（超上限）", 2: "STOP（估算缺项）"}[rc], "problems": problems, "exit_code": rc}
    text = json.dumps(doc, ensure_ascii=False, indent=1)
    print(text)
    if a.out:
        Path(a.out).write_text(text + "\n", encoding="utf-8")
    return rc


if __name__ == "__main__":
    sys.exit(main())
