"""9B 的语言代价并排表（D125-3：同一套健康检查与 T 文本读数，E3b+C3、W2-c、W2-c 加回放三列；另加 DEV-0064 的大题量读数）。
管理员的分析脚本，只读结果目录，不跑模型。每一列都与同一节点上的基座配对（D93-1）：
- 健康检查（简报 8.11，固定 200 题）：各系统的 health --compare 输出（标红照它的规则）；
- 大题量（MMLU-Pro 2,000 题、IFEval 541 条）：lang_battery --compare 的输出；
- T 文本：定位（refexp_box，acc@0.5，56,857 句）与计数（CountBench 442 条）。逐题配对 bootstrap 的差（百分点），95%，10,000 次，
  种子 src/constants.BOOTSTRAP_SEED；W2-c 与回放变体配 RUN-0069 在 06 上的基座，E3b+C3 配 RUN-0056 在 06 上的基座。
缺的格子（例如回放变体还没跑完）写"待 RUN-0076"。

用法：python scripts/mgr/lang_cost_table.py [--out docs/analysis/lang_cost_9b]  → <out>.json 与 <out>.md
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src import constants as C  # noqa: E402
from src import stats as S  # noqa: E402

W2 = "experiments/W2"
E3 = "experiments/E3/0056_hpc_9b_eval_hpcgpu06"
SYSTEMS = {
    "E3b+C3": {
        "health": f"{W2}/0074_lang_battery_hpcgpu06/health/health_compare_e3b_c3.json",
        "battery": f"{W2}/0074_lang_battery_hpcgpu06/lb_compare_e3b_c3.json",
        "ttext": f"{E3}/t_text/e3b_c3/seed0/qwen3.5-9b", "ttext_base": f"{E3}/qwen3.5-9b", "pending": "—"},
    "W2-c": {
        "health": f"{W2}/0069_w2c_hpcgpu06/health/health_compare.json",
        "battery": f"{W2}/0074_lang_battery_hpcgpu06/lb_compare_w2c.json",
        "ttext": f"{W2}/0069_w2c_hpcgpu06/0069_ttext/t_text/w2c/seed0/qwen3.5-9b",
        "ttext_base": f"{W2}/0069_w2c_hpcgpu06/0069_ttext/qwen3.5-9b", "pending": "—"},
    "W2-c+回放": {
        "health": f"{W2}/0076_w2c_replay_eval_hpcgpu06/health/health_compare.json",
        "battery": f"{W2}/0076_w2c_replay_eval_hpcgpu06/lb_compare_w2c_replay.json",
        "ttext": f"{W2}/0076_w2c_replay_ttext_hpcgpu06/t_text/w2c_replay/seed0/qwen3.5-9b",
        "ttext_base": f"{W2}/0069_w2c_hpcgpu06/0069_ttext/qwen3.5-9b", "pending": "待 RUN-0076"},
}
HEALTH_ROWS = [("mmlu_pro_letter", "accuracy"), ("gsm8k", "flexible"), ("ifeval", "prompt_level_strict")]
BATTERY_ROWS = [("mmlu_pro_letter", "accuracy"), ("ifeval", "prompt_level_strict"), ("ifeval", "instruction_level_strict")]
TTEXT = [("refexp_box", "定位 acc@0.5"), ("count", "计数准确率")]
N_RES, LEVEL = 10_000, 0.95


def jload(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def correct_map(leaf: Path) -> dict[str, bool] | None:
    for name, op in (("predictions.jsonl", open), ("predictions.jsonl.gz", gzip.open)):
        p = leaf / name
        if p.exists():
            with op(p, "rt", encoding="utf-8") as f:
                return {r["id"]: bool(r["score"]["correct"]) for r in map(json.loads, f) if r.get("score")}
    return None


def ttext_pair(sys_leaf: Path, base_leaf: Path) -> dict | None:
    a, b = correct_map(sys_leaf), correct_map(base_leaf)
    if a is None or b is None:
        return None
    if set(a) != set(b):
        raise SystemExit(f"{sys_leaf} 与 {base_leaf} 的题目集合不同")
    ids = sorted(a)
    x = np.array([a[i] for i in ids], dtype=np.float64) * 100
    y = np.array([b[i] for i in ids], dtype=np.float64) * 100
    idx = S._resample_indices(len(ids), N_RES, C.BOOTSTRAP_SEED)
    boot = x[idx].mean(1) - y[idx].mean(1)
    ci = S._percentile_ci(float(x.mean() - y.mean()), boot, LEVEL, N_RES)
    return {"n": len(ids), "base": float(y.mean()), "sys": float(x.mean()), "delta_pp": ci.estimate, "ci95_pp": [ci.low, ci.high]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="docs/analysis/lang_cost_9b")
    a = ap.parse_args(argv)
    doc: dict = {"systems": {}}
    for name, src in SYSTEMS.items():
        e: dict = {"sources": src, "health": {}, "battery": {}, "ttext": {}}
        h = jload(ROOT / src["health"])
        if h:
            for bench, metric in HEALTH_ROWS:
                r = next((r for r in h["rows"] if r["reading"] == bench and r["metric"] == metric), None)
                e["health"][f"{bench}.{metric}"] = None if r is None else {k: r[k] for k in ("base_pct", "merged_pct", "delta_pp", "ci95_pp", "flag", "n")}
            e["health_any_flag"] = h.get("any_flag")
        bt = jload(ROOT / src["battery"])
        if bt:
            for reading, metric in BATTERY_ROWS:
                r = next((r for r in bt["rows"] if r["reading"] == reading and r["metric"] == metric), None)
                e["battery"][f"{reading}.{metric}"] = None if r is None else {k: r.get(k) for k in ("base", "other", "delta_pp", "ci95_pp", "n")}
        for task, _ in TTEXT:
            e["ttext"][task] = ttext_pair(ROOT / src["ttext"] / task, ROOT / src["ttext_base"] / task)
        doc["systems"][name] = e
    out = Path(a.out) if Path(a.out).is_absolute() else ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    def cell(r, pending, flag=False):
        if r is None:
            return pending
        d, ci = r.get("delta_pp"), r.get("ci95_pp") or [None, None]
        if d is None:
            return pending
        s = f"{d:+.1f} [{ci[0]:+.1f}, {ci[1]:+.1f}]"
        return s + (" **标红**" if flag and r.get("flag") else "")
    names = list(SYSTEMS)
    L = ["# 9B 的语言代价并排（D125-3；`scripts/mgr/lang_cost_table.py` 生成）", "",
         "每格是相对同节点基座的差（百分点）与 95% CI。健康检查的标红照简报 8.11 的规则。", "",
         "| 读数 | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for bench, metric in HEALTH_ROWS:
        k = f"{bench}.{metric}"
        L.append(f"| 健康检查 {bench} {metric}（n {100 if bench == 'mmlu_pro_letter' else 50}） | "
                 + " | ".join(cell(doc["systems"][n]["health"].get(k), SYSTEMS[n]["pending"], True) for n in names) + " |")
    for reading, metric in BATTERY_ROWS:
        k = f"{reading}.{metric}"
        L.append(f"| 大题量 {reading} {metric} | " + " | ".join(cell(doc["systems"][n]["battery"].get(k), SYSTEMS[n]["pending"]) for n in names) + " |")
    for task, label in TTEXT:
        L.append(f"| T 文本 {label} | " + " | ".join(cell(doc["systems"][n]["ttext"].get(task), SYSTEMS[n]["pending"]) for n in names) + " |")
    base_bits = []
    for n in names:
        t = doc["systems"][n]["ttext"]
        if t.get("refexp_box") and t.get("count"):
            base_bits.append(f"{n} 的基座：定位 {t['refexp_box']['base']:.1f}、计数 {t['count']['base']:.1f}")
    L += ["", "T 文本的基座读数（同节点）：" + "；".join(base_bits) + "。"]
    out.with_suffix(".md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(out.with_suffix(".md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
