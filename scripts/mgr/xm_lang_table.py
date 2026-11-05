"""X-M 的语言代价表（D182 第 2、3 点）。管理员的分析脚本，只读结果目录，不跑模型。

D182 要求：不以阈值作论据，直接给出 X-M 相对同节点基座的读数，附置信区间，并与 X-2 三个种子的范围并列；补上计数，与定位一并给出。
- 大题量（MMLU-Pro 2,000 题、IFEval 541 条）：
  - 与四方表同格式：lang_battery --compare 的输出，07 的基座（RUN-0084）对 X-M（RUN-0107），由 RUN-0114 生成（1,000 次，取整到 0.1）；
  - 另列 G-XM 判定用的 10,000 次读数（gate_xm.json 第 4 条，D170）。
- T 文本：定位 acc@0.5（refexp_box，56,857 句）与计数 CountBench（442 题），RUN-0114 逐题配对：在 06 上时对 RUN-0069 的 T 文本基座，
  在 07 上时对同一运行在 07 上跑的基座（D183），
  算法同 scripts/mgr/lang_cost_table.ttext_pair（10,000 次）。
- 健康检查（简报 §8.11，固定 200 题）：RUN-0110 在 06 上对 RUN-0074 的基座（X-2 的健康检查在 07 上对 RUN-0080 的基座）。
- X-2 的三个种子：docs/analysis/seed_variance.json 各种子的 language 与 health（RUN-0080/0084/0091、0093、0094、0096）。
- 缺的格子（RUN-0114 还没跑完）写"待 RUN-0114"。

用法：python scripts/mgr/xm_lang_table.py [--out docs/analysis/xm_lang]  → <out>.json 与 <out>.md
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location("lang_cost_table", ROOT / "scripts/mgr/lang_cost_table.py")
LCT = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(LCT)

PENDING = "待 RUN-0114"
def xm_sources() -> dict:
    """RUN-0114 落在哪个节点（D183）：06 上对 RUN-0069 的 T 文本基座；07 上对它自己在 07 上跑的基座（同节点）。都没有时按 06 写路径。"""
    t07 = "experiments/XM/0114_xm_ttext_hpcgpu07"
    if (ROOT / t07).exists():
        tt, base, node = t07, f"{t07}/qwen3.5-9b", "hpcgpu07"
    else:
        tt, base, node = "experiments/XM/0114_xm_ttext_hpcgpu06", "experiments/W2/0069_w2c_hpcgpu06/0069_ttext/qwen3.5-9b", "hpcgpu06"
    return {"health": "experiments/XM/0110_xm_health_hpcgpu06/health/health_compare.json", "battery": f"{tt}/lb_compare_xm.json",
            "gate": "experiments/XM/0108_gate_xm_hpcgpu07/0108_gate/gate_xm.json", "ttext": f"{tt}/t_text/xm/seed0/qwen3.5-9b",
            "ttext_base": base, "ttext_node": node}
SEEDS = "docs/analysis/seed_variance.json"
BATTERY = [("mmlu_pro_letter.accuracy", "大题量 MMLU-Pro 字母准确率（2,000 题）"), ("ifeval.prompt_level_strict", "大题量 IFEval prompt 严格（541 条）"),
           ("ifeval.instruction_level_strict", "大题量 IFEval 指令级严格")]
TTEXT = [("count", "计数 CountBench（442 题）"), ("refexp_box", "定位 acc@0.5（56,857 句）")]
HEALTH = [("mmlu_pro_letter.accuracy", "健康检查 MMLU-Pro（100 题）"), ("gsm8k.flexible", "健康检查 GSM8K（50 题）"),
          ("ifeval.prompt_level_strict", "健康检查 IFEval prompt 严格（50 条）")]
GATE_ROWS = {"mmlu_pro_letter.accuracy": "mmlu_pro", "ifeval.prompt_level_strict": "ifeval"}


def battery_rows(path: Path) -> dict:
    bt = LCT.jload(path)
    out = {}
    if not bt:
        return out
    for key, _ in BATTERY:
        reading, metric = key.split(".")
        r = next((r for r in bt["rows"] if r["reading"] == reading and r["metric"] == metric), None)
        out[key] = None if r is None else {"base": r.get("base_pct"), "sys": r.get("other_pct"), "delta_pp": r.get("delta_pp"),
                                           "ci95_pp": r.get("ci95_pp"), "n": r.get("n")}
    return out


def health_rows(path: Path) -> dict:
    h = LCT.jload(path)
    out = {}
    if not h:
        return out
    for key, _ in HEALTH:
        reading, metric = key.split(".")
        r = next((r for r in h["rows"] if r["reading"] == reading and r["metric"] == metric), None)
        out[key] = None if r is None else {"base": r["base_pct"], "sys": r["merged_pct"], "delta_pp": r["delta_pp"], "ci95_pp": r["ci95_pp"],
                                           "flag": r["flag"], "n": r["n"]}
    out["any_flag"] = h.get("any_flag")
    return out


def gate_rows(path: Path) -> dict:
    g = LCT.jload(path)
    if not g:
        return {}
    rows = g["criteria"]["4"]["rows"]
    return {k: {"base": rows[v]["control_pct"], "sys": rows[v]["xm_pct"], "delta_pp": rows[v]["diff_pp"],
                "ci95_pp": [rows[v]["ci_low_pp"], rows[v]["ci_high_pp"]], "n": rows[v]["n"], "n_resamples": 10_000}
            for k, v in GATE_ROWS.items()}


def x2_seeds() -> dict:
    d = LCT.jload(ROOT / SEEDS) or {"seeds": {}}
    out: dict = {}
    for sd, e in sorted(d["seeds"].items()):
        L = e.get("language") or {}
        vals = {k: (v or {}).get("delta_pp") for k, v in (L.get("battery") or {}).items()}
        vals |= {k: v.get("delta_pp") if isinstance(v, dict) else v for k, v in (L.get("ttext") or {}).items()}
        vals |= {f"health:{k}": (v or {}).get("delta_pp") for k, v in ((e.get("health") or {}).get("rows") or {}).items()}
        out[sd] = vals
    return out


def rng(seeds: dict, key: str):
    v = [s.get(key) for s in seeds.values() if s.get(key) is not None]
    return None if not v else {"min": min(v), "max": max(v), "values": v}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="docs/analysis/xm_lang")
    a = ap.parse_args(argv)
    seeds = x2_seeds()
    XM = xm_sources()
    doc = {"sources": XM, "x2_seeds_source": SEEDS, "battery": battery_rows(ROOT / XM["battery"]), "battery_gate": gate_rows(ROOT / XM["gate"]),
           "ttext": {t: LCT.ttext_pair(ROOT / XM["ttext"] / t, ROOT / XM["ttext_base"] / t) for t, _ in TTEXT},
           "health": health_rows(ROOT / XM["health"]), "x2_seeds": seeds,
           "x2_range": {k: rng(seeds, k) for k in [b for b, _ in BATTERY] + [t for t, _ in TTEXT] + [f"health:{h}" for h, _ in HEALTH]}}

    def within(d, r):
        if not d or d.get("delta_pp") is None or not r:
            return "—"
        lo, hi = min(r["min"], r["max"]), max(r["min"], r["max"])
        x = d["delta_pp"]
        return "在范围内" if lo - 1e-9 <= x <= hi + 1e-9 else ("好于范围" if x > hi else "差于范围")

    def cell(d, pending=PENDING):
        if not d:
            return pending
        ci = d.get("ci95_pp")
        return f"{d['sys']:.1f}（基座 {d['base']:.1f}）：{d['delta_pp']:+.1f}" + (f" [{ci[0]:+.1f}, {ci[1]:+.1f}]" if ci else "")

    def rcell(r):
        return "—" if not r else f"{r['max']:+.1f} 到 {r['min']:+.1f}"

    L = ["# X-M 的语言代价（D182；管理员的分析脚本 `scripts/mgr/xm_lang_table.py` 生成，不要手改）", "",
         "- X-M 相对同节点基座的差，百分点，95% CI。大题量与健康检查同四方表的口径；T 文本逐题配对，10,000 次。",
         "- 最后两列：X-2 配方三个种子相对各自同节点基座的范围（`docs/analysis/seed_variance.json`），以及 X-M 落在范围内、好于范围还是差于范围。不以阈值作论据（D182）。", "",
         "| 读数 | X-M（基座）：差 [95% CI] | X-2 三个种子的范围 | X-M 相对这个范围 |", "|---|---|---|---|"]
    for k, name in BATTERY:
        L.append(f"| {name} | {cell(doc['battery'].get(k))} | {rcell(doc['x2_range'][k])} | {within(doc['battery'].get(k), doc['x2_range'][k])} |")
    for k, name in TTEXT:
        L.append(f"| {name} | {cell(doc['ttext'].get(k))} | {rcell(doc['x2_range'][k])} | {within(doc['ttext'].get(k), doc['x2_range'][k])} |")
    for k, name in HEALTH:
        hk = f"health:{k}"
        L.append(f"| {name} | {cell(doc['health'].get(k), '缺')} | {rcell(doc['x2_range'][hk])} | {within(doc['health'].get(k), doc['x2_range'][hk])} |")
    L += ["", "G-XM 判定用的大题量读数（gate_xm.json 第 4 条，10,000 次，不取整）：", ""]
    for k, name in BATTERY:
        g = doc["battery_gate"].get(k)
        if g:
            L.append(f"- {name}：{g['sys']:.2f}（基座 {g['base']:.2f}）：{g['delta_pp']:+.2f} [{g['ci95_pp'][0]:+.2f}, {g['ci95_pp'][1]:+.2f}]")
    L += ["", f"- 健康检查有无标红：{doc['health'].get('any_flag')}（X-M 对 06 的 RUN-0074 基座；X-2 的对 07 的 RUN-0080 基座）。",
          f"- 节点：大题量 X-M 与基座都在 07；T 文本都在 {XM['ttext_node'][-2:]}（{'对 RUN-0069 的基座' if XM['ttext_node'] == 'hpcgpu06' else '基座同在 07 上跑，D183'}）；健康检查都在 06。"]
    out = Path(a.out) if Path(a.out).is_absolute() else ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    out.with_suffix(".md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(out.with_suffix(".md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
