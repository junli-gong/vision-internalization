"""主方法 C1（X-2 的配方）的种子波动（D150、D151；探索日志"种子波动"一节的预测）。管理员的分析脚本，只读结果目录，不跑模型。

- 种子：seed 0 = RUN-0080（07）、seed 1 = RUN-0093（07）、seed 2 = RUN-0094（06）；没跑完的跳过并注明。
- 划分：explore_val 三层（400 / 400 / 200）、devtune800、devcheck2，合并共 2,300 句。不含 test 与 devcheck3。
- 读数：每个划分与合并的 cIoU（ratio-of-sums）；合并的 95% CI；与 seed 0 的配对差、与同节点 W2-c 参照（RUN-0082 的 0082_w2c_ref，
  06、07 逐位相同）的配对差，95%，10,000 次，种子 src/constants.BOOTSTRAP_SEED；3 个种子合并读数的样本标准差；健康检查的标红。
- 语言读数（D153）：每个种子的大题量（lb_compare，与同节点 lb_base 配对）与 T 文本定位、计数（与 RUN-0069 在 06 上的 T 文本基座逐题配对，
  口径同 scripts/mgr/lang_cost_table.py 的 ttext_pair：10,000 次，95%）。没跑完的写"缺"。
- 只描述，不做检验。

用法：python scripts/mgr/seed_variance.py [--out docs/analysis/seed_variance]  → <out>.json 与 <out>.md
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src import constants as C  # noqa: E402

_spec = importlib.util.spec_from_file_location("xtt", ROOT / "scripts/mgr/x_test_table.py")
XT = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(XT)
W2T = XT.W2T

M = "qwen3.5-9b"
SPLITS = ["explore_val_refcoco_unc_val", "explore_val_refcoco+_unc_val", "explore_val_refcocog_umd_val", "devtune800", "devcheck2"]
SETTINGS = ["x2_spline_trimap", "x2_ellipse_trimap", "x2_select", "x2_select5"]
SEEDS = {0: ("experiments/X/0080_x2_hpcgpu07/0080_eval", "experiments/X/0080_x2_hpcgpu07/0080_eval/health/health_compare.json"),
         1: ("experiments/X/0093_x2s1_hpcgpu07/0093_eval", "experiments/X/0093_x2s1_hpcgpu07/0093_eval/health/health_compare.json"),
         2: ("experiments/X/0094_x2s2_hpcgpu06/0094_eval", "experiments/X/0094_x2s2_hpcgpu06/0094_eval/health/health_compare.json")}
W2C_REF = "experiments/X/0082_x3_hpcgpu07/0082_w2c_ref"
TT_BASE = "experiments/W2/0069_w2c_hpcgpu06/0069_ttext/qwen3.5-9b"
LANG = {0: ("experiments/X/0084_lb_explore_hpcgpu07/lb_compare_x2.json", "experiments/X/0091_x_ttext_hpcgpu06/t_text/x2/seed0/qwen3.5-9b"),
        1: ("experiments/X/0096_lang_seeds_hpcgpu07/lb_compare_x2s1.json", "experiments/X/0096_lang_seeds_hpcgpu06/t_text/x2s1/seed1/qwen3.5-9b"),
        2: ("experiments/X/0096_lang_seeds_hpcgpu06/lb_compare_x2s2.json", "experiments/X/0096_lang_seeds_hpcgpu06/t_text/x2s2/seed2/qwen3.5-9b")}
_spec_l = importlib.util.spec_from_file_location("lct", ROOT / "scripts/mgr/lang_cost_table.py")
LCT = importlib.util.module_from_spec(_spec_l)
_spec_l.loader.exec_module(LCT)


def lang_of(s: int) -> dict:
    """一个种子的语言读数：大题量两行、T 文本两项；缺的为 None。"""
    lb, tt = LANG[s]
    out: dict = {"sources": {"battery": lb, "ttext": tt, "ttext_base": TT_BASE}, "battery": {}, "ttext": {}}
    bj = LCT.jload(ROOT / lb)
    for reading, metric in LCT.BATTERY_ROWS:
        r = None if not bj else next((r for r in bj["rows"] if r["reading"] == reading and r["metric"] == metric), None)
        out["battery"][f"{reading}.{metric}"] = None if r is None else {
            "base": r.get("base_pct"), "other": r.get("other_pct"), "delta_pp": r.get("delta_pp"), "ci95_pp": r.get("ci95_pp"), "n": r.get("n")}
    for task, _ in LCT.TTEXT:
        out["ttext"][task] = LCT.ttext_pair(ROOT / tt / task, ROOT / TT_BASE / task)
    return out


def leaf(run: str, setting: str, sp: str) -> Path:
    return ROOT / run / M / "refseg" / setting / sp


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="docs/analysis/seed_variance")
    a = ap.parse_args(argv)
    ref = {sp: W2T.preds(leaf(W2C_REF, "w2c", sp)) for sp in SPLITS}
    if not all(ref.values()):
        raise SystemExit("W2-c 参照不全")
    ref_pool = {k: v for sp in SPLITS for k, v in ref[sp].items()}
    doc: dict = {"n_resamples": XT.N_RES, "level": XT.LEVEL, "seed": C.BOOTSTRAP_SEED, "splits": SPLITS,
                 "w2c_ref": {"source": W2C_REF, "per_split": {sp: XT.single_ci(ref[sp]) for sp in SPLITS}, "pooled": XT.single_ci(ref_pool)},
                 "seeds": {}, "across_seeds": {}}
    preds: dict = {}
    for s, (run, health) in SEEDS.items():
        per = {st: {sp: W2T.preds(leaf(run, st, sp)) for sp in SPLITS} for st in SETTINGS}
        if not all(per[st][sp] for st in SETTINGS for sp in SPLITS):
            doc["seeds"][s] = {"status": "未跑完", "source": run}
            continue
        preds[s] = per
        e: dict = {"status": "完成", "source": run, "settings": {}}
        for st in SETTINGS:
            pool = {k: v for sp in SPLITS for k, v in per[st][sp].items()}
            if set(pool) != set(ref_pool):
                raise SystemExit(f"seed {s} {st}：句子集合与 W2-c 参照不同")
            e["settings"][st] = {"per_split": {sp: XT.single_ci(per[st][sp])["ciou"] for sp in SPLITS},
                                 "pooled": XT.single_ci(pool),
                                 "minus_w2c_pooled": XT.paired(pool, ref_pool)}
        h = ROOT / health
        if h.exists():
            hj = json.loads(h.read_text(encoding="utf-8"))
            e["health"] = {"any_flag": hj.get("any_flag"),
                           "rows": {f"{r['reading']}.{r['metric']}": {"delta_pp": r.get("delta_pp"), "flag": r.get("flag")}
                                    for r in hj.get("rows", []) if r.get("headline")}}
        e["language"] = lang_of(s)
        doc["seeds"][s] = e
    if 0 in preds:
        for s in preds:
            if s == 0:
                continue
            for st in SETTINGS:
                p0 = {k: v for sp in SPLITS for k, v in preds[0][st][sp].items()}
                ps = {k: v for sp in SPLITS for k, v in preds[s][st][sp].items()}
                doc["seeds"][s]["settings"][st]["minus_seed0_pooled"] = XT.paired(ps, p0)
                doc["seeds"][s]["settings"][st]["minus_seed0_per_split"] = {
                    sp: XT.single_ci(preds[s][st][sp])["ciou"] - XT.single_ci(preds[0][st][sp])["ciou"] for sp in SPLITS}
    done = [s for s in preds]
    for st in SETTINGS:
        vals = [doc["seeds"][s]["settings"][st]["pooled"]["ciou"] for s in done]
        doc["across_seeds"][st] = {"seeds": done, "pooled": vals,
                                   "mean": statistics.mean(vals) if vals else None,
                                   "sd": statistics.stdev(vals) if len(vals) >= 2 else None}
    out = Path(a.out) if Path(a.out).is_absolute() else ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    def d(x):
        return f"{x['diff']:+.3f} [{x['ci'][0]:+.3f}, {x['ci'][1]:+.3f}]"
    L = ["# 主方法 C1 的种子波动（`scripts/mgr/seed_variance.py` 生成；只描述；D150、D151）", "",
         f"5 个划分合并 2,300 句；cIoU = Σinter / Σunion；配对差为逐句配对 bootstrap，{XT.N_RES:,} 次，种子 {C.BOOTSTRAP_SEED}。"
         f"W2-c 参照：`{W2C_REF}`，合并 {doc['w2c_ref']['pooled']['ciou']:.3f}。", ""]
    for st in SETTINGS:
        L += [f"## {st}", "", "| 种子 | " + " | ".join(SPLITS) + " | 合并 [95% CI] | − W2-c 参照 | − seed 0 |",
              "|---|" + "---|" * (len(SPLITS) + 3)]
        for s, e in doc["seeds"].items():
            if e["status"] != "完成":
                L.append(f"| {s} | 未跑完 |" + " |" * (len(SPLITS) + 2))
                continue
            x = e["settings"][st]
            p = x["pooled"]
            L.append(f"| {s} | " + " | ".join(f"{x['per_split'][sp]:.3f}" for sp in SPLITS) +
                     f" | {p['ciou']:.3f} [{p['ci'][0]:.3f}, {p['ci'][1]:.3f}] | {d(x['minus_w2c_pooled'])} | "
                     f"{d(x['minus_seed0_pooled']) if 'minus_seed0_pooled' in x else '—'} |")
        a_ = doc["across_seeds"][st]
        L += ["", f"合并读数：种子 {a_['seeds']}，均值 {a_['mean']:.4f}，样本标准差 "
              f"{'—' if a_['sd'] is None else format(a_['sd'], '.4f')}。", ""]
    L += ["## 健康检查（与各自节点的基座配对）", "", "| 种子 | 标红 | 各项的差（百分点） |", "|---|---|---|"]
    for s, e in doc["seeds"].items():
        if "health" in e:
            L.append(f"| {s} | {e['health']['any_flag']} | " +
                     "；".join(f"{k} {v['delta_pp']:+.1f}{' 标红' if v['flag'] else ''}" for k, v in e["health"]["rows"].items()) + " |")
    L += ["", "## 语言读数（相对同节点基座，百分点 [95% CI]；D153）", "",
          "| 种子 | " + " | ".join(f"{r}.{m}" for r, m in LCT.BATTERY_ROWS) + " | " + " | ".join(t for t, _ in LCT.TTEXT) + " |",
          "|---|" + "---|" * (len(LCT.BATTERY_ROWS) + len(LCT.TTEXT))]
    for s_, e in doc["seeds"].items():
        if "language" not in e:
            continue
        lg = e["language"]
        cells = [("缺" if v is None else f"{v['delta_pp']:+.1f} [{v['ci95_pp'][0]:+.1f}, {v['ci95_pp'][1]:+.1f}]")
                 for v in list(lg["battery"].values()) + list(lg["ttext"].values())]
        L.append(f"| {s_} | " + " | ".join(cells) + " |")
    out.with_suffix(".md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(out.with_suffix(".md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
