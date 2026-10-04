"""RUN-0030 report helper: debiased letter readout table and usage table (reads run outputs only; no scoring of its own).

Debiased (per model x task): taken verbatim from <model>/<task>__debiased/metrics.json["debiased"] (written by run_e1):
  headline (debiased, full eval set, CI), original_letter.headline (same run, order s = 0, CI), paired_diff (CI, answer changes),
  random_level, per_order (accuracy, most common letter and its share), answer_letters (most common answer letter share).
Usage: per-step wall seconds from logs/driver.log ("=== <model>__<step> rc=.. (<s>s; ...)"), against estimate.json
  (scripts/summarize_e1_posthoc.py estimate --main experiments/E1/0027_e1_main).
usage: python p30_debiased_table.py <run dir> <driver.log> <out prefix>
"""
import json
import re
import sys
from pathlib import Path

ORDER = ["qwen3.5-0.8b", "qwen3.5-2b", "qwen3.5-4b", "qwen3.5-9b", "internvl3.5-2b-hf", "internvl3.5-4b-hf",
         "gemma-4-e2b-it", "gemma-4-e4b-it"]
TASKS = ["rel_depth_nyu", "rel_depth_blink", "spatial_cvbench", "spatial_embspatial"]


def ci(c):
    return f"[{c['low']:.4f}, {c['high']:.4f}]" if c else "—"


def main(run, driver_log, prefix):
    run = Path(run)
    out, md = {"debiased": {}, "usage": {}}, []
    md += ["# RUN-0030 去偏字母读数（逐模型 × 任务，取自各 `<任务>__debiased/metrics.json`）", "",
           "## 1. 去偏读数、同次运行的原字母读数（s = 0）与配对差", "",
           "| 模型 | 任务 | n | 随机水平 | 去偏读数 | 95% CI | 原字母读数 | 95% CI | 差（去偏 − 原） | 95% CI | 改答 | 错→对 | 对→错 | 去偏答案最常见字母 | 占比 |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    per_order_md = ["## 2. 每种顺序 s 单独的读数（k > s 的样本）", "",
                    "| 模型 | 任务 | s | n | 准确率 | 95% CI | 最常见字母 | 占比 |", "|---|---|---|---|---|---|---|---|"]
    for m in ORDER:
        for t in TASKS:
            p = run / m / f"{t}__debiased" / "metrics.json"
            if not p.exists():
                continue
            mt = json.loads(p.read_text())
            d, h = mt["debiased"], mt["headline"][0]
            o = d["original_letter"]["headline"][0]
            pdf, al = d["paired_diff"], d["answer_letters"]
            row = {"n": d["n"], "k_counts": d["k_counts"], "random_level": d["random_level"],
                   "debiased": {"value": h["value"], "ci": h["ci"]}, "original": {"value": o["value"], "ci": o["ci"]},
                   "paired_diff": pdf, "answer_letters": al,
                   "per_order": [{"order": po["order"], "n": po["n"], "accuracy": po["accuracy"]["value"],
                                  "ci": po["accuracy"]["ci"], "most_common_letter": po["most_common_letter"],
                                  "most_common_letter_share": po["most_common_letter_share"]} for po in d["per_order"]]}
            out["debiased"].setdefault(m, {})[t] = row
            md.append(f"| {m} | {t} | {d['n']} | {d['random_level']:.4f} | {h['value']:.4f} | {ci(h['ci'])} | {o['value']:.4f} | "
                      f"{ci(o['ci'])} | {pdf['value']:+.4f} | {ci(pdf['ci'])} | {pdf['n_answer_changed']} | "
                      f"{pdf['n_wrong_to_right']} | {pdf['n_right_to_wrong']} | {al['most_common_letter']} | "
                      f"{al['most_common_letter_share']:.4f} |")
            for po in row["per_order"]:
                per_order_md.append(f"| {m} | {t} | {po['order']} | {po['n']} | {po['accuracy']:.4f} | {ci(po['ci'])} | "
                                    f"{po['most_common_letter']} | {po['most_common_letter_share']:.4f} |")
    md += [""] + per_order_md + [""]

    est = json.loads((run / "estimate.json").read_text())
    steps = {}
    for line in Path(driver_log).read_text().splitlines():
        g = re.match(r"=== (\S+)__(debiased|forced) rc=(\d+) \S+ \((\d+)s;", line)
        if g:
            steps[(g[1], g[2])] = (int(g[3]), int(g[4]))
    est_deb, est_forced = {}, {}
    for r in est["rows"]:
        if r["kind"] == "debiased":
            est_deb[r["model"]] = est_deb.get(r["model"], 0.0) + r["debiased_wall_s"]
        else:
            est_forced[r["model"]] = r["restricted_s"] + r["structured_s"] + r["prefill_upper_s"]
    md += ["## 3. 用量（run_e1 墙钟，含加载模型；估算取 `estimate.json`，不含加载模型）", "",
           "| 模型 | 去偏 rc | 去偏 s | 去偏估算 s | 格式强制 rc | 格式强制 s | 格式强制估算（上界）s | 合计 s | 合计估算 s |",
           "|---|---|---|---|---|---|---|---|---|"]
    tot = tot_e = 0.0
    for m in ORDER:
        if (m, "debiased") not in steps:
            continue
        rc_d, s_d = steps[(m, "debiased")]
        rc_f, s_f = steps.get((m, "forced"), (None, 0))
        e_d, e_f = est_deb.get(m, 0.0), est_forced.get(m, 0.0)
        tot += s_d + s_f
        tot_e += e_d + e_f
        out["usage"][m] = {"debiased_s": s_d, "debiased_rc": rc_d, "debiased_est_s": e_d, "forced_s": s_f if rc_f is not None else None,
                           "forced_rc": rc_f, "forced_est_s": e_f if m in est_forced else None}
        md.append(f"| {m} | {rc_d} | {s_d} | {e_d:.1f} | {'—' if rc_f is None else rc_f} | {'—' if rc_f is None else s_f} | "
                  f"{f'{e_f:.1f}' if m in est_forced else '—'} | {s_d + s_f} | {e_d + e_f:.1f} |")
    md.append(f"| **合计** | | | | | | | **{tot:.0f} s = {tot / 3600:.3f} h** | **{tot_e:.1f} s = {tot_e / 3600:.3f} h** |")
    out["usage_total"] = {"actual_s": tot, "actual_h": tot / 3600, "estimate_s": tot_e, "estimate_h": tot_e / 3600}
    Path(prefix + ".json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    Path(prefix + ".md").write_text("\n".join(md) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:4])
