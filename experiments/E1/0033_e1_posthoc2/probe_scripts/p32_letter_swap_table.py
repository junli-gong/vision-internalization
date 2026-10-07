"""RUN-0033 report helper: letter-swap NYU table and usage (reads run outputs only; no scoring of its own).

Letter swap (per model): from <model>/rel_depth_nyu__letter_swap/metrics.json["letter_swap"] verbatim: headline (debiased, full eval set,
CI), original_letter (same run, original image, CI), paired_diff (CI, answer changes), per_view (accuracy [CI], most common letter and
share), answer_letters; plus validity_d72.json's order0_vs_main_letter (original pass vs RUN-0027 letter readout, per item).
Usage: per-step wall seconds from logs/driver.log, against the task book's estimate (letter_swap 0.45 h for 8 models, forced_poly 0.18 h).
usage: python p32_letter_swap_table.py <run dir> <driver.log> <out prefix>
"""
import json
import re
import sys
from pathlib import Path

ORDER = ["qwen3.5-0.8b", "qwen3.5-2b", "qwen3.5-4b", "qwen3.5-9b", "internvl3.5-2b-hf", "internvl3.5-4b-hf",
         "gemma-4-e2b-it", "gemma-4-e4b-it"]
EST = {"letter_swap": 0.45 * 3600, "forced_poly": 0.18 * 3600}


def ci(c):
    return f"[{c['low']:.4f}, {c['high']:.4f}]" if c else "—"


def main(run, driver_log, prefix):
    run = Path(run)
    val = {(c["model"], c["task"]): c for c in json.loads((run / "validity_d72.json").read_text())["cells"]}
    out = {"letter_swap": {}, "usage": {}}
    md = ["# RUN-0033 对调标记的 NYU 点对（取自各 `rel_depth_nyu__letter_swap/metrics.json` 的 letter_swap）", "",
          "## 1. 去偏读数（全量）、同次运行原图那一遍的字母读数与配对差", "",
          "| 模型 | n | 去偏读数 | 95% CI | 原图字母读数 | 95% CI | 差（去偏 − 原图） | 95% CI | 改答 | 错→对 | 对→错 | 去偏答案最常见字母 | 占比 | 原图那一遍与 RUN-0027 字母读数逐条一致 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    views = ["## 2. 两张图各自的读数", "", "| 模型 | 图 | n | 准确率 | 95% CI | 最常见字母 | 占比 |", "|---|---|---|---|---|---|---|"]
    for m in ORDER:
        p = run / m / "rel_depth_nyu__letter_swap" / "metrics.json"
        if not p.exists():
            continue
        mt = json.loads(p.read_text())
        d, h = mt["letter_swap"], mt["headline"][0]
        o, pdf, al = d["original_letter"]["headline"][0], d["paired_diff"], d["answer_letters"]
        agree = val.get((m, "rel_depth_nyu"), {}).get("order0_vs_main_letter")
        out["letter_swap"][m] = {"n": d["n"], "debiased": {"value": h["value"], "ci": h["ci"]}, "original": {"value": o["value"], "ci": o["ci"]},
                                 "paired_diff": pdf, "answer_letters": al, "per_view": d["per_view"], "order0_vs_main_letter": agree}
        ag = f"{agree['n_agree']}/{agree['n']} = {agree['rate']:.4f}" if agree else "—"
        md.append(f"| {m} | {d['n']} | {h['value']:.4f} | {ci(h['ci'])} | {o['value']:.4f} | {ci(o['ci'])} | {pdf['value']:+.4f} | {ci(pdf['ci'])} | "
                  f"{pdf['n_answer_changed']} | {pdf['n_wrong_to_right']} | {pdf['n_right_to_wrong']} | {al['most_common_letter']} | "
                  f"{al['most_common_letter_share']:.4f} | {ag} |")
        for v in d["per_view"]:
            views.append(f"| {m} | {v['view']} | {v['n']} | {v['accuracy']['value']:.4f} | {ci(v['accuracy']['ci'])} | "
                         f"{v['most_common_letter']} | {v['most_common_letter_share']:.4f} |")
    md += [""] + views + [""]
    steps = {}
    for line in Path(driver_log).read_text().splitlines():
        g = re.match(r"=== (\S+)__(letter_swap|forced_poly) rc=(\d+) \S+ \((\d+)s;", line)
        if g:
            steps[(g[1], g[2])] = (int(g[3]), int(g[4]))
    md += ["## 3. 用量（run_e1 墙钟，含加载模型；`logs/driver.log`）", "",
           "| 模型 | letter_swap rc | letter_swap s | forced_poly rc | forced_poly s | 合计 s |", "|---|---|---|---|---|---|"]
    tot = {"letter_swap": 0, "forced_poly": 0}
    for m in ORDER:
        if (m, "letter_swap") not in steps:
            continue
        r1, s1 = steps[(m, "letter_swap")]
        r2, s2 = steps.get((m, "forced_poly"), (None, 0))
        tot["letter_swap"] += s1
        tot["forced_poly"] += s2
        out["usage"][m] = {"letter_swap_s": s1, "letter_swap_rc": r1, "forced_poly_s": s2 if r2 is not None else None, "forced_poly_rc": r2}
        md.append(f"| {m} | {r1} | {s1} | {'—' if r2 is None else r2} | {'—' if r2 is None else s2} | {s1 + s2} |")
    md.append(f"| **合计** | | **{tot['letter_swap']}** | | **{tot['forced_poly']}** | **{sum(tot.values())}** |")
    md += ["", "| 读法 | 实测 GPU 小时 | 任务书估算 GPU 小时 |", "|---|---|---|"]
    for k in ("letter_swap", "forced_poly"):
        md.append(f"| {k} | {tot[k] / 3600:.3f} | {EST[k] / 3600:.2f} |")
    md.append(f"| 合计 | {sum(tot.values()) / 3600:.3f} | {sum(EST.values()) / 3600:.2f} |")
    out["usage_total_s"] = tot
    Path(prefix + ".md").write_text("\n".join(md) + "\n")
    Path(prefix + ".json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:4])
