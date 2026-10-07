"""RUN-0032 report helper: E5 test tables, read verbatim from run_e5's metrics.json files (no scoring of its own).

1. Per model x system (zeroshot, cot) x task: n, valid rate (parse n_ok / n), parse failures by reason, truncation rate,
   thinking_end four classes (cot), headline-scope mask IoU (grid) strict [CI] and conditional, relief error (a, b) strict [CI],
   conditional and conditional median.
2. Gate (metrics["gate"]): prerequisite (strongest non-GT baseline, value, system, paired diff [CI], pass); judgment (a, b:
   copy_flat, gt_affine_upper, copy_convex paired diff [CI], beats; pass / status); imagining; headline n.
3. LISA (external/lisa-7b-v1): mask IoU (grid) and IoU against the current mask (D59-5), parse, gate.
4. Trivial baselines: per task x bin (and viewpoint per move group) mask IoU (grid) and relief error.
5. Usage: per-step wall seconds from logs/driver.log.
usage: python p31_e5_summary.py <run dir> <driver.log> <out prefix>
"""
import json
import re
import sys
from pathlib import Path

MODELS = ["qwen3.5-2b", "qwen3.5-0.8b", "qwen3.5-4b", "qwen3.5-9b", "internvl3.5-2b-hf", "internvl3.5-4b-hf",
          "gemma-4-e2b-it", "gemma-4-e4b-it"]
TASKS = [("a", "rotation3d"), ("b", "viewpoint"), ("c", "occlusion"), ("d", "physics"), ("e", "imagery")]
ENDS = ["natural", "forced", "not_opened", "unclosed"]


def f(x, nd=4):
    return "—" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def ci(c):
    return "—" if not c else f"[{c['low']:.4f}, {c['high']:.4f}]"


def load(p):
    return json.loads(p.read_text()) if p.exists() else None


def gate_rows(label, sysname, task, m, out):
    g = m["gate"]
    pr = g["prerequisite"]
    jd = g["judgment"]
    jb = jd.get("baselines") or {}
    cells = []
    for b in ("copy_flat", "gt_affine_upper", "copy_convex"):
        r = jb.get(b)
        cells.append("—" if r is None else f"{r['diff']:+.4f} {ci(r['diff_ci'])} {'胜' if r['beats'] else '否'}")
    out.append(f"| {label} | {sysname} | {task} | {g['n']} | {pr['strongest']} {f(pr['strongest_value'])} | {f(pr['system'])} | "
               f"{pr['diff']:+.4f} {ci(pr['diff_ci'])} | {pr['pass']} | " + " | ".join(cells) +
               f" | {jd.get('status')} / {jd.get('pass')} | {g['imagining']} |")
    return {"n": g["n"], "scope": g["scope"], "prerequisite": pr, "judgment": jd, "imagining": g["imagining"]}


def main(run, driver_log, prefix):
    run = Path(run)
    J = {"systems": {}, "gate": {}, "lisa": {}, "baselines": {}, "usage": {}}
    t1 = ["## 1. 逐模型 × 系统 × 任务（标题口径：a 只算 30° 以上三档，b 只算环绕 [20,180] 三档，c、d、e 全部样本）", "",
          "| 模型 | 系统 | 任务 | n | 有效率 | 解析失败（原因：条数） | 截断率 | thinking_end（自然/强制/没开/未闭合） | 标题 n | IoU 严格 | 95% CI | IoU 条件 | 起伏 严格 | 95% CI | 起伏 条件 | 起伏 条件中位数 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    t2 = ["## 2. 门槛（metrics 的 gate；差 = 系统 − 基线，配对 bootstrap 95% CI；IoU 下界 > 0、起伏上界 < 0 才算胜过）", "",
          "| 模型 | 系统 | 任务 | 标题 n | 先决：最强非真值基线 | 系统 | 差 [CI] | 先决通过 | 判定：平面 差 [CI] | 判定：真值仿射 差 [CI] | 判定：凸起先验 差 [CI] | 判定 状态 / 通过 | imagining |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for mdl in MODELS:
        for sysname in ("zeroshot", "cot"):
            for letter, task in TASKS:
                m = load(run / sysname / mdl / task / "metrics.json")
                if m is None:
                    continue
                pa, h = m["parse"], m["headline"]
                iou, rel = h["mask_iou_grid"], h.get("relief_error")
                te = m.get("thinking_end")
                te_s = "/".join(str(te["counts"].get(k, 0)) for k in ENDS) if te else "—"
                fails = ", ".join(f"{k}: {v}" for k, v in sorted(pa["failures"].items(), key=lambda kv: -kv[1])) or "—"
                row = {"n": m["n"], "complete": m["complete"], "n_expected": m["n_expected"], "valid_rate": pa["n_ok"] / pa["n"],
                       "parse": pa, "truncation": m.get("truncation"), "thinking_end": te, "headline_scope": m["headline_scope"],
                       "headline": h}
                J["systems"].setdefault(mdl, {}).setdefault(sysname, {})[task] = row
                t1.append(f"| {mdl} | {sysname} | {letter} {task} | {m['n']} | {pa['n_ok'] / pa['n']:.4f} | {fails} | "
                          f"{f(m['truncation']['rate'])} | {te_s} | {iou['n']} | {f(iou['strict'])} | {ci(iou['strict_ci'])} | "
                          f"{f(iou['conditional'])} | {f(rel['strict']) if rel else '—'} | {ci(rel['strict_ci']) if rel else '—'} | "
                          f"{f(rel['conditional']) if rel else '—'} | {f(rel['conditional_median']) if rel else '—'} |")
                J["gate"].setdefault(mdl, {}).setdefault(sysname, {})[task] = gate_rows(mdl, sysname, f"{letter} {task}", m, t2)
    # LISA
    t3 = ["## 3. LISA-7B-v1（外部对照，a–d；标题口径同上）", "",
          "| 任务 | n | 解析失败（原因：条数） | 标题 n | IoU 严格 | 95% CI | IoU 条件 | 对当前 mask 的 IoU 严格 | 95% CI | 对当前 mask 的 IoU 条件 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for letter, task in TASKS[:4]:
        m = load(run / "external" / "lisa-7b-v1" / task / "metrics.json")
        if m is None:
            continue
        pa, h = m["parse"], m["headline"]
        a, c = h["mask_iou_grid"], h["mask_iou_current_grid"]
        fails = ", ".join(f"{k}: {v}" for k, v in sorted(pa["failures"].items(), key=lambda kv: -kv[1])) or "—"
        t3.append(f"| {letter} {task} | {m['n']} | {fails} | {a['n']} | {f(a['strict'])} | {ci(a['strict_ci'])} | {f(a['conditional'])} | "
                  f"{f(c['strict'])} | {ci(c['strict_ci'])} | {f(c['conditional'])} |")
        J["lisa"][task] = {"n": m["n"], "parse": pa, "headline": h, "headline_scope": m["headline_scope"],
                           "gate": gate_rows("lisa-7b-v1", "external", f"{letter} {task}", m, t2)}
    # baselines
    t4 = ["## 4. 平凡基线（`baseline/<基线>/<任务>/metrics.json` 的 by_bin；档号从 0 数，档界见 E5 预注册 §2）", "",
          "| 基线 | 任务 | 档 | n | IoU（网格） | 起伏误差 严格 | 起伏误差 条件中位数 |", "|---|---|---|---|---|---|---|"]
    for bdir in sorted((run / "baseline").iterdir()):
        for letter, task in TASKS:
            m = load(bdir / task / "metrics.json")
            if m is None:
                continue
            J["baselines"].setdefault(bdir.name, {})[task] = {"headline": m["headline"], "headline_scope": m["headline_scope"],
                                                               "by_bin": m["by_bin"], "by_group": m.get("by_group")}
            items = [(f"档 {k}", v) for k, v in sorted(m["by_bin"].items(), key=lambda kv: int(kv[0]))]
            if task == "viewpoint":
                items += [(f"move={k}", v) for k, v in sorted((m.get("by_group") or {}).get("move", {}).items())]
            items.append((f"标题（{m['headline_scope']}）", m["headline"]))
            for lab, v in items:
                r = v.get("relief_error")
                t4.append(f"| {bdir.name} | {letter} {task} | {lab} | {v['mask_iou_grid']['n']} | {f(v['mask_iou_grid']['strict'])} | "
                          f"{f(r['strict']) if r else '—'} | {f(r['conditional_median']) if r else '—'} |")
    # usage
    t5 = ["## 5. 用量（`logs/driver.log`；GPU 时间 = 零样本/思维链两条命令与 LISA 推理的墙钟，含加载模型；导出、评分、基线是 CPU，不计）", "",
          "| 步骤 | rc | 墙钟 s | 计入 GPU | 累计 GPU s |", "|---|---|---|---|---|"]
    for line in Path(driver_log).read_text().splitlines():
        g = re.match(r"=== (\S+) rc=(\d+) \S+ \((\d+)s; gpu=(\d); GPU cumulative (\d+)s\)", line)
        if g:
            J["usage"][g[1]] = {"rc": int(g[2]), "wall_s": int(g[3]), "gpu": g[4] == "1", "cumulative_gpu_s": int(g[5])}
            t5.append(f"| {g[1]} | {g[2]} | {g[3]} | {'是' if g[4] == '1' else '否'} | {g[5]} |")
    per_item = {}
    for k, v in J["usage"].items():
        if v["gpu"]:
            item = k.split("__")[0]
            per_item[item] = per_item.get(item, 0) + v["wall_s"]
    t5 += ["", "| 项目 | GPU 墙钟 s | GPU 小时 |", "|---|---|---|"]
    for item, s in per_item.items():
        t5.append(f"| {item} | {s} | {s / 3600:.3f} |")
    tot = sum(per_item.values())
    t5.append(f"| **合计** | **{tot}** | **{tot / 3600:.3f}** |")
    J["usage_per_item_gpu_s"] = per_item
    J["usage_total_gpu_s"] = tot
    md = ["# RUN-0032 E5 test 汇总（取自各 metrics.json 原值）", ""] + t1 + [""] + t2 + [""] + t3 + [""] + t4 + [""] + t5 + [""]
    Path(prefix + ".md").write_text("\n".join(md) + "\n")
    Path(prefix + ".json").write_text(json.dumps(J, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:4])
