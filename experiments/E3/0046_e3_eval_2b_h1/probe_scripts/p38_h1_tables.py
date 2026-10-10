"""RUN-0046 report helper (server CC): tables read verbatim from the run's merge_check.json / metrics.json / steps.log (no re-scoring).
E3b seed 0 depth (structured) is not in this run (eval.sh skips it); with <RUN-0039 dir> it is read from there and labelled.
usage: python p38_h1_tables.py <run dir> <out prefix> [<RUN-0039 dir>]"""
import json
import re
import sys
from pathlib import Path

M = "qwen3.5-2b"
TASKS = ["refexp_box", "refexp_poly", "grid16", "count", "ovd", "rel_depth_nyu", "rel_depth_blink", "spatial_cvbench", "spatial_embspatial"]
LET = ["rel_depth_nyu", "rel_depth_blink", "spatial_cvbench", "spatial_embspatial"]
TT = [("e3b_c3", 0), ("e3b_c3", 1), ("e3b_c3", 2), ("c3", 1), ("c3", 2)]
GRID = [("e3b_c3", 0), ("e3b_c3", 1), ("e3b_c3", 2), ("e3b", 0), ("e3b", 1), ("e3b", 2)]


def load(p):
    return json.loads(p.read_text()) if p.is_file() else None


def f(v, nd=4):
    return "—" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def vci(m, nd=4):
    if not m or m.get("value") is None:
        return "—"
    c = m.get("ci") or {}
    return f"{f(m['value'], nd)} [{f(c.get('low'), nd)}, {f(c.get('high'), nd)}]" if c else f(m["value"], nd)


def main(run, prefix, ref39=None):
    run = Path(run)
    md, J = ["# RUN-0046 评测表（各 metrics.json 原样取值）", ""], {}
    # 1. merge checks
    md += ["## 1. 合并核对（merge_check.json）", "", "| 位置 | pass | 各项 | 跳过 | 跳过原因 | error |", "|---|---|---|---|---|---|"]
    J["merge_check"] = {}
    for p in sorted(run.glob("**/merge_check.json")):
        m = load(p)
        ch = {k: v.get("pass") for k, v in (m.get("checks") or {}).items() if isinstance(v, dict)}
        rel = str(p.parent.relative_to(run))
        J["merge_check"][rel] = {"pass": m.get("pass"), "checks": ch, "skipped_checks": m.get("skipped_checks"), "skip_reason": m.get("skip_reason"),
                                 "error": m.get("error"), "checkpoint": (m.get("checkpoint") or {}).get("dir")}
        md.append(f"| `{rel}` | {m.get('pass')} | {ch or '—'} | {m.get('skipped_checks') or '—'} | {(m.get('skip_reason') or '—')[:60]} | {m.get('error')} |")
    # 2. T-text
    md += ["", "## 2. T 文本（run_e1 --checkpoint；三列取 three_columns 的第一行）", "",
           "| 变体 | 种子 | 任务 | 指标（范围） | n | 严格 [CI] | 有效率 | 条件 [CI] |", "|---|---|---|---|---|---|---|---|"]
    extra = ["", "### 2.1 去偏读数、free 200 条与 NYU 对调标记", "",
             "| 变体 | 种子 | 任务 | 去偏读数（全量）[CI] | free：严格 [CI]，有效率 | NYU 对调标记 [CI] |", "|---|---|---|---|---|---|"]
    J["ttext"] = {}
    for v, s in TT:
        base = run / "t_text" / v / f"seed{s}" / M
        J["ttext"].setdefault(v, {})[s] = {}
        for t in TASKS:
            d = load(base / t / "metrics.json")
            if d is None:
                continue
            c = d["three_columns"][0]
            J["ttext"][v][s][t] = {"three_columns": d["three_columns"], "parse": d.get("parse")}
            md.append(f"| {v} | {s} | {t} | {c['name']}（{c['scope']}） | {c['n']} | {vci(c['strict'])} | {f(c['valid_rate'])} | {vci(c['conditional'])} |")
        for t in LET:
            db = load(base / f"{t}__debiased" / "metrics.json")
            fr = load(base / f"{t}__free" / "metrics.json")
            ls = load(base / f"{t}__letter_swap" / "metrics.json") if t == "rel_depth_nyu" else None
            if not (db or fr or ls):
                continue
            frc = fr["three_columns"][0] if fr else None
            J["ttext"][v][s][f"{t}__debiased"] = db["headline"] if db else None
            J["ttext"][v][s][f"{t}__free"] = fr["three_columns"] if fr else None
            if ls:
                J["ttext"][v][s][f"{t}__letter_swap"] = ls["headline"]
            extra.append(f"| {v} | {s} | {t} | {vci(db['headline'][0]) if db else '—'} | "
                         f"{(vci(frc['strict']) + '，' + f(frc['valid_rate'])) if frc else '—'} | {vci(ls['headline'][0]) if ls else '—'} |")
    md += extra
    # 3. structured grid
    md += ["", "## 3. 网格（结构强制，eval_e3b --decode structured，单轮）", "",
           "| 变体 | 种子 | 任务 | n | 有效率 | 解析失败 | guided | bilinear |", "|---|---|---|---|---|---|---|---|"]
    J["grid"] = {}
    for v, s in GRID:
        for t in ("refseg", "depth"):
            d, lab = load(run / "eval" / v / M / f"seed{s}" / f"{t}__structured" / "metrics.json"), v
            if d is None and ref39 and (v, s, t) == ("e3b", 0, "depth"):
                d, lab = load(Path(ref39) / "eval" / v / M / f"seed{s}" / f"{t}__structured" / "metrics.json"), f"{v}（RUN-0039）"
            if d is None:
                continue
            g, b = d["single"]["guided"], d["single"]["bilinear"]
            st = d["stages"]["round1"]
            J["grid"].setdefault(v, {}).setdefault(s, {})[t] = {"n": d["n"], "single": d["single"], "stage": st}
            if t == "refseg":
                gs = f"cIoU {vci(g['cIoU'])}; gIoU {vci(g['gIoU'])}"
                bs = f"cIoU {vci(b['cIoU'])}; gIoU {vci(b['gIoU'])}"
            else:
                gs = f"AbsRel {vci(g['abs_rel'])}; δ1 {vci(g['d1'])}"
                bs = f"AbsRel {vci(b['abs_rel'])}; δ1 {vci(b['d1'])}"
            md.append(f"| {lab} | {s} | {t} | {d['n']} | {f(g['valid_rate'])} | {st['parse_failures'] or '—'} | {gs} | {bs} |")
    # 4. usage
    md += ["", "## 4. 用量（steps.log 的 wall_s；每条命令含加载模型）", "", "| 组 | 命令数 | rc≠0 | 墙钟 s | 小时 |", "|---|---|---|---|---|"]
    groups, tot, bad = {}, 0, 0
    for line in (run / "steps.log").read_text().splitlines():
        g = re.match(r"\S+ (\S+) rc=(\d+) wall_s=(\d+)", line)
        if not g:
            continue
        name, rc, w = g[1], int(g[2]), int(g[3])
        gm = re.match(r"(ttext|grid)_(.+_s\d+)_", name)
        key = f"{gm[1]} {gm[2]}" if gm else name
        e = groups.setdefault(key, [0, 0, 0])
        e[0] += 1; e[1] += rc != 0; e[2] += w
        tot += w; bad += rc != 0
    for k, (n, nb, w) in groups.items():
        md.append(f"| {k} | {n} | {nb} | {w} | {w / 3600:.3f} |")
    md.append(f"| **合计** | **{sum(e[0] for e in groups.values())}** | **{bad}** | **{tot}** | **{tot / 3600:.3f}** |")
    J["usage"] = {"groups": groups, "total_s": tot, "n_failed": bad}
    Path(prefix + ".md").write_text("\n".join(md) + "\n")
    Path(prefix + ".json").write_text(json.dumps(J, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:4])
