"""RUN-0038 report helper (server CC): tables read verbatim from the run's merge_check.json / metrics.json / spotcheck files (no re-scoring).
usage: python p36_eval_tables.py <run dir> <out prefix>"""
import json
import sys
from pathlib import Path

M, S = "qwen3.5-2b", 0


def load(p):
    return json.loads(p.read_text()) if p.is_file() else None


def f(v, nd=4):
    return "—" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def vci(m, nd=4):
    if not m or m.get("value") is None:
        return "—"
    c = m.get("ci") or {}
    return f"{f(m['value'], nd)} [{f(c.get('low'), nd)}, {f(c.get('high'), nd)}]" if c else f(m["value"], nd)


def pci(d, nd=4):
    return "—" if not d else f"{d['estimate']:+.{nd}f} [{d['low']:+.{nd}f}, {d['high']:+.{nd}f}]"


def main(run, prefix):
    run = Path(run)
    md, J = ["# RUN-0038 评测表（各 metrics.json 原样取值）", ""], {}
    # 1. merge checks
    md += ["## 1. 合并核对（merge_check.json）", "", "| 检查点（运行目录内的位置） | pass | 各项 | 跳过 | 跳过原因 | error |", "|---|---|---|---|---|---|"]
    J["merge_check"] = {}
    for p in sorted(run.glob("**/merge_check.json")):
        m = load(p)
        checks = {k: v.get("pass") for k, v in (m.get("checks") or {}).items() if isinstance(v, dict)}
        rel = str(p.parent.relative_to(run))
        J["merge_check"][rel] = {"pass": m.get("pass"), "checks": checks, "skipped_checks": m.get("skipped_checks"), "skip_reason": m.get("skip_reason"),
                                 "error": m.get("error"), "checkpoint": (m.get("checkpoint") or {}).get("dir")}
        md.append(f"| `{rel}` | {m.get('pass')} | {checks or '—'} | {m.get('skipped_checks') or '—'} | {(m.get('skip_reason') or '—')[:60]} | {m.get('error')} |")
    # 2. grid single round
    md += ["", "## 2. 网格单轮（eval_e3b --k-values 1）", "",
           "| 变体 | 任务 | n | 有效率 | 解析失败 | guided | bilinear |", "|---|---|---|---|---|---|---|"]
    J["grid"] = {}
    for v in ("e3b", "c1prime"):
        for t in ("refseg", "depth"):
            d = load(run / "eval" / v / M / f"seed{S}" / t / "metrics.json")
            if d is None:
                continue
            g, b = d["single"]["guided"], d["single"]["bilinear"]
            st = d["stages"]["round1"]
            J["grid"].setdefault(v, {})[t] = {"n": d["n"], "single": d["single"], "stage": st}
            if t == "refseg":
                gs = f"cIoU {vci(g['cIoU'])}; gIoU {vci(g['gIoU'])}"
                bs = f"cIoU {vci(b['cIoU'])}; gIoU {vci(b['gIoU'])}"
            else:
                gs = f"AbsRel {vci(g['abs_rel'])}; δ1 {vci(g['d1'])}; δ1 严格 {vci(g['d1_strict'])}"
                bs = f"AbsRel {vci(b['abs_rel'])}; δ1 {vci(b['d1'])}; δ1 严格 {vci(b['d1_strict'])}"
            md.append(f"| {v} | {t} | {d['n']} | {f(g['valid_rate'])} | {st['parse_failures'] or '—'} | {gs} | {bs} |")
    # 3. dense
    md += ["", "## 3. 稠密（eval_e3a；主读数为 guided 上采样）", "",
           "| 变体 | 任务 | n | 主指标（guided） | bilinear |", "|---|---|---|---|---|"]
    leak = ["", "### 3.1 泄漏检查（leak_checks；差 = 正常 − 检查，配对 bootstrap 95% CI）", "",
            "| 变体 | 任务 | 检查 | 检查时的读数 | 正常 − 检查 |", "|---|---|---|---|---|"]
    J["dense"] = {}
    keys = {"refseg": ("cIoU", "gIoU"), "depth": ("abs_rel", "d1"), "semseg": ("miou", "pixel_acc"), "normals": ("mean", "median", "within_11.25")}
    for v in ("full", "c1", "c2"):
        for t in ("refseg", "depth", "semseg", "normals"):
            d = load(run / "eval" / "e3a" / v / M / f"seed{S}" / t / "metrics.json")
            if d is None:
                continue
            h, bl = d["headline"], d["by_upsample"].get("bilinear", {})
            J["dense"].setdefault(v, {})[t] = {"n": d["n"], "headline": h, "by_upsample": d["by_upsample"], "leak_checks": d["leak_checks"]}
            hs = "; ".join(f"{k} {vci(h.get(k))}" for k in keys[t] if k in h)
            def bval(k):
                x = bl.get(k)
                if isinstance(x, dict):
                    return vci(x)
                return f(x) if x is not None else "—"
            bs = "; ".join(f"{k} {bval(k)}" for k in keys[t] if k in bl or isinstance(bl.get("metrics"), dict))
            md.append(f"| {v} | {t} | {d['n']} | {hs} | {bs or '—'} |")
            for lk, lv in d["leak_checks"].items():
                mt = lv.get("metrics", {})
                ms = []
                for k in keys[t]:
                    x = mt.get(k)
                    if isinstance(x, dict):
                        ms.append(f"{k} {vci(x)}")
                    elif x is not None:
                        ms.append(f"{k} {f(x)}")
                pd = lv.get("paired_normal_minus_check") or {}
                ps = "; ".join(f"{k} {pci(pd[k])}" for k in keys[t] if k in pd)
                leak.append(f"| {v} | {t} | {lk} | {'; '.join(ms) or '—'} | {ps or '—'} |")
    md += leak
    # 4. T-text
    TASKS = ["refexp_box", "refexp_poly", "grid16", "count", "ovd", "rel_depth_nyu", "rel_depth_blink", "spatial_cvbench", "spatial_embspatial"]
    LET = ["rel_depth_nyu", "rel_depth_blink", "spatial_cvbench", "spatial_embspatial"]
    md += ["", "## 4. T 文本（run_e1 --checkpoint；三列取 three_columns 的第一行）", "",
           "| 变体 | 任务 | 指标（范围） | n | 严格 [CI] | 有效率 | 条件 |", "|---|---|---|---|---|---|---|"]
    extra = ["", "### 4.1 去偏读数、free 200 条与 NYU 对调标记", "",
             "| 变体 | 任务 | 去偏读数（全量）[CI] | free：严格 [CI]，有效率 | NYU 对调标记 [CI] |", "|---|---|---|---|---|"]
    J["ttext"] = {}
    for v in ("e3b", "c3", "e3a"):
        base = run / "t_text" / v / f"seed{S}" / M
        for t in TASKS:
            d = load(base / t / "metrics.json")
            if d is None:
                continue
            c = d["three_columns"][0]
            J["ttext"].setdefault(v, {})[t] = {"three_columns": d["three_columns"], "parse": d.get("parse")}
            md.append(f"| {v} | {t} | {c['name']}（{c['scope']}） | {c['n']} | {vci(c['strict'])} | {f(c['valid_rate'])} | {vci(c['conditional'])} |")
        for t in LET:
            db = load(base / f"{t}__debiased" / "metrics.json")
            fr = load(base / f"{t}__free" / "metrics.json")
            ls = load(base / f"{t}__letter_swap" / "metrics.json") if t == "rel_depth_nyu" else None
            frc = fr["three_columns"][0] if fr else None
            J["ttext"][v][f"{t}__debiased"] = db["headline"] if db else None
            J["ttext"][v][f"{t}__free"] = fr["three_columns"] if fr else None
            if ls:
                J["ttext"][v][f"{t}__letter_swap"] = ls["headline"]
            extra.append(f"| {v} | {t} | {vci(db['headline'][0]) if db else '—'} | "
                         f"{(vci(frc['strict']) + '，' + f(frc['valid_rate'])) if frc else '—'} | {vci(ls['headline'][0]) if ls else '—'} |")
    md += extra
    # 5. spot checks
    md += ["", "## 5. 冻结对照的抽查（spotcheck/*.md 的判定）", "", "| 文件 | 判定行 | 原始输出逐条相同 |", "|---|---|---|"]
    J["spotcheck"] = {}
    for p in sorted((run / "spotcheck").glob("*.md")):
        lines = p.read_text().splitlines()
        title = lines[0].lstrip("# ").strip() if lines else ""
        same = next((l.strip("- ").strip() for l in lines if "原始输出逐条相同" in l), "—")
        J["spotcheck"][p.name] = {"title": title, "same": same}
        md.append(f"| {p.name} | {title} | {same} |")
    Path(prefix + ".md").write_text("\n".join(md) + "\n")
    Path(prefix + ".json").write_text(json.dumps(J, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:3])
