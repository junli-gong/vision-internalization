"""RUN-0039 report helper (server CC): tables read verbatim from the run's metrics.json / merge_check.json (no re-scoring).
usage: python p37_diag_tables.py <run dir> <out prefix>"""
import json, sys
from pathlib import Path
M, S = "qwen3.5-2b", 0
TASKS = ["refexp_box", "refexp_poly", "grid16", "count", "ovd", "rel_depth_nyu", "rel_depth_blink", "spatial_cvbench", "spatial_embspatial"]


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
    md, J = ["# RUN-0039 诊断表（各 metrics.json 原样取值）", ""], {}
    md += ["## 1. 合并核对（merge_check.json）", "", "| 位置 | pass | 各项 | 跳过 |", "|---|---|---|---|"]
    J["merge_check"] = {}
    for p in sorted(run.glob("**/merge_check.json")):
        m = load(p); ch = {k: v.get("pass") for k, v in (m.get("checks") or {}).items() if isinstance(v, dict)}
        rel = str(p.parent.relative_to(run)); J["merge_check"][rel] = {"pass": m.get("pass"), "checks": ch, "skipped": m.get("skipped_checks")}
        md.append(f"| `{rel}` | {m.get('pass')} | {ch or '—'} | {m.get('skipped_checks') or '—'} |")
    md += ["", "## 2. 指令检查（E3b，refseg，前 2,048 条，与 RUN-0038 的正常评测逐条配对；guided；差 = 正常 − 检查）", "",
           "| 检查 | 评分对象 | 调用 / 跳过 / 配对 | 检查：cIoU | 检查：gIoU | 检查有效率 | 正常：cIoU | 正常：gIoU | 差 cIoU [CI] | 差 gIoU [CI] |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    J["instr"] = {}
    for kind in ("none", "swap"):
        m = load(run / "eval" / "e3b" / M / f"seed{S}" / f"refseg__instr_{kind}" / "metrics.json")
        if m is None:
            continue
        ic = m["instruction_check"]; J["instr"][kind] = {"instruction_check": {k: v for k, v in ic.items() if k != "rule"}, "targets": m["targets"],
                                                          "normal": {k: v for k, v in m["normal"].items() if k.startswith("n")}}
        for tgt, t in m["targets"].items():
            c, n, pd = t["check"], t["normal"], t["paired_normal_minus_check"]
            md.append(f"| {kind} | {tgt} | {ic.get('n_called')} / {ic.get('n_skipped')} / {m['normal'].get('n_paired')} | {vci(c['cIoU'])} | {vci(c['gIoU'])} | "
                      f"{f(c.get('valid_rate'))} | {vci(n['cIoU'])} | {vci(n['gIoU'])} | {pci(pd['cIoU'])} | {pci(pd['gIoU'])} |")
    md += ["", "## 3. 结构强制的深度（NYU 654，单轮，--decode structured）", "",
           "| 变体 | 有效率 | 解析失败 | guided：AbsRel | guided：δ1 | bilinear：AbsRel | bilinear：δ1 |", "|---|---|---|---|---|---|---|"]
    J["struct_depth"] = {}
    for v in ("e3b", "c1prime"):
        d = load(run / "eval" / v / M / f"seed{S}" / "depth__structured" / "metrics.json")
        if d is None:
            continue
        g, b, st = d["single"]["guided"], d["single"]["bilinear"], d["stages"]["round1"]
        J["struct_depth"][v] = {"single": d["single"], "stage": st}
        md.append(f"| {v} | {f(g['valid_rate'])} | {st['parse_failures'] or '—'} | {vci(g['abs_rel'])} | {vci(g['d1'])} | {vci(b['abs_rel'])} | {vci(b['d1'])} |")
    md += ["", "## 4. LoRA 缩放（派生检查点 lora_B × s）", "",
           "### 4.1 网格：refseg（受限解码，前 2,048 条，guided）与 depth（结构强制，654）", "",
           "| s | refseg n | refseg 有效率 | refseg cIoU | refseg gIoU | depth 有效率 | depth AbsRel（guided） | depth δ1（guided） |", "|---|---|---|---|---|---|---|---|"]
    J["scale"] = {}
    for tag, sc in (("e3b_s025", 0.25), ("e3b_s05", 0.5), ("e3b_s075", 0.75)):
        r = load(run / "eval" / tag / M / f"seed{S}" / "refseg" / "metrics.json")
        d = load(run / "eval" / tag / M / f"seed{S}" / "depth__structured" / "metrics.json")
        rg, dg = r["single"]["guided"], d["single"]["guided"]
        J["scale"][tag] = {"s": sc, "refseg": {"n": r["n"], "guided": rg}, "depth": {"n": d["n"], "guided": dg}}
        md.append(f"| {sc} | {r['n']} | {f(rg['valid_rate'])} | {vci(rg['cIoU'])} | {vci(rg['gIoU'])} | {f(dg['valid_rate'])} | {vci(dg['abs_rel'])} | {vci(dg['d1'])} |")
    md += ["", "### 4.2 T 文本（run_e1 --checkpoint，前 192 条；严格读数，three_columns 第一行；括号里是有效率）", "",
           "| s | " + " | ".join(TASKS) + " |", "|---|" + "---|" * len(TASKS)]
    for tag, sc in (("e3b_s025", 0.25), ("e3b_s05", 0.5), ("e3b_s075", 0.75)):
        cells = []
        for t in TASKS:
            m = load(run / "t_text" / tag / f"seed{S}" / M / t / "metrics.json")
            if m is None:
                cells.append("—"); continue
            c = m["three_columns"][0]
            J["scale"][tag].setdefault("ttext", {})[t] = m["three_columns"]
            cells.append(f"{f(c['strict']['value'])}（{f(c['valid_rate'], 3)}）")
        md.append(f"| {sc} | " + " | ".join(cells) + " |")
    md += ["", "T 文本各任务的指标：refexp_box acc@0.5、refexp_poly cIoU、grid16 gIoU、count accuracy、ovd F1@0.5（coco_val2017）、其余 accuracy；n 为前 192 条（BLINK 112）。"]
    Path(prefix + ".md").write_text("\n".join(md) + "\n")
    Path(prefix + ".json").write_text(json.dumps(J, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:3])
