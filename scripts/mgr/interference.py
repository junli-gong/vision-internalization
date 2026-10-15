#!/usr/bin/env python
"""管理员用：E3 预注册 §13 的"干扰随尺寸的变化"（D93）。只读各评测运行的 metrics.json，不重算。

定义（§13，写于 0.8B、4B 的任何评测之前）：
- 语言干扰 L = C3 −（E3b+C3），单位是点（×100），取下列 8 个 T 文本读数之差的不加权平均：
  定位 acc@0.5、多边形 cIoU、计数、检测 F1@0.5（严格读数）；NYU 点对的对调标记；BLINK、CV-Bench、EmbSpatial 的全量去偏读数。
  grid16 不计入，单独报告。逐项差也报告；选择题另报 free 200 条的差。
- 像素干扰（两边结构强制解码、单轮、guided）：P_seg = E3b 的 refseg cIoU − E3b+C3 的；P_depth = E3b+C3 的 depth AbsRel − E3b 的。
- 逐种子配对；只用两臂都有有效读数的种子（没过合并核对的种子没有读数）。
- 趋势（L、P_seg、P_depth 各判一次）："随尺寸缩小" = 种子平均 0.8B > 2B > 4B 严格递减；"明显缩小" = 再加 4B ≤ 0.8B 的一半；其余为"不缩小"。
- 事后补充（D96-4，2026-10-15，描述性，不参与判定）：相对干扰 R_seg = P_seg ÷ E3b 的 refseg cIoU，R_depth = P_depth ÷ E3b 的 depth AbsRel
  （混入文本之后像素能力丢掉几成）；逐种子算、取种子平均，趋势用同一套写法描述，标"描述"。

用法：
  python scripts/mgr/interference.py --size 2B:B200:qwen3.5-2b:<T 文本与网格运行目录…> \\
      --size 0.8B:hpcgpu07:qwen3.5-0.8b:<…> --size 4B:hpcgpu06:qwen3.5-4b:<…> [--json out.json]
  每个 --size 是"尺寸:机器:模型:目录1,目录2,…"。各臂的读数在这些目录里按下面的相对路径找，先找到的为准：
    T 文本：<目录>/t_text/<变体>/seed<k>/<模型>/<任务>/metrics.json
    网格：  <目录>/eval/<变体>/<模型>/seed<k>/<refseg|depth>__structured/metrics.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

L_ITEMS = [  # (名字, T 文本的任务目录名)
    ("定位 acc@0.5", "refexp_box"),
    ("多边形 cIoU", "refexp_poly"),
    ("计数", "count"),
    ("检测 F1@0.5", "ovd"),
    ("NYU 对调标记", "rel_depth_nyu__letter_swap"),
    ("BLINK 去偏", "rel_depth_blink__debiased"),
    ("CV-Bench 去偏", "spatial_cvbench__debiased"),
    ("EmbSpatial 去偏", "spatial_embspatial__debiased"),
]
FREE_ITEMS = [("BLINK free", "rel_depth_blink__free"), ("CV-Bench free", "spatial_cvbench__free"),
              ("EmbSpatial free", "spatial_embspatial__free"), ("NYU free", "rel_depth_nyu__free")]
SEEDS = (0, 1, 2)


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def headline(d):
    """E1 metrics.json 的第一个标题读数（严格值）。"""
    if d is None:
        return None
    h = d.get("headline") or []
    first = h[0] if isinstance(h, list) and h else (h if isinstance(h, dict) else None)
    return None if not first else first.get("value")


def find(dirs, rel):
    for d in dirs:
        p = Path(d) / rel
        if p.is_file():
            return load(p)
    return None


def ttext(dirs, variant, seed, model, task):
    return headline(find(dirs, f"t_text/{variant}/seed{seed}/{model}/{task}/metrics.json"))


def grid(dirs, variant, seed, model, task, key):
    d = find(dirs, f"eval/{variant}/{model}/seed{seed}/{task}__structured/metrics.json")
    if d is None:
        return None
    v = (d.get("headline") or {}).get(key)
    return v.get("value") if isinstance(v, dict) else v


def size_block(label, machine, model, dirs):
    out = {"size": label, "machine": machine, "model": model, "L": {}, "L_items": {}, "free_items": {}, "grid16": {},
           "P_seg": {}, "P_depth": {}, "R_seg": {}, "R_depth": {}}
    for s in SEEDS:
        diffs = []
        for name, task in L_ITEMS:
            a, b = ttext(dirs, "c3", s, model, task), ttext(dirs, "e3b_c3", s, model, task)
            if a is None or b is None:
                diffs = None
                break
            out["L_items"].setdefault(name, {})[s] = 100 * (a - b)
            diffs.append(100 * (a - b))
        if diffs:
            out["L"][s] = sum(diffs) / len(diffs)
        for name, task in FREE_ITEMS:
            a, b = ttext(dirs, "c3", s, model, task), ttext(dirs, "e3b_c3", s, model, task)
            if a is not None and b is not None:
                out["free_items"].setdefault(name, {})[s] = 100 * (a - b)
        a, b = ttext(dirs, "c3", s, model, "grid16"), ttext(dirs, "e3b_c3", s, model, "grid16")
        if a is not None and b is not None:
            out["grid16"][s] = 100 * (a - b)
        e, m = grid(dirs, "e3b", s, model, "refseg", "cIoU"), grid(dirs, "e3b_c3", s, model, "refseg", "cIoU")
        if e is not None and m is not None:
            out["P_seg"][s] = e - m
            if e:
                out["R_seg"][s] = (e - m) / e
        e, m = grid(dirs, "e3b", s, model, "depth", "abs_rel"), grid(dirs, "e3b_c3", s, model, "depth", "abs_rel")
        if e is not None and m is not None:
            out["P_depth"][s] = m - e
            if e:
                out["R_depth"][s] = (m - e) / e
    return out


def summary(vals):
    v = list(vals.values())
    if not v:
        return None, None
    return statistics.mean(v), (statistics.stdev(v) if len(v) > 1 else None)


def trend(means):
    """means = [(尺寸, 平均)]，按 0.8B、2B、4B 的顺序；缺一个就判不了。"""
    if len(means) < 3 or any(m is None for _, m in means):
        return "判不了（缺尺寸）"
    a, b, c = (m for _, m in means)
    if a > b > c:
        return "明显缩小" if c <= 0.5 * a else "随尺寸缩小"
    return "不缩小"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--size", action="append", required=True, help="尺寸:机器:模型:目录1,目录2,…")
    p.add_argument("--json")
    a = p.parse_args(argv)
    blocks = []
    for spec in a.size:
        label, machine, model, dirs = spec.split(":", 3)
        blocks.append(size_block(label, machine, model, dirs.split(",")))
    order = {"0.8B": 0, "2B": 1, "4B": 2}
    blocks.sort(key=lambda b: order.get(b["size"], 9))
    lines = ["# 干扰随尺寸的变化（E3 预注册 §13）", "",
             "| 尺寸 | 机器 | 量 | 种子 0 | 种子 1 | 种子 2 | 平均（标准差） |", "|---|---|---|---:|---:|---:|---:|"]
    means = {"L": [], "P_seg": [], "P_depth": []}
    rel_means = {"R_seg": [], "R_depth": []}
    for b in blocks:
        for q, unit in (("L", "点"), ("P_seg", "cIoU"), ("P_depth", "AbsRel"), ("R_seg", "比例，事后补充"), ("R_depth", "比例，事后补充")):
            m, sd = summary(b[q])
            (means if q in means else rel_means)[q].append((b["size"], m))
            cells = [(f"{b[q][s]:.1f}" if q == "L" else f"{b[q][s]:.3f}") if s in b[q] else "—" for s in SEEDS]
            ms = "—" if m is None else (f"{m:.1f}" if q == "L" else f"{m:.3f}") + ("" if sd is None else f"（{sd:.1f}）" if q == "L" else f"（{sd:.3f}）")
            lines.append(f"| {b['size']} | {b['machine']} | {q}（{unit}） | " + " | ".join(cells) + f" | {ms} |")
    lines += ["", "**趋势**（§13 的判定，现在定死）：" + "；".join(f"{q} {trend(v)}" for q, v in means.items()),
              "", "**相对干扰的趋势**（D96-4 事后补充，描述性，不参与判定，不改 §13a 的触发）：" +
              "；".join(f"{q} {trend(v)}（描述）" for q, v in rel_means.items()), ""]
    for b in blocks:
        lines += [f"## {b['size']}（{b['machine']}）：L 的逐项与 free（点，C3 −（E3b+C3）；种子 0 / 1 / 2）", ""]
        for name, d in list(b["L_items"].items()) + list(b["free_items"].items()):
            lines.append(f"- {name}：" + " / ".join(f"{d[s]:.1f}" if s in d else "—" for s in SEEDS))
        lines.append("- grid16（不计入 L）：" + " / ".join(f"{b['grid16'][s]:.1f}" if s in b["grid16"] else "—" for s in SEEDS))
        lines.append("")
    print("\n".join(lines))
    if a.json:
        Path(a.json).write_text(json.dumps({"blocks": blocks, "trend": {q: trend(v) for q, v in means.items()},
                                            "relative_trend_descriptive": {q: trend(v) for q, v in rel_means.items()}},
                                           ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
