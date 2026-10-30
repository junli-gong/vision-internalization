"""统一主表（D154；预注册 U）：所有系统在同样的 5 个 test 划分 × 300 句上（configs/w2_llmseg_subsets.json，1,500 句）。
管理员的分析脚本，只读结果目录，不跑模型。

- 系统清单照预注册 U-2，顺序固定，不增不减；没跑完的写"缺"，跑坏的照样列出。
- 读数：每个划分与合并的 cIoU（Σinter / Σunion）、gIoU（逐句 IoU 均值）与 95% CI；每个系统对主方法 X-2 的配对差（cIoU、gIoU）。
  bootstrap 10,000 次，种子 src/constants.BOOTSTRAP_SEED；口径同 scripts/mgr/x_test_table.py。只描述。
- 新跑三项（E1 结构强制多边形、9B E3b 网格、LISA-7B）的结果目录由 --e1-poly、--e3b-grid、--lisa 给出（RUN 编号确定后写进下面的默认值）。

用法：python scripts/mgr/unified_table.py [--out docs/analysis/unified_table] [--e1-poly DIR] [--e3b-grid DIR] [--lisa DIR]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src import constants as C  # noqa: E402

_spec = importlib.util.spec_from_file_location("xtt", ROOT / "scripts/mgr/x_test_table.py")
XT = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(XT)
W2T, SPLITS, M = XT.W2T, XT.SPLITS, XT.M
SUBSETS = ROOT / "configs/w2_llmseg_subsets.json"
REF = "experiments/W2/0072_w2_test_hpcgpu06"
X2, X3 = "experiments/X/0092_x2_test_hpcgpu06", "experiments/X/0092_x3_test_hpcgpu06"
W2A = ROOT / "experiments/W2/0095_w2a_scores"
MAIN = "X-2（C1，主方法）"


def subset_ids() -> dict[str, list[str]]:
    d = json.loads(SUBSETS.read_text(encoding="utf-8"))
    return {sp: [f"{v['dataset']}_{v['split_by']}/{v['split']}/{i}" for i in v["ids"]] for sp, v in d["splits"].items()}


def leaf_preds(run: str | None, setting: str):
    return lambda sp: None if run is None else W2T.preds(XT.leaf(run, setting, sp))


def lisa_preds(run: str | None):
    """LISA 的叶目录在 <run>/lisa-7b-v1/refseg/lisa_7b/<划分>/（src/lisa/common 的布局，DEV-0075），不在 qwen3.5-9b/ 下。"""
    return lambda sp: None if run is None else W2T.preds(ROOT / run / "lisa-7b-v1" / "refseg" / "lisa_7b" / sp)


def w2a(sp: str):
    p = W2A / f"{sp}.jsonl"
    if not p.exists():
        return None
    return {r["id"]: {"inter": int(r["inter"]), "union": int(r["union"])}
            for r in map(json.loads, p.read_text(encoding="utf-8").splitlines()) if r}


def llmseg(which: int):
    return lambda sp: (XT.llmseg_preds(sp) or (None, None))[which]


def raw_path(name: str, a) -> object:
    """该系统逐句预测文件的路径（split → Path），用于数失败与退回；W2-a 的逐句预测只在 HPC（RUN-0095 只抽回交并），返回 None。"""
    leafp = lambda run, setting: (lambda sp: ROOT / run / M / "refseg" / setting / sp / "predictions.jsonl")  # noqa: E731
    table = {MAIN: leafp(X2, XT.MAIN), "X-3（C2，消融）": leafp(X3, XT.MAIN), "W2-c": leafp(REF, "w2c"),
             "模型自己的框": leafp(REF, "ownbox"), "W1-F": leafp(REF, "w1f"), "C1 的椭圆消融": leafp(X2, XT.ABL),
             "LLMSeg 修正后（主配置）": lambda sp: ROOT / XT.LLMSEG / M / "refseg" / "llmseg" / sp / "predictions.jsonl",
             "LLMSeg 修正前": lambda sp: ROOT / XT.LLMSEG / M / "refseg" / "llmseg" / sp / "predictions.jsonl"}
    if a.e1_poly:
        table["E1 结构强制多边形（9B 零样本）"] = leafp(a.e1_poly, "u_e1_poly_structured")
    if a.e3b_grid:
        table["9B E3b 像素 token 网格"] = leafp(a.e3b_grid, "u_e3b_grid")
    if a.lisa:
        table["LISA-7B（xinlai/LISA-7B-v1）"] = lambda sp: ROOT / a.lisa / "lisa-7b-v1" / "refseg" / "lisa_7b" / sp / "predictions.jsonl"
    return table.get(name)


def failure_counts(fn, ids: dict[str, list[str]]) -> dict | None:
    """U-1"失败率单列"：子集里 failure 非空或 status 为 failed 的句数（按空 mask 计分），与 fallback 非空的句数（退回替补，不是空 mask）。"""
    if fn is None:
        return None
    failed = fallback = n = 0
    for sp in SPLITS:
        p = fn(sp)
        if not p.exists():
            return None
        want = set(ids[sp])
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("id") not in want:
                continue
            n += 1
            failed += bool(r.get("failure")) or r.get("status") == "failed"
            fallback += bool(r.get("fallback"))
    return {"n": n, "failed": failed, "fallback": fallback}


def systems(a) -> list[tuple[str, str, object, str]]:
    """(组, 名, 取预测的函数 split → {id: {inter, union}}, 来源)；顺序即预注册 U-2。"""
    return [("本文的方法", MAIN, leaf_preds(X2, XT.MAIN), f"RUN-0092 {X2}"),
            ("本文的方法", "X-3（C2，消融）", leaf_preds(X3, XT.MAIN), f"RUN-0092 {X3}"),
            ("本文的方法", "W2-c", leaf_preds(REF, "w2c"), "RUN-0072"),
            ("本文的方法", "W2-a", w2a, "RUN-0072（逐句交并取 RUN-0095）"),
            ("各种手", "模型自己的框", leaf_preds(REF, "ownbox"), "RUN-0072"),
            ("各种手", "W1-F", leaf_preds(REF, "w1f"), "RUN-0072"),
            ("各种手", "E1 结构强制多边形（9B 零样本）", leaf_preds(a.e1_poly, "u_e1_poly_structured"), a.e1_poly or "待跑"),
            ("各种手", "9B E3b 像素 token 网格", leaf_preds(a.e3b_grid, "u_e3b_grid"), a.e3b_grid or "待跑"),
            ("外部方法", "LLMSeg 修正后（主配置）", llmseg(0), "RUN-0078"),
            ("外部方法", "LLMSeg 修正前", llmseg(1), "RUN-0078"),
            ("外部方法", "LISA-7B（xinlai/LISA-7B-v1）", lisa_preds(a.lisa), a.lisa or "待跑"),
            ("附带", "C1 的椭圆消融", leaf_preds(X2, XT.ABL), f"RUN-0092 {X2}")]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="docs/analysis/unified_table")
    ap.add_argument("--e1-poly", default="experiments/W2/0100_u_hands_hpcgpu06/0100_u_e1poly")  # RUN-0100
    ap.add_argument("--e3b-grid", default="experiments/W2/0100_u_hands_hpcgpu06/0100_u_e3bgrid")  # RUN-0100
    ap.add_argument("--lisa", default="experiments/W2/0097_lisa_hpcgpu06")  # RUN-0097
    a = ap.parse_args(argv)
    ids = subset_ids()
    sub = {}
    for group, name, fn, src in systems(a):
        per = {}
        for sp in SPLITS:
            p = fn(sp)
            if p is None:
                per = None
                break
            missing = [i for i in ids[sp] if i not in p]
            if missing:
                raise SystemExit(f"{name} {sp}：子集里 {len(missing)} 句没有预测（例 {missing[0]}）")
            per[sp] = {i: p[i] for i in ids[sp]}
        sub[name] = (group, src, per)
    main_per = sub[MAIN][2]
    doc: dict = {"subset": str(SUBSETS.relative_to(ROOT)), "n_per_split": 300, "n_resamples": XT.N_RES, "level": XT.LEVEL,
                 "seed": C.BOOTSTRAP_SEED, "main": MAIN, "systems": []}
    for name, (group, src, per) in sub.items():
        e: dict = {"group": group, "name": name, "source": src}
        if per is None:
            e["status"] = "缺"
            doc["systems"].append(e)
            continue
        pool = {k: v for sp in SPLITS for k, v in per[sp].items()}
        mpool = {k: v for sp in SPLITS for k, v in main_per[sp].items()}
        e["failures"] = failure_counts(raw_path(name, a), ids)
        e |= {"status": "完成",
              "per_split": {sp: {"cIoU": XT.single_ci(per[sp]), "gIoU": XT.single_g(per[sp])} for sp in SPLITS},
              "pooled": {"cIoU": XT.single_ci(pool), "gIoU": XT.single_g(pool)}}
        if name != MAIN:
            e["main_minus"] = {"pooled": {"cIoU": XT.paired(mpool, pool), "gIoU": XT.paired_g(mpool, pool)},
                               "per_split": {sp: XT.paired(main_per[sp], per[sp]) for sp in SPLITS}}
        doc["systems"].append(e)
    out = Path(a.out) if Path(a.out).is_absolute() else ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    def c(x):
        return f"{x['ciou']:.3f}"

    def ci(x, k):
        return f"{x[k]:.3f} [{x['ci'][0]:.3f}, {x['ci'][1]:.3f}]"

    def d(x):
        return f"{x['diff']:+.3f} [{x['ci'][0]:+.3f}, {x['ci'][1]:+.3f}]"
    L = ["# 统一主表（D154；预注册 U；`scripts/mgr/unified_table.py` 生成；只描述）", "",
         f"5 个 test 划分 × 300 句 = 1,500 句（`{doc['subset']}`）；cIoU = Σinter / Σunion，gIoU = 逐句 IoU 均值；"
         f"95% CI 与配对差：bootstrap {XT.N_RES:,} 次，种子 {C.BOOTSTRAP_SEED}。配对差 = X-2 − 该系统。", "",
         "| 组 | 系统 | " + " | ".join(SPLITS) + " | 合并 cIoU | 合并 gIoU | X-2 − 它（cIoU） | X-2 − 它（gIoU） |",
         "|---|---|" + "---|" * (len(SPLITS) + 4)]
    for e in doc["systems"]:
        if e["status"] != "完成":
            L.append(f"| {e['group']} | {e['name']} | 缺（{e['source']}） |" + " |" * (len(SPLITS) + 3))
            continue
        mm = e.get("main_minus", {}).get("pooled")
        L.append(f"| {e['group']} | {e['name']} | " + " | ".join(c(e["per_split"][sp]["cIoU"]) for sp in SPLITS) +
                 f" | {ci(e['pooled']['cIoU'], 'ciou')} | {ci(e['pooled']['gIoU'], 'giou')} | "
                 f"{d(mm['cIoU']) if mm else '—'} | {d(mm['gIoU']) if mm else '—'} |")
    L += ["", "## 失败率（预注册 U-1：单列）", "",
          "失败 = 解析或管线失败，按空 mask 计分；退回 = 输出解析不出、按规则改用替补（如 1 号候选），不是空 mask。", "",
          "| 系统 | 句数 | 失败 | 退回 |", "|---|---|---|---|"]
    for e in doc["systems"]:
        f = e.get("failures")
        L.append(f"| {e['name']} | " + ("— | — | — |" if not f else f"{f['n']:,} | {f['failed']} ({f['failed'] / f['n']:.1%}) | {f['fallback']} ({f['fallback'] / f['n']:.1%}) |"))
    L += ["", "W2-a 的逐句预测只在 HPC（RUN-0095 只抽回交并），这里记为 —；它挑选的解析失败按规则退回 W1-F，全 test 上 9B 为 2.9%（144 / 4,897，见 tab_w2_test）。",
          "LLMSeg 修正前后是同一次运行的两个读数，失败数相同。"]
    L += ["", "专家 SOTA 的灰色参考线见 `docs/references/refseg_sota_notes.md`（预注册 U-3a；完整 test 划分上的文献数字，不参与配对）。"]
    out.with_suffix(".md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(out.with_suffix(".md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
