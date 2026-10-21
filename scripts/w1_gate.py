#!/usr/bin/env python
"""W1 的去留判定（DEV-0051）：照 D103 §三原样实现，规则写死，不留参数（原文 docs/lead_answers/2026-10-20-w1-directive.md 第三节，
写于任何 W1 结果之前，不得修改）。

    python scripts/w1_gate.py --w1 <out>/qwen3.5-9b/refseg/full/devcheck \\
        --ownbox <out>/qwen3.5-9b/refseg/baseline_ownbox/devcheck \\
        --poly <out>/qwen3.5-9b/refseg/baseline_poly_structured/devcheck [--out-json g.json] [--out-md g.md]

判定点：开发-检查集（devcheck）的 500 句，Qwen3.5-9B 的完整工作台（W1-1 = 设置 full）。继续需要同时满足两条：
1. W1-1 的 cIoU 显著高于同一模型"自己的框"基线：配对 bootstrap 的 95% CI 下界 > 0
   （cIoU 差 W1-1 − 自己的框，src/stats.paired_bootstrap_ci，statistic = ratio_of_sums，重采样 1,000 次、水平 0.95、种子取
   src/constants.BOOTSTRAP_SEED；这三个数在这里写死，与 src/constants 不一致就报错）。
2. W1-1 的 cIoU 比同一模型在 E1 结构强制多边形上的 cIoU 至少高 0.2（点估计之差 ≥ 0.2；cIoU = Σinter / Σunion 用整数精确相除，
   比较没有浮点误差）。
两条都满足 → 继续（按三段式投 CVPR）；否则 → 放弃（停掉 W1，这一轮不投 CVPR）。

核对（不满足就报错，退出码 2）：三个目录的 predictions 都写完；id 集合相同；同一个模型（键与 revision）；都是 refseg、同一个划分；
设置依次是 full、baseline_ownbox、baseline_poly_structured。是不是判定点（模型 qwen3.5-9b、划分 devcheck、500 句、不是 --tiny、没有 --limit）
另行报出：不是判定点时照算两条，但结论标为"彩排，不作数"。
"""

from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src import constants as C  # noqa: E402
from src.e1.runner import PREDICTIONS, read_jsonl_tolerant  # noqa: E402
from src.stats import paired_bootstrap_ci, ratio_of_sums  # noqa: E402
from src.w1.runner import read_identity  # noqa: E402
from src.w1.scoring import aligned_pairs, exact_ciou  # noqa: E402

# D103 §三（不得修改）
GATE_MODEL = "qwen3.5-9b"
GATE_SPLIT = "devcheck"
GATE_N = 500
GATE_TASK = "refseg"
SETTINGS = {"w1": "full", "ownbox": "baseline_ownbox", "poly": "baseline_poly_structured"}
COND1_LEVEL = 0.95
COND1_N_RESAMPLES = 1000
COND2_MIN_DIFF = Fraction(1, 5)  # 0.2
CONTINUE, ABANDON = "继续", "放弃"


class GateInputError(ValueError):
    """三个运行不能放在一起判定（没写完、id 集合不同、模型或划分不同……）。"""


def _check_constants() -> None:
    if (C.BOOTSTRAP_CI_LEVEL, C.BOOTSTRAP_N_RESAMPLES) != (COND1_LEVEL, COND1_N_RESAMPLES):
        raise RuntimeError(f"src/constants 的 bootstrap 设置 {(C.BOOTSTRAP_CI_LEVEL, C.BOOTSTRAP_N_RESAMPLES)} 与判定规则写死的 "
                           f"{(COND1_LEVEL, COND1_N_RESAMPLES)} 不一致")


def decide(w1, ownbox, poly) -> dict:
    """三组记录（各自 predictions.jsonl 的行，score 里有 inter / union）→ 两条条件的读数与结论。id 集合必须相同。"""
    _check_constants()
    ids_w, pw, po = aligned_pairs(w1, ownbox)
    ids_p, _, pp = aligned_pairs(w1, poly)
    if ids_w != ids_p:
        raise GateInputError("内部错误：对齐后的 id 顺序不同")
    ci = paired_bootstrap_ci(pw, po, statistic=ratio_of_sums, n_resamples=COND1_N_RESAMPLES, level=COND1_LEVEL,
                             seed=C.BOOTSTRAP_SEED)
    c1 = ci.low > 0
    cw, cp = exact_ciou(w1), exact_ciou(poly)
    diff2 = cw - cp
    c2 = diff2 >= COND2_MIN_DIFF
    return {
        "n": len(ids_w),
        "ciou": {"w1_full": float(cw), "own_box": ratio_of_sums(po), "poly_structured": float(cp)},
        "condition_1": {"rule": "W1-1 − 自己的框 的 cIoU 差，配对 bootstrap 95% CI 下界 > 0", "diff": ci.estimate,
                        "ci_low": ci.low, "ci_high": ci.high, "level": ci.level, "n_resamples": ci.n_resamples,
                        "seed": C.BOOTSTRAP_SEED, "pass": bool(c1)},
        "condition_2": {"rule": "W1-1 − E1 结构强制多边形 的 cIoU 差 ≥ 0.2（精确值）", "diff": float(diff2),
                        "diff_exact": f"{diff2.numerator}/{diff2.denominator}", "min_diff": float(COND2_MIN_DIFF),
                        "pass": bool(c2)},
        "decision": CONTINUE if (c1 and c2) else ABANDON,
    }


def load_run(leaf: Path, what: str) -> tuple[dict, list]:
    if not (leaf / PREDICTIONS).exists():
        raise GateInputError(f"{what}：{leaf} 里没有 {PREDICTIONS}")
    ident = read_identity(leaf)
    recs = read_jsonl_tolerant(leaf / PREDICTIONS, repair=False)
    if ident["complete"] is not True or len(recs) != ident["n_samples"]:
        raise GateInputError(f"{what}：{leaf} 没有写完（{len(recs)}/{ident['n_samples']} 条，complete = {ident['complete']}）")
    return ident, recs


def evaluate(w1_leaf: Path, ownbox_leaf: Path, poly_leaf: Path) -> dict:
    runs = {k: load_run(Path(p), k) for k, p in (("w1", w1_leaf), ("ownbox", ownbox_leaf), ("poly", poly_leaf))}
    idents = {k: v[0] for k, v in runs.items()}
    for k, want in SETTINGS.items():
        if idents[k]["setting"] != want or idents[k]["task"] != GATE_TASK:
            raise GateInputError(f"{k}：应是 {GATE_TASK}/{want}，收到 {idents[k]['task']}/{idents[k]['setting']}")
    for key in ("model", "revision", "split", "split_sha256", "tiny", "limit"):
        vals = {k: v[key] for k, v in idents.items()}
        if len(set(map(str, vals.values()))) != 1:
            raise GateInputError(f"三个运行的 {key} 不同：{vals}")
    id_sets = {k: sorted(r["id"] for r in v[1]) for k, v in runs.items()}
    if not (id_sets["w1"] == id_sets["ownbox"] == id_sets["poly"]):
        raise GateInputError(f"三个运行的 id 集合不同：{ {k: len(v) for k, v in id_sets.items()} }")
    out = decide(runs["w1"][1], runs["ownbox"][1], runs["poly"][1])
    i = idents["w1"]
    checks = {"model_is_qwen3.5-9b": i["model"] == GATE_MODEL, "split_is_devcheck": i["split"] == GATE_SPLIT,
              "n_is_500": out["n"] == GATE_N, "not_tiny": not i["tiny"], "no_limit": i["limit"] is None}
    out["gate_point"] = {"checks": checks, "is_gate_point": all(checks.values())}
    out["verdict"] = out["decision"] if all(checks.values()) else f"彩排，不作数（规则给出：{out['decision']}）"
    out["runs"] = {k: {"leaf": str(v["leaf"]), "setting": v["setting"]} for k, v in idents.items()}
    out["identity"] = {k: i[k] for k in ("model", "revision", "split", "split_sha256", "tiny", "limit")}
    return out


def markdown(doc: dict) -> str:
    c1, c2 = doc["condition_1"], doc["condition_2"]
    yes = {True: "满足", False: "不满足"}
    return "\n".join([
        "# W1 去留判定（D103 §三，规则写死）", "",
        f"- 模型 {doc['identity']['model']}@{doc['identity']['revision'][:12]}…，划分 {doc['identity']['split']}，n = {doc['n']}；"
        f"判定点核对：{doc['gate_point']['checks']}", "",
        "| 读数 | 值 |", "|---|---|",
        f"| W1-1（full）cIoU | {doc['ciou']['w1_full']:.4f} |",
        f"| 自己的框 cIoU | {doc['ciou']['own_box']:.4f} |",
        f"| E1 结构强制多边形 cIoU | {doc['ciou']['poly_structured']:.4f} |", "",
        f"1. {c1['rule']}：差 {c1['diff']:.4f}，95% CI [{c1['ci_low']:.4f}, {c1['ci_high']:.4f}] → **{yes[c1['pass']]}**",
        f"2. {c2['rule']}：差 {c2['diff']:.4f}（{c2['diff_exact']}）→ **{yes[c2['pass']]}**", "",
        f"**结论：{doc['verdict']}**", ""])


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--w1", required=True, help="W1-1（refseg/full/<划分>）的输出目录")
    p.add_argument("--ownbox", required=True, help="\"自己的框\"（refseg/baseline_ownbox/<划分>）的输出目录")
    p.add_argument("--poly", required=True, help="E1 结构强制多边形（refseg/baseline_poly_structured/<划分>）的输出目录")
    p.add_argument("--out-json")
    p.add_argument("--out-md")
    a = p.parse_args(argv)
    try:
        doc = evaluate(Path(a.w1), Path(a.ownbox), Path(a.poly))
    except (GateInputError, ValueError) as e:
        print(f"不能判定：{e}", file=sys.stderr)
        return 2
    if a.out_json:
        Path(a.out_json).write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    md = markdown(doc)
    if a.out_md:
        Path(a.out_md).write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
