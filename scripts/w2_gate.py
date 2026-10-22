#!/usr/bin/env python
"""W2 的去留判定（DEV-0059）：照 D113 第六条（负责人原文 docs/lead_answers/2026-10-22-w2-directive.md 第三节，写于任何 W2 结果之前，
不得修改）原样实现，规则写死，不留参数。

    python scripts/w2_gate.py --main-name W2-a --main <out>/qwen3.5-9b/refseg/w2a/devcheck2 \\
        --w1f <out>/qwen3.5-9b/refseg/w1f/devcheck2 --out-dir <目录>   # 写 <目录>/gate.json 与 <目录>/gate.md

规则：主配置（W2-a 或 W2-c）的 cIoU 显著高于 W1-F：两者 cIoU 之差（主配置 − W1-F），配对 bootstrap（src/stats.paired_bootstrap_ci，
statistic = ratio_of_sums，即 Σinter / Σunion，同 w1_gate.py），置信水平 0.975（两个主配置的多重比较校正），重抽样 10,000 次，种子取
src/constants.BOOTSTRAP_SEED；CI 下界 > 0 才算通过。两个主配置各跑一次；"至少一个通过"不在这里合并，由管理员在门报告里写。

核对（不满足就报错，退出码 2）：两个目录的 predictions 都写完；id 集合相同；同一个模型键与 revision；都是 refseg、同一个划分（名字与
sha256）、同样的 tiny 与 --limit；主配置的设置是 w2a、W1-F 的设置是 w1f 且由 E1 协议的框得出（propose mode = box）；
W2-c 的主配置必须带训练的检查点、W2-a 与 W1-F 不能带。
判定点核对（另行报出；不是判定点时照算，但结论标为"彩排，不作数"）：模型 qwen3.5-9b、划分 devcheck2、n = 500、不是 --tiny、没有
--limit、两个运行用的 configs/w2.json 都已冻结（frozen = true）且是同一份（sha256 相同）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src import constants as C  # noqa: E402
from src.e1.runner import PREDICTIONS, read_jsonl_tolerant  # noqa: E402
from src.stats import paired_bootstrap_ci, ratio_of_sums  # noqa: E402
from src.w1.scoring import aligned_pairs, exact_ciou  # noqa: E402
from src.w2.runner import read_identity  # noqa: E402

# D113 第六条 / 负责人原文第三节（不得修改）
GATE_MODEL = "qwen3.5-9b"
GATE_SPLIT = "devcheck2"
GATE_N = 500
GATE_TASK = "refseg"
LEVEL = 0.975
N_RESAMPLES = 10_000
MAIN_SETTING = "w2a"
BASE_SETTING = "w1f"
BASE_PROPOSE_MODE = "box"
MAIN_NAMES = ("W2-a", "W2-c")
PASS, FAIL = "通过", "不通过"
RULE = ("主配置 − W1-F 的 cIoU 差（Σinter / Σunion），配对 bootstrap 97.5% CI（重抽样 10,000 次，种子 src/constants.BOOTSTRAP_SEED），"
        "下界 > 0 才算通过")


class GateInputError(ValueError):
    """两个运行不能放在一起判定（没写完、id 集合不同、模型或划分不同、设置不对……）。"""


def decide(main, w1f) -> dict:
    """两组记录（predictions.jsonl 的行，score 里有 inter / union）→ 判定的读数与结论。id 集合必须相同。"""
    ids, pm, pb = aligned_pairs(main, w1f)
    ci = paired_bootstrap_ci(pm, pb, statistic=ratio_of_sums, n_resamples=N_RESAMPLES, level=LEVEL, seed=C.BOOTSTRAP_SEED)
    if ci is None:
        raise GateInputError("没有样本")
    if (ci.level, ci.n_resamples) != (LEVEL, N_RESAMPLES):  # 写死的两个数必须真的生效
        raise RuntimeError(f"bootstrap 的水平与次数 {(ci.level, ci.n_resamples)} 不是规则写死的 {(LEVEL, N_RESAMPLES)}")
    cm, cb = exact_ciou(main), exact_ciou(w1f)
    ok = ci.low > 0
    return {"n": len(ids), "ciou": {"main": float(cm), "w1f": float(cb)},
            "rule": RULE, "diff": ci.estimate, "diff_exact": f"{(cm - cb).numerator}/{(cm - cb).denominator}", "ci_low": ci.low,
            "ci_high": ci.high, "level": ci.level, "n_resamples": ci.n_resamples, "seed": C.BOOTSTRAP_SEED, "pass": bool(ok),
            "decision": PASS if ok else FAIL}


def load_run(leaf: Path, what: str) -> tuple[dict, list]:
    if not (leaf / PREDICTIONS).exists():
        raise GateInputError(f"{what}：{leaf} 里没有 {PREDICTIONS}")
    ident = read_identity(leaf)
    recs = read_jsonl_tolerant(leaf / PREDICTIONS, repair=False)
    if ident["complete"] is not True or len(recs) != ident["n_samples"]:
        raise GateInputError(f"{what}：{leaf} 没有写完（{len(recs)}/{ident['n_samples']} 条，complete = {ident['complete']}）")
    return ident, recs


def evaluate(main_leaf: Path, w1f_leaf: Path, main_name: str) -> dict:
    if main_name not in MAIN_NAMES:
        raise GateInputError(f"--main-name 必须是 {MAIN_NAMES} 之一")
    (im, rm), (ib, rb) = load_run(Path(main_leaf), "主配置"), load_run(Path(w1f_leaf), "W1-F")
    for what, i, want in (("主配置", im, MAIN_SETTING), ("W1-F", ib, BASE_SETTING)):
        if i["task"] != GATE_TASK or i["setting"] != want:
            raise GateInputError(f"{what}：应是 {GATE_TASK}/{want}，收到 {i['task']}/{i['setting']}")
    if ib["propose_mode"] != BASE_PROPOSE_MODE:
        raise GateInputError(f"W1-F 必须由 E1 协议的框得出（propose mode = {BASE_PROPOSE_MODE}），收到 {ib['propose_mode']!r}")
    if ib["checkpoint"] is not None:
        raise GateInputError("W1-F 不能带训练的检查点（W1-F 是基座模型的框加默认 GrabCut）")
    if (main_name == "W2-c") != (im["checkpoint"] is not None):
        raise GateInputError(f"{main_name}：W2-c 的主配置必须带训练的检查点，W2-a 不能带（收到 checkpoint = {im['checkpoint']!r}）")
    for key in ("model", "revision", "split", "split_sha256", "tiny", "limit", "n_samples"):
        if str(im[key]) != str(ib[key]):
            raise GateInputError(f"两个运行的 {key} 不同：主配置 {im[key]!r}，W1-F {ib[key]!r}")
    if sorted(r["id"] for r in rm) != sorted(r["id"] for r in rb):
        raise GateInputError("两个运行的 id 集合不同")
    out = decide(rm, rb)
    wc_m, wc_b = im["w2_config"] or {}, ib["w2_config"] or {}
    checks = {"model_is_qwen3.5-9b": im["model"] == GATE_MODEL, "split_is_devcheck2": im["split"] == GATE_SPLIT,
              "n_is_500": out["n"] == GATE_N, "not_tiny": not im["tiny"], "no_limit": im["limit"] is None,
              "w2_config_frozen": wc_m.get("frozen") is True and wc_b.get("frozen") is True,
              "same_w2_config": bool(wc_m.get("config_sha256")) and wc_m.get("config_sha256") == wc_b.get("config_sha256")}
    out["main_name"] = main_name
    out["gate_point"] = {"checks": checks, "is_gate_point": all(checks.values())}
    out["verdict"] = out["decision"] if all(checks.values()) else f"彩排，不作数（规则给出：{out['decision']}）"
    out["runs"] = {"main": {"leaf": im["leaf"], "setting": im["setting"], "checkpoint": im["checkpoint"]},
                   "w1f": {"leaf": ib["leaf"], "setting": ib["setting"]}}
    out["identity"] = {k: im[k] for k in ("model", "revision", "split", "split_sha256", "tiny", "limit")}
    out["w2_config"] = {"main": wc_m, "w1f": wc_b}
    return out


def markdown(doc: dict) -> str:
    i = doc["identity"]
    return "\n".join([
        f"# W2 去留判定：{doc['main_name']}（D113 第六条，规则写死）", "",
        f"- 模型 {i['model']}@{str(i['revision'])[:12]}…，划分 {i['split']}，n = {doc['n']}；判定点核对：{doc['gate_point']['checks']}", "",
        "| 读数 | 值 |", "|---|---|",
        f"| {doc['main_name']}（w2a）cIoU | {doc['ciou']['main']:.4f} |",
        f"| W1-F（w1f）cIoU | {doc['ciou']['w1f']:.4f} |", "",
        f"- {doc['rule']}：差 {doc['diff']:.4f}（{doc['diff_exact']}），{doc['level'] * 100:.1f}% CI [{doc['ci_low']:.4f}, "
        f"{doc['ci_high']:.4f}]（{doc['n_resamples']} 次，种子 {doc['seed']}）→ **{doc['decision']}**", "",
        f"**结论：{doc['verdict']}**", "",
        "两个主配置（W2-a、W2-c）各判一次；\"至少一个通过\"由管理员在门报告里合起来写。", ""])


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--main-name", required=True, choices=MAIN_NAMES, help="主配置：W2-a（零样本）或 W2-c（训练过的挑选）")
    p.add_argument("--main", required=True, help="主配置（refseg/w2a/<划分>）的输出目录")
    p.add_argument("--w1f", required=True, help="W1-F（refseg/w1f/<划分>）的输出目录")
    p.add_argument("--out-dir", help="写 gate.json 与 gate.md 的目录")
    a = p.parse_args(argv)
    try:
        doc = evaluate(Path(a.main), Path(a.w1f), a.main_name)
    except (GateInputError, ValueError) as e:
        print(f"不能判定：{e}", file=sys.stderr)
        return 2
    md = markdown(doc)
    if a.out_dir:
        d = Path(a.out_dir)
        d.mkdir(parents=True, exist_ok=True)
        (d / "gate.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        (d / "gate.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
