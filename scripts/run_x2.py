#!/usr/bin/env python
"""X-2 的推理（DEV-0069；探索 X-2；预注册 X）：一个检查点 × 一个划分，全部设置一次跑完（模型只加载一次）。规则见 src/explore/x2_runner.py，
参数 configs/x2_run.json（第一个检查点的读数出来之前不调）。输出 W2 格式，scripts/gate_explore.py 能直接读：
<out>/<模型>/refseg/<设置>/<划分>/（predictions.jsonl、metrics.json、compute.json、config.yaml、meta.json、README.md）；
轨迹 <out>/<模型>/refseg/x2_trace/<划分>/；汇总 <out>/<模型>/refseg/x2_summary/<划分>.json。

服务器（--checkpoint 是 scripts/train_w2c.py --config configs/x2_train.json 的 <叶目录>/ckpt/final/model）：
    for S in explore_val_refcoco_unc_val explore_val_refcoco+_unc_val explore_val_refcocog_umd_val; do
      python scripts/run_x2.py --model qwen3.5-9b --checkpoint $CK --splits-file configs/explore_splits.json --split $S \\
          --data-root $DATA --out experiments/X/<NNNN>_x2_eval --candidate-workers 8 --tool-workers 8 [--resume]
    done
    python scripts/run_x2.py --model qwen3.5-9b --checkpoint $CK --split devtune800 --data-root $DATA \\
        --out experiments/X/<NNNN>_x2_eval_devtune --candidate-workers 8 --tool-workers 8
    # 先做评测前的合并核对（同 run_w2 --setting w2c：<out>/<模型>/w2c_merge_check.json，不通过退出码 1），再 load_trained(merge=True)。
    # 提出与候选与 run_w2.py --setting w2c 同名（propose_w2c、candidates_w2c）：同一个 --out 里已有同一个检查点的就直接用。
    # 设置（20 个）：x2_select、x2_select_cor{1,2,3}、x2_spline_raster、x2_spline_trimap、x2_select5、x2_select5_cor{1,2,3}；诊断
    #   x2_oracle4、x2_oracle5；消融 x2_select_cor{1,2,3}_randbox、x2_select5_cor{1,2,3}_randbox、x2_ellipse_raster、x2_ellipse_trimap。
    #   --settings 只导出其中几个（轨迹照样全跑）。
    # 判定集（devcheck3）：只经 --unseal-devcheck3 G-EX（门名记进运行身份，放行记录追加进 <out>/<模型>/unseal_devcheck3.jsonl），
    #   且 W2 配置必须冻结。devcheck2 同 run_w2 要求冻结的配置。
    # test（DEV-0072，D147）：不给 --test-gate 与 --test-approval 时一律拒绝（同以前）。两者都给且都合格才放行：
    #   --test-approval 恰是 D147；--test-gate 是 scripts/gate_explore.py 写的 gate_explore.json，结论"通过"（不是"彩排，不作数"）、
    #   判定点核对全部为真、模型键相同、候选在 devcheck3 上的检查点 manifest sha256 等于 --checkpoint 的；要给 --settings，其中必须有
    #   候选判定过的设置，其余只能是 D147 批准的消融（x2_ellipse_trimap）；W2 配置必须冻结；清单只用 configs/w2_splits.json 的 test_*
    #   （与 run_w2 同一份清单、同一个取样函数，句子与 W2-c 的 test 逐 id 相同）。任何一条不满足：退出码 2，什么都不写。
    #   放行时批准编号、gate_explore.json 的路径与 sha256 记进运行身份（config.yaml 的 w2.test_unlock），放行记录追加进
    #   <out>/<模型>/test_unlock.jsonl。例（X-2 = C1，5 个 test 划分，各 1,000 句）：
    #     for S in test_refcoco_unc_testA test_refcoco_unc_testB test_refcoco+_unc_testA test_refcoco+_unc_testB test_refcocog_umd_test; do
    #       python scripts/run_x2.py --model qwen3.5-9b --checkpoint $CK --split $S --data-root $DATA --out experiments/X/<NNNN>_x2_test \\
    #           --settings x2_spline_trimap x2_ellipse_trimap --test-gate <RUN-0090 的 gate_C1>/gate_explore.json --test-approval D147 \\
    #           --candidate-workers 8 --tool-workers 8 [--resume]
    #     done
    # 用时（9B、A100、explore_val 1,000 句，全部设置；每句的模型调用：提出 1、挑选 K = 4 一次、描样条一次、挑选 K = 5 一次、两条修正轨迹
    #   各至多 3 次）：
    #   - 提出约 6 分钟（W2-c：500 句约 3 分钟）；候选 1,000 × 4.5 s / 8 进程约 10 分钟（CPU）；
    #   - 挑选两次约 6 + 8 分钟（K = 5 是 6 张图）；描样条约 160 个输出 token、批 16，约 10–15 分钟；
    #   - 修正：每轮一张叠加图 + 约 50 个输出 token，约 4–5 分钟一轮；两条轨迹各至多 3 轮，提前停使后几轮的句子变少，约 15–30 分钟；
    #   - 工具（trimap、局部 GrabCut、随机框）：每句约 1–2.5 s 的 CPU，--tool-workers 8 时与模型调用交替，约 5–10 分钟；
    #   合计约 1.0–1.5 GPU 小时 / 1,000 句；devtune800 约 0.8–1.2 小时。第一组（64 句）跑完先看 logs 的耗时再外推。
    # 只训了样条的检查点（DEV-0070，X-3：scripts/train_w2c.py --config configs/x3_train.json；叶目录 config.yaml 的 x2 一节没有 correct）：
    #   命令同上（--checkpoint 换成 X-3 的）。轨迹不调修正；涉及修正的设置（x2_select_cor*、x2_select5_cor*、*_randbox，共 12 个）跳过，
    #   不写输出目录，x2_summary/<划分>.json 的 skipped 写明"检查点没有修正流"；其余 8 个设置照常，输出格式不变。少了两条修正轨迹
    #   （RUN-0080 的 refcoco 层：修正调用每句约 0.6 + 0.6 s，另加工具），用时约省三成。
开发机彩排（--tiny：小型随机模型、CPU、fixture 的小数据；候选与挑选图的渲染可以换成 tests._w2_fakes）：
    python scripts/run_x2.py --tiny --model qwen3.5-0.8b --checkpoint <tiny 训练的检查点> --split devcheck2 --out <dir> \\
        --candidates-module tests._w2_fakes --render-module tests._w2_fakes
退出码：0 = 跑完；1 = 合并核对没有通过，拒绝评测；2 = 出错（报错先写进 <out>/<模型>/errors/，argparse 用法错误也是 2；test 的放行不合格
也是用法错误，什么都不写）。
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from src.adapters import registry  # noqa: E402
from src.data import explore_splits as EX  # noqa: E402
from src.e1.prompts import PROMPTS_PATH, load_prompt_config  # noqa: E402
from src.explore import x2_runner as XR  # noqa: E402
from src.runlog import sha256_file  # noqa: E402
from src.w2 import runner as R  # noqa: E402
from src.w2 import w2c as W2C  # noqa: E402
from src.w2.config import W2_CONFIG_PATH, load_w2_config  # noqa: E402
from src.w2.data import W2_SPLITS_PATH, needs_frozen, split_names, w2_items  # noqa: E402

DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}
RUN_DIR_NAME = re.compile(r"[0-9]{4}_[A-Za-z0-9][A-Za-z0-9_.-]*")
TASK = "refseg"
TINY_E1_BATCH_SIZE = 4


def _load_run_w2():
    """scripts/run_w2.py 的划分清单与数据路径的读法（同一套函数，不复制）。scripts 不是包，按路径加载。"""
    import importlib.util

    spec = importlib.util.spec_from_file_location("_run_w2_for_run_x2", REPO_ROOT / "scripts" / "run_w2.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


RW2 = _load_run_w2()


def is_test_split(split: str) -> bool:
    return split.startswith("test")


# ---------------------------------------------------------------------------
# test 的放行（DEV-0072）
# ---------------------------------------------------------------------------

# 依据：DECISIONS.md D147（项目负责人 2026-10-28 答复 N45，原文 docs/lead_answers/2026-10-28-n45.md）：
#   D147-2：通过 G-EX 判定的候选跑与 W2-c 相同的 test 子集（W2 的 5 个 test 划分）；D147-4：样条换成框内椭圆的消融（x2_ellipse_trimap）
#   在 test 上与候选同一次运行。预注册 X-5、X-5a。别的编号一律不认。
TEST_APPROVAL = "D147"
TEST_ABLATIONS: tuple[str, ...] = ("x2_ellipse_trimap",)  # D147-4：test 上与候选同一次导出的消融
TEST_UNLOCK_LOG = "test_unlock.jsonl"
# scripts/gate_explore.py 写进 gate_explore.json 的判定点核对（gate_point.checks）；全部在且全部为真才算判定点
GATE_POINT_CHECKS: tuple[str, ...] = ("manifest_not_tiny", "n_is_500", "not_tiny", "no_limit", "same_model", "same_host",
                                      "reference_w2_config_frozen", "unsealed_with_G-EX")
GATE_PASS = "通过"  # scripts/gate_explore.py 的 PASS（= scripts/w2_gate.PASS）
GATE_REHEARSAL = "彩排，不作数"  # scripts/gate_explore.py：判定点核对不全为真时 verdict 的前缀


class X2TestGateError(ValueError):
    """test 的放行不合格（D147）。"""


def check_test_gate(gate: str | Path | None, approval: str | None, *, checkpoint: str | Path, model: str, settings, rounds: int) -> dict:
    """test 的放行（D147）：两个参数都给、批准编号恰是 D147、gate_explore.json 结论"通过"且不是彩排、判定点核对全部为真、模型键相同、
    候选在 devcheck3 上的检查点 manifest sha256 等于 --checkpoint 的、--settings 里有候选判定过的设置且其余只有 D147 批准的消融。
    不满足抛 X2TestGateError。返回 {"identity"（进运行身份 w2.test_unlock：批准编号、gate_explore.json 的路径与 sha256、候选名与设置）,
    "gate"（gate_explore.json 的结论与 CI，进 test_unlock.jsonl）, "checkpoint_manifest_sha256"}。"""
    if gate is None or approval is None:
        raise X2TestGateError("--test-gate 与 --test-approval 要一起给（D147：test 只经通过 G-EX 判定的候选与负责人的批准编号放行），"
                            f"收到 --test-gate {gate!r}、--test-approval {approval!r}")
    if approval != TEST_APPROVAL:
        raise X2TestGateError(f"--test-approval 必须恰是 {TEST_APPROVAL}（负责人批准候选上 test 的裁定编号），收到 {approval!r}")
    p = Path(gate)
    try:
        raw = p.read_bytes()
        doc = json.loads(raw)
    except (OSError, ValueError) as e:
        raise X2TestGateError(f"--test-gate {gate} 读不了：{e}") from e
    if not isinstance(doc, dict) or "verdict" not in doc or "gate_point" not in doc:
        raise X2TestGateError(f"--test-gate {gate} 不是 scripts/gate_explore.py 写的 gate_explore.json")
    verdict, decision = doc.get("verdict"), doc.get("decision")
    if isinstance(verdict, str) and verdict.startswith(GATE_REHEARSAL):
        raise X2TestGateError(f"--test-gate {gate} 是彩排，不作数（verdict = {verdict!r}）：test 只认判定点上的通过")
    if verdict != GATE_PASS or decision != GATE_PASS or doc.get("pass") is not True:
        raise X2TestGateError(f"--test-gate {gate} 的结论不是“{GATE_PASS}”（verdict = {verdict!r}，decision = {decision!r}，pass = {doc.get('pass')!r}）")
    gp = doc.get("gate_point") if isinstance(doc.get("gate_point"), dict) else {}
    checks = gp.get("checks") if isinstance(gp.get("checks"), dict) else {}
    bad = [k for k in GATE_POINT_CHECKS if checks.get(k) is not True] + [k for k, v in checks.items() if v is not True and k not in GATE_POINT_CHECKS]
    if bad or gp.get("is_gate_point") is not True:
        raise X2TestGateError(f"--test-gate {gate} 的判定点核对没有全部为真：{bad or 'is_gate_point 不是 true'}")
    got_model = (doc.get("identity") or {}).get("model")
    if got_model != model:
        raise X2TestGateError(f"--test-gate {gate} 判的是模型 {got_model!r}，本次 --model {model!r}")
    cands = (doc.get("runs") or {}).get("candidate")
    if not isinstance(cands, list) or not cands or not all(isinstance(c, dict) for c in cands):
        raise X2TestGateError(f"--test-gate {gate} 里没有候选的运行（runs.candidate）")
    man = Path(checkpoint) / "manifest.json"
    if not man.is_file():
        raise X2TestGateError(f"--checkpoint {checkpoint} 里没有 manifest.json，核对不了检查点")
    want = sha256_file(man)
    have = sorted({str((c.get("checkpoint") or {}).get("manifest_sha256")) for c in cands})
    if have != [want]:
        raise X2TestGateError(f"--test-gate {gate} 判定的候选检查点（manifest sha256 {[h[:16] for h in have]}）不是本次的 --checkpoint"
                            f"（{want[:16]}…）")
    cand_settings = sorted({str(c.get("setting")) for c in cands})
    defs = XR.setting_defs(rounds)
    if len(cand_settings) != 1 or cand_settings[0] not in defs:
        raise X2TestGateError(f"--test-gate {gate} 的候选设置 {cand_settings} 不是 X-2 的一个设置")
    setting = cand_settings[0]
    if not settings:
        raise X2TestGateError(f"test 上要用 --settings 只导出判定过的设置 {setting}（与 D147 批准的消融 {list(TEST_ABLATIONS)}）")
    if setting not in settings:
        raise X2TestGateError(f"--settings {list(settings)} 里没有判定过的设置 {setting}（gate_explore.json 的候选）")
    extra = [s for s in settings if s != setting and s not in TEST_ABLATIONS]
    if extra:
        raise X2TestGateError(f"--settings 里的 {extra} 没有经过判定，也不是 D147 批准的消融 {list(TEST_ABLATIONS)}：test 上不导出")
    ident = {"approval": TEST_APPROVAL, "gate_explore": {"path": str(gate), "sha256": hashlib.sha256(raw).hexdigest()},
             "candidate_name": doc.get("candidate_name"), "candidate_setting": setting}
    summary = {"verdict": verdict, "n": doc.get("n"), "diff": doc.get("diff"), "ci_low": doc.get("ci_low"), "ci_high": doc.get("ci_high"),
               "rules": {k: (v or {}).get("result") for k, v in (doc.get("rules") or {}).items()}}
    return {"identity": ident, "gate": summary, "checkpoint_manifest_sha256": want}


def check_test_split_args(a) -> dict:
    """命令行上 test 的放行（parse_args 与 cmd 共用）：清单只能是默认的 configs/w2_splits.json（与 run_w2 的 test 划分同一份），再
    check_test_gate。不合格抛 X2TestGateError。"""
    if not RW2.is_default_splits(a):
        raise X2TestGateError(f"test 划分只用 configs/w2_splits.json 的 test_*（与 run_w2 同一份清单），不配 --splits-file {a.splits_file}")
    try:
        rounds = XR.load_run_config(a.run_config).rounds
    except (OSError, ValueError) as e:
        raise X2TestGateError(f"--run-config {a.run_config} 读不了：{e}") from e
    return check_test_gate(a.test_gate, a.test_approval, checkpoint=a.checkpoint, model=a.model, settings=a.settings, rounds=rounds)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="注册键（src/adapters/registry.py）")
    p.add_argument("--checkpoint", required=True, help="X-2 的检查点 <训练叶目录>/ckpt/final/model（scripts/train_w2c.py --config "
                                                        "configs/x2_train.json；DEV-0070：或 configs/x3_train.json，只训了样条，修正类设置跳过）")
    p.add_argument("--split", required=True, help="划分名（--splits-file 的 splits；test 一律拒绝）")
    p.add_argument("--splits-file", default=str(W2_SPLITS_PATH), help="划分清单（默认 configs/w2_splits.json；探索：configs/explore_splits.json）")
    p.add_argument("--unseal-devcheck3", metavar="GATE", help=f"密封的 devcheck3 只在判定时用：门名必须是 {EX.GATE}")
    p.add_argument("--out", required=True, help="本次运行的目录（experiments/X/<NNNN_slug>）")
    p.add_argument("--data-root", help="落地目录（默认环境变量 VISINT_DATA_ROOT；--tiny 时取 fixture）")
    p.add_argument("--w2-config", default=str(W2_CONFIG_PATH), help="默认 configs/w2.json（必须与检查点训练时的相同）")
    p.add_argument("--run-config", default=str(XR.X2_RUN_CONFIG_PATH), help="默认 configs/x2_run.json")
    p.add_argument("--settings", nargs="+", help="只导出这几个设置（轨迹照样全跑）")
    p.add_argument("--candidate-workers", type=int, default=1, help="候选生成的进程数（同 run_w2）")
    p.add_argument("--tool-workers", type=int, default=1, help="样条 trimap、局部修正的进程数（spawn，子进程单线程，结果与单进程相同）")
    p.add_argument("--limit", type=int, help="只取划分的前 N 条（调试）")
    p.add_argument("--resume", action="store_true", help="续跑：跳过已完成的组（身份必须相同）")
    p.add_argument("--tiny", action="store_true", help="小型随机模型 + CPU + tests/fixtures 的小数据（开发机彩排）")
    p.add_argument("--device", help="默认：真模型 cuda，--tiny 时 cpu")
    p.add_argument("--dtype", choices=sorted(DTYPES), help="默认：真模型 bf16，--tiny 时 fp32")
    p.add_argument("--candidates-module", default=R.DEFAULT_CANDIDATES_MODULE, help="候选生成器的模块（非默认只配 --tiny）")
    p.add_argument("--render-module", default=R.DEFAULT_RENDER_MODULE, help="挑选图渲染的模块（非默认只配 --tiny）")
    p.add_argument("--merge-check-n", type=int, help="合并核对的开发样本数（默认 configs/e3b_train.json 的 eval.merge_check.n）")
    p.add_argument("--test-gate", metavar="GATE_JSON", help="test 划分（D147）：scripts/gate_explore.py 写的 gate_explore.json（通过、判定点、"
                                                            "候选检查点与 --checkpoint 相同）；与 --test-approval 一起给")
    p.add_argument("--test-approval", metavar="ID", help=f"test 划分：负责人的批准编号，必须恰是 {TEST_APPROVAL}")
    a = p.parse_args(argv)
    if a.model not in registry.MODELS:
        p.error(f"未登记的模型键 {a.model!r}")
    if is_test_split(a.split) and a.test_gate is None and a.test_approval is None:
        p.error(f"X-2 是探索：不碰 test（预注册 X §1），拒绝 --split {a.split}")
    if not a.tiny and (a.candidates_module != R.DEFAULT_CANDIDATES_MODULE or a.render_module != R.DEFAULT_RENDER_MODULE):
        p.error("--candidates-module、--render-module（非默认）只配 --tiny")
    if not Path(a.checkpoint).is_dir():
        p.error(f"--checkpoint {a.checkpoint!r} 不是目录")
    for name in ("limit", "candidate_workers", "tool_workers", "merge_check_n"):
        v = getattr(a, name)
        if v is not None and v < 1:
            p.error(f"--{name.replace('_', '-')} 必须是正整数")
    if a.settings:
        known = XR.setting_defs(XR.load_run_config(a.run_config).rounds)
        bad = [s for s in a.settings if s not in known]
        if bad:
            p.error(f"未知的设置 {bad}；可选 {list(known)}")
    if not a.tiny and not (a.data_root or os.environ.get("VISINT_DATA_ROOT")):
        p.error("真模型运行要给 --data-root（或设 VISINT_DATA_ROOT）")
    if not a.tiny and not RUN_DIR_NAME.fullmatch(Path(a.out).name):
        p.error(f"--out 的目录名必须带运行编号，形如 0081_x2_eval，收到 {a.out!r}")
    if not a.tiny and (needs_frozen(a.split) or EX.needs_frozen(a.split)):
        try:
            frozen = load_w2_config(a.w2_config).data["frozen"]
        except (OSError, ValueError) as e:
            p.error(f"--w2-config {a.w2_config} 读不了：{e}")
        if frozen is not True:
            p.error(f"--split {a.split} 要求 {a.w2_config} 已冻结（frozen = true）")
    if EX.is_sealed(a.split):  # 密封守卫（--tiny 也查）
        try:
            EX.require_unsealed(a.split, a.unseal_devcheck3)
        except EX.SealedSplitError as e:
            p.error(str(e))
    elif a.unseal_devcheck3 is not None:
        p.error(f"--unseal-devcheck3 只配密封的 devcheck3 划分（--split {a.split} 不是）")
    if is_test_split(a.split):  # test 的放行（DEV-0072，D147）：--tiny 也查
        try:
            check_test_split_args(a)
        except X2TestGateError as e:
            p.error(str(e))
    elif a.test_gate is not None or a.test_approval is not None:
        p.error(f"--test-gate、--test-approval 只配 test 划分（--split {a.split} 不是）")
    a.device = a.device or ("cpu" if a.tiny else "cuda")
    a.dtype = a.dtype or ("fp32" if a.tiny else "bf16")
    a.task = TASK
    return a


def cmd(a, argv) -> int:
    cfg = load_w2_config(a.w2_config)
    rc = XR.load_run_config(a.run_config)
    splits_doc = RW2.load_splits_doc(a, cfg)
    unlock = None
    if is_test_split(a.split):  # parse_args 已核对；这里再核对一次（直接调用 cmd 时）
        if getattr(a, "test_gate", None) is None and getattr(a, "test_approval", None) is None:
            raise XR.X2RunError("X-2 不碰 test")
        unlock = check_test_split_args(a)
        if not a.tiny and cfg.data.get("frozen") is not True:
            raise ValueError(f"test 划分只能用冻结的配置跑：{a.w2_config} 的 frozen 不是 true")
    if a.split not in split_names(TASK, splits_doc):
        raise ValueError(f"--split {a.split} 不用于 {TASK}；可用：{split_names(TASK, splits_doc)}")
    unseal = EX.require_unsealed(a.split, a.unseal_devcheck3, entry=splits_doc["splits"][a.split])
    if unseal is not None:
        if not a.tiny and cfg.data.get("frozen") is not True:
            raise ValueError(f"密封的 {a.split} 只能用冻结的配置跑")
        a.unseal = EX.seal_identity(unseal)
        log = Path(a.out) / a.model / "unseal_devcheck3.jsonl"
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "a", encoding="utf-8") as f:
            f.write(json.dumps({**unseal, "split": a.split, "task": TASK, "setting": "x2", "argv": list(argv)}, ensure_ascii=False) + "\n")
    if unlock is not None:  # test 的放行（D147）：身份进 w2.test_unlock，放行记录追加进 test_unlock.jsonl
        a.test_unlock = unlock["identity"]
        log = Path(a.out) / a.model / TEST_UNLOCK_LOG
        log.parent.mkdir(parents=True, exist_ok=True)
        rec = {**unlock["identity"], "gate_explore_resolved": str(Path(a.test_gate).resolve()), "gate": unlock["gate"],
               "checkpoint_manifest_sha256": unlock["checkpoint_manifest_sha256"],
               "unlocked_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "split": a.split, "task": TASK,
               "setting": "x2", "settings": list(a.settings), "argv": list(argv)}
        with open(log, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    x2 = XR.checkpoint_x2_params(a.checkpoint)
    paths = RW2.data_paths(a, splits_doc)
    samples = w2_items(TASK, a.split, paths, splits=splits_doc)
    if a.limit is not None:
        samples = samples[: a.limit]
    prompts = load_prompt_config(PROMPTS_PATH)
    entry = registry.get_model(a.model)
    W2C.refuse_mixing(a.out, entry.key, a.split, a.checkpoint)
    ad, ck_info, extra = W2C.load_checkpoint_adapter(
        a.checkpoint, model_key=a.model, tiny=a.tiny, device=a.device, dtype=a.dtype, torch_dtype=DTYPES[a.dtype], w2=cfg,
        merge_check_json=Path(a.out) / a.model / W2C.MERGE_CHECK_JSON, data_root=a.data_root or os.environ.get("VISINT_DATA_ROOT"),
        merge_check_n=a.merge_check_n)
    pol = ad.image_policy
    spec = R.W2RunSpec(model_key=entry.key, family=entry.family, repo=entry.repo, revision=entry.revision,
                       adapter_model_key=str(getattr(ad, "model_key", entry.key)), task=TASK, split=a.split, setting="x2", tiny=a.tiny,
                       limit=a.limit, image_policy={"tag": "checkpoint", "type": type(pol).__name__, **pol.as_dict()},
                       unseal=getattr(a, "unseal", None), test_unlock=getattr(a, "test_unlock", None))
    frozen_bs = prompts.batch_protocol()["batch_size"]
    impl = R.load_impl(a.candidates_module, a.render_module)
    summ = XR.run_split(ad, entry, spec, cfg, rc, samples, paths, a.out, checkpoint=ck_info, x2=x2, splits_doc=splits_doc, prompts=prompts,
                        impl=impl, e1_batch_size=TINY_E1_BATCH_SIZE if a.tiny else frozen_bs, resume=a.resume, tool_workers=a.tool_workers,
                        candidate_workers=a.candidate_workers, argv=argv, run_extra=extra, settings=a.settings)
    head = "; ".join(f"{k}={v['cIoU']}" for k, v in summ["settings"].items())
    sk = summ.get("skipped")
    tail = "" if sk is None else f"（跳过 {len(sk['settings'])} 个设置：{sk['reason']}）"
    print(f"X-2 {a.model} {a.split} n={summ['n']} {head}{tail}")
    return 0


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    a = parse_args(argv)
    from src.e3.merge_check import MergeCheckFailed

    try:
        return cmd(a, argv)
    except MergeCheckFailed as e:
        print(f"拒绝评测：{e}", file=sys.stderr)
        return 1
    except Exception as e:  # noqa: BLE001 — 证据先落盘，再以非零码退出
        err_dir = Path(a.out) / a.model / "errors"
        err_dir.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        path = err_dir / f"error_{stamp}.json"
        path.write_text(json.dumps({"argv": argv, "error": repr(e), "traceback": traceback.format_exc()[-6000:]},
                                   indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"失败：{e!r}（已写出 {path}）", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
