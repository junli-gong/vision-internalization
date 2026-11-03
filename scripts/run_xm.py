#!/usr/bin/env python
"""X-M 的推理与评测（DEV-0080；指令第三、四、五节；D165）：一个模型 × 一个划分，一次跑完一个系统的全部设置（模型只加载一次，各设置共用
模型输出）。规则见 src/xm/runner.py（流程、设置、记账）、src/xm/prompts.py（两次调用）、src/xm/post.py（NMS、贴边）、src/xm/metrics.py
（主读数 f1_rexomni，XM-3a）、src/xm/gate.py（密封）。

输出：<out>/<模型>/xm/<设置>/<划分>/（predictions.jsonl、metrics.json、compute.json、config.yaml、meta.json、README.md）；轨迹
<out>/<模型>/xm/{xm_trace,base_trace}/<划分>/；汇总 <out>/<模型>/xm/xm_summary/<划分>__<轨迹>.json。

系统与设置（--setting 不给时跑这个系统的全部设置）：
- X-M 的检查点（--checkpoint <训练叶目录>/ckpt/final/model，DEV-0081）：xm_spline_trimap、xm_box_grabcut、xm_ellipse_trimap、xm_spline_raster；
- 基座零样本（不给 --checkpoint）：base_det、base_box_grabcut。一次只跑一个系统。

服务器（HPC；$VI/data 见 DEV-0079；--out 的末级目录名必须以四位运行编号开头）：
    # 调参集（xm_tune，300 张）与判定集（xm_decision，500 张，密封：要 --gate G-XM，且 src/xm/prompts.FROZEN = True）
    python scripts/run_xm.py --model qwen3.5-9b --split xm_tune --data-root $VI/data --out experiments/XM/<NNNN>_xm_base_tune \\
        --tool-workers 8 [--resume]                                                  # 基座：base_det、base_box_grabcut
    python scripts/run_xm.py --model qwen3.5-9b --checkpoint $CK --split xm_tune --data-root $VI/data \\
        --out experiments/XM/<NNNN>_xm_tune --tool-workers 8 [--resume]               # X-M：4 个 xm_* 设置
    python scripts/run_xm.py --model qwen3.5-9b [--checkpoint $CK] --split xm_decision --gate G-XM --data-root $VI/data \\
        --out experiments/XM/<NNNN>_xm_decision --tool-workers 8
    # 最终报告集、LVIS 子集（判定通过之后）：--xm-gate <DEV-0081 写的 gate_xm.json>
    python scripts/run_xm.py --model qwen3.5-9b --checkpoint $CK --split xm_final --xm-gate <gate_xm.json> --data-root $VI/data \\
        --out experiments/XM/<NNNN>_xm_final --tool-workers 8
    # ReasonSeg val 的指代分割（X-2 的原协议：src/explore/x2_runner.run_split，导出 x2_spline_trimap；W2 格式，输出在
    # <out>/<模型>/refseg/<设置>/reasonseg_val/；扣掉忽略区域的 gIoU、cIoU 另写 <out>/<模型>/reasonseg_readout/reasonseg_val.json）。
    # --system x2：X-2 的检查点（描述性，不要 gate）；--system xm：X-M 的检查点（要 --xm-gate）。
    # 两个系统用不同的 --out（同一个 --out 里不混两个检查点）。
    python scripts/run_xm.py --model qwen3.5-9b --checkpoint $X2CK --split reasonseg_val --system x2 --data-root $VI/data \\
        --out experiments/XM/<NNNN>_reasonseg_x2 --candidate-workers 8 --tool-workers 8
开发机彩排（--tiny：小型随机模型、CPU）：数据是 --pairs-json 给的对（tests/_xm_run_fakes.write_fixture），或 DEV-0079 的假数据根经
load_pairs（与真运行同一条路径：python -m tests._xm_fakes <root>，scripts/xm/build_configs.py --tiny 写划分清单）：
    python scripts/run_xm.py --tiny --model qwen3.5-0.8b --split xm_tune --pairs-json <dir>/pairs.json --out <dir>/0001_xm_tiny
    python scripts/run_xm.py --tiny --model qwen3.5-0.8b --split xm_tune --data-root <root> --splits-file <dir>/xm_splits.json \
        --out <dir>/0002_xm_tiny
退出码：0 = 跑完；1 = 检查点的合并核对没有通过，拒绝评测；2 = 出错（报错先写进 <out>/<模型>/errors/）或用法错误（含密封与 gate 不合格：
什么都不写）。
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from src.adapters import registry  # noqa: E402
from src.e1.prompts import PROMPTS_PATH, load_prompt_config  # noqa: E402
from src.xm import data as XD  # noqa: E402
from src.xm import gate as XG  # noqa: E402
from src.xm import runner as XR  # noqa: E402

DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}
RUN_DIR_NAME = XD.RUN_DIR_RE  # 末级目录名以四位运行编号开头（同 src/xm/data.check_run_dir）
EVAL_SPLITS = tuple(s for s in XD.SPLITS if s != "train")
SYSTEMS_REASONSEG = ("x2", "xm")
REASONSEG_DEFAULT_SETTINGS = ("x2_spline_trimap",)
TINY_INIT_STD = 0.05  # 同 scripts/run_e1.py
TINY_E1_BATCH_SIZE = 4  # 同 scripts/run_x2.py（ReasonSeg 的提出）
MERGE_CHECK_JSON = "xm_merge_check.json"


def _x2_runner():
    from src.explore import x2_runner as X2

    return X2


def system_of_args(a) -> str:
    if a.split == XG.REASONSEG:
        return a.system
    if a.setting:
        return XR.system_of(a.setting)
    return "xm" if a.checkpoint else "base"


def check_gates(a, system: str) -> dict:
    """密封与 gate（parse_args 与 cmd 共用）：不合格抛 XMGateError。返回 {"unseal", "xm_gate"}（放行记录或 None）。"""
    unseal = XG.require_unsealed(a.split, a.gate)
    xm_gate = None
    if XG.needs_xm_gate(a.split, system):
        xm_gate = XG.check_xm_gate(a.xm_gate, checkpoint=a.checkpoint if system == "xm" else None)
    elif a.xm_gate is not None:
        raise XG.XMGateError(f"--xm-gate 只配 {list(XG.GATED)} 与 X-M 检查点的 {XG.REASONSEG}（--split {a.split}、系统 {system} 不要）")
    if system != "x2":
        XG.require_frozen_prompts(a.split, tiny=a.tiny)
    return {"unseal": unseal, "xm_gate": xm_gate}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="注册键（src/adapters/registry.py）")
    p.add_argument("--checkpoint", help="X-M 的检查点 <训练叶目录>/ckpt/final/model（base_* 设置不配）；ReasonSeg 的 --system x2 时是 X-2 的检查点")
    p.add_argument("--split", required=True, choices=EVAL_SPLITS, help="划分（src/xm/data.load_pairs）")
    p.add_argument("--setting", nargs="+", help="只导出这几个设置（轨迹照样全跑）；ReasonSeg 时是 X-2 的设置名（缺省 x2_spline_trimap）")
    p.add_argument("--system", choices=SYSTEMS_REASONSEG, help="只配 reasonseg_val：x2 = X-2 的检查点，xm = X-M 的检查点")
    p.add_argument("--data-root", help="落地目录（默认环境变量 VISINT_DATA_ROOT；--tiny 时取 --pairs-json 所在目录）")
    p.add_argument("--pairs-json", help="只配 --tiny：对的 JSON（src/xm/runner.pairs_from_json；tests/_xm_run_fakes.write_fixture 写）")
    p.add_argument("--splits-file", help="划分清单（默认 configs/xm_splits.json；src/xm/data.load_pairs 的 splits_path）")
    p.add_argument("--out", required=True, help="本次运行的目录（experiments/XM/<NNNN_slug>）")
    p.add_argument("--limit", type=int, help="只取划分的前 N 个对（调试）")
    p.add_argument("--resume", action="store_true", help="续跑：跳过已完成的组（身份必须相同）")
    p.add_argument("--tiny", action="store_true", help="小型随机模型 + CPU（开发机彩排）")
    p.add_argument("--device", help="默认：真模型 cuda，--tiny 时 cpu")
    p.add_argument("--dtype", choices=sorted(DTYPES), help="默认：真模型 bf16，--tiny 时 fp32")
    p.add_argument("--candidate-workers", type=int, default=1, help="ReasonSeg（X-2 的运行器）的候选生成进程数（同 run_x2）")
    p.add_argument("--tool-workers", type=int, default=1, help="贴边工具的进程数（spawn，子进程单线程，结果与单进程相同；同 run_x2）")
    p.add_argument("--gate", metavar="GATE", help=f"密封的 xm_decision 只在判定时用：门名必须是 {XG.GATE}")
    p.add_argument("--xm-gate", metavar="GATE_XM_JSON", help="xm_final、lvis_sub、X-M 检查点的 reasonseg_val：DEV-0081 写的 gate_xm.json")
    p.add_argument("--w2-config", help="默认 configs/w2.json（X-M 检查点的核对与 ReasonSeg；必须与训练时的相同）")
    p.add_argument("--run-config", help="ReasonSeg：X-2 的运行配置（默认 configs/x2_run.json）")
    p.add_argument("--merge-check-n", type=int, help="合并核对的开发样本数（默认 configs/e3b_train.json 的 eval.merge_check.n）")
    p.add_argument("--merge-check-data-root", help="合并核对读 E1 开发样本的数据根（缺省 VISINT_DATA_ROOT，再缺省 --data-root）")
    p.add_argument("--candidates-module", help="ReasonSeg：候选生成器的模块（非默认只配 --tiny）")
    p.add_argument("--render-module", help="ReasonSeg：挑选图渲染的模块（非默认只配 --tiny）")
    a = p.parse_args(argv)
    if a.model not in registry.MODELS:
        p.error(f"未登记的模型键 {a.model!r}")
    if not RUN_DIR_NAME.fullmatch(Path(a.out).name):
        p.error(f"--out 的末级目录名必须以四位运行编号开头，形如 0103_xm_tune，收到 {a.out!r}")
    for name in ("limit", "candidate_workers", "tool_workers", "merge_check_n"):
        v = getattr(a, name)
        if v is not None and v < 1:
            p.error(f"--{name.replace('_', '-')} 必须是正整数")
    if a.pairs_json is not None and not a.tiny:
        p.error("--pairs-json 只配 --tiny（真运行经 src/xm/data.load_pairs）")
    if a.tiny and a.pairs_json is None and a.data_root is None:
        p.error("--tiny 要给 --pairs-json，或 --data-root（加 --splits-file：tests/_xm_fakes 的假数据根，经 load_pairs）")
    if a.pairs_json is not None and (a.data_root is not None or a.splits_file is not None):
        p.error("--pairs-json 不与 --data-root、--splits-file 同给")
    if not a.tiny and not (a.data_root or os.environ.get("VISINT_DATA_ROOT")):
        p.error("真模型运行要给 --data-root（或设 VISINT_DATA_ROOT）")
    if not a.tiny and (a.candidates_module or a.render_module):
        p.error("--candidates-module、--render-module 只配 --tiny")
    if a.checkpoint is not None and not Path(a.checkpoint).is_dir():
        p.error(f"--checkpoint {a.checkpoint!r} 不是目录")
    if a.split == XG.REASONSEG:
        if a.system is None:
            p.error(f"--split {XG.REASONSEG} 要给 --system（x2 = X-2 的检查点，xm = X-M 的检查点）")
        if a.checkpoint is None:
            p.error(f"--split {XG.REASONSEG} 要给 --checkpoint（--system {a.system} 的检查点）")
        X2 = _x2_runner()
        known = X2.setting_defs(X2.load_run_config(a.run_config or X2.X2_RUN_CONFIG_PATH).rounds)
        bad = [s for s in (a.setting or ()) if s not in known]
        if bad:
            p.error(f"ReasonSeg 走 X-2 的协议：未知的 X-2 设置 {bad}")
    else:
        if a.system is not None:
            p.error(f"--system 只配 --split {XG.REASONSEG}")
        try:
            system = system_of_args(a)
        except XR.XMRunError as e:
            p.error(str(e))
        if system == "xm" and a.checkpoint is None:
            p.error(f"xm_* 设置要 --checkpoint（X-M 的检查点）：{a.setting}")
        if system == "base" and a.checkpoint is not None:
            p.error(f"base_* 设置不配检查点（基座零样本，不加 LoRA）：{a.setting}")
    try:
        check_gates(a, system_of_args(a))
    except XG.XMGateError as e:
        p.error(str(e))
    a.device = a.device or ("cpu" if a.tiny else "cuda")
    a.dtype = a.dtype or ("fp32" if a.tiny else "bf16")
    return a


# ---------------------------------------------------------------------------
# 数据与模型
# ---------------------------------------------------------------------------


def data_root(a) -> Path:
    if a.data_root:
        return Path(a.data_root)
    if a.pairs_json is not None:
        return Path(a.pairs_json).resolve().parent
    return Path(os.environ["VISINT_DATA_ROOT"])


def load_split_pairs(a, gates: dict) -> list:
    """对：--pairs-json（只配 --tiny）读 JSON；否则 src/xm/data.load_pairs（DEV-0079；它对 xm_decision 再核一次门名，对 xm_final 与 lvis_sub
    要 check_gates 认过的 XMGatePass）。"""
    if a.pairs_json is not None:
        pairs = XR.pairs_from_json(a.pairs_json)
    else:
        xg = gates["xm_gate"]["pass"] if gates["xm_gate"] is not None and a.split in XG.GATED else None
        pairs = XD.load_pairs(a.split, data_root(a), gate=a.gate, xm_gate=xg, splits_path=a.splits_file)
    wrong = sorted({p.split for p in pairs} - {a.split})
    if wrong:
        raise XR.XMRunError(f"对的 split {wrong} 不是 --split {a.split}")
    return pairs[: a.limit] if a.limit is not None else pairs


def merge_check_root(a) -> str | None:
    """评测前合并核对（src/e3/merge_check，E1 的开发样本）的数据根：--tiny 时用它自己的 fixture（None）；真运行取 --merge-check-data-root，
    缺省环境变量 VISINT_DATA_ROOT，再缺省 --data-root（X-M 的数据根不一定有 E1 的开发数据）。"""
    if a.tiny:
        return None
    return a.merge_check_data_root or os.environ.get("VISINT_DATA_ROOT") or a.data_root


def log_unlocks(a, model_key: str, gates: dict, argv) -> None:
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    if gates["unseal"] is not None:
        XG.append_log(a.out, model_key, XG.UNSEAL_LOG, {**gates["unseal"], "split": a.split, "argv": list(argv)})
    if gates["xm_gate"] is not None:
        XG.append_log(a.out, model_key, XG.UNLOCK_LOG, {**gates["xm_gate"]["identity"], "gate": gates["xm_gate"]["summary"],
                                                        "resolved": str(Path(a.xm_gate).resolve()), "unlocked_utc": now, "split": a.split,
                                                        "argv": list(argv)})


def _w2_config(a):
    from src.w2.config import W2_CONFIG_PATH, load_w2_config

    return load_w2_config(a.w2_config or W2_CONFIG_PATH)


def cmd_xm(a, argv) -> int:
    from src.adapters.factory import default_image_policy, make_adapter
    from src.runlog import sha256_file

    system = system_of_args(a)
    gates = check_gates(a, system)
    entry = registry.get_model(a.model)
    pairs = load_split_pairs(a, gates)
    prompts = load_prompt_config(PROMPTS_PATH)
    ck_info = None
    if system == "xm":
        from src.w2 import w2c as W2C

        XR.refuse_mixing(a.out, entry.key, a.split, sha256_file(Path(a.checkpoint) / "manifest.json"))
        xm_params = XR.checkpoint_xm_params(a.checkpoint)
        ad, ck_info, _ = W2C.load_checkpoint_adapter(
            a.checkpoint, model_key=a.model, tiny=a.tiny, device=a.device, dtype=a.dtype, torch_dtype=DTYPES[a.dtype], w2=_w2_config(a),
            merge_check_json=Path(a.out) / entry.key / MERGE_CHECK_JSON, data_root=merge_check_root(a),
            merge_check_n=a.merge_check_n)
        ck_info = {**ck_info, "xm": xm_params}
        pol, tag = ad.image_policy, "checkpoint"
    else:
        pol, tag = default_image_policy(entry.family), "default"
        ad = make_adapter(a.model, device=a.device, dtype=DTYPES[a.dtype], image_policy=pol, tiny=a.tiny,
                          **({"init_std": TINY_INIT_STD} if a.tiny else {}))
    log_unlocks(a, entry.key, gates, argv)
    model = {"key": entry.key, "family": entry.family, "repo": entry.repo, "revision": entry.revision,
             "adapter_model_key": str(getattr(ad, "model_key", entry.key)), "tiny": bool(a.tiny)}
    summ = XR.run_split(ad, pairs, a.out, system=system, split=a.split, model=model,
                        image_policy={"tag": tag, "type": type(pol).__name__, **pol.as_dict()}, checkpoint=ck_info, prompts=prompts,
                        data_root=data_root(a), tiny=a.tiny, list_batch_size=prompts.batch_protocol()["batch_size"], resume=a.resume,
                        tool_workers=a.tool_workers, settings=a.setting, limit=a.limit, unseal=XG.seal_identity(gates["unseal"]),
                        xm_gate=None if gates["xm_gate"] is None else gates["xm_gate"]["identity"],
                        files=[f for f in (a.pairs_json, a.splits_file) if f is not None], argv=argv)
    head = "; ".join(f"{k}: det {v['det']['f1_rexomni']}" + ("" if v["seg"] is None else f", seg {v['seg']['f1_rexomni']}")
                     for k, v in summ["settings"].items())
    print(f"X-M {a.model} {a.split} n={summ['n_pairs']} {head}")
    return 0


def reasonseg_splits_doc(a, model_key: str, samples) -> object:
    """X-2 的运行器要的 W2 格式划分清单（只有 reasonseg_val 一项）：写进 <out>/<模型>/reasonseg_splits.json，内容只由对的 id 定。"""
    from src.data.subsets import population_digest
    from src.w2.data import SplitsDoc

    doc = {"version": 1, "source": "src/xm/data.load_pairs（DEV-0079）经 scripts/run_xm.py（DEV-0080）",
           "splits": {a.split: {"tasks": ["refseg", "refexp_box"], "status": "built", "n": len(samples),
                                "sha256": population_digest([s.id for s in samples])}}}
    text = json.dumps(doc, indent=1, ensure_ascii=False) + "\n"
    path = Path(a.out) / model_key / "reasonseg_splits.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise XR.XMRunError(f"{path} 已有、且与本次的对不同：换一个 --out")
    path.write_text(text, encoding="utf-8")
    sd = SplitsDoc(doc)
    sd.path = path
    return sd


def cmd_reasonseg(a, argv) -> int:
    """ReasonSeg val 的指代分割：X-2 的原协议（src/explore/x2_runner.run_split，同 scripts/run_x2.py 的 cmd），检查点按 --system。"""
    from src.e1.tasks import DataPaths
    from src.w2 import runner as R
    from src.w2 import w2c as W2C

    X2 = _x2_runner()
    gates = check_gates(a, a.system)
    cfg = _w2_config(a)
    rc = X2.load_run_config(a.run_config or X2.X2_RUN_CONFIG_PATH)
    x2 = X2.checkpoint_x2_params(a.checkpoint)
    if a.system == "xm":
        XR.checkpoint_xm_params(a.checkpoint)
    entry = registry.get_model(a.model)
    pairs = load_split_pairs(a, gates)
    samples = XR.reasonseg_samples(pairs, data_root(a))
    splits_doc = reasonseg_splits_doc(a, entry.key, samples)
    prompts = load_prompt_config(PROMPTS_PATH)
    W2C.refuse_mixing(a.out, entry.key, a.split, a.checkpoint)
    ad, ck_info, extra = W2C.load_checkpoint_adapter(
        a.checkpoint, model_key=a.model, tiny=a.tiny, device=a.device, dtype=a.dtype, torch_dtype=DTYPES[a.dtype], w2=cfg,
        merge_check_json=Path(a.out) / entry.key / W2C.MERGE_CHECK_JSON, data_root=merge_check_root(a),
        merge_check_n=a.merge_check_n)
    log_unlocks(a, entry.key, gates, argv)
    unseal = None if gates["xm_gate"] is None else {"gate": f"{XG.GATE} 通过（gate_xm.json）", "xm_gate": gates["xm_gate"]["identity"]}
    pol = ad.image_policy
    spec = R.W2RunSpec(model_key=entry.key, family=entry.family, repo=entry.repo, revision=entry.revision,
                       adapter_model_key=str(getattr(ad, "model_key", entry.key)), task="refseg", split=a.split, setting="x2", tiny=a.tiny,
                       limit=a.limit, image_policy={"tag": "checkpoint", "type": type(pol).__name__, **pol.as_dict()}, unseal=unseal)
    impl = R.load_impl(a.candidates_module or R.DEFAULT_CANDIDATES_MODULE, a.render_module or R.DEFAULT_RENDER_MODULE)
    summ = X2.run_split(ad, entry, spec, cfg, rc, samples, DataPaths(root=data_root(a)), a.out, checkpoint=ck_info, x2=x2,
                        splits_doc=splits_doc, prompts=prompts, impl=impl,
                        e1_batch_size=TINY_E1_BATCH_SIZE if a.tiny else prompts.batch_protocol()["batch_size"], resume=a.resume,
                        tool_workers=a.tool_workers, candidate_workers=a.candidate_workers, argv=argv, run_extra=extra,
                        settings=list(a.setting or REASONSEG_DEFAULT_SETTINGS))
    rs = {}
    for name in summ["settings"]:
        leaf = R.w2_leaf(a.out, entry.key, "refseg", name, a.split)
        rs[name] = XR.reasonseg_readout([json.loads(x) for x in (leaf / "predictions.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()],
                                        pairs)
    rp = Path(a.out) / entry.key / "reasonseg_readout" / f"{a.split}.json"
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps({"system": a.system, "checkpoint": ck_info.get("manifest_sha256"), "settings": rs}, indent=1, ensure_ascii=False)
                  + "\n", encoding="utf-8")
    head = "; ".join(f"{k}: cIoU {v['with_ignore']['cIoU']}, gIoU {v['with_ignore']['gIoU']}" for k, v in rs.items())
    print(f"ReasonSeg（--system {a.system}）{a.model} n={summ['n']} {head}")
    return 0


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    a = parse_args(argv)
    from src.e3.merge_check import MergeCheckFailed

    try:
        return cmd_reasonseg(a, argv) if a.split == XG.REASONSEG else cmd_xm(a, argv)
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
