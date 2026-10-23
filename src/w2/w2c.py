"""W2-c 的评测接入（DEV-0061；D113 第二节第 4 条；docs/w2_plan.md §5）：scripts/run_w2.py --setting w2c --checkpoint <检查点>。

- 检查点（load_checkpoint_adapter）：口径同 E3 的 T 文本评测（scripts/run_e1.py --checkpoint 的 build_checkpoint_adapter）：
  manifest 的构造方式必须与 --model、--tiny 相符 → 评测前的合并核对（src/e3/merge_check.run_merge_check，检查项与 n 取
  configs/e3b_train.json 的 eval.merge_check 的 t_text 口径；不通过抛 MergeCheckFailed，拒绝评测）→ factory.load_trained(merge=True)
  （LoRA 状态必须是 merged）→ 非 --tiny 时检查点的图像口径必须是 E1 冻结的口径（D28-3）。另核对它是 W2-c 的训练结果：检查点在
  <叶目录>/ckpt/final/model 下，叶目录 config.yaml 的 variant 是 w2c，且训练用的 W2 配置的 sha256 等于本次评测用的那一份（候选生成器、
  渲染、K 都是 W2-a 冻结的同一份）。检查点的记录（checkpoint_record：路径、manifest 的 sha256、LoRA、合并核对的口径、训练叶目录的
  变体 / 种子 / 运行身份 sha256）写进提出与挑选两个目录的 config.yaml 的 checkpoint。
- 提出（ensure_proposal）：训练过的模型按 E1 refexp_box 协议重新出框（E1 的运行器原样：关思考、贪心、E1 冻结的批大小、图在前），输出放在
  <out>/<模型>/refexp_box/propose_w2c/<划分>，与 base 9B 的 baseline_e1 分开；已写完且是同一个检查点就直接读，检查点不同就拒绝混用。
- 候选与挑选：src/w2/runner.ensure_candidates(cand_setting = candidates_w2c) 与 run_candidate_setting(setting = w2c)：候选生成器、渲染、K
  都是 W2-a 冻结的那一份（同一个 --w2-config）；挑选关思考（W2Config.select_protocol_for("w2c")）；"都不对"或解析失败交 1 号。
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml

from src.data.records import Sample
from src.e1.prompts import PromptConfig
from src.e1.runner import CONFIG, PREDICTIONS, read_jsonl_tolerant, run_task
from src.e1.tasks import DataPaths
from src.w1 import baselines as B
from src.w2 import runner as R
from src.w2.config import W2C_PROPOSE_SETTING, W2Config

BOX_TASK = "refexp_box"
W2C_VARIANT = "w2c"  # src/w2/w2c_train.VARIANT（这里不 import 训练模块：评测不需要它的依赖）
MERGE_CHECK_JSON = "w2c_merge_check.json"  # <out>/<模型>/w2c_merge_check.json


class W2CCheckpointError(ValueError):
    """检查点不是 W2-c 的训练结果，或与本次评测的 W2 配置、模型对不上。"""


def training_record(checkpoint: str | Path) -> dict[str, Any]:
    """检查点所在的训练叶目录（<叶>/ckpt/final/model）的 config.yaml 与 train_summary.json → 变体、种子、运行身份 sha256、W2 配置 sha256。
    不在这个布局下、或叶目录没有 config.yaml 时报错（核对不了它是不是 W2-c）。"""
    from src.e3.train import checkpoint_leaf, identity_sha256

    leaf = checkpoint_leaf(checkpoint)
    if leaf is None or not (leaf / CONFIG).is_file():
        raise W2CCheckpointError(f"检查点 {checkpoint} 不在 <训练叶目录>/ckpt/final/model 下，或叶目录没有 {CONFIG}：核对不了它是不是 W2-c")
    doc = yaml.safe_load((leaf / CONFIG).read_text(encoding="utf-8")) or {}
    out = {"variant": doc.get("variant"), "seed": doc.get("seed"), "identity_sha256": identity_sha256(doc),
           "w2_config_sha256": (doc.get("w2_config") or {}).get("config_sha256"), "config_sha256": (doc.get("config") or {}).get("sha256")}
    s = leaf / "train_summary.json"
    if s.is_file():
        sm = json.loads(s.read_text(encoding="utf-8"))
        out.update(complete=sm.get("complete"), steps=sm.get("steps"))
    return out


def check_training(training: Mapping[str, Any], w2: W2Config) -> None:
    if training.get("variant") != W2C_VARIANT:
        raise W2CCheckpointError(f"检查点来自变体 {training.get('variant')!r}，--setting w2c 要 W2-c 的训练结果（variant = {W2C_VARIANT}）")
    if training.get("w2_config_sha256") != w2.sha256:
        raise W2CCheckpointError(f"检查点训练时用的 W2 配置 sha256 {str(training.get('w2_config_sha256'))[:16]}… 不是本次评测的 "
                                 f"{w2.identity()['config']}（{w2.sha256[:16]}…）：W2-c 的候选、渲染、K 必须与训练时是同一份")


def checkpoint_record(ck: Path, man: Mapping[str, Any], *, merge_summary: Mapping[str, Any], n_grid_tokens: int,
                      training: Mapping[str, Any]) -> dict[str, Any]:
    """config.yaml 的 checkpoint 一节（键同 run_e1 --checkpoint 的记录，另加路径与训练叶目录的摘要）。"""
    from src.runlog import sha256_file

    return {"path": str(Path(ck).resolve()), "manifest_sha256": sha256_file(Path(ck) / "manifest.json"), "label": W2C_VARIANT,
            "build_spec": man.get("build_spec"), "image_policy": man.get("image_policy"), "dtype": man.get("dtype"),
            "lora": man.get("lora"), "trainable_token_rows": man.get("trainable_token_rows"), "n_grid_tokens": int(n_grid_tokens),
            "merge_check": {"criteria_version": merge_summary["criteria_version"], "n": merge_summary["n"],
                            "checks": merge_summary["checks"]},
            "training": dict(training)}


def load_checkpoint_adapter(checkpoint: str | Path, *, model_key: str, tiny: bool, device: str, dtype: str, torch_dtype: Any,
                            w2: W2Config, merge_check_json: str | Path, data_root: str | Path | None, merge_check_n: int | None = None,
                            log: Callable[[str], None] | None = None) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """规则见模块说明。返回 (合并后的适配器, config.yaml 的 checkpoint 记录, 运行记录的附加项)。合并核对不通过抛 MergeCheckFailed。"""
    from src.adapters import registry
    from src.adapters.checkpoint import policy_spec, read_manifest
    from src.adapters.factory import default_image_policy, load_trained
    from src.e3.config import load_config
    from src.e3.merge_check import run_merge_check, settings_from_config

    ck = Path(checkpoint)
    man = read_manifest(ck)
    bs = man.get("build_spec") or {}
    if bs.get("key") != model_key or bool(bs.get("tiny")) != bool(tiny):
        raise W2CCheckpointError(f"检查点的构造方式（key {bs.get('key')!r}，tiny {bs.get('tiny')}）与 --model {model_key!r}、--tiny {tiny} 不符")
    training = training_record(ck)
    check_training(training, w2)
    mc = settings_from_config(load_config().doc, tiny=tiny, n=merge_check_n)  # t_text 口径（同 run_e1 --checkpoint）
    summ = run_merge_check(ck, model_key=model_key, tiny=tiny, n=mc["n"], checks=mc["checks"], out_json=Path(merge_check_json),
                           device=device, dtype=dtype, data_root=data_root, dev_subsets=None, log=log)
    ad, _ = load_trained(ck, device=device, dtype=torch_dtype, merge=True)
    if man.get("lora") is not None and ad.lora_state != "merged":
        raise RuntimeError(f"load_trained(merge=True) 之后 LoRA 状态是 {ad.lora_state}，应为 merged")
    if ad.grid_token_ids:
        raise W2CCheckpointError("W2-c 的检查点不该有网格 token（只有文本监督、不加新 token）")
    frozen = default_image_policy(registry.get_model(model_key).family)
    if not tiny and policy_spec(ad.image_policy) != policy_spec(frozen):
        raise W2CCheckpointError(f"检查点的图像口径 {policy_spec(ad.image_policy)} 不是 E1 冻结的口径 {policy_spec(frozen)}（D28-3）")
    rec = checkpoint_record(ck, man, merge_summary=summ, n_grid_tokens=len(ad.grid_token_ids), training=training)
    return ad, rec, {"checkpoint_dir": str(ck), "merge_check": summ}


def proposal_leaf(out: str | Path, spec: R.W2RunSpec) -> Path:
    return R.w2_leaf(out, spec.model_key, BOX_TASK, W2C_PROPOSE_SETTING, spec.split)


def refuse_mixing(out: str | Path, model_key: str, split: str, checkpoint: str | Path) -> None:
    """加载与合并核对之前先看同一个 --out 里已有的 W2-c 目录（提出与 w2c）：来自另一个检查点（manifest 的 sha256 不同）就拒绝，
    不覆盖 <out>/<模型>/w2c_merge_check.json。"""
    from src.runlog import sha256_file

    want = sha256_file(Path(checkpoint) / "manifest.json")
    for task, setting in ((BOX_TASK, W2C_PROPOSE_SETTING), ("refseg", W2C_VARIANT)):
        cfg = R.w2_leaf(out, model_key, task, setting, split) / CONFIG
        if cfg.exists():
            have = ((yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}).get("checkpoint") or {}).get("manifest_sha256")
            if have != want:
                raise W2CCheckpointError(f"{cfg.parent}：已有的结果来自另一个检查点（manifest sha256 {str(have)[:16]}…，本次 "
                                         f"{want[:16]}…），拒绝混用；换一个 --out")


def proposal_complete(leaf: Path, samples: Sequence[Sample], spec: R.W2RunSpec, checkpoint: Mapping[str, Any]) -> bool:
    """提出的目录已经写完、是同一个模型与划分（R.e1_run_complete）。已有但检查点不同（manifest sha256）就报错：拒绝混用。"""
    if not (leaf / CONFIG).exists():
        return False
    c = yaml.safe_load((leaf / CONFIG).read_text(encoding="utf-8")) or {}
    have = (c.get("checkpoint") or {}).get("manifest_sha256")
    if have != checkpoint["manifest_sha256"]:
        raise W2CCheckpointError(f"{leaf}：已有的提出来自另一个检查点（manifest sha256 {str(have)[:16]}…，本次 "
                                 f"{checkpoint['manifest_sha256'][:16]}…），拒绝混用；换一个 --out")
    return R.e1_run_complete(leaf, samples, spec)


def ensure_proposal(adapter: Any, entry: Any, spec: R.W2RunSpec, cfg: W2Config, samples: Sequence[Sample], paths: DataPaths,
                    out: str | Path, *, checkpoint: Mapping[str, Any], splits_doc: Mapping[str, Any], batch_size: int,
                    prompts: PromptConfig, argv: Sequence[str] = (), run_extra: Mapping[str, Any] | None = None,
                    log: Callable[[str], None] | None = None, warmup: bool = True) -> tuple[Path, list[dict[str, Any]]]:
    """W2-c 的提出（模块说明）：训练过的模型（adapter）按 E1 refexp_box 协议出框；已写完就直接读。返回 (目录, 记录)。"""
    log = log or (lambda m: print(m, file=sys.stderr, flush=True))
    leaf = proposal_leaf(out, spec)
    if not proposal_complete(leaf, samples, spec, checkpoint):
        pspec = dataclasses.replace(spec, task=BOX_TASK, setting=W2C_PROPOSE_SETTING)
        rs = dataclasses.replace(B.e1_spec(entry, adapter, adapter.image_policy, BOX_TASK, "eval", batch_size, tiny=spec.tiny,
                                           limit=spec.limit), checkpoint=dict(checkpoint))
        sel = R.e1_selection(BOX_TASK, samples, R.w2_record(pspec, splits_doc, samples))
        log(f"[W2-c {spec.model_key}] 训练过的模型按 E1 refexp_box 协议出框：{leaf}")
        run_task(adapter, rs, sel, prompts, paths, out, resume=True, argv=argv, leaf=leaf, warmup=warmup, run_extra=run_extra)
        if not proposal_complete(leaf, samples, spec, checkpoint):
            raise RuntimeError(f"{leaf}：W2-c 的提出跑完之后仍不完整")
    return leaf, read_jsonl_tolerant(leaf / PREDICTIONS, repair=False)


__all__ = ["BOX_TASK", "MERGE_CHECK_JSON", "W2CCheckpointError", "W2C_VARIANT", "check_training", "checkpoint_record", "ensure_proposal",
           "load_checkpoint_adapter", "proposal_complete", "proposal_leaf", "refuse_mixing", "training_record"]
