"""E2 探针接训练后的检查点（DEV-0038 §2；D77-2(b)：E3 的机制测量，不计入成功标准；E3 预注册 §10a）。

scripts/run_e2.py --checkpoint <检查点> --checkpoint-label <标签> [--layers L24 L8 …]：探针设置与 E2 v1 完全相同（同一个配置
configs/e2_probes.json、同一套超参与步数、同一个种子规则；src/e2/probes.probe_seed 按 (基础种子, 任务, 特征键, 种类) 定初值，与别的层
无关），只把冻结的 base 换成"检查点合并后的模型"。

**哪些检查点、怎么加载**（resolve_checkpoint；变体从结构与训练叶目录的 config.yaml 认）：
- E3b、C3、C1′、网格格式预热（scripts/train_e3b.py 的 <叶>/ckpt/final/model）：factory.load_trained(merge=True) 之后取隐藏状态。
  C1′ 没有 LoRA，只训了网格 token 的行；E2 的输入里没有网格 token，所以它的隐藏状态与 base 逐位相同（只是复核 base 读数）。
- E3a（scripts/train_e3a.py 的 <叶>/ckpt/final：model/ + readout/；给 final 或 final/model 都行）：只用 model/ 里的 LoRA
  （load_trained(final/model, merge=True)，与 run_e1 --checkpoint 同一种加载：表里只有查询占位符的行、没有查询个数，build_inputs
  不追加查询 token），不用查询与读出（readout/ 只读 readout.json 认变体，sha256 记进运行身份）。
- C1、C2（E3a 的 c1、c2：冻结 LLM）：LLM 权重没动，拒绝（RefusedCheckpoint），提示引用 E2 的 base 读数，不重复跑。更一般地，
  没有 LoRA 也没有网格 token 行的检查点（LLM 权重与 base 相同）一律拒绝。
- E5 的检查点（final/ 里有 e5_stack.json）不能单独加载（D75-3），拒绝。

**合并核对**（与 E3 评测同一套：src/e3/merge_check.run_merge_check，判据 v1 = D50-3）：检查项取 E3b 配置 eval.merge_check.checks 的
t_text（只有 merge：E2 读隐藏状态，用 base 的文本 token，同 run_e1 --checkpoint）；没有 LoRA 的检查点（C1′）merge 一项跳过。
不通过抛 MergeCheckFailed，调用方拒绝（退出码 1，不建任何任务目录）。输出 <out>/<标签>/<模型>/merge_check.json。

**运行身份**（checkpoint_record → config.yaml 的 checkpoint；不给 --checkpoint 时 config.yaml 没有这个键，与原来逐字节相同）：
检查点 manifest 的 sha256、标签、认出的变体、训练叶目录 config.yaml 记的种子、构造方式、图像口径、dtype、LoRA、网格 token 数、
E3a 的 readout.json sha256、合并核对的摘要。路径不进（与机器有关），只进运行记录。输出目录 <out>/<标签>/<模型>/<任务>/，规则同
run_e1 --checkpoint-label（CHECKPOINT_LABEL，与 scripts/run_e1.py 的同一个正则，tests 钉住）。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from src.adapters.checkpoint import MANIFEST, TOKENS_JSON, file_sha256, read_manifest

CHECKPOINT_LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*(/[A-Za-z0-9][A-Za-z0-9_.-]*)*")  # 同 scripts/run_e1.py（tests 钉住）
MERGE_CHECK_FILE = "merge_check.json"
MERGE_CHECK_PURPOSE = "t_text"  # src/e3/merge_check.PURPOSES：E2 读隐藏状态，同 T 文本评测只核对 merge
E3A_READOUT = "readout"
E3A_READOUT_MANIFEST = "readout.json"
E5_STACK_FILE = "e5_stack.json"
FROZEN_LLM_VARIANTS = ("c1", "c2")  # E3a 的冻结 LLM 对照
USE_TEXT = {
    "e3a": "load_trained(<final>/model, merge=True) 之后取隐藏状态：只用 LoRA，不用查询与读出",
    "default": "load_trained(检查点, merge=True) 之后取隐藏状态",
}


class RefusedCheckpoint(ValueError):
    """这个检查点不该跑 E2 探针（LLM 权重没动，或不能单独加载）。"""


@dataclass(frozen=True)
class CheckpointRef:
    model_dir: Path
    experiment: str | None  # E3b / E3a（认不出时 None）
    variant: str | None  # e3b、c3、c1prime、warmup；E3a 的 full、c1、c2
    kind: str  # 记进运行身份的种类：e3b、c3、c1prime、warmup、e3a_full、unknown
    manifest: Mapping[str, Any]
    manifest_sha256: str
    leaf_config: str | None = None
    leaf_seed: int | None = None
    readout_sha256: str | None = None
    n_grid_tokens: int = 0
    extra: Mapping[str, Any] = field(default_factory=dict)


def _leaf_doc(model_dir: Path) -> tuple[dict[str, Any] | None, str | None]:
    """<叶>/ckpt/final/model → <叶>/config.yaml（训练的运行身份）。"""
    p = model_dir.resolve()
    if p.name != "model" or p.parent.name != "final" or p.parent.parent.name != "ckpt":
        return None, None
    cfg = p.parents[2] / "config.yaml"
    if not cfg.is_file():
        return None, None
    return (yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}), str(cfg)


def resolve_checkpoint(path: str | Path) -> CheckpointRef:
    """检查点目录 → CheckpointRef（规则见模块说明）；C1、C2、E5 检查点与 LLM 权重没动的检查点抛 RefusedCheckpoint。"""
    d = Path(path)
    if (d / E5_STACK_FILE).is_file() or (d.parent / E5_STACK_FILE).is_file():
        raise RefusedCheckpoint(f"{d} 是 E5 的检查点（有 {E5_STACK_FILE}）：它的 model/ 不能单独加载（D75-3），E2 探针不接")
    readout_dir = None
    if (d / MANIFEST).is_file():
        model_dir = d
        if (d.parent / E3A_READOUT / E3A_READOUT_MANIFEST).is_file():
            readout_dir = d.parent / E3A_READOUT
    elif (d / "model" / MANIFEST).is_file():
        model_dir = d / "model"
        if not (d / E3A_READOUT / E3A_READOUT_MANIFEST).is_file():
            raise RefusedCheckpoint(f"{d} 里有 model/ 却没有 {E3A_READOUT}/{E3A_READOUT_MANIFEST}：认不出是哪种检查点；"
                                    "E3b 系的检查点给 <叶>/ckpt/final/model")
        readout_dir = d / E3A_READOUT
    else:
        raise RefusedCheckpoint(f"{d} 里没有 {MANIFEST}（也没有 model/{MANIFEST}）：不是训练后的检查点")
    man = read_manifest(model_dir)  # 逐个核对文件的 sha256 与字节数
    table = json.loads((model_dir / TOKENS_JSON).read_text(encoding="utf-8"))
    n_grid = len(table.get("grid_tokens") or [])
    doc, leaf_cfg = _leaf_doc(model_dir)
    experiment = None if doc is None else doc.get("experiment")
    variant = None if doc is None else doc.get("variant")
    readout_sha = None
    if readout_dir is not None:  # E3a：readout.json 认变体（逐个核对它记的文件），只记 sha256，不加载读出
        from src.readout.train import read_readout_manifest

        rman = read_readout_manifest(readout_dir)
        rv = str(rman["spec"]["variant"])
        if variant is not None and variant != rv:
            raise RefusedCheckpoint(f"{leaf_cfg} 的变体 {variant!r} 与 readout.json 的 {rv!r} 不同")
        if rman.get("model_manifest_sha256") != file_sha256(model_dir / MANIFEST):
            raise RefusedCheckpoint(f"{readout_dir} 与 {model_dir} 不配对（readout.json 记的 model/manifest.json sha256 不同）")
        experiment, variant = "E3a", rv
        readout_sha = file_sha256(readout_dir / E3A_READOUT_MANIFEST)
    if experiment == "E3a" and variant in FROZEN_LLM_VARIANTS:
        raise RefusedCheckpoint(f"{d} 是 E3a 的 {variant.upper()}（冻结 LLM）：LLM 权重没动，E2 探针的读数就是 base 的；"
                                "请引用 E2 的 base 读数（同一模型、同一层），不重复跑")
    if man.get("lora") is None and n_grid == 0:
        raise RefusedCheckpoint(f"{d} 没有 LoRA、也没有网格 token 的行：LLM 权重与 base 相同（如 C1、C2），请引用 E2 的 base 读数，"
                                "不重复跑")
    if experiment == "E3a":
        kind = f"e3a_{variant}"
    else:
        kind = str(variant) if variant is not None else "unknown"
    seed = None if doc is None else doc.get("seed")
    return CheckpointRef(model_dir=model_dir, experiment=experiment, variant=variant, kind=kind, manifest=man,
                         manifest_sha256=file_sha256(model_dir / MANIFEST), leaf_config=leaf_cfg,
                         leaf_seed=None if seed is None or isinstance(seed, bool) else int(seed), readout_sha256=readout_sha,
                         n_grid_tokens=n_grid)


def checkpoint_record(ref: CheckpointRef, *, label: str, merge_check: Mapping[str, Any]) -> dict[str, Any]:
    """进 config.yaml 的 checkpoint（模块说明；不含路径）。"""
    m = ref.manifest
    return {"manifest_sha256": ref.manifest_sha256, "label": label, "kind": ref.kind, "experiment": ref.experiment,
            "variant": ref.variant, "train_seed": ref.leaf_seed, "build_spec": m.get("build_spec"),
            "image_policy": m.get("image_policy"), "dtype": m.get("dtype"), "lora": m.get("lora"),
            "trainable_token_rows": m.get("trainable_token_rows"), "n_grid_tokens": ref.n_grid_tokens,
            "readout_manifest_sha256": ref.readout_sha256,
            "use": USE_TEXT["e3a" if ref.experiment == "E3a" else "default"],
            "merge_check": {"criteria_version": merge_check.get("criteria_version"), "n": merge_check.get("n"),
                            "checks": merge_check.get("checks"), "skipped_checks": merge_check.get("skipped_checks"),
                            "purpose": MERGE_CHECK_PURPOSE}}


def load_for_probes(ref: CheckpointRef, *, device: str, dtype: Any) -> Any:
    """factory.load_trained(model/, merge=True)；有 LoRA 时核对已合并；不许带查询 token（E3a 不用查询）。"""
    from src.adapters.factory import load_trained

    ad, _ = load_trained(ref.model_dir, device=device, dtype=dtype, merge=True)
    if ref.manifest.get("lora") is not None and ad.lora_state != "merged":
        raise RuntimeError(f"load_trained(merge=True) 之后 LoRA 状态是 {ad.lora_state}，应为 merged")
    q = ad.token_table().get("query") or {}
    if q.get("n") is not None:
        raise RuntimeError(f"检查点载入后登记了 {q['n']} 个查询 token：E2 不用查询（E3a 只用 LoRA）")
    ad.eval_mode()
    return ad


__all__ = ["CHECKPOINT_LABEL", "CheckpointRef", "FROZEN_LLM_VARIANTS", "MERGE_CHECK_FILE", "MERGE_CHECK_PURPOSE", "RefusedCheckpoint",
           "checkpoint_record", "load_for_probes", "resolve_checkpoint"]
