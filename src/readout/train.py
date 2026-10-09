"""E3a 的训练循环（DEV-0034 §2）：一个模型 × 一个变体 × 一个种子 → <out>/e3a/<变体>/<模型>/seed<k>/。结构借自 E3b（src/e3/train.py）。

**顺序**（适配器的顺序约束照旧检查）：加载 base → torch.manual_seed(种子) → inject_query_tokens(n, 家族的位置模式)（full、c1）→
inject_lora（full）→ 冻结 LLM 的其余参数（c1、c2：全部；full：peft 已冻结 LoRA 以外的参数）→ 查询嵌入初始化（种子决定的噪声）→
读出模块（Readout；构造时断言参数量、做像素通路的结构检查，再带上适配器的模型复查一次）→ 梯度检查点（full、c1）→ 优化器。
续跑：load_trained(<latest>/model, merge=False)（恢复 token 表、挂 LoRA，不合并）→ inject_query_tokens → 读出与查询嵌入从
<latest>/readout 载入 → 优化器、调度器与随机数状态。

**一步**：样本按 micro_batch_size 切成微批（语义分割的样本与其余样本分开成批：类别名区间只在语义分割的行上取），各自前向
（features.llm_inputs 或 vision_inputs → Readout → 各样本自己任务的头）、反向（Σ 任务权重 × 逐样本损失 ÷ 这一步的样本数），然后一次
裁剪与 step。四个参数组（config.groups_for：没有的组不建）：lora、query（查询嵌入，fp32）、mix（混合的 logits 与 gamma）、readout
（解码器、类别投影、各头）。warmup（线性）+ 余弦衰减（src/e3/train.lr_factor）。

**检查点**（D34：不存合并后的权重）：<ckpt>/latest/ 与 <ckpt>/final/ 都是 model/ + readout/（latest 另有 resume_state.pt）：
- model/：adapter.save_trained，存之前临时取消查询登记（clear_query_tokens，存完再登记回来），所以表里只有查询占位符的行、没有查询个数：
  scripts/run_e1.py --checkpoint 能直接加载它（只用合并后的 LoRA，build_inputs 不追加查询 token），c1、c2 没有 LoRA；
- readout/：query_embeds、layer_mix、decoder（含类别投影）、heads 分别存（safetensors），readout.json 记结构、参数量、各文件的 sha256
  与 model/manifest.json 的 sha256（两者必须配对）。
**运行期审计**（进 logs/run_<k>.json）：LLM 里求梯度的参数只能是 LoRA（c1、c2 一个也没有）；LoRA 是 fp32；base 权重的指纹
（src/adapters/checkpoint.base_fingerprint，LoRA 之外的参数）在每次存检查点时与开训时相同，否则报错（c1 下 LLM 逐位不变的运行期证据）；
读出模块的参数量与像素通路检查的摘要。

**续跑**：叶目录非空时必须给 resume 且 config.yaml 与本次的运行身份完全相同；从 latest 的步数接着跑，train_log.jsonl 里之后的行截掉
重算。样本日程无状态、随机数状态随检查点保存，所以 CPU 上续跑与一次跑完逐位一致（tests 钉住）。

**记账**：train_log.jsonl（每步：lr、损失、各任务损失与样本数、梯度范数、混合权重、各类 token 数与像素数；CPU 上续跑与一次跑完
逐字节相同）；logs/run_<k>.json、logs/timing_run_<k>.jsonl（每步的 data_s、build_inputs_s、fwd_s、bwd_s、opt_s、step_s 与显存峰值）；
accounting.json、compute.json（src/accounting：每条样本一次调用，generated_tokens = 0，prompt_tokens = 送进 LLM 的提示长度（c2 不跑 LLM，
记 0），visual_tokens、image_pixels 按图）。

**E3a+C3**（DEV-0044；运行变体 e3a_c3，run_cfg 的 mixed 一节；配置的 mixed 节）：搭建、检查点、续跑都同 full（Trainer 的读出变体是
full：LoRA、查询嵌入、混合、解码器与各头，四个参数组）；样本来自 data.MixedSchedule（每步 = C3 同一种子的文本流 per_step.text 条 +
E3a full 同一种子的稠密流 per_step.dense 条，与两者单独训练逐个相同），文本样本由 C3 的构造（src/e3/data.ExampleBuilder，原样调用）给出。
一步之内按流分开切微批（每个微批只含一种监督，流的先后按 mixed.order，文本在前）：
- 文本微批：build_inputs 之前临时取消查询登记（clear_query_tokens，之后登记回来），所以输入里没有查询 token，与 C3 的输入逐个相同
  （入口断言：每一行的 query_span 都是 None）；adapter.training_forward（不给 query_embeds）→ C3 的文本损失（src/e3/train.Trainer
  的 _task_loss 文本分支，原样调用：逐个任务 chunked_lm_loss 全词表求和）。不经过读出。
- 稠密微批：照 E3a 原样（语义分割单独成批；llm_inputs → 读出 → 各任务的头与损失）。
每种监督各自归一（文本 ÷ 这一步的文本监督 token 总数，同 C3 的 token_mean；稠密 ÷ 这一步的稠密样本数，同 E3a 的 sample_mean），
再乘 loss_weights（1 : 1）相加：loss = 1 · L_text + 1 · L_dense。train_log 的 loss 是这个和；loss_by_task 的 text、dense 两项是 L_text、
L_dense，text/<任务> 是分任务的逐 token 平均，dense/<任务> 是分任务的逐样本平均（同 E3a）；n_by_task 是分流、分任务的样本数；
n_supervised_by_task 是文本分任务的监督 token 数；by_supervision 记两种监督各自的样本数与 token 数（accounting.json、train_summary.json
另有合计）。compute.json 里文本样本的 generated_tokens 是教师强制的目标 token 数（同 C3），稠密样本为 0。学习率的 warmup 与余弦按
mixed.steps 排：日程长度是 E3a full、C3 的 2 倍（运行身份 training.mixed 与 README 写明）。运行身份另有 text_stream 一节（C3 的配置、
指令、C3 节、停止 token）。
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import yaml

from src.accounting import Compute, merge, write_compute_json
from src.adapters.checkpoint import base_fingerprint, file_sha256, read_manifest
from src.e3 import config as E3C
from src.e3 import train as E3T
from src.e3.data import check_dedupe, ids_digest
from src.e3.targets import NoSupervision as TextNoSupervision
from src.e3.train import identity_sha256, lr_factor
from src.readout import data as D
from src.readout import losses as LS
from src.readout.config import (
    E3A_C3,
    MIXED_STREAMS,
    TEXT_SUPERVISION,
    E3aConfig,
    groups_for,
    micro_batch_groups,
    readout_spec,
    readout_variant,
    text_stream_config,
)
from src.readout.features import ClassSpans, ReadoutInputs, c2_class_side, llm_inputs, vision_inputs
from src.readout.model import Readout, check_no_pixel_path
from src.readout.targets import NoSupervision
from src.runlog import write_run_metadata

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_SUB = "e3a"
CONFIG = "config.yaml"
TRAIN_LOG = "train_log.jsonl"
SUMMARY = "train_summary.json"
ACCOUNTING = "accounting.json"
COMPUTE = "compute.json"
README = "README.md"
LATEST = "latest"
FINAL = "final"
MODEL_SUB = "model"
READOUT_SUB = "readout"
RESUME_STATE = "resume_state.pt"
READOUT_FORMAT = "visint-e3a-readout/1"
READOUT_MANIFEST = "readout.json"
READOUT_FILES = {"query": "query_embeds.safetensors", "mix": "layer_mix.safetensors", "decoder": "decoder.safetensors",
                 "heads": "heads.safetensors"}
QUERY_STD_ROWS = 4096  # 查询嵌入初始化：估计输入嵌入逐元素标准差时均匀取的基础词表行数


class StopRequested(Exception):
    """--stop-after-steps 到了：检查点已写，下次 resume 接着跑。"""


class CheckpointMismatch(RuntimeError):
    """readout/ 与 model/ 不配对、结构与本次不同、文件哈希不符。"""


@dataclass(frozen=True)
class RunSpec:
    model_key: str
    family: str
    repo: str
    revision: str
    adapter_model_key: str
    tiny: bool
    image_policy: str
    model_dtype: str
    variant: str
    seed: int


def leaf_dir(out: str | Path, variant: str, model_key: str, seed: int) -> Path:
    return Path(out) / EXPERIMENT_SUB / variant / model_key / f"seed{int(seed)}"


def _sync(device: Any) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(torch.device(device))


def _identity(cfg: Mapping[str, Any]) -> dict[str, Any]:
    return yaml.safe_load(yaml.safe_dump(dict(cfg), sort_keys=False, allow_unicode=True))


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _jsonl(rec: Mapping[str, Any]) -> str:
    return json.dumps(rec, ensure_ascii=False, sort_keys=False) + "\n"


def _read_log(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    raw = path.read_bytes()
    keep = raw.rfind(b"\n") + 1
    return [json.loads(x) for x in raw[:keep].decode("utf-8").splitlines() if x.strip()]


def _append(path: Path, lines: Sequence[str]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write("".join(lines))
        f.flush()
        os.fsync(f.fileno())


def _dir_bytes(d: Path) -> int:
    return sum(p.stat().st_size for p in d.rglob("*") if p.is_file()) if d.exists() else 0


def layers_for(ad: Any) -> list[int]:
    """逐层混合的来源：全部 LLM 层，层号 0..num_layers()（D19：含第 0 层，最后一层是最终归一化之后）。"""
    return list(range(int(ad.num_layers()) + 1))


def init_query_embeds(ad: Any, n: int, ratio: float, seed: int) -> torch.Tensor:
    """查询嵌入的初始值 [n, H]（fp32，在适配器的设备上）：占位符行经输入嵌入查表的值 + N(0, (ratio · s)²)，s = 均匀取的
    QUERY_STD_ROWS 个基础词表行（查表空间）逐元素的标准差。噪声用以种子为种的独立生成器（不动全局随机数）。"""
    emb = ad.model.get_input_embeddings()
    qid = ad.query_token_id
    if qid is None:
        raise RuntimeError("还没有注入查询 token")
    n_vocab = min(int(emb.weight.shape[0]), len(ad.tokenizer))
    ids = torch.linspace(0, n_vocab - 1, steps=min(QUERY_STD_ROWS, n_vocab)).round().long().to(ad.device)
    with torch.no_grad():
        base = emb(torch.tensor([int(qid)], device=ad.device))[0].float()
        s = float(emb(ids).float().std())
    g = torch.Generator().manual_seed(int(seed))
    noise = torch.randn((int(n), base.shape[0]), generator=g, dtype=torch.float32) * float(ratio) * s
    return (base[None].cpu() + noise).to(ad.device)


# ---------------------------------------------------------------------------
# 读出的存取（检查点的 readout/ 子目录）
# ---------------------------------------------------------------------------


def _save_st(tensors: Mapping[str, torch.Tensor], path: Path) -> None:
    from safetensors.torch import save_file

    save_file({k: v.detach().to("cpu").contiguous().clone() for k, v in tensors.items()}, str(path), metadata={"format": "pt"})


def _load_st(path: Path) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    return load_file(str(path))


def readout_tensors(readout: Readout, query_embeds: torch.Tensor | None) -> dict[str, dict[str, torch.Tensor]]:
    """读出的四份文件的内容（READOUT_FILES）。c2 没有查询嵌入与可训练的混合权重。"""
    out: dict[str, dict[str, torch.Tensor]] = {
        "decoder": {**{f"decoder.{k}": v for k, v in readout.decoder.state_dict().items()},
                    **{f"class_proj.{k}": v for k, v in readout.class_proj.state_dict().items()}},
        "heads": dict(readout.heads.state_dict()),
    }
    if readout.mix.learnable:
        out["mix"] = dict(readout.mix.state_dict())
    if query_embeds is not None:
        out["query"] = {"query_embeds": query_embeds}
    return out


def save_readout(directory: str | Path, readout: Readout, query_embeds: torch.Tensor | None, *, model_manifest_sha256: str,
                 query: Mapping[str, Any] | None, layers: Sequence[int] | None) -> Path:
    d = Path(directory)
    if d.exists() and (not d.is_dir() or any(d.iterdir())):
        raise CheckpointMismatch(f"{d} 已存在且不是空目录")
    d.mkdir(parents=True, exist_ok=True)
    files = {}
    for kind, tensors in readout_tensors(readout, query_embeds).items():
        p = d / READOUT_FILES[kind]
        _save_st(tensors, p)
        files[READOUT_FILES[kind]] = {"sha256": file_sha256(p), "bytes": p.stat().st_size}
    man = {"format": READOUT_FORMAT, "spec": readout.spec.as_dict(), "sources": list(readout.sources),
           "params": readout.params, "layers": None if layers is None else [int(x) for x in layers],
           "query": None if query is None else dict(query), "layer_mix": readout.mix.snapshot() if readout.mix.learnable else None,
           "model_manifest_sha256": model_manifest_sha256, "files": files}
    tmp = d / (READOUT_MANIFEST + ".tmp")
    tmp.write_text(json.dumps(man, ensure_ascii=False, indent=1, sort_keys=True))
    tmp.replace(d / READOUT_MANIFEST)
    return d


def read_readout_manifest(directory: str | Path) -> dict[str, Any]:
    d = Path(directory)
    p = d / READOUT_MANIFEST
    if not p.is_file():
        raise CheckpointMismatch(f"{d} 里没有 {READOUT_MANIFEST}")
    man = json.loads(p.read_text(encoding="utf-8"))
    if man.get("format") != READOUT_FORMAT:
        raise CheckpointMismatch(f"不认识的读出格式 {man.get('format')!r}")
    actual = {x.name for x in d.iterdir() if x.is_file() and not x.name.startswith(".")} - {READOUT_MANIFEST}
    if actual != set(man["files"]):
        raise CheckpointMismatch(f"{d} 的文件与 readout.json 不符：多出 {sorted(actual - set(man['files']))}，"
                                 f"缺少 {sorted(set(man['files']) - actual)}")
    for name, info in man["files"].items():
        if (d / name).stat().st_size != info["bytes"] or file_sha256(d / name) != info["sha256"]:
            raise CheckpointMismatch(f"{name} 的字节数或 sha256 与 readout.json 不符")
    return man


def load_readout(directory: str | Path, readout: Readout, *, model_dir: str | Path | None = None) -> tuple[torch.Tensor | None, dict]:
    """把 readout/ 载入已构造的读出模块（结构必须相同，strict）；给了 model_dir 时核对两者配对（model/manifest.json 的 sha256）。
    返回 (查询嵌入（CPU，fp32；c2 为 None）, readout.json)。"""
    d = Path(directory)
    man = read_readout_manifest(d)
    if man["spec"] != readout.spec.as_dict():
        raise CheckpointMismatch(f"读出结构与本次不同：{man['spec']} ≠ {readout.spec.as_dict()}")
    if model_dir is not None:
        sha = file_sha256(Path(model_dir) / "manifest.json")
        if man["model_manifest_sha256"] != sha:
            raise CheckpointMismatch(f"{d} 与 {model_dir} 不是同一次保存的（model/manifest.json 的 sha256 不符）")
    t = {kind: _load_st(d / READOUT_FILES[kind]) for kind in READOUT_FILES if READOUT_FILES[kind] in man["files"]}
    dec = t["decoder"]
    readout.decoder.load_state_dict({k[len("decoder."):]: v for k, v in dec.items() if k.startswith("decoder.")}, strict=True)
    readout.class_proj.load_state_dict({k[len("class_proj."):]: v for k, v in dec.items() if k.startswith("class_proj.")},
                                       strict=True)
    readout.heads.load_state_dict(t["heads"], strict=True)
    if readout.mix.learnable:
        if "mix" not in t:
            raise CheckpointMismatch("检查点里没有混合权重")
        readout.mix.load_state_dict(t["mix"], strict=True)
    elif "mix" in t:
        raise CheckpointMismatch("c2 的检查点不该有混合权重")
    q = t.get("query", {}).get("query_embeds")
    return q, man


def save_checkpoint(ad: Any, readout: Readout, query_embeds: torch.Tensor | None, directory: str | Path, *,
                    query: Mapping[str, Any] | None, layers: Sequence[int] | None) -> Path:
    """<directory>/model（save_trained；存之前取消查询登记、存完再登记回来）+ <directory>/readout。"""
    d = Path(directory)
    model_dir = d / MODEL_SUB
    if query is not None:
        ad.clear_query_tokens()
    try:
        ad.save_trained(model_dir)
    finally:
        if query is not None:
            ad.inject_query_tokens(int(query["n"]), str(query["position_mode"]))
    save_readout(d / READOUT_SUB, readout, query_embeds, model_manifest_sha256=file_sha256(model_dir / "manifest.json"),
                 query=query, layers=layers)
    return d


# ---------------------------------------------------------------------------
# 训练器
# ---------------------------------------------------------------------------


def stack_targets(task: str, examples: Sequence[D.Example], device: Any) -> dict[str, torch.Tensor]:
    """一组同任务样本的目标（模型所在设备）。"""
    ts = [e.target for e in examples]
    if task == "refseg":
        return {"mask": torch.from_numpy(np.stack([t.values for t in ts])).to(device)}
    if task == "semseg":
        return {"label": torch.from_numpy(np.stack([t.values for t in ts])).to(device)}
    if task == "depth":
        return {"log_gt": torch.from_numpy(np.stack([t.values for t in ts])).to(device),
                "valid": torch.from_numpy(np.stack([t.valid for t in ts])).to(device)}
    if task == "normals":
        return {"gt": torch.from_numpy(np.stack([t.values for t in ts])).to(device),
                "valid": torch.from_numpy(np.stack([t.valid for t in ts])).to(device)}
    raise ValueError(task)


def task_losses(task: str, out: torch.Tensor, tgt: Mapping[str, torch.Tensor], loss_cfg: Mapping[str, Any]) -> torch.Tensor:
    """[b] 的逐样本损失（losses.py 的四个函数）。"""
    if task == "refseg":
        return LS.refseg_loss(out, tgt["mask"], loss_cfg)
    if task == "depth":
        return LS.depth_loss(out, tgt["log_gt"], tgt["valid"], loss_cfg)
    if task == "semseg":
        return LS.semseg_loss(out, tgt["label"])
    if task == "normals":
        return LS.normals_loss(out, tgt["gt"], tgt["valid"])
    raise ValueError(task)


class Trainer:
    """持有适配器、读出模块、查询嵌入、优化器与调度器；optimize(examples) 做一步（含微批与梯度累积）。
    variant 是运行变体：e3a_c3 时读出变体（self.variant）是 full，另持有文本流的配置（text_cfg）与 C3 的训练器（self.c3，只用它的
    文本损失与 build_inputs，不搭建、不训练）。"""

    def __init__(self, adapter: Any, cfg: E3aConfig, run_cfg: Mapping[str, Any], *, variant: str, seed: int,
                 tiny: bool = False, text_cfg: E3C.E3Config | None = None) -> None:
        self.ad = adapter
        self.cfg = cfg
        self.rc = dict(run_cfg)
        self.run_variant = variant
        self.variant = readout_variant(variant)
        self.seed = int(seed)
        self.tiny = bool(tiny)
        self.mixed = variant == E3A_C3  # E3a+C3（模块说明）
        if (self.rc.get("variant") == E3A_C3) != self.mixed or self.mixed != ("mixed" in self.rc):
            raise ValueError(f"run_cfg（variant {self.rc.get('variant')!r}，mixed 一节 {'有' if 'mixed' in self.rc else '没有'}）"
                             f"与训练器的变体 {variant!r} 不符")
        self.text_cfg = text_cfg
        self.c3: E3T.Trainer | None = None
        self.c3_rc: dict[str, Any] | None = None
        if self.mixed:
            ref = self.rc["mixed"]["reference"]["text"]["config"]
            if text_cfg is None or text_cfg.sha256 != ref["sha256"]:
                raise ValueError(f"E3a+C3 要给文本流的配置，sha256 必须是 run_cfg 记下的 {ref['sha256'][:16]}…，收到 "
                                 f"{None if text_cfg is None else text_cfg.sha256[:16]}")
            self.c3_rc = E3C.resolve_run(text_cfg, supervision=TEXT_SUPERVISION, tiny=self.tiny)
            self.c3 = E3T.Trainer(adapter, text_cfg, self.c3_rc, supervision=TEXT_SUPERVISION, seed=self.seed)
        elif text_cfg is not None:
            raise ValueError(f"{variant} 没有文本流，不该给 text_cfg")
        self.readout: Readout | None = None
        self.query_embeds: torch.nn.Parameter | None = None
        self.layers: list[int] | None = None
        self.query: dict[str, Any] | None = None
        self.class_spans = ClassSpans(cfg.class_names)
        self.c2_class: tuple[torch.Tensor, torch.Tensor] | None = None
        self.opt: torch.optim.Optimizer | None = None
        self.sched: torch.optim.lr_scheduler.LambdaLR | None = None
        self.audit: dict[str, Any] = {}
        self._fingerprint: str | None = None
        self.mix_init: dict[str, Any] | None = None

    # ----- 搭建 -----
    def _query_spec(self) -> dict[str, Any] | None:
        if self.variant == "c2":
            return None
        return {"n": self.cfg.query_n, "grid": self.cfg.query_grid, "position_mode": self.cfg.position_mode(self.ad.family)}

    def _build_readout(self) -> Readout:
        spec = readout_spec(self.cfg.doc, variant=self.variant, hidden_size=self.ad.hidden_size(), n_layers=self.ad.num_layers(),
                            tiny=self.tiny)
        return Readout(spec).to(self.ad.device)

    def _freeze_llm(self) -> None:
        for n, p in self.ad.model.named_parameters():
            if self.variant == "full" and "lora_" in n:
                continue
            p.requires_grad_(False)

    def setup_fresh(self) -> None:
        torch.manual_seed(self.seed)
        self.query = self._query_spec()
        if self.query is not None:
            self.ad.inject_query_tokens(int(self.query["n"]), str(self.query["position_mode"]))
        if self.variant == "full":
            lo = self.rc["lora"]
            self.ad.inject_lora(r=int(lo["r"]), lora_alpha=float(lo["alpha"]), lora_dropout=float(lo["dropout"]))
        self._freeze_llm()
        if self.query is not None:
            q = init_query_embeds(self.ad, int(self.query["n"]), float(self.rc["query"]["init_noise_ratio"]), self.seed)
            self.query_embeds = torch.nn.Parameter(q)
        self.readout = self._build_readout()
        self._finish_setup()

    def setup_resume(self, ckpt_dir: Path) -> None:
        self.ad.load_trained(ckpt_dir / MODEL_SUB, merge=False)
        want = "injected" if self.variant == "full" else "none"
        if self.ad.lora_state != want:
            raise RuntimeError(f"续跑：检查点载入后 LoRA 状态是 {self.ad.lora_state}，{self.variant} 应为 {want}")
        self.query = self._query_spec()
        if self.query is not None:
            self.ad.inject_query_tokens(int(self.query["n"]), str(self.query["position_mode"]))
        self._freeze_llm()
        self.readout = self._build_readout()
        q, _ = load_readout(ckpt_dir / READOUT_SUB, self.readout, model_dir=ckpt_dir / MODEL_SUB)
        if (q is None) != (self.query is None):
            raise CheckpointMismatch("检查点的查询嵌入与本次的变体不符")
        if q is not None:
            self.query_embeds = torch.nn.Parameter(q.to(self.ad.device, torch.float32))
        self._finish_setup()

    def _finish_setup(self) -> None:
        ad = self.ad
        if self.variant != "c2":
            self.layers = layers_for(ad)
            if self.readout.spec.n_sources != len(self.layers):
                raise RuntimeError("读出的来源数与 LLM 层数不符")
        else:
            self.c2_class = c2_class_side(ad, self.cfg.instruction("semseg"), self.cfg.class_names, order=self.cfg.order("semseg"))
        if self.rc["gradient_checkpointing"]:
            ad.set_gradient_checkpointing(True)
        lora = [p for n, p in ad.model.named_parameters() if "lora_" in n]
        groups = {"lora": lora, "query": [self.query_embeds] if self.query_embeds is not None else [],
                  "mix": list(self.readout.mix.parameters()),
                  "readout": [p for m in (self.readout.decoder, self.readout.class_proj, self.readout.heads)
                              for p in m.parameters()]}
        want = groups_for(self.variant)
        for g, ps in groups.items():
            if (g in want) != bool(ps):
                raise RuntimeError(f"{self.variant} 的参数组 {g}：{'应有' if g in want else '不应有'}参数，实际 {len(ps)} 个")
        o = self.rc["optim"]
        pg = [{"params": groups[g], "lr": float(o["groups"][g]["lr"]), "weight_decay": float(o["groups"][g]["weight_decay"]),
               "name": g} for g in want]
        self.opt = torch.optim.AdamW(pg, betas=tuple(float(b) for b in o["betas"]), eps=float(o["eps"]))
        warm, total, mn = int(self.rc["warmup_steps"]), int(self.rc["steps"]), float(o["min_lr_ratio"])
        self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, lambda s: lr_factor(s, warmup=warm, total=total, min_ratio=mn))
        self.audit = self.setup_audit(lora)
        self._fingerprint = base_fingerprint(ad.model)
        self.mix_init = self.readout.mix.snapshot() if self.readout.mix.learnable else None

    def setup_audit(self, lora: Sequence[torch.nn.Parameter]) -> dict[str, Any]:
        """运行期审计（进运行记录）：LLM 里求梯度的参数只能是 LoRA（c1、c2 一个也没有）；LoRA 是 fp32；读出模块的像素通路检查
        （带上适配器的模型）与参数量。"""
        ad = self.ad
        lora_ids = {id(p) for p in lora}
        bad = [n for n, p in ad.model.named_parameters() if p.requires_grad and id(p) not in lora_ids]
        if bad:
            raise RuntimeError(f"LoRA 以外的 LLM 参数求梯度：{bad[:5]}")
        if self.variant != "full" and lora:
            raise RuntimeError(f"{self.variant} 不该有 LoRA")
        if self.variant == "full" and (not lora or not all(p.requires_grad for p in lora)):
            raise RuntimeError("LoRA 参数不存在或没有设为求梯度")
        not32 = sorted({str(p.dtype) for p in lora if p.dtype != torch.float32})
        if not32:
            raise RuntimeError(f"LoRA 参数的 dtype 是 {not32}，应为 torch.float32")
        pix = check_no_pixel_path(self.readout, adapter_model=ad.model)
        out = {"variant": self.run_variant, "n_lora_params": int(sum(p.numel() for p in lora)),
               "lora_dtypes": sorted({str(p.dtype) for p in lora}),
               "query": None if self.query_embeds is None else {**dict(self.query), "shape": list(self.query_embeds.shape),
                                                                "dtype": str(self.query_embeds.dtype)},
               "layers": self.layers, "readout_params": self.readout.params, "pixel_path_check": pix,
               "gradient_checkpointing": bool(self.rc["gradient_checkpointing"])}
        if self.mixed:
            out["readout_variant"] = self.variant
        return out

    def check_base_unchanged(self) -> None:
        """base 权重（LoRA 以外的参数）的指纹与开训时相同（存检查点前调用）。"""
        fp = base_fingerprint(self.ad.model)
        if fp != self._fingerprint:
            raise RuntimeError(f"{self.variant}：LLM 的 base 权重变了（指纹 {fp[:12]} ≠ 开训时 {self._fingerprint[:12]}）")

    # ----- 前向 -----
    def features(self, inputs: Sequence[Any], *, semseg: bool) -> ReadoutInputs:
        if self.variant == "c2":
            return vision_inputs(self.ad, inputs, grid=self.cfg.query_grid, class_side=self.c2_class if semseg else None)
        return llm_inputs(self.ad, inputs, self.query_embeds, layers=self.layers,
                          class_spans=self.class_spans if semseg else None, enable_thinking=bool(self.rc["enable_thinking"]))

    def build_inputs(self, e: D.Example) -> Any:
        return self.ad.build_inputs([e.image], e.instruction, e.order)

    def lr_now(self) -> dict[str, float]:
        return {g.get("name", str(k)): float(g["lr"]) for k, g in enumerate(self.opt.param_groups)}

    def micro_batches(self, examples: Sequence[D.Example]) -> list[list[D.Example]]:
        """语义分割的样本与其余样本分开、各自按原顺序切成微批（语义分割在前；config.micro_batch_groups）。"""
        groups = micro_batch_groups([e.task for e in examples], int(self.rc["micro_batch_size"]))
        return [[examples[i] for i in g] for g in groups]

    def text_inputs(self, group: Sequence[Any]) -> list[Any]:
        """E3a+C3 的文本微批的输入（模块说明）：临时取消查询登记，用 C3 的 build_inputs 构造，再把查询登记回来（只改个数与模式，
        不碰任何行）。入口断言：每一行都没有查询 token。"""
        q = self.query
        if q is None or self.c3 is None:
            raise RuntimeError("文本微批只在 E3a+C3 里有")
        self.ad.clear_query_tokens()
        try:
            inputs = [self.c3.build_inputs(e) for e in group]
        finally:
            self.ad.inject_query_tokens(int(q["n"]), str(q["position_mode"]))
        bad = [i for i, x in enumerate(inputs) if x.query_span is not None]
        if bad:
            raise RuntimeError(f"文本样本 {bad} 的输入里有查询 token（C3 没有）")
        return inputs

    def optimize(self, examples: Sequence[D.Example], *, device: Any = None) -> tuple[dict[str, Any], dict[str, Any], list[Compute]]:
        """一步优化（规则见模块说明）。返回 (确定性的记录, 计时, 每条样本的 Compute，按 examples 的顺序)。"""
        if self.mixed:
            return self._optimize_mixed(examples, device=device)
        if not examples:
            raise ValueError("这一步没有样本")
        dev = device if device is not None else self.ad.device
        n = len(examples)
        tw = self.rc["loss"]["task_weights"]
        self.opt.zero_grad(set_to_none=True)
        self.readout.train()
        loss_sum = 0.0
        per: dict[str, list[float]] = {}
        counts = {"n_prompt_tokens": 0, "n_visual_tokens": 0, "n_image_pixels": 0, "n_query_tokens": 0, "n_class_rows": 0}
        times = {"build_inputs_s": 0.0, "fwd_s": 0.0, "bwd_s": 0.0, "opt_s": 0.0}
        comp_of: dict[int, Compute] = {}
        for group in self.micro_batches(examples):
            t0 = time.perf_counter()
            inputs = [self.build_inputs(e) for e in group]
            _sync(dev)
            t1 = time.perf_counter()
            semseg = group[0].task == "semseg"
            feats = self.features(inputs, semseg=semseg)
            ro = self.readout(*feats.args())
            total = ro.embed.new_zeros((), dtype=torch.float32)
            for task in dict.fromkeys(e.task for e in group):
                rows = [b for b, e in enumerate(group) if e.task == task]
                idx = torch.tensor(rows, dtype=torch.long, device=ro.embed.device)
                out = self.readout.task_output(task, ro.embed.index_select(0, idx),
                                               None if ro.class_emb is None else ro.class_emb.index_select(0, idx))
                ls = task_losses(task, out, stack_targets(task, [group[b] for b in rows], ro.embed.device),
                                 self.cfg.task(task)["loss"])
                total = total + float(tw[task]) * ls.sum()
                acc = per.setdefault(task, [0.0, 0])
                acc[0] += float(ls.detach().sum())
                acc[1] += len(rows)
            _sync(dev)
            t2 = time.perf_counter()
            (total / float(n)).backward()
            _sync(dev)
            t3 = time.perf_counter()
            times["build_inputs_s"] += t1 - t0
            times["fwd_s"] += t2 - t1
            times["bwd_s"] += t3 - t2
            loss_sum += float(total.detach())
            share = (t3 - t1) / len(group)
            for b, (e, inp) in enumerate(zip(group, inputs)):
                nv = sum(len(i.span) for i in inp.images)
                npx = sum(int(i.n_pixels or 0) for i in inp.images)
                counts["n_prompt_tokens"] += feats.prompt_lengths[b]
                counts["n_visual_tokens"] += nv
                counts["n_image_pixels"] += npx
                counts["n_query_tokens"] += 0 if inp.query_span is None else len(inp.query_span)
                counts["n_class_rows"] += int(semseg)
                comp_of[e.plan.k] = Compute.call(generated_tokens=0, prompt_tokens=int(feats.prompt_lengths[b]),
                                                 visual_tokens=int(nv), image_pixels=int(npx), call_time_s=share,
                                                 new_round=True)
            del feats, ro, total
        t4 = time.perf_counter()
        params = [p for g in self.opt.param_groups for p in g["params"]]
        mx = self.rc["optim"]["max_grad_norm"]
        gn = torch.nn.utils.clip_grad_norm_(params, float("inf") if mx is None else float(mx))
        lr = self.lr_now()
        self.opt.step()
        self.sched.step()
        self.opt.zero_grad(set_to_none=True)
        _sync(dev)
        times["opt_s"] = time.perf_counter() - t4
        rec = {"lr": lr, "loss": loss_sum / float(n),
               "loss_by_task": {k: v[0] / v[1] for k, v in sorted(per.items())},
               "n_by_task": {k: int(v[1]) for k, v in sorted(per.items())},
               "n_samples": n, **{k: int(v) for k, v in counts.items()}, "grad_norm": float(gn),
               "mix": self.readout.mix.snapshot() if self.readout.mix.learnable else None}
        return rec, times, [comp_of[e.plan.k] for e in examples]

    def _optimize_mixed(self, examples: Sequence[Any], *, device: Any = None) -> tuple[dict[str, Any], dict[str, Any], list[Compute]]:
        """E3a+C3 的一步（模块说明）：按流切微批（文本每 micro_batch_size 条一批，稠密按 E3a 的 micro_batches），文本微批的损失
        ÷ 这一步的文本监督 token 总数、稠密微批的损失 ÷ 这一步的稠密样本数，各乘 loss_weights 后反传；然后一次裁剪与 step。"""
        if not examples:
            raise ValueError("这一步没有样本")
        dev = device if device is not None else self.ad.device
        mcfg = self.rc["mixed"]
        by = {s: [e for e in examples if D.stream_of(e.plan) == s] for s in MIXED_STREAMS}
        n_text = int(sum(e.n_supervised for e in by["text"]))
        n_dense = len(by["dense"])
        weights = {s: float(mcfg["loss_weights"][s]) for s in MIXED_STREAMS}
        micro = int(self.rc["micro_batch_size"])
        chunks: list[tuple[str, list[Any]]] = []
        for s in mcfg["order"]:
            if s == "text":
                chunks += [("text", by["text"][i: i + micro]) for i in range(0, len(by["text"]), micro)]
            else:
                chunks += [("dense", g) for g in self.micro_batches(by["dense"])]
        tw = self.rc["loss"]["task_weights"]
        self.opt.zero_grad(set_to_none=True)
        self.readout.train()
        sums = {s: 0.0 for s in MIXED_STREAMS}
        per: dict[str, list[float]] = {}
        counts = {"n_prompt_tokens": 0, "n_visual_tokens": 0, "n_image_pixels": 0, "n_query_tokens": 0, "n_class_rows": 0}
        cnt_by = {"text": {"n_samples": len(by["text"]), "n_supervised_tokens": n_text, "n_target_tokens": 0, "n_prompt_tokens": 0,
                           "n_visual_tokens": 0, "n_image_pixels": 0},
                  "dense": {"n_samples": n_dense, "n_prompt_tokens": 0, "n_visual_tokens": 0, "n_image_pixels": 0,
                            "n_query_tokens": 0, "n_class_rows": 0}}
        times = {"build_inputs_s": 0.0, "fwd_s": 0.0, "bwd_s": 0.0, "opt_s": 0.0}
        comp_of: dict[int, Compute] = {}
        for key, group in chunks:
            t0 = time.perf_counter()
            if key == "text":  # C3：没有查询 token、全词表语言损失、不经过读出
                inputs = self.text_inputs(group)
                _sync(dev)
                t1 = time.perf_counter()
                out = self.ad.training_forward(inputs, [e.target_ids for e in group], enable_thinking=bool(self.c3_rc["enable_thinking"]))
                ls, p, _ = self.c3._task_loss(out, group, grid=False)
                _sync(dev)
                t2 = time.perf_counter()
                (ls / float(max(n_text, 1)) * weights["text"]).backward()
                _sync(dev)
                t3 = time.perf_counter()
                sums["text"] += float(ls.detach())
                for k, v in p.items():
                    acc = per.setdefault(f"text/{k}", [0.0, 0.0])
                    acc[0] += v[0]
                    acc[1] += v[1]
                share = (t3 - t1) / len(group)
                for b, (e, inp) in enumerate(zip(group, inputs)):
                    nv = sum(len(i.span) for i in inp.images)
                    npx = sum(int(i.n_pixels or 0) for i in inp.images)
                    tl, pl = int(out.target_lengths[b]), int(out.prompt_lengths[b])
                    counts["n_prompt_tokens"] += pl
                    counts["n_visual_tokens"] += nv
                    counts["n_image_pixels"] += npx
                    c = cnt_by["text"]
                    c["n_target_tokens"] += tl
                    c["n_prompt_tokens"] += pl
                    c["n_visual_tokens"] += nv
                    c["n_image_pixels"] += npx
                    comp_of[id(e)] = Compute.call(generated_tokens=tl, prompt_tokens=pl, visual_tokens=int(nv), image_pixels=int(npx),
                                                  call_time_s=share, new_round=True)
                del out, ls
            else:  # E3a 原样
                inputs = [self.build_inputs(e) for e in group]
                _sync(dev)
                t1 = time.perf_counter()
                semseg = group[0].task == "semseg"
                feats = self.features(inputs, semseg=semseg)
                ro = self.readout(*feats.args())
                total = ro.embed.new_zeros((), dtype=torch.float32)
                for task in dict.fromkeys(e.task for e in group):
                    rows = [b for b, e in enumerate(group) if e.task == task]
                    idx = torch.tensor(rows, dtype=torch.long, device=ro.embed.device)
                    o = self.readout.task_output(task, ro.embed.index_select(0, idx),
                                                 None if ro.class_emb is None else ro.class_emb.index_select(0, idx))
                    ls = task_losses(task, o, stack_targets(task, [group[b] for b in rows], ro.embed.device),
                                     self.cfg.task(task)["loss"])
                    total = total + float(tw[task]) * ls.sum()
                    acc = per.setdefault(f"dense/{task}", [0.0, 0])
                    acc[0] += float(ls.detach().sum())
                    acc[1] += len(rows)
                _sync(dev)
                t2 = time.perf_counter()
                (total / float(max(n_dense, 1)) * weights["dense"]).backward()
                _sync(dev)
                t3 = time.perf_counter()
                sums["dense"] += float(total.detach())
                share = (t3 - t1) / len(group)
                c = cnt_by["dense"]
                for b, (e, inp) in enumerate(zip(group, inputs)):
                    nv = sum(len(i.span) for i in inp.images)
                    npx = sum(int(i.n_pixels or 0) for i in inp.images)
                    nq = 0 if inp.query_span is None else len(inp.query_span)
                    pl = int(feats.prompt_lengths[b])
                    for d in (counts, c):
                        d["n_prompt_tokens"] += pl
                        d["n_visual_tokens"] += nv
                        d["n_image_pixels"] += npx
                        d["n_query_tokens"] += nq
                        d["n_class_rows"] += int(semseg)
                    comp_of[id(e)] = Compute.call(generated_tokens=0, prompt_tokens=pl, visual_tokens=int(nv), image_pixels=int(npx),
                                                  call_time_s=share, new_round=True)
                del feats, ro, total
            times["build_inputs_s"] += t1 - t0
            times["fwd_s"] += t2 - t1
            times["bwd_s"] += t3 - t2
        t4 = time.perf_counter()
        params = [p for g in self.opt.param_groups for p in g["params"]]
        mx = self.rc["optim"]["max_grad_norm"]
        gn = torch.nn.utils.clip_grad_norm_(params, float("inf") if mx is None else float(mx))
        lr = self.lr_now()
        self.opt.step()
        self.sched.step()
        self.opt.zero_grad(set_to_none=True)
        _sync(dev)
        times["opt_s"] = time.perf_counter() - t4
        loss_by = {"text": sums["text"] / n_text if n_text else None, "dense": sums["dense"] / n_dense if n_dense else None}
        loss = sum(weights[s] * loss_by[s] for s in mcfg["order"] if loss_by[s] is not None)
        n_by: dict[str, int] = {}
        for e in examples:
            k = f"{D.stream_of(e.plan)}/{e.plan.task}"
            n_by[k] = n_by.get(k, 0) + 1
        rec = {"lr": lr, "loss": loss,
               "loss_by_task": dict(sorted({**{k: v[0] / v[1] for k, v in per.items()}, **loss_by}.items())),
               "n_by_task": dict(sorted(n_by.items())),
               "n_samples": len(examples), **{k: int(v) for k, v in counts.items()}, "grad_norm": float(gn),
               "mix": self.readout.mix.snapshot() if self.readout.mix.learnable else None,
               "n_supervised_by_task": {k: int(v[1]) for k, v in sorted(per.items()) if k.startswith("text/")},
               "by_supervision": cnt_by}
        return rec, times, [comp_of[id(e)] for e in examples]

    # ----- 检查点与续跑状态 -----
    def save(self, directory: Path) -> Path:
        self.check_base_unchanged()
        return save_checkpoint(self.ad, self.readout, None if self.query_embeds is None else self.query_embeds.detach(),
                               directory, query=self.query, layers=self.layers)

    def state_dict(self) -> dict[str, Any]:
        st = {"optimizer": self.opt.state_dict(), "scheduler": self.sched.state_dict(), "rng_cpu": torch.get_rng_state()}
        if torch.cuda.is_available():
            st["rng_cuda"] = torch.cuda.get_rng_state_all()
        return st

    def load_state_dict(self, st: Mapping[str, Any]) -> None:
        self.opt.load_state_dict(st["optimizer"])
        self.sched.load_state_dict(st["scheduler"])
        torch.set_rng_state(st["rng_cpu"])
        if "rng_cuda" in st and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(st["rng_cuda"])


# ---------------------------------------------------------------------------
# 运行身份与输出
# ---------------------------------------------------------------------------


def _scalars(d: Mapping[str, Any]) -> dict[str, Any]:
    out = {}
    for k, v in d.items():
        if k in ("dedupe_file", "dedupe_excluded"):
            continue
        if isinstance(v, (int, float, str, bool)) or v is None:
            out[k] = v
        elif isinstance(v, Mapping):
            out[k] = _scalars(v)
    return out


TEXT_BOX_FORMAT = "adapter.format_boxes（本家族原生格式，label = 指代句；DEV-0030 §3）"  # 与 C3 的运行身份同一句（tests 钉住）


def text_stream_identity(trainer: Trainer, builder: Any) -> dict[str, Any]:
    """E3a+C3 的运行身份里文本流的一节：C3 的配置（路径、sha256、版本）、决定样本流与损失的 C3 训练参数、指令与 C3 节（含实际用的
    E1 提示与出处）、停止 token。各项与同一种子的 C3 运行身份（src/e3/train.run_identity）的对应项相同（tests 钉住）。"""
    tc = trainer.text_cfg
    d = tc.doc
    c3 = {k: v for k, v in d["c3"].items() if k != "basis"}
    c3["polygon_template"] = builder.polygon_template
    c3["polygon_template_source"] = builder.polygon_source
    c3["box_template"] = builder.box_template
    c3["box_template_source"] = builder.box_source
    c3["box_format"] = TEXT_BOX_FORMAT
    return {"variant": E3C.VARIANT_OF[TEXT_SUPERVISION], "supervision": TEXT_SUPERVISION,
            "config": {"path": tc.path, "sha256": tc.sha256, "version": d["version"]},
            "training": {k: trainer.c3_rc[k] for k in ("steps", "batch_size", "micro_batch_size", "mix", "loss_reduction",
                                                        "enable_thinking")},
            "instructions": {t: {k: v for k, v in d["tasks"][t].items() if k.endswith(("template", "instruction")) or k == "order"}
                             for t in tc.tasks},
            "c3": c3, "stop_id": trainer.c3.stop_id}


def run_identity(spec: RunSpec, cfg: E3aConfig, run_cfg: Mapping[str, Any], trainer: Trainer, pools: Mapping[str, Any],
                 dedupe: Mapping[str, Any], class_check: Mapping[str, Any], *, text_builder: Any = None) -> dict[str, Any]:
    """运行身份（config.yaml）。E3a+C3 另加 text_stream 一节（text_stream_identity；要给 text_builder），其余变体不加键。"""
    d = cfg.doc
    instr = {t: {"order": cfg.order(t), "instruction_template": d["tasks"][t]["instruction_template"]} for t in pools}
    if "semseg" in pools:
        instr["semseg"]["class_names_sha256"] = hashlib.sha256("\n".join(cfg.class_names).encode("utf-8")).hexdigest()
    out = {
        "experiment": "E3a",
        "framework": "src/readout（DEV-0034）",
        "model": {"key": spec.model_key, "family": spec.family, "repo": spec.repo, "revision": spec.revision,
                  "adapter_model_key": spec.adapter_model_key, "tiny": spec.tiny, "dtype": spec.model_dtype},
        "image_policy": spec.image_policy,
        "variant": spec.variant,
        "seed": spec.seed,
        "training": dict(run_cfg),
        "readout": {"spec": trainer.readout.spec.as_dict(), "sources": list(trainer.readout.sources),
                    "params": trainer.readout.params, "layers": trainer.layers, "query": trainer.query},
        "instructions": instr,
        "class_names_check": dict(class_check),
        "losses": {t: {k: v for k, v in d["tasks"][t]["loss"].items()} for t in pools},
        "gt": {t: {k: d["tasks"][t][k] for k in ("gt", "min_valid_fraction") if k in d["tasks"][t]} for t in pools},
        "config": {"path": cfg.path, "sha256": cfg.sha256, "version": d["version"]},
        "data": {t: {"pool": _scalars(pools[t].stats), "n": len(pools[t]), "ids_sha256": ids_digest([s.id for s in pools[t].records]),
                     "dedupe": dict(dedupe[t])} for t in pools},
    }
    if trainer.mixed:
        if text_builder is None:
            raise ValueError("E3a+C3 的运行身份要给 text_builder")
        out["text_stream"] = text_stream_identity(trainer, text_builder)
    return out


def accounting_doc(leaf: Path) -> tuple[dict[str, Any], Compute]:
    """logs/ 的合计：各次调用的分阶段墙钟相加、显存峰值取最大；compute.json 用的 Compute 由每步的逐样本记录相加。"""
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((leaf / "logs").glob("run_*.json"))]
    tot: dict[str, float] = {}
    peaks: list[int] = []
    computes: list[Compute] = []
    n_steps = 0
    for p in sorted((leaf / "logs").glob("timing_run_*.jsonl")):
        for r in _read_log(p):
            n_steps += 1
            for k in ("data_s", "build_inputs_s", "fwd_s", "bwd_s", "opt_s", "step_s"):
                tot[k] = tot.get(k, 0.0) + float(r[k])
            if r.get("peak_mem_bytes") is not None:
                peaks.append(int(r["peak_mem_bytes"]))
            computes.append(Compute(**r["compute"]))
    comp = merge(computes)
    log = _read_log(leaf / TRAIN_LOG)
    sums = {k: int(sum(r[k] for r in log)) for k in ("n_samples", "n_prompt_tokens", "n_visual_tokens", "n_image_pixels",
                                                     "n_query_tokens", "n_class_rows")}
    wall = sum(float(r.get("wall_s") or 0.0) for r in runs)
    step_s = [float(r["step_s"]) for p in sorted((leaf / "logs").glob("timing_run_*.jsonl")) for r in _read_log(p)]
    doc = {"n_runs": len(runs), "n_steps_timed": n_steps, "n_steps_logged": len(log), **sums, "phases_s": tot,
           "step_s_last": step_s[-1] if step_s else None,
           "step_s_median": float(sorted(step_s)[len(step_s) // 2]) if step_s else None,
           "wall_s": wall, "gpu_hours": wall / 3600.0, "peak_mem_bytes": max(peaks) if peaks else None,
           "device": runs[-1].get("device") if runs else None, "gpu": runs[-1].get("gpu") if runs else None,
           "note": "phases_s 按步相加：data_s = 主线程等样本（读图、解码 GT、降到 128）；build_inputs_s = 处理器；fwd_s、bwd_s = 前向"
                   "（LLM、读出与损失）与反向（CUDA 前后同步）；opt_s = 裁剪、step；step_s = 整步。compute.json 的 generated_tokens 恒为 0"
                   "（没有生成），prompt_tokens 是送进 LLM 的提示长度（c2 不跑 LLM，记 0）"}
    if any("by_supervision" in r for r in log):  # E3a+C3：两种监督各自的样本数与 token 数（每步的 by_supervision 相加）
        doc["by_supervision"] = by_supervision_totals(log)
        doc["note"] += ("。E3a+C3：文本样本（C3）的 generated_tokens 是教师强制的目标 token 数（同 C3），稠密样本为 0；"
                        "by_supervision 是两种监督各自的合计")
    return doc, comp


def by_supervision_totals(log: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]] | None:
    """E3a+C3：train_log 每步的 by_supervision 相加。"""
    if not log:
        return None
    return {s: {k: int(sum(r["by_supervision"][s][k] for r in log)) for k in log[0]["by_supervision"][s]} for s in MIXED_STREAMS}


def readme_text(spec: RunSpec, ident: Mapping[str, Any], summary: Mapping[str, Any], log: Sequence[Mapping[str, Any]]) -> str:
    tr = ident["training"]
    first, last = (log[0], log[-1]) if log else ({}, {})
    p = ident["readout"]["params"]
    lines = [f"# E3a 训练 · {spec.variant} · {spec.model_key} · seed {spec.seed}", "",
             f"- 模型：{spec.repo}@{spec.revision}（{spec.family}；适配器 {spec.adapter_model_key}；{spec.model_dtype}）；图像口径 {spec.image_policy}",
             f"- 变体：{spec.variant}；参数组 {list(tr['optim']['groups'])}；LoRA {tr['lora']}",
             f"- 训练：{tr['steps']} 步 × {tr['batch_size']} 条（微批 {tr['micro_batch_size']} × 累积 {tr['grad_accum_steps']}），"
             f"warmup {tr['warmup_steps']} 步 + 余弦；混合 {tr['mix']}；任务权重 {tr['loss']['task_weights']}",
             f"- 读出：解码器（含混合与类别投影）{p['decoder_budget_total']} 个参数（上限 {p['limits']['decoder_budget_total']}）；"
             f"线性头 {p['heads']}（每个上限 {p['limits']['each_head']}）；来源 {ident['readout']['sources']}",
             f"- 第一步损失 {first.get('loss')}，最后一步损失 {last.get('loss')}（{len(log)} 步）",
             f"- 混合权重：训练前 {summary.get('layer_mix_init')}；训练后 {summary.get('layer_mix_final')}",
             f"- 检查点：{summary.get('final_checkpoint')}（{summary.get('final_checkpoint_bytes')} 字节；model/ + readout/）", ""]
    if "mixed" in tr:  # E3a+C3
        m = tr["mixed"]
        ref = m["reference"]
        ts = ident["text_stream"]
        lines += ["## E3a+C3 的两条样本流", "",
                  f"- 文本：{m['streams']['text']}；稠密：{m['streams']['dense']}。先后 {m['order']}，每个微批只含一种监督"
                  f"（文本 {m['micro_batches']['text']} 个微批；稠密 {m['micro_batches']['dense']['per_step'] or m['micro_batches']['dense']['rule']}）",
                  f"- 全程两条流各 {ref['text']['samples']}、{ref['dense']['samples']} 条 = C3、E3a full 单独训练的全部样本，顺序相同；文本流的"
                  f"配置 {ts['config']['path']}（{ts['config']['version']}，sha256 {ts['config']['sha256']}）",
                  "- 两条流的槽位与样本的对应只由种子与 k 决定：同一条样本会以两种监督各出现一次（第 0 步在同一步里，之后文本流在前、"
                  "稠密流约 2 倍步数之后），不另换数据种子",
                  f"- 损失：每步 = {m['loss_weights']['text']:g} × 文本的逐 token 平均（同 C3）+ {m['loss_weights']['dense']:g} × "
                  "稠密的逐样本平均（同 E3a）（事先固定，D83-2）；train_log 的 loss 是这个和，loss_by_task 的 text、dense 两项是两种各自的平均",
                  f"- 学习率：warmup {tr['warmup_steps']} 步 + 余弦，按 {tr['steps']} 步排，日程长度是 C3（{ref['text']['steps']} 步）的 "
                  f"{m['schedule_length_ratio']['text']:g} 倍、E3a full（{ref['dense']['steps']} 步）的 {m['schedule_length_ratio']['dense']:g} 倍",
                  "- 文本样本不插查询 token、不经过读出；检查点同 E3a full（eval_e3a --variant full、run_e1 --checkpoint <final/model>）",
                  f"- 本次合计（train_log 的 by_supervision）：{summary.get('by_supervision')}", ""]
    lines += ["## 文件", "",
              "- train_log.jsonl：每步的确定性记录；logs/：每次调用的记录与逐步计时；accounting.json、compute.json：记账；",
              "- config.yaml：运行身份；meta.json：git、主机、包版本。", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 正式运行
# ---------------------------------------------------------------------------


def run_training(adapter: Any, cfg: E3aConfig, spec: RunSpec, run_cfg: Mapping[str, Any], paths: D.E3aPaths,
                 out_dir: str | Path, *, resume: bool = False, argv: Sequence[str] = (), ckpt_root: str | Path | None = None,
                 stop_after_steps: int | None = None, keep_optimizer_state: bool = False,
                 log: Callable[[str], None] | None = None, _fail_after_step: int | None = None) -> dict[str, Any] | None:
    """跑一次训练；返回 train_summary（stop_after_steps 到了时返回 None）。adapter 必须是刚挂好的 base（规则见模块说明）。
    spec.variant 是运行变体（e3a_c3 时 run_cfg 要有 mixed 一节，文本流的配置按配置的 mixed.text_stream 读取）。"""
    log = log or (lambda m: print(m, file=sys.stderr, flush=True))
    rc = dict(run_cfg)
    mixed = spec.variant == E3A_C3
    if mixed and rc.get("variant") != E3A_C3:
        raise ValueError(f"run_cfg 的变体 {rc.get('variant')!r} 与 RunSpec 的 {spec.variant!r} 不符")
    tag = f"[e3a {spec.variant} {spec.model_key} seed{spec.seed}]"
    leaf = leaf_dir(out_dir, spec.variant, spec.model_key, spec.seed)
    ckpt = (Path(ckpt_root) / spec.variant / spec.model_key / f"seed{spec.seed}") if ckpt_root else leaf / "ckpt"
    latest = ckpt / LATEST

    pools, dedupe = {}, {}
    for t in cfg.tasks:
        pools[t] = D.load_pool(t, paths, cfg)
        dedupe[t] = check_dedupe(t, pools[t].stats.get("dedupe_file"), cfg.doc["dedupe"]["required_eval_sets"][t],
                                 real_model=not spec.tiny)
        if not pools[t].records:
            raise ValueError(f"{t} 的训练池是空的")
    class_check = D.check_class_names(paths.semseg_root, cfg.class_names) if "semseg" in pools else {"checked": False}

    exists = leaf.exists() and any(leaf.iterdir())
    if exists and not resume:
        raise FileExistsError(f"{leaf} 已存在且非空；续跑请加 --resume")
    if exists and (leaf / SUMMARY).exists():
        s = json.loads((leaf / SUMMARY).read_text(encoding="utf-8"))
        if s.get("complete"):
            log(f"{tag} 已完成，跳过（{leaf / SUMMARY}）")
            return s

    text_cfg = text_stream_config(cfg.doc) if mixed else None
    if text_cfg is not None and int(spec.seed) not in text_cfg.seeds:
        raise ValueError(f"种子 {spec.seed} 不在文本流配置的 seeds {list(text_cfg.seeds)} 里")
    trainer = Trainer(adapter, cfg, rc, variant=spec.variant, seed=spec.seed, tiny=spec.tiny, text_cfg=text_cfg)
    st = None
    if exists and (latest / RESUME_STATE).exists():
        trainer.setup_resume(latest)
        st = torch.load(latest / RESUME_STATE, map_location="cpu", weights_only=False)
    else:
        trainer.setup_fresh()
    records = {t: p.records for t, p in pools.items()}
    builder: Any = D.ExampleBuilder(cfg, records, out_size=trainer.readout.spec.out_size)
    text_builder = None
    if mixed:  # E3a+C3：C3 的文本流 + E3a full 的稠密流（data.MixedSchedule、MixedBuilder）
        text_builder = D.c3_text_builder(text_cfg, records, seed=spec.seed, adapter=adapter, family=spec.family)
        builder = D.MixedBuilder(builder, text_builder)
        schedule: Any = D.MixedSchedule(records, cfg, text_cfg, seed=spec.seed, per_step=rc["mixed"]["per_step"],
                                        order=rc["mixed"]["order"])
        if schedule.batch_size != int(rc["batch_size"]):
            raise ValueError(f"E3a+C3 每步 {schedule.batch_size} 条，与 run_cfg 的 batch_size {rc['batch_size']} 不同")
    else:
        schedule = D.Schedule(records, cfg, seed=spec.seed, batch_size=int(rc["batch_size"]))
    ident = run_identity(spec, cfg, rc, trainer, pools, dedupe, class_check, text_builder=text_builder)
    ident_sha = identity_sha256(ident)
    if exists:
        if not (leaf / CONFIG).exists():
            raise FileExistsError(f"{leaf} 里没有 {CONFIG}，不能续跑")
        old = yaml.safe_load((leaf / CONFIG).read_text(encoding="utf-8"))
        if _identity(old) != _identity(ident):
            diff = sorted(k for k in set(_identity(old)) | set(_identity(ident)) if _identity(old).get(k) != _identity(ident).get(k))
            raise ValueError(f"{leaf}：续跑的运行身份与已有的不同（{diff}），拒绝混写")
    else:
        leaf.mkdir(parents=True, exist_ok=True)
        files = [REPO_ROOT / cfg.path if not Path(cfg.path).is_absolute() else Path(cfg.path)]
        if text_cfg is not None:
            files.append(REPO_ROOT / text_cfg.path if not Path(text_cfg.path).is_absolute() else Path(text_cfg.path))
        files += sorted({Path(p.stats["dedupe_file"]) for p in pools.values() if p.stats.get("dedupe_file")})
        write_run_metadata(leaf, config=ident, seed=spec.seed, data_files=files, repo_root=REPO_ROOT)
    start = 0
    if st is not None:
        if st["identity_sha256"] != ident_sha:
            raise ValueError(f"{latest / RESUME_STATE} 的运行身份与本次不同，拒绝续跑")
        trainer.load_state_dict(st)
        start = int(st["step"])
        trainer.mix_init = st.get("mix_init")
        log(f"{tag} 从第 {start} 步续跑")
    tl = [r for r in _read_log(leaf / TRAIN_LOG) if r["step"] < start]
    _atomic_write_text(leaf / TRAIN_LOG, "".join(_jsonl(r) for r in tl))

    (leaf / "logs").mkdir(exist_ok=True)
    dev = torch.device(adapter.device)
    run_k = len(list((leaf / "logs").glob("run_*.json"))) + 1
    run_path = leaf / "logs" / f"run_{run_k:03d}.json"
    timing_path = leaf / "logs" / f"timing_run_{run_k:03d}.jsonl"
    run_log: dict[str, Any] = {"argv": list(argv), "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                               "device": str(dev), "gpu": torch.cuda.get_device_name(dev) if dev.type == "cuda" else None,
                               "paths": paths.as_dict(), "ckpt_dir": str(ckpt), "start_step": start, "finished_utc": None,
                               "audit": trainer.audit}
    run_path.write_text(json.dumps(run_log, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    t_all = time.perf_counter()
    done_box = [0]

    def finish_log(status: str) -> None:
        _sync(dev)
        run_log.update(status=status, finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                       wall_s=round(time.perf_counter() - t_all, 3), steps_this_run=done_box[0])
        run_path.write_text(json.dumps(run_log, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        acc, comp = accounting_doc(leaf)
        (leaf / ACCOUNTING).write_text(json.dumps(acc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        write_compute_json(leaf / COMPUTE, comp)

    def save_latest(step_done: int) -> None:
        tmp = ckpt / (LATEST + ".tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        trainer.save(tmp)
        torch.save({"identity_sha256": ident_sha, "step": step_done, "mix_init": trainer.mix_init, **trainer.state_dict()},
                   tmp / RESUME_STATE)
        old = ckpt / (LATEST + ".old")
        if old.exists():
            shutil.rmtree(old)
        if latest.exists():
            latest.replace(old)
        tmp.replace(latest)
        if old.exists():
            shutil.rmtree(old)

    steps = int(rc["steps"])
    every = int(rc["ckpt_every"])
    workers = int(rc["data_workers"])
    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="e3a-data") if workers > 0 else None

    def build_one(plan: Any) -> tuple[Any, dict[str, Any] | None]:
        try:
            return builder.build(plan), None
        except (NoSupervision, TextNoSupervision) as e:  # 后者只来自 C3 的构造（E3a+C3 的文本流：抽不到点对）
            rec = {"k": plan.k, "id": plan.sample_id, "reason": str(e)}
            return None, ({"stream": D.stream_of(plan), **rec} if mixed else rec)

    def submit(step: int) -> list[Any]:
        plans = schedule.step_plans(step)
        return [pool.submit(build_one, p) for p in plans] if pool is not None else plans

    def collect(handle: list[Any]) -> tuple[list[D.Example], list[dict[str, Any]]]:
        res = [h.result() for h in handle] if pool is not None else [build_one(p) for p in handle]
        return [e for e, _ in res if e is not None], [s for _, s in res if s is not None]

    try:
        adapter.train_mode()
        done = 0
        pending = submit(start) if start < steps else None
        for step in range(start, steps):
            if dev.type == "cuda":
                torch.cuda.reset_peak_memory_stats(dev)
            t0 = time.perf_counter()
            examples, skipped = collect(pending)
            pending = submit(step + 1) if step + 1 < steps else None
            t_data = time.perf_counter() - t0
            rec, times, comps = trainer.optimize(examples, device=dev)
            step_s = time.perf_counter() - t0
            comps = [c.as_sample(wall_time_s=step_s / len(examples)) for c in comps]
            if step == start and not mixed:
                trainer.audit["first_step_examples"] = [{"id": e.plan.sample_id, "task": e.task, "image_hw": list(e.image.shape[:2]),
                                                         "instruction_chars": len(e.instruction)} for e in examples[:4]]
            elif step == start:  # E3a+C3：两条流各前 4 条
                shown = [e for s in MIXED_STREAMS for e in [x for x in examples if D.stream_of(x.plan) == s][:4]]
                trainer.audit["first_step_examples"] = [
                    {"stream": D.stream_of(e.plan), "id": e.plan.sample_id, "task": e.plan.task,
                     "image_hw": list((e.images[0] if D.stream_of(e.plan) == "text" else e.image).shape[:2]),
                     "instruction_chars": len(e.instruction)} for e in shown]
            line = {"step": step, **rec, "n_skipped": len(skipped), "skipped": skipped}
            _append(leaf / TRAIN_LOG, [_jsonl(line)])
            _append(timing_path, [_jsonl({"step": step, "data_s": t_data, **times, "step_s": step_s,
                                          "peak_mem_bytes": int(torch.cuda.max_memory_allocated(dev)) if dev.type == "cuda" else None,
                                          "compute": {k: v for k, v in merge(comps).to_json_dict().items()
                                                      if k not in ("schema_version", "per_sample_mean")}})])
            done += 1
            done_box[0] = done
            s1 = step + 1
            if s1 % 10 == 0 or s1 == steps:
                log(f"{tag} 第 {s1}/{steps} 步 loss {rec['loss']:.4f}（{step_s:.2f} s/步）")
            stop = stop_after_steps is not None and done >= stop_after_steps and s1 < steps
            if (s1 % every == 0 and s1 < steps) or stop:
                save_latest(s1)
            if _fail_after_step is not None and s1 == _fail_after_step:
                raise RuntimeError(f"测试注入的中断（第 {s1} 步之后）")
            if stop:
                raise StopRequested(f"本次跑了 {done} 步（到第 {s1}/{steps} 步），检查点已写")
        adapter.eval_mode()
        trainer.readout.eval()
        final = ckpt / FINAL
        if final.exists():
            shutil.rmtree(final)
        final.parent.mkdir(parents=True, exist_ok=True)
        trainer.save(final)
        if not keep_optimizer_state and latest.exists():
            shutil.rmtree(latest)
        man = read_manifest(final / MODEL_SUB)
        tlog = _read_log(leaf / TRAIN_LOG)
        summary = {"complete": True, "steps": steps, "variant": spec.variant, "model": spec.model_key, "seed": spec.seed,
                   "identity_sha256": ident_sha, "final_loss": tlog[-1]["loss"] if tlog else None,
                   "final_checkpoint": str(final.relative_to(leaf)) if final.is_relative_to(leaf) else str(final),
                   "final_checkpoint_bytes": _dir_bytes(final),
                   "final_checkpoint_files": {**{f"{MODEL_SUB}/{k}": v["bytes"] for k, v in man["files"].items()},
                                              **{f"{READOUT_SUB}/{k}": v["bytes"] for k, v in
                                                 read_readout_manifest(final / READOUT_SUB)["files"].items()}},
                   "lora": man.get("lora"), "readout_params": trainer.readout.params,
                   "layer_mix_init": trainer.mix_init,
                   "layer_mix_final": trainer.readout.mix.snapshot() if trainer.readout.mix.learnable else None,
                   "layers": trainer.layers}
        if mixed:  # E3a+C3：两种监督各自的合计（每步的 by_supervision 相加）
            summary["by_supervision"] = by_supervision_totals(tlog)
        (leaf / README).write_text(readme_text(spec, ident, summary, tlog), encoding="utf-8")
        _atomic_write_text(leaf / SUMMARY, json.dumps(summary, indent=1, ensure_ascii=False) + "\n")
        finish_log("complete")
        return summary
    except StopRequested as e:
        log(f"[停止] {e}")
        adapter.eval_mode()
        finish_log("stopped")
        return None
    except BaseException:
        finish_log("error")
        raise
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)


__all__ = [
    "CheckpointMismatch",
    "EXPERIMENT_SUB",
    "FINAL",
    "LATEST",
    "MODEL_SUB",
    "READOUT_SUB",
    "RunSpec",
    "StopRequested",
    "TEXT_BOX_FORMAT",
    "Trainer",
    "accounting_doc",
    "by_supervision_totals",
    "init_query_embeds",
    "layers_for",
    "leaf_dir",
    "load_readout",
    "read_readout_manifest",
    "readout_tensors",
    "run_identity",
    "run_training",
    "save_checkpoint",
    "save_readout",
    "text_stream_identity",
]
