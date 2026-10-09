"""E3b 的训练循环（DEV-0025 §3）：一个模型 × 一种监督 × 一个种子 → <out>/<变体>/<模型>/seed<k>/。

**顺序**（适配器的顺序约束照旧检查）：加载 base → torch.manual_seed(种子) → add_grid_tokens（网格监督；C3 不加）→
inject_lora（秩、alpha、dropout 取配置；目标 = lora_target_modules()）→ trainable_token_rows(网格 token)（网格监督）→
梯度检查点 → 优化器。续跑：load_trained(ckpt/latest/model, merge=False)（restore_tokens → 挂 LoRA，不合并）→
trainable_token_rows → 梯度检查点 → 优化器，再载入优化器、调度器、token 行主副本与随机数状态。

**损失**（一步之内全部监督 token 的平均：各微批的损失和除以整步的监督 token 总数，梯度累积与一次大批等价）：
- 网格：training_forward → 按行的任务分组 → restricted_logits(隐藏状态, 本任务的允许集合) 的交叉熵 × 逐 token 权重
  （深度不可监督的格子权重 0），只算目标位置（training_forward 只返回目标位置的隐藏状态）。
- C3：chunked_lm_loss（全词表，分块）。

**优化**：AdamW，两个参数组：LoRA（lr、weight_decay 取配置），网格 token 行（weight_decay = 0，D45-3）。token 行的两种模式
（optim.token_rows.mode）：
- fp32_master_rows（默认）：为 trainable_token_rows 打开的这几行建 fp32 的主副本（嵌入，不共享时另有 lm_head 的同一组行），
  优化器只管主副本；每步反向之后从嵌入矩阵的稠密梯度里取这几行给主副本（其余行的梯度被适配器的掩码钩子置 0，不用），
  step 之后把主副本写回权重（模型 dtype）。冻结行从不被写，save_trained 的冻结行核对照旧。
- dense_param：把 trainable_token_rows 返回的整个矩阵交给优化器（weight_decay = 0）；冻结行梯度为 0，Adam 的更新恰为 0。
  fp32 上两种模式逐位相同（tests 钉住）；bf16 上 dense_param 的更新小于半个 ulp 时被舍掉，优化器状态是整个 [V, H]。
warmup（线性，从 1/warmup 起）+ 余弦衰减到 min_lr_ratio；梯度裁剪 max_grad_norm（null 时只记范数不裁）；bf16 权重 + fp32 的
LoRA（peft 的默认）；梯度检查点。

**检查点**（D34：只存 LoRA、token 行、manifest，不存合并后的权重）：每 ckpt_every 步与 --stop-after-steps 停下时写
<ckpt>/latest/：model/（save_trained）+ resume_state.pt（优化器、调度器、主副本、随机数状态、步数、运行身份的 sha256），
先写到 latest.tmp 再换名，只留最新一份。训练完成后写 <ckpt>/final/model/（只有 save_trained），并删掉 latest（优化器状态只为续跑；
--keep-optimizer-state 保留）。<ckpt> 默认是 <叶目录>/ckpt（.gitignore 的 experiments/E3/**/ckpt/）。

**续跑**：叶目录非空时必须给 resume 且 config.yaml 与本次的运行身份完全相同；从 latest 的步数接着跑，train_log.jsonl 里之后的行
截掉重算。样本日程无状态、随机数状态随检查点保存，所以 CPU 上续跑与一次跑完逐位一致（tests 钉住）。

**记账**：
- train_log.jsonl：每步一行，只含由训练本身决定的量（步、lr、损失、各任务损失、样本数、各类样本数、监督 / 目标 / 提示 / 视觉
  token 数、像素数、梯度范数、网格 token 的教师强制准确率），CPU 上续跑与一次跑完逐字节相同。
- logs/run_<k>.json：每次调用一份（argv、设备、GPU、起止时间、状态、本次跑的步数、运行期审计）；
  logs/timing_run_<k>.jsonl：每步的墙钟（data_s、fwd_s、bwd_s、opt_s、step_s，CUDA 前后同步）与显存峰值
  （torch.cuda.max_memory_allocated，每步重置；CPU 上为 null）。
- 数据：下一步的样本由线程池（data.workers 个线程）在本步的 GPU 计算期间构造；每条样本由 (种子, k) 唯一决定，与线程的先后无关。
  data_s 是主线程等样本的时间（与 GPU 重叠的部分不计）。
- accounting.json：各次调用合计（步数、样本、各类 token、分阶段墙钟、GPU 小时、显存峰值）；compute.json：src/accounting 的
  Compute（每条样本记一次调用：generated_tokens = 教师强制的目标 token 数，prompt_tokens = 提示行长度，visual_tokens、
  image_pixels 按行；call_time_s = 所在微批前向 + 反向墙钟的等分，wall_time_s = 整步墙钟的等分）。

**C1′**（DEV-0035 §1；变体 c1prime，run_cfg 的 frozen_llm = true；配置的 c1prime 节）：网格监督，数据、日程、步数、批、调度都同 E3b，
只是不注入 LoRA：加载 base → torch.manual_seed → add_grid_tokens → frozen_llm_token_rows（LLM 的全部参数设为不求梯度，只让输入嵌入、
不共享时另有 lm_head 的权重求梯度，不挂梯度掩码）→ TokenRows（必须是 fp32_master_rows：主副本只取网格 token 那几行的梯度、step 后
只写回那几行，其余行逐位不变）→ 梯度检查点 → 优化器（只有 token_rows 一组）。每次存检查点之前核对冻结行的抽样（FrozenRowsProbe）
与开始时逐位相同。检查点只有 token 行与 manifest（save_trained 在没有 LoRA 时就这样存）；续跑：load_trained(merge=False) 之后 LoRA
状态必须是 none。

**网格格式预热**（DEV-0035 §2；监督 warmup，变体 warmup）：LoRA + 网格 token 行、受限交叉熵同 E3b 的网格监督；样本来自
src/e3/warmup.WarmupSource（纯文字，不给图像；不读训练池、没有去重清单）。每 warmup.eval_every 步在固定的留出集上做受限解码
（warmup.evaluate_heldout），读数逐行写进 warmup_eval.jsonl（只含由模型输出决定的量，CPU 上续跑与一次跑完逐字节相同；墙钟在
logs/run_<k>.json 的 warmup_evals 与 accounting.json 的 warmup_eval_s）；格式错误率连续 consecutive 次低于阈值（warmup.StopRule）就
停在这一步，直接存 final；steps（= warmup.stop.max_steps）是上限。train_summary.json 另有 warmup 一节（是否满足、停在第几步、读数）。

**E3b+C3**（DEV-0042 §1；监督 grid+text，变体 e3b_c3；配置的 mixed 节）：搭建同 E3b（网格 token + LoRA + token 行）；样本来自
src/e3/data.MixedSchedule（每步 = C3 同一种子的文本流 per_step.text 条 + E3b 同一种子的网格流 per_step.grid 条，与两者单独训练逐个相同）。
一步之内按流分开切微批（每个微批只含一种监督，文本在前），文本微批用 chunked_lm_loss（全词表）、网格微批用受限交叉熵；每种监督各自
除以这一步里它自己的监督 token 总数（与 C3、E3b 单独训练的 token_mean 同一个定义），再乘 loss_weights（1 : 1）相加：
loss = 1 · L_text + 1 · L_grid。train_log 的 loss 是这个和；loss_by_task 的 text、grid 两项是 L_text、L_grid，text/<任务>、grid/<任务>
是分任务的逐 token 平均；n_supervised_by_task 同样的键（不含两项合计）；by_supervision 记两种监督各自的样本数与 token 数（accounting.json
另有合计）。学习率的 warmup 与余弦按 mixed.steps（8000）排：日程长度是 C3、E3b 的 2 倍（运行身份 training.mixed 与 README 写明）。

**C3+预热**（DEV-0042 §2；监督 warmup，变体 c3_warmup；RunSpec.parent_checkpoint = C3 的 ckpt/final/model）：开训前核对起点（resolve_parent：
构造方式同本次、有 LoRA、没有新增 token 行与查询、叶目录 config.yaml 的 variant 是 c3、seed 等于本次的种子），把它的 LoRA 合并进 base
（src/e5/stack.merge_start_lora，与 E5 合并起点同一串调用，不改适配器的 LoRA 状态），再 torch.manual_seed → 加网格 token → 注入预热的
LoRA（与基座上的预热同一个种子、同一份初始值），之后的预热（数据、留出集、停止规则）与基座上的完全相同。续跑时先同样合并起点，再
load_trained(latest, merge=False)。检查点：latest/model 只有预热的 LoRA（相对"base + C3"，extra 里记起点 manifest 的 sha256，只为续跑）；
final/model = src/adapters/checkpoint.write_rank_concat(C3, 预热)：秩 32 + 32 = 64 的一个 LoRA（alpha / r 不变）加预热之后的网格 token
行，对原 base 独立可读（eval_e3b、run_e1 --checkpoint、E5 的起点）；写完用 check_rank_concat 逐位核对两半。起点检查点的 manifest sha256
进运行身份：续跑时起点换了（sha256 不同）就拒绝。
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
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
import torch.nn.functional as F
import yaml

from src.accounting import Compute, merge, write_compute_json
from src.adapters.checkpoint import (
    EXTRA,
    MANIFEST,
    TOKENS_JSON,
    CheckpointError,
    check_rank_concat,
    file_sha256,
    read_manifest,
    write_rank_concat,
)
from src.adapters.training import restricted_targets
from src.e3 import data as D
from src.e3 import warmup as W
from src.e3.config import C1PRIME, C3_WARMUP, C3_WARMUP_PARENT, MIXED, STREAMS, E3Config, VARIANT_OF
from src.e3.targets import NoSupervision
from src.e3.tokens import GRID_TASKS, GridTokens, add_vocab, attach_vocab, family_stop_id
from src.runlog import write_run_metadata

REPO_ROOT = Path(__file__).resolve().parents[2]
GRID_LIKE: tuple[str, ...] = ("grid", "warmup")  # 有网格 token、全部样本都用受限交叉熵的监督
WITH_GRID_TOKENS: tuple[str, ...] = GRID_LIKE + (MIXED,)  # 加网格 token 的监督（E3b+C3 的文本微批用全词表损失）
PARENT_MARKER = "c3_warmup_parent_manifest_sha256"  # C3+预热的 latest/model 的 extra：起点 manifest 的 sha256（uint8[32]）
CONFIG = "config.yaml"
TRAIN_LOG = "train_log.jsonl"
WARMUP_EVAL = "warmup_eval.jsonl"
SUMMARY = "train_summary.json"
ACCOUNTING = "accounting.json"
COMPUTE = "compute.json"
README = "README.md"
TOKENS_INIT = "tokens_init.json"
LATEST = "latest"
FINAL = "final"
MODEL_SUB = "model"
RESUME_STATE = "resume_state.pt"


class StopRequested(Exception):
    """--stop-after-steps 到了：检查点已写，下次 resume 接着跑。"""


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
    supervision: str
    seed: int
    frozen_llm: bool = False  # C1′（DEV-0035 §1）：变体 c1prime
    parent_checkpoint: str | None = None  # C3+预热（DEV-0042 §2）：起点 C3 的 ckpt/final/model；变体 c3_warmup（路径不进运行身份）

    @property
    def variant(self) -> str:
        if self.frozen_llm:
            if self.supervision != "grid" or self.parent_checkpoint is not None:
                raise ValueError("C1′（冻结 LLM）只配网格监督、没有起点")
            return C1PRIME
        if self.parent_checkpoint is not None:
            if self.supervision != "warmup":
                raise ValueError(f"起点检查点（{C3_WARMUP}）只配预热监督，收到 {self.supervision!r}")
            return C3_WARMUP
        return VARIANT_OF[self.supervision]


def leaf_dir(out: str | Path, variant: str, model_key: str, seed: int) -> Path:
    return Path(out) / variant / model_key / f"seed{int(seed)}"


# ---------------------------------------------------------------------------
# C3+预热的起点（模块说明）
# ---------------------------------------------------------------------------


class ParentError(ValueError):
    """C3+预热的起点检查点不合规：构造方式不同、没有 LoRA、带新增 token、不是 C3、种子不符、manifest sha256 变了。"""


@dataclass(frozen=True)
class ParentRef:
    checkpoint: Path
    manifest_sha256: str
    manifest: Mapping[str, Any]
    leaf_config: str
    variant: str
    seed: int
    leaf_identity_sha256: str

    def identity(self) -> dict[str, Any]:
        """进运行身份的部分（不含路径：路径与机器有关，只进运行记录与 train_summary）。"""
        m = self.manifest
        return {"variant": self.variant, "seed": self.seed, "manifest_sha256": self.manifest_sha256, "lora": m.get("lora"),
                "build_spec": m.get("build_spec"), "dtype": m.get("dtype"), "image_policy": m.get("image_policy"),
                "base_fingerprint": m.get("base_fingerprint"), "leaf_identity_sha256": self.leaf_identity_sha256}

    def record(self) -> dict[str, Any]:
        return {**self.identity(), "checkpoint": str(self.checkpoint), "leaf_config": self.leaf_config}


def checkpoint_leaf(ckpt: str | Path) -> Path | None:
    """检查点在 <叶>/ckpt/final/model 下（run_training 的布局）时返回 <叶>，否则 None。"""
    p = Path(ckpt).resolve()
    if p.name != MODEL_SUB or p.parent.name != FINAL or p.parent.parent.name != "ckpt":
        return None
    return p.parents[2]


def resolve_parent(checkpoint: str | Path, *, model_key: str, tiny: bool, seed: int) -> ParentRef:
    """C3+预热的起点核对（模块说明）：manifest 与文件逐个核对；构造方式与本次相同；有 LoRA；没有新增 token 行、查询与 extra；
    叶目录 config.yaml 的 variant 是 c3、seed 等于本次的种子（找不到叶目录的 config.yaml 也拒绝：核对不了）。"""
    d = Path(checkpoint)
    try:
        man = read_manifest(d)
    except (CheckpointError, OSError, ValueError) as e:
        raise ParentError(f"起点检查点 {d} 读不了：{e}") from None
    bs = man.get("build_spec") or {}
    if bs.get("key") != model_key or bool(bs.get("tiny")) != bool(tiny):
        raise ParentError(f"起点检查点的构造方式（key {bs.get('key')!r}，tiny {bs.get('tiny')}）与本次（{model_key!r}，tiny {tiny}）不符")
    if man.get("lora") is None:
        raise ParentError(f"起点检查点 {d} 没有 LoRA（C1′ 之类不能作起点）")
    table = json.loads((d / TOKENS_JSON).read_text(encoding="utf-8"))
    if table.get("grid_tokens") or table.get("row_ids") or (table.get("query") or {}).get("n") is not None or man.get("extra_keys"):
        raise ParentError(f"起点检查点 {d} 带新增 token 行、查询占位符或 extra：{C3_WARMUP} 的起点必须是 C3（只有 LoRA）")
    leaf = checkpoint_leaf(d)
    if leaf is None or not (leaf / CONFIG).is_file():
        raise ParentError(f"起点检查点 {d} 不在 <训练叶目录>/ckpt/final/model 下，或叶目录没有 {CONFIG}：核对不了它的变体与种子")
    doc = yaml.safe_load((leaf / CONFIG).read_text(encoding="utf-8")) or {}
    if doc.get("variant") != C3_WARMUP_PARENT:
        raise ParentError(f"起点检查点来自变体 {doc.get('variant')!r}（{leaf / CONFIG}），{C3_WARMUP} 的起点必须是 {C3_WARMUP_PARENT!r}")
    s = doc.get("seed")
    if isinstance(s, bool) or not isinstance(s, int) or s != int(seed):
        raise ParentError(f"起点检查点来自种子 {s!r}（{leaf / CONFIG}），本次是 {seed}：{C3_WARMUP} 的种子 i 用 C3 的种子 i")
    return ParentRef(checkpoint=d, manifest_sha256=file_sha256(d / MANIFEST), manifest=man, leaf_config=str(leaf / CONFIG),
                     variant=str(doc["variant"]), seed=int(s), leaf_identity_sha256=identity_sha256(doc))


def merge_parent(ad: Any, parent: ParentRef) -> dict[str, Any]:
    """起点 C3 的 LoRA 合并进适配器的 base（src/e5/stack.merge_start_lora：与 E5 合并起点同一串调用，不改适配器的 LoRA 状态）。
    合并时读到的 manifest sha256 必须与 resolve_parent 时相同。"""
    from src.e5.stack import StartError, merge_start_lora  # 延迟 import：只有 C3+预热用到

    try:
        info = merge_start_lora(ad, parent.checkpoint)
    except StartError as e:
        raise ParentError(f"合并起点 LoRA 失败：{e}") from None
    if info["manifest_sha256"] != parent.manifest_sha256:
        raise ParentError(f"起点检查点的 manifest sha256 在核对之后变了（{parent.manifest_sha256[:16]}… → {info['manifest_sha256'][:16]}…）")
    return info


def parent_marker(parent: ParentRef) -> dict[str, torch.Tensor]:
    return {PARENT_MARKER: torch.tensor(list(bytes.fromhex(parent.manifest_sha256)), dtype=torch.uint8)}


def read_parent_marker(model_dir: str | Path) -> str | None:
    from safetensors.torch import load_file

    p = Path(model_dir) / EXTRA
    if not p.is_file():
        return None
    t = load_file(str(p)).get(PARENT_MARKER)
    return None if t is None else bytes(t.to(torch.uint8).tolist()).hex()


def _sync(device: Any) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(torch.device(device))


def _identity(cfg: Mapping[str, Any]) -> dict[str, Any]:
    return yaml.safe_load(yaml.safe_dump(dict(cfg), sort_keys=False, allow_unicode=True))


def identity_sha256(cfg: Mapping[str, Any]) -> str:
    return hashlib.sha256(yaml.safe_dump(_identity(cfg), sort_keys=True, allow_unicode=True).encode("utf-8")).hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _jsonl(rec: Mapping[str, Any]) -> str:
    return json.dumps(rec, ensure_ascii=False, sort_keys=False) + "\n"


def lr_factor(step: int, *, warmup: int, total: int, min_ratio: float) -> float:
    """第 step 步（0 起）用的 lr 系数：warmup 内线性 (step + 1) / warmup；之后余弦从 1 衰减到 min_ratio（最后一步之后到 min_ratio）。"""
    if warmup > 0 and step < warmup:
        return (step + 1) / warmup
    span = max(1, total - warmup)
    prog = min(1.0, max(0.0, (step - warmup) / span))
    return float(min_ratio) + (1.0 - float(min_ratio)) * 0.5 * (1.0 + math.cos(math.pi * prog))


# ---------------------------------------------------------------------------
# token 行的优化（两种模式，见模块说明）
# ---------------------------------------------------------------------------


class TokenRows:
    """trainable_token_rows 打开的行在优化器里的代理。"""

    def __init__(self, row_params: Sequence[torch.nn.Parameter], row_ids: Sequence[int], mode: str) -> None:
        self.params = list(row_params)
        self.mode = mode
        self.ids = sorted(int(i) for i in row_ids)
        dev = self.params[0].device
        self.idx = torch.tensor(self.ids, dtype=torch.long, device=dev)
        if mode == "fp32_master_rows":
            self.masters = [torch.nn.Parameter(p.detach().index_select(0, self.idx).to(torch.float32).clone())
                            for p in self.params]
        elif mode == "dense_param":
            self.masters = []
        else:
            raise ValueError(f"未知的 token 行模式 {mode!r}")

    def optim_params(self) -> list[torch.nn.Parameter]:
        return self.masters if self.mode == "fp32_master_rows" else self.params

    def after_backward(self) -> None:
        """主副本模式：从稠密梯度里取这几行，清掉稠密梯度（它只是中转）。"""
        if self.mode != "fp32_master_rows":
            return
        for p, m in zip(self.params, self.masters):
            g = p.grad
            m.grad = (torch.zeros_like(m) if g is None else g.detach().index_select(0, self.idx).to(torch.float32))
            p.grad = None

    def after_step(self) -> None:
        if self.mode != "fp32_master_rows":
            return
        with torch.no_grad():
            for p, m in zip(self.params, self.masters):
                p.index_copy_(0, self.idx, m.detach().to(p.dtype))

    def zero_dense_grads(self) -> None:
        for p in self.params:
            p.grad = None

    def state_dict(self) -> dict[str, Any]:
        return {"mode": self.mode, "ids": list(self.ids), "masters": [m.detach().cpu().clone() for m in self.masters]}

    def load_state_dict(self, st: Mapping[str, Any]) -> None:
        if st["mode"] != self.mode or list(st["ids"]) != self.ids or len(st["masters"]) != len(self.masters):
            raise ValueError("续跑状态里的 token 行与本次不符（模式、行或个数）")
        with torch.no_grad():
            for m, v in zip(self.masters, st["masters"]):
                m.copy_(v.to(m.device, m.dtype))
        self.after_step()


def frozen_llm_token_rows(ad: Any) -> list[torch.nn.Parameter]:
    """C1′（模块说明）：LLM 的全部参数设为不求梯度，只让输入嵌入（lm_head 不共享时另有 lm_head）的权重求梯度；返回这些权重
    （共享时只有一个）。不挂梯度掩码：交给 TokenRows 的 fp32 主副本模式，它只取、只写网格 token 那几行。
    （适配器的 trainable_token_rows 要求先注入 LoRA，C1′ 没有 LoRA，所以不用它；嵌入与 lm_head 取 HF 的 get_input_embeddings 与
    适配器的 lm_head_weight，与 trainable_token_rows 取的是同两个张量。）"""
    for p in ad.model.parameters():
        p.requires_grad_(False)
    emb = ad.model.get_input_embeddings().weight
    head = ad.lm_head_weight()
    if (emb is head) != bool(ad.tied_word_embeddings):
        raise RuntimeError(f"tied_word_embeddings = {ad.tied_word_embeddings}，但嵌入与 lm_head {'是' if emb is head else '不是'}同一个张量")
    params = [emb] if emb is head else [emb, head]
    for w in params:
        w.requires_grad_(True)
    return params


class FrozenRowsProbe:
    """C1′ 的冻结行核对：开始时记下嵌入（与不共享的 lm_head）里均匀抽的至多 n 行（网格 token 的行除外），存检查点之前核对逐位不变
    （save_trained 的冻结行核对只对 trainable_token_rows 打开的行生效，C1′ 不经过它）。"""

    def __init__(self, params: Sequence[torch.Tensor], trainable_ids: Sequence[int], n: int = 512) -> None:
        skip = {int(i) for i in trainable_ids}
        self.items = []
        for w in params:
            rows = sorted({int(round(x)) for x in np.linspace(0, int(w.shape[0]) - 1, n)} - skip)
            idx = torch.tensor(rows, dtype=torch.long, device=w.device)
            self.items.append((w, idx, w.detach().index_select(0, idx).clone()))

    def check(self) -> None:
        for w, idx, rows in self.items:
            if not torch.equal(w.detach().index_select(0, idx), rows):
                raise RuntimeError("C1′：网格 token 以外的嵌入 / lm_head 行变了（冻结 LLM 只应改网格 token 的行）")


# ---------------------------------------------------------------------------
# 训练器：一个适配器上的一步优化（网格与 C3 共用）
# ---------------------------------------------------------------------------


class Trainer:
    """持有适配器、网格 token 布局、优化器与调度器；optimize(examples) 做一步（含微批与梯度累积）。"""

    def __init__(self, adapter: Any, cfg: E3Config, run_cfg: Mapping[str, Any], *, supervision: str, seed: int,
                 parent: ParentRef | None = None) -> None:
        self.ad = adapter
        self.cfg = cfg
        self.rc = dict(run_cfg)
        self.supervision = supervision
        self.seed = int(seed)
        self.tokens: GridTokens | None = None
        self.rows: TokenRows | None = None
        self.opt: torch.optim.Optimizer | None = None
        self.sched: torch.optim.lr_scheduler.LambdaLR | None = None
        self.stop_id = family_stop_id(adapter)
        self.audit: dict[str, Any] = {}
        self.grid_like = supervision in GRID_LIKE
        self.mixed = supervision == MIXED  # E3b+C3（模块说明）：按流切微批、两种损失各自归一后相加
        self.with_grid_tokens = supervision in WITH_GRID_TOKENS
        self.frozen_llm = bool(self.rc.get("frozen_llm", False))  # C1′（模块说明）
        if self.frozen_llm and supervision != "grid":
            raise ValueError("C1′（冻结 LLM）只配网格监督")
        if self.mixed and "mixed" not in self.rc:
            raise ValueError("E3b+C3 的 run_cfg 没有 mixed 一节（resolve_run(supervision = grid+text)）")
        self.parent = parent  # C3+预热（模块说明）：搭建之前先把起点的 LoRA 合并进 base
        if parent is not None and supervision != "warmup":
            raise ValueError("起点检查点只配预热监督")
        self.parent_merge: dict[str, Any] | None = None
        self.frozen_probe: FrozenRowsProbe | None = None

    # ----- 搭建 -----
    def _merge_parent(self) -> None:
        if self.parent is not None:  # 在 torch.manual_seed 之前：peft 建起点的 LoRA 层时会用随机数
            self.parent_merge = merge_parent(self.ad, self.parent)

    def setup_fresh(self) -> None:
        self._merge_parent()
        torch.manual_seed(self.seed)
        if self.with_grid_tokens:
            self.tokens = add_vocab(self.ad, self.cfg.vocab)
        if not self.frozen_llm:
            lo = self.rc["lora"]
            self.ad.inject_lora(r=int(lo["r"]), lora_alpha=float(lo["alpha"]), lora_dropout=float(lo["dropout"]))
        self._finish_setup()

    def setup_resume(self, model_dir: Path) -> None:
        if self.parent is not None:  # C3+预热：latest/model 是相对"base + C3"的预热 LoRA，先合并同一个起点
            mark = read_parent_marker(model_dir)
            if mark != self.parent.manifest_sha256:
                raise ParentError(f"续跑：{model_dir} 记下的起点 manifest sha256 {str(mark)[:16]}… 与本次的起点 "
                                  f"{self.parent.manifest_sha256[:16]}… 不符")
        self._merge_parent()
        self.ad.load_trained(model_dir, merge=False)
        want = "none" if self.frozen_llm else "injected"
        if self.ad.lora_state != want:
            raise RuntimeError(f"续跑：检查点载入后 LoRA 状态是 {self.ad.lora_state}，应为 {want}")
        if self.with_grid_tokens:
            self.tokens = attach_vocab(self.ad, self.cfg.vocab)
        elif self.ad.grid_token_ids:
            raise RuntimeError("续跑：C3 的检查点里不该有 grid token")
        self._finish_setup()

    def _finish_setup(self) -> None:
        ad = self.ad
        lora = [p for n, p in ad.model.named_parameters() if "lora_" in n]
        if self.frozen_llm:  # C1′：没有 LoRA，只有网格 token 的行（模块说明）
            if lora or ad.lora_state != "none":
                raise RuntimeError(f"C1′ 不该有 LoRA（LoRA 状态 {ad.lora_state}，{len(lora)} 个 LoRA 参数）")
            if self.tokens is None:
                raise RuntimeError("C1′ 要有网格 token")
            mode = self.rc["optim"]["token_rows"]["mode"]
            if mode != "fp32_master_rows":
                raise ValueError(f"C1′ 的 token 行必须是 fp32_master_rows，收到 {mode!r}")
            self.rows = TokenRows(frozen_llm_token_rows(ad), self.tokens.grid_ids, mode)
            self.frozen_probe = FrozenRowsProbe(self.rows.params, self.rows.ids)
        else:
            if not lora or not all(p.requires_grad for p in lora):
                raise RuntimeError("LoRA 参数不存在或没有设为求梯度")
            if self.tokens is not None:
                row_params = ad.trainable_token_rows(list(self.cfg.vocab.names))
                self.rows = TokenRows(row_params, ad.trainable_token_row_ids, self.rc["optim"]["token_rows"]["mode"])
        ad.set_gradient_checkpointing(bool(self.rc["gradient_checkpointing"]))
        o = self.rc["optim"]
        groups = []
        if not self.frozen_llm:
            groups.append({"params": lora, "lr": float(o["lora"]["lr"]), "weight_decay": float(o["lora"]["weight_decay"]),
                           "name": "lora"})
        if self.rows is not None:
            groups.append({"params": self.rows.optim_params(), "lr": float(o["token_rows"]["lr"]),
                           "weight_decay": 0.0, "name": "token_rows"})
        if o["optimizer"] != "adamw":
            raise ValueError(f"只实现了 adamw，配置是 {o['optimizer']!r}")
        self.opt = torch.optim.AdamW(groups, betas=tuple(float(b) for b in o["betas"]), eps=float(o["eps"]))
        warm, total, mn = int(self.rc["warmup_steps"]), int(self.rc["steps"]), float(o["min_lr_ratio"])
        self.sched = torch.optim.lr_scheduler.LambdaLR(
            self.opt, lambda s: lr_factor(s, warmup=warm, total=total, min_ratio=mn))
        self.audit = self.setup_audit(lora)

    def setup_audit(self, lora: Sequence[torch.nn.Parameter]) -> dict[str, Any]:
        """运行期审计（进运行记录）：求梯度的参数只能是 LoRA 与 token 行所在的矩阵；参数的设备与 dtype 集合。"""
        ad = self.ad
        rows = {id(p) for p in (self.rows.params if self.rows else [])}
        lora_ids = {id(p) for p in lora}
        bad = [n for n, p in ad.model.named_parameters() if p.requires_grad and id(p) not in rows | lora_ids]
        if bad:
            raise RuntimeError(f"LoRA 与 token 行以外的参数求梯度：{bad[:5]}")
        # 入口断言：LoRA 的参数必须是 fp32（bf16 的 base 上由 peft 的 autocast_adapter_dtype 升上来）；peft 的行为一变就报错，
        # 不静默地用 bf16 的 LoRA 训练（更新小于半个 ulp 时被舍掉）
        not32 = sorted({str(p.dtype) for p in lora if p.dtype != torch.float32})
        if not32:
            raise RuntimeError(f"LoRA 参数的 dtype 是 {not32}，应为 torch.float32")
        out = {"n_lora_params": int(sum(p.numel() for p in lora)),
               "lora_dtypes": sorted({str(p.dtype) for p in lora}), "lora_devices": sorted({str(p.device) for p in lora}),
               "token_row_ids": None if self.rows is None else list(self.rows.ids),
               "token_row_mode": None if self.rows is None else self.rows.mode,
               "token_row_tensors": None if self.rows is None else len(self.rows.params),
               "tied_word_embeddings": bool(ad.tied_word_embeddings), "lora_targets": len(ad.lora_target_modules())}
        if self.frozen_llm:  # C1′：求梯度的只有嵌入（与不共享的 lm_head），优化器里只有网格 token 行的主副本
            out.update(frozen_llm=True, n_requires_grad_tensors=sum(1 for p in ad.model.parameters() if p.requires_grad),
                       n_optim_params=int(sum(p.numel() for p in self.rows.optim_params())))
        return out

    def check_frozen(self) -> None:
        """存检查点之前（C1′）：冻结行的抽样逐位不变。其余监督什么都不做（save_trained 自己核对）。"""
        if self.frozen_probe is not None:
            self.frozen_probe.check()

    def latest_extra(self) -> dict[str, torch.Tensor] | None:
        """续跑检查点（latest/model）的 extra：C3+预热记起点 manifest 的 sha256（模块说明），其余没有。"""
        return None if self.parent is None else parent_marker(self.parent)

    # ----- 一步 -----
    def lr_now(self) -> dict[str, float]:
        return {g.get("name", str(k)): float(g["lr"]) for k, g in enumerate(self.opt.param_groups)}

    def _loss(self, out: Any, group: Sequence[D.Example]) -> tuple[torch.Tensor, dict[str, list[float]], list[int]]:
        """一个微批的损失和（未归一化）、各任务的 [损失和, 监督 token 数]、网格 token 的 [对的个数, 计入的个数]。
        grid、warmup 用受限交叉熵，text 用 chunked_lm_loss（E3b+C3 按微批所属的流调 _task_loss）。"""
        return self._task_loss(out, group, grid=self.grid_like)

    def _task_loss(self, out: Any, group: Sequence[D.Example], *, grid: bool) -> tuple[torch.Tensor, dict[str, list[float]], list[int]]:
        ad = self.ad
        th, tid, trow = out.target_hidden, out.target_ids, out.target_rows
        w = torch.tensor([x for e in group for x in e.weights], dtype=torch.float32, device=th.device)
        total = th.new_zeros((), dtype=torch.float32)
        per: dict[str, list[float]] = {}
        acc = [0, 0]
        for task in GRID_TASKS:
            rows = [b for b, e in enumerate(group) if e.plan.task == task]
            if not rows:
                continue
            sel = torch.isin(trow, torch.tensor(rows, dtype=torch.long, device=trow.device)).nonzero().flatten()
            h, t, ws = th.index_select(0, sel), tid.index_select(0, sel), w.index_select(0, sel)
            if grid:
                allowed = self.tokens.allowed_ids(task, self.rc["allowed_scope"])
                lg = ad.restricted_logits(h, allowed)
                tt = restricted_targets(t, allowed)
                ce = F.cross_entropy(lg, tt, reduction="none")
                ls = (ce * ws).sum()
                with torch.no_grad():
                    hit = (lg.argmax(-1) == tt) & (ws > 0)
                    acc[0] += int(hit.sum())
                    acc[1] += int((ws > 0).sum())
            else:
                ls = ad.chunked_lm_loss(h, t, chunk_size=int(self.cfg.doc["c3"]["chunk_size"]), reduction="sum")
            total = total + ls
            per[task] = [float(ls.detach()), float(ws.sum())]
        return total, per, acc

    def build_inputs(self, e: D.Example) -> Any:
        return self.ad.build_inputs(e.images, e.instruction, e.order)

    def optimize(self, examples: Sequence[D.Example], *, device: Any = None) -> tuple[dict[str, Any], dict[str, Any], list[Compute]]:
        """一步优化：examples 按 micro_batch_size 切成微批，各自前向、反向（损失 / 整步监督 token 总数），然后一次 step。
        E3b+C3（模块说明）：按流分开切微批（每个微批只含一种监督，流的先后按 mixed.order），每个微批的损失除以这一步里它那种监督的
        监督 token 总数、再乘损失权重（1 : 1），即一步的损失 = L_text + L_grid。返回 (确定性的记录, 计时, 每条样本的 Compute)。"""
        if not examples:
            raise ValueError("这一步没有样本")
        dev = device if device is not None else self.ad.device
        n_sup = sum(e.n_supervised for e in examples)
        if n_sup <= 0:
            raise ValueError("这一步没有监督 token")
        micro = int(self.rc["micro_batch_size"])
        n_by: dict[str, int] = {}
        weights: dict[str, float] = {}
        if self.mixed:
            mcfg = self.rc["mixed"]
            by = {s: [e for e in examples if D.stream_of(e.plan) == s] for s in STREAMS}
            n_by = {s: sum(e.n_supervised for e in by[s]) for s in STREAMS}
            weights = {s: float(mcfg["loss_weights"][s]) for s in STREAMS}
            chunks: list[tuple[str | None, Sequence[D.Example]]] = [
                (s, by[s][i: i + micro]) for s in mcfg["order"] for i in range(0, len(by[s]), micro)]
            cnt_by = {s: {"n_samples": len(by[s]), "n_supervised_tokens": int(n_by[s]), "n_target_tokens": 0, "n_prompt_tokens": 0,
                          "n_visual_tokens": 0, "n_image_pixels": 0} for s in STREAMS}
        else:
            chunks = [(None, examples[i: i + micro]) for i in range(0, len(examples), micro)]
        self.opt.zero_grad(set_to_none=True)
        if self.rows is not None:
            self.rows.zero_dense_grads()
        loss_sum = 0.0
        sums = {s: 0.0 for s in STREAMS}
        per: dict[str, list[float]] = {}
        acc = [0, 0]
        counts = {"n_target_tokens": 0, "n_prompt_tokens": 0, "n_visual_tokens": 0, "n_image_pixels": 0}
        times = {"build_inputs_s": 0.0, "fwd_s": 0.0, "bwd_s": 0.0, "opt_s": 0.0}
        computes: list[Compute] = []
        for key, group in chunks:
            t0 = time.perf_counter()
            inputs = [self.build_inputs(e) for e in group]
            _sync(dev)
            t1 = time.perf_counter()
            out = self.ad.training_forward(inputs, [e.target_ids for e in group], enable_thinking=bool(self.rc["enable_thinking"]))
            ls, p, a = self._loss(out, group) if key is None else self._task_loss(out, group, grid=key == "grid")
            _sync(dev)
            t2 = time.perf_counter()
            if key is None:
                (ls / float(n_sup)).backward()
            else:  # 这种监督没有监督 token 时（网格的权重全 0）ls 恰为 0，分母取 1 只为不除以 0
                (ls / float(max(n_by[key], 1)) * weights[key]).backward()
            _sync(dev)
            t3 = time.perf_counter()
            times["build_inputs_s"] += t1 - t0
            times["fwd_s"] += t2 - t1
            times["bwd_s"] += t3 - t2
            loss_sum += float(ls.detach())
            if key is not None:
                sums[key] += float(ls.detach())
            for k, v in p.items():
                acc_k = per.setdefault(k if key is None else f"{key}/{k}", [0.0, 0.0])
                acc_k[0] += v[0]
                acc_k[1] += v[1]
            acc[0] += a[0]
            acc[1] += a[1]
            share = (t3 - t1) / len(group)
            for b, (e, inp) in enumerate(zip(group, inputs)):
                nv = sum(len(i.span) for i in inp.images)
                npx = sum(int(i.n_pixels or 0) for i in inp.images)
                counts["n_target_tokens"] += out.target_lengths[b]
                counts["n_prompt_tokens"] += out.prompt_lengths[b]
                counts["n_visual_tokens"] += nv
                counts["n_image_pixels"] += npx
                if key is not None:
                    c = cnt_by[key]
                    c["n_target_tokens"] += int(out.target_lengths[b])
                    c["n_prompt_tokens"] += int(out.prompt_lengths[b])
                    c["n_visual_tokens"] += int(nv)
                    c["n_image_pixels"] += int(npx)
                computes.append(Compute.call(generated_tokens=int(out.target_lengths[b]), prompt_tokens=int(out.prompt_lengths[b]),
                                             visual_tokens=int(nv), image_pixels=int(npx), call_time_s=share, new_round=True))
            del out, ls
        t4 = time.perf_counter()
        if self.rows is not None:
            self.rows.after_backward()
        params = [p for g in self.opt.param_groups for p in g["params"]]
        mx = self.rc["optim"]["max_grad_norm"]
        gn = torch.nn.utils.clip_grad_norm_(params, float("inf") if mx is None else float(mx))
        lr = self.lr_now()
        self.opt.step()
        self.sched.step()
        if self.rows is not None:
            self.rows.after_step()
        self.opt.zero_grad(set_to_none=True)
        _sync(dev)
        times["opt_s"] = time.perf_counter() - t4
        kinds: dict[str, int] = {}
        for e in examples:
            kinds[e.plan.kind] = kinds.get(e.plan.kind, 0) + 1
        n_empty = sum(1 for e in examples if e.info.get("n_fg_cells") == 0)  # 分割目标全 0（物体小于半格时会这样）
        by_task = {k: (v[0] / v[1] if v[1] else None) for k, v in sorted(per.items())}
        if self.mixed:  # 两种监督各自的逐 token 平均（loss_by_task 的 text、grid 两项），loss = 按权重相加
            loss_by = {s: (sums[s] / n_by[s] if n_by[s] else None) for s in STREAMS}
            loss = sum(weights[s] * loss_by[s] for s in self.rc["mixed"]["order"] if loss_by[s] is not None)
            by_task = dict(sorted({**by_task, **loss_by}.items()))
        else:
            loss = loss_sum / float(n_sup)
        rec = {"lr": lr, "loss": loss,
               "loss_by_task": by_task,
               "n_supervised_by_task": {k: int(v[1]) for k, v in sorted(per.items())},
               "n_samples": len(examples), "n_by_kind": dict(sorted(kinds.items())), "n_empty_seg_grid": n_empty,
               "n_supervised_tokens": int(n_sup),
               **{k: int(v) for k, v in counts.items()}, "grad_norm": float(gn),
               "grid_token_acc": (acc[0] / acc[1]) if acc[1] else None}
        if self.mixed:
            rec["by_supervision"] = cnt_by
        return rec, times, computes

    # ----- 续跑状态 -----
    def state_dict(self) -> dict[str, Any]:
        st = {"optimizer": self.opt.state_dict(), "scheduler": self.sched.state_dict(),
              "token_rows": None if self.rows is None else self.rows.state_dict(),
              "rng_cpu": torch.get_rng_state()}
        if torch.cuda.is_available():
            st["rng_cuda"] = torch.cuda.get_rng_state_all()
        return st

    def load_state_dict(self, st: Mapping[str, Any]) -> None:
        self.opt.load_state_dict(st["optimizer"])
        self.sched.load_state_dict(st["scheduler"])
        if self.rows is not None:
            self.rows.load_state_dict(st["token_rows"])
        elif st.get("token_rows") is not None:
            raise ValueError("续跑状态里有 token 行，本次没有")
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


def token_identity(ad: Any, tokens: GridTokens | None, scope: str) -> dict[str, Any] | None:
    """运行身份里的 token 部分：token_table 去掉行哈希（续跑时行已训练过），加上 id 布局与各任务的允许集合。"""
    if tokens is None:
        return None
    table = {k: v for k, v in ad.token_table().items() if k != "row_tables"}
    return {"vocab": list(tokens.vocab.names), "budget": tokens.vocab.budget(), "layout": tokens.layout(),
            "allowed": {t: tokens.allowed_ids(t, scope) for t in GRID_TASKS}, "table": table}


def warmup_identity(spec: RunSpec, cfg: E3Config, run_cfg: Mapping[str, Any], trainer: Trainer,
                    source: W.WarmupSource) -> dict[str, Any]:
    """网格格式预热的运行身份（键与 E3b 的相同；数据一节换成生成器与留出集的指纹，没有训练池、去重与加噪）。"""
    held = source.heldout
    out = {
        "experiment": "E3b",
        "framework": "src/e3（DEV-0025；网格格式预热 src/e3/warmup.py，DEV-0035）",
        "model": {"key": spec.model_key, "family": spec.family, "repo": spec.repo, "revision": spec.revision,
                  "adapter_model_key": spec.adapter_model_key, "tiny": spec.tiny, "dtype": spec.model_dtype},
        "image_policy": spec.image_policy,
        "supervision": spec.supervision,
        "variant": spec.variant,
        "seed": spec.seed,
        "training": dict(run_cfg),
        "grid_size": cfg.grid_n,
        "tokens": token_identity(trainer.ad, trainer.tokens, run_cfg["allowed_scope"]),
        "stop_id": trainer.stop_id,
        "instructions": {"orders": dict(source.orders), "templates": "data.warmup（配置的 warmup 节）"},
        "c3": None,
        "noise": None,
        "config": {"path": cfg.path, "sha256": cfg.sha256, "version": cfg.doc["version"]},
        "data": {"source": "src/e3/warmup.WarmupSource（纯文字，不给图像；与任何视觉数据无关）", "images": 0,
                 "warmup": source.spec.identity(),
                 "heldout": {"n": len(held), "sha256": W.items_digest(held),
                             "by_task": {t: sum(1 for it in held if it.task == t) for t in GRID_TASKS}}},
    }
    if trainer.parent is not None:  # C3+预热：起点（不含路径）；基座上的预热不加这个键（运行身份逐字节不变）
        out["parent"] = trainer.parent.identity()
    return out


def run_identity(spec: RunSpec, cfg: E3Config, run_cfg: Mapping[str, Any], trainer: Trainer,
                 pools: Mapping[str, Any], dedupe: Mapping[str, Any], builder: Any) -> dict[str, Any]:
    if spec.supervision == "warmup":
        return warmup_identity(spec, cfg, run_cfg, trainer, builder)
    d = cfg.doc
    instr = {t: {k: v for k, v in d["tasks"][t].items() if k.endswith(("template", "instruction")) or k == "order"}
             for t in pools}
    c3 = None
    if spec.supervision in ("text", MIXED):
        c3 = {k: v for k, v in d["c3"].items() if k != "basis"}
        c3["polygon_template"] = builder.polygon_template
        c3["polygon_template_source"] = builder.polygon_source
        c3["box_template"] = builder.box_template
        c3["box_template_source"] = builder.box_source
        c3["box_format"] = "adapter.format_boxes（本家族原生格式，label = 指代句；DEV-0030 §3）"
    return {
        "experiment": "E3b",
        "framework": "src/e3（DEV-0025）",
        "model": {"key": spec.model_key, "family": spec.family, "repo": spec.repo, "revision": spec.revision,
                  "adapter_model_key": spec.adapter_model_key, "tiny": spec.tiny, "dtype": spec.model_dtype},
        "image_policy": spec.image_policy,
        "supervision": spec.supervision,
        "variant": spec.variant,
        "seed": spec.seed,
        "training": dict(run_cfg),
        "grid_size": cfg.grid_n,
        "tokens": token_identity(trainer.ad, trainer.tokens, run_cfg["allowed_scope"]),
        "stop_id": trainer.stop_id,
        "instructions": instr,
        "c3": c3,
        "noise": {"correction": {k: v for k, v in d["correction"].items() if k != "basis"},
                  "depth_noise": {k: v for k, v in d["depth_noise"].items() if k != "basis"}}
        if spec.supervision in ("grid", MIXED) else None,
        "config": {"path": cfg.path, "sha256": cfg.sha256, "version": d["version"]},
        "data": {t: {"pool": _scalars(pools[t].stats), "n": len(pools[t]),
                     "ids_sha256": D.ids_digest([s.id for s in pools[t].records]), "dedupe": dict(dedupe[t])}
                 for t in pools},
    }


def _read_log(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    raw = path.read_bytes()
    keep = raw.rfind(b"\n") + 1
    return [json.loads(l) for l in raw[:keep].decode("utf-8").splitlines() if l.strip()]


def _append(path: Path, lines: Sequence[str]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write("".join(lines))
        f.flush()
        os.fsync(f.fileno())


def _dir_bytes(d: Path) -> int:
    return sum(p.stat().st_size for p in d.rglob("*") if p.is_file()) if d.exists() else 0


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
    sums = {k: int(sum(r[k] for r in log)) for k in ("n_samples", "n_supervised_tokens", "n_target_tokens", "n_prompt_tokens",
                                                     "n_visual_tokens", "n_image_pixels")}
    wall = sum(float(r.get("wall_s") or 0.0) for r in runs)
    doc = {"n_runs": len(runs), "n_steps_timed": n_steps, "n_steps_logged": len(log), **sums, "phases_s": tot,
           "wall_s": wall, "gpu_hours": wall / 3600.0, "peak_mem_bytes": max(peaks) if peaks else None,
           "device": runs[-1].get("device") if runs else None, "gpu": runs[-1].get("gpu") if runs else None,
           "note": "phases_s 按步相加：data_s = 构造样本（读图、解码 GT、加噪、渲染叠加图）；build_inputs_s = 处理器；fwd_s、bwd_s = 前向"
                   "（含损失）与反向（CUDA 前后同步）；opt_s = 取主副本梯度、裁剪、step、写回；step_s = 整步。计入 n_steps_timed 的步可能"
                   "多于 n_steps_logged（被中断的步的计时也记了，但没有写进 train_log）。compute.json 的 generated_tokens 是教师强制的"
                   "目标 token 数"}
    if any("by_supervision" in r for r in log):  # E3b+C3：两种监督各自的样本数与 token 数（每步的 by_supervision 相加）
        doc["by_supervision"] = {s: {k: int(sum(r["by_supervision"][s][k] for r in log)) for k in log[0]["by_supervision"][s]}
                                 for s in STREAMS}
    if any("warmup_evals" in r for r in runs):  # 网格格式预热：留出集的受限解码（不在 phases_s 里，含在 wall_s 里）
        evs = [e for r in runs for e in r.get("warmup_evals") or []]
        doc["warmup_eval_s"] = sum(float(e["wall_s"]) for e in evs)
        doc["warmup_eval_n"] = len(evs)
    return doc, comp


def readme_text(spec: RunSpec, ident: Mapping[str, Any], summary: Mapping[str, Any], log: Sequence[Mapping[str, Any]]) -> str:
    tr = ident["training"]
    first, last = (log[0], log[-1]) if log else ({}, {})
    sup = {"grid": "GT 网格 + 修正对", "text": "C3：稀疏文本", "warmup": "网格格式预热：纯文字给坐标 → 网格，不给图像",
           MIXED: "E3b+C3：C3 的文本样本流 + E3b 的网格样本流（GT 网格 + 修正对）"}[spec.supervision]
    if tr.get("frozen_llm"):  # C1′
        sup += "；C1′：冻结 LLM，只训练网格 token 的行"
        opt = (f"不注入 LoRA，AdamW 只有 token 行一组：lr {tr['optim']['token_rows']['lr']}（{tr['optim']['token_rows']['mode']}），"
               f"warmup {tr['warmup_steps']} 步 + 余弦")
        ck = "只有 token 行、manifest"
    else:
        opt = (f"LoRA r {tr['lora']['r']} / alpha {tr['lora']['alpha']} / dropout {tr['lora']['dropout']}，"
               f"AdamW lr {tr['optim']['lora']['lr']}（token 行 {tr['optim']['token_rows']['lr']}，{tr['optim']['token_rows']['mode']}），"
               f"warmup {tr['warmup_steps']} 步 + 余弦")
        ck = "只有 LoRA、token 行、manifest"
    if spec.variant == C3_WARMUP:
        sup += "；C3+预热：先把起点 C3 的 LoRA 合并进 base，再在其上预热"
        lo = summary.get("lora") or {}
        ck = f"C3 与预热的 LoRA 按秩拼接成一个（r {lo.get('r')}、alpha {lo.get('lora_alpha')}）、预热之后的 token 行、manifest"
    if spec.supervision == MIXED:
        m = tr["mixed"]
        steps_line = (f"- 训练：{tr['steps']} 步 × {tr['batch_size']} 条（文本 {m['per_step']['text']} + 网格 {m['per_step']['grid']}；"
                      f"微批 {tr['micro_batch_size']} × 累积 {tr['grad_accum_steps']}，每个微批只含一种监督：{m['micro_batches']}，"
                      f"先后 {m['order']}；日程长度是 C3、E3b 的 {m['schedule_length_ratio']:g} 倍），")
    elif spec.supervision != "warmup":
        steps_line = f"- 训练：{tr['steps']} 步 × {tr['batch_size']} 条（微批 {tr['micro_batch_size']} × 累积 {tr['grad_accum_steps']}），"
    else:
        steps_line = (f"- 训练：至多 {tr['steps']} 步 × {tr['batch_size']} 条（微批 {tr['micro_batch_size']} × 累积 {tr['grad_accum_steps']}；"
                      f"实际 {summary.get('steps')} 步），")
    lines = [f"# E3b 训练 · {spec.variant} · {spec.model_key} · seed {spec.seed}", "",
             f"- 模型：{spec.repo}@{spec.revision}（{spec.family}；适配器 {spec.adapter_model_key}；{spec.model_dtype}）",
             f"- 监督：{spec.supervision}（{sup}）；图像口径 {spec.image_policy}",
             steps_line + opt,
             f"- 混合：{tr['mix']}；修正对比例：{tr['correction_fraction']}",
             f"- 第一步损失 {first.get('loss')}，最后一步损失 {last.get('loss')}（{len(log)} 步）",
             f"- 检查点：{summary.get('final_checkpoint')}（{summary.get('final_checkpoint_bytes')} 字节；{ck}）", ""]
    if spec.supervision == MIXED:
        m = tr["mixed"]
        ref = m["reference"]
        tot = {s: {k: int(sum(r["by_supervision"][s][k] for r in log)) for k in ("n_samples", "n_supervised_tokens")}
               for s in STREAMS} if log else {}
        lines += ["## E3b+C3 的两条样本流", "",
                  f"- 第 t 步：文本 = {ref['variants']['text']} 同一种子的样本流第 {m['per_step']['text']}t … "
                  f"{m['per_step']['text']}t + {m['per_step']['text'] - 1} 个槽位；网格 = {ref['variants']['grid']} 同一种子的样本流第 "
                  f"{m['per_step']['grid']}t … {m['per_step']['grid']}t + {m['per_step']['grid'] - 1} 个槽位（含修正对）。全程两条流各 "
                  f"{ref['samples_per_stream']} 条 = {ref['variants']['text']}、{ref['variants']['grid']} 单独训练的全部样本，顺序相同",
                  "- 两条流的槽位与样本的对应只由种子与 k 决定，所以同一步里同一张图以两种监督各出现一次（不另换数据种子）",
                  f"- 损失：每步 = {m['loss_weights']['text']:g} × 文本的逐 token 平均 + {m['loss_weights']['grid']:g} × 网格的逐 token 平均"
                  "（事先固定，D83-2）；train_log 的 loss 是这个和，loss_by_task 的 text、grid 两项是两种各自的平均",
                  f"- 学习率：warmup {tr['warmup_steps']} 步 + 余弦，按 {tr['steps']} 步排，日程长度是 {ref['variants']['text']}、"
                  f"{ref['variants']['grid']}（{ref['steps']} 步）的 {m['schedule_length_ratio']:g} 倍",
                  f"- 本次合计（train_log 的 by_supervision）：{tot}", ""]
    par = summary.get("parent")
    if par:
        lc = summary.get("lora_concat") or {}
        lines += ["## C3+预热的起点", "",
                  f"- 起点：变体 {par['variant']}、种子 {par['seed']}，manifest sha256 {par['manifest_sha256']}",
                  f"- 起点的 LoRA 在 torch.manual_seed 之前合并进 base，再加网格 token、注入预热的 LoRA；预热的数据、留出集、停止规则与基座上的相同",
                  f"- 存法：{lc.get('rule')}；拼接后核对：{lc.get('check')}", ""]
    wsum = summary.get("warmup")
    if wsum:
        r = wsum["rule"]
        lines += ["## 停止规则（网格格式预热）", "",
                  f"- 规则：每 {r['eval_every']} 步在留出集（{r['heldout_n']} 条）上受限解码，格式错误率连续 {r['consecutive']} 次 "
                  f"< {r['threshold']} 就停；上限 {r['max_steps']} 步",
                  f"- 结果：stop_rule_met = {wsum['stop_rule_met']}，停在第 {wsum['stopped_at_step']} 步（{wsum['reason']}），"
                  f"共 {wsum['n_evals']} 次读数；最后一次格式错误率 {(wsum.get('last') or {}).get('format_error_rate')}",
                  "- 逐次读数：warmup_eval.jsonl", ""]
    if ident.get("tokens"):
        lay = ident["tokens"]["layout"]
        lines += ["## token 布局", "", f"- mask：{lay['mask']}；深度 {lay['depth_first']}–{lay['depth_last']}"
                  f"（连续：{lay['depth_contiguous']}）；行分隔 {lay['row']}；停止 {lay['stop']}", ""]
    lines += ["## 文件", "", "- train_log.jsonl：每步的确定性记录；logs/：每次调用的记录与逐步计时；accounting.json、compute.json：记账；",
              "- config.yaml：运行身份；meta.json：git、主机、包版本；tokens_init.json：初始化时的 token 表（含行哈希）。", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 正式运行
# ---------------------------------------------------------------------------


def run_training(adapter: Any, cfg: E3Config, spec: RunSpec, run_cfg: Mapping[str, Any], paths: D.E3Paths,
                 out_dir: str | Path, *, resume: bool = False, argv: Sequence[str] = (), ckpt_root: str | Path | None = None,
                 stop_after_steps: int | None = None, keep_optimizer_state: bool = False,
                 log: Callable[[str], None] | None = None, _fail_after_step: int | None = None,
                 _warmup_eval: W.EvalFn | None = None) -> dict[str, Any] | None:
    """跑一次训练；返回 train_summary（stop_after_steps 到了时返回 None）。adapter 必须是刚挂好的 base（规则见模块说明）。
    _warmup_eval（只给测试）：替换留出集的读数 (adapter, tokens, 留出集) → 读数，用来核对停止规则的流程。"""
    log = log or (lambda m: print(m, file=sys.stderr, flush=True))
    rc = dict(run_cfg)
    if rc["variant"] != spec.variant or bool(rc.get("frozen_llm", False)) != spec.frozen_llm or rc["supervision"] != spec.supervision:
        raise ValueError(f"run_cfg（{rc['supervision']}/{rc['variant']}）与 RunSpec（{spec.supervision}/{spec.variant}）不符")
    warm = spec.supervision == "warmup"
    leaf = leaf_dir(out_dir, spec.variant, spec.model_key, spec.seed)
    ckpt = (Path(ckpt_root) / spec.variant / spec.model_key / f"seed{spec.seed}") if ckpt_root else leaf / "ckpt"
    latest = ckpt / LATEST

    # 数据（硬约束 7：加载器没有剔除清单就报错；这里另核对清单比较过本任务的评测集）。网格格式预热不读训练池（纯文字，src/e3/warmup）
    pools, dedupe = {}, {}
    for t in ([] if warm else cfg.tasks):
        pools[t] = D.load_pool(t, paths, cfg.task(t))
        dedupe[t] = D.check_dedupe(t, pools[t].stats.get("dedupe_file"), cfg.doc["dedupe"]["required_eval_sets"].get(t, []),
                                   real_model=not spec.tiny)
        if not pools[t].records:
            raise ValueError(f"{t} 的训练池是空的")

    exists = leaf.exists() and any(leaf.iterdir())
    if exists and not resume:
        raise FileExistsError(f"{leaf} 已存在且非空；续跑请加 --resume")
    if exists and (leaf / SUMMARY).exists():
        s = json.loads((leaf / SUMMARY).read_text(encoding="utf-8"))
        if s.get("complete"):
            log(f"[{spec.variant} {spec.model_key} seed{spec.seed}] 已完成，跳过（{leaf / SUMMARY}）")
            return s

    parent = None
    if spec.parent_checkpoint is not None:  # C3+预热：起点核对（模块说明）
        if "parent" not in rc:
            raise ValueError(f"run_cfg 没有 parent 一节（resolve_run(variant = {C3_WARMUP})）")
        parent = resolve_parent(spec.parent_checkpoint, model_key=spec.model_key, tiny=spec.tiny, seed=spec.seed)
    trainer = Trainer(adapter, cfg, rc, supervision=spec.supervision, seed=spec.seed, parent=parent)
    st = None
    if exists and (latest / RESUME_STATE).exists():
        trainer.setup_resume(latest / MODEL_SUB)
        st = torch.load(latest / RESUME_STATE, map_location="cpu", weights_only=False)
    else:
        trainer.setup_fresh()
    if warm:
        source = W.WarmupSource(W.WarmupSpec.from_doc(cfg.doc, n=cfg.grid_n, tiny=spec.tiny), seed=spec.seed,
                                tokens=trainer.tokens, batch_size=int(rc["batch_size"]),
                                orders={t: cfg.task(t)["order"] for t in GRID_TASKS})
        if source.spec.stop["max_steps"] != int(rc["steps"]):
            raise ValueError(f"run_cfg 的 steps {rc['steps']} 不是 warmup 的 max_steps {source.spec.stop['max_steps']}")
        builder, schedule = source, source
    else:
        builder = D.ExampleBuilder(cfg, {t: p.records for t, p in pools.items()}, seed=spec.seed, supervision=spec.supervision,
                                   stop_id=trainer.stop_id, tokens=trainer.tokens,
                                   encode=lambda s: list(adapter.tokenizer(s, add_special_tokens=False)["input_ids"]),
                                   family=spec.family, format_boxes=adapter.format_boxes)
        if spec.supervision == MIXED:  # E3b+C3：C3 的文本流 + E3b 的网格流（src/e3/data.MixedSchedule）
            schedule = D.MixedSchedule({t: p.records for t, p in pools.items()}, cfg, seed=spec.seed,
                                       per_step=rc["mixed"]["per_step"], order=rc["mixed"]["order"])
            if schedule.batch_size != int(rc["batch_size"]):
                raise ValueError(f"E3b+C3 每步 {schedule.batch_size} 条，与 run_cfg 的 batch_size {rc['batch_size']} 不同")
        else:
            schedule = D.Schedule({t: p.records for t, p in pools.items()}, cfg, seed=spec.seed, supervision=spec.supervision,
                                  batch_size=int(rc["batch_size"]))
    ident = run_identity(spec, cfg, rc, trainer, pools, dedupe, builder)
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
        files += sorted({Path(p.stats["dedupe_file"]) for p in pools.values() if p.stats.get("dedupe_file")})
        if parent is not None:
            files.append(parent.checkpoint / MANIFEST)
        write_run_metadata(leaf, config=ident, seed=spec.seed, data_files=files, repo_root=REPO_ROOT)
        if trainer.tokens is not None:
            (leaf / TOKENS_INIT).write_text(json.dumps(adapter.token_table(), ensure_ascii=False, indent=1) + "\n",
                                            encoding="utf-8")
    start = 0
    if st is not None:
        if st["identity_sha256"] != ident_sha:
            raise ValueError(f"{latest / RESUME_STATE} 的运行身份与本次不同，拒绝续跑")
        trainer.load_state_dict(st)
        start = int(st["step"])
        log(f"[{spec.variant} {spec.model_key} seed{spec.seed}] 从第 {start} 步续跑")
    tl = [r for r in _read_log(leaf / TRAIN_LOG) if r["step"] < start]
    _atomic_write_text(leaf / TRAIN_LOG, "".join(_jsonl(r) for r in tl))
    history: list[dict[str, Any]] = []
    rule = None
    if warm:  # 网格格式预热：留出集的读数（第 s 步做完之后的读数记 step = s；续跑时保留 step ≤ start 的）
        wcfg = rc["warmup"]
        rule = W.StopRule(threshold=float(wcfg["threshold"]), consecutive=int(wcfg["consecutive"]))
        history = [r for r in _read_log(leaf / WARMUP_EVAL) if r["step"] <= start]
        _atomic_write_text(leaf / WARMUP_EVAL, "".join(_jsonl(r) for r in history))
        if rule.met(history):
            raise RuntimeError(f"{leaf / WARMUP_EVAL}：第 {start} 步之前已满足停止规则，却没有完成的检查点")

        def heldout_eval(ad: Any, tokens: GridTokens, items: Sequence[W.WarmupItem]) -> Mapping[str, Any]:
            return W.evaluate_heldout(ad, tokens, items, n=cfg.grid_n, scope=rc["allowed_scope"], orders=source.orders,
                                      enable_thinking=bool(rc["enable_thinking"]), batch_size=int(wcfg["heldout_batch_size"]))

        eval_fn = _warmup_eval or heldout_eval

    (leaf / "logs").mkdir(exist_ok=True)
    dev = torch.device(adapter.device)
    run_k = len(list((leaf / "logs").glob("run_*.json"))) + 1
    run_path = leaf / "logs" / f"run_{run_k:03d}.json"
    timing_path = leaf / "logs" / f"timing_run_{run_k:03d}.jsonl"
    run_log: dict[str, Any] = {"argv": list(argv), "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                               "device": str(dev), "gpu": torch.cuda.get_device_name(dev) if dev.type == "cuda" else None,
                               "paths": paths.as_dict(), "ckpt_dir": str(ckpt), "start_step": start, "finished_utc": None,
                               "audit": trainer.audit}
    if parent is not None:
        run_log["parent"] = {**parent.record(), "merge": trainer.parent_merge}
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
        trainer.check_frozen()
        tmp = ckpt / (LATEST + ".tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        adapter.save_trained(tmp / MODEL_SUB, extra=trainer.latest_extra())
        torch.save({"identity_sha256": ident_sha, "step": step_done, **trainer.state_dict()}, tmp / RESUME_STATE)
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
    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="e3-data") if workers > 0 else None

    def build_one(plan: D.SlotPlan) -> tuple[D.Example | None, dict[str, Any] | None]:
        try:
            return builder.build(plan), None
        except NoSupervision as e:
            return None, {"k": plan.k, "id": plan.sample_id, "reason": str(e)}

    def submit(step: int) -> list[Any]:
        """第 step 步的样本：有线程池时提交（与上一步的 GPU 计算重叠），否则留到取用时再构造。每条样本都是确定的，与线程的先后无关。"""
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
            n = len(examples)
            comps = [c.as_sample(wall_time_s=step_s / n) for c in comps]
            if step == start:
                shown = examples[:4] if not trainer.mixed else [e for s in STREAMS
                                                                 for e in [x for x in examples if D.stream_of(x.plan) == s][:4]]
                trainer.audit["first_step_examples"] = [{"id": e.plan.sample_id, "task": e.plan.task, "kind": e.plan.kind,
                                                         "n_images": len(e.images), "n_target": len(e.target_ids),
                                                         "n_supervised": e.n_supervised} for e in shown]
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
                log(f"[{spec.variant} {spec.model_key} seed{spec.seed}] 第 {s1}/{steps} 步 loss {rec['loss']:.4f}（{step_s:.2f} s/步）")
            met = False
            if warm and s1 % int(wcfg["eval_every"]) == 0:  # 留出集的格式错误率与停止规则（模块说明）
                t_e = time.perf_counter()
                adapter.eval_mode()
                ev = dict(eval_fn(adapter, trainer.tokens, source.heldout))
                adapter.train_mode()
                _sync(dev)
                ev_rec = {"step": s1, **ev, "below_threshold": rule.below(ev["format_error_rate"])}
                history.append(ev_rec)
                met = rule.met(history)
                ev_rec["stop_rule_met"] = met
                _append(leaf / WARMUP_EVAL, [_jsonl(ev_rec)])
                run_log.setdefault("warmup_evals", []).append({"step": s1, "wall_s": round(time.perf_counter() - t_e, 3)})
                log(f"[{spec.variant} {spec.model_key} seed{spec.seed}] 第 {s1} 步 留出集格式错误率 {ev['format_error_rate']}"
                    f"（{ev['n_format_errors']}/{ev['n']}）" + ("，满足停止规则" if met else ""))
            stop = not met and stop_after_steps is not None and done >= stop_after_steps and s1 < steps
            if not met and ((s1 % every == 0 and s1 < steps) or stop):
                save_latest(s1)
            if _fail_after_step is not None and s1 == _fail_after_step:
                raise RuntimeError(f"测试注入的中断（第 {s1} 步之后）")
            if stop:
                raise StopRequested(f"本次跑了 {done} 步（到第 {s1}/{steps} 步），检查点已写")
            if met:
                break
        adapter.eval_mode()
        trainer.check_frozen()
        final = ckpt / FINAL / MODEL_SUB
        if final.exists():
            shutil.rmtree(final)
        final.parent.mkdir(parents=True, exist_ok=True)
        concat = None
        if parent is None:
            adapter.save_trained(final)
        else:  # C3+预热（模块说明）：先存预热的 LoRA（相对 base + C3），再与 C3 的 LoRA 按秩拼接成 final，逐位核对两半后删掉中间目录
            child = ckpt / (FINAL + ".child")
            if child.exists():
                shutil.rmtree(child)
            adapter.save_trained(child / MODEL_SUB)
            write_rank_concat(parent.checkpoint, child / MODEL_SUB, final,
                              record={"variant": spec.variant, "seed": spec.seed, "parent": parent.identity(),
                                      "training_identity_sha256": ident_sha})
            chk = check_rank_concat(final, parent.checkpoint, child / MODEL_SUB)
            shutil.rmtree(child)
            concat = {"rule": read_manifest(final)["lora_concat"]["rule"],
                      "check": f"check_rank_concat 通过（{chk['checked']}，{chk['n_tensors']} 个张量逐位相同）",
                      "r": chk["r"], "lora_alpha": chk["lora_alpha"]}
        if not keep_optimizer_state and latest.exists():
            shutil.rmtree(latest)
        man = json.loads((final / "manifest.json").read_text(encoding="utf-8"))
        tlog = _read_log(leaf / TRAIN_LOG)
        steps_done = len(tlog) if warm else steps
        summary = {"complete": True, "steps": steps_done, "variant": spec.variant, "model": spec.model_key, "seed": spec.seed,
                   "identity_sha256": ident_sha, "final_loss": tlog[-1]["loss"] if tlog else None,
                   "final_checkpoint": str(final.relative_to(leaf)) if final.is_relative_to(leaf) else str(final),
                   "final_checkpoint_bytes": _dir_bytes(final),
                   "final_checkpoint_files": {k: v["bytes"] for k, v in man["files"].items()},
                   "lora": man.get("lora"), "trainable_token_rows": man.get("trainable_token_rows")}
        if trainer.frozen_llm:  # C1′：没有 LoRA，manifest 不记 trainable_token_rows（不经过适配器的开关），这里补上
            summary.update(frozen_llm=True, trainable_token_rows=list(trainer.rows.ids))
        if parent is not None:  # C3+预热：起点与拼接
            summary.update(parent=parent.record(), lora_concat=concat)
        if trainer.mixed:  # E3b+C3：两种监督各自的合计（每步的 by_supervision 相加）
            summary["by_supervision"] = {s: {k: int(sum(r["by_supervision"][s][k] for r in tlog)) for k in tlog[0]["by_supervision"][s]}
                                         for s in STREAMS} if tlog else None
        if warm:
            ok = rule.met(history)
            summary["warmup"] = {
                "stop_rule_met": ok, "stopped_at_step": steps_done, "max_steps": steps,
                "reason": "format_error_below_threshold" if ok else "max_steps", "n_evals": len(history),
                "last": ({k: history[-1][k] for k in ("step", "n", "n_format_errors", "format_error_rate")} if history else None),
                "history": [{"step": h["step"], "format_error_rate": h["format_error_rate"]} for h in history],
                "rule": {**{k: wcfg[k] for k in ("eval_every", "threshold", "consecutive", "max_steps", "heldout_n")},
                         "text": "每 eval_every 步在留出集上受限解码，格式错误率连续 consecutive 次 < threshold 就停；max_steps 是上限"}}
            if not ok:
                log(f"[{spec.variant} {spec.model_key} seed{spec.seed}] 到了上限 {steps} 步仍未满足停止规则（检查点照存，"
                    "train_summary.json 记 stop_rule_met = false）")
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
    "FrozenRowsProbe",
    "GRID_LIKE",
    "PARENT_MARKER",
    "ParentError",
    "ParentRef",
    "RunSpec",
    "StopRequested",
    "TokenRows",
    "Trainer",
    "WARMUP_EVAL",
    "WITH_GRID_TOKENS",
    "accounting_doc",
    "checkpoint_leaf",
    "frozen_llm_token_rows",
    "identity_sha256",
    "leaf_dir",
    "lr_factor",
    "merge_parent",
    "parent_marker",
    "read_parent_marker",
    "resolve_parent",
    "run_identity",
    "run_training",
    "warmup_identity",
]
