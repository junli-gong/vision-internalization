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

import torch
import torch.nn.functional as F
import yaml

from src.accounting import Compute, merge, write_compute_json
from src.adapters.training import restricted_targets
from src.e3 import data as D
from src.e3.config import E3Config, VARIANT_OF
from src.e3.targets import NoSupervision
from src.e3.tokens import GRID_TASKS, GridTokens, add_vocab, attach_vocab, family_stop_id
from src.runlog import write_run_metadata

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = "config.yaml"
TRAIN_LOG = "train_log.jsonl"
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

    @property
    def variant(self) -> str:
        return VARIANT_OF[self.supervision]


def leaf_dir(out: str | Path, variant: str, model_key: str, seed: int) -> Path:
    return Path(out) / variant / model_key / f"seed{int(seed)}"


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


# ---------------------------------------------------------------------------
# 训练器：一个适配器上的一步优化（网格与 C3 共用）
# ---------------------------------------------------------------------------


class Trainer:
    """持有适配器、网格 token 布局、优化器与调度器；optimize(examples) 做一步（含微批与梯度累积）。"""

    def __init__(self, adapter: Any, cfg: E3Config, run_cfg: Mapping[str, Any], *, supervision: str, seed: int) -> None:
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

    # ----- 搭建 -----
    def setup_fresh(self) -> None:
        torch.manual_seed(self.seed)
        if self.supervision == "grid":
            self.tokens = add_vocab(self.ad, self.cfg.vocab)
        lo = self.rc["lora"]
        self.ad.inject_lora(r=int(lo["r"]), lora_alpha=float(lo["alpha"]), lora_dropout=float(lo["dropout"]))
        self._finish_setup()

    def setup_resume(self, model_dir: Path) -> None:
        self.ad.load_trained(model_dir, merge=False)
        if self.ad.lora_state != "injected":
            raise RuntimeError(f"续跑：检查点载入后 LoRA 状态是 {self.ad.lora_state}，应为 injected")
        if self.supervision == "grid":
            self.tokens = attach_vocab(self.ad, self.cfg.vocab)
        elif self.ad.grid_token_ids:
            raise RuntimeError("续跑：C3 的检查点里不该有 grid token")
        self._finish_setup()

    def _finish_setup(self) -> None:
        ad = self.ad
        lora = [p for n, p in ad.model.named_parameters() if "lora_" in n]
        if not lora or not all(p.requires_grad for p in lora):
            raise RuntimeError("LoRA 参数不存在或没有设为求梯度")
        if self.tokens is not None:
            row_params = ad.trainable_token_rows(list(self.cfg.vocab.names))
            self.rows = TokenRows(row_params, ad.trainable_token_row_ids, self.rc["optim"]["token_rows"]["mode"])
        ad.set_gradient_checkpointing(bool(self.rc["gradient_checkpointing"]))
        o = self.rc["optim"]
        groups = [{"params": lora, "lr": float(o["lora"]["lr"]), "weight_decay": float(o["lora"]["weight_decay"]),
                   "name": "lora"}]
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
        return {"n_lora_params": int(sum(p.numel() for p in lora)),
                "lora_dtypes": sorted({str(p.dtype) for p in lora}), "lora_devices": sorted({str(p.device) for p in lora}),
                "token_row_ids": None if self.rows is None else list(self.rows.ids),
                "token_row_mode": None if self.rows is None else self.rows.mode,
                "token_row_tensors": None if self.rows is None else len(self.rows.params),
                "tied_word_embeddings": bool(ad.tied_word_embeddings), "lora_targets": len(ad.lora_target_modules())}

    # ----- 一步 -----
    def lr_now(self) -> dict[str, float]:
        return {g.get("name", str(k)): float(g["lr"]) for k, g in enumerate(self.opt.param_groups)}

    def _loss(self, out: Any, group: Sequence[D.Example]) -> tuple[torch.Tensor, dict[str, list[float]], list[int]]:
        """一个微批的损失和（未归一化）、各任务的 [损失和, 监督 token 数]、网格 token 的 [对的个数, 计入的个数]。"""
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
            if self.supervision == "grid":
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
        返回 (确定性的记录, 计时, 每条样本的 Compute)。"""
        if not examples:
            raise ValueError("这一步没有样本")
        dev = device if device is not None else self.ad.device
        n_sup = sum(e.n_supervised for e in examples)
        if n_sup <= 0:
            raise ValueError("这一步没有监督 token")
        micro = int(self.rc["micro_batch_size"])
        self.opt.zero_grad(set_to_none=True)
        if self.rows is not None:
            self.rows.zero_dense_grads()
        loss_sum = 0.0
        per: dict[str, list[float]] = {}
        acc = [0, 0]
        counts = {"n_target_tokens": 0, "n_prompt_tokens": 0, "n_visual_tokens": 0, "n_image_pixels": 0}
        times = {"build_inputs_s": 0.0, "fwd_s": 0.0, "bwd_s": 0.0, "opt_s": 0.0}
        computes: list[Compute] = []
        for s in range(0, len(examples), micro):
            group = examples[s: s + micro]
            t0 = time.perf_counter()
            inputs = [self.build_inputs(e) for e in group]
            _sync(dev)
            t1 = time.perf_counter()
            out = self.ad.training_forward(inputs, [e.target_ids for e in group], enable_thinking=bool(self.rc["enable_thinking"]))
            ls, p, a = self._loss(out, group)
            _sync(dev)
            t2 = time.perf_counter()
            (ls / float(n_sup)).backward()
            _sync(dev)
            t3 = time.perf_counter()
            times["build_inputs_s"] += t1 - t0
            times["fwd_s"] += t2 - t1
            times["bwd_s"] += t3 - t2
            loss_sum += float(ls.detach())
            for k, v in p.items():
                acc_k = per.setdefault(k, [0.0, 0.0])
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
        rec = {"lr": lr, "loss": loss_sum / float(n_sup),
               "loss_by_task": {k: (v[0] / v[1] if v[1] else None) for k, v in sorted(per.items())},
               "n_supervised_by_task": {k: int(v[1]) for k, v in sorted(per.items())},
               "n_samples": len(examples), "n_by_kind": dict(sorted(kinds.items())), "n_empty_seg_grid": n_empty,
               "n_supervised_tokens": int(n_sup),
               **{k: int(v) for k, v in counts.items()}, "grad_norm": float(gn),
               "grid_token_acc": (acc[0] / acc[1]) if acc[1] else None}
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


def run_identity(spec: RunSpec, cfg: E3Config, run_cfg: Mapping[str, Any], trainer: Trainer,
                 pools: Mapping[str, Any], dedupe: Mapping[str, Any], builder: D.ExampleBuilder) -> dict[str, Any]:
    d = cfg.doc
    instr = {t: {k: v for k, v in d["tasks"][t].items() if k.endswith(("template", "instruction")) or k == "order"}
             for t in pools}
    c3 = None
    if spec.supervision == "text":
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
                  "depth_noise": {k: v for k, v in d["depth_noise"].items() if k != "basis"}} if spec.supervision == "grid" else None,
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
    return ({"n_runs": len(runs), "n_steps_timed": n_steps, "n_steps_logged": len(log), **sums, "phases_s": tot,
             "wall_s": wall, "gpu_hours": wall / 3600.0, "peak_mem_bytes": max(peaks) if peaks else None,
             "device": runs[-1].get("device") if runs else None, "gpu": runs[-1].get("gpu") if runs else None,
             "note": "phases_s 按步相加：data_s = 构造样本（读图、解码 GT、加噪、渲染叠加图）；build_inputs_s = 处理器；fwd_s、bwd_s = 前向"
                     "（含损失）与反向（CUDA 前后同步）；opt_s = 取主副本梯度、裁剪、step、写回；step_s = 整步。计入 n_steps_timed 的步可能"
                     "多于 n_steps_logged（被中断的步的计时也记了，但没有写进 train_log）。compute.json 的 generated_tokens 是教师强制的"
                     "目标 token 数"}, comp)


def readme_text(spec: RunSpec, ident: Mapping[str, Any], summary: Mapping[str, Any], log: Sequence[Mapping[str, Any]]) -> str:
    tr = ident["training"]
    first, last = (log[0], log[-1]) if log else ({}, {})
    lines = [f"# E3b 训练 · {spec.variant} · {spec.model_key} · seed {spec.seed}", "",
             f"- 模型：{spec.repo}@{spec.revision}（{spec.family}；适配器 {spec.adapter_model_key}；{spec.model_dtype}）",
             f"- 监督：{spec.supervision}（{'GT 网格 + 修正对' if spec.supervision == 'grid' else 'C3：稀疏文本'}）；图像口径 {spec.image_policy}",
             f"- 训练：{tr['steps']} 步 × {tr['batch_size']} 条（微批 {tr['micro_batch_size']} × 累积 {tr['grad_accum_steps']}），"
             f"LoRA r {tr['lora']['r']} / alpha {tr['lora']['alpha']} / dropout {tr['lora']['dropout']}，"
             f"AdamW lr {tr['optim']['lora']['lr']}（token 行 {tr['optim']['token_rows']['lr']}，{tr['optim']['token_rows']['mode']}），"
             f"warmup {tr['warmup_steps']} 步 + 余弦",
             f"- 混合：{tr['mix']}；修正对比例：{tr['correction_fraction']}",
             f"- 第一步损失 {first.get('loss')}，最后一步损失 {last.get('loss')}（{len(log)} 步）",
             f"- 检查点：{summary.get('final_checkpoint')}（{summary.get('final_checkpoint_bytes')} 字节；只有 LoRA、token 行、manifest）", ""]
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
                 log: Callable[[str], None] | None = None, _fail_after_step: int | None = None) -> dict[str, Any] | None:
    """跑一次训练；返回 train_summary（stop_after_steps 到了时返回 None）。adapter 必须是刚挂好的 base（规则见模块说明）。"""
    log = log or (lambda m: print(m, file=sys.stderr, flush=True))
    rc = dict(run_cfg)
    leaf = leaf_dir(out_dir, spec.variant, spec.model_key, spec.seed)
    ckpt = (Path(ckpt_root) / spec.variant / spec.model_key / f"seed{spec.seed}") if ckpt_root else leaf / "ckpt"
    latest = ckpt / LATEST

    # 数据（硬约束 7：加载器没有剔除清单就报错；这里另核对清单比较过本任务的评测集）
    pools, dedupe = {}, {}
    for t in cfg.tasks:
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

    trainer = Trainer(adapter, cfg, rc, supervision=spec.supervision, seed=spec.seed)
    st = None
    if exists and (latest / RESUME_STATE).exists():
        trainer.setup_resume(latest / MODEL_SUB)
        st = torch.load(latest / RESUME_STATE, map_location="cpu", weights_only=False)
    else:
        trainer.setup_fresh()
    builder = D.ExampleBuilder(cfg, {t: p.records for t, p in pools.items()}, seed=spec.seed, supervision=spec.supervision,
                               stop_id=trainer.stop_id, tokens=trainer.tokens,
                               encode=lambda s: list(adapter.tokenizer(s, add_special_tokens=False)["input_ids"]),
                               family=spec.family, format_boxes=adapter.format_boxes)
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
        adapter.save_trained(tmp / MODEL_SUB)
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
                trainer.audit["first_step_examples"] = [{"id": e.plan.sample_id, "task": e.plan.task, "kind": e.plan.kind,
                                                         "n_images": len(e.images), "n_target": len(e.target_ids),
                                                         "n_supervised": e.n_supervised} for e in examples[:4]]
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
            stop = stop_after_steps is not None and done >= stop_after_steps and s1 < steps
            if (s1 % every == 0 and s1 < steps) or stop:
                save_latest(s1)
            if _fail_after_step is not None and s1 == _fail_after_step:
                raise RuntimeError(f"测试注入的中断（第 {s1} 步之后）")
            if stop:
                raise StopRequested(f"本次跑了 {done} 步（到第 {s1}/{steps} 步），检查点已写")
        adapter.eval_mode()
        final = ckpt / FINAL / MODEL_SUB
        if final.exists():
            shutil.rmtree(final)
        final.parent.mkdir(parents=True, exist_ok=True)
        adapter.save_trained(final)
        if not keep_optimizer_state and latest.exists():
            shutil.rmtree(latest)
        man = json.loads((final / "manifest.json").read_text(encoding="utf-8"))
        tlog = _read_log(leaf / TRAIN_LOG)
        summary = {"complete": True, "steps": steps, "variant": spec.variant, "model": spec.model_key, "seed": spec.seed,
                   "identity_sha256": ident_sha, "final_loss": tlog[-1]["loss"] if tlog else None,
                   "final_checkpoint": str(final.relative_to(leaf)) if final.is_relative_to(leaf) else str(final),
                   "final_checkpoint_bytes": _dir_bytes(final),
                   "final_checkpoint_files": {k: v["bytes"] for k, v in man["files"].items()},
                   "lora": man.get("lora"), "trainable_token_rows": man.get("trainable_token_rows")}
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
    "RunSpec",
    "StopRequested",
    "TokenRows",
    "Trainer",
    "accounting_doc",
    "identity_sha256",
    "leaf_dir",
    "lr_factor",
    "run_identity",
    "run_training",
]
