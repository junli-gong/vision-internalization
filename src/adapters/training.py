"""E3 的训练侧公共部分（DEV-0020 第一部分，D21-2）：训练前向、唯一的受限 logits、分块交叉熵、受限解码。

与模型家族无关；家族相关的部分（每行的提示怎么拼、输入嵌入怎么并入视觉特征、位置编码、前向的关键字）由各适配器的钩子提供
（见 TrainingSupport 的"家族钩子"一节）。三个适配器都继承 TrainingSupport，所以公开接口与规则只在这里写一份。

约定（三个家族一致）：

- **训练前向 = 教师强制版的 generate**：每行的提示（思考前缀、system 轮、user 轮、助手头）由适配器的 `_prompt_row` 给出，与
  generate_batch 用的是同一个函数；目标 token 直接接在提示后面；整行（提示 + 目标）左填充，填充 id 与 attention_mask 的
  规则与 generate_batch 相同（batching.pad_left）。位置编码：一维位置取 cumsum(mask) − 1、填充处记 0（HF generate 的
  _prepare_position_ids_for_generation）；Qwen 的 M-RoPE 由适配器按自己的规则构造并与 get_rope_index 逐行核对。
  于是"位置 t 的最后一层隐藏状态预测第 t+1 个 token"：第 k 个目标 token 的预测位置 = 该行最后一个提示 token 的位置 + k。
- **logits 只有一种算法**（`project`）：`F.linear(hidden.float(), weight.float())`，即 hidden 与 lm_head 的行都升到 fp32 再乘，
  结果是 fp32；有 final_logit_softcapping 的家族（Gemma）再做 `tanh(z / cap) · cap`（fp32）。lm_head 是否与嵌入共享不影响算法：
  一律取 `model.get_output_embeddings().weight` 的行（共享时就是嵌入矩阵本身，Gemma 的 embed_scale 只作用在查表上，不作用在
  lm_head 上）。受限 logits（`restricted_logits`）只对 allowed 的行做这个乘法；全词表的分块交叉熵（`chunked_lm_loss`）对全部行
  分块做同一个乘法。训练损失与受限解码都调同一个函数。
  与 HF 的差别（写明，不是 bug）：HF 的 lm_head 在模型 dtype 下乘、再转 fp32（softcap 在模型 dtype 下做）。fp32 模型上两者
  只差浮点求和顺序；bf16 模型上差 bf16 舍入，所以 GPU 上"受限解码 = generate_batch(allowed_token_ids)"写成一致率，不要求逐位相同。
- **受限解码**（`generate_restricted`）：手写的 greedy 循环，带 KV cache（GDN 的状态也在同一个 cache 里），每一步调**内层模型**
  （`model.model`，不经过 lm_head）取最后一层最后一个位置的隐藏状态，再调 `restricted_logits`；不经过 HF 的全词表 logits，也不经过
  任何 logits 处理器。每一步送进内层模型的输入与 HF generate 逐项相同：预填用左填充的整批提示（与 generate_batch 同一段拼批
  `_batch_kw`）、HF 规则的位置（Qwen 为 [文本位置; get_rope_index]），之后每步只送新 token、attention_mask 末尾补 1、
  位置 = 上一步 + 1（HF 的 _update_model_kwargs_for_generation）；已结束的行补 pad（与 HF 相同）。平局取 id 小的 token
  （allowed 先排序，argmax 取第一个最大值，与 HF greedy 在 -inf 掩码下的 torch.argmax 相同）。
- 模式：`train_mode()` 把 LLM 切到训练模式（dropout 生效、梯度检查点生效），视觉塔与投影层保持 eval（默认冻结）；
  `eval_mode()` 全部切回 eval。生成类方法（generate、generate_batch、option_logits、generate_restricted）在训练模式下报错：
  训练中途做验证要先 eval_mode()（D21-2）。梯度检查点用 `set_gradient_checkpointing(True)`（use_reentrant=False）。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import torch
import torch.nn.functional as F

from src.adapters import batching
from src.adapters.base import GenerationResult, ModelInputs, Span

# 内置的隐藏状态区域名（HiddenRequest.regions）
REGION_QUERY = "query"
REGION_IMAGE = "image"
REGIONS: tuple[str, ...] = (REGION_QUERY, REGION_IMAGE)

# E3a 开放词汇头：类别名在提示里的拼法（class_name_spans）。类别名之间用 ", " 分隔，整串前面紧挨一个空格，
# 例如 "Segment these classes: cat, traffic light, dog."。每个类别名的区间是 " 名字" 这一段的 token（含前导空格并进去的那个 token）。
CLASS_NAME_SEPARATOR = ", "


class TrainingInputError(ValueError):
    """training_forward / generate_restricted / restricted_logits 的入参不合规（目标 id、查询嵌入、allowed 等）。"""


class TrainingModeError(RuntimeError):
    """模式不对：训练模式下调用生成类方法（先 eval_mode()）。"""


class TokenOrderError(RuntimeError):
    """顺序约束（DEV-0020）：add_grid_tokens、inject_query_tokens、restore_tokens 必须在 inject_lora 之前调用。
    理由：新增 token 的行（可能扩表）与查询登记要在 LoRA 包装模型之前定下来，检查点的 token 表才与训练时一致。"""


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HiddenRequest:
    """training_forward 另外要读的隐藏状态（带梯度；E3a 的逐层加权混合用）。

    - layers：层号（base.py 的层号约定：0 = 嵌入输出，i = 第 i 个 decoder 层之后，num_layers() 取最终归一化之后），不许重复。
    - regions：内置区域，"query"（ModelInputs.query_span）与 / 或 "image"（各图 ImageTokenInfo.span 按图的顺序拼接）。
    - spans：每行一个 {名字: Span}（下标相对于该行 ModelInputs 的 input_ids，即 build_inputs 给的区间；例如 class_name_spans 的结果），
      各行的名字集合必须相同，不能与内置区域重名。
    结果在 TrainingOutput.hidden[区域或名字][层号] 里，是每行一个 [区间长度, hidden] 的张量列表。
    """

    layers: tuple[int, ...]
    regions: tuple[str, ...] = ()
    spans: tuple[Mapping[str, Span], ...] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.layers, (str, bytes)) or not self.layers:
            raise TrainingInputError("HiddenRequest.layers 必须是非空的层号序列")
        object.__setattr__(self, "layers", tuple(self.layers))
        object.__setattr__(self, "regions", tuple(self.regions))
        bad = [r for r in self.regions if r not in REGIONS]
        if bad or len(set(self.regions)) != len(self.regions):
            raise TrainingInputError(f"regions 只能取 {REGIONS} 且不重复，收到 {self.regions!r}")
        if self.spans is not None:
            spans = tuple(dict(s) for s in self.spans)
            keys = [tuple(sorted(s)) for s in spans]
            if len(set(keys)) > 1:
                raise TrainingInputError(f"HiddenRequest.spans 各行的名字集合不同：{sorted(set(keys))}")
            clash = sorted(set(keys[0]) & set(REGIONS)) if keys else []
            if clash:
                raise TrainingInputError(f"HiddenRequest.spans 的名字与内置区域重名：{clash}")
            for s in spans:
                for k, v in s.items():
                    if not isinstance(k, str) or not k or not isinstance(v, Span):
                        raise TrainingInputError(f"HiddenRequest.spans 的项必须是 名字 → Span，收到 {k!r}: {v!r}")
            object.__setattr__(self, "spans", spans)
        if not self.regions and not self.spans:
            raise TrainingInputError("HiddenRequest 至少要有一个区域或区间")


@dataclass
class TrainingOutput:
    """training_forward 的结果。行的顺序与 inputs_list 相同；N = 全部目标 token 的个数（各行按顺序拼接）。

    - target_hidden [N, hidden]：预测各目标 token 的最后一层隐藏状态（lm_head 之前、最终归一化之后），带梯度。
      第 k 个目标 token（行 b）由 target_positions[n] 这一列的隐藏状态预测；行 b 的目标是 target_offsets[b]:target_offsets[b+1]。
    - target_ids [N]、target_rows [N]、target_positions [N]（long，模型所在设备）。
    - input_ids、attention_mask [B, L]：实际送进模型的整批（提示 + 目标，左填充）。n_padding、prompt_lengths、target_lengths 按行。
    - hidden：HiddenRequest 要的隐藏状态（见 HiddenRequest），没要时为 None。
    """

    target_hidden: torch.Tensor
    target_ids: torch.Tensor
    target_rows: torch.Tensor
    target_positions: torch.Tensor
    target_offsets: tuple[int, ...]
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    n_padding: tuple[int, ...]
    prompt_lengths: tuple[int, ...]
    target_lengths: tuple[int, ...]
    hidden: dict[str, dict[int, list[torch.Tensor]]] | None = None

    def row_ids(self, b: int) -> list[int]:
        """第 b 行实际送进模型的 token（去掉左填充）= 提示 + 目标。"""
        return self.input_ids[b, self.n_padding[b]:].tolist()

    def row_target_hidden(self, b: int) -> torch.Tensor:
        return self.target_hidden[self.target_offsets[b]: self.target_offsets[b + 1]]


@dataclass(frozen=True)
class _Assembled:
    """整批（提示 + 目标）左填充之后的张量与对齐。"""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    mm_token_type_ids: torch.Tensor | None
    n_padding: tuple[int, ...]
    prompt_lengths: tuple[int, ...]
    target_lengths: tuple[int, ...]
    target_rows: torch.Tensor
    target_cols: torch.Tensor
    target_ids: torch.Tensor
    target_offsets: tuple[int, ...]


# ---------------------------------------------------------------------------
# logits：唯一的算法
# ---------------------------------------------------------------------------


def project(hidden: torch.Tensor, weight: torch.Tensor, *, softcap: float | None) -> torch.Tensor:
    """hidden [..., H] × weight [V', H] → fp32 logits [..., V']：两者升到 fp32 再乘；softcap 非空时 tanh(z / cap) · cap（fp32）。"""
    z = F.linear(hidden.float(), weight.float())
    if softcap is not None:
        z = torch.tanh(z / float(softcap)) * float(softcap)
    return z


def check_allowed_ids(allowed_ids: Any, vocab_rows: int, *, device: torch.device) -> torch.Tensor:
    """allowed_ids：非空、不重复、落在 [0, vocab_rows) 里的整数（list 或 long 张量）→ 设备上的 long 张量（顺序不变）。"""
    if torch.is_tensor(allowed_ids):
        if allowed_ids.dim() != 1 or allowed_ids.dtype not in (torch.int64, torch.int32):
            raise TrainingInputError(f"allowed_ids 张量必须是一维整数，收到 {tuple(allowed_ids.shape)} {allowed_ids.dtype}")
        ids = [int(i) for i in allowed_ids.tolist()]
    else:
        if isinstance(allowed_ids, (str, bytes)):
            raise TrainingInputError("allowed_ids 必须是整数序列")
        ids = list(allowed_ids)
        if any(isinstance(i, bool) or not isinstance(i, int) for i in ids):
            raise TrainingInputError(f"allowed_ids 必须是整数，收到 {ids[:5]}…")
    if not ids:
        raise TrainingInputError("allowed_ids 为空")
    if len(set(ids)) != len(ids):
        raise TrainingInputError("allowed_ids 有重复")
    bad = [i for i in ids if not 0 <= i < vocab_rows]
    if bad:
        raise TrainingInputError(f"allowed_ids 越界：{bad[:5]}，lm_head 行数 {vocab_rows}")
    return torch.tensor(ids, dtype=torch.long, device=device)


def restricted_logits(hidden: torch.Tensor, weight: torch.Tensor, allowed_ids: Any, *,
                      softcap: float | None) -> torch.Tensor:
    """受限 logits [..., len(allowed)]（fp32，列的顺序 = allowed_ids 的顺序）：只对 allowed 的 lm_head 行做 project。"""
    if not torch.is_tensor(hidden) or not hidden.is_floating_point() or hidden.dim() < 1:
        raise TrainingInputError("hidden 必须是浮点张量 [..., hidden_size]")
    if hidden.shape[-1] != weight.shape[1]:
        raise TrainingInputError(f"hidden 的最后一维 {hidden.shape[-1]} 与 lm_head 的宽度 {weight.shape[1]} 不一致")
    if hidden.device != weight.device:
        raise TrainingInputError(f"hidden 在 {hidden.device}，lm_head 在 {weight.device}")
    idx = check_allowed_ids(allowed_ids, int(weight.shape[0]), device=weight.device)
    return project(hidden, weight.index_select(0, idx), softcap=softcap)


def restricted_targets(target_ids: Any, allowed_ids: Any) -> torch.Tensor:
    """把目标 token id 换成它在 allowed_ids 里的下标（受限网格损失：F.cross_entropy(restricted_logits(h, allowed), 这个下标)）。
    目标不在 allowed 里就报错。"""
    allowed = [int(i) for i in (allowed_ids.tolist() if torch.is_tensor(allowed_ids) else allowed_ids)]
    pos = {t: k for k, t in enumerate(allowed)}
    if len(pos) != len(allowed):
        raise TrainingInputError("allowed_ids 有重复")
    tgt = target_ids.tolist() if torch.is_tensor(target_ids) else list(target_ids)
    missing = sorted({int(t) for t in tgt} - set(pos))
    if missing:
        raise TrainingInputError(f"目标 token {missing[:5]} 不在 allowed_ids 里")
    device = target_ids.device if torch.is_tensor(target_ids) else None
    return torch.tensor([pos[int(t)] for t in tgt], dtype=torch.long, device=device)


def _chunk_ce_sum(h: torch.Tensor, t: torch.Tensor, weight: torch.Tensor, softcap: float | None) -> torch.Tensor:
    return F.cross_entropy(project(h, weight, softcap=softcap), t, reduction="sum")


def chunked_lm_loss(hidden: torch.Tensor, target_ids: torch.Tensor, weight: torch.Tensor, *, softcap: float | None,
                    chunk_size: int, reduction: str = "mean") -> torch.Tensor:
    """全词表 LM 交叉熵（fp32），按位置分块：每块 [chunk_size, V] 的 fp32 logits 用完即弃；有梯度时每块套
    torch.utils.checkpoint（use_reentrant=False），反向时重算这一块，不把整批的 logits 存下来（简报 §5.3、§12）。

    hidden [N, H]、target_ids [N]（long）；reduction="mean"（÷ N）或 "sum"。与不分块的 F.cross_entropy(project(hidden, W), t)
    在 fp32 下只差求和顺序（测试的容差写在 tests/test_adapter_training_logits.py）。
    """
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise TrainingInputError(f"chunk_size 必须是正整数，收到 {chunk_size!r}")
    if reduction not in ("mean", "sum"):
        raise TrainingInputError(f"reduction 只能是 mean 或 sum，收到 {reduction!r}")
    if hidden.dim() != 2 or hidden.shape[1] != weight.shape[1]:
        raise TrainingInputError(f"hidden 必须是 [N, {weight.shape[1]}]，收到 {tuple(hidden.shape)}")
    if target_ids.dim() != 1 or target_ids.shape[0] != hidden.shape[0] or target_ids.dtype != torch.long:
        raise TrainingInputError(f"target_ids 必须是 [N] 的 long，收到 {tuple(target_ids.shape)} {target_ids.dtype}")
    n = hidden.shape[0]
    if n == 0:
        raise TrainingInputError("没有目标 token")
    if int(target_ids.min()) < 0 or int(target_ids.max()) >= weight.shape[0]:
        raise TrainingInputError(f"target_ids 越界（lm_head 行数 {weight.shape[0]}）")
    grad = torch.is_grad_enabled() and (hidden.requires_grad or weight.requires_grad)
    total = hidden.new_zeros((), dtype=torch.float32)
    for s in range(0, n, chunk_size):
        h, t = hidden[s: s + chunk_size], target_ids[s: s + chunk_size]
        if grad:
            total = total + torch.utils.checkpoint.checkpoint(_chunk_ce_sum, h, t, weight, softcap, use_reentrant=False)
        else:
            total = total + _chunk_ce_sum(h, t, weight, softcap)
    return total / n if reduction == "mean" else total


# ---------------------------------------------------------------------------
# 入参检查与拼批
# ---------------------------------------------------------------------------


def check_training_inputs(inputs_list: Any, *, where: str) -> list[ModelInputs]:
    """非空的 ModelInputs 序列（训练前向允许带查询 token）。"""
    if isinstance(inputs_list, ModelInputs):
        raise TrainingInputError(f"{where}：inputs_list 必须是 ModelInputs 的序列（单条也放进列表）")
    items = list(inputs_list)
    if not items:
        raise TrainingInputError(f"{where}：inputs_list 为空")
    for k, inp in enumerate(items):
        if not isinstance(inp, ModelInputs):
            raise TypeError(f"{where}：inputs_list[{k}] 必须是 ModelInputs，收到 {type(inp).__name__}")
    return items


def check_targets(target_ids_list: Any, n_rows: int, vocab_rows: int) -> list[list[int]]:
    """每行的目标 token：整数序列（可以为空），与输入等长，id 落在 [0, vocab_rows)。"""
    if isinstance(target_ids_list, (str, bytes)) or torch.is_tensor(target_ids_list):
        raise TrainingInputError("target_ids_list 必须是每行一个整数序列的列表")
    rows = list(target_ids_list)
    if len(rows) != n_rows:
        raise TrainingInputError(f"target_ids_list 有 {len(rows)} 行，输入有 {n_rows} 条")
    out = []
    for k, r in enumerate(rows):
        if isinstance(r, (str, bytes)):
            raise TrainingInputError(f"target_ids_list[{k}] 必须是整数序列")
        ids = r.tolist() if torch.is_tensor(r) else list(r)
        if any(isinstance(i, bool) or not isinstance(i, int) for i in ids):
            raise TrainingInputError(f"target_ids_list[{k}] 必须是整数，收到 {ids[:5]}…")
        bad = [i for i in ids if not 0 <= i < vocab_rows]
        if bad:
            raise TrainingInputError(f"target_ids_list[{k}] 越界：{bad[:5]}，lm_head 行数 {vocab_rows}")
        out.append([int(i) for i in ids])
    return out


def assemble(rows: Sequence[batching.PromptRow], targets: Sequence[Sequence[int]], *, pad_id: int,
             device: torch.device) -> _Assembled:
    """提示 + 目标拼成整行，左填充（与 generate_batch 的 batch_prompt 同一个 pad_left）；mm_token_type_ids 在目标处为 0。"""
    full = [list(r.ids) + list(t) for r, t in zip(rows, targets)]
    width = max(len(f) for f in full)
    ids = batching.pad_left(full, pad_value=pad_id, device=device)
    mask = batching.pad_left([[1] * len(f) for f in full], pad_value=0, device=device)
    have_mm = [r.mm_token_type_ids is not None for r in rows]
    mm = None
    if any(have_mm):
        if not all(have_mm):
            raise RuntimeError("有的行给了 mm_token_type_ids、有的没给")
        mm = batching.pad_left([list(r.mm_token_type_ids) + [0] * len(t) for r, t in zip(rows, targets)], pad_value=0,
                               device=device)
    pads = tuple(width - len(f) for f in full)
    t_rows, t_cols, t_ids, offsets = [], [], [], [0]
    for b, (r, t) in enumerate(zip(rows, targets)):
        first = pads[b] + len(r.ids) - 1  # 预测第一个目标 token 的位置 = 最后一个提示 token
        t_rows += [b] * len(t)
        t_cols += list(range(first, first + len(t)))
        t_ids += list(t)
        offsets.append(offsets[-1] + len(t))
    long = dict(dtype=torch.long, device=device)
    return _Assembled(ids, mask, mm, pads, tuple(len(r.ids) for r in rows), tuple(len(t) for t in targets),
                      torch.tensor(t_rows, **long), torch.tensor(t_cols, **long), torch.tensor(t_ids, **long),
                      tuple(offsets))


def cumsum_positions(attention_mask: torch.Tensor) -> torch.Tensor:
    """HF generate 的一维位置：cumsum(mask) − 1，填充处记 0（GenerationMixin._prepare_position_ids_for_generation）。"""
    pos = attention_mask.long().cumsum(-1) - 1
    return pos.masked_fill(attention_mask == 0, 0)


def scatter_rows(embeds: torch.Tensor, mask: torch.Tensor, values: torch.Tensor, *, what: str) -> torch.Tensor:
    """embeds [B, L, H] 在 mask [B, L] 为真的位置按行主序换成 values [n, H]（masked_scatter，对 values 可求导）。"""
    n = int(mask.sum())
    if values.shape != (n, embeds.shape[-1]):
        raise RuntimeError(f"{what}：要替换 {n} 个位置，给了 {tuple(values.shape)}")
    return embeds.masked_scatter(mask.unsqueeze(-1), values.to(embeds.dtype))


def _scoped_checkpoint(ckpt: Callable[..., Any], scope: Callable[[], Any]) -> Callable[..., Any]:
    """检查点函数 ckpt(fn, *args) 的包装：被检查点的 fn 每次运行（前向，以及反向时的重算）都在 scope() 里。"""

    def run(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        def scoped(*a: Any, **k: Any) -> Any:
            with scope():
                return fn(*a, **k)

        return ckpt(scoped, *args, **kwargs)

    run.scoped = True  # type: ignore[attr-defined]  # 测试据此确认已包装
    return run


# ---------------------------------------------------------------------------
# 受限解码的循环（家族无关；每步的输入由 _decode_* 钩子给出）
# ---------------------------------------------------------------------------


def greedy_restricted_loop(*, forward: Callable[..., Any], prefill_kw: dict[str, Any],
                           step_extra: Callable[[int], dict[str, Any]], logits_fn: Callable[[torch.Tensor], torch.Tensor],
                           allowed: torch.Tensor, stop_ids: Sequence[int], pad_id: int,
                           max_new_tokens: int) -> torch.Tensor:
    """greedy 循环，返回 [B, T] 的生成 token（CPU）；规则同 HF _sample：已结束的行补 pad，全部结束或到 max_new_tokens 停。

    forward(**kw) 是内层模型（不经过 lm_head），返回 last_hidden_state 与 past_key_values。prefill_kw 含 input_ids、
    attention_mask、position_ids 等；之后每步送 input_ids=[B, 1]、attention_mask 末尾补 1、position_ids = 上一步最后一列 + 1。
    allowed 是升序的 allowed id（logits_fn 的列与它对应）。
    """
    mask = prefill_kw["attention_mask"]
    pos = prefill_kw["position_ids"]
    batch = mask.shape[0]
    device = mask.device
    stop = torch.tensor(sorted(set(int(s) for s in stop_ids)), dtype=torch.long, device=device)
    out = forward(**prefill_kw, use_cache=True, return_dict=True)
    cache = out.past_key_values
    hid = out.last_hidden_state[:, -1]
    unfinished = torch.ones(batch, dtype=torch.bool, device=device)
    steps: list[torch.Tensor] = []
    for k in range(max_new_tokens):
        nxt = allowed[logits_fn(hid).argmax(dim=-1)]
        nxt = torch.where(unfinished, nxt, torch.full_like(nxt, pad_id))
        steps.append(nxt)
        if stop.numel():
            unfinished = unfinished & ~torch.isin(nxt, stop)
        if k + 1 == max_new_tokens or not bool(unfinished.any()):
            break
        mask = torch.cat([mask, mask.new_ones((batch, 1))], dim=-1)
        pos = pos[..., -1:] + 1
        out = forward(input_ids=nxt.unsqueeze(-1), attention_mask=mask, position_ids=pos, past_key_values=cache,
                      use_cache=True, return_dict=True, **step_extra(batch))
        cache = out.past_key_values
        hid = out.last_hidden_state[:, -1]
    return torch.stack(steps, dim=1).cpu()


# ---------------------------------------------------------------------------
# 三个适配器共用的实现
# ---------------------------------------------------------------------------


class TrainingSupport:
    """training_forward、restricted_logits、chunked_lm_loss、generate_restricted、train_mode / eval_mode、梯度检查点、
    class_name_spans 的公共实现（规则见模块说明与 base.ModelAdapter 里的接口说明）。

    家族钩子（各适配器实现）：
    - `_prompt_row(inputs, enable_thinking)`：与 generate_batch 同一个函数；
    - `_user_index_shift(enable_thinking) -> (offset, start)`：ModelInputs 的下标 i（i ≥ start）在提示行里是 i + offset；
    - `_pad_id()`、`_stop_ids()`、`_forward_scope()`（Qwen：CPU 上走 GDN 的 torch 实现）；
    - `_image_kw(items)`：各行图像张量按行拼接（与 _batch_kw 相同）；`_image_features_from(tensors, sizes)`；
    - `_training_forward_kwargs(items, asm, image_kw, query_mask, query_embeds)`：送进内层模型的关键字（输入嵌入、位置等）；
    - `_decode_positions(kw)`、`_decode_step_extra(batch)`：受限解码每一步的位置与额外输入；
    - `_logit_softcap()`：final_logit_softcapping（没有就是 None）；`_vision_modules()`：train_mode 下保持 eval 的模块。
    """

    # ------------------------------------------------------------------
    # 模式
    # ------------------------------------------------------------------

    def train_mode(self) -> None:
        """LLM 切到训练模式（dropout、梯度检查点生效）；视觉塔与投影层保持 eval（默认冻结，D21-2、简报 §8.4）。"""
        self._require_loaded()
        self._model.train()
        for m in self._vision_modules():
            m.eval()

    def eval_mode(self) -> None:
        """全部切回 eval（dropout 关闭；训练中途做验证、生成之前调用）。"""
        self._require_loaded()
        self._model.eval()

    @property
    def training(self) -> bool:
        self._require_loaded()
        return bool(self._model.training)

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        """打开 / 关闭梯度检查点（HF gradient_checkpointing_enable，use_reentrant=False；只在训练模式下生效）。

        检查点的重算发生在反向时，已经出了 training_forward 的前向作用域（Qwen 在 CPU 上的 torch_gdn()）。所以每个层的检查点函数
        再包一层：被检查点的函数每次运行（前向与反向里的重算）都进入 _forward_scope()（嵌套进入是允许的）。否则装了 fla 的机器上，
        CPU 的重算会把 CPU 张量送进 fla 的 Triton 内核（server-parity 条件下复现过）。
        """
        self._require_loaded()
        if not isinstance(enabled, bool):
            raise TypeError(f"enabled 必须是 bool，收到 {enabled!r}")
        if not enabled:
            self._model.gradient_checkpointing_disable()
            return
        self._model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        n = 0
        for m in self._model.modules():
            fn = getattr(m, "_gradient_checkpointing_func", None)
            if getattr(m, "gradient_checkpointing", False) and fn is not None:
                m._gradient_checkpointing_func = _scoped_checkpoint(fn, self._forward_scope)
                n += 1
        if n == 0:
            raise RuntimeError("gradient_checkpointing_enable 之后没有找到任何检查点层")

    @property
    def gradient_checkpointing(self) -> bool:
        self._require_loaded()
        return bool(getattr(self._model, "is_gradient_checkpointing", False))

    def _require_eval(self, where: str) -> None:
        if self._model is not None and self._model.training:
            raise TrainingModeError(f"{where}：模型在训练模式（dropout 生效），先调用 eval_mode()")

    def _check_tokens_mutable(self, where: str) -> None:
        """顺序约束：LoRA 注入（或合并）之后不能再加 token、注入查询或恢复 token 表（TokenOrderError）。"""
        state = getattr(self, "_lora_state", "none")
        if state != "none":
            raise TokenOrderError(f"{where} 必须在 inject_lora 之前调用（当前 LoRA 状态：{state}）：先 add_grid_tokens / "
                                  "inject_query_tokens（或 restore_tokens），再 inject_lora")

    # ------------------------------------------------------------------
    # logits
    # ------------------------------------------------------------------

    def lm_head_weight(self) -> torch.Tensor:
        """lm_head 的权重 [V, H]（共享时就是嵌入矩阵）。有 bias 就报错（三个家族都没有）。"""
        self._require_loaded()
        head = self._model.get_output_embeddings()
        if getattr(head, "bias", None) is not None:
            raise RuntimeError("lm_head 有 bias，restricted_logits 的算法没有考虑")
        return head.weight

    def restricted_logits(self, hidden: torch.Tensor, allowed_ids: Any) -> torch.Tensor:
        """唯一的受限 logits（训练损失与受限解码都调它）：只对 allowed 的 lm_head 行做矩阵乘，fp32，含 softcapping。
        规则见模块说明（training.project）。"""
        return restricted_logits(hidden, self.lm_head_weight(), allowed_ids, softcap=self._logit_softcap())

    def chunked_lm_loss(self, hidden: torch.Tensor, target_ids: torch.Tensor, *, chunk_size: int,
                        reduction: str = "mean") -> torch.Tensor:
        """全词表 LM 交叉熵，按位置分块（training.chunked_lm_loss；logits 与 restricted_logits 同一个算法）。"""
        return chunked_lm_loss(hidden, target_ids, self.lm_head_weight(), softcap=self._logit_softcap(),
                               chunk_size=chunk_size, reduction=reduction)

    # ------------------------------------------------------------------
    # 训练前向
    # ------------------------------------------------------------------

    def training_forward(self, inputs_list: Sequence[ModelInputs], target_ids_list: Sequence[Sequence[int]], *,
                         enable_thinking: bool, return_hidden: HiddenRequest | None = None,
                         query_embeds: torch.Tensor | None = None) -> TrainingOutput:
        """教师强制版的 generate（约定见模块说明与 base.ModelAdapter.training_forward）。不强制 no_grad。"""
        where = "training_forward"
        self._require_loaded()
        batching.check_enable_thinking(enable_thinking)
        items = check_training_inputs(inputs_list, where=where)
        if return_hidden is not None and not isinstance(return_hidden, HiddenRequest):
            raise TypeError(f"return_hidden 必须是 HiddenRequest，收到 {type(return_hidden).__name__}")
        has_q = [inp.query_span is not None for inp in items]
        if any(has_q) and not all(has_q):
            raise TrainingInputError(f"{where}：一批里有的行带查询 token、有的不带")
        if all(has_q):
            if query_embeds is None:
                raise TrainingInputError(f"{where}：输入里有查询 token，必须显式传 query_embeds（D21-2）")
            if len({len(inp.query_span) for inp in items}) != 1:
                raise TrainingInputError(f"{where}：各行的查询 token 个数不同")
        for inp in items:
            self._check_entry(inp, query_embeds if inp.query_span is not None else None, where=where)
        if query_embeds is not None and not all(has_q):
            raise TrainingInputError(f"{where}：给了 query_embeds，但输入里没有查询 token")
        vocab_rows = int(self.lm_head_weight().shape[0])
        targets = check_targets(target_ids_list, len(items), vocab_rows)
        layers = self._check_layers(return_hidden.layers) if return_hidden is not None else []
        if return_hidden is not None and return_hidden.spans is not None and len(return_hidden.spans) != len(items):
            raise TrainingInputError(f"HiddenRequest.spans 有 {len(return_hidden.spans)} 行，输入有 {len(items)} 条")
        if return_hidden is not None and REGION_QUERY in return_hidden.regions and not all(has_q):
            raise TrainingInputError("HiddenRequest 要 query 区域，但输入里没有查询 token")

        rows = [self._prompt_row(inp, enable_thinking) for inp in items]
        asm = assemble(rows, targets, pad_id=self._pad_id(), device=self._device)
        offset, start = self._user_index_shift(enable_thinking)

        def to_cols(b: int, span: Span) -> torch.Tensor:
            if span.start < start and len(span):
                raise TrainingInputError(f"区间 [{span.start}, {span.end}) 落在提示的固定前缀里")
            if span.end > items[b].tensors["input_ids"].shape[1]:
                raise TrainingInputError(f"区间 [{span.start}, {span.end}) 超出第 {b} 行 user 轮的长度")
            s = asm.n_padding[b] + span.start + offset
            return torch.arange(s, s + len(span), device=self._device)

        query_mask = None
        if all(has_q):
            query_mask = torch.zeros_like(asm.attention_mask, dtype=torch.bool)
            for b, inp in enumerate(items):
                query_mask[b, to_cols(b, inp.query_span)] = True
        image_kw = self._image_kw(items)
        with self._forward_scope():
            fwd = self._training_forward_kwargs(items, asm, image_kw, query_mask, query_embeds)
            out = self._model.model(**fwd, output_hidden_states=return_hidden is not None, use_cache=False,
                                    return_dict=True)
        last = out.last_hidden_state
        hidden = None
        if return_hidden is not None:
            hs = out.hidden_states
            if hs is None or len(hs) != self.num_layers() + 1:
                raise RuntimeError(f"hidden_states 有 {None if hs is None else len(hs)} 项，期望 num_layers()+1")
            regions: dict[str, list[torch.Tensor]] = {}
            for r in return_hidden.regions:
                if r == REGION_QUERY:
                    regions[r] = [to_cols(b, inp.query_span) for b, inp in enumerate(items)]
                else:
                    regions[r] = [torch.cat([to_cols(b, i.span) for i in inp.images]) if inp.images
                                  else torch.empty(0, dtype=torch.long, device=self._device)
                                  for b, inp in enumerate(items)]
            for name in (return_hidden.spans[0] if return_hidden.spans else {}):
                regions[name] = [to_cols(b, return_hidden.spans[b][name]) for b in range(len(items))]
            hidden = {name: {l: [hs[l][b].index_select(0, cols) for b, cols in enumerate(per_row)] for l in layers}
                      for name, per_row in regions.items()}
        return TrainingOutput(
            target_hidden=last[asm.target_rows, asm.target_cols],
            target_ids=asm.target_ids,
            target_rows=asm.target_rows,
            target_positions=asm.target_cols,
            target_offsets=asm.target_offsets,
            input_ids=asm.input_ids,
            attention_mask=asm.attention_mask,
            n_padding=asm.n_padding,
            prompt_lengths=asm.prompt_lengths,
            target_lengths=asm.target_lengths,
            hidden=hidden,
        )

    # ------------------------------------------------------------------
    # 受限解码
    # ------------------------------------------------------------------

    def generate_restricted(self, inputs_list: Sequence[ModelInputs], allowed_ids: Any, *, max_new_tokens: int,
                            stop_ids: Sequence[int], enable_thinking: bool) -> list[GenerationResult]:
        """受限 greedy 解码（规则见模块说明）。返回与 generate_batch 同样的 GenerationResult（停止、拆分、记账的规则相同）。

        allowed_ids：非空、不重复；内部按升序用（平局取 id 小的 token）。stop_ids：生成到其中任何一个就结束这一行（可以为空：
        一律生成满 max_new_tokens）；想让它能停，调用方要把停止 token 放进 allowed_ids。带查询 token 的输入不能生成。
        """
        where = "generate_restricted"
        self._require_loaded()
        batching.check_generate_args(enable_thinking, max_new_tokens)
        self._require_eval(where)
        items = batching.check_inputs_list(inputs_list, where=where)
        if isinstance(stop_ids, (str, bytes)) or torch.is_tensor(stop_ids):
            raise TrainingInputError("stop_ids 必须是整数序列")
        stops = list(stop_ids)
        if any(isinstance(s, bool) or not isinstance(s, int) for s in stops):
            raise TrainingInputError(f"stop_ids 必须是整数，收到 {stops!r}")
        for inp in items:
            self._check_entry(inp, None, where=where)
        weight = self.lm_head_weight()
        allowed = check_allowed_ids(allowed_ids, int(weight.shape[0]), device=weight.device)
        allowed = torch.sort(allowed).values
        bad_stop = [s for s in stops if not 0 <= s < weight.shape[0]]
        if bad_stop:
            raise TrainingInputError(f"stop_ids 越界：{bad_stop}")
        rows, kw = self._batch_kw(items, enable_thinking)
        prefill = dict(kw)
        prefill["position_ids"] = self._decode_positions(kw)
        pad_id = self._pad_id()
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)
        t0 = time.perf_counter()
        with self._forward_scope(), torch.no_grad():
            gen = greedy_restricted_loop(
                forward=self._model.model, prefill_kw=prefill, step_extra=self._decode_step_extra,
                logits_fn=lambda h: self.restricted_logits(h, allowed), allowed=allowed, stop_ids=stops,
                pad_id=pad_id, max_new_tokens=max_new_tokens)
        wall = time.perf_counter() - t0
        out = torch.cat([kw["input_ids"].cpu(), gen], dim=1)
        return batching.make_results(rows, out, kw["input_ids"].shape[1], wall=wall, stop_ids=stops, pad_id=pad_id,
                                     max_new_tokens=max_new_tokens, split_answer=self.split_answer,
                                     enable_thinking=enable_thinking)

    # ------------------------------------------------------------------
    # 类别名的区间（E3a 开放词汇头）
    # ------------------------------------------------------------------

    def class_name_spans(self, inputs: ModelInputs, names: Sequence[str]) -> list[Span]:
        """每个类别名在 inputs（ModelInputs 的 input_ids）里的 token 区间，按 names 的顺序。

        拼法（CLASS_NAME_SEPARATOR）：指令里恰好出现一次 `" " + ", ".join(names)`（前面紧挨一个空格），例如
        "Segment these classes: cat, traffic light, dog."。区间是 " 名字" 这一段的 token（前导空格并进第一个 token，与名字
        在句中的分词相同）。做法：把指令的 token 解码回文本（必须能原样编码回去），按类别名切成若干段分别编码，拼接必须与指令的
        token 逐个相同——名字与相邻文字在分词时合并、区间无法精确定位时报错，不猜。
        """
        self._require_loaded()
        if not isinstance(inputs, ModelInputs):
            raise TypeError(f"inputs 必须是 ModelInputs，收到 {type(inputs).__name__}")
        joined = join_class_names(names)
        ids = inputs.tensors["input_ids"][0].tolist()
        ins = ids[inputs.instruction_span.start: inputs.instruction_span.end]
        text = self._tok.decode(ins, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if self._encode(text) != ins:
            raise TrainingInputError("指令的 token 解码后不能原样编码回去，类别名区间无法定位")
        needle = " " + joined
        if text.count(needle) != 1:
            raise TrainingInputError(f"指令里应恰好出现一次 {needle!r}（类别名的拼法见 CLASS_NAME_SEPARATOR），"
                                     f"实际 {text.count(needle)} 次")
        at = text.index(needle)
        pieces: list[tuple[str, bool]] = [(text[:at], False)]
        for k, name in enumerate(names):
            if k:
                pieces.append((CLASS_NAME_SEPARATOR.rstrip(" "), False))
            pieces.append((" " + name, True))
        pieces.append((text[at + len(needle):], False))
        spans, cat = [], []
        for piece, is_name in pieces:
            enc = self._encode(piece) if piece else []
            if is_name:
                if not enc:
                    raise TrainingInputError(f"类别名 {piece!r} 编码为空")
                s = inputs.instruction_span.start + len(cat)
                spans.append(Span(s, s + len(enc)))
            cat += enc
        if cat != ins:
            raise TrainingInputError("类别名与相邻文字在分词时合并，区间无法精确定位")
        return spans


def first_divergence(a: Sequence[int], b: Sequence[int]) -> int | None:
    """两个 token 序列第一个不同的位置（长度不同时短的那个的末尾算不同）；完全相同为 None。"""
    for k, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return k
    return None if len(a) == len(b) else min(len(a), len(b))


def restricted_consistency(ad: Any, inputs_list: Sequence[ModelInputs], allowed_ids: Sequence[int], *,
                           max_new_tokens: int, stop_ids: Sequence[int], enable_thinking: bool) -> dict[str, Any]:
    """训练与推理的受限 logits 一致性读数（D21-2；简报 §12）。CPU fp32 上两项都应为 1.0；GPU bf16 上写成一致率。

    - decode_vs_generate_batch：generate_restricted 与 generate_batch(allowed_token_ids=allowed) 的 token 序列逐行比较：
      exact_rows（整行相同的比例）、token_agreement（第一个分歧之前的 token 数 / 两者较长者的 token 数，按行求和再比）、
      first_divergence（每行第一个分歧的位置，相同为 None）。
    - teacher_forcing：把受限解码的序列当目标做 training_forward，每个位置 argmax(restricted_logits) 是否等于解码出的 token：
      agreement（相同的位置数 / 总数；教师强制下每个位置独立，不累积）、disagreements（不同位置的 (行, 位置, 解码 token,
      教师强制 argmax, 两者在教师强制 logits 上的差)；bf16 下应当都是近乎平局的位置）。
    """
    allowed = sorted(int(i) for i in allowed_ids)
    gen = ad.generate_batch(list(inputs_list), enable_thinking=enable_thinking, max_new_tokens=max_new_tokens,
                            allowed_token_ids=allowed)
    res = ad.generate_restricted(list(inputs_list), allowed, max_new_tokens=max_new_tokens, stop_ids=list(stop_ids),
                                 enable_thinking=enable_thinking)
    div = [first_divergence(g.token_ids, r.token_ids) for g, r in zip(gen, res)]
    longer = [max(len(g.token_ids), len(r.token_ids)) for g, r in zip(gen, res)]
    agree = [n if d is None else d for d, n in zip(div, longer)]
    with torch.no_grad():
        tf = ad.training_forward(list(inputs_list), [list(r.token_ids) for r in res], enable_thinking=enable_thinking)
        logits = ad.restricted_logits(tf.target_hidden, allowed)
    allowed_t = torch.tensor(allowed, dtype=torch.long)
    am = allowed_t[logits.argmax(dim=-1).cpu()]
    tgt = tf.target_ids.cpu()
    col = {t: k for k, t in enumerate(allowed)}
    bad = (am != tgt).nonzero().flatten().tolist()
    disagreements = []
    for n in bad:
        b = int(tf.target_rows[n])
        lg = logits[n]
        disagreements.append((b, n - tf.target_offsets[b], int(tgt[n]), int(am[n]),
                              float(lg[col[int(am[n])]] - lg[col[int(tgt[n])]])))
    n_tok = int(tgt.numel())
    return {
        "n_rows": len(res),
        "decode_vs_generate_batch": {"exact_rows": sum(d is None for d in div) / len(div),
                                     "token_agreement": sum(agree) / max(1, sum(longer)),
                                     "first_divergence": div},
        "teacher_forcing": {"n_tokens": n_tok, "agreement": 1.0 if n_tok == 0 else 1.0 - len(bad) / n_tok,
                            "disagreements": disagreements},
    }


def join_class_names(names: Sequence[str]) -> str:
    """类别名的拼法：", ".join(names)。名字非空、没有首尾空白、不含分隔符 ","、互不相同。"""
    if isinstance(names, str):
        raise TrainingInputError("names 必须是类别名的序列，不能是单个字符串")
    names = list(names)
    if not names:
        raise TrainingInputError("names 为空")
    for n in names:
        if not isinstance(n, str) or not n or n != n.strip() or CLASS_NAME_SEPARATOR.strip() in n:
            raise TrainingInputError(f"类别名必须是非空、没有首尾空白、不含 {CLASS_NAME_SEPARATOR.strip()!r} 的字符串：{n!r}")
    if len(set(names)) != len(names):
        raise TrainingInputError(f"类别名有重复：{names!r}")
    return CLASS_NAME_SEPARATOR.join(names)


__all__ = [
    "CLASS_NAME_SEPARATOR",
    "HiddenRequest",
    "REGIONS",
    "REGION_IMAGE",
    "REGION_QUERY",
    "TrainingInputError",
    "TrainingModeError",
    "TrainingOutput",
    "TrainingSupport",
    "TokenOrderError",
    "assemble",
    "check_allowed_ids",
    "check_targets",
    "check_training_inputs",
    "chunked_lm_loss",
    "cumsum_positions",
    "first_divergence",
    "greedy_restricted_loop",
    "restricted_consistency",
    "join_class_names",
    "project",
    "restricted_logits",
    "restricted_targets",
    "scatter_rows",
]
