"""Qwen3.5 家族适配器（简报 §8.1；DEV-0003）。

模型事实一律以 `docs/model_notes.md`（RUN-0001 实测）为准；运行时从 tokenizer / config 读取并断言，
不写死 token id，不写死层号（简报 §4 第 6 条）：

- 加载类：`AutoModelForImageTextToText` → `Qwen3_5ForConditionalGeneration`，`attn_implementation="sdpa"`，
  eval 模式；processor 由 `AutoProcessor` 加载（§5.1）。load 一律带注册表里的 revision（D9-2）。
- 图像口径（§5.3、简报 §12、D28-3；src/adapters/image_policy.py）：每个 LLM 视觉 token 对应缩放后图像的 32×32 像素
  （patch 16 × merge 2）。构造时二选一：
  - 固定尺寸 `image_size=(H, W)`（= FixedSize）：先用 PIL 把图缩放到 (H, W)，再以 min_pixels = max_pixels = H·W 调图像处理器，
    处理器不再缩放；全部图的网格都是 (H/32, W/32)，`llm_grid_hw` 只在这个模式下可用。
  - 像素预算 `image_policy=PixelBudget(n)`（DEV-0011，E1 用 n=300）：不预先缩放，以 min_pixels = max_pixels = n·32² 调图像处理器，
    由它的 smart resize 保持长宽比；每张图有自己的网格（ImageTokenInfo.grid_hw，取自 image_grid_thw，并与 smart_resize 的结果核对）。
    位置编码、hidden_states、vision_features、查询 token 都按每张图自己的网格工作，多张图网格不同也可以（测试核对
    position_ids 与 get_rope_index 逐项相同、vision_features 与第 0 层逐位相同）。
  ImageTokenInfo.n_pixels = 缩放后的像素数 = (grid_h·16)·(grid_w·16)。
- M-RoPE（§5.4）：文本 token 的 (T, H, W) 都等于顺序位置；图像 token 的 T = s、H = s + 行、W = s + 列；
  图像之后从 s + max(h, w) 接着编号。position_ids 形状 (4, B, L)，第 0 行是 0..L-1 的文本位置，
  第 1–3 行是 M-RoPE 位置，与 generate 内部的做法相同。本适配器自己构造并显式传入，
  每次都在查询块之前的部分与 `get_rope_index` 逐项比对。
- 聊天模板（§5.2、D9-4）：思考开关直接作为 `apply_chat_template` 的关键字参数传，不用 `chat_template_kwargs`；
  0.8B/2B 与 4B/9B 的默认值相反，所以 generate 必须显式给出。
- 新增 token（D9-3 + 管理员对 DEV-0003 的补充规则）：先复用超出 tokenizer 的预留嵌入行（0.8B–9B 都是 243 行）；
  累计需要的行数超过现有行数时，把嵌入（不共享时连同 lm_head）扩到恰好需要的行数，已有的行逐位不变；
  任何时候都不缩表，不调用 `resize_token_embeddings`。新行一律初始化为基础词表行的精确均值。
  查询占位符固定取预留区的末行（248319），grid token 跳过它（DEV-0020，规则见 ReservedRowTokens）。
- E3 的训练侧接口（DEV-0020）：training_forward、restricted_logits、generate_restricted 等见 src/adapters/training.py，
  token 表、LoRA 与检查点见 src/adapters/checkpoint.py；本模块只提供家族钩子（"训练侧钩子"一节与 ReservedRowTokens）。
- 层号（base.py 的层号约定）：就是 HF `output_hidden_states` 元组的下标。0 = 视觉特征并入之后的嵌入输出，
  i = 第 i 个 decoder 层之后；最后一个下标（num_layers()）是最终归一化之后的值（HF 用 last_hidden_state
  覆盖元组末项）。
- Gated DeltaNet 的实现（DEV-0003-a1，§5.6）：transformers 在 import 时把 GDN 的两个模块级函数绑到 fla 的 Triton
  内核（装了 fla 时，只收 GPU 张量）或 torch 原函数，与张量在哪个设备无关。适配器在 CPU 上的前向与生成一律套
  `torch_gdn()`，显式换成 torch 原函数；GPU 上不动，保持 transformers 的绑定。实际走哪个见 `gdn_implementation()`。
- 坐标（D13-4、§5.5；DEV-0006）：JSON `[{"bbox_2d": [x1, y1, x2, y2], "label": …}, …]`，0–1000，x 在前，按原图宽高
  换算成内部的绝对像素 xyxy，见 parse_boxes / parse_one_box（模块级函数，也是适配器方法；通用 JSON 规则在 src/parsing）。
- 生成（DEV-0011）：generate 与 generate_batch 走同一条路径（src/adapters/batching.py；generate 是一行的批）。批量时左填充，
  M-RoPE 位置由 HF 按 attention_mask 跳过填充计算；生成前逐行核对"非填充 token 的三维位置 = 本适配器自己按规则构造的位置"
  （_check_batch_positions），不一致就报错。
- DEV-0027：generate_batch_budgeted（预算强制收尾，公共实现在 src/adapters/budget.py；本家族的思考段开头 <think> 在助手头里，
  强制收尾补 `</think>` + THINK_SEPARATOR）；pre_projection_features（merger 之前的 patch 网格，重排成行主序）；format_boxes
  （parse_boxes 的逆）；end_of_turn_token_id / stop_token_ids。

处理器的用法：本适配器只用处理器的三个部件——tokenizer、image_processor、chat_template——自己完成图像占位符的
展开与 mm_token_type_ids，做法与 `ProcessorMixin.__call__` 相同。原因：开发机没有 torchvision，
`Qwen3VLProcessor` 的视频部件在那里构造不出来，测试没法走官方处理器。服务器上 load() 仍用官方
`AutoProcessor` 加载，并把它留在 `hf_processor` 上；冒烟脚本逐项比对两者的输出。开发机上的比对对象见 reference_processor()。
"""

from __future__ import annotations

import contextlib
import importlib
import math
import types
from typing import Any, Callable, Iterable, Iterator, Sequence

import numpy as np
import torch
from PIL import Image
from transformers import LogitsProcessor

from src.adapters import batching, registry
from src.adapters.budget import BudgetSupport
from src.adapters.image_policy import QWEN_PIXELS_PER_TOKEN, FixedSize, PixelBudget
from src.adapters.checkpoint import CheckpointSupport, TokenRestoreError, base_fingerprint
from src.adapters.training import TrainingSupport, cumsum_positions, scatter_rows
from src.adapters.base import (
    ORDERS,
    POSITION_MODES,
    PRE_NORM_LAST,
    GenerationResult,
    HiddenStates,
    ImageTokenInfo,
    LayerKey,
    ModelAdapter,
    ModelInputs,
    OptionScores,
    Order,
    PositionMode,
    Span,
    capture_output,
    captured_output,
    check_box_labels,
    check_grid_token_names,
    native_box_ints,
    resolve_layer_keys,
)
from src.executors.geometry import norm1000_to_abs
from src.parsing import BoxParse, OneBoxParse, one_box, parse_json_boxes

# ---------------------------------------------------------------------------
# Qwen3.5 家族事实（字符串来自 docs/model_notes.md §5.1–5.2；id 一律在 attach 时从 tokenizer 查）
# ---------------------------------------------------------------------------

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
END_OF_TEXT = "<|endoftext|>"
THINK_END = "</think>"
IMAGE_PAD = "<|image_pad|>"
VIDEO_PAD = "<|video_pad|>"
VISION_START = "<|vision_start|>"
VISION_END = "<|vision_end|>"
# 停止 token（任务书：取 tokenizer 里的 <|im_end|> 与 <|endoftext|>）。第一个是回合结束 token（end_of_turn_token_id）
STOP_TOKENS: tuple[str, ...] = (IM_END, END_OF_TEXT)
# 预算强制收尾（DEV-0027）：思考结束标记 </think> 后面紧跟的固定分隔。依据：聊天模板关思考时的助手头是 `<think>\n\n</think>\n\n`
# （model_notes §5.2），把历史里的思考段渲染成 `<think>\n…\n</think>\n\n` + 回答；RUN-0005 的开思考输出在 </think> 之后都是 "\n\n"。
# 思考段开头 <think> 在提示里（助手头），不在输出里。tests/test_adapter_budget.py 对照模板核对。
THINK_SEPARATOR = "\n\n"
# 指令里不许出现的结构性 token：出现就会打乱区间或模板结构
_STRUCTURAL_TOKENS: tuple[str, ...] = (IM_START, IM_END, END_OF_TEXT, IMAGE_PAD, VIDEO_PAD, VISION_START, VISION_END)

# E3a 查询 token 的占位符。前向时它的嵌入被 query_embeds 替换（必须显式传，D21-2），不计入 E3b 的 300 个 token 预算，
# 但占用一个嵌入行：固定取预留区的末行（attach 时嵌入的最后一行，Qwen 为 248319；DEV-0020），所以 grid token 的 id
# 与"先注入查询还是先加 grid token"无关。它平时不在 tokenizer 里（build_inputs 直接把这个 id 拼进序列）；只有 grid token
# 的分配越过这一行时，才把 QUERY_PLACEHOLDER 登记进 tokenizer 占住它（tokenizer 的 id 必须连续），之后的 grid token 跳过它。
QUERY_PLACEHOLDER = "<|visint_query|>"
_QUERY_SLOT = "VISINTQUERYSLOT"  # 渲染时查询块的哨兵：分词时在这里切开，拼进 [占位符 id] × n

# 每个 LLM 视觉 token 对应缩放后图像的边长（像素），= patch_size × spatial_merge_size（§5.1）。
# 定义在 image_policy（FixedSize 用它检查 image_size）；attach 时与 config 比对。
PIXELS_PER_LLM_TOKEN = QWEN_PIXELS_PER_TOKEN

# generate / generate_batch 的 gen_kwargs 不能覆盖的参数（适配器自己控制）
_RESERVED_GEN_KWARGS = frozenset({"input_ids", "attention_mask", "pixel_values", "image_grid_thw", "mm_token_type_ids",
                                  "position_ids", "inputs_embeds", "max_new_tokens", "eos_token_id", "pad_token_id"})

# 小型随机模型：从哪个登记模型的真实 config 缩出来（任务书：取固定 revision 的真实 config）
TINY_SOURCE_KEY = "qwen3.5-0.8b"

_SENTINEL = "VISINTINSTRUCTIONSLOT"
_MEAN_CHUNK_BYTES = 1 << 28  # 求基础行均值时每块 float64 的字节上限


class AdapterDeviceError(RuntimeError):
    """入口断言：输入与模型的 device / dtype 不一致（lessons：跨机器差异）。消息里报出两个值。"""


def _canonical_device(device: str | torch.device) -> torch.device:
    """把 "cuda" 这类不带下标的写法规范成张量实际报告的设备（如 cuda:0），便于逐项比较。"""
    return torch.empty(0, device=device).device


def mrope_rows(seq_len: int, segments: Iterable[tuple[int, int, tuple[int, int]]]) -> torch.Tensor:
    """本适配器自己的 M-RoPE 规则（§5.4），与 HF 的 get_rope_index 无关地构造 (3, seq_len) 的 (T, H, W) 位置。

    segments：若干 (start, end, (gh, gw))，每段是一块按行主序排成 gh×gw 网格的 token（图像，或 grid2d 的查询块）。
    文本 token 三维都等于顺序位置；网格块的 T = s、H = s + 行、W = s + 列（s 是块前的下一个位置），块之后从 s + max(gh, gw)
    接着编号。position_ids 与 generate_batch 的逐行核对都用它。
    """
    rows = torch.empty((3, seq_len), dtype=torch.long)
    cur = 0
    i = 0
    for s, e, (gh, gw) in sorted(segments):
        if s < i or e - s != gh * gw or e > seq_len:
            raise RuntimeError(f"段 [{s}, {e}) 与网格 {gh}×{gw}、前一段或序列长度 {seq_len} 不相容")
        rows[:, i:s] = torch.arange(cur, cur + (s - i))
        cur += s - i
        r = torch.arange(gh).repeat_interleave(gw)
        c = torch.arange(gw).repeat(gh)
        rows[0, s:e] = cur
        rows[1, s:e] = cur + r
        rows[2, s:e] = cur + c
        cur += max(gh, gw)
        i = e
    rows[:, i:] = torch.arange(cur, cur + (seq_len - i))
    return rows


def _runs(ids: Sequence[int], token_id: int) -> list[tuple[int, int]]:
    """ids 里 token_id 的连续段，返回半开区间列表。"""
    out: list[tuple[int, int]] = []
    start = None
    for i, t in enumerate(ids):
        if t == token_id and start is None:
            start = i
        elif t != token_id and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(ids)))
    return out


class _AllowedTokensProcessor(LogitsProcessor):
    """把 allowed_token_ids 以外的 logits 置为 -inf（任务书）。

    在 HF generate 里，自定义处理器排在温度 / top-k / top-p 之前，所以采样时也先限制再截断。
    """

    def __init__(self, allowed_ids: Sequence[int], vocab_rows: int):
        ids = [int(i) for i in allowed_ids]
        if len(ids) == 0:
            raise ValueError("allowed_token_ids 为空；不限制请传 None")
        if len(set(ids)) != len(ids):
            raise ValueError("allowed_token_ids 有重复")
        bad = [i for i in ids if not 0 <= i < vocab_rows]
        if bad:
            raise ValueError(f"allowed_token_ids 越界：{bad}，logits 宽度 {vocab_rows}")
        self.allowed = torch.tensor(ids, dtype=torch.long)
        self.vocab_rows = vocab_rows
        self._keep: torch.Tensor | None = None

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        if scores.shape[-1] != self.vocab_rows:
            raise RuntimeError(f"logits 宽度 {scores.shape[-1]} 与 lm_head 行数 {self.vocab_rows} 不一致")
        if self._keep is None or self._keep.device != scores.device:
            keep = torch.zeros(scores.shape[-1], dtype=torch.bool, device=scores.device)
            keep[self.allowed.to(scores.device)] = True
            self._keep = keep
        return scores.masked_fill(~self._keep, float("-inf"))


# ---------------------------------------------------------------------------
# Gated DeltaNet 的实现选择（DEV-0003-a1；docs/model_notes.md §5.6，RUN-0003 的服务器诊断）
#
# transformers 5.17 在 import 下面这个模块时，用 use_kernel_func_from_hub_with_fallback 把两个 GDN 函数包成
# "分派包装"：装了 fla 就调 fla 的 Triton 内核（只收 GPU 张量），没装就调 torch 原函数。这在 import 时一次决定，
# 与张量在哪个设备无关，所以服务器（装了 fla）上 CPU 的小模型会把 CPU 张量送进 Triton。
# 包装用 functools.wraps 包住 torch 原函数：`__wrapped__` 就是 torch 原函数，而包装自己的 __module__/__qualname__
# 抄自 torch 原函数，只看名字分不出实际调用哪个，要看闭包里的 implementation（kernel_implementation）。
# GDN 层的 forward 在调用时才按名字查模块全局，所以临时替换模块属性就能换掉实际调用的函数（§5.6.2 的做法）。
# ---------------------------------------------------------------------------

QWEN35_MODELING = "transformers.models.qwen3_5.modeling_qwen3_5"
# 只换这两个。causal_conv1d_fn / causal_conv1d_update 在两台机器上本来都走 torch（都没装 causal-conv1d），不动。
GDN_FUNCTIONS: tuple[str, ...] = ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule")


class GDNKernelError(RuntimeError):
    """GDN 实现切换失败：取不到 torch 原函数、换完不是预期的函数、认不出绑定，或退出时绑定已被别处改动。"""


def _modeling_module() -> types.ModuleType:
    return importlib.import_module(QWEN35_MODELING)


def _qualified(fn: Any) -> str:
    return f"{getattr(fn, '__module__', '?')}.{getattr(fn, '__qualname__', type(fn).__name__)}"


def _is_torch_reference(fn: Any, module: types.ModuleType, name: str) -> bool:
    """fn 是否正是 modeling_qwen3_5 源码里定义的 torch 原函数 `name`。

    要求：普通 Python 函数、名字相同、全局命名空间就是该模块（hub_kernels 里的分派包装、测试替身都不满足）、
    自己没有再包一层（没有 __wrapped__）。
    """
    return (
        isinstance(fn, types.FunctionType)
        and fn.__name__ == name
        and fn.__qualname__ == name
        and fn.__globals__ is vars(module)
        and not hasattr(fn, "__wrapped__")
    )


def _torch_reference(module: types.ModuleType, name: str) -> Callable[..., Any]:
    """模块级函数 `name` 对应的 torch 原函数：当前绑定已经是它就原样返回（嵌套进入），否则取 `__wrapped__` 并校验。"""
    bound = getattr(module, name, None)
    if bound is None:
        raise GDNKernelError(f"{module.__name__} 里没有 {name}（transformers 版本变了？）")
    if _is_torch_reference(bound, module, name):
        return bound
    try:
        target = bound.__wrapped__
    except AttributeError:
        raise GDNKernelError(f"{name} 当前绑定的 {_qualified(bound)} 没有 __wrapped__，取不到 torch 原函数") from None
    if not _is_torch_reference(target, module, name):
        raise GDNKernelError(
            f"{name}.__wrapped__ 是 {_qualified(target)}，不是 {module.__name__} 里定义的 torch 原函数")
    return target


def kernel_implementation(fn: Callable[..., Any]) -> Callable[..., Any]:
    """模块级绑定 fn 实际调用的函数。

    fn 是 transformers 的分派包装（有 `__wrapped__`）时，返回它闭包里的 implementation（服务器上是 fla 的内核，
    开发机上是 torch 原函数）；否则（例如 torch_gdn() 换上的 torch 原函数）就是 fn 自己。有 `__wrapped__` 却认不出
    implementation 时报错，不猜。
    """
    if not hasattr(fn, "__wrapped__"):
        return fn
    code = getattr(fn, "__code__", None)
    cells = dict(zip(code.co_freevars, fn.__closure__ or ())) if code is not None else {}
    if "implementation" not in cells:
        raise GDNKernelError(f"{_qualified(fn)} 有 __wrapped__，但不是认得的 transformers 分派包装（闭包里没有 implementation）")
    return cells["implementation"].cell_contents


@contextlib.contextmanager
def torch_gdn() -> Iterator[None]:
    """临时把 modeling_qwen3_5 的两个 GDN 模块级函数（GDN_FUNCTIONS）换成 torch 原函数；退出时恢复进入时的绑定，
    异常退出也恢复。

    - 先把两个目标都取到并校验完才动手：任何一个取不到 `__wrapped__`、或它不是预期的 torch 原函数，就报 GDNKernelError，
      一个也不换。换完再核对一遍模块属性。
    - 嵌套：可以。内层进入时绑定已经是 torch 原函数，目标就是它自己；内层退出时恢复成它进入时看到的绑定（仍是 torch），
      外层退出时才恢复 transformers 的原绑定。
    - 线程安全：不是。换的是进程级的模块属性，对本进程所有模型、所有线程同时生效：在这段时间里别的线程跑 GPU 前向
      也会走 torch 实现，两个线程交错进出会恢复错。本项目单进程单线程使用。退出时若发现绑定已不是本层换上的函数
      （交错使用的迹象），照样恢复进入时的绑定，然后抛 GDNKernelError。
    """
    mq = _modeling_module()
    saved = {n: getattr(mq, n, None) for n in GDN_FUNCTIONS}
    targets = {n: _torch_reference(mq, n) for n in GDN_FUNCTIONS}
    try:
        for n, fn in targets.items():
            setattr(mq, n, fn)
        for n, fn in targets.items():
            now = getattr(mq, n)
            if now is not fn or not _is_torch_reference(now, mq, n):
                raise GDNKernelError(f"换完后 {n} 是 {_qualified(now)}，不是预期的 torch 原函数")
        yield
    finally:
        clobbered = [n for n in GDN_FUNCTIONS if getattr(mq, n) is not targets[n]]
        for n, fn in saved.items():
            setattr(mq, n, fn)
        if clobbered:
            raise GDNKernelError(
                f"退出 torch_gdn() 时 {clobbered} 的绑定已不是本层换上的 torch 原函数（多线程或非后进先出的交错使用？）；"
                "已恢复进入时的绑定")


class ReservedRowTokens:
    """Qwen3.5 与 InternVL3.5 共用的新增 token 分配（D9-3、D13-1、DEV-0020）：两家的嵌入都有超出 tokenizer 的预留行。

    - 查询占位符固定取预留区的末行（attach 时嵌入的最后一行），平时不在 tokenizer 里；
    - grid token 从 len(tokenizer) 起连续取 id，跳过占位符行：越过它时把 QUERY_PLACEHOLDER 登记进 tokenizer 占住这一行
      （tokenizer 的 id 必须连续）；所以 grid token 的 id 只取决于已加的 grid token 个数，与是否、何时注入查询无关；
    - 行数不够时扩表（不共享时连同 lm_head），已有行逐位不变，从不缩表。
    需要宿主提供：_tok、_model、_tied、_query_row、_query_row_ready、_encode、_base_mean、_init_rows。
    """

    def _require_query_row(self) -> int:
        if self._query_row is None:
            raise RuntimeError("attach 时嵌入没有超出 tokenizer 的预留行，查询占位符没有固定的行可用")
        return self._query_row

    def _placeholder_in_tokenizer(self) -> bool:
        return QUERY_PLACEHOLDER in self._tok.get_vocab()

    def _allocate_query_row(self, *, init: bool) -> None:
        """分配查询占位符的行（预留区末行，attach 时就在嵌入里）；init 时按均值初始化（只做一次）。"""
        q = self._require_query_row()
        if q >= self._model.get_input_embeddings().weight.shape[0]:
            raise RuntimeError(f"查询占位符的行 {q} 超出嵌入行数")
        if not self._query_row_ready:
            if init:
                self._init_rows([q])
            self._query_row_ready = True

    def _grid_id_plan(self, n: int) -> tuple[list[int], int | None]:
        """接下来 n 个 grid token 的 id 与"越过占位符行"的位置：从 len(tokenizer) 起连续取，遇到预留区末行（占位符还没登记进
        tokenizer 时）跳过它；返回 (ids, cross)，cross 是第一个排在占位符之后的名字的下标（没越过为 None）。"""
        cur = len(self._tok)
        slot = None
        if self._query_row is not None and not self._placeholder_in_tokenizer():
            if cur > self._query_row:
                raise RuntimeError(f"len(tokenizer)={cur} 已越过占位符行 {self._query_row}，但占位符不在 tokenizer 里")
            slot = self._query_row
        ids, cross = [], None
        for k in range(n):
            if cur == slot:
                cross, cur = k, cur + 1
            ids.append(cur)
            cur += 1
        return ids, cross

    def _register_tokens(self, names: list[str], expected: list[int]) -> None:
        if not names:
            return
        n_added = self._tok.add_tokens(names, special_tokens=True)
        ids = [self._tok.convert_tokens_to_ids(n) for n in names]
        if n_added != len(names) or ids != expected:
            raise RuntimeError(f"add_tokens 加了 {n_added} 个，id {ids[:3]}…，期望 {expected[:3]}…")

    def _add_new_tokens(self, names: list[str], *, init: bool = True) -> list[int]:
        """加 grid token 的公共路径：分词器（id 规则见 _grid_id_plan）→ 行数（必要时扩表）→ 行初始化。
        init=False 只给 restore_tokens 用：只登记、扩表，不写均值（随后由保存的行覆盖）。"""
        vocab = self._tok.get_vocab()
        clash = [n for n in names if n in vocab or n in (QUERY_PLACEHOLDER, _QUERY_SLOT)]
        if clash:
            raise ValueError(f"这些 token 已在词表里或是保留名：{clash[:5]}")
        ids, cross = self._grid_id_plan(len(names))
        if cross is None:
            self._register_tokens(names, ids)
        else:
            self._register_tokens(names[:cross], ids[:cross])
            self._register_tokens([QUERY_PLACEHOLDER], [self._query_row])
            self._register_tokens(names[cross:], ids[cross:])
            if self._encode(QUERY_PLACEHOLDER) != [self._query_row]:
                raise RuntimeError(f"{QUERY_PLACEHOLDER!r} 没有分词成单个 token")
        for name, i in zip(names, ids):
            if self._encode(name) != [i]:
                raise RuntimeError(f"{name!r} 没有分词成单个 token")
        if self._encode("".join(names)) != ids:
            raise RuntimeError("新 token 首尾相接时没有逐个分词")
        self._ensure_rows(max(ids) + 1, fill_mean=init)
        rows = self._model.get_input_embeddings().weight.shape[0]
        if max(ids) >= rows or self._model.get_output_embeddings().weight.shape[0] != rows:
            raise RuntimeError(f"新 id 超出嵌入行数 {rows}")
        if init:
            self._init_rows(ids)
        if cross is not None:
            self._allocate_query_row(init=init)
        return ids

    # ---- 检查点钩子（DEV-0020 第二部分；公共实现见 src/adapters/checkpoint.py 的 CheckpointSupport）----

    def _vocab_facts(self) -> dict[str, Any]:
        """attach 时的规模事实：恢复时必须相同（grid token 的 id 由它们决定）。"""
        return {"n_tok_at_attach": self._n_tok_at_attach, "n_rows_at_attach": self._n_rows_at_attach,
                "query_row": self._query_row}

    def _token_state_fresh(self) -> bool:
        return (self._n_grid_tokens == 0 and self._query_token_id is None and not self._query_row_ready
                and len(self._tok) == self._n_tok_at_attach)

    def _token_registry(self) -> dict[str, Any]:
        grid = [{"name": n, "id": i, "surface": n} for n, i in sorted(self._grid_token_ids.items(), key=lambda kv: kv[1])]
        injected = self._query_token_id is not None
        query = {"id": self._query_row if self._query_row_ready else None, "surface": QUERY_PLACEHOLDER,
                 "row_allocated": self._query_row_ready, "in_tokenizer": self._placeholder_in_tokenizer(),
                 "injected": injected, "n": self._query_n, "position_mode": self._current_query_mode()}
        return {"grid_tokens": grid, "query": query, "vocab": self._vocab_facts()}

    def _restore_registration(self, table: Any) -> None:
        """restore_tokens 的登记一步：先按 _grid_id_plan 推算 id，与表不同就报错（什么都不改）；再登记（不初始化）。"""
        if table["vocab"] != self._vocab_facts():
            raise TokenRestoreError(f"规模事实不同（表 {table['vocab']}，本适配器 {self._vocab_facts()}）")
        grid = sorted(table["grid_tokens"], key=lambda g: g["id"])
        names, want = [g["name"] for g in grid], [g["id"] for g in grid]
        if any(g["surface"] != g["name"] for g in grid):
            raise TokenRestoreError("grid token 的 surface 应与名字相同")
        plan, cross = self._grid_id_plan(len(names))
        if plan != want:
            raise TokenRestoreError(f"按本适配器的规则重新登记，grid token 的 id 是 {plan[:3]}…，表里是 {want[:3]}…")
        q = table["query"]
        if q["row_allocated"] and q["id"] != self._query_row:
            raise TokenRestoreError(f"查询占位符的 id {q['id']} 与本适配器的 {self._query_row} 不同")
        if bool(q["in_tokenizer"]) != (cross is not None) or (cross is not None and not q["row_allocated"]):
            raise TokenRestoreError("查询占位符是否在 tokenizer 里与 grid token 的个数不相容")
        if names:
            self._add_new_tokens(names, init=False)
        if q["row_allocated"]:
            self._allocate_query_row(init=False)
        if q["injected"]:
            self._query_token_id = self._require_query_row()
        self._n_grid_tokens = len(names)
        self._grid_token_ids = dict(zip(names, want))

    def _row_tables(self) -> list[tuple[str, Any]]:
        return [("embed", self._model.get_input_embeddings().weight),
                ("lm_head", self._model.get_output_embeddings().weight)]

    def _ensure_rows(self, needed: int, *, fill_mean: bool = True) -> None:
        """行数不够时扩到恰好 needed 行（不共享时 lm_head 同步），已有行逐位不变；够了什么都不做，从不缩表。
        扩出来的行先填基础行均值（fill_mean=False 时填 0：restore_tokens 随后写回保存的行，不做初始化）。"""
        emb = self._model.get_input_embeddings()
        head = self._model.get_output_embeddings()
        n_old = emb.weight.shape[0]
        if needed <= n_old:
            return
        with torch.no_grad():
            pairs = [(emb, "num_embeddings")] + ([] if self._tied else [(head, "out_features")])
            for mod, size_attr in pairs:
                old = mod.weight.data
                new = torch.empty((needed, old.shape[1]), dtype=old.dtype, device=old.device)
                new[:n_old] = old
                new[n_old:] = self._base_mean(old) if fill_mean else 0
                mod.weight.data = new  # 同一个 Parameter 对象：共享的 lm_head 跟着变
                setattr(mod, size_attr, needed)
                if not torch.equal(mod.weight[:n_old], old):
                    raise RuntimeError("扩表后已有的行不再逐位相同")
        if self._tied and head.weight is not emb.weight:
            raise RuntimeError("扩表后 lm_head 与嵌入不再共享")
        if head.weight.shape[0] != needed or emb.weight.shape[0] != needed:
            raise RuntimeError("扩表后嵌入与 lm_head 行数不一致")
        head.out_features = needed  # 共享时 Parameter 已随嵌入变了，这里只同步 nn.Linear 的属性
        self._model.config.get_text_config().vocab_size = needed


class Qwen35Adapter(ReservedRowTokens, CheckpointSupport, TrainingSupport, BudgetSupport, ModelAdapter):
    """Qwen3.5（0.8B/2B/4B/9B）适配器。ModelInputs 是单条样本；generate_batch 左填充拼批（DEV-0011）。vLLM 另开任务。

    设备与图像口径在构造时给定，dtype 在 load 时给定。图像口径二选一：`image_size=(H, W)`（固定尺寸，= FixedSize）或
    `image_policy=PixelBudget(n)`（像素预算；也接受 FixedSize）。进入任何模型算子之前，入口断言检查输入与模型的 device/dtype，
    不一致就抛 AdapterDeviceError 并报出两个值。设备是 CPU 时，前向与生成显式走 GDN 的 torch 实现（gdn_scope）。
    """

    family = registry.FAMILY_QWEN35

    def __init__(self, *, device: str | torch.device, image_size: tuple[int, int] | None = None,
                 image_policy: FixedSize | PixelBudget | None = None):
        if (image_size is None) == (image_policy is None):
            raise ValueError("image_size（固定尺寸）与 image_policy 必须恰好给一个")
        if image_size is not None:
            if not isinstance(image_size, (tuple, list)) or len(image_size) != 2:
                raise ValueError(f"image_size 必须是 (H, W)，收到 {image_size!r}")
            h, w = image_size
            for name, v in (("H", h), ("W", w)):
                if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v <= 0:
                    raise ValueError(f"image_size 的 {name} 必须是正整数，收到 {v!r}")
                if int(v) % PIXELS_PER_LLM_TOKEN != 0:
                    raise ValueError(f"image_size 的 {name}={v} 不是 {PIXELS_PER_LLM_TOKEN} 的倍数")
            image_policy = FixedSize(int(h), int(w))
        if not isinstance(image_policy, (FixedSize, PixelBudget)):
            raise TypeError(f"Qwen3.5 的 image_policy 必须是 FixedSize 或 PixelBudget，收到 {image_policy!r}")
        self._policy: FixedSize | PixelBudget = image_policy
        self._device = _canonical_device(device)
        self._dtype: torch.dtype | None = None
        self._model: Any = None
        self._tok: Any = None
        self._ip: Any = None
        self._template: str | None = None
        self._model_key: str | None = None
        self.hf_processor: Any = None  # 只有 load() 设置：官方 AutoProcessor，供冒烟脚本比对
        self._ids: dict[str, int] = {}
        self._n_base_vocab: int | None = None
        self._tied: bool | None = None
        self._resample: Any = None
        self._header_cache: dict[bool, tuple[int, ...]] = {}
        self._n_grid_tokens = 0
        self._grid_token_ids: dict[str, int] = {}
        self._query_token_id: int | None = None
        self._query_n: int | None = None
        self._query_mode: PositionMode | None = None
        self._n_tok_at_attach: int | None = None  # attach 时的 len(tokenizer)：预留区 = [它, 嵌入行数)
        self._n_rows_at_attach: int | None = None
        self._query_row: int | None = None  # 查询占位符的固定 id = 预留区末行（DEV-0020）
        self._query_row_ready = False  # 这一行已分配（注入查询或 grid token 越过它时；add 路径同时按均值初始化）

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        self._require_loaded()
        return self._dtype

    @property
    def image_policy(self) -> FixedSize | PixelBudget:
        """构造时给定的图像口径（FixedSize 或 PixelBudget）。"""
        return self._policy

    @property
    def fixed_size(self) -> bool:
        return isinstance(self._policy, FixedSize)

    @property
    def image_size(self) -> tuple[int, int]:
        """固定尺寸 (H, W)。只在固定尺寸模式下可用；像素预算模式下每张图的尺寸不同，报错。"""
        self._require_fixed("image_size")
        return (self._policy.h, self._policy.w)

    @property
    def llm_grid_hw(self) -> tuple[int, int]:
        """每张图送进 LLM 的视觉 token 网格 (H/32, W/32)。只在固定尺寸模式下可用：像素预算模式下每张图有自己的网格，
        请读 ModelInputs.images[k].grid_hw。"""
        self._require_fixed("llm_grid_hw")
        return (self._policy.h // PIXELS_PER_LLM_TOKEN, self._policy.w // PIXELS_PER_LLM_TOKEN)

    def _require_fixed(self, what: str) -> None:
        if not isinstance(self._policy, FixedSize):
            raise ValueError(f"{what} 只在固定尺寸模式下有定义；当前是 {self._policy!r}，每张图的网格见 ImageTokenInfo.grid_hw")

    @property
    def chat_template(self) -> str:
        self._require_loaded()
        return self._template

    @property
    def model(self) -> Any:
        self._require_loaded()
        return self._model

    @property
    def tokenizer(self) -> Any:
        self._require_loaded()
        return self._tok

    @property
    def image_processor(self) -> Any:
        self._require_loaded()
        return self._ip

    @property
    def model_key(self) -> str | None:
        return self._model_key

    @property
    def tied_word_embeddings(self) -> bool:
        self._require_loaded()
        return self._tied

    @property
    def n_base_vocab(self) -> int:
        """基础词表大小（不含 added tokens），新行初始化取这些行的均值（D9-3）。"""
        self._require_loaded()
        return self._n_base_vocab

    @property
    def grid_token_ids(self) -> dict[str, int]:
        return dict(self._grid_token_ids)

    @property
    def n_grid_tokens(self) -> int:
        """累计已加的 grid token 个数（计入 300 的预算；查询占位符不计）。"""
        return self._n_grid_tokens

    @property
    def query_token_id(self) -> int | None:
        return self._query_token_id

    def token_id(self, token: str) -> int:
        """词表里任意一个 token 的 id（DEV-0011；RUN-0005 发现原来只认 attach 时缓存的那几个）。

        attach 时查好的特殊 token（IM_END、THINK_END 等）直接返回；其他按 token 的字面（词表里的写法，如 `<think>`、`Ġthe`）
        经 tokenizer 查，要求 id → token 能原样转回，查不到报 KeyError。注意它查的是词表项，不是"这段文字分词后的 id"。
        """
        self._require_loaded()
        if token in self._ids:
            return self._ids[token]
        if not isinstance(token, str) or not token:
            raise KeyError(f"token 必须是非空字符串，收到 {token!r}")
        i = self._tok.convert_tokens_to_ids(token)
        if isinstance(i, bool) or not isinstance(i, int) or i < 0 or self._tok.convert_ids_to_tokens(i) != token:
            raise KeyError(f"{token!r} 不在词表里")
        return i

    # ------------------------------------------------------------------
    # 加载
    # ------------------------------------------------------------------

    def load(self, model_id: str, dtype: Any) -> None:
        """按注册键加载（带固定 revision）。dtype 由调用方给出，如 torch.bfloat16。"""
        entry = registry.get_model(model_id)
        if entry.family != registry.FAMILY_QWEN35:
            raise ValueError(f"{model_id} 属于 {entry.family} 家族，不是 {registry.FAMILY_QWEN35}")
        if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
            raise TypeError(f"dtype 必须是浮点 torch.dtype，收到 {dtype!r}")
        from transformers import AutoModelForImageTextToText, AutoProcessor

        processor = AutoProcessor.from_pretrained(entry.repo, revision=entry.revision)
        model = AutoModelForImageTextToText.from_pretrained(
            entry.repo,
            revision=entry.revision,
            dtype=dtype,
            attn_implementation="sdpa",
            device_map=str(self._device),
        )
        self.attach(
            model,
            tokenizer=processor.tokenizer,
            image_processor=processor.image_processor,
            chat_template=processor.chat_template,
            model_key=entry.key,
            expected_dtype=dtype,
        )
        self.hf_processor = processor

    def attach(
        self,
        model: Any,
        *,
        tokenizer: Any,
        image_processor: Any,
        chat_template: str,
        model_key: str | None = None,
        expected_dtype: torch.dtype | None = None,
    ) -> None:
        """挂上已经构造好的模型与处理器部件（load 与小型随机模型共用这条路径）。

        模型若不在构造时给定的设备上，会被整体搬过去；dtype 不做转换，取模型参数的 dtype，
        要求全部浮点参数同一个 dtype（buffer 如 RoPE 的 inv_freq 按 HF 的做法保持 float32，不检查）。
        """
        if self._model is not None:
            raise RuntimeError("这个适配器已经挂过模型；新模型请新建适配器")
        if type(model).__name__ != "Qwen3_5ForConditionalGeneration":
            raise TypeError(f"期望 Qwen3_5ForConditionalGeneration，收到 {type(model).__name__}")
        if not isinstance(chat_template, str) or not chat_template:
            raise ValueError("chat_template 为空")
        cfg = model.config
        tc, vc = cfg.text_config, cfg.vision_config
        attn = getattr(cfg, "_attn_implementation", None)
        if attn != "sdpa":
            raise ValueError(f"attn_implementation 必须是 sdpa（任务书），收到 {attn!r}")

        # 视觉几何：config、图像处理器与 PIXELS_PER_LLM_TOKEN 三方一致
        unit = vc.patch_size * vc.spatial_merge_size
        if unit != PIXELS_PER_LLM_TOKEN:
            raise ValueError(f"config 的 patch_size×spatial_merge_size={unit}，与 {PIXELS_PER_LLM_TOKEN} 不一致")
        for ip_attr, cfg_val in (
            ("patch_size", vc.patch_size),
            ("merge_size", vc.spatial_merge_size),
            ("temporal_patch_size", vc.temporal_patch_size),
        ):
            if getattr(image_processor, ip_attr) != cfg_val:
                raise ValueError(f"图像处理器 {ip_attr}={getattr(image_processor, ip_attr)} 与 config 的 {cfg_val} 不一致")

        # 特殊 token：每个都必须分词成单个 token，并与 config 登记的 id 一致
        ids: dict[str, int] = {}
        for s in (IM_START, IM_END, END_OF_TEXT, THINK_END, IMAGE_PAD, VIDEO_PAD, VISION_START, VISION_END):
            enc = tokenizer(s, add_special_tokens=False)["input_ids"]
            if len(enc) != 1 or tokenizer.convert_tokens_to_ids(s) != enc[0]:
                raise ValueError(f"{s!r} 没有分词成单个已知 token：{enc}")
            ids[s] = enc[0]
        for s, cfg_name in (
            (IMAGE_PAD, "image_token_id"),
            (VIDEO_PAD, "video_token_id"),
            (VISION_START, "vision_start_token_id"),
            (VISION_END, "vision_end_token_id"),
        ):
            if getattr(cfg, cfg_name) != ids[s]:
                raise ValueError(f"config.{cfg_name}={getattr(cfg, cfg_name)} 与 tokenizer 的 {s} id {ids[s]} 不一致")

        # 嵌入行与词表
        emb = model.get_input_embeddings()
        head = model.get_output_embeddings()
        n_rows = emb.weight.shape[0]
        if head.weight.shape[0] != n_rows:
            raise ValueError(f"lm_head 行数 {head.weight.shape[0]} 与嵌入行数 {n_rows} 不一致")
        if tc.vocab_size != n_rows:
            raise ValueError(f"text_config.vocab_size={tc.vocab_size} 与嵌入行数 {n_rows} 不一致")
        if len(tokenizer) > n_rows:
            raise ValueError(f"len(tokenizer)={len(tokenizer)} 超过嵌入行数 {n_rows}")
        tied = head.weight is emb.weight
        if tied != (head.weight.data_ptr() == emb.weight.data_ptr()):
            raise ValueError("lm_head 与嵌入的共享状态不一致（对象与存储判断矛盾）")
        cfg_tied = bool(getattr(cfg, "tie_word_embeddings", False))
        if tied != cfg_tied:
            raise ValueError(f"config.tie_word_embeddings={cfg_tied}，但 lm_head 与嵌入{'共享' if tied else '不共享'}")
        n_base = int(tokenizer.vocab_size)
        if not 0 < n_base <= len(tokenizer):
            raise ValueError(f"tokenizer.vocab_size={n_base} 不在 (0, len(tokenizer)={len(tokenizer)}]")

        # dtype：全部浮点参数同一个 dtype
        dtypes = {p.dtype for p in model.parameters() if p.is_floating_point()}
        if len(dtypes) != 1:
            raise ValueError(f"模型浮点参数的 dtype 不唯一：{sorted(map(str, dtypes))}")
        (dtype,) = dtypes
        if expected_dtype is not None and dtype != expected_dtype:
            raise AdapterDeviceError(f"加载后参数 dtype 是 {dtype}，调用方要求 {expected_dtype}")

        # 设备：整体搬到构造时给定的设备，再逐个参数 / buffer 核对
        if any(t.device != self._device for t in _params_and_buffers(model)):
            model.to(self._device)
        bad = [(n, t.device) for n, t in _named_params_and_buffers(model) if t.device != self._device]
        if bad:
            raise AdapterDeviceError(f"模型张量 {bad[0][0]} 在 {bad[0][1]}，构造时给定的设备是 {self._device}")
        model.eval()

        self._model = model
        self._tok = tokenizer
        self._ip = image_processor
        self._template = chat_template
        self._model_key = model_key
        self._dtype = dtype
        self._ids = ids
        self._n_base_vocab = n_base
        self._tied = tied
        self._resample = Image.Resampling(int(image_processor.resample))
        self._n_tok_at_attach = len(tokenizer)
        self._n_rows_at_attach = int(n_rows)
        self._query_row = int(n_rows) - 1 if n_rows > len(tokenizer) else None
        self._base_fingerprint = base_fingerprint(model)  # 检查点核对 base 用（DEV-0020）

        # 模板自检：user 轮渲染与思考开关无关，以 <|im_end|>\n 结束；两种 assistant 头都能拆出来。失败就撤销挂载。
        try:
            probe = [{"type": "text", "text": "x"}]
            plain = self._render(probe)
            for flag in (False, True):
                if self._render(probe, enable_thinking=flag) != plain:
                    raise ValueError("user 轮的渲染随思考开关变化，与 model_notes §5.2 不符")
            if not plain.endswith(IM_END + "\n"):
                raise ValueError(f"user 轮渲染没有以 {IM_END!r}+换行 结束：{plain[-40:]!r}")
            for flag in (False, True):
                self.assistant_header_ids(flag)
        except Exception:
            self._model = None
            self._header_cache.clear()
            raise

    # ------------------------------------------------------------------
    # 输入
    # ------------------------------------------------------------------

    def to_rgb(self, image: Any) -> Image.Image:
        """把一张图转成 RGB 的 PIL 图，不缩放。"""
        if isinstance(image, Image.Image):
            return image if image.mode == "RGB" else image.convert("RGB")
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(f"numpy 图必须是 uint8 的 (H, W, 3)，收到 dtype={image.dtype} shape={image.shape}")
            return Image.fromarray(image)
        raise TypeError(f"图像必须是 PIL.Image 或 numpy 数组，收到 {type(image).__name__}")

    def resize_image(self, image: Any) -> Image.Image:
        """固定尺寸模式：把一张图转成 RGB 并缩放到 (H, W)。缩放用 PIL，重采样方式取图像处理器的 resample（执行器白名单）。
        像素预算模式下不预先缩放（由处理器的 smart resize 做），这里报错。"""
        self._require_loaded()
        self._require_fixed("resize_image")
        pil = self.to_rgb(image)
        h, w = self.image_size
        if pil.size != (w, h):
            pil = pil.resize((w, h), resample=self._resample)
        return pil

    def processor_pixels(self) -> int:
        """调图像处理器时显式传入的 min_pixels = max_pixels：固定尺寸 H·W；像素预算 n·32²。"""
        if isinstance(self._policy, FixedSize):
            return self._policy.h * self._policy.w
        return self._policy.pixels

    def expected_grid(self, image_hw: tuple[int, int]) -> tuple[int, int]:
        """一张原图（高, 宽）在当前口径下应得的 patch 网格 (grid_h, grid_w)：固定尺寸是 (H/16, W/16)；像素预算按图像处理器
        模块自己的 smart_resize（factor = patch × merge，min_pixels = max_pixels = 预算）算。build_inputs 用它核对处理器的输出。"""
        self._require_loaded()
        p = int(self._ip.patch_size)
        if isinstance(self._policy, FixedSize):
            return self._policy.h // p, self._policy.w // p
        # 优先取图像处理器所在模块自己的 smart_resize（服务器：torchvision 后端；开发机：PIL 后端；5.17 两份逐字相同），
        # 没有时退回 PIL 后端模块里的那份
        smart_resize = getattr(importlib.import_module(type(self._ip).__module__), "smart_resize", None)
        if smart_resize is None:
            from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import smart_resize
        n = self._policy.pixels
        rh, rw = smart_resize(int(image_hw[0]), int(image_hw[1]), factor=p * int(self._ip.merge_size),
                              min_pixels=n, max_pixels=n)
        return rh // p, rw // p

    def build_inputs(self, images: Sequence[Any], instruction: str, order: Order) -> ModelInputs:
        """组装单条输入。区间都是 input_ids 里的 token 下标（半开）。

        user 内容的顺序：instruction_first → [指令, 图1, 图2, …]；image_first → [图1, …, 指令]。
        已调用 inject_query_tokens 时，查询 token 放在 user 内容最后一项之后、<|im_end|> 之前。
        只渲染到 user 轮结束（<|im_end|>\\n），不含 assistant 头；assistant 头由 generate 按思考开关追加。
        图像按构造时的口径处理（模块说明）：固定尺寸先用 PIL 缩放；像素预算不预先缩放。两种都以 min_pixels = max_pixels
        显式调图像处理器，每张图的网格取自 image_grid_thw，并与 expected_grid（固定尺寸 / smart_resize）逐张核对。
        """
        self._require_loaded()
        if order not in ORDERS:
            raise ValueError(f"order 必须是 {ORDERS} 之一，收到 {order!r}")
        if not isinstance(instruction, str):
            raise TypeError(f"instruction 必须是 str，收到 {type(instruction).__name__}")
        self._check_instruction(instruction)
        if isinstance(images, (Image.Image, np.ndarray)):
            raise TypeError("images 必须是图像的序列（单张图也要放进列表）")
        if isinstance(self._policy, FixedSize):
            pil = [self.resize_image(im) for im in images]
        else:
            pil = [self.to_rgb(im) for im in images]
        n_img = len(pil)

        # 1. 渲染：指令槽先放哨兵，确认模板对指令原样输出（Qwen 模板会 trim 整段内容）
        content = self._content(n_img, instruction, order)
        text = self._render(content)
        sent_text = self._render(self._content(n_img, _SENTINEL, order))
        if sent_text.count(_SENTINEL) != 1:
            raise RuntimeError("哨兵在渲染结果里出现的次数不是 1")
        cut = sent_text.index(_SENTINEL)
        prefix, suffix = sent_text[:cut], sent_text[cut + len(_SENTINEL) :]
        if text != prefix + instruction + suffix:
            raise ValueError("聊天模板改写了指令文本（例如首尾空白被 trim），指令区间无法精确定位；请去掉首尾空白")
        if not text.endswith(IM_END + "\n"):
            raise RuntimeError(f"渲染没有以 {IM_END!r}+换行 结束")

        # 2. 图像：min_pixels = max_pixels 显式传入；逐张核对网格（固定尺寸：处理器不再缩放；像素预算：= smart_resize）
        tensors: dict[str, torch.Tensor] = {}
        n_tok_per_image: list[int] = []
        llm_grids: list[tuple[int, int]] = []
        n_pixels: list[int] = []
        if n_img:
            px = self.processor_pixels()
            enc = self._ip(images=pil, min_pixels=px, max_pixels=px, return_tensors="pt")
            grid = enc["image_grid_thw"]
            p, merge = int(self._ip.patch_size), int(self._ip.merge_size)
            want = torch.tensor([[1, *self.expected_grid((im.height, im.width))] for im in pil], dtype=grid.dtype)
            if not torch.equal(grid, want):
                raise RuntimeError(f"image_grid_thw={grid.tolist()}，期望 {want.tolist()}（{self._policy!r}）：处理器没有按给定的像素数缩放")
            for _, gh_p, gw_p in grid.tolist():
                if gh_p % merge or gw_p % merge:
                    raise RuntimeError(f"patch 网格 {gh_p}×{gw_p} 不是 merge_size={merge} 的倍数")
                llm_grids.append((gh_p // merge, gw_p // merge))
                n_pixels.append(gh_p * p * gw_p * p)
            n_tok_per_image = [gh * gw for gh, gw in llm_grids]
            tensors["pixel_values"] = enc["pixel_values"].to(device=self._device, dtype=self._dtype)
            tensors["image_grid_thw"] = grid.to(self._device)

        # 3. 展开图像占位符并分词；指令两侧分别分词，拼接必须与整段分词逐个相同。
        #    查询块（DEV-0020）：渲染时是哨兵 _QUERY_SLOT，在这里切开，拼进 [占位符 id] × n（与把占位符当作特殊 token 分词相同：
        #    特殊 token 两侧的文字各自分词）。
        n_pre = prefix.count(IMAGE_PAD)
        if n_pre + suffix.count(IMAGE_PAD) != n_img:
            raise RuntimeError("模板里的图像占位符个数与图数不一致")
        pre_exp = self._expand(prefix, n_tok_per_image[:n_pre])
        post_exp = self._expand(suffix, n_tok_per_image[n_pre:])
        post_head, post_tail = post_exp, ""
        if self._query_n is not None:
            if _QUERY_SLOT in pre_exp or post_exp.count(_QUERY_SLOT) != 1:
                raise RuntimeError("查询块的哨兵没有恰好出现在指令之后一次")
            post_head, post_tail = post_exp.split(_QUERY_SLOT)
        ids_pre, ids_ins, ids_head = self._encode(pre_exp), self._encode(instruction), self._encode(post_head)
        if ids_pre + ids_ins + ids_head != self._encode(pre_exp + instruction + post_head):
            raise ValueError("指令与相邻模板文本在分词时合并，指令区间无法精确定位")
        ids_post = ids_head
        if self._query_n is not None:
            ids_post = ids_head + [self._query_token_id] * self._query_n + self._encode(post_tail)
        ids = ids_pre + ids_ins + ids_post
        instruction_span = Span(len(ids_pre), len(ids_pre) + len(ids_ins))

        # 4. 图像区间：<|image_pad|> 的连续段，个数、长度、前后标记逐项核对
        runs = _runs(ids, self._ids[IMAGE_PAD])
        if [e - s for s, e in runs] != n_tok_per_image:
            raise RuntimeError(f"图像 token 段 {runs} 与每张图的 token 数 {n_tok_per_image} 不一致")
        infos = []
        for (s, e), (gh, gw), npx in zip(runs, llm_grids, n_pixels):
            if ids[s - 1] != self._ids[VISION_START] or ids[e] != self._ids[VISION_END]:
                raise RuntimeError(f"图像段 [{s}, {e}) 前后不是 vision_start / vision_end")
            if e - s != gh * gw:
                raise RuntimeError(f"图像段长度 {e - s} 与网格 {gh}×{gw} 不一致")
            infos.append(ImageTokenInfo(Span(s, e), (gh, gw), n_pixels=npx))

        # 5. 查询区间
        query_span = None
        if self._query_n is not None:
            qruns = _runs(ids, self._query_token_id)
            if len(qruns) != 1 or qruns[0][1] - qruns[0][0] != self._query_n:
                raise RuntimeError(f"查询 token 段 {qruns} 与登记的 {self._query_n} 个不一致")
            qs, qe = qruns[0]
            if ids[qe] != self._ids[IM_END] or qe + 2 != len(ids):
                raise RuntimeError("查询 token 没有紧挨在 <|im_end|>\\n 之前")
            if qs < instruction_span.end or any(qs < info.span.end for info in infos):
                raise RuntimeError("查询 token 不在 user 内容的最后")
            query_span = Span(qs, qe)

        input_ids = torch.tensor([ids], dtype=torch.long, device=self._device)
        tensors["input_ids"] = input_ids
        tensors["attention_mask"] = torch.ones_like(input_ids)
        # 与 ProcessorMixin.create_mm_token_type_ids 相同：图像 token 记 1，其余 0（int64）
        tensors["mm_token_type_ids"] = (input_ids == self._ids[IMAGE_PAD]).to(torch.long)
        return ModelInputs(
            tensors=tensors,
            order=order,
            images=tuple(infos),
            instruction_span=instruction_span,
            query_span=query_span,
            query_position_mode=self._query_mode if query_span is not None else None,
        )

    # ------------------------------------------------------------------
    # 位置
    # ------------------------------------------------------------------

    def position_ids(self, inputs: ModelInputs) -> torch.Tensor:
        """自己构造的 (4, 1, L) position_ids：第 0 行 0..L-1，第 1–3 行 M-RoPE 的 (T, H, W)。

        文本 token 三维都等于顺序位置；图像 token T = s、H = s + 行、W = s + 列，之后从 s + max(h, w) 接着编号。
        查询块：grid2d 时 q×q 网格像图像一样编码——块前最后一个 token 的位置是 p，t0 = p + 1，
        查询 (r, c) 的 (T, H, W) = (t0, t0 + r, t0 + c)，块之后从 t0 + q 接着编号；seq1d 时按文本顺序编号。
        查询块之前的部分（seq1d 时整条）与 get_rope_index 逐项比对，不一致就报错。
        """
        self._require_loaded()
        t = inputs.tensors
        ids = t["input_ids"]
        if ids.dim() != 2 or ids.shape[0] != 1:
            raise ValueError(f"只支持单条样本，input_ids 形状 {tuple(ids.shape)}")
        seq_len = ids.shape[1]
        segs: list[tuple[int, int, tuple[int, int]]] = [(i.span.start, i.span.end, i.grid_hw) for i in inputs.images]
        qs = None
        if inputs.query_span is not None:
            qs, qe = inputs.query_span.start, inputs.query_span.end
            if any(i.span.start <= qs - 1 < i.span.end for i in inputs.images):
                raise RuntimeError("查询块前一个 token 是图像 token，t0 = p + 1 的规则无定义")
            if inputs.query_position_mode == "grid2d":
                q = math.isqrt(qe - qs)
                if q * q != qe - qs:
                    raise ValueError(f"grid2d 的查询个数 {qe - qs} 不是完全平方数")
                segs.append((qs, qe, (q, q)))
        rows = mrope_rows(seq_len, segs)
        pos = torch.cat([torch.arange(seq_len)[None], rows], dim=0).unsqueeze(1)

        # 与模型自己的 get_rope_index 比对（查询 token 在 mm_token_type_ids 里记为文本）
        ref, _ = self._model.model.get_rope_index(
            ids,
            mm_token_type_ids=t["mm_token_type_ids"],
            image_grid_thw=t.get("image_grid_thw"),
            attention_mask=t["attention_mask"],
        )
        check_end = seq_len if (qs is None or inputs.query_position_mode == "seq1d") else qs
        if not torch.equal(pos[1:, :, :check_end], ref[:, :, :check_end].cpu()):
            raise RuntimeError("自构 M-RoPE 位置与 get_rope_index 不一致")
        return pos.to(ids.device)

    # ------------------------------------------------------------------
    # 前向：隐藏状态与视觉特征
    # ------------------------------------------------------------------

    def hidden_states(
        self, inputs: ModelInputs, layers: Sequence[LayerKey], *, query_embeds: torch.Tensor | None = None
    ) -> HiddenStates:
        """指定层在图像 token 与查询 token 位置的隐藏状态（层号约定见 base.py）。

        layers 里可以有 PRE_NORM_LAST（base.py）：最后一个 decoder 层的原始输出，在 _last_decoder_layer() 上挂前向钩子取得。
        不强制 no_grad：沿用调用方的梯度模式（E3a 训练要让梯度流回 query_embeds 与 LoRA）；
        只做推理时请在 torch.no_grad() 里调用。
        """
        keys, layer_list, want_pre = resolve_layer_keys(layers, self._check_layers)
        with capture_output(self._last_decoder_layer()) if want_pre else contextlib.nullcontext({}) as cap:
            hs = self._run(inputs, query_embeds, where="hidden_states")
        rows = {l: hs[l][0] for l in layer_list}
        if want_pre:
            rows[PRE_NORM_LAST] = captured_output(cap, what="hidden_states")[0]
        img_idx = self._image_index(inputs)
        image = {k: rows[k].index_select(0, img_idx) for k in keys}
        query = None
        if inputs.query_span is not None:
            q_idx = torch.arange(inputs.query_span.start, inputs.query_span.end, device=img_idx.device)
            query = {k: rows[k].index_select(0, q_idx) for k in keys}
        return HiddenStates(layers=keys, image=image, query=query)

    def sequence_hidden_states(
        self, inputs: ModelInputs, layers: Sequence[int], *, query_embeds: torch.Tensor | None = None
    ) -> dict[int, torch.Tensor]:
        """诊断用：指定层在整条序列上的隐藏状态，每层形状 [L, hidden]。与 hidden_states 走同一条前向路径。"""
        layer_list = self._check_layers(layers)
        hs = self._run(inputs, query_embeds, where="sequence_hidden_states")
        return {l: hs[l][0] for l in layer_list}

    def vision_features(self, inputs: ModelInputs) -> torch.Tensor:
        """进入 LLM 之前、经过 patch merger 投影之后的视觉 token，[n_image_tokens, hidden]，多图按顺序拼接。

        就是前向时并入嵌入序列的那组向量，所以与 hidden_states(inputs, [0]) 在图像位置上逐位相同（测试与冒烟都查）。
        """
        self._check_entry(inputs, None, where="vision_features")
        if not inputs.images:
            return torch.empty((0, self.hidden_size()), dtype=self._dtype, device=self._device)
        with self.gdn_scope():
            return self._image_features(inputs)

    def pre_projection_features(self, inputs: ModelInputs) -> tuple[torch.Tensor, tuple[int, int]]:
        """patch merger 之前的视觉塔输出（D59-8，E2 的 pre_proj 参考线）：([gh·gw, D_v], (gh, gw))，gh × gw 是 patch 网格
        （= LLM 网格的 merge 倍，每个 patch 16 px），行主序。只接受单图。

        取的位置：`model.visual` 的 last_hidden_state（Qwen3_5VisionModel.forward 在 merger 之前的 hidden_states），调用方式与
        get_image_features 相同（pixel_values 转成视觉塔的 dtype）。视觉塔里的 patch 顺序是 merge×merge 窗口序（每 merge² 个连续
        patch 是同一个窗口）；重排成行主序用模型自己的元数据：视觉 RoPE 的 (行, 列)（vision_patch_row_major_order）。
        merger（LayerNorm → 每 merge² 个拼接 → MLP）作用在窗口序上，对本结果按 vision_patch_row_major_order 逆排再过 merger 就是
        vision_features（测试逐位核对）。"""
        where = "pre_projection_features"
        self._check_entry(inputs, None, where=where)
        if len(inputs.images) != 1:
            raise ValueError(f"{where} 只接受单图，收到 {len(inputs.images)} 张")
        t = inputs.tensors
        grid = t["image_grid_thw"]
        visual = self._model.model.visual
        order = vision_patch_row_major_order(grid, int(visual.spatial_merge_size))
        with self.gdn_scope():
            out = visual(t["pixel_values"].type(visual.dtype), grid_thw=grid, return_dict=True)
        x = out.last_hidden_state
        _, gh, gw = (int(v) for v in grid[0].tolist())
        if tuple(x.shape[:1]) != (gh * gw,) or x.dim() != 2:
            raise RuntimeError(f"视觉塔输出形状 {tuple(x.shape)}，期望 [{gh * gw}, D_v]")
        merge = int(visual.spatial_merge_size)
        if (gh // merge, gw // merge) != inputs.images[0].grid_hw:
            raise RuntimeError(f"patch 网格 {gh}×{gw} 与 LLM 网格 {inputs.images[0].grid_hw} 不是 {merge} 倍的关系")
        return x.index_select(0, order.to(x.device)), (gh, gw)

    # ------------------------------------------------------------------
    # 生成
    # ------------------------------------------------------------------

    def assistant_header_ids(self, enable_thinking: bool) -> tuple[int, ...]:
        """assistant 头的 token：apply_chat_template(add_generation_prompt=True, enable_thinking=X) 比不加时多出的部分。"""
        self._require_loaded()
        if not isinstance(enable_thinking, bool):
            raise TypeError(f"enable_thinking 必须是 bool，收到 {enable_thinking!r}")
        if enable_thinking not in self._header_cache:
            probe = [{"type": "text", "text": "x"}]
            plain = self._render(probe)
            full = self._render(probe, add_generation_prompt=True, enable_thinking=enable_thinking)
            if not full.startswith(plain) or not full[len(plain) :].startswith(IM_START):
                raise RuntimeError("加了生成提示的渲染不是 user 轮 + <|im_start|>… 的形式")
            header = self._encode(full[len(plain) :])
            if self._encode(plain) + header != self._encode(full):
                raise RuntimeError("assistant 头单独分词与整段分词不一致")
            self._header_cache[enable_thinking] = tuple(header)
        return self._header_cache[enable_thinking]

    def generate(
        self,
        inputs: ModelInputs,
        *,
        enable_thinking: bool,
        allowed_token_ids: Sequence[int] | None = None,
        max_new_tokens: int,
        **gen_kwargs: Any,
    ) -> GenerationResult:
        """单条样本生成。默认 greedy；停止 token 是 <|im_end|> 与 <|endoftext|>（id 从 tokenizer 查）。

        allowed_token_ids 非空时，每一步把其他 id 的 logits 置为 -inf；这时若希望能停下，调用方要把停止 token
        放进允许集合。返回的 token_ids 含生成出的停止 token；n_prompt_tokens 含 assistant 头。
        与 generate_batch 走同一条路径（一行的批，没有填充）；stopped、n_image_pixels 等字段见 base.GenerationResult。
        """
        (res,) = self._generate_rows([inputs], enable_thinking=enable_thinking, allowed_token_ids=allowed_token_ids,
                                     max_new_tokens=max_new_tokens, gen_kwargs=gen_kwargs, where="generate")
        return res

    def generate_batch(
        self,
        inputs_list: Sequence[ModelInputs],
        *,
        enable_thinking: bool,
        max_new_tokens: int,
        allowed_token_ids: Sequence[int] | None = None,
        **gen_kwargs: Any,
    ) -> list[GenerationResult]:
        """批量生成（约定见 base.ModelAdapter.generate_batch）。左填充用 <|endoftext|>（也是 HF 给已结束的行补的 pad）；
        mm_token_type_ids 在填充位置为 0；各行的 pixel_values / image_grid_thw 按行的顺序拼接，每行可以有不同个数、不同网格的图。
        M-RoPE：HF 按 attention_mask 跳过填充算 (T, H, W)；生成前逐行核对它与本适配器自己的规则（mrope_rows）逐个相等。
        """
        return self._generate_rows(inputs_list, enable_thinking=enable_thinking, allowed_token_ids=allowed_token_ids,
                                   max_new_tokens=max_new_tokens, gen_kwargs=gen_kwargs, where="generate_batch")

    def _prompt_row(self, inputs: ModelInputs, enable_thinking: bool) -> batching.PromptRow:
        """一行送进模型的提示：user 轮（inputs）+ assistant 头；mm_token_type_ids 在头上为 0。"""
        t = inputs.tensors
        if t["input_ids"].dim() != 2 or t["input_ids"].shape[0] != 1:
            raise ValueError(f"只支持单条样本的 ModelInputs，input_ids 形状 {tuple(t['input_ids'].shape)}")
        header = list(self.assistant_header_ids(enable_thinking))
        return batching.PromptRow(
            ids=tuple(t["input_ids"][0].tolist() + header),
            n_visual_tokens=sum(len(i.span) for i in inputs.images),
            n_image_pixels=batching.image_pixels(inputs),
            mm_token_type_ids=tuple(t["mm_token_type_ids"][0].tolist() + [0] * len(header)),
        )

    def _generate_rows(self, inputs_list: Sequence[ModelInputs], *, enable_thinking: bool,
                       allowed_token_ids: Sequence[int] | None, max_new_tokens: int, gen_kwargs: dict[str, Any],
                       where: str) -> list[GenerationResult]:
        self._require_loaded()
        batching.check_generate_args(enable_thinking, max_new_tokens)
        self._require_eval(where)
        items = batching.check_inputs_list(inputs_list, where=where)
        batching.check_reserved(gen_kwargs, _RESERVED_GEN_KWARGS)
        for inp in items:
            self._check_entry(inp, None, where=where)
        rows, kw = self._batch_kw(items, enable_thinking)
        processors = list(gen_kwargs.pop("logits_processor", None) or [])
        if allowed_token_ids is not None:
            vocab_rows = self._model.get_output_embeddings().weight.shape[0]
            processors.insert(0, _AllowedTokensProcessor(allowed_token_ids, vocab_rows))
        gen_kwargs.setdefault("do_sample", False)
        stop_ids = [self._ids[s] for s in STOP_TOKENS]
        pad_id = self._ids[END_OF_TEXT]
        out, wall = batching.run_generate(self._model, kw, device=self._device, max_new_tokens=max_new_tokens,
                                          stop_ids=stop_ids, pad_id=pad_id, processors=processors,
                                          gen_kwargs=gen_kwargs, scope=self.gdn_scope())
        return batching.make_results(rows, out, kw["input_ids"].shape[1], wall=wall, stop_ids=stop_ids, pad_id=pad_id,
                                     max_new_tokens=max_new_tokens, split_answer=self.split_answer,
                                     enable_thinking=enable_thinking)

    def _batch_kw(self, items: Sequence[ModelInputs],
                  enable_thinking: bool) -> tuple[list[batching.PromptRow], dict[str, Any]]:
        """generate / generate_batch / option_logits 共用的拼批：每行的提示（_prompt_row）左填充成一批，各行的图像张量按行的顺序
        拼接，并逐行核对 M-RoPE 位置（_check_batch_positions）。"""
        rows = [self._prompt_row(inp, enable_thinking) for inp in items]
        kw: dict[str, Any] = batching.batch_prompt(rows, pad_id=self._ids[END_OF_TEXT], device=self._device)
        kw.update(self._image_kw(items))
        self._check_batch_positions(kw, items)
        return rows, kw

    def _image_kw(self, items: Sequence[ModelInputs]) -> dict[str, torch.Tensor]:
        """各行的 pixel_values / image_grid_thw 按行的顺序拼接（没有图时为空）。_batch_kw 与训练前向共用。"""
        with_images = [inp for inp in items if inp.images]
        if not with_images:
            return {}
        return {"pixel_values": torch.cat([inp.tensors["pixel_values"] for inp in with_images], dim=0),
                "image_grid_thw": torch.cat([inp.tensors["image_grid_thw"] for inp in with_images], dim=0)}

    def option_logits(
        self,
        inputs_list: Sequence[ModelInputs],
        options: Sequence[Sequence[str]],
        *,
        enable_thinking: bool,
    ) -> list[OptionScores]:
        """受限选项读数（约定见 base.ModelAdapter.option_logits；DEV-0016）。提示与左填充同 generate_batch（_batch_kw，含 M-RoPE
        逐行核对）；第一个生成位置 = assistant 头最后一个 token 之后：关思考时 assistant 头以空思考块 `<think>\\n\\n</think>\\n\\n`
        结尾，开思考时以 `<think>\\n` 结尾。前向套 gdn_scope()（CPU 上显式走 torch 的 GDN 实现）。"""
        self._require_loaded()
        batching.check_enable_thinking(enable_thinking)
        self._require_eval("option_logits")
        items = batching.check_inputs_list(inputs_list, where="option_logits")
        opts = batching.check_options(options, len(items))
        for inp in items:
            self._check_entry(inp, None, where="option_logits")
        rows, kw = self._batch_kw(items, enable_thinking)
        return batching.option_logits_rows(self._model, self._tok, rows, kw, opts, device=self._device,
                                           stop_ids=[self._ids[s] for s in STOP_TOKENS], pad_id=self._ids[END_OF_TEXT],
                                           scope=self.gdn_scope())

    def _check_batch_positions(self, kw: dict[str, Any], items: Sequence[ModelInputs]) -> None:
        """有填充的批：HF 的 get_rope_index（带 attention_mask，与 generate 内部同一种调用）在每行非填充位置上的 (T, H, W)
        必须与本适配器按规则构造的位置（mrope_rows，图像段取各图自己的网格，assistant 头是文本）逐个相等。不一致就报错。"""
        pads = batching.n_padding(kw["attention_mask"])
        if not any(pads):
            return
        ref, _ = self._model.model.get_rope_index(
            kw["input_ids"], mm_token_type_ids=kw["mm_token_type_ids"], image_grid_thw=kw.get("image_grid_thw"),
            attention_mask=kw["attention_mask"])
        ref = ref.cpu()
        width = kw["input_ids"].shape[1]
        for b, (inp, p) in enumerate(zip(items, pads)):
            own = mrope_rows(width - p, [(i.span.start, i.span.end, i.grid_hw) for i in inp.images])
            if not torch.equal(ref[:, b, p:], own):
                bad = int((ref[:, b, p:] != own).any(dim=0).nonzero()[0])
                raise RuntimeError(f"左填充后第 {b} 行的 M-RoPE 位置与单独生成时不同（行内第 {bad} 个 token："
                                   f"HF {ref[:, b, p + bad].tolist()}，本适配器 {own[:, bad].tolist()}）")

    def split_answer(self, token_ids: Sequence[int], *, enable_thinking: bool) -> tuple[str, str, bool]:
        """把生成的 token 拆成 (text, raw_text, thinking_unclosed)（D10）。

        raw_text：全部生成 token 的解码，含思考段与特殊 token。
        text：最后一个 </think> 之后的内容；末尾的停止 token 去掉，其余特殊 token（包括 grid token）保留，
        首尾换行去掉（与 Qwen 官方示例的 strip("\\n") 相同）。
        开了思考却没有 </think>：text 为空串，thinking_unclosed = True。关思考时没有 </think> 就取全部。
        """
        self._require_loaded()
        ids = [int(i) for i in token_ids]
        raw_text = self._tok.decode(ids, skip_special_tokens=False)
        end = self._ids[THINK_END]
        last = max((k for k, v in enumerate(ids) if v == end), default=None)
        if last is not None:
            ans = ids[last + 1 :]
        elif enable_thinking:
            return "", raw_text, True
        else:
            ans = ids
        stop = {self._ids[s] for s in STOP_TOKENS}
        if ans and ans[-1] in stop:
            ans = ans[:-1]
        text = self._tok.decode(ans, skip_special_tokens=False).strip("\n")
        return text, raw_text, False

    # ------------------------------------------------------------------
    # 查询 token 与新增 token
    # ------------------------------------------------------------------

    def inject_query_tokens(self, n: int, position_mode: PositionMode) -> None:
        """登记 n 个查询 token；之后的 build_inputs 把它们放在 user 内容最后一项之后。可重复调用以改变 n 或模式。

        占位符固定是预留区的末行（QUERY_PLACEHOLDER 的说明；DEV-0020），首次调用时分配，嵌入行（不共享时连同 lm_head 行）按
        D9-3 初始化为基础词表行的均值（grid token 的分配先越过了这一行时已经初始化过）。它不计入 E3b 的 300 个 token 预算，
        但占用一个嵌入行。前向时它的嵌入一律由 query_embeds 替换（必须显式传，D21-2）。首次调用（分配占位符的行）必须在 inject_lora 之前；之后再调只改个数与模式，不碰任何行，LoRA 之后也可以。
        """
        self._require_loaded()
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise ValueError(f"n 必须是正整数，收到 {n!r}")
        if position_mode not in POSITION_MODES:
            raise ValueError(f"position_mode 必须是 {POSITION_MODES} 之一，收到 {position_mode!r}")
        if position_mode == "grid2d" and math.isqrt(n) ** 2 != n:
            raise ValueError(f"grid2d 要求 n 是完全平方数，收到 {n}")
        if self._query_token_id is None:
            self._check_tokens_mutable("inject_query_tokens")  # 首次注入要分配行：必须在 inject_lora 之前
            q = self._require_query_row()
            self._allocate_query_row(init=True)
            self._query_token_id = q
        self._query_n = n
        self._query_mode = position_mode

    def clear_query_tokens(self) -> None:
        """取消查询 token：之后的 build_inputs 不再追加（占位符 token 仍留在词表里）。"""
        self._query_n = None
        self._query_mode = None

    def add_grid_tokens(self, names: Sequence[str]) -> dict[str, int]:
        """加入像素 token 通道用的新 token，返回 名字 → id。

        预算按累计计（≤ MAX_NEW_GRID_TOKENS，查询占位符不计）。先复用预留嵌入行，id 从 attach 时的 len(tokenizer) 起连续取，
        跳过预留区末行（查询占位符的固定 id；越过它时把占位符登记进 tokenizer 占住，DEV-0020），所以 id 与是否注入过查询无关；
        累计需要的行数超过现有行数时，嵌入（lm_head 不共享时连同 lm_head）扩到恰好需要的行数，已有的行逐位不变，从不缩表。
        新行初始化为基础词表行的精确均值；lm_head 与嵌入共享（0.8B/2B）时只写一次。必须在 inject_lora 之前调用。
        """
        self._require_loaded()
        self._check_tokens_mutable("add_grid_tokens")
        if isinstance(names, str):
            raise TypeError("names 必须是字符串的序列，不能是单个字符串")
        names = list(names)
        check_grid_token_names(names, n_existing=self._n_grid_tokens)
        ids = self._add_new_tokens(names)
        self._n_grid_tokens += len(names)
        added = dict(zip(names, ids))
        self._grid_token_ids.update(added)
        return added

    # ------------------------------------------------------------------
    # 规模
    # ------------------------------------------------------------------

    def num_layers(self) -> int:
        self._require_loaded()
        n = int(self._model.config.text_config.num_hidden_layers)
        if len(self._model.model.language_model.layers) != n:
            raise RuntimeError("decoder 层数与 text_config.num_hidden_layers 不一致")
        return n

    def _last_decoder_layer(self) -> torch.nn.Module:
        """PRE_NORM_LAST 的钩子位置：LLM 的最后一个 decoder 层。Qwen3_5TextModel.forward 在层循环之后紧接 self.norm，
        中间没有别的运算，所以它的输出就是最终归一化的输入。"""
        return self._model.model.language_model.layers[self.num_layers() - 1]

    def hidden_size(self) -> int:
        self._require_loaded()
        return int(self._model.config.text_config.hidden_size)

    # ------------------------------------------------------------------
    # 输出解析（DEV-0006）
    # ------------------------------------------------------------------

    def parse_boxes(self, text: str, *, width: float, height: float) -> BoxParse:
        """本家族的原生框（JSON 的 bbox_2d，0–1000，x 在前），规则见模块级的 parse_boxes。
        不需要加载模型；parse_one_box 由基类在它之上给出。"""
        return parse_boxes(text, width=width, height=height)  # 模块级函数（方法体里的名字按全局作用域查找）

    def format_boxes(self, boxes: Sequence[Sequence[float]], *, width: float, height: float,
                     labels: Sequence[str | None] | None = None) -> str:
        """parse_boxes 的逆（DEV-0027），格式见模块级的 format_boxes。不需要加载模型。"""
        return format_boxes(boxes, width=width, height=height, labels=labels)

    # ------------------------------------------------------------------
    # 停止 token（DEV-0027：公开接口，替代私有的 _stop_ids()）
    # ------------------------------------------------------------------

    def end_of_turn_token_id(self) -> int:
        """回合结束 token：<|im_end|>（STOP_TOKENS 的第一个）。"""
        self._require_loaded()
        return self._ids[IM_END]

    def stop_token_ids(self) -> tuple[int, ...]:
        """全部停止 token：<|im_end|>、<|endoftext|>（STOP_TOKENS 的顺序）。"""
        self._require_loaded()
        return tuple(self._ids[s] for s in STOP_TOKENS)

    # ------------------------------------------------------------------
    # GDN 实现（DEV-0003-a1）
    # ------------------------------------------------------------------

    def gdn_scope(self) -> contextlib.AbstractContextManager[None]:
        """本适配器前向与生成所用的 GDN 实现作用域。

        设备是 CPU 时返回 torch_gdn()：显式走 torch 原函数，不管 transformers 在 import 时绑到了什么（服务器装了 fla，
        绑定的是只收 GPU 张量的 Triton 内核）。其他设备返回空作用域，保持 transformers 的绑定（服务器 GPU 上是 fla）。
        _run、vision_features、generate 都套在它里面；绕过适配器直接调 self.model(...) 做对照时也要套上它，
        两条路径才走同一个实现。
        """
        return torch_gdn() if self._device.type == "cpu" else contextlib.nullcontext()

    def gdn_implementation(self) -> dict[str, dict[str, Any]]:
        """只读：本适配器的前向与生成此刻会走的 GDN 实现，按模块级函数名给出实际调用的函数。

        在与前向相同的作用域（gdn_scope）里读 modeling_qwen3_5 的绑定，读完退出，绑定原样恢复，不改任何状态。
        绑定是 transformers 的分派包装时穿过它（kernel_implementation），报实际调用的函数。每项的字段：
        module、name（实际调用的函数的模块路径与名字）；is_torch_reference（是否就是 transformers 的 torch 原函数）；
        switched_to_torch（本适配器的设备是否触发 CPU 上的切换）。不需要已挂模型。
        """
        mq = _modeling_module()
        switched = self._device.type == "cpu"
        out: dict[str, dict[str, Any]] = {}
        with self.gdn_scope():
            for name in GDN_FUNCTIONS:
                impl = kernel_implementation(getattr(mq, name))
                out[name] = {
                    "module": getattr(impl, "__module__", None),
                    "name": getattr(impl, "__qualname__", type(impl).__name__),
                    "is_torch_reference": _is_torch_reference(impl, mq, name),
                    "switched_to_torch": switched,
                }
        return out

    # ------------------------------------------------------------------
    # 训练侧钩子（DEV-0020；公共实现见 src/adapters/training.py 的 TrainingSupport）
    # ------------------------------------------------------------------

    def _pad_id(self) -> int:
        return self._ids[END_OF_TEXT]

    def _stop_ids(self) -> list[int]:
        return [self._ids[s] for s in STOP_TOKENS]

    def _think_markers(self) -> tuple[str | None, str, str]:
        """预算强制收尾（src/adapters/budget.py）：(输出里的思考段开头（Qwen 在提示里，None）, 结束标记, 结束标记后的固定分隔)。"""
        return None, THINK_END, THINK_SEPARATOR

    def _forward_scope(self) -> contextlib.AbstractContextManager[None]:
        return self.gdn_scope()

    def _user_index_shift(self, enable_thinking: bool) -> tuple[int, int]:
        """提示行 = user 轮（ModelInputs）+ 助手头：ModelInputs 的下标在提示行里不变。"""
        return 0, 0

    def _logit_softcap(self) -> float | None:
        return None  # Qwen3.5 没有 final_logit_softcapping

    def _current_query_mode(self) -> PositionMode | None:
        return self._query_mode

    def _vision_modules(self) -> list[Any]:
        return [self._model.model.visual]  # 视觉塔（含 patch merger 投影）

    def _training_forward_kwargs(self, items: Sequence[ModelInputs], asm: Any, image_kw: dict[str, torch.Tensor],
                                 query_mask: torch.Tensor | None,
                                 query_embeds: torch.Tensor | None) -> dict[str, Any]:
        """内层模型的输入：嵌入查表 → 图像位置并入视觉特征 → 查询位置换成 query_embeds（与 _input_embeds 同一种并入方式）；
        位置 = (4, B, L)：第 0 行 cumsum(mask) − 1，第 1–3 行按本适配器的 M-RoPE 规则逐行构造，并与 get_rope_index 核对。"""
        ids = asm.input_ids
        embeds = self._model.get_input_embeddings()(ids)
        if image_kw:
            feats = self._image_features_from(image_kw, [len(i.span) for inp in items for i in inp.images])
            embeds = scatter_rows(embeds, ids == self._ids[IMAGE_PAD], feats, what="图像特征")
        if query_mask is not None:
            embeds = scatter_rows(embeds, query_mask, query_embeds.repeat(len(items), 1), what="查询嵌入")
        pos = self._sequence_positions(items, asm.input_ids, asm.attention_mask, asm.mm_token_type_ids,
                                       image_kw.get("image_grid_thw"), asm.n_padding)
        return {"inputs_embeds": embeds, "attention_mask": asm.attention_mask, "position_ids": pos}

    def _sequence_positions(self, items: Sequence[ModelInputs], ids: torch.Tensor, mask: torch.Tensor,
                            mm: torch.Tensor, grid_thw: torch.Tensor | None, pads: Sequence[int]) -> torch.Tensor:
        """左填充整批的 (4, B, L) 位置：每行非填充部分按 mrope_rows（图像段 + grid2d 的查询块），填充处记 0（与 HF 相同）。
        与 get_rope_index（带 attention_mask，即 generate 的算法）逐行核对：没有 grid2d 查询时整行，有时查询块之前。"""
        b_, width = ids.shape
        pos = torch.zeros((4, b_, width), dtype=torch.long)
        stops = []
        for b, (inp, p) in enumerate(zip(items, pads)):
            segs = [(i.span.start, i.span.end, i.grid_hw) for i in inp.images]
            stop = width - p
            if inp.query_span is not None and inp.query_position_mode == "grid2d":
                qs, qe = inp.query_span.start, inp.query_span.end
                q = math.isqrt(qe - qs)
                if q * q != qe - qs:
                    raise ValueError(f"grid2d 的查询个数 {qe - qs} 不是完全平方数")
                segs.append((qs, qe, (q, q)))
                stop = qs
            pos[1:, b, p:] = mrope_rows(width - p, segs)
            pos[0, b, p:] = torch.arange(width - p)
            stops.append(stop)
        ref, _ = self._model.model.get_rope_index(ids, mm_token_type_ids=mm, image_grid_thw=grid_thw, attention_mask=mask)
        ref = ref.cpu()
        for b, (p, stop) in enumerate(zip(pads, stops)):
            if not torch.equal(pos[1:, b, p: p + stop], ref[:, b, p: p + stop]):
                raise RuntimeError(f"第 {b} 行的 M-RoPE 位置与 get_rope_index 不一致")
        if not torch.equal(pos[0], cumsum_positions(mask.cpu())):
            raise RuntimeError("文本位置与 cumsum(mask) − 1 不一致")
        return pos.to(ids.device)

    def _decode_positions(self, kw: dict[str, Any]) -> torch.Tensor:
        """受限解码预填的位置，与 HF generate 相同（Qwen3_5ForConditionalGeneration._prepare_position_ids_for_generation）：
        [cumsum(mask) − 1; get_rope_index(...)]；没有图时三维都取文本位置。"""
        text = cumsum_positions(kw["attention_mask"])
        if kw.get("image_grid_thw") is not None:
            vis, _ = self._model.model.get_rope_index(kw["input_ids"], mm_token_type_ids=kw["mm_token_type_ids"],
                                                      image_grid_thw=kw["image_grid_thw"],
                                                      attention_mask=kw["attention_mask"])
        else:
            vis = text.unsqueeze(0).expand(3, -1, -1)
        return torch.cat([text.unsqueeze(0), vis.to(text.device)], dim=0)

    def _decode_step_extra(self, batch: int) -> dict[str, Any]:
        """解码每步的额外输入：mm_token_type_ids 只取新 token 的一列（文本，0），与 HF 相同。"""
        return {"mm_token_type_ids": torch.zeros((batch, 1), dtype=torch.long, device=self._device)}

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _require_loaded(self) -> None:
        if self._model is None:
            raise RuntimeError("还没有 load / attach 模型")

    def _content(self, n_images: int, instruction: str, order: Order) -> list[dict[str, str]]:
        imgs = [{"type": "image"} for _ in range(n_images)]
        instr = {"type": "text", "text": instruction}
        content = [instr, *imgs] if order == "instruction_first" else [*imgs, instr]
        if self._query_n is not None:
            content.append({"type": "text", "text": _QUERY_SLOT})  # build_inputs 在这里拼进 [占位符 id] × n
        return content

    def _render(self, content: list[dict[str, str]], *, add_generation_prompt: bool = False,
                enable_thinking: bool | None = None) -> str:
        kwargs = {} if enable_thinking is None else {"enable_thinking": enable_thinking}
        return self._tok.apply_chat_template(
            [{"role": "user", "content": content}],
            chat_template=self._template,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )

    def _encode(self, text: str) -> list[int]:
        return list(self._tok(text, add_special_tokens=False)["input_ids"])

    def _expand(self, text: str, counts: Sequence[int]) -> str:
        parts = text.split(IMAGE_PAD)
        if len(parts) != len(counts) + 1:
            raise RuntimeError("图像占位符个数与展开计数不一致")
        out = [parts[0]]
        for n, part in zip(counts, parts[1:]):
            out.append(IMAGE_PAD * n)
            out.append(part)
        return "".join(out)

    def _check_instruction(self, instruction: str) -> None:
        for sentinel in (_SENTINEL, _QUERY_SLOT):
            if sentinel in instruction:
                raise ValueError(f"指令里含有内部哨兵字符串 {sentinel!r}")
        ids = set(self._encode(instruction))
        forbidden = [s for s in _STRUCTURAL_TOKENS if self._ids[s] in ids]
        # 查询占位符：不论是否已登记进 tokenizer，字面出现就算（DEV-0020 起它平时不在 tokenizer 里）
        if QUERY_PLACEHOLDER in instruction or (self._query_token_id is not None and self._query_token_id in ids):
            forbidden.append(QUERY_PLACEHOLDER)
        if forbidden:
            raise ValueError(f"指令里含有结构性 token：{forbidden}")

    def _check_layers(self, layers: Iterable[int]) -> list[int]:
        self._require_loaded()
        if isinstance(layers, (str, bytes)):
            raise TypeError("layers 必须是整数序列")
        n = self.num_layers()
        out = []
        for layer in layers:
            if isinstance(layer, bool) or not isinstance(layer, (int, np.integer)):
                raise TypeError(f"层号必须是整数，收到 {layer!r}")
            if not 0 <= int(layer) <= n:
                raise ValueError(f"层号 {int(layer)} 越界，有效范围 [0, {n}]")
            out.append(int(layer))
        if not out:
            raise ValueError("layers 为空")
        if len(set(out)) != len(out):
            raise ValueError(f"layers 有重复：{out}")
        return out

    def _model_device_dtype(self) -> tuple[torch.device, torch.dtype]:
        w = self._model.get_input_embeddings().weight
        return w.device, w.dtype

    def _check_entry(self, inputs: ModelInputs, query_embeds: Any, *, where: str) -> None:
        """入口断言：在任何模型算子之前比较 device / dtype，纯 Python，任何机器上都会触发。"""
        self._require_loaded()
        m_dev, m_dtype = self._model_device_dtype()
        if m_dev != self._device:
            raise AdapterDeviceError(f"{where}：模型在 {m_dev}，构造时给定的设备是 {self._device}")
        if m_dtype != self._dtype:
            raise AdapterDeviceError(f"{where}：模型 dtype 是 {m_dtype}，load 时给定的是 {self._dtype}")
        for mod_name, mod in (("lm_head", self._model.get_output_embeddings()),
                              ("visual.patch_embed", self._model.model.visual.patch_embed.proj)):
            if mod.weight.device != m_dev:
                raise AdapterDeviceError(f"{where}：{mod_name} 在 {mod.weight.device}，嵌入层在 {m_dev}")
        if not isinstance(inputs, ModelInputs):
            raise TypeError(f"inputs 必须是 ModelInputs，收到 {type(inputs).__name__}")
        for name, v in inputs.tensors.items():
            if not torch.is_tensor(v):
                continue
            if v.device != m_dev:
                raise AdapterDeviceError(f"{where}：输入 {name} 在 {v.device}，模型在 {m_dev}")
            if v.is_floating_point() and v.dtype != m_dtype:
                raise AdapterDeviceError(f"{where}：输入 {name} 的 dtype 是 {v.dtype}，模型是 {m_dtype}")
        if query_embeds is not None:
            if not torch.is_tensor(query_embeds):
                raise TypeError(f"query_embeds 必须是张量，收到 {type(query_embeds).__name__}")
            if query_embeds.device != m_dev:
                raise AdapterDeviceError(f"{where}：query_embeds 在 {query_embeds.device}，模型在 {m_dev}")
            if query_embeds.dtype != m_dtype:
                raise AdapterDeviceError(f"{where}：query_embeds 的 dtype 是 {query_embeds.dtype}，模型是 {m_dtype}")
            if inputs.query_span is None:
                raise ValueError(f"{where}：给了 query_embeds，但 inputs 里没有查询 token")
            want = (len(inputs.query_span), self.hidden_size())
            if tuple(query_embeds.shape) != want:
                raise ValueError(f"{where}：query_embeds 形状 {tuple(query_embeds.shape)}，期望 {want}")

    def _image_features(self, inputs: ModelInputs) -> torch.Tensor:
        return self._image_features_from(inputs.tensors, [len(i.span) for i in inputs.images])

    def _image_features_from(self, t: Any, sizes: Sequence[int]) -> torch.Tensor:
        """pixel_values / image_grid_thw（一条输入的，或整批按行拼接的）→ [Σ sizes, hidden]，与 HF 前向的并入方式相同。"""
        out = self._model.model.get_image_features(t["pixel_values"], t["image_grid_thw"], return_dict=True)
        feats = torch.cat(out.pooler_output, dim=0).to(device=self._device, dtype=self._dtype)
        n_expected = sum(sizes)
        if tuple(feats.shape) != (n_expected, self.hidden_size()):
            raise RuntimeError(f"视觉特征形状 {tuple(feats.shape)}，期望 ({n_expected}, {self.hidden_size()})")
        return feats

    def _input_embeds(self, inputs: ModelInputs, query_embeds: torch.Tensor | None) -> torch.Tensor:
        """与 Qwen3_5Model.forward 相同的并入方式：嵌入查表 → 图像位置 masked_scatter 视觉特征；再替换查询位置。"""
        ids = inputs.tensors["input_ids"]
        embeds = self._model.get_input_embeddings()(ids)
        if inputs.images:
            feats = self._image_features(inputs)
            mask = (ids == self._ids[IMAGE_PAD]).unsqueeze(-1)
            embeds = embeds.masked_scatter(mask, feats)
        if query_embeds is not None:
            qs, qe = inputs.query_span.start, inputs.query_span.end
            embeds = torch.cat([embeds[:, :qs], query_embeds.unsqueeze(0), embeds[:, qe:]], dim=1)
        return embeds

    def _run(self, inputs: ModelInputs, query_embeds: torch.Tensor | None, *, where: str) -> tuple[torch.Tensor, ...]:
        self._check_entry(inputs, query_embeds, where=where)
        if inputs.query_span is not None and query_embeds is None:
            raise ValueError(f"{where}：输入里有查询 token，必须显式传 query_embeds（D21-2；不再静默用占位符的行）")
        with self.gdn_scope():
            embeds = self._input_embeds(inputs, query_embeds)
            pos = self.position_ids(inputs)
            out = self._model(
                inputs_embeds=embeds,
                attention_mask=inputs.tensors["attention_mask"],
                position_ids=pos,
                output_hidden_states=True,
                logits_to_keep=1,
                use_cache=False,
                return_dict=True,
            )
        hs = out.hidden_states
        if len(hs) != self.num_layers() + 1:
            raise RuntimeError(f"hidden_states 有 {len(hs)} 项，期望 num_layers()+1 = {self.num_layers() + 1}")
        return hs

    def _image_index(self, inputs: ModelInputs) -> torch.Tensor:
        idx = [torch.arange(i.span.start, i.span.end) for i in inputs.images]
        cat = torch.cat(idx) if idx else torch.empty(0, dtype=torch.long)
        return cat.to(inputs.tensors["input_ids"].device)

    def _base_mean(self, weight: torch.Tensor) -> torch.Tensor:
        """基础词表行 [0, n_base_vocab) 的均值：float64 分块累加，再转回参数 dtype。"""
        n, dim = self._n_base_vocab, weight.shape[1]
        step = max(1, _MEAN_CHUNK_BYTES // (dim * 8))
        acc = torch.zeros(dim, dtype=torch.float64, device=weight.device)
        for s in range(0, n, step):
            acc += weight[s : min(s + step, n)].to(torch.float64).sum(dim=0)
        return (acc / n).to(weight.dtype)

    def _init_rows(self, ids: list[int]) -> None:
        """新 token 的嵌入行（不共享时连同 lm_head 行）写成基础词表行的均值；共享时只写一次。"""
        idx = torch.tensor(ids, dtype=torch.long, device=self._device)
        with torch.no_grad():
            emb = self._model.get_input_embeddings().weight
            emb[idx] = self._base_mean(emb)
            if not self._tied:
                head = self._model.get_output_embeddings().weight
                head[idx] = self._base_mean(head)


def _params_and_buffers(model: Any) -> Iterable[torch.Tensor]:
    yield from model.parameters()
    yield from model.buffers()


def _named_params_and_buffers(model: Any) -> Iterable[tuple[str, torch.Tensor]]:
    yield from model.named_parameters()
    yield from model.named_buffers()


# ---------------------------------------------------------------------------
# 原生坐标 → 内部绝对像素 xyxy（D13-4；docs/model_notes.md §5.5；DEV-0006）
# ---------------------------------------------------------------------------

# 原生框的键：[x1, y1, x2, y2]，0–1000，x 在前（§5.5 零样本实测；前代 Qwen3-VL cookbook 的 0–1000 相对坐标）
BOX_KEY = "bbox_2d"


def native_box_to_xyxy(quad: Sequence[float] | np.ndarray, *, width: float, height: float) -> np.ndarray:
    """原生四元组 [x1, y1, x2, y2]（0–1000，x 在前）→ 内部绝对像素 xyxy：x = x_norm / 1000 · W，y = y_norm / 1000 · H
    （geometry.norm1000_to_abs，浮点，不取整）。只做线性换算，不裁剪、不交换。"""
    return norm1000_to_abs(quad, width=width, height=height)


def parse_boxes(text: str, *, width: float, height: float) -> BoxParse:
    """解析 Qwen3.5 的原生框，换算成内部绝对像素 xyxy。width、height 是原图的宽高（按原图归一化，§5.5）。

    原生格式：JSON，形如 `[{"bbox_2d": [x1, y1, x2, y2], "label": "…"}, …]`，常包在 ```json 代码块里（§5.5 的四条实测都是）。
    接受单个对象与对象的列表，label 可缺，其他键忽略；另外接受裸的 [x1, y1, x2, y2] 或它的列表，同样按 x 在前、0–1000 解释。
    数值是有限的 JSON 数，整数或小数都接受（JSON 本身允许小数；不同于 InternVL 的整数语法）。
    失败判定（通用规则见 src.parsing.parse_json_boxes 与 extract_json）：没有 JSON → no_box；多个代码块或多个顶层 JSON → ambiguous；
    JSON 坏了、对象缺 bbox_2d、值不是 4 个数 → malformed；坐标 < 0 或 > 1000 → out_of_range；x2 < x1 或 y2 < y1 → inverted。
    空列表 [] 是"没有框"的明确回答：成功、零个框（只要一个框时 parse_one_box 判 no_box）。
    """
    return parse_json_boxes(text, key=BOX_KEY, to_xyxy=native_box_to_xyxy, width=width, height=height)


def parse_one_box(text: str, *, width: float, height: float) -> OneBoxParse:
    """只要一个框（如 RefCOCO 的 REC）：规则见 src.parsing.one_box，与适配器方法 parse_one_box 相同。"""
    return one_box(parse_boxes(text, width=width, height=height))


# 原生框的文本格式（format_boxes；DEV-0027）：Qwen3-VL cookbook 的写法，也是 RUN-0005 关思考时的实测输出，如
# '```json\n[\n\t{"bbox_2d": [14, 114, 492, 983], "label": "cat on the left"}\n]\n```'（tab 缩进，一行一个对象）。
JSON_FENCE_OPEN = "```json\n"
JSON_FENCE_CLOSE = "\n```"
QWEN_BOX_INDENT = "\t"


def json_box_text(objs: Sequence[str], *, indent: str) -> str:
    """JSON 框对象（已序列化）→ 代码块文本：`[` 换行、每行 indent + 一个对象（对象之间 ",\\n"）、`]`；零个对象写 `[]`。
    Gemma 用同一个函数（两个空格缩进）。"""
    body = "[\n" + ",\n".join(indent + o for o in objs) + "\n]" if objs else "[]"
    return JSON_FENCE_OPEN + body + JSON_FENCE_CLOSE


def json_box_object(key: str, quad: Sequence[int], label: str | None) -> str:
    """一个 JSON 框对象：{"<key>": [a, b, c, d], "label": "<label>"}（label 为 None 时不写 label；字符串按 JSON 转义）。"""
    import json

    s = "{" + json.dumps(key) + ": [" + ", ".join(str(int(v)) for v in quad) + "]"
    if label is not None:
        s += ', "label": ' + json.dumps(label, ensure_ascii=False)
    return s + "}"


def format_boxes(boxes: Sequence[Sequence[float]], *, width: float, height: float,
                 labels: Sequence[str | None] | None = None) -> str:
    """parse_boxes 的逆（DEV-0027）：绝对像素 xyxy → Qwen 的原生框文本（E1 定位提示 "Report bbox coordinates in JSON format." 要求
    的输出格式）：```json 代码块，`[{"bbox_2d": [x1, y1, x2, y2], "label": …}, …]`，0–1000 的整数，x 在前（换算与取整见
    base.native_box_ints）。零个框写 `[]`（parse_boxes 判为成功、零个框）。"""
    quads = native_box_ints(boxes, width=width, height=height)
    labs = check_box_labels(labels, len(quads))
    return json_box_text([json_box_object(BOX_KEY, q, lab) for q, lab in zip(quads, labs)], indent=QWEN_BOX_INDENT)


def vision_patch_row_major_order(grid_thw: torch.Tensor, spatial_merge_size: int) -> torch.Tensor:
    """视觉塔里 patch 的序列顺序 → 行主序（pre_projection_features 用）：返回 order，使 x_row_major = x_seq[order]。

    用模型自己的元数据：Qwen3_5VisionModel 的视觉 RoPE 位置 get_vision_position_ids(grid_thw, merge)（modeling_qwen3_5 里的那个
    函数）给出每个序列位置的 (行, 列)；行主序的键 = 行 · gw + 列，必须恰好是 0 … gh·gw − 1 的一个排列，否则报错。只接受单图、T = 1。"""
    grid = torch.as_tensor(grid_thw).reshape(-1, 3)
    if grid.shape[0] != 1 or int(grid[0, 0]) != 1:
        raise ValueError(f"只接受单图（T = 1），收到 grid_thw={grid.tolist()}")
    gh, gw = int(grid[0, 1]), int(grid[0, 2])
    get_pos = getattr(_modeling_module(), "get_vision_position_ids", None)
    if get_pos is None:
        raise RuntimeError(f"{QWEN35_MODELING} 里没有 get_vision_position_ids（transformers 版本变了？）")
    pos = get_pos(grid.cpu(), int(spatial_merge_size))
    if tuple(pos.shape) != (gh * gw, 2):
        raise RuntimeError(f"视觉 RoPE 位置形状 {tuple(pos.shape)}，期望 ({gh * gw}, 2)")
    key = pos[:, 0].long() * gw + pos[:, 1].long()
    if not torch.equal(torch.sort(key).values, torch.arange(gh * gw)):
        raise RuntimeError("视觉 RoPE 的 (行, 列) 不是 patch 网格的一个排列")
    return torch.argsort(key)


# ---------------------------------------------------------------------------
# 小型随机 Qwen3.5：CPU 测试与冒烟脚本 --tiny 的彩排（不下载权重）
# ---------------------------------------------------------------------------

TINY_TEXT = {
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 4,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "linear_num_key_heads": 2,
    "linear_num_value_heads": 2,
    "linear_key_head_dim": 16,
    "linear_value_head_dim": 16,
}
TINY_VISION = {"depth": 1, "hidden_size": 64, "intermediate_size": 128, "num_heads": 2, "out_hidden_size": 64}


def load_pinned_components(model_key: str = TINY_SOURCE_KEY) -> tuple[Any, Any, Any, str]:
    """从登记的固定 revision 下载 config、tokenizer、图像处理器与聊天模板（都是小文件，不含权重）。

    图像处理器固定用 PIL 后端（Qwen2VLImageProcessorPil），开发机没有 torchvision 也能跑；服务器上的 load()
    走官方 AutoProcessor（torchvision 后端）。两者只在像素值的舍入上可能不同，接口相同。
    """
    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil

    entry = registry.get_model(model_key)
    if entry.family != registry.FAMILY_QWEN35:
        raise ValueError(f"{model_key} 不是 Qwen3.5")
    config = AutoConfig.from_pretrained(entry.repo, revision=entry.revision)
    tokenizer = AutoTokenizer.from_pretrained(entry.repo, revision=entry.revision)
    image_processor = Qwen2VLImageProcessorPil.from_pretrained(entry.repo, revision=entry.revision)
    template = tokenizer.chat_template
    if not isinstance(template, str) or not template:
        raise RuntimeError(f"{entry.repo}@{entry.revision} 没有取到聊天模板")
    return config, tokenizer, image_processor, template


def tiny_config(real_config: Any, *, tie_word_embeddings: bool) -> Any:
    """把真实 config 缩成小型随机模型的 config：4 层（3 层 linear_attention + 1 层 full_attention）、hidden 64、
    视觉塔 1 层。词表、特殊 token id、patch/merge、head_dim 与 RoPE 参数保持真实值，位置编码的结构与真模型相同。"""
    d = real_config.to_dict()
    tc, vc = d["text_config"], d["vision_config"]
    types = list(tc["layer_types"][: TINY_TEXT["num_hidden_layers"]])
    if types.count("linear_attention") != 3 or types.count("full_attention") != 1:
        raise ValueError(f"真实 config 的前 4 层类型不是 3 linear + 1 full：{types}")
    tc.update(TINY_TEXT, layer_types=types, tie_word_embeddings=bool(tie_word_embeddings))
    vc.update(TINY_VISION)
    d["tie_word_embeddings"] = bool(tie_word_embeddings)
    return type(real_config).from_dict(d)


def build_tiny_model(real_config: Any, *, tie_word_embeddings: bool, dtype: torch.dtype = torch.float32,
                     seed: int = 0) -> Any:
    """按 tiny_config 构造随机初始化的 Qwen3_5ForConditionalGeneration（sdpa，eval）。"""
    from transformers import AutoModelForImageTextToText

    cfg = tiny_config(real_config, tie_word_embeddings=tie_word_embeddings)
    torch.manual_seed(seed)
    model = AutoModelForImageTextToText.from_config(cfg, attn_implementation="sdpa", dtype=dtype)
    return model.eval()


def make_tiny_adapter(
    components: tuple[Any, Any, Any, str],
    *,
    image_size: tuple[int, int] | None = None,
    image_policy: FixedSize | PixelBudget | None = None,
    tie_word_embeddings: bool = True,
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
    device: str | torch.device = "cpu",
    model: Any = None,
    source_key: str = TINY_SOURCE_KEY,
) -> Qwen35Adapter:
    """用 load_pinned_components 的结果挂一个小型随机模型的适配器。图像口径与 Qwen35Adapter 相同：image_size 与
    image_policy 恰好给一个。source_key 是 components 的来源（只写进 model_key）。

    注意：add_grid_tokens / inject_query_tokens 会改 tokenizer；需要彼此独立的适配器时，传入各自的 tokenizer 副本。
    """
    config, tokenizer, image_processor, template = components
    if model is None:
        model = build_tiny_model(config, tie_word_embeddings=tie_word_embeddings, dtype=dtype, seed=seed)
    adapter = Qwen35Adapter(device=device, image_size=image_size, image_policy=image_policy)
    adapter.attach(model, tokenizer=tokenizer, image_processor=image_processor, chat_template=template,
                   model_key=f"tiny({source_key})")
    return adapter


def reference_processor(tokenizer: Any, image_processor: Any, chat_template: str) -> Any:
    """官方 `Qwen3VLProcessor` 类的一个实例，只供比对（测试与冒烟 --tiny）；适配器本身从不用它（同 internvl35.reference_processor）。

    开发机没有 torchvision，视频处理器构造不出来，ProcessorMixin 的类型检查又不接受别的视频处理器。这里在构造期间临时跳过
    那一项类型检查、用 `BaseVideoProcessor()` 占位（本项目不送视频）；图像与文本路径——`__call__`、占位符展开、
    mm_token_type_ids、分词——都是官方代码。服务器上的比对对象是 load() 留下的 hf_processor（AutoProcessor）。
    """
    from unittest import mock

    from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
    from transformers.processing_utils import ProcessorMixin
    from transformers.video_processing_utils import BaseVideoProcessor

    with mock.patch.object(ProcessorMixin, "check_argument_for_proper_class", lambda self, name, arg: type(arg)):
        return Qwen3VLProcessor(image_processor=image_processor, tokenizer=tokenizer,
                                video_processor=BaseVideoProcessor(), chat_template=chat_template)


__all__ = [
    "AdapterDeviceError",
    "BOX_KEY",
    "GDNKernelError",
    "GDN_FUNCTIONS",
    "PIXELS_PER_LLM_TOKEN",
    "QUERY_PLACEHOLDER",
    "QWEN35_MODELING",
    "Qwen35Adapter",
    "ReservedRowTokens",
    "THINK_SEPARATOR",
    "build_tiny_model",
    "format_boxes",
    "json_box_object",
    "json_box_text",
    "kernel_implementation",
    "load_pinned_components",
    "make_tiny_adapter",
    "mrope_rows",
    "native_box_to_xyxy",
    "parse_boxes",
    "parse_one_box",
    "reference_processor",
    "tiny_config",
    "torch_gdn",
    "vision_patch_row_major_order",
]
