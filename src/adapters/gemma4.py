"""Gemma 4 家族适配器（简报 §8.1；DEV-0004）。结构照 qwen35.py，差异按下面各条处理。

模型事实一律以 `docs/model_notes.md` §6（RUN-0002 实测）与 D13 为准；运行时从 tokenizer / config 读取并断言，
不写死 token id，不写死层号（简报 §4 第 6 条）：

- 加载：`AutoModelForImageTextToText` → `Gemma4ForConditionalGeneration`，`attn_implementation="sdpa"`，eval；
  processor 由 `AutoProcessor` 加载；一律带注册表里的 revision（D9-2）。
- 视觉 token（D13-5，§6.3）：构造时给定 `max_soft_tokens`，只接受 70/140/280/560/1120，每次调用图像处理器都显式传。
  图像处理器按宽高比缩放（高、宽都是 patch 16 × pool 3 = 48 的倍数，不超过 max_soft_tokens×9 个 patch），
  网格 (h, w) 由 `image_position_ids` 推出：第 0 维是列（x），第 1 维是行（y），网格 = (行 patch 数/3, 列 patch 数/3)。
  序列里的视觉 token 按行优先排列：pooler 的核下标 = 列块 + w × 行块（modeling_gemma4.Gemma4VisionPooler）。
  口径对象是 image_policy.SoftTokenBudget（DEV-0011；E1 用 280，D28-3）。ImageTokenInfo.n_pixels = 有效 patch 数 × 16²
  = (h·48)·(w·48)，即缩放后的像素数（每个视觉 token 对应 48×48 px，D21-3）。
- 序列布局（§6.2–6.3）：`<bos><|turn>user\\n{内容}<turn|>\\n`；每张图是 `<|image>` + h·w 个 `<|image|>` + `<image|>`，
  两张图直接相连。ImageTokenInfo.span 只覆盖 `<|image|>` 那一段（长度恰为 h·w），不含前后的 `<|image>`、`<image|>`。
- 思考开关（D13-2，§6.2）：`enable_thinking` 直接传给 `apply_chat_template`。与 Qwen 不同，Gemma 的开关不在
  assistant 头里，而是在 `<bos>` 之后插入一个 system 轮 `<|turn>system\\n<|think|>\\n<turn|>\\n`；生成提示
  `<|turn>model\\n` 两种开关相同。build_inputs 按 enable_thinking=False 渲染（不含 system 轮）；generate 在
  enable_thinking=True 时把 system 轮的 token 插到 `<bos>` 之后，再接生成提示。插入的 token 与生成提示都在 attach 时
  从模板的两次渲染差分得到，测试与冒烟逐个核对"拼出来的 ids == apply_chat_template(..., enable_thinking=X) 的分词"。
- 思考段（固定 revision 的 tokenizer_config.json 里 response_template：open `<|channel>thought\\n`、close `<channel|>`；
  聊天模板的 strip_thinking 按 `<|channel>`…`<channel|>` 剥离）：text 取最后一个 `<channel|>` 之后的内容。
- 生成（D13-3）：显式 greedy（仓库的 generation_config 是 do_sample=True, top_k=64, top_p=0.95）；
  停止 token 是 `<turn|>` 与 `<eos>`，id 从 tokenizer 查。generate 与 generate_batch 走同一条路径（src/adapters/batching.py；
  DEV-0011）：左填充用 `<pad>`，各行的图按行的顺序拼进同一个 pixel_values（每张图都按 max_soft_tokens 补到同样的 patch 数），
  一维位置由 HF 按 attention_mask 推出（cumsum − 1），E2B/E4B 是纯因果注意力（use_bidirectional_attention=None）。
- 位置（§6.4）：LLM 位置是一维的 0..L-1，没有 M-RoPE。查询 token 只支持 seq1d；grid2d 直接报错（见 inject_query_tokens）。
- 新增 token（D13-1）：没有超出 tokenizer 的预留行，复用 6227 个 `<unusedN>`（普通词表项）。grid token 按 id 从小到大取，
  查询占位符取 id 最大的那一个（与 grid token 的取用顺序无关，grid token 的 id 不随是否注入过查询而变）。
  先 `add_tokens(..., special_tokens=True)` 注册：注册后仍是原 id、len(tokenizer) 不变、不 resize。
  初始化两处，都写成基础词表（全部 id 去掉 `<unusedN>` 与 tokenizer 的特殊 token）行的精确均值：
  主嵌入行（lm_head 与它共享，只写一次）与 PLE 的 `embed_tokens_per_layer` 对应行。
- E3 的训练侧接口（DEV-0020）：见 src/adapters/training.py 与 src/adapters/checkpoint.py；本模块只提供家族钩子
  （logits 含 final_logit_softcapping；token 表另记 PLE 行 per_layer）。
- vision_features 与第 0 层的关系（按 transformers 5.17 `modeling_gemma4` 源码，测试与冒烟钉住）：
  **在图像位置上逐位相同**。理由：`Gemma4Model.forward` 先用 `embed_tokens`（`Gemma4TextScaledWordEmbedding`，
  输出 = 权重行 × embed_scale = sqrt(hidden_size)）查 `llm_input_ids`（图像位置换成 pad），再用 `masked_scatter`
  把 `embed_vision` 的输出原样写进图像位置，所以 embed_scale 只作用在文本 token 上，不作用在视觉特征上；
  PLE（per-layer input）不属于第 0 层，它在每个 decoder 层内部、MLP 之后才加进残差，只影响第 1 层起的输出。
  另外，PLE 并不是"不作用在图像位置上"：图像位置的 per-layer input =
  (RMSNorm(per_layer_model_projection(视觉特征)/sqrt(hidden)) + embed_tokens_per_layer[pad] × sqrt(ple_dim)) / sqrt(2)，
  其中 token 身份那一半用的是 pad（id 0）的 PLE 行。
- 层号（base.py 的层号约定）：就是 HF `output_hidden_states` 元组的下标，共 num_layers()+1 个（E2B 为 36，§6.1）。
  0 = 视觉特征并入之后的嵌入输出，i = 第 i 个 decoder 层之后，最后一个下标取最终归一化之后的值。
- 坐标（D13-4，§6.5）：`box_2d` 是 [y1, x1, y2, x2]，0–1000，**y 在前**；`box2d_to_xyxy` 换算成内部绝对像素 xyxy。
  从回答里解析框见 parse_boxes / parse_one_box（模块级函数，也是适配器方法；通用 JSON 规则在 src/parsing，DEV-0006）。

处理器的用法同 qwen35.py：本适配器只用 tokenizer、image_processor、chat_template 三个部件，自己完成图像占位符的展开与
mm_token_type_ids（做法与 `Gemma4Processor` / `ProcessorMixin` 相同）。开发机没有 torchvision，`Gemma4Processor`
构造不出来（它的视频部件需要 torchvision），测试用 PIL 后端的 `Gemma4ImageProcessorPil`；服务器上 load() 用官方
`AutoProcessor`，把它留在 `hf_processor` 上，冒烟脚本逐项比对两者的输出（官方 processor 要求图按样本嵌套传入，D13-6）。

冒烟：scripts/smoke_adapter.py 按家族分派到本模块的 `smoke_setup`，检查项编号与 Qwen 相同，见文件末尾。
"""

from __future__ import annotations

import contextlib
import copy
import math
import re
from typing import Any, Iterable, NamedTuple, Sequence

import numpy as np
import torch
from PIL import Image

from src.adapters import batching, registry
from src.adapters.image_policy import SoftTokenBudget
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
    check_grid_token_names,
    resolve_layer_keys,
)

# 与 Qwen 共用：入口断言的异常类型（调用方跨家族只需捕获一种），以及 allowed_token_ids 的 logits 限制
# （简报 §12：限制 logits 的实现处处一致）。二者目前定义在 qwen35.py；挪到公共模块要改 base.py，不在本任务范围内。
from src.adapters.qwen35 import AdapterDeviceError, _AllowedTokensProcessor
from src.executors.geometry import norm1000_to_abs
from src.parsing import BoxParse, OneBoxParse, one_box, parse_json_boxes

# ---------------------------------------------------------------------------
# Gemma 4 家族事实（字符串来自 docs/model_notes.md §6.1–6.3 与固定 revision 的 tokenizer_config.json；
# id 一律在 attach 时从 tokenizer 查）
# ---------------------------------------------------------------------------

BOS = "<bos>"
EOS = "<eos>"
PAD = "<pad>"
TURN_OPEN = "<|turn>"
TURN_CLOSE = "<turn|>"
IMAGE_TOKEN = "<|image|>"  # 模板里每张图渲染成一个它，展开后是 h·w 个
BOI = "<|image>"  # 每张图的开始
EOI = "<image|>"  # 每张图的结束
THINK = "<|think|>"  # enable_thinking=True 时出现在插入的 system 轮里
CHANNEL_OPEN = "<|channel>"  # 思考段开始（tokenizer 的 soc_token）
CHANNEL_CLOSE = "<channel|>"  # 思考段结束（eoc_token）
THOUGHT_OPEN = CHANNEL_OPEN + "thought\n"  # response_template.fields.thinking.open（固定 revision 的 tokenizer_config.json）
# 停止 token（D13-3）
STOP_TOKENS: tuple[str, ...] = (TURN_CLOSE, EOS)
# attach 时逐个核对"分词成单个 token"的特殊 token
_NAMED_TOKENS: tuple[str, ...] = (
    BOS, EOS, PAD, TURN_OPEN, TURN_CLOSE, IMAGE_TOKEN, BOI, EOI, THINK, CHANNEL_OPEN, CHANNEL_CLOSE,
)

# max_soft_tokens 的可选档位（D13-5；model_notes §6.3，源码 `_SUPPORTED_SOFT_TOKENS`；tests 与二者核对）
MAX_SOFT_TOKENS_CHOICES: tuple[int, ...] = (70, 140, 280, 560, 1120)

# 复用的占位 token（D13-1）：词表里形如 <unusedN> 的普通词表项
UNUSED_TOKEN_RE = re.compile(r"<unused\d+>")

# generate / generate_batch 的 gen_kwargs 不能覆盖的参数（适配器自己控制）
_RESERVED_GEN_KWARGS = frozenset({"input_ids", "attention_mask", "pixel_values", "image_position_ids", "mm_token_type_ids",
                                  "position_ids", "inputs_embeds", "per_layer_inputs", "max_new_tokens", "eos_token_id",
                                  "pad_token_id"})

# 小型随机模型：从哪个登记模型的真实 config 缩出来
TINY_SOURCE_KEY = "gemma-4-e2b-it"

_SENTINEL = "VISINTINSTRUCTIONSLOT"
_MEAN_CHUNK_BYTES = 1 << 28  # 求基础行均值时每块 float64 的字节上限


def _canonical_device(device: str | torch.device) -> torch.device:
    """把 "cuda" 这类不带下标的写法规范成张量实际报告的设备（如 cuda:0），便于逐项比较。"""
    return torch.empty(0, device=device).device


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


def grid_from_position_ids(position_ids: torch.Tensor, pooling_kernel_size: int) -> tuple[int, int]:
    """一张图的 `image_position_ids`（[max_patches, 2]，每行 (x, y)，填充行为 (-1, -1)）→ LLM 视觉 token 网格 (h, w)。

    第 0 维是列 x，第 1 维是行 y（model_notes §6.3）。逐项核对：有效行在前、填充行在后；有效行恰为
    x 变化最快的完整网格（与图像处理器的 meshgrid(indexing="xy") 相同）；行、列 patch 数都是 pooling_kernel_size 的倍数。
    """
    pos = position_ids.detach().cpu()
    if pos.dim() != 2 or pos.shape[1] != 2:
        raise ValueError(f"image_position_ids 的形状应为 [max_patches, 2]，收到 {tuple(pos.shape)}")
    pad = (pos == -1).all(dim=1)
    valid = (pos >= 0).all(dim=1)
    if not bool((pad | valid).all()):
        raise RuntimeError("image_position_ids 里有一半是 -1 的行")
    n_valid = int(valid.sum())
    if n_valid == 0 or not bool(valid[:n_valid].all()) or bool(valid[n_valid:].any()):
        raise RuntimeError("image_position_ids 的有效行不是连续的前缀")
    xy = pos[:n_valid]
    n_x, n_y = int(xy[:, 0].max()) + 1, int(xy[:, 1].max()) + 1
    want = torch.stack([torch.arange(n_x).repeat(n_y), torch.arange(n_y).repeat_interleave(n_x)], dim=1)
    if n_x * n_y != n_valid or not torch.equal(xy, want):
        raise RuntimeError(f"有效 patch 位置不是 {n_y}×{n_x} 的行优先网格（x 变化最快）")
    k = int(pooling_kernel_size)
    if n_x % k or n_y % k:
        raise RuntimeError(f"patch 网格 {n_y}×{n_x} 不是 pooling_kernel_size={k} 的倍数")
    return n_y // k, n_x // k


def box2d_to_xyxy(box_2d: Sequence[float] | np.ndarray, *, width: float, height: float) -> np.ndarray:
    """Gemma 原生框 `box_2d` → 内部绝对像素 xyxy（浮点，不取整；D13-4、model_notes §6.5）。

    box_2d = [y1, x1, y2, x2]，0–1000，**y 在前**（官方文档的解析代码 `y1, x1, y2, x2 = ...`）。
    换算：先换成 x 在前的 [x1, y1, x2, y2]，再按 src/executors/geometry.norm1000_to_abs 缩放
    （x = x_norm / 1000 × W，y = y_norm / 1000 × H；官方示例最后取整，内部格式是浮点，这里不取整）。
    接受形状 (4,) 或 (..., 4)。只做线性换算：不裁剪到图内、不交换颠倒的坐标（这些由解析器 / 指标决定，不在这里猜）。
    长度不是 4、含非数值 / 布尔 / 非有限值时报错。
    """
    if isinstance(box_2d, (str, bytes)):
        raise TypeError("box_2d 必须是数值序列")
    raw = np.asarray(box_2d, dtype=object)
    if raw.size and any(isinstance(v, (bool, np.bool_)) for v in raw.ravel()):
        raise TypeError(f"box_2d 不能含布尔值：{box_2d!r}")
    try:
        arr = np.asarray(box_2d, dtype=np.float64)
    except (TypeError, ValueError) as e:
        raise TypeError(f"box_2d 必须是数值序列：{box_2d!r}") from e
    if arr.ndim == 0 or arr.shape[-1] != 4:
        raise ValueError(f"box_2d 的最后一维必须是 4（[y1, x1, y2, x2]），收到形状 {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"box_2d 含非有限值：{box_2d!r}")
    xyxy_norm = arr[..., [1, 0, 3, 2]]
    return norm1000_to_abs(xyxy_norm, width=width, height=height)


# 原生框的键：[y1, x1, y2, x2]，0–1000，y 在前（§6.5 官方文档与零样本实测；D13-4）
BOX_KEY = "box_2d"


def parse_boxes(text: str, *, width: float, height: float) -> BoxParse:
    """解析 Gemma 4 的原生框，换算成内部绝对像素 xyxy（DEV-0006）。width、height 是原图的宽高（按原图归一化，§6.5）。

    原生格式：JSON，形如 `[{"box_2d": [y1, x1, y2, x2], "label": "…"}, …]`，**y 在前**，0–1000，常包在 ```json 代码块里
    （§6.5 的两条实测都是）。换算走 box2d_to_xyxy。接受单个对象与对象的列表，label 可缺，其他键忽略；另外接受裸的四元组
    或它的列表，同样按 y 在前、0–1000 解释。其余规则与 Qwen 相同（src.parsing.parse_json_boxes 与 extract_json）：
    没有 JSON → no_box；多个代码块或多个顶层 JSON → ambiguous；JSON 坏了、对象缺 box_2d、值不是 4 个有限数 → malformed；
    坐标 < 0 或 > 1000 → out_of_range；y2 < y1 或 x2 < x1 → inverted；空列表 [] 成功、零个框。整数或小数都接受。
    """
    return parse_json_boxes(text, key=BOX_KEY, to_xyxy=box2d_to_xyxy, width=width, height=height)


def parse_one_box(text: str, *, width: float, height: float) -> OneBoxParse:
    """只要一个框（如 RefCOCO 的 REC）：规则见 src.parsing.one_box，与适配器方法 parse_one_box 相同。"""
    return one_box(parse_boxes(text, width=width, height=height))


class Gemma4Adapter(CheckpointSupport, TrainingSupport, ModelAdapter):
    """Gemma 4（E2B-it / E4B-it）适配器。ModelInputs 是单条样本；generate_batch 左填充拼批（DEV-0011）。vLLM 另开任务。

    设备在构造时给定，dtype 在 load 时给定。进入任何模型算子之前，入口断言检查输入与模型的 device/dtype，
    不一致就抛 AdapterDeviceError 并报出两个值。
    """

    family = registry.FAMILY_GEMMA4

    def __init__(self, *, device: str | torch.device, max_soft_tokens: int):
        if (
            isinstance(max_soft_tokens, bool)
            or not isinstance(max_soft_tokens, (int, np.integer))
            or int(max_soft_tokens) not in MAX_SOFT_TOKENS_CHOICES
        ):
            raise ValueError(f"max_soft_tokens 只能取 {MAX_SOFT_TOKENS_CHOICES} 之一（D13-5），收到 {max_soft_tokens!r}")
        self._max_soft_tokens = int(max_soft_tokens)
        self._device = _canonical_device(device)
        self._dtype: torch.dtype | None = None
        self._model: Any = None
        self._tok: Any = None
        self._ip: Any = None
        self._template: str | None = None
        self._model_key: str | None = None
        self.hf_processor: Any = None  # 只有 load() 设置：官方 AutoProcessor，供冒烟脚本比对
        self._ids: dict[str, int] = {}
        self._special_ids: frozenset[int] = frozenset()
        self._unused_ids: tuple[int, ...] = ()
        self._unused_surface: dict[int, str] = {}
        self._base_idx: torch.Tensor | None = None
        self._patch = 0
        self._pool = 0
        self._prefix: dict[bool, tuple[int, ...]] = {}
        self._header: tuple[int, ...] = ()
        self._taken: set[int] = set()
        self._n_grid_tokens = 0
        self._grid_token_ids: dict[str, int] = {}
        self._grid_token_surfaces: dict[str, str] = {}
        self._query_token_id: int | None = None
        self._query_n: int | None = None
        self._query_mode: PositionMode | None = None

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
    def max_soft_tokens(self) -> int:
        """每张图的视觉 token 预算（构造时给定，每次调用图像处理器都显式传）。实际个数 = h·w ≤ 它。"""
        return self._max_soft_tokens

    @property
    def image_policy(self) -> SoftTokenBudget:
        """构造时给定的图像口径（src/adapters/image_policy.py）。"""
        return SoftTokenBudget(self._max_soft_tokens)

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
        """lm_head 与主嵌入共享（attach 时断言；E2B/E4B 都是，model_notes §6.1）。"""
        self._require_loaded()
        return True

    @property
    def n_base_vocab(self) -> int:
        """新行初始化取均值的基础行数：全部 id 去掉 `<unusedN>` 与 tokenizer 的特殊 token（D13-1）。"""
        self._require_loaded()
        return int(self._base_idx.numel())

    @property
    def unused_token_ids(self) -> tuple[int, ...]:
        """可复用的 `<unusedN>` 的 id，从小到大（attach 时从词表查）。"""
        self._require_loaded()
        return self._unused_ids

    @property
    def grid_token_ids(self) -> dict[str, int]:
        """grid token：调用方给的名字 → id。"""
        return dict(self._grid_token_ids)

    @property
    def grid_token_surfaces(self) -> dict[str, str]:
        """grid token：调用方给的名字 → 它在文本里的样子（复用的 `<unusedN>`；decode 出来的就是它）。"""
        return dict(self._grid_token_surfaces)

    @property
    def n_grid_tokens(self) -> int:
        """累计已加的 grid token 个数（计入 300 的预算；查询占位符不计）。"""
        return self._n_grid_tokens

    @property
    def query_token_id(self) -> int | None:
        return self._query_token_id

    @property
    def query_token_surface(self) -> str | None:
        """查询占位符在文本里的样子（id 最大的 `<unusedN>`）；尚未注入查询时为 None。"""
        return None if self._query_token_id is None else self._unused_surface[self._query_token_id]

    def token_id(self, token: str) -> int:
        """attach 时查好的特殊 token id（如 TURN_CLOSE、CHANNEL_CLOSE）。"""
        self._require_loaded()
        return self._ids[token]

    # ------------------------------------------------------------------
    # 加载
    # ------------------------------------------------------------------

    def load(self, model_id: str, dtype: Any) -> None:
        """按注册键加载（带固定 revision）。dtype 由调用方给出，如 torch.bfloat16。"""
        entry = registry.get_model(model_id)
        if entry.family != registry.FAMILY_GEMMA4:
            raise ValueError(f"{model_id} 属于 {entry.family} 家族，不是 {registry.FAMILY_GEMMA4}")
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

        tokenizer 必须是没有注册过 `<unusedN>` 的原样 tokenizer（add_grid_tokens / inject_query_tokens 会改它；
        需要彼此独立的适配器时传各自的副本）。模型若不在构造时给定的设备上，会被整体搬过去；dtype 不做转换，
        要求全部浮点参数同一个 dtype。
        """
        if self._model is not None:
            raise RuntimeError("这个适配器已经挂过模型；新模型请新建适配器")
        if type(model).__name__ != "Gemma4ForConditionalGeneration":
            raise TypeError(f"期望 Gemma4ForConditionalGeneration，收到 {type(model).__name__}")
        if not isinstance(chat_template, str) or not chat_template:
            raise ValueError("chat_template 为空")
        cfg = model.config
        tc, vc = cfg.text_config, cfg.vision_config
        if vc is None:
            raise ValueError("config 没有 vision_config")
        attn = getattr(cfg, "_attn_implementation", None)
        if attn != "sdpa":
            raise ValueError(f"attn_implementation 必须是 sdpa，收到 {attn!r}")
        if tc.use_bidirectional_attention is not None:
            raise ValueError(
                f"use_bidirectional_attention={tc.use_bidirectional_attention!r}：本适配器按 E2B/E4B 的纯因果注意力写"
                "（model_notes §6.1 为 None）"
            )
        ple_dim = int(tc.hidden_size_per_layer_input or 0)
        if ple_dim <= 0:
            raise ValueError("config 没有 PLE（hidden_size_per_layer_input）；本适配器按 E2B/E4B 写，要初始化 PLE 行（D13-1）")

        # 视觉几何：config 与图像处理器一致
        for ip_attr, cfg_val in (("patch_size", vc.patch_size), ("pooling_kernel_size", vc.pooling_kernel_size)):
            if getattr(image_processor, ip_attr, None) != cfg_val:
                raise ValueError(
                    f"图像处理器 {ip_attr}={getattr(image_processor, ip_attr, None)} 与 vision_config 的 {cfg_val} 不一致"
                )

        # 特殊 token：每个都必须分词成单个 token，并与 config 登记的 id 一致
        ids: dict[str, int] = {}
        for s in _NAMED_TOKENS:
            enc = tokenizer(s, add_special_tokens=False)["input_ids"]
            if len(enc) != 1 or tokenizer.convert_tokens_to_ids(s) != enc[0]:
                raise ValueError(f"{s!r} 没有分词成单个已知 token：{enc}")
            ids[s] = enc[0]
        for s, cfg_name in ((IMAGE_TOKEN, "image_token_id"), (BOI, "boi_token_id"), (EOI, "eoi_token_id")):
            if getattr(cfg, cfg_name) != ids[s]:
                raise ValueError(f"config.{cfg_name}={getattr(cfg, cfg_name)} 与 tokenizer 的 {s} id {ids[s]} 不一致")
        if tokenizer.bos_token != BOS:
            raise ValueError(f"tokenizer.bos_token={tokenizer.bos_token!r}，期望 {BOS!r}")
        if tc.pad_token_id != ids[PAD] or tokenizer.pad_token_id != ids[PAD]:
            raise ValueError(f"pad id 不一致：text_config {tc.pad_token_id}，tokenizer {tokenizer.pad_token_id}，{PAD} {ids[PAD]}")
        # 思考段标记：与 tokenizer 自带的 response_template 一致（固定 revision 的 tokenizer_config.json）
        rt = getattr(tokenizer, "response_template", None)
        if rt is not None:
            thinking = (rt.get("fields") or {}).get("thinking") or {}
            if thinking.get("open") != THOUGHT_OPEN or thinking.get("close") != CHANNEL_CLOSE:
                raise ValueError(f"tokenizer.response_template 的思考段标记 {thinking} 与 {THOUGHT_OPEN!r}/{CHANNEL_CLOSE!r} 不一致")

        # 行数：嵌入 = lm_head = PLE = vocab = len(tokenizer)，没有超出 tokenizer 的行（§6.1）
        emb = model.get_input_embeddings()
        head = model.get_output_embeddings()
        ple = model.get_per_layer_input_embeddings()
        n_rows = emb.weight.shape[0]
        if head.weight.shape[0] != n_rows or tc.vocab_size != n_rows:
            raise ValueError(f"嵌入 {n_rows} 行、lm_head {head.weight.shape[0]} 行、text_config.vocab_size {tc.vocab_size} 不一致")
        if ple.weight.shape[0] != n_rows or tc.vocab_size_per_layer_input != n_rows:
            raise ValueError(f"PLE {ple.weight.shape[0]} 行、vocab_size_per_layer_input {tc.vocab_size_per_layer_input} 与嵌入 {n_rows} 行不一致")
        if ple.weight.shape[1] != tc.num_hidden_layers * ple_dim:
            raise ValueError(f"PLE 宽度 {ple.weight.shape[1]} ≠ 层数 {tc.num_hidden_layers} × {ple_dim}")
        if len(tokenizer) != n_rows:
            raise ValueError(f"len(tokenizer)={len(tokenizer)} 与嵌入行数 {n_rows} 不一致（Gemma 没有预留行，§6.1）")
        tied = head.weight is emb.weight
        if tied != (head.weight.data_ptr() == emb.weight.data_ptr()):
            raise ValueError("lm_head 与嵌入的共享状态不一致（对象与存储判断矛盾）")
        if not tied or not bool(getattr(cfg, "tie_word_embeddings", False)):
            raise ValueError("lm_head 与主嵌入必须共享（D13-1；E2B/E4B 的 config 都是 tie_word_embeddings=True）")

        # 可复用的 <unusedN> 与基础行
        vocab = tokenizer.get_vocab()
        special = frozenset(int(i) for i in tokenizer.added_tokens_decoder)
        unused = sorted((int(i), s) for s, i in vocab.items() if UNUSED_TOKEN_RE.fullmatch(s))
        if not unused:
            raise ValueError("词表里没有 <unusedN>")
        reg = [s for i, s in unused if i in special]
        if reg:
            raise ValueError(f"tokenizer 里已经注册过 {len(reg)} 个 <unusedN>（如 {reg[:3]}）：请传入未改动过的 tokenizer")
        if unused[-1][0] >= n_rows:
            raise ValueError(f"<unusedN> 的 id {unused[-1][0]} 超出嵌入行数 {n_rows}")
        base_mask = torch.ones(n_rows, dtype=torch.bool)
        base_mask[[i for i, _ in unused]] = False
        base_mask[sorted(special)] = False

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
        self._special_ids = special
        self._unused_ids = tuple(i for i, _ in unused)
        self._unused_surface = dict(unused)
        self._base_idx = base_mask.nonzero().squeeze(1).to(self._device)
        self._patch = int(vc.patch_size)
        self._pool = int(vc.pooling_kernel_size)
        self._base_fingerprint = base_fingerprint(model)  # 检查点核对 base 用（DEV-0020）

        # 模板自检：思考开关只在 <bos> 之后插一个 system 轮，生成提示两种开关相同。失败就撤销挂载。
        try:
            self._derive_prompt_parts()
        except Exception:
            self._model = None
            self._prefix, self._header = {}, ()
            raise

    def _derive_prompt_parts(self) -> None:
        """从模板的渲染差分出 enable_thinking=True 时插入的 system 轮 token，以及生成提示的 token（§6.2）。"""
        probe = [{"type": "text", "text": "x"}]
        renders: dict[bool, tuple[str, str]] = {}
        for flag in (False, True):
            plain = self._render(probe, enable_thinking=flag)
            full = self._render(probe, add_generation_prompt=True, enable_thinking=flag)
            if not plain.startswith(BOS) or not plain.endswith(TURN_CLOSE + "\n"):
                raise ValueError(f"user 轮渲染不是 {BOS!r}…{TURN_CLOSE!r}+换行 的形式：{plain[:40]!r}…{plain[-20:]!r}")
            if not full.startswith(plain):
                raise ValueError("加了生成提示的渲染不以不加时的渲染为前缀")
            renders[flag] = (plain, full[len(plain) :])
        (off, header_off), (on, header_on) = renders[False], renders[True]
        if header_off != header_on or not header_off.startswith(TURN_OPEN):
            raise ValueError(f"生成提示随思考开关变化或不以 {TURN_OPEN!r} 开头：{header_off!r} / {header_on!r}（与 §6.2 不符）")
        body = off[len(BOS) :]
        if not on.endswith(body) or len(on) <= len(off):
            raise ValueError("enable_thinking=True 的渲染不是在 <bos> 之后插入一段、其余不变（与 §6.2 不符）")
        system_turn = on[len(BOS) : len(on) - len(body)]
        if not (system_turn.startswith(TURN_OPEN) and THINK in system_turn and system_turn.endswith(TURN_CLOSE + "\n")):
            raise ValueError(f"插入的 system 轮 {system_turn!r} 不是 {TURN_OPEN}…{THINK}…{TURN_CLOSE} 的形式")
        header = self._encode(header_off)
        prefix = self._encode(system_turn)
        bos, body_ids = self._ids[BOS], self._encode(body)
        for flag, pre in ((False, []), (True, prefix)):
            plain, h = renders[flag]
            if self._encode(plain + h) != [bos] + pre + body_ids + header:
                raise RuntimeError("system 轮 / 生成提示单独分词后拼接，与整段分词不一致")
        self._prefix = {False: (), True: tuple(prefix)}
        self._header = tuple(header)

    # ------------------------------------------------------------------
    # 输入
    # ------------------------------------------------------------------

    def prepare_image(self, image: Any) -> Image.Image:
        """把一张图转成 RGB 的 PIL 图，不缩放：缩放由图像处理器按 max_soft_tokens 做（保持宽高比，§6.3）。"""
        if isinstance(image, Image.Image):
            return image if image.mode == "RGB" else image.convert("RGB")
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(f"numpy 图必须是 uint8 的 (H, W, 3)，收到 dtype={image.dtype} shape={image.shape}")
            return Image.fromarray(image)
        raise TypeError(f"图像必须是 PIL.Image 或 numpy 数组，收到 {type(image).__name__}")

    def build_inputs(self, images: Sequence[Any], instruction: str, order: Order) -> ModelInputs:
        """组装单条输入。区间都是 input_ids 里的 token 下标（半开）。

        user 内容的顺序：instruction_first → [指令, 图1, 图2, …]；image_first → [图1, …, 指令]。
        已调用 inject_query_tokens 时，查询 token 放在 user 内容最后一项之后、`<turn|>` 之前。
        按 enable_thinking=False 渲染到 user 轮结束（`<turn|>\\n`），不含生成提示；思考开关由 generate 处理。
        """
        self._require_loaded()
        if order not in ORDERS:
            raise ValueError(f"order 必须是 {ORDERS} 之一，收到 {order!r}")
        if not isinstance(instruction, str):
            raise TypeError(f"instruction 必须是 str，收到 {type(instruction).__name__}")
        self._check_instruction(instruction)
        if isinstance(images, (Image.Image, np.ndarray)):
            raise TypeError("images 必须是图像的序列（单张图也要放进列表）")
        pil = [self.prepare_image(im) for im in images]
        n_img = len(pil)

        # 1. 渲染：指令槽先放哨兵，确认模板对指令原样输出（Gemma 模板会 trim 每段文本）
        text = self._render(self._content(n_img, instruction, order), enable_thinking=False)
        sent_text = self._render(self._content(n_img, _SENTINEL, order), enable_thinking=False)
        if sent_text.count(_SENTINEL) != 1:
            raise RuntimeError("哨兵在渲染结果里出现的次数不是 1")
        cut = sent_text.index(_SENTINEL)
        prefix, suffix = sent_text[:cut], sent_text[cut + len(_SENTINEL) :]
        if text != prefix + instruction + suffix:
            raise ValueError("聊天模板改写了指令文本（例如首尾空白被 trim），指令区间无法精确定位；请去掉首尾空白")
        if not text.startswith(BOS) or not text.endswith(TURN_CLOSE + "\n"):
            raise RuntimeError(f"渲染不是 {BOS!r}…{TURN_CLOSE!r}+换行 的形式")

        # 2. 图像：显式传 max_soft_tokens，逐张从 image_position_ids 推网格并核对
        tensors: dict[str, torch.Tensor] = {}
        grids: list[tuple[int, int]] = []
        if n_img:
            enc = self._ip(images=pil, max_soft_tokens=self._max_soft_tokens, return_tensors="pt")
            pv, pos = enc["pixel_values"], enc["image_position_ids"]
            max_patches = self._max_soft_tokens * self._pool * self._pool
            if tuple(pv.shape) != (n_img, max_patches, 3 * self._patch * self._patch) or tuple(pos.shape) != (n_img, max_patches, 2):
                raise RuntimeError(
                    f"pixel_values {tuple(pv.shape)} / image_position_ids {tuple(pos.shape)} 与 {n_img} 张图、"
                    f"max_soft_tokens={self._max_soft_tokens} 不符：处理器没有用给定的档位"
                )
            grids = [grid_from_position_ids(pos[i], self._pool) for i in range(n_img)]
            n_soft = enc.get("num_soft_tokens_per_image")
            if n_soft is not None and [int(v) for v in n_soft] != [h * w for h, w in grids]:
                raise RuntimeError(f"num_soft_tokens_per_image={list(n_soft)} 与网格 {grids} 不一致")
            if any(h * w > self._max_soft_tokens for h, w in grids):
                raise RuntimeError(f"网格 {grids} 超出 max_soft_tokens={self._max_soft_tokens}")
            tensors["pixel_values"] = pv.to(device=self._device, dtype=self._dtype)
            tensors["image_position_ids"] = pos.to(device=self._device, dtype=torch.long)

        # 3. 展开图像占位符并分词；指令两侧分别分词，拼接必须与整段分词逐个相同
        counts = [h * w for h, w in grids]
        n_pre = prefix.count(IMAGE_TOKEN)
        if n_pre + suffix.count(IMAGE_TOKEN) != n_img:
            raise RuntimeError("模板里的图像占位符个数与图数不一致")
        pre_exp = self._expand(prefix, counts[:n_pre])
        post_exp = self._expand(suffix, counts[n_pre:])
        ids_pre, ids_ins, ids_post = self._encode(pre_exp), self._encode(instruction), self._encode(post_exp)
        ids = ids_pre + ids_ins + ids_post
        if ids != self._encode(pre_exp + instruction + post_exp):
            raise ValueError("指令与相邻模板文本在分词时合并，指令区间无法精确定位")
        if ids[0] != self._ids[BOS] or ids.count(self._ids[BOS]) != 1:
            raise RuntimeError("序列不是以唯一的一个 <bos> 开头")
        instruction_span = Span(len(ids_pre), len(ids_pre) + len(ids_ins))

        # 4. 图像区间：<|image|> 的连续段，个数、长度、前后标记逐项核对
        runs = _runs(ids, self._ids[IMAGE_TOKEN])
        if [e - s for s, e in runs] != counts:
            raise RuntimeError(f"图像 token 段 {runs} 与每张图的 token 数 {counts} 不一致")
        infos = []
        side = self._patch * self._pool  # 每个视觉 token 对应缩放后图像的边长（48 px）
        for (s, e), (gh, gw) in zip(runs, grids):
            if ids[s - 1] != self._ids[BOI] or ids[e] != self._ids[EOI]:
                raise RuntimeError(f"图像段 [{s}, {e}) 前后不是 {BOI} / {EOI}")
            infos.append(ImageTokenInfo(Span(s, e), (gh, gw), n_pixels=gh * side * gw * side))

        # 5. 查询区间
        query_span = None
        if self._query_n is not None:
            qruns = _runs(ids, self._query_token_id)
            if len(qruns) != 1 or qruns[0][1] - qruns[0][0] != self._query_n:
                raise RuntimeError(f"查询 token 段 {qruns} 与登记的 {self._query_n} 个不一致")
            qs, qe = qruns[0]
            if ids[qe] != self._ids[TURN_CLOSE] or qe + 2 != len(ids):
                raise RuntimeError("查询 token 没有紧挨在 <turn|>\\n 之前")
            if qs < instruction_span.end or any(qs <= info.span.end for info in infos):
                raise RuntimeError("查询 token 不在 user 内容的最后")
            query_span = Span(qs, qe)

        input_ids = torch.tensor([ids], dtype=torch.long, device=self._device)
        tensors["input_ids"] = input_ids
        tensors["attention_mask"] = torch.ones_like(input_ids)
        # 与 ProcessorMixin.create_mm_token_type_ids 相同：<|image|> 记 1（<|image>、<image|> 记 0），其余 0（int64）
        tensors["mm_token_type_ids"] = (input_ids == self._ids[IMAGE_TOKEN]).to(torch.long)
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
        """一维位置 (1, L) = 0..L-1（§6.4：图像 token 与文本一样逐个加 1）。查询 token 也按顺序编号（seq1d）。

        与 Gemma4Model.forward 在不传 position_ids 时自己生成的 arange 相同；冒烟逐位比对"传它"与"不传"。
        """
        self._require_loaded()
        ids = inputs.tensors["input_ids"]
        if ids.dim() != 2 or ids.shape[0] != 1:
            raise ValueError(f"只支持单条样本，input_ids 形状 {tuple(ids.shape)}")
        return torch.arange(ids.shape[1], device=ids.device).unsqueeze(0)

    # ------------------------------------------------------------------
    # 前向：隐藏状态与视觉特征
    # ------------------------------------------------------------------

    def hidden_states(
        self, inputs: ModelInputs, layers: Sequence[LayerKey], *, query_embeds: torch.Tensor | None = None
    ) -> HiddenStates:
        """指定层在图像 token 与查询 token 位置的隐藏状态（层号约定见 base.py）。

        layers 里可以有 PRE_NORM_LAST（base.py）：最后一个 decoder 层的原始输出（含它的 PLE 与层缩放），在 _last_decoder_layer()
        上挂前向钩子取得。
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
        """进入 LLM 之前的视觉 token：`embed_vision` 之后的特征，[n_image_tokens, hidden]，多图按顺序拼接。

        与 hidden_states(inputs, [0]) 在图像位置上逐位相同（模块开头写明了源码依据；测试与冒烟都查）。
        """
        self._check_entry(inputs, None, where="vision_features")
        if not inputs.images:
            return torch.empty((0, self.hidden_size()), dtype=self._dtype, device=self._device)
        return self._image_features(inputs)

    # ------------------------------------------------------------------
    # 生成
    # ------------------------------------------------------------------

    def thinking_prefix_ids(self, enable_thinking: bool) -> tuple[int, ...]:
        """enable_thinking=True 时插在 `<bos>` 之后的 system 轮 token（`<|turn>system\\n<|think|>\\n<turn|>\\n`）；False 时为空。"""
        self._require_loaded()
        if not isinstance(enable_thinking, bool):
            raise TypeError(f"enable_thinking 必须是 bool，收到 {enable_thinking!r}")
        return self._prefix[enable_thinking]

    def assistant_header_ids(self, enable_thinking: bool) -> tuple[int, ...]:
        """生成提示的 token（`<|turn>model\\n`）。两种思考开关相同（attach 时核对过）；参数只为与 Qwen 的接口对称。"""
        self._require_loaded()
        if not isinstance(enable_thinking, bool):
            raise TypeError(f"enable_thinking 必须是 bool，收到 {enable_thinking!r}")
        return self._header

    def prompt_ids(self, inputs: ModelInputs, *, enable_thinking: bool) -> torch.Tensor:
        """generate 实际送进模型的 input_ids（1, L'）：`<bos>` + 思考 system 轮（仅开启时）+ 其余 + 生成提示。

        它与 apply_chat_template(..., add_generation_prompt=True, enable_thinking=X) 的分词（图像占位符展开后）逐个相同。
        """
        ids = inputs.tensors["input_ids"]
        if ids.dim() != 2 or ids.shape[0] != 1:
            raise ValueError(f"只支持单条样本，input_ids 形状 {tuple(ids.shape)}")
        if int(ids[0, 0]) != self._ids[BOS]:
            raise RuntimeError("input_ids 不以 <bos> 开头")
        prefix = torch.tensor([self.thinking_prefix_ids(enable_thinking)], dtype=torch.long, device=ids.device)
        header = torch.tensor([self.assistant_header_ids(enable_thinking)], dtype=torch.long, device=ids.device)
        return torch.cat([ids[:, :1], prefix, ids[:, 1:], header], dim=1)

    def generate(
        self,
        inputs: ModelInputs,
        *,
        enable_thinking: bool,
        allowed_token_ids: Sequence[int] | None = None,
        max_new_tokens: int,
        **gen_kwargs: Any,
    ) -> GenerationResult:
        """单条样本生成。默认 greedy（显式 do_sample=False，盖过仓库 generation_config 的采样设置）；
        停止 token 是 `<turn|>` 与 `<eos>`（id 从 tokenizer 查）。

        allowed_token_ids 非空时，每一步把其他 id 的 logits 置为 -inf；这时若希望能停下，调用方要把停止 token
        放进允许集合。返回的 token_ids 含生成出的停止 token；n_prompt_tokens 含思考 system 轮与生成提示。
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
        """批量生成（约定见 base.ModelAdapter.generate_batch）。每行的提示 = prompt_ids（思考 system 轮只在开启时插在 `<bos>` 之后）；
        左填充用 `<pad>`；图像按行的顺序拼接（相当于官方 processor 的"按样本嵌套传图"，D13-6），每行的图数可以不同。
        """
        return self._generate_rows(inputs_list, enable_thinking=enable_thinking, allowed_token_ids=allowed_token_ids,
                                   max_new_tokens=max_new_tokens, gen_kwargs=gen_kwargs, where="generate_batch")

    def _prompt_row(self, inputs: ModelInputs, enable_thinking: bool) -> batching.PromptRow:
        ids = self.prompt_ids(inputs, enable_thinking=enable_thinking)[0].tolist()
        image_id = self._ids[IMAGE_TOKEN]
        return batching.PromptRow(
            ids=tuple(ids),
            n_visual_tokens=sum(len(i.span) for i in inputs.images),
            n_image_pixels=batching.image_pixels(inputs),
            mm_token_type_ids=tuple(int(t == image_id) for t in ids),
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
        pad_id = self._ids[PAD]
        out, wall = batching.run_generate(self._model, kw, device=self._device, max_new_tokens=max_new_tokens,
                                          stop_ids=stop_ids, pad_id=pad_id, processors=processors, gen_kwargs=gen_kwargs)
        return batching.make_results(rows, out, kw["input_ids"].shape[1], wall=wall, stop_ids=stop_ids, pad_id=pad_id,
                                     max_new_tokens=max_new_tokens, split_answer=self.split_answer,
                                     enable_thinking=enable_thinking)

    def _batch_kw(self, items: Sequence[ModelInputs],
                  enable_thinking: bool) -> tuple[list[batching.PromptRow], dict[str, Any]]:
        """generate / generate_batch / option_logits 共用的拼批：每行的提示（prompt_ids）左填充成一批，各行的图按行的顺序拼接。"""
        rows = [self._prompt_row(inp, enable_thinking) for inp in items]
        kw: dict[str, Any] = batching.batch_prompt(rows, pad_id=self._ids[PAD], device=self._device)
        kw.update(self._image_kw(items))
        return rows, kw

    def _image_kw(self, items: Sequence[ModelInputs]) -> dict[str, torch.Tensor]:
        """各行的图按行的顺序拼进 pixel_values / image_position_ids（没有图时为空）。_batch_kw 与训练前向共用。"""
        with_images = [inp for inp in items if inp.images]
        if not with_images:
            return {}
        return {"pixel_values": torch.cat([inp.tensors["pixel_values"] for inp in with_images], dim=0),
                "image_position_ids": torch.cat([inp.tensors["image_position_ids"] for inp in with_images], dim=0)}

    def option_logits(
        self,
        inputs_list: Sequence[ModelInputs],
        options: Sequence[Sequence[str]],
        *,
        enable_thinking: bool,
    ) -> list[OptionScores]:
        """受限选项读数（约定见 base.ModelAdapter.option_logits；DEV-0016）。提示与左填充同 generate_batch（_batch_kw）；
        第一个生成位置 = 生成提示 `<|turn>model\\n` 之后（两种思考开关相同；开思考时 system 轮插在 `<bos>` 之后）。"""
        self._require_loaded()
        batching.check_enable_thinking(enable_thinking)
        self._require_eval("option_logits")
        items = batching.check_inputs_list(inputs_list, where="option_logits")
        opts = batching.check_options(options, len(items))
        for inp in items:
            self._check_entry(inp, None, where="option_logits")
        rows, kw = self._batch_kw(items, enable_thinking)
        return batching.option_logits_rows(self._model, self._tok, rows, kw, opts, device=self._device,
                                           stop_ids=[self._ids[s] for s in STOP_TOKENS], pad_id=self._ids[PAD])

    def split_answer(self, token_ids: Sequence[int], *, enable_thinking: bool) -> tuple[str, str, bool]:
        """把生成的 token 拆成 (text, raw_text, thinking_unclosed)（D10）。

        raw_text：全部生成 token 的解码，含思考段与特殊 token。
        思考段是 `<|channel>`…`<channel|>`（开头标记后面是 "thought\\n"，见模块开头）。规则：
        - 最后一个 `<|channel>` 之后没有 `<channel|>`（思考段开了没关，例如 max_new_tokens 用完）：text 为空串，
          thinking_unclosed = True；
        - 否则 text 取最后一个 `<channel|>` 之后的内容（没有思考段就取全部）；末尾的停止 token 去掉，
          其余特殊 token（包括 grid token）保留，首尾换行去掉（与 Qwen 适配器相同）。
        与 Qwen 不同，思考段的开头在模型输出里而不在提示里，所以规则只看输出，不看 enable_thinking
        （参数只为与 Qwen 的接口对称）：E2B/E4B 关闭思考时不输出思考段（model card，§6.2），开启时也可能直接作答。
        """
        self._require_loaded()
        if not isinstance(enable_thinking, bool):
            raise TypeError(f"enable_thinking 必须是 bool，收到 {enable_thinking!r}")
        ids = [int(i) for i in token_ids]
        raw_text = self._tok.decode(ids, skip_special_tokens=False)
        open_id, close_id = self._ids[CHANNEL_OPEN], self._ids[CHANNEL_CLOSE]
        last_open = max((k for k, v in enumerate(ids) if v == open_id), default=None)
        last_close = max((k for k, v in enumerate(ids) if v == close_id), default=None)
        if last_open is not None and (last_close is None or last_open > last_close):
            return "", raw_text, True
        ans = ids[last_close + 1 :] if last_close is not None else ids
        stop = {self._ids[s] for s in STOP_TOKENS}
        if ans and ans[-1] in stop:
            ans = ans[:-1]
        text = self._tok.decode(ans, skip_special_tokens=False).strip("\n")
        return text, raw_text, False

    # ------------------------------------------------------------------
    # 查询 token 与新增 token
    # ------------------------------------------------------------------

    def inject_query_tokens(self, n: int, position_mode: PositionMode) -> None:
        """登记 n 个查询 token；之后的 build_inputs 把它们放在 user 内容最后一项之后。可重复调用以改变 n。

        只支持 position_mode="seq1d"。Gemma 4 的 LLM 位置是一维的（§6.4），没有 M-RoPE；简报 §8.1/§12 规定二维网格
        位置只用于 M-RoPE 模型，其他模型用一维位置。"grid2d" 直接报 ValueError（不静默当成 seq1d，免得配置里写着
        grid2d、实际跑的是 seq1d），报错前不改任何状态。
        占位符复用 id 最大的 `<unusedN>`，首次调用时注册，主嵌入行与 PLE 行按 D13-1 初始化为基础行均值；
        它不计入 E3b 的 300 个 token 预算。前向时它的嵌入一律由 query_embeds 替换（必须显式传，D21-2）。
        首次调用（分配占位符的行）必须在 inject_lora 之前；之后再调只改个数与模式，不碰任何行，LoRA 之后也可以。
        """
        self._require_loaded()
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise ValueError(f"n 必须是正整数，收到 {n!r}")
        if position_mode not in POSITION_MODES:
            raise ValueError(f"position_mode 必须是 {POSITION_MODES} 之一，收到 {position_mode!r}")
        if position_mode != "seq1d":
            raise ValueError(
                "Gemma 4 的 LLM 位置是一维的（model_notes §6.4），没有 M-RoPE；按简报 §8.1/§12，非 M-RoPE 模型的查询 token "
                f"用一维位置：请传 position_mode='seq1d'（收到 {position_mode!r}）"
            )
        if self._query_token_id is None:
            self._check_tokens_mutable("inject_query_tokens")  # 首次注入要分配行：必须在 inject_lora 之前
            top = max(i for i in self._unused_ids if i not in self._taken)
            (qid,) = self._register_unused([self._unused_surface[top]])
            self._init_rows([qid])
            self._taken.add(qid)
            self._query_token_id = qid
        self._query_n = n
        self._query_mode = position_mode

    def clear_query_tokens(self) -> None:
        """取消查询 token：之后的 build_inputs 不再追加（占位符 token 仍留在词表里）。"""
        self._query_n = None
        self._query_mode = None

    def add_grid_tokens(self, names: Sequence[str]) -> dict[str, int]:
        """加入像素 token 通道用的 token，返回 名字 → id（D13-1）。

        复用 `<unusedN>`：按 id 从小到大取尚未使用的，`add_tokens(..., special_tokens=True)` 注册后逐项断言
        仍是原 id、len(tokenizer) 不变、每个都分词成单个 token、嵌入 / lm_head / PLE 行数不变（不 resize）。
        名字只是调用方的标签：token 在文本里的样子是 `<unusedN>`（decode 出来就是它），对应关系见 grid_token_surfaces；
        解析网格时请按 token id（grid_token_ids）而不是按名字的字符串。
        预算按累计计（≤ MAX_NEW_GRID_TOKENS，查询占位符不计）。主嵌入行（lm_head 共享）与 PLE 行都写成基础行的精确均值。
        必须在 inject_lora 之前调用。
        """
        self._require_loaded()
        self._check_tokens_mutable("add_grid_tokens")
        if isinstance(names, str):
            raise TypeError("names 必须是字符串的序列，不能是单个字符串")
        names = list(names)
        check_grid_token_names(names, n_existing=self._n_grid_tokens)
        vocab = self._tok.get_vocab()
        clash = [n for n in names if n in vocab or n in self._grid_token_ids]
        if clash:
            raise ValueError(f"这些名字已在词表里或已登记为 grid token：{clash[:5]}")
        free = [i for i in self._unused_ids if i not in self._taken]
        if len(free) < len(names):
            raise RuntimeError(f"可复用的 <unusedN> 只剩 {len(free)} 个，要 {len(names)} 个")
        surfaces = [self._unused_surface[i] for i in free[: len(names)]]
        ids = self._register_unused(surfaces)
        self._init_rows(ids)
        self._taken.update(ids)
        self._n_grid_tokens += len(names)
        added = dict(zip(names, ids))
        self._grid_token_ids.update(added)
        self._grid_token_surfaces.update(zip(names, surfaces))
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
        """PRE_NORM_LAST 的钩子位置：LLM 的最后一个 decoder 层。Gemma4TextModel.forward 在层循环之后紧接 self.norm（PLE 在层内
        加进残差），中间没有别的运算，所以它的输出就是最终归一化的输入。"""
        return self._model.model.language_model.layers[self.num_layers() - 1]

    def hidden_size(self) -> int:
        self._require_loaded()
        return int(self._model.config.text_config.hidden_size)

    # ------------------------------------------------------------------
    # 输出解析（DEV-0006）
    # ------------------------------------------------------------------

    def parse_boxes(self, text: str, *, width: float, height: float) -> BoxParse:
        """本家族的原生框（JSON 的 box_2d，0–1000，y 在前），规则见模块级的 parse_boxes。
        不需要加载模型；parse_one_box 由基类在它之上给出。"""
        return parse_boxes(text, width=width, height=height)  # 模块级函数（方法体里的名字按全局作用域查找）

    # ------------------------------------------------------------------
    # 训练侧钩子（DEV-0020；公共实现见 src/adapters/training.py 的 TrainingSupport）
    # ------------------------------------------------------------------

    def _pad_id(self) -> int:
        return self._ids[PAD]

    def _stop_ids(self) -> list[int]:
        return [self._ids[s] for s in STOP_TOKENS]

    def _forward_scope(self) -> Any:
        return contextlib.nullcontext()

    def _user_index_shift(self, enable_thinking: bool) -> tuple[int, int]:
        """提示行 = <bos> + 思考 system 轮（开思考时）+ ModelInputs 的其余部分 + 生成提示：下标 ≥ 1 的整体后移 system 轮的长度。"""
        return len(self.thinking_prefix_ids(enable_thinking)), 1

    def _logit_softcap(self) -> float | None:
        """final_logit_softcapping（E2B/E4B 为 30.0；小模型保持真实值）：logits = tanh(z / cap) · cap。"""
        cap = self._model.config.get_text_config().final_logit_softcapping
        return None if cap is None else float(cap)

    def _vision_modules(self) -> list[Any]:
        inner = self._model.model
        mods = [inner.vision_tower, inner.embed_vision]
        for name in ("audio_tower", "embed_audio"):  # 真实 E2B/E4B 带音频塔；小模型没有
            if getattr(inner, name, None) is not None:
                mods.append(getattr(inner, name))
        return mods

    def _training_forward_kwargs(self, items: Sequence[ModelInputs], asm: Any, image_kw: dict[str, torch.Tensor],
                                 query_mask: torch.Tensor | None,
                                 query_embeds: torch.Tensor | None) -> dict[str, Any]:
        """内层模型的输入，与 _input_embeds / _per_layer_inputs 同一种并入方式：图像位置换成 pad 再查表（× embed_scale），
        图像位置并入视觉特征，查询位置换成 query_embeds；PLE 的 token 身份取同一组 id（查询位置用占位符自己的行，D17-11）；
        一维位置 cumsum(mask) − 1（与 generate 相同）。"""
        ids = asm.input_ids
        llm_ids = torch.where(ids == self._ids[IMAGE_TOKEN], self._ids[PAD], ids)
        embeds = self._model.get_input_embeddings()(llm_ids)
        if image_kw:
            feats = self._image_features_from(image_kw, [len(i.span) for inp in items for i in inp.images])
            embeds = scatter_rows(embeds, ids == self._ids[IMAGE_TOKEN], feats, what="图像特征")
        if query_mask is not None:
            embeds = scatter_rows(embeds, query_mask, query_embeds.repeat(len(items), 1), what="查询嵌入")
        per_layer = self._model.model.language_model.get_per_layer_inputs(llm_ids, None)
        return {"inputs_embeds": embeds, "per_layer_inputs": per_layer, "attention_mask": asm.attention_mask,
                "mm_token_type_ids": asm.mm_token_type_ids, "position_ids": cumsum_positions(asm.attention_mask)}

    def _decode_positions(self, kw: dict[str, Any]) -> torch.Tensor:
        return cumsum_positions(kw["attention_mask"])

    def _decode_step_extra(self, batch: int) -> dict[str, Any]:
        return {}  # HF 在预填之后不再传 mm_token_type_ids（Gemma4ForConditionalGeneration.prepare_inputs_for_generation）

    # ---- 检查点钩子（DEV-0020 第二部分；公共实现见 src/adapters/checkpoint.py 的 CheckpointSupport）----

    def _vocab_facts(self) -> dict[str, Any]:
        """规模事实：恢复时必须相同（grid token 与查询占位符的 id 由 <unusedN> 的集合决定）。"""
        return {"n_rows": int(self._model.get_input_embeddings().weight.shape[0]), "len_tokenizer": len(self._tok),
                "n_unused": len(self._unused_ids), "unused_first": self._unused_ids[0], "unused_last": self._unused_ids[-1]}

    def _token_state_fresh(self) -> bool:
        return self._n_grid_tokens == 0 and not self._taken and self._query_token_id is None

    def _token_registry(self) -> dict[str, Any]:
        grid = [{"name": n, "id": i, "surface": self._grid_token_surfaces[n]}
                for n, i in sorted(self._grid_token_ids.items(), key=lambda kv: kv[1])]
        qid = self._query_token_id
        query = {"id": qid, "surface": self.query_token_surface, "row_allocated": qid is not None,
                 "in_tokenizer": qid is not None, "injected": qid is not None, "n": self._query_n,
                 "position_mode": self._query_mode}
        return {"grid_tokens": grid, "query": query, "vocab": self._vocab_facts()}

    def _restore_registration(self, table: Any) -> None:
        """restore_tokens 的登记一步：grid token 取最小的几个 <unusedN>、查询占位符取余下最大的；与表不同就报错（什么都不改）；
        再注册（不初始化）。"""
        if table["vocab"] != self._vocab_facts():
            raise TokenRestoreError(f"规模事实不同（表 {table['vocab']}，本适配器 {self._vocab_facts()}）")
        grid = sorted(table["grid_tokens"], key=lambda g: g["id"])
        free = [i for i in self._unused_ids if i not in self._taken]
        plan = free[: len(grid)]
        if [g["id"] for g in grid] != plan:
            raise TokenRestoreError(f"按本适配器的规则重新登记，grid token 的 id 是 {plan[:3]}…，表里是 {[g['id'] for g in grid][:3]}…")
        if any(g["surface"] != self._unused_surface[g["id"]] for g in grid):
            raise TokenRestoreError("grid token 的 surface 与 <unusedN> 对不上")
        vocab = self._tok.get_vocab()
        clash = [g["name"] for g in grid if g["name"] in vocab]
        if clash:
            raise TokenRestoreError(f"这些名字已在词表里：{clash[:5]}")
        q = table["query"]
        if q["row_allocated"]:
            rest = [i for i in free if i not in set(plan)]
            if q["id"] != max(rest) or q["surface"] != self._unused_surface[q["id"]]:
                raise TokenRestoreError(f"查询占位符的 id {q['id']} 与本适配器推算的 {max(rest)} 不同")
        if grid:
            ids = self._register_unused([g["surface"] for g in grid])
            self._taken.update(ids)
            names = [g["name"] for g in grid]
            self._n_grid_tokens = len(names)
            self._grid_token_ids = dict(zip(names, ids))
            self._grid_token_surfaces = {g["name"]: g["surface"] for g in grid}
        if q["row_allocated"]:
            (qid,) = self._register_unused([q["surface"]])
            self._taken.add(qid)
            self._query_token_id = qid

    def _row_tables(self) -> list[tuple[str, Any]]:
        return [("embed", self._model.get_input_embeddings().weight),
                ("lm_head", self._model.get_output_embeddings().weight),
                ("per_layer", self._model.get_per_layer_input_embeddings().weight)]


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
            content.append({"type": "text", "text": self.query_token_surface * self._query_n})
        return content

    def _render(self, content: list[dict[str, str]], *, add_generation_prompt: bool = False,
                enable_thinking: bool) -> str:
        return self._tok.apply_chat_template(
            [{"role": "user", "content": content}],
            chat_template=self._template,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )

    def _encode(self, text: str) -> list[int]:
        return list(self._tok(text, add_special_tokens=False)["input_ids"])

    def _expand(self, text: str, counts: Sequence[int]) -> str:
        """与 Gemma4Processor.replace_image_token 相同：每个 <|image|> → <|image>` + n×<|image|> + `<image|>。"""
        parts = text.split(IMAGE_TOKEN)
        if len(parts) != len(counts) + 1:
            raise RuntimeError("图像占位符个数与展开计数不一致")
        out = [parts[0]]
        for n, part in zip(counts, parts[1:]):
            out.append(BOI + IMAGE_TOKEN * n + EOI)
            out.append(part)
        return "".join(out)

    def _check_instruction(self, instruction: str) -> None:
        """指令里不许出现 tokenizer 的特殊 token（轮次、图像、思考段标记等，出现就会打乱区间或模板结构）与查询占位符。"""
        if _SENTINEL in instruction:
            raise ValueError(f"指令里含有内部哨兵字符串 {_SENTINEL!r}")
        ids = set(self._encode(instruction))
        forbidden = [self._tok.convert_ids_to_tokens(i) for i in sorted(self._special_ids & ids)]
        if self._query_token_id is not None and self._query_token_id in ids:
            forbidden.append(self.query_token_surface)
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
                              ("embed_tokens_per_layer", self._model.get_per_layer_input_embeddings()),
                              ("vision_tower.patch_embedder", self._model.model.vision_tower.patch_embedder.input_proj),
                              ("embed_vision", self._model.model.embed_vision.embedding_projection)):
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

    def _image_features_from(self, t: Any, want: Sequence[int]) -> torch.Tensor:
        """pixel_values / image_position_ids（一条输入的，或整批按行拼接的）→ [Σ want, hidden]；每张图的特征个数与 want 逐个核对。"""
        out = self._model.model.get_image_features(t["pixel_values"], t["image_position_ids"], return_dict=True)
        sizes = [int(f.shape[0]) for f in out.pooler_output]
        if sizes != list(want):
            raise RuntimeError(f"每张图的视觉特征个数 {sizes} 与图像区间 {list(want)} 不一致")
        feats = torch.cat(out.pooler_output, dim=0).to(device=self._device, dtype=self._dtype)
        n_expected = sum(sizes)
        if tuple(feats.shape) != (n_expected, self.hidden_size()):
            raise RuntimeError(f"视觉特征形状 {tuple(feats.shape)}，期望 ({n_expected}, {self.hidden_size()})")
        return feats

    def _llm_input_ids(self, inputs: ModelInputs) -> torch.Tensor:
        """与 Gemma4Model.forward 相同：图像 token 换成 pad 之后的 id（主嵌入查表与 PLE 的 token 身份都用它）。"""
        ids = inputs.tensors["input_ids"]
        return torch.where(ids == self._ids[IMAGE_TOKEN], self._ids[PAD], ids)

    def _input_embeds(self, inputs: ModelInputs, query_embeds: torch.Tensor | None) -> torch.Tensor:
        """与 Gemma4Model.forward 相同的并入方式：查表（× embed_scale）→ 图像位置 masked_scatter 视觉特征；再替换查询位置。"""
        ids = inputs.tensors["input_ids"]
        embeds = self._model.get_input_embeddings()(self._llm_input_ids(inputs))
        if inputs.images:
            feats = self._image_features(inputs)
            mask = (ids == self._ids[IMAGE_TOKEN]).unsqueeze(-1).expand_as(embeds)
            embeds = embeds.masked_scatter(mask, feats)
        if query_embeds is not None:
            qs, qe = inputs.query_span.start, inputs.query_span.end
            embeds = torch.cat([embeds[:, :qs], query_embeds.unsqueeze(0), embeds[:, qe:]], dim=1)
        return embeds

    def _per_layer_inputs(self, inputs: ModelInputs) -> torch.Tensor:
        """PLE 的 token 身份部分，与 Gemma4Model.forward 相同：`get_per_layer_inputs(图像位置换成 pad 的 id)`。

        查询位置用占位符自己的 PLE 行（与普通 token 相同）；上下文投影部分由模型在 language_model 里从 inputs_embeds 算。
        """
        return self._model.model.language_model.get_per_layer_inputs(self._llm_input_ids(inputs), None)

    def _run(self, inputs: ModelInputs, query_embeds: torch.Tensor | None, *, where: str) -> tuple[torch.Tensor, ...]:
        self._check_entry(inputs, query_embeds, where=where)
        if inputs.query_span is not None and query_embeds is None:
            raise ValueError(f"{where}：输入里有查询 token，必须显式传 query_embeds（D21-2；不再静默用占位符的行）")
        embeds = self._input_embeds(inputs, query_embeds)
        per_layer = self._per_layer_inputs(inputs)
        t = inputs.tensors
        out = self._model(
            inputs_embeds=embeds,
            per_layer_inputs=per_layer,
            attention_mask=t["attention_mask"],
            mm_token_type_ids=t["mm_token_type_ids"],
            position_ids=self.position_ids(inputs),
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

    def _rows(self) -> tuple[int, int, int, int]:
        m = self._model
        return (m.get_input_embeddings().weight.shape[0], m.get_output_embeddings().weight.shape[0],
                m.get_per_layer_input_embeddings().weight.shape[0], int(m.config.text_config.vocab_size))

    def _register_unused(self, surfaces: list[str]) -> list[int]:
        """注册若干 `<unusedN>`（D13-1）：注册后仍是原 id、len(tokenizer) 不变、单个 token、行数不变。"""
        vocab = self._tok.get_vocab()
        want = [vocab[s] for s in surfaces]
        n_before, rows_before = len(self._tok), self._rows()
        n_added = self._tok.add_tokens(list(surfaces), special_tokens=True)
        got = [self._tok.convert_tokens_to_ids(s) for s in surfaces]
        if n_added != len(surfaces) or got != want or len(self._tok) != n_before:
            raise RuntimeError(
                f"add_tokens 返回 {n_added}，id {got[:3]}…（期望原 id {want[:3]}…），len(tokenizer) {n_before}→{len(self._tok)}"
            )
        for s, i in zip(surfaces, got):
            if self._encode(s) != [i]:
                raise RuntimeError(f"{s!r} 注册后没有分词成单个 token")
        if self._encode("".join(surfaces)) != got:
            raise RuntimeError("注册的 token 首尾相接时没有逐个分词")
        if self._rows() != rows_before:
            raise RuntimeError(f"注册 token 后行数变了：{rows_before} → {self._rows()}（不许 resize）")
        return got

    def _base_mean(self, weight: torch.Tensor) -> torch.Tensor:
        """基础行（全部 id 去掉 <unusedN> 与特殊 token）的均值：float64 分块累加，再转回参数 dtype。"""
        idx = self._base_idx.to(weight.device)
        dim = weight.shape[1]
        step = max(1, _MEAN_CHUNK_BYTES // (dim * 8))
        acc = torch.zeros(dim, dtype=torch.float64, device=weight.device)
        for s in range(0, idx.numel(), step):
            acc += weight.index_select(0, idx[s : s + step]).to(torch.float64).sum(dim=0)
        return (acc / idx.numel()).to(weight.dtype)

    def _init_rows(self, ids: list[int]) -> None:
        """新 token 的主嵌入行（lm_head 与它共享，只写一次）与 PLE 行，都写成各自基础行的均值（D13-1）。"""
        idx = torch.tensor(ids, dtype=torch.long, device=self._device)
        with torch.no_grad():
            emb = self._model.get_input_embeddings().weight
            emb[idx] = self._base_mean(emb)
            ple = self._model.get_per_layer_input_embeddings().weight
            ple[idx] = self._base_mean(ple)


def _params_and_buffers(model: Any) -> Iterable[torch.Tensor]:
    yield from model.parameters()
    yield from model.buffers()


def _named_params_and_buffers(model: Any) -> Iterable[tuple[str, torch.Tensor]]:
    yield from model.named_parameters()
    yield from model.named_buffers()


# ---------------------------------------------------------------------------
# 小型随机 Gemma 4：CPU 测试与冒烟脚本 --tiny 的彩排（不下载权重）
# ---------------------------------------------------------------------------

# 4 层：sliding / full 交替，后 2 层 KV 共享（第 2 层复用第 0 层的 KV，第 3 层复用第 1 层的），最后一层是 full
# （config 要求）。真实 E2B 的前 4 层全是 sliding，取前缀凑不出"两种层 + KV 共享"，所以这里重排为最小的交替结构。
TINY_LAYER_TYPES: tuple[str, ...] = ("sliding_attention", "full_attention", "sliding_attention", "full_attention")
TINY_TEXT = {
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 4,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "head_dim": 16,
    "global_head_dim": 32,
    "hidden_size_per_layer_input": 8,
    "num_kv_shared_layers": 2,
}
TINY_VISION = {
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "num_key_value_heads": 2,
    "head_dim": 32,
    "global_head_dim": 32,
}
# 彩排用最小档位：测试与冒烟里的小图每张 60 个视觉 token 左右（如 6×10），前向很快
TINY_MAX_SOFT_TOKENS = 70


class PinnedComponents(NamedTuple):
    """固定 revision 的小文件：config、tokenizer、图像处理器（PIL 后端）、聊天模板、generation_config。"""

    config: Any
    tokenizer: Any
    image_processor: Any
    chat_template: str
    generation_config: Any


def load_pinned_components(model_key: str = TINY_SOURCE_KEY) -> PinnedComponents:
    """从登记的固定 revision 下载 config、tokenizer、图像处理器、聊天模板与 generation_config（都是小文件，不含权重）。

    图像处理器固定用 PIL 后端（Gemma4ImageProcessorPil），开发机没有 torchvision 也能跑；服务器上的 load() 走官方
    AutoProcessor（torchvision 后端）。两者只在缩放的插值实现上可能不同（antialias 与 PIL 的 BICUBIC），接口相同。
    generation_config 用来让小模型也带上仓库默认的采样设置（do_sample=True），检查适配器确实显式 greedy。
    """
    from transformers import AutoConfig, AutoTokenizer, GenerationConfig
    from transformers.models.gemma4.image_processing_pil_gemma4 import Gemma4ImageProcessorPil

    entry = registry.get_model(model_key)
    if entry.family != registry.FAMILY_GEMMA4:
        raise ValueError(f"{model_key} 不是 Gemma 4")
    config = AutoConfig.from_pretrained(entry.repo, revision=entry.revision)
    tokenizer = AutoTokenizer.from_pretrained(entry.repo, revision=entry.revision)
    image_processor = Gemma4ImageProcessorPil.from_pretrained(entry.repo, revision=entry.revision)
    generation_config = GenerationConfig.from_pretrained(entry.repo, revision=entry.revision)
    template = tokenizer.chat_template
    if not isinstance(template, str) or not template:
        raise RuntimeError(f"{entry.repo}@{entry.revision} 没有取到聊天模板")
    return PinnedComponents(config, tokenizer, image_processor, template, generation_config)


def tiny_config(real_config: Any, *, sliding_window: int | None = None) -> Any:
    """把真实 config 缩成小型随机模型的 config：4 层（TINY_LAYER_TYPES，2 层 KV 共享）、hidden 64、PLE 维度 8、
    视觉塔 1 层 hidden 64，去掉音频塔。词表、特殊 token id、patch/pool、RoPE 参数、sliding_window、softcapping、
    double-wide MLP 保持真实值。sliding_window 只在测试滑动窗口截断时改小。"""
    d = copy.deepcopy(real_config.to_dict())
    tc, vc = d["text_config"], d["vision_config"]
    types = set(tc["layer_types"])
    if types != {"sliding_attention", "full_attention"} or not tc.get("num_kv_shared_layers"):
        raise ValueError(f"真实 config 不是 sliding + full 且带 KV 共享：{sorted(types)}, {tc.get('num_kv_shared_layers')}")
    tc.pop("per_layer_config", None)  # 按新的层类型与 global_head_dim 重建
    tc.update(TINY_TEXT, layer_types=list(TINY_LAYER_TYPES))
    if sliding_window is not None:
        tc["sliding_window"] = int(sliding_window)
    vc.update(TINY_VISION)
    d["audio_config"] = None
    return type(real_config).from_dict(d)


def build_tiny_model(real_config: Any, *, dtype: torch.dtype = torch.float32, seed: int = 0,
                     generation_config: Any = None, sliding_window: int | None = None) -> Any:
    """按 tiny_config 构造随机初始化的 Gemma4ForConditionalGeneration（sdpa，eval）。

    generation_config 给了就换上（通常是 load_pinned_components 取到的仓库原版，默认采样）。
    """
    from transformers import AutoModelForImageTextToText

    cfg = tiny_config(real_config, sliding_window=sliding_window)
    torch.manual_seed(seed)
    model = AutoModelForImageTextToText.from_config(cfg, attn_implementation="sdpa", dtype=dtype)
    if generation_config is not None:
        model.generation_config = copy.deepcopy(generation_config)
    return model.eval()


def make_tiny_adapter(
    components: PinnedComponents,
    *,
    max_soft_tokens: int = TINY_MAX_SOFT_TOKENS,
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
    device: str | torch.device = "cpu",
    model: Any = None,
    source_key: str = TINY_SOURCE_KEY,
) -> Gemma4Adapter:
    """用 load_pinned_components 的结果挂一个小型随机模型的适配器。source_key 是 components 的来源（只写进 model_key）。

    注意：add_grid_tokens / inject_query_tokens 会改 tokenizer；需要彼此独立的适配器时，传入各自的 tokenizer 副本。
    """
    if model is None:
        model = build_tiny_model(components.config, dtype=dtype, seed=seed,
                                 generation_config=components.generation_config)
    adapter = Gemma4Adapter(device=device, max_soft_tokens=max_soft_tokens)
    adapter.attach(model, tokenizer=components.tokenizer, image_processor=components.image_processor,
                   chat_template=components.chat_template, model_key=f"tiny({source_key})")
    return adapter


# ---------------------------------------------------------------------------
# 冒烟：scripts/smoke_adapter.py 在 setup() 里按家族分派到 smoke_setup。
# 检查项编号与 Qwen 相同（见冒烟脚本的 docstring）；2（任务调制仪器）与 8（设备审计）与家族无关，沿用脚本里的实现，
# 其余 7 项换成下面的 Gemma 版本。Gemma 特有的两项：6 查 vision_features 与第 0 层的精确关系；4 查 grid2d 报错。
# ---------------------------------------------------------------------------

# 真模型冒烟的档位：默认档 280，与 RUN-0002 §6.6 的仪器同设置（640×480 → 14×19 = 266 个视觉 token）
SMOKE_MAX_SOFT_TOKENS = 280


def smoke_setup(args: Any, dtype: torch.dtype, ctx: dict, smoke: Any) -> tuple[Gemma4Adapter, dict]:
    """构造适配器（--tiny：小型随机模型，config 取 --model 的固定 revision；否则按注册键加载真模型），
    返回 (adapter, ctx)。ctx["checks"] 是 Gemma 的检查表，ctx["smoke"] 是冒烟脚本模块（取指令常量与通用检查）。"""
    from src.adapters.factory import make_adapter  # 函数内 import：factory 按家族 import 本模块

    policy = SoftTokenBudget(TINY_MAX_SOFT_TOKENS if args.tiny else SMOKE_MAX_SOFT_TOKENS)
    ad = make_adapter(args.model, device=args.device, dtype=dtype, image_policy=policy, tiny=args.tiny)
    checks = dict(smoke.CHECKS)
    checks.update({
        "0_facts": smoke_check_0_facts,
        "1_build_generate": smoke_check_1_build_generate,
        "3_grid_tokens": smoke_check_3_grid_tokens,
        "4_query_tokens": smoke_check_4_query_tokens,
        "5_position_ids": smoke_check_5_position_ids,
        "6_vision_features": smoke_check_6_vision_features,
        "7_thinking": smoke_check_7_thinking,
    })
    return ad, {**ctx, "smoke": smoke, "checks": checks}


def _smoke_layers(ad: Gemma4Adapter) -> list[int]:
    return list(range(ad.num_layers() + 1))


def _smoke_content(n_images: int, instruction: str, order: str) -> list[dict[str, str]]:
    imgs = [{"type": "image"}] * n_images
    text = {"type": "text", "text": instruction}
    return [text, *imgs] if order == "instruction_first" else [*imgs, text]


def collapse_image_blocks(ids: Sequence[int], *, boi: int, image: int, eoi: int) -> list[int]:
    """把每段 `<|image>` + n×`<|image|>` + `<image|>` 折回模板里的单个 `<|image|>`（与 apply_chat_template 的分词对照用）。"""
    out: list[int] = []
    i, ids = 0, list(ids)
    while i < len(ids):
        if ids[i] == boi:
            j = i + 1
            while j < len(ids) and ids[j] == image:
                j += 1
            if j == i + 1 or j >= len(ids) or ids[j] != eoi:
                raise ValueError(f"位置 {i} 的 <|image> 后面不是 <|image|>…<image|>")
            out.append(image)
            i = j + 1
        else:
            out.append(ids[i])
            i += 1
    return out


def _independent_base_mean(ad: Gemma4Adapter, weight: torch.Tensor) -> torch.Tensor:
    """冒烟里独立重算的基础行均值：基础行按 tokenizer 现状另算（全部 id − 形如 <unusedN> 的 − added_tokens_decoder），
    float64 一次性 index_select 求和，不复用适配器的实现。"""
    tok = ad.tokenizer
    drop = {int(i) for s, i in tok.get_vocab().items() if UNUSED_TOKEN_RE.fullmatch(s)}
    drop |= {int(i) for i in tok.added_tokens_decoder}
    keep = torch.tensor([i for i in range(weight.shape[0]) if i not in drop], dtype=torch.long, device=weight.device)
    acc = torch.zeros(weight.shape[1], dtype=torch.float64, device=weight.device)
    for s in range(0, keep.numel(), 8192):
        acc += weight.index_select(0, keep[s : s + 8192]).double().sum(0)
    return (acc / keep.numel()).to(weight.dtype)


def smoke_check_0_facts(ad: Gemma4Adapter, ctx: dict) -> dict:
    import platform

    import transformers

    m = ad.model
    tc = m.config.text_config
    lm = m.model.language_model
    ple = m.get_per_layer_input_embeddings()
    gc = m.generation_config
    entry = registry.MODELS.get(ad.model_key)
    ip = ad.image_processor
    inp = ad.build_inputs([ctx["img"]], ctx["smoke"].INSTR_A, "image_first")
    return {
        "ok": True,
        "mode": "tiny" if ctx["args"].tiny else "real",
        "model_key": ad.model_key,
        "repo": entry.repo if entry else None,
        "revision": entry.revision if entry else None,
        "device": str(ad.device),
        "dtype": str(ad.dtype),
        "max_soft_tokens": ad.max_soft_tokens,
        "image_grid_hw": list(inp.images[0].grid_hw),
        "n_image_tokens": len(inp.images[0].span),
        "num_layers": ad.num_layers(),
        "n_hidden_states": ad.num_layers() + 1,
        "hidden_size": ad.hidden_size(),
        "layer_types": list(tc.layer_types),
        "kv_shared_layers": [i for i, layer in enumerate(lm.layers) if layer.self_attn.is_kv_shared_layer],
        "sliding_window": tc.sliding_window,
        "use_bidirectional_attention": tc.use_bidirectional_attention,
        "hidden_size_per_layer_input": tc.hidden_size_per_layer_input,
        "ple_shape": list(ple.weight.shape),
        "embed_scale": float(lm.embed_tokens.embed_scale),
        "ple_embed_scale": float(ple.embed_scale),
        "tied_word_embeddings": ad.tied_word_embeddings,
        "len_tokenizer": len(ad.tokenizer),
        "embed_rows": m.get_input_embeddings().weight.shape[0],
        "lm_head_rows": m.get_output_embeddings().weight.shape[0],
        "n_unused_pool": len(ad.unused_token_ids),
        "n_base_rows_for_mean": ad.n_base_vocab,
        "attn_implementation": m.config._attn_implementation,
        "image_processor_class": type(ip).__name__,
        "image_processor_attrs": {k: getattr(ip, k, None) for k in (
            "patch_size", "pooling_kernel_size", "do_resize", "resample", "do_rescale", "rescale_factor", "do_normalize")},
        "hf_processor_class": type(ad.hf_processor).__name__ if ad.hf_processor is not None else None,
        "thinking_prefix_ids": {"thinking_off": list(ad.thinking_prefix_ids(False)),
                                "thinking_on": list(ad.thinking_prefix_ids(True))},
        "assistant_header_ids": list(ad.assistant_header_ids(False)),
        "stop_token_ids": {s: ad.token_id(s) for s in STOP_TOKENS},
        "repo_generation_config": {"do_sample": gc.do_sample, "top_k": gc.top_k, "top_p": gc.top_p,
                                   "temperature": gc.temperature, "eos_token_id": gc.eos_token_id},
        "image": ctx["image_info"],
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "transformers": transformers.__version__},
    }


def _smoke_compare_official(ad: Gemma4Adapter, images: list, instruction: str, order: str, thinking: bool,
                            inp: ModelInputs) -> dict:
    """load 路径：适配器的输入（插入思考 system 轮与生成提示后）与官方 AutoProcessor 的输出逐项相同。
    官方 processor 按样本嵌套传图（D13-6），显式传 max_soft_tokens。"""
    proc = ad.hf_processor
    pil = [ad.prepare_image(im) for im in images]
    text = proc.apply_chat_template([{"role": "user", "content": _smoke_content(len(images), instruction, order)}],
                                    tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
    enc = proc(text=[text], images=[pil], max_soft_tokens=ad.max_soft_tokens, return_tensors="pt")
    t = inp.tensors
    ids = ad.prompt_ids(inp, enable_thinking=thinking).cpu()
    res = {
        "input_ids": torch.equal(enc["input_ids"].cpu(), ids),
        "attention_mask": torch.equal(enc["attention_mask"].cpu(), torch.ones_like(ids)),
        "mm_token_type_ids": torch.equal(enc["mm_token_type_ids"].cpu(), (ids == ad.token_id(IMAGE_TOKEN)).long()),
        "pixel_values": torch.equal(enc["pixel_values"].to(ad.dtype).cpu(), t["pixel_values"].cpu()),
        "image_position_ids": torch.equal(enc["image_position_ids"].cpu().long(), t["image_position_ids"].cpu()),
    }
    return {"ok": all(res.values()), **res}


def smoke_check_1_build_generate(ad: Gemma4Adapter, ctx: dict) -> dict:
    args, img, tok = ctx["args"], ctx["img"], ad.tokenizer
    instr = ctx["smoke"].INSTR_A
    marks = {"boi": ad.token_id(BOI), "image": ad.token_id(IMAGE_TOKEN), "eoi": ad.token_id(EOI)}
    cases = []
    for order in ORDERS:
        inp = ad.build_inputs([img], instr, order)
        for thinking in (False, True):
            res = ad.generate(inp, enable_thinking=thinking, max_new_tokens=args.max_new_tokens)
            full = ad.prompt_ids(inp, enable_thinking=thinking)[0].tolist()
            ref = tok.apply_chat_template([{"role": "user", "content": _smoke_content(1, instr, order)}],
                                          tokenize=True, add_generation_prompt=True, enable_thinking=thinking)["input_ids"]
            template_equal = collapse_image_blocks(full, **marks) == list(ref)
            official = (_smoke_compare_official(ad, [img], instr, order, thinking, inp)
                        if ad.hf_processor is not None else None)
            raw_ok = res.raw_text == tok.decode(list(res.token_ids), skip_special_tokens=False)
            ok = (template_equal and raw_ok and 1 <= res.n_generated_tokens <= args.max_new_tokens
                  and res.n_prompt_tokens == len(full) and (official is None or official["ok"]))
            cases.append({
                "order": order, "enable_thinking": thinking, "ok": bool(ok),
                "template_equal": template_equal, "official_processor": official, "raw_text_is_full_decode": raw_ok,
                "instruction_span": [inp.instruction_span.start, inp.instruction_span.end],
                "image_spans": [[i.span.start, i.span.end] for i in inp.images],
                "image_grids": [list(i.grid_hw) for i in inp.images],
                "n_prompt_tokens": res.n_prompt_tokens, "n_generated_tokens": res.n_generated_tokens,
                "n_visual_tokens": res.n_visual_tokens, "wall_time_s": round(res.wall_time_s, 4),
                "thinking_unclosed": res.thinking_unclosed, "text": res.text[:200], "raw_text": res.raw_text[:300],
            })
    two_images = None  # 多图：适配器（一次调图像处理器）与官方 processor（按样本嵌套传图，D13-6）逐项相同
    if ad.hf_processor is not None:
        imgs = [img, ctx["img2"]]
        two_images = _smoke_compare_official(ad, imgs, instr, "instruction_first", False,
                                             ad.build_inputs(imgs, instr, "instruction_first"))
    return {"ok": all(c["ok"] for c in cases) and (two_images is None or two_images["ok"]),
            "official_processor_compared": ad.hf_processor is not None, "official_two_images": two_images,
            "cases": cases}


def smoke_check_3_grid_tokens(ad: Gemma4Adapter, ctx: dict) -> dict:
    args, img, tok = ctx["args"], ctx["img"], ad.tokenizer
    names = list(ctx["smoke"].GRID_NAMES)
    m = ad.model
    rows_before = [m.get_input_embeddings().weight.shape[0], m.get_output_embeddings().weight.shape[0],
                   m.get_per_layer_input_embeddings().weight.shape[0], len(tok)]
    # 独立推算应取的 id：词表里形如 <unusedN> 的 id 从小到大的前几个（此前没有注册过任何 token）
    expect_ids = sorted(int(i) for s, i in tok.get_vocab().items() if UNUSED_TOKEN_RE.fullmatch(s))[: len(names)]
    ids = ad.add_grid_tokens(names)
    allowed = list(ids.values())
    surfaces = ad.grid_token_surfaces
    single = {n: tok(surfaces[n], add_special_tokens=False)["input_ids"] == [i] for n, i in ids.items()}
    emb = m.get_input_embeddings().weight
    ple = m.get_per_layer_input_embeddings().weight
    with torch.no_grad():
        emb_ok = bool(torch.equal(emb[allowed], _independent_base_mean(ad, emb).expand(len(allowed), -1)))
        ple_ok = bool(torch.equal(ple[allowed], _independent_base_mean(ad, ple).expand(len(allowed), -1)))
    rows_after = [emb.shape[0], m.get_output_embeddings().weight.shape[0], ple.shape[0], len(tok)]
    inp = ad.build_inputs([img], ctx["smoke"].INSTR_A, "instruction_first")
    res = ad.generate(inp, enable_thinking=False, max_new_tokens=args.max_new_tokens, allowed_token_ids=allowed)
    in_set = set(res.token_ids) <= set(allowed)
    ok = (in_set and res.n_generated_tokens == args.max_new_tokens and all(single.values()) and emb_ok and ple_ok
          and rows_after == rows_before and allowed == expect_ids and m.get_output_embeddings().weight is emb)
    return {"ok": bool(ok), "ids": ids, "surfaces": surfaces, "expected_lowest_unused_ids": expect_ids,
            "single_token": single, "rows_before(embed,lm_head,ple,len_tok)": rows_before, "rows_after": rows_after,
            "mean_init_embed(tied_lm_head)": emb_ok, "mean_init_ple": ple_ok, "all_generated_in_allowed": in_set,
            "generated_ids": list(res.token_ids), "text": res.text[:200]}


def smoke_check_4_query_tokens(ad: Gemma4Adapter, ctx: dict) -> dict:
    img, smoke = ctx["img"], ctx["smoke"]
    n_query, layers = smoke.N_QUERY, _smoke_layers(ad)
    n_layers, hidden = ad.num_layers(), ad.hidden_size()
    tok = ad.tokenizer
    # grid2d：本适配器的决定是报错（一维位置的模型没有二维网格位置），且报错前不改任何状态
    len_before, qid_before = len(tok), ad.query_token_id
    try:
        ad.inject_query_tokens(n_query, "grid2d")
        grid2d = {"raised": False}
    except ValueError as e:
        grid2d = {"raised": True, "message": str(e)[:200]}
    grid2d["state_unchanged"] = (len(tok) == len_before and ad.query_token_id == qid_before
                                 and ad.build_inputs([img], smoke.INSTR_A, "instruction_first").query_span is None)
    grid2d["ok"] = bool(grid2d["raised"] and grid2d["state_unchanged"])

    ad.inject_query_tokens(n_query, "seq1d")
    inp = ad.build_inputs([img], smoke.INSTR_A, "instruction_first")
    span = inp.query_span
    ids = inp.tensors["input_ids"][0].tolist()
    L = len(ids)
    pos = ad.position_ids(inp)
    pos_ok = bool(torch.equal(pos.cpu(), torch.arange(L).unsqueeze(0)))
    place_ok = (span.end + 2 == L and ids[span.end] == ad.token_id(TURN_CLOSE)
                and span.start == inp.images[-1].span.end + 1 and inp.query_position_mode == "seq1d")
    top_unused = max(int(i) for s, i in tok.get_vocab().items() if UNUSED_TOKEN_RE.fullmatch(s))
    g = torch.Generator().manual_seed(0)
    qa = (torch.randn(n_query, hidden, generator=g) / math.sqrt(hidden)).to(device=ad.device, dtype=ad.dtype)
    qb = (torch.randn(n_query, hidden, generator=g) / math.sqrt(hidden)).to(device=ad.device, dtype=ad.dtype)
    with torch.no_grad():
        sa = ad.sequence_hidden_states(inp, layers, query_embeds=qa)
        sb = ad.sequence_hidden_states(inp, layers, query_embeds=qb)
        hs = ad.hidden_states(inp, [0, n_layers], query_embeds=qa)
    qs, qe = span.start, span.end
    prefix = [bool(torch.equal(sa[l][:qs], sb[l][:qs])) for l in layers]
    q_differ = [not torch.equal(sa[l][qs:qe], sb[l][qs:qe]) for l in layers]
    shape_ok = (tuple(hs.query[n_layers].shape) == (n_query, hidden)
                and tuple(hs.image[n_layers].shape) == (sum(len(i.span) for i in inp.images), hidden))
    layer0_ok = bool(torch.equal(hs.query[0], qa))
    ad.clear_query_tokens()
    ok = (grid2d["ok"] and pos_ok and place_ok and all(prefix) and q_differ[-1] and shape_ok and layer0_ok
          and ad.query_token_id == top_unused)
    return {"ok": bool(ok), "grid2d": grid2d, "n_query": n_query, "query_token_id": ad.query_token_id,
            "query_token_surface": ad.query_token_surface, "expected_top_unused_id": top_unused,
            "seq1d": {"query_span": [qs, qe], "position_ids_arange": pos_ok, "placement_ok": place_ok,
                      "shape_ok": shape_ok, "layer0_query_equals_query_embeds": layer0_ok},
            "prefix_bitwise_equal_by_layer(two_query_embeds)": prefix, "query_states_differ_by_layer": q_differ}


def smoke_check_5_position_ids(ad: Gemma4Adapter, ctx: dict) -> dict:
    imgs = [ctx["img"], ctx["img2"]]
    layers = _smoke_layers(ad)
    native_keys = ("input_ids", "attention_mask", "mm_token_type_ids", "pixel_values", "image_position_ids")
    out = {}
    for order in ORDERS:
        inp = ad.build_inputs(imgs, ctx["smoke"].INSTR_A, order)
        t = inp.tensors
        native = {k: t[k] for k in native_keys}
        pos = ad.position_ids(inp)
        L = t["input_ids"].shape[1]
        arange_ok = bool(torch.equal(pos.cpu(), torch.arange(L).unsqueeze(0)))
        kw = dict(output_hidden_states=True, logits_to_keep=1)
        with torch.no_grad():
            ours = ad.sequence_hidden_states(inp, layers)
            auto = ad.model(**native, use_cache=False, **kw).hidden_states
            given = ad.model(**native, position_ids=pos, use_cache=False, **kw).hidden_states
            default = ad.model(**native, **kw).hidden_states  # HF 默认（use_cache 取 config），只记录
        given_eq = [bool(torch.equal(auto[l][0], given[l][0])) for l in layers]
        ours_eq = [bool(torch.equal(auto[l][0], ours[l])) for l in layers]
        default_eq = [bool(torch.equal(default[l][0], ours[l])) for l in layers]
        out[order] = {"ok": all(given_eq) and all(ours_eq) and arange_ok and len(auto) == len(layers),
                      "position_ids_are_arange": arange_ok, "n_hidden_states": len(auto),
                      "own_position_ids_vs_model_computed": given_eq,
                      "adapter_path(inputs_embeds+per_layer_inputs)_vs_model_path(input_ids+pixel_values)": ours_eq,
                      "info_adapter_path_vs_hf_default_use_cache": default_eq, "seq_len": int(L)}
    out["ok"] = all(out[o]["ok"] for o in ORDERS)
    return out


def smoke_check_6_vision_features(ad: Gemma4Adapter, ctx: dict) -> dict:
    imgs = [ctx["img"], ctx["img2"]]
    lm = ad.model.model.language_model
    emb = ad.model.get_input_embeddings()
    out = {"relation": ("layer 0 at image positions == vision_features (embed_vision output), bitwise: embed_scale "
                        "multiplies only the token-embedding lookup (text positions), image rows are overwritten by "
                        "masked_scatter; PLE enters inside each decoder layer, so only layers >= 1 see it, and its "
                        "token-identity part at image positions is the pad row")}
    for order in ORDERS:
        inp = ad.build_inputs(imgs, ctx["smoke"].INSTR_A, order)
        ids = inp.tensors["input_ids"]
        img_mask = ids[0] == ad.token_id(IMAGE_TOKEN)
        captured = {}

        def grab(mod, args, kwargs):
            captured["per_layer_input"] = args[1] if len(args) > 1 else kwargs["per_layer_input"]

        h = lm.layers[0].register_forward_pre_hook(grab, with_kwargs=True)
        try:
            with torch.no_grad():
                vf = ad.vision_features(inp)
                seq0 = ad.sequence_hidden_states(inp, [0])[0]
        finally:
            h.remove()
        h0 = seq0[img_mask]
        n_img = sum(len(i.span) for i in inp.images)
        equal = bool(torch.equal(vf, h0))
        with torch.no_grad():
            text_ids = ids[0][~img_mask]
            text_scaled = bool(torch.equal(seq0[~img_mask], emb.weight[text_ids] * emb.embed_scale.to(emb.weight.dtype)))
            pad_ids = torch.where(ids == ad.token_id(IMAGE_TOKEN), ad.token_id(PAD), ids)
            ple_pad = lm.project_per_layer_inputs(seq0.unsqueeze(0), lm.get_per_layer_inputs(pad_ids, None))[0, :, 0]
            ple_img = lm.project_per_layer_inputs(seq0.unsqueeze(0), lm.get_per_layer_inputs(ids, None))[0, :, 0]
        got = captured["per_layer_input"][0]
        ple_uses_pad = bool(torch.equal(got[img_mask], ple_pad[img_mask]))
        ple_not_image_row = not torch.equal(got[img_mask], ple_img[img_mask])
        d = (vf.float() - h0.float()).abs()
        out[order] = {"ok": bool(tuple(vf.shape) == (n_img, ad.hidden_size()) and equal and text_scaled
                                 and ple_uses_pad and ple_not_image_row),
                      "shape": list(vf.shape), "bitwise_equal_layer0_image_positions": equal,
                      "max_abs_delta": d.max().item() if d.numel() else 0.0,
                      "layer0_text_positions_equal_embedding_row_times_embed_scale": text_scaled,
                      "layer0_ple_identity_at_image_positions_is_pad_row": ple_uses_pad,
                      "layer0_ple_differs_from_image_token_row": ple_not_image_row}
    out["ok"] = all(out[o]["ok"] for o in ORDERS)
    return out


def smoke_check_7_thinking(ad: Gemma4Adapter, ctx: dict) -> dict:
    args, img, tok = ctx["args"], ctx["img"], ad.tokenizer
    inp = ad.build_inputs([img], ctx["smoke"].INSTR_THINK, "image_first")
    open_id, close_id = ad.token_id(CHANNEL_OPEN), ad.token_id(CHANNEL_CLOSE)
    stops = {ad.token_id(s) for s in STOP_TOKENS}

    def expected(ids):
        """独立重写的剥离规则（与 split_answer 的 docstring 同义）：(text, unclosed)。"""
        lo = max((k for k, v in enumerate(ids) if v == open_id), default=None)
        lc = max((k for k, v in enumerate(ids) if v == close_id), default=None)
        if lo is not None and (lc is None or lo > lc):
            return "", True
        ans = ids[lc + 1 :] if lc is not None else list(ids)
        if ans and ans[-1] in stops:
            ans = ans[:-1]
        return tok.decode(ans, skip_special_tokens=False).strip("\n"), False

    res = ad.generate(inp, enable_thinking=True, max_new_tokens=args.think_max_new_tokens)
    ids = list(res.token_ids)
    raw_ok = res.raw_text == tok.decode(ids, skip_special_tokens=False)
    want_text, want_unclosed = expected(ids)
    closed = close_id in ids
    ok = (raw_ok and res.text == want_text and res.thinking_unclosed == want_unclosed
          and CHANNEL_CLOSE not in res.text and (not closed or CHANNEL_CLOSE in res.raw_text))
    off = ad.generate(inp, enable_thinking=False, max_new_tokens=args.max_new_tokens)
    off_text, off_unclosed = expected(list(off.token_ids))
    off_ok = off.text == off_text and off.thinking_unclosed == off_unclosed and CHANNEL_CLOSE not in off.text
    return {"ok": bool(ok and off_ok), "channel_opened": open_id in ids, "closed": closed,
            "closed_case_exercised": closed, "n_generated_tokens": res.n_generated_tokens,
            "max_new_tokens": args.think_max_new_tokens, "n_prompt_tokens": res.n_prompt_tokens,
            "thinking_unclosed": res.thinking_unclosed, "text": res.text[:300], "raw_text": res.raw_text[:800],
            "thinking_off_control": {"ok": bool(off_ok), "text": off.text[:200],
                                     "thinking_unclosed": off.thinking_unclosed,
                                     "channel_opened": open_id in off.token_ids}}


__all__ = [
    "AdapterDeviceError",
    "BOX_KEY",
    "CHANNEL_CLOSE",
    "CHANNEL_OPEN",
    "Gemma4Adapter",
    "MAX_SOFT_TOKENS_CHOICES",
    "PinnedComponents",
    "STOP_TOKENS",
    "box2d_to_xyxy",
    "build_tiny_model",
    "collapse_image_blocks",
    "grid_from_position_ids",
    "load_pinned_components",
    "make_tiny_adapter",
    "parse_boxes",
    "parse_one_box",
    "smoke_setup",
    "tiny_config",
]
