"""InternVL3.5 家族适配器（简报 §8.1；DEV-0005）。

模型事实一律以 `docs/model_notes.md` §7（RUN-0002 在 `OpenGVLab/InternVL3_5-2B-HF` 上实测）与 D13 为准；运行时从
tokenizer / config 读取并断言，不写死 token id，不写死层号（简报 §4 第 6 条）：

- 加载类：`AutoModelForImageTextToText` → `InternVLForConditionalGeneration`（HF 原生格式，D9-1），`attn_implementation="sdpa"`，
  eval 模式；processor 由 `AutoProcessor` 加载。load 一律带注册表里的 revision（D9-2）。
- 视觉 token（D13-5、§7.3）：tile 策略（`crop_to_patches`、`max_patches`）在构造时给定，每次调用图像处理器都显式传入，
  连同 `min_patches=MIN_PATCHES` 与 `size=TILE_PX`。**处理器的默认值（`InternVLProcessorKwargs` 里 crop_to_patches=True）
  与 preprocessor_config.json（False）相反，本适配器从不依赖任何一个默认值。**
  - 每个 tile 是 TILE_PX×TILE_PX 像素，产生 TOKENS_PER_TILE 个 LLM 视觉 token，排成 TILE_GRID_HW 网格，行主序：
    tile 内第 k 个 token 在第 r = k // 16 行（y 方向）、第 c = k % 16 列（x 方向），覆盖 tile 内的像素
    y ∈ [28r, 28r + 28)、x ∈ [28c, 28c + 28)，即 2×2 个 14 px 的 patch（pixel shuffle 0.5；
    行主序由 tests 里对 `get_image_features` 的桩测试钉住）。
  - `crop_to_patches=False`（或 `max_patches=1`）：整图直接缩放到 TILE_PX×TILE_PX，1 个 tile。
  - `crop_to_patches=True` 且 `max_patches>1`：处理器用 `get_optimal_tiled_canvas` 选 cols×rows 画布（tile 数 ≤ max_patches），
    画布 tile 按行主序排列；画布 tile 多于 1 个时末尾再加一张整图缩略图。所以同一策略下 tile 数随图的宽高比与面积而变。
  - 口径对象是 image_policy.TilePolicy（DEV-0011）。E1 用单 tile `TilePolicy(False, 1)`；原生动态 tile
    `INTERNVL_NATIVE_TILES = TilePolicy(True, 12)`（preprocessor_config 的 12；640×480 切成 13 个 tile、3328 个 token）只用于
    子集对照（D28-3）：生成可用，探针类方法照旧拒绝多 tile。ImageTokenInfo.n_pixels = tile 数 × 448²（含缩略图）。
- `ImageTokenInfo`：span 是一张图的 `<IMG_CONTEXT>` 连续段（不含两侧的 `<img>`、`</img>`）；grid_hw 的含义：
  - 单 tile：(16, 16)，就是整图（缩放到 448×448 之后）的视觉 token 网格，行主序；
  - n 个 tile（n > 1）：(16·n, 16)，是 n 个 16×16 的 tile **纵向堆叠**，第 16t–16t+15 行是第 t 个 tile，tile 按处理器的顺序
    （画布 tile 行主序，最后一个是缩略图）。它只保证"行主序 = 序列顺序"与 h·w = len(span)，**不是整图的空间网格**。
    因此查询 / 探针相关的方法（hidden_states、vision_features、带查询 token 的 build_inputs）拒绝多 tile 的图；
    generate 与诊断用的 sequence_hidden_states 接受。
- 位置（§7.4）：一维 RoPE，position_ids = 0..L-1，图像 token 逐个加 1。查询 token 只支持 seq1d：grid2d 只对 M-RoPE 模型有定义
  （简报 §12），这里直接报错，不静默当作 seq1d（DEV-0005 的二选一，见完成总结）。
- 思考开关（D13-2、§7.2）：模板没有开关。enable_thinking=True = 在 user 轮之前加一轮 system，内容是 model card 的
  R1_SYSTEM_PROMPT（原文逐字，出处见 R1_SYSTEM_PROMPT_SOURCE）；False = 不加 system 轮。两种都用同一个 assistant 头。
  ModelInputs 只含 user 轮，system 轮由 generate 按开关加在最前面。
- 思考段的剥离（D10；与 Gemma 适配器同一条规则，DEV-0005 第 1 轮审查）：思考段的开头 `<think>` 由模型自己写出（不在提示里），
  所以规则只看输出：最后一个 `<think>` 之后没有 `</think>`（写了思考段却没关上，例如 max_new_tokens 用完）才算未闭合，
  text 为空、thinking_unclosed=True；否则 text 取最后一个 `</think>` 之后的内容（模型没写思考段、直接作答时取全文），
  去掉末尾的停止 token 与首尾换行。`<think>`、`</think>` 在这个 tokenizer 里是单个、非 special 的 token（§7.1 表）。
- 生成（D13-3）：显式 greedy；停止 token 只有 `<|im_end|>`，pad 用 tokenizer 的 pad token（`<|endoftext|>`），id 都从 tokenizer 查。
  generate 与 generate_batch 走同一条路径（src/adapters/batching.py；DEV-0011）：左填充用 pad，各行的 tile 按行的顺序拼进
  同一个 pixel_values（每行的 tile 数可以不同），一维位置由 HF 按 attention_mask 推出（cumsum − 1）。
- 新增 token（D13-1）：先复用嵌入里超出 tokenizer 的 257 个预留行（id 151679–151935）；累计需要的行数超过现有行数时，把嵌入
  （不共享时连同 lm_head）扩到恰好需要的行数，已有行逐位不变；从不缩表，不调用 `resize_token_embeddings`。新 token 的嵌入行与
  lm_head 行各自写成自己基础词表行的精确均值。lm_head 是否与嵌入共享以运行时为准：真实 2B 的 config 顶层写 tie，但 checkpoint
  里两者不同，transformers 没有绑定（§7.1），所以这里不拿 config 校验共享状态，只记录。
- 层号（base.py 的层号约定，D15-2）：就是 HF `output_hidden_states` 元组的下标。0 = 视觉特征并入之后的嵌入输出，
  i = 第 i 个 decoder 层之后；最后一个下标（num_layers()）是最终归一化之后的值。
- vision_features = `multi_modal_projector` 的输出，就是前向时并入嵌入序列的那组向量，与第 0 层图像位置逐位相同。
- 坐标（D13-4、§7.5）：`<box>[[x1, y1, x2, y2], …]</box>`，0–1000，x 在前，按原图宽高换算成内部的绝对像素 xyxy，
  见 parse_boxes / parse_one_box（模块级函数，也是适配器方法；结果类型在 src/parsing，DEV-0006）。label 取紧挨在
  `<box>` 前面的 `<ref>…</ref>`。

处理器的用法：本适配器只用处理器的三个部件——tokenizer、image_processor、chat_template——自己完成 `<IMG_CONTEXT>` 占位符的展开
（与 `InternVLProcessor.replace_image_token` 相同：`<img>` + `<IMG_CONTEXT>`×(256·tile 数) + `</img>`）。原因：开发机没有
torchvision，`InternVLVideoProcessor` 构造不出来，官方处理器没法照常实例化。服务器上 load() 仍用官方 `AutoProcessor` 加载，并把它
留在 `hf_processor` 上；冒烟脚本逐项比对两者。开发机上的比对对象见 reference_processor()。
"""

from __future__ import annotations

import functools
import hashlib
import json
import platform
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from PIL import Image

from src.adapters import batching, registry
from src.adapters.image_policy import TilePolicy
from src.adapters.base import (
    ORDERS,
    POSITION_MODES,
    GenerationResult,
    HiddenStates,
    ImageTokenInfo,
    ModelAdapter,
    ModelInputs,
    OptionScores,
    Order,
    PositionMode,
    Span,
    check_grid_token_names,
)

# 与 Qwen 适配器共用同一份实现，不另写：设备/dtype 入口断言的异常类型、E3a 查询占位符的名字、
# allowed_token_ids 的 logits 限制（简报 §12：限制的实现要处处一致），以及几个小工具。
from src.adapters.qwen35 import (
    QUERY_PLACEHOLDER,
    AdapterDeviceError,
    _AllowedTokensProcessor,
    _canonical_device,
    _named_params_and_buffers,
    _params_and_buffers,
    _runs,
)
from src.constants import NORM1000_SCALE
from src.executors.geometry import check_box, norm1000_to_abs
from src.parsing import BoxParse, OneBoxParse, ParseFailure, one_box

# ---------------------------------------------------------------------------
# InternVL3.5 家族事实（字符串来自 docs/model_notes.md §7.1–7.2；id 一律在 attach 时从 tokenizer 查）
# ---------------------------------------------------------------------------

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
END_OF_TEXT = "<|endoftext|>"  # tokenizer 的 pad token；不是停止 token（D13-3）
THINK_START = "<think>"
THINK_END = "</think>"
IMG_CONTEXT = "<IMG_CONTEXT>"  # 图像占位符 = config.image_token_id
IMG_START = "<img>"
IMG_END = "</img>"
VIDEO = "<video>"
# 停止 token（D13-3：InternVL 只有 <|im_end|>）
STOP_TOKENS: tuple[str, ...] = (IM_END,)
# 指令里不许出现的结构性 token：出现就会打乱区间或模板结构（官方处理器会把文本里任何 <IMG_CONTEXT>/<video> 当占位符展开）
_STRUCTURAL_TOKENS: tuple[str, ...] = (IM_START, IM_END, END_OF_TEXT, IMG_CONTEXT, IMG_START, IMG_END, VIDEO)

# 视觉几何（§7.1、§7.3）：attach 时与 config（image_size、patch_size、downsample_ratio、image_seq_length）和图像处理器的 size 比对。
TILE_PX = 448  # 每个 tile 的边长（像素）
TOKENS_PER_TILE = 256  # 每个 tile 的 LLM 视觉 token 数
TILE_GRID_HW = (16, 16)  # 每个 tile 的 LLM 视觉 token 网格（行主序，行 = y）
MIN_PATCHES = 1  # 处理器的 min_patches：固定为 1（preprocessor_config 与 §7.3 的全部实测都是 1），每次显式传入

# 思考模式的 system prompt（D13-2）。原文取自下面这份 model card 的 "Thinking Mode" 一节：
# card 里写的是 R1_SYSTEM_PROMPT = """\n…\n""".strip()，这里存 .strip() 之后的值。4B-HF 在其固定 revision 上的 card 与此逐字相同
# （tests/test_adapter_internvl35.py 下载两份 card 比对，并与 RUN-0002 在服务器上渲染出的 system 轮逐字比对）。
R1_SYSTEM_PROMPT_SOURCE = (
    "https://huggingface.co/OpenGVLab/InternVL3_5-2B-HF/blob/3f301ffcf3dcbb47893afae6650ea3e78d96fb6d/README.md"
    " （§Thinking Mode，R1_SYSTEM_PROMPT）"
)
R1_SYSTEM_PROMPT = (
    "You are an AI assistant that rigorously follows this response protocol:\n\n"
    "1. First, conduct a detailed analysis of the question. Consider different angles, potential solutions, and reason "
    "through the problem step-by-step. Enclose this entire thinking process within <think> and </think> tags.\n\n"
    "2. After the thinking section, provide a clear, concise, and direct answer to the user's question. Separate the "
    "answer from the think section with a newline.\n\n"
    "Ensure that the thinking process is thorough but remains focused on the query. The final answer should be "
    "standalone and not reference the thinking section."
)

# 小型随机模型：从哪个登记模型的真实 config 缩出来（任务书：取固定 revision 的真实 config）
TINY_SOURCE_KEY = "internvl3.5-2b-hf"

_SENTINEL = "VISINTINSTRUCTIONSLOT"
_MEAN_CHUNK_BYTES = 1 << 28  # 求基础行均值时每块 float64 的字节上限

# generate / generate_batch 的 gen_kwargs 不能覆盖的参数（适配器自己控制）
_RESERVED_GEN_KWARGS = frozenset({"input_ids", "attention_mask", "pixel_values", "position_ids", "inputs_embeds",
                                  "max_new_tokens", "eos_token_id", "pad_token_id"})

_GRID2D_UNSUPPORTED = (
    "InternVL3.5 的 LLM 是一维 RoPE（model_notes §7.4），没有二维网格位置；grid2d 只对 M-RoPE 模型有定义（简报 §12）。"
    "本适配器不把 grid2d 静默当作 seq1d，请显式用 seq1d。"
)


def _size_hw(size: Any) -> tuple[Any, Any]:
    """图像处理器的 size（SizeDict 或 dict）→ (height, width)。"""
    if isinstance(size, dict):
        return size.get("height"), size.get("width")
    return getattr(size, "height", None), getattr(size, "width", None)


def _as_pair(v: Any) -> tuple[int, int]:
    if isinstance(v, (list, tuple)):
        if len(v) != 2:
            raise ValueError(f"期望长度为 2 的 (h, w)，收到 {v!r}")
        return int(v[0]), int(v[1])
    return int(v), int(v)


def tiles_of(info: ImageTokenInfo) -> int:
    """一张图切成的 tile 数（含缩略图），由 grid_hw 的约定反推：h = 16·tile 数、w = 16。"""
    h, w = info.grid_hw
    if w != TILE_GRID_HW[1] or h % TILE_GRID_HW[0] != 0 or h == 0:
        raise ValueError(f"grid_hw={info.grid_hw} 不符合 (16·tile 数, 16) 的约定")
    return h // TILE_GRID_HW[0]


class InternVL35Adapter(ModelAdapter):
    """InternVL3.5-HF（2B/4B）适配器。ModelInputs 是单条样本；generate_batch 左填充拼批（DEV-0011）。vLLM 另开任务。

    设备与 tile 策略在构造时给定，dtype 在 load 时给定。进入任何模型算子之前，入口断言检查输入与模型的 device/dtype，
    不一致就抛 AdapterDeviceError 并报出两个值。
    """

    family = registry.FAMILY_INTERNVL35

    def __init__(self, *, device: str | torch.device, crop_to_patches: bool, max_patches: int):
        if not isinstance(crop_to_patches, bool):
            raise TypeError(f"crop_to_patches 必须是 bool，收到 {crop_to_patches!r}")
        if isinstance(max_patches, bool) or not isinstance(max_patches, (int, np.integer)) or max_patches < 1:
            raise ValueError(f"max_patches 必须是正整数，收到 {max_patches!r}")
        self._crop = crop_to_patches
        self._max_patches = int(max_patches)
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
        self._config_tie: dict[str, bool] = {}
        self._prefix_cache: dict[bool, tuple[int, ...]] = {}
        self._header: tuple[int, ...] | None = None
        self._n_grid_tokens = 0
        self._grid_token_ids: dict[str, int] = {}
        self._query_token_id: int | None = None
        self._query_n: int | None = None

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
    def crop_to_patches(self) -> bool:
        return self._crop

    @property
    def max_patches(self) -> int:
        return self._max_patches

    @property
    def always_single_tile(self) -> bool:
        """策略本身保证每张图只有 1 个 tile（crop_to_patches=False 或 max_patches=1）。否则 tile 数随图而变。"""
        return (not self._crop) or self._max_patches == 1

    @property
    def image_policy(self) -> TilePolicy:
        """构造时给定的图像口径（src/adapters/image_policy.py）。"""
        return TilePolicy(self._crop, self._max_patches)

    @property
    def tile_policy(self) -> dict[str, Any]:
        """构造时给定并每次显式传给图像处理器的参数（E1 预注册与 G0 报告要列出，D13-5）。"""
        return {"crop_to_patches": self._crop, "max_patches": self._max_patches, "min_patches": MIN_PATCHES,
                "tile_px": TILE_PX}

    def processor_kwargs(self) -> dict[str, Any]:
        """每次调用图像处理器都显式传入的全部 tile 参数（不依赖处理器或 preprocessor_config 的任何默认值）。"""
        return {"crop_to_patches": self._crop, "max_patches": self._max_patches, "min_patches": MIN_PATCHES,
                "size": {"height": TILE_PX, "width": TILE_PX}}

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
    def chat_template(self) -> str:
        self._require_loaded()
        return self._template

    @property
    def model_key(self) -> str | None:
        return self._model_key

    @property
    def tied_word_embeddings(self) -> bool:
        """运行时 lm_head 与嵌入是否共享同一个参数（以运行时为准，不信 config，见模块说明）。"""
        self._require_loaded()
        return self._tied

    @property
    def config_tie_word_embeddings(self) -> dict[str, bool]:
        """config 里写的共享开关（顶层与 text_config 各一个），只记录；真实 2B 是 {top: True, text: False}，运行时不共享。"""
        self._require_loaded()
        return dict(self._config_tie)

    @property
    def n_base_vocab(self) -> int:
        """基础词表大小（不含 added tokens），新行初始化取这些行的均值（D13-1）。"""
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
        """attach 时查好的特殊 token id（如 IM_END、THINK_END、IMG_CONTEXT）。"""
        self._require_loaded()
        return self._ids[token]

    # ------------------------------------------------------------------
    # 加载
    # ------------------------------------------------------------------

    def load(self, model_id: str, dtype: Any) -> None:
        """按注册键加载（带固定 revision）。dtype 由调用方给出，如 torch.bfloat16。"""
        entry = registry.get_model(model_id)
        if entry.family != registry.FAMILY_INTERNVL35:
            raise ValueError(f"{model_id} 属于 {entry.family} 家族，不是 {registry.FAMILY_INTERNVL35}")
        if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
            raise TypeError(f"dtype 必须是浮点 torch.dtype，收到 {dtype!r}")
        from transformers import AutoModelForImageTextToText, AutoProcessor

        processor = AutoProcessor.from_pretrained(entry.repo, revision=entry.revision)
        if type(processor).__name__ != "InternVLProcessor":
            raise TypeError(f"期望 InternVLProcessor，收到 {type(processor).__name__}")
        facts = {"image_seq_length": (processor.image_seq_length, TOKENS_PER_TILE),
                 "image_token": (processor.image_token, IMG_CONTEXT),
                 "start_image_token": (processor.start_image_token, IMG_START),
                 "end_image_token": (processor.end_image_token, IMG_END)}
        bad = {k: v for k, v in facts.items() if v[0] != v[1]}
        if bad:
            raise ValueError(f"官方处理器与本适配器的约定不一致（实际, 期望）：{bad}")
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

        模型若不在构造时给定的设备上，会被整体搬过去；dtype 不做转换，取模型参数的 dtype，要求全部浮点参数同一个 dtype。
        同一个模型可以挂到多个适配器上（例如不同的 tile 策略），但加 token / 注入查询会改模型与 tokenizer，那时请各用各的副本。
        """
        if self._model is not None:
            raise RuntimeError("这个适配器已经挂过模型；新模型请新建适配器")
        if type(model).__name__ != "InternVLForConditionalGeneration":
            raise TypeError(f"期望 InternVLForConditionalGeneration，收到 {type(model).__name__}")
        if not isinstance(chat_template, str) or not chat_template:
            raise ValueError("chat_template 为空")
        cfg = model.config
        tc, vc = cfg.text_config, cfg.vision_config
        for name, c in (("config", cfg), ("text_config", tc), ("vision_config", vc)):
            attn = getattr(c, "_attn_implementation", None)
            if attn != "sdpa":
                raise ValueError(f"{name} 的 attn_implementation 必须是 sdpa（任务书），收到 {attn!r}")

        # 视觉几何：config、图像处理器与 TILE_PX / TOKENS_PER_TILE 三方一致
        tile_hw = _as_pair(vc.image_size)
        patch_hw = _as_pair(vc.patch_size)
        if tile_hw != (TILE_PX, TILE_PX):
            raise ValueError(f"vision_config.image_size={vc.image_size}，与 tile 边长 {TILE_PX} 不一致")
        if tile_hw[0] % patch_hw[0] or tile_hw[1] % patch_hw[1]:
            raise ValueError(f"tile {tile_hw} 不能被 patch {patch_hw} 整除")
        side = [t // p * float(cfg.downsample_ratio) for t, p in zip(tile_hw, patch_hw)]
        if tuple(side) != tuple(float(s) for s in TILE_GRID_HW) or TILE_GRID_HW[0] * TILE_GRID_HW[1] != TOKENS_PER_TILE:
            raise ValueError(f"config 推出每个 tile 的网格是 {side}，与 {TILE_GRID_HW} 不一致")
        if int(cfg.image_seq_length) != TOKENS_PER_TILE:
            raise ValueError(f"config.image_seq_length={cfg.image_seq_length}，与 {TOKENS_PER_TILE} 不一致")
        if not type(image_processor).__name__.startswith("GotOcr2ImageProcessor"):
            raise TypeError(f"期望 GotOcr2ImageProcessor（torchvision 或 PIL 后端），收到 {type(image_processor).__name__}")
        if _size_hw(image_processor.size) != (TILE_PX, TILE_PX):
            raise ValueError(f"图像处理器 size={image_processor.size}，与 tile 边长 {TILE_PX} 不一致")

        # 特殊 token：每个都必须分词成单个 token；与 config / tokenizer 登记的一致
        ids: dict[str, int] = {}
        for s in (IM_START, IM_END, END_OF_TEXT, THINK_START, THINK_END, IMG_CONTEXT, IMG_START, IMG_END, VIDEO):
            enc = tokenizer(s, add_special_tokens=False)["input_ids"]
            if len(enc) != 1 or tokenizer.convert_tokens_to_ids(s) != enc[0]:
                raise ValueError(f"{s!r} 没有分词成单个已知 token：{enc}")
            ids[s] = enc[0]
        if cfg.image_token_id != ids[IMG_CONTEXT]:
            raise ValueError(f"config.image_token_id={cfg.image_token_id} 与 tokenizer 的 {IMG_CONTEXT} id {ids[IMG_CONTEXT]} 不一致")
        for attr, want in (("context_image_token", IMG_CONTEXT), ("start_image_token", IMG_START),
                           ("end_image_token", IMG_END), ("eos_token", IM_END), ("pad_token", END_OF_TEXT)):
            if getattr(tokenizer, attr, None) != want:
                raise ValueError(f"tokenizer.{attr}={getattr(tokenizer, attr, None)!r}，期望 {want!r}（model_notes §7.1）")

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
        self._config_tie = {"top": bool(getattr(cfg, "tie_word_embeddings", False)),
                            "text": bool(getattr(tc, "tie_word_embeddings", False))}

        # 模板自检：user 轮以 <|im_start|>user\n 开头、<|im_end|>\n 结束；assistant 头与 system 轮都能干净地拆出来。
        # 失败就撤销挂载。
        try:
            plain = self._render([_user_msg([{"type": "text", "text": "x"}])])
            if not plain.startswith(IM_START + "user\n") or not plain.endswith(IM_END + "\n"):
                raise ValueError(f"user 轮的渲染不是 <|im_start|>user\\n…<|im_end|>\\n 的形式：{plain!r}")
            for flag in (False, True):
                self.assistant_header_ids(flag)
                self.system_prefix_ids(flag)
        except Exception:
            self._model = None
            self._header = None
            self._prefix_cache.clear()
            raise

    # ------------------------------------------------------------------
    # 输入
    # ------------------------------------------------------------------

    def to_rgb(self, image: Any) -> Image.Image:
        """把一张图转成 RGB 的 PIL 图，不缩放：缩放与切 tile 由图像处理器按 tile 策略完成。"""
        if isinstance(image, Image.Image):
            return image if image.mode == "RGB" else image.convert("RGB")
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(f"numpy 图必须是 uint8 的 (H, W, 3)，收到 dtype={image.dtype} shape={image.shape}")
            return Image.fromarray(image)
        raise TypeError(f"图像必须是 PIL.Image 或 numpy 数组，收到 {type(image).__name__}")

    def build_inputs(self, images: Sequence[Any], instruction: str, order: Order) -> ModelInputs:
        """组装单条输入（只含 user 轮）。区间都是 input_ids 里的 token 下标（半开）。

        user 内容的顺序：instruction_first → [指令, 图1, 图2, …]；image_first → [图1, …, 指令]。模板把每张图渲染成
        `<IMG_CONTEXT>\\n`，这里展开成 `<img>` + `<IMG_CONTEXT>`×(256·tile 数) + `</img>`；ImageTokenInfo 的 span 是中间那段，
        grid_hw 的含义见模块说明（单 tile (16, 16)；n 个 tile (16·n, 16)，是 tile 的纵向堆叠，不是整图网格）。
        已调用 inject_query_tokens 时，查询 token 放在 user 内容最后一项之后、<|im_end|> 之前；这时每张图都必须是单 tile。
        system 轮（思考开关）与 assistant 头由 generate 追加，不在这里。
        """
        self._require_loaded()
        if order not in ORDERS:
            raise ValueError(f"order 必须是 {ORDERS} 之一，收到 {order!r}")
        if not isinstance(instruction, str):
            raise TypeError(f"instruction 必须是 str，收到 {type(instruction).__name__}")
        self._check_instruction(instruction)
        if isinstance(images, (Image.Image, np.ndarray)):
            raise TypeError("images 必须是图像的序列（单张图也要放进列表）")
        pil = [self.to_rgb(im) for im in images]
        n_img = len(pil)

        # 1. 渲染：指令槽先放哨兵，确认模板对指令原样输出
        text = self._render([_user_msg(self._content(n_img, instruction, order))])
        sent_text = self._render([_user_msg(self._content(n_img, _SENTINEL, order))])
        if sent_text.count(_SENTINEL) != 1:
            raise RuntimeError("哨兵在渲染结果里出现的次数不是 1")
        cut = sent_text.index(_SENTINEL)
        prefix, suffix = sent_text[:cut], sent_text[cut + len(_SENTINEL) :]
        if text != prefix + instruction + suffix:
            raise ValueError("聊天模板改写了指令文本，指令区间无法精确定位")
        if not text.endswith(IM_END + "\n"):
            raise RuntimeError(f"渲染没有以 {IM_END!r}+换行 结束")

        # 2. 图像：tile 参数全部显式传入（D13-5）
        tensors: dict[str, torch.Tensor] = {}
        n_tiles: list[int] = []
        if n_img:
            enc = self._ip(images=pil, return_tensors="pt", **self.processor_kwargs())
            n_tiles = [int(n) for n in enc["num_patches"]]
            pv = enc["pixel_values"]
            if len(n_tiles) != n_img or any(n < 1 for n in n_tiles):
                raise RuntimeError(f"图像处理器返回的 tile 数 {n_tiles} 与图数 {n_img} 不符")
            if self.always_single_tile and any(n != 1 for n in n_tiles):
                raise RuntimeError(f"单 tile 策略下处理器切出了 {n_tiles} 个 tile")
            if tuple(pv.shape) != (sum(n_tiles), 3, TILE_PX, TILE_PX):
                raise RuntimeError(f"pixel_values 形状 {tuple(pv.shape)}，期望 ({sum(n_tiles)}, 3, {TILE_PX}, {TILE_PX})")
            tensors["pixel_values"] = pv.to(device=self._device, dtype=self._dtype)

        # 3. 展开图像占位符并分词；指令两侧分别分词，拼接必须与整段分词逐个相同
        n_pre = prefix.count(IMG_CONTEXT)
        if n_pre + suffix.count(IMG_CONTEXT) != n_img:
            raise RuntimeError("模板里的图像占位符个数与图数不一致")
        pre_exp = self._expand(prefix, n_tiles[:n_pre])
        post_exp = self._expand(suffix, n_tiles[n_pre:])
        ids_pre, ids_ins, ids_post = self._encode(pre_exp), self._encode(instruction), self._encode(post_exp)
        ids = ids_pre + ids_ins + ids_post
        if ids != self._encode(pre_exp + instruction + post_exp):
            raise ValueError("指令与相邻模板文本在分词时合并，指令区间无法精确定位")
        instruction_span = Span(len(ids_pre), len(ids_pre) + len(ids_ins))

        # 4. 图像区间：<IMG_CONTEXT> 的连续段，个数、长度、前后标记逐项核对
        runs = _runs(ids, self._ids[IMG_CONTEXT])
        if [e - s for s, e in runs] != [TOKENS_PER_TILE * n for n in n_tiles]:
            raise RuntimeError(f"图像 token 段 {runs} 与每张图的 tile 数 {n_tiles} 不一致")
        infos = []
        for (s, e), n in zip(runs, n_tiles):
            if ids[s - 1] != self._ids[IMG_START] or ids[e] != self._ids[IMG_END]:
                raise RuntimeError(f"图像段 [{s}, {e}) 前后不是 {IMG_START} / {IMG_END}")
            infos.append(ImageTokenInfo(Span(s, e), (TILE_GRID_HW[0] * n, TILE_GRID_HW[1]), n_pixels=n * TILE_PX * TILE_PX))

        # 5. 查询区间
        query_span = None
        if self._query_n is not None:
            multi = [(k, n) for k, n in enumerate(n_tiles) if n != 1]
            if multi:
                raise ValueError(f"查询 token 只支持单 tile 的图：第 {multi[0][0]} 张图切成了 {multi[0][1]} 个 tile，"
                                 "多 tile 的视觉 token 不构成整图网格")
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
        return ModelInputs(
            tensors=tensors,
            order=order,
            images=tuple(infos),
            instruction_span=instruction_span,
            query_span=query_span,
            query_position_mode="seq1d" if query_span is not None else None,
        )

    # ------------------------------------------------------------------
    # 位置
    # ------------------------------------------------------------------

    def position_ids(self, inputs: ModelInputs) -> torch.Tensor:
        """(1, L) 的一维位置 0..L-1（§7.4：图像 token 与查询 token 都逐个加 1，与模型不传 position_ids 时自己算的相同）。"""
        self._require_loaded()
        ids = inputs.tensors["input_ids"]
        if ids.dim() != 2 or ids.shape[0] != 1:
            raise ValueError(f"只支持单条样本，input_ids 形状 {tuple(ids.shape)}")
        return torch.arange(ids.shape[1], device=ids.device).unsqueeze(0)

    # ------------------------------------------------------------------
    # 前向：隐藏状态与视觉特征
    # ------------------------------------------------------------------

    def hidden_states(
        self, inputs: ModelInputs, layers: Sequence[int], *, query_embeds: torch.Tensor | None = None
    ) -> HiddenStates:
        """指定层在图像 token 与查询 token 位置的隐藏状态（层号约定见 base.py）。只接受单 tile 的图（探针要整图网格）。

        不强制 no_grad：沿用调用方的梯度模式（E3a 训练要让梯度流回 query_embeds 与 LoRA）；
        只做推理时请在 torch.no_grad() 里调用。
        """
        layer_list = self._check_layers(layers)
        self._require_single_tile(inputs, where="hidden_states")
        hs = self._run(inputs, query_embeds, where="hidden_states")
        img_idx = self._image_index(inputs)
        image = {l: hs[l][0].index_select(0, img_idx) for l in layer_list}
        query = None
        if inputs.query_span is not None:
            q_idx = torch.arange(inputs.query_span.start, inputs.query_span.end, device=img_idx.device)
            query = {l: hs[l][0].index_select(0, q_idx) for l in layer_list}
        return HiddenStates(layers=tuple(layer_list), image=image, query=query)

    def sequence_hidden_states(
        self, inputs: ModelInputs, layers: Sequence[int], *, query_embeds: torch.Tensor | None = None
    ) -> dict[int, torch.Tensor]:
        """诊断用：指定层在整条序列上的隐藏状态，每层形状 [L, hidden]。与 hidden_states 走同一条前向路径。

        不涉及视觉 token 的网格含义，所以多 tile 的输入也接受（冒烟用它核对多 tile 时适配器路径 = HF 原生路径）。
        """
        layer_list = self._check_layers(layers)
        hs = self._run(inputs, query_embeds, where="sequence_hidden_states")
        return {l: hs[l][0] for l in layer_list}

    def vision_features(self, inputs: ModelInputs) -> torch.Tensor:
        """进入 LLM 之前、经过 `multi_modal_projector` 投影之后的视觉 token，[n_image_tokens, hidden]，多图按顺序拼接。

        就是前向时 masked_scatter 进嵌入序列的那组向量，所以与 hidden_states(inputs, [0]) 在图像位置上逐位相同
        （测试与冒烟都查）。E2 参考线 / E3a 对照用，属于探针相关，只接受单 tile 的图。
        """
        self._check_entry(inputs, None, where="vision_features")
        self._require_single_tile(inputs, where="vision_features")
        if not inputs.images:
            return torch.empty((0, self.hidden_size()), dtype=self._dtype, device=self._device)
        return self._image_features(inputs)

    # ------------------------------------------------------------------
    # 生成
    # ------------------------------------------------------------------

    def system_prefix_ids(self, enable_thinking: bool) -> tuple[int, ...]:
        """思考开关对应的 system 轮 token（加在 user 轮之前）：开 = R1 system prompt 的整轮；关 = 空（D13-2）。"""
        self._require_loaded()
        if not isinstance(enable_thinking, bool):
            raise TypeError(f"enable_thinking 必须是 bool，收到 {enable_thinking!r}")
        if enable_thinking not in self._prefix_cache:
            if not enable_thinking:
                prefix: tuple[int, ...] = ()
            else:
                probe = _user_msg([{"type": "text", "text": "x"}])
                user = self._render([probe])
                sys_turn = self._render([_system_msg()])
                both = self._render([_system_msg(), probe])
                if sys_turn != IM_START + "system\n" + R1_SYSTEM_PROMPT + IM_END + "\n":
                    raise RuntimeError(f"system 轮的渲染不是 <|im_start|>system\\n<R1><|im_end|>\\n：{sys_turn[:80]!r}…")
                if both != sys_turn + user:
                    raise RuntimeError("system 轮 + user 轮的渲染不等于两段各自渲染的拼接")
                prefix = tuple(self._encode(sys_turn))
                if list(prefix) + self._encode(user) != self._encode(both):
                    raise RuntimeError("system 轮单独分词与整段分词不一致")
            self._prefix_cache[enable_thinking] = prefix
        return self._prefix_cache[enable_thinking]

    def assistant_header_ids(self, enable_thinking: bool) -> tuple[int, ...]:
        """assistant 头的 token：apply_chat_template(add_generation_prompt=True) 比不加时多出的部分（`<|im_start|>assistant\\n`）。

        InternVL 的模板没有思考开关，两种取值的头相同；参数只为与其他家族的接口一致，思考开关体现在 system_prefix_ids。
        """
        self._require_loaded()
        if not isinstance(enable_thinking, bool):
            raise TypeError(f"enable_thinking 必须是 bool，收到 {enable_thinking!r}")
        if self._header is None:
            probe = [_user_msg([{"type": "text", "text": "x"}])]
            plain = self._render(probe)
            full = self._render(probe, add_generation_prompt=True)
            if not full.startswith(plain) or not full[len(plain) :].startswith(IM_START):
                raise RuntimeError("加了生成提示的渲染不是 user 轮 + <|im_start|>… 的形式")
            header = self._encode(full[len(plain) :])
            if self._encode(plain) + header != self._encode(full):
                raise RuntimeError("assistant 头单独分词与整段分词不一致")
            self._header = tuple(header)
        return self._header

    def prompt_ids(self, inputs: ModelInputs, enable_thinking: bool) -> list[int]:
        """generate 实际送进模型的完整提示：system 轮（开思考时）+ user 轮（inputs）+ assistant 头。"""
        self._require_loaded()
        ids = inputs.tensors["input_ids"]
        if ids.dim() != 2 or ids.shape[0] != 1:
            raise ValueError(f"只支持单条样本，input_ids 形状 {tuple(ids.shape)}")
        return (list(self.system_prefix_ids(enable_thinking)) + ids[0].tolist()
                + list(self.assistant_header_ids(enable_thinking)))

    def generate(
        self,
        inputs: ModelInputs,
        *,
        enable_thinking: bool,
        allowed_token_ids: Sequence[int] | None = None,
        max_new_tokens: int,
        **gen_kwargs: Any,
    ) -> GenerationResult:
        """单条样本生成。默认 greedy；停止 token 只有 <|im_end|>（id 从 tokenizer 查，D13-3）。

        enable_thinking=True 时在 user 轮之前加 R1 system 轮（D13-2）。allowed_token_ids 非空时，每一步把其他 id 的 logits
        置为 -inf；这时若希望能停下，调用方要把停止 token 放进允许集合。返回的 token_ids 含生成出的停止 token；
        n_prompt_tokens 含 system 轮与 assistant 头；n_visual_tokens 只数 <IMG_CONTEXT>。
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
        """批量生成（约定见 base.ModelAdapter.generate_batch）。每行的提示 = prompt_ids（system 轮只在开思考时加）；
        左填充用 pad（`<|endoftext|>`）；各行的 tile 按行的顺序拼接，每行的 tile 数可以不同（多 tile 策略下生成可用）。
        """
        return self._generate_rows(inputs_list, enable_thinking=enable_thinking, allowed_token_ids=allowed_token_ids,
                                   max_new_tokens=max_new_tokens, gen_kwargs=gen_kwargs, where="generate_batch")

    def _prompt_row(self, inputs: ModelInputs, enable_thinking: bool) -> batching.PromptRow:
        return batching.PromptRow(
            ids=tuple(self.prompt_ids(inputs, enable_thinking)),
            n_visual_tokens=sum(len(i.span) for i in inputs.images),
            n_image_pixels=batching.image_pixels(inputs),
        )

    def _generate_rows(self, inputs_list: Sequence[ModelInputs], *, enable_thinking: bool,
                       allowed_token_ids: Sequence[int] | None, max_new_tokens: int, gen_kwargs: dict[str, Any],
                       where: str) -> list[GenerationResult]:
        self._require_loaded()
        batching.check_generate_args(enable_thinking, max_new_tokens)
        items = batching.check_inputs_list(inputs_list, where=where)
        batching.check_reserved(gen_kwargs, _RESERVED_GEN_KWARGS)
        for inp in items:
            self._check_entry(inp, None, where=where)
        rows, kw = self._batch_kw(items, enable_thinking)
        pad_id = self._ids[END_OF_TEXT]
        processors = list(gen_kwargs.pop("logits_processor", None) or [])
        if allowed_token_ids is not None:
            vocab_rows = self._model.get_output_embeddings().weight.shape[0]
            processors.insert(0, _AllowedTokensProcessor(allowed_token_ids, vocab_rows))
        gen_kwargs.setdefault("do_sample", False)
        stop_ids = [self._ids[s] for s in STOP_TOKENS]
        out, wall = batching.run_generate(self._model, kw, device=self._device, max_new_tokens=max_new_tokens,
                                          stop_ids=stop_ids, pad_id=pad_id, processors=processors, gen_kwargs=gen_kwargs)
        return batching.make_results(rows, out, kw["input_ids"].shape[1], wall=wall, stop_ids=stop_ids, pad_id=pad_id,
                                     max_new_tokens=max_new_tokens, split_answer=self.split_answer,
                                     enable_thinking=enable_thinking)

    def _batch_kw(self, items: Sequence[ModelInputs],
                  enable_thinking: bool) -> tuple[list[batching.PromptRow], dict[str, Any]]:
        """generate / generate_batch / option_logits 共用的拼批：每行的提示（prompt_ids）左填充成一批，各行的 tile 按行的顺序拼接。"""
        rows = [self._prompt_row(inp, enable_thinking) for inp in items]
        kw: dict[str, Any] = batching.batch_prompt(rows, pad_id=self._ids[END_OF_TEXT], device=self._device)
        with_images = [inp for inp in items if inp.images]
        if with_images:
            kw["pixel_values"] = torch.cat([inp.tensors["pixel_values"] for inp in with_images], dim=0)
        return rows, kw

    def option_logits(
        self,
        inputs_list: Sequence[ModelInputs],
        options: Sequence[Sequence[str]],
        *,
        enable_thinking: bool,
    ) -> list[OptionScores]:
        """受限选项读数（约定见 base.ModelAdapter.option_logits；DEV-0016）。提示与左填充同 generate_batch（_batch_kw）；
        第一个生成位置 = assistant 头 `<|im_start|>assistant\\n` 之后（开思考时 R1 system 轮在 user 轮之前）。多 tile 的图可以用
        （与 generate 相同）。"""
        self._require_loaded()
        batching.check_enable_thinking(enable_thinking)
        items = batching.check_inputs_list(inputs_list, where="option_logits")
        opts = batching.check_options(options, len(items))
        for inp in items:
            self._check_entry(inp, None, where="option_logits")
        rows, kw = self._batch_kw(items, enable_thinking)
        return batching.option_logits_rows(self._model, self._tok, rows, kw, opts, device=self._device,
                                           stop_ids=[self._ids[s] for s in STOP_TOKENS], pad_id=self._ids[END_OF_TEXT])

    def split_answer(self, token_ids: Sequence[int], *, enable_thinking: bool) -> tuple[str, str, bool]:
        """把生成的 token 拆成 (text, raw_text, thinking_unclosed)（D10；规则与 Gemma 适配器相同）。

        R1 system prompt 要求的写法（§7.2 的原文）："Enclose this entire thinking process within <think> and </think> tags"，
        "Separate the answer from the think section with a newline"。思考段的开头 <think> 由模型写出、不在提示里，
        所以规则只看输出（按 token id 找标记），不看 enable_thinking（参数只为与其他家族的接口对称）：
        - 最后一个 <think> 之后没有 </think>（思考段开了没关，例如 max_new_tokens 用完）：text 为空串，thinking_unclosed = True；
        - 否则 text 取最后一个 </think> 之后的内容；没有思考段（模型没写 <think> 就直接作答，或关了思考）就取全部。
          末尾的停止 token 去掉，其余特殊 token（包括 grid token）保留，首尾换行去掉；thinking_unclosed = False。
        raw_text：全部生成 token 的解码，含思考段与特殊 token。
        """
        self._require_loaded()
        if not isinstance(enable_thinking, bool):
            raise TypeError(f"enable_thinking 必须是 bool，收到 {enable_thinking!r}")
        ids = [int(i) for i in token_ids]
        raw_text = self._tok.decode(ids, skip_special_tokens=False)
        last_open = max((k for k, v in enumerate(ids) if v == self._ids[THINK_START]), default=None)
        last_close = max((k for k, v in enumerate(ids) if v == self._ids[THINK_END]), default=None)
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

        position_mode 只接受 "seq1d"（一维顺序位置，与文本相同）；"grid2d" 报 ValueError（见模块说明）。
        占位符是 add_tokens 新增的一个专用 token（QUERY_PLACEHOLDER，与 Qwen 适配器同名），首次调用时加入，嵌入行（与 lm_head 行）
        按 D13-1 初始化为基础词表行的均值。它不计入 E3b 的 300 个 token 预算，但占用一个嵌入行。
        """
        self._require_loaded()
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise ValueError(f"n 必须是正整数，收到 {n!r}")
        if position_mode not in POSITION_MODES:
            raise ValueError(f"position_mode 必须是 {POSITION_MODES} 之一，收到 {position_mode!r}")
        if position_mode == "grid2d":
            raise ValueError(_GRID2D_UNSUPPORTED)
        if self._query_token_id is None:
            (self._query_token_id,) = self._add_new_tokens([QUERY_PLACEHOLDER])
        self._query_n = n

    def clear_query_tokens(self) -> None:
        """取消查询 token：之后的 build_inputs 不再追加（占位符 token 仍留在词表里）。"""
        self._query_n = None

    def add_grid_tokens(self, names: Sequence[str]) -> dict[str, int]:
        """加入像素 token 通道用的新 token，返回 名字 → id。

        预算按累计计（≤ MAX_NEW_GRID_TOKENS，查询占位符不计）。先复用 257 个预留嵌入行；累计需要的行数超过现有行数时，
        嵌入（lm_head 不共享时连同 lm_head）扩到恰好需要的行数，已有的行逐位不变，从不缩表（D13-1）。
        新 token 的嵌入行与 lm_head 行各自写成自己基础词表行的精确均值；共享时只写一次。
        """
        self._require_loaded()
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

    def hidden_size(self) -> int:
        self._require_loaded()
        return int(self._model.config.text_config.hidden_size)

    # ------------------------------------------------------------------
    # 输出解析（DEV-0006）
    # ------------------------------------------------------------------

    def parse_boxes(self, text: str, *, width: float, height: float) -> BoxParse:
        """本家族的原生框 `<ref>…</ref><box>[[x1, y1, x2, y2], …]</box>`，规则见模块级的 parse_boxes（D19）。
        不需要加载模型；parse_one_box 由基类在它之上给出。"""
        return parse_boxes(text, width=width, height=height)  # 模块级函数（方法体里的名字按全局作用域查找）

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
            content.append({"type": "text", "text": QUERY_PLACEHOLDER * self._query_n})
        return content

    def _render(self, messages: list[dict[str, Any]], *, add_generation_prompt: bool = False) -> str:
        return self._tok.apply_chat_template(
            messages,
            chat_template=self._template,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    def _encode(self, text: str) -> list[int]:
        return list(self._tok(text, add_special_tokens=False)["input_ids"])

    def _expand(self, text: str, counts: Sequence[int]) -> str:
        """每个 <IMG_CONTEXT> 占位符 → <img> + <IMG_CONTEXT>×(256·tile 数) + </img>（同 InternVLProcessor.replace_image_token）。"""
        parts = text.split(IMG_CONTEXT)
        if len(parts) != len(counts) + 1:
            raise RuntimeError("图像占位符个数与展开计数不一致")
        out = [parts[0]]
        for n, part in zip(counts, parts[1:]):
            out.append(IMG_START + IMG_CONTEXT * (TOKENS_PER_TILE * n) + IMG_END)
            out.append(part)
        return "".join(out)

    def _check_instruction(self, instruction: str) -> None:
        if _SENTINEL in instruction:
            raise ValueError(f"指令里含有内部哨兵字符串 {_SENTINEL!r}")
        ids = set(self._encode(instruction))
        forbidden = [s for s in _STRUCTURAL_TOKENS if self._ids[s] in ids]
        if self._query_token_id is not None and self._query_token_id in ids:
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

    def _require_single_tile(self, inputs: ModelInputs, *, where: str) -> None:
        """查询 / 探针相关的方法只接受单 tile 的图：多 tile 的 (16·n, 16) 是 tile 堆叠，不是整图网格。"""
        if not isinstance(inputs, ModelInputs):
            raise TypeError(f"inputs 必须是 ModelInputs，收到 {type(inputs).__name__}")
        for k, info in enumerate(inputs.images):
            if info.grid_hw != TILE_GRID_HW:
                raise ValueError(
                    f"{where}：第 {k} 张图切成了 {tiles_of(info)} 个 tile，grid_hw={info.grid_hw} 是 tile 的纵向堆叠，不是整图网格；"
                    "查询 / 探针相关的方法只接受单 tile 的图（crop_to_patches=False 或 max_patches=1，或这张图本来就只切出 1 个 tile）"
                )

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
        inner = self._model.model
        for mod_name, weight in (
            ("lm_head", self._model.get_output_embeddings().weight),
            ("vision_tower.patch_embeddings", inner.vision_tower.embeddings.patch_embeddings.projection.weight),
            ("multi_modal_projector.linear_1", inner.multi_modal_projector.linear_1.weight),
        ):
            if weight.device != m_dev:
                raise AdapterDeviceError(f"{where}：{mod_name} 在 {weight.device}，嵌入层在 {m_dev}")
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
        """与 InternVLModel.forward 相同的调用：视觉塔 → 去 CLS → pixel shuffle → multi_modal_projector；参数取 config 的值。"""
        cfg = self._model.config
        out = self._model.model.get_image_features(
            pixel_values=inputs.tensors["pixel_values"],
            vision_feature_layer=cfg.vision_feature_layer,
            vision_feature_select_strategy=cfg.vision_feature_select_strategy,
            return_dict=True,
        )
        feats = out.pooler_output.reshape(-1, self.hidden_size()).to(device=self._device, dtype=self._dtype)
        n_expected = sum(len(i.span) for i in inputs.images)
        if tuple(feats.shape) != (n_expected, self.hidden_size()):
            raise RuntimeError(f"视觉特征形状 {tuple(feats.shape)}，期望 ({n_expected}, {self.hidden_size()})")
        return feats

    def _input_embeds(self, inputs: ModelInputs, query_embeds: torch.Tensor | None) -> torch.Tensor:
        """与 InternVLModel.forward 相同的并入方式：嵌入查表 → 图像位置 masked_scatter 视觉特征；再替换查询位置。"""
        ids = inputs.tensors["input_ids"]
        embeds = self._model.get_input_embeddings()(ids)
        if inputs.images:
            feats = self._image_features(inputs)
            mask = (ids == self._ids[IMG_CONTEXT]).unsqueeze(-1)
            embeds = embeds.masked_scatter(mask, feats)
        if query_embeds is not None:
            qs, qe = inputs.query_span.start, inputs.query_span.end
            embeds = torch.cat([embeds[:, :qs], query_embeds.unsqueeze(0), embeds[:, qe:]], dim=1)
        return embeds

    def _run(self, inputs: ModelInputs, query_embeds: torch.Tensor | None, *, where: str) -> tuple[torch.Tensor, ...]:
        self._check_entry(inputs, query_embeds, where=where)
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

    def _add_new_tokens(self, names: list[str]) -> list[int]:
        """加 token 的公共路径（grid token 与查询占位符）：分词器 → 行数（必要时扩表）→ 行初始化。"""
        vocab = self._tok.get_vocab()
        clash = [n for n in names if n in vocab]
        if clash:
            raise ValueError(f"这些 token 已在词表里：{clash[:5]}")
        n_before = len(self._tok)
        expected = list(range(n_before, n_before + len(names)))
        n_added = self._tok.add_tokens(names, special_tokens=True)
        ids = [self._tok.convert_tokens_to_ids(n) for n in names]
        if n_added != len(names) or ids != expected:
            raise RuntimeError(f"add_tokens 加了 {n_added} 个，id {ids[:3]}…，期望从 {n_before} 起连续")
        for name, i in zip(names, ids):
            if self._encode(name) != [i]:
                raise RuntimeError(f"{name!r} 没有分词成单个 token")
        if self._encode("".join(names)) != ids:
            raise RuntimeError("新 token 首尾相接时没有逐个分词")
        self._ensure_rows(n_before + len(names))
        rows = self._model.get_input_embeddings().weight.shape[0]
        if max(ids) >= rows or self._model.get_output_embeddings().weight.shape[0] != rows:
            raise RuntimeError(f"新 id 超出嵌入行数 {rows}")
        self._init_rows(ids)
        return ids

    def _ensure_rows(self, needed: int) -> None:
        """行数不够时扩到恰好 needed 行（不共享时 lm_head 同步），已有行逐位不变；够了什么都不做，从不缩表。"""
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
                new[n_old:] = self._base_mean(old)
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

    def _base_mean(self, weight: torch.Tensor) -> torch.Tensor:
        """基础词表行 [0, n_base_vocab) 的均值：float64 分块累加，再转回参数 dtype。"""
        n, dim = self._n_base_vocab, weight.shape[1]
        step = max(1, _MEAN_CHUNK_BYTES // (dim * 8))
        acc = torch.zeros(dim, dtype=torch.float64, device=weight.device)
        for s in range(0, n, step):
            acc += weight[s : min(s + step, n)].to(torch.float64).sum(dim=0)
        return (acc / n).to(weight.dtype)

    def _init_rows(self, ids: list[int]) -> None:
        """新 token 的嵌入行（不共享时连同 lm_head 行）写成各自基础词表行的均值；共享时只写一次。"""
        idx = torch.tensor(ids, dtype=torch.long, device=self._device)
        with torch.no_grad():
            emb = self._model.get_input_embeddings().weight
            emb[idx] = self._base_mean(emb)
            if not self._tied:
                head = self._model.get_output_embeddings().weight
                head[idx] = self._base_mean(head)


def _user_msg(content: list[dict[str, str]]) -> dict[str, Any]:
    return {"role": "user", "content": content}


def _system_msg() -> dict[str, Any]:
    """card 的写法：system 轮的内容是一段 text（§7.2 的渲染即由此得到）。"""
    return {"role": "system", "content": [{"type": "text", "text": R1_SYSTEM_PROMPT}]}


# ---------------------------------------------------------------------------
# 原生坐标 → 内部绝对像素 xyxy（D13-4；docs/model_notes.md §7.5）
# ---------------------------------------------------------------------------

BOX_OPEN = "<box>"
BOX_CLOSE = "</box>"
REF_OPEN = "<ref>"
REF_CLOSE = "</ref>"

# 失败原因与结果类型的定义在 src/parsing（DEV-0006）；这里的旧名字保持可导入。BoxParseFailure 是 ParseFailure 的别名，
# 原有的 6 个取值（no_box、unbalanced_tags、malformed、out_of_range、inverted、multiple_boxes）不变。
BoxParseFailure = ParseFailure

_BOX_TAG_RE = re.compile(r"</?box>")
_QUAD_NC = r"\[\s*[0-9]+\s*,\s*[0-9]+\s*,\s*[0-9]+\s*,\s*[0-9]+\s*\]"
_BOX_LIST_RE = re.compile(rf"\s*\[\s*{_QUAD_NC}(?:\s*,\s*{_QUAD_NC})*\s*\]\s*")
_QUAD_RE = re.compile(r"\[\s*([0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-9]+)\s*\]")


def _ref_label(text: str, box_open: int) -> str | None:
    """紧挨在 text[box_open] 处的 <box> 前面的 `<ref>…</ref>` 的内容（原样，不去空白）；两者之间只允许空白。没有就是 None。"""
    head = text[:box_open].rstrip()
    if not head.endswith(REF_CLOSE):
        return None
    body_end = len(head) - len(REF_CLOSE)
    start = head.rfind(REF_OPEN, 0, body_end)
    if start < 0:
        return None
    label = head[start + len(REF_OPEN) : body_end]
    return None if REF_CLOSE in label else label


def parse_boxes(text: str, *, width: float, height: float) -> BoxParse:
    """解析 InternVL 的原生框 `<box>[[x1, y1, x2, y2], …]</box>`，换算成内部绝对像素 xyxy（src/constants.py 的 BOX_FORMAT）。

    格式依据（§7.5 原文）："we normalize each coordinate box to the range [0, 1000] using the width and height of the image,
    and round the normalized coordinates to integers"，回答写成 `<ref>class name</ref><box>[[x1, y1, x2, y2], ...]</box>`。
    width、height 是原图的宽高（像素）：归一化按原图算，与 tile 切法无关。换算 x = x_norm · W / 1000（norm1000_to_abs，不取整）。

    规则（不补齐、不截断、不猜）：
    - `<box>` 标签之外的文字（停止 token、说明文字）一律忽略；
    - 多个框：一个标签里的多个 [..] 与多个标签都接受，按出现顺序全部返回；只要一个框的场景（REC）用 parse_one_box；
    - 检查顺序与失败原因：没有标签 → NO_BOX；标签不成对 → UNBALANCED_TAGS；任一标签内容不是 [[四个非负整数], …] → MALFORMED；
      任一坐标 > 1000 → OUT_OF_RANGE；任一框 x2 < x1 或 y2 < y1 → INVERTED。任何一个框坏了整条判失败，不返回其余的框。
    - 小数（如 24.5）与负数算格式损坏（MALFORMED），不是越界：原生格式是 0–1000 的整数（D19）。x1 == x2 的退化框照样返回。
    - label（DEV-0006）：紧挨在这个 `<box>` 前面的 `<ref>…</ref>` 的内容，中间只允许空白；没有就是 None。同一个标签里的
      几个框共用这个 label。label 只是附带信息，不影响成功与否。
    """
    if not isinstance(text, str):
        raise TypeError(f"text 必须是 str，收到 {type(text).__name__}")
    if not (width > 0 and height > 0):
        raise ValueError(f"原图宽高必须为正：W={width}, H={height}")
    tags = list(_BOX_TAG_RE.finditer(text))
    if not tags:
        return BoxParse(None, None, BoxParseFailure.NO_BOX)
    if len(tags) % 2 != 0 or any(m.group() != (BOX_OPEN if k % 2 == 0 else BOX_CLOSE) for k, m in enumerate(tags)):
        return BoxParse(None, None, BoxParseFailure.UNBALANCED_TAGS)
    quads: list[tuple[int, int, int, int]] = []
    labels: list[str | None] = []
    for k in range(0, len(tags), 2):
        segment = text[tags[k].end() : tags[k + 1].start()]
        if not _BOX_LIST_RE.fullmatch(segment):
            return BoxParse(None, None, BoxParseFailure.MALFORMED)
        found = [tuple(int(v) for v in m.groups()) for m in _QUAD_RE.finditer(segment)]
        quads.extend(found)
        labels.extend([_ref_label(text, tags[k].start())] * len(found))
    if any(v > NORM1000_SCALE for q in quads for v in q):
        return BoxParse(None, None, BoxParseFailure.OUT_OF_RANGE)
    if any(q[2] < q[0] or q[3] < q[1] for q in quads):
        return BoxParse(None, None, BoxParseFailure.INVERTED)
    boxes = tuple(check_box(norm1000_to_abs(list(q), width=width, height=height)) for q in quads)
    return BoxParse(boxes, tuple(labels), None)


def parse_one_box(text: str, *, width: float, height: float) -> OneBoxParse:
    """只要一个框（如 RefCOCO 的 REC）：parse_boxes 失败就原样给出失败原因；框多于一个 → MULTIPLE_BOXES，不猜用哪个
    （规则见 src.parsing.one_box，与适配器方法 parse_one_box 相同）。"""
    return one_box(parse_boxes(text, width=width, height=height))


# ---------------------------------------------------------------------------
# 小型随机 InternVL3.5：CPU 测试与冒烟脚本 --tiny 的彩排（不下载权重）
# ---------------------------------------------------------------------------

TINY_TEXT = {
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 4,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "max_window_layers": 4,
}
TINY_VISION = {"hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 1, "num_attention_heads": 2}


def load_pinned_components(model_key: str = TINY_SOURCE_KEY) -> tuple[Any, Any, Any, str]:
    """从登记的固定 revision 下载 config、tokenizer、图像处理器与聊天模板（都是小文件，不含权重）。

    图像处理器固定用 PIL 后端（GotOcr2ImageProcessorPil），开发机没有 torchvision 也能跑；服务器上的 load() 走官方
    AutoProcessor（torchvision 后端）。两者切 tile 的规则相同，只在缩放的舍入上可能不同，接口相同。
    """
    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.got_ocr2.image_processing_pil_got_ocr2 import GotOcr2ImageProcessorPil

    entry = registry.get_model(model_key)
    if entry.family != registry.FAMILY_INTERNVL35:
        raise ValueError(f"{model_key} 不是 InternVL3.5")
    config = AutoConfig.from_pretrained(entry.repo, revision=entry.revision)
    tokenizer = AutoTokenizer.from_pretrained(entry.repo, revision=entry.revision)
    image_processor = GotOcr2ImageProcessorPil.from_pretrained(entry.repo, revision=entry.revision)
    template = tokenizer.chat_template
    if not isinstance(template, str) or not template:
        raise RuntimeError(f"{entry.repo}@{entry.revision} 没有取到聊天模板")
    return config, tokenizer, image_processor, template


def reference_processor(components: tuple[Any, Any, Any, str], model_key: str = TINY_SOURCE_KEY) -> Any:
    """官方 `InternVLProcessor` 类的一个实例，只供比对（测试与冒烟 --tiny）；适配器本身从不用它。

    开发机没有 torchvision，`InternVLVideoProcessor` 构造不出来，ProcessorMixin 的类型检查又不接受别的视频处理器。
    这里在构造期间临时跳过那一项类型检查、用 `BaseVideoProcessor()` 占位（本项目不送视频）；图像与文本路径——
    `__call__`、`replace_image_token`、占位符替换、分词——都是官方代码。image_seq_length 取固定 revision 的
    processor_config.json，与服务器上 AutoProcessor 的来源相同。
    """
    from unittest import mock

    from huggingface_hub import hf_hub_download
    from transformers.models.internvl.processing_internvl import InternVLProcessor
    from transformers.processing_utils import ProcessorMixin
    from transformers.video_processing_utils import BaseVideoProcessor

    entry = registry.get_model(model_key)
    if entry.family != registry.FAMILY_INTERNVL35:
        raise ValueError(f"{model_key} 不是 InternVL3.5")
    path = hf_hub_download(entry.repo, "processor_config.json", revision=entry.revision)
    seq_len = int(json.loads(Path(path).read_text())["image_seq_length"])
    _, tokenizer, image_processor, template = components
    with mock.patch.object(ProcessorMixin, "check_argument_for_proper_class", lambda self, name, arg: type(arg)):
        return InternVLProcessor(image_processor=image_processor, tokenizer=tokenizer, video_processor=BaseVideoProcessor(),
                                 image_seq_length=seq_len, chat_template=template)


def tiny_config(real_config: Any, *, tie_word_embeddings: bool) -> Any:
    """把真实 config 缩成小型随机模型的 config：LLM 4 层全注意力、hidden 64；视觉塔 1 层、hidden 64。

    词表、特殊 token id、tile 边长 448、patch 14、downsample 0.5、每 tile 256 token、head_dim 与 RoPE 参数保持真实值。
    共享开关顶层与 text_config 一起设（真实 2B 的 config 顶层是 True、运行时却不共享，见模块说明；小模型默认按运行时建成不共享）。
    视觉塔的 initializer_range 改用 text_config 的值：真实 config 里是 1e-10（反正加载权重），随机初始化下视觉特征只有 1e-8 量级，
    图像内容几乎不影响任何输出，测试就查不出图像相关的错误。
    """
    d = real_config.to_dict()
    tc, vc = d["text_config"], d["vision_config"]
    n = TINY_TEXT["num_hidden_layers"]
    types = list(tc["layer_types"][:n])
    if set(types) != {"full_attention"}:
        raise ValueError(f"真实 config 的前 {n} 层不全是 full_attention：{types}")
    tc.update(TINY_TEXT, layer_types=types, tie_word_embeddings=bool(tie_word_embeddings))
    vc.update(TINY_VISION, initializer_range=tc["initializer_range"])
    d["tie_word_embeddings"] = bool(tie_word_embeddings)
    return type(real_config).from_dict(d)


def build_tiny_model(real_config: Any, *, tie_word_embeddings: bool = False, dtype: torch.dtype = torch.float32,
                     seed: int = 0) -> Any:
    """按 tiny_config 构造随机初始化的 InternVLForConditionalGeneration（sdpa，eval）。"""
    from transformers import AutoModelForImageTextToText

    cfg = tiny_config(real_config, tie_word_embeddings=tie_word_embeddings)
    torch.manual_seed(seed)
    model = AutoModelForImageTextToText.from_config(cfg, attn_implementation="sdpa", dtype=dtype)
    return model.eval()


def make_tiny_adapter(
    components: tuple[Any, Any, Any, str],
    *,
    crop_to_patches: bool,
    max_patches: int,
    tie_word_embeddings: bool = False,
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
    device: str | torch.device = "cpu",
    model: Any = None,
    source_key: str = TINY_SOURCE_KEY,
) -> InternVL35Adapter:
    """用 load_pinned_components 的结果挂一个小型随机模型的适配器。

    注意：add_grid_tokens / inject_query_tokens 会改 tokenizer 与模型；需要彼此独立的适配器时，传入各自的副本。
    """
    config, tokenizer, image_processor, template = components
    if model is None:
        model = build_tiny_model(config, tie_word_embeddings=tie_word_embeddings, dtype=dtype, seed=seed)
    adapter = InternVL35Adapter(device=device, crop_to_patches=crop_to_patches, max_patches=max_patches)
    adapter.attach(model, tokenizer=tokenizer, image_processor=image_processor, chat_template=template,
                   model_key=f"tiny({source_key})")
    return adapter


# ---------------------------------------------------------------------------
# 冒烟：scripts/smoke_adapter.py 在按家族分派的那一处调用 smoke_suite（DEV-0005）
# ---------------------------------------------------------------------------
# 检查项沿用 Qwen 冒烟的编号 0–8，外加 9：
#   0 环境与模型事实（只记录；含 tile 策略、R1 prompt 的出处与 sha256、config 与运行时的 lm_head 共享状态）
#   1 两种顺序 × 思考开关各 build + generate 一次；完整提示（system 轮 + user 轮 + assistant 头）与 apply_chat_template
#     逐个相同，并与官方处理器的输出逐项比对（真模型：AutoProcessor；--tiny：reference_processor）
#   2 D1 仪器（图在前逐位相同 / 指令在前第 0 层相同之后不同）——直接用冒烟脚本的通用实现
#   3 add_grid_tokens 后限定生成——通用实现
#   4 查询 token：grid2d 被拒且不留下占位符；seq1d 能前向、形状对、位置 = 0..L-1；两组不同的 query_embeds 下查询块之前
#     各层逐位相同（因果注意力，精确为零的仪器），查询位置不同
#   5 一维位置：自构 position_ids = 0..L-1；传它与让模型自己算、以及适配器整条路径与 HF 原生路径（input_ids + pixel_values），
#     各层隐藏状态逐位相同（两张图、两种顺序）
#   6 vision_features 与第 0 层图像位置的隐藏状态逐位相同
#   7 思考：R1 system 轮在提示里；text / thinking_unclosed 与独立重写的剥离规则一致（开、关思考各一次）；记录属于哪种情形
#     （closed / unclosed / no_think）与思考段的实际格式（服务器上确认 R1 写法）
#   8 设备审计——通用实现
#   9 多 tile：另挂一个多 tile 策略的适配器（共享模型与 tokenizer，只读）：tile 数与处理器一致、grid_hw = (16·n, 16)、
#     与官方处理器逐项相同、适配器路径 = HF 原生路径、generate 可用；hidden_states / vision_features 拒绝多 tile

SMOKE_POLICY = {"crop_to_patches": False, "max_patches": 1}  # 主适配器：单 tile
SMOKE_MULTI_TILE_POLICY = {"crop_to_patches": True, "max_patches": 6}  # 检查 9（640×480 与合成图都切成 3×2 + 缩略图 = 7 个 tile）
SMOKE_EXTRA_CHECKS = ("9_multi_tile",)
SMOKE_RUN_ORDER = (
    "0_facts", "1_build_generate", "2_task_modulation", "5_position_ids", "6_vision_features", "7_thinking",
    "9_multi_tile", "3_grid_tokens", "4_query_tokens", "8_device_audit",
)
_SMOKE_GENERIC = ("2_task_modulation", "3_grid_tokens", "8_device_audit")


def smoke_suite(smoke: Any) -> tuple[Any, list[str], list[str]]:
    """供冒烟脚本的 family_suite 调用：返回 (setup, check_names, run_order)；setup(args) 返回 (adapter, ctx)，
    ctx["checks"] 是 InternVL 的检查表。

    smoke 是冒烟脚本模块本身：共用的准备（base_context）、通用检查（2、3、8）与常量（INSTR_A、DTYPES、N_QUERY…）
    从它取，src 不 import scripts。
    """
    checks = {name: smoke.CHECKS[name] for name in _SMOKE_GENERIC}
    checks.update({
        "0_facts": _smoke_facts,
        "1_build_generate": _smoke_build_generate,
        "4_query_tokens": _smoke_query_tokens,
        "5_position_ids": _smoke_position_ids,
        "6_vision_features": _smoke_vision_features,
        "7_thinking": _smoke_thinking,
        "9_multi_tile": _smoke_multi_tile,
    })
    names = list(smoke.CHECK_NAMES) + list(SMOKE_EXTRA_CHECKS)
    if set(names) != set(checks) or set(SMOKE_RUN_ORDER) != set(names):
        raise RuntimeError(f"InternVL 冒烟的检查项不齐：{sorted(set(names) ^ set(checks))}")
    return functools.partial(_smoke_setup, smoke, checks), names, list(SMOKE_RUN_ORDER)


def chat_messages(n_images: int, instruction: str, order: Order, *, enable_thinking: bool) -> list[dict[str, Any]]:
    """比对用的 HF 消息（写法同 model card 与 RUN-0002 的探针），与适配器内部的拼法各写各的。"""
    imgs = [{"type": "image"} for _ in range(n_images)]
    text = {"type": "text", "text": instruction}
    user = {"role": "user", "content": [text, *imgs] if order == "instruction_first" else [*imgs, text]}
    system = {"role": "system", "content": [{"type": "text", "text": R1_SYSTEM_PROMPT}]}
    return [system, user] if enable_thinking else [user]


def collapse_image_tokens(ids: Sequence[int], *, start_id: int, context_id: int, end_id: int) -> list[int]:
    """把每段 <img> <IMG_CONTEXT>×k </img> 收成一个 <IMG_CONTEXT>，便于与模板渲染（未展开）逐个比较。"""
    out: list[int] = []
    k = 0
    ids = list(ids)
    while k < len(ids):
        if ids[k] == start_id:
            j = k + 1
            while j < len(ids) and ids[j] == context_id:
                j += 1
            if j < len(ids) and ids[j] == end_id and j > k + 1:
                out.append(context_id)
                k = j + 1
                continue
        out.append(ids[k])
        k += 1
    return out


def _collapse(ad: InternVL35Adapter, ids: Sequence[int]) -> list[int]:
    return collapse_image_tokens(ids, start_id=ad.token_id(IMG_START), context_id=ad.token_id(IMG_CONTEXT),
                                 end_id=ad.token_id(IMG_END))


def compare_with_processor(ad: InternVL35Adapter, proc: Any, images: Sequence[Any], instruction: str, order: Order,
                           enable_thinking: bool, inp: ModelInputs) -> dict[str, Any]:
    """适配器送进模型的完整提示与像素，和处理器（官方类）对同一条消息的输出逐项比较。"""
    pil = [ad.to_rgb(im) for im in images]
    text = proc.apply_chat_template(chat_messages(len(pil), instruction, order, enable_thinking=enable_thinking),
                                    tokenize=False, add_generation_prompt=True)
    enc = proc(text=[text], images=pil or None, return_tensors="pt", **ad.processor_kwargs())
    ids = torch.tensor([ad.prompt_ids(inp, enable_thinking)], dtype=torch.long)
    want_keys = ["attention_mask", "input_ids"] + (["pixel_values"] if pil else [])
    res = {
        "keys": sorted(enc.keys()) == want_keys,
        "input_ids": torch.equal(enc["input_ids"].cpu(), ids),
        "attention_mask": torch.equal(enc["attention_mask"].cpu(), torch.ones_like(ids)),
    }
    if pil:
        res["pixel_values"] = torch.equal(enc["pixel_values"].to(ad.dtype).cpu(), inp.tensors["pixel_values"].cpu())
    return {"ok": all(res.values()), **res}


def _smoke_setup(smoke: Any, checks: dict[str, Any], args: Any) -> tuple[InternVL35Adapter, dict[str, Any]]:
    from src.adapters.factory import make_adapter  # 函数内 import：factory 按家族 import 本模块

    ctx = smoke.base_context(args)  # 固定种子、测试图与第二张图：三个家族共用同一份准备
    dtype = smoke.DTYPES[args.dtype]
    key = args.model or TINY_SOURCE_KEY
    ad = make_adapter(key, device=args.device, dtype=dtype, image_policy=TilePolicy(**SMOKE_POLICY), tiny=args.tiny)
    if args.tiny:
        ref = reference_processor((None, ad.tokenizer, ad.image_processor, ad.chat_template), key)
        ref_kind = "InternVLProcessor 官方类（开发机：绕过视频处理器的类型检查）+ GotOcr2ImageProcessorPil"
    else:
        ref = ad.hf_processor
        ref_kind = f"AutoProcessor（{type(ref).__name__} + {type(ref.image_processor).__name__}）"
    return ad, {**ctx, "smoke": smoke, "checks": checks, "ref_processor": ref, "ref_kind": ref_kind}


def _all_layers(ad: InternVL35Adapter) -> list[int]:
    return list(range(ad.num_layers() + 1))


def _smoke_facts(ad: InternVL35Adapter, ctx: dict[str, Any]) -> dict[str, Any]:
    import transformers

    args = ctx["args"]
    entry = registry.MODELS.get(ad.model_key) if ad.model_key in registry.MODELS else None
    cfg, tok = ad.model.config, ad.tokenizer
    return {
        "ok": True,
        "mode": "tiny" if args.tiny else "real",
        "model_key": ad.model_key,
        "repo": entry.repo if entry else None,
        "revision": entry.revision if entry else None,
        "device": str(ad.device),
        "dtype": str(ad.dtype),
        "tile_policy": ad.tile_policy,
        "multi_tile_policy_check9": SMOKE_MULTI_TILE_POLICY,
        "image_size_arg_ignored": {"value": list(args.image_size),
                                   "note": "--image-size 只对 Qwen 有效；InternVL 由图像处理器按 tile 策略缩放"},
        "tile_px": TILE_PX, "tokens_per_tile": TOKENS_PER_TILE, "tile_grid_hw": list(TILE_GRID_HW),
        "num_layers": ad.num_layers(),
        "hidden_size": ad.hidden_size(),
        "tied_word_embeddings_runtime": ad.tied_word_embeddings,
        "config_tie_word_embeddings": ad.config_tie_word_embeddings,
        "len_tokenizer": len(tok),
        "n_base_vocab": ad.n_base_vocab,
        "embed_rows": ad.model.get_input_embeddings().weight.shape[0],
        "lm_head_rows": ad.model.get_output_embeddings().weight.shape[0],
        "attn_implementation": {"top": cfg._attn_implementation, "text": cfg.text_config._attn_implementation,
                                "vision": cfg.vision_config._attn_implementation},
        "image_processor_class": type(ad.image_processor).__name__,
        "hf_processor_class": type(ad.hf_processor).__name__ if ad.hf_processor is not None else None,
        "reference_processor": ctx["ref_kind"],
        "system_prefix_n_tokens": {"thinking_off": len(ad.system_prefix_ids(False)),
                                   "thinking_on": len(ad.system_prefix_ids(True))},
        "assistant_header_ids": list(ad.assistant_header_ids(False)),
        "stop_token_ids": {IM_END: ad.token_id(IM_END)},
        "pad_token_id": {END_OF_TEXT: ad.token_id(END_OF_TEXT)},
        "think_token_ids": {THINK_START: ad.token_id(THINK_START), THINK_END: ad.token_id(THINK_END)},
        "r1_system_prompt_sha256": hashlib.sha256(R1_SYSTEM_PROMPT.encode()).hexdigest(),
        "r1_system_prompt_source": R1_SYSTEM_PROMPT_SOURCE,
        "image": ctx["image_info"],
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "transformers": transformers.__version__},
    }


def _smoke_build_generate(ad: InternVL35Adapter, ctx: dict[str, Any]) -> dict[str, Any]:
    smoke, args, img, tok = ctx["smoke"], ctx["args"], ctx["img"], ad.tokenizer
    cases = []
    for order in ORDERS:
        inp = ad.build_inputs([img], smoke.INSTR_A, order)
        for thinking in (False, True):
            res = ad.generate(inp, enable_thinking=thinking, max_new_tokens=args.max_new_tokens)
            full = ad.prompt_ids(inp, thinking)
            ref = tok.apply_chat_template(chat_messages(1, smoke.INSTR_A, order, enable_thinking=thinking),
                                          tokenize=True, add_generation_prompt=True)["input_ids"]
            template_equal = _collapse(ad, full) == list(ref)
            official = compare_with_processor(ad, ctx["ref_processor"], [img], smoke.INSTR_A, order, thinking, inp)
            raw_ok = res.raw_text == tok.decode(list(res.token_ids), skip_special_tokens=False)
            prefix = list(ad.system_prefix_ids(thinking))
            system_ok = (len(prefix) > 0) is thinking and full[: len(prefix)] == prefix
            ok = (template_equal and official["ok"] and raw_ok and system_ok and res.n_prompt_tokens == len(full)
                  and 1 <= res.n_generated_tokens <= args.max_new_tokens)
            cases.append({
                "order": order, "enable_thinking": thinking, "ok": ok,
                "template_equal": template_equal, "processor_compare": official, "raw_text_is_full_decode": raw_ok,
                "system_turn_ok": system_ok, "n_system_prefix_tokens": len(prefix),
                "instruction_span": [inp.instruction_span.start, inp.instruction_span.end],
                "image_spans": [[i.span.start, i.span.end] for i in inp.images],
                "image_grid_hw": [list(i.grid_hw) for i in inp.images],
                "n_prompt_tokens": res.n_prompt_tokens, "n_generated_tokens": res.n_generated_tokens,
                "n_visual_tokens": res.n_visual_tokens, "wall_time_s": round(res.wall_time_s, 4),
                "thinking_unclosed": res.thinking_unclosed, "text": res.text[:200], "raw_text": res.raw_text[:300],
            })
    return {"ok": all(c["ok"] for c in cases), "reference_processor": ctx["ref_kind"], "cases": cases}


def _smoke_query_tokens(ad: InternVL35Adapter, ctx: dict[str, Any]) -> dict[str, Any]:
    smoke, img = ctx["smoke"], ctx["img"]
    n, layers = smoke.N_QUERY, _all_layers(ad)
    n_layers, hidden = ad.num_layers(), ad.hidden_size()
    had_placeholder = ad.query_token_id is not None
    try:
        ad.inject_query_tokens(n, "grid2d")
        grid2d = {"rejected": False}
    except ValueError as e:
        grid2d = {"rejected": True, "error": str(e)[:200]}
    grid2d["left_no_placeholder"] = had_placeholder or ad.query_token_id is None
    ad.inject_query_tokens(n, "seq1d")
    try:
        inp = ad.build_inputs([img], smoke.INSTR_A, "instruction_first")
        span, ids = inp.query_span, inp.tensors["input_ids"][0]
        seq_len = ids.shape[0]
        pos_ok = bool(torch.equal(ad.position_ids(inp).cpu(), torch.arange(seq_len).unsqueeze(0)))
        placement_ok = span.end + 2 == seq_len and int(ids[span.end]) == ad.token_id(IM_END)
        g = torch.Generator().manual_seed(0)
        scale = float(ad.model.config.text_config.initializer_range)  # 与嵌入初始化同尺度
        qa = (torch.randn(n, hidden, generator=g) * scale).to(device=ad.device, dtype=ad.dtype)
        qb = (torch.randn(n, hidden, generator=g) * scale).to(device=ad.device, dtype=ad.dtype)
        with torch.no_grad():
            sa = ad.sequence_hidden_states(inp, layers, query_embeds=qa)
            sb = ad.sequence_hidden_states(inp, layers, query_embeds=qb)
            hs = ad.hidden_states(inp, [0, n_layers], query_embeds=qa)
        qs, qe = span.start, span.end
        prefix = [bool(torch.equal(sa[l][:qs], sb[l][:qs])) for l in layers]
        differ = [not torch.equal(sa[l][qs:qe], sb[l][qs:qe]) for l in layers]
        shape_ok = (tuple(hs.query[n_layers].shape) == (n, hidden)
                    and tuple(hs.image[n_layers].shape) == (TOKENS_PER_TILE, hidden))
        layer0 = bool(torch.equal(hs.query[0], qa))
    finally:
        ad.clear_query_tokens()
    ok = (grid2d["rejected"] and grid2d["left_no_placeholder"] and pos_ok and placement_ok and all(prefix)
          and all(differ) and shape_ok and layer0 and inp.query_position_mode == "seq1d")
    return {"ok": bool(ok), "n_query": n, "query_token_id": ad.query_token_id, "grid2d": grid2d,
            "query_span": [span.start, span.end], "position_ids_are_arange": pos_ok, "placement_ok": placement_ok,
            "shape_ok": shape_ok, "layer0_query_equals_query_embeds": layer0,
            "prefix_bitwise_equal_by_layer": prefix, "query_states_differ_by_layer": differ}


def _smoke_position_ids(ad: InternVL35Adapter, ctx: dict[str, Any]) -> dict[str, Any]:
    imgs = [ctx["img"], ctx["img2"]]
    layers = _all_layers(ad)
    out: dict[str, Any] = {}
    for order in ORDERS:
        inp = ad.build_inputs(imgs, ctx["smoke"].INSTR_A, order)
        t = inp.tensors
        pos = ad.position_ids(inp)
        seq_len = t["input_ids"].shape[1]
        arange = bool(torch.equal(pos.cpu(), torch.arange(seq_len).unsqueeze(0)))
        kw = dict(output_hidden_states=True, logits_to_keep=1)
        with torch.no_grad():
            ours = ad.sequence_hidden_states(inp, layers)
            auto = ad.model(**t, use_cache=False, **kw).hidden_states
            given = ad.model(**t, position_ids=pos, use_cache=False, **kw).hidden_states
            default = ad.model(**t, **kw).hidden_states  # HF 默认（use_cache 取 config），只记录
        given_eq = [bool(torch.equal(auto[l][0], given[l][0])) for l in layers]
        ours_eq = [bool(torch.equal(auto[l][0], ours[l])) for l in layers]
        default_eq = [bool(torch.equal(default[l][0], ours[l])) for l in layers]
        out[order] = {"ok": arange and all(given_eq) and all(ours_eq), "position_ids_are_arange": arange,
                      "own_position_ids_vs_model_computed": given_eq, "adapter_path_vs_model_path": ours_eq,
                      "info_adapter_path_vs_hf_default_use_cache": default_eq, "seq_len": int(seq_len)}
    out["ok"] = all(out[o]["ok"] for o in ORDERS)
    return out


def _smoke_vision_features(ad: InternVL35Adapter, ctx: dict[str, Any]) -> dict[str, Any]:
    imgs = [ctx["img"], ctx["img2"]]
    out: dict[str, Any] = {}
    for order in ORDERS:
        inp = ad.build_inputs(imgs, ctx["smoke"].INSTR_A, order)
        with torch.no_grad():
            vf = ad.vision_features(inp)
            h0 = ad.hidden_states(inp, [0]).image[0]
        shape_ok = tuple(vf.shape) == (2 * TOKENS_PER_TILE, ad.hidden_size())
        out[order] = {"ok": bool(shape_ok and torch.equal(vf, h0)), "shape": list(vf.shape),
                      "equal": bool(torch.equal(vf, h0))}
    out["ok"] = all(out[o]["ok"] for o in ORDERS)
    return out


def _smoke_thinking(ad: InternVL35Adapter, ctx: dict[str, Any]) -> dict[str, Any]:
    smoke, args, img, tok = ctx["smoke"], ctx["args"], ctx["img"], ad.tokenizer
    inp = ad.build_inputs([img], smoke.INSTR_THINK, "image_first")
    res = ad.generate(inp, enable_thinking=True, max_new_tokens=args.think_max_new_tokens)
    ids = list(res.token_ids)
    start, end, stop = ad.token_id(THINK_START), ad.token_id(THINK_END), ad.token_id(IM_END)
    raw_ok = res.raw_text == tok.decode(ids, skip_special_tokens=False)
    prompt = ad.prompt_ids(inp, True)
    system_text = tok.decode(list(ad.system_prefix_ids(True)), skip_special_tokens=False)
    system_ok = (res.n_prompt_tokens == len(prompt) and R1_SYSTEM_PROMPT in system_text
                 and prompt[: len(ad.system_prefix_ids(True))] == list(ad.system_prefix_ids(True)))

    def expected(out: list[int]) -> tuple[str, bool, str]:
        """独立重写的剥离规则（与 split_answer 的 docstring 同义）：(text, unclosed, 属于哪种情形)。"""
        lo = max((k for k, v in enumerate(out) if v == start), default=None)
        lc = max((k for k, v in enumerate(out) if v == end), default=None)
        if lo is not None and (lc is None or lo > lc):
            return "", True, "unclosed"
        ans = out[lc + 1 :] if lc is not None else list(out)
        if ans and ans[-1] == stop:
            ans = ans[:-1]
        return tok.decode(ans, skip_special_tokens=False).strip("\n"), False, "closed" if lc is not None else "no_think"

    want_text, want_unclosed, case = expected(ids)
    fmt: dict[str, Any] = {
        "first_token_is_think_start": bool(ids) and ids[0] == start,
        "n_think_start": ids.count(start), "n_think_end": ids.count(end),
        "ended_with_stop_token": bool(ids) and ids[-1] == stop,
    }
    if case == "closed":  # 服务器上用来确认 R1 写法：</think> 之后隔了什么
        k = len(ids) - 1 - ids[::-1].index(end)
        after = tok.decode([t for t in ids[k + 1 :] if t != stop], skip_special_tokens=False)
        fmt["separator_after_think_end"] = after[: len(after) - len(after.lstrip("\n"))]
        fmt["answer_starts_with_space"] = after.lstrip("\n")[:1] == " "
    ok = raw_ok and res.text == want_text and res.thinking_unclosed == want_unclosed and THINK_END not in res.text
    off = ad.generate(inp, enable_thinking=False, max_new_tokens=args.max_new_tokens)
    off_text, off_unclosed, off_case = expected(list(off.token_ids))
    off_ok = (off.text == off_text and off.thinking_unclosed == off_unclosed and THINK_END not in off.text
              and off.n_prompt_tokens == len(ad.prompt_ids(inp, False)) == len(prompt) - len(ad.system_prefix_ids(True)))
    return {"ok": bool(ok and off_ok and system_ok), "case": case, "closed": case == "closed",
            "closed_case_exercised": case == "closed", "system_turn_in_prompt": system_ok, "format_facts": fmt,
            "n_generated_tokens": res.n_generated_tokens, "max_new_tokens": args.think_max_new_tokens,
            "thinking_unclosed": res.thinking_unclosed, "text": res.text[:300], "raw_text": res.raw_text[:1500],
            "thinking_off_control": {"ok": off_ok, "case": off_case, "text": off.text[:200],
                                     "thinking_unclosed": off.thinking_unclosed}}


def _smoke_multi_tile(ad: InternVL35Adapter, ctx: dict[str, Any]) -> dict[str, Any]:
    smoke, args, img = ctx["smoke"], ctx["args"], ctx["img"]
    mt = InternVL35Adapter(device=ad.device, **SMOKE_MULTI_TILE_POLICY)
    mt.attach(ad.model, tokenizer=ad.tokenizer, image_processor=ad.image_processor, chat_template=ad.chat_template,
              model_key=ad.model_key)
    inp = mt.build_inputs([img], smoke.INSTR_A, "instruction_first")
    info = inp.images[0]
    n = tiles_of(info)
    w, h = mt.to_rgb(img).size
    expect_n = int(mt.image_processor.get_number_of_image_patches(h, w, mt.processor_kwargs()))
    layout_ok = (n > 1 and n == expect_n and info.grid_hw == (TILE_GRID_HW[0] * n, TILE_GRID_HW[1])
                 and len(info.span) == TOKENS_PER_TILE * n and inp.tensors["pixel_values"].shape[0] == n)
    compare = {str(flag): compare_with_processor(mt, ctx["ref_processor"], [img], smoke.INSTR_A, "instruction_first",
                                                 flag, inp) for flag in (False, True)}
    layers = _all_layers(mt)
    with torch.no_grad():
        ours = mt.sequence_hidden_states(inp, layers)
        auto = mt.model(**inp.tensors, use_cache=False, output_hidden_states=True, logits_to_keep=1).hidden_states
    path_eq = [bool(torch.equal(auto[l][0], ours[l])) for l in layers]
    res = mt.generate(inp, enable_thinking=False, max_new_tokens=args.max_new_tokens)
    gen_ok = 1 <= res.n_generated_tokens <= args.max_new_tokens and res.n_visual_tokens == TOKENS_PER_TILE * n
    refused = {}
    for name, call in (("hidden_states", lambda: mt.hidden_states(inp, [0])),
                       ("vision_features", lambda: mt.vision_features(inp))):
        try:
            call()
            refused[name] = False
        except ValueError:
            refused[name] = True
    ok = layout_ok and all(c["ok"] for c in compare.values()) and all(path_eq) and gen_ok and all(refused.values())
    return {"ok": bool(ok), "policy": mt.tile_policy, "image_size_wh": [w, h], "n_tiles": n,
            "n_tiles_from_image_processor": expect_n, "grid_hw": list(info.grid_hw), "span_len": len(info.span),
            "layout_ok": layout_ok, "processor_compare_by_thinking": compare,
            "adapter_path_vs_model_path": path_eq, "generate_ok": gen_ok, "n_generated_tokens": res.n_generated_tokens,
            "probe_methods_refuse_multi_tile": refused,
            "note": "带查询 token 的 build_inputs 拒绝多 tile 由单元测试覆盖（这里不加 token，保持共享的模型与 tokenizer 不变）"}


__all__ = [
    "AdapterDeviceError",
    "BoxParse",
    "BoxParseFailure",
    "InternVL35Adapter",
    "OneBoxParse",
    "QUERY_PLACEHOLDER",
    "R1_SYSTEM_PROMPT",
    "R1_SYSTEM_PROMPT_SOURCE",
    "TILE_GRID_HW",
    "TILE_PX",
    "TOKENS_PER_TILE",
    "build_tiny_model",
    "load_pinned_components",
    "make_tiny_adapter",
    "parse_boxes",
    "parse_one_box",
    "reference_processor",
    "smoke_suite",
    "tiles_of",
    "tiny_config",
]
