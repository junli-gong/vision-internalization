"""模型适配器接口（简报 §8.1）的抽象基类与数据结构。

这里只有签名、类型注解和 docstring，不实现任何模型，不 import transformers。
各家族的实现放在 src/adapters/<family>.py，等 RUN-0001 报回真实 schema 后再写（DEV-0003 起）。

约定：
- 层号只能作为参数传入；本模块不出现任何具体层号（简报 §4 第 6 条）。
- **层号约定（D10，各家族必须一致，否则 E2 的逐层曲线会错位）**：
  - 层号 0 = 视觉特征并入序列之后的嵌入输出，即第一个 decoder 层的输入；
  - 层号 i（1 ≤ i ≤ num_layers()）= 第 i 个 decoder 层的输出；
  - 其中最后一个层号 num_layers() 按 HF `output_hidden_states` 元组的惯例，取经过最终归一化（final norm）
    之后的值。三个家族的 HF 实现都是这个惯例，所以层号与 HF 元组下标一一对应；
  - 有效范围 [0, num_layers()]，越界报错。
- `generate` 的 `enable_thinking` 与 `max_new_tokens` 是必填关键字参数，没有默认值（D1：每次调用都
  显式设置）。子类若给它们加默认值，定义类时就报 TypeError（见 `ModelAdapter.__init_subclass__`）。
- token 区间一律是半开区间 [start, end)，下标相对于单条输入序列。
"""

from __future__ import annotations

import abc
import inspect
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

from src.constants import MAX_NEW_GRID_TOKENS

Order = Literal["instruction_first", "image_first"]
PositionMode = Literal["grid2d", "seq1d"]

ORDERS: tuple[str, ...] = ("instruction_first", "image_first")
POSITION_MODES: tuple[str, ...] = ("grid2d", "seq1d")


@dataclass(frozen=True)
class Span:
    """token 半开区间 [start, end)。"""

    start: int
    end: int

    def __post_init__(self) -> None:
        if not (0 <= self.start <= self.end):
            raise ValueError(f"非法区间 [{self.start}, {self.end})")

    def __len__(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class ImageTokenInfo:
    """一张图在序列里的视觉 token 区间，以及这些 token 排成的 (h, w) 网格。

    grid_hw 指送进 LLM 的视觉 token 网格（即合并/下采样之后的网格），不是 patch 网格。
    某些家族在行间插分隔 token，这时 len(span) 可能不等于 h*w；本基类不强制两者相等，
    由各家族适配器在 docstring 里写明。
    """

    span: Span
    grid_hw: tuple[int, int]


@dataclass
class ModelInputs:
    """送进模型的一条输入：张量加元数据。

    `tensors` 是处理器产出的张量字典（input_ids、attention_mask、pixel_values 等），键名由适配器决定。
    元数据描述单条序列（batch 为 1）；批处理的元数据形式等 DEV-0003 按真实 schema 定。
    """

    tensors: Mapping[str, Any]
    order: Order
    images: tuple[ImageTokenInfo, ...]
    instruction_span: Span
    query_span: Span | None = None
    query_position_mode: PositionMode | None = None

    def __post_init__(self) -> None:
        if self.order not in ORDERS:
            raise ValueError(f"order 必须是 {ORDERS} 之一，收到 {self.order!r}")
        if (self.query_span is None) != (self.query_position_mode is None):
            raise ValueError("query_span 与 query_position_mode 必须同时给出或同时为空")
        if self.query_position_mode is not None and self.query_position_mode not in POSITION_MODES:
            raise ValueError(f"position_mode 必须是 {POSITION_MODES} 之一，收到 {self.query_position_mode!r}")


@dataclass(frozen=True)
class GenerationResult:
    """一次 generate 调用的结果与计算量（记账字段见 src/accounting.py）。

    text：剥离思考段之后的最终回答（各家族的剥离规则写在适配器里），解析器只看它（D10）。
    raw_text：完整解码结果，含思考段与特殊 token。适配器必须填写；默认 None 只为兼容旧的构造方式。
    thinking_unclosed：开启了思考、却没有生成思考段的结束标记（例如 max_new_tokens 用完）。此时 text 为空串。
    """

    text: str
    token_ids: tuple[int, ...]
    n_generated_tokens: int
    n_prompt_tokens: int
    n_visual_tokens: int
    wall_time_s: float
    raw_text: str | None = None
    thinking_unclosed: bool = False

    def __post_init__(self) -> None:
        if self.n_generated_tokens != len(self.token_ids):
            raise ValueError(
                f"n_generated_tokens={self.n_generated_tokens} 与 len(token_ids)={len(self.token_ids)} 不一致"
            )
        for name in ("n_generated_tokens", "n_prompt_tokens", "n_visual_tokens"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} 不能为负")
        if self.wall_time_s < 0:
            raise ValueError("wall_time_s 不能为负")
        if self.thinking_unclosed and self.text != "":
            raise ValueError("思考段未闭合时 text 必须为空串")


@dataclass(frozen=True)
class HiddenStates:
    """按层给出的隐藏状态。

    image[layer] 形状 [n_image_tokens, hidden_size]，按 ModelInputs.images 的顺序拼接所有图的视觉 token。
    query[layer] 形状 [n_query_tokens, hidden_size]；输入里没有查询 token 时 query 为 None。
    layers 是调用方传入的层号，原样保留顺序。
    """

    layers: tuple[int, ...]
    image: Mapping[int, Any]
    query: Mapping[int, Any] | None = None

    def __post_init__(self) -> None:
        if set(self.image) != set(self.layers):
            raise ValueError(f"image 的层号 {sorted(self.image)} 与 layers {list(self.layers)} 不一致")
        if self.query is not None and set(self.query) != set(self.layers):
            raise ValueError(f"query 的层号 {sorted(self.query)} 与 layers {list(self.layers)} 不一致")


def check_grid_token_names(names: Sequence[str], n_existing: int = 0) -> None:
    """add_grid_tokens 实现里调用：名字不重复、非空，总数不超过 MAX_NEW_GRID_TOKENS（简报 §4 第 4 条）。

    预算跨调用累计（D10）：n_existing 是此前已经加过的 grid token 个数，由适配器自己记；
    判据是 n_existing + len(names) ≤ MAX_NEW_GRID_TOKENS。
    """
    if isinstance(names, str):
        raise TypeError("names 必须是字符串的序列，不能是单个字符串")
    if isinstance(n_existing, bool) or not isinstance(n_existing, int) or n_existing < 0:
        raise ValueError(f"n_existing 必须是非负整数，收到 {n_existing!r}")
    if len(names) == 0:
        raise ValueError("names 为空")
    if any(not isinstance(n, str) or n == "" for n in names):
        raise ValueError("grid token 名字必须是非空字符串")
    if len(set(names)) != len(names):
        raise ValueError("grid token 名字有重复")
    if n_existing + len(names) > MAX_NEW_GRID_TOKENS:
        raise ValueError(
            f"累计新增 {n_existing} + {len(names)} = {n_existing + len(names)} 个 token，超过上限 {MAX_NEW_GRID_TOKENS}"
        )


# 子类必须保持为"无默认值的关键字参数"的参数：方法名 -> 参数名
_REQUIRED_KWONLY: dict[str, tuple[str, ...]] = {
    "generate": ("enable_thinking", "max_new_tokens"),
}
# 子类必须保持为"无默认值"的参数（层号只能由调用方给出）
_REQUIRED_NO_DEFAULT: dict[str, tuple[str, ...]] = {
    "hidden_states": ("layers",),
}


class ModelAdapter(abc.ABC):
    """每个模型家族实现一个。模型相关逻辑只能出现在子类里（简报 §4 第 6 条）。"""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        for meth, params in _REQUIRED_KWONLY.items():
            if meth in cls.__dict__:
                sig = inspect.signature(cls.__dict__[meth])
                for p in params:
                    par = sig.parameters.get(p)
                    if par is None or par.kind is not inspect.Parameter.KEYWORD_ONLY or par.default is not par.empty:
                        raise TypeError(f"{cls.__name__}.{meth} 的 {p} 必须是无默认值的关键字参数（D1）")
        for meth, params in _REQUIRED_NO_DEFAULT.items():
            if meth in cls.__dict__:
                sig = inspect.signature(cls.__dict__[meth])
                for p in params:
                    par = sig.parameters.get(p)
                    if par is None or par.default is not par.empty:
                        raise TypeError(f"{cls.__name__}.{meth} 的 {p} 必须由调用方显式传入，不能有默认值")

    @abc.abstractmethod
    def load(self, model_id: str, dtype: Any) -> None:
        """加载模型与处理器。dtype 由调用方给出（如 torch.bfloat16），适配器不自行选择。"""

    @abc.abstractmethod
    def build_inputs(self, images: Sequence[Any], instruction: str, order: Order) -> ModelInputs:
        """把若干张图（叠加图与原图一起送入时为多张）和一条指令组装成模型输入。

        order="instruction_first"：指令在图像之前，图像 token 能看到指令（简报 §12 因果注意力）。
        order="image_first"：图像在指令之前。
        若已调用 inject_query_tokens，查询 token 追加在指令与图像之后，并填好 query_span。
        分辨率与视觉 token 数由处理器参数固定，所有对比保持一致（简报 §12）。
        """

    @abc.abstractmethod
    def generate(
        self,
        inputs: ModelInputs,
        *,
        enable_thinking: bool,
        allowed_token_ids: Sequence[int] | None = None,
        max_new_tokens: int,
        **gen_kwargs: Any,
    ) -> GenerationResult:
        """生成。enable_thinking、max_new_tokens 必须每次显式给出（D1）。

        allowed_token_ids 非空时，每一步只在这组 token 上计算 logits 并采样（简报 §4 第 4 条）；
        训练与推理的限制实现必须一致（简报 §12）。
        """

    @abc.abstractmethod
    def hidden_states(
        self, inputs: ModelInputs, layers: Sequence[int], *, query_embeds: Any | None = None
    ) -> HiddenStates:
        """返回指定层在图像 token 位置和查询 token 位置的隐藏状态。

        layers 由调用方给出（来自探针结果或可学习逐层加权的配置），适配器不设默认层。
        层号按本模块开头的层号约定：层号 0 是视觉特征并入之后的嵌入输出，层号 i 是第 i 个 decoder 层的
        输出（最后一个取最终归一化之后的值），有效范围 [0, num_layers()]，越界报错。

        query_embeds（D10 的接口扩展，DEV-0003）：形状 [n_query, hidden_size]。非 None 时，前向前把查询位置的
        输入嵌入替换成它，E3a 的可学习查询 token 从这里进；inputs 里必须有查询 token，个数一致。
        """

    def vision_features(self, inputs: ModelInputs) -> Any:
        """进入 LLM 之前、经过投影/合并之后的视觉 token（D10，DEV-0003）。

        形状 [n_image_tokens, hidden_size]，多图按 inputs.images 的顺序拼接。E2 的参考线、E3a 的 C2 对照用它。
        各家族在适配器里写明它与 hidden_states(inputs, [0]) 在图像位置上的关系（Qwen3.5 要求逐位相同）。

        注意：任务书要求这是抽象方法。tests/test_adapters_base.py 里的测试替身没有实现它，而那个文件不在
        DEV-0003 的改动范围内，所以这里暂用"默认抛 NotImplementedError"的普通方法，待管理员裁定。
        """
        raise NotImplementedError(f"{type(self).__name__} 没有实现 vision_features")

    @abc.abstractmethod
    def inject_query_tokens(self, n: int, position_mode: PositionMode) -> None:
        """登记 n 个可学习查询 token，之后 build_inputs 把它们追加在指令与图像之后。

        position_mode="grid2d"：对 M-RoPE 模型，查询 token 取二维网格位置，放在文本之后的新时间索引上；
        position_mode="seq1d"：一维顺序位置。两种都要能切换（简报 §12）。
        """

    @abc.abstractmethod
    def add_grid_tokens(self, names: Sequence[str]) -> dict[str, int]:
        """加入像素 token 通道用的新 token，返回 名字 -> token id。

        优先复用预留 token，否则扩展嵌入（简报 §8.1）。实现须调用 check_grid_token_names，
        并确认分词器把每个新 token 当作单个 token（简报 §12）。预算按累计计：适配器自己记已加的总数，
        作为 n_existing 传给 check_grid_token_names（D10）。
        """

    @abc.abstractmethod
    def num_layers(self) -> int:
        """LLM 的 Transformer 层数。"""

    @abc.abstractmethod
    def hidden_size(self) -> int:
        """LLM 隐藏维度。"""
