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
  - **可选的层键 PRE_NORM_LAST = "pre_norm_last"（DEV-0024，E2 审查 I1）**：最后一个 decoder 层的原始输出，即最终归一化
    之前的值；norm(它) 就是层号 num_layers() 的读数。只有 hidden_states 接受它（与整数层号混在同一个 layers 序列里，
    HiddenStates.image / query 用同一个键返回）；其余接受层号的接口（sequence_hidden_states、training_forward 的
    HiddenRequest）照旧只收整数。解析规则见 resolve_layer_keys。
- `generate` 与 `generate_batch` 的 `enable_thinking` 与 `max_new_tokens` 是必填关键字参数，没有默认值（D1：每次调用都
  显式设置）。子类若给它们加默认值，定义类时就报 TypeError（见 `ModelAdapter.__init_subclass__`）。
- token 区间一律是半开区间 [start, end)，下标相对于单条输入序列。
- 计时口径（D21，DEV-0011）：GenerationResult.wall_time_s 只含 model.generate 本身（CUDA 上前后各同步一次），不含 build_inputs；
  进程里第一次生成含 JIT、内核编译与库初始化，记账之前先调用 warmup()，它的耗时不进记账。
- 受限选项读数（DEV-0016，D36-4、D36-6）：option_logits 只做一次前向，读第一个生成位置的 logits，每个选项取它所有单 token
  写法（OPTION_SPELLING_TEMPLATES）里 logit 的最大值，在选项集合上做 softmax。结果是 OptionScores，规则写在它的说明里。
- E3 的训练侧接口（DEV-0020，D21-2）：training_forward、restricted_logits、chunked_lm_loss、generate_restricted、train_mode /
  eval_mode、set_gradient_checkpointing、class_name_spans（第一部分，实现在 src/adapters/training.py）；token_table、restore_tokens、
  lora_target_modules、inject_lora、merge_lora、save_trained、load_trained、trainable_token_rows（第二部分，实现在
  src/adapters/checkpoint.py）。这里只写接口说明；基类的实现一律报 NotImplementedError（不是抽象方法：只做推理的替身不必实现
  它们），三个家族都实现。
- DEV-0027 的接口（同样不是抽象方法，三个家族都实现）：generate_batch_budgeted（预算强制收尾的思考生成，公共实现在
  src/adapters/budget.py）、pre_projection_features（E2 的 pre_proj 参考线）、format_boxes（parse_boxes 的逆）、
  end_of_turn_token_id / stop_token_ids（公开的停止 token，替代私有的 _stop_ids()）。
- 作答开头预填（DEV-0033，D66-2 的格式强制诊断）：generate / generate_batch 的可选关键字参数 answer_prefix（默认 None = 不预填，
  送进模型的输入与以前逐 token 相同）。给了时只用于关思考：预填文字按本家族的分词器编码，接在每行提示（助手头；Qwen 关思考时
  助手头以空思考块结尾，所以在思考段之后）的末尾，作为输入的一部分，不计入生成；结果的 text、raw_text 按"预填的 token 是模型
  已经生成的前几个 token"来解码（split_answer 作用在 预填 + 生成 上），token_ids 只含生成的 token，GenerationResult.answer_prefix_ids
  记下预填的 token。公共部分在 src/adapters/batching.py（encode_answer_prefix、with_answer_prefix、make_results 的 prefix_ids）。
- 多轮对话（DEV-0049，W1；接口约定 docs/w1_plan.md §1）：chat_turn_batch / chat_turn。消息、工具说明、结果的数据结构与工具调用的解析在
  src/adapters/chat.py。不是抽象方法：基类报 NotImplementedError，目前只有 Qwen3.5 实现。tools、thinking_budget、answer_budget
  是无默认值的关键字参数（同 D1 的规则，__init_subclass__ 核对）。
"""

from __future__ import annotations

import abc
import inspect
import math
import time
from dataclasses import dataclass
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable, Iterator, Literal, Mapping, NamedTuple, Sequence

from src.constants import MAX_NEW_GRID_TOKENS
from src.parsing import BoxParse, OneBoxParse, one_box

if TYPE_CHECKING:  # chat.py 引用本模块的数据结构，运行时不反向 import
    from src.adapters.chat import ChatTurnResult, ToolSpec

Order = Literal["instruction_first", "image_first"]
PositionMode = Literal["grid2d", "seq1d"]

ORDERS: tuple[str, ...] = ("instruction_first", "image_first")
POSITION_MODES: tuple[str, ...] = ("grid2d", "seq1d")

# 预算强制收尾的思考生成（DEV-0027）：思考段怎样结束（GenerationResult.thinking_end；规则见 src/adapters/budget.py）
THINKING_ENDS: tuple[str, ...] = ("natural", "forced", "not_opened", "unclosed")

# hidden_states 的可选层键（层号约定见模块说明）：最后一个 decoder 层的原始输出，最终归一化之前
PRE_NORM_LAST = "pre_norm_last"
LayerKey = int | str  # 整数层号，或 PRE_NORM_LAST


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
    n_pixels（DEV-0011，D28-3"报告里写明各家族实际看到的像素数"）：这张图送进视觉编码器的像素数，即缩放 / 切 tile 之后
    的像素总数（含 InternVL 的缩略图 tile）；各家族的算法写在适配器里。None 只为兼容旧的构造方式，三个适配器一律填写。
    """

    span: Span
    grid_hw: tuple[int, int]
    n_pixels: int | None = None

    def __post_init__(self) -> None:
        if self.n_pixels is not None and (isinstance(self.n_pixels, bool) or not isinstance(self.n_pixels, int)
                                          or self.n_pixels <= 0):
            raise ValueError(f"n_pixels 必须是正整数或 None，收到 {self.n_pixels!r}")


@dataclass
class ModelInputs:
    """送进模型的一条输入：张量加元数据。

    `tensors` 是处理器产出的张量字典（input_ids、attention_mask、pixel_values 等），键名由适配器决定。
    元数据描述单条序列（batch 为 1）。批处理不改这个结构：generate_batch 接收若干条 ModelInputs，在内部左填充拼批（DEV-0011）。
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
    thinking_unclosed：输出里的思考段开了却没有闭合（例如 max_new_tokens 用完）。此时 text 为空串。
        各家族的判定规则写在适配器里（D19）：Qwen3.5 的思考段开头由提示预填，开启思考时输出里没有 `</think>` 即为未闭合；
        Gemma 4 与 InternVL3.5 看输出本身，最后一个思考段开头之后没有结束标记才算未闭合，没写思考段就直接作答取全文。

    DEV-0011 新增的字段（都有默认值，旧的构造方式照旧可用；三个适配器的 generate 与 generate_batch 一律填写）：
    stopped：停止原因（D21、D22-3）。True = 最后一个 token 是本家族的停止 token；False = 没有生成停止 token，被
        max_new_tokens 截断（allowed_token_ids 把停止 token 排除在外时也总是 False）。None = 未记录（旧的构造方式与测试替身）。
        src.parsing.parse_generation 把 stopped is False 判为 truncated（E1 的协议：截断一律判失败）。
    n_image_pixels：模型实际看到的像素数，即各图 ImageTokenInfo.n_pixels 之和（没有图为 0；D28-3）。None = 未记录。
    batch_size：这一行所在批的行数（generate 为 1）。
    batch_wall_time_s：整批 model.generate 的墙钟。wall_time_s 是按行均摊的份额 = batch_wall_time_s / batch_size
        （等分；理由：左填充下每一行在整批的每个解码步里都占一个槽位，直到最长的一行结束）。generate 时两者相等。

    DEV-0027 新增的字段：预算强制收尾的思考生成（generate_batch_budgeted，D59-1、D59-5；规则见 src/adapters/budget.py）。
    thinking_end 为 None 表示不是这种生成，其余新字段必须保持默认值；不为 None 时：
    thinking_end：思考段怎样结束，取 THINKING_ENDS 之一——natural（模型自己写出思考结束标记）、forced（思考写满预算，适配器补上
        结束标记与固定分隔）、not_opened（思考段开头在输出里的家族（Gemma、InternVL），第一个 token 不是思考段开头：直接作答，
        全部 token 都是作答段）、unclosed（思考段里写出停止 token，没有收尾就结束了；没有作答段）。
    forced_close：thinking_end == "forced"（任务书固定的字段名）。natural_close 是只读属性。
    token_ids / n_generated_tokens：只含模型生成的 token（思考段 + 作答段），补的 token 不计入（单独记）。
    n_thinking_tokens：思考段里模型生成的 token 数（含思考段开头与结束标记，凡是模型写出的）；n_answer_tokens：作答段的 token 数
        （含末尾的停止 token）。二者之和 = n_generated_tokens。
    n_inserted_tokens、inserted_token_ids：补的 token（只有 forced 才有）；inserted_at：它们插在 token_ids 的哪个下标之前
        （= n_thinking_tokens）。完整序列 = token_ids[:inserted_at] + inserted_token_ids + token_ids[inserted_at:]。
    raw_text：完整序列的解码（含补的 token）；inserted_span：补的那段文字在 raw_text 里的字符区间 [start, end)，用来辨认补上的标记。
        forced 时 raw_text = 思考段的解码 + 补的 token 的解码 + 作答段的解码（分段解码，所以区间是精确的）。
    text：完整序列按本家族的剥离规则（split_answer）得到的回答，与 generate 的口径相同；answer_text：作答段（收尾之后，forced 时
        在补的 token 之后）的解码，去掉末尾的停止 token 与首尾换行（与 text 同样的清理）。unclosed 时为空串。

    DEV-0033 新增的字段（作答开头预填，generate / generate_batch 的 answer_prefix）：
    answer_prefix_ids：预填的 token（接在提示末尾，是输入的一部分，计入 n_prompt_tokens，不计入 token_ids / n_generated_tokens）。
        非空时 text、raw_text 是 预填 + 生成 一起按 split_answer 解码的结果（等价于预填的 token 由模型自己写出）。默认 () = 没有预填。
    """

    text: str
    token_ids: tuple[int, ...]
    n_generated_tokens: int
    n_prompt_tokens: int
    n_visual_tokens: int
    wall_time_s: float
    raw_text: str | None = None
    thinking_unclosed: bool = False
    stopped: bool | None = None
    n_image_pixels: int | None = None
    batch_size: int = 1
    batch_wall_time_s: float | None = None
    thinking_end: str | None = None
    forced_close: bool = False
    n_thinking_tokens: int | None = None
    n_answer_tokens: int | None = None
    n_inserted_tokens: int = 0
    inserted_token_ids: tuple[int, ...] = ()
    inserted_at: int | None = None
    inserted_span: tuple[int, int] | None = None
    answer_text: str | None = None
    answer_prefix_ids: tuple[int, ...] = ()

    @property
    def natural_close(self) -> bool:
        """模型自己写出了思考结束标记（预算强制收尾的生成；其他生成恒为 False）。"""
        return self.thinking_end == "natural"

    def _check_budget_fields(self) -> None:
        defaults = {"forced_close": False, "n_thinking_tokens": None, "n_answer_tokens": None, "n_inserted_tokens": 0,
                    "inserted_token_ids": (), "inserted_at": None, "inserted_span": None, "answer_text": None}
        if self.thinking_end is None:
            bad = [k for k, v in defaults.items() if getattr(self, k) != v]
            if bad:
                raise ValueError(f"thinking_end 为 None（不是预算强制收尾的生成），这些字段却不是默认值：{bad}")
            return
        if self.thinking_end not in THINKING_ENDS:
            raise ValueError(f"thinking_end 必须是 {THINKING_ENDS} 之一，收到 {self.thinking_end!r}")
        if self.forced_close != (self.thinking_end == "forced"):
            raise ValueError(f"forced_close={self.forced_close} 与 thinking_end={self.thinking_end!r} 不符")
        for name in ("n_thinking_tokens", "n_answer_tokens", "n_inserted_tokens"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int) or v < 0:
                raise ValueError(f"{name} 必须是非负整数，收到 {v!r}")
        if self.n_thinking_tokens + self.n_answer_tokens != self.n_generated_tokens:
            raise ValueError(f"n_thinking_tokens + n_answer_tokens = {self.n_thinking_tokens} + {self.n_answer_tokens} "
                             f"≠ n_generated_tokens = {self.n_generated_tokens}")
        if self.n_inserted_tokens != len(self.inserted_token_ids):
            raise ValueError(f"n_inserted_tokens={self.n_inserted_tokens} 与补的 token {self.inserted_token_ids} 不符")
        if not isinstance(self.answer_text, str):
            raise TypeError(f"answer_text 必须是 str，收到 {self.answer_text!r}")
        if self.forced_close:
            if not self.inserted_token_ids or self.inserted_at != self.n_thinking_tokens or self.inserted_span is None:
                raise ValueError("forced：必须有补的 token，inserted_at = n_thinking_tokens，并给出 inserted_span")
            s, e = self.inserted_span
            if self.raw_text is None or not 0 <= s < e <= len(self.raw_text):
                raise ValueError(f"inserted_span={self.inserted_span} 不在 raw_text 之内")
        elif self.inserted_token_ids or self.inserted_at is not None or self.inserted_span is not None:
            raise ValueError(f"thinking_end={self.thinking_end!r} 不应有补的 token")
        if self.thinking_end == "not_opened" and self.n_thinking_tokens != 0:
            raise ValueError("not_opened：没有思考段，n_thinking_tokens 必须为 0")
        if self.thinking_end == "unclosed" and (self.n_answer_tokens != 0 or self.answer_text != ""):
            raise ValueError("unclosed：没有作答段，n_answer_tokens 必须为 0、answer_text 必须为空串")
        if self.thinking_end in ("natural", "forced") and self.n_thinking_tokens < 1:
            raise ValueError(f"{self.thinking_end}：思考段至少有 1 个 token")

    def __post_init__(self) -> None:
        self._check_budget_fields()
        if not isinstance(self.answer_prefix_ids, tuple) or any(
                isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in self.answer_prefix_ids):
            raise ValueError(f"answer_prefix_ids 必须是非负整数的元组，收到 {self.answer_prefix_ids!r}")
        if self.answer_prefix_ids and self.thinking_end is not None:
            raise ValueError("预填（answer_prefix_ids）只用于关思考的 generate / generate_batch，不与预算强制收尾同用")
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
        if self.stopped is not None:
            if not isinstance(self.stopped, bool):
                raise TypeError(f"stopped 必须是 bool 或 None，收到 {self.stopped!r}")
            if self.stopped and not self.token_ids:
                raise ValueError("stopped=True 却一个 token 也没有生成")
        if self.n_image_pixels is not None:
            if isinstance(self.n_image_pixels, bool) or not isinstance(self.n_image_pixels, int) or self.n_image_pixels < 0:
                raise ValueError(f"n_image_pixels 必须是非负整数或 None，收到 {self.n_image_pixels!r}")
            if (self.n_image_pixels == 0) != (self.n_visual_tokens == 0):
                raise ValueError(f"n_image_pixels={self.n_image_pixels} 与 n_visual_tokens={self.n_visual_tokens} 不相容"
                                 "（有视觉 token 就必须有像素，反之亦然）")
        if isinstance(self.batch_size, bool) or not isinstance(self.batch_size, int) or self.batch_size < 1:
            raise ValueError(f"batch_size 必须是正整数，收到 {self.batch_size!r}")
        if self.batch_wall_time_s is not None:
            if self.batch_wall_time_s < 0:
                raise ValueError("batch_wall_time_s 不能为负")
            if not math.isclose(self.wall_time_s, self.batch_wall_time_s / self.batch_size, rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError(f"wall_time_s={self.wall_time_s} 不是整批墙钟 {self.batch_wall_time_s} 的 1/{self.batch_size}")


class BoxPointsParse(NamedTuple):
    """一个框加若干前景点、背景点的解析结果（DEV-0059，W2 的 box_points 提出；适配器的 parse_box_points）。box 是内部格式
    （绝对像素、浮点、xyxy），点是绝对像素 (x, y)；n_points_dropped 是丢掉的点数（不合格或超过上限）。失败时前四项都是 None。"""

    box: tuple[float, float, float, float] | None
    fg_points: tuple[tuple[float, float], ...] | None
    bg_points: tuple[tuple[float, float], ...] | None
    n_points_dropped: int | None
    failure: Any


# 受限选项读数（DEV-0016，D36-4）：一个选项的候选写法，"{}" 换成选项本身。至少包括不带空格与带一个前导空格两种
# （同一字母常有这两种 token）；编码不是单个 token 的写法不用，记在 OptionScores.skipped_spellings 里。
OPTION_SPELLING_TEMPLATES: tuple[str, ...] = ("{}", " {}")


@dataclass(frozen=True)
class OptionSpelling:
    """选项的一种单 token 写法：文本与它的 token id（tokenizer 编码恰为这一个 id，且解码回来就是这段文本）。"""

    text: str
    token_id: int


@dataclass(frozen=True)
class OptionScores:
    """option_logits 的一行结果（DEV-0016，D36-4）。各字段按 options 的顺序。

    - spellings[i]：选项 i 的单 token 写法（至少一个）；skipped_spellings[i]：编码不是单个 token 的写法（不用，只记录）。
    - spelling_logits[i][j]：第一个生成位置上 spellings[i][j] 的 logit（fp32；bf16 模型取 lm_head 输出转成的 fp32）。
    - logits[i] = max_j spelling_logits[i][j]；token_ids[i] 是取到最大值的那个写法的 token id（相等时取 id 小的）。
    - probs：logits 在选项集合上的 softmax（fp32）。
    - choice：logit 最大的选项；相等时取所用 token id 较小的选项。这与 HF greedy 在"只允许全部写法"时的选择相同
      （torch.argmax 取第一个最大值，即 id 最小的 token），所以 choice 恒等于
      generate(allowed_token_ids=全部写法, max_new_tokens=1) 生成的 token 所属的选项。
    - 记账字段同 GenerationResult：n_prompt_tokens 是该行（不含填充）的提示长度；wall_time_s 是整批墙钟按行等分的份额；
      n_generated_tokens 恒为 1（计算量等于只生成 1 个 token 的一次 prefill；src.accounting.Compute.from_generation 可直接用）。
    """

    options: tuple[str, ...]
    spellings: tuple[tuple[OptionSpelling, ...], ...]
    skipped_spellings: tuple[tuple[str, ...], ...]
    spelling_logits: tuple[tuple[float, ...], ...]
    logits: tuple[float, ...]
    token_ids: tuple[int, ...]
    probs: tuple[float, ...]
    choice: str
    n_prompt_tokens: int
    n_visual_tokens: int
    n_image_pixels: int
    wall_time_s: float
    batch_size: int = 1
    batch_wall_time_s: float | None = None

    def __post_init__(self) -> None:
        n = len(self.options)
        if n < 2 or len(set(self.options)) != n:
            raise ValueError(f"options 须是至少 2 个互不相同的选项，收到 {self.options!r}")
        for name in ("spellings", "skipped_spellings", "spelling_logits", "logits", "token_ids", "probs"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"{name} 的长度 {len(getattr(self, name))} 与选项数 {n} 不一致")
        for i, (sp, lg) in enumerate(zip(self.spellings, self.spelling_logits)):
            if not sp or len(sp) != len(lg):
                raise ValueError(f"选项 {self.options[i]!r}：写法 {sp!r} 与 logit {lg!r} 不配套")
            best = max(range(len(sp)), key=lambda j: (lg[j], -sp[j].token_id))
            if self.logits[i] != lg[best] or self.token_ids[i] != sp[best].token_id:
                raise ValueError(f"选项 {self.options[i]!r}：logit / token 不是各写法里的最大值")
        all_ids = [s.token_id for sp in self.spellings for s in sp]
        if len(set(all_ids)) != len(all_ids):
            raise ValueError(f"不同写法用到了同一个 token id：{all_ids}")
        top = max(range(n), key=lambda i: (self.logits[i], -self.token_ids[i]))
        if self.choice != self.options[top]:
            raise ValueError(f"choice={self.choice!r} 不是 logit 最大的选项 {self.options[top]!r}")
        if any(not 0.0 <= p <= 1.0 for p in self.probs) or not math.isclose(sum(self.probs), 1.0, abs_tol=1e-5):
            raise ValueError(f"probs 不是选项集合上的概率分布：{self.probs!r}")
        for name in ("n_prompt_tokens", "n_visual_tokens", "n_image_pixels"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int) or v < 0:
                raise ValueError(f"{name} 必须是非负整数，收到 {v!r}")
        if (self.n_image_pixels == 0) != (self.n_visual_tokens == 0):
            raise ValueError(f"n_image_pixels={self.n_image_pixels} 与 n_visual_tokens={self.n_visual_tokens} 不相容")
        if self.wall_time_s < 0:
            raise ValueError("wall_time_s 不能为负")
        if isinstance(self.batch_size, bool) or not isinstance(self.batch_size, int) or self.batch_size < 1:
            raise ValueError(f"batch_size 必须是正整数，收到 {self.batch_size!r}")
        if self.batch_wall_time_s is not None and not math.isclose(
                self.wall_time_s, self.batch_wall_time_s / self.batch_size, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(f"wall_time_s={self.wall_time_s} 不是整批墙钟 {self.batch_wall_time_s} 的 1/{self.batch_size}")

    @property
    def n_generated_tokens(self) -> int:
        """记账口径：一次 option_logits 的计算量等于只生成 1 个 token。"""
        return 1

    @property
    def choice_token_id(self) -> int:
        """choice 所用的 token id（= generate(allowed_token_ids=全部写法, max_new_tokens=1) 生成的 token）。"""
        return self.token_ids[self.options.index(self.choice)]

    def prob(self, option: str) -> float:
        return self.probs[self.options.index(option)]


@dataclass(frozen=True)
class HiddenStates:
    """按层给出的隐藏状态。

    image[layer] 形状 [n_image_tokens, hidden_size]，按 ModelInputs.images 的顺序拼接所有图的视觉 token。
    query[layer] 形状 [n_query_tokens, hidden_size]；输入里没有查询 token 时 query 为 None。
    layers 是调用方传入的层键（整数层号，或 PRE_NORM_LAST），原样保留顺序；image / query 的键与它相同。
    """

    layers: tuple[LayerKey, ...]
    image: Mapping[LayerKey, Any]
    query: Mapping[LayerKey, Any] | None = None

    def __post_init__(self) -> None:
        if set(self.image) != set(self.layers):
            raise ValueError(f"image 的层号 {sorted(self.image, key=str)} 与 layers {list(self.layers)} 不一致")
        if self.query is not None and set(self.query) != set(self.layers):
            raise ValueError(f"query 的层号 {sorted(self.query, key=str)} 与 layers {list(self.layers)} 不一致")


def resolve_layer_keys(layers: Sequence[LayerKey], check_ints: Callable[[Sequence[Any]], list[int]]
                       ) -> tuple[tuple[LayerKey, ...], list[int], bool]:
    """hidden_states 的 layers → (层键，原样顺序、整数已规范化；其中的整数层号；是否要 PRE_NORM_LAST)。

    整数部分（以及 PRE_NORM_LAST 以外的任何项）原样交给家族的 check_ints（_check_layers：类型、越界、重复、为空），
    所以只给整数时行为与以前完全相同；PRE_NORM_LAST 至多一次；只给 PRE_NORM_LAST 也可以（不再调 check_ints）。
    """
    if isinstance(layers, (str, bytes)):
        check_ints(layers)  # 家族的报错（TypeError："layers 必须是整数序列"）
        raise TypeError("layers 必须是层键的序列")
    items = list(layers)
    is_pre = [isinstance(x, str) and x == PRE_NORM_LAST for x in items]
    if sum(is_pre) > 1:
        raise ValueError(f"layers 里 {PRE_NORM_LAST!r} 出现了 {sum(is_pre)} 次")
    rest = [x for x, p in zip(items, is_pre) if not p]
    ints = check_ints(rest) if (rest or not any(is_pre)) else []
    it = iter(ints)
    keys = tuple(PRE_NORM_LAST if p else next(it) for p in is_pre)
    return keys, list(ints), any(is_pre)


@contextmanager
def capture_output(module: Any) -> Iterator[dict[str, Any]]:
    """在 module 上挂一个前向钩子，把它（最后一次）前向的输出放进返回的字典的 "out"（输出是元组时取第一项）；退出时摘掉钩子。
    不 detach：沿用调用方的梯度模式。hidden_states 取 PRE_NORM_LAST 用（挂在最后一个 decoder 层上）。"""
    box: dict[str, Any] = {}

    def hook(_m: Any, _args: Any, out: Any) -> None:
        box["out"] = out[0] if isinstance(out, tuple) else out
        box["n_calls"] = box.get("n_calls", 0) + 1

    handle = module.register_forward_hook(hook)
    try:
        yield box
    finally:
        handle.remove()


def captured_output(box: Mapping[str, Any], *, what: str) -> Any:
    """capture_output 抓到的输出；这次前向里模块必须恰好被调用一次，否则 PRE_NORM_LAST 无定义，报错。"""
    n = int(box.get("n_calls", 0))
    if n != 1:
        raise RuntimeError(f"{what}：最后一个 decoder 层在这次前向里被调用了 {n} 次（应为 1 次），{PRE_NORM_LAST} 无定义")
    return box["out"]


# format_boxes（DEV-0027）：换算到 0–1000 之后允许的越界量（单位：0–1000 的刻度）。越界不超过它的坐标裁到 [0, 1000]（浮点误差、
# 标注框贴边时 x + w 略大于 W）；超过就报错，不猜。
NATIVE_BOX_TOLERANCE = 0.5


def native_box_ints(boxes: Sequence[Sequence[float]], *, width: float, height: float) -> list[tuple[int, int, int, int]]:
    """format_boxes 的公共一步：绝对像素 xyxy → x 在前的 0–1000 整数四元组（x_norm = x / W · 1000，Python round 取整，与 InternVL
    文档的归一化公式相同）。每个框必须是 4 个有限数、x2 ≥ x1、y2 ≥ y1；换算后落在 [−tol, 1000 + tol]（tol = NATIVE_BOX_TOLERANCE）
    之外就报错，之内的裁到 [0, 1000]。取整是单调的，所以 x2 ≥ x1 保持。"""
    import numpy as np

    from src.executors.geometry import abs_to_norm1000

    if not (width > 0 and height > 0):
        raise ValueError(f"原图宽高必须为正：W={width}, H={height}")
    if isinstance(boxes, (str, bytes)):
        raise TypeError("boxes 必须是框的序列")
    out = []
    for k, b in enumerate(boxes):
        arr = np.asarray(b, dtype=np.float64)
        if arr.shape != (4,) or not np.all(np.isfinite(arr)):
            raise ValueError(f"第 {k} 个框必须是 4 个有限数 [x1, y1, x2, y2]，收到 {b!r}")
        if arr[2] < arr[0] or arr[3] < arr[1]:
            raise ValueError(f"第 {k} 个框颠倒：{b!r}")
        norm = abs_to_norm1000(arr, width=width, height=height)
        if np.any(norm < -NATIVE_BOX_TOLERANCE) or np.any(norm > 1000.0 + NATIVE_BOX_TOLERANCE):
            raise ValueError(f"第 {k} 个框 {b!r} 超出原图 {width}×{height}（换算到 0–1000 是 {norm.tolist()}）")
        out.append(tuple(min(1000, max(0, int(round(float(v))))) for v in norm))
    return out


def check_box_labels(labels: Sequence[str | None] | None, n: int) -> list[str | None]:
    """format_boxes 的 labels：None（都没有标签）或与框等长的序列，每项是字符串或 None。"""
    if labels is None:
        return [None] * n
    if isinstance(labels, (str, bytes)):
        raise TypeError("labels 必须是与 boxes 等长的序列，不能是单个字符串")
    labels = list(labels)
    if len(labels) != n:
        raise ValueError(f"labels 有 {len(labels)} 项，boxes 有 {n} 个")
    bad = [x for x in labels if x is not None and not isinstance(x, str)]
    if bad:
        raise TypeError(f"label 必须是字符串或 None：{bad!r}")
    return labels


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
    "generate_batch": ("enable_thinking", "max_new_tokens"),  # DEV-0011：规则与 generate 相同
    "option_logits": ("enable_thinking",),  # DEV-0016：思考开关每次显式给出（D1）
    "parse_boxes": ("width", "height"),  # 原图宽高必须由调用方显式给出（DEV-0006）
    "format_boxes": ("width", "height"),  # 同上（DEV-0027）
    "chat_turn_batch": ("tools", "thinking_budget", "answer_budget"),  # DEV-0049：每次显式给出
    "chat_turn": ("tools", "thinking_budget", "answer_budget"),
}
# 子类必须保持为"无默认值"的参数（层号只能由调用方给出）
_REQUIRED_NO_DEFAULT: dict[str, tuple[str, ...]] = {
    "hidden_states": ("layers",),
}
# DEV-0020 的训练侧接口：实现通常来自 mixin（不在子类的 __dict__ 里），所以按 MRO 解析到的函数检查（只要不是基类的占位实现）
_REQUIRED_KWONLY_RESOLVED: dict[str, tuple[str, ...]] = {
    "training_forward": ("enable_thinking",),
    "generate_restricted": ("enable_thinking", "max_new_tokens", "stop_ids"),
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
        for meth, params in _REQUIRED_KWONLY_RESOLVED.items():
            fn = getattr(cls, meth, None)
            if fn is None or fn is ModelAdapter.__dict__.get(meth):
                continue
            sig = inspect.signature(fn)
            for p in params:
                par = sig.parameters.get(p)
                if par is None or par.kind is not inspect.Parameter.KEYWORD_ONLY or par.default is not par.empty:
                    raise TypeError(f"{cls.__name__}.{meth} 的 {p} 必须是无默认值的关键字参数（D1）")

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
        answer_prefix: str | None = None,
        **gen_kwargs: Any,
    ) -> GenerationResult:
        """生成。enable_thinking、max_new_tokens 必须每次显式给出（D1）。

        allowed_token_ids 非空时，每一步只在这组 token 上计算 logits 并采样（简报 §4 第 4 条）；
        训练与推理的限制实现必须一致（简报 §12）。
        answer_prefix（DEV-0033）：作答开头预填，规则见模块说明；None 时与以前逐 token 相同。
        """

    @abc.abstractmethod
    def generate_batch(
        self,
        inputs_list: Sequence[ModelInputs],
        *,
        enable_thinking: bool,
        max_new_tokens: int,
        allowed_token_ids: Sequence[int] | None = None,
        answer_prefix: str | None = None,
        **gen_kwargs: Any,
    ) -> list[GenerationResult]:
        """批量生成（DEV-0011，D28-5），返回与 inputs_list 等长、同顺序的结果。

        约定（三个家族一致）：
        - enable_thinking、max_new_tokens、allowed_token_ids、answer_prefix 与 gen_kwargs 对整批相同；必填参数的规则与 generate
          相同（D1）。answer_prefix（DEV-0033）的规则见模块说明：预填接在每一行提示的末尾（左填充之前），max_new_tokens 不含它。
        - 每一行送进模型的提示（思考开关、system 轮、助手头）与 generate 对同一条输入送进去的逐个相同；行间左填充，
          attention_mask 把填充位置置 0；每一行非填充 token 的位置编码与单独生成时逐个相等（Qwen 的 M-RoPE 三维位置亦然）。
        - 逐行拆分：token_ids 从第一个生成 token 到第一个停止 token（含），之后的填充不计；text、raw_text、thinking_unclosed、
          stopped 按行各自给出；n_prompt_tokens 是该行自己（不含填充）的提示长度；n_visual_tokens、n_image_pixels 按行计。
        - 墙钟：batch_wall_time_s 是整批的值，wall_time_s 按行等分（见 GenerationResult）。
        - 带查询 token 的输入不能生成（与 generate 相同）。
        """

    @abc.abstractmethod
    def option_logits(
        self,
        inputs_list: Sequence[ModelInputs],
        options: Sequence[Sequence[str]],
        *,
        enable_thinking: bool,
    ) -> list[OptionScores]:
        """受限选项读数（DEV-0016，D36-4、D36-6），返回与 inputs_list 等长、同顺序的 OptionScores。

        约定（三个家族一致）：
        - options[k] 是第 k 条输入自己的选项（如 ["A", "B", "C", "D"]），至少 2 个、互不相同、非空、没有首尾空白；
          不能是单个字符串。
        - 送进模型的提示（思考开关、system 轮、助手头）与左填充和 generate_batch 对同一批输入送进去的逐个相同（共用同一段构造）。
        - 只做一次前向：取 HF generate 在 max_new_tokens=1 下的原始 logits（output_logits，logits 处理器之前；HF 把 lm_head 的输出
          转成 fp32），即**第一个生成位置**（提示最后一个 token 之后）的 logits。不经过任何采样或 logits 处理器；显式 greedy、
          num_beams=1。
        - 每个选项的写法由 tokenizer 查出（OPTION_SPELLING_TEMPLATES）：编码恰为一个 token、解码回来就是原文的写法才用；
          不是单 token 的写法记在 skipped_spellings 里；一个可用写法都没有时报错。不同选项不能共用 token。
        - 每个选项的分数是它所有可用写法 logit 的最大值；softmax 在 fp32 下、只在选项集合上做；choice 的规则见 OptionScores。
        - 带查询 token 的输入不能用（与 generate 相同）；入口断言与 generate 相同。
        """

    def warmup(self, images: Sequence[Any] = (), *, instruction: str = "Describe the image.",
               order: Order = "instruction_first", batch_size: int = 1, max_new_tokens: int = 2) -> float:
        """预热（D21）：用给定的图（建议用与正式运行同口径、同尺寸的一张代表图）跑一次不计账的 greedy 生成，返回它的墙钟。

        进程里第一次生成含 JIT、内核编译与库初始化；记账之前调用一次。batch_size > 1 时走 generate_batch（把同一条输入
        复制 batch_size 份），也预热批量路径。结果丢弃，不改适配器的任何状态。
        """
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError(f"batch_size 必须是正整数，收到 {batch_size!r}")
        inputs = self.build_inputs(list(images), instruction, order)
        t0 = time.perf_counter()
        if batch_size == 1:
            self.generate(inputs, enable_thinking=False, max_new_tokens=max_new_tokens, do_sample=False)
        else:
            self.generate_batch([inputs] * batch_size, enable_thinking=False, max_new_tokens=max_new_tokens,
                                do_sample=False)
        return time.perf_counter() - t0

    @abc.abstractmethod
    def hidden_states(
        self, inputs: ModelInputs, layers: Sequence[LayerKey], *, query_embeds: Any | None = None
    ) -> HiddenStates:
        """返回指定层在图像 token 位置和查询 token 位置的隐藏状态。

        layers 由调用方给出（来自探针结果或可学习逐层加权的配置），适配器不设默认层。
        层号按本模块开头的层号约定：层号 0 是视觉特征并入之后的嵌入输出，层号 i 是第 i 个 decoder 层的
        输出（最后一个取最终归一化之后的值），有效范围 [0, num_layers()]，越界报错。
        layers 里还可以有一项 PRE_NORM_LAST（DEV-0024）：最后一个 decoder 层的原始输出（最终归一化之前），与整数层号同一次前向，
        在最后一个 decoder 层上挂前向钩子取得；norm(它) = 层号 num_layers() 的读数（tests/test_adapter_pre_norm_last.py 钉住）。
        层键的解析用 resolve_layer_keys；HiddenStates 的键与传入的层键相同。

        query_embeds（D10 的接口扩展，DEV-0003）：形状 [n_query, hidden_size]。非 None 时，前向前把查询位置的
        输入嵌入替换成它，E3a 的可学习查询 token 从这里进；inputs 里必须有查询 token，个数一致。
        反过来，inputs 里有查询 token 时必须给 query_embeds，不给就报错（DEV-0020，D21-2：原来静默用占位符的行）。
        """

    @abc.abstractmethod
    def vision_features(self, inputs: ModelInputs) -> Any:
        """进入 LLM 之前、经过投影/合并之后的视觉 token（D10，DEV-0003；D15-2 起为抽象方法）。

        形状 [n_image_tokens, hidden_size]，多图按 inputs.images 的顺序拼接。E2 的参考线、E3a 的 C2 对照用它。
        各家族在适配器里写明它与 hidden_states(inputs, [0]) 在图像位置上的关系（Qwen3.5 要求逐位相同）。
        """

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

    @abc.abstractmethod
    def parse_boxes(self, text: str, *, width: float, height: float) -> BoxParse:
        """解析本家族的原生框格式，换算成内部的绝对像素 xyxy（constants.BOX_FORMAT；DEV-0006）。不需要加载模型。

        text 是 GenerationResult.text（剥离思考段之后的回答，D10）；思考段未闭合与空回答由 src.parsing.parse_generation 先判。
        width、height 是原图的宽高（像素）：三个家族都按原图归一化到 0–1000（model_notes §5.5、§6.5、§7.5 已核实），
        与模型输入的缩放、tile 切法无关。返回 BoxParse(boxes, labels, failure)，labels 与 boxes 等长（没有标签的框为 None）。
        规则"不猜"：任何一个框坏了整条判失败，不返回其余的（D19）；各家族的格式与失败判定写在各自的实现里。
        """

    def parse_one_box(self, text: str, *, width: float, height: float) -> OneBoxParse:
        """只要一个框（如 RefCOCO 的 REC），建在 parse_boxes 之上（规则见 src.parsing.one_box）：
        parse_boxes 失败 → 原样给出失败原因（没有框就是 no_box）；成功但零个框 → no_box；多于一个框 → multiple_boxes。"""
        return one_box(self.parse_boxes(text, width=width, height=height))

    def parse_box_points(self, text: str, *, width: float, height: float, max_points: int) -> BoxPointsParse:
        """一个框加至多 max_points 个前景点与 max_points 个背景点（DEV-0059，W2 的 box_points 提出），换算成原图绝对像素。
        本家族的原生格式与规则写在实现里（目前只有 Qwen3.5）；对空串必须返回失败（src.parsing.parse_generation 要求）。
        不需要加载模型。不是抽象方法：其余家族没有实现。"""
        raise NotImplementedError(f"{type(self).__name__} 没有实现 parse_box_points（DEV-0059 只实现 Qwen3.5）")

    # ------------------------------------------------------------------
    # DEV-0027 的接口：预算强制收尾、投影前的视觉特征、原生框的格式化、公开的停止 token。不是抽象方法（只做推理的替身与旧的
    # 测试替身不必实现）；三个家族都实现。
    # ------------------------------------------------------------------

    def generate_batch_budgeted(self, inputs_list: Sequence[ModelInputs], *, thinking_budget: int, answer_budget: int,
                                allowed_token_ids: Sequence[int] | None = None) -> list[GenerationResult]:
        """预算强制收尾的思考生成（D59-1、D59-5；实现与逐条规则见 src/adapters/budget.py）。开思考、greedy，逐行：
        思考段在自然收尾（本家族的思考结束标记）或写满 thinking_budget 个 token 时结束；写满还没收尾的行由适配器补上本家族的
        思考结束标记与紧跟的固定分隔（不计入模型生成，单独记）；作答段在收尾之后最多 answer_budget 个 token，到停止 token 结束。
        allowed_token_ids 只作用于作答段（None = 全词表）。提示、左填充与位置编码与 generate_batch(enable_thinking=True) 相同。
        结果的新字段见 GenerationResult。"""
        raise NotImplementedError(f"{type(self).__name__} 没有实现 generate_batch_budgeted")

    def pre_projection_features(self, inputs: ModelInputs) -> tuple[Any, tuple[int, int]]:
        """投影 / 合并之前的视觉塔输出（D59-8，E2 的 pre_proj 参考线）：(张量 [h·w, D_v], (h, w))，只接受单图，行主序，去掉填充，
        张量在适配器的设备上。可以在 no_grad 下调用（不强制）。各家族取的位置写在实现里。"""
        raise NotImplementedError(f"{type(self).__name__} 没有实现 pre_projection_features")

    def format_boxes(self, boxes: Sequence[Sequence[float]], *, width: float, height: float,
                     labels: Sequence[str | None] | None = None) -> str:
        """parse_boxes 的逆（DEV-0027，D56-5：C3 对照用本家族原生的框格式）：绝对像素 xyxy 的框 → 本家族原生框格式的文本，与 E1
        定位提示要求的输出格式相同。坐标按原图宽高换算到 0–1000 并取整（原生格式是整数），所以往返在本家族的坐标精度内
        （每个坐标差不超过半个单位，即 W/2000、H/2000 像素）。labels 可选，与 boxes 等长。不需要加载模型。"""
        raise NotImplementedError(f"{type(self).__name__} 没有实现 format_boxes")

    # ------------------------------------------------------------------
    # 多轮对话（DEV-0049，W1；docs/w1_plan.md §1；数据结构与解析见 src/adapters/chat.py）。不是抽象方法，目前只有 Qwen3.5 实现。
    # ------------------------------------------------------------------

    def chat_turn_batch(self, conversations: Sequence[Sequence[dict]], *,
                        tools: Sequence[Sequence["ToolSpec"] | None], thinking_budget: int,
                        answer_budget: int) -> list["ChatTurnResult"]:
        """多条对话各生成一轮 assistant 回答（按轮批量），返回与 conversations 等长、同顺序的 ChatTurnResult。

        约定：
        - conversations[k] 是一条完整对话（消息格式与校验见 chat 模块说明），最后一条是 user 或 tool；tools[k] 是它自己的工具
          （None = 没有工具），各条可以不同。
        - 用本家族的对话模板渲染整条对话（tools、开思考、生成提示），每张图（包括 tool 消息里的图）都按适配器的图像口径处理。
        - 生成：开思考、greedy，规则与 generate_batch_budgeted 完全相同（思考段写满 thinking_budget 强制收尾，作答段至多
          answer_budget 个 token，到停止 token 结束）；多条对话左填充成一批锁步生成，每行各自收尾，结果与逐条生成逐 token 相同。
        """
        raise NotImplementedError(f"{type(self).__name__} 没有实现 chat_turn_batch（DEV-0049 只实现 Qwen3.5）")

    def chat_turn(self, messages: Sequence[dict], *, tools: Sequence["ToolSpec"] | None, thinking_budget: int,
                  answer_budget: int) -> "ChatTurnResult":
        """一条对话生成一轮：chat_turn_batch 的批大小 1。"""
        (res,) = self.chat_turn_batch([messages], tools=[tools], thinking_budget=thinking_budget,
                                      answer_budget=answer_budget)
        return res

    def end_of_turn_token_id(self) -> int:
        """本家族的回合结束 token（Qwen、InternVL 是 <|im_end|>，Gemma 是 <turn|>）：E3b 目标序列的最后一个 token。"""
        raise NotImplementedError(f"{type(self).__name__} 没有实现 end_of_turn_token_id")

    def stop_token_ids(self) -> tuple[int, ...]:
        """generate / generate_batch 的全部停止 token（第一个就是 end_of_turn_token_id）。"""
        raise NotImplementedError(f"{type(self).__name__} 没有实现 stop_token_ids")

    # ------------------------------------------------------------------
    # E3 的训练侧接口（DEV-0020，D21-2）。实现与规则的细节：第一部分在 src/adapters/training.py，第二部分在
    # src/adapters/checkpoint.py；三个家族都实现，基类只报 NotImplementedError。
    # ------------------------------------------------------------------

    def training_forward(self, inputs_list: Sequence[ModelInputs], target_ids_list: Sequence[Sequence[int]], *,
                         enable_thinking: bool, return_hidden: Any = None, query_embeds: Any = None) -> Any:
        """教师强制版的 generate：每行的提示（思考前缀、system 轮、助手头）与左填充、位置编码都与 generate_batch 相同，目标
        token 接在提示后面；返回预测各目标 token 的最后一层隐藏状态（最终归一化之后、lm_head 之前，带梯度），位置 t 预测
        第 t+1 个 token（training.TrainingOutput）。return_hidden（training.HiddenRequest）另外返回指定层在查询 / 图像 / 给定区间上的
        隐藏状态（带梯度）。输入里有查询 token 时必须给 query_embeds（[n_query, hidden]，各行共用）。不强制 no_grad。"""
        raise NotImplementedError(f"{type(self).__name__} 没有实现 training_forward")

    def restricted_logits(self, hidden: Any, allowed_ids: Any) -> Any:
        """唯一的受限 logits（训练损失与受限解码都调它）：只对 allowed 的 lm_head 行做矩阵乘，fp32，含 final_logit_softcapping。"""
        raise NotImplementedError(f"{type(self).__name__} 没有实现 restricted_logits")

    def chunked_lm_loss(self, hidden: Any, target_ids: Any, *, chunk_size: int, reduction: str = "mean") -> Any:
        """全词表 LM 交叉熵，按位置分块（与 restricted_logits 同一种 logits 算法）。"""
        raise NotImplementedError(f"{type(self).__name__} 没有实现 chunked_lm_loss")

    def generate_restricted(self, inputs_list: Sequence[ModelInputs], allowed_ids: Sequence[int], *, max_new_tokens: int,
                            stop_ids: Sequence[int], enable_thinking: bool) -> list[GenerationResult]:
        """受限 greedy 解码：每一步的 logits 都由 restricted_logits 算，不经过全词表 logits 与 logits 处理器；
        提示、填充、位置编码与 generate_batch 相同。"""
        raise NotImplementedError(f"{type(self).__name__} 没有实现 generate_restricted")

    def train_mode(self) -> None:
        """LLM 切到训练模式（视觉塔默认冻结，保持 eval）。"""
        raise NotImplementedError(f"{type(self).__name__} 没有实现 train_mode")

    def eval_mode(self) -> None:
        """全部切回 eval；生成类方法要求 eval 模式。"""
        raise NotImplementedError(f"{type(self).__name__} 没有实现 eval_mode")

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        """梯度检查点（use_reentrant=False）。"""
        raise NotImplementedError(f"{type(self).__name__} 没有实现 set_gradient_checkpointing")

    def class_name_spans(self, inputs: ModelInputs, names: Sequence[str]) -> list[Span]:
        """每个类别名在提示（ModelInputs 的 input_ids）里的 token 区间（E3a 开放词汇头；拼法见 training.CLASS_NAME_SEPARATOR）。"""
        raise NotImplementedError(f"{type(self).__name__} 没有实现 class_name_spans")

    def token_table(self) -> dict[str, Any]:
        """新增 token 的表：名字 → id、每行在输入嵌入与 lm_head（Gemma 另有 PLE）里的 sha256（按 bf16 原始字节）、查询 token 的
        登记、家族与模型 revision。"""
        raise NotImplementedError(f"{type(self).__name__} 没有实现 token_table")

    def restore_tokens(self, table: Mapping[str, Any], rows: Mapping[str, Any]) -> None:
        """只核对、不初始化的恢复：重新登记同样的名字（id 必须与表相同），把保存的行写回去，写回后哈希必须与表相同。"""
        raise NotImplementedError(f"{type(self).__name__} 没有实现 restore_tokens")

    def lora_target_modules(self) -> list[str]:
        """LoRA 的目标模块：只含 LLM 的线性层（视觉塔、投影层与 lm_head 不含）。"""
        raise NotImplementedError(f"{type(self).__name__} 没有实现 lora_target_modules")

    def inject_lora(self, *, r: int, lora_alpha: float, lora_dropout: float = 0.0, init_lora_weights: Any = True) -> None:
        """注入 LoRA（必须在 add_grid_tokens、inject_query_tokens 之后；顺序反了报错）。"""
        raise NotImplementedError(f"{type(self).__name__} 没有实现 inject_lora")

    def merge_lora(self) -> None:
        """在内存里合并 LoRA（merge_and_unload）；合并后的权重不落盘（D34）。"""
        raise NotImplementedError(f"{type(self).__name__} 没有实现 merge_lora")

    def save_trained(self, directory: Any, *, extra: Mapping[str, Any] | None = None, query_embeds: Any = None) -> Any:
        """检查点：LoRA（peft 格式）、token 表与新增行、查询 token 的嵌入、extra 张量，加 manifest；不存合并后的权重（D34）。"""
        raise NotImplementedError(f"{type(self).__name__} 没有实现 save_trained")

    def load_trained(self, directory: Any, *, merge: bool = True) -> Any:
        """从检查点恢复：base（固定 revision）→ restore_tokens → 挂 LoRA（merge=True 时在内存里合并）→ 核对 manifest。"""
        raise NotImplementedError(f"{type(self).__name__} 没有实现 load_trained")

    def trainable_token_rows(self, names_or_ids: Sequence[str | int]) -> list[Any]:
        """新增 token 行的训练开关（DEV-0021，D45-3；规则见 checkpoint.TRAINABLE_ROWS_RULE）：让这些 grid token（名字或 id）在
        输入嵌入（不共享时另有 lm_head）里的行求梯度，其余行由梯度掩码钩子冻结；返回要交给优化器的参数（共享权重时只返回一个），
        这些参数必须放进 weight_decay = 0 的参数组。必须在 inject_lora 之后、合并之前调用；再调一次就换成新的行集合。
        save_trained 核对冻结行没有被改动。"""
        raise NotImplementedError(f"{type(self).__name__} 没有实现 trainable_token_rows")
