"""生成的公共部分：参数检查、左填充拼批、调用 model.generate、逐行拆分与结果（DEV-0011，D28-5）。

与模型家族无关；三个适配器的 generate（= 一行的批）与 generate_batch 都走这里，所以单条与批量送进模型的是同一种构造，
差别只在行数与填充。家族相关的部分（每行的提示怎么拼、图像张量怎么拼、停止 token、思考段剥离、位置编码的核对）留在各适配器。

左填充的约定：
- 每行的提示放在右端，左边用该家族的 pad id 补齐到整批最长的一行；attention_mask 在填充位置为 0、其余为 1；
  mm_token_type_ids（Qwen、Gemma）在填充位置为 0（文本）。
- 位置编码由 HF 的 generate 按 attention_mask 推出：一维位置取 cumsum(mask) − 1，所以每行非填充 token 的位置与单独生成时
  逐个相等；Qwen 的 M-RoPE 三维位置由 get_rope_index 按 mask 跳过填充后计算（Qwen 适配器在生成前逐行核对，测试另用 hook 核对
  每一步实际送进语言模型的位置）。
- 逐行拆分：第一个停止 token（含）之后必须全是 pad（HF 对已结束的行补 pad），否则报错；没有停止 token 的行必须恰好有
  max_new_tokens 个 token（被截断），否则报错。

受限选项读数（DEV-0016，D36-4）：三个适配器的 option_logits 与 generate_batch 用同一段拼批（提示、助手头、左填充、图像张量），
然后走 first_step_logits：调一次 model.generate(max_new_tokens=1, output_logits=True)，取原始 logits（logits 处理器之前；HF 在
_sample 里把 lm_head 的输出转成 fp32），也就是第一个生成位置的 logits；位置编码与 generate_batch 的第一步逐个相同。
写法查找（option_spellings）、取最大值与 softmax（score_options）与家族无关，都在这里。
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

import torch

from src.adapters.base import (
    OPTION_SPELLING_TEMPLATES,
    GenerationResult,
    ModelInputs,
    OptionScores,
    OptionSpelling,
)


@dataclass(frozen=True)
class PromptRow:
    """一行送进模型的完整提示（不含填充）与这一行的记账。"""

    ids: tuple[int, ...]
    n_visual_tokens: int
    n_image_pixels: int
    mm_token_type_ids: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if not self.ids:
            raise ValueError("提示为空")
        if self.mm_token_type_ids is not None and len(self.mm_token_type_ids) != len(self.ids):
            raise ValueError("mm_token_type_ids 与提示不等长")


def check_generate_args(enable_thinking: Any, max_new_tokens: Any) -> None:
    if not isinstance(enable_thinking, bool):
        raise TypeError(f"enable_thinking 必须是 bool，收到 {enable_thinking!r}")
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens < 1:
        raise ValueError(f"max_new_tokens 必须是正整数，收到 {max_new_tokens!r}")


def check_inputs_list(inputs_list: Any, *, where: str = "generate_batch") -> list[ModelInputs]:
    """inputs_list：非空的 ModelInputs 序列，都不带查询 token。where 是调用方的方法名，写进报错。"""
    if isinstance(inputs_list, ModelInputs):
        raise TypeError("inputs_list 必须是 ModelInputs 的序列（单条请用 generate，或放进列表）")
    items = list(inputs_list)
    if not items:
        raise ValueError("inputs_list 为空")
    for k, inp in enumerate(items):
        row = "" if where == "generate" else f"（inputs_list[{k}]）"
        if not isinstance(inp, ModelInputs):
            raise TypeError(f"inputs{row} 必须是 ModelInputs，收到 {type(inp).__name__}")
        if inp.query_span is not None:
            raise ValueError(f"带查询 token 的输入{row}不能用于 {where}（查询 token 只供 hidden_states 读出）")
    return items


def check_reserved(gen_kwargs: dict[str, Any], reserved: Iterable[str]) -> None:
    clash = sorted(set(reserved) & set(gen_kwargs))
    if clash:
        raise ValueError(f"gen_kwargs 不能覆盖适配器控制的参数：{clash}")


def pad_left(rows: Sequence[Sequence[int]], *, pad_value: int, device: torch.device) -> torch.Tensor:
    """左填充成 [B, L] 的 long 张量（L = 最长一行）。"""
    width = max(len(r) for r in rows)
    out = torch.full((len(rows), width), int(pad_value), dtype=torch.long)
    for b, r in enumerate(rows):
        if len(r):
            out[b, width - len(r):] = torch.tensor(list(r), dtype=torch.long)
    return out.to(device)


def batch_prompt(rows: Sequence[PromptRow], *, pad_id: int, device: torch.device) -> dict[str, torch.Tensor]:
    """input_ids、attention_mask（以及各行都给了时的 mm_token_type_ids）的左填充拼批。"""
    ids = pad_left([r.ids for r in rows], pad_value=pad_id, device=device)
    mask = pad_left([[1] * len(r.ids) for r in rows], pad_value=0, device=device)
    kw = {"input_ids": ids, "attention_mask": mask}
    have_mm = [r.mm_token_type_ids is not None for r in rows]
    if any(have_mm):
        if not all(have_mm):
            raise RuntimeError("有的行给了 mm_token_type_ids、有的没给")
        kw["mm_token_type_ids"] = pad_left([r.mm_token_type_ids for r in rows], pad_value=0, device=device)
    return kw


def n_padding(attention_mask: torch.Tensor) -> list[int]:
    """每行左边的填充个数（并核对填充都在左边：mask 是 0…0 1…1 的形式）。"""
    out = []
    for b, row in enumerate(attention_mask.tolist()):
        p = row.index(1) if 1 in row else len(row)
        if any(v != 0 for v in row[:p]) or any(v != 1 for v in row[p:]):
            raise RuntimeError(f"第 {b} 行的 attention_mask 不是左填充的形式")
        out.append(p)
    return out


def run_generate(model: Any, kw: dict[str, Any], *, device: torch.device, max_new_tokens: int, stop_ids: Sequence[int],
                 pad_id: int, processors: list[Any], gen_kwargs: dict[str, Any],
                 scope: contextlib.AbstractContextManager | None = None) -> tuple[torch.Tensor, float]:
    """调 model.generate，返回 (CPU 上的输出 [B, L + T], 墙钟秒)。墙钟只含 generate（CUDA 上前后各同步一次）。"""
    from transformers import LogitsProcessorList

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    with (scope if scope is not None else contextlib.nullcontext()), torch.no_grad():
        out = model.generate(
            **kw,
            max_new_tokens=max_new_tokens,
            eos_token_id=list(stop_ids),
            pad_token_id=pad_id,
            logits_processor=LogitsProcessorList(processors),
            **gen_kwargs,
        )
    out = out.cpu()  # 隐式同步
    wall = time.perf_counter() - t0
    ids = kw["input_ids"]
    if out.dim() != 2 or out.shape[0] != ids.shape[0] or not torch.equal(out[:, : ids.shape[1]], ids.cpu()):
        raise RuntimeError("generate 的输出不以输入为前缀")
    return out, wall


def split_generated(gen: Sequence[int], *, stop_ids: Iterable[int], pad_id: int,
                    max_new_tokens: int) -> tuple[list[int], bool]:
    """一行生成的 token → (到第一个停止 token 为止的 token（含）, stopped)。

    第一个停止 token 之后必须全是 pad；没有停止 token 时必须恰好 max_new_tokens 个（被截断）。两条都是对 HF 行为的核对，
    不成立就报错，不猜。
    """
    gen = [int(t) for t in gen]
    stops = set(int(s) for s in stop_ids)
    k = next((i for i, t in enumerate(gen) if t in stops), None)
    if k is None:
        if len(gen) != max_new_tokens:
            raise RuntimeError(f"没有停止 token 的行有 {len(gen)} 个 token，应恰为 max_new_tokens={max_new_tokens}")
        return gen, False
    tail = gen[k + 1:]
    if any(t != pad_id for t in tail):
        raise RuntimeError(f"停止 token 之后不全是 pad（{pad_id}）：{tail[:8]}…")
    return gen[: k + 1], True


def make_results(rows: Sequence[PromptRow], out: torch.Tensor, prompt_len: int, *, wall: float,
                 stop_ids: Sequence[int], pad_id: int, max_new_tokens: int,
                 split_answer: Callable[..., tuple[str, str, bool]], enable_thinking: bool) -> list[GenerationResult]:
    """逐行拆分并组装 GenerationResult。墙钟按行等分（batch_wall_time_s / batch_size）。"""
    n = len(rows)
    share = wall / n
    results = []
    for b, row in enumerate(rows):
        ids, stopped = split_generated(out[b, prompt_len:].tolist(), stop_ids=stop_ids, pad_id=pad_id,
                                       max_new_tokens=max_new_tokens)
        text, raw_text, unclosed = split_answer(ids, enable_thinking=enable_thinking)
        results.append(GenerationResult(
            text=text,
            token_ids=tuple(ids),
            n_generated_tokens=len(ids),
            n_prompt_tokens=len(row.ids),
            n_visual_tokens=row.n_visual_tokens,
            wall_time_s=share,
            raw_text=raw_text,
            thinking_unclosed=unclosed,
            stopped=stopped,
            n_image_pixels=row.n_image_pixels,
            batch_size=n,
            batch_wall_time_s=wall,
        ))
    return results


def image_pixels(inputs: ModelInputs) -> int:
    """一条输入里各图 n_pixels 之和；某张图没记录就报错（适配器一律填写）。"""
    missing = [k for k, i in enumerate(inputs.images) if i.n_pixels is None]
    if missing:
        raise RuntimeError(f"第 {missing} 张图没有记录 n_pixels")
    return sum(int(i.n_pixels) for i in inputs.images)


# ---------------------------------------------------------------------------
# 受限选项读数（DEV-0016，D36-4）
# ---------------------------------------------------------------------------


class OptionSpellingError(ValueError):
    """某个选项没有一个单 token 写法，或两个选项用到了同一个 token。"""


class OptionLogitsError(RuntimeError):
    """first_step_logits 拿到的 logits 不是预期的形状或 dtype（入口断言，任何机器上都会触发），或输出不以输入为前缀。"""


def check_enable_thinking(enable_thinking: Any) -> None:
    if not isinstance(enable_thinking, bool):
        raise TypeError(f"enable_thinking 必须是 bool，收到 {enable_thinking!r}")


def check_options(options: Any, n_rows: int) -> list[tuple[str, ...]]:
    """options：与输入等长的序列，每项是该行的选项（至少 2 个、互不相同、非空、没有首尾空白的字符串）。"""
    if isinstance(options, str):
        raise TypeError("options 必须是每条输入各自的选项列表（如 [['A', 'B'], …]），不能是单个字符串")
    rows = list(options)
    if len(rows) != n_rows:
        raise ValueError(f"options 有 {len(rows)} 行，输入有 {n_rows} 条")
    out = []
    for k, row in enumerate(rows):
        if isinstance(row, str):
            raise TypeError(f"options[{k}] 必须是字符串的序列（如 ['A', 'B']），不能是单个字符串 {row!r}")
        row = tuple(row)
        if len(row) < 2:
            raise ValueError(f"options[{k}] 至少要有 2 个选项，收到 {row!r}")
        bad = [o for o in row if not isinstance(o, str) or o == "" or o != o.strip()]
        if bad:
            raise ValueError(f"options[{k}] 的选项必须是非空、没有首尾空白的字符串：{bad!r}")
        if len(set(row)) != len(row):
            raise ValueError(f"options[{k}] 有重复的选项：{row!r}")
        out.append(row)
    return out


def option_spellings(tokenizer: Any, option: str, *, vocab_rows: int) -> tuple[tuple[OptionSpelling, ...], tuple[str, ...]]:
    """一个选项的写法（OPTION_SPELLING_TEMPLATES）→ (可用的单 token 写法, 不用的写法)。

    可用 = tokenizer.encode(写法, add_special_tokens=False) 恰为一个 id，且 decode 回来就是这段文本。一个可用写法都没有时
    报 OptionSpellingError。id 必须落在 logits 的宽度（lm_head 行数）以内，否则报错（不是"不用"，是对不上）。
    """
    usable: list[OptionSpelling] = []
    skipped: list[str] = []
    for template in OPTION_SPELLING_TEMPLATES:
        text = template.format(option)
        ids = list(tokenizer.encode(text, add_special_tokens=False))
        if len(ids) == 1 and tokenizer.decode(ids) == text:
            tid = int(ids[0])
            if not 0 <= tid < vocab_rows:
                raise OptionSpellingError(f"写法 {text!r} 的 token id {tid} 超出 logits 宽度 {vocab_rows}")
            if all(s.token_id != tid for s in usable):
                usable.append(OptionSpelling(text, tid))
        else:
            skipped.append(text)
    if not usable:
        raise OptionSpellingError(f"选项 {option!r} 的写法 {[t.format(option) for t in OPTION_SPELLING_TEMPLATES]} "
                                  f"都不是单个 token")
    return tuple(usable), tuple(skipped)


def first_step_logits(model: Any, kw: dict[str, Any], *, device: torch.device, stop_ids: Sequence[int], pad_id: int,
                      vocab_rows: int, scope: contextlib.AbstractContextManager | None = None) -> tuple[torch.Tensor, float]:
    """第一个生成位置的原始 logits：(CPU 上 fp32 的 [B, vocab_rows], 墙钟秒)。

    调一次 model.generate(max_new_tokens=1, output_logits=True, return_dict_in_generate=True)：只有 prefill 一次前向；
    output_logits 是 logits 处理器之前的值（HF _sample 里 `outputs.logits[:, -1].to(dtype=torch.float32)` 的拷贝），所以与
    run_generate 的第一步是同一个位置、同一种位置编码与填充。显式 greedy、num_beams=1，不传任何 logits 处理器。
    入口断言：logits 必须恰有一步、形状 [B, vocab_rows]、dtype fp32，输出必须以输入为前缀；不符就报 OptionLogitsError，报出两个值。
    墙钟口径同 run_generate（只含 generate，CUDA 上前后各同步一次）。
    """
    from transformers import LogitsProcessorList

    ids = kw["input_ids"]
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    with (scope if scope is not None else contextlib.nullcontext()), torch.no_grad():
        out = model.generate(
            **kw,
            max_new_tokens=1,
            eos_token_id=list(stop_ids),
            pad_token_id=pad_id,
            do_sample=False,
            num_beams=1,
            output_logits=True,
            return_dict_in_generate=True,
            logits_processor=LogitsProcessorList(),
        )
    steps = getattr(out, "logits", None)
    if steps is None or len(steps) != 1:
        raise OptionLogitsError(f"generate 应给出恰好 1 步的 logits，收到 {None if steps is None else len(steps)} 步")
    logits = steps[0]
    if logits.dtype != torch.float32:
        raise OptionLogitsError(f"第一个生成位置的 logits 应为 torch.float32，收到 {logits.dtype}")
    want = (ids.shape[0], vocab_rows)
    if tuple(logits.shape) != want:
        raise OptionLogitsError(f"logits 形状 {tuple(logits.shape)}，应为 {want}（批大小，lm_head 行数）")
    logits = logits.detach().cpu()  # 隐式同步
    wall = time.perf_counter() - t0
    seq = out.sequences.cpu()
    if seq.dim() != 2 or seq.shape[0] != ids.shape[0] or not torch.equal(seq[:, : ids.shape[1]], ids.cpu()):
        raise OptionLogitsError("generate 的输出不以输入为前缀")
    return logits, wall


def score_options(row_logits: torch.Tensor, spellings: Sequence[Sequence[OptionSpelling]]) -> dict[str, Any]:
    """一行的 fp32 logits [V] → 每个选项的最大 logit、所用 token、选项集合上的 softmax、argmax 选项的下标。

    取最大值与 argmax 的平局规则与 HF greedy 的 torch.argmax 相同（取 id 小的 token），见 base.OptionScores。
    """
    if row_logits.dim() != 1 or row_logits.dtype != torch.float32:
        raise OptionLogitsError(f"score_options 要一维 fp32 的 logits，收到 {tuple(row_logits.shape)} {row_logits.dtype}")
    ids = [s.token_id for sp in spellings for s in sp]
    if len(set(ids)) != len(ids):
        raise OptionSpellingError(f"不同选项的写法用到了同一个 token：{ids}")
    spelling_logits = [tuple(float(v) for v in row_logits[[s.token_id for s in sp]].tolist()) for sp in spellings]
    best = [max(range(len(sp)), key=lambda j, sp=sp, lg=lg: (lg[j], -sp[j].token_id))
            for sp, lg in zip(spellings, spelling_logits)]
    logits = [lg[j] for lg, j in zip(spelling_logits, best)]
    token_ids = [sp[j].token_id for sp, j in zip(spellings, best)]
    # softmax 在 fp32 下做：logits 本来就是 fp32 的值，原样放回 fp32 张量
    probs = torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=0).tolist()
    top = max(range(len(logits)), key=lambda i: (logits[i], -token_ids[i]))
    return {"spelling_logits": tuple(spelling_logits), "logits": tuple(logits), "token_ids": tuple(token_ids),
            "probs": tuple(float(p) for p in probs), "top": top}


def make_option_scores(rows: Sequence[PromptRow], logits: torch.Tensor, options: Sequence[tuple[str, ...]],
                       spelling_table: Sequence[Sequence[tuple[tuple[OptionSpelling, ...], tuple[str, ...]]]], *,
                       wall: float) -> list[OptionScores]:
    """逐行组装 OptionScores。spelling_table[b][i] 是第 b 行第 i 个选项的 (可用写法, 不用的写法)。墙钟按行等分。"""
    n = len(rows)
    share = wall / n
    out = []
    for b, (row, opts, table) in enumerate(zip(rows, options, spelling_table)):
        spellings = [usable for usable, _ in table]
        s = score_options(logits[b], spellings)
        out.append(OptionScores(
            options=tuple(opts),
            spellings=tuple(spellings),
            skipped_spellings=tuple(skipped for _, skipped in table),
            spelling_logits=s["spelling_logits"],
            logits=s["logits"],
            token_ids=s["token_ids"],
            probs=s["probs"],
            choice=opts[s["top"]],
            n_prompt_tokens=len(row.ids),
            n_visual_tokens=row.n_visual_tokens,
            n_image_pixels=row.n_image_pixels,
            wall_time_s=share,
            batch_size=n,
            batch_wall_time_s=wall,
        ))
    return out


def option_logits_rows(model: Any, tokenizer: Any, rows: Sequence[PromptRow], kw: dict[str, Any],
                       options: Sequence[tuple[str, ...]], *, device: torch.device, stop_ids: Sequence[int], pad_id: int,
                       scope: contextlib.AbstractContextManager | None = None) -> list[OptionScores]:
    """option_logits 的公共部分：查写法（先查，写法有问题时不跑模型）→ first_step_logits → make_option_scores。"""
    vocab_rows = int(model.get_output_embeddings().weight.shape[0])
    cache: dict[str, tuple[tuple[OptionSpelling, ...], tuple[str, ...]]] = {}
    table = []
    for opts in options:
        for o in opts:
            if o not in cache:
                cache[o] = option_spellings(tokenizer, o, vocab_rows=vocab_rows)
        ids = [s.token_id for o in opts for s in cache[o][0]]
        if len(set(ids)) != len(ids):
            raise OptionSpellingError(f"选项 {opts!r} 的写法用到了同一个 token：{ids}")
        table.append([cache[o] for o in opts])
    logits, wall = first_step_logits(model, kw, device=device, stop_ids=stop_ids, pad_id=pad_id, vocab_rows=vocab_rows,
                                     scope=scope)
    return make_option_scores(rows, logits, options, table, wall=wall)


__all__ = [
    "OptionLogitsError",
    "OptionSpellingError",
    "PromptRow",
    "batch_prompt",
    "check_enable_thinking",
    "check_generate_args",
    "check_inputs_list",
    "check_options",
    "check_reserved",
    "first_step_logits",
    "image_pixels",
    "make_option_scores",
    "make_results",
    "n_padding",
    "option_logits_rows",
    "option_spellings",
    "pad_left",
    "run_generate",
    "score_options",
    "split_generated",
]
