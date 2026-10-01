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
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

import torch

from src.adapters.base import GenerationResult, ModelInputs


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


__all__ = [
    "PromptRow",
    "batch_prompt",
    "check_generate_args",
    "check_inputs_list",
    "check_reserved",
    "image_pixels",
    "make_results",
    "n_padding",
    "pad_left",
    "run_generate",
    "split_generated",
]
