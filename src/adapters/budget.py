"""预算强制收尾的思考生成（DEV-0027；D59-1 E1 的思考子集、D59-5 E5 的思维链）：generate_batch_budgeted 的公共实现。

与模型家族无关；家族相关的只有思考段的格式（ThinkFormat：开头标记在不在输出里、结束标记、结束标记后面的固定分隔），由各适配器的
`_think_markers()` 给出，以及与 generate_restricted 共用的钩子（`_batch_kw`、`_decode_positions`、`_decode_step_extra`、
`_forward_scope`、`_stop_ids`、`_pad_id`、`split_answer`）。

逐行的规则（开思考，greedy）：
1. 思考段：从第一个生成的 token 起。
   - 思考段开头在提示里的家族（Qwen：助手头以 `<think>\\n` 结尾），一开始就在思考段里。
   - 开头在输出里的家族（Gemma 的 `<|channel>`、InternVL 的 `<think>`）：第一个 token 是开头标记才进入思考段；否则这一行没有思考段，
     直接作答（thinking_end = not_opened），第一个 token 就是作答段的第一个 token。依据：RUN-0005 的 think_greedy 记录里，凡是写了
     思考段的输出都以开头标记开始（experiments/P0/0005_thinking_probe/records.jsonl；InternVL-2B 12 条里 4 条直接作答）。
     给了 allowed_token_ids 时，这第一步在 allowed ∪ {开头标记} 里取：要么开始思考，要么作答段的第一个 token（受限）。
   - 思考段里模型写出结束标记 → 自然收尾（natural），转入作答段；写出停止 token → 没有收尾就结束（unclosed，没有作答段）；
     写满 thinking_budget 个 token 仍没收尾 → 强制收尾（forced）：适配器把结束标记与固定分隔（ThinkFormat.insert_ids）逐个喂进去，
     这几个 token 不计入模型生成，单独记（GenerationResult.inserted_*）。第 thinking_budget 个 token 恰是结束标记时算自然收尾。
   - thinking_budget 计的是思考段里模型生成的全部 token：Gemma、InternVL 含开头标记，Qwen 的开头标记在提示里不计。
2. 作答段：自然收尾时从模型的结束标记之后起（模型自己写的分隔算作答 token），强制收尾时从补的 token 之后起；最多 answer_budget
   个 token，到停止 token 结束（stopped = True），写满预算仍没有停止 token 时 stopped = False（E1 判为截断）。自然收尾的行也是同样的
   作答上限，不因为思考短而多拿预算。allowed_token_ids 只作用于作答段：每一步只在 allowed 里取 argmax。

实现：一次预填 + 续用 KV cache 的 greedy 循环（同 training.greedy_restricted_loop 的做法），整批锁步：每一步每行追加一个 token——
模型选的（思考段用全词表 argmax，作答段用 allowed 里的 argmax），或者强制收尾时要补的 token，已结束的行补 pad（与 HF 相同）。
- 每一步调**外层模型**（`self._model`，含 lm_head 与 Gemma 的 final_logit_softcapping），取最后一个位置的 logits 转 fp32 再 argmax，
  与 HF greedy 的 `_sample` 相同（`outputs.logits[:, -1].to(float32)`；allowed 的平局取 id 小的，与 -inf 掩码下的 argmax 相同）。
  三个家族的固定 revision 都没有会在 greedy 下生效的 logits 处理器（Qwen 没有 generation_config；Gemma、InternVL 的只有采样参数与
  eos，RUN-0005；入口用 check_greedy_generation_config 核对），所以作答段与思考段的每一步都与 generate_batch 的 greedy 逐个相同
  （CPU fp32 上由测试核对）。
- 预填与 generate_batch 同一段拼批（`_batch_kw`：提示、助手头、左填充、图像张量；Qwen 的 M-RoPE 逐行核对），位置用 HF generate 的
  规则（`_decode_positions`），之后每步只送新 token、attention_mask 末尾补 1、位置 = 上一步 + 1（HF 的
  _update_model_kwargs_for_generation）。补的 token 也是这样一步一个喂进去，所以位置编码与"模型自己写出这几个 token"时完全相同，
  不需要重新预填。
- 记账：n_generated_tokens 只计模型生成的 token；补的 token 是送进模型的输入（与提示 token 同类），记在 n_inserted_tokens；没有重新
  预填。墙钟只含循环本身（CUDA 上前后各同步一次），按行等分。锁步里已结束或正在补 token 的行占用的槽位不单独记（与 generate_batch 的
  填充同一口径）。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import torch

from src.adapters import batching
from src.adapters.base import GenerationResult, ModelInputs
from src.adapters.training import check_allowed_ids

# 行的状态
_PRE, _THINK, _INSERT, _ANSWER, _DONE = "pre", "think", "insert", "answer", "done"


class BudgetError(ValueError):
    """generate_batch_budgeted 的入参不合规（预算、allowed_token_ids）。"""


@dataclass(frozen=True)
class ThinkFormat:
    """本家族的思考段格式（attach 之后由 tokenizer 查出 id；文本原文定义在各适配器模块里）。

    - open_text / open_id：思考段开头标记在输出里时（Gemma、InternVL）的标记与它的 id；开头在提示里（Qwen）时为 None。
    - close_text / close_id：思考结束标记（单个 token）。
    - separator_text：结束标记后面紧跟的固定分隔（可以为空串）。
    - insert_ids：强制收尾时补的 token = 结束标记 + 固定分隔的分词（整段分词必须等于分开分词的拼接）。
    """

    open_text: str | None
    open_id: int | None
    close_text: str
    close_id: int
    separator_text: str
    insert_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if (self.open_text is None) != (self.open_id is None):
            raise ValueError("open_text 与 open_id 必须同时给出或同时为空")
        if not self.insert_ids or self.insert_ids[0] != self.close_id:
            raise ValueError(f"insert_ids {self.insert_ids} 必须以结束标记 {self.close_id} 开头")

    @property
    def insert_text(self) -> str:
        return self.close_text + self.separator_text

    def as_dict(self) -> dict[str, Any]:
        return {"open_text": self.open_text, "open_id": self.open_id, "close_text": self.close_text,
                "close_id": self.close_id, "separator_text": self.separator_text, "insert_text": self.insert_text,
                "insert_ids": list(self.insert_ids)}


def make_think_format(encode: Callable[[str], list[int]], *, open_text: str | None, close_text: str,
                      separator_text: str) -> ThinkFormat:
    """由本家族的 tokenizer（encode = 不加特殊 token 的分词）查出 id，并核对：标记各是单个 token；结束标记 + 分隔整段分词等于
    分开分词的拼接（补进去的 token 就是模型自己写出这段文字时的 token）。"""
    close = encode(close_text)
    if len(close) != 1:
        raise ValueError(f"思考结束标记 {close_text!r} 不是单个 token：{close}")
    open_id = None
    if open_text is not None:
        op = encode(open_text)
        if len(op) != 1:
            raise ValueError(f"思考段开头标记 {open_text!r} 不是单个 token：{op}")
        open_id = op[0]
    sep = encode(separator_text) if separator_text else []
    ins = encode(close_text + separator_text)
    if ins != close + sep:
        raise ValueError(f"{close_text + separator_text!r} 整段分词 {ins} 与分开分词的拼接 {close + sep} 不同")
    return ThinkFormat(open_text, open_id, close_text, close[0], separator_text, tuple(ins))


# 模型 generation_config 里会让 HF 的 greedy 生成多出 logits 处理器的项，与它们的中性值（None 或下面的值都算没开）。预算循环不经过
# HF 的处理器列表，只在这些项都是中性值时才与 generate_batch 的 greedy 相同（入口核对；三家族的固定 revision 都满足，RUN-0005）。
GREEDY_PROCESSOR_FIELDS: dict[str, tuple[Any, ...]] = {
    "repetition_penalty": (1.0,), "encoder_repetition_penalty": (1.0,), "no_repeat_ngram_size": (0,),
    "encoder_no_repeat_ngram_size": (0,), "bad_words_ids": (), "min_length": (0,), "min_new_tokens": (0,),
    "suppress_tokens": (), "begin_suppress_tokens": (), "forced_bos_token_id": (), "forced_eos_token_id": (),
    "sequence_bias": (), "exponential_decay_length_penalty": (), "renormalize_logits": (False,),
    "diversity_penalty": (0.0,), "guidance_scale": (1.0,), "watermarking_config": (), "remove_invalid_values": (False,),
}


def check_greedy_generation_config(gen_cfg: Any) -> None:
    """generation_config 里不能有会改 greedy logits 的项（GREEDY_PROCESSOR_FIELDS）；有就报错，并报出项名与值。"""
    if gen_cfg is None:
        return
    bad = {k: getattr(gen_cfg, k) for k, neutral in GREEDY_PROCESSOR_FIELDS.items()
           if getattr(gen_cfg, k, None) is not None and getattr(gen_cfg, k) not in neutral}
    if bad:
        raise RuntimeError(f"generation_config 里有会改 greedy logits 的项 {bad}：预算循环没有实现它们，与 generate_batch 不再相同")


def check_budget(name: str, v: Any) -> int:
    if isinstance(v, bool) or not isinstance(v, int) or v < 1:
        raise BudgetError(f"{name} 必须是正整数，收到 {v!r}")
    return int(v)


@dataclass
class RowTrace:
    """一行的生成过程（循环的产物）：think = 思考段里模型生成的 token；inserted = 补的 token；answer = 作答段的 token。"""

    phase: str
    think: list[int] = field(default_factory=list)
    inserted: list[int] = field(default_factory=list)
    answer: list[int] = field(default_factory=list)
    queue: list[int] = field(default_factory=list)
    end: str | None = None  # thinking_end
    stopped: bool | None = None

    @property
    def done(self) -> bool:
        return self.phase == _DONE

    @property
    def model_ids(self) -> list[int]:
        return self.think + self.answer

    @property
    def full_ids(self) -> list[int]:
        return self.think + self.inserted + self.answer


class _Rules:
    """逐行的状态转移（模块说明的第 1、2 条）。advance 返回这一行这一步要追加的 token。"""

    def __init__(self, fmt: ThinkFormat, stop_ids: Sequence[int], pad_id: int, thinking_budget: int, answer_budget: int):
        self.fmt, self.stops, self.pad = fmt, frozenset(int(s) for s in stop_ids), int(pad_id)
        self.tb, self.ab = thinking_budget, answer_budget

    def new_row(self) -> RowTrace:
        return RowTrace(phase=_THINK if self.fmt.open_id is None else _PRE)

    def advance(self, r: RowTrace, *, full: int, restricted: int, pre: int) -> int:
        if r.phase == _DONE:
            return self.pad
        if r.phase == _INSERT:
            t = r.queue.pop(0)
            r.inserted.append(t)
            if not r.queue:
                r.phase = _ANSWER
            return t
        if r.phase == _PRE:
            if pre == self.fmt.open_id:
                r.phase = _THINK
                return self._think(r, pre)
            r.end = "not_opened"
            r.phase = _ANSWER
            return self._answer(r, pre)
        if r.phase == _THINK:
            return self._think(r, full)
        return self._answer(r, restricted)

    def _think(self, r: RowTrace, t: int) -> int:
        r.think.append(t)
        if t == self.fmt.close_id:
            r.end, r.phase = "natural", _ANSWER
        elif t in self.stops:
            r.end, r.phase, r.stopped = "unclosed", _DONE, True
        elif len(r.think) == self.tb:
            r.end, r.phase, r.queue = "forced", _INSERT, list(self.fmt.insert_ids)
        return t

    def _answer(self, r: RowTrace, t: int) -> int:
        r.answer.append(t)
        if t in self.stops:
            r.phase, r.stopped = _DONE, True
        elif len(r.answer) == self.ab:
            r.phase, r.stopped = _DONE, False
        return t


def budgeted_greedy_loop(*, forward: Callable[..., Any], prefill_kw: dict[str, Any],
                         step_extra: Callable[[int], dict[str, Any]], fmt: ThinkFormat, stop_ids: Sequence[int], pad_id: int,
                         thinking_budget: int, answer_budget: int, allowed: torch.Tensor | None) -> list[RowTrace]:
    """锁步的 greedy 循环（规则见模块说明）。forward(**kw) 是外层模型（返回 logits 与 past_key_values）；prefill_kw 含 input_ids、
    attention_mask、position_ids 与图像张量；allowed 是升序的 allowed id（设备上的 long 张量）或 None。返回每行的 RowTrace。"""
    rules = _Rules(fmt, stop_ids, pad_id, thinking_budget, answer_budget)
    mask = prefill_kw["attention_mask"]
    pos = prefill_kw["position_ids"]
    batch = mask.shape[0]
    device = mask.device
    rows = [rules.new_row() for _ in range(batch)]
    pre_ids = None
    if allowed is not None and fmt.open_id is not None:
        pre_ids = torch.unique(torch.cat([allowed, allowed.new_tensor([fmt.open_id])]))  # 升序
    max_steps = thinking_budget + len(fmt.insert_ids) + answer_budget
    out = forward(**prefill_kw, use_cache=True, logits_to_keep=1, return_dict=True)
    for _ in range(max_steps):
        logits = out.logits[:, -1, :].to(dtype=torch.float32)  # HF _sample 的取法
        full = logits.argmax(dim=-1)
        restricted = full if allowed is None else allowed[logits.index_select(-1, allowed).argmax(dim=-1)]
        pre = full if pre_ids is None else pre_ids[logits.index_select(-1, pre_ids).argmax(dim=-1)]
        full_l, res_l, pre_l = full.tolist(), restricted.tolist(), pre.tolist()  # 一次同步
        nxt = [rules.advance(r, full=f, restricted=x, pre=p) for r, f, x, p in zip(rows, full_l, res_l, pre_l)]
        if all(r.done for r in rows):
            break
        mask = torch.cat([mask, mask.new_ones((batch, 1))], dim=-1)
        pos = pos[..., -1:] + 1
        out = forward(input_ids=torch.tensor(nxt, dtype=torch.long, device=device).unsqueeze(-1), attention_mask=mask,
                      position_ids=pos, past_key_values=out.past_key_values, use_cache=True, logits_to_keep=1,
                      return_dict=True, **step_extra(batch))
    if not all(r.done for r in rows):
        raise RuntimeError(f"{max_steps} 步之后仍有行没有结束（状态转移有误）")
    return rows


class BudgetSupport:
    """generate_batch_budgeted 的公共实现（三个适配器都继承）。家族钩子：`_think_markers() -> (open_text | None, close_text,
    separator_text)`，以及 generate_restricted 用的那几个（见模块说明）。"""

    def think_format(self) -> ThinkFormat:
        """本家族的思考段格式（第一次调用时由 tokenizer 查出 id 并核对，之后缓存）。"""
        self._require_loaded()
        cached = getattr(self, "_think_format_cache", None)
        if cached is None:
            open_text, close_text, sep = self._think_markers()
            cached = make_think_format(self._encode, open_text=open_text, close_text=close_text, separator_text=sep)
            self._think_format_cache = cached
        return cached

    def generate_batch_budgeted(self, inputs_list: Sequence[ModelInputs], *, thinking_budget: int, answer_budget: int,
                                allowed_token_ids: Sequence[int] | None = None) -> list[GenerationResult]:
        """预算强制收尾的思考生成（规则见模块说明；接口说明见 base.ModelAdapter.generate_batch_budgeted）。"""
        where = "generate_batch_budgeted"
        self._require_loaded()
        tb = check_budget("thinking_budget", thinking_budget)
        ab = check_budget("answer_budget", answer_budget)
        self._require_eval(where)
        items = batching.check_inputs_list(inputs_list, where=where)
        for inp in items:
            self._check_entry(inp, None, where=where)
        check_greedy_generation_config(getattr(self._model, "generation_config", None))
        fmt = self.think_format()
        head = self._model.get_output_embeddings().weight
        allowed = None
        if allowed_token_ids is not None:
            if isinstance(allowed_token_ids, (str, bytes)) or any(isinstance(i, bool) for i in allowed_token_ids):
                raise BudgetError("allowed_token_ids 必须是整数序列")
            ids = [int(i) for i in allowed_token_ids]  # 与 generate_batch 的 _AllowedTokensProcessor 一样接受 numpy 整数
            allowed = torch.sort(check_allowed_ids(ids, int(head.shape[0]), device=head.device)).values
        stops = [int(s) for s in self._stop_ids()]
        pad_id = int(self._pad_id())
        rows, kw = self._batch_kw(items, True)
        prefill = dict(kw)
        prefill["position_ids"] = self._decode_positions(kw)
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)
        t0 = time.perf_counter()
        with self._forward_scope(), torch.no_grad():
            traces = budgeted_greedy_loop(forward=self._model, prefill_kw=prefill, step_extra=self._decode_step_extra,
                                          fmt=fmt, stop_ids=stops, pad_id=pad_id, thinking_budget=tb, answer_budget=ab,
                                          allowed=allowed)
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)
        wall = time.perf_counter() - t0
        return [self._budget_result(row, tr, fmt, stops, wall=wall, n_rows=len(rows)) for row, tr in zip(rows, traces)]

    def _budget_result(self, row: batching.PromptRow, tr: RowTrace, fmt: ThinkFormat, stops: Sequence[int], *, wall: float,
                       n_rows: int) -> GenerationResult:
        text, raw_full, unclosed = self.split_answer(tr.full_ids, enable_thinking=True)
        span = None
        if tr.end == "forced":
            pieces = [self._tok.decode(ids, skip_special_tokens=False) for ids in (tr.think, tr.inserted, tr.answer)]
            raw = "".join(pieces)
            span = (len(pieces[0]), len(pieces[0]) + len(pieces[1]))
        else:
            raw = raw_full
        ans = list(tr.answer)
        if ans and ans[-1] in set(stops):
            ans = ans[:-1]
        answer_text = self._tok.decode(ans, skip_special_tokens=False).strip("\n") if tr.end != "unclosed" else ""
        model_ids = tr.model_ids
        return GenerationResult(
            text=text,
            token_ids=tuple(model_ids),
            n_generated_tokens=len(model_ids),
            n_prompt_tokens=len(row.ids),
            n_visual_tokens=row.n_visual_tokens,
            wall_time_s=wall / n_rows,
            raw_text=raw,
            thinking_unclosed=unclosed,
            stopped=tr.stopped,
            n_image_pixels=row.n_image_pixels,
            batch_size=n_rows,
            batch_wall_time_s=wall,
            thinking_end=tr.end,
            forced_close=tr.end == "forced",
            n_thinking_tokens=len(tr.think),
            n_answer_tokens=len(tr.answer),
            n_inserted_tokens=len(tr.inserted),
            inserted_token_ids=tuple(tr.inserted),
            inserted_at=len(tr.think) if tr.end == "forced" else None,
            inserted_span=span,
            answer_text=answer_text,
        )


__all__ = [
    "BudgetError",
    "BudgetSupport",
    "GREEDY_PROCESSOR_FIELDS",
    "RowTrace",
    "ThinkFormat",
    "budgeted_greedy_loop",
    "check_budget",
    "check_greedy_generation_config",
    "make_think_format",
]
