"""W2 的两轮回路（DEV-0059；D113；接口约定 docs/w2_plan.md §0、§3）：提出 → 候选 → 挑选，以及三种不用模型的挑法与评分。

- 提出（propose）：mode = box 时就是 E1 refexp_box 的原生框（记录来自同一输出目录里的 E1 运行）；mode = box_points 时另跑一次单轮生成，
  由适配器的 parse_box_points 解析（本家族的原生格式与 0–1000 换算都在 src/adapters/ 里；这里只调用）。proposal_of 把两种记录统一成
  {mode, box, failure, fg_points, bg_points}（原图绝对像素）。
- 候选（candidates）：generate_candidates（DEV-0058，src/w2/candidates.py；测试与彩排注入假实现）在提出的框与点上产出 K 个；
  candidates_row 存成 candidates.jsonl 的一行（每个候选的 COCO RLE、名字、ok、info，生成器给的原样），decode_candidates 还原。
  回路里用的是 effective_masks（管理员审查 DEV-0059 的裁定 A：与 W1-F 完全一致）：1 号候选 ok = False（GrabCut 报错，生成器按
  §1 用框代替）时，回路一律把它当作空 mask——w1f 交空，w2a、random_pick、oracle_pick 里选中或回退到 1 号也交空，评分与"最优"按空
  mask 算，渲染给 LLM 的 1 号也画空叠加（selection_candidates，info 记原因）；其余编号的失败候选仍按 §1 用框代替。
- 挑选（select）：run_selection 一组样本一起跑：
  - thinking = true：消息 = [system（可缺）] + user（挑选图在前、提示在后）；逐条用 adapter.chat_inputs 预检（拒收的记解析失败，原因
    input_rejected，不进批），按 src/w1/loop.plan_sub_batches 拆子批（行数 ≤ batch_size，行数 × 最长提示 ≤ max_batch_tokens），
    逐个子批调 adapter.chat_turn_batch（不给工具；开思考、贪心，思考写满 thinking_budget 强制收尾，作答至多 answer_budget）；
  - thinking = false：adapter.build_inputs(挑选图, 提示, image_first) + E1 的 generate_rows（关思考、贪心，E1 的解码参数，至多
    answer_budget 个 token），按顺序每 batch_size 条一批（build_inputs 报错的那条记 input_rejected）。
  批的组成只取决于顺序（与提示长度），是确定的。
- 解析（parse_selection，先经 E1 的 parse_generation 判思考段未闭合、截断、空回答）：去掉首尾的 strip_chars、不分大小写之后，恰是
  none_words 之一 → "none"；恰是一个十进制整数且在 1..K 内 → 编号；超出 1..K → out_of_range；其余 → no_choice。
- 挑法（choose）：w2a = 解析出的编号，none 或解析失败交 1 号（fallback = none / parse_failure）；random_pick = random_index
  （1 + int(sha256(salt|id), 16) mod K）；oracle_pick = oracle_index（真值 IoU 最高的，并列取编号小的，精确比较）；w1f = 1 号。
- 每条样本的记录（candidate_record）：提出、各候选与真值的 inter / union / IoU、最优编号、挑选原文与思考、挑中的编号、fallback、
  最终 mask 的 RLE、计分；计算记账：每次模型调用一条 Compute（提出一次、挑选一次），工具调用 = 候选生成（有框时 1 次）。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from fractions import Fraction
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np

from src.accounting import Compute
from src.executors.raster import mask_to_rle, rle_to_mask
from src.metrics.seg import mask_inter_union, sample_iou
from src.parsing import ParseFailure, ParseResult, parse_generation
from src.w1.loop import image_digest, plan_sub_batches
from src.w2.config import PROPOSE_BOX, PROPOSE_BOX_POINTS, SelectProtocol

FALLBACK_NONE = "none"
FALLBACK_PARSE = "parse_failure"
FALLBACKS: tuple[str | None, ...] = (None, FALLBACK_NONE, FALLBACK_PARSE)
INPUT_REJECTED = "input_rejected"  # 预检（chat_inputs / build_inputs）拒收：记成解析失败的原因
NONE_ANSWER = "none"
METHOD_CHAT = "chat_turn_batch"
METHOD_GENERATE = "generate_batch"
# 裁定 A：1 号候选 ok = False 时回路把它当作空 mask（与 W1-F 的 tool_error 按空 mask 计分一致）
FIRST_NOT_OK_EMPTY = "1 号候选 ok = False（生成器失败）：与 W1-F 一致，回路按空 mask 用（评分、挑选图都是空）"


# ---------------------------------------------------------------------------
# 解析
# ---------------------------------------------------------------------------


def parse_selection(text: str, *, k: int, none_words: Sequence[str], strip_chars: str) -> ParseResult:
    """挑选的作答 → ParseResult(value = 1..k 的编号或 "none", failure)。规则见模块说明；空串判 no_choice（parse_generation 要求）。"""
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError(f"k 必须是正整数，收到 {k!r}")
    s = text.strip().strip(strip_chars).strip().lower()
    if s in none_words:
        return ParseResult(NONE_ANSWER, None)
    if s and s.isascii() and s.isdigit():
        v = int(s)
        return ParseResult(v, None) if 1 <= v <= k else ParseResult(None, ParseFailure.OUT_OF_RANGE)
    return ParseResult(None, ParseFailure.NO_CHOICE)


def parse_box_points_generation(res: Any, adapter: Any, *, width: float, height: float, max_points: int) -> Any:
    """box_points 的作答 → adapter.parse_box_points（本家族的原生格式与 0–1000 换算在适配器里，DEV-0059 只有 Qwen3.5 实现），
    先经 E1 的 parse_generation 判思考段未闭合、截断、空回答。返回 src.adapters.base.BoxPointsParse。"""
    return parse_generation(res, adapter.parse_box_points, width=width, height=height, max_points=max_points)


# ---------------------------------------------------------------------------
# 提出与候选
# ---------------------------------------------------------------------------


def proposal_of(mode: str, rec: Mapping[str, Any]) -> dict[str, Any]:
    """提出的记录（mode = box：E1 refexp_box 的预测记录；box_points：propose_box_points 的记录）→ {mode, box, failure, fg_points, bg_points}。"""
    parsed = rec.get("parsed") or {}
    box = parsed.get("box")
    out = {"mode": mode, "box": None if box is None else [float(v) for v in box], "failure": rec["failure"],
           "fg_points": [], "bg_points": []}
    if mode == PROPOSE_BOX_POINTS:
        out["fg_points"] = [[float(x), float(y)] for x, y in (parsed.get("fg_points") or [])]
        out["bg_points"] = [[float(x), float(y)] for x, y in (parsed.get("bg_points") or [])]
    elif mode != PROPOSE_BOX:
        raise ValueError(f"未知的提出方式 {mode!r}")
    if (out["box"] is None) == (out["failure"] is None):
        raise RuntimeError(f"{rec.get('id')}：提出的记录要么有框、要么有失败原因（box = {box!r}，failure = {rec['failure']!r}）")
    return out


def proposal_compute(rec: Mapping[str, Any], wall_time_s: float) -> Compute:
    """提出这一次模型调用（E1 记录的 token 数；墙钟取那次运行的 logs/timing.jsonl）。"""
    return Compute.call(generated_tokens=int(rec["n_generated_tokens"]),
                        prompt_tokens=int(rec["n_prompt_tokens"]) + int(rec.get("n_inserted_tokens") or 0),
                        visual_tokens=int(rec["n_visual_tokens"]), image_pixels=int(rec["n_image_pixels"]),
                        call_time_s=float(wall_time_s), new_round=True)


def candidates_row(sample_id: str, image_size: tuple[int, int], proposal: Mapping[str, Any], cands: Sequence[Any] | None,
                   k: int) -> dict[str, Any]:
    """candidates.jsonl 的一行。cands 为 None：提出失败，没有候选。入口核对：恰 K 个、mask 是原图分辨率的 bool、ok 是 bool。"""
    w, h = image_size
    row: dict[str, Any] = {"id": sample_id, "image_size": [int(w), int(h)], "proposal": dict(proposal), "k": int(k),
                           "n_tool_calls": 0, "candidates": None}
    if cands is None:
        return row
    if len(cands) != k:
        raise RuntimeError(f"{sample_id}：generate_candidates 应产出 K = {k} 个候选，收到 {len(cands)} 个")
    out = []
    for i, c in enumerate(cands, start=1):
        m = np.asarray(c.mask)
        if m.dtype != bool or m.shape != (h, w):
            raise RuntimeError(f"{sample_id}：候选 {i}（{c.name}）的 mask 应是 (H, W) = {(h, w)} 的 bool，收到 {m.dtype} {m.shape}")
        if not isinstance(c.ok, bool) or not isinstance(c.name, str):
            raise RuntimeError(f"{sample_id}：候选 {i} 的 name / ok 类型不对")
        out.append({"index": i, "name": c.name, "ok": c.ok, "info": json.loads(json.dumps(dict(c.info), default=str)),
                    "mask": {"rle": mask_to_rle(m), "area": int(np.count_nonzero(m))}})
    row.update(n_tool_calls=1, candidates=out)
    return row


def decode_candidates(row: Mapping[str, Any]) -> list[np.ndarray]:
    w, h = row["image_size"]
    out = []
    for c in row["candidates"] or []:
        m = rle_to_mask(c["mask"]["rle"])
        if m.shape != (h, w):
            raise RuntimeError(f"{row['id']}：候选 {c['index']} 的 RLE 尺寸 {m.shape} 与原图 {(h, w)} 不符")
        out.append(m)
    return out


def first_as_empty(row: Mapping[str, Any]) -> bool:
    """这一行的 1 号候选是否按空 mask 用（裁定 A：ok = False）。"""
    cands = row["candidates"] or []
    return bool(cands) and cands[0]["ok"] is False


def effective_masks(row: Mapping[str, Any]) -> list[np.ndarray]:
    """回路里用的候选 mask：decode_candidates，1 号候选 ok = False 时换成空 mask（裁定 A），其余原样（失败的仍是框，§1）。"""
    masks = decode_candidates(row)
    if first_as_empty(row):
        masks[0] = np.zeros_like(masks[0])
    return masks


def selection_candidates(row: Mapping[str, Any], candidate_cls: Any) -> list[Any]:
    """交给 render_selection 的候选（candidate_cls 是候选生成器模块的 Candidate）：mask 用 effective_masks；按空 mask 用的 1 号候选
    在 info 里加 w2_as_empty = 原因。"""
    out = []
    for c, m in zip(row["candidates"], effective_masks(row)):
        info = dict(c["info"])
        if c["index"] == 1 and first_as_empty(row):
            info["w2_as_empty"] = FIRST_NOT_OK_EMPTY
        out.append(candidate_cls(name=c["name"], mask=m, ok=c["ok"], info=info))
    return out


def candidates_digest_rows(rows: Sequence[Mapping[str, Any]]) -> str:
    """候选内容的指纹（不含 info 里的耗时）：每行的 id、提出、每个候选的编号、名字、ok、RLE。下游设置的运行身份用它。"""
    h = hashlib.sha256()
    for r in rows:
        doc = {"id": r["id"], "proposal": r["proposal"],
               "candidates": None if r["candidates"] is None else [[c["index"], c["name"], c["ok"], c["mask"]["rle"]]
                                                                  for c in r["candidates"]]}
        h.update(json.dumps(doc, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n")
    return h.hexdigest()


# ---------------------------------------------------------------------------
# 挑法
# ---------------------------------------------------------------------------


def random_index(salt: str, sample_id: str, k: int) -> int:
    """random_pick：1 + int(sha256(salt|id), 16) mod K。"""
    return 1 + int(hashlib.sha256(f"{salt}|{sample_id}".encode("utf-8")).hexdigest(), 16) % int(k)


def _iou_frac(inter: int, union: int) -> Fraction:
    return Fraction(int(inter), int(union)) if union else Fraction(0)


def oracle_index(stats: Sequence[tuple[int, int]]) -> int:
    """真值 IoU 最高的候选编号（1 起），并列取编号小的；IoU 用整数精确比较。"""
    if not stats:
        raise ValueError("没有候选")
    best, best_v = 1, _iou_frac(*stats[0])
    for i, st in enumerate(stats[1:], start=2):
        v = _iou_frac(*st)
        if v > best_v:
            best, best_v = i, v
    return best


@dataclass
class SelectOutcome:
    """一条样本挑选一步的结果。answer：编号、"none" 或 None（解析失败）；record 进 predictions（不含墙钟）。"""

    id: str
    answer: int | str | None
    parse_failure: str | None
    record: dict[str, Any]
    compute: Compute | None = None
    wall_time_s: float = 0.0

    @property
    def fallback(self) -> str | None:
        if self.answer == NONE_ANSWER:
            return FALLBACK_NONE
        return FALLBACK_PARSE if self.answer is None else None

    @property
    def pick(self) -> int | None:
        return self.answer if isinstance(self.answer, int) else None


def choose(setting: str, *, sample_id: str, k: int, stats: Sequence[tuple[int, int]], salt: str,
           outcome: SelectOutcome | None = None) -> tuple[int, int | None, str | None]:
    """(最终编号, 挑中的编号, fallback)。w2a 要 outcome；none 或解析失败交 1 号。"""
    if setting == "w2a":
        if outcome is None:
            raise ValueError("w2a 要挑选的结果")
        return (outcome.pick if outcome.pick is not None else 1), outcome.pick, outcome.fallback
    if setting == "random_pick":
        i = random_index(salt, sample_id, k)
        return i, i, None
    if setting == "oracle_pick":
        i = oracle_index(stats)
        return i, i, None
    if setting == "w1f":
        return 1, 1, None
    raise ValueError(f"{setting!r} 不是用候选的设置")


# ---------------------------------------------------------------------------
# 挑选一步（模型）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SelectItem:
    """一条样本的挑选输入：挑选图（render_selection 的输出，按顺序）、渲染好的提示、候选数 K。"""

    id: str
    images: tuple[Any, ...]
    prompt: str
    k: int


def select_messages(item: SelectItem, system: str | None) -> list[dict[str, Any]]:
    msgs: list[dict[str, Any]] = []
    if system is not None:
        msgs.append({"role": "system", "content": [{"type": "text", "text": system}]})
    msgs.append({"role": "user", "content": [{"type": "image", "image": im} for im in item.images]
                 + [{"type": "text", "text": item.prompt}]})
    return msgs


def _pixels(res: Any, where: str) -> int:
    px = getattr(res, "n_image_pixels", None)
    if px is None:
        if res.n_visual_tokens:
            raise RuntimeError(f"{where}：结果有 {res.n_visual_tokens} 个视觉 token，却没有 n_image_pixels（D28-3）")
        return 0
    return int(px)


def _outcome(item: SelectItem, res: Any, protocol: SelectProtocol, method: str, batch_rows: int) -> SelectOutcome:
    chat = method == METHOD_CHAT
    view = SimpleNamespace(thinking_unclosed=(res.thinking_end == "unclosed") if chat else bool(res.thinking_unclosed),
                           stopped=res.stopped, text=res.text)
    p = parse_generation(view, parse_selection, k=item.k, none_words=protocol.none_words, strip_chars=protocol.strip_chars)
    px = _pixels(res, item.id)
    inserted = int(getattr(res, "n_inserted_tokens", 0) or 0)
    rec = {"method": method, "batch_rows": batch_rows, "prompt": item.prompt, "images": [image_digest(im) for im in item.images],
           "thinking": res.thinking if chat else None, "thinking_end": res.thinking_end if chat else None, "text": res.text,
           "stopped": res.stopped, "n_prompt_tokens": int(res.n_prompt_tokens), "n_generated_tokens": int(res.n_generated_tokens),
           "n_inserted_tokens": inserted, "n_visual_tokens": int(res.n_visual_tokens), "n_image_pixels": px,
           "answer": p.value, "parse_failure": None if p.failure is None else p.failure.value}
    if chat and (res.tool_calls or res.tool_call_errors):
        rec["tool_call_like_text"] = True  # 没给工具，作答里却写了工具调用：照常按文字解析（多半是解析失败）
    comp = Compute.call(generated_tokens=int(res.n_generated_tokens), prompt_tokens=int(res.n_prompt_tokens) + inserted,
                        visual_tokens=int(res.n_visual_tokens), image_pixels=px, call_time_s=float(res.wall_time_s), new_round=True)
    return SelectOutcome(item.id, p.value, rec["parse_failure"], rec, comp, float(res.wall_time_s))


def _rejected(item: SelectItem, method: str, err: Exception) -> SelectOutcome:
    rec = {"method": method, "batch_rows": 0, "prompt": item.prompt, "images": [image_digest(im) for im in item.images],
           "answer": None, "parse_failure": INPUT_REJECTED, "input_error": f"{type(err).__name__}: {err}"}
    return SelectOutcome(item.id, None, INPUT_REJECTED, rec, None, 0.0)


def run_selection(adapter: Any, items: Sequence[SelectItem], protocol: SelectProtocol, *, system: str | None, batch_size: int,
                  max_batch_tokens: int | None, decoding: Mapping[str, Any] | None = None,
                  order: str = "image_first") -> list[SelectOutcome]:
    """一组样本的挑选（规则见模块说明），返回与 items 同序的结果。decoding 是 E1 的解码参数（thinking = false 时用）。"""
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError(f"batch_size 必须是正整数，收到 {batch_size!r}")
    out: dict[str, SelectOutcome] = {}
    if protocol.thinking:
        ready, widths, convs = [], [], []
        for it in items:
            msgs = select_messages(it, system)
            try:
                ci = adapter.chat_inputs(msgs, tools=None)
            except ValueError as e:  # ChatMessageError 是 ValueError 的子类（D104）
                out[it.id] = _rejected(it, METHOD_CHAT, e)
                continue
            ready.append(it)
            widths.append(len(ci.row.ids))
            convs.append(msgs)
        for part in plan_sub_batches(widths, batch_size, max_batch_tokens):
            res = adapter.chat_turn_batch([convs[i] for i in part], tools=[None] * len(part),
                                          thinking_budget=protocol.thinking_budget, answer_budget=protocol.answer_budget)
            if len(res) != len(part):
                raise RuntimeError(f"chat_turn_batch 返回 {len(res)} 行，输入 {len(part)} 行")
            for i, r in zip(part, res):
                out[ready[i].id] = _outcome(ready[i], r, protocol, METHOD_CHAT, len(part))
    else:
        if system is not None:
            raise ValueError("thinking = false 走 build_inputs + generate_batch（E1 的单轮），不支持 system 提示")
        from src.e1.runner import generate_rows

        if decoding is None:
            raise ValueError("thinking = false 要 E1 的解码参数（decoding）")
        ready, inputs = [], []
        for it in items:
            try:
                inputs.append(adapter.build_inputs(list(it.images), it.prompt, order))
            except ValueError as e:
                out[it.id] = _rejected(it, METHOD_GENERATE, e)
                continue
            ready.append(it)
        for a in range(0, len(ready), batch_size):
            part = list(range(a, min(a + batch_size, len(ready))))
            res = generate_rows(adapter, [inputs[i] for i in part], batch_size=batch_size, enable_thinking=False,
                                max_new_tokens=protocol.answer_budget, decoding=decoding)
            if len(res) != len(part):
                raise RuntimeError(f"generate_batch 返回 {len(res)} 行，输入 {len(part)} 行")
            for i, r in zip(part, res):
                out[ready[i].id] = _outcome(ready[i], r, protocol, METHOD_GENERATE, len(part))
    return [out[it.id] for it in items]


# ---------------------------------------------------------------------------
# 一条样本的记录与计分
# ---------------------------------------------------------------------------


def _tokens(calls: Sequence[Compute], inserted: int) -> dict[str, int]:
    return {"prompt": int(sum(c.prompt_tokens for c in calls)) - inserted, "generated": int(sum(c.generated_tokens for c in calls)),
            "inserted": inserted, "visual": int(sum(c.visual_tokens for c in calls)),
            "image_pixels": int(sum(c.image_pixels for c in calls))}


def candidate_record(setting: str, sample: Any, gt: np.ndarray, row: Mapping[str, Any], *, salt: str,
                     proposal_rec: Mapping[str, Any], proposal_wall_s: float = 0.0,
                     outcome: SelectOutcome | None = None) -> tuple[dict[str, Any], list[Compute]]:
    """一条样本在用候选的设置（w2a、random_pick、oracle_pick、w1f）下的记录与这条样本的模型调用（提出一次，墙钟 proposal_wall_s；
    w2a 另加挑选一次）。记录里不含墙钟。"""
    w, h = row["image_size"]
    if gt.shape != (h, w):
        raise ValueError(f"{sample.id}：GT mask 形状 {gt.shape} 与原图 (H, W) = {(h, w)} 不符")
    prop = row["proposal"]
    calls = [proposal_compute(proposal_rec, proposal_wall_s)]
    inserted = int(proposal_rec.get("n_inserted_tokens") or 0)
    rec: dict[str, Any] = {"id": sample.id, "source": sample.source, "split": sample.split, "setting": setting, "task": "refseg",
                           "instruction": sample.fields["expression"], "image_size": [int(w), int(h)], "proposal": dict(prop),
                           "n_candidates": 0, "candidates": [], "best_index": None, "best_iou": None}
    mask = None
    if row["candidates"] is None:
        rec.update(pick=None, fallback=None, final_index=None, final_name=None, failure=prop["failure"], answer=None)
    else:
        masks = effective_masks(row)
        stats = [mask_inter_union(m, gt) for m in masks]
        pairs = [(int(s.inter), int(s.union)) for s in stats]
        cands = [{"index": c["index"], "name": c["name"], "ok": c["ok"], "inter": s.inter, "union": s.union,
                  "iou": sample_iou(s.inter, s.union)} for c, s in zip(row["candidates"], stats)]
        if first_as_empty(row):
            cands[0]["as_empty"] = True  # 裁定 A：1 号候选失败，按空 mask 用
        best = oracle_index(pairs)
        if setting == "w2a":
            if outcome is None or outcome.id != sample.id:
                raise RuntimeError(f"{sample.id}：w2a 缺这条样本的挑选结果")
            rec["select"] = dict(outcome.record)
            if outcome.compute is not None:
                calls.append(outcome.compute)
                inserted += int(outcome.record.get("n_inserted_tokens") or 0)
        final, pick, fb = choose(setting, sample_id=sample.id, k=len(masks), stats=pairs, salt=salt, outcome=outcome)
        mask = masks[final - 1]
        rec.update(n_candidates=len(masks), candidates=cands, best_index=best, best_iou=cands[best - 1]["iou"], pick=pick,
                   fallback=fb, final_index=final, final_name=row["candidates"][final - 1]["name"], failure=None,
                   answer={"kind": "mask", "candidate": final, "mask": {"rle": mask_to_rle(mask), "area": int(np.count_nonzero(mask))}})
        rec["final_is_best"] = _iou_frac(*pairs[final - 1]) == _iou_frac(*pairs[best - 1])
    st = mask_inter_union(mask if rec["failure"] is None else None, gt)
    rec.update(tokens=_tokens(calls, inserted), n_model_calls=len(calls), n_tool_calls=int(row["n_tool_calls"]),
               score={"inter": st.inter, "union": st.union, "iou": sample_iou(*st)})
    return rec, calls


__all__ = [
    "FALLBACKS",
    "FALLBACK_NONE",
    "FALLBACK_PARSE",
    "FIRST_NOT_OK_EMPTY",
    "INPUT_REJECTED",
    "METHOD_CHAT",
    "METHOD_GENERATE",
    "NONE_ANSWER",
    "SelectItem",
    "SelectOutcome",
    "candidate_record",
    "candidates_digest_rows",
    "candidates_row",
    "choose",
    "decode_candidates",
    "effective_masks",
    "first_as_empty",
    "oracle_index",
    "parse_box_points_generation",
    "parse_selection",
    "proposal_compute",
    "proposal_of",
    "random_index",
    "run_selection",
    "select_messages",
    "selection_candidates",
]
