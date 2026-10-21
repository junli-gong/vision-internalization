"""W1 的回路（DEV-0051；D103；接口约定 docs/w1_plan.md §1–§3）：冻结的 LLM 经适配器的多轮对话接口调用工作台，按轮同步、按批生成。

协议（Protocol，取值在 configs/w1.json 的 protocol；src/w1/config.py 读取）：
- 第 1 轮：system（W1 的系统提示，对所有任务相同）+ user（原图在前，按任务的提示在后）。
- 每一轮：还没结束的样本按原顺序先逐条预检（adapter.chat_inputs；D104：消息里有结构性 token 等，chat_turn_batch 会让整批报错，所以
  拒收的那条单独记 failure = input_rejected、ended_by = input_rejected，不进批），再拆成子批（plan_sub_batches：按顺序贪心，行数
  ≤ batch_size，且 行数 × 子批里最长的提示 ≤ max_batch_tokens；预填不分块，上限按显存定，D104），逐个子批调 adapter.chat_turn_batch
  （开思考、贪心；思考写满 thinking_budget 时适配器强制收尾，作答至多 answer_budget）；之后逐条样本、按出现顺序执行作答段里的工具调用
  （Workbench.call），每次调用的结果追加为一条 tool 消息（文字 + 图）。同一条样本的调用总是按顺序执行；不同样本之间可以并行
  （tool_workers，只影响墙钟）。批的组成只取决于顺序与提示长度，是确定的。
- 历史（D104 的回填规则）：assistant 消息 = content [作答原文]（含模型自己写的 <tool_call> 块）+ reasoning_content（这一轮的思考段），
  不给 tool_calls；模板自己决定保留哪些轮的思考。
- 结束：(1) 某次调用之后 workbench.submission 不为空（提交了答案）——这一段里其后的调用不再执行（记 refused = after_submission）；
  (2) 作答段里既没有解析出的调用、也没有解析失败的调用：这一段就是最终答案。
- 上限：
  - 计数的工具调用最多 max_tool_calls 次；submit_tools（submit_mask、submit_box、submit_count）在 submit_exempt_from_cap 时不计数、
    总是执行（否则用完 8 次之后指代分割再也交不了答案）。计满之后的调用不执行，返回 prompts.tool_cap（refused = tool_cap）。
  - 解析失败的调用（ChatTurnResult.tool_call_errors）不执行、不计数，各返回一条 prompts.tool_call_error；这一轮不算给出最终答案。
  - 第 max_rounds 轮之后还没有答案：failure = no_answer（ended_by = round_cap）。进入第 max_rounds 轮之前，prompts.last_round（不为 None
    时）追加在上一轮最后一条 tool 消息的末尾。
- 最终答案：refseg = 提交的 mask（原图分辨率 bool）；refexp_box = submit_box 的框，或最终答案文字里本家族的原生框（adapter.parse_one_box，
  E1 的解析器）；count = submit_count 的数，或最终答案文字里的单个整数（E1 的 parse_count）。最终答案文字先按 E1 的
  parse_generation 判思考段未闭合、截断、空回答。refseg 给了文字却没有提交 mask：failure = not_submitted；提交了别的种类（例如
  refseg 交了框）：failure = wrong_submission。回路本身不做任何与任务相关的修补。
- 记录（EpisodeRecord）：每轮的思考与作答原文、thinking_end、工具调用（名字、参数、原文、是否计数 / 执行 / 被拒）与结果摘要（文字、
  报错、图只记尺寸与像素 sha256）、最终答案（mask 记 COCO RLE）、失败原因、轮数与调用数；compute（src/accounting.Compute）：每轮记一次
  调用（prompt_tokens 含强制收尾补入的 token，同 E1），样本的墙钟 = 各轮这一行分到的生成墙钟（ChatTurnResult.wall_time_s）+ 这条样本
  工具调用的墙钟。to_json() 不含任何墙钟（predictions.jsonl 因而逐字节可复现），墙钟在 timing()。

ChatTurnResult.n_image_pixels（模型实际看到的像素数，D28-3；D104 已加进约定）：这里入口断言，有视觉 token 却没有像素就报错，
不把"未记录"当 0。
"""

from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from src.accounting import Compute, merge
from src.data.records import Sample
from src.executors.raster import mask_to_rle
from src.parsing import ParseFailure, ParseResult, parse_count, parse_generation

TASKS: tuple[str, ...] = ("refseg", "refexp_box", "count")
ANSWER_KIND = {"refseg": "mask", "refexp_box": "box", "count": "count"}  # 每个任务要的答案种类（Submission 的字段名）
SUBMISSION_FIELDS: tuple[str, ...] = ("mask", "box", "count")
ENDED_BY: tuple[str, ...] = ("submission", "final_answer", "round_cap", "input_rejected")
FAILURE_NO_ANSWER = "no_answer"  # 第 max_rounds 轮之后还没有答案
FAILURE_INPUT_REJECTED = "input_rejected"  # 拼批前的预检（adapter.chat_inputs）拒收这条对话（D104：结构性 token 等）
FAILURE_NOT_SUBMITTED = "not_submitted"  # refseg 给了最终答案文字，没有提交 mask
FAILURE_WRONG_SUBMISSION = "wrong_submission"  # 提交了别的种类（例如 refseg 交了框）
REFUSED_TOOL_CAP = "tool_cap"
REFUSED_AFTER_SUBMISSION = "after_submission"
THINKING_UNCLOSED = "unclosed"  # ChatTurnResult.thinking_end 的取值（同 GenerationResult，src/adapters/base.THINKING_ENDS）


@dataclass(frozen=True)
class Protocol:
    """回路协议（configs/w1.json 的 protocol；docs/w1_plan.md §3）。在开发-调参集上定，跑 test 之前冻结（D103 §二.4）。
    submit_tools、submit_exempt_from_cap 是 DEV-0051 加的两项（见模块说明的"上限"）。"""

    max_rounds: int = 6
    max_tool_calls: int = 8
    thinking_budget: int = 2048
    answer_budget: int = 1024
    disabled_groups: frozenset[str] = frozenset()
    submit_tools: frozenset[str] = frozenset({"submit_mask", "submit_box", "submit_count"})
    submit_exempt_from_cap: bool = True

    def __post_init__(self) -> None:
        for name in ("max_rounds", "max_tool_calls", "thinking_budget", "answer_budget"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise ValueError(f"Protocol.{name} 必须是正整数，收到 {v!r}")
        object.__setattr__(self, "disabled_groups", frozenset(self.disabled_groups))
        object.__setattr__(self, "submit_tools", frozenset(self.submit_tools))
        if "submit" in self.disabled_groups:
            raise ValueError("submit 组不能关（关了就交不了答案）")

    def as_dict(self) -> dict[str, Any]:
        """config.yaml 的 protocol 一节（集合排序，便于逐字节比较）。"""
        return {"max_rounds": self.max_rounds, "max_tool_calls": self.max_tool_calls,
                "thinking_budget": self.thinking_budget, "answer_budget": self.answer_budget,
                "disabled_groups": sorted(self.disabled_groups), "submit_tools": sorted(self.submit_tools),
                "submit_exempt_from_cap": self.submit_exempt_from_cap}


@dataclass(frozen=True)
class LoopPrompts:
    """回路用的提示（src/w1/config.W1Config.loop_prompts 由 configs/w1.json 渲染）：system 与 tool_cap 已按协议填好上限；
    tasks 是按任务的模板；tool_call_error 含 {error}；last_round 为 None 时不追加。"""

    system: str
    tasks: Mapping[str, str]
    tool_cap: str
    tool_call_error: str
    last_round: str | None

    def task_text(self, task: str, sample: Sample) -> str:
        from src.e1.prompts import COUNTBENCHQA_FIELD

        if task not in TASKS:
            raise ValueError(f"未知任务 {task!r}；可用 {TASKS}")
        f = sample.fields
        if task == "count":
            if COUNTBENCHQA_FIELD not in f:
                raise KeyError(f"{sample.id}：计数要 E1 的人工问句（fields[{COUNTBENCHQA_FIELD!r}]，D36-2）")
            return self.tasks[task].format(question=f[COUNTBENCHQA_FIELD].strip()).strip()
        return self.tasks[task].format(expression=f["expression"]).strip()


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------


def image_digest(im: Any) -> dict[str, Any]:
    """图的摘要（不存图）：尺寸 [W, H]、模式与像素 sha256（sha256(模式|W|H|原始字节)）。"""
    w, h = im.size
    h_ = hashlib.sha256(f"{im.mode}|{w}|{h}|".encode("ascii"))
    h_.update(im.tobytes())
    return {"size": [int(w), int(h)], "mode": im.mode, "sha256": h_.hexdigest()}


def tools_digest(specs: Sequence[Any]) -> str:
    """工具清单的 sha256：按 specs() 的顺序，每个工具的 (name, description, parameters) 规范 JSON（键排序、保留中文）。"""
    doc = [{"name": s.name, "description": s.description, "parameters": s.parameters} for s in specs]
    return hashlib.sha256(json.dumps(doc, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _text_msg(role: str, text: str) -> dict[str, Any]:
    return {"role": role, "content": [{"type": "text", "text": text}]}


def _tool_msg(text: str, images: Sequence[Any] = ()) -> dict[str, Any]:
    return {"role": "tool", "content": [{"type": "text", "text": text}] + [{"type": "image", "image": im} for im in images]}


def plan_sub_batches(widths: Sequence[int], batch_size: int, max_batch_tokens: int | None) -> list[list[int]]:
    """一轮里还没结束的样本（按原顺序，widths 是各自的提示长度）→ 子批（下标列表）。按顺序贪心：当前子批再加一行之后，行数不超过
    batch_size，且 行数 × 子批里最长的提示（左填充后的宽度）≤ max_batch_tokens，就加进来，否则另起一批（D104：预填不分块，按显存拆
    子批）。一行本身就超过上限时单独成批。max_batch_tokens 为 None 时只按 batch_size 切。确定：只取决于顺序与宽度。"""
    out: list[list[int]] = []
    cur: list[int] = []
    cur_w = 0
    for i, w in enumerate(widths):
        w = int(w)
        if cur:
            new_w = max(cur_w, w)
            if len(cur) + 1 > batch_size or (max_batch_tokens is not None and (len(cur) + 1) * new_w > max_batch_tokens):
                out.append(cur)
                cur, cur_w = [], 0
        cur.append(i)
        cur_w = max(cur_w, w)
    if cur:
        out.append(cur)
    return out


def _check_box(box: Any, where: str) -> list[float]:
    try:
        vals = [float(v) for v in box]
    except TypeError:
        raise RuntimeError(f"{where}：提交的框不是数的序列：{box!r}") from None
    if len(vals) != 4 or not np.all(np.isfinite(vals)):
        raise RuntimeError(f"{where}：提交的框应是 4 个有限的数（绝对像素 xyxy），收到 {box!r}")
    return vals


# ---------------------------------------------------------------------------
# 记录
# ---------------------------------------------------------------------------


@dataclass
class EpisodeRecord:
    """一条样本的回路记录。to_json() 是 predictions.jsonl 里回路的部分（不含墙钟）；mask 是 refseg 提交的 mask（原图分辨率 bool），
    不进 JSON（JSON 里是 RLE）；timing() 与 workbench_log 写进 logs/。"""

    id: str
    task: str
    instruction: str
    image_size: tuple[int, int]
    tools_sha256: str
    rounds: list[dict[str, Any]] = field(default_factory=list)
    ended_by: str | None = None
    failure: str | None = None
    answer: dict[str, Any] | None = None
    mask: np.ndarray | None = None
    box: list[float] | None = None
    count: int | None = None
    n_tool_calls: int = 0  # 计数的调用（执行了的、不在 submit_tools 里的；submit_exempt_from_cap 为 False 时含提交）
    n_executed: int = 0  # 实际执行的调用（含提交）
    n_refused: int = 0
    n_tool_call_errors: int = 0
    input_error: dict[str, Any] | None = None  # 预检拒收时：{round, error}
    compute: Compute = field(default_factory=Compute)
    round_walls: list[float] = field(default_factory=list)
    tool_times: list[list[float]] = field(default_factory=list)
    workbench_log: list[dict[str, Any]] = field(default_factory=list)

    @property
    def n_rounds(self) -> int:
        return len(self.rounds)

    @property
    def tool_time_s(self) -> float:
        return float(sum(sum(t) for t in self.tool_times))

    def tokens(self) -> dict[str, int]:
        r = self.rounds
        return {"prompt": sum(x["n_prompt_tokens"] for x in r), "generated": sum(x["n_generated_tokens"] for x in r),
                "inserted": sum(x["n_inserted_tokens"] for x in r), "visual": sum(x["n_visual_tokens"] for x in r),
                "image_pixels": sum(x["n_image_pixels"] for x in r), "max_images": max((x["n_images"] for x in r), default=0)}

    def to_json(self) -> dict[str, Any]:
        return {"id": self.id, "task": self.task, "instruction": self.instruction, "image_size": list(self.image_size),
                "tools_sha256": self.tools_sha256, "ended_by": self.ended_by, "failure": self.failure, "answer": self.answer,
                "n_rounds": self.n_rounds, "n_tool_calls": self.n_tool_calls, "n_executed": self.n_executed,
                "n_refused": self.n_refused, "n_tool_call_errors": self.n_tool_call_errors, "input_error": self.input_error,
                "thinking_end": [x["thinking_end"] for x in self.rounds], "tokens": self.tokens(), "rounds": self.rounds}

    def timing(self) -> dict[str, Any]:
        return {"id": self.id, "round_walls": [float(v) for v in self.round_walls],
                "tool_times": [[float(v) for v in t] for t in self.tool_times], "tool_time_s": self.tool_time_s,
                "wall_time_s": self.compute.wall_time_s, "call_time_s": self.compute.call_time_s}


# ---------------------------------------------------------------------------
# 一条样本的状态
# ---------------------------------------------------------------------------


class _Episode:
    def __init__(self, sample: Sample, image: Any, wb: Any, task: str, protocol: Protocol, prompts: LoopPrompts,
                 tools_sha: str) -> None:
        self.sample, self.wb, self.task, self.protocol, self.prompts = sample, wb, task, protocol, prompts
        instr = prompts.task_text(task, sample)
        self.messages: list[dict[str, Any]] = [
            _text_msg("system", prompts.system),
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": instr}]},
        ]
        self.rec = EpisodeRecord(id=sample.id, task=task, instruction=instr, image_size=tuple(image.size), tools_sha256=tools_sha)
        self.calls: list[Compute] = []
        self.done = False

    # -- 结束 ---------------------------------------------------------------

    def _finish(self, ended_by: str, failure: str | None) -> None:
        self.done = True
        self.rec.ended_by = ended_by
        self.rec.failure = failure
        wall = float(sum(self.rec.round_walls)) + self.rec.tool_time_s
        self.rec.compute = merge(self.calls).as_sample(wall_time_s=wall)
        self.rec.workbench_log = list(self.wb.log())

    def finish_submission(self, tool: str) -> None:
        sub = self.wb.submission
        parts = {k: getattr(sub, k, None) for k in SUBMISSION_FIELDS}
        kind = ANSWER_KIND[self.task]
        given = [k for k in SUBMISSION_FIELDS if parts[k] is not None]
        ans: dict[str, Any] = {"source": tool, "kind": kind, "submitted": given}
        val = parts[kind]
        if val is None:
            self.rec.answer = ans
            self._finish("submission", FAILURE_WRONG_SUBMISSION)
            return
        where = f"{self.sample.id}（{tool}）"
        if kind == "mask":
            m = np.asarray(val)
            w, h = self.rec.image_size
            if m.dtype != bool or m.shape != (h, w):
                raise RuntimeError(f"{where}：提交的 mask 应是原图分辨率 (H, W) = {(h, w)} 的 bool 数组，收到 {m.dtype} {m.shape}")
            self.rec.mask = m
            ans["mask"] = {"rle": mask_to_rle(m), "area": int(np.count_nonzero(m))}
        elif kind == "box":
            self.rec.box = _check_box(val, where)
            ans["box"] = list(self.rec.box)
        else:
            if isinstance(val, bool) or not isinstance(val, (int, np.integer)) or int(val) < 0:
                raise RuntimeError(f"{where}：提交的计数应是非负整数，收到 {val!r}")
            self.rec.count = int(val)
            ans["count"] = self.rec.count
        self.rec.answer = ans
        self._finish("submission", None)

    def finish_final(self, res: Any, adapter: Any) -> None:
        """作答段里没有工具调用：这一段就是最终答案（按 E1 的 parse_generation 先判未闭合、截断、空回答）。"""
        view = SimpleNamespace(thinking_unclosed=res.thinking_end == THINKING_UNCLOSED, stopped=res.stopped, text=res.text)
        kind = ANSWER_KIND[self.task]
        w, h = self.rec.image_size
        if kind == "mask":
            p = parse_generation(view, _text_is_not_an_answer)
            failure = FAILURE_NOT_SUBMITTED if p.failure is None else p.failure.value
            self.rec.answer = {"source": "text", "kind": kind}
            self._finish("final_answer", failure)
            return
        if kind == "box":
            p = parse_generation(view, adapter.parse_one_box, width=w, height=h)
            if p.failure is None:
                self.rec.box = [float(v) for v in p.box]
            self.rec.answer = {"source": "text", "kind": kind, **({"box": list(self.rec.box)} if p.failure is None else {})}
        else:
            p = parse_generation(view, parse_count)
            if p.failure is None:
                self.rec.count = int(p.value)
            self.rec.answer = {"source": "text", "kind": kind, **({"count": self.rec.count} if p.failure is None else {})}
        self._finish("final_answer", None if p.failure is None else p.failure.value)

    # -- 一轮 ---------------------------------------------------------------

    def tool_specs(self) -> list[Any]:
        return list(self.wb.specs())

    def add_round(self, r: int, res: Any, batch_rows: int) -> None:
        pixels = getattr(res, "n_image_pixels", None)
        if pixels is None:
            if res.n_visual_tokens:
                raise RuntimeError(f"{self.sample.id} 第 {r} 轮：结果有 {res.n_visual_tokens} 个视觉 token，却没有 n_image_pixels"
                                   "（ChatTurnResult 要记模型实际看到的像素数，D28-3；见 src/w1/loop.py 模块说明）")
            pixels = 0
        self.calls.append(Compute.call(generated_tokens=int(res.n_generated_tokens),
                                       prompt_tokens=int(res.n_prompt_tokens) + int(res.n_inserted_tokens),
                                       visual_tokens=int(res.n_visual_tokens), image_pixels=int(pixels),
                                       call_time_s=float(res.wall_time_s), new_round=True))
        self.rec.round_walls.append(float(res.wall_time_s))
        self.rec.tool_times.append([])
        self.rec.rounds.append({
            "round": r, "batch_rows": batch_rows, "thinking": res.thinking, "text": res.text, "thinking_end": res.thinking_end,
            "stopped": res.stopped, "n_prompt_tokens": int(res.n_prompt_tokens), "n_generated_tokens": int(res.n_generated_tokens),
            "n_inserted_tokens": int(res.n_inserted_tokens), "n_visual_tokens": int(res.n_visual_tokens),
            "n_images": int(res.n_images), "n_image_pixels": int(pixels), "tool_calls": [],
            "tool_call_errors": [str(e) for e in res.tool_call_errors]})

    def step(self, r: int, res: Any, batch_rows: int, adapter: Any) -> None:
        """处理这一轮的结果：记录、回填历史、执行工具调用、判结束。"""
        self.add_round(r, res, batch_rows)
        calls, errors = list(res.tool_calls), list(res.tool_call_errors)
        # 回填（D104）：作答原文（含模型自己写的 <tool_call> 块）+ 思考段，不给 tool_calls，最贴近模型的原样输出
        msg = _text_msg("assistant", res.text)
        msg["reasoning_content"] = res.thinking
        self.messages.append(msg)
        if not calls and not errors:
            self.finish_final(res, adapter)
            return
        rnd = self.rec.rounds[-1]
        p = self.protocol
        submitted_by: str | None = None
        for c in calls:
            entry: dict[str, Any] = {"name": c.name, "arguments": dict(c.arguments), "raw": c.raw, "counted": False,
                                     "executed": False, "refused": None, "result": None}
            rnd["tool_calls"].append(entry)
            if submitted_by is not None:
                entry["refused"] = REFUSED_AFTER_SUBMISSION
                self.rec.n_refused += 1
                continue
            counts = not (p.submit_exempt_from_cap and c.name in p.submit_tools)
            if counts and self.rec.n_tool_calls >= p.max_tool_calls:
                entry["refused"] = REFUSED_TOOL_CAP
                self.rec.n_refused += 1
                self.messages.append(_tool_msg(self.prompts.tool_cap))
                continue
            t0 = time.perf_counter()
            out = self.wb.call(c.name, dict(c.arguments))
            self.rec.tool_times[-1].append(time.perf_counter() - t0)
            self.rec.n_executed += 1
            if counts:
                self.rec.n_tool_calls += 1
            entry.update(counted=counts, executed=True,
                         result={"text": out.text, "error": out.error, "images": [image_digest(im) for im in out.images]})
            self.messages.append(_tool_msg(out.text, out.images))
            if self.wb.submission is not None:
                submitted_by = c.name
        if submitted_by is not None:
            self.finish_submission(submitted_by)
            return
        for e in errors:
            self.rec.n_tool_call_errors += 1
            self.messages.append(_tool_msg(self.prompts.tool_call_error.format(error=e)))
        if r >= p.max_rounds:
            self._finish("round_cap", FAILURE_NO_ANSWER)

    def reject_input(self, r: int, err: Exception) -> None:
        """拼批前的预检拒收这条对话（D104）：不进这一轮的批，记 failure = input_rejected 与原因。"""
        self.rec.input_error = {"round": r, "error": f"{type(err).__name__}: {err}"}
        self._finish("input_rejected", FAILURE_INPUT_REJECTED)

    def add_last_round_notice(self, text: str) -> None:
        last = self.messages[-1]
        if last["role"] != "tool":
            raise RuntimeError(f"{self.sample.id}：进入最后一轮之前，最后一条消息应是 tool 消息，收到 {last['role']}")
        last["content"].append({"type": "text", "text": text})
        self.rec.rounds[-1]["last_round_notice"] = True


def _text_is_not_an_answer(text: str) -> ParseResult:
    """refseg 的最终答案文字不是答案（mask 只能经 submit_mask 提交）。只为走 E1 parse_generation 的同一套判定顺序（思考段未闭合 →
    截断 → 空回答）：文字非空时返回 failure = None，调用方记 not_submitted。"""
    return ParseResult(None, None if text.strip() else ParseFailure.EMPTY)


# ---------------------------------------------------------------------------
# 按轮同步
# ---------------------------------------------------------------------------


def default_workbench_factory() -> Callable[..., Any]:
    """真工作台（DEV-0050，src/w1/workbench.Workbench）；按需 import。"""
    from src.w1.workbench import Workbench

    return Workbench


def run_episodes(adapter: Any, samples: Sequence[Sample], task: str, protocol: Protocol, prompts: LoopPrompts, *,
                 batch_size: int, max_batch_tokens: int | None = None, workbench_factory: Callable[..., Any] | None = None,
                 tool_workers: int = 1, expected_tools_sha256: str | None = None, images: Sequence[Any] | None = None,
                 log: Callable[[str], None] | None = None) -> list[EpisodeRecord]:
    """samples 一起按轮同步地跑完（规则见模块说明），返回与 samples 同序的 EpisodeRecord。

    每一轮：还没结束的样本按原顺序逐条预检（adapter.chat_inputs；拒收的记 input_rejected，不进批，D104），再按 plan_sub_batches
    拆子批（行数 ≤ batch_size，且 行数 × 子批里最长的提示 ≤ max_batch_tokens），逐个子批调 chat_turn_batch、执行工具调用。
    workbench_factory(image, disabled_groups=…) 构造每条样本自己的工作台（默认 DEV-0050 的 Workbench；工作台拿原图的副本）。
    expected_tools_sha256 给了时，每条样本的工具清单（tools_digest(specs())）必须等于它（运行身份里记的那一份）。
    images 给了时按它取原图（与 samples 同序、每条一张 PIL 图），否则用 E1 的 load_images。"""
    if task not in TASKS:
        raise ValueError(f"未知任务 {task!r}；可用 {TASKS}")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError(f"batch_size 必须是正整数，收到 {batch_size!r}")
    if max_batch_tokens is not None and (isinstance(max_batch_tokens, bool) or not isinstance(max_batch_tokens, int)
                                         or max_batch_tokens < 1):
        raise ValueError(f"max_batch_tokens 必须是正整数或 None，收到 {max_batch_tokens!r}")
    if isinstance(tool_workers, bool) or not isinstance(tool_workers, int) or tool_workers < 1:
        raise ValueError(f"tool_workers 必须是正整数，收到 {tool_workers!r}")
    factory = workbench_factory or default_workbench_factory()
    if images is None:
        from src.e1.runner import load_images

        loaded = []
        for s in samples:
            ims = load_images(s)
            if len(ims) != 1:
                raise ValueError(f"{s.id}：W1 的任务都是单图，收到 {len(ims)} 张")
            loaded.append(ims[0])
        images = loaded
    if len(images) != len(samples):
        raise ValueError("images 与 samples 条数不同")
    eps: list[_Episode] = []
    for s, im in zip(samples, images):
        wb = factory(im.copy(), disabled_groups=protocol.disabled_groups)
        sha = tools_digest(wb.specs())
        if expected_tools_sha256 is not None and sha != expected_tools_sha256:
            raise RuntimeError(f"{s.id}：工具清单的 sha256 {sha[:12]}… 与运行身份里的 {expected_tools_sha256[:12]}… 不同")
        eps.append(_Episode(s, im, wb, task, protocol, prompts, sha))
    pool = ThreadPoolExecutor(max_workers=tool_workers) if tool_workers > 1 else None
    try:
        for r in range(1, protocol.max_rounds + 1):
            active = [e for e in eps if not e.done]
            if not active:
                break
            if r == protocol.max_rounds and r > 1 and prompts.last_round:
                for e in active:
                    e.add_last_round_notice(prompts.last_round)
            ready, widths, specs = [], [], []
            for e in active:  # 预检（D104）：有结构性 token 等，chat_turn_batch 会让整批报错；这一条单独记失败
                sp = e.tool_specs()
                try:
                    ci = adapter.chat_inputs(e.messages, tools=sp)
                except ValueError as err:  # ChatMessageError 是 ValueError 的子类
                    e.reject_input(r, err)
                    continue
                ready.append(e)
                widths.append(len(ci.row.ids))
                specs.append(sp)
            for part_idx in plan_sub_batches(widths, batch_size, max_batch_tokens):
                part = [ready[i] for i in part_idx]
                results = adapter.chat_turn_batch([list(e.messages) for e in part], tools=[specs[i] for i in part_idx],
                                                  thinking_budget=protocol.thinking_budget,
                                                  answer_budget=protocol.answer_budget)
                if len(results) != len(part):
                    raise RuntimeError(f"chat_turn_batch 返回 {len(results)} 行，输入 {len(part)} 行")
                jobs = list(zip(part, results))
                if pool is None:
                    for e, res in jobs:
                        e.step(r, res, len(part), adapter)
                else:
                    for f in [pool.submit(e.step, r, res, len(part), adapter) for e, res in jobs]:
                        f.result()
            if log is not None:
                log(f"[W1 {task}] 第 {r} 轮：{len(active)} 条在跑（预检拒收 {len(active) - len(ready)}），"
                    f"{sum(1 for e in eps if e.done)}/{len(eps)} 条已结束")
    finally:
        if pool is not None:
            pool.shutdown(wait=True)
    left = [e.sample.id for e in eps if not e.done]
    if left:  # 不会发生：最后一轮的 step 总会结束每一条
        raise RuntimeError(f"回路结束时还有 {len(left)} 条没有结束：{left[:5]}")
    return [e.rec for e in eps]


__all__ = [
    "ANSWER_KIND",
    "ENDED_BY",
    "EpisodeRecord",
    "FAILURE_INPUT_REJECTED",
    "FAILURE_NOT_SUBMITTED",
    "FAILURE_NO_ANSWER",
    "FAILURE_WRONG_SUBMISSION",
    "LoopPrompts",
    "Protocol",
    "REFUSED_AFTER_SUBMISSION",
    "REFUSED_TOOL_CAP",
    "SUBMISSION_FIELDS",
    "TASKS",
    "default_workbench_factory",
    "plan_sub_batches",
    "image_digest",
    "run_episodes",
    "tools_digest",
]
