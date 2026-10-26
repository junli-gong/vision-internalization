"""LLMSeg 的本地提供方（DEV-0066）：官方管线的每次视觉调用 → 我们的适配器（Qwen3.5，关思考、贪心），锁步合批。

注入（不改官方文件，install() 只替换模块属性，restore() 还原）：
- `llm._complete(provider, model, prompt, path, timeout)` ← LocalProvider.complete。这是官方离线测试自己用的接缝（tests/test_offline.py
  把它换成脚本化的假模型）；官方 `llm.query` 照常负责计时、日志与"空回复报错"，之后的解析、换算、光栅化全是官方代码。
- `llm._call_gemini`、`llm._call_anthropic`、`llm._call_openai_compatible` ← 一律抛 ExternalCallForbidden（BaseException 的子类，官方
  `_complete` 的 `except Exception` 重试接不住它，管线的 except 也接不住）：即使有人设了 key，也到不了任何闭源 API。
- `llm.configure(provider="local", timeout=CALL_TIMEOUT_S)`：官方的公开函数；provider 名只进日志（`[llm:local:…]`），timeout 放大，
  因为合批时一次调用要等同一轮的别的请求（真正的看门狗在调度器里）。
- `llm.log`、`pipeline.log`、`geometry.log` ← LogRouter：每个管线线程的日志进它自己的 run.log（官方评测是每条一个子进程、stderr
  写进 work/<item>/run.log，这里同一个进程里多线程，按线程分流）。

坐标（管理员补充第 1 条）：官方 llm.py 按模型名决定线上的坐标约定，名字的最后一段以 qwen 开头 → 0–1000 网格、x 在前，提示里写明
"NORMALIZED 0-1000 grid"，回答由 normalize_boxes / rescale_spline_groups 换回像素（判据："所有数都 ≤ 1000 才按网格换算"）。我们传给管线
的模型名是注册键（qwen3.5-9b 等），check_coordinate_convention 入口核对它落在这条约定上；这与本家族适配器的原生约定相同（qwen35.parse_boxes：
0–1000、x 在前，x = x_norm / 1000 · W）。每次回复另做一次只读的坐标统计（coord_audit：坐标值个数、> 1000 的个数），不改官方的换算。

输出长度（管理员补充第 2 条）：官方的 OpenAI 兼容调用不传 max_tokens，本地生成必须自己给。按阶段给上限（answer_cap）：
    上限 = min(CAP_CEIL, 基数 + 每单位 × 单位数)，单位数 = 提示里以 "- id=" 开头的行数（阶段 2 = 实例数，3a = 实例数，3b = 焦点框数）。
每单位的数取官方提示自己的上限（阶段 2 每个轮廓至多 64 个控制点，3b 每框至多 12 个，3a 每实例至多 6 个框），按 Qwen 分词器
（数字逐位成 token）在缩进 2 的 JSON、一位小数下的最长合理回复留出余量（tests/test_llmseg_provider.py 用真分词器核对上限够用）。
截断（stopped = False）照实记录，回复原样交给官方代码（它会抢救截断回复里完整的条目）。

合批（锁步，确定）：每个管线线程跑一条样本；调度器等到"所有在跑的管线都在等模型"（或已跑完）时，把这一刻的全部请求作为一轮：
按 (阶段, 上限, 样本序号, 调用序号) 排序，按 (阶段, 上限) 分组，组内按 src.w1.loop.plan_sub_batches（行数 ≤ batch_size，行数 × 最长提示
≤ max_batch_tokens）切子批，逐个子批生成，每个子批生成完立刻把结果交回（那几条管线接着跑 CPU 部分，下一次请求进下一轮）。空出来的
槽位在轮与轮之间按样本顺序补新样本。批的组成只由样本顺序与管线逻辑决定，与线程的快慢无关，所以同一配置跑两遍逐字节相同。
同一组里每行的 max_new_tokens 相同（= 该行自己的上限），每行只看自己的 logits：合批与逐条（slots = batch_size = 1，走 generate）的结果
逐 token 相同（贪心；CPU fp32 上由测试核对；GPU bf16 上批的组成可能带来数值噪声，同 E1 的 D36-1）。
"""

from __future__ import annotations

import contextlib
import hashlib
import itertools
import json
import re
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

from PIL import Image

from src.w1.loop import plan_sub_batches

LOCAL_PROVIDER = "local"
CALL_TIMEOUT_S = 7 * 24 * 3600.0  # 官方 query 的单次超时（放大；不让排队被当成超时）
WATCHDOG_S = 3 * 3600.0  # 调度器：一轮之内迟迟等不到全部管线就报错（防死锁）

# ---------------------------------------------------------------------------
# 阶段：按官方提示模板里各自独有的一句话判定（测试核对每句恰好只出现在一个模板里）
# ---------------------------------------------------------------------------

STAGE_DETECT = "detect"
STAGE_SEGMENT = "segment"
STAGE_BOX_ADJUST = "box_adjust"
STAGE_FOCUS = "focus_boxes"
STAGE_FOCUS_SPLINES = "focus_splines"
STAGES: tuple[str, ...] = (STAGE_DETECT, STAGE_SEGMENT, STAGE_BOX_ADJUST, STAGE_FOCUS, STAGE_FOCUS_SPLINES)
STAGE2_STAGES: tuple[str, ...] = (STAGE_DETECT, STAGE_SEGMENT)  # 修正前（官方 seg_*）用到的调用
NEW_ROUND_STAGES: tuple[str, ...] = (STAGE_DETECT, STAGE_SEGMENT, STAGE_BOX_ADJUST, STAGE_FOCUS)  # 3b 与 3a 同一轮
STAGE_TEMPLATES: Mapping[str, str] = {STAGE_DETECT: "PROMPT_DETECT", STAGE_SEGMENT: "PROMPT_SEGMENT",
                                      STAGE_BOX_ADJUST: "PROMPT_BOX_ADJUST", STAGE_FOCUS: "PROMPT_FOCUS",
                                      STAGE_FOCUS_SPLINES: "PROMPT_FOCUS_SPLINE"}
STAGE_MARKERS: Mapping[str, str] = {
    STAGE_DETECT: "Find every DISTINCT instance of the following in the image",
    STAGE_SEGMENT: "We have already DETECTED",
    STAGE_BOX_ADJUST: "Return a SIGNED side delta",
    STAGE_FOCUS: "Find FOCUS BOXES that mark where the mask is wrong",
    STAGE_FOCUS_SPLINES: "FOCUS BOX(ES). For EACH box, trace the OUTLINE",
}
UNIT_LINE = "- id="  # 官方 format_instances 与 focus_splines 每个实例 / 焦点框一行，都以它开头

# 各阶段的输出上限：(基数, 每单位)。依据见模块说明；tests/test_llmseg_provider.py 用真分词器核对最长合理回复不超过上限。
ANSWER_CAP_RULE: Mapping[str, tuple[int, int]] = {
    STAGE_DETECT: (2048, 0),          # 约 10 个检测（每个含形态描述，缩进 JSON 约 170 token）
    STAGE_SEGMENT: (256, 1792),       # 每实例一个 64 点的轮廓（一位小数、缩进 JSON 约 1,600 token）
    STAGE_BOX_ADJUST: (256, 128),     # 每实例 4 个带符号的数（主配置不用）
    STAGE_FOCUS: (256, 640),          # 每实例至多 6 个焦点框（缩进 JSON 约 480 token）
    STAGE_FOCUS_SPLINES: (256, 384),  # 每框一个 12 点的轮廓（约 350 token）
}
CAP_CEIL = 16384

COORD_BOX_KEYS: tuple[str, ...] = ("box_2d", "bbox_2d", "box", "bbox", "box2d")  # 同官方 llm._BOX_KEYS
DELTA_KEYS: tuple[str, ...] = ("x1", "x2", "y1", "y2")
GRID_MAX = 1000.0  # 官方 llm.NORMALIZED_GRID（check_coordinate_convention 核对）


class LocalProviderError(Exception):
    """本地提供方的基础设施错误（生成报错、调度器异常、请求对不上样本）。不是 RuntimeError / ValueError / TimeoutError 的子类，
    官方管线的 except 接不住它，整次运行停下来，而不是被当成"这条没分割出来"按空 mask 计分。"""


class ExternalCallForbidden(BaseException):
    """官方的闭源 API 调用被禁用（D125-4：不用任何闭源 API）。BaseException：官方 _complete 的 except Exception 重试接不住。"""


def stage_of(prompt: str) -> str:
    hits = [s for s, m in STAGE_MARKERS.items() if m in prompt]
    if len(hits) != 1:
        raise LocalProviderError(f"认不出提示属于哪个阶段（命中 {hits}）：{prompt[:120]!r}…")
    return hits[0]


def n_units(prompt: str) -> int:
    return sum(1 for line in prompt.splitlines() if line.startswith(UNIT_LINE))


def answer_cap(stage: str, prompt: str, *, cap_limit: int | None = None) -> int:
    """这次调用的 max_new_tokens（规则见模块说明）。cap_limit（只给 --tiny 彩排）再取一次 min。"""
    base, per = ANSWER_CAP_RULE[stage]
    cap = min(CAP_CEIL, base + per * n_units(prompt))
    return cap if cap_limit is None else min(cap, int(cap_limit))


def check_coordinate_convention(llm: Any, model_name: str) -> dict[str, Any]:
    """传给官方管线的模型名必须落在"0–1000 网格、x 在前"的约定上（官方 llm.py 的 Qwen-VL 一支），与本家族适配器的原生约定相同。"""
    if float(llm.NORMALIZED_GRID) != GRID_MAX:
        raise LocalProviderError(f"官方 NORMALIZED_GRID = {llm.NORMALIZED_GRID}，不是 {GRID_MAX}")
    if not llm.is_normalized_grid_model(model_name) or llm.is_yfirst_box_model(model_name):
        raise LocalProviderError(f"模型名 {model_name!r} 在官方 llm.py 里不是 0–1000 网格、x 在前的约定（Qwen-VL 一支）")
    if llm.box_field_name(model_name) != "box" or llm.box_order_literal(model_name) != "[x1, y1, x2, y2]":
        raise LocalProviderError(f"模型名 {model_name!r} 的框键 / 顺序不是 box、[x1, y1, x2, y2]")
    return {"model_name": model_name, "grid": "0-1000", "box_order": "[x1, y1, x2, y2]（x 在前）", "points": "[x, y]",
            "converted_by": "官方 llm.normalize_boxes / rescale_spline_groups（判据：全部数 ≤ 1000 才按网格换算）",
            "adapter_native": "src/adapters/qwen35.native_box_to_xyxy：x = x_norm / 1000 · W（同一换算，测试逐值核对）"}


# ---------------------------------------------------------------------------
# 坐标统计（只读，不改官方的换算）
# ---------------------------------------------------------------------------


def _num(v: Any) -> float | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return float(v)


def _collect(obj: Any, stage: str, out: list[float]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in COORD_BOX_KEYS and isinstance(v, (list, tuple)):
                out.extend(x for x in (_num(t) for t in list(v)[:4]) if x is not None)
            elif k == "control_points" and isinstance(v, (list, tuple)):
                for p in v:
                    if isinstance(p, (list, tuple)):
                        out.extend(x for x in (_num(t) for t in list(p)[:2]) if x is not None)
            elif stage == STAGE_BOX_ADJUST and k in DELTA_KEYS:
                x = _num(v)
                if x is not None:
                    out.append(x)
            else:
                _collect(v, stage, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _collect(v, stage, out)


def coord_audit(llm: Any, text: str, stage: str) -> dict[str, Any]:
    """一次回复里的坐标值：严格解析（官方 extract_json）成功记 json_ok；失败时用官方的结构修复（parse_repaired_json）再数。
    n_values / n_gt_1000（绝对值 > 1000，官方会把这一组当像素、不换算）/ max_abs；两种解析都失败时坐标记 None。"""
    json_ok = True
    try:
        payload = llm.extract_json(text)
    except ValueError:
        json_ok = False
        payload = llm.parse_repaired_json(text)
    if payload is None:
        return {"json_ok": False, "parsed": False, "n_values": None, "n_gt_1000": None, "max_abs": None}
    vals: list[float] = []
    _collect(payload, stage, vals)
    return {"json_ok": json_ok, "parsed": True, "n_values": len(vals), "n_gt_1000": sum(1 for v in vals if abs(v) > GRID_MAX),
            "max_abs": max((abs(v) for v in vals), default=None)}


# ---------------------------------------------------------------------------
# 生成后端（适配器）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GenerationConfig:
    """本地生成的协议（进运行身份）。主配置：关思考，E1 的 greedy 解码参数，图在前；thinking_budget 只配 enable_thinking。"""

    decoding: Mapping[str, Any]
    order: str = "image_first"
    enable_thinking: bool = False
    thinking_budget: int | None = None
    batch_size: int = 16
    max_batch_tokens: int | None = 48000
    cap_limit: int | None = None

    def __post_init__(self) -> None:
        if self.enable_thinking != (self.thinking_budget is not None):
            raise ValueError("thinking_budget 只配 enable_thinking = True，且此时必须给")
        if self.decoding.get("do_sample") is not False:
            raise ValueError(f"只用贪心解码（do_sample = False），收到 {dict(self.decoding)}")
        for name in ("batch_size", "max_batch_tokens", "cap_limit", "thinking_budget"):
            v = getattr(self, name)
            if v is not None and (isinstance(v, bool) or not isinstance(v, int) or v < 1):
                raise ValueError(f"{name} 必须是正整数，收到 {v!r}")

    def identity(self) -> dict[str, Any]:
        return {"enable_thinking": self.enable_thinking, "thinking_budget": self.thinking_budget, "decoding": dict(self.decoding),
                "order": self.order, "answer_cap_rule": {k: list(v) for k, v in ANSWER_CAP_RULE.items()}, "cap_ceil": CAP_CEIL,
                "cap_limit": self.cap_limit,
                "method": ("generate_batch_budgeted" if self.enable_thinking else
                           ("generate" if self.batch_size == 1 else "generate_batch"))}


class AdapterBackend:
    """把请求交给适配器：build_inputs（一张图 + 官方提示，图在前）与 src.e1.runner.generate_rows（batch_size = 1 走 generate，
    否则 generate_batch；开思考时走 generate_batch_budgeted，思考预算 thinking_budget、作答预算 = 这一组的上限）。"""

    def __init__(self, adapter: Any, gen: GenerationConfig):
        self.adapter = adapter
        self.gen = gen

    def build(self, image: Image.Image, prompt: str) -> Any:
        return self.adapter.build_inputs([image], prompt, self.gen.order)

    @staticmethod
    def width(inputs: Any) -> int:
        return int(inputs.tensors["input_ids"].shape[1])

    def generate(self, inputs: Sequence[Any], cap: int) -> list[Any]:
        from src.e1.runner import generate_rows

        g = self.gen
        if g.enable_thinking:
            return generate_rows(self.adapter, list(inputs), batch_size=g.batch_size, enable_thinking=True,
                                 max_new_tokens=int(g.thinking_budget) + cap, decoding=g.decoding,
                                 think_budget=(int(g.thinking_budget), cap))
        return generate_rows(self.adapter, list(inputs), batch_size=g.batch_size, enable_thinking=False, max_new_tokens=cap,
                             decoding=g.decoding)


# ---------------------------------------------------------------------------
# 锁步调度器
# ---------------------------------------------------------------------------


@dataclass
class Request:
    item_index: int
    seq: int
    stage: str
    cap: int
    prompt: str
    image: Image.Image
    t_submit: float = 0.0
    inputs: Any = None
    result: Any = None
    error: BaseException | None = None
    info: dict[str, Any] = field(default_factory=dict)
    done: threading.Event = field(default_factory=threading.Event)

    def sort_key(self) -> tuple:
        return (STAGES.index(self.stage), self.cap, self.item_index, self.seq)


class LockstepScheduler:
    """规则见模块说明。run(items, work) 用 slots 个槽位跑 work(item)（每条一个线程），work 里的模型调用经 submit 进来。"""

    def __init__(self, backend: AdapterBackend, *, slots: int, watchdog_s: float = WATCHDOG_S,
                 on_batch: Callable[[dict[str, Any]], None] | None = None):
        if isinstance(slots, bool) or not isinstance(slots, int) or slots < 1:
            raise ValueError(f"slots 必须是正整数，收到 {slots!r}")
        self.backend = backend
        self.slots = slots
        self.watchdog_s = float(watchdog_s)
        self.on_batch = on_batch
        self._cv = threading.Condition()
        self._pending: list[Request] = []
        self._running = 0
        self._active = 0
        self._abort: BaseException | None = None
        self._in_run = False
        self.round = 0
        self.sub_batch = 0

    # 管线线程（经官方 query 的工作线程）调用
    def submit(self, req: Request) -> Any:
        with self._cv:
            if not self._in_run:
                raise LocalProviderError("调度器没有在跑（submit 只能在 run 的 work 里调用）")
            if self._abort is not None:
                raise LocalProviderError(f"运行已中止：{self._abort!r}")
            req.t_submit = time.perf_counter()
            self._pending.append(req)
            self._running -= 1
            self._cv.notify_all()
        req.done.wait()
        if req.error is not None:
            raise req.error
        return req.result

    def _deliver(self, reqs: Sequence[Request]) -> None:
        with self._cv:
            for r in reqs:
                self._running += 1
                r.done.set()
            self._cv.notify_all()

    def _fail(self, reqs: Sequence[Request], err: BaseException) -> None:
        for r in reqs:
            r.error = err
        self._deliver(reqs)

    def _serve(self, batch: list[Request]) -> None:
        """一轮：逐条 build_inputs（拒收的交回 RuntimeError，官方按这次调用失败处理），按 (阶段, 上限) 分组切子批，逐个生成并交回。"""
        ready: list[Request] = []
        for r in batch:
            try:
                r.inputs = self.backend.build(r.image, r.prompt)
            except (ValueError, TypeError) as e:
                r.info["input_rejected"] = f"{type(e).__name__}: {e}"
                self._fail([r], RuntimeError(f"local provider rejected the input: {e}"))
                continue
            ready.append(r)
        groups: dict[tuple[str, int], list[Request]] = {}
        for r in ready:
            groups.setdefault((r.stage, r.cap), []).append(r)
        todo = [(k, g) for k, g in sorted(groups.items(), key=lambda kv: (STAGES.index(kv[0][0]), kv[0][1]))]
        parts: list[tuple[str, int, list[Request]]] = []
        for (stage, cap), g in todo:
            widths = [self.backend.width(r.inputs) for r in g]
            for idx in plan_sub_batches(widths, self.backend.gen.batch_size, self.backend.gen.max_batch_tokens):
                parts.append((stage, cap, [g[i] for i in idx]))
        for k, (stage, cap, part) in enumerate(parts):
            t0 = time.perf_counter()
            try:
                res = self.backend.generate([r.inputs for r in part], cap)
                if len(res) != len(part):
                    raise RuntimeError(f"生成返回 {len(res)} 行，输入 {len(part)} 行")
            except BaseException as e:  # noqa: BLE001 — 基础设施错误：整次运行中止
                err = LocalProviderError(f"本地生成失败（第 {self.round} 轮，{stage}，{len(part)} 行）：{e!r}")
                err.__cause__ = e
                with self._cv:
                    self._abort = err
                rest = [r for _, _, p in parts[k:] for r in p]
                self._fail(rest, err)
                return
            wall = time.perf_counter() - t0
            self.sub_batch += 1
            info = {"round": self.round, "sub_batch": self.sub_batch, "stage": stage, "cap": cap, "rows": len(part),
                    "max_prompt": max(self.backend.width(r.inputs) for r in part),
                    "max_generated": max(int(x.n_generated_tokens) for x in res),
                    "n_truncated": sum(1 for x in res if x.stopped is False), "wall_s": wall}
            if self.on_batch is not None:
                self.on_batch(dict(info))
            for r, x in zip(part, res):
                r.result = x
                r.info.update(round=self.round, sub_batch=self.sub_batch, batch_rows=len(part), batch_wall_s=wall,
                              queue_s=t0 - r.t_submit)
            self._deliver(part)

    def run(self, items: Sequence[Any], work: Callable[[int, Any], Any]) -> list[Any]:
        """work(序号, item) 在各自的线程里跑；返回与 items 同序的结果。work 抛出的异常（应只有 LocalProviderError）在全部线程结束后
        重新抛出（中止优先）。"""
        n = len(items)
        results: list[Any] = [None] * n
        errors: dict[int, BaseException] = {}
        queue = deque(enumerate(items))
        threads: list[threading.Thread] = []

        def body(i: int, it: Any) -> None:
            try:
                results[i] = work(i, it)
            except BaseException as e:  # noqa: BLE001
                errors[i] = e
            finally:
                with self._cv:
                    self._active -= 1
                    self._running -= 1
                    self._cv.notify_all()

        with self._cv:
            if self._in_run:
                raise LocalProviderError("调度器不能嵌套 run")
            self._in_run, self._abort, self._pending, self._running, self._active = True, None, [], 0, 0
        watchdog_fired = False
        try:
            while True:
                batch: list[Request] = []
                with self._cv:
                    if not self._cv.wait_for(lambda: self._running == 0, timeout=self.watchdog_s):
                        if watchdog_fired:  # 中止之后仍有管线卡住：不再等
                            break
                        watchdog_fired = True
                        if self._abort is None:
                            self._abort = LocalProviderError(f"看门狗：{self.watchdog_s:.0f} s 内仍有 {self._running} 条管线没有交请求")
                        batch, self._pending = self._pending, []
                    else:
                        started = 0
                        while self._abort is None and self._active < self.slots and queue:
                            i, it = queue.popleft()
                            self._active += 1
                            self._running += 1
                            t = threading.Thread(target=body, args=(i, it), name=f"llmseg-slot-{i}", daemon=True)
                            threads.append(t)
                            t.start()
                            started += 1
                        if started:
                            continue
                        if not self._pending:
                            if self._active == 0:
                                break
                            if self._abort is None:
                                self._abort = LocalProviderError(f"调度器状态不一致：{self._active} 条管线在跑却没有请求")
                            break
                        batch = sorted(self._pending, key=Request.sort_key)
                        self._pending = []
                        self.round += 1
                    abort = self._abort
                if abort is not None:  # 中止之后进来的请求一律交回错误，等管线各自退出
                    self._fail(batch, abort if isinstance(abort, LocalProviderError) else LocalProviderError(repr(abort)))
                    continue
                self._serve(batch)
        finally:
            for t in threads:
                t.join(timeout=60.0)
            with self._cv:
                self._in_run = False
                abort = self._abort
        if abort is not None:
            raise abort
        if errors:
            raise errors[min(errors)]
        return results


# ---------------------------------------------------------------------------
# 日志分流
# ---------------------------------------------------------------------------


class LogRouter:
    """官方 log(msg) 的替身：当前线程在 capture 里就记进它的缓冲，否则照旧写 stderr。"""

    def __init__(self) -> None:
        self._tls = threading.local()

    def __call__(self, msg: str) -> None:
        buf = getattr(self._tls, "buf", None)
        if buf is None:
            print(msg, file=sys.stderr)
        else:
            buf.append(str(msg))

    @contextlib.contextmanager
    def capture(self, buf: list[str]) -> Iterator[list[str]]:
        old = getattr(self._tls, "buf", None)
        self._tls.buf = buf
        try:
            yield buf
        finally:
            self._tls.buf = old


# ---------------------------------------------------------------------------
# 提供方
# ---------------------------------------------------------------------------


@dataclass
class _Item:
    index: int
    work_dir: Path
    t0: float
    calls: list[dict[str, Any]] = field(default_factory=list)
    seq: Iterator[int] = field(default_factory=lambda: itertools.count(1))


class LocalProvider:
    """官方 `_complete` 的替身。每条样本先 register(key, 序号, 工作目录)：官方管线的每次调用都带一个图像路径（原图拷贝或叠加图），
    都在这条样本的工作目录里，据此认出是哪条样本（模块说明）。每次调用的记录（阶段、上限、token、截断、坐标统计、合批信息、回复原文）
    存在 calls_of(key)。"""

    def __init__(self, scheduler: LockstepScheduler, *, llm: Any, model_name: str):
        self.scheduler = scheduler
        self.llm = llm
        self.model_name = model_name
        self.gen = scheduler.backend.gen
        self._items: dict[Path, _Item] = {}
        self._lock = threading.Lock()

    def register(self, work_dir: str | Path, index: int) -> None:
        d = Path(work_dir).resolve()
        with self._lock:
            if d in self._items:
                raise LocalProviderError(f"工作目录 {d} 已登记")
            self._items[d] = _Item(index=int(index), work_dir=d, t0=time.perf_counter())

    def release(self, work_dir: str | Path) -> list[dict[str, Any]]:
        with self._lock:
            it = self._items.pop(Path(work_dir).resolve())
        return it.calls

    def _item_for(self, path: Path) -> _Item:
        with self._lock:
            it = self._items.get(path.resolve().parent)
        if it is None:
            raise LocalProviderError(f"图像 {path} 不在任何登记的工作目录里")
        return it

    def complete(self, provider: str, model: str, prompt: str, path: Any, timeout: float) -> str:
        if provider != LOCAL_PROVIDER:
            raise LocalProviderError(f"provider 应为 {LOCAL_PROVIDER!r}（llm.configure），收到 {provider!r}")
        if model != self.model_name:
            raise LocalProviderError(f"模型名 {model!r} 与本地提供方的 {self.model_name!r} 不同")
        p = Path(path)
        item = self._item_for(p)
        stage = stage_of(prompt)
        cap = answer_cap(stage, prompt, cap_limit=self.gen.cap_limit)
        with Image.open(p) as im:
            img = im.convert("RGB")
        seq = next(item.seq)
        req = Request(item_index=item.index, seq=seq, stage=stage, cap=cap, prompt=prompt, image=img)
        rec: dict[str, Any] = {"seq": seq, "stage": stage, "units": n_units(prompt), "cap": cap,
                               "image": p.name, "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest()}
        try:
            res = self.scheduler.submit(req)
        except RuntimeError as e:  # 拒收（build_inputs 报错）：官方按这次调用失败处理
            rec.update(input_rejected=req.info.get("input_rejected", str(e)), t_end_s=time.perf_counter() - item.t0)
            item.calls.append(rec)
            raise
        text = res.text
        rec.update(n_prompt_tokens=int(res.n_prompt_tokens), n_generated_tokens=int(res.n_generated_tokens),
                   n_inserted_tokens=int(getattr(res, "n_inserted_tokens", 0) or 0), n_visual_tokens=int(res.n_visual_tokens),
                   n_image_pixels=int(res.n_image_pixels or 0), stopped=res.stopped, truncated=res.stopped is False,
                   thinking_end=getattr(res, "thinking_end", None), empty_reply=not text,
                   coords=coord_audit(self.llm, text, stage) if text else None,
                   wall_s=float(res.wall_time_s), **{k: req.info[k] for k in ("round", "sub_batch", "batch_rows", "batch_wall_s",
                                                                              "queue_s") if k in req.info},
                   t_end_s=time.perf_counter() - item.t0, text=text)
        item.calls.append(rec)
        return text


# ---------------------------------------------------------------------------
# 注入
# ---------------------------------------------------------------------------

PATCHED_PROVIDER_CALLS: tuple[str, ...] = ("_call_gemini", "_call_anthropic", "_call_openai_compatible")


def _forbidden(name: str) -> Callable[..., Any]:
    def call(*_a: Any, **_k: Any) -> Any:
        raise ExternalCallForbidden(f"llm.{name}：闭源 API 调用已禁用（DEV-0066 / D125-4）")
    call.__name__ = f"forbidden_{name}"
    return call


@dataclass
class Installation:
    mods: Any
    saved: dict[tuple[str, str], Any]

    def restore(self) -> None:
        for (mod, attr), v in self.saved.items():
            setattr(getattr(self.mods, mod), attr, v)


def install(mods: Any, provider: LocalProvider | Callable[..., str], router: LogRouter | None = None) -> Installation:
    """把本地提供方注入官方模块（规则见模块说明），返回可 restore() 的记录。"""
    llm = mods.llm
    saved: dict[tuple[str, str], Any] = {("llm", "_complete"): llm._complete, ("llm", "_provider_override"): llm._provider_override,
                                         ("llm", "_timeout"): llm._timeout}
    for name in PATCHED_PROVIDER_CALLS:
        saved[("llm", name)] = getattr(llm, name)
        setattr(llm, name, _forbidden(name))
    llm._complete = provider.complete if isinstance(provider, LocalProvider) else provider
    llm.configure(provider=LOCAL_PROVIDER, timeout=CALL_TIMEOUT_S)
    if router is not None:
        for mod in ("llm", "pipeline", "geometry"):
            saved[(mod, "log")] = getattr(mods, mod).log
            getattr(mods, mod).log = router
    return Installation(mods, saved)


def calls_summary(calls: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """一条样本的调用记录 → predictions 里的计数（不含墙钟）：各阶段的调用数、token、截断、格式失败、坐标统计。"""
    by_stage = {s: 0 for s in STAGES}
    tok_stage = {s: {"prompt": 0, "generated": 0} for s in STAGES}
    trunc = {s: 0 for s in STAGES}
    fmt_fail = {s: 0 for s in STAGES}
    tokens = {"prompt": 0, "generated": 0, "inserted": 0, "visual": 0, "image_pixels": 0}
    coords = {"n_values": 0, "n_gt_1000": 0, "n_calls_gt_1000": 0, "n_calls_unparsed": 0, "max_abs": None}
    n_rej = n_empty = 0
    for c in calls:
        s = c["stage"]
        by_stage[s] += 1
        if "input_rejected" in c:
            n_rej += 1
            continue
        tokens["prompt"] += c["n_prompt_tokens"]
        tokens["generated"] += c["n_generated_tokens"]
        tokens["inserted"] += c["n_inserted_tokens"]
        tokens["visual"] += c["n_visual_tokens"]
        tokens["image_pixels"] += c["n_image_pixels"]
        tok_stage[s]["prompt"] += c["n_prompt_tokens"]
        tok_stage[s]["generated"] += c["n_generated_tokens"]
        trunc[s] += int(bool(c["truncated"]))
        n_empty += int(bool(c["empty_reply"]))
        a = c.get("coords")
        if a is None or not a["json_ok"]:
            fmt_fail[s] += 1
        if a is not None and a["parsed"]:
            coords["n_values"] += a["n_values"]
            coords["n_gt_1000"] += a["n_gt_1000"]
            coords["n_calls_gt_1000"] += int(a["n_gt_1000"] > 0)
            if a["max_abs"] is not None:
                coords["max_abs"] = a["max_abs"] if coords["max_abs"] is None else max(coords["max_abs"], a["max_abs"])
        elif a is not None:
            coords["n_calls_unparsed"] += 1
    return {"n_model_calls": len(calls), "calls_by_stage": by_stage, "tokens": tokens, "tokens_by_stage": tok_stage,
            "truncated_by_stage": trunc, "format_fail_by_stage": fmt_fail, "n_empty_reply": n_empty, "n_input_rejected": n_rej,
            "coords": coords}


def json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


__all__ = [
    "ANSWER_CAP_RULE",
    "AdapterBackend",
    "CALL_TIMEOUT_S",
    "CAP_CEIL",
    "ExternalCallForbidden",
    "GenerationConfig",
    "Installation",
    "LOCAL_PROVIDER",
    "LocalProvider",
    "LocalProviderError",
    "LockstepScheduler",
    "LogRouter",
    "NEW_ROUND_STAGES",
    "PATCHED_PROVIDER_CALLS",
    "Request",
    "STAGE2_STAGES",
    "STAGES",
    "STAGE_MARKERS",
    "STAGE_TEMPLATES",
    "answer_cap",
    "calls_summary",
    "check_coordinate_convention",
    "coord_audit",
    "install",
    "n_units",
    "stage_of",
]
