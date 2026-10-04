"""网格格式预热（DEV-0035 §2；D36-5，N5 第 2 处）：E5 主比较里"基座 + 网格格式预热"那一臂的训练数据、留出集与停止规则。

**任务**（纯符号，不给图像）：用文字给出要涂的格子坐标（分割）或若干区域的深度档号（深度），目标是对应的 32×32 网格 token 序列，
与 E3b 网格监督同一套 token、同一种格式（src/e3/targets.grid_to_ids：光栅顺序、每行之后一个行分隔、最后接本家族的回合结束 token，
长度 1057）。输入只有文字（build_inputs 的图像列表为空），没有任何视觉内容。措辞全部写在 configs/e3b_train.json 的 warmup 节：
- 坐标：行与列都从左上角起编号 0..n−1；区间写成 range（"{lo}-{hi}"，一行 / 一列时也写成 "5-5"，格式统一）。
- 分割（refseg），三种（refseg.kinds 的整数权重）：
  - rect：一个矩形，"Rows 3-5, columns 10-12 are foreground."；高、宽在 [1, n] 均匀取，左上角在放得下的位置里均匀取；
  - rects：refseg.rects_count 个矩形（含两端，均匀取；可重叠，取并集），每个的高、宽在 [1, rects_max_side] 均匀取，
    "Foreground cells: rows 3-5, columns 10-12; rows 20-24, columns 0-7."；
  - shape：一个简单形状（refseg.shapes 里均匀取：disk = 欧氏距离 ≤ s、diamond = 曼哈顿距离 ≤ s、triangle = 顶点在 (r0, c0)、
    往下 s 行、第 i 行左右各伸 i 格），s 在 refseg.shape_size 均匀取，中心在全图均匀取、出界的格子裁掉；逐行列出，
    "Foreground cells, row by row: row 7: columns 3-9; row 8: columns 2-10."（一行有多段时用 run_joiner 连接；这三种形状每行一段）。
  模板 refseg.template 再把这句话包成整条指令（说明网格的编号，结尾 "Output the {n}x{n} mask grid." 与 E3b 的网格指令同一句）。
- 深度（depth）：背景档号在 [0, 63] 均匀取；depth.regions_count 个矩形区域（高、宽在 [1, n] 均匀取），各自的档号在 [0, 63] 均匀取，
  后面的区域覆盖前面的；"Every cell is at level 40 except rows 3-5, columns 10-12 at level 12; rows 20-31, columns 0-31 at level 5."
  档号与 src/grid/codec.depth_to_bins 同一方向（0 = 最近，63 = 最远），64 档 = src/constants.DEPTH_NUM_BINS。
- 混合：warmup.mix 的整数权重（默认分割 : 深度 = 1 : 1），与 E3b 同一个平滑加权轮转（src/e3/data.mix_cycle、slot_task），第 k 个
  样本的任务由 k 算出。

**随机数**（与任何视觉数据无关）：训练样本 k 的随机数发生器是 np.random.default_rng([data_seed, 0, 运行种子, k, attempt])；留出集
第 i 条是 default_rng([data_seed, 1, heldout.seed, i])，与运行种子无关（三个种子的预热用同一个留出集）。训练样本的指令与留出集的
某一条逐字相同时 attempt + 1 重抽（确定的），所以留出集不进训练。

**停止规则**（warmup.stop）：每 eval_every 步（第 s 步做完之后，s 是 eval_every 的倍数）在留出集上做受限解码（adapter.generate_restricted，
允许集合 = tokens.allowed_ids(任务, allowed_scope)，max_new_tokens = 目标序列长度 1057，停止 token = 本家族的回合结束 token，与
E3b 的单轮评测同一个设置），统计格式错误 = targets.parse_target 解析失败（没有以停止 token 结尾、长度不对、行分隔的位置不对即行数
或列数不对、格子里出现本任务以外的 token）。格式错误率 = 格式错误的条数 / 留出集条数（两个任务合计；另按任务分列）。连续
consecutive 次低于 threshold（严格小于）就停，记下步数；max_steps 是步数上限（学习率的余弦按它排），到了上限还没满足就照样存检查点，
train_summary.json 记 stop_rule_met = false。另报（不作判据）：与目标网格逐格相同的比例、整格全对的比例。

本模块只造样本、留出集与读数；训练循环（src/e3/train.py）负责在规定的步数上调 evaluate_heldout 并按 StopRule 停。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from src.constants import DEPTH_NUM_BINS
from src.e3.data import Example, SlotPlan, mix_cycle, slot_task
from src.e3.targets import grid_to_ids, parse_target, seq_len
from src.e3.tokens import GRID_TASKS, GridTokens

SEG_KINDS: tuple[str, ...] = ("rect", "rects", "shape")
SHAPES: tuple[str, ...] = ("disk", "diamond", "triangle")
DEPTH_KIND = "regions"
TRAIN_STREAM, HELDOUT_STREAM = 0, 1
MAX_ATTEMPTS = 100  # 训练样本与留出集撞上时重抽的上限（撞上的概率很小；到上限说明配置坏了）


class WarmupConfigError(ValueError):
    pass


def _pair(v: Any, name: str, lo: int, hi: int) -> tuple[int, int]:
    if not isinstance(v, (list, tuple)) or len(v) != 2 or any(isinstance(x, bool) or int(x) != x for x in v):
        raise WarmupConfigError(f"warmup.{name} 必须是两个整数 [lo, hi]，收到 {v!r}")
    a, b = int(v[0]), int(v[1])
    if not lo <= a <= b <= hi:
        raise WarmupConfigError(f"warmup.{name} = {v} 必须满足 {lo} ≤ lo ≤ hi ≤ {hi}")
    return a, b


def _weights(w: Mapping[str, Any], allowed: Sequence[str], name: str) -> dict[str, int]:
    if not isinstance(w, Mapping) or set(w) - set(allowed) or any(isinstance(v, bool) or int(v) != v or int(v) < 0
                                                                   for v in w.values()) or sum(int(v) for v in w.values()) < 1:
        raise WarmupConfigError(f"warmup.{name} 必须是 {list(allowed)} 的非负整数权重、至少一个为正，收到 {w!r}")
    return {k: int(w[k]) for k in allowed if int(w.get(k, 0)) > 0}


def _check_template(t: Any, keys: Sequence[str], name: str) -> str:
    """模板必须恰好用到这些占位符（各至少一次），且能用它们 format。"""
    import string

    if not isinstance(t, str) or not t.strip():
        raise WarmupConfigError(f"warmup.{name} 必须是非空字符串")
    used = {f for _, f, _, _ in string.Formatter().parse(t) if f is not None}
    if used != set(keys):
        raise WarmupConfigError(f"warmup.{name} 的占位符 {sorted(used)} 应恰好是 {sorted(keys)}")
    return t


@dataclass(frozen=True)
class WarmupSpec:
    """configs/e3b_train.json 的 warmup 节（结构核对在 from_doc）。"""

    doc: Mapping[str, Any]
    n: int
    mix: Mapping[str, int]
    seg_kinds: Mapping[str, int]
    rects_count: tuple[int, int]
    rects_max_side: int
    shapes: tuple[str, ...]
    shape_size: tuple[int, int]
    depth_regions: tuple[int, int]
    data_seed: int
    heldout_n: int
    heldout_seed: int
    stop: Mapping[str, Any]

    @classmethod
    def from_doc(cls, doc: Mapping[str, Any], *, n: int, tiny: bool = False) -> "WarmupSpec":
        w = doc["warmup"]
        mix = _weights(w["mix"], GRID_TASKS, "mix")
        r, d = w["refseg"], w["depth"]
        kinds = _weights(r["kinds"], SEG_KINDS, "refseg.kinds")
        shapes = tuple(r["shapes"])
        if not shapes or set(shapes) - set(SHAPES) or len(set(shapes)) != len(shapes):
            raise WarmupConfigError(f"warmup.refseg.shapes 必须是 {SHAPES} 里不重复的非空子集，收到 {shapes}")
        side = int(r["rects_max_side"])
        if not 1 <= side <= n:
            raise WarmupConfigError(f"warmup.refseg.rects_max_side 必须在 [1, {n}]")
        for name, keys in (("template", ("n", "last", "cells")), ("rect", ("rows", "cols")), ("rects", ("regions",)),
                           ("region", ("rows", "cols")), ("shape", ("rows",)), ("shape_row", ("row", "cols"))):
            _check_template(r[name], keys, f"refseg.{name}")
        _check_template(d["template"], ("n", "last", "max_level", "background", "regions"), "depth.template")
        _check_template(d["region"], ("rows", "cols", "level"), "depth.region")
        _check_template(w["range"], ("lo", "hi"), "range")
        for k in ("joiner", "run_joiner"):
            if not isinstance(w[k], str) or not w[k]:
                raise WarmupConfigError(f"warmup.{k} 必须是非空字符串")
        st = dict(w["stop"])
        if tiny:
            st.update(max_steps=int(w["tiny"]["max_steps"]), eval_every=int(w["tiny"]["eval_every"]))
        if not (int(st["eval_every"]) >= 1 and int(st["consecutive"]) >= 1 and int(st["max_steps"]) >= 1
                and 0.0 < float(st["threshold"]) <= 1.0):
            raise WarmupConfigError(f"warmup.stop 不合规：{st}")
        h = w["heldout"]
        hn = int(w["tiny"]["heldout_n"]) if tiny else int(h["n"])
        if hn < 1:
            raise WarmupConfigError("留出集至少 1 条")
        return cls(doc=w, n=int(n), mix=mix, seg_kinds=kinds, rects_count=_pair(r["rects_count"], "refseg.rects_count", 1, 64),
                   rects_max_side=side, shapes=shapes, shape_size=_pair(r["shape_size"], "refseg.shape_size", 0, n),
                   depth_regions=_pair(d["regions_count"], "depth.regions_count", 1, 64), data_seed=int(w["data_seed"]),
                   heldout_n=hn, heldout_seed=int(h["seed"]),
                   stop={"eval_every": int(st["eval_every"]), "threshold": float(st["threshold"]),
                         "consecutive": int(st["consecutive"]), "max_steps": int(st["max_steps"])})

    def identity(self) -> dict[str, Any]:
        """写进运行身份的部分（配置的 warmup 节去掉 basis 与 tiny，加上本次生效的停止规则与留出集条数）。"""
        def strip(x: Any) -> Any:
            if isinstance(x, Mapping):
                return {k: strip(v) for k, v in x.items() if k not in ("basis", "tiny")}
            return x

        return {**strip(self.doc), "effective": {"stop": dict(self.stop), "heldout_n": self.heldout_n, "grid_size": self.n}}


# ---------------------------------------------------------------------------
# 一条样本：网格 + 文字
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WarmupItem:
    task: str
    kind: str
    grid: np.ndarray
    instruction: str
    info: dict[str, Any] = field(default_factory=dict)


def _rng(*key: int) -> np.random.Generator:
    return np.random.default_rng([int(k) for k in key])


def _pick(rng: np.random.Generator, weights: Mapping[str, int]) -> str:
    """按整数权重选一个（整数运算，与平台的浮点无关）：u 在 [0, Σw) 均匀取，落在哪一段就是哪一个（按 weights 的顺序）。"""
    u = int(rng.integers(0, sum(weights.values())))
    for k, w in weights.items():
        if u < w:
            return k
        u -= w
    raise AssertionError("不可达")


def _range(spec: WarmupSpec, lo: int, hi: int) -> str:
    return spec.doc["range"].format(lo=int(lo), hi=int(hi))


def _rect(rng: np.random.Generator, n: int, max_side: int) -> tuple[int, int, int, int]:
    """(r0, r1, c0, c1)，含两端：高、宽在 [1, max_side] 均匀取，左上角在放得下的位置里均匀取。"""
    h, w = int(rng.integers(1, max_side + 1)), int(rng.integers(1, max_side + 1))
    r0, c0 = int(rng.integers(0, n - h + 1)), int(rng.integers(0, n - w + 1))
    return r0, r0 + h - 1, c0, c0 + w - 1


def _shape_mask(shape: str, n: int, r0: int, c0: int, s: int) -> np.ndarray:
    rr, cc = np.mgrid[0:n, 0:n]
    dr, dc = rr - r0, cc - c0
    if shape == "disk":
        m = dr * dr + dc * dc <= s * s
    elif shape == "diamond":
        m = np.abs(dr) + np.abs(dc) <= s
    elif shape == "triangle":
        m = (dr >= 0) & (dr <= s) & (np.abs(dc) <= dr)
    else:
        raise ValueError(f"未知形状 {shape!r}")
    return m.astype(np.int64)


def _runs(row: np.ndarray) -> list[tuple[int, int]]:
    out, start = [], None
    for c, v in enumerate(list(row) + [0]):
        if v and start is None:
            start = c
        elif not v and start is not None:
            out.append((start, c - 1))
            start = None
    return out


def seg_item(spec: WarmupSpec, rng: np.random.Generator) -> WarmupItem:
    n, r = spec.n, spec.doc["refseg"]
    kind = _pick(rng, spec.seg_kinds)
    g = np.zeros((n, n), dtype=np.int64)
    info: dict[str, Any] = {}
    if kind == "rect":
        r0, r1, c0, c1 = _rect(rng, n, n)
        g[r0:r1 + 1, c0:c1 + 1] = 1
        cells = r["rect"].format(rows=_range(spec, r0, r1), cols=_range(spec, c0, c1))
        info["rects"] = [[r0, r1, c0, c1]]
    elif kind == "rects":
        k = int(rng.integers(spec.rects_count[0], spec.rects_count[1] + 1))
        rects = [_rect(rng, n, spec.rects_max_side) for _ in range(k)]
        for r0, r1, c0, c1 in rects:
            g[r0:r1 + 1, c0:c1 + 1] = 1
        cells = r["rects"].format(regions=spec.doc["joiner"].join(
            r["region"].format(rows=_range(spec, r0, r1), cols=_range(spec, c0, c1)) for r0, r1, c0, c1 in rects))
        info["rects"] = [list(x) for x in rects]
    else:
        shape = spec.shapes[int(rng.integers(0, len(spec.shapes)))]
        s = int(rng.integers(spec.shape_size[0], spec.shape_size[1] + 1))
        r0, c0 = int(rng.integers(0, n)), int(rng.integers(0, n))
        g = _shape_mask(shape, n, r0, c0, s)
        rows = []
        for i in range(n):
            runs = _runs(g[i])
            if runs:
                rows.append(r["shape_row"].format(row=i, cols=spec.doc["run_joiner"].join(_range(spec, a, b) for a, b in runs)))
        cells = r["shape"].format(rows=spec.doc["joiner"].join(rows))
        info.update(shape=shape, size=s, center=[r0, c0])
    text = r["template"].format(n=n, last=n - 1, cells=cells)
    info["n_fg_cells"] = int(g.sum())
    return WarmupItem("refseg", kind, g, text, info)


def depth_item(spec: WarmupSpec, rng: np.random.Generator) -> WarmupItem:
    n, d = spec.n, spec.doc["depth"]
    bg = int(rng.integers(0, DEPTH_NUM_BINS))
    g = np.full((n, n), bg, dtype=np.int64)
    k = int(rng.integers(spec.depth_regions[0], spec.depth_regions[1] + 1))
    regions = []
    for _ in range(k):
        r0, r1, c0, c1 = _rect(rng, n, n)
        lv = int(rng.integers(0, DEPTH_NUM_BINS))
        g[r0:r1 + 1, c0:c1 + 1] = lv
        regions.append((r0, r1, c0, c1, lv))
    text = d["template"].format(n=n, last=n - 1, max_level=DEPTH_NUM_BINS - 1, background=bg,
                                regions=spec.doc["joiner"].join(
                                    d["region"].format(rows=_range(spec, r0, r1), cols=_range(spec, c0, c1), level=lv)
                                    for r0, r1, c0, c1, lv in regions))
    return WarmupItem("depth", DEPTH_KIND, g, text, {"background": bg, "regions": [list(x) for x in regions]})


def make_item(spec: WarmupSpec, task: str, rng: np.random.Generator) -> WarmupItem:
    if task == "refseg":
        return seg_item(spec, rng)
    if task == "depth":
        return depth_item(spec, rng)
    raise ValueError(f"预热没有任务 {task!r}（有：{GRID_TASKS}）")


def heldout_items(spec: WarmupSpec) -> list[WarmupItem]:
    """固定的留出集：第 i 条的任务按 warmup.mix 的轮转，随机数只由 (data_seed, 1, heldout.seed, i) 决定。"""
    cycle = mix_cycle(spec.mix)
    return [make_item(spec, slot_task(i, cycle)[0], _rng(spec.data_seed, HELDOUT_STREAM, spec.heldout_seed, i))
            for i in range(spec.heldout_n)]


def items_digest(items: Sequence[WarmupItem]) -> str:
    """留出集的指纹：每条的任务、指令与目标网格。"""
    h = hashlib.sha256()
    for it in items:
        h.update(f"{it.task}\n{it.instruction}\n".encode("utf-8"))
        h.update(np.ascontiguousarray(it.grid, dtype=np.int64).tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# 训练样本的日程与构造（与 src/e3/data 的 Schedule / ExampleBuilder 同一个接口：step_plans(step)、build(plan)）
# ---------------------------------------------------------------------------


class WarmupSource:
    """训练样本：第 k 个样本 k = 步 × batch_size + 批内下标，一切由 (data_seed, 运行种子, k) 算出（无状态，续跑按步号重算）。"""

    def __init__(self, spec: WarmupSpec, *, seed: int, tokens: GridTokens, batch_size: int, orders: Mapping[str, str]) -> None:
        """orders：任务 → build_inputs 的 order（取 E3b 配置的 tasks.<任务>.order；没有图像时只影响模板，照 E3b 同一个值）。"""
        self.spec = spec
        self.seed = int(seed)
        self.tokens = tokens
        self.batch_size = int(batch_size)
        self.orders = {t: str(orders[t]) for t in spec.mix}
        self.cycle = mix_cycle(spec.mix)
        self.heldout = heldout_items(spec)
        self._heldout_texts = {it.instruction for it in self.heldout}

    def item(self, k: int) -> tuple[WarmupItem, int]:
        task, _ = slot_task(k, self.cycle)
        for attempt in range(MAX_ATTEMPTS):
            it = make_item(self.spec, task, _rng(self.spec.data_seed, TRAIN_STREAM, self.seed, k, attempt))
            if it.instruction not in self._heldout_texts:
                return it, attempt
        raise RuntimeError(f"第 {k} 个预热样本重抽 {MAX_ATTEMPTS} 次都与留出集撞上：配置的随机范围太窄")

    def plan(self, k: int) -> SlotPlan:
        task, j = slot_task(k, self.cycle)
        return SlotPlan(k=k, step=k // self.batch_size, task=task, j=j, index=k, epoch=0,
                        sample_id=f"warmup/seed{self.seed}/k{k}", kind="warmup")

    def step_plans(self, step: int) -> list[SlotPlan]:
        return [self.plan(step * self.batch_size + i) for i in range(self.batch_size)]

    def build(self, plan: SlotPlan) -> Example:
        it, attempt = self.item(plan.k)
        if it.task != plan.task:
            raise RuntimeError(f"日程与样本对不上：{plan.task} ≠ {it.task}")
        ids = target_ids(it, self.tokens, self.spec.n)
        plan = SlotPlan(k=plan.k, step=plan.step, task=plan.task, j=plan.j, index=plan.index, epoch=0,
                        sample_id=plan.sample_id, kind=it.kind)
        info = {**it.info, "attempt": attempt}
        return Example(plan, [], it.instruction, self.orders[it.task], ids, [1] * len(ids), info=info)


def target_ids(it: WarmupItem, tokens: GridTokens, n: int) -> list[int]:
    return grid_to_ids(it.grid, tokens.value_ids(it.task), tokens.row_id, tokens.stop_id, n=n)


# ---------------------------------------------------------------------------
# 留出集上的格式错误率与停止规则
# ---------------------------------------------------------------------------


def evaluate_heldout(ad: Any, tokens: GridTokens, items: Sequence[WarmupItem], *, n: int, scope: str,
                     orders: Mapping[str, str], enable_thinking: bool, batch_size: int) -> dict[str, Any]:
    """留出集上的受限解码与格式错误（规则见模块说明）。按任务分组、组内按留出集的顺序连续成批。adapter 要在 eval 模式。"""
    fails: dict[str, int] = {}
    by_task: dict[str, dict[str, Any]] = {}
    gen_tokens = 0
    for task in GRID_TASKS:
        idx = [i for i, it in enumerate(items) if it.task == task]
        if not idx:
            continue
        allowed = tokens.allowed_ids(task, scope)
        nf = n_exact = 0
        cell_hits = 0
        for s in range(0, len(idx), batch_size):
            chunk = idx[s: s + batch_size]
            inputs = [ad.build_inputs([], items[i].instruction, orders[task]) for i in chunk]
            res = ad.generate_restricted(inputs, allowed, max_new_tokens=seq_len(n), stop_ids=[int(tokens.stop_id)],
                                         enable_thinking=enable_thinking)
            if len(res) != len(chunk):
                raise RuntimeError("generate_restricted 返回的行数与输入不同")
            for i, r in zip(chunk, res):
                gen_tokens += int(r.n_generated_tokens)
                p = parse_target(task, r.token_ids, tokens, n=n)
                if p.failure is not None:
                    nf += 1
                    fails[p.failure.value] = fails.get(p.failure.value, 0) + 1
                    continue
                hit = int((p.grid == items[i].grid).sum())
                cell_hits += hit
                n_exact += int(hit == n * n)
        m = len(idx)
        by_task[task] = {"n": m, "n_format_errors": nf, "format_error_rate": nf / m, "exact_rate": n_exact / m,
                         "cell_accuracy": cell_hits / (m * n * n)}
    total = len(items)
    nfe = sum(v["n_format_errors"] for v in by_task.values())
    return {"n": total, "n_format_errors": nfe, "format_error_rate": nfe / total if total else None,
            "failures": dict(sorted(fails.items())), "by_task": by_task, "generated_tokens": gen_tokens,
            "note": "by_task 的 cell_accuracy、exact_rate 把格式错误的条记 0，只作诊断（停止规则只看 format_error_rate）"}


class StopRule:
    """连续 consecutive 次格式错误率 < threshold 就停（模块说明）。"""

    def __init__(self, *, threshold: float, consecutive: int) -> None:
        self.threshold = float(threshold)
        self.consecutive = int(consecutive)

    def below(self, rate: float | None) -> bool:
        return rate is not None and float(rate) < self.threshold

    def met(self, history: Sequence[Mapping[str, Any]]) -> bool:
        """history：按步数排好的读数（每条有 format_error_rate）。"""
        tail = list(history)[-self.consecutive:]
        return len(tail) == self.consecutive and all(self.below(h["format_error_rate"]) for h in tail)


EvalFn = Callable[[Any, GridTokens, Sequence[WarmupItem]], Mapping[str, Any]]


__all__ = [
    "DEPTH_KIND",
    "EvalFn",
    "SEG_KINDS",
    "SHAPES",
    "StopRule",
    "WarmupConfigError",
    "WarmupItem",
    "WarmupSource",
    "WarmupSpec",
    "depth_item",
    "evaluate_heldout",
    "heldout_items",
    "items_digest",
    "make_item",
    "seg_item",
    "target_ids",
]
