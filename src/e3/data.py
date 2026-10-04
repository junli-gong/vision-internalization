"""E3b 的训练样本（DEV-0025 §2）：训练池、无状态的样本日程、修正对、C3 的文本目标。只调用 src/data 的加载器，不改它们。

**训练池**（硬约束 7：没有去重剔除清单就报错，由 src/data/train_loaders 负责，这里不绕过；开跑前另核对剔除清单比较过本任务的
评测集，check_dedupe）：
- refseg = refcoco_train 里 tasks.refseg.datasets（默认 RefCOCO、RefCOCO+、RefCOCOg）的 train 句子，按数据集依次拼接；指代句去掉
  首尾空白后为空的剔除并计数；
- depth = nyu_depth_train（BTS 清单，按场景过滤）。
- 其余任务：配置的 mix.weights 里加任务名，并在这里与 src/e3/targets 补上加载与目标；没有实现的任务报错（ADE20K 的类别网格
  要 src/constants.ADE_NUM_CLASSES 个类别 token，属于后续任务）。

**样本日程**（无状态：第 k 个样本 k = 步 × batch_size + 批内下标，一切都由 (种子, k) 算出，续跑时按步号重算）：
1. 任务：mix.weights 展开成固定的循环（mix_cycle，平滑加权轮转），k 落在循环的第 k mod L 个槽；该任务内的序号
   j = (k // L) · w_任务 + 这个槽在循环里是该任务的第几个。
2. 样本：第 j 个样本取第 j // N 个 epoch 的置换（np.random.default_rng([种子, 任务号, epoch]).permutation(N)）的第 j mod N 个
   （与 E2 的 batch_indices 同一规则）。
3. 每条样本的随机数一律由 (种子, 样本 id, epoch, 用途) 决定（sample_rng）：是否做成修正对（kind）、严重度（severity）、加噪
   （corrupt）、C3 选框还是多边形（c3_kind）、点对的抽样键。同一条样本在不同 epoch 里拿到不同的修正对。
C3 与 E3b 用同一个种子时，每一步的任务、样本 id、epoch 逐个相同（匹配曝光量）；只有目标不同。

**修正对**（supervision = grid，概率 correction.fraction）：
- 分割：GT mask → mask_noise.corrupt(mask, s, rng, others)（s 按 sample_severity 的噪声日程；others 是同一张图上其他被指代的物体）
  → mask_to_grid → feedback_image("refseg", 原图, 带错的网格)。
- 深度：GT 的档号网格 → depth_noise.corrupt_depth_grid → feedback_image("depth", 原图, 带错的网格)。
- 输入图 = [原图, 叠加图]，指令用 correction 模板（写明"第二张图是候选结果，输出修正后的完整网格"），目标是干净网格。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from PIL import Image

from src.data import train_dedupe as TD
from src.data import train_loaders as TL
from src.data.records import LoadResult, Sample
from src.e3 import c3 as C3
from src.e3.config import E3Config
from src.e3.depth_noise import corrupt_depth_grid
from src.e3.feedback import feedback_image
from src.e3.targets import (
    NoSupervision,
    depth_target,
    seg_grid,
    seg_target,
    target_from_grid,
)
from src.e3.tokens import GRID_TASKS, GridTokens
from src.noising.mask_noise import corrupt, sample_severity

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_REFSEG_ROOT = REPO_ROOT / "tests" / "fixtures" / "e3" / "root"
FIXTURE_DEPTH_ROOT = REPO_ROOT / "tests" / "fixtures" / "e2" / "root"
# --tiny 的两个训练池共用 E3 fixture 的剔除清单（eval_sets 同 RUN-0026，含 dedupe.required_eval_sets；DEV-0032）。E2 fixture 的清单只比较过
# E2 的评测集，按 E3b 的要求会挡住深度池；E2 的 fixture 不动（E2 的运行身份不变）
FIXTURE_EXCLUSIONS = FIXTURE_REFSEG_ROOT / TD.EXCLUSIONS_REL
TASK_INDEX = {t: i for i, t in enumerate(GRID_TASKS)}
KINDS: tuple[str, ...] = ("grid", "correction", "text")


@dataclass(frozen=True)
class E3Paths:
    """refseg_root / depth_root：DEV-0022 的落地根（含 dedupe/train_exclusions.json）；真实运行两者相同（--data-root），
    --tiny 时分别是 tests/fixtures/e3/root（RefCOCO 家族，图够大，能切 32×32）与 tests/fixtures/e2/root（NYU 深度），剔除清单两者都用
    FIXTURE_EXCLUSIONS。"""

    refseg_root: Path
    depth_root: Path
    exclusions: str | Path | None = "auto"
    require_dedupe: bool = True
    strict: bool = True

    @classmethod
    def single_root(cls, root: str | Path) -> "E3Paths":
        return cls(refseg_root=Path(root), depth_root=Path(root))

    @classmethod
    def fixtures(cls) -> "E3Paths":
        return cls(refseg_root=FIXTURE_REFSEG_ROOT, depth_root=FIXTURE_DEPTH_ROOT, exclusions=FIXTURE_EXCLUSIONS)

    def root_for(self, task: str) -> Path:
        if task == "refseg":
            return Path(self.refseg_root)
        if task == "depth":
            return Path(self.depth_root)
        raise ValueError(f"未知任务 {task!r}")

    def as_dict(self) -> dict[str, Any]:
        return {"refseg_root": str(self.refseg_root), "depth_root": str(self.depth_root), "exclusions": str(self.exclusions),
                "require_dedupe": self.require_dedupe, "strict": self.strict}


# ---------------------------------------------------------------------------
# 训练池与去重核对
# ---------------------------------------------------------------------------


def load_pool(task: str, paths: E3Paths, task_cfg: Mapping[str, Any]) -> LoadResult:
    kw = {"exclusions": paths.exclusions, "require_dedupe": paths.require_dedupe, "strict": paths.strict}
    root = paths.root_for(task)
    if task == "refseg":
        recs: list[Sample] = []
        stats: dict[str, Any] = {"set": "refcoco_train/train", "datasets": {}}
        n_empty = 0
        for ds in task_cfg["datasets"]:
            res = TL.load_refcoco_train(root, ds, **kw)
            keep = [s for s in res.records if str(s.fields.get("expression", "")).strip()]
            n_empty += len(res.records) - len(keep)
            recs += keep
            stats["datasets"][ds] = {k: v for k, v in res.stats.items() if k != "dedupe_excluded"}
            stats["dedupe_file"] = res.stats.get("dedupe_file")
        stats.update(n_kept=len(recs), n_empty_expression=n_empty)
        return LoadResult(recs, stats)
    if task == "depth":
        res = TL.load_nyu_depth_train(root, **kw)
        return LoadResult(res.records, {k: v for k, v in res.stats.items() if k != "dedupe_excluded"}
                          | {"n_dedupe_excluded": res.stats.get("n_dedupe_excluded")})
    raise ValueError(f"任务 {task!r} 还没有训练池（有：{GRID_TASKS}）")


def check_dedupe(task: str, dedupe_file: str | Path | None, required_eval_sets: Sequence[str], *,
                 real_model: bool) -> dict[str, Any]:
    """开跑前的断言：训练池用的剔除清单比较过 required_eval_sets（与 src/e2/data.check_dedupe_coverage 同一规则）。
    没有清单：真模型报错；旧格式（没有 eval_sets）：真模型报错，--tiny 照读；新格式：缺一个就报错。返回写进 config.yaml 的摘要。"""
    need = list(required_eval_sets)
    if dedupe_file is None:
        if real_model:
            raise ValueError(f"{task}：训练池没有用去重剔除清单，真模型不跑（硬约束 7）")
        return {"file_sha256": None, "schema": None, "legacy": None, "max_hamming": None, "eval_sets": None,
                "required_eval_sets": need}
    meta = TD.exclusions_meta(dedupe_file)
    out = {"file_sha256": meta["sha256"], "schema": meta["schema"], "legacy": meta["legacy"],
           "max_hamming": meta["max_hamming"], "eval_sets": meta["eval_sets"], "required_eval_sets": need}
    if meta["legacy"]:
        if real_model:
            raise ValueError(f"{task}：剔除清单 {dedupe_file} 是旧格式（schema {meta['schema']}），无法核实它比较过评测集 {need}")
        return out
    missing = [s for s in need if s not in meta["eval_sets"]]
    if missing:
        raise ValueError(f"{task}：评测集 {missing} 不在剔除清单 {dedupe_file} 的 eval_sets 里（{meta['eval_sets']}）；"
                         "把它们的哈希清单一起给 scripts/data/dedupe_report.py 重新比较")
    return out


def ids_digest(ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# 无状态的样本日程
# ---------------------------------------------------------------------------


def mix_cycle(weights: Mapping[str, int]) -> tuple[str, ...]:
    """整数权重 → 长度 Σw 的固定循环（平滑加权轮转：每轮各任务加上自己的权重，取当前值最大者（平局按 GRID_TASKS 的顺序），
    被取者减去总权重）。权重为 0 的任务不出现。"""
    items = [(t, int(weights[t])) for t in GRID_TASKS if int(weights.get(t, 0)) > 0]
    extra = sorted(set(weights) - set(GRID_TASKS))
    if extra:
        raise ValueError(f"mix.weights 里有未知任务 {extra}")
    total = sum(w for _, w in items)
    if total < 1:
        raise ValueError("mix.weights 至少要有一个正权重")
    cur = {t: 0 for t, _ in items}
    out = []
    for _ in range(total):
        for t, w in items:
            cur[t] += w
        pick = max(items, key=lambda tw: (cur[tw[0]], -TASK_INDEX[tw[0]]))[0]
        cur[pick] -= total
        out.append(pick)
    return tuple(out)


def slot_task(k: int, cycle: Sequence[str]) -> tuple[str, int]:
    """全局下标 k → (任务, 该任务内的序号 j)。"""
    if k < 0:
        raise ValueError("k 必须 ≥ 0")
    L = len(cycle)
    pos = k % L
    task = cycle[pos]
    per = sum(1 for t in cycle if t == task)
    rank = sum(1 for t in cycle[:pos] if t == task)
    return task, (k // L) * per + rank


def pool_position(j: int, n: int, *, seed: int, task: str) -> tuple[int, int]:
    """任务内的第 j 个样本 → (训练池下标, epoch)。"""
    if n < 1:
        raise ValueError(f"{task}：训练池为空")
    ep = j // n
    perm = np.random.default_rng([int(seed), TASK_INDEX[task], ep]).permutation(n)
    return int(perm[j % n]), ep


def sample_rng(seed: int, sample_id: str, epoch: int, purpose: str) -> np.random.Generator:
    """一条样本在一个 epoch 里某个用途的随机数发生器（与 numpy 版本无关的种子：sha256 的前 4 个 32 位字）。"""
    h = hashlib.sha256(f"visint-e3|{purpose}|{sample_id}".encode("utf-8")).digest()
    words = [int.from_bytes(h[4 * i: 4 * i + 4], "big") for i in range(4)]
    return np.random.default_rng([int(seed), int(epoch), *words])


@dataclass(frozen=True)
class SlotPlan:
    """日程里的一格：第 k 个样本（第 step 步）。"""

    k: int
    step: int
    task: str
    j: int
    index: int
    epoch: int
    sample_id: str
    kind: str
    text_kind: str | None = None


class Schedule:
    def __init__(self, pools: Mapping[str, Sequence[Sample]], cfg: E3Config, *, seed: int, supervision: str,
                 batch_size: int) -> None:
        self.pools = {t: list(v) for t, v in pools.items()}
        self.cycle = mix_cycle(cfg.mix_weights)
        missing = [t for t in set(self.cycle) if t not in self.pools or not self.pools[t]]
        if missing:
            raise ValueError(f"任务 {missing} 的训练池为空或没有加载")
        self.seed = int(seed)
        self.supervision = supervision
        self.batch_size = int(batch_size)
        self.fraction = float(cfg.doc["correction"]["fraction"])
        kinds = cfg.doc["c3"]["seg_kinds"]
        self.c3_kinds = [(k, int(kinds[k])) for k in ("box", "polygon") if int(kinds.get(k, 0)) > 0]

    def plan(self, k: int) -> SlotPlan:
        task, j = slot_task(k, self.cycle)
        pool = self.pools[task]
        idx, ep = pool_position(j, len(pool), seed=self.seed, task=task)
        sid = pool[idx].id
        text_kind = None
        if self.supervision == "grid":
            kind = "correction" if sample_rng(self.seed, sid, ep, "kind").random() < self.fraction else "grid"
        else:
            kind = "text"
            if task == "refseg":
                total = sum(w for _, w in self.c3_kinds)
                u = sample_rng(self.seed, sid, ep, "c3_kind").random() * total
                acc = 0.0
                text_kind = self.c3_kinds[-1][0]
                for name, w in self.c3_kinds:
                    acc += w
                    if u < acc:
                        text_kind = name
                        break
            else:
                text_kind = "pairs"
        return SlotPlan(k=k, step=k // self.batch_size, task=task, j=j, index=idx, epoch=ep, sample_id=sid, kind=kind,
                        text_kind=text_kind)

    def step_plans(self, step: int) -> list[SlotPlan]:
        return [self.plan(step * self.batch_size + i) for i in range(self.batch_size)]


# ---------------------------------------------------------------------------
# 一条训练样本
# ---------------------------------------------------------------------------


@dataclass
class Example:
    """送进 build_inputs 的图与指令，以及目标 token（含停止 token）与逐 token 的损失权重（0 / 1）。"""

    plan: SlotPlan
    images: list[np.ndarray]
    instruction: str
    order: str
    target_ids: list[int]
    weights: list[int]
    target_text: str | None = None
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def n_supervised(self) -> int:
        return int(sum(self.weights))


def load_rgb(path: str | Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8).copy()


def refseg_mask(sample: Sample) -> np.ndarray:
    """GT mask（与 E1 同一套解码：src/e1/scoring.gt_mask）。"""
    from src.e1.scoring import gt_mask

    return gt_mask(sample)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    u = int(np.logical_or(a, b).sum())
    return 1.0 if u == 0 else float(np.logical_and(a, b).sum()) / u


class ExampleBuilder:
    """SlotPlan → Example。grid 监督要给 tokens（网格词表的 id 布局）；text 监督要给 encode（适配器 tokenizer 的无特殊 token 编码）、
    family（取 E1 的多边形统一提示与本家族的定位提示）与 format_boxes（适配器的 format_boxes：C3 的框目标用本家族原生格式，
    DEV-0030 §3）。"""

    def __init__(self, cfg: E3Config, pools: Mapping[str, Sequence[Sample]], *, seed: int, supervision: str, stop_id: int,
                 tokens: GridTokens | None = None, encode: Callable[[str], list[int]] | None = None,
                 family: str | None = None, format_boxes: Callable[..., str] | None = None) -> None:
        self.cfg = cfg
        self.doc = cfg.doc
        self.pools = {t: list(v) for t, v in pools.items()}
        self.seed = int(seed)
        self.supervision = supervision
        self.tokens = tokens
        self.encode = encode
        self.format_boxes = format_boxes
        self.stop_id = int(stop_id)
        self.n = cfg.grid_n
        if supervision == "grid" and tokens is None:
            raise ValueError("网格监督要给 tokens")
        if supervision == "text":
            if encode is None or family is None or format_boxes is None:
                raise ValueError("文本监督（C3）要给 encode、family 与 format_boxes")
            self.polygon_template, self.polygon_source = C3.load_polygon_template(self.doc, family, REPO_ROOT)
            self.box_template, self.box_source = C3.load_box_template(self.doc, family, REPO_ROOT)
        self._others = self._index_others(self.pools.get("refseg", []))

    @staticmethod
    def _index_others(pool: Sequence[Sample]) -> dict[Any, dict[int, Mapping[str, Any]]]:
        """image_id → {ann_id: mask 的 GT 描述}（RefCOCO 家族的 ann 是 COCO 的标注，跨数据集同一个 id 同一个 mask）。"""
        out: dict[Any, dict[int, Mapping[str, Any]]] = {}
        for s in pool:
            out.setdefault(s.fields["image_id"], {}).setdefault(int(s.fields["ann_id"]), s.gt["mask"])
        return out

    def _merge_others(self, sample: Sample, shape: tuple[int, int]) -> list[np.ndarray]:
        from src.e1.scoring import gt_mask

        cands = self._others.get(sample.fields["image_id"], {})
        mine = int(sample.fields["ann_id"])
        keep = [a for a in sorted(cands) if a != mine][: int(self.doc["tasks"]["refseg"]["merge_candidates_max"])]
        out = []
        for a in keep:
            m = gt_mask(Sample(id=f"ann:{a}", task="refexp", images=(), image_keys=(), fields={}, gt={"mask": cands[a]},
                               source="", split=""))
            if m.shape == shape:
                out.append(m)
        return out

    def build(self, plan: SlotPlan) -> Example:
        sample = self.pools[plan.task][plan.index]
        if sample.id != plan.sample_id:
            raise RuntimeError(f"日程与训练池对不上：{plan.sample_id} ≠ {sample.id}")
        image = load_rgb(sample.images[0])
        if plan.kind == "text":
            return self._text(plan, sample, image)
        if plan.task == "refseg":
            return self._refseg(plan, sample, image)
        if plan.task == "depth":
            return self._depth(plan, sample, image)
        raise ValueError(f"任务 {plan.task!r} 没有样本构造")

    # ----- 网格 -----
    def _refseg(self, plan: SlotPlan, sample: Sample, image: np.ndarray) -> Example:
        tc = self.doc["tasks"]["refseg"]
        mask = refseg_mask(sample)
        if mask.shape != image.shape[:2]:
            raise ValueError(f"{sample.id}：mask {mask.shape} 与图像 {image.shape[:2]} 尺寸不同")
        tgt = seg_target(mask, self.tokens, n=self.n)
        expr = str(sample.fields["expression"]).strip()
        info: dict[str, Any] = {"n_fg_cells": tgt.info["n_fg_cells"]}
        if plan.kind == "grid":
            return Example(plan, [image], tc["grid_template"].format(expression=expr, n=self.n), tc["order"],
                           list(tgt.ids), list(tgt.weights), info=info)
        s = sample_severity(sample_rng(self.seed, sample.id, plan.epoch, "severity"))
        noisy, rec = corrupt(mask, s, sample_rng(self.seed, sample.id, plan.epoch, "corrupt"),
                             others=self._merge_others(sample, mask.shape))
        g_noisy = seg_grid(noisy, n=self.n)
        overlay = feedback_image("refseg", image, g_noisy)
        info.update(severity=s, noise=rec, candidate_cell_iou=_iou(g_noisy.astype(bool), tgt.grid.astype(bool)),
                    n_changed_cells=int((g_noisy != tgt.grid).sum()))
        return Example(plan, [image, overlay], tc["correction_template"].format(expression=expr, n=self.n), tc["order"],
                       list(tgt.ids), list(tgt.weights), info=info)

    def _depth(self, plan: SlotPlan, sample: Sample, image: np.ndarray) -> Example:
        tc = self.doc["tasks"]["depth"]
        depth = TL.read_depth_m(sample.gt["depth"])
        if depth.shape != image.shape[:2]:
            raise ValueError(f"{sample.id}：深度 {depth.shape} 与图像 {image.shape[:2]} 尺寸不同")
        tgt = depth_target(depth, self.tokens, min_valid_fraction=float(tc["cell"]["min_valid_fraction"]), n=self.n)
        info: dict[str, Any] = dict(tgt.info)
        if plan.kind == "grid":
            return Example(plan, [image], tc["grid_instruction"].format(n=self.n), tc["order"], list(tgt.ids),
                           list(tgt.weights), info=info)
        s = sample_severity(sample_rng(self.seed, sample.id, plan.epoch, "severity"))
        noisy, rec = corrupt_depth_grid(tgt.grid, s, sample_rng(self.seed, sample.id, plan.epoch, "corrupt"),
                                        self.doc["depth_noise"])
        overlay = feedback_image("depth", image, noisy)
        info.update(severity=s, noise=rec, n_changed_cells=int((noisy != tgt.grid).sum()),
                    mean_abs_bin_error=float(np.abs(noisy - tgt.grid).mean()))
        target = target_from_grid("depth", tgt.grid, tgt.supervised, self.tokens, n=self.n)
        return Example(plan, [image, overlay], tc["correction_instruction"].format(n=self.n), tc["order"],
                       list(target.ids), list(target.weights), info=info)

    # ----- C3 -----
    def _text(self, plan: SlotPlan, sample: Sample, image: np.ndarray) -> Example:
        c = self.doc["c3"]
        h, w = image.shape[:2]
        info: dict[str, Any] = {"text_kind": plan.text_kind}
        if plan.task == "refseg":
            tc = self.doc["tasks"]["refseg"]
            expr = str(sample.fields["expression"]).strip()
            text = None
            kind = plan.text_kind
            if kind == "polygon":
                polys = sample.gt["mask"].get("polygons")
                text = C3.polygon_text(polys, w, h, max_points=int(c["polygon_max_points"])) if polys else None
                if text is None:
                    kind = "box"
                    info["fallback"] = "rle_or_degenerate_polygon"
                else:
                    instruction = self.polygon_template.format(expression=expr)
            if kind == "box":
                text = C3.box_target(self.format_boxes, sample.gt["box"], w, h, expr)
                instruction = C3.render_e1_template(self.box_template, expr)
            info["text_kind_used"] = kind
            order = tc["order"]
        elif plan.task == "depth":
            tc = self.doc["tasks"]["depth"]
            depth = TL.read_depth_m(sample.gt["depth"])
            pairs = C3.depth_pairs(depth, key=f"{sample.id}#e{plan.epoch}", n_pairs=int(c["depth_pairs_per_image"]),
                                   seed=self.seed)
            if not pairs:
                raise NoSupervision(f"{sample.id}：抽不到满足约束的点对")
            instruction = C3.depth_pairs_prompt(c["depth_template"], c["depth_pair_line"], pairs, w, h)
            text = C3.depth_pairs_answer(c["depth_answer_line"], pairs)
            info["n_pairs"] = len(pairs)
            order = tc["order"]
        else:
            raise ValueError(f"任务 {plan.task!r} 没有 C3 目标")
        ids = list(self.encode(text)) + [self.stop_id]
        return Example(plan, [image], instruction, order, ids, [1] * len(ids), target_text=text, info=info)


__all__ = [
    "E3Paths",
    "Example",
    "ExampleBuilder",
    "KINDS",
    "Schedule",
    "SlotPlan",
    "TASK_INDEX",
    "check_dedupe",
    "ids_digest",
    "load_pool",
    "load_rgb",
    "mix_cycle",
    "pool_position",
    "refseg_mask",
    "sample_rng",
    "slot_task",
]
