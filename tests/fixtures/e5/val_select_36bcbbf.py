"""E5 小档的 val 选检查点（DEV-0038 §1；D77-3；configs/e5_train.json 的 selection 一节，设置与理由写在那里的 basis）。

只对受步数下限影响的档（training.min_steps.tiers：1K、4K）。训练循环（src/e5/train.py）每 val_every 步（以及最后一步）调一次
ValSelector.evaluate，按规则算分数，留下分数最高那一步的检查点作为 final/。

**划分**：只用生成器的 val 划分（E5 评测的 dev 子集，src/e5/protocol.SUBSETS；与 test 按场景种子与物体身份不相交，D23-2）。
入口断言：每条样本生成之前核对划分是 val（guard_val；不是就抛 SplitGuardError），训练进程里不生成任何 test 样本（tests 钉住：
生成器的划分核对处记录全部调用，一次训练里没有 test）。test 只在选定的检查点上由 scripts/eval_e5_trained.py 评一次。

**样本**（selection_items）：每类 n_per_task 条，都落在标题口径里（src/e5/protocol.in_headline，与 test 同一个函数；a 另要求目标起伏
不全平，起伏误差才有定义）。清单 = val 序号升序（档的轮转同评测集，protocol.eval_items 的 dev）里前 n_per_task 条合格的；口径外的样本
不进分数，所以不解码。只用本次训练的任务。清单的 id 指纹进运行身份。

**一次评测**（ValSelector.evaluate）：与 test 评测同一条路径（src/e5/trained_eval）：提示 = PromptFormat.render、指令在前；单轮受限解码
generate_restricted（允许集合 = E5Tokens.allowed_ids、max_new_tokens = n(n + 1) + 1、本家族的停止 token、关思考、贪心）；按格式
（二值 c、d、e / 起伏 a、b）连续成批（同一批同一个允许集合），批大小 batch_size；逐条记录 trained_eval.record，汇总
scoring.aggregate（n_resamples = 0），取标题口径（headline）的读数。调用方负责 eval 模式、no_grad 与随机数状态（src/e5/train.py）。

**规则 mean_headline_v1**（selection_score）：分数 = 训练任务各类标题指标 h_t 的平均，越高越好。
- b、c、d、e：h = 标题口径的网格 mask IoU，严格读数（解析失败记 0）——b 的标题指标就是环绕上的 mask IoU（E5 预注册 §2 的表）；
- a：h = 1 − 起伏误差（严格读数，失败记 scoring.RELIEF_FAIL_VALUE = 2.0）/ RELIEF_FAIL_VALUE，线性映到 [0, 1]。
选择（best_of）：分数最高的一步，同分取较早的一步。每次评测的全部读数逐行写进 val_selection.jsonl（只含由模型输出决定的量，
CPU 上续跑与一次跑完逐字节相同；墙钟在运行记录里）。
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from src.e5 import scoring as SC
from src.e5.protocol import E5_TASKS, SUBSETS, EvalItem, eval_items, num_bins
from src.e5.runner import check_platform, make_sample
from src.e5.token_grid import BINARY, RELIEF, E5Tokens, PromptFormat, max_new_tokens, sample_images, task_format
from src.synth.relief import RELIEF_TARGET_KEY, relief_score
from src.synth.sample import Sample
from src.synth.seeding import canonical_task

VAL_SUBSET = "dev"
VAL_SPLIT = SUBSETS[VAL_SUBSET]  # 生成器的 val 划分
VAL_LOG = "val_selection.jsonl"
RULE = "mean_headline_v1"
RULE_TEXT = ("分数 = 训练任务各类标题指标的平均（越高越好）：b、c、d、e 取标题口径的网格 mask IoU（严格读数）；a 取 1 − 起伏误差"
             "（严格读数，失败记 2.0）/ 2.0。选分数最高的一步，同分取较早的一步（src/e5/val_select.py）")
# 各类的标题指标（E5 预注册 §2 的表：a 是起伏误差，b–e 是 mask IoU）
HEADLINE_METRIC: dict[str, str] = {t: ("relief" if t == "rotation3d" else "iou") for t in E5_TASKS}
SCAN_FACTOR = 32  # 找 n 条合格样本时最多看 SCAN_FACTOR × n 条候选（b 约 1/4 合格，a 约 3/4）


class SplitGuardError(AssertionError):
    """val 选择的样本不是 val 划分（例如 test）：拒绝生成。"""


def guard_val(item: EvalItem) -> None:
    """入口断言：val 选择只碰 val 划分（模块说明）。"""
    if item.split != VAL_SPLIT:
        raise SplitGuardError(f"val 选检查点只能用 {VAL_SPLIT} 划分，收到 {item.id}（划分 {item.split}）：test 只在决策门用")


def val_sample(item: EvalItem) -> Sample:
    guard_val(item)
    return make_sample(item)


def target_relief_flat(sample: Sample) -> bool:
    """目标起伏是否全平（同 scoring.score 的 target_flat：起伏误差没有定义）。"""
    target = sample.extras[RELIEF_TARGET_KEY]
    return relief_score(target, target).reason == "flat_target"


def qualifies(sample: Sample) -> bool:
    """这条 val 样本是否进选择：在标题口径里（protocol.in_headline，经 scoring.sample_fields）；标题指标是起伏误差的类另要求目标起伏不全平。"""
    if not SC.sample_fields(sample)["headline"]:
        return False
    return not (HEADLINE_METRIC[canonical_task(sample.task)] == "relief" and target_relief_flat(sample))


def selection_items(tasks: Sequence[str], n_per_task: int, *, tiny: bool) -> dict[str, list[EvalItem]]:
    """每类前 n_per_task 条合格的 val 样本（模块说明）。d 的数据只在规范平台上生成（同评测）。"""
    if isinstance(n_per_task, bool) or not isinstance(n_per_task, int) or n_per_task < 1:
        raise ValueError(f"n_per_task 必须是正整数，收到 {n_per_task!r}")
    out: dict[str, list[EvalItem]] = {}
    for t in tasks:
        t = canonical_task(t)
        check_platform(t, tiny)
        cands = eval_items(t, VAL_SUBSET, math.ceil(SCAN_FACTOR * n_per_task / num_bins(t)))
        got: list[EvalItem] = []
        for it in cands:
            if qualifies(val_sample(it)):
                got.append(it)
                if len(got) == n_per_task:
                    break
        if len(got) < n_per_task:
            raise RuntimeError(f"{t}：前 {len(cands)} 条 val 样本里只有 {len(got)} 条落在标题口径里，不够 {n_per_task} 条")
        out[t] = got
    return out


def ids_digest(items: Sequence[EvalItem]) -> str:
    return hashlib.sha256("\n".join(it.id for it in items).encode("utf-8")).hexdigest()


def task_value(task: str, head: Mapping[str, Any]) -> float | None:
    """一类的标题指标 h_t（规则 mean_headline_v1）：标题口径的读数块（scoring.blocks）→ [0, 1]，越高越好。"""
    if HEADLINE_METRIC[canonical_task(task)] == "relief":
        r = (head.get("relief_error") or {}).get("strict")
        return None if r is None else 1.0 - float(r) / SC.RELIEF_FAIL_VALUE
    v = head["mask_iou_grid"]["strict"]
    return None if v is None else float(v)


def selection_score(by_task: Mapping[str, Mapping[str, Any]], rule: str = RULE) -> float:
    """各类 h_t 的平均（规则见模块说明）。"""
    if rule != RULE:
        raise ValueError(f"只实现了 {RULE}，收到 {rule!r}")
    vals = [by_task[t]["h"] for t in by_task]
    if not vals or any(v is None for v in vals):
        raise ValueError(f"有的类没有标题读数：{ {t: by_task[t]['h'] for t in by_task} }")
    return float(sum(vals) / len(vals))


def best_of(history: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """分数最高的一条，同分取步数较小的。"""
    best = None
    for h in history:
        if best is None or float(h["score"]) > float(best["score"]) or \
                (float(h["score"]) == float(best["score"]) and int(h["step"]) < int(best["step"])):
            best = h
    return best


@dataclass
class ValSelector:
    items: Mapping[str, Sequence[EvalItem]]
    rule: str
    val_every: int
    batch_size: int
    n_per_task: int

    @classmethod
    def build(cls, sel: Mapping[str, Any], tasks: Sequence[str], *, tiny: bool) -> "ValSelector":
        if sel.get("split") != VAL_SPLIT:
            raise SplitGuardError(f"selection.split 必须是 {VAL_SPLIT}，收到 {sel.get('split')!r}")
        if sel["rule"] != RULE:
            raise ValueError(f"只实现了 {RULE}，收到 {sel['rule']!r}")
        items = selection_items(tasks, int(sel["n_per_task"]), tiny=tiny)
        return cls(items=items, rule=str(sel["rule"]), val_every=int(sel["val_every"]), batch_size=int(sel["batch_size"]),
                   n_per_task=int(sel["n_per_task"]))

    def is_eval_step(self, step_done: int, steps: int) -> bool:
        """完成第 step_done 步之后是否评一次：val_every 的倍数，以及最后一步。"""
        return step_done % self.val_every == 0 or step_done == steps

    def eval_steps(self, steps: int) -> list[int]:
        return [s for s in range(1, steps + 1) if self.is_eval_step(s, steps)]

    def identity(self) -> dict[str, Any]:
        return {"rule": self.rule, "rule_text": RULE_TEXT, "split": VAL_SPLIT, "val_every": self.val_every,
                "n_per_task": self.n_per_task, "batch_size": self.batch_size, "headline_metric": {t: HEADLINE_METRIC[t] for t in self.items},
                "items": {t: {"n": len(v), "ids_sha256": ids_digest(v), "first": v[0].id, "last": v[-1].id} for t, v in self.items.items()},
                "rule_source": "src/e5/val_select.py（configs/e5_train.json 的 selection.basis）"}

    def evaluate(self, ad: Any, e5tok: E5Tokens, prompt: PromptFormat, n: int, *,
                 log: Callable[[str], None] | None = None) -> dict[str, Any]:
        """一次评测（模块说明）。返回 {score, by_task}；只含由模型输出决定的量。"""
        from src.e5.trained_eval import record

        mnt = max_new_tokens(n)
        recs: dict[str, list[dict[str, Any]]] = {t: [] for t in self.items}
        for fmt in (BINARY, RELIEF):
            group = [it for t in E5_TASKS if t in self.items and task_format(t) == fmt for it in self.items[t]]
            allowed = e5tok.allowed_ids_of(fmt)
            for lo in range(0, len(group), self.batch_size):
                batch = [val_sample(it) for it in group[lo: lo + self.batch_size]]
                texts = [prompt.render(s, n) for s in batch]
                inputs = [ad.build_inputs(sample_images(s), txt, prompt.order) for s, txt in zip(batch, texts)]
                res = ad.generate_restricted(inputs, allowed, max_new_tokens=mnt, stop_ids=[e5tok.stop_id], enable_thinking=False)
                if len(res) != len(batch):
                    raise RuntimeError("generate_restricted 返回的行数与输入不同")
                for s, txt, r in zip(batch, texts, res):
                    rec = record(s, txt, r, e5tok, n)
                    if rec["split"] != VAL_SPLIT or not rec["headline"]:
                        raise SplitGuardError(f"{rec['id']}：val 选择的样本必须是 val 划分且在标题口径里")
                    recs[canonical_task(s.task)].append(rec)
        by_task: dict[str, dict[str, Any]] = {}
        for t, rs in recs.items():
            agg = SC.aggregate(t, rs, n_resamples=0, enable_thinking=False)
            head = agg["headline"]
            if head["mask_iou_grid"]["n"] != len(self.items[t]):
                raise RuntimeError(f"{t}：标题口径的条数 {head['mask_iou_grid']['n']} 不是 {len(self.items[t])}")
            row = {"n": len(rs), "metric": HEADLINE_METRIC[t], "h": task_value(t, head),
                   "iou_grid_strict": head["mask_iou_grid"]["strict"], "iou_grid_valid_rate": head["mask_iou_grid"]["valid_rate"],
                   "parse_failure_rate": agg["parse"]["failure_rate"],
                   "mean_generated_tokens": (agg.get("generated_tokens") or {}).get("mean")}
            if "relief_error" in head:
                row.update(relief_strict=head["relief_error"]["strict"], relief_valid_rate=head["relief_error"]["valid_rate"],
                           relief_n=head["relief_error"]["n"])
            by_task[t] = row
        out = {"score": selection_score(by_task, self.rule), "by_task": by_task}
        if log:
            log("[val 选择] 分数 " + f"{out['score']:.4f}（" + "，".join(f"{t} {v['h']:.3f}" for t, v in by_task.items()) + "）")
        return out


__all__ = ["HEADLINE_METRIC", "RULE", "RULE_TEXT", "SplitGuardError", "VAL_LOG", "VAL_SPLIT", "VAL_SUBSET", "ValSelector", "best_of",
           "guard_val", "ids_digest", "qualifies", "selection_items", "selection_score", "task_value", "val_sample"]
