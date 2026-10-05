"""NYU 点对"对调图上的 A、B 标记"的去偏读数（DEV-0036；D72-2，N26 → 问题 1 选 (a)；事后读数，不改 G1 的结论）。

为什么：循环移位去偏（src/e1/debiased.py）只换选项文字的位置，而 NYU 点对的选项文字 "A is closer" / "B is closer" 指的是图上的
标记——"按标记作答"的模型在移位后被误罚（N26）。这里改为换图上的标记，选项文字不变，按同一个物理点平均。

口径逐字照 configs/e1_posthoc.json 的 letter_swap（规则文字在配置里，这里实现）：
- 每题两张图（v = 0、1）：original = 主评测的那张渲染图；labels_swapped = 同一张原图经 nyu_pairs.render_pair(…, swap_labels=True)，
  两个标签的位置与环都不变，只把字形互换（物理点 1 = 原图里标 A 的点，旁边改写 B；物理点 2 = 原图里标 B 的点，旁边改写 A）。提示不变。
- 每张图读一次受限字母读数（option_logits，两种写法取大，D36-4），只在 A、B 上 softmax；第 v 张图的第 i 个字母对应物理点
  perm_v[i] = (i + v) mod 2——与 debiased_letter 在 k = 2 时的循环移位是同一个排列，所以合成（按物理点取平均、平票取点 1）、原字母
  读数（v = 0 的 argmax）与指标直接复用 src/e1/debiased 的函数，只换读法名与图的名字。
- 运行时逐条核对（check_swapped）：labels_swapped 与 original 在两个标签框之外逐像素相同、框里都有不同。

本模块只有纯函数（不跑模型、不写文件）；运行在 src/e1/posthoc_runner.run_letter_swap_task。
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from src.data.nyu_pairs import LETTERS, label_boxes, render_pair
from src.data.records import Sample
from src.e1 import debiased as DB
from src.e1.posthoc import LETTER_SWAP_READOUT, LETTER_SWAP_VIEWS
from src.e1.prompts import Rendered


class LetterSwapError(RuntimeError):
    """对调标记的图或读数不合约定（字母不是 A、B，换标记的图在标签框外有不同、框里没变，读数的图不对）。"""


def view_prompts(rendered: Rendered) -> list[DB.OrderedPrompt]:
    """两张图各自的"顺序"：指令都是主评测的原文（不换选项），perm_v = cyclic_perm(2, v)（第 i 个字母 → 物理点 (i + v) mod 2）。"""
    letters = tuple(rendered.extras["letters"])
    if letters != LETTERS:
        raise LetterSwapError(f"NYU 点对的字母应是 {LETTERS}，收到 {letters}")
    return [DB.OrderedPrompt(v, DB.cyclic_perm(len(LETTERS), v), rendered.text) for v in range(len(LETTER_SWAP_VIEWS))]


def _points(sample: Sample) -> tuple[list[int], list[int]]:
    pts = sample.fields["points"]
    return list(pts["A"]), list(pts["B"])


def swapped_image(sample: Sample) -> np.ndarray:
    """labels_swapped 那张图：未加标记的原图（fields.raw_image）上 render_pair(A, B, swap_labels=True)。"""
    from PIL import Image

    with Image.open(sample.fields["raw_image"]) as im:
        raw = np.asarray(im.convert("RGB"))
    A, B = _points(sample)
    return render_pair(raw, A, B, swap_labels=True)


def check_swapped(original: np.ndarray, swapped: np.ndarray, sample: Sample) -> dict[str, Any]:
    """两张图只差标签框里的字形（规则见 configs/e1_posthoc.json 的 letter_swap.image_check）。返回不同像素的个数。"""
    o, s = np.asarray(original), np.asarray(swapped)
    if o.shape != s.shape or o.dtype != s.dtype:
        raise LetterSwapError(f"{sample.id}：换标记的图形状 {s.shape}/{s.dtype} 与原图 {o.shape}/{o.dtype} 不同")
    A, B = _points(sample)
    boxes = label_boxes(o.shape[:2], A, B)
    outside = np.ones(o.shape[:2], dtype=bool)
    for x0, y0, x1, y1 in boxes.values():
        outside[y0:y1, x0:x1] = False
    if not np.array_equal(o[outside], s[outside]):
        raise LetterSwapError(f"{sample.id}：换标记的图在标签框之外有像素不同")
    for k, (x0, y0, x1, y1) in boxes.items():
        if np.array_equal(o[y0:y1, x0:x1], s[y0:y1, x0:x1]):
            raise LetterSwapError(f"{sample.id}：标签 {k} 的框里没有变化（没换成另一个字母？）")
    return {"n_changed_pixels": int(np.any(o != s, axis=-1).sum()), "label_boxes": {k: list(v) for k, v in boxes.items()}}


def combine_sample(task: str, sample: Sample, rendered: Rendered, image_size: Sequence[int],
                   readings: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """一条样本的两次读数 → predictions.jsonl 的一行：合成同 debiased.combine_sample（按物理点取平均、平票取点 1），读法名换成
    letter_swap，orders 每项另记它是哪张图。"""
    for r in readings:
        if r.get("view") != LETTER_SWAP_VIEWS[int(r["order"])]:
            raise LetterSwapError(f"{sample.id}：第 {r['order']} 次读数的图是 {r.get('view')!r}，应为 {LETTER_SWAP_VIEWS[int(r['order'])]}")
        if r["instruction"] != rendered.text:
            raise LetterSwapError(f"{sample.id}：对调标记不换提示，读数的指令应与主评测的渲染相同")
    rec = DB.combine_sample(task, sample, rendered, image_size, readings)
    rec["readout"] = LETTER_SWAP_READOUT
    rec["parsed"]["route"] = LETTER_SWAP_READOUT
    for o in rec["orders"]:
        o["view"] = LETTER_SWAP_VIEWS[o["order"]]
    return rec


def letter_swap_metrics(task: str, samples: Sequence[Sample], recs: Sequence[Mapping[str, Any]], *,
                        has_gt: bool) -> dict[str, Any]:
    """metrics.json 的主体：同 debiased.debiased_metrics（标题 = 对调标记的去偏读数；原字母读数 = original 那张图的 argmax；配对差），
    事后一节改叫 letter_swap，per_order 改叫 per_view（每项另记图名）：两张图各自的准确率与最常见字母占比。"""
    body = DB.debiased_metrics(task, samples, recs, has_gt=has_gt)
    post = body.pop("debiased")
    post["readout"] = LETTER_SWAP_READOUT
    if "per_order" in post:
        post["per_view"] = [{"view": LETTER_SWAP_VIEWS[e["order"]], **e} for e in post.pop("per_order")]
    body["letter_swap"] = post
    return body


__all__ = [
    "LetterSwapError",
    "check_swapped",
    "combine_sample",
    "letter_swap_metrics",
    "swapped_image",
    "view_prompts",
]
