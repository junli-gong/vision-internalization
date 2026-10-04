"""去位置偏好的字母读数（E1 预注册 §10.1，D66-3，N22 → (b)；DEV-0033）：选项的循环移位、逐顺序的字母概率、映射回选项内容后取平均。

口径逐字照预注册 §10.1 与 configs/e1_posthoc.json 的 debiased_letter（规则文字在配置里，这里实现）：
- 每题 k 个选项，读 k 种循环移位 s = 0 … k−1：第 s 种顺序在第 i 个字母的位置放原顺序的第 (i + s) mod k 个选项（cyclic_perm），
  s = 0 就是原顺序。字母 A、B、… 的位置、题干与图都不变。
- 每种顺序下用 adapter.option_logits 取每个合法字母的 logit（"X" 与 " X" 两种写法取较大，D36-4），只在合法字母上 softmax（fp32），
  第 i 个字母的概率记到选项内容 perm_s[i] 上；k 种顺序的概率按选项内容取平均（s 升序累加），最大者为答案，平票取原顺序里靠前的。
- 选项的重排（ordered_prompts）：
  - BLINK、CV-Bench（how = prompt_field）：提示是数据集自带的整段文字。prompt 字段首尾的空白（如有）单独留出、原样放回；
    split_option_lines 把其余部分的最后 k 行认作选项行（每行必须逐字等于 "({letter}) {text}"，text = 数据集的 choices[i]），
    其余是题干；按新顺序拼回后照原模板渲染。
  - NYU 点对（rendered_template）：对渲染好的统一提示用同一条规则拆（选项是任务条目的 choices："A is closer"、"B is closer"，
    指图上的标记，图不变，只换两行的顺序）。
  - EmbSpatial（option_format）：把样本的 choices 换成新顺序，照原模板渲染（e1_prompts.json 的 option_format、option_joiner）。
  - 每条样本都断言：原顺序拼回与原提示逐字节相同（prompt_field 还核对 prompt 字段本身），且与主评测的渲染逐字节相同；
    拆不出就报 OptionSplitError，不跳过。
- 原字母读数在同一次运行里顺带算出（original_letter_control = same_run_order0）：s = 0 那一种顺序的 argmax（OptionScores.choice）。

本模块只有纯函数（不碰模型与文件）：ordered_prompts、reading_rows（读数序列与分批键，order_major）、combine_sample（一条样本的 k 次读数 →
predictions.jsonl 的一行）、debiased_metrics（metrics.json 的事后一节）。运行（批、续跑、输出）在 src/e1/posthoc_runner.py。
"""

from __future__ import annotations

import dataclasses
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Hashable, Mapping, Sequence

import numpy as np

from src.data.records import Sample
from src.e1.posthoc import DEBIASED_READOUT, PosthocConfig
from src.e1.prompts import PromptConfig, Rendered
from src.e1.scoring import aggregate, choice_gt, ci_dict, metric_of
from src.stats import paired_bootstrap_ci

DEBIASED_ROUTE = DEBIASED_READOUT  # predictions.jsonl 的 parsed.route
_OPTION_LINE_START = re.compile(r"\([A-Z]\) ")  # 题干最后一行不能也是选项行（选项行比 choices 多时报错）


class OptionSplitError(ValueError):
    """选项行拆不出来，或按原顺序拼回与原提示不是逐字节相同。"""


def cyclic_perm(k: int, s: int) -> tuple[int, ...]:
    """第 s 种循环移位：perm[i] = 第 i 个字母位置上放的原选项下标 = (i + s) mod k。"""
    if isinstance(k, bool) or not isinstance(k, int) or k < 2:
        raise ValueError(f"选项数 k 必须是 ≥ 2 的整数，收到 {k!r}")
    if isinstance(s, bool) or not isinstance(s, int) or not 0 <= s < k:
        raise ValueError(f"顺序 s 必须在 [0, {k}) 内，收到 {s!r}")
    return tuple((i + s) % k for i in range(k))


@dataclass(frozen=True)
class OptionBlock:
    """一段提示 = 题干 + joiner + 选项行（每行 line_format.format(letter=字母, text=选项)，行间 joiner）。"""

    head: str
    letters: tuple[str, ...]
    options: tuple[str, ...]
    line_format: str
    joiner: str

    def compose(self, perm: Sequence[int]) -> str:
        """按 perm 拼回：第 i 个字母后面放 options[perm[i]]。"""
        if sorted(perm) != list(range(len(self.options))):
            raise ValueError(f"perm {list(perm)} 不是 0…{len(self.options) - 1} 的排列")
        lines = [self.line_format.format(letter=lt, text=self.options[j]) for lt, j in zip(self.letters, perm)]
        return self.head + self.joiner + self.joiner.join(lines)


def split_option_lines(text: str, letters: Sequence[str], options: Sequence[str], *, line_format: str,
                       joiner: str, where: str = "") -> OptionBlock:
    """拆出选项行（规则见模块说明与 configs/e1_posthoc.json 的 option_split.rule）。拆不出或还原不了：OptionSplitError。"""
    letters, options = tuple(letters), tuple(options)
    k = len(options)
    if k < 2 or len(letters) != k:
        raise OptionSplitError(f"{where}：字母 {letters} 与选项 {options} 的个数不对")
    for o in options:
        if not isinstance(o, str) or joiner in o or o == "":
            raise OptionSplitError(f"{where}：选项 {o!r} 为空或含行分隔符")
    lines = text.split(joiner)
    if len(lines) < k + 1:
        raise OptionSplitError(f"{where}：提示只有 {len(lines)} 行，放不下题干 + {k} 行选项：{text!r}")
    head_lines, tail = lines[:-k], lines[-k:]
    for i, (lt, o, got) in enumerate(zip(letters, options, tail)):
        want = line_format.format(letter=lt, text=o)
        if got != want:
            raise OptionSplitError(f"{where}：倒数第 {k - i} 行是 {got!r}，应为选项行 {want!r}（提示 {text!r}）")
    head = joiner.join(head_lines)
    if not head.strip():
        raise OptionSplitError(f"{where}：选项行之前没有题干：{text!r}")
    if _OPTION_LINE_START.match(head_lines[-1]):
        raise OptionSplitError(f"{where}：题干的最后一行 {head_lines[-1]!r} 也像选项行（选项行比 {k} 个多？）")
    block = OptionBlock(head, letters, options, line_format, joiner)
    if block.compose(tuple(range(k))) != text:  # 原顺序拼回必须逐字节相同
        raise OptionSplitError(f"{where}：原顺序拼回与原提示不是逐字节相同")
    return block


@dataclass(frozen=True)
class OrderedPrompt:
    """一种顺序下送进模型的指令。perm[i] = 第 i 个字母位置上的原选项下标。"""

    order: int
    perm: tuple[int, ...]
    text: str


def ordered_prompts(prompts: PromptConfig, posthoc: PosthocConfig, task: str, family: str,
                    sample: Sample) -> tuple[Rendered, list[OrderedPrompt]]:
    """一条样本的原渲染（主评测同一份）与 k 种顺序的指令。每条都断言原顺序与主评测的渲染逐字节相同。"""
    rendered = prompts.render(task, family, sample)
    letters = tuple(rendered.extras["letters"])
    options = tuple(rendered.extras["options"])
    k = len(options)
    how = posthoc.split_how(task)
    fmt, joiner = posthoc.line_format, posthoc.line_joiner
    where = f"{task} {sample.id}"
    if how == "prompt_field":
        raw = sample.fields["prompt"]
        if tuple(sample.fields["choices"]) != options:
            raise OptionSplitError(f"{where}：渲染的选项 {options} 与数据集的 choices {sample.fields['choices']} 不同")
        # prompt 字段首尾的空白（如有）单独留出、原样放回；渲染时本来就会去掉（模板 "{prompt}" 渲染后 strip）
        core = raw.strip()
        lead, trail = raw[: len(raw) - len(raw.lstrip())], raw[len(raw.rstrip()):]
        block = split_option_lines(core, letters, options, line_format=fmt, joiner=joiner, where=where)
        if lead + block.compose(tuple(range(k))) + trail != raw:
            raise OptionSplitError(f"{where}：原顺序拼回与 prompt 字段不是逐字节相同")

        def text_of(perm: tuple[int, ...]) -> str:
            s = dataclasses.replace(sample, fields={**sample.fields, "prompt": lead + block.compose(perm) + trail})
            return prompts.render(task, family, s).text
    elif how == "rendered_template":
        block = split_option_lines(rendered.text, letters, options, line_format=fmt, joiner=joiner, where=where)

        def text_of(perm: tuple[int, ...]) -> str:
            return block.compose(perm)
    elif how == "option_format":
        if tuple(sample.fields["choices"]) != options:
            raise OptionSplitError(f"{where}：渲染的选项 {options} 与数据集的 choices {sample.fields['choices']} 不同")

        def text_of(perm: tuple[int, ...]) -> str:
            s = dataclasses.replace(sample, fields={**sample.fields, "choices": [options[j] for j in perm]})
            return prompts.render(task, family, s).text
    else:
        raise ValueError(f"未知的拆分方式 {how!r}")
    out = [OrderedPrompt(s, cyclic_perm(k, s), text_of(cyclic_perm(k, s))) for s in range(k)]
    if out[0].text != rendered.text:
        raise OptionSplitError(f"{where}：原顺序重拼的指令与主评测的渲染不是逐字节相同：{out[0].text!r} ≠ {rendered.text!r}")
    if len({o.text for o in out}) != k:
        raise OptionSplitError(f"{where}：{k} 种顺序的指令有重复（选项内容相同？）")
    return rendered, out


def reading_rows(ks: Sequence[int], letter_sets: Sequence[Sequence[str]]) -> tuple[list[tuple[int, int]], list[Hashable]]:
    """order_major 的读数序列与分批键：rows = [(样本下标 j, 顺序 s)]，先 s = 0 的全部样本（选择顺序），再 s = 1 …，s 只取 k_j > s 的
    样本；keys[r] = (s, 本题的字母集合)，交给 runner.batch_plan，同一批的顺序与字母集合都相同；s = 0 那一段与主评测的批划分相同。"""
    if len(ks) != len(letter_sets):
        raise ValueError("ks 与 letter_sets 不等长")
    rows: list[tuple[int, int]] = []
    keys: list[Hashable] = []
    for s in range(max(ks, default=0)):
        for j, (k, lts) in enumerate(zip(ks, letter_sets)):
            if s < k:
                rows.append((j, s))
                keys.append((s, tuple(lts)))
    return rows, keys


def reading_record(sample: Sample, op: OrderedPrompt, letters: Sequence[str], scores: Any) -> dict[str, Any]:
    """readings.jsonl 的一行：一条样本在一种顺序下的读数（OptionScores）。letter_probs 按字母顺序（A、B、…）。"""
    if tuple(scores.options) != tuple(letters):
        raise RuntimeError(f"{sample.id}：option_logits 的选项 {scores.options} 不是本题的字母 {tuple(letters)}")
    i = list(letters).index(scores.choice)
    return {"id": sample.id, "order": op.order, "k": len(letters), "perm": list(op.perm), "instruction": op.text,
            "letter_logits": list(scores.logits), "letter_token_ids": list(scores.token_ids),
            "letter_probs": list(scores.probs), "choice_letter": scores.choice, "choice_option": op.perm[i],
            "n_prompt_tokens": scores.n_prompt_tokens, "n_visual_tokens": scores.n_visual_tokens,
            "n_image_pixels": scores.n_image_pixels}


def mean_option_probs(readings: Sequence[Mapping[str, Any]], k: int) -> list[float]:
    """k 种顺序的字母概率映射回选项内容后的平均（s 升序累加再除以 k；规则见模块说明）。"""
    if [int(r["order"]) for r in readings] != list(range(k)):
        raise ValueError(f"读数的顺序 {[r['order'] for r in readings]} 不是 0…{k - 1}")
    total = [0.0] * k
    for r in readings:
        perm = list(r["perm"])
        if perm != list(cyclic_perm(k, int(r["order"]))):
            raise ValueError(f"第 {r['order']} 种顺序的 perm {perm} 不是循环移位")
        for i, p in enumerate(r["letter_probs"]):
            total[perm[i]] += float(p)
    return [t / k for t in total]


def argmax_first(values: Sequence[float]) -> int:
    """最大值的下标；完全相等时取下标小的（原顺序里靠前的选项）。"""
    return max(range(len(values)), key=lambda j: (values[j], -j))


def combine_sample(task: str, sample: Sample, rendered: Rendered, image_size: Sequence[int],
                   readings: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """一条样本的 k 次读数 → predictions.jsonl 的一行（字段与主评测同形，aggregate 能直接用）。

    - parsed.letter = 去偏读数的答案（原顺序的字母），parsed.original_letter = s = 0 的 argmax（原字母读数）；
    - score.correct / original_correct；p_gold = 正确选项的平均概率，original_p_gold = 它在 s = 0 下的概率；
    - n_generated_tokens = k（每种顺序一次 option_logits，记账口径同 D37：一次记作生成 1 个 token），n_prompt_tokens = k 次之和；
      n_visual_tokens、n_image_pixels 是一次读数的值（k 次相同）。"""
    letters = list(rendered.extras["letters"])
    k = len(letters)
    if [r["id"] for r in readings] != [sample.id] * k:
        raise ValueError(f"{sample.id}：读数的 id 不对")
    vis = {(int(r["n_visual_tokens"]), int(r["n_image_pixels"])) for r in readings}
    if len(vis) != 1:
        raise RuntimeError(f"{sample.id}：k 种顺序的视觉 token / 像素数不同 {sorted(vis)}（图应当不变）")
    mean = mean_option_probs(readings, k)
    j = argmax_first(mean)
    gt = choice_gt(task, sample)
    g = letters.index(gt)
    orig = readings[0]
    orig_letter = orig["choice_letter"]
    n_vis, n_pix = vis.pop()
    return {
        "id": sample.id, "source": sample.source, "split": sample.split, "instruction": rendered.text,
        "extras": rendered.extras, "image_size": list(image_size), "readout": DEBIASED_READOUT,
        "truncated": False, "thinking_unclosed": False,
        "n_generated_tokens": k, "n_prompt_tokens": int(sum(int(r["n_prompt_tokens"]) for r in readings)),
        "n_visual_tokens": n_vis, "n_image_pixels": n_pix, "k": k,
        "orders": [{"order": int(r["order"]), "perm": list(r["perm"]), "letter_probs": list(r["letter_probs"]),
                    "choice_letter": r["choice_letter"], "choice_option": int(r["choice_option"])} for r in readings],
        "mean_probs": mean,
        "failure": None,
        "parsed": {"letter": letters[j], "route": DEBIASED_ROUTE, "option_index": j, "original_letter": orig_letter},
        "score": {"gt": gt, "correct": letters[j] == gt, "original_correct": orig_letter == gt,
                  "p_gold": mean[g], "original_p_gold": float(orig["letter_probs"][g])},
    }


def _original_shadow(rec: Mapping[str, Any]) -> dict[str, Any]:
    """原字母读数（s = 0）的影子记录：同一行，只把 parsed 与 score 换成 s = 0 的读数（aggregate 用）。"""
    out = dict(rec)
    out["readout"] = "letter_order0"
    out["parsed"] = {"letter": rec["parsed"]["original_letter"], "route": "letter_order0"}
    out["score"] = {"gt": rec["score"]["gt"], "correct": rec["score"]["original_correct"]}
    return out


def _letter_counts(letters: Sequence[str]) -> dict[str, Any]:
    c = Counter(letters)
    top = sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))
    n = len(letters)
    return {"n": n, "letter_counts": dict(sorted(c.items())),
            "most_common_letter": top[0][0] if top else None,
            "most_common_letter_share": (top[0][1] / n) if n else None}


def per_order(task: str, samples: Sequence[Sample], recs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """每种顺序 s 单独的读数：该顺序下 argmax 映射回选项内容后的准确率（只含 k > s 的样本）、最常见字母占比，并按 k 分开。"""
    out = []
    for s in range(max((r["k"] for r in recs), default=0)):
        rows = [(smp, r, r["orders"][s]) for smp, r in zip(samples, recs) if r["k"] > s]
        correct = [float(r["extras"]["letters"][o["choice_option"]] == r["score"]["gt"]) for _, r, o in rows]
        e = {"order": s, "accuracy": metric_of("accuracy", [{"score": {"correct": c}} for c in correct]),
             **_letter_counts([o["choice_letter"] for _, _, o in rows]), "by_k": {}}
        for k in sorted({r["k"] for _, r, _ in rows}):
            sub = [(r, o) for _, r, o in rows if r["k"] == k]
            acc = [float(r["extras"]["letters"][o["choice_option"]] == r["score"]["gt"]) for r, o in sub]
            e["by_k"][str(k)] = {"accuracy": float(np.mean(acc)), **_letter_counts([o["choice_letter"] for _, o in sub])}
        out.append(e)
    return out


def debiased_metrics(task: str, samples: Sequence[Sample], recs: Sequence[Mapping[str, Any]], *,
                     has_gt: bool) -> dict[str, Any]:
    """metrics.json 的主体：aggregate（标题 = 去偏读数的严格读数与 CI，三列、分档同主评测）+ 事后一节：
    original_letter（同一次运行 s = 0 的原字母读数：标题、分档与 CV-Bench 的官方合成准确率）、paired_diff（去偏 − 原字母的配对差与配对
    bootstrap CI、答案变了的条数）、per_order、random_level（逐题 1/k 的平均）、k 的分布、mean_p_gold（正确选项的平均概率）。"""
    body = aggregate(task, samples, recs, has_gt=has_gt, enable_thinking=False)
    ks = [int(r["k"]) for r in recs]
    post: dict[str, Any] = {"readout": DEBIASED_READOUT, "n": len(recs),
                            "k_counts": {str(k): c for k, c in sorted(Counter(ks).items())},
                            "random_level": float(np.mean([1.0 / k for k in ks])) if ks else None}
    if has_gt and recs:
        shadow = aggregate(task, samples, [_original_shadow(r) for r in recs], has_gt=True, enable_thinking=False)
        post["original_letter"] = {k: shadow[k] for k in ("headline", "breakdown", "cvbench_official_combined") if k in shadow}
        a = np.array([float(r["score"]["correct"]) for r in recs])
        b = np.array([float(r["score"]["original_correct"]) for r in recs])
        changed = [r for r in recs if r["parsed"]["letter"] != r["parsed"]["original_letter"]]
        post["paired_diff"] = {
            "metric": "accuracy", "what": "去偏读数 − 原字母读数（同一批样本、同一次运行；配对 bootstrap 95% CI）",
            "value": float(a.mean() - b.mean()), "ci": ci_dict(paired_bootstrap_ci(a, b)), "n": len(recs),
            "n_answer_changed": len(changed),
            "n_wrong_to_right": sum(1 for r in changed if r["score"]["correct"] and not r["score"]["original_correct"]),
            "n_right_to_wrong": sum(1 for r in changed if r["score"]["original_correct"] and not r["score"]["correct"])}
        post["mean_p_gold"] = {"debiased": float(np.mean([r["score"]["p_gold"] for r in recs])),
                               "original": float(np.mean([r["score"]["original_p_gold"] for r in recs]))}
        post["per_order"] = per_order(task, samples, recs)
    post["answer_letters"] = _letter_counts([r["parsed"]["letter"] for r in recs])
    body["debiased"] = post
    return body


__all__ = [
    "DEBIASED_ROUTE",
    "OptionBlock",
    "OptionSplitError",
    "OrderedPrompt",
    "argmax_first",
    "combine_sample",
    "cyclic_perm",
    "debiased_metrics",
    "mean_option_probs",
    "ordered_prompts",
    "per_order",
    "reading_record",
    "reading_rows",
    "split_option_lines",
]
