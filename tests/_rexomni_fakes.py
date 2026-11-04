"""DEV-0083：Rex-Omni 测试用的假模型（同 src/rexomni/model.Engine 的接口：.info 与 .generate(items)）。

ScriptedRexOmniEngine：按（图像, 所问的类别列表）给官方格式的回答（多个类别时各组之间 ", "，protocol.format_groups）（src/rexomni/protocol.format_boxes：<|object_ref_start|>类名<|object_ref_end|>
<|box_start|><x0><y0><x1><y1>, …<|box_end|>），框是这张图这个类别的真值框经官方的 框 → 网格（rex_omni/parser.py 的
convert_boxes_to_normalized_bins）换算。可选：在末尾重复第 1 个框（官方不去重，记 FP）、另加一个框（如 crowd 区域上的框，忽略）、
另加一个别的类名的组（丢掉并记数）。类名带 "junk" 时回答不合格式的文本，带 "silent" 时只回答 <|im_end|>。
"""

from __future__ import annotations

from typing import Any

from src.rexomni import protocol as P
from tests import _xm_run_fakes as F


class ScriptedRexOmniEngine:
    def __init__(self, answers: dict[tuple[str, str], list[list[float]]], *, duplicate_first: bool = False,
                 extra_box: list[float] | None = None, other_phrase: str | None = None) -> None:
        self.answers, self.dup, self.extra, self.other = answers, duplicate_first, extra_box, other_phrase
        self.info = {"kind": "scripted", "duplicate_first": duplicate_first, "extra_box": extra_box, "other_phrase": other_phrase}
        self.calls: list[dict[str, Any]] = []

    def _answer(self, img: Any, cats: list[str]) -> str:
        if any("junk" in c for c in cats):
            return "There are two of them."
        if any("silent" in c for c in cats):
            return ""
        w, h = img.size
        groups = []
        for cat in cats:
            boxes = [P.xyxy_to_bins(b, w, h) for b in self.answers.get((F.image_id(img), cat), [])]
            if self.dup and boxes:
                boxes = boxes + [list(boxes[0])]
            if self.extra is not None:
                boxes = boxes + [P.xyxy_to_bins(self.extra, w, h)]
            groups.append((cat, boxes))
        if self.other:
            groups.append((self.other, [[0, 0, 500, 500]]))
        return P.format_groups(groups)

    def generate(self, items):
        out = []
        for img, cats in items:
            assert not isinstance(cats, str)
            raw = self._answer(img, list(cats)).split(P.EOS_TEXT)[0]
            self.calls.append({"categories": list(cats), "size": img.size, "batch": len(items)})
            g = P.expected_grid(*img.size)
            out.append({"prompt": P.detect_prompt(list(cats)), "raw_text": raw, "text": raw, "stop_reason": "eos", "stop_string": None,
                        "n_prompt_tokens": 40 + P.n_visual_tokens(g), "n_generated_tokens": len(raw) // 4 + 1,
                        "n_visual_tokens": P.n_visual_tokens(g), "image_grid_thw": list(g), "max_new_tokens": P.MAX_NEW_TOKENS,
                        "wall_s": 0.001, "batch_wall_s": 0.001 * len(items), "batch_size": len(items)})
        return out
