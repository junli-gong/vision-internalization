"""DEV-0075 的替身运行器（scripts/run_lisa.py --runner-factory tests._lisa_fakes:<名>）：不加载权重，返回固定 mask。

infer(row) 的返回与 scripts/run_lisa.LisaRunner.infer 同形：mask（原图尺寸的 bool，中央一半的矩形）、n_seg、n_masks、conversation、
n_input_ids、n_visual_tokens、image_hw_decoded、sam_resize、forwards、forward_s。
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

NO_SEG_SENT_IDS = frozenset({"107", "405"})  # tiny 清单里：testA 的第 2 句、refcocog 的第 2 句
RAISE_SENT_IDS = frozenset({"406"})  # refcocog 的第 3 句


def fixed_mask(h: int, w: int) -> np.ndarray:
    m = np.zeros((h, w), dtype=bool)
    m[h // 4: (3 * h) // 4, w // 4: (3 * w) // 4] = True
    return m


class FixedMaskRunner:
    def __init__(self, *, no_seg: frozenset[str] = frozenset(), raise_ids: frozenset[str] = frozenset(), always_raise: bool = False):
        self.no_seg, self.raise_ids, self.always_raise = no_seg, raise_ids, always_raise
        self.fixes: list[str] = []
        self.info: dict[str, Any] = {"versions": None, "stand_in": "tests/_lisa_fakes.FixedMaskRunner"}

    def infer(self, row: Mapping[str, Any]) -> dict[str, Any]:
        if self.always_raise or row["sent_id"] in self.raise_ids:
            raise RuntimeError(f"替身：{row['id']} 推理异常")
        h, w = int(row["height"]), int(row["width"])
        n_seg = 0 if row["sent_id"] in self.no_seg else 1
        return {"mask": fixed_mask(h, w) if n_seg else None, "n_seg": n_seg, "n_masks": n_seg,
                "conversation": f"替身 USER: <im_start><image><im_end>\n What is {row['sent'].strip().lower()} in this image? ...",
                "n_input_ids": 64, "n_visual_tokens": 256, "image_hw_decoded": [h, w], "sam_resize": [h, w],
                "forwards": {"llm": 1, "sam_image_encoder": 1, "sam_mask_decoder": n_seg}, "forward_s": 0.01}

    def stats(self) -> dict[str, Any]:
        return {"peak_mem_inference_bytes": 0}


def make_fixed(args: Any) -> FixedMaskRunner:
    return FixedMaskRunner()


def make_with_failures(args: Any) -> FixedMaskRunner:
    return FixedMaskRunner(no_seg=NO_SEG_SENT_IDS, raise_ids=RAISE_SENT_IDS)


def make_always_raise(args: Any) -> FixedMaskRunner:
    return FixedMaskRunner(always_raise=True)
