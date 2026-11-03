"""X-M 测试用的假数据与假适配器（DEV-0080）。

- write_fixture(dir)：两张 128×96 的合成图（带纹理的背景上画红色圆盘与蓝色方块）与它们的"图像 × 类别"对（XMPair 的 JSON，
  src/xm/runner.pairs_from_json 读）：图 A 有 2 个圆盘、1 个方块；图 B 有 1 个圆盘和一块 crowd 区域（3 个挤在一起的小圆盘）。
- ScriptedXMAdapter：按提示给确定的回答——调用 1 回答这张图这个类别的真值框（网格整数，可选在末尾重复第 1 个框，测 NMS）；类别名带
  "junk" 时回答不合格式的文本；调用 2 给每个框的内切椭圆（n 个网格控制点）；类别名带 "short" 时第 1 条只有 n − 1 个点（逐条失败）。
  基座（system = base）的回答写成 Qwen 的原生格式（代码块里的 bbox_2d 对象），parse_boxes 用 Qwen 适配器的同一个函数。
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image

from src.executors.raster import mask_to_rle
from src.explore import x2_prompts as XP
from src.explore import x2_tools as XT

W, H = 128, 96
DISK, SQUARE = (1, "red disk"), (2, "blue square")


def _disk(cx: float, cy: float, r: float) -> np.ndarray:
    yy, xx = np.mgrid[0:H, 0:W] + 0.5
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r


def _rect(x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    m = np.zeros((H, W), bool)
    m[y0:y1, x0:x1] = True
    return m


def _bbox(m: np.ndarray) -> list[float]:
    ys, xs = np.nonzero(m)
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def _image(objs: list[tuple[np.ndarray, tuple[int, int, int]]], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = (np.stack(np.meshgrid(np.linspace(40, 120, W), np.linspace(60, 140, H)), -1).mean(-1)[..., None] + rng.normal(0, 6, (H, W, 3)))
    img = np.clip(img, 0, 255)
    for m, col in objs:
        img[m] = np.clip(np.asarray(col, float) + rng.normal(0, 5, (int(m.sum()), 3)), 0, 255)
    return img.astype(np.uint8)


def write_fixture(root: str | Path, *, split: str = "xm_tune") -> Path:
    """写图与 pairs.json，返回 pairs.json 的路径（图像路径相对它所在的目录）。"""
    root = Path(root)
    (root / "images").mkdir(parents=True, exist_ok=True)
    a_disks = [_disk(30, 30, 12), _disk(80, 60, 14)]
    a_sq = [_rect(95, 10, 120, 35)]
    b_disks = [_disk(40, 50, 15)]
    b_crowd = _disk(95, 30, 6) | _disk(105, 34, 6) | _disk(100, 44, 6)
    imgs = {"a.png": _image([(m, (220, 40, 40)) for m in a_disks] + [(a_sq[0], (40, 60, 220))], 0),
            "b.png": _image([(b_disks[0], (220, 40, 40)), (b_crowd, (220, 40, 40))], 1)}
    for name, arr in imgs.items():
        Image.fromarray(arr).save(root / "images" / name)
    pairs = []

    def add(img: str, iid: int, cat: tuple[int, str], masks: list[np.ndarray], crowd: list[np.ndarray] = ()):
        insts = [{"ann_id": iid * 100 + k, "box_xyxy": _bbox(m), "rle": mask_to_rle(m), "area": int(m.sum())} for k, m in enumerate(masks)]
        pairs.append({"id": f"fixture/{split}/{iid}/{cat[0]}", "source": "fixture", "split": split, "image": f"images/{img}", "width": W,
                      "height": H, "category_id": cat[0], "category": cat[1], "instances": insts,
                      "crowd_rles": [mask_to_rle(c) for c in crowd]})

    add("a.png", 1, DISK, a_disks)
    add("a.png", 1, SQUARE, a_sq)
    add("b.png", 2, DISK, b_disks, [b_crowd])
    p = root / "pairs.json"
    p.write_text(json.dumps(pairs, ensure_ascii=False), encoding="utf-8")
    return p


def image_id(img: Image.Image) -> str:
    return hashlib.sha256(np.asarray(img.convert("RGB")).tobytes()).hexdigest()[:16]


class ScriptedXMAdapter:
    """见模块说明。answers：{(image_id(图), 类别名): 真值框（原图像素）}。记下每次调用的提示、图数与 max_new_tokens。"""

    model_key = "fake-xm"

    def __init__(self, answers: dict[tuple[str, str], list[list[float]]], *, system: str = "xm", duplicate_first: bool = False,
                 extra_box: list[float] | None = None) -> None:
        self.answers, self.system, self.dup, self.extra = answers, system, duplicate_first, extra_box
        self.calls: list[dict[str, Any]] = []

    def build_inputs(self, images, instruction, order):
        return SimpleNamespace(images=list(images), instruction=instruction, order=order)

    def parse_boxes(self, text, *, width, height):
        from src.adapters.qwen35 import parse_boxes

        return parse_boxes(text, width=width, height=height)

    def _answer(self, img: Image.Image, prompt: str) -> str:
        cat = re.search(r'category "([^"]+)"', prompt).group(1)
        if prompt.startswith("Find every instance"):
            if "junk" in cat:
                return "There are two of them."
            boxes = [XP.box_q(b, W, H) for b in self.answers.get((image_id(img), cat), [])]
            if self.dup and boxes:
                boxes = boxes + [list(boxes[0])]
            if self.extra is not None:
                boxes = boxes + [XP.box_q(self.extra, W, H)]
            if self.system == "base":
                return "```json\n" + json.dumps([{"bbox_2d": b, "label": cat} for b in boxes]) + "\n```"
            return json.dumps(boxes)
        n = int(re.search(r"exactly (\d+) control points", prompt).group(1))
        boxes = [[int(v) for v in m.groups()] for m in re.finditer(r"^\d+: \[(\d+), (\d+), (\d+), (\d+)\]$", prompt, re.M)]
        sp = [XT.ellipse_ctrl_q(b, 1000, 1000, n) for b in boxes]  # 网格上的椭圆：宽高取 1000 即网格本身
        if "short" in cat and sp:
            sp[0] = sp[0][:-1]
        return json.dumps(sp)

    def generate_batch(self, inputs_list, *, enable_thinking, max_new_tokens, **kw):
        out = []
        for inp in inputs_list:
            text = self._answer(inp.images[0], inp.instruction)
            self.calls.append({"prompt": inp.instruction, "n_images": len(inp.images), "enable_thinking": enable_thinking,
                               "max_new_tokens": max_new_tokens, "batch": len(inputs_list)})
            vis = 64 * len(inp.images)
            out.append(SimpleNamespace(text=text, raw_text=text, thinking_unclosed=False, stopped=True,
                                       n_prompt_tokens=len(inp.instruction) // 4 + vis, n_generated_tokens=len(text) // 4 + 1,
                                       n_visual_tokens=vis, n_image_pixels=vis * 1024, wall_time_s=0.001 * len(inputs_list)))
        return out

    def generate(self, inputs, *, enable_thinking, max_new_tokens, **kw):
        return self.generate_batch([inputs], enable_thinking=enable_thinking, max_new_tokens=max_new_tokens, **kw)[0]


def gt_answers(pairs, data_root: str | Path) -> dict[tuple[str, str], list[list[float]]]:
    out = {}
    for p in pairs:
        with Image.open(Path(data_root) / p.image) as im:
            out[(image_id(im), p.category)] = [list(i.box_xyxy) for i in p.instances]
    return out
