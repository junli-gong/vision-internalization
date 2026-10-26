"""X-2 测试用的假数据（DEV-0069）：tests/_w2c_fakes 的 §5 格式假行，加上 DEV-0060 的行里有、而 W2-c 训练不读的 gt_rle（真值 mask 的
COCO RLE），键序同 src/w2/w2c_data.ROW_KEYS。图像是 E3 训练池的 fixture（tests/fixtures/e3/root，128×96 的合成图形）。

命令行：python -m tests._x2_fakes <dir>（写 <dir>/*.jsonl 与 manifest.json）。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

from src.e1.scoring import gt_mask
from src.executors.raster import mask_to_rle
from src.w2.config import W2_CONFIG_PATH, W2Config, load_w2_config
from src.w2.w2c_data import ROW_KEYS
from tests import _w2c_fakes as FK

ROOT = FK.ROOT


def fake_rows(w2: W2Config, *, n: int | None = None, failed: Sequence[int] = (1, 7), first_not_ok: Sequence[int] = (2, 9)) -> list[dict[str, Any]]:
    pool = FK.train_pool()[:n]
    out = []
    for i, s in enumerate(pool):
        r = FK.make_row(s, w2, failed=i in failed, first_not_ok=i in first_not_ok, rot=i)
        r["gt_rle"] = mask_to_rle(gt_mask(s))
        out.append({k: r[k] for k in ROW_KEYS})
    return out


def write_fake_w2c_data(out_dir: str | Path, *, w2_config: str | Path = W2_CONFIG_PATH, n: int | None = None, shard_size: int = 12,
                        failed: Sequence[int] = (1, 7), first_not_ok: Sequence[int] = (2, 9)) -> Path:
    w2 = load_w2_config(w2_config)
    return FK.write_rows(out_dir, fake_rows(w2, n=n, failed=failed, first_not_ok=first_not_ok), shard_size=shard_size)


# ---------------------------------------------------------------------------
# 推理的假适配器（tests/test_x2_runner.py）：关思考的 build_inputs + generate_batch，按提示的种类与指代句给确定的回答
# ---------------------------------------------------------------------------


class ScriptedX2Adapter:
    """描样条：回答提示里那个框的内切椭圆（n 个网格控制点，n 读自提示）；指修正框：按指代句里的关键词——"+" 整个框（plus）、[]（empty）、
    不合格式的文本（junk）、其余给一个 "-" 的左上四分之一框；挑选：回答 select（缺省 "2"）。记下每次调用的提示、图数与 max_new_tokens。"""

    model_key = "fake-x2"

    def __init__(self, select: str = "2") -> None:
        self.select = select
        self.calls: list[dict[str, Any]] = []

    def build_inputs(self, images, instruction, order):
        return SimpleNamespace(images=list(images), instruction=instruction, order=order)

    def _answer(self, prompt: str) -> str:
        import re

        from src.explore import x2_prompts as XP
        from src.explore import x2_tools as XT

        found = re.search(r"\[(\d+), (\d+), (\d+), (\d+)\]", prompt)
        box = None if found is None else [int(v) for v in found.groups()]
        if prompt.startswith("The object"):
            n = int(re.search(r"exactly (\d+) control points", prompt).group(1))
            return XP.spline_text(XT.ellipse_ctrl_q(box, 1000, 1000, n))  # 网格上的椭圆：宽高取 1000 即网格本身
        if "current mask" in prompt:
            if "plus" in prompt:
                return XP.corrections_text([{"sign": "+", "box": box}])
            if "empty" in prompt:
                return "[]"
            if "junk" in prompt:
                return "I would remove the left part."
            x0, y0, x1, y1 = box
            return XP.corrections_text([{"sign": "-", "box": [x0, y0, (x0 + x1) // 2, (y0 + y1) // 2]}])
        return self.select

    def generate_batch(self, inputs_list, *, enable_thinking, max_new_tokens, **kw):
        out = []
        for inp in inputs_list:
            text = self._answer(inp.instruction)
            self.calls.append({"prompt": inp.instruction, "n_images": len(inp.images), "enable_thinking": enable_thinking,
                               "max_new_tokens": max_new_tokens, "image_sizes": [im.size for im in inp.images]})
            vis = 64 * len(inp.images)
            out.append(SimpleNamespace(text=text, raw_text=text, thinking_unclosed=False, stopped=True, n_prompt_tokens=len(inp.instruction) // 4 + vis,
                                       n_generated_tokens=len(text) // 4 + 1, n_visual_tokens=vis, n_image_pixels=vis * 1024,
                                       wall_time_s=0.001 * len(inputs_list)))
        return out

    def generate(self, inputs, *, enable_thinking, max_new_tokens, **kw):
        return self.generate_batch([inputs], enable_thinking=enable_thinking, max_new_tokens=max_new_tokens, **kw)[0]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out_dir")
    p.add_argument("--w2-config", default=str(W2_CONFIG_PATH))
    a = p.parse_args(argv)
    d = write_fake_w2c_data(a.out_dir, w2_config=a.w2_config)
    print(f"写出 {d}（{sum(1 for _ in d.glob('*.jsonl'))} 个分片）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
