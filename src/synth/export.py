"""可选的导出：图像 PNG、数组 npz、JSONL 清单。生成器本身按需即时生成，不要求落盘（N1）。

每个样本的文件名主干 stem = "{task}_{split}_{index:09d}"：
- {stem}.png：图像（e 没有图像，不写）；
- {stem}.npz：target_mask、target_grid、object_masks、baseline_<名>_mask、baseline_<名>_grid（压缩）；
- manifest.jsonl：每个样本一行，{"stem", "task", "split", "index", "instruction", "image", "arrays", "digest", "meta"}。
PNG 的字节依赖 zlib 版本，逐字节复现以解码后的数组（与 Sample.digest）为准。
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from PIL import Image

from src.synth.sample import Sample, canonical_json


def sample_stem(s: Sample) -> str:
    return f"{s.task}_{s.split}_{s.index:09d}"


def sample_arrays(s: Sample) -> dict[str, np.ndarray]:
    arrays = {"target_mask": s.target_mask, "target_grid": s.target_grid, "object_masks": s.object_masks}
    for name, p in sorted(s.baselines.items()):
        arrays[f"baseline_{name}_mask"] = p.mask
        arrays[f"baseline_{name}_grid"] = p.grid
    return arrays


def export_samples(samples: Iterable[Sample], out_dir: str | Path, *, manifest_name: str = "manifest.jsonl") -> Path:
    """写出样本，返回清单路径。清单以追加方式写入，同一目录可以分批导出。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / manifest_name
    with manifest.open("a", encoding="utf-8") as fh:
        for s in samples:
            stem = sample_stem(s)
            image_name = None
            if s.image is not None:
                image_name = f"{stem}.png"
                Image.fromarray(s.image, mode="RGB").save(out / image_name)
            np.savez_compressed(out / f"{stem}.npz", **sample_arrays(s))
            row = {
                "stem": stem,
                "task": s.task,
                "split": s.split,
                "index": s.index,
                "instruction": s.instruction,
                "image": image_name,
                "arrays": f"{stem}.npz",
                "digest": s.digest(),
                "meta": s.meta,
            }
            fh.write(canonical_json(row) + "\n")
    return manifest


def load_manifest(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
