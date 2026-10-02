"""E5 的外部预测（LISA 式对照系统，简报 §4 第 1 条例外、§8.8；DEV-0018）：导出输入、读入 mask，只做评分。

硬约束 1：外部系统的输出绝不能进入我们的训练数据、标签或奖励。所以
- 外部预测放在独立目录（"外部预测目录"），目录里必须有标记文件 EXTERNAL_MARKER；本模块只在评分时读它。
- 只有 ALLOWED_READERS 里的文件可以 import 本模块或引用标记文件名（tests/test_e5_external.py 静态扫描 src/ 与 scripts/）；
  生成器、E5 的数据与提示代码、任何训练代码都不读这个目录（同一测试在打开文件的系统调用上设桩，跑一遍评测集的生成、
  提示渲染与基线评分，核对没有任何路径落进外部预测目录）。

外部预测目录的格式（format_version 1）
- EXTERNAL_PREDICTIONS.json：{"format_version": 1, "system": "<外部系统名，如 lisa-7b-v1>", "note": "…"}，可另加任意说明字段。
- <任务>.jsonl（任务用规范名：rotation3d、viewpoint、occlusion、physics、imagery）：每行一条，三种之一：
  * {"id": "rotation3d/test/12", "mask_rle": {"size": [H, W], "counts": "<COCO 压缩 RLE>"}}（或未压缩 RLE：counts 为整数列表）；
  * {"id": "...", "mask_png": "<相对本目录的 PNG 路径>"}：单通道或调色板图，非零 = 前景；也接受 RGB，转灰度后非零 = 前景；
  * {"id": "...", "failure": "<原因>"}：外部系统没有给出 mask。
  mask 的尺寸必须等于画布（生成器默认 512×512）。同一 id 出现两次报错。评测集里有、文件里没有的 id 记失败 "missing"；
  文件里有、评测集里没有的 id 忽略，条数记进统计。
- 评测集的输入可以用 export_inputs 导出（图像 PNG + inputs.jsonl：id、图像文件、生成器指令、被点名物体的描述、画布尺寸），
  交给外部系统运行；导出目录与外部预测目录是两个目录。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

from src.executors.raster import rle_to_mask

EXTERNAL_MARKER = "EXTERNAL_PREDICTIONS.json"
EXTERNAL_FORMAT_VERSION = 1
# 可以读外部预测目录的文件（相对仓库根）。其余代码一律不许 import 本模块、不许引用 EXTERNAL_MARKER。
ALLOWED_READERS = ("src/e5/external.py", "src/e5/runner.py", "scripts/run_e5.py")
MISSING = "missing"


class ExternalPredictionError(ValueError):
    pass


def read_manifest(root: str | Path) -> dict[str, Any]:
    p = Path(root) / EXTERNAL_MARKER
    if not p.is_file():
        raise ExternalPredictionError(f"{root} 不是外部预测目录：没有标记文件 {EXTERNAL_MARKER}")
    doc = json.loads(p.read_text(encoding="utf-8"))
    if doc.get("format_version") != EXTERNAL_FORMAT_VERSION or not isinstance(doc.get("system"), str) or not doc["system"]:
        raise ExternalPredictionError(f"{p}：format_version 必须是 {EXTERNAL_FORMAT_VERSION}，system 必须是非空字符串")
    return doc


def _mask_from_entry(root: Path, e: dict[str, Any], height: int, width: int) -> np.ndarray:
    if "mask_rle" in e:
        rle = dict(e["mask_rle"])
        m = rle_to_mask(rle)
    elif "mask_png" in e:
        path = (root / e["mask_png"]).resolve()
        if root.resolve() not in path.parents:
            raise ExternalPredictionError(f"{e['id']}：PNG 路径 {e['mask_png']!r} 不在外部预测目录里")
        with Image.open(path) as im:
            arr = np.asarray(im.convert("L") if im.mode not in ("1", "L", "P", "I", "I;16") else im)
        m = arr != 0
    else:
        raise ExternalPredictionError(f"{e.get('id')}：要给 mask_rle、mask_png 或 failure 之一")
    if m.shape != (height, width):
        raise ExternalPredictionError(f"{e['id']}：mask 尺寸 {m.shape} 与画布 {(height, width)} 不同")
    return m.astype(bool)


def load_task(root: str | Path, task: str, ids: Iterable[str], canvas_hw: tuple[int, int]) -> tuple[dict[str, np.ndarray | str], dict[str, Any]]:
    """读一个任务的外部预测：返回 (id → mask 或失败原因字符串, 统计)。只读 ids 里的样本；规则见模块说明。"""
    root = Path(root)
    manifest = read_manifest(root)
    path = root / f"{task}.jsonl"
    want = list(ids)
    wanted = set(want)
    got: dict[str, np.ndarray | str] = {}
    n_extra = 0
    if path.is_file():
        for k, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            e = json.loads(line)
            sid = e.get("id")
            if not isinstance(sid, str):
                raise ExternalPredictionError(f"{path}:{k + 1}：缺 id")
            if sid in got:
                raise ExternalPredictionError(f"{path}：id {sid} 出现了两次")
            if sid not in wanted:
                n_extra += 1
                got[sid] = "__extra__"
                continue
            if "failure" in e:
                got[sid] = str(e["failure"])
            else:
                got[sid] = _mask_from_entry(root, e, *canvas_hw)
    out = {sid: got.get(sid, MISSING) for sid in want}
    stats = {"system": manifest["system"], "file": path.name, "file_exists": path.is_file(),
             "n_expected": len(want), "n_missing": sum(1 for sid in want if sid not in got), "n_extra_ignored": n_extra,
             "n_failure": sum(1 for v in out.values() if isinstance(v, str))}
    return out, stats


def export_inputs(samples: Iterable[Any], out_dir: str | Path) -> Path:
    """把评测样本导出给外部系统：<out>/images/<任务>_<序号>.png（e 没有图像，不写）与 <out>/inputs.jsonl。返回 inputs.jsonl 的路径。"""
    from src.e5.protocol import sample_id

    out = Path(out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    lines = []
    for s in samples:
        sid = sample_id(s.task, s.split, s.index)
        img = None
        if s.image is not None:
            img = f"images/{s.task}_{s.split}_{s.index:09d}.png"
            Image.fromarray(s.image).save(out / img)
        lines.append(json.dumps({"id": sid, "task": s.task, "image": img, "instruction": s.instruction,
                                 "target_description": s.meta.get("target_description"),
                                 "canvas_hw": list(s.target_mask.shape)}, ensure_ascii=False))
    p = out / "inputs.jsonl"
    p.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return p
