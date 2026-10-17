"""E2/E3 训练数据的加载器（DEV-0022 任务书 §1）：每个落地源一个，返回统一的记录（src/data/records.Sample）。

共同行为：
- 训练划分（refcoco_train、nyu_depth_train、ade20k/train、nyu_normals/train）一律读去重报告的训练剔除清单
  （默认 <root>/dedupe/train_exclusions.json，src/data/train_dedupe），自动排除其中的样本，逐条写进
  stats["dedupe_excluded"]。清单不存在时报错（硬约束 7：没做去重不训练）；清单在、但没有这个训练集的键（没比较过）也报错
  （DEV-0024）；require_dedupe=False 只给开发机小样本与测试用。
- 评测划分（ade20k/val、nyu_normals/test）不读剔除清单。
- strict=True（默认）时图像或 GT 文件缺失直接报错；strict=False 时计入 stats["n_missing"] 并跳过（--limit 小样本用）。
  文件在不在按父目录一次性列出来查（_DirListing，DEV-0047），结果与逐个 Path.exists() 相同。
- 稠密 GT 不在加载时读进内存：gt 里放文件路径，用 read_depth_m / read_normals / read_ade_label 解码：
  深度 → 米（float32，0 = 无效）；法向 → 单位向量（float32 H×W×3）与有效掩码；ADE 标签 → uint8，0 到 src/constants.ADE_NUM_CLASSES（0 不参与评测）。
- RefCOCO 的 gt 与 E1 相同：框是绝对像素 xyxy（浮点），mask 是 COCO 多边形或 RLE，附 height、width。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from src.data.land import ade20k as ade_land
from src.data.land import nyu_depth_train as nyu_land
from src.data.land.manifest import read_manifest
from src.data.land.refcoco import SPEC as REFCOCO_SPEC
from src.data.land.refcoco_train import DATASETS as REFCOCO_DATASETS
from src.data.land.refcoco_train import SOURCE as REFCOCO_TRAIN
from src.data.records import LoadResult, Sample, xywh_to_xyxy
from src.data.train_dedupe import EXCLUSIONS_REL, TRAIN_SETS, load_exclusions


def _train_exclusions(root: Path, set_name: str, exclusions: str | Path | None, require_dedupe: bool) -> tuple[dict[str, dict[str, Any]], str | None]:
    if set_name not in TRAIN_SETS:
        raise ValueError(f"{set_name} 不是训练集")
    path = root / EXCLUSIONS_REL if exclusions == "auto" else (Path(exclusions) if exclusions else None)
    if path is None or not path.exists():
        if require_dedupe:
            raise FileNotFoundError(f"{set_name}：没有训练剔除清单 {path}（先跑 scripts/data/dedupe_report.py，--out <root>/dedupe）；硬约束 7：没做去重不训练")
        return {}, None
    excl = load_exclusions(path)
    if set_name not in excl:
        if require_dedupe:
            raise ValueError(f"{set_name}：训练剔除清单 {path} 里没有这个训练集的键（有 {sorted(excl)}），即这个训练集没有和评测集"
                             "比较过；把它的哈希清单一起给 scripts/data/dedupe_report.py 重新比较；硬约束 7：没做去重不训练")
        return {}, str(path)
    return excl[set_name], str(path)


def _finish(samples: list[Sample], rel_of: Callable[[Sample], str], *, set_name: str, excl: dict[str, dict[str, Any]], excl_path: str | None, n_missing: int, extra: dict[str, Any] | None = None) -> LoadResult:
    kept, dropped = [], []
    for s in samples:
        rec = excl.get(rel_of(s))
        if rec is None:
            kept.append(s)
        else:
            dropped.append({"id": s.id, "path": rec["path"], "key": rec.get("key"), "reasons": rec.get("reasons")})
    stats = {
        "set": set_name,
        "n_total": len(samples) + n_missing,
        "n_missing": n_missing,
        "dedupe_file": excl_path,
        "n_dedupe_excluded": len(dropped),
        "dedupe_excluded": dropped,
        "n_kept": len(kept),
        **(extra or {}),
    }
    return LoadResult(sorted(kept, key=lambda s: s.id), stats)


class _DirListing:
    """一次加载器调用内的"磁盘上有没有"（DEV-0047）：每个父目录 os.scandir 一次，在内存里按名字查，代替逐个 Path.exists()
    （共享盘的元数据慢时，逐个 stat 十几万次要几个小时；按目录列只要几百次）。结果与 Path.exists() 相同（Linux）：
    - 名字按字节比较（scandir 与 exists 用同一套文件系统编码；不折叠大小写、不做 Unicode 规范化）；
    - 列出来是符号链接的条目：这一条退回 Path.exists()（跟随链接，断链仍判不存在）；
    - 父目录不存在或不是目录（FileNotFoundError、NotADirectoryError）：其下全都不存在（exists() 同样是 False）；
      列不出来的其他原因（权限、链接环路、非法路径等）：这个目录退回逐个 Path.exists()，结果与异常都同旧做法；
    - 名字是空、"."、".." 的路径（scandir 不列这些）：退回 Path.exists()。
    缓存只活在一次加载器调用里（加载期间落地不变）。"""

    def __init__(self) -> None:
        self._dirs: dict[Path, dict[str, os.DirEntry[str]] | None] = {}

    def _entries(self, d: Path) -> dict[str, os.DirEntry[str]] | None:
        if d not in self._dirs:
            try:
                with os.scandir(d) as it:
                    self._dirs[d] = {e.name: e for e in it}
            except (FileNotFoundError, NotADirectoryError):
                self._dirs[d] = {}
            except (OSError, ValueError):
                self._dirs[d] = None
        return self._dirs[d]

    def exists(self, path: Path) -> bool:
        if path.name in ("", ".", ".."):
            return path.exists()
        entries = self._entries(path.parent)
        if entries is None:
            return path.exists()
        e = entries.get(path.name)
        if e is None:
            return False
        return path.exists() if e.is_symlink() else True


def _have(src: Path, rels: tuple[str, ...], files: dict[str, Any], strict: bool, what: str, disk: _DirListing) -> bool:
    """rels 都在清单 files 里、且磁盘上存在（disk：本次加载器调用的 _DirListing）。"""
    if all(r in files and disk.exists(src / r) for r in rels):
        return True
    if strict:
        raise FileNotFoundError(f"{what}：{[r for r in rels if not disk.exists(src / r)]} 不存在；落地不全（strict=False 可跳过并计数）")
    return False


# ---------------------------------------------------------------------------
# RefCOCO 训练（一句一条；D11-1 已在落地时做完）
# ---------------------------------------------------------------------------


def load_refcoco_train(root: str | Path, dataset: str, *, exclusions: str | Path | None = "auto", require_dedupe: bool = True, strict: bool = True) -> LoadResult:
    if dataset not in REFCOCO_DATASETS:
        raise ValueError(f"dataset 必须是 {REFCOCO_DATASETS}")
    root = Path(root)
    src = root / REFCOCO_TRAIN
    split_by = REFCOCO_SPEC["datasets"][dataset]["split_by"]
    doc = json.loads((src / f"{dataset}_{split_by}_train.json").read_text())
    files = read_manifest(src)["files"]
    excl, excl_path = _train_exclusions(root, f"{REFCOCO_TRAIN}/train", exclusions, require_dedupe)
    cats = doc["categories"]
    out, n_missing, disk = [], 0, _DirListing()
    for r in doc["refs"]:
        ann = doc["anns"][str(r["ann_id"])]
        im = doc["images"][str(r["image_id"])]
        rel = f"images/{im['file_name']}"
        seg = ann["segmentation"]
        mask = {"polygons": seg} if isinstance(seg, list) else {"rle": seg}
        mask.update(height=im["height"], width=im["width"])
        if not _have(src, (rel,), files, strict, f"{dataset} ref {r['ref_id']}", disk):
            n_missing += len(r["sentences"])
            continue
        for s in r["sentences"]:
            out.append(
                Sample(
                    id=f"{dataset}_{split_by}/train/{s['sent_id']}",
                    task="refexp",
                    images=(str(src / rel),),
                    image_keys=(f"coco:{r['image_id']}",),
                    phash=(files[rel].get("phash"),),
                    fields={
                        "expression": s["sent"],
                        "expression_raw": s["raw"],
                        "sent_id": s["sent_id"],
                        "ref_id": r["ref_id"],
                        "ann_id": r["ann_id"],
                        "image_id": r["image_id"],
                        "category": cats[str(r["category_id"])],
                        "image_size": [im["width"], im["height"]],
                    },
                    gt={"box": xywh_to_xyxy(ann["bbox"]), "mask": mask, "iscrowd": int(ann["iscrowd"])},
                    source=dataset,
                    split="train",
                )
            )
    return _finish(out, lambda s: "images/" + Path(s.images[0]).name, set_name=f"{REFCOCO_TRAIN}/train", excl=excl, excl_path=excl_path, n_missing=n_missing, extra={"dataset": dataset})


# ---------------------------------------------------------------------------
# NYU 深度训练（BTS 清单，D11-2 已在落地时做完）
# ---------------------------------------------------------------------------


def read_depth_m(path: str | Path, scale: float = nyu_land.SPEC["depth_scale"]) -> np.ndarray:
    """BTS 的 16 位深度 PNG → 米（float32，H×W）；0 表示无效。"""
    with Image.open(path) as im:
        a = np.asarray(im)
    if a.dtype != np.uint16 or a.ndim != 2:
        raise ValueError(f"{path}：应为 16 位单通道 PNG，实际 {a.dtype} {a.shape}")
    return (a.astype(np.float32) / np.float32(scale)).astype(np.float32)


def load_nyu_depth_train(root: str | Path, *, exclusions: str | Path | None = "auto", require_dedupe: bool = True, strict: bool = True) -> LoadResult:
    """D11-2 复查（DEV-0024，D53-3 的小问题 C）：每一条的基础场景名（nyu_land.base_scene(scene)，必须与清单里的 base_scene 相同）
    都不能在官方 test 场景集合 nyu_land.TEST_SCENES 里；不满足就报错（落地时已按场景过滤，这里不静默跳过）。"""
    root = Path(root)
    src = root / nyu_land.SOURCE
    files = read_manifest(src)["files"]
    excl, excl_path = _train_exclusions(root, f"{nyu_land.SOURCE}/train", exclusions, require_dedupe)
    H, W = nyu_land.SPEC["hw"]
    test_scenes = frozenset(nyu_land.TEST_SCENES)
    out, n_missing, disk = [], 0, _DirListing()
    for line in (src / "pairs_train.jsonl").read_text().splitlines():
        p = json.loads(line)
        base = nyu_land.base_scene(p["scene"])
        if base != p["base_scene"]:
            raise ValueError(f"nyu pair {p['id']}：清单的 base_scene {p['base_scene']!r} 与按场景名算出的 {base!r} 不同")
        if base in test_scenes:
            raise ValueError(f"nyu pair {p['id']}：基础场景 {base!r} 属于官方 test 场景（D11-2），不能进训练集；落地的场景过滤没生效")
        if not _have(src, (p["rgb"], p["depth"]), files, strict, f"nyu pair {p['id']}", disk):
            n_missing += 1
            continue
        out.append(
            Sample(
                id=f"{nyu_land.SOURCE}/train/{p['id']}",
                task="depth",
                images=(str(src / p["rgb"]),),
                image_keys=(f"nyuraw:{p['id']}",),
                phash=(files[p["rgb"]].get("phash"),),
                fields={"scene": p["scene"], "base_scene": p["base_scene"], "frame": p["frame"], "focal": p["focal"], "image_size": [W, H]},
                gt={"depth": str(src / p["depth"]), "depth_units": "meters", "depth_scale": nyu_land.SPEC["depth_scale"], "invalid_value": 0.0, "decode": "src.data.train_loaders.read_depth_m"},
                source=nyu_land.SOURCE,
                split="train",
            )
        )
    return _finish(out, lambda s: Path(s.images[0]).relative_to(src).as_posix(), set_name=f"{nyu_land.SOURCE}/train", excl=excl, excl_path=excl_path, n_missing=n_missing)


# ---------------------------------------------------------------------------
# NYU 法向（Bae 2021 / GeoNet；train 795 / test 654）
# ---------------------------------------------------------------------------


def read_normals(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """法向 PNG → (单位法向 float32 H×W×3，有效掩码 bool H×W)。无效像素（RGB 全 0）的法向置 0。"""
    with Image.open(path) as im:
        a = np.asarray(im.convert("RGB"))
    valid = ~np.all(a == 0, axis=-1)
    n = a.astype(np.float32) / np.float32(255.0) * np.float32(2.0) - np.float32(1.0)
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    n = np.where(valid[..., None] & (norm > 0), n / np.maximum(norm, np.float32(1e-12)), np.float32(0.0)).astype(np.float32)
    return n, valid


def load_nyu_normals(root: str | Path, split: str, *, exclusions: str | Path | None = "auto", require_dedupe: bool = True, strict: bool = True) -> LoadResult:
    from src.data.land import nyu_normals as nn_land

    if split not in nn_land.SPLITS:
        raise ValueError(f"split 必须是 {nn_land.SPLITS}")
    root = Path(root)
    src = root / nn_land.SOURCE
    files = read_manifest(src)["files"]
    set_name = f"{nn_land.SOURCE}/{split}"
    excl, excl_path = _train_exclusions(root, set_name, exclusions, require_dedupe) if nn_land.ROLE[split] == "train" else ({}, None)
    H, W = nn_land.SPEC["hw"]
    out, n_missing, disk = [], 0, _DirListing()
    for r in json.loads((src / "index.json").read_text())["images"]:
        if r["split"] != split:
            continue
        if not _have(src, (r["img"], r["norm"]), files, strict, f"nyu_normals {r['idx']}", disk):
            n_missing += 1
            continue
        out.append(
            Sample(
                id=f"{set_name}/{r['idx']}",
                task="normal",
                images=(str(src / r["img"]),),
                image_keys=(f"nyuv2:{r['idx']}",),
                phash=(files[r["img"]].get("phash"),),
                fields={"idx": r["idx"], "image_size": [W, H]},
                gt={"normal": str(src / r["norm"]), "decode": "src.data.train_loaders.read_normals", "invalid": "RGB == (0,0,0)"},
                source=nn_land.SOURCE,
                split=split,
            )
        )
    return _finish(out, lambda s: Path(s.images[0]).relative_to(src).as_posix(), set_name=set_name, excl=excl, excl_path=excl_path, n_missing=n_missing, extra={"role": nn_land.ROLE[split]})


# ---------------------------------------------------------------------------
# ADE20K（SceneParse150）
# ---------------------------------------------------------------------------


def read_ade_label(path: str | Path) -> np.ndarray:
    """ADE 标注 PNG → uint8 H×W，取值 0 到 ADE_NUM_CLASSES（0 = other objects，不参与评测）。"""
    with Image.open(path) as im:
        if im.mode != "L":
            raise ValueError(f"{path}：标注应为 mode L，实际 {im.mode}")
        a = np.asarray(im)
    if a.max(initial=0) > ade_land.N_CLASSES:
        raise ValueError(f"{path}：标签超过 {ade_land.N_CLASSES}")
    return a


def load_ade20k(root: str | Path, split: str, *, exclusions: str | Path | None = "auto", require_dedupe: bool = True, strict: bool = True) -> LoadResult:
    if split not in ade_land.SPLIT_DIRS:
        raise ValueError(f"split 必须是 {tuple(ade_land.SPLIT_DIRS)}")
    root = Path(root)
    src = root / ade_land.SOURCE
    files = read_manifest(src)["files"]
    set_name = f"{ade_land.SOURCE}/{split}"
    excl, excl_path = _train_exclusions(root, set_name, exclusions, require_dedupe) if ade_land.ROLE[split] == "train" else ({}, None)
    out, n_missing, disk = [], 0, _DirListing()
    for line in (src / f"{split}.jsonl").read_text().splitlines():
        r = json.loads(line)
        if not _have(src, (r["image"], r["label"]), files, strict, f"ade20k {r['stem']}", disk):
            n_missing += 1
            continue
        out.append(
            Sample(
                id=f"{set_name}/{r['stem']}",
                task="semseg",
                images=(str(src / r["image"]),),
                image_keys=(f"ade20k:{r['stem']}",),
                phash=(files[r["image"]].get("phash"),),
                fields={"stem": r["stem"], "scene": r.get("scene"), "image_size": [r["width"], r["height"]]},
                gt={"label": str(src / r["label"]), "n_classes": ade_land.N_CLASSES, "ignore_index": 0, "decode": "src.data.train_loaders.read_ade_label"},
                source=ade_land.SOURCE,
                split=split,
            )
        )
    return _finish(out, lambda s: Path(s.images[0]).relative_to(src).as_posix(), set_name=set_name, excl=excl, excl_path=excl_path, n_missing=n_missing, extra={"role": ade_land.ROLE[split]})
