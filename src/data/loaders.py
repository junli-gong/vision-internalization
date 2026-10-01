"""E1 每个任务一个加载器，返回统一的记录（src/data/records.Sample；DEV-0012 任务书 §4）。

共同行为：
- subset="eval"（默认）：剔除开发子集；"dev"：只要开发子集；"all"：不按开发子集剔除（用于统计总体）。
- 一律剔除 D20 的探针用图（规范键 + 感知哈希两路），可再传 extra_exclusion（例如以后的训练集图像）。
- 被剔除的样本逐条写进 LoadResult.stats["exclusion"]，不静默丢弃；stats 里写明各原因剔除了多少、剩余的 n。
- strict=True（默认）时，落地不全（图像文件缺失）直接报错；strict=False 时把缺图的样本计入 stats["n_missing_image"] 并跳过
  （只给开发机小样本用）。
- 不做任何模型相关的事；提示的拼法属于 E1 评测框架。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Iterable

from src.data.dedupe import REASON_DEV, Exclusion, apply_exclusions, probe_exclusion
from src.data.land.manifest import read_manifest
from src.data.records import LoadResult, Sample, xywh_to_xyxy
from src.data.subsets import DEV_SUBSETS_PATH, load_dev_subsets, sort_ids

SUBSETS = ("eval", "dev", "all")


# ---------------------------------------------------------------------------
# 共用
# ---------------------------------------------------------------------------


def _file_index(src_dir: Path) -> dict[str, dict[str, Any]]:
    return read_manifest(src_dir)["files"]


def _phash(files: dict[str, dict[str, Any]], rel: str) -> str | None:
    rec = files.get(rel)
    return rec.get("phash") if rec else None


def _dev_ids(name: str, dev_config: str | Path | None) -> set[str]:
    cfg = load_dev_subsets(dev_config or DEV_SUBSETS_PATH)
    if name not in cfg["subsets"]:
        raise KeyError(f"开发子集配置里没有 {name}")
    return set(cfg["subsets"][name]["ids"])


def _finish(
    samples: list[Sample],
    native_id: Callable[[Sample], str],
    *,
    subset: str,
    dev_name: str | None,
    dev_config: str | Path | None,
    extra_exclusion: Exclusion | None,
    n_missing: int,
    extra_stats: dict[str, Any] | None = None,
) -> LoadResult:
    """按 subset 处理开发子集，再做 D20 与额外的剔除；返回带完整统计的 LoadResult。"""
    if subset not in SUBSETS:
        raise ValueError(f"subset 必须是 {SUBSETS}，收到 {subset!r}")
    stats: dict[str, Any] = {"n_total": len(samples) + n_missing, "n_missing_image": n_missing}
    dev = _dev_ids(dev_name, dev_config) if (dev_name and subset != "all") else set()
    if subset == "dev":
        if not dev_name:
            raise ValueError("这个数据源没有开发子集")
        kept_dev = [s for s in samples if native_id(s) in dev]
        stats["n_not_in_dev"] = len(samples) - len(kept_dev)
        samples = kept_dev
    ex = probe_exclusion()
    if subset == "eval" and dev:
        ex.add_ids((s.id for s in samples if native_id(s) in dev), REASON_DEV)
    if extra_exclusion is not None:
        ex.keys.update({k: v for k, v in extra_exclusion.keys.items() if k not in ex.keys})
        ex.phashes.update({k: v for k, v in extra_exclusion.phashes.items() if k not in ex.phashes})
        ex.ids.update({k: v for k, v in extra_exclusion.ids.items() if k not in ex.ids})
    kept, ex_stats = apply_exclusions(samples, ex)
    stats["exclusion"] = ex_stats
    stats["n_dev_excluded"] = ex_stats["excluded_by_reason"].get(REASON_DEV, 0)
    stats["n_probe_excluded"] = ex_stats["excluded_by_reason"].get("D20 探针用图", 0)
    stats["n_kept"] = len(kept)
    stats["subset"] = subset
    if extra_stats:
        stats.update(extra_stats)
    order = {i: k for k, i in enumerate(sort_ids(s.id for s in kept))}
    return LoadResult(sorted(kept, key=lambda s: order[s.id]), stats)


def _exists(root: Path, rel: str | None, strict: bool, what: str) -> bool:
    if rel and (root / rel).exists():
        return True
    if strict:
        raise FileNotFoundError(f"{what}：图像 {root / rel if rel else '(未落地)'} 不存在；落地不全（strict=False 可跳过并计数）")
    return False


# ---------------------------------------------------------------------------
# RefCOCO 系列：指代定位（框）与指代分割（多边形）共用一个加载器，一句一条
# ---------------------------------------------------------------------------

REFCOCO_FILES = {"refcoco": "refcoco_unc.json", "refcoco+": "refcoco+_unc.json", "refcocog": "refcocog_umd.json"}


def refcoco_dev_name(dataset: str) -> str:
    return {"refcoco": "refcoco_unc_val", "refcoco+": "refcoco+_unc_val", "refcocog": "refcocog_umd_val"}[dataset]


def load_refcoco(
    root: str | Path,
    dataset: str,
    split: str,
    *,
    subset: str = "eval",
    extra_exclusion: Exclusion | None = None,
    dev_config: str | Path | None = None,
    strict: bool = True,
) -> LoadResult:
    src = Path(root) / "refcoco"
    doc = json.loads((src / REFCOCO_FILES[dataset]).read_text())
    if split not in doc["eval_splits"]:
        raise ValueError(f"{dataset} 的评测划分是 {doc['eval_splits']}，没有 {split}")
    files = _file_index(src)
    cats = doc["categories"]
    out, n_missing = [], 0
    for r in doc["refs"]:
        if r["split"] != split:
            continue
        ann = doc["anns"][str(r["ann_id"])]
        im = doc["images"][str(r["image_id"])]
        rel = f"images/{im['file_name']}"
        seg = ann["segmentation"]
        mask = {"polygons": seg} if isinstance(seg, list) else {"rle": seg}
        mask.update(height=im["height"], width=im["width"])
        for s in r["sentences"]:
            if not _exists(src, rel if rel in files else None, strict, f"{dataset} sent {s['sent_id']}"):
                n_missing += 1
                continue
            out.append(
                Sample(
                    id=f"{dataset}_{doc['splitBy']}/{split}/{s['sent_id']}",
                    task="refexp",
                    images=(str(src / rel),),
                    image_keys=(f"coco:{r['image_id']}",),
                    phash=(_phash(files, rel),),
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
                    split=split,
                )
            )
    dev_name = refcoco_dev_name(dataset) if split == "val" else None
    if subset == "dev" and dev_name is None:
        raise ValueError(f"{dataset} 的开发子集只在 val 上")
    return _finish(out, lambda s: str(s.fields["sent_id"]), subset=subset, dev_name=dev_name, dev_config=dev_config, extra_exclusion=extra_exclusion, n_missing=n_missing)


# ---------------------------------------------------------------------------
# 开放词汇检测：COCO val2017 子集、ODinW13 子集（一图一条）
# ---------------------------------------------------------------------------


def _ovd_samples(src: Path, files: dict, coco: dict, *, source: str, split: str, id_prefix: str, rel_dir: str, key_fn, cat_names: list[str], strict: bool) -> tuple[list[Sample], int]:
    cats = sorted(coco["categories"], key=lambda c: c["id"])
    name_of = {c["id"]: n for c, n in zip(cats, cat_names)}
    anns: dict[int, list[dict[str, Any]]] = {}
    for a in coco["annotations"]:
        anns.setdefault(a["image_id"], []).append(a)
    out, n_missing = [], 0
    for im in sorted(coco["images"], key=lambda im: im["id"]):
        rel = f"{rel_dir}/{im['file_name']}"
        if not _exists(src, rel if rel in files else None, strict, f"{source} image {im['id']}"):
            n_missing += 1
            continue
        boxes = [
            {"box": xywh_to_xyxy(a["bbox"]), "category_id": a["category_id"], "category": name_of[a["category_id"]], "iscrowd": int(a.get("iscrowd", 0)), "ann_id": a["id"]}
            for a in sorted(anns.get(im["id"], []), key=lambda a: a["id"])
        ]
        out.append(
            Sample(
                id=f"{id_prefix}/{im['id']}",
                task="ovd",
                images=(str(src / rel),),
                image_keys=(key_fn(im),),
                phash=(_phash(files, rel),),
                fields={
                    "categories": [name_of[c["id"]] for c in cats],
                    "category_ids": [c["id"] for c in cats],
                    "image_id": im["id"],
                    "image_size": [im["width"], im["height"]],
                },
                gt={"boxes": boxes},
                source=source,
                split=split,
            )
        )
    return out, n_missing


def load_coco_ovd(root: str | Path, *, extra_exclusion: Exclusion | None = None, strict: bool = True) -> LoadResult:
    """COCO val2017 的 500 张子集。stats["coco_json"] 是子集标注（pycocotools 直接可读，评测用）。没有开发子集。"""
    from src.data.land.coco import SUBSET_FILE

    src = Path(root) / "coco_val2017"
    coco = json.loads((src / SUBSET_FILE).read_text())
    files = _file_index(src)
    names = [c["name"] for c in sorted(coco["categories"], key=lambda c: c["id"])]
    out, n_missing = _ovd_samples(src, files, coco, source="coco_val2017", split="val2017", id_prefix="coco_val2017/val2017", rel_dir="images", key_fn=lambda im: f"coco:{im['id']}", cat_names=names, strict=strict)
    return _finish(out, lambda s: str(s.fields["image_id"]), subset="all", dev_name=None, dev_config=None, extra_exclusion=extra_exclusion, n_missing=n_missing, extra_stats={"coco_json": str(src / SUBSET_FILE)})


def load_odinw(root: str | Path, config: str, *, category_names: str = "glip", extra_exclusion: Exclusion | None = None, strict: bool = True) -> LoadResult:
    """ODinW13 的一个数据集（GLIP 测试划分里的 ≤50 张）。

    category_names="glip"（默认）用 GLIP 配置的 OVERRIDE_CATEGORY 名字（GLIP 的评测提示就用它）；"raw" 用标注 json 的名字。
    fields 里两套都给（categories 与 categories_raw）。没有开发子集。
    """
    src = Path(root) / "odinw13"
    coco = json.loads((src / config / "annotations_e1sub.json").read_text())
    files = _file_index(src)
    raw = [c["name"] for c in sorted(coco["categories"], key=lambda c: c["id"])]
    glip = coco["glip"]["override_category_names"]
    if category_names not in ("glip", "raw"):
        raise ValueError(category_names)
    names = glip if category_names == "glip" else raw
    out, n_missing = _ovd_samples(src, files, coco, source=f"odinw13/{config}", split=coco["glip"]["test_split"], id_prefix=f"odinw13/{config}", rel_dir=f"{config}/images", key_fn=lambda im: f"odinw:{config}:{im['file_name']}", cat_names=names, strict=strict)
    for s in out:
        s.fields["categories_raw"] = raw
        s.fields["categories_glip"] = glip
    return _finish(out, lambda s: str(s.fields["image_id"]), subset="all", dev_name=None, dev_config=None, extra_exclusion=extra_exclusion, n_missing=n_missing, extra_stats={"coco_json": str(src / config / "annotations_e1sub.json"), "category_names": category_names})


# ---------------------------------------------------------------------------
# 计数：CountBench（491 张有图，D11-4）
# ---------------------------------------------------------------------------


def load_countbench(root: str | Path, *, subset: str = "eval", extra_exclusion: Exclusion | None = None, dev_config: str | Path | None = None, strict: bool = True) -> LoadResult:
    src = Path(root) / "countbench"
    files = _file_index(src)
    out, n_missing, n_no_image = [], 0, 0
    for line in (src / "countbench.jsonl").read_text().splitlines():
        r = json.loads(line)
        if not r["has_image"]:
            n_no_image += 1
            continue
        if not _exists(src, r.get("image"), strict, f"countbench row {r['row']}"):
            n_missing += 1
            continue
        out.append(
            Sample(
                id=f"countbench/all/{r['row']}",
                task="count",
                images=(str(src / r["image"]),),
                image_keys=(f"countbench:{r['row']}",),
                phash=(_phash(files, r["image"]),),
                fields={"text": r["text"], "row": r["row"], "image_url": r["image_url"]},
                gt={"count": int(r["number"])},
                source="countbench",
                split="all",
            )
        )
    return _finish(out, lambda s: str(s.fields["row"]), subset=subset, dev_name="countbench", dev_config=dev_config, extra_exclusion=extra_exclusion, n_missing=n_missing, extra_stats={"n_rows_without_image_in_hf": n_no_image})


# ---------------------------------------------------------------------------
# 相对深度：NYUv2 点对、BLINK Relative_Depth val
# ---------------------------------------------------------------------------


def load_nyu_pairs(
    root: str | Path,
    *,
    subset: str = "eval",
    pairs_config: str | Path | None = None,
    cache_dir: str | Path | None = None,
    verify_render: bool = True,
    extra_exclusion: Exclusion | None = None,
    strict: bool = True,
) -> LoadResult:
    """NYUv2 点对（configs/e1_nyu_pairs.json）。eval = Eigen test 上的点对；dev = 开发子集（官方 train 图）上的点对。

    渲染图（带 A、B 标记）按需生成到 cache_dir，已存在且像素 sha256 与配置相符就复用。默认放 $TMPDIR/visint_nyu_pairs_rendered
    （服务器上是内存盘）：约 3,500 张 640×480 PNG、1.4 GB 左右，不占根盘份额；渲染是确定性的，丢了随时重建。
    fields["raw_image"] 是未加标记的 RGB，fields["points"] 是 A、B 的像素坐标 (x, y)。
    """
    import numpy as np
    from PIL import Image

    from src.data.nyu_pairs import PAIRS_PATH, load_pairs, pixels_sha256, render_pair

    if subset not in SUBSETS:
        raise ValueError(subset)
    src = Path(root) / "nyuv2"
    cfg = load_pairs(pairs_config or PAIRS_PATH)
    files = _file_index(src)
    index = {r["idx"]: r for r in json.loads((src / "index.json").read_text())["images"]}
    cache = Path(cache_dir) if cache_dir else Path(os.environ.get("TMPDIR", "/tmp")) / "visint_nyu_pairs_rendered"
    want_split = {"eval": {"test"}, "dev": {"train"}, "all": {"test", "train"}}[subset]
    out, n_missing = [], 0
    rgb_cache: dict[int, Any] = {}
    for p in cfg["pairs"]:
        if p["split"] not in want_split:
            continue
        im = index.get(p["idx"])
        if im is None or not _exists(src, im["rgb"] if im else None, strict, f"nyu pair {p['id']}"):
            n_missing += 1
            continue
        path = cache / f"{p['id']}.png"
        need = True
        if path.exists() and not verify_render:
            need = False
        elif path.exists():
            with Image.open(path) as f:
                need = pixels_sha256(np.asarray(f.convert("RGB"))) != p.get("render_sha256")
        if need:
            if p["idx"] not in rgb_cache:
                rgb_cache = {p["idx"]: np.asarray(Image.open(src / im["rgb"]).convert("RGB"))}
            img = render_pair(rgb_cache[p["idx"]], p["A"], p["B"])
            if p.get("render_sha256") and pixels_sha256(img) != p["render_sha256"]:
                raise ValueError(f"{p['id']}：渲染结果与配置记录的 render_sha256 不符（渲染常量或 RGB 变了）")
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(img).save(path, format="PNG")
        out.append(
            Sample(
                id=f"nyuv2_pairs/{p['split']}/{p['id']}",
                task="rel_depth",
                images=(str(path),),
                image_keys=(f"nyuv2:{p['idx']}",),
                phash=(_phash(files, im["rgb"]),),
                fields={"points": {"A": list(p["A"]), "B": list(p["B"])}, "raw_image": str(src / im["rgb"]), "idx": p["idx"], "scene": im["scene"]},
                gt={"closer": p["closer"], "depth_A": p["depth_A"], "depth_B": p["depth_B"]},
                source="nyuv2_pairs",
                split=p["split"],
            )
        )
    # 开发子集按"图"定义（configs/e1_dev_subsets.json 的 nyuv2_train），点对的 split 已经区分了 test 与 train
    return _finish(out, lambda s: str(s.fields["idx"]), subset="all", dev_name=None, dev_config=None, extra_exclusion=extra_exclusion, n_missing=n_missing, extra_stats={"subset": subset, "dev_rule": "dev = 开发子集 nyuv2_train 那 50 张官方 train 图上的点对；eval = Eigen test 上的点对"})


def load_blink_reldepth(root: str | Path, *, subset: str = "eval", extra_exclusion: Exclusion | None = None, dev_config: str | Path | None = None, strict: bool = True) -> LoadResult:
    src = Path(root) / "blink_reldepth"
    files = _file_index(src)
    out, n_missing = [], 0
    for line in (src / "val.jsonl").read_text().splitlines():
        r = json.loads(line)
        if not _exists(src, r.get("image"), strict, f"blink {r['idx']}"):
            n_missing += 1
            continue
        out.append(
            Sample(
                id=f"blink_reldepth/val/{r['idx']}",
                task="rel_depth",
                images=(str(src / r["image"]),),
                image_keys=(f"blink:{r['idx']}",),
                phash=(_phash(files, r["image"]),),
                fields={"idx": r["idx"], "question": r["question"], "choices": r["choices"], "prompt": r["prompt"]},
                gt={"answer": r["answer_letter"], "closer": r["answer_letter"]},
                source="blink_reldepth",
                split="val",
            )
        )
    return _finish(out, lambda s: s.fields["idx"], subset=subset, dev_name="blink_reldepth", dev_config=dev_config, extra_exclusion=extra_exclusion, n_missing=n_missing)


# ---------------------------------------------------------------------------
# 空间问答：CV-Bench、EmbSpatial-Bench
# ---------------------------------------------------------------------------


def load_cvbench(root: str | Path, *, subset: str = "eval", extra_exclusion: Exclusion | None = None, dev_config: str | Path | None = None, strict: bool = True) -> LoadResult:
    src = Path(root) / "cvbench"
    files = _file_index(src)
    out, n_missing = [], 0
    for line in (src / "test.jsonl").read_text().splitlines():
        r = json.loads(line)
        if not _exists(src, r.get("image"), strict, f"cvbench {r['idx']}"):
            n_missing += 1
            continue
        out.append(
            Sample(
                id=f"cvbench/test/{r['idx']}",
                task="spatial_qa",
                images=(str(src / r["image"]),),
                image_keys=(r["image_key"],),
                phash=(_phash(files, r["image"]),),
                fields={
                    "idx": r["idx"],
                    "question": r["question"],
                    "choices": r["choices"],
                    "prompt": r["prompt"],
                    "type": r["type"],
                    "task": r["task"],
                    "source_dataset": r["source_dataset"],
                    "bbox": r["bbox"],
                },
                gt={"answer": r["answer_letter"]},
                source="cvbench",
                split="test",
            )
        )
    return _finish(out, lambda s: str(s.fields["idx"]), subset=subset, dev_name="cvbench", dev_config=dev_config, extra_exclusion=extra_exclusion, n_missing=n_missing)


def load_embspatial(root: str | Path, *, subset: str = "eval", extra_exclusion: Exclusion | None = None, dev_config: str | Path | None = None, strict: bool = True) -> LoadResult:
    src = Path(root) / "embspatial"
    files = _file_index(src)
    out, n_missing = [], 0
    for line in (src / "test.jsonl").read_text().splitlines():
        r = json.loads(line)
        if not _exists(src, r.get("image"), strict, f"embspatial {r['question_id']}"):
            n_missing += 1
            continue
        out.append(
            Sample(
                id=f"embspatial/test/{r['question_id']}",
                task="spatial_qa",
                images=(str(src / r["image"]),),
                image_keys=(f"embspatial:{r['question_id']}",),
                phash=(_phash(files, r["image"]),),
                fields={
                    "question_id": r["question_id"],
                    "question": r["question"],
                    "choices": r["answer_options"],
                    "relation": r["relation"],
                    "data_source": r["data_source"],
                    "scene_id": r["scene_id"],
                    "objects": r.get("objects"),
                },
                gt={"answer": r["answer_letter"], "answer_index": int(r["answer"])},
                source="embspatial",
                split="test",
            )
        )
    return _finish(out, lambda s: s.fields["question_id"], subset=subset, dev_name="embspatial", dev_config=dev_config, extra_exclusion=extra_exclusion, n_missing=n_missing)


# ---------------------------------------------------------------------------
# 总体 id（生成与核对开发子集用；不需要图像落地）
# ---------------------------------------------------------------------------


def populations(root: str | Path, names: Iterable[str] | None = None) -> tuple[dict[str, list[str]], dict[str, str]]:
    """从落地目录读各开发子集的总体 id 清单（与图像是否落全无关）。返回 (总体, 出处说明)。"""
    root = Path(root)
    pops: dict[str, list[str]] = {}
    prov: dict[str, str] = {}
    want = set(names) if names is not None else None

    def on(n: str) -> bool:
        return want is None or n in want

    if on("cvbench"):
        rows = [json.loads(l) for l in (root / "cvbench/test.jsonl").read_text().splitlines()]
        pops["cvbench"] = [str(r["idx"]) for r in rows]
        prov["cvbench"] = "nyu-visionx/CV-Bench@bc284db5 test_2d/3d.parquet 的 idx"
    if on("embspatial"):
        rows = [json.loads(l) for l in (root / "embspatial/test.jsonl").read_text().splitlines()]
        pops["embspatial"] = [r["question_id"] for r in rows]
        prov["embspatial"] = "Phineas476/EmbSpatial-Bench@e2733cac embspatial_bench.json 的 question_id"
    if on("blink_reldepth"):
        rows = [json.loads(l) for l in (root / "blink_reldepth/val.jsonl").read_text().splitlines()]
        pops["blink_reldepth"] = [r["idx"] for r in rows]
        prov["blink_reldepth"] = "BLINK-Benchmark/BLINK@a3666eb2 Relative_Depth val 的 idx"
    if on("countbench"):
        rows = [json.loads(l) for l in (root / "countbench/countbench.jsonl").read_text().splitlines()]
        pops["countbench"] = [str(r["row"]) for r in rows if r["has_image"]]
        prov["countbench"] = "nielsr/countbench@e32ce254 中 image 非空的行（官方 CountBench.json 的 0 起行号）"
    for ds, fn in REFCOCO_FILES.items():
        name = refcoco_dev_name(ds)
        if on(name):
            doc = json.loads((root / "refcoco" / fn).read_text())
            pops[name] = [str(s["sent_id"]) for r in doc["refs"] if r["split"] == "val" for s in r["sentences"]]
            prov[name] = f"{ds} refs pickle（Wayback 官方 zip）val 划分的 sent_id"
    if on("nyuv2_train"):
        from src.data.land.nyuv2 import read_splits

        train, _ = read_splits(root / "nyuv2/splits.mat")
        pops["nyuv2_train"] = [str(i) for i in train]
        prov["nyuv2_train"] = "官方 splits.mat 的 trainNdxs（1-based）"
    return pops, prov
