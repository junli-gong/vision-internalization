#!/usr/bin/env python
"""W2 test 的逐句元数据（DEV-0076；预注册 U-6；D154 (4)）。只读：不跑模型，不改任何结果目录，只写 --out。

句子：W2 的 5 个 test 划分（configs/w2.json 的 splits.test）的全部句子，取样与 scripts/run_w2.py 相同（src/w2/data.w2_items，
加载器的固定顺序，id 指纹与清单核对），所以逐 id 与 RUN-0072、RUN-0092 的 predictions.jsonl 相同（--check-preds 逐划分核对）。

每句一行（<out>/test_metadata.jsonl）：
- id、w2_split、sent_id、ref_id、ann_id、image_id；
- 类别：ref → ann → COCO 的 category_id（落地的 refcoco JSON 的 anns[ann_id].category_id，并核对 refs 里同一 ref 的 category_id 与
  ann_id 相同）；名称取同一文件的 categories；大类（supercategory）取 COCO 原样的 categories 表（--coco-categories，默认
  <data-root>/coco_train2014_e1dev/instances_train2014_e1dev.json，src/data/land/coco_train2014_dev 落地，categories 原样），
  按 id 核对两边的名称逐字相同；
- image_size [W, H]（标注里的图像尺寸，与真值 mask 的形状核对）；
- 真值 mask（src/e1/scoring.gt_mask，同 W2 的计分）的面积 A（前景像素数）、周长 P（OpenCV findContours(RETR_EXTERNAL,
  CHAIN_APPROX_NONE) 的每条外轮廓 arcLength(closed=True) 之和）、c = P² / A、相对面积 A / (W·H)、连通块数（8 邻接，与
  findContours 追踪前景的邻接相同）；另记外轮廓条数（RETR_EXTERNAL 不含洞里的块，所以可以少于连通块数）。A = 0 时 c 记 null。

汇总（<out>/summary.json）：句数（合计与各划分）、各类别与大类的句数（合计与各划分）、c 与相对面积的分位数（numpy.quantile 的
默认 linear 插值；A = 0 的句子不进 c 的分位数，单列条数）、输入文件的 sha256、本脚本的 sha256 与 git commit、库版本。
不做分档的统计（U-6 的分档由管理员另写脚本）。

运行（HPC 登录节点或 06 的 CPU，几分钟）：
    python scripts/analysis/test_metadata.py --data-root $VI/b200/data --out experiments/U/NNNN_test_metadata \\
        --check-preds experiments/X/0092_x2_test_hpcgpu06/qwen3.5-9b/refseg/x2_spline_trimap \\
        --check-preds experiments/W2/0072_w2_test_hpcgpu06/qwen3.5-9b/refseg/w2c
开发机彩排：--tiny（tests/fixtures/w2 的 RefCOCO，划分按 run_w2 --tiny 的同一套规则现算；大类取 tests/fixtures/e1 的 COCO 表）。
--out 的最后一级目录名必须以四位运行编号开头（如 0097_test_metadata），否则退出码 2，什么都不写；输出文件已存在时也拒绝。
退出码：0 = 写完；1 = --check-preds 不一致（什么都不写）；2 = 用法或数据错误。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from src.data.loaders import REFCOCO_FILES  # noqa: E402
from src.data.records import Sample  # noqa: E402
from src.e1.scoring import gt_mask  # noqa: E402
from src.e1.tasks import DataPaths, FIXTURE_DIR, _sent_id  # noqa: E402
from src.llmseg.runner import test_splits  # noqa: E402
from src.w2.config import load_w2_config  # noqa: E402
from src.w2.data import W2_SPLITS_PATH, load_splits, tiny_paths, tiny_splits, w2_items  # noqa: E402

RUN_DIR_NAME = re.compile(r"[0-9]{4}_[A-Za-z0-9][A-Za-z0-9_.-]*")  # 同 scripts/run_w2.py
TASK = "refseg"
OUT_ROWS = "test_metadata.jsonl"
OUT_SUMMARY = "summary.json"
COCO_CATEGORIES_REL = "coco_train2014_e1dev/instances_train2014_e1dev.json"  # src/data/land/coco_train2014_dev 的落地布局
TINY_COCO_CATEGORIES = FIXTURE_DIR / "root" / COCO_CATEGORIES_REL
QUANTILES: tuple[float, ...] = (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)
QUANTILE_METHOD = "linear"  # numpy.quantile 的默认
CONNECTIVITY = 8
PREDICTIONS = "predictions.jsonl"


class MetadataError(ValueError):
    """输入不合格（类别对不上、mask 形状不对、文件已存在……）。"""


# ---------------------------------------------------------------------------
# 形状（纯函数）
# ---------------------------------------------------------------------------


def shape_stats(mask: np.ndarray) -> dict[str, Any]:
    """二值 mask → {area, perimeter, complexity, n_components, n_external_contours}。规则见模块说明。"""
    import cv2

    m = np.asarray(mask).astype(bool)
    if m.ndim != 2:
        raise MetadataError(f"mask 必须是二维，收到形状 {m.shape}")
    u8 = m.astype(np.uint8)
    area = int(np.count_nonzero(m))
    found = cv2.findContours(u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = found[0] if len(found) == 2 else found[1]  # OpenCV 3 返回 3 项
    perimeter = float(sum(cv2.arcLength(c, True) for c in contours))
    n_cc = int(cv2.connectedComponents(u8, connectivity=CONNECTIVITY)[0]) - 1
    return {"area": area, "perimeter": perimeter, "complexity": (perimeter * perimeter / area) if area > 0 else None,
            "n_components": n_cc, "n_external_contours": len(contours)}


# ---------------------------------------------------------------------------
# 类别
# ---------------------------------------------------------------------------


def load_coco_categories(path: str | Path) -> dict[int, dict[str, str]]:
    """COCO 格式文件的 categories → {id: {"name", "supercategory"}}。"""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    cats = doc.get("categories") if isinstance(doc, Mapping) else None
    if not isinstance(cats, list) or not cats:
        raise MetadataError(f"{path} 没有 COCO 格式的 categories 列表")
    out: dict[int, dict[str, str]] = {}
    for c in cats:
        if not {"id", "name", "supercategory"} <= set(c):
            raise MetadataError(f"{path} 的类别缺 id、name 或 supercategory：{c}")
        if int(c["id"]) in out:
            raise MetadataError(f"{path} 的类别 id {c['id']} 重复")
        out[int(c["id"])] = {"name": str(c["name"]), "supercategory": str(c["supercategory"])}
    return out


class RefcocoIndex:
    """一个落地的 refcoco JSON（src/data/loaders.load_refcoco 读的同一个文件）：ref_id → ref、ann_id → ann、category id → 名称。"""

    def __init__(self, path: Path) -> None:
        doc = json.loads(path.read_text(encoding="utf-8"))
        self.path = path
        self.refs = {int(r["ref_id"]): r for r in doc["refs"]}
        self.anns = doc["anns"]
        self.names = {int(k): str(v) for k, v in doc["categories"].items()}


def category_of(sample: Sample, idx: RefcocoIndex, coco: Mapping[int, Mapping[str, str]]) -> dict[str, Any]:
    """ref → ann → COCO 的 category_id；名称与大类。任何一处对不上都报错，不猜。"""
    f = sample.fields
    ann = idx.anns.get(str(f["ann_id"]))
    ref = idx.refs.get(int(f["ref_id"]))
    if ann is None or ref is None:
        raise MetadataError(f"{sample.id}：{idx.path} 里没有 ann {f['ann_id']} 或 ref {f['ref_id']}")
    cid = int(ann["category_id"])
    if int(ref["ann_id"]) != int(f["ann_id"]) or int(ref["category_id"]) != cid or int(ann["image_id"]) != int(f["image_id"]):
        raise MetadataError(f"{sample.id}：ref {ref['ref_id']}（ann {ref['ann_id']}，类别 {ref['category_id']}）与 ann {ann['id']}"
                            f"（类别 {cid}，图 {ann['image_id']}）对不上")
    name = idx.names.get(cid)
    if name is None or name != f["category"]:
        raise MetadataError(f"{sample.id}：类别 {cid} 在 {idx.path} 里叫 {name!r}，加载器给的是 {f['category']!r}")
    c = coco.get(cid)
    if c is None or c["name"] != name:
        raise MetadataError(f"{sample.id}：类别 {cid}（{name}）在 COCO 的类别表里是 {None if c is None else c['name']!r}")
    return {"category_id": cid, "category": name, "supercategory": c["supercategory"]}


def check_category_tables(indexes: Iterable[RefcocoIndex], coco: Mapping[int, Mapping[str, str]]) -> dict[str, Any]:
    """refcoco JSON 的 categories 与 COCO 表按 id 逐个比名称（refcoco 里有的 id 都必须在 COCO 表里、名称相同）。"""
    seen: dict[int, str] = {}
    for idx in indexes:
        for cid, nm in idx.names.items():
            if cid in seen and seen[cid] != nm:
                raise MetadataError(f"类别 {cid} 在两份 refcoco JSON 里名称不同：{seen[cid]!r} / {nm!r}")
            seen[cid] = nm
    bad = [(cid, nm, coco.get(cid, {}).get("name")) for cid, nm in sorted(seen.items()) if coco.get(cid, {}).get("name") != nm]
    if bad:
        raise MetadataError(f"refcoco 的类别与 COCO 表的名称不同（id, refcoco, coco）：{bad[:5]}")
    return {"n_refcoco_ids": len(seen), "n_coco_ids": len(coco), "names_identical": True}


# ---------------------------------------------------------------------------
# 逐句
# ---------------------------------------------------------------------------


def sample_record(sample: Sample, split: str, idx: RefcocoIndex, coco: Mapping[int, Mapping[str, str]]) -> dict[str, Any]:
    gt = gt_mask(sample)
    w, h = (int(v) for v in sample.fields["image_size"])
    if gt.shape != (h, w):
        raise MetadataError(f"{sample.id}：真值 mask 形状 {gt.shape} 与图像尺寸 (H, W) = ({h}, {w}) 不同")
    st = shape_stats(gt)
    return {"id": sample.id, "w2_split": split, "sent_id": _sent_id(sample), "ref_id": int(sample.fields["ref_id"]),
            "ann_id": int(sample.fields["ann_id"]), "image_id": int(sample.fields["image_id"]), **category_of(sample, idx, coco),
            "image_size": [w, h], "area": st["area"], "perimeter": st["perimeter"], "complexity": st["complexity"],
            "rel_area": st["area"] / float(w * h), "n_components": st["n_components"], "n_external_contours": st["n_external_contours"]}


def collect(paths: DataPaths, splits_doc: Mapping[str, Any], splits: Sequence[str], coco: Mapping[int, Mapping[str, str]],
            log=print) -> tuple[list[dict[str, Any]], dict[str, RefcocoIndex]]:
    """全部 test 句子的元数据（划分按 splits 的顺序，划分内按 w2_items 的顺序）。"""
    indexes: dict[str, RefcocoIndex] = {}
    rows: list[dict[str, Any]] = []
    for split in splits:
        samples = w2_items(TASK, split, paths, splits=splits_doc)
        ds = splits_doc["splits"][split]["dataset"]
        if ds not in indexes:
            indexes[ds] = RefcocoIndex(Path(paths.root) / "refcoco" / REFCOCO_FILES[ds])
        rows += [sample_record(s, split, indexes[ds], coco) for s in samples]
        log(f"[metadata] {split}: {len(samples)} 句")
    return rows, indexes


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------


def _sha(p: str | Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _rel(p: str | Path) -> str:
    q = Path(p).resolve()
    return str(q.relative_to(REPO_ROOT)) if q.is_relative_to(REPO_ROOT) else str(p)


def quantiles(vals: Sequence[float]) -> dict[str, float] | None:
    if not vals:
        return None
    q = np.quantile(np.asarray(vals, dtype=np.float64), QUANTILES, method=QUANTILE_METHOD)
    return {f"{p:g}": float(v) for p, v in zip(QUANTILES, q)}


def _counts(vals: Iterable[str]) -> dict[str, int]:
    c = Counter(vals)
    return {k: c[k] for k in sorted(c, key=lambda k: (-c[k], k))}


def git_commit() -> str | None:
    try:
        r = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        return r.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def summarize(rows: Sequence[Mapping[str, Any]], splits: Sequence[str], inputs: Mapping[str, Any], extra: Mapping[str, Any]) -> dict[str, Any]:
    import cv2
    import pycocotools

    by_split = {s: [r for r in rows if r["w2_split"] == s] for s in splits}
    cs = [r["complexity"] for r in rows if r["complexity"] is not None]
    return {
        "experiment": "U-6 元数据（DEV-0076）", "prereg": "PREREGISTRATION.md U-6", "framework": _rel(__file__),
        "n": len(rows), "n_by_split": {s: len(v) for s, v in by_split.items()},
        "category": _counts(r["category"] for r in rows), "supercategory": _counts(r["supercategory"] for r in rows),
        "category_by_split": {s: _counts(r["category"] for r in v) for s, v in by_split.items()},
        "supercategory_by_split": {s: _counts(r["supercategory"] for r in v) for s, v in by_split.items()},
        "complexity_quantiles": quantiles(cs), "rel_area_quantiles": quantiles([r["rel_area"] for r in rows]),
        "n_zero_area": sum(1 for r in rows if r["area"] == 0),
        "n_multi_component": sum(1 for r in rows if r["n_components"] > 1),
        "n_components_ne_external_contours": sum(1 for r in rows if r["n_components"] != r["n_external_contours"]),
        "rules": {
            "perimeter": "OpenCV findContours(RETR_EXTERNAL, CHAIN_APPROX_NONE)，每条外轮廓 arcLength(closed=True) 之和（像素中心的 8 邻接链，"
                         "斜步记 √2）",
            "complexity": "P² / A（A = 前景像素数；A = 0 记 null）", "rel_area": "A / (W·H)，W、H 取标注的图像尺寸",
            "n_components": f"cv2.connectedComponents(connectivity={CONNECTIVITY}) − 1",
            "category": "ref → ann → COCO category_id（落地的 refcoco JSON）；大类取 COCO 原样的 categories 表，按 id 核对名称",
            "quantiles": f"numpy.quantile(method={QUANTILE_METHOD!r})，分位点 {list(QUANTILES)}；c 的分位数不含 A = 0 的句子",
            "sampling": "src/w2/data.w2_items（同 scripts/run_w2.py），划分 = configs/w2.json 的 splits.test",
        },
        "inputs": dict(inputs), **dict(extra),
        "code": {"script_sha256": _sha(__file__), "git_commit": git_commit()},
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "opencv": cv2.__version__,
                     "pycocotools": getattr(pycocotools, "__version__", None)},
    }


def check_preds(rows: Sequence[Mapping[str, Any]], dirs: Sequence[str | Path], splits: Sequence[str]) -> dict[str, Any]:
    """每个目录的 <split>/predictions.jsonl 的 id（按行序）必须与元数据的 id 逐个相同。返回 {目录: {划分: {...}}}。"""
    out: dict[str, Any] = {}
    for d in dirs:
        res = {}
        for s in splits:
            p = Path(d) / s / PREDICTIONS
            want = [r["id"] for r in rows if r["w2_split"] == s]
            if not p.is_file():
                res[s] = {"same": False, "reason": f"没有 {p}"}
                continue
            got = [json.loads(line)["id"] for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
            res[s] = {"same": got == want, "n": len(got), "sha256": _sha(p)}
            if got != want:
                res[s]["reason"] = "同一组 id、顺序不同" if sorted(got) == sorted(want) else "id 集合不同"
        out[_rel(d)] = res
    return out


# ---------------------------------------------------------------------------
# 命令行
# ---------------------------------------------------------------------------


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="输出目录（最后一级以四位运行编号开头，如 experiments/U/0097_test_metadata）")
    p.add_argument("--data-root", help="落地目录（默认环境变量 VISINT_DATA_ROOT；--tiny 时取 fixture）")
    p.add_argument("--splits-file", default=str(W2_SPLITS_PATH), help="W2 的划分清单（默认 configs/w2_splits.json；--tiny 时现算）")
    p.add_argument("--coco-categories", help=f"COCO 格式的类别表（默认 <data-root>/{COCO_CATEGORIES_REL}）")
    p.add_argument("--check-preds", action="append", default=[], metavar="DIR",
                   help="<DIR>/<划分>/predictions.jsonl 的 id 必须与元数据逐个相同（可给多次）")
    p.add_argument("--tiny", action="store_true", help="tests/fixtures 的小数据（开发机彩排）")
    a = p.parse_args(argv)
    if not RUN_DIR_NAME.fullmatch(Path(a.out).name):
        p.error(f"--out 的目录名必须以四位运行编号开头，形如 0097_test_metadata，收到 {a.out!r}")
    if not a.tiny and not (a.data_root or os.environ.get("VISINT_DATA_ROOT")):
        p.error("要给 --data-root（或设 VISINT_DATA_ROOT）")
    if a.tiny and Path(a.splits_file).resolve() != W2_SPLITS_PATH.resolve():
        p.error("--tiny 的划分在 fixture 上现算，不能配 --splits-file")
    return a


def main(argv=None) -> int:
    a = parse_args(argv)
    out = Path(a.out)
    for name in (OUT_ROWS, OUT_SUMMARY):
        if (out / name).exists():
            print(f"{out / name} 已存在，不覆盖", file=sys.stderr)
            return 2
    try:
        if a.tiny:
            paths, splits_doc = (DataPaths(root=Path(a.data_root)) if a.data_root else tiny_paths()), tiny_splits(load_w2_config())
            splits_src: dict[str, Any] = {"path": "tiny（src/w2/data.tiny_splits，tests/fixtures 上现算）"}
        else:
            paths = DataPaths(root=Path(a.data_root or os.environ["VISINT_DATA_ROOT"]))
            splits_doc = load_splits(a.splits_file)
            splits_src = {"path": _rel(a.splits_file), "sha256": _sha(a.splits_file)}
        coco_path = Path(a.coco_categories) if a.coco_categories else (TINY_COCO_CATEGORIES if a.tiny and not a.data_root
                                                                       else Path(paths.root) / COCO_CATEGORIES_REL)
        coco = load_coco_categories(coco_path)
        splits = test_splits()
        rows, indexes = collect(paths, splits_doc, splits, coco)
        cat_check = check_category_tables(indexes.values(), coco)
    except (MetadataError, OSError, KeyError, ValueError) as e:
        print(f"失败：{e}", file=sys.stderr)
        return 2
    preds = check_preds(rows, a.check_preds, splits) if a.check_preds else {}
    bad = [(d, s) for d, r in preds.items() for s, v in r.items() if not v["same"]]
    for d, r in preds.items():
        for s, v in r.items():
            print(f"[check-preds] {d} {s}: {'same' if v['same'] else 'DIFFERENT（' + v.get('reason', '') + '）'}")
    if bad:
        print(f"--check-preds 不一致：{bad}；什么都不写", file=sys.stderr)
        return 1
    inputs = {"splits": splits_src, "dev_subsets": {"path": _rel(paths.dev_subsets), "sha256": _sha(paths.dev_subsets)},
              "refcoco": {ds: {"path": str(idx.path), "sha256": _sha(idx.path)} for ds, idx in indexes.items()},
              "coco_categories": {"path": str(coco_path), "sha256": _sha(coco_path)},
              "w2_config": {"path": _rel(load_w2_config().path), "sha256": _sha(load_w2_config().path)}}
    summ = summarize(rows, splits, inputs, {"tiny": bool(a.tiny), "category_tables": cat_check, "check_preds": preds})
    out.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    (out / OUT_ROWS).write_text(text, encoding="utf-8")
    summ["output"] = {"path": OUT_ROWS, "n": len(rows), "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}
    (out / OUT_SUMMARY).write_text(json.dumps(summ, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    q = summ["complexity_quantiles"] or {}
    print(f"写出 {out / OUT_ROWS}（{len(rows)} 句，sha256 {summ['output']['sha256']}）与 {out / OUT_SUMMARY}；"
          f"c 的四分位 {q.get('0.25')} / {q.get('0.5')} / {q.get('0.75')}；人 {summ['category'].get('person', 0)} 句")
    return 0


if __name__ == "__main__":
    sys.exit(main())
