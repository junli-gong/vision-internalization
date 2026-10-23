"""W2-c 测试用的假数据（DEV-0061）：按 docs/w2_plan.md §5 的格式造几条训练数据（真数据由并行的 DEV-0060 生成，这里不碰它）。

- write_fake_w2c_data(out_dir, w2_config=…)：句子取 E3 训练池的 fixture（tests/fixtures/e3/root 的 RefCOCO 系 train，图像路径相对这个根），
  K = 配置的候选数。候选 mask 由 GT 框做几种确定的变换（缩放、平移、GT mask 本身、框本身），IoU 与标签照 §5 的规则（真值 IoU 最高、并列取编号
  小的，src/w2/loop.oracle_index 的精确比较）；1 号失败（ok = False）的那几行，1 号存的是空 mask（§5："1 号失败时为空 mask"）；提出失败的那几行
  candidates 与 label 都是 null。每行的 w2_config_sha256、spec_sha256 取给定的 W2 配置。分片每片 shard_size 行，另写 manifest.json。
- from_candidates_row(row, sample, root)：把 W2 评测的 candidates.jsonl 的一行（src/w2/loop.candidates_row 的格式）照 DEV-0060 的做法换成
  §5 的一行（1 号失败存空 mask），给"挑选图与 W2-a 评测逐位相同"的测试用。

命令行：python -m tests._w2c_fakes <dir> [--w2-config configs/w2.json]（写 <dir>/*.jsonl 与 manifest.json）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.data.records import Sample
from src.e1.scoring import gt_mask
from src.e3 import data as D
from src.e3.config import load_config
from src.executors.raster import box_to_mask, mask_to_rle
from src.metrics.seg import mask_inter_union, sample_iou
from src.w2.config import W2_CONFIG_PATH, W2Config, load_w2_config
from src.w2.loop import oracle_index

ROOT = D.FIXTURE_REFSEG_ROOT  # 假数据的图像根（--data-root）
SOURCES = {"refcoco_unc": "refcoco", "refcoco+_unc": "refcoco+", "refcocog_umd": "refcocog"}


def train_pool() -> list[Sample]:
    return list(D.load_pool("refseg", D.E3Paths.fixtures(), load_config().task("refseg")).records)


def _scaled(box: Sequence[float], f: float, dx: float, h: int, w: int) -> np.ndarray:
    x0, y0, x1, y1 = (float(v) for v in box)
    cx, cy, hw, hh = (x0 + x1) / 2 + dx, (y0 + y1) / 2, (x1 - x0) / 2 * f, (y1 - y0) / 2 * f
    return box_to_mask([max(0.0, cx - hw), max(0.0, cy - hh), min(float(w), cx + hw), min(float(h), cy + hh)], h, w)


def candidate_masks(sample: Sample, k: int, rot: int = 0) -> list[np.ndarray]:
    """K 个确定的候选：框本身、GT mask、框缩放 0.7 / 1.3、框平移，从第 rot 个起循环取够 K 个（有重复就照常编号，同 §1"不去重"；
    rot 随句子变，标签就分散在各个编号上，也有并列）。"""
    gt = gt_mask(sample)
    h, w = gt.shape
    box = sample.gt["box"]
    pool = [box_to_mask([float(v) for v in box], h, w), gt.copy(), _scaled(box, 0.7, 0.0, h, w), _scaled(box, 1.3, 0.0, h, w),
            _scaled(box, 1.0, 4.0, h, w), _scaled(box, 1.0, -4.0, h, w)]
    return [np.asarray(pool[(i + rot) % len(pool)], dtype=bool) for i in range(k)]


def make_row(sample: Sample, w2: W2Config, *, root: Path = ROOT, failed: bool = False, first_not_ok: bool = False,
             masks: Sequence[np.ndarray] | None = None, names: Sequence[str] | None = None, rot: int = 0) -> dict[str, Any]:
    gt = gt_mask(sample)
    h, w = gt.shape
    x0, y0, x1, y1 = (float(v) for v in sample.gt["box"])
    src = SOURCES[sample.id.split("/")[0]]
    row: dict[str, Any] = {"id": sample.id, "source": src, "split": "train",
                           "image": str(Path(sample.images[0]).resolve().relative_to(Path(root).resolve())), "image_size": [w, h],
                           "expression": sample.fields["expression"], "gt_box": [x0, y0, x1, y1]}
    if failed:
        row.update(proposal={"box": None, "failure": "no_box", "raw_text": "I cannot find it."}, candidates=None, label=None)
    else:
        ms = list(masks) if masks is not None else candidate_masks(sample, w2.k, rot)
        if first_not_ok:
            ms[0] = np.zeros_like(ms[0])  # §5：1 号失败时为空 mask
        pairs = []
        cands = []
        for i, m in enumerate(ms, start=1):
            st = mask_inter_union(m, gt)
            pairs.append((int(st.inter), int(st.union)))
            cands.append({"index": i, "name": (names[i - 1] if names else f"fake{i}"), "ok": not (first_not_ok and i == 1),
                          "rle": mask_to_rle(m), "iou": sample_iou(st.inter, st.union)})
        row.update(proposal={"box": [x0 + 1, y0 + 1, x1 - 1, y1 - 1], "failure": None,
                             "raw_text": f'```json\n[{{"bbox_2d": [1, 1, 2, 2], "label": "{sample.fields["expression"]}"}}]\n```'},
                   candidates=cands, label=oracle_index(pairs))
    row.update(spec_sha256=w2.candidates_digest(), w2_config_sha256=w2.sha256)
    return row


def from_candidates_row(crow: Mapping[str, Any], sample: Sample, w2: W2Config, *, root: Path) -> dict[str, Any]:
    """W2 评测的 candidates.jsonl 的一行 → §5 的一行（DEV-0060 的做法：1 号失败时存空 mask；IoU 与标签按 effective 的 mask 算）。"""
    from src.executors.raster import rle_to_mask

    masks = [rle_to_mask(c["mask"]["rle"]) for c in crow["candidates"]]
    first_not_ok = crow["candidates"][0]["ok"] is False
    row = make_row(sample, w2, root=root, masks=masks, names=[c["name"] for c in crow["candidates"]], first_not_ok=first_not_ok)
    for c, src in zip(row["candidates"], crow["candidates"]):
        c["ok"] = src["ok"]
    return row


def write_rows(out_dir: str | Path, rows: Sequence[Mapping[str, Any]], *, shard_size: int = 12, manifest: bool = True) -> Path:
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    for i in range(0, len(rows), shard_size):
        (d / f"shard_{i // shard_size:04d}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                                                                      for r in rows[i: i + shard_size]), encoding="utf-8")
    if manifest:
        (d / "manifest.json").write_text(json.dumps({"fake": True, "n": len(rows), "note": "tests/_w2c_fakes（DEV-0061 的测试数据）"},
                                                    ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return d


def fake_rows(w2: W2Config, *, n: int | None = None, failed: Sequence[int] = (1, 7), first_not_ok: Sequence[int] = (2, 9)
              ) -> list[dict[str, Any]]:
    pool = train_pool()[:n]
    return [make_row(s, w2, failed=i in failed, first_not_ok=i in first_not_ok, rot=i) for i, s in enumerate(pool)]


def write_fake_w2c_data(out_dir: str | Path, *, w2_config: str | Path = W2_CONFIG_PATH, n: int | None = None, shard_size: int = 12,
                        failed: Sequence[int] = (1, 7), first_not_ok: Sequence[int] = (2, 9)) -> Path:
    w2 = load_w2_config(w2_config)
    return write_rows(out_dir, fake_rows(w2, n=n, failed=failed, first_not_ok=first_not_ok), shard_size=shard_size)


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
