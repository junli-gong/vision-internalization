"""DEV-0076：scripts/analysis/test_metadata.py（W2 test 的逐句元数据，预注册 U-6）。

- 形状：已知形状（矩形、圆、两块、环里套块、空）的面积、周长、c = P² / A、连通块数，与 OpenCV 的直接调用相同；
- fixture 上端到端（--tiny）：句子与 run_w2 的取样逐 id 相同、类别取 ref → ann → category_id、大类取 COCO 表；--check-preds 一致与
  不一致；类别表对不上、输出已存在、--out 不带运行编号时都拒绝。
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
cv2 = pytest.importorskip("cv2")


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


TM = _load("_test_metadata_under_test", "scripts/analysis/test_metadata.py")


# ---------------------------------------------------------------------------
# 形状
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("h,w", [(20, 30), (1, 7), (13, 13)])
def test_rectangle_area_perimeter(h, w):
    m = np.zeros((40, 50), bool)
    m[5:5 + h, 8:8 + w] = True
    st = TM.shape_stats(m)
    # 外轮廓沿像素中心走：横向 w − 1 步、纵向 h − 1 步，各两次
    p = 2 * (w - 1) + 2 * (h - 1)
    assert st["area"] == h * w
    assert st["perimeter"] == pytest.approx(p)
    assert st["complexity"] == pytest.approx(p * p / (h * w))
    assert (st["n_components"], st["n_external_contours"]) == (1, 1)


@pytest.mark.parametrize("r", [10, 20, 40])
def test_disk_area_perimeter(r):
    n = 2 * r + 11
    yy, xx = np.mgrid[:n, :n]
    c = n // 2
    m = (xx - c) ** 2 + (yy - c) ** 2 <= r * r
    st = TM.shape_stats(m)
    assert st["area"] == int(m.sum())
    assert abs(st["area"] - math.pi * r * r) / (math.pi * r * r) < 0.02
    # 8 邻接链的长度比圆周长略长（斜步记 √2），在 [2πr, 1.1 · 2πr] 内
    assert 2 * math.pi * r <= st["perimeter"] <= 1.1 * 2 * math.pi * r
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    assert st["perimeter"] == pytest.approx(sum(cv2.arcLength(x, True) for x in cs))
    assert st["complexity"] >= 4 * math.pi  # 圆最紧凑


def test_two_blocks_ring_island_and_empty():
    m = np.zeros((30, 40), bool)
    m[2:6, 2:12] = True  # 4 × 10
    m[20:25, 30:33] = True  # 5 × 3
    st = TM.shape_stats(m)
    assert st["area"] == 40 + 15
    assert st["perimeter"] == pytest.approx((2 * 9 + 2 * 3) + (2 * 2 + 2 * 4))
    assert (st["n_components"], st["n_external_contours"]) == (2, 2)
    ring = np.zeros((30, 30), bool)
    ring[2:28, 2:28] = True
    ring[8:22, 8:22] = False
    ring[12:18, 12:18] = True  # 洞里的块：RETR_EXTERNAL 不数它，连通块数数它
    st = TM.shape_stats(ring)
    assert (st["n_components"], st["n_external_contours"]) == (2, 1)
    assert st["perimeter"] == pytest.approx(4 * 25)
    diag = np.zeros((6, 6), bool)
    diag[1, 1] = diag[2, 2] = True  # 只靠角相连：8 邻接是一块
    assert TM.shape_stats(diag)["n_components"] == 1
    e = TM.shape_stats(np.zeros((5, 5), bool))
    assert (e["area"], e["perimeter"], e["complexity"], e["n_components"]) == (0, 0.0, None, 0)


# ---------------------------------------------------------------------------
# fixture 上端到端
# ---------------------------------------------------------------------------


def _tiny_samples():
    from src.llmseg.runner import test_splits
    from src.w2.config import load_w2_config
    from src.w2.data import tiny_paths, tiny_splits, w2_items

    doc = tiny_splits(load_w2_config())
    return {s: w2_items("refseg", s, tiny_paths(), splits=doc) for s in test_splits()}


def test_tiny_end_to_end(tmp_path):
    out = tmp_path / "0001_meta"
    assert TM.main(["--tiny", "--out", str(out)]) == 0
    rows = [json.loads(x) for x in (out / TM.OUT_ROWS).read_text().splitlines()]
    want = _tiny_samples()
    assert [r["id"] for r in rows] == [s.id for ss in want.values() for s in ss]  # 取样同 run_w2，逐 id、同顺序
    assert [r["w2_split"] for r in rows] == [k for k, ss in want.items() for _ in ss]
    by = {r["id"]: r for r in rows}
    # fixture：ref 3（sent 106、107）→ ann 7003 → category 2 = dog，COCO 表里大类 animal；其余是 person
    assert {k: by["refcoco_unc/testA/106"][k] for k in ("ref_id", "ann_id", "category_id", "category", "supercategory")} == \
        {"ref_id": 3, "ann_id": 7003, "category_id": 2, "category": "dog", "supercategory": "animal"}
    assert by["refcoco_unc/testB/108"]["supercategory"] == "person"
    from src.e1.scoring import gt_mask

    for ss in want.values():
        for s in ss:
            r = by[s.id]
            g = gt_mask(s)
            st = TM.shape_stats(g)
            assert (r["area"], r["perimeter"], r["n_components"]) == (st["area"], st["perimeter"], st["n_components"])
            assert r["rel_area"] == pytest.approx(st["area"] / (g.shape[0] * g.shape[1]))
            assert r["image_size"] == [g.shape[1], g.shape[0]]
    # 106 的真值是 32 × 24 的矩形（polygon [4,4,36,4,36,28,4,28]）
    assert (by["refcoco_unc/testA/106"]["area"], by["refcoco_unc/testA/106"]["perimeter"]) == (768, 2 * 31 + 2 * 23)
    summ = json.loads((out / TM.OUT_SUMMARY).read_text())
    assert summ["n"] == len(rows) and summ["n_by_split"] == {k: len(v) for k, v in want.items()}
    assert summ["category"] == {"person": sum(r["category"] == "person" for r in rows), "dog": 2}
    assert set(summ["inputs"]) >= {"splits", "refcoco", "coco_categories", "dev_subsets", "w2_config"}
    assert all(len(v["sha256"]) == 64 for v in summ["inputs"]["refcoco"].values())
    assert summ["complexity_quantiles"]["0.5"] == pytest.approx(float(np.median([r["complexity"] for r in rows])))
    assert summ["output"]["n"] == len(rows)
    # 已存在就不覆盖
    assert TM.main(["--tiny", "--out", str(out)]) == 2


def _fake_preds(root: Path, samples: dict, *, drop: str | None = None) -> Path:
    for split, ss in samples.items():
        d = root / split
        d.mkdir(parents=True)
        d.joinpath(TM.PREDICTIONS).write_text("".join(json.dumps({"id": s.id}) + "\n" for s in ss if s.id != drop))
    return root


def test_check_preds(tmp_path):
    samples = _tiny_samples()
    good = _fake_preds(tmp_path / "good", samples)
    out = tmp_path / "0002_meta"
    assert TM.main(["--tiny", "--out", str(out), "--check-preds", str(good)]) == 0
    summ = json.loads((out / TM.OUT_SUMMARY).read_text())
    assert all(v["same"] for r in summ["check_preds"].values() for v in r.values())
    bad = _fake_preds(tmp_path / "bad", samples, drop="refcoco_unc/testB/109")
    out2 = tmp_path / "0003_meta"
    assert TM.main(["--tiny", "--out", str(out2), "--check-preds", str(good), "--check-preds", str(bad)]) == 1
    assert not out2.exists()  # 不一致时什么都不写


def test_category_table_mismatch_refused(tmp_path):
    doc = json.loads(TM.TINY_COCO_CATEGORIES.read_text())
    for c in doc["categories"]:
        if c["name"] == "dog":
            c["name"] = "hound"
    p = tmp_path / "cats.json"
    p.write_text(json.dumps(doc))
    out = tmp_path / "0004_meta"
    assert TM.main(["--tiny", "--out", str(out), "--coco-categories", str(p)]) == 2
    assert not out.exists()


def test_out_needs_run_number(tmp_path):
    with pytest.raises(SystemExit) as e:
        TM.parse_args(["--tiny", "--out", str(tmp_path / "meta")])
    assert e.value.code == 2
    with pytest.raises(SystemExit):
        TM.parse_args(["--tiny", "--out", str(tmp_path / "76_meta")])
    assert TM.parse_args(["--tiny", "--out", str(tmp_path / "0076_meta")]).tiny


def test_real_run_needs_data_root(tmp_path, monkeypatch):
    monkeypatch.delenv("VISINT_DATA_ROOT", raising=False)
    with pytest.raises(SystemExit):
        TM.parse_args(["--out", str(tmp_path / "0005_meta")])
