"""DEV-0079：X-M 的训练目标（src/xm/targets.py）与统计（scripts/xm/data_stats.py）。"""

from __future__ import annotations

import importlib.util
import json

import numpy as np
import pytest

from src.executors.raster import mask_to_rle
from src.explore import oracle_sim as OS
from src.explore import x2_data as XD
from src.explore import x2_prompts as XP
from src.xm import data as D
from src.xm import exclusions as X
from src.xm import targets as T
from tests._xm_fakes import build_fake_root


def _inst(aid, mask, box=None):
    ys, xs = np.nonzero(mask)
    b = box or (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))
    return D.XMInstance(ann_id=aid, box_xyxy=tuple(b), rle=mask_to_rle(mask), area=int(mask.sum()))


def _pair(insts, w, h, pid="coco2017/train2017/1/1"):
    return D.XMPair(id=pid, source="coco2017", split="train", image="coco2017/images/train2017/000000000001.jpg", width=w, height=h,
                    category_id=1, category="person", instances=tuple(insts), crowd_rles=(), meta={"image_id": 1})


def _rect_mask(h, w, x0, y0, x1, y1):
    m = np.zeros((h, w), bool)
    m[y0:y1, x0:x1] = True
    return m


# ---------------------------------------------------------------------------
# 阅读顺序与上限
# ---------------------------------------------------------------------------


def test_reading_order_rows_then_left_to_right():
    # 两行：上行的两个框中心 y 差 4 < min(20, 30)/2 = 10，同一行；下行与上行差很多
    boxes = [[60, 10, 80, 40], [10, 15, 30, 35], [40, 100, 60, 130], [5, 105, 25, 125]]
    assert T.reading_order(boxes) == [1, 0, 3, 2]
    # 刚好不到一半：差 = 10，min 高 = 20 → 10 < 10 不成立，分两行（上面的先）
    assert T.reading_order([[50, 10, 60, 30], [0, 20, 10, 40]]) == [0, 1]
    assert T.reading_order([[50, 10, 60, 30], [0, 19, 10, 39]]) == [1, 0]
    # 不做链式合并：A–B、B–C 各自同行，A–C 不同行 → C 另起一行
    a, b, c = [0, 0, 10, 40], [20, 15, 30, 55], [40, 30, 50, 70]
    assert T.reading_order([c, b, a]) == [2, 1, 0]  # A、B 一行（A 在左），C 下一行
    # 并列：同一位置按原序号
    assert T.reading_order([[0, 0, 10, 10], [0, 0, 10, 10]]) == [0, 1]
    assert T.reading_order([]) == []


def test_reading_order_degenerate_heights():
    # 高度为 0 的框阈值是 0：中心 y 完全相同也不算同一行（|差| < 0 不成立），各自成行；行按 (中心 y, 中心 x) 的先后建立
    assert T.reading_order([[30, 5, 40, 5], [0, 5, 10, 5]]) == [1, 0]
    assert T.reading_order([[30, 5, 40, 5], [0, 5, 10, 5], [50, 0, 60, 10]]) == [1, 0, 2]  # 第三个框与第一行（只有 [1]）不同行


def test_max_30_keeps_reading_order_prefix():
    h, w = 120, 340
    insts = []
    for r in range(3):
        for c in range(11):
            insts.append(_inst(100 + r * 11 + c, _rect_mask(h, w, 5 + 30 * c, 5 + 40 * r, 25 + 30 * c, 35 + 40 * r)))
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(insts))
    row = T.pair_targets(_pair([insts[i] for i in perm], w, h))
    assert row["n_instances"] == 33 and row["n_kept"] == 30 and row["truncated"]
    assert row["ann_ids"] == [100 + k for k in range(30)]  # 第 1、2 行全部 + 第 3 行左起 8 个
    assert len(row["boxes_q"]) == 30 and len(row["ctrl_q"]) == 30
    assert T.parse_box_list(row["box_text"]) == row["boxes_q"]
    assert T.row_problems(row, T.DEFAULT_PARAMS) == []
    few = T.pair_targets(_pair(insts[:5], w, h))
    assert not few["truncated"] and few["n_kept"] == 5


# ---------------------------------------------------------------------------
# 样条
# ---------------------------------------------------------------------------


def _disk(h, w, cx, cy, r):
    yy, xx = np.mgrid[0:h, 0:w]
    return (xx + 0.5 - cx) ** 2 + (yy + 0.5 - cy) ** 2 <= r * r


def test_spline_fit_is_x2_fit_and_clamped_into_box():
    h, w = 96, 128
    m = _disk(h, w, 60, 50, 30)
    # 一个比 mask 小的框：拟合出的控制点有的在框外，夹进来之后都在框内
    box = (40.0, 30.0, 80.0, 70.0)
    inst = _inst(7, m, box=box)
    bq = XP.box_q(box, w, h)
    ctrl, fit, why = T.instance_spline(inst, bq, w, h, 16)
    assert why is None and len(ctrl) == 16
    assert all(bq[0] <= x <= bq[2] and bq[1] <= y <= bq[3] for x, y in ctrl)
    raw = OS.oracle_spline(m, 16, parts=1, holes=False)["loops_q"][0]["ctrl"]
    assert any(not (bq[0] <= x <= bq[2] and bq[1] <= y <= bq[3]) for x, y in raw)  # 确实有夹的必要
    assert ctrl == T.clamp_ctrl(raw, bq)
    # 框就是 mask 的外接框时，与 X-2 的样条目标（x2_data.spline_row）逐个相同、fit_iou 也相同
    inst2 = _inst(8, m)
    bq2 = XP.box_q(inst2.box_xyxy, w, h)
    ctrl2, fit2, _ = T.instance_spline(inst2, bq2, w, h, 16)
    row = {"id": "x", "source": "refcoco", "image": "i.jpg", "image_size": [w, h], "expression": "e"}
    x2row, _ = XD.spline_row(row, m, XP.box_abs(bq2, w, h), 1.0, 16)
    assert ctrl2 == T.clamp_ctrl(x2row["ctrl_q"], bq2)
    if ctrl2 == x2row["ctrl_q"]:
        assert fit2 == pytest.approx(x2row["fit_iou"], abs=0)
    assert 0.85 < fit2 <= 1.0


def test_spline_text_format_matches_x2():
    c1 = [[i, 2 * i] for i in range(16)]
    c2 = [[100 + i, 3 * i] for i in range(16)]
    t = T.spline_list_text([c1, c2])
    assert t == "[" + XP.spline_text(c1) + ", " + XP.spline_text(c2) + "]"
    assert json.loads(t) == [c1, c2] and T.parse_spline_list(t) == [c1, c2]
    one = T.spline_list_text([c1])
    assert one.startswith("[[[") and T.parse_spline_list(one) == [c1]
    with pytest.raises(ValueError):
        T.parse_spline_list("[[[1, 2]]]")  # 点数不是 16
    assert T.box_list_text([[1, 2, 3, 4], [5, 6, 7, 8]]) == "[[1, 2, 3, 4], [5, 6, 7, 8]]"


def test_empty_mask_makes_spline_unavailable():
    h, w = 40, 60
    good = _inst(1, _rect_mask(h, w, 5, 5, 25, 30))
    empty = D.XMInstance(ann_id=2, box_xyxy=(30.0, 5.0, 31.0, 6.0), rle=mask_to_rle(np.zeros((h, w), bool)), area=0)
    row = T.pair_targets(_pair([good, empty], w, h))
    assert not row["spline_ok"] and row["spline_text"] is None and row["ctrl_q"] is None
    assert row["spline_fail"] == [{"ann_id": 2, "reason": T.FAIL_MASK_EMPTY}]
    k = row["ann_ids"].index(2)
    assert len(row["boxes_q"]) == 2 and row["fit_iou"][k] is None and row["fit_iou"][1 - k] > 0.9
    assert T.row_problems(row, T.DEFAULT_PARAMS) == []


def test_row_problems_catch_bad_rows():
    h, w = 40, 60
    row = T.pair_targets(_pair([_inst(1, _rect_mask(h, w, 5, 5, 25, 30)), _inst(2, _rect_mask(h, w, 30, 6, 50, 32))], w, h))
    assert T.row_problems(row, T.DEFAULT_PARAMS) == []
    r = json.loads(json.dumps(row))
    r["boxes_q"] = r["boxes_q"][::-1]
    r["box_text"] = T.box_list_text(r["boxes_q"])
    assert any("阅读顺序" in b for b in T.row_problems(r, T.DEFAULT_PARAMS))
    r = json.loads(json.dumps(row))
    r["ctrl_q"][0][0] = [0, 0]
    r["spline_text"] = T.spline_list_text(r["ctrl_q"])
    assert any("框内" in b for b in T.row_problems(r, T.DEFAULT_PARAMS))
    r = json.loads(json.dumps(row))
    r["spline_text"] = r["spline_text"].replace(", ", ",")
    assert T.row_problems(r, T.DEFAULT_PARAMS)


# ---------------------------------------------------------------------------
# 生成、续跑、读取（假数据，同 scripts/xm/build_targets.py --tiny）
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    d = tmp_path_factory.mktemp("xmt")
    build_fake_root(d / "root")
    sp, ep = d / "xm_splits.json", d / "xm_train_exclusions.json"
    sp.write_text(D.dump_config(D.build_splits_doc(d / "root", tiny=True)), encoding="utf-8")
    ep.write_text(D.dump_config(X.build_exclusions(d / "root", splits_path=sp, tiny=True, workers=1)), encoding="utf-8")
    pairs = D.load_pairs("train", d / "root", splits_path=sp, exclusions_path=ep)
    src = T.source_identity(pairs, splits_path=sp, exclusions_path=ep, train_json_sha256="x")
    return {"dir": d, "root": d / "root", "splits": sp, "excl": ep, "pairs": pairs, "src": src}


def _files(out):
    return {p.name: p.read_bytes() for p in sorted((out / T.DATA_DIR).iterdir())}


def test_build_workers_and_resume_identical(tiny, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    man = T.build(tiny["pairs"], a, params=T.DEFAULT_PARAMS, source=tiny["src"], tiny=True, workers=1, shard_pairs=3, log=lambda m: None)
    assert man["counts"]["n_pairs"] == len(tiny["pairs"]) and man["counts"]["n_truncated_pairs"] == 1
    assert T.build(tiny["pairs"], b, params=T.DEFAULT_PARAMS, source=tiny["src"], tiny=True, workers=2, shard_pairs=3, stop_after_shards=1,
                   log=lambda m: None) is None
    T.build(tiny["pairs"], b, params=T.DEFAULT_PARAMS, source=tiny["src"], tiny=True, workers=2, shard_pairs=3, resume=True, log=lambda m: None)
    assert _files(a) == _files(b)
    with pytest.raises(T.XMTargetsError, match="--resume"):
        T.build(tiny["pairs"], a, params=T.DEFAULT_PARAMS, source=tiny["src"], tiny=True, shard_pairs=3, log=lambda m: None)
    with pytest.raises(T.XMTargetsError, match="不同"):
        T.build(tiny["pairs"], a, params={**T.DEFAULT_PARAMS, "max_boxes": 5}, source=tiny["src"], tiny=True, shard_pairs=3, resume=True,
                log=lambda m: None)
    rows, man2 = T.load_targets(a, data_root=tiny["root"])
    assert [r["id"] for r in rows] == [p.id for p in tiny["pairs"]] and man2 == man
    # 篡改一个文件：sha256 对不上
    f = sorted((a / T.DATA_DIR).glob("targets_*.jsonl"))[0]
    f.write_text(f.read_text().replace('"truncated": false', '"truncated": true', 1))
    with pytest.raises(T.XMTargetsError, match="sha256"):
        T.load_targets(a)


def test_build_targets_script_tiny(tiny, tmp_path):
    spec = importlib.util.spec_from_file_location("xm_build_targets", D.REPO_ROOT / "scripts" / "xm" / "build_targets.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    out = tmp_path / "tg"
    rc = m.main(["--tiny", "--data-root", str(tiny["root"]), "--splits", str(tiny["splits"]), "--exclusions", str(tiny["excl"]),
                 "--out", str(out), "--workers", "1"])
    assert rc == 0
    rows, man = T.load_targets(out)
    assert man["source"]["xm_splits_sha256"] == D.file_sha256(tiny["splits"]) and man["source"]["limit"] is None
    assert {r["image"].rsplit("/", 1)[1] for r in rows}.isdisjoint({f"{i:012d}.jpg" for i in (903, 920, 3006, 3007, 3008)})


class _CharTok:
    """假分词器：一个字符一个 token（只测统计的管线；真分词器在服务器上）。"""

    def __call__(self, texts, add_special_tokens=False):
        return {"input_ids": [list(range(len(t))) for t in texts]}


def test_data_stats_build(tiny, tmp_path):
    spec = importlib.util.spec_from_file_location("xm_data_stats", D.REPO_ROOT / "scripts" / "xm" / "data_stats.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    out = tmp_path / "tg"
    T.build(tiny["pairs"], out, params=T.DEFAULT_PARAMS, source=tiny["src"], tiny=True, log=lambda m_: None)
    doc = m.build(data_root=tiny["root"], targets_dir=out, splits_path=tiny["splits"], exclusions_path=tiny["excl"], tokenizer=_CharTok(),
                  tokenizer_record={"model": "char"}, log=lambda m_: None)
    assert list(doc["splits"]) == ["train", "xm_tune", "xm_decision", "xm_final", "lvis_sub"]
    tr = doc["splits"]["train"]
    assert tr["n_pairs"] == len(tiny["pairs"]) and tr["n_truncated_pairs"] == 1 and tr["n_dropped_by_cap"] == 3
    rows, _ = T.load_targets(out)
    assert tr["tokens"]["box_text"]["max"] == max(len(r["box_text"]) for r in rows)
    assert tr["instances_per_pair"]["max"] == 33
    md = m.render_md(doc)
    assert "| train |" in md and "xm_decision" in md


def test_xm_data_modules_import_no_learned_modules():
    """数据层只用不含学习的工具（硬约束；同 X-2 的 test_x2_tool_modules_import_no_learned_modules）。"""
    import subprocess
    import sys

    from src.explore import guidance as G

    code = ("import json, sys; import src.xm.data as D, src.xm.targets, src.xm.exclusions; import src.data.land.coco; "
            "import src.explore.guidance as G; "
            "print(json.dumps({'loaded': G.loaded_learned_modules(), 'roots': sorted({m.split('.')[0] for m in sys.modules})}))")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=D.REPO_ROOT, check=True)
    doc = json.loads(r.stdout.strip().splitlines()[-1])
    assert doc["loaded"] == [] and not set(doc["roots"]) & set(G.BANNED_MODULE_ROOTS)
    assert "pycocotools" in doc["roots"]
