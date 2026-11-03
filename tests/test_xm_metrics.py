"""DEV-0080：X-M 的指标（src/xm/metrics.py；XM-3a）——手算的小例子（COCOeval 式匹配、贪心与匈牙利、crowd 忽略、空预测、空真值、
LVIS 非穷尽）、与 pycocotools COCOeval 的匹配逐个相同、Rex-Omni 的宏平均（含 safe_mean 的诊断列、F1@mIoU）、微平均、
按图像重抽的 bootstrap 与 src.stats 的参考实现逐位相同、指代分割的 gIoU / cIoU。"""

from __future__ import annotations

import contextlib
import io

import numpy as np
import pytest

from src import stats as S
from src.executors.raster import mask_to_rle
from src.metrics.seg import ciou, giou
from src.xm import metrics as M


def _m(ious, crowd=None, t=0.5, **kw):
    ious = np.asarray(ious, float)
    crowd = np.zeros((ious.shape[0], 0)) if crowd is None else np.asarray(crowd, float)
    return M.match(ious, crowd, t, **kw)


def test_constants():
    assert M.MATCHING == "coco" and M.MAX_DETS == 100 and M.HEADLINE_IOU == 0.5 and len(M.IOU_THRESHOLDS) == 10
    assert M.IOU_THRESHOLDS[0] == 0.5 and abs(M.IOU_THRESHOLDS[-1] - 0.95) < 1e-12 and M.DECISION_N_RESAMPLES == 10_000
    assert M.PRIMARY == "f1_rexomni" and M.identity()["seed"] == 20260929


def test_matching_hand_examples():
    ious = [[0.9, 0.6], [0.95, 0.1], [0.2, 0.3]]
    # COCOeval：按输出顺序，预测 0 先占了真值 0（它对真值 0 的 IoU 最大），预测 1 只剩真值 1（0.1 < 0.5）→ FP
    co = _m(ious)
    assert co.gt_of == (0, -1, -1) and co.counts(2) == {"tp": 1, "fp": 2, "fn": 1, "ignored": 0}
    # 按 IoU 从高到低的贪心：(1, 0) = 0.95 先配，预测 0 退到真值 1（0.6）
    assert _m(ious, method="greedy_iou").gt_of == (1, 0, -1)
    # 匈牙利：先使对数最多（2 对），再使 IoU 之和最大（0.6 + 0.95 > 0.9 + 0.1 不可行）
    assert _m(ious, method="hungarian").gt_of == (1, 0, -1)
    # 贪心与匈牙利不同的例子：贪心先拿 0.9，匈牙利为了多配一对改走 0.8 + 0.7
    ious2 = [[0.9, 0.8], [0.7, 0.1]]
    assert _m(ious2, method="greedy_iou").gt_of == (0, -1) and _m(ious2, method="hungarian").gt_of == (1, 0)
    assert _m(ious2).gt_of == (0, -1)  # COCOeval：预测 0 拿最大的 0.9
    # 重复的框记 FP
    dup = _m([[0.8], [0.8], [0.8]])
    assert dup.counts(1) == {"tp": 1, "fp": 2, "fn": 0, "ignored": 0}
    # COCOeval 的同值取后一个真值
    assert _m([[0.7, 0.7]]).gt_of == (1,)
    # 阈值：IoU 恰为 t 算配上
    assert _m([[0.5]]).gt_of == (0,) and _m([[0.4999]]).gt_of == (-1,)
    with pytest.raises(ValueError):
        _m(ious, method="magic")


def test_crowd_and_edge_cases():
    # 预测 1 没配上非 crowd 真值，但与 crowd 区域的 IoA ≥ 0.5 → 忽略；预测 2 的 IoA 0.4 → FP
    ious = [[0.8], [0.1], [0.0]]
    crowd = [[0.9], [0.6], [0.4]]
    for method in M.MATCHINGS:
        m = _m(ious, crowd, method=method)
        assert m.gt_of == (0, -1, -1) and m.ignored == (False, True, False), method
        assert m.counts(1) == {"tp": 1, "fp": 1, "fn": 0, "ignored": 1}
    # COCOeval：已经能配上非 crowd 的预测不会被 crowd 抢走
    assert _m([[0.55]], [[0.99]]).gt_of == (0,)
    # crowd 可以被多个预测重复"配上"
    assert _m([[0.0], [0.0]], [[0.7], [0.7]]).ignored == (True, True)
    # 空预测：真值全 FN；空真值：预测全 FP（crowd 上的除外）
    e = M.eval_pair(np.zeros((0, 3)), np.zeros((0, 0)), 3)
    assert e["tp"] == [0] * 10 and e["fp"] == [0] * 10 and e["fn"] == [3] * 10 and e["matched_iou"] == []
    e = M.eval_pair(np.zeros((2, 0)), np.array([[0.0], [0.8]]), 0)
    assert e["tp"][0] == 0 and e["fp"][0] == 1 and e["ignored"][0] == 1 and e["fn"][0] == 0
    # 非穷尽类别（LVIS）：没配上的预测一律忽略
    e = M.eval_pair(np.array([[0.9], [0.1]]), np.zeros((2, 0)), 1, not_exhaustive=True)
    assert e["tp"][0] == 1 and e["fp"][0] == 0 and e["ignored"][0] == 1
    # 阈值逐档：IoU 0.72 在 0.7 配上、0.75 不配上
    e = M.eval_pair(np.array([[0.72]]), np.zeros((1, 0)), 1)
    assert e["tp"] == [1, 1, 1, 1, 1, 0, 0, 0, 0, 0] and e["matched_iou"] == [0.72]
    # maxDets：多于 100 个只看前 100 个
    e = M.eval_pair(np.zeros((105, 1)), np.zeros((105, 0)), 1)
    assert e["fp"][0] == 100


def _random_case(rng, n_img=6, n_cat=3):
    gts, dts, crowds = {}, {}, {}
    for i in range(n_img):
        for c in range(n_cat):
            k = int(rng.integers(0, 4))
            g = []
            for _ in range(k):
                x, y = rng.uniform(0, 60, 2)
                g.append([x, y, x + rng.uniform(5, 30), y + rng.uniform(5, 30)])
            d = []
            for b in g:
                if rng.random() < 0.8:
                    j = rng.normal(0, 3, 4)
                    d.append([b[0] + j[0], b[1] + j[1], b[2] + j[2], b[3] + j[3]])
                    if rng.random() < 0.3:
                        d.append(list(d[-1]))  # 重复
            for _ in range(int(rng.integers(0, 2))):
                x, y = rng.uniform(0, 60, 2)
                d.append([x, y, x + 10, y + 10])
            perm = rng.permutation(len(d))
            dts[(i, c)] = [d[q] for q in perm]
            gts[(i, c)] = g
            crowds[(i, c)] = [[20.0, 20.0, 60.0, 60.0]] if rng.random() < 0.3 else []
    return gts, dts, crowds


def test_coco_matching_equals_pycocotools_cocoeval():
    """COCOeval（scores 按输出顺序递减）的 dtMatches / dtIgnore 与本模块的 coco 匹配逐个相同，10 个阈值都比。"""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    rng = np.random.default_rng(3)
    gts, dts, crowds = _random_case(rng)
    images = [{"id": i + 1, "width": 100, "height": 100} for i in range(6)]
    anns, dets, aid = [], [], 1
    for (i, c), g in gts.items():
        for b in g + crowds[(i, c)]:
            crowd = b in crowds[(i, c)]
            anns.append({"id": aid, "image_id": i + 1, "category_id": c + 1, "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]],
                         "area": (b[2] - b[0]) * (b[3] - b[1]), "iscrowd": int(crowd)})
            aid += 1
        for k, b in enumerate(dts[(i, c)]):
            dets.append({"image_id": i + 1, "category_id": c + 1, "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]], "score": 1.0 - k / 1000})
    with contextlib.redirect_stdout(io.StringIO()):
        cg = COCO()
        cg.dataset = {"images": images, "annotations": anns, "categories": [{"id": c + 1} for c in range(3)]}
        cg.createIndex()
        cd = cg.loadRes(dets)
        ev = COCOeval(cg, cd, iouType="bbox")
        ev.params.areaRng, ev.params.areaRngLbl, ev.params.maxDets = [[0, 1e10]], ["all"], [100]
        ev.evaluate()
    by = {(e["image_id"] - 1, e["category_id"] - 1): e for e in ev.evalImgs if e is not None}
    n_checked = 0
    for key, d in dts.items():
        crowd_rles = [mask_to_rle(np.pad(np.ones((40, 40), bool), ((20, 40), (20, 40)))) for _ in crowds[key]]
        ious, ioa = M.box_ious(d, gts[key], crowd_rles)
        e = M.eval_pair(ious, ioa, len(gts[key]))
        if key not in by:
            assert not d and not gts[key]
            continue
        ce = by[key]
        for t in range(10):
            tp = int(((ce["dtMatches"][t] > 0) & ~ce["dtIgnore"][t]).sum())
            fp = int(((ce["dtMatches"][t] == 0) & ~ce["dtIgnore"][t]).sum())
            assert (e["tp"][t], e["fp"][t]) == (tp, fp), (key, t)
            assert e["tp"][t] + e["fn"][t] == int((~ce["gtIgnore"].astype(bool)).sum())
        n_checked += 1
    assert n_checked >= 10


def _rows(spec):
    """spec：[(图, 类别, tp, fp, fn)]（10 档相同），matched_iou 每个 TP 记 0.8。"""
    return [{"id": f"{i}/{c}/{k}", "image_key": f"img{i}", "category_id": c,
             "det": {"tp": [tp] * 10, "fp": [fp] * 10, "fn": [fn] * 10, "ignored": [0] * 10, "matched_iou": [0.8] * tp,
                     "flags": [[1] * tp + [0] * fp] * 10}}
            for k, (i, c, tp, fp, fn) in enumerate(spec)]


def test_rexomni_macro_hand_example():
    # 类别 1：TP 3、FP 1、FN 1 → P 0.75、R 0.75；类别 2：TP 1、FP 3、FN 0 → P 0.25、R 1；类别 3：没有预测（TP 0、FP 0、FN 2）→ 不进 P，R 0
    rows = _rows([(0, 1, 2, 0, 1), (1, 1, 1, 1, 0), (1, 2, 1, 3, 0), (2, 3, 0, 0, 2)])
    tb = M.table(rows)
    p, r, f = M.prf_from_totals(tb.totals(), "f1_rexomni")
    assert p == pytest.approx(0.5) and r == pytest.approx((0.75 + 1 + 0) / 3) and f == pytest.approx(2 * 0.5 * r / (0.5 + r))
    # Rex-Omni 的复现（诊断）：P_c 是插值精度在最高的 > 0 召回档的值——类别 1 的 [TP, TP, TP, FP] 在召回 0.75 那一档是 1.0（不是 0.75），
    # 类别 2 的 [TP, FP, FP, FP] 是 1.0；类别 3 没有预测，precision 不清零、沿用类别 2 的 1.0，召回 0 被 safe_mean 丢掉
    ex = M.rexomni_exact(rows)["f1_rexomni_exact"]
    assert ex["precision"] == pytest.approx(1.0) and ex["recall"] == pytest.approx(0.875) and ex["value"] == pytest.approx(1.75 / 1.875)
    rows_r = [dict(r, det={**r["det"], "flags": [list(reversed(f)) for f in r["det"]["flags"]]}) for r in rows]  # FP 在前：顺序有影响
    assert M.rexomni_exact(rows_r)["f1_rexomni_exact"]["precision"] == pytest.approx((0.75 + 0.25 + 0.25) / 3)
    # 微平均：ΣTP 4、ΣFP 4、ΣFN 3 → 2·4 / (2·4 + 4 + 3)
    assert M.prf_from_totals(tb.totals(), "f1_micro")[2] == pytest.approx(8 / 15)
    assert M.point(tb, "matched_mean_iou") == pytest.approx(0.8)
    # F1@mIoU：P、R 先在 10 档上平均，再调和平均
    rows2 = [{"id": "a", "image_key": "i", "category_id": 1, "det": {"tp": [2] * 5 + [0] * 5, "fp": [0] * 5 + [2] * 5, "fn": [0] * 5 + [2] * 5,
                                                                     "ignored": [0] * 10, "matched_iou": [0.8, 0.8]}}]
    p, r, f = M.prf_from_totals(M.table(rows2).totals(), "f1_rexomni_miou")
    assert p == pytest.approx(0.5) and r == pytest.approx(0.5) and f == pytest.approx(0.5)
    rows2[0]["det"]["flags"] = [[1, 1]] * 5 + [[0, 0]] * 5
    ex = M.rexomni_exact(rows2)["f1_rexomni_exact_miou"]  # 后 5 档没有 TP：精度沿用前一档的 1.0（不清零），召回 0 照加
    assert ex["precision"] == pytest.approx(1.0) and ex["recall"] == pytest.approx(0.5) and ex["value"] == pytest.approx(2 / 3)
    # 没有任何预测：P 无定义，F1 记 0
    p, r, f = M.prf_from_totals(M.table(_rows([(0, 1, 0, 0, 3)])).totals(), "f1_rexomni")
    assert p is None and r == 0 and f == 0.0
    out = M.readout(rows, n_resamples=200)
    assert out["f1_rexomni_exact"]["value"] == pytest.approx(1.75 / 1.875) and "ci" not in out["f1_rexomni_exact"]
    assert out["f1_rexomni"]["value"] == pytest.approx(f1 := 2 * 0.5 * (1.75 / 3) / (0.5 + 1.75 / 3)) and out["primary"] == "f1_rexomni"
    assert out["totals_iou50"] == {"tp": 4, "fp": 4, "fn": 3, "ignored": 0} and out["n_images"] == 3 and out["n_categories"] == 3
    assert out["f1_rexomni"]["ci"]["n_resamples"] == 200 and out["f1_rexomni"]["ci"]["low"] <= f1 <= out["f1_rexomni"]["ci"]["high"]


def _random_rows(rng, n_img=25, seed_shift=0):
    rows = []
    for i in range(n_img):
        for c in range(4):
            if rng.random() < 0.5:
                tp = [int(rng.integers(0, 4))]
                for _ in range(9):
                    tp.append(max(0, tp[-1] - int(rng.integers(0, 2))))
                fp = [int(rng.integers(0, 3)) + (tp[0] - t) for t in tp]
                n_gt = tp[0] + int(rng.integers(0, 3))
                rows.append({"id": f"{i}/{c}", "image_key": f"im{i}", "category_id": c,
                             "det": {"tp": tp, "fp": fp, "fn": [n_gt - t for t in tp], "ignored": [0] * 10,
                                     "matched_iou": list(rng.uniform(0.5, 1.0, tp[0])), "flags": [[1] * t + [0] * f for t, f in zip(tp, fp)]}})
    return rows


def test_bootstrap_equals_src_stats_reference():
    rng = np.random.default_rng(11)
    rows = _random_rows(rng)
    tb = M.table(rows)
    for st in M.STATISTICS:
        data = tb.iou if st == "matched_mean_iou" else tb.dense()
        a = M.bootstrap_readout(tb, st, n_resamples=300)
        b = S.bootstrap_ci(data, statistic=M.statistic_fn(st), n_resamples=300)
        assert (a.estimate, a.low, a.high) == (b.estimate, b.low, b.high), st
    # 配对：两个系统同一批对（换掉 B 的计数）
    rows_b = [dict(r, det={**r["det"], "fp": [v + 1 for v in r["det"]["fp"]]}) for r in rows]
    tb_b = M.table(rows_b)
    for st in ("f1_rexomni", "f1_micro", "f1_rexomni_miou", "matched_mean_iou"):
        da, db = (tb.iou, tb_b.iou) if st == "matched_mean_iou" else (tb.dense(), tb_b.dense())
        a = M.paired_bootstrap(tb, tb_b, st, n_resamples=300)
        b = S.paired_bootstrap_ci(da, db, statistic=M.statistic_fn(st), n_resamples=300)
        assert (a.estimate, a.low, a.high) == (b.estimate, b.low, b.high), st
    assert M.paired_bootstrap(tb, tb_b, "f1_rexomni").n_resamples == 10_000  # 判定缺省 10,000 次
    # 微平均的统计量与 src.stats.f1_statistic 一致（按图的 (tp, fp, n_gt)）
    per_img = np.zeros((len(tb.images), 3))
    for r in rows:
        k = tb.images.index(r["image_key"])
        per_img[k] += [r["det"]["tp"][0], r["det"]["fp"][0], r["det"]["tp"][0] + r["det"]["fn"][0]]
    assert M.point(tb, "f1_micro") == pytest.approx(S.f1_statistic(per_img), abs=1e-15)
    # 对不齐就拒绝
    with pytest.raises(ValueError, match="配对"):
        M.paired_bootstrap(tb, M.table(rows[1:]), "f1_rexomni")


def test_mask_ious_and_refseg_readout():
    a = np.zeros((10, 10), bool)
    a[2:6, 2:6] = True
    b = np.zeros((10, 10), bool)
    b[2:6, 4:8] = True
    crowd = np.zeros((10, 10), bool)
    crowd[:, :5] = True
    iou, ioa = M.mask_ious([mask_to_rle(a)], [mask_to_rle(b)], [mask_to_rle(crowd)])
    assert iou[0, 0] == pytest.approx(8 / 24) and ioa[0, 0] == pytest.approx(12 / 16)
    unc = {"size": [10, 10], "counts": [20, 10, 70]}  # 未压缩的 RLE（COCO 的 crowd 写法）
    assert M.mask_ious([mask_to_rle(a)], [], [unc])[1].shape == (1, 1)
    recs = [{"score": {"inter": 5, "union": 10}, "answer": {}}, {"score": {"inter": 0, "union": 4}, "answer": None, "failure": "no_box"},
            {"score": {"inter": 0, "union": 0}, "answer": {}}]
    out = M.refseg_readout(recs)
    assert out["cIoU"] == ciou([(5, 10), (0, 4), (0, 0)]) and out["gIoU"] == giou([(5, 10, False), (0, 4, True), (0, 0, False)])
    assert out["gIoU"] == pytest.approx((0.5 + 0 + 1) / 3)
