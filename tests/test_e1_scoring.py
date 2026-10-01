"""DEV-0014：E1 的逐条评分与汇总，在构造的小例子上与手算对照；编造类别、截断判失败、思考段未闭合单列。"""

from __future__ import annotations

import math
import types

import numpy as np
import pytest

from src.adapters import gemma4, internvl35, qwen35
from src.adapters.base import GenerationResult
from src.data.records import Sample
from src.e1 import scoring as S
from src.e1.acceptance import answers_agree, compare
from src.parsing import ParseFailure

QWEN = types.SimpleNamespace(parse_one_box=qwen35.parse_one_box, parse_boxes=qwen35.parse_boxes)
GEMMA = types.SimpleNamespace(parse_one_box=gemma4.parse_one_box, parse_boxes=gemma4.parse_boxes)
INTERNVL = types.SimpleNamespace(parse_one_box=internvl35.parse_one_box, parse_boxes=internvl35.parse_boxes)


def gen(text: str, *, stopped: bool = True, unclosed: bool = False, ntok: int = 5) -> GenerationResult:
    return GenerationResult(text="" if unclosed else text, token_ids=tuple(range(ntok)), n_generated_tokens=ntok,
                            n_prompt_tokens=10, n_visual_tokens=4, wall_time_s=0.1, raw_text=text,
                            thinking_unclosed=unclosed, stopped=stopped, n_image_pixels=16)


def ref_sample(k: int = 0, split: str = "val", ds: str = "refcoco_unc") -> Sample:
    """12×9 的图，GT 框 = GT 多边形 = [1, 1, 9, 7]（8×6 = 48 个像素）。"""
    return Sample(id=f"{ds}/{split}/{k}", task="refexp", images=("x",), image_keys=("k",),
                  fields={"expression": "the thing", "sent_id": k, "image_size": [12, 9]},
                  gt={"box": [1.0, 1.0, 9.0, 7.0], "mask": {"polygons": [[1, 1, 9, 1, 9, 7, 1, 7]], "height": 9, "width": 12}},
                  source="refcoco", split=split)


def record(task, parser, sample, res, adapter=QWEN, extras=None, size=(12, 9)):
    rec = {"id": sample.id, "raw_text": res.raw_text, "truncated": res.stopped is False,
           "thinking_unclosed": res.thinking_unclosed, "n_generated_tokens": res.n_generated_tokens,
           "n_visual_tokens": res.n_visual_tokens, "n_image_pixels": res.n_image_pixels, "image_size": list(size),
           "extras": extras or {}}
    rec.update(S.score_generation(task, parser, sample, res, adapter=adapter, image_size=size, extras=extras or {}))
    return rec


# ---------------------------------------------------------------------------
# 定位
# ---------------------------------------------------------------------------


def test_box_iou_by_hand_all_families():
    s = ref_sample()
    # 0–1000 的 [100, 125, 750, 750] → 绝对像素 [1.2, 1.125, 9, 6.75]；与 GT [1, 1, 9, 7] 的交 7.8 × 5.625 = 43.875，
    # 预测面积也是 43.875，并 = 48 → IoU = 0.9140625
    want = 43.875 / 48
    outs = [(QWEN, '```json\n[{"bbox_2d": [100, 125, 750, 750], "label": "x"}]\n```'),
            (GEMMA, '```json\n[{"box_2d": [125, 100, 750, 750], "label": "x"}]\n```'),  # Gemma：y 在前
            (INTERNVL, "<ref>x</ref><box>[[100, 125, 750, 750]]</box>")]
    for ad, text in outs:
        r = record("refexp_box", "one_box", s, gen(text), adapter=ad)
        assert r["failure"] is None and math.isclose(r["score"]["iou"], want, rel_tol=1e-12) and r["score"]["correct"]
        assert r["parsed"]["box"] == pytest.approx([1.2, 1.125, 9.0, 6.75])
    # [0, 0, 500, 500] → [0, 0, 6, 4.5]：交 5 × 3.5 = 17.5，并 27 + 48 − 17.5 = 57.5
    r = record("refexp_box", "one_box", s, gen('[{"bbox_2d": [0, 0, 500, 500]}]'))
    assert math.isclose(r["score"]["iou"], 17.5 / 57.5) and not r["score"]["correct"]
    r = record("refexp_box", "one_box", s, gen('[{"bbox_2d": [0, 0, 5, 5]}, {"bbox_2d": [1, 1, 5, 5]}]'))
    assert r["failure"] == "multiple_boxes" and r["score"] == {"iou": 0.0, "correct": False}


def test_truncated_is_failure_even_if_parseable_and_unclosed_is_separate():
    s = ref_sample()
    good = '[{"bbox_2d": [100, 125, 750, 750]}]'
    r_trunc = record("refexp_box", "one_box", s, gen(good, stopped=False))
    assert r_trunc["failure"] == "truncated" and r_trunc["parsed"] is None and r_trunc["truncated"]
    r_unc = record("refexp_box", "one_box", s, gen("<think>...", unclosed=True, stopped=False))
    assert r_unc["failure"] == "thinking_unclosed"  # 先判思考段未闭合，再判截断
    r_ok = record("refexp_box", "one_box", s, gen(good))
    samples = [ref_sample(0), ref_sample(1), ref_sample(2)]
    for k, r in enumerate((r_trunc, r_unc, r_ok)):
        r["id"] = samples[k].id
    m = S.aggregate("refexp_box", samples, [r_trunc, r_unc, r_ok], has_gt=True)
    assert m["parse"]["failures"] == {"thinking_unclosed": 1, "truncated": 1} and m["parse"]["n_failed"] == 2
    assert m["truncation"] == {"n_truncated": 2, "rate": 2 / 3}  # 截断率按 stopped is False 计（含未闭合的那条）
    assert m["thinking_unclosed"] == {"n": 1, "rate": 1 / 3}
    h = m["headline"][0]
    assert h["name"] == "acc@0.5" and math.isclose(h["value"], 1 / 3) and h["n"] == 3 and h["ci"]["low"] <= h["value"]


def test_coordinate_diagnostic():
    assert S.coord_all_le1_decimal('[{"bbox_2d": [0.1, 0.2, 0.5, 0.9], "label": "a"}]')
    assert S.coord_all_le1_decimal("<ref>a</ref><box>[[0.1, 0.2, 0.3, 0.4]]</box>")
    assert not S.coord_all_le1_decimal("[0, 1, 1, 1]")  # 没有小数
    assert not S.coord_all_le1_decimal("[0.5, 2]")  # 有大于 1 的
    assert not S.coord_all_le1_decimal("no numbers") and not S.coord_all_le1_decimal("[]")
    r = record("refexp_box", "one_box", ref_sample(), gen("<box>[[0.1, 0.2, 0.3, 0.4]]</box>"), adapter=INTERNVL)
    assert r["failure"] == "malformed" and r["coord_le1_decimal"] is True  # InternVL 只认整数，诊断照样计
    r = record("refexp_box", "one_box", ref_sample(), gen('[{"bbox_2d": [0.1, 0.2, 0.9, 0.9]}]'))
    assert r["failure"] is None and r["coord_le1_decimal"] and r["score"]["iou"] < 0.01  # 被当成 0–1000：定位错，不是解析失败


# ---------------------------------------------------------------------------
# 多边形与网格
# ---------------------------------------------------------------------------


def test_polygon_ciou_giou_by_hand():
    s = ref_sample()
    # 左半边矩形 [0, 6] × [0, 9]（54 个像素）与 GT [1, 9] × [1, 7]（48 个）：交 = 列 1..5 × 行 1..6 = 30，并 = 72
    r1 = record("refexp_poly", "polygon", s, gen("[[0, 0], [500, 0], [500, 1000], [0, 1000]]"))
    assert r1["failure"] is None and (r1["score"]["inter"], r1["score"]["union"]) == (30, 72)
    assert math.isclose(r1["score"]["iou"], 30 / 72)
    # 完全重合（两个多边形的并集写法也接受）
    s2 = ref_sample(1)
    r2 = record("refexp_poly", "polygon", s2, gen("[[[83.33333333333333, 111.11111111111111], [750, 111.11111111111111], "
                                                  "[750, 777.7777777777778], [83.33333333333333, 777.7777777777778]]]"))
    assert (r2["score"]["inter"], r2["score"]["union"]) == (48, 48)
    s3 = ref_sample(2)
    r3 = record("refexp_poly", "polygon", s3, gen("[[0, 0], [10, 10]]"))
    assert r3["failure"] == "too_few_points" and (r3["score"]["inter"], r3["score"]["union"]) == (0, 48)
    m = S.aggregate("refexp_poly", [s, s2, s3], [r1, r2, r3], has_gt=True)
    c = next(h for h in m["headline"] if h["name"] == "cIoU")
    g = next(h for h in m["headline"] if h["name"] == "gIoU")
    assert math.isclose(c["value"], (30 + 48 + 0) / (72 + 48 + 48))
    assert math.isclose(g["value"], (30 / 72 + 1 + 0) / 3)
    assert m["headline"][0]["name"] == "cIoU"  # 标题 cIoU
    assert set(m["breakdown"]["split"]) == {"refcoco_unc/val"}
    assert m["diagnostics"]["coord_all_le1_with_decimal"]["n"] == 0


def _grid_sample(k: int) -> Sample:
    """32×18 的图（网格回填要求边长 ≥ 16），GT = [8, 0, 24, 9]：列 8..23 × 行 0..8 = 144 个像素。"""
    return Sample(id=f"refcoco_unc/val/{k}", task="refexp", images=("x",), image_keys=("k",),
                  fields={"expression": "t", "sent_id": k, "image_size": [32, 18]},
                  gt={"box": [8.0, 0.0, 24.0, 9.0], "mask": {"polygons": [[8, 0, 24, 0, 24, 9, 8, 9]], "height": 18, "width": 32}},
                  source="refcoco", split="val")


def test_grid16_upsampled_to_original_image():
    ss = [_grid_sample(0), _grid_sample(1), _grid_sample(2)]
    ones = "\n".join(["1" * 16] * 16)
    r = record("grid16", "grid16", ss[0], gen(f"Here is the grid:\n{ones}"), size=(32, 18))
    assert r["failure"] is None and (r["score"]["inter"], r["score"]["union"]) == (144, 576)  # 全 1 网格回填成整张图
    top = "\n".join(["1" * 16] * 8 + ["0" * 16] * 8)  # 上半 8 行 → 原图行 0..floor(8·18/16)−1 = 0..8，全部 32 列
    r2 = record("grid16", "grid16", ss[1], gen(top), size=(32, 18))
    assert (r2["score"]["inter"], r2["score"]["union"]) == (144, 288)
    r3 = record("grid16", "grid16", ss[2], gen("\n".join(["1" * 15] * 16)), size=(32, 18))
    assert r3["failure"] == "wrong_col_count"
    m = S.aggregate("grid16", ss, [r, r2, r3], has_gt=True)
    assert m["headline"][0]["name"] == "gIoU" and math.isclose(m["headline"][0]["value"], (0.25 + 0.5 + 0) / 3)
    c = m["headline"][1]
    assert c["name"] == "cIoU" and math.isclose(c["value"], (144 + 144) / (576 + 288 + 144))
    assert m["parse"]["failures"] == {"wrong_col_count": 1}


# ---------------------------------------------------------------------------
# 计数与多选
# ---------------------------------------------------------------------------


def _count_sample(k, gt):
    return Sample(id=f"countbench/all/{k}", task="count", images=("x",), image_keys=("k",), fields={"text": "x"},
                  gt={"count": gt}, source="countbench", split="all")


def test_count_accuracy_and_bins():
    ss = [_count_sample(0, 2), _count_sample(1, 3), _count_sample(2, 3), _count_sample(3, 5)]
    outs = ["2", "There are 3 dogs.", "three or four", "5.5"]
    ex = [{"object": "x", "object_rule": r} for r in ("after_number", "after_number", "before_number", "fallback_no_phrase")]
    recs = [record("count", "count", s, gen(o), extras=e) for s, o, e in zip(ss, outs, ex)]
    assert [r["failure"] for r in recs] == [None, None, "multiple_numbers", "malformed"]
    m = S.aggregate("count", ss, recs, has_gt=True)
    assert m["headline"][0]["value"] == 0.5
    assert m["breakdown"]["count"]["3"]["accuracy"]["value"] == 0.5 and m["breakdown"]["count"]["5"]["accuracy"]["value"] == 0.0
    assert m["breakdown"]["object_rule"]["after_number"] == {"n": 2, "accuracy": 1.0}


def _cv(k, ans, typ, src, choices=("1", "2", "3")):
    return Sample(id=f"cvbench/test/{k}", task="spatial_qa", images=("x",), image_keys=("k",),
                  fields={"prompt": "p", "choices": list(choices), "type": typ, "task": "Count", "source_dataset": src},
                  gt={"answer": ans}, source="cvbench", split="test")


def test_mcq_letter_then_option_text_and_cvbench_combined():
    letters_opts = {"letters": ["A", "B", "C"], "options": ["1", "2", "3"]}
    ss = [_cv(0, "C", "2D", "ADE20K Validation Set"), _cv(1, "B", "2D", "COCO 2017 Validation Set"),
          _cv(2, "A", "3D", "Omni3D_SUNRGBD"), _cv(3, "A", "3D", "Omni3D_Hypersim")]
    outs = ["(C)", "2", "A or B", "The answer is A"]
    recs = [record("spatial_cvbench", "choice", s, gen(o), extras=letters_opts) for s, o in zip(ss, outs)]
    assert [r["parsed"] for r in recs] == [{"letter": "C", "route": "letter"}, {"letter": "B", "route": "option_text"},
                                           None, {"letter": "A", "route": "letter"}]
    assert recs[2]["failure"] == "no_choice"
    m = S.aggregate("spatial_cvbench", ss, recs, has_gt=True)
    assert m["headline"][0]["value"] == 0.75 and m["parse_routes"] == {"letter": 2, "option_text": 1}
    assert m["breakdown"]["type"]["2D"]["accuracy"]["value"] == 1.0 and m["breakdown"]["type"]["3D"]["accuracy"]["value"] == 0.5
    # 官方合成：1/2 · ((ADE 1 + COCO 1)/2 + Omni3D 0.5) = 0.75
    assert m["cvbench_official_combined"] == 0.75


def test_nyu_pairs_option_text_route():
    s = Sample(id="nyuv2_pairs/test/p0", task="rel_depth", images=("x",), image_keys=("k",), fields={},
               gt={"closer": "B"}, source="nyuv2_pairs", split="test")
    ex = {"letters": ["A", "B"], "options": ["A is closer", "B is closer"]}
    for text, letter, route in [("B", "B", "letter"), ("(A)", "A", "letter"), ("B is closer", "B", "option_text"),
                                ("Point B is closer to the camera.", "B", "option_text")]:
        r = record("rel_depth_nyu", "choice", s, gen(text), extras=ex)
        assert r["parsed"] == {"letter": letter, "route": route} and r["score"]["correct"] == (letter == "B")
    r = record("rel_depth_nyu", "choice", s, gen("A is closer than B is closer"), extras=ex)
    assert r["failure"] == "multiple_choices"


def test_parse_mcq_empty_string_contract():
    assert S.parse_mcq("", letters=["A", "B"], options=["x", "y"]) == S.MCQParse(None, None, ParseFailure.NO_CHOICE)


# ---------------------------------------------------------------------------
# 检测
# ---------------------------------------------------------------------------


def _ovd(k, boxes, *, size=(100, 100), source="coco_val2017"):
    return Sample(id=f"{source}/val2017/{k}", task="ovd", images=("x",), image_keys=("k",),
                  fields={"categories": ["person", "car"], "category_ids": [1, 3], "image_id": k, "image_size": list(size)},
                  gt={"boxes": [{"box": b, "category_id": c, "category": n, "iscrowd": 0, "ann_id": i}
                                for i, (b, c, n) in enumerate(boxes)]}, source=source, split="val2017")


def test_detection_fabricated_unlabeled_and_f1_by_hand():
    s = _ovd(1, [([10, 10, 50, 50], 1, "person"), ([60, 60, 90, 90], 3, "car")])
    text = ('```json\n[{"bbox_2d": [100, 100, 500, 500], "label": "Person"}, {"bbox_2d": [600, 600, 900, 900], '
            '"label": "truck"}, {"bbox_2d": [0, 0, 50, 50]}]\n```')
    r = record("ovd", "boxes", s, gen(text), size=(100, 100))
    assert r["parsed"]["category_ids"] == [1, S.UNKNOWN_CATEGORY_ID, S.UNKNOWN_CATEGORY_ID]  # "Person" 规范化后命中
    assert r["score"] == {"n_pred": 3, "n_fabricated": 1, "n_unlabeled": 1}
    s2 = _ovd(2, [([0, 0, 40, 40], 3, "car")])
    r2 = record("ovd", "boxes", s2, gen('[{"bbox_2d": [0, 0, 400, 400], "label": "car"}]'), size=(100, 100))
    s3 = _ovd(3, [([0, 0, 10, 10], 1, "person")])
    r3 = record("ovd", "boxes", s3, gen('[{"bbox_2d": [0, 0, 1, 1]', stopped=False), size=(100, 100))
    assert r3["failure"] == "truncated" and r3["score"]["n_pred"] == 0
    m = S.aggregate("ovd", [s, s2, s3], [r, r2, r3], has_gt=True)
    d = m["breakdown"]["dataset"]["coco_val2017"]
    # TP = 2（person、car），FP = 2（truck、无标签），n_gt = 4 → P = 1/2，R = 1/2，F1 = 2·2/(4 + 4) = 0.5
    assert (d["n_tp"], d["n_fp"], d["n_gt"]) == (2, 2, 4) and d["P50"] == 0.5 and d["R50"] == 0.5
    assert d["F1_50"]["value"] == 0.5 and d["F1_50"]["ci"] is not None
    assert d["fabricated_rate"] == 1 / 4 and d["n_unlabeled"] == 1
    assert m["headline"][0] == {"name": "F1@0.5", "scope": "coco_val2017", **d["F1_50"]}
    assert m["fabricated"] == {"n_pred_boxes": 4, "n_fabricated": 1, "rate": 0.25, "n_unlabeled": 1}
    assert d["AP"]["mean"] is not None and d["n_tiebreaks"] == 20  # 无分数 → K 次打破平局
    assert d["AP_ci"] is None  # 默认不算 mAP 的 CI
    m5 = S.aggregate("ovd", [s, s2, s3], [r, r2, r3], has_gt=True, map_ci_resamples=5)
    assert m5["breakdown"]["dataset"]["coco_val2017"]["AP_ci"]["n_resamples"] == 5


def test_detection_odinw_mean_and_no_gt_dev():
    a = _ovd(1, [([10, 10, 50, 50], 1, "person")], source="odinw13/A")
    b = _ovd(2, [([10, 10, 50, 50], 1, "person")], source="odinw13/B")
    ra = record("ovd", "boxes", a, gen('[{"bbox_2d": [100, 100, 500, 500], "label": "person"}]'), size=(100, 100))
    rb = record("ovd", "boxes", b, gen("[]"), size=(100, 100))
    m = S.aggregate("ovd", [a, b], [ra, rb], has_gt=True)
    h = next(x for x in m["headline"] if x["scope"] == "odinw13_mean_over_datasets")
    assert h["value"] == 0.5 and h["n"] == 2  # (1 + 0) / 2
    dev = Sample(id="ovd_dev/x/5", task="ovd", images=("x",), image_keys=("k",),
                 fields={"categories": ["person"], "category_ids": [1], "image_id": 5, "image_size": [100, 100]},
                 gt={"boxes": None}, source="ovd_dev/x", split="dev")
    rd = record("ovd", "boxes", dev, gen('[{"bbox_2d": [1, 1, 5, 5], "label": "dragon"}]'), size=(100, 100))
    md = S.aggregate("ovd", [dev], [rd], has_gt=False)
    assert md["headline"] == [] and md["fabricated"]["rate"] == 1.0


# ---------------------------------------------------------------------------
# 验收的一致判据
# ---------------------------------------------------------------------------


def test_acceptance_agreement_rules():
    s = ref_sample()
    a = record("refexp_poly", "polygon", s, gen("[[0, 0], [500, 0], [500, 1000], [0, 1000]]"))
    b = record("refexp_poly", "polygon", s, gen("[[0, 0], [510, 0], [510, 1000], [0, 1000]]"))  # 多 1 列以内
    c = record("refexp_poly", "polygon", s, gen("[[0, 0], [1000, 0], [1000, 1000], [0, 1000]]"))
    f1 = record("refexp_poly", "polygon", s, gen("x"))
    f2 = record("refexp_poly", "polygon", s, gen("[]"))
    assert answers_agree("refexp_poly", a, a) and not answers_agree("refexp_poly", a, c)
    assert answers_agree("refexp_poly", a, b)  # x 到 6.12：像素中心 6.5 仍在外，mask 相同
    d = record("refexp_poly", "polygon", s, gen("[[0, 0], [590, 0], [590, 1000], [0, 1000]]"))  # 多 1 列：54/63 < 0.9
    assert not answers_agree("refexp_poly", a, d)
    assert answers_agree("refexp_poly", f1, f2) and not answers_agree("refexp_poly", f1, a)  # 都失败算一致
    x = record("refexp_box", "one_box", s, gen('[{"bbox_2d": [100, 125, 750, 750]}]'))
    y = record("refexp_box", "one_box", s, gen('[{"bbox_2d": [100, 125, 750, 751]}]'))
    assert answers_agree("refexp_box", x, x) and not answers_agree("refexp_box", x, y)  # 框要逐位相等
    out = compare("refexp_box", [s], [x], [y], has_gt=True, batch_size=4)
    assert out["answer_agreement"] == 0.0 and out["pass"] is False and out["diff"] == 0.0 and out["diff_ci_contains_0"]
    same = compare("refexp_box", [s], [x], [x], has_gt=True, batch_size=4)
    assert same["pass"] and same["raw_text_equal_rate"] == 1.0
