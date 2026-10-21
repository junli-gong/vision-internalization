"""DEV-0051：W1 的评分（src/w1/scoring.py）。手算的小例子上 cIoU、gIoU、Acc@0.5、计数准确率与 E1 的函数一致；配对 cIoU 的方向对；
id 集合不同报错；精确的 cIoU 没有浮点误差。"""

from __future__ import annotations

from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

from src.data.records import Sample
from src.e1.scoring import metric_of
from src.metrics.box import box_acc_at_iou
from src.metrics.seg import ciou, giou
from src.w1.scoring import aggregate_w1, exact_ciou, paired_ciou, score_episode

H, W = 10, 10


def seg_sample(i: int, poly) -> Sample:
    return Sample(id=f"s/{i}", task="refexp", images=("/x",), image_keys=("k",), fields={"expression": "e"},
                  gt={"box": [0.0, 0.0, 5.0, 5.0], "mask": {"polygons": [poly], "height": H, "width": W}}, source="s", split="x")


def ep(**kw) -> SimpleNamespace:
    base = {"failure": None, "mask": None, "box": None, "count": None, "image_size": (W, H)}
    return SimpleNamespace(**{**base, **kw})


def loop_row(i: int, score: dict, failure=None) -> dict:
    """aggregate_w1 读的 predictions 行（只填它用到的字段）。"""
    return {"id": f"s/{i}", "failure": failure, "ended_by": "submission" if failure is None else "round_cap", "n_rounds": 2,
            "n_tool_calls": 1, "n_executed": 2, "n_refused": 0, "n_tool_call_errors": 0, "thinking_end": ["natural", "forced"],
            "tokens": {"prompt": 10, "generated": 5, "inserted": 0, "visual": 300}, "answer": {"source": "submit_mask"},
            "rounds": [{"tool_calls": [{"name": "draw_box", "executed": True, "refused": None, "result": {"error": None}}]}],
            "score": score}


def test_refseg_scores_hand_example():
    # GT：左上 5×5 的方块（25 像素）
    gt_poly = [0, 0, 5, 0, 5, 5, 0, 5]
    s = [seg_sample(k, gt_poly) for k in range(3)]
    m0 = np.zeros((H, W), bool)
    m0[0:5, 0:5] = True  # 完全对：inter 25，union 25
    m1 = np.zeros((H, W), bool)
    m1[0:5, 0:10] = True  # inter 25，union 50
    sc = [score_episode("refseg", s[0], ep(mask=m0)), score_episode("refseg", s[1], ep(mask=m1)),
          score_episode("refseg", s[2], ep(failure="no_answer"))]
    assert [(x["inter"], x["union"]) for x in sc] == [(25, 25), (25, 50), (0, 25)]
    assert [x["iou"] for x in sc] == [1.0, 0.5, 0.0]
    rows = [loop_row(k, x, None if k < 2 else "no_answer") for k, x in enumerate(sc)]
    m = aggregate_w1("refseg", rows)
    c, g = m["headline"]
    assert (c["name"], g["name"]) == ("cIoU", "gIoU")
    assert c["value"] == pytest.approx(50 / 100) == metric_of("cIoU", rows)["value"]
    assert c["value"] == ciou([(25, 25), (25, 50), (0, 25, True)])
    assert g["value"] == pytest.approx(0.5) == giou([(25, 25), (25, 50), (0, 25, True)])
    assert c["ci"]["level"] == 0.95 and c["ci"]["low"] <= c["value"] <= c["ci"]["high"]
    assert m["failures"] == {"no_answer": 1, "ok": 2} and m["n_failed"] == 1
    assert m["thinking_end"] == {"forced": 3, "natural": 3} and m["tool_usage"]["executed_by_name"] == {"draw_box": 3}
    with pytest.raises(ValueError, match="GT mask 形状"):
        score_episode("refseg", s[0], ep(mask=m0, image_size=(W, H + 1)))
    with pytest.raises(RuntimeError, match="没有 mask"):
        score_episode("refseg", s[0], ep())


def test_box_and_count_scores_match_e1():
    s = seg_sample(0, [0, 0, 5, 0, 5, 5, 0, 5])
    boxes = [[0.0, 0.0, 5.0, 5.0], [0.0, 0.0, 5.0, 10.0], [0.0, 0.0, 2.0, 2.0]]  # IoU 1、0.5、0.16
    sc = [score_episode("refexp_box", s, ep(box=b)) for b in boxes] + [score_episode("refexp_box", s, ep(failure="no_box"))]
    assert [round(x["iou"], 4) for x in sc] == [1.0, 0.5, 0.16, 0.0] and [x["correct"] for x in sc] == [True, True, False, False]
    rows = [loop_row(k, x) for k, x in enumerate(sc)]
    m = aggregate_w1("refexp_box", rows)
    assert m["headline"][0]["name"] == "acc@0.5" and m["headline"][0]["value"] == 0.5 == box_acc_at_iou([x["iou"] for x in sc])
    cs = Sample(id="c/0", task="count", images=("/x",), image_keys=("k",), fields={}, gt={"count": 3}, source="c", split="a")
    got = [score_episode("count", cs, ep(count=3)), score_episode("count", cs, ep(count=4)),
           score_episode("count", cs, ep(failure="no_number"))]
    assert [x["correct"] for x in got] == [True, False, False] and got[0]["gt"] == 3
    m = aggregate_w1("count", [loop_row(k, x) for k, x in enumerate(got)])
    assert m["headline"][0]["name"] == "accuracy" and m["headline"][0]["value"] == pytest.approx(1 / 3)


def _rows(pairs, prefix="s"):
    return [{"id": f"{prefix}/{k}", "score": {"inter": i, "union": u}} for k, (i, u) in enumerate(pairs)]


def test_paired_ciou_direction_and_ci():
    rng = np.random.default_rng(0)
    union = rng.integers(80, 120, 200)
    good = _rows([(int(u * 0.8), int(u)) for u in union])
    bad = _rows([(int(u * 0.4), int(u)) for u in union])
    d = paired_ciou(good, bad)
    assert d["n"] == 200 and d["diff"] > 0.3 and d["ci"]["low"] > 0 and d["a"] > d["b"]
    r = paired_ciou(bad, good)
    assert r["diff"] == pytest.approx(-d["diff"]) and r["ci"]["high"] < 0
    # 顺序无关（按 id 对齐，顺序按 a）
    shuffled = list(reversed(bad))
    assert paired_ciou(good, shuffled)["diff"] == pytest.approx(d["diff"])
    # 噪声里没有差别：CI 跨 0
    noisy_a = _rows([(int(rng.integers(0, u)), int(u)) for u in union])
    noisy_b = _rows([(int(rng.integers(0, u)), int(u)) for u in union])
    n = paired_ciou(noisy_a, noisy_b)
    assert n["ci"]["low"] < 0 < n["ci"]["high"]


def test_paired_ciou_rejects_different_ids():
    a = _rows([(1, 2), (1, 2)])
    b = _rows([(1, 2), (1, 2)], prefix="t")
    with pytest.raises(ValueError, match="id 集合不同"):
        paired_ciou(a, b)
    with pytest.raises(ValueError, match="重复"):
        paired_ciou(a + a[:1], a)


def test_exact_ciou():
    rows = _rows([(1, 2), (1, 2), (1, 6)])
    assert exact_ciou(rows) == Fraction(3, 10)
    assert 0.7 - 0.5 < 0.2  # 浮点下 0.7 − 0.5 = 0.19999999999999996：判定"至少高 0.2"要用精确值
    assert exact_ciou(_rows([(7, 10)])) - exact_ciou(_rows([(1, 2)])) == Fraction(1, 5)
    with pytest.raises(ValueError):
        exact_ciou(_rows([(0, 0)]))
