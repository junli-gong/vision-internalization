"""DEV-0080：X-M 的两次调用（src/xm/prompts.py）——提示原文、目标文本、解析（正常、空、越界、超过 30、条数不符、逐条失败）、生成上限。"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from src.explore import x2_prompts as XP
from src.parsing import parse_generation
from src.xm import prompts as P


def test_constants_and_prompt_text():
    assert P.SPLINE_N == 16 == XP.SPLINE_N_DEFAULT and P.MAX_INSTANCES == 30 and P.GRID_MAX == 1000
    assert P.JSON_SEPARATORS == XP.JSON_SEPARATORS
    t = P.list_prompt(" traffic light ")
    assert t == ('Find every instance of the category "traffic light" in the image. Give the bounding box of each instance in reading order '
                 "(top to bottom, then left to right), at most 30 boxes. Coordinates are integers on a 0-1000 grid of the image, x first. "
                 "If there is no instance, output []. Output only JSON: [[x1, y1, x2, y2], ...].")
    u = P.trace_prompt("dog", [[1, 2, 30, 40], [500, 0, 1000, 999]])
    assert u.startswith('Each numbered box below contains one instance of the category "dog" in the image (2 listed):\n1: [1, 2, 30, 40]\n'
                        '2: [500, 0, 1000, 999]\n')
    assert P.call1_prompt("dog") == P.list_prompt("dog") and P.call2_prompt("dog", [[1, 2, 30, 40]]) == P.trace_prompt("dog", [[1, 2, 30, 40]])
    assert P.format_boxes([[1, 2, 3, 4]]) == P.boxes_text([[1, 2, 3, 4]]) and P.format_splines is P.splines_text
    assert "exactly 16 control points" in u and "a list of 2 splines" in u
    with pytest.raises(ValueError):
        P.trace_prompt("dog", [])
    with pytest.raises(ValueError):
        P.trace_prompt("dog", [[10, 10, 5, 20]])  # 颠倒的框不进提示
    with pytest.raises(ValueError):
        P.list_prompt("  ")
    ident = P.identity()
    assert ident["version"] == P.PROMPT_VERSION and len(ident["templates_sha256"]) == 64 and ident["frozen"] is P.FROZEN


def test_targets_round_trip():
    boxes = [[0, 0, 10, 10], [5, 6, 700, 1000]]
    assert P.boxes_text(boxes) == "[[0, 0, 10, 10], [5, 6, 700, 1000]]" and P.boxes_text([]) == "[]"
    assert P.parse_boxes(P.boxes_text(boxes)).value["boxes_q"] == boxes
    sp = [[[i, 2 * i] for i in range(16)], [[1000 - i, i] for i in range(16)]]
    txt = P.splines_text(sp)
    assert txt.startswith("[[[0, 0], [1, 2], ") and json.loads(txt) == sp
    v = P.parse_splines(txt, k=2).value
    assert v["splines"] == sp and v["failures"] == [None, None] and v["n_clipped"] == 0
    # 单条的写法与 X-2 的样条目标逐字相同；两个目标与训练目标（src/xm/targets，DEV-0079）是同一个函数
    assert json.dumps(sp[0], separators=P.JSON_SEPARATORS) == XP.spline_text(sp[0])
    from src.xm import targets as T

    assert P.splines_text(sp) == T.spline_list_text(sp) and T.parse_spline_list(txt) == sp and P.boxes_text(boxes) == T.box_list_text(boxes)
    assert P.MAX_INSTANCES == T.MAX_BOXES and P.SPLINE_N == T.SPLINE_N


def test_parse_boxes_rules():
    ok = P.parse_boxes("  [[1, 2, 3, 4], [10, 20, 30, 40]] \n")
    assert ok.failure is None and ok.value == {"boxes_q": [[1, 2, 3, 4], [10, 20, 30, 40]], "n_raw": 2, "n_capped": 0, "n_clipped": 0,
                                               "n_inverted": 0}
    assert P.parse_boxes("[]").value["boxes_q"] == [] and P.parse_boxes("[]").failure is None
    clip = P.parse_boxes("[[-5, 10, 1200, 990], [3, 3, 3, 9]]").value
    assert clip["boxes_q"] == [[0, 10, 1000, 990], [3, 3, 3, 9]] and clip["n_clipped"] == 1  # 面积为 0 的框保留
    inv = P.parse_boxes("[[50, 10, 20, 40], [1, 1, 2, 2], [1200, 0, 1100, 5]]").value
    assert inv["boxes_q"] == [[1, 1, 2, 2], [1000, 0, 1000, 5]] and inv["n_inverted"] == 1 and inv["n_clipped"] == 1
    many = [[i, i, i + 1, i + 1] for i in range(35)]
    cap = P.parse_boxes(json.dumps(many)).value
    assert cap["boxes_q"] == many[:30] and cap["n_raw"] == 35 and cap["n_capped"] == 5
    for text, why in [("", "empty"), ("   ", "empty"), ("```json\n[[1, 2, 3, 4]]\n```", "malformed_json"), ("[[1,2,3,4]] extra", "malformed_json"),
                      ('{"bbox_2d": [1, 2, 3, 4]}', "not_list"), ("[[1, 2, 3]]", "bad_box"), ("[[1, 2, 3, 4], 5]", "bad_box"),
                      ("[[1.0, 2, 3, 4]]", "non_integer"), ("[[true, 2, 3, 4]]", "non_integer"), ('[["1", 2, 3, 4]]', "non_integer")]:
        r = P.parse_boxes(text)
        assert r.value is None and P.failure_name(r.failure) == why, text


def test_parse_splines_rules():
    good = [[i * 10, i * 20] for i in range(16)]
    r = P.parse_splines(json.dumps([good, good[:15], good]), k=3)
    assert r.failure is None and r.value["splines"] == [good, None, good] and r.value["failures"] == [None, "wrong_count", None]
    r = P.parse_splines(json.dumps([good, [[1, 2]] * 15 + [[1.5, 2]], [[1, 2, 3]] * 16, "x"]), k=4)
    assert r.value["failures"] == [None, "non_integer", "bad_point", "bad_point"]
    oob = [[-3, 5]] + good[1:]
    r = P.parse_splines(json.dumps([oob]), k=1)
    assert r.value["splines"][0][0] == [0, 5] and r.value["n_clipped"] == 1
    for text, k, why in [(json.dumps([good]), 2, "count_mismatch"), (json.dumps(good), 1, "count_mismatch"), ("", 1, "empty"),
                         ("[[[1, 2]]", 1, "malformed_json"), ('{"a": 1}', 1, "not_list")]:
        r = P.parse_splines(text, k=k)
        assert r.value is None and P.failure_name(r.failure) == why, text
    with pytest.raises(ValueError):
        P.parse_splines("[]", k=0)


def test_parse_generation_and_budgets():
    trunc = SimpleNamespace(text="[[1, 2, 3, 4], [5", thinking_unclosed=False, stopped=False)
    r = parse_generation(trunc, P.parse_boxes)
    assert r.value is None and P.failure_name(r.failure) == "truncated"
    r = parse_generation(SimpleNamespace(text="[[1, 2, 3, 4]]", thinking_unclosed=False, stopped=True), lambda t: P.parse_splines(t, k=1))
    assert r.failure is None and r.value["failures"] == ["bad_point"]  # 逐条失败不算整条失败
    assert P.max_new_tokens_list() == P.LIST_TOKENS_BASE + P.LIST_TOKENS_PER_BOX * 30 == 1504
    # D167：DEV-0079 实测一条 16 点样条约 157 个 token → 200 / 条 + 基数；30 个实例 6,064 ≥ 实测约 4,700
    assert P.max_new_tokens_trace(1) == 64 + 200 and P.max_new_tokens_trace(30) == 64 + 200 * 30 > 157 * 30
    assert "200" in P.identity()["max_new_tokens"]["rule"]
    with pytest.raises(ValueError):
        P.max_new_tokens_trace(0)
