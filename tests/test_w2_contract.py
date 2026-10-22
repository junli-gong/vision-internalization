"""DEV-0059：候选生成器与渲染的真实现（DEV-0058 的 src/w2/candidates.py、src/w2/render.py）合入之后，核对它们与这里的假实现
（tests/_w2_fakes.py）同一套签名（docs/w2_plan.md §1、§2），configs/w2.json 的生成器类型都是真实现认得的，并且不注入假实现时
run_w2.py --tiny 的 w2a 也能跑通。合入之前这些测试跳过。"""

from __future__ import annotations

import dataclasses
import importlib.util
import inspect
import json
from pathlib import Path

import pytest

from src.w2.config import load_w2_config
from tests import _w2_fakes as F

REPO = Path(__file__).resolve().parents[1]
HAVE_CANDIDATES = importlib.util.find_spec("src.w2.candidates") is not None
HAVE_RENDER = importlib.util.find_spec("src.w2.render") is not None


def _params(fn) -> list[tuple[str, str]]:
    return [(p.name, p.kind.name) for p in inspect.signature(fn).parameters.values()]


def test_fakes_follow_the_plan_signatures():
    assert [f.name for f in dataclasses.fields(F.Candidate)] == ["name", "mask", "ok", "info"]
    assert [f.name for f in dataclasses.fields(F.CandidateSpec)] == ["generators"]
    assert _params(F.generate_candidates) == [("image_rgb", "POSITIONAL_OR_KEYWORD"), ("box_abs", "POSITIONAL_OR_KEYWORD"),
                                              ("fg_points_abs", "KEYWORD_ONLY"), ("bg_points_abs", "KEYWORD_ONLY"), ("spec", "KEYWORD_ONLY")]
    assert _params(F.render_selection) == [("image", "POSITIONAL_OR_KEYWORD"), ("candidates", "POSITIONAL_OR_KEYWORD"),
                                           ("layout", "KEYWORD_ONLY"), ("style", "KEYWORD_ONLY")]


@pytest.mark.skipif(not HAVE_CANDIDATES, reason="DEV-0058（src/w2/candidates.py）还没合入")
def test_real_candidates_match_the_fakes():
    from src.w2 import candidates as RC

    assert [f.name for f in dataclasses.fields(RC.Candidate)] == ["name", "mask", "ok", "info"]
    assert "generators" in [f.name for f in dataclasses.fields(RC.CandidateSpec)]
    assert _params(RC.generate_candidates) == _params(F.generate_candidates)
    assert RC.DEFAULT_SPEC.generators[0] == {"type": "grabcut_default"}
    types = set(RC.GENERATOR_TYPES)
    unknown = [g["type"] for g in load_w2_config().generators if g["type"] not in types]
    assert not unknown, f"configs/w2.json 里有真实现不认得的生成器类型 {unknown}（占位没换？）"


@pytest.mark.skipif(not HAVE_RENDER, reason="DEV-0058（src/w2/render.py）还没合入")
def test_real_render_matches_the_fakes():
    from src.w2 import render as RR

    assert _params(RR.render_selection) == _params(F.render_selection)


@pytest.mark.skipif(not (HAVE_CANDIDATES and HAVE_RENDER), reason="DEV-0058 还没合入")
def test_tiny_w2a_with_the_real_implementations(tmp_path):
    """不注入假的候选与渲染（默认模块）：框换成合法的之后，w2a 用真实现生成候选、渲染挑选图，w1f 交 1 号。"""
    from tests.test_w2_cli import inject_boxes, jsonl, leaf, run, w2

    cfg = REPO / "configs" / "w2.json"
    out = tmp_path / "o"
    run(w2("refexp_box", "devtune800", "baseline_e1", out, cfg), tmp_path)
    inject_boxes(leaf(out, "refexp_box", "baseline_e1", "devtune800"), "devtune800")
    run(w2("refseg", "devtune800", "w2a", out, cfg, "--adapter-factory", "tests._w2_fakes:make_scripted_select_adapter"), tmp_path)
    run(w2("refseg", "devtune800", "w1f", out, cfg), tmp_path)
    recs = jsonl(leaf(out, "refseg", "w2a", "devtune800") / "predictions.jsonl")
    k = load_w2_config().k
    assert all(r["n_candidates"] == k for r in recs[1:]) and all(r["select"]["images"] for r in recs[1:])
    m = json.loads((leaf(out, "refseg", "w2a", "devtune800") / "metrics.json").read_text())
    assert m["complete"] is True and m["n_with_candidates"] == 4
    assert all(r["final_index"] == 1 for r in jsonl(leaf(out, "refseg", "w1f", "devtune800") / "predictions.jsonl")[1:])
