"""DEV-0014：E1 提示配置 configs/e1_prompts.json（结构、出处字段、sha256、渲染、CountBench 问句规则）。"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path

import pytest

from src.adapters import registry
from src.constants import GRID_SIZE_E1, GRID_FG_AREA_THRESHOLD
from src.data.records import Sample
from src.e1 import prompts as P
from src.e1.tasks import DataPaths, select

REPO = Path(__file__).resolve().parents[1]
CFG_PATH = REPO / "configs" / "e1_prompts.json"
# 配置的 sha256：配置改了（包括按截断率调 max_new_tokens）就要更新这里，并在 max_new_tokens_adjustments 里记理由；
# 冻结后写进 PREREGISTRATION.md 的 E1 条目。
E1_PROMPTS_SHA256 = "974d3d730e903a6f0a0727a13fceaa52437b3a048914b90836f8b918afcdc0fa"
FAM = registry.FAMILIES


@pytest.fixture(scope="module")
def cfg():
    return P.load_prompt_config(CFG_PATH)


def test_sha256_is_pinned_and_stable(cfg):
    raw = CFG_PATH.read_bytes()
    assert cfg.sha256 == hashlib.sha256(raw).hexdigest() == P.file_sha256(CFG_PATH) == E1_PROMPTS_SHA256
    assert P.load_prompt_config(CFG_PATH).sha256 == cfg.sha256  # 读两次同一个值


def test_structure_tasks_by_family_with_sources(cfg):
    d = cfg.data
    assert set(d["tasks"]) == set(P.TASKS) and len(P.TASKS) == 9
    for t, spec in d["tasks"].items():
        assert set(spec["families"]) == set(FAM), t
        for fam, e in spec["families"].items():
            for k in P.REQUIRED_FAMILY_FIELDS:
                assert k in e, (t, fam, k)
            assert e["source_url"].strip() and e["source_quote"].strip() and e["note"].strip()
            assert e["parse"] in P.PARSERS and e["max_new_tokens"] >= 1
            assert e["native"] == (e["provenance"] in P.NATIVE_PROVENANCES)
            if not e["native"]:
                assert "非原生" in e["note"] or "统一提示" in e["note"], (t, fam)  # 非原生的一律标明


def test_max_new_tokens_match_preregistration_draft(cfg):
    """预注册草稿 §3：定位 128；检测 512；多边形 1024；数字网格 1024；计数、二选一、多选 64。思考 4096 / 8192（D28-10）。"""
    want = {"refexp_box": 128, "ovd": 512, "refexp_poly": 1024, "grid16": 1024, "count": 64, "rel_depth_nyu": 64,
            "rel_depth_blink": 64, "spatial_cvbench": 64, "spatial_embspatial": 64}
    for t, n in want.items():
        for fam in FAM:
            assert cfg.max_new_tokens(t, fam, "eval") == n
            assert cfg.max_new_tokens(t, fam, "dev") == n and cfg.max_new_tokens(t, fam, "extra") == n
            assert cfg.max_new_tokens(t, fam, "think") == 4096 and cfg.max_new_tokens(t, fam, "think8192") == 8192
    assert cfg.data["max_new_tokens_adjustments"] == []


def test_thinking_only_on_think_subsets_and_greedy(cfg):
    assert {s: cfg.enable_thinking(s) for s in P.SUBSET_KINDS} == {"eval": False, "dev": False, "think": True,
                                                                  "think8192": True, "extra": False}
    dec = cfg.decoding()
    assert dec["do_sample"] is False and dec["num_beams"] == 1 and dec["repetition_penalty"] == 1.0
    assert cfg.order == "image_first"


def test_native_grounding_prompts_equal_model_notes(cfg):
    """定位用 model_notes 已记的三条原文（§5.5、§6.5、§7.5）：模板里的 {expression} 换成 {obj} 后逐字出现在 model_notes 里。"""
    notes = (REPO / "docs" / "model_notes.md").read_text()
    for fam in FAM:
        tpl = cfg.entry("refexp_box", fam)["template"].replace("{expression}", "{obj}")
        assert f"`{tpl}`" in notes, fam


def test_detection_prompts(cfg):
    q = cfg.entry("ovd", registry.FAMILY_QWEN35)
    assert q["template"].startswith("Locate every instance that belongs to the following categories: \"{categories}\".")
    assert "Report bbox coordinates in JSON format." in q["source_quote"]
    g = cfg.entry("ovd", registry.FAMILY_GEMMA4)
    assert g["template"] == "detect {categories}" and g["category_joiner"] == ", "
    i = cfg.entry("ovd", registry.FAMILY_INTERNVL35)
    assert i["native"] is False and "detect and label all objects" in i["source_quote"]


def test_unified_prompts_identical_across_families(cfg):
    for t in ("refexp_poly", "grid16", "count", "rel_depth_nyu", "spatial_embspatial"):
        tpls = {cfg.entry(t, f)["template"] for f in FAM}
        assert len(tpls) == 1, t
    poly = cfg.entry("refexp_poly", FAM[0])["template"]
    assert "[[x1, y1], [x2, y2], ...]" in poly and "0 to 1000" in poly and "x first" in poly and "only one JSON block" in poly
    grid = cfg.entry("grid16", FAM[0])["template"]
    n = GRID_SIZE_E1
    assert f"{n}x{n}" in grid and f"exactly {n} lines, each with {n} digits" in grid and f"{n} rows" in grid
    assert GRID_FG_AREA_THRESHOLD == 0.5 and "mostly covered" in grid  # "mostly" = 前景占比 ≥ 0.5


def test_benchmark_prompts_are_verbatim(cfg):
    for t in ("rel_depth_blink", "spatial_cvbench"):
        for f in FAM:
            assert cfg.entry(t, f)["template"] == "{prompt}"
    s = Sample(id="cvbench/test/7", task="spatial_qa", images=("x",), image_keys=("k",),
               fields={"prompt": "How many cats?\nSelect from the following choices.\n(A) 1\n(B) 2", "choices": ["1", "2"]},
               gt={"answer": "B"}, source="cvbench", split="test")
    r = cfg.render("spatial_cvbench", registry.FAMILY_GEMMA4, s)
    assert r.text == s.fields["prompt"] and r.extras == {"letters": ["A", "B"], "options": ["1", "2"]}


def test_embspatial_render_follows_vlmevalkit_layout(cfg):
    s = Sample(id="embspatial/test/q", task="spatial_qa", images=("x",), image_keys=("k",),
               fields={"question": "Which object is closest?", "choices": ["chair", "table", "lamp", "bed"]},
               gt={"answer": "C"}, source="embspatial", split="test")
    r = cfg.render("spatial_embspatial", registry.FAMILY_INTERNVL35, s)
    assert r.text == ("Question: Which object is closest?\nOptions:\nA. chair\nB. table\nC. lamp\nD. bed\n"
                      "Please select the correct answer from the options above.")
    assert r.extras["letters"] == ["A", "B", "C", "D"]


def test_render_on_fixtures_strips_and_fills_placeholders(cfg):
    paths = DataPaths.fixtures()
    for t in ("refexp_box", "ovd", "count", "spatial_embspatial", "rel_depth_blink"):
        for s in select(t, "eval", paths).samples[:2]:
            for f in FAM:
                r = cfg.render(t, f, s)
                assert r.text == r.text.strip() and "{" not in r.text.replace("{categories}", "") or t == "ovd"
    s = select("ovd", "eval", paths).samples[0]
    assert cfg.render("ovd", registry.FAMILY_GEMMA4, s).text == "detect " + ", ".join(s.fields["categories"])
    assert "\"" + ", ".join(s.fields["categories"]) + "\"" in cfg.render("ovd", registry.FAMILY_QWEN35, s).text
    ref = select("refexp_box", "eval", paths).samples[0]
    assert cfg.render("refexp_box", registry.FAMILY_INTERNVL35, ref).text.endswith(f"<ref>{ref.fields['expression']}</ref>")


def test_validate_reports_problems(cfg, tmp_path):
    bad = copy.deepcopy(dict(cfg.data))
    del bad["tasks"]["count"]["families"]["gemma4"]["source_quote"]
    bad["tasks"]["refexp_box"]["families"]["qwen3.5"]["template"] = "Locate {thing}."
    bad["tasks"]["grid16"]["families"]["gemma4"]["parse"] = "nope"
    bad["tasks"]["ovd"]["families"]["internvl3.5"]["native"] = True
    bad["generation"]["enable_thinking"]["eval"] = True
    with pytest.raises(P.PromptConfigError) as ei:
        P.validate(bad)
    msg = str(ei.value)
    for frag in ("count/gemma4: 缺字段", "未声明的占位符", "parse 'nope'", "native=True 与 provenance", "只有思考子集开思考"):
        assert frag in msg, frag
    p = tmp_path / "x.json"
    p.write_text(json.dumps(bad))
    with pytest.raises(P.PromptConfigError):
        P.load_prompt_config(p)


# ---------------------------------------------------------------------------
# CountBench 问句
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("caption,count,obj,rule", [
    ("background photo of three light bulbs", 3, "light bulbs", "after_number"),
    ("City prints: Set of three big prints - $150.00 USD", 3, "big prints", "after_number"),
    ("five 5 holiday nail polishes", 5, "holiday nail polishes", "after_number"),  # 跳过重复的数字
    ("Multiple lot of ten (10) coins: three (3) AR", 10, "coins", "after_number"),  # 跳过括号里的重复
    ("Seven of the American League All-Star players", 7, "American League All-Star players", "after_number"),
    ("two kittens sitting", 2, "kittens", "after_number"),  # 停用词
    ("2 apples", 2, "apples", "after_number"),  # 阿拉伯数字
    ("ten pigeon silhouette collection on white", 10, "pigeon silhouette collection", "after_number"),
    ("Set of nine abstract bicycles Ilustrace wall", 9, "abstract bicycles Ilustrace wall", "after_number"),  # 最多 4 个词
    ("Vintage Crocheted Doilies Pot Holders set of three", 3, "Crocheted Doilies Pot Holders", "before_number"),
    ("RIEDEL Vinum Chablis/Chardonnay glasses set of eight", 8, "Chardonnay glasses", "before_number"),
    ("d10 set of ten - Black", 10, "objects", "fallback_no_phrase"),
    ("a photo of some dogs", 4, "objects", "fallback_no_mention"),
    ("Four business people, one young woman and three older men", 3, "older men", "after_number"),  # 与 GT 相等的那个数
])
def test_count_object_rule(caption, count, obj, rule):
    assert P.count_object(caption, count) == (obj, rule)


def test_count_rule_text_matches_code(cfg):
    r = cfg.data["count_question_rule"]
    assert r["version"] == P.COUNT_RULE_VERSION
    text = r["rule"]
    assert f"{P.COUNT_MAX_OBJECT_WORDS} 个词" in text and f"取 {P.COUNT_FALLBACK_OBJECT}" in text
    for w in sorted(P._BACK_SKIP | P._DETERMINERS):
        assert re.search(rf"(?<![a-z]){w}(?![a-z])", text), w
    assert set(P.COUNT_RULE_BRANCHES) == {"after_number", "before_number", "fallback_no_phrase", "fallback_no_mention"}
    assert "How many {object class} are there in the image?" in cfg.entry("count", FAM[0])["source_quote"]


def test_count_render_records_object_and_branch(cfg):
    s = Sample(id="countbench/all/9", task="count", images=("x",), image_keys=("k",),
               fields={"text": "Set of six gift boxes. Vector."}, gt={"count": 6}, source="countbench", split="all")
    r = cfg.render("count", registry.FAMILY_QWEN35, s)
    assert r.text == "How many gift boxes are there in the image? Answer with a single number."
    assert r.extras == {"object": "gift boxes", "object_rule": "after_number"}
    with pytest.raises(ValueError):
        P.count_object("x", -1)
