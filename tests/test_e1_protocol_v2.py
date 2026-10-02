"""DEV-0017：E1 协议 v2（D36-1、D36-2、D36-6）。

- 配置与代码的钉子：letter_readout、report_columns、refusal_by_person、batch_protocol、acceptance 的结构与数字；
- 受限字母读数：字母表（假分词器与三个家族的真分词器）、按字母集合分批、allowed_token_ids、映射回字母、不算截断；
- free 子集（think:<任务> 的 id，关思考，自由作答）；
- 三列（严格读数、有效率、条件读数）与拒答按人物分列，在手算的小例子上核对；
- 检测的开发核对：coco_train2014_e1dev 的 GT 进样本，按图算 TP/FP；
- 验收 v3 的部件：take、逐位重跑、按组的数值噪声底（±δ）、总判定。
"""

from __future__ import annotations

import copy
import dataclasses
import json
import math
import shutil
import types
from pathlib import Path

import numpy as np
import pytest

from src.adapters import qwen35
from src.adapters.base import GenerationResult
from src.data.records import Sample
from src.e1 import acceptance as A
from src.e1 import prompts as P
from src.e1 import runner as R
from src.e1 import scoring as S
from src.e1.letters import LetterTableError, build_letter_table
from src.e1.tasks import CHOICE_TASKS, FIXTURE_DIR, OVD_DEV_SOURCE, TASK_SUBSETS, DataPaths, select
from tests.test_e1_runner import FakeAdapter, FakeTokenizer, leaf, spec

REPO = Path(__file__).resolve().parents[1]
CFG = P.load_prompt_config()
QWEN = types.SimpleNamespace(parse_one_box=qwen35.parse_one_box, parse_boxes=qwen35.parse_boxes)


@pytest.fixture(scope="module")
def paths(tmp_path_factory):
    return DataPaths.fixtures(nyu_cache=tmp_path_factory.mktemp("nyu"))


# ---------------------------------------------------------------------------
# 配置与代码的钉子
# ---------------------------------------------------------------------------


def test_letter_readout_config_pins():
    lr = CFG.data["letter_readout"]
    assert tuple(lr["tasks"]) == CHOICE_TASKS and S.MCQ_TASKS == frozenset(CHOICE_TASKS)
    assert lr["subsets"] == ["eval", "dev"] and lr["max_new_tokens"] == 1 and lr["parse"] == "letter_token" == S.LETTER_ROUTE
    assert CFG.letter_spellings() == ("{letter}", " {letter}")  # D36-4：同一字母带空格与不带空格两种写法
    for t in P.TASKS:
        for sub in P.SUBSET_KINDS:
            want = "letter" if t in CHOICE_TASKS and sub in ("eval", "dev") else "text"
            assert CFG.readout(t, sub) == want, (t, sub)
            assert CFG.parser(t, "qwen3.5", sub) == ("letter_token" if want == "letter" else CFG.entry(t, "qwen3.5")["parse"])
    assert {t for t, subs in TASK_SUBSETS.items() if "free" in subs} == set(CHOICE_TASKS)


def test_report_and_refusal_config_pins():
    rp = CFG.data["refusal_by_person"]
    assert tuple(rp["tasks"]) == S.REFUSAL_BY_PERSON_TASKS and rp["person_category"] == S.PERSON_CATEGORY == "person"
    rc = CFG.data["report_columns"]
    assert "failure_detail" in rc["valid_rate"] and "有效 = 解析成功" in rc["valid_rate"]


def test_batch_protocol_and_acceptance_config_pins():
    """D36-1 的数字：批大小 16（RUN-0012/0013 用的就是它）、左填充、按 select 的顺序；B 12；δ 准确率 2 个点、IoU 0.02；
    短输出 n ≥ 300（按输出类型合并），多边形与网格 500。"""
    bp, ac = CFG.batch_protocol(), CFG.acceptance()
    assert (bp["batch_size"], bp["padding_side"], bp["order"], bp["letter_batches"]) == (64, "left", "selection", "same_letters")
    assert ac["version"] == P.ACCEPTANCE_VERSION == 3 and ac["composition_b"]["batch_size"] == 48
    assert ac["delta"] == {"accuracy": 0.02, "iou": 0.02}
    g = {x["name"]: x for x in ac["groups"]}
    assert list(g) == ["short_text", "letter", "polygon", "grid", "detection"]
    assert g["short_text"]["tasks"] == ["refexp_box", "count"] and g["letter"]["tasks"] == list(CHOICE_TASKS)
    assert [(g[k]["n_min"], g[k]["take"]) for k in g] == [(300, "all"), (300, "all"), (500, 500), (500, 500), (None, 500)]
    assert [g[k]["metric"] for k in g] == ["accuracy_pooled", "accuracy_pooled", "cIoU", "gIoU", "F1@0.5"]
    assert [g[k]["metric_class"] for k in g] == ["accuracy", "accuracy", "iou", "iou", "accuracy"]


def test_acceptance_group_sizes_in_rules_match_dev_configs():
    """组规则里写的开发样本数（600 句、49 条、250 + 12 + 50 + 50 = 362）与入库的开发子集清单、NYU 点对一致，且够 n_min。"""
    dev = json.loads((REPO / "configs" / "e1_dev_subsets.json").read_text())["subsets"]
    nyu = json.loads((REPO / "configs" / "e1_nyu_pairs.json").read_text())["stats"]["train"]["n_pairs"]
    ref = sum(dev[k]["k"] for k in ("refcoco_unc_val", "refcoco+_unc_val", "refcocog_umd_val"))
    letter = nyu + dev["blink_reldepth"]["k"] + dev["cvbench"]["k"] + dev["embspatial"]["k"]
    g = {x["name"]: x for x in CFG.acceptance()["groups"]}
    assert ref == 600 and f"{ref} 句" in g["short_text"]["rule"] and f"{dev['countbench']['k']} 条" in g["short_text"]["rule"]
    assert f"{nyu} + {dev['blink_reldepth']['k']} + {dev['cvbench']['k']} + {dev['embspatial']['k']} = {letter}" in g["letter"]["rule"]
    assert ref + dev["countbench"]["k"] >= g["short_text"]["n_min"] and letter >= g["letter"]["n_min"]
    assert ref >= g["polygon"]["take"] == g["polygon"]["n_min"] and f"共 {ref} 句" in g["polygon"]["rule"]


@pytest.mark.parametrize("path,value,frag", [
    (("batch_protocol", "padding_side"), "right", "padding_side"),
    (("acceptance", "composition_b", "batch_size"), 64, "composition_b"),
    (("letter_readout", "max_new_tokens"), 2, "max_new_tokens 必须是 1"),
    (("letter_readout", "spellings"), ["{letter}", "{letter}"], "spellings"),
    (("letter_readout", "subsets"), ["eval", "think"], "letter_readout.subsets"),
    (("acceptance", "delta"), {"accuracy": 0.02}, "acceptance.delta"),
    (("countbenchqa_questions", "provenance_note"), " ", "provenance_note"),
])
def test_validate_rejects_bad_protocol_sections(path, value, frag):
    bad = copy.deepcopy(dict(CFG.data))
    d = bad
    for k in path[:-1]:
        d = d[k]
    d[path[-1]] = value
    with pytest.raises(P.PromptConfigError, match=frag):
        P.validate(bad)


def test_validate_rejects_groups_not_covering_tasks_once():
    bad = copy.deepcopy(dict(CFG.data))
    bad["acceptance"]["groups"][0]["tasks"] = ["refexp_box"]  # 计数没有组了
    with pytest.raises(P.PromptConfigError, match="恰好覆盖每个任务一次"):
        P.validate(bad)
    bad = copy.deepcopy(dict(CFG.data))
    bad["acceptance"]["groups"][0]["tasks"] = ["refexp_box", "count", "refexp_poly"]
    bad["acceptance"]["groups"][2]["tasks"] = []
    with pytest.raises(P.PromptConfigError, match="不进 0/1 的池"):
        P.validate(bad)


# ---------------------------------------------------------------------------
# 字母表
# ---------------------------------------------------------------------------


def test_letter_table_with_fake_tokenizer():
    t = build_letter_table(FakeTokenizer(), ["B", "A", "A"], CFG.letter_spellings())
    assert t.letters == ("A", "B") and t.allowed_ids(["A"]) == [1065, 2065] and t.allowed_ids("AB") == [1065, 1066, 2065, 2066]
    assert (t.letter_of(2066), t.spelling_of(2066), t.letter_of(1067)) == ("B", " B", None)
    d = t.as_dict()
    assert d["entries"]["A"] == [{"spelling": "A", "token_id": 1065, "decoded": "A"},
                                 {"spelling": " A", "token_id": 2065, "decoded": " A"}] and d["skipped"] == []
    t2 = build_letter_table(FakeTokenizer(), ["A"], ["{letter}", "({letter}"])  # "(A" 不是单 token：跳过并记下
    assert t2.allowed_ids(["A"]) == [1065] and t2.as_dict()["skipped"] == [{"letter": "A", "spelling": "(A", "token_ids": [40, 65]}]
    with pytest.raises(LetterTableError, match="没有单 token"):
        build_letter_table(FakeTokenizer(), ["A"], ["({letter}"])
    with pytest.raises(KeyError):
        t.allowed_ids(["C"])

    class Collide(FakeTokenizer):
        def encode(self, text, add_special_tokens=False):
            return [7]

        def decode(self, ids):
            return "?"

    with pytest.raises(LetterTableError, match="同时是字母"):
        build_letter_table(Collide(), ["A", "B"], ["{letter}"])
    assert build_letter_table(Collide(), ["A"], CFG.letter_spellings()).allowed_ids("A") == [7]  # 同一字母两写法同 id：记一次


@pytest.fixture(scope="module", params=["qwen3.5-0.8b", "gemma-4-e2b-it", "internvl3.5-2b-hf"])
def real_tokenizer(request):
    import torch

    from src.adapters.factory import make_adapter

    return make_adapter(request.param, device="cpu", dtype=torch.float32, tiny=True).tokenizer


def test_letter_table_real_tokenizers_both_spellings_single_token(real_tokenizer):
    """三个家族的分词器上，A–F 的两种写法都是单个 token，彼此不同（所以每个字母恰好两个允许的 id）。"""
    t = build_letter_table(real_tokenizer, "ABCDEF", CFG.letter_spellings())
    assert not t.skipped and all(len(t.entries[x]) == 2 for x in "ABCDEF") and len(t.allowed_ids("ABCDEF")) == 12
    assert [[e["decoded"] for e in t.as_dict()["entries"][x]] for x in "AB"] == [["A", " A"], ["B", " B"]]


# ---------------------------------------------------------------------------
# 分批与受限字母读数
# ---------------------------------------------------------------------------


def test_batch_plan_cuts_by_size_and_letter_set():
    assert R.batch_plan([None] * 5, 2) == [(0, 2), (2, 4), (4, 5)]
    assert R.batch_plan([None] * 4, 16) == [(0, 4)] and R.batch_plan([], 3) == []
    keys = ["AB", "AB", "ABCD", "ABCD", "ABCD", "AB", "AB"]
    assert R.batch_plan(keys, 2) == [(0, 2), (2, 4), (4, 5), (5, 7)]
    assert R.batch_plan(keys, 16) == [(0, 2), (2, 5), (5, 7)]
    with pytest.raises(ValueError):
        R.batch_plan(keys, 0)


def _mixed_letter_selection(paths):
    """EmbSpatial 的 fixture 样本改出 2、4、4、2、2 个选项：字母集合变化处必须另起一批。"""
    base = select("spatial_embspatial", "eval", paths).samples[0]
    out = []
    for k, n in enumerate((2, 4, 4, 2, 2)):
        opts = [f"opt{j}" for j in range(n)]
        out.append(dataclasses.replace(base, id=f"embspatial/test/mix{k}", fields={**base.fields, "choices": opts},
                                       gt={"answer": "B"}))
    sel = select("spatial_embspatial", "eval", paths)
    sel.samples = out
    return sel


def test_letter_readout_end_to_end_with_fake_adapter(paths, tmp_path):
    sel = _mixed_letter_selection(paths)
    fake = FakeAdapter()
    m = R.run_task(fake, spec("spatial_embspatial", bs=4), sel, CFG, paths, tmp_path, log=lambda _: None)
    lf = leaf(tmp_path, "spatial_embspatial")
    recs = R.read_jsonl_tolerant(lf / "predictions.jsonl", repair=False)
    assert [n for _, n, _ in fake.calls] == [1, 2, 2] and set(fake.max_new) == {1}  # (0,1) (1,3) (3,5)
    ab, abcd = [1065, 1066, 2065, 2066], [1065, 1066, 1067, 1068, 2065, 2066, 2067, 2068]
    assert fake.allowed == [ab, abcd, ab]
    for r in recs:
        assert r["readout"] == "letter" and r["truncated"] is False and r["stopped"] is False and r["failure"] is None
        assert r["n_generated_tokens"] == 1 and r["parsed"]["route"] == "letter_token"
        assert r["parsed"]["letter"] in r["extras"]["letters"] and r["score"]["correct"] == (r["parsed"]["letter"] == "B")
        assert FakeTokenizer().encode(r["parsed"]["spelling"]) == [r["parsed"]["token_id"]]
    assert m["truncation"]["rate"] == 0.0 and m["valid"]["rate"] == 1.0 and m["parse_routes"] == {"letter_token": 5}
    assert m["readout"] == {"letter": 5} and m["generation"]["max_new_tokens"] == 1
    cfg = (lf / "config.yaml").read_text()
    assert "kind: letter" in cfg and "token_id: 2068" in cfg and "frozen: false" in cfg
    assert "读法 letter（解析 letter_token）" in (lf / "README.md").read_text()


def test_free_subset_is_free_form_on_think_ids(paths, tmp_path):
    from src.data.eval_subsets import subset_ids

    sel = select("rel_depth_nyu", "free", paths)
    assert [s.id.rsplit("/", 1)[1] for s in sel.samples] == subset_ids("think:rel_depth_nyu", paths.eval_subsets)
    fake = FakeAdapter()
    m = R.run_task(fake, spec("rel_depth_nyu", "free"), sel, CFG, paths, tmp_path, log=lambda _: None)
    assert set(fake.allowed) == {None} and set(fake.max_new) == {1024} and {e for _, _, e in fake.calls} == {False}
    assert m["readout"] == {"text": len(sel.samples)} and set(m["parse_routes"]) <= {"letter", "option_text"}
    with pytest.raises(ValueError, match="不支持子集"):
        select("count", "free", paths)


def test_score_letter_entry_assertions():
    s = Sample(id="nyuv2_pairs/test/p0", task="rel_depth", images=("x",), image_keys=("k",), fields={}, gt={"closer": "B"},
               source="nyuv2_pairs", split="test")
    t = build_letter_table(FakeTokenizer(), "AB", CFG.letter_spellings())

    def res(ids):
        return GenerationResult(text="x", token_ids=tuple(ids), n_generated_tokens=len(ids), n_prompt_tokens=3,
                                n_visual_tokens=1, wall_time_s=0.0, raw_text="x", stopped=False, n_image_pixels=4)

    assert S.score_letter("rel_depth_nyu", s, res([2066]), table=t, letters=["A", "B"])["score"] == {"gt": "B", "correct": True}
    with pytest.raises(RuntimeError, match="恰好生成 1 个"):
        S.score_letter("rel_depth_nyu", s, res([2066, 1]), table=t, letters=["A", "B"])
    with pytest.raises(RuntimeError, match="不是本题字母"):
        S.score_letter("rel_depth_nyu", s, res([1066]), table=t, letters=["A"])
    with pytest.raises(ValueError):
        S.score_letter("count", s, res([2066]), table=t, letters=["A", "B"])


# ---------------------------------------------------------------------------
# 三列与拒答按人物分列
# ---------------------------------------------------------------------------


def _gen(text, stopped=True):
    return GenerationResult(text=text, token_ids=(1, 2), n_generated_tokens=2, n_prompt_tokens=10, n_visual_tokens=4,
                            wall_time_s=0.1, raw_text=text, stopped=stopped, n_image_pixels=16)


def _ref(k, cat):
    return Sample(id=f"refcoco_unc/val/{k}", task="refexp", images=("x",), image_keys=("k",),
                  fields={"expression": "x", "sent_id": k, "image_size": [12, 9], "category": cat},
                  gt={"box": [1.0, 1.0, 9.0, 7.0], "mask": {"polygons": [[1, 1, 9, 1, 9, 7, 1, 7]], "height": 9, "width": 12}},
                  source="refcoco", split="val")


def _rec(task, parser, s, res):
    rec = {"id": s.id, "raw_text": res.raw_text, "text": res.text, "truncated": res.stopped is False,
           "thinking_unclosed": False, "n_generated_tokens": 2, "n_visual_tokens": 4, "n_image_pixels": 16,
           "image_size": [12, 9], "extras": {}}
    rec.update(S.score_generation(task, parser, s, res, adapter=QWEN, image_size=(12, 9), extras={}))
    return rec


def test_three_columns_and_refusal_by_person_by_hand():
    good, wrong = '[{"bbox_2d": [100, 125, 750, 750]}]', '[{"bbox_2d": [0, 0, 100, 100]}]'
    refuse = "I'm sorry, but I cannot help with locating people."
    rows = [("person", refuse), ("person", refuse), ("person", good), ("dog", good), ("dog", wrong), ("dog", "garbage")]
    ss = [_ref(k, c) for k, (c, _) in enumerate(rows)]
    recs = [_rec("refexp_box", "one_box", s, _gen(t)) for s, (_, t) in zip(ss, rows)]
    m = S.aggregate("refexp_box", ss, recs, has_gt=True)
    assert m["valid"] == {"n": 6, "n_valid": 3, "rate": 0.5, "n_invalid": 3, "n_refusal": 2, "n_other_failure": 1}
    (c,) = m["three_columns"]
    assert c["strict"]["value"] == m["headline"][0]["value"] == 2 / 6 and (c["n_valid"], c["n"], c["valid_rate"]) == (3, 6, 0.5)
    assert math.isclose(c["conditional"]["value"], 2 / 3) and c["conditional"]["n"] == 3 and c["conditional"]["ci"] is not None
    rbp = m["refusal_by_person"]
    assert rbp["metric"] == "acc@0.5" and "unknown" not in rbp
    assert (rbp["person"]["n"], rbp["person"]["n_refusal"], rbp["person"]["refusal_rate"]) == (3, 2, 2 / 3)
    assert rbp["person"]["conditional"]["value"] == 1.0 and math.isclose(rbp["person"]["strict"]["value"], 1 / 3)
    assert (rbp["non_person"]["n_refusal"], rbp["non_person"]["valid_rate"]) == (0, 2 / 3)
    assert rbp["non_person"]["conditional"]["value"] == 0.5 and math.isclose(rbp["non_person"]["strict"]["value"], 1 / 3)


def test_three_columns_polygon_ciou_and_unknown_category():
    ss = [_ref(0, None), _ref(1, "dog"), _ref(2, "dog")]
    texts = ["[[0, 0], [500, 0], [500, 1000], [0, 1000]]", "x", "[[0, 0], [1000, 0], [1000, 1000], [0, 1000]]"]
    recs = [_rec("refexp_poly", "polygon", s, _gen(t)) for s, t in zip(ss, texts)]
    m = S.aggregate("refexp_poly", ss, recs, has_gt=True)
    ci, gi = m["three_columns"]
    valid = [recs[0], recs[2]]
    want_c = sum(r["score"]["inter"] for r in valid) / sum(r["score"]["union"] for r in valid)
    assert ci["name"] == "cIoU" and math.isclose(ci["conditional"]["value"], want_c) and ci["n_valid"] == 2
    assert gi["name"] == "gIoU" and math.isclose(gi["conditional"]["value"], np.mean([r["score"]["iou"] for r in valid]))
    assert m["refusal_by_person"]["unknown"]["n"] == 1 and m["refusal_by_person"]["metric"] == "cIoU"
    count = Sample(id="countbench/all/0", task="count", images=("x",), image_keys=("k",), fields={}, gt={"count": 2},
                   source="countbench", split="all")
    rc = {**_rec("count", "count", count, _gen("2")), "extras": {"object": "x", "object_rule": "countbenchqa"}}
    assert "refusal_by_person" not in S.aggregate("count", [count], [rc], has_gt=True)


def _ovd_sample(k, boxes, source="coco_val2017"):
    return Sample(id=f"{source}/x/{k}", task="ovd", images=("x",), image_keys=("k",),
                  fields={"categories": ["person", "car"], "category_ids": [1, 3], "image_id": k, "image_size": [100, 100]},
                  gt={"boxes": [{"box": b, "category_id": c, "category": "x", "iscrowd": 0, "ann_id": i}
                                for i, (b, c) in enumerate(boxes)]}, source=source, split="x")


def test_detection_conditional_f1_uses_valid_images_only():
    """图 1：TP 1；图 2：解析失败（有 1 个 GT）；图 3：FP 1、GT 1 没检到。严格 F1 = 2·1/(2 + 3) = 0.4；
    有效的只有图 1、图 3：条件 F1 = 2·1/(2 + 2) = 0.5。"""
    ss = [_ovd_sample(1, [([10, 10, 50, 50], 1)]), _ovd_sample(2, [([0, 0, 30, 30], 3)]), _ovd_sample(3, [([0, 0, 20, 20], 1)])]
    texts = ['[{"bbox_2d": [100, 100, 500, 500], "label": "person"}]', "nope",
             '[{"bbox_2d": [600, 600, 900, 900], "label": "car"}]']
    recs = []
    for s, t in zip(ss, texts):
        r = {"id": s.id, "raw_text": t, "text": t, "truncated": False, "thinking_unclosed": False, "n_generated_tokens": 2,
             "n_visual_tokens": 4, "n_image_pixels": 16, "image_size": [100, 100], "extras": {}}
        r.update(S.score_generation("ovd", "boxes", s, _gen(t), adapter=QWEN, image_size=(100, 100), extras={}))
        recs.append(r)
    m = S.aggregate("ovd", ss, recs, has_gt=True)
    (c,) = m["three_columns"]
    assert c["scope"] == "coco_val2017" and math.isclose(c["strict"]["value"], 0.4) and (c["n_valid"], c["n"]) == (2, 3)
    assert math.isclose(c["conditional"]["value"], 0.5) and c["conditional"]["n"] == 2
    assert S.ovd_per_image_counts(ss, recs) == [(1, 0, 1), (0, 0, 1), (0, 1, 1)]
    assert S.f1_from_counts([]) == {"value": None, "ci": None, "n": 0}
    assert S.f1_from_counts([(0, 2, 0)])["value"] is None


# ---------------------------------------------------------------------------
# 检测的开发核对：coco_train2014_e1dev 的 GT
# ---------------------------------------------------------------------------


def test_ovd_dev_selection_has_full_gt_from_fixture(paths):
    sel = select("ovd", "dev", paths)
    assert sel.has_gt and [s.id for s in sel.samples] == [f"{OVD_DEV_SOURCE}/901", f"{OVD_DEV_SOURCE}/902"]
    s901, s902 = sel.samples
    assert [b["ann_id"] for b in s901.gt["boxes"]] == [7000, 7001] and [b["ann_id"] for b in s902.gt["boxes"]] == [7002, 9002, 9003]
    assert s902.gt["boxes"][1]["box"] == [0.0, 0.0, 4.0, 3.0] and s902.gt["boxes"][2]["iscrowd"] == 1
    assert s901.fields["categories"] == ["person", "dog", "car"] and s901.fields["category_ids"] == [1, 2, 3]
    assert sel.stats["annotations"]["source"] == "coco_train2014_e1dev" and len(sel.stats["annotations"]["sha256"]) == 64
    # 三个 RefCOCO 开发子集的图都在（refcoco、refcoco+ 的开发句在 901，refcocog 的在 902）
    assert {int(s.fields["image_id"]) for s in sel.samples} == {901, 902}


def test_ovd_dev_without_annotations_or_with_bad_hash(paths, tmp_path):
    root = tmp_path / "root"
    shutil.copytree(FIXTURE_DIR / "root", root, ignore=shutil.ignore_patterns("coco_train2014_e1dev"))
    p = DataPaths(root=root, dev_subsets=paths.dev_subsets, eval_subsets=paths.eval_subsets, nyu_pairs=paths.nyu_pairs)
    sel = select("ovd", "dev", p)
    assert not sel.has_gt and all(s.gt["boxes"] is None for s in sel.samples) and "没有落地" in sel.stats["note"]
    shutil.copytree(FIXTURE_DIR / "root" / "coco_train2014_e1dev", root / "coco_train2014_e1dev")
    f = root / "coco_train2014_e1dev" / "instances_train2014_e1dev.json"
    f.write_bytes(f.read_bytes().replace(b'"dog"', b'"cat"'))
    with pytest.raises(ValueError, match="sha256"):
        select("ovd", "dev", p)


# ---------------------------------------------------------------------------
# 验收 v3 的部件
# ---------------------------------------------------------------------------


def _acc_recs(task, ss, correct):
    return [{"id": s.id, "raw_text": str(c), "failure": None, "parsed": {"letter": "A"}, "score": {"correct": bool(c)}}
            for s, c in zip(ss, correct)]


def _plain(k):
    return Sample(id=f"cvbench/test/{k}", task="spatial_qa", images=("x",), image_keys=("k",), fields={}, gt={"answer": "A"},
                  source="cvbench", split="test")


def test_take_samples_rule():
    proto = CFG.acceptance()
    ss = [_ref(k, "dog") for k in range(600)]
    got, st = A.take_samples("refexp_poly", ss, proto)
    assert len(got) == 500 and st == {"group": "polygon", "take": 500, "n_before": 600, "n": 500,
                                      "purpose": "e1_acceptance_v3:polygon"}
    assert [s.id for s in got] == [s.id for s in ss if s in got]  # 保持原顺序
    again, _ = A.take_samples("refexp_poly", list(reversed(ss)), proto)
    assert {s.id for s in again} == {s.id for s in got}  # 由 id 的哈希决定，与输入顺序无关
    assert A.take_samples("refexp_box", ss, proto)[0] == ss  # take = all
    assert A.take_samples("grid16", ss[:7], proto)[1]["n"] == 7  # 不够 take：全取


def test_rerun_identical_and_file_hash(tmp_path):
    ss = [_plain(k) for k in range(3)]
    a1 = _acc_recs("spatial_cvbench", ss, [1, 0, 1])
    a2 = copy.deepcopy(a1)
    assert A.rerun_identical("t", a1, a2, sha_a1="x", sha_a2="x")["identical"]
    a2[1]["raw_text"] = "changed"
    r = A.rerun_identical("t", a1, a2)
    assert (r["identical"], r["n_diff"], r["diff_ids"], r["raw_text_diff"]) == (False, 1, [ss[1].id], 1)
    assert not A.rerun_identical("t", a1, a1, sha_a1="x", sha_a2="y")["identical"]  # 记录相同而文件不同也不算
    f = tmp_path / "p.jsonl"
    f.write_bytes(b"abc")
    assert A.file_sha256(f) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad" and A.file_sha256(tmp_path / "no") is None


def test_noise_floor_within_and_outside_delta():
    """300 条的池：A1 与 B 一条不差 → CI = [0, 0]，通过；翻 12 条（全是 A1 对、B 错）→ 差 0.04，CI 在 +δ 之外，不通过。"""
    g = {x["name"]: x for x in CFG.acceptance()["groups"]}["letter"]
    ss = [_plain(k) for k in range(300)]
    base = [k % 2 for k in range(300)]
    a = [("spatial_cvbench", ss, _acc_recs("spatial_cvbench", ss, base))]
    same = A.noise_floor(g, 0.02, a, a)
    assert (same["diff"], same["diff_ci"]["low"], same["diff_ci"]["high"], same["pass"]) == (0.0, 0.0, 0.0, True)
    flipped = list(base)
    for k in [i for i in range(300) if base[i] == 1][:12]:
        flipped[k] = 0
    b = [("spatial_cvbench", ss, _acc_recs("spatial_cvbench", ss, flipped))]
    out = A.noise_floor(g, 0.02, a, b)
    assert math.isclose(out["diff"], 0.04) and out["diff_ci"]["low"] > 0 and out["diff_ci"]["high"] > 0.02
    assert out["within_delta"] is False and out["pass"] is False and out["n"] == 300 and out["n_below_min"] is False
    with pytest.raises(ValueError, match="不同或不同序"):
        A.noise_floor(g, 0.02, a, [("spatial_cvbench", ss[::-1], _acc_recs("spatial_cvbench", ss[::-1], base))])
    empty = A.noise_floor(g, 0.02, [("spatial_cvbench", [], [])], [("spatial_cvbench", [], [])])
    assert empty["pass"] is None and empty["note"] == "空组"


def test_group_values_per_metric():
    ss = [_ref(k, "dog") for k in range(2)]
    poly = [{"id": s.id, "score": {"inter": i, "union": u, "iou": i / u}} for s, (i, u) in zip(ss, [(1, 2), (3, 4)])]
    g = {x["name"]: x for x in CFG.acceptance()["groups"]}
    arr, stat = A.group_values(g["polygon"], [("refexp_poly", ss, poly)])
    assert stat(arr) == 4 / 6
    arr, stat = A.group_values(g["grid"], [("grid16", ss, poly)])
    assert stat is None and float(np.mean(arr)) == (0.5 + 0.75) / 2
    pooled, stat = A.group_values(g["short_text"], [("refexp_box", ss, [{"score": {"correct": True}}] * 2),
                                                    ("count", ss[:1], [{"score": {"correct": False}}])])
    assert stat is None and list(pooled) == [1.0, 1.0, 0.0]
    with pytest.raises(ValueError, match="单个任务"):
        A.group_values(g["polygon"], [("refexp_poly", ss, poly), ("refexp_poly", ss, poly)])


def test_overall_v3():
    tr = [{"task": "a", "rerun": {"identical": True}}, {"task": "b", "rerun": {"identical": True}}]
    gr = [{"group": "x", "pass": True, "n_below_min": False}, {"group": "y", "pass": None, "n_below_min": True}]
    o = A.overall_v3(tr, gr)
    assert o["all_pass"] is True and o["groups_not_evaluated"] == ["y"] and o["groups_below_n_min"] == ["y"]
    assert A.overall_v3([{"task": "a", "rerun": {"identical": False}}], gr)["all_pass"] is False
    assert A.overall_v3(tr, [{"group": "x", "pass": False, "n_below_min": False}])["groups_failed"] == ["x"]
    assert A.overall_v3(tr, [{"group": "y", "pass": None, "n_below_min": None}])["all_pass"] is None
