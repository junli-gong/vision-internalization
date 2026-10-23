"""DEV-0059：src/w2/loop.py（挑选的解析、回退、三种不用模型的挑法、挑选一步的批与预检、每条样本的记录），用假候选与假适配器
（tests/_w2_fakes.py，签名照 docs/w2_plan.md §1、§2）。

DEV-0062：挑选的协议与提示取草稿夹具（tests/_w2_draft：思考开、预算 2048 / 64），不取仓库配置，所以 configs/w2.json 冻结成任何一个臂
（例如关思考）之后照样成立。"""

from __future__ import annotations

import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from src.e1.runner import load_images
from src.e1.scoring import gt_mask
from src.parsing import ParseFailure, parse_generation
from src.w1 import baselines as B
from src.w2 import loop as Lp
from src.w2.data import tiny_paths, tiny_splits, w2_items
from tests import _w2_fakes as F
from tests._w2_draft import W2_DRAFT

CFG = W2_DRAFT  # 草稿夹具（思考开、预算 2048 / 64；DEV-0062：仓库配置冻结之后照样成立）
P = CFG.select_protocol()
GENS = ({"type": "grabcut_default"}, {"type": "box_scale", "percent": 20}, {"type": "box_scale", "percent": -20}, {"type": "box"})


def ps(text, k=4):
    return Lp.parse_selection(text, k=k, none_words=P.none_words, strip_chars=P.strip_chars)


@pytest.mark.parametrize("text, want", [("3", 3), (" 3. ", 3), ("**2**", 2), ("(1)", 1), ("#4", 4), ("\"2\"", 2), ("2\n", 2),
                                        ("None", "none"), ("NONE.", "none"), ("none!", "none"), ("`none`", "none")])
def test_parse_numbers_and_none(text, want):
    r = ps(text)
    assert r.failure is None and r.value == want


@pytest.mark.parametrize("text, why", [("0", ParseFailure.OUT_OF_RANGE), ("5", ParseFailure.OUT_OF_RANGE), ("12", ParseFailure.OUT_OF_RANGE),
                                       ("Candidate 3", ParseFailure.NO_CHOICE), ("3 or 4", ParseFailure.NO_CHOICE),
                                       ("none of them", ParseFailure.NO_CHOICE), ("three", ParseFailure.NO_CHOICE),
                                       ("2.5", ParseFailure.NO_CHOICE), ("-1", ParseFailure.NO_CHOICE), ("３", ParseFailure.NO_CHOICE),
                                       ("", ParseFailure.NO_CHOICE)])
def test_parse_garbage_is_a_failure(text, why):
    r = ps(text)
    assert r.value is None and r.failure == why


def test_parse_generation_order():
    """思考段未闭合 → 截断 → 空回答 → 解析（E1 的 parse_generation）。"""
    kw = dict(k=4, none_words=P.none_words, strip_chars=P.strip_chars)
    view = lambda **a: SimpleNamespace(**{"thinking_unclosed": False, "stopped": True, "text": "2", **a})  # noqa: E731
    assert parse_generation(view(), Lp.parse_selection, **kw).value == 2
    assert parse_generation(view(thinking_unclosed=True, text=""), Lp.parse_selection, **kw).failure == ParseFailure.THINKING_UNCLOSED
    assert parse_generation(view(stopped=False), Lp.parse_selection, **kw).failure == ParseFailure.TRUNCATED
    assert parse_generation(view(text="  "), Lp.parse_selection, **kw).failure == ParseFailure.EMPTY


def test_random_pick_is_reproducible_and_matches_the_rule():
    ids = [f"refcoco_unc/val/{i}" for i in range(400)]
    got = [Lp.random_index("w2_random_pick_v1", i, 6) for i in ids]
    assert got == [Lp.random_index("w2_random_pick_v1", i, 6) for i in ids]
    assert got == [1 + int(hashlib.sha256(f"w2_random_pick_v1|{i}".encode()).hexdigest(), 16) % 6 for i in ids]
    assert set(got) == {1, 2, 3, 4, 5, 6}  # 用到全部编号
    assert got != [Lp.random_index("other_salt", i, 6) for i in ids]


def test_oracle_ties_go_to_the_smaller_index_and_compare_exactly():
    assert Lp.oracle_index([(1, 3), (2, 6), (1, 4)]) == 1  # 1/3 == 2/6：取编号小的
    assert Lp.oracle_index([(1, 4), (2, 6), (1, 3)]) == 2
    assert Lp.oracle_index([(333333333, 1000000000), (1, 3)]) == 2  # 1/3 > 0.333333333（精确比较）
    with pytest.raises(ValueError):
        Lp.oracle_index([])


def test_choose_and_fallbacks():
    stats = [(1, 4), (3, 4), (2, 4)]
    mk = lambda ans: Lp.SelectOutcome("x", ans, None, {})  # noqa: E731
    assert Lp.choose("w2a", sample_id="x", k=3, stats=stats, salt="s", outcome=mk(3)) == (3, 3, None)
    assert Lp.choose("w2a", sample_id="x", k=3, stats=stats, salt="s", outcome=mk("none")) == (1, None, "none")
    assert Lp.choose("w2a", sample_id="x", k=3, stats=stats, salt="s", outcome=mk(None)) == (1, None, "parse_failure")
    assert Lp.choose("oracle_pick", sample_id="x", k=3, stats=stats, salt="s") == (2, 2, None)
    assert Lp.choose("w1f", sample_id="x", k=3, stats=stats, salt="s") == (1, 1, None)
    r = Lp.random_index("s", "x", 3)
    assert Lp.choose("random_pick", sample_id="x", k=3, stats=stats, salt="s") == (r, r, None)
    with pytest.raises(ValueError):
        Lp.choose("ownbox", sample_id="x", k=3, stats=stats, salt="s")


# ---------------------------------------------------------------------------
# fixture 上的样本：候选、挑选、记录
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def samples():
    return w2_items("refseg", "devtune800", tiny_paths(), splits=tiny_splits(CFG))


def box_rec(s, box, failure=None):
    """一条假的 E1 refexp_box 记录（提出）。"""
    return {"id": s.id, "failure": failure, "parsed": None if failure else {"box": list(box)}, "n_prompt_tokens": 100,
            "n_generated_tokens": 20, "n_visual_tokens": 30, "n_image_pixels": 30 * 1024}


def rows_for(samples, *, fail_first=False):
    rows, props = {}, {}
    spec = F.CandidateSpec(generators=GENS)
    for k, s in enumerate(samples):
        im = load_images(s)[0]
        gt_box = s.gt["box"]
        rec = box_rec(s, [gt_box[0] + 1, gt_box[1] + 1, gt_box[2] - 1, gt_box[3]], "no_box" if (fail_first and k == 0) else None)
        prop = Lp.proposal_of("box", rec)
        cands = None if prop["box"] is None else F.generate_candidates(np.asarray(im), prop["box"], spec=spec)
        rows[s.id] = Lp.candidates_row(s.id, im.size, prop, cands, len(GENS))
        props[s.id] = rec
    return rows, props


def test_candidate_one_is_w1f_bit_for_bit(samples):
    """1 号候选（假实现走 W1-F 的同一条调用路径）与 src/w1/baselines.w1f_episode 的 mask 逐位相同；w1f 设置交的就是它。"""
    rows, props = rows_for(samples)
    for s in samples:
        im = load_images(s)[0]
        ref, _ = B.w1f_episode(s, props[s.id], im, B.workbench_grabcut())
        m1 = Lp.decode_candidates(rows[s.id])[0]
        assert ref["answer"]["mask"]["rle"] == rows[s.id]["candidates"][0]["mask"]["rle"]
        rec, _ = Lp.candidate_record("w1f", s, gt_mask(s), rows[s.id], salt="s", proposal_rec=props[s.id])
        assert rec["final_index"] == 1 and rec["answer"]["mask"]["rle"] == ref["answer"]["mask"]["rle"]
        assert rec["score"] == ref["score"] and m1.shape == gt_mask(s).shape


def test_oracle_matches_ground_truth_and_records(samples):
    rows, props = rows_for(samples, fail_first=True)
    s0 = samples[0]
    rec0, calls0 = Lp.candidate_record("oracle_pick", s0, gt_mask(s0), rows[s0.id], salt="s", proposal_rec=props[s0.id])
    assert rec0["failure"] == "no_box" and rec0["n_candidates"] == 0 and rec0["answer"] is None and rec0["n_tool_calls"] == 0
    assert rec0["score"]["inter"] == 0 and rec0["score"]["union"] == int(gt_mask(s0).sum()) and len(calls0) == 1
    for s in samples[1:]:
        g = gt_mask(s)
        rec, calls = Lp.candidate_record("oracle_pick", s, g, rows[s.id], salt="s", proposal_rec=props[s.id])
        ious = [c["iou"] for c in rec["candidates"]]
        assert rec["final_index"] == rec["best_index"] == 1 + int(np.argmax(ious)) and rec["final_is_best"]
        assert rec["best_iou"] == max(ious) == rec["score"]["iou"]
        assert len(rec["candidates"]) == 4 and [c["index"] for c in rec["candidates"]] == [1, 2, 3, 4]
        assert rec["n_model_calls"] == 1 and rec["n_tool_calls"] == 1 and rec["tokens"]["prompt"] == 100
        for c, m in zip(rec["candidates"], Lp.decode_candidates(rows[s.id])):  # 各候选的 IoU 就是与真值的 IoU
            assert c["inter"] == int((m & g).sum()) and c["union"] == int((m | g).sum())
        rr, _ = Lp.candidate_record("random_pick", s, g, rows[s.id], salt="s", proposal_rec=props[s.id])
        assert rr["final_index"] == Lp.random_index("s", s.id, 4) and rr["score"]["iou"] <= rec["score"]["iou"]


def _items(samples, rows):
    out = []
    for s in samples:
        row = rows[s.id]
        cands = [F.Candidate(c["name"], m, c["ok"], c["info"]) for c, m in zip(row["candidates"], Lp.decode_candidates(row))]
        imgs = F.render_selection(load_images(s)[0], cands, layout="grid", style={})
        out.append(Lp.SelectItem(s.id, tuple(imgs), CFG.select_prompt(len(cands), s.fields["expression"]), len(cands)))
    return out


def test_selection_batches_parses_and_falls_back(samples):
    rows, props = rows_for(samples)
    ad = F.ScriptedSelectAdapter({"the thing 100": "2", "the thing 102": "none", "the thing 103": "Candidate 3",
                                  "the thing 104": {"text": "", "thinking_end": "unclosed"}, "the thing 105": "4"})
    items = _items(samples, rows)
    outs = Lp.run_selection(ad, items, P, system=None, batch_size=2, max_batch_tokens=None)
    assert [o.id for o in outs] == [s.id for s in samples]
    assert ad.batch_sizes == [2, 2, 1] and all(t is None for b in ad.tools_seen for t in b)  # 不给工具
    assert ad.budgets == [(2048, 64)] * 3
    got = {o.id.rsplit("/", 1)[1]: (o.answer, o.fallback, o.parse_failure) for o in outs}
    assert got == {"100": (2, None, None), "102": ("none", "none", None), "103": (None, "parse_failure", "no_choice"),
                   "104": (None, "parse_failure", "thinking_unclosed"), "105": (4, None, None)}
    finals = {}
    for s, o in zip(samples, outs):
        rec, calls = Lp.candidate_record("w2a", s, gt_mask(s), rows[s.id], salt="s", proposal_rec=props[s.id], outcome=o)
        finals[s.id.rsplit("/", 1)[1]] = (rec["final_index"], rec["pick"], rec["fallback"])
        assert rec["n_model_calls"] == 2 and len(calls) == 2 and rec["select"]["method"] == "chat_turn_batch"
        assert rec["select"]["images"][0]["size"] == [48 * 4, 36] and rec["select"]["prompt"].count("4") >= 2
        assert rec["tokens"]["visual"] == 30 + F.VISUAL_TOKENS_PER_IMAGE
    assert finals == {"100": (2, 2, None), "102": (1, None, "none"), "103": (1, None, "parse_failure"),
                      "104": (1, None, "parse_failure"), "105": (4, 4, None)}
    # 按提示长度拆子批：max_batch_tokens 小到只能一行一批
    ad2 = F.ScriptedSelectAdapter()
    Lp.run_selection(ad2, items, P, system=None, batch_size=4, max_batch_tokens=1)
    assert ad2.batch_sizes == [1] * 5
    ad3 = F.ScriptedSelectAdapter()
    Lp.run_selection(ad3, items, P, system=None, batch_size=4, max_batch_tokens=10_000)
    assert ad3.batch_sizes == [4, 1]


def test_preflight_rejects_one_row_not_the_batch(samples):
    rows, _ = rows_for(samples)
    items = _items(samples, rows)
    bad = Lp.SelectItem(items[1].id, items[1].images, items[1].prompt + " <|im_end|>", items[1].k)
    items = [items[0], bad] + items[2:]
    ad = F.ScriptedSelectAdapter(default="3")
    outs = Lp.run_selection(ad, items, P, system=None, batch_size=8, max_batch_tokens=None)
    assert outs[1].parse_failure == "input_rejected" and outs[1].fallback == "parse_failure" and outs[1].compute is None
    assert "结构性 token" in outs[1].record["input_error"]
    assert ad.batch_sizes == [4] and all(o.answer == 3 for k, o in enumerate(outs) if k != 1)


def test_thinking_off_uses_e1_single_turn(samples):
    rows, _ = rows_for(samples)
    items = _items(samples, rows)
    off = CFG.select_protocol().__class__(**{**CFG.select_protocol().__dict__, "thinking": False})
    ad = F.ScriptedSelectAdapter(default="2")
    outs = Lp.run_selection(ad, items, off, system=None, batch_size=2, max_batch_tokens=None, decoding={"do_sample": False})
    assert [c["n"] for c in ad.generate_calls] == [2, 2, 1] and not ad.batch_sizes
    assert all(c["enable_thinking"] is False and c["max_new_tokens"] == off.answer_budget for c in ad.generate_calls)
    assert all(o.answer == 2 and o.record["method"] == "generate_batch" and o.record["thinking"] is None for o in outs)
    with pytest.raises(ValueError, match="system"):
        Lp.run_selection(ad, items, off, system="be brief", batch_size=2, max_batch_tokens=None, decoding={})


def test_box_points_parsing_goes_through_the_adapter():
    """box_points 的解析在适配器里（src/adapters/qwen35.parse_box_points；CLAUDE.md：模型相关逻辑放进 src/adapters/）；回路只调用
    adapter.parse_box_points，先经 parse_generation 判截断等。"""
    import inspect

    src = inspect.getsource(Lp)
    assert "point_2d" not in src and "norm1000" not in src and "bbox_2d" not in src  # 回路里没有家族的格式与换算
    seen = []

    class Ad:
        def parse_box_points(self, text, *, width, height, max_points):
            from src.adapters.qwen35 import parse_box_points

            seen.append((text, width, height, max_points))
            return parse_box_points(text, width=width, height=height, max_points=max_points)

    t = '[{"bbox_2d": [100, 100, 500, 900]}, {"point_2d": [300, 500]}, {"point_2d": [900, 900], "label": "background"}]'
    r = Lp.parse_box_points_generation(SimpleNamespace(thinking_unclosed=False, stopped=True, text=t), Ad(), width=200, height=100,
                                       max_points=2)
    assert r.failure is None and r.box == (20.0, 10.0, 100.0, 90.0) and r.fg_points == ((60.0, 50.0),) and r.bg_points == ((180.0, 90.0),)
    assert seen[-1] == (t, 200, 100, 2)
    cut = SimpleNamespace(thinking_unclosed=False, stopped=False, text=t)
    assert Lp.parse_box_points_generation(cut, Ad(), width=200, height=100, max_points=2).failure == ParseFailure.TRUNCATED
    p = Lp.proposal_of("box_points", {"id": "x", "failure": None, "parsed": {"box": list(r.box), "fg_points": [list(v) for v in r.fg_points],
                                                                        "bg_points": [list(v) for v in r.bg_points]}})
    assert p["fg_points"] == [[60.0, 50.0]] and p["bg_points"] == [[180.0, 90.0]]


# ---------------------------------------------------------------------------
# 裁定 A：1 号候选 ok = False 时回路按空 mask 用（与 W1-F 一致），其余编号的失败候选仍是框
# ---------------------------------------------------------------------------


def _first_failed(samples):
    """把每行的 1 号候选改成"失败"：mask 换成 GT 本身（若按原样用，它会是最优），ok = False；3 号也失败（仍按框用）。"""
    from src.executors.raster import mask_to_rle

    rows, props = rows_for(samples)
    for s in samples:
        r = rows[s.id]
        g = gt_mask(s)
        r["candidates"][0].update(ok=False, mask={"rle": mask_to_rle(g), "area": int(g.sum())},
                                  info={**r["candidates"][0]["info"], "error": "ValueError: 假的 GrabCut 失败"})
        r["candidates"][2]["ok"] = False
    return rows, props


def test_first_not_ok_is_empty_for_w1f_and_oracle(samples):
    rows, props = _first_failed(samples)
    for s in samples:
        g = gt_mask(s)
        row = rows[s.id]
        assert Lp.first_as_empty(row) and Lp.decode_candidates(row)[0].sum() == g.sum()  # 缓存里存的是生成器给的原样
        eff = Lp.effective_masks(row)
        assert eff[0].sum() == 0 and (eff[2] == Lp.decode_candidates(row)[2]).all()  # 只有 1 号按空，3 号失败仍按原样（框）
        w, _ = Lp.candidate_record("w1f", s, g, row, salt="s", proposal_rec=props[s.id])
        assert w["final_index"] == 1 and w["answer"]["mask"]["area"] == 0 and w["score"] == {"inter": 0, "union": int(g.sum()), "iou": 0.0}
        assert w["candidates"][0]["as_empty"] is True and w["candidates"][0]["iou"] == 0.0 and "as_empty" not in w["candidates"][2]
        o, _ = Lp.candidate_record("oracle_pick", s, g, row, salt="s", proposal_rec=props[s.id])
        assert o["best_index"] != 1 and o["final_index"] == o["best_index"] and o["score"]["iou"] > 0  # 原样用的话 1 号是最优（IoU 1）


def test_first_not_ok_is_empty_for_w2a_pick_and_fallback(samples):
    rows, props = _first_failed(samples)
    s0, s1 = samples[1], samples[2]
    for s, ans, fb in ((s0, 1, None), (s1, "none", "none")):
        g = gt_mask(s)
        o = Lp.SelectOutcome(s.id, ans, None, {"method": "chat_turn_batch"})
        r, _ = Lp.candidate_record("w2a", s, g, rows[s.id], salt="s", proposal_rec=props[s.id], outcome=o)
        assert r["final_index"] == 1 and r["fallback"] == fb and r["answer"]["mask"]["area"] == 0 and r["score"]["inter"] == 0
    # 渲染给 LLM 的 1 号候选是空叠加，info 记原因；其余照旧
    cands = Lp.selection_candidates(rows[s0.id], F.Candidate)
    assert cands[0].mask.sum() == 0 and cands[0].info["w2_as_empty"] == Lp.FIRST_NOT_OK_EMPTY and cands[0].ok is False
    assert "w2_as_empty" not in cands[2].info and cands[2].mask.sum() > 0 and cands[1].mask.sum() > 0
    sep = F.render_selection(load_images(s0)[0], cands, layout="separate", style={})
    assert np.array_equal(np.asarray(sep[0]), np.asarray(load_images(s0)[0]))  # 空 mask：1 号那张与原图相同


def test_first_not_ok_is_empty_for_random_pick(samples):
    rows, props = _first_failed(samples)
    s = samples[1]
    salt1 = next(f"salt{i}" for i in range(1000) if Lp.random_index(f"salt{i}", s.id, 4) == 1)
    r, _ = Lp.candidate_record("random_pick", s, gt_mask(s), rows[s.id], salt=salt1, proposal_rec=props[s.id])
    assert r["final_index"] == 1 and r["answer"]["mask"]["area"] == 0 and r["score"]["inter"] == 0
    salt3 = next(f"salt{i}" for i in range(1000) if Lp.random_index(f"salt{i}", s.id, 4) == 3)
    r3, _ = Lp.candidate_record("random_pick", s, gt_mask(s), rows[s.id], salt=salt3, proposal_rec=props[s.id])
    assert r3["final_index"] == 3 and r3["answer"]["mask"]["area"] == rows[s.id]["candidates"][2]["mask"]["area"] > 0  # 其余失败仍是框


def test_candidates_row_entry_checks():
    im = Image.new("RGB", (8, 6))
    prop = {"mode": "box", "box": [1.0, 1.0, 5.0, 5.0], "failure": None, "fg_points": [], "bg_points": []}
    good = [F.Candidate("a", np.zeros((6, 8), bool), True, {}), F.Candidate("b", np.ones((6, 8), bool), False, {"t": 1})]
    row = Lp.candidates_row("x", im.size, prop, good, 2)
    assert [c["index"] for c in row["candidates"]] == [1, 2] and row["n_tool_calls"] == 1
    assert [m.sum() for m in Lp.decode_candidates(row)] == [0, 48]
    with pytest.raises(RuntimeError, match="K = 3"):
        Lp.candidates_row("x", im.size, prop, good, 3)
    with pytest.raises(RuntimeError, match="bool"):
        Lp.candidates_row("x", im.size, prop, [F.Candidate("a", np.zeros((6, 8), np.uint8), True, {})] * 2, 2)
    with pytest.raises(RuntimeError, match="(H, W)"):
        Lp.candidates_row("x", im.size, prop, [F.Candidate("a", np.zeros((8, 6), bool), True, {})] * 2, 2)
    assert Lp.candidates_digest_rows([row]) == Lp.candidates_digest_rows([{**row, "candidates": [{**c, "info": {"elapsed_s": 9}}
                                                                                                  for c in row["candidates"]]}])
