"""DEV-0051：W1 回路（src/w1/loop.py）。假适配器按脚本逐轮返回 ChatTurnResult，假工作台按约定实现（tests/_w1_fakes.py）。

- 上限：轮数（第 6 轮之后 no_answer）、计数的调用（满 8 次之后不执行、返回上限说明；提交不计数、总是执行；不豁免时提交也被拒）。
- 按轮批量：样本先后结束，每轮的批按原顺序收缩；批大小上限生效。
- 结束与失败：提交（mask / 框 / 计数）、最终答案文字（框用本家族原生 JSON、计数用单个整数）、解析失败、refseg 没提交、提交了别的种类、
  截断、思考段未闭合；提交之后同一段里的调用不执行。
- 解析失败的调用：不执行、不计数、返回报错说明，这一轮不算最终答案。
- 历史：assistant 的 content 去掉了调用原文、带 tool_calls 与 reasoning_content；tool 消息带图；最后一轮之前追加提示。
- compute 字段齐全且与每轮的 token 数一致；to_json 不含墙钟；tool_workers 不改变输出；入口断言（没有像素、工具清单不符）。
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from src.accounting import COMPUTE_JSON_FIELDS, Compute
from src.data.records import Sample
from src.w1.config import load_w1_config
from src.w1.loop import (
    FAILURE_NO_ANSWER,
    FAILURE_NOT_SUBMITTED,
    FAILURE_WRONG_SUBMISSION,
    FAILURE_INPUT_REJECTED,
    REFUSED_AFTER_SUBMISSION,
    REFUSED_TOOL_CAP,
    LoopPrompts,
    Protocol,
    plan_sub_batches,
    run_episodes,
    tools_digest,
)
from tests._w1_fakes import ChatTurnResult, FakeWorkbench, ScriptedChatAdapter, raw_call, turn

CFG = load_w1_config()
W, H = 40, 30


def image(seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 60, (H, W, 3), dtype=np.uint8)
    arr[8:22, 10:30] = 220
    return Image.fromarray(arr)


def sample(key: str, task: str = "refseg") -> Sample:
    fields = {"expression": f"the thing {key}", "countbenchqa_question": f"How many things {key} are there in the image?"}
    poly = [[10.0, 8.0, 30.0, 8.0, 30.0, 22.0, 10.0, 22.0]]
    return Sample(id=f"t/x/{key}", task="refexp", images=("/nonexistent.png",), image_keys=(f"k:{key}",), fields=fields,
                  gt={"box": [10.0, 8.0, 30.0, 22.0], "mask": {"polygons": poly, "height": H, "width": W}, "count": 3},
                  source="t", split="x")


def protocol(**kw) -> Protocol:
    return Protocol(**{"max_rounds": 6, "max_tool_calls": 8, "thinking_budget": 2048, "answer_budget": 1024, **kw})


def prompts(p: Protocol) -> LoopPrompts:
    return CFG.loop_prompts(p)


def run(script, task="refseg", p=None, batch_size=8, **kw):
    p = p or protocol()
    keys = list(script)
    ad = ScriptedChatAdapter({f"the thing {k}" if task != "count" else f"things {k} " : v for k, v in script.items()})
    ss = [sample(k, task) for k in keys]
    recs = run_episodes(ad, ss, task, p, prompts(p), batch_size=batch_size, workbench_factory=FakeWorkbench,
                        images=[image(i) for i in range(len(ss))], **kw)
    return ad, {k: r for k, r in zip(keys, recs)}


ZOOM = ("zoom", {"box": "[0, 0, 500, 500]"})
DRAW = ("draw_box", {"box": "[250, 266, 750, 733]", "out": "m"})
SUBMIT_M = ("submit_mask", {"name": "m"})


# ---------------------------------------------------------------------------
# 上限
# ---------------------------------------------------------------------------


def test_round_cap_gives_no_answer():
    _, r = run({"a": [turn(calls=[ZOOM])] * 6})
    a = r["a"]
    assert (a.ended_by, a.failure, a.n_rounds, a.n_tool_calls) == ("round_cap", FAILURE_NO_ANSWER, 6, 6)
    assert a.answer is None and a.mask is None
    js = a.to_json()
    assert js["failure"] == "no_answer" and [x["round"] for x in js["rounds"]] == [1, 2, 3, 4, 5, 6]


def test_round_cap_respects_protocol_value():
    _, r = run({"a": [turn(calls=[ZOOM])] * 3}, p=protocol(max_rounds=3))
    assert (r["a"].n_rounds, r["a"].failure) == (3, FAILURE_NO_ANSWER)


def test_tool_call_cap_refuses_and_submit_is_exempt():
    three = turn(calls=[ZOOM, ZOOM, ZOOM])
    ad, r = run({"a": [three, three, three, turn(calls=[DRAW, SUBMIT_M]), turn(text="done")]})
    a = r["a"]
    calls = [c for rd in a.rounds for c in rd["tool_calls"]]
    assert [c["executed"] for c in calls[:9]] == [True] * 8 + [False]
    assert calls[8]["refused"] == REFUSED_TOOL_CAP and a.n_tool_calls == 8 and a.n_refused == 2  # 第 9 次与 draw_box
    assert calls[9]["name"] == "draw_box" and calls[9]["refused"] == REFUSED_TOOL_CAP and not calls[9]["counted"]
    # 提交不计数、总是执行；draw_box 被拒，mask m 不存在 → 提交报错，没有答案；第 5 轮给了文字 → not_submitted
    assert calls[10]["name"] == "submit_mask" and calls[10]["executed"] and calls[10]["result"]["error"] == "unknown_mask"
    assert (a.ended_by, a.failure, a.n_executed) == ("final_answer", FAILURE_NOT_SUBMITTED, 9)
    conv = ad.conversations[-1][0]
    cap = CFG.loop_prompts(protocol()).tool_cap
    cap_msgs = [m for m in conv if m["role"] == "tool" and m["content"][0]["text"] == cap]
    assert len(cap_msgs) == 2 and "8 calls" in cap  # 第 9 次 zoom 与 draw_box 各一条上限说明


def test_submit_after_cap_still_ends_the_episode():
    three = turn(calls=[ZOOM, ZOOM, ZOOM])
    _, r = run({"a": [turn(calls=[DRAW, ZOOM]), three, three, turn(calls=[ZOOM, SUBMIT_M])]})
    a = r["a"]
    assert a.n_tool_calls == 8 and a.n_refused == 1
    assert (a.ended_by, a.failure) == ("submission", None) and a.mask is not None and a.n_executed == 9


def test_submit_counts_when_not_exempt():
    three = turn(calls=[ZOOM, ZOOM, ZOOM])
    p = protocol(submit_exempt_from_cap=False, max_rounds=4)
    _, r = run({"a": [turn(calls=[DRAW, ZOOM]), three, three, turn(calls=[SUBMIT_M])]}, p=p)
    a = r["a"]
    last = a.rounds[-1]["tool_calls"][0]
    assert last["name"] == "submit_mask" and last["refused"] == REFUSED_TOOL_CAP
    assert (a.ended_by, a.failure) == ("round_cap", FAILURE_NO_ANSWER)


# ---------------------------------------------------------------------------
# 按轮批量
# ---------------------------------------------------------------------------


def test_batches_shrink_as_samples_finish():
    fin = turn(calls=[DRAW, SUBMIT_M])
    z = turn(calls=[ZOOM])
    script = {"a": [fin], "b": [z, fin], "c": [z, z, z, fin], "d": [z, fin], "e": [z, z, fin]}
    ad, r = run(script, batch_size=2)
    # 第 1 轮 5 条：2 + 2 + 1；第 2 轮 b c d e：2 + 2；第 3 轮 c e：2；第 4 轮 c：1
    assert ad.batch_sizes == [2, 2, 1, 2, 2, 2, 1]
    assert {k: v.n_rounds for k, v in r.items()} == {"a": 1, "b": 2, "c": 4, "d": 2, "e": 3}
    assert all(v.ended_by == "submission" and v.failure is None for v in r.values())
    assert [rd["batch_rows"] for rd in r["c"].rounds] == [2, 2, 2, 1]
    # 每一批里的行按样本的原顺序
    firsts = [[next(c["text"] for c in conv[1]["content"] if c["type"] == "text") for conv in b] for b in ad.conversations]
    assert ["thing c" in t for t in firsts[-1]] == [True]
    assert ["thing b" in firsts[3][0], "thing c" in firsts[3][1]] == [True, True]


def test_budgets_passed_to_adapter():
    ad, _ = run({"a": [turn(calls=[ZOOM]), turn(calls=[DRAW, SUBMIT_M])]}, p=protocol(thinking_budget=123, answer_budget=45))
    assert ad.budgets == [(123, 45), (123, 45)]


# ---------------------------------------------------------------------------
# 结束与失败
# ---------------------------------------------------------------------------


def test_refseg_submission_mask_is_recorded():
    _, r = run({"a": [turn(calls=[DRAW]), turn(calls=[SUBMIT_M])]})
    a = r["a"]
    assert (a.ended_by, a.failure, a.n_rounds, a.n_tool_calls, a.n_executed) == ("submission", None, 2, 1, 2)
    assert a.mask.dtype == bool and a.mask.shape == (H, W) and a.mask.sum() > 0
    assert a.answer["source"] == "submit_mask" and a.answer["kind"] == "mask" and a.answer["mask"]["area"] == int(a.mask.sum())
    from src.executors.raster import rle_to_mask

    assert np.array_equal(rle_to_mask(a.answer["mask"]["rle"]), a.mask)


def test_refseg_text_answer_is_not_submitted():
    _, r = run({"a": [turn(text="It is the cat on the left.")]})
    assert (r["a"].ended_by, r["a"].failure) == ("final_answer", FAILURE_NOT_SUBMITTED)


@pytest.mark.parametrize("t, want", [
    (turn(text="partial", stopped=False), "truncated"),
    (turn(text="", thinking_end="unclosed"), "thinking_unclosed"),
    (turn(text="   "), "empty"),
])
def test_refseg_text_failures_follow_e1_precedence(t, want):
    _, r = run({"a": [t]})
    assert r["a"].failure == want


def test_refseg_wrong_submission_kind():
    _, r = run({"a": [turn(calls=[("submit_box", {"box": "[100, 100, 900, 900]"})])]})
    a = r["a"]
    assert (a.ended_by, a.failure) == ("submission", FAILURE_WRONG_SUBMISSION) and a.answer["submitted"] == ["box"]


def test_box_text_answer_and_parse_failure():
    ok = turn(text='```json\n[{"bbox_2d": [250, 266, 750, 733], "label": "thing"}]\n```')
    _, r = run({"a": [ok], "b": [turn(text="I cannot find it.")], "c": [turn(calls=[("submit_box", {"box": "[250, 266, 750, 733]"})])]},
               task="refexp_box")
    assert (r["a"].failure, r["a"].answer["source"]) == (None, "text")
    np.testing.assert_allclose(r["a"].box, [10.0, 7.98, 30.0, 21.99], atol=0.02)
    assert r["b"].failure == "no_box" and r["b"].box is None
    assert r["c"].failure is None and r["c"].answer["source"] == "submit_box" and r["c"].box == pytest.approx(r["a"].box)


def test_count_text_and_submission():
    _, r = run({"a": [turn(text="There are 3.")], "b": [turn(text="3 or 4")], "c": [turn(calls=[("submit_count", {"count": "5"})])]},
               task="count")
    assert (r["a"].failure, r["a"].count) == (None, 3)
    assert r["b"].failure == "multiple_numbers" and r["b"].count is None
    assert (r["c"].failure, r["c"].count, r["c"].answer["source"]) == (None, 5, "submit_count")


def test_calls_after_submission_are_not_executed():
    _, r = run({"a": [turn(calls=[DRAW, SUBMIT_M, ZOOM, DRAW])]})
    a = r["a"]
    calls = a.rounds[0]["tool_calls"]
    assert [c["executed"] for c in calls] == [True, True, False, False]
    assert [c["refused"] for c in calls] == [None, None, REFUSED_AFTER_SUBMISSION, REFUSED_AFTER_SUBMISSION]
    assert (a.ended_by, a.n_refused, a.n_tool_calls) == ("submission", 2, 1)


def test_tool_call_parse_errors_continue_the_episode():
    ad, r = run({"a": [turn(errors=["bad xml"]), turn(calls=[DRAW, SUBMIT_M])]})
    a = r["a"]
    assert a.n_tool_call_errors == 1 and a.n_rounds == 2 and a.failure is None and a.n_tool_calls == 1
    conv = ad.conversations[1][0]
    assert conv[-1]["role"] == "tool" and conv[-1]["content"][0]["text"] == \
        CFG.data["prompts"]["messages"]["tool_call_error"].format(error="bad xml")
    assert a.rounds[0]["tool_call_errors"] == ["bad xml"]


def test_disabled_group_tools_are_absent_and_calls_error():
    _, r = run({"a": [turn(calls=[ZOOM]), turn(calls=[DRAW, SUBMIT_M])]}, p=protocol(disabled_groups=frozenset({"eye"})))
    c = r["a"].rounds[0]["tool_calls"][0]
    assert c["executed"] and c["result"]["error"] == "unknown_tool" and r["a"].failure is None


# ---------------------------------------------------------------------------
# 历史与提示
# ---------------------------------------------------------------------------


def test_history_format_and_last_round_notice():
    z = turn(text="Zooming.", thinking="where is it", calls=[ZOOM])
    ad, r = run({"a": [z, z, z, z, z, turn(calls=[DRAW, SUBMIT_M])]})
    last = ad.conversations[-1][0]
    p = prompts(protocol())
    assert last[0] == {"role": "system", "content": [{"type": "text", "text": p.system}]}
    assert last[1]["content"][1]["text"] == p.task_text("refseg", sample("a"))
    a1 = last[2]
    # D104 的回填：content = 作答原文（含模型自己写的 <tool_call> 块），reasoning_content = 思考段，不给 tool_calls
    assert set(a1) == {"role", "content", "reasoning_content"} and a1["reasoning_content"] == "where is it"
    assert a1["content"] == [{"type": "text", "text": "Zooming.\n" + raw_call(*ZOOM)}] == [{"type": "text", "text": r["a"].rounds[0]["text"]}]
    t1 = last[3]
    assert t1["role"] == "tool" and [c["type"] for c in t1["content"]] == ["text", "image"]
    assert last[-1]["role"] == "tool" and last[-1]["content"][-1] == {"type": "text", "text": p.last_round}
    notices = [m for conv in ad.conversations for m in conv[0] if m["role"] == "tool"
               and m["content"][-1].get("text") == p.last_round]
    assert len(notices) == 1  # 只在进入第 6 轮之前加一次
    assert r["a"].rounds[4].get("last_round_notice") is True and r["a"].failure is None
    img = r["a"].rounds[0]["tool_calls"][0]["result"]["images"][0]
    assert set(img) == {"size", "mode", "sha256"} and len(img["sha256"]) == 64


def test_system_prompt_identical_across_tasks_and_counts_question():
    p = prompts(protocol())
    assert "{" not in p.system and "6 turns" in p.system and "8 tool calls" in p.system
    s = sample("q")
    assert p.task_text("count", s).startswith("Task: How many things q are there in the image?")
    s2 = Sample(id="x", task="count", images=(), image_keys=(), fields={"text": "two cats"}, gt={"count": 2}, source="c", split="a")
    with pytest.raises(KeyError, match="人工问句"):
        p.task_text("count", s2)


# ---------------------------------------------------------------------------
# 预检与按 token 上限拆子批（D104）
# ---------------------------------------------------------------------------


def test_precheck_rejects_structural_tokens_without_breaking_the_batch():
    fin = turn(calls=[DRAW, SUBMIT_M])
    bad = turn(text="look <|vision_start|> here", calls=[ZOOM])  # 回填之后下一轮的对话里有结构性 token
    ad, r = run({"a": [turn(calls=[ZOOM]), fin], "b": [bad, fin], "c": [turn(calls=[ZOOM]), fin]}, batch_size=8)
    assert (r["b"].ended_by, r["b"].failure) == ("input_rejected", FAILURE_INPUT_REJECTED)
    assert r["b"].input_error["round"] == 2 and "<|vision_start|>" in r["b"].input_error["error"]
    assert r["b"].n_rounds == 1 and r["b"].to_json()["input_error"] == r["b"].input_error
    assert all(r[k].failure is None and r[k].ended_by == "submission" for k in ("a", "c"))
    assert ad.batch_sizes == [3, 2]  # 第 2 轮 b 不进批（假适配器与真的一样，批里有结构性 token 就整批报错）
    assert ad.n_chat_inputs == 3 + 3  # 每轮每条还没结束的样本预检一次


def test_plan_sub_batches_rules():
    assert plan_sub_batches([10] * 5, 8, None) == [[0, 1, 2, 3, 4]]
    assert plan_sub_batches([10] * 5, 2, None) == [[0, 1], [2, 3], [4]]
    assert plan_sub_batches([10, 10, 10, 10], 8, 30) == [[0, 1, 2], [3]]
    # 填充后的宽度按子批里最长的一条算：加入 25 之后 2 × 25 = 50 > 40，另起一批
    assert plan_sub_batches([10, 25, 10, 10], 8, 40) == [[0], [1], [2, 3]]
    assert plan_sub_batches([100, 5, 5], 8, 50) == [[0], [1, 2]]  # 一条本身就超过上限：单独成批
    assert plan_sub_batches([], 4, 10) == []
    rng = np.random.default_rng(0)
    for _ in range(200):
        w = rng.integers(1, 60, int(rng.integers(1, 30))).tolist()
        bs, cap = int(rng.integers(1, 6)), int(rng.integers(20, 200))
        plan = plan_sub_batches(w, bs, cap)
        assert [i for p in plan for i in p] == list(range(len(w)))  # 顺序不变、不重不漏
        for p in plan:
            assert len(p) <= bs and (len(p) == 1 or len(p) * max(w[i] for i in p) <= cap)
        assert plan == plan_sub_batches(w, bs, cap)  # 确定


def test_token_cap_splits_batches_without_changing_records():
    fin = turn(calls=[DRAW, SUBMIT_M])
    z = turn(calls=[ZOOM])
    script = {"a": [z, fin], "b": [z, z, fin], "c": [fin], "d": [z, fin], "e": [z, z, z, fin]}
    ad0, r0 = run(script, batch_size=8)
    w1 = ad0.widths[0][0]  # 第 1 轮每条的提示长度（都一样）
    cap = 2 * w1 + 1
    ad1, r1 = run(script, batch_size=8, max_batch_tokens=cap)
    assert ad0.batch_sizes[0] == 5 and ad1.batch_sizes[:3] == [2, 2, 1]
    for rows in ad1.widths:
        assert len(rows) == 1 or len(rows) * max(rows) <= cap
    assert len(ad1.batch_sizes) > len(ad0.batch_sizes)  # 后几轮带了图，提示变长，拆得更细

    def strip(rec):
        js = rec.to_json()
        for rd in js["rounds"]:
            rd.pop("batch_rows")
        return json.dumps(js)

    assert {k: strip(v) for k, v in r0.items()} == {k: strip(v) for k, v in r1.items()}  # 只有批的组成不同


# ---------------------------------------------------------------------------
# 记录与计算记账
# ---------------------------------------------------------------------------


def test_compute_fields_complete_and_consistent():
    _, r = run({"a": [turn(calls=[ZOOM]), turn(thinking_end="forced", calls=[DRAW]), turn(calls=[SUBMIT_M])]})
    a = r["a"]
    c = a.compute
    assert isinstance(c, Compute) and c.n_samples == 1 and c.n_calls == c.n_rounds == 3
    tok = a.tokens()
    assert c.generated_tokens == tok["generated"] and c.prompt_tokens == tok["prompt"] + tok["inserted"]
    assert tok["inserted"] == 2 and c.visual_tokens == tok["visual"] and c.image_pixels == tok["image_pixels"] > 0
    assert c.wall_time_s == pytest.approx(sum(a.round_walls) + a.tool_time_s)
    assert tuple(c.to_json_dict()) == COMPUTE_JSON_FIELDS
    js = a.to_json()
    assert {"n_rounds", "n_tool_calls", "n_executed", "n_refused", "n_tool_call_errors", "tokens", "thinking_end"} <= set(js)
    assert js["thinking_end"] == ["natural", "forced", "natural"]
    text = json.dumps(js)
    assert "wall" not in text and "elapsed" not in text  # 墙钟不进 predictions
    t = a.timing()
    assert set(t) == {"id", "round_walls", "tool_times", "tool_time_s", "wall_time_s", "call_time_s"} and len(t["round_walls"]) == 3
    assert [len(x) for x in t["tool_times"]] == [1, 1, 1] and len(a.workbench_log) == 3


def test_tool_workers_do_not_change_outputs():
    fin = turn(calls=[DRAW, SUBMIT_M])
    z = turn(calls=[ZOOM])
    script = {"a": [fin], "b": [z, fin], "c": [z, z, fin], "d": [z, fin]}
    _, r1 = run(script, batch_size=3)
    _, r3 = run(script, batch_size=3, tool_workers=3)
    assert {k: json.dumps(v.to_json()) for k, v in r1.items()} == {k: json.dumps(v.to_json()) for k, v in r3.items()}


def test_missing_pixels_is_an_error():
    class NoPixels(ScriptedChatAdapter):
        def _result(self, conv, spec, rows):
            res = super()._result(conv, spec, rows)
            return ChatTurnResult(**{**res.__dict__, "n_image_pixels": None})

    ad = NoPixels({"the thing a": [turn(text="x")]})
    with pytest.raises(RuntimeError, match="n_image_pixels"):
        run_episodes(ad, [sample("a")], "refseg", protocol(), prompts(protocol()), batch_size=1, workbench_factory=FakeWorkbench,
                     images=[image()])


def test_tools_digest_checked_and_workbench_gets_a_copy():
    seen = []

    def factory(im, **kw):
        seen.append(im)
        return FakeWorkbench(im, **kw)

    im = image()
    want = tools_digest(FakeWorkbench(im).specs())
    run_episodes(ScriptedChatAdapter({"the thing a": [turn(text="x")]}), [sample("a")], "refseg", protocol(), prompts(protocol()),
                 batch_size=1, workbench_factory=factory, images=[im], expected_tools_sha256=want)
    assert seen and seen[0] is not im  # 工作台拿的是副本
    with pytest.raises(RuntimeError, match="工具清单"):
        run_episodes(ScriptedChatAdapter({"the thing a": [turn(text="x")]}), [sample("a")], "refseg", protocol(),
                     prompts(protocol()), batch_size=1, workbench_factory=FakeWorkbench, images=[im],
                     expected_tools_sha256="0" * 64)
    other = tools_digest(FakeWorkbench(im, disabled_groups=frozenset({"eye"})).specs())
    assert other != want


def test_protocol_validation():
    with pytest.raises(ValueError):
        Protocol(max_rounds=0)
    with pytest.raises(ValueError, match="submit"):
        Protocol(disabled_groups=frozenset({"submit"}))
    d = protocol(disabled_groups=frozenset({"snap", "code"})).as_dict()
    assert d["disabled_groups"] == ["code", "snap"] and d["submit_tools"] == ["submit_box", "submit_count", "submit_mask"]
