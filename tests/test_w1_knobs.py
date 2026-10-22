"""DEV-0053：W1 回路的协议开关、工作台选项透传与冻结保护（configs/w1.json；src/w1/loop.py、config.py、runner.py；scripts/run_w1.py）。

- final_round_tools = submit_only：最后一轮交给 chat_inputs 与 chat_turn_batch 的工具清单只有提交工具（顺序照 specs()）；这一轮里的
  非提交调用不执行、不计数，记 final_round，回复 final_round_refused；同一段里排在后面的提交照常执行；文字答案照旧有效；max_rounds = 1
  时不生效；缺 final_round_refused 报错。
- capped_tools = submit_only：某一轮开始时计数的调用已满，这一轮只给提交工具；被拒的调用照旧记 tool_cap。
- status：每轮执行完、样本还没结束时追加在这一轮最后一条 tool 消息的末尾，六个占位符的值；与 last_round 的先后；只能用六个占位符。
- 记录：只在非默认时加 tools_offered / status_notice；Protocol.as_dict() 带两个开关。
- 工作台选项：默认不传；非默认时传给回路里的每个工作台与运行身份的探测（命令行彩排里核对）。
- 冻结保护：未冻结时 devcheck、test_* 退出码 2（全部设置）；devtune 照常；--tiny 照常；--w1-config 的路径与 sha256 进运行身份。
- 全部开关打开的临时配置跑一遍 tiny 彩排（假适配器不主动收尾，开关都被走到），rc = 0。
- DEV-0056（D111）：configs/w1.json 已冻结为调参臂 U2。测"默认、未冻结"行为与冻结保护拒跑的测试读冻结前的草稿夹具
  （tests.test_w1_loop.DRAFT）；另有三项钉住冻结后的仓库配置：frozen 为 true、与 U2 除 name/status/frozen 外相同、冻结保护放行。
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import importlib.util
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from src.w1.config import W1_CONFIG_PATH, WORKBENCH_OPTIONS, W1ConfigError, file_sha256, load_w1_config, validate
from src.w1.loop import (
    REFUSED_AFTER_SUBMISSION,
    REFUSED_FINAL_ROUND,
    REFUSED_TOOL_CAP,
    STATUS_FIELDS,
    LoopPrompts,
    Protocol,
    _Episode,
    run_episodes,
)
from tests._w1_fakes import TOOLS, FakeWorkbench, OptionsWorkbench, ScriptedChatAdapter, turn
from tests.test_w1_loop import DRAFT, DRAFT_PATH, DRAW, SUBMIT_M, ZOOM, image, protocol, sample

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "run_w1.py"
CFG = load_w1_config()  # 仓库配置（D111 起已冻结）；测默认、未冻结行为的读 DRAFT
ALL_TOOLS = list(TOOLS)  # FakeWorkbench.specs() 的顺序
SUBMIT_ORDER = [n for n in ALL_TOOLS if n.startswith("submit_")]  # specs() 里的提交工具，顺序照 specs()
REFUSAL = "REFUSED-FINAL-ROUND"
STATUS = "S r={round}/{max_rounds} left={rounds_left} used={calls_used}/{max_tool_calls} calls_left={calls_left}"


def prompts(p: Protocol, *, cfg=CFG, **kw) -> LoopPrompts:
    return dataclasses.replace(cfg.loop_prompts(p), **kw)


def run(script, p, *, task="refseg", status=None, refusal=REFUSAL, batch_size=8, cfg=CFG, **kw):
    keys = list(script)
    ad = ScriptedChatAdapter({f"the thing {k}" if task != "count" else f"things {k} ": v for k, v in script.items()})
    ss = [sample(k, task) for k in keys]
    recs = run_episodes(ad, ss, task, p, prompts(p, cfg=cfg, status=status, final_round_refused=refusal), batch_size=batch_size,
                        workbench_factory=FakeWorkbench, images=[image(i) for i in range(len(ss))], **kw)
    return ad, {k: r for k, r in zip(keys, recs)}


def episode_messages(script_turns, p, *, status=None, refusal=REFUSAL, cfg=CFG):
    """一条样本照 run_episodes 的步骤逐轮走，返回 (_Episode, adapter)；最后一轮之后的历史（模型看不到）也能查。"""
    ad = ScriptedChatAdapter({"the thing a": script_turns})
    pr = prompts(p, cfg=cfg, status=status, final_round_refused=refusal)
    im = image()
    e = _Episode(sample("a"), im, FakeWorkbench(im.copy()), "refseg", p, pr, "sha")
    for r in range(1, p.max_rounds + 1):
        if e.done:
            break
        if r == p.max_rounds and r > 1 and pr.last_round:
            e.add_last_round_notice(pr.last_round)
        sp = e.tool_specs(r)
        res = ad.chat_turn_batch([list(e.messages)], tools=[sp], thinking_budget=p.thinking_budget, answer_budget=p.answer_budget)[0]
        e.step(r, res, 1, ad)
    return e, ad


def texts(msg) -> list[str]:
    return [c["text"] for c in msg["content"] if c["type"] == "text"]


# ---------------------------------------------------------------------------
# final_round_tools
# ---------------------------------------------------------------------------


def test_final_round_offers_only_submit_tools_and_refuses_others():
    p = protocol(final_round_tools="submit_only")
    script = {"a": [turn(calls=[DRAW])] + [turn(calls=[ZOOM])] * 4 + [turn(calls=[ZOOM, SUBMIT_M, ZOOM])]}
    ad, r = run(script, p)
    a = r["a"]
    # 交给 chat_turn_batch 与 chat_inputs 的工具清单：前 5 轮全部，第 6 轮只有提交工具（顺序照 specs()）
    assert [rows[0] for rows in ad.tools_seen] == [ALL_TOOLS] * 5 + [SUBMIT_ORDER]
    assert ad.chat_inputs_tools == [ALL_TOOLS] * 5 + [SUBMIT_ORDER]
    last = a.rounds[-1]
    assert last["tools_offered"] == "submit_only" and all("tools_offered" not in rd for rd in a.rounds[:-1])
    calls = last["tool_calls"]
    assert [c["name"] for c in calls] == ["zoom", "submit_mask", "zoom"]
    assert [c["refused"] for c in calls] == [REFUSED_FINAL_ROUND, None, REFUSED_AFTER_SUBMISSION]
    assert [c["executed"] for c in calls] == [False, True, False] and not calls[0]["counted"] and calls[0]["result"] is None
    # 排在被拒调用后面的提交照常执行
    assert (a.ended_by, a.failure, a.n_tool_calls, a.n_refused, a.n_executed) == ("submission", None, 5, 2, 6)
    assert a.mask is not None and a.mask.sum() > 0


def test_final_round_refusal_message_in_history():
    p = protocol(final_round_tools="submit_only", max_rounds=3)
    e, _ = episode_messages([turn(calls=[DRAW]), turn(calls=[ZOOM]), turn(calls=[ZOOM, DRAW, SUBMIT_M])], p)
    tools = [m for m in e.messages if m["role"] == "tool"]
    # 第 3 轮：zoom 与 draw_box 各一条拒绝回复（没有占位符，原样），然后是提交的结果
    assert [texts(m)[0] for m in tools[-3:]] == [REFUSAL, REFUSAL, "Submitted."]
    assert [c["refused"] for c in e.rec.rounds[-1]["tool_calls"]] == [REFUSED_FINAL_ROUND, REFUSED_FINAL_ROUND, None]
    assert e.rec.failure is None and e.rec.n_tool_calls == 2


def test_final_round_without_submission_is_no_answer_and_text_answer_still_counts():
    p = protocol(final_round_tools="submit_only", max_rounds=2)
    box = turn(text='```json\n[{"bbox_2d": [250, 266, 750, 733], "label": "thing"}]\n```')
    _, r = run({"a": [turn(calls=[ZOOM]), turn(calls=[ZOOM])], "b": [turn(calls=[ZOOM]), box]}, p, task="refexp_box")
    assert (r["a"].ended_by, r["a"].failure, r["a"].n_tool_calls) == ("round_cap", "no_answer", 1)
    assert r["a"].rounds[-1]["tool_calls"][0]["refused"] == REFUSED_FINAL_ROUND
    assert (r["b"].ended_by, r["b"].failure, r["b"].answer["source"]) == ("final_answer", None, "text")
    assert r["b"].rounds[-1]["tools_offered"] == "submit_only"
    _, c = run({"a": [turn(calls=[ZOOM]), turn(text="There are 3.")]}, p, task="count")
    assert (c["a"].failure, c["a"].count) == (None, 3)


def test_final_round_switch_has_no_effect_with_one_round():
    p = protocol(final_round_tools="submit_only", max_rounds=1)
    ad, r = run({"a": [turn(calls=[DRAW, SUBMIT_M])]}, p)
    assert ad.tools_seen == [[ALL_TOOLS]] and "tools_offered" not in r["a"].rounds[0]
    assert (r["a"].ended_by, r["a"].failure, r["a"].n_tool_calls) == ("submission", None, 1)


def test_final_round_default_offers_all_tools():
    ad, r = run({"a": [turn(calls=[ZOOM])] * 6}, protocol())
    assert [rows[0] for rows in ad.tools_seen] == [ALL_TOOLS] * 6
    assert all("tools_offered" not in rd and "status_notice" not in rd for rd in r["a"].rounds)


def test_final_round_requires_refusal_text():
    p = protocol(final_round_tools="submit_only")
    with pytest.raises(ValueError, match="final_round_refused"):
        run({"a": [turn(text="x")]}, p, refusal=None)
    run({"a": [turn(calls=[DRAW, SUBMIT_M])]}, protocol(final_round_tools="submit_only", max_rounds=1), refusal=None)  # 不生效时不要


# ---------------------------------------------------------------------------
# capped_tools
# ---------------------------------------------------------------------------


def test_capped_tools_offers_only_submit_tools_after_cap():
    three = turn(calls=[ZOOM, ZOOM, ZOOM])
    p = protocol(capped_tools="submit_only")
    # 第 3 轮开始时 6 次（< 8）：全部工具；这一轮里计满 8 次（第 3 次 zoom 被拒）；第 4 轮只给提交工具，非提交调用照旧记 tool_cap
    ad, r = run({"a": [three, three, turn(calls=[DRAW, ZOOM, ZOOM]), turn(calls=[ZOOM, SUBMIT_M])]}, p)
    a = r["a"]
    assert [rows[0] for rows in ad.tools_seen] == [ALL_TOOLS] * 3 + [SUBMIT_ORDER]
    assert ad.chat_inputs_tools[-1] == SUBMIT_ORDER
    assert [rd.get("tools_offered") for rd in a.rounds] == [None, None, None, "submit_only"]
    assert a.rounds[2]["tool_calls"][2]["refused"] == REFUSED_TOOL_CAP  # 满额发生在轮中：这一轮给的仍是全部
    c4 = a.rounds[3]["tool_calls"]
    assert [x["refused"] for x in c4] == [REFUSED_TOOL_CAP, None] and c4[1]["executed"]
    assert (a.ended_by, a.failure, a.n_tool_calls, a.n_refused) == ("submission", None, 8, 2)


def test_capped_tools_default_keeps_all_tools_after_cap():
    three = turn(calls=[ZOOM, ZOOM, ZOOM])
    ad, r = run({"a": [turn(calls=[DRAW, ZOOM, ZOOM]), three, three, turn(calls=[ZOOM, SUBMIT_M])]}, protocol())
    assert [rows[0] for rows in ad.tools_seen] == [ALL_TOOLS] * 4
    assert all("tools_offered" not in rd for rd in r["a"].rounds)
    assert [c["refused"] for c in r["a"].rounds[-1]["tool_calls"]] == [REFUSED_TOOL_CAP, None] and r["a"].failure is None


def test_both_switches_in_final_round_refuse_as_final_round():
    """最后一轮又已满额：非提交调用记 final_round（先后：after_submission → final_round → tool_cap）。"""
    three = turn(calls=[ZOOM, ZOOM, ZOOM])
    p = protocol(final_round_tools="submit_only", capped_tools="submit_only", max_rounds=4)
    ad, r = run({"a": [three, three, three, turn(calls=[ZOOM, SUBMIT_M])]}, p)
    assert ad.tools_seen[-1] == [SUBMIT_ORDER]
    assert [c["refused"] for c in r["a"].rounds[-1]["tool_calls"]] == [REFUSED_FINAL_ROUND, None]


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


def expected_status(rnd: int, used: int, max_rounds: int = 6, max_calls: int = 8) -> str:
    return f"S r={rnd}/{max_rounds} left={max_rounds - rnd} used={used}/{max_calls} calls_left={max(0, max_calls - used)}"


def test_status_notice_values_and_position():
    three = turn(calls=[ZOOM, ZOOM, ZOOM])
    script = {"a": [turn(calls=[DRAW, ZOOM, ZOOM]), three, turn(errors=["bad xml"]), three, turn(calls=[ZOOM]), turn(calls=[SUBMIT_M])]}
    ad, r = run(script, protocol(), status=STATUS, cfg=DRAFT)  # 默认走法：两种提示都走 tool
    a = r["a"]
    used = [3, 6, 6, 8, 8]  # 第 1–5 轮结束时计数的调用
    last_round = DRAFT.loop_prompts(protocol()).last_round
    for k in range(1, 6):  # 第 k+1 轮的对话：最后一条是 tool 消息，末尾是第 k 轮的状态
        conv = ad.conversations[k][0]
        assert conv[-1]["role"] == "tool"
        tail = texts(conv[-1])
        want = expected_status(k, used[k - 1])
        if k < 5:
            assert tail[-1] == want
        else:  # 进入最后一轮：status 在前，last_round 在后
            assert tail[-2:] == [want, last_round]
        # 每轮只追加一次，追加在这一轮最后一条 tool 消息上
        assert sum(t == want for m in conv for t in texts(m)) == 1
    assert texts(ad.conversations[3][0][-1])[0].startswith("This tool call could not be parsed")  # 第 3 轮只有解析失败的回复
    assert [rd.get("status_notice") for rd in a.rounds] == [True] * 5 + [None]  # 第 6 轮之后样本结束，不追加
    assert a.rounds[4]["last_round_notice"] is True and a.failure is None


def test_status_not_added_after_the_episode_ends():
    _, r = run({"a": [turn(calls=[DRAW]), turn(calls=[DRAW, SUBMIT_M])], "b": [turn(text="It is there.")]}, protocol(),
               status=STATUS)
    assert [rd.get("status_notice") for rd in r["a"].rounds] == [True, None]
    assert "status_notice" not in r["b"].rounds[0]


def test_status_with_partial_placeholders_and_with_final_round_switch():
    p = protocol(final_round_tools="submit_only", max_rounds=3)
    e, ad = episode_messages([turn(calls=[DRAW]), turn(calls=[ZOOM]), turn(calls=[ZOOM, SUBMIT_M])], p,
                             status="{rounds_left} turns and {calls_left} calls left.", cfg=DRAFT)
    assert texts(ad.conversations[1][0][-1])[-1] == "2 turns and 7 calls left."
    assert texts(ad.conversations[2][0][-1])[-2:] == ["1 turns and 6 calls left.", DRAFT.loop_prompts(p).last_round]
    assert ad.tools_seen[-1] == [SUBMIT_ORDER] and e.rec.failure is None


def test_status_placeholders_are_checked():
    with pytest.raises(ValueError, match="占位符"):
        prompts(protocol(), status="{round} {error}")
    with pytest.raises(ValueError):
        prompts(protocol(), status="{round")
    assert set(STATUS_FIELDS) == {"round", "max_rounds", "rounds_left", "calls_used", "max_tool_calls", "calls_left"}


def test_status_requires_a_tool_message():
    p = protocol()
    e, _ = episode_messages([turn(text="done")], p, status=STATUS)
    e.messages.append({"role": "assistant", "content": [{"type": "text", "text": "x"}], "reasoning_content": ""})
    with pytest.raises(RuntimeError, match="tool 消息"):
        e.add_status_notice(1)


# ---------------------------------------------------------------------------
# Protocol 与配置
# ---------------------------------------------------------------------------


def test_protocol_switches_validated_and_in_identity():
    d = protocol(final_round_tools="submit_only").as_dict()
    assert (d["final_round_tools"], d["capped_tools"]) == ("submit_only", "all")
    assert list(protocol().as_dict())[-2:] == ["final_round_tools", "capped_tools"]
    for bad in ({"final_round_tools": "none"}, {"capped_tools": "submit"}):
        with pytest.raises(ValueError):
            protocol(**bad)


def test_default_config_keeps_previous_behaviour():
    """未冻结的草稿（DEV-0053 之后的默认）：两个开关 all，没有 status，工作台什么都不传。"""
    pr = DRAFT.data["protocol"]
    assert (pr["final_round_tools"], pr["capped_tools"]) == ("all", "all")
    p = DRAFT.protocol("full")
    assert (p.final_round_tools, p.capped_tools) == ("all", "all") and not p.final_round_submit_only(6)
    lp = DRAFT.loop_prompts(p)
    assert lp.status is None and DRAFT.data["prompts"]["messages"]["status"] is None
    assert lp.final_round_refused and "{" not in lp.final_round_refused
    assert DRAFT.data["workbench"] == {"view_frames": False} and DRAFT.workbench_options() == {}
    assert dict(WORKBENCH_OPTIONS) == {"view_frames": False}


def _cfg(path: Path = W1_CONFIG_PATH) -> dict:
    return copy.deepcopy(json.loads(path.read_text(encoding="utf-8")))


def _set_status(d: dict, text: str | None, declared: list[str]) -> None:
    d["prompts"]["messages"]["status"] = text
    d["prompts"]["placeholders"]["status"] = declared


@pytest.mark.parametrize("mutate", [
    lambda d: d["protocol"].__setitem__("final_round_tools", "none"),
    lambda d: d["protocol"].pop("capped_tools"),
    lambda d: _set_status(d, "Turn {round}, error {error}", ["round", "error"]),  # 六个以外的占位符
    lambda d: _set_status(d, "Turn {round}", []),  # 用了没声明
    lambda d: _set_status(d, "", []),
    lambda d: (d["protocol"].__setitem__("final_round_tools", "submit_only"),
               d["prompts"]["messages"].__setitem__("final_round_refused", None)),
    lambda d: (d["prompts"]["messages"].__setitem__("final_round_refused", "Not executed ({max_rounds})."),
               d["prompts"]["placeholders"].__setitem__("final_round_refused", ["max_rounds"])),
    lambda d: d["prompts"]["placeholders"].pop("status"),
    lambda d: d.pop("workbench"),
    lambda d: d["workbench"].__setitem__("view_frames", 1),
    lambda d: d["workbench"].__setitem__("zoom_cache", True),
])
def test_validation_catches_switch_mutations(mutate):
    d = _cfg()
    validate(d)
    mutate(d)
    with pytest.raises(W1ConfigError):
        validate(d)


def test_valid_variants_render(tmp_path):
    d = _cfg()
    d["protocol"].update(final_round_tools="submit_only", capped_tools="submit_only")
    _set_status(d, "Turn {round} of {max_rounds} done; {calls_left} calls left.", ["round", "max_rounds", "calls_left"])
    d["workbench"]["view_frames"] = True
    path = tmp_path / "variant.json"
    path.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    c = load_w1_config(path)
    p = c.protocol("no_eye", tiny=True)
    assert (p.final_round_tools, p.capped_tools) == ("submit_only", "submit_only") and p.final_round_submit_only(6)
    assert c.loop_prompts(p).status_text(2, p, 3) == "Turn 2 of 6 done; 5 calls left."
    assert c.workbench_options() == {"view_frames": True}
    d["prompts"]["messages"]["final_round_refused"] = None  # 不用 submit_only 时可以不给
    d["protocol"]["final_round_tools"] = "all"
    validate(d)


def test_workbench_option_defaults_match_the_workbench():
    """WORKBENCH_OPTIONS 的默认值与真工作台的关键字参数默认值相同（DEV-0054 合入之后生效；之前只核对已有的）。"""
    if importlib.util.find_spec("src.w1.workbench") is None:
        pytest.skip("src/w1/workbench.py 不在")
    from src.w1.workbench import Workbench

    params = inspect.signature(Workbench).parameters
    present = [k for k in WORKBENCH_OPTIONS if k in params]
    if not present:
        pytest.skip("Workbench 还没有 view_frames（DEV-0054 未合入）")
    for k in present:
        assert params[k].default == WORKBENCH_OPTIONS[k], k


# ---------------------------------------------------------------------------
# 工作台选项透传
# ---------------------------------------------------------------------------


def test_workbench_options_passthrough():
    seen: list[dict] = []

    def factory(im, **kw):
        seen.append(dict(kw))
        return OptionsWorkbench(im, **kw)

    def go(**kw):
        seen.clear()
        ad = ScriptedChatAdapter({"the thing a": [turn(calls=[DRAW, SUBMIT_M])], "the thing b": [turn(text="x")]})
        recs = run_episodes(ad, [sample("a"), sample("b")], "refseg", protocol(), prompts(protocol()), batch_size=2,
                            workbench_factory=factory, images=[image(0), image(1)], **kw)
        return ad, recs

    _, recs = go()
    assert seen == [{"disabled_groups": frozenset()}] * 2  # 默认什么都不传
    _, recs_empty = go(workbench_options={})
    assert seen == [{"disabled_groups": frozenset()}] * 2
    assert [json.dumps(r.to_json()) for r in recs_empty] == [json.dumps(r.to_json()) for r in recs]
    ad, recs_on = go(workbench_options={"view_frames": True})
    assert seen == [{"disabled_groups": frozenset(), "view_frames": True}] * 2
    assert recs_on[0].workbench_log[0]["options"] == {"view_frames": True}
    assert recs_on[0].tools_sha256 != recs[0].tools_sha256  # 假工作台把选项写进了工具说明
    assert all("[options" in t.description for t in OptionsWorkbench(image(), view_frames=True).specs())


# ---------------------------------------------------------------------------
# 冻结保护（scripts/run_w1.py）
# ---------------------------------------------------------------------------


def _script():
    spec = importlib.util.spec_from_file_location("_run_w1_for_knobs", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


RUN_W1 = _script()
FROZEN_MSG = "协议未冻结，不能碰开发-检查集与 test（D103 §二.4、§三）"


def _args(task, split, setting, *extra, tiny=False, out="/nonexistent/0999_w1_knobs"):
    return (["--tiny"] if tiny else ["--data-root", "/nonexistent"]) + [
        "--model", "qwen3.5-0.8b", "--task", task, "--split", split, "--setting", setting, "--out", out, *extra]


UNFROZEN = ("--w1-config", str(DRAFT_PATH))  # 冻结前的草稿（frozen = false）
# 冻结保护管的 (task, split, setting)：devcheck 与 test_*，回路设置与各种基线
GUARDED = [
    ("refseg", "devcheck", "full"),
    ("refseg", "devcheck", "baseline_ownbox"),
    ("refseg", "devcheck", "w1f"),
    ("refseg", "devcheck", "baseline_poly_structured"),
    ("refexp_box", "devcheck", "baseline_e1"),
    ("refseg", "test_refcoco_unc_testA", "no_snap"),
    ("refexp_box", "test_refcocog_umd_test", "full"),
    ("count", "test_countbench", "baseline_e1"),
    ("count", "test_countbench", "no_eye"),
]


def _frozen_copy(tmp_path: Path) -> Path:
    """草稿只把 frozen 改成 true（与 UNFROZEN 只差这一处）。"""
    d = _cfg(DRAFT_PATH)
    d["frozen"] = True
    p = tmp_path / "w1_frozen.json"
    p.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    return p


@pytest.mark.parametrize("task, split, setting", GUARDED)
def test_unfrozen_config_refuses_devcheck_and_test(task, split, setting, capsys, tmp_path):
    with pytest.raises(SystemExit) as ex:
        RUN_W1.parse_args(_args(task, split, setting, *UNFROZEN))
    assert ex.value.code == 2 and FROZEN_MSG in capsys.readouterr().err
    a = RUN_W1.parse_args(_args(task, split, setting, "--w1-config", str(_frozen_copy(tmp_path))))  # 冻结的配置照常
    assert a.split == split
    assert RUN_W1.parse_args(_args(task, split, setting, *UNFROZEN, tiny=True)).tiny  # --tiny 不受此限


@pytest.mark.parametrize("task, split, setting", [
    ("refseg", "devtune", "full"), ("refseg", "devtune", "baseline_ownbox"), ("count", "devtune_countbench", "full")])
def test_devtune_runs_unfrozen(task, split, setting):
    assert RUN_W1.parse_args(_args(task, split, setting, *UNFROZEN)).split == split


def test_unreadable_config_is_a_usage_error(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text("{", encoding="utf-8")
    with pytest.raises(SystemExit) as ex:
        RUN_W1.parse_args(_args("refseg", "devcheck", "full", "--w1-config", str(bad)))
    assert ex.value.code == 2 and "读不了" in capsys.readouterr().err


def _cli(args, tmp: Path) -> subprocess.CompletedProcess:
    env = {**os.environ, "TMPDIR": str(tmp), "HF_HUB_OFFLINE": "1"}
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True, cwd=REPO, env=env, timeout=1800)


def test_frozen_guard_exit_code_from_the_command_line(tmp_path):
    out = tmp_path / "0999_w1_knobs"
    p = _cli(_args("refseg", "devcheck", "full", *UNFROZEN, out=str(out)), tmp_path)
    assert p.returncode == 2 and FROZEN_MSG in p.stderr and not out.exists()  # 用法错误：什么都不写


# ---------------------------------------------------------------------------
# 冻结后的仓库配置（D111：configs/w1.json = 调参臂 U2）
# ---------------------------------------------------------------------------

U2_PATH = REPO / "configs" / "w1_tune" / "U2.json"
ARM_ONLY_KEYS = ("name", "status", "frozen")  # 冻结时只改这三项


def test_repo_config_is_frozen():
    assert Path(CFG.path) == W1_CONFIG_PATH and CFG.data["frozen"] is True


def test_repo_config_is_tuning_arm_u2():
    """除 name、status、frozen 外与 U2 完全相同（含键的顺序）；status 里写的 U2 的 sha256 就是这份文件的。"""
    frozen, u2 = (json.loads(p.read_text(encoding="utf-8")) for p in (W1_CONFIG_PATH, U2_PATH))
    rest = [{k: v for k, v in d.items() if k not in ARM_ONLY_KEYS} for d in (frozen, u2)]
    assert list(frozen) == list(u2) and rest[0] == rest[1]
    assert json.dumps(rest[0], ensure_ascii=False) == json.dumps(rest[1], ensure_ascii=False)
    assert (frozen["frozen"], u2["frozen"]) == (True, False) and frozen["name"] != u2["name"]
    assert file_sha256(U2_PATH) in frozen["status"]


def test_repo_config_passes_the_frozen_guard():
    """默认 --w1-config（冻结后的仓库配置）：devcheck 与 test_* 不再被冻结保护拒绝（只到参数检查这一步）。"""
    for task, split, setting in GUARDED:
        assert RUN_W1.needs_frozen(split)
        a = RUN_W1.parse_args(_args(task, split, setting))
        assert (a.split, a.w1_config, a.tiny) == (split, str(W1_CONFIG_PATH), False)


# ---------------------------------------------------------------------------
# 命令行彩排：全部开关打开的临时配置；工作台选项进运行身份
# ---------------------------------------------------------------------------


def _jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def test_tiny_rehearsal_with_all_switches(tmp_path):
    d = _cfg()
    d["protocol"].update(final_round_tools="submit_only", capped_tools="submit_only")
    _set_status(d, "Turn {round} of {max_rounds} is done ({rounds_left} left); tool calls used: {calls_used} of {max_tool_calls} "
                "({calls_left} left).", list(STATUS_FIELDS))
    cfg_path = tmp_path / "w1_all_switches.json"
    cfg_path.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    out = tmp_path / "out"
    fakes = ["--adapter-factory", "tests._w1_fakes:make_persistent_adapter", "--workbench", "tests._w1_fakes:FakeWorkbench"]
    for task, split in (("refseg", "devcheck"), ("refexp_box", "test_refcocog_umd_test"), ("count", "test_countbench")):
        p = _cli(_args(task, split, "full", "--w1-config", str(cfg_path), *fakes, tiny=True, out=str(out)), tmp_path)
        assert p.returncode == 0, p.stderr[-3000:]
    refused, offered, status = set(), set(), 0
    for task, split in (("refseg", "devcheck"), ("refexp_box", "test_refcocog_umd_test"), ("count", "test_countbench")):
        leaf = out / "qwen3.5-0.8b" / task / "full" / split
        cfg = yaml.safe_load((leaf / "config.yaml").read_text())
        assert (cfg["protocol"]["final_round_tools"], cfg["protocol"]["capped_tools"]) == ("submit_only", "submit_only")
        assert cfg["prompts"]["config"] == str(cfg_path)  # --w1-config 的路径与 sha256 进运行身份
        assert cfg["prompts"]["config_sha256"] == hashlib.sha256(cfg_path.read_bytes()).hexdigest()
        assert cfg["prompts"]["status"] == d["prompts"]["messages"]["status"] and cfg["workbench"]["options"] == {"view_frames": False}
        for rec in _jsonl(leaf / "predictions.jsonl"):
            assert rec["failure"] is None and rec["ended_by"] == "submission"  # 假适配器只在只给提交工具时才提交
            for rd in rec["rounds"]:
                status += bool(rd.get("status_notice"))
                offered |= {(rd["round"], rd["tools_offered"])} if "tools_offered" in rd else set()
                refused |= {c["refused"] for c in rd["tool_calls"] if c["refused"]}
    assert {REFUSED_FINAL_ROUND, REFUSED_TOOL_CAP} <= refused and (6, "submit_only") in offered and (5, "submit_only") in offered
    assert status > 0


def test_tiny_rehearsal_passes_workbench_options(tmp_path):
    d = _cfg()
    d["workbench"]["view_frames"] = True
    cfg_path = tmp_path / "w1_view_frames.json"
    cfg_path.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    out = tmp_path / "out"
    fakes = ["--adapter-factory", "tests._w1_fakes:make_policy_adapter", "--workbench", "tests._w1_fakes:OptionsWorkbench"]
    for name, cfg_args in (("on", ["--w1-config", str(cfg_path)]), ("off", [])):
        p = _cli(_args("refseg", "devtune", "full", *cfg_args, *fakes, tiny=True, out=str(out / name)), tmp_path)
        assert p.returncode == 0, p.stderr[-3000:]
    on, off = (out / n / "qwen3.5-0.8b" / "refseg" / "full" / "devtune" for n in ("on", "off"))
    con, coff = (yaml.safe_load((x / "config.yaml").read_text()) for x in (on, off))
    assert con["workbench"]["options"] == {"view_frames": True} and coff["workbench"]["options"] == {"view_frames": False}
    # 运行身份的探测与回路里的工作台拿到同样的选项（否则回路的工具清单核对会报错）；选项进了每次调用的日志
    assert con["workbench"]["tools_sha256"] != coff["workbench"]["tools_sha256"]
    logs_on = [e for row in _jsonl(on / "logs" / "workbench.jsonl") for e in row["log"]]
    logs_off = [e for row in _jsonl(off / "logs" / "workbench.jsonl") for e in row["log"]]
    assert logs_on and all(e["options"] == {"view_frames": True} for e in logs_on)
    assert logs_off and all(e["options"] == {} for e in logs_off)
    assert all(r["tools_sha256"] == con["workbench"]["tools_sha256"] for r in _jsonl(on / "predictions.jsonl"))
