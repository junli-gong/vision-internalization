"""DEV-0055：每轮的状态提示（prompts.messages.status）与"最后一轮"提示（prompts.messages.last_round）走哪种消息
（configs/w1.json 的 prompts.status_role、prompts.last_round_role；src/w1/loop.py、config.py、runner.py）。

- tool（默认）：追加在这一轮最后一条 tool 消息的末尾（现状，记录逐字节不变，不加 notice_message）。
- user：这一轮的工具结果全部追加之后，再追加一条只有文字的 user 消息；同一轮至多一条：两种都走 user 时 status 在前、last_round 在后，
  中间空一行；一个走 tool、一个走 user 时各走各的；有 user 提示的那一轮记 notice_message = user。
- 每种取值组合；与 final_round_tools、capped_tools 一起开；样本结束之后不追加；只跟在 tool 消息之后。
- 配置：默认两个键都是 tool；不给时取 tool（DEV-0055 之前生成的配置照旧可读）；取值校验；进运行身份。
- 命令行彩排：全部开关打开（两个都走 user）的临时配置，假适配器（开关都被走到）与真的小模型各跑一遍，rc = 0。
（真的 Qwen3.5 对话模板怎样渲染 tool → user、前几轮的思考段是否保留：tests/test_adapter_chat.py 的 DEV-0055 一节。）
- DEV-0056（D111）：configs/w1.json 冻结后 last_round 走 user；测默认（两种都走 tool）的测试读冻结前的草稿夹具
  （tests.test_w1_loop.DRAFT）。
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
import yaml

from src.w1.config import NOTICE_ROLE_KEYS, W1ConfigError, load_w1_config, validate
from src.w1.loop import NOTICE_ROLES, NOTICE_SEPARATOR, STATUS_FIELDS, LoopPrompts, Protocol, _Episode, run_episodes
from tests._w1_fakes import FakeWorkbench, ScriptedChatAdapter, turn
from tests.test_w1_knobs import REFUSAL, STATUS, SUBMIT_ORDER, UNFROZEN, _args, _cfg, _cli, _jsonl, expected_status, texts
from tests.test_w1_loop import DRAFT, DRAFT_PATH, DRAFT_SHA256, DRAW, SUBMIT_M, ZOOM, image, protocol, sample

REPO = Path(__file__).resolve().parents[1]
CFG = load_w1_config()
LAST = CFG.loop_prompts(protocol()).last_round  # configs/w1.json 的 last_round 文字
COMBOS = [(s, lr) for s in NOTICE_ROLES for lr in NOTICE_ROLES]
_KEEP = object()


def lp(p: Protocol, *, status=STATUS, status_role="tool", last_round_role="tool", last_round=_KEEP, refusal=REFUSAL,
       cfg=CFG) -> LoopPrompts:
    kw = {} if last_round is _KEEP else {"last_round": last_round}
    return dataclasses.replace(cfg.loop_prompts(p), status=status, final_round_refused=refusal, status_role=status_role,
                               last_round_role=last_round_role, **kw)


def go(script, p, pr, *, task="refseg", batch_size=8):
    keys = list(script)
    ad = ScriptedChatAdapter({f"the thing {k}": v for k, v in script.items()})
    ss = [sample(k, task) for k in keys]
    recs = run_episodes(ad, ss, task, p, pr, batch_size=batch_size, workbench_factory=FakeWorkbench,
                        images=[image(i) for i in range(len(ss))])
    return ad, {k: r for k, r in zip(keys, recs)}


def walk(script_turns, p, pr):
    """一条样本照 run_episodes 的步骤逐轮走（同 test_w1_knobs.episode_messages），最后一轮之后的历史也能查。"""
    ad = ScriptedChatAdapter({"the thing a": script_turns})
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


def user_msg(text: str) -> dict:
    return {"role": "user", "content": [{"type": "text", "text": text}]}


THREE = turn(calls=[ZOOM, ZOOM, ZOOM])
# 6 轮：第 3 轮只有解析失败的调用（这一轮最后一条 tool 消息是解析失败的回复）；第 6 轮提交
SCRIPT = [turn(calls=[DRAW, ZOOM, ZOOM]), THREE, turn(errors=["bad xml"]), THREE, turn(calls=[ZOOM]), turn(calls=[SUBMIT_M])]
USED = [3, 6, 6, 8, 8]  # 第 1–5 轮结束时计数的调用


# ---------------------------------------------------------------------------
# 每种取值组合
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status_role, last_round_role", COMBOS)
def test_each_combination_places_notices(status_role, last_round_role):
    p = protocol()
    ad, r = go({"a": SCRIPT}, p, lp(p, status_role=status_role, last_round_role=last_round_role))
    a = r["a"]
    assert (a.ended_by, a.failure, a.n_rounds) == ("submission", None, 6)
    assert len(ad.conversations[0][0]) == 2  # 第 1 轮：system + user
    for k in range(1, 6):  # 第 k+1 轮的对话：末尾是第 k 轮的工具结果与提示
        conv = ad.conversations[k][0]
        status = expected_status(k, USED[k - 1])
        to_user = ([status] if status_role == "user" else []) + ([LAST] if k == 5 and last_round_role == "user" else [])
        to_tool = ([status] if status_role == "tool" else []) + ([LAST] if k == 5 and last_round_role == "tool" else [])
        if to_user:  # 只有一条 user 消息、只有文字；status 在前、last_round 在后，中间空一行
            assert conv[-1] == user_msg(NOTICE_SEPARATOR.join(to_user))
            tool = conv[-2]
        else:
            tool = conv[-1]
        assert tool["role"] == "tool"
        tail = texts(tool)
        assert tail[len(tail) - len(to_tool):] == to_tool
        assert not to_tool or len(tail) > len(to_tool)  # 提示追加在工具结果之后，不替换它
        if not to_tool:
            assert tail[-1] not in (status, LAST)
        # 每种提示只出现一次（不会既在 tool 里又在 user 里）
        everywhere = [t for m in conv for t in texts(m)]
        assert sum(status in t for t in everywhere) == 1
        assert sum(LAST in t for t in everywhere) == (1 if k == 5 else 0)
        # 对话里的 user 消息：任务 + 之前每一轮（含这一轮）各至多一条提示
        n_user = 1 + sum(1 for j in range(1, k + 1) if status_role == "user" or (j == 5 and last_round_role == "user"))
        assert sum(m["role"] == "user" for m in conv) == n_user
        # user 提示紧跟在这一轮的工具结果之后：它前面一条是 tool，后面（下一轮的回填）是 assistant
        for i, m in enumerate(conv[2:], start=2):
            if m["role"] == "user":
                assert conv[i - 1]["role"] == "tool" and (i + 1 == len(conv) or conv[i + 1]["role"] == "assistant")
    want_user = [status_role == "user" or (k == 5 and last_round_role == "user") for k in range(1, 7)]
    want_user[5] = False  # 第 6 轮之后样本结束
    assert [rd.get("notice_message") for rd in a.rounds] == ["user" if w else None for w in want_user]
    assert [rd.get("status_notice") for rd in a.rounds] == [True] * 5 + [None]
    assert [rd.get("last_round_notice") for rd in a.rounds] == [None] * 4 + [True, None]


def test_default_roles_are_tool_and_records_unchanged():
    """不给 status_role / last_round_role 与显式给 tool 完全相同；默认时不加 notice_message，对话里没有提示用的 user 消息。
    默认取冻结前的草稿（冻结后的仓库配置 last_round 走 user）。"""
    p = protocol()
    base = dataclasses.replace(DRAFT.loop_prompts(p), status=STATUS)
    assert (base.status_role, base.last_round_role) == ("tool", "tool")
    ad0, r0 = go({"a": SCRIPT, "b": [turn(calls=[ZOOM])] * 6}, p, base)
    ad1, r1 = go({"a": SCRIPT, "b": [turn(calls=[ZOOM])] * 6}, p, lp(p, cfg=DRAFT))
    assert [json.dumps(r.to_json()) for r in r0.values()] == [json.dumps(r.to_json()) for r in r1.values()]
    assert all("notice_message" not in rd for r in r0.values() for rd in r.rounds)
    assert all(sum(m["role"] == "user" for m in conv) == 1 for batch in ad0.conversations for conv in batch)


def test_only_last_round_as_user():
    p = protocol()
    ad, r = go({"a": [turn(calls=[ZOOM])] * 5 + [turn(calls=[SUBMIT_M])]}, p, lp(p, status=None, last_round_role="user"))
    for k in range(1, 5):
        assert ad.conversations[k][0][-1]["role"] == "tool"
    assert ad.conversations[5][0][-1] == user_msg(LAST) and ad.conversations[5][0][-2]["role"] == "tool"
    assert [rd.get("notice_message") for rd in r["a"].rounds] == [None] * 4 + ["user", None]
    assert "status_notice" not in r["a"].rounds[0]


def test_only_status_as_user_without_last_round():
    p = protocol()
    ad, r = go({"a": SCRIPT}, p, lp(p, status_role="user", last_round_role="user", last_round=None))
    assert ad.conversations[5][0][-1] == user_msg(expected_status(5, 8))  # 没有 last_round：只有状态
    assert all(rd.get("last_round_notice") is None for rd in r["a"].rounds)


def test_mixed_roles_keep_the_tool_notice_on_the_tool_message():
    """status 走 user、last_round 走 tool：last_round 照旧追加在上一轮最后一条 tool 消息上（它后面已经有这一轮的 user 提示）。"""
    p = protocol(max_rounds=3)
    e, ad = walk([turn(calls=[DRAW]), turn(calls=[ZOOM]), turn(calls=[SUBMIT_M])], p, lp(p, status_role="user"))
    conv = ad.conversations[2][0]
    assert conv[-1] == user_msg(expected_status(2, 2, max_rounds=3))
    assert conv[-2]["role"] == "tool" and texts(conv[-2])[-1] == LAST
    assert e.rec.rounds[1]["notice_message"] == "user" and e.rec.failure is None


def test_user_notice_not_added_after_the_episode_ends():
    p = protocol()
    pr = lp(p, status_role="user", last_round_role="user")
    _, r = go({"a": [turn(calls=[DRAW]), turn(calls=[DRAW, SUBMIT_M])], "b": [turn(text="It is there.")]}, p, pr)
    assert [rd.get("notice_message") for rd in r["a"].rounds] == ["user", None]
    assert "notice_message" not in r["b"].rounds[0] and "status_notice" not in r["b"].rounds[0]


def test_user_notice_requires_tool_results():
    p = protocol()
    e, _ = walk([turn(text="done")], p, lp(p, status_role="user"))
    e.messages.append({"role": "assistant", "content": [{"type": "text", "text": "x"}], "reasoning_content": ""})
    with pytest.raises(RuntimeError, match="tool 消息"):
        e.add_status_notice(1)
    with pytest.raises(RuntimeError, match="tool 消息"):
        e.add_last_round_notice("last")  # last_round 走 tool 时同样要求这一轮最后一条是 tool 消息


# ---------------------------------------------------------------------------
# 与 final_round_tools、capped_tools 一起开
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status_role, last_round_role", COMBOS)
def test_with_final_round_and_capped_switches(status_role, last_round_role):
    p = protocol(final_round_tools="submit_only", capped_tools="submit_only", max_rounds=4)
    script = [turn(calls=[DRAW, ZOOM, ZOOM]), THREE, THREE, turn(calls=[ZOOM, SUBMIT_M])]  # 第 3 轮计满 8 次（第 3 个被拒）
    e, ad = walk(script, p, lp(p, status_role=status_role, last_round_role=last_round_role))
    rec = e.rec
    assert ad.tools_seen[-1] == [SUBMIT_ORDER]  # 最后一轮只给提交工具（满额也只给提交工具）
    assert [c["refused"] for c in rec.rounds[-1]["tool_calls"]] == ["final_round", None]
    assert (rec.ended_by, rec.failure, rec.n_tool_calls) == ("submission", None, 8)
    conv = ad.conversations[3][0]
    status = expected_status(3, 8, max_rounds=4)
    to_user = ([status] if status_role == "user" else []) + ([LAST] if last_round_role == "user" else [])
    if to_user:
        assert conv[-1] == user_msg(NOTICE_SEPARATOR.join(to_user))
    else:
        assert conv[-1]["role"] == "tool" and texts(conv[-1])[-2:] == [status, LAST]
    # 最后一轮之后的历史：user 提示 → assistant（第 4 轮）→ 拒绝的回复 → 提交的结果
    tail = e.messages[-4:] if to_user else e.messages[-3:]
    assert [m["role"] for m in tail] == (["user"] if to_user else []) + ["assistant", "tool", "tool"]
    assert texts(e.messages[-2]) == [REFUSAL]
    assert rec.rounds[2].get("tools_offered") is None and rec.rounds[3]["tools_offered"] == "submit_only"
    assert rec.rounds[2].get("notice_message") == ("user" if to_user else None)


# ---------------------------------------------------------------------------
# 配置与运行身份
# ---------------------------------------------------------------------------


def test_config_defaults_and_roles():
    pr = DRAFT.data["prompts"]
    assert (pr["status_role"], pr["last_round_role"]) == ("tool", "tool")  # 写在冻结前的 configs/w1.json（草稿夹具）里
    assert dict(NOTICE_ROLE_KEYS) == {"status_role": "tool", "last_round_role": "tool"} and NOTICE_ROLES == ("tool", "user")
    assert DRAFT.notice_roles() == {"status_role": "tool", "last_round_role": "tool"}
    lpr = DRAFT.loop_prompts(DRAFT.protocol("full"))
    assert (lpr.status_role, lpr.last_round_role) == ("tool", "tool")
    for name in ("status_role", "last_round_role"):
        with pytest.raises(ValueError, match=name):
            dataclasses.replace(lpr, **{name: "system"})


@pytest.mark.parametrize("key", ["status_role", "last_round_role"])
@pytest.mark.parametrize("bad", ["assistant", "User", None, 1])
def test_validation_rejects_bad_roles(key, bad):
    d = _cfg()
    validate(d)
    d["prompts"][key] = bad
    with pytest.raises(W1ConfigError, match=key):
        validate(d)


def test_missing_role_keys_default_to_tool(tmp_path):
    """DEV-0055 之前生成的配置（例如 configs/w1_tune/T1–T4）没有这两个键：照旧可读，取 tool。"""
    d = _cfg()
    del d["prompts"]["status_role"], d["prompts"]["last_round_role"]
    path = tmp_path / "old_style.json"
    path.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    c = load_w1_config(path)
    assert c.notice_roles() == {"status_role": "tool", "last_round_role": "tool"}
    for arm in sorted((REPO / "configs" / "w1_tune").glob("T*.json")):
        assert load_w1_config(arm).notice_roles() == {"status_role": "tool", "last_round_role": "tool"}, arm.name


def test_user_roles_render(tmp_path):
    d = _cfg()
    d["prompts"].update(status_role="user", last_round_role="user")
    path = tmp_path / "user.json"
    path.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    c = load_w1_config(path)
    lpr = c.loop_prompts(c.protocol("full"))
    assert (lpr.status_role, lpr.last_round_role) == ("user", "user") and lpr.last_round == LAST


# ---------------------------------------------------------------------------
# 命令行彩排：全部开关打开（两种提示都走 user）
# ---------------------------------------------------------------------------


def _all_switches(tmp_path: Path) -> tuple[Path, dict]:
    d = _cfg()
    d["protocol"].update(final_round_tools="submit_only", capped_tools="submit_only")
    d["prompts"]["messages"]["status"] = ("Turn {round} of {max_rounds} is done ({rounds_left} left); tool calls used: {calls_used} of "
                                          "{max_tool_calls} ({calls_left} left).")
    d["prompts"]["placeholders"]["status"] = list(STATUS_FIELDS)
    d["prompts"].update(status_role="user", last_round_role="user")
    path = tmp_path / "w1_all_switches_user.json"
    path.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    return path, d


def test_tiny_rehearsal_all_switches_user_notices(tmp_path):
    cfg_path, d = _all_switches(tmp_path)
    out = tmp_path / "out"
    fakes = ["--adapter-factory", "tests._w1_fakes:make_persistent_adapter", "--workbench", "tests._w1_fakes:FakeWorkbench"]
    runs = (("refseg", "devcheck"), ("refexp_box", "test_refcocog_umd_test"), ("count", "test_countbench"))
    for task, split in runs:
        p = _cli(_args(task, split, "full", "--w1-config", str(cfg_path), *fakes, tiny=True, out=str(out)), tmp_path)
        assert p.returncode == 0, p.stderr[-3000:]
    n_user, n_status, last = 0, 0, 0
    for task, split in runs:
        leaf = out / "qwen3.5-0.8b" / task / "full" / split
        cfg = yaml.safe_load((leaf / "config.yaml").read_text())
        assert (cfg["prompts"]["status_role"], cfg["prompts"]["last_round_role"]) == ("user", "user")  # 进运行身份
        assert (cfg["protocol"]["final_round_tools"], cfg["protocol"]["capped_tools"]) == ("submit_only", "submit_only")
        assert "status → user" in (leaf / "README.md").read_text() and "last_round → user" in (leaf / "README.md").read_text()
        for rec in _jsonl(leaf / "predictions.jsonl"):
            assert rec["failure"] is None and rec["ended_by"] == "submission"
            for rd in rec["rounds"]:
                n_user += rd.get("notice_message") == "user"
                n_status += bool(rd.get("status_notice"))
                last += bool(rd.get("last_round_notice"))
                assert ("notice_message" in rd) == bool(rd.get("status_notice") or rd.get("last_round_notice"))
    assert n_user > 0 and n_user == n_status and last > 0


def test_tiny_rehearsal_all_switches_real_tiny_model(tmp_path):
    """真的小模型（真 chat_turn_batch）+ 真工作台，全部开关打开（两种提示都走 user）：rc = 0，运行身份记下两个开关。"""
    cfg_path, _ = _all_switches(tmp_path)
    out = tmp_path / "out"
    p = _cli(_args("refseg", "devtune", "full", "--w1-config", str(cfg_path), tiny=True, out=str(out)), tmp_path)
    assert p.returncode == 0, p.stderr[-3000:]
    leaf = out / "qwen3.5-0.8b" / "refseg" / "full" / "devtune"
    cfg = yaml.safe_load((leaf / "config.yaml").read_text())
    assert (cfg["prompts"]["status_role"], cfg["prompts"]["last_round_role"]) == ("user", "user")
    assert _jsonl(leaf / "predictions.jsonl")


def test_default_identity_records_tool_roles(tmp_path):
    """冻结前的草稿（两种提示都走 tool）：运行身份记 tool、tool，记录里没有 notice_message。"""
    out = tmp_path / "out"
    fakes = ["--adapter-factory", "tests._w1_fakes:make_persistent_adapter", "--workbench", "tests._w1_fakes:FakeWorkbench"]
    p = _cli(_args("refseg", "devtune", "full", *UNFROZEN, *fakes, tiny=True, out=str(out)), tmp_path)
    assert p.returncode == 0, p.stderr[-3000:]
    leaf = out / "qwen3.5-0.8b" / "refseg" / "full" / "devtune"
    cfg = yaml.safe_load((leaf / "config.yaml").read_text())
    assert (cfg["prompts"]["status_role"], cfg["prompts"]["last_round_role"]) == ("tool", "tool")
    recs = _jsonl(leaf / "predictions.jsonl")
    assert any(rd.get("last_round_notice") for r in recs for rd in r["rounds"])  # 默认的 last_round 照旧走 tool
    assert all("notice_message" not in rd for r in recs for rd in r["rounds"])
    assert REPO / cfg["prompts"]["config"] == DRAFT_PATH  # 仓库内的路径记成相对仓库根
    assert (cfg["prompts"]["config_sha256"], cfg["prompts"]["frozen"]) == (DRAFT_SHA256, False)
