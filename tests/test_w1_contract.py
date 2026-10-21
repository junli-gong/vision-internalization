"""DEV-0051：假实现与真实现的约定一致（docs/w1_plan.md §1、§2）。DEV-0049（src/adapters/chat.py、chat_turn_batch）与 DEV-0050
（src/w1/workbench.py）还没合入时整个跳过；合入之后自动生效，是集成彩排的第一道核对：

- ChatTurnResult、ToolCall、ToolSpec 的字段与假的一致（合入之后假的直接用真的数据结构）；适配器基类有 chat_turn_batch，
  tools / thinking_budget / answer_budget 是关键字参数，Qwen 有 chat_inputs（预检与拆子批，D104）；回路回填的对话过得了真的
  check_messages；
- 工作台：GROUPS 与假的相同、包含 configs/w1.json 各设置关掉的组；构造签名；specs() 里有三个提交工具；Submission 有 mask / box /
  count；W1-F 用的 grabcut_mask 与 GRABCUT_ITERATIONS 在；
- 真的小模型 + 假工作台（只要 DEV-0049）、真的小模型 + 真工作台（两个都要）各跑一遍 --tiny 的回路。
"""

from __future__ import annotations

import dataclasses
import importlib.util
import inspect
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests import _w1_fakes as F

REPO = Path(__file__).resolve().parents[1]
HAS_CHAT = importlib.util.find_spec("src.adapters.chat") is not None
HAS_WB = importlib.util.find_spec("src.w1.workbench") is not None
need_chat = pytest.mark.skipif(not HAS_CHAT, reason="DEV-0049（src/adapters/chat.py）还没合入")
need_wb = pytest.mark.skipif(not HAS_WB, reason="DEV-0050（src/w1/workbench.py）还没合入")


def _fields(cls) -> set[str]:
    return {f.name for f in dataclasses.fields(cls)}


@need_chat
def test_chat_dataclasses_match_fakes():
    from src.adapters import chat

    for name in ("ToolSpec", "ToolCall"):
        assert _fields(getattr(F, name)) <= _fields(getattr(chat, name)), name
    real = _fields(chat.ChatTurnResult)
    assert _fields(F.ChatTurnResult) - {"n_image_pixels"} <= real
    assert "n_image_pixels" in real, "ChatTurnResult 要有 n_image_pixels（DEV-0051 的提议；回路的入口断言要它）"


@need_chat
def test_adapter_has_chat_turn_batch_and_chat_inputs():
    from src.adapters.base import ModelAdapter
    from src.adapters.qwen35 import Qwen35Adapter

    sig = inspect.signature(ModelAdapter.chat_turn_batch)
    for p in ("tools", "thinking_budget", "answer_budget"):
        assert sig.parameters[p].kind is inspect.Parameter.KEYWORD_ONLY, p
    ci = inspect.signature(Qwen35Adapter.chat_inputs)  # 回路拼批前的预检与拆子批用它（D104）
    assert ci.parameters["tools"].kind is inspect.Parameter.KEYWORD_ONLY


@need_chat
def test_loop_conversations_pass_the_real_message_check():
    """回路回填出来的多轮对话（D104 的写法）过得了真适配器的 check_messages。"""
    from src.adapters import chat
    from tests.test_w1_loop import DRAW, SUBMIT_M, ZOOM, run

    ad, r = run({"a": [F.turn(text="Look.", calls=[ZOOM]), F.turn(errors=["bad"]), F.turn(calls=[DRAW, SUBMIT_M])]})
    assert r["a"].failure is None
    for batch in ad.conversations:
        for conv in batch:
            chat.check_messages(conv, check_text=lambda s: None)


@need_wb
def test_workbench_matches_fake_and_config():
    from src.w1 import workbench as WB
    from src.w1.config import LOOP_SETTINGS, load_w1_config

    assert tuple(WB.GROUPS) == F.GROUPS
    cfg = load_w1_config()
    for s in LOOP_SETTINGS:
        assert set(cfg.data["settings"][s]) <= set(WB.GROUPS), s
    sig = inspect.signature(WB.Workbench)
    assert list(sig.parameters)[0] == "image" and {"disabled_groups", "sandbox"} <= set(sig.parameters)
    assert {"text", "images", "error"} <= _fields(WB.ToolResult)
    from PIL import Image

    names = {s.name for s in WB.Workbench(Image.new("RGB", (64, 48))).specs()}
    assert set(cfg.data["protocol"]["submit_tools"]) <= names
    sub = WB.Submission(kind="mask", mask=None)
    assert {"mask", "box", "count"} <= {f.name for f in dataclasses.fields(sub)}  # 回路按这三个字段读答案
    assert callable(WB.grabcut_mask) and isinstance(WB.GRABCUT_ITERATIONS, int)  # W1-F 用的同一个函数（D106）


def _run_tiny(tmp: Path, *extra: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "TMPDIR": str(tmp), "HF_HUB_OFFLINE": "1"}
    args = ["--tiny", "--model", "qwen3.5-0.8b", "--task", "refseg", "--split", "devcheck", "--setting", "full", "--out",
            str(tmp / "out"), *extra]
    return subprocess.run([sys.executable, str(REPO / "scripts" / "run_w1.py"), *args], capture_output=True, text=True, cwd=REPO,
                          env=env, timeout=3600)


@need_chat
def test_tiny_real_adapter_with_fake_workbench(tmp_path):
    p = _run_tiny(tmp_path, "--workbench", "tests._w1_fakes:FakeWorkbench")
    assert p.returncode == 0, p.stderr[-3000:]


@need_chat
@need_wb
def test_tiny_real_adapter_with_real_workbench(tmp_path):
    p = _run_tiny(tmp_path)
    assert p.returncode == 0, p.stderr[-3000:]
