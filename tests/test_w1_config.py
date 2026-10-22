"""DEV-0051：configs/w1.json（src/w1/config.py）。配置里的数字由这里钉住；改坏任一处都报错；按设置给出的协议与提示。"""

from __future__ import annotations

import copy
import json

import pytest

from src.w1.config import LOOP_SETTINGS, W1_CONFIG_PATH, W1ConfigError, load_w1_config, validate
from src.w1.loop import TASKS

CFG = load_w1_config()


def test_protocol_numbers_pinned_to_d103():
    pr = CFG.data["protocol"]
    assert (pr["max_rounds"], pr["max_tool_calls"], pr["thinking_budget"], pr["answer_budget"]) == (6, 8, 2048, 1024)
    p = CFG.protocol("full")
    assert (p.max_rounds, p.max_tool_calls, p.thinking_budget, p.answer_budget) == (6, 8, 2048, 1024)
    assert p.disabled_groups == frozenset() and p.submit_tools == {"submit_mask", "submit_box", "submit_count"}
    assert p.submit_exempt_from_cap is True
    assert "6 轮" in CFG.data["basis"] and "8 次" in CFG.data["basis"] and "2,048" in CFG.data["basis"]
    assert CFG.data["frozen"] is True  # D111：开发-调参集上选定 U2 后冻结（冻结前是 False）


def test_settings_map_to_ablation_groups():
    got = {s: CFG.protocol(s).disabled_groups for s in LOOP_SETTINGS}
    assert got == {"full": frozenset(), "no_eye": {"eye"}, "no_check": {"check"}, "no_snap": {"snap", "code"}}
    with pytest.raises(ValueError):
        CFG.protocol("baseline_e1")


def test_tiny_only_shrinks_budgets_and_batches():
    t, f = CFG.protocol("no_eye", tiny=True), CFG.protocol("no_eye")
    assert (t.max_rounds, t.max_tool_calls, t.disabled_groups) == (f.max_rounds, f.max_tool_calls, f.disabled_groups)
    assert t.thinking_budget < f.thinking_budget and t.answer_budget < f.answer_budget
    assert CFG.batching() == {"batch_size": 16, "chunk_size": 64, "max_batch_tokens": 48000, "tool_workers": 1}
    assert CFG.batching(tiny=True)["batch_size"] == 2 and CFG.batching(tiny=True)["max_batch_tokens"] == 48000
    assert CFG.data["protocol"]["answer_budget"] == 1024  # D104：先保留 1024


def test_prompts_are_task_agnostic_and_render():
    p = CFG.loop_prompts(CFG.protocol("full"))
    low = p.system.lower()
    for word in ("segment", "locate", "how many", "mask", "box", "number", "refseg", "count:"):
        assert word not in low, word  # 系统提示不写任何任务（也不写任务的解法）
    assert "0 to 1000" in p.system and "x first" in p.system and "original image" in p.system
    assert set(p.tasks) == set(TASKS)
    assert "{expression}" in p.tasks["refseg"] and "submit_mask" in p.tasks["refseg"]
    assert "JSON" in p.tasks["refexp_box"] and "{question}" in p.tasks["count"]
    assert p.tool_cap.startswith("Tool call limit reached (8 calls)")
    assert "{error}" in p.tool_call_error and p.last_round
    assert CFG.data["prompts"]["status"].startswith("初稿，开发-调参集上待定")


def test_splits_and_baselines_section():
    sp = CFG.splits
    assert (sp["salt"], sp["purpose"], sp["population"], sp["devtune_n"], sp["devcheck_n"]) == \
        ("w1_dev_v1", "w1_dev", "poly:refcoco_unc_val", 300, 500)
    assert sp["test"] == ["refcoco_unc_testA", "refcoco_unc_testB", "refcoco+_unc_testA", "refcoco+_unc_testB", "refcocog_umd_test"]
    assert {t: CFG.e1_task(t) for t in TASKS} == {"refseg": "refexp_poly", "refexp_box": "refexp_box", "count": "count"}
    assert CFG.baselines["w1f"]["function"] == "src.w1.workbench.grabcut_mask"  # W1-F 用工作台的同一个函数（D106）


@pytest.mark.parametrize("mutate", [
    lambda d: d["protocol"].__setitem__("max_rounds", 0),
    lambda d: d["batching"].__setitem__("max_batch_tokens", 0),
    lambda d: d["protocol"].__setitem__("submit_tools", ["finish"]),
    lambda d: d["settings"].__setitem__("full", ["eye"]),
    lambda d: d["settings"].__setitem__("no_eye", ["submit"]),
    lambda d: d["settings"].pop("no_check"),
    lambda d: d["prompts"].__setitem__("system", "Use {tool} wisely."),
    lambda d: d["prompts"]["tasks"].__setitem__("refseg", "Segment it."),  # 少了 {expression}
    lambda d: d["prompts"]["tasks"].pop("count"),
    lambda d: d["splits"].__setitem__("devcheck_n", 0),
    lambda d: d["baselines"]["e1_task"].__setitem__("refseg", "grid16"),
    lambda d: d["baselines"]["poly_structured"].__setitem__("variant", "poly_structured_min8"),
    lambda d: d["baselines"]["w1f"].__setitem__("function", "cv2.grabCut"),
    lambda d: d["baselines"]["w1f"].__setitem__("iterations", 5),
    lambda d: d["tiny"]["protocol"].__setitem__("max_rounds", 2),
])
def test_validation_catches_mutations(mutate):
    d = copy.deepcopy(json.loads(W1_CONFIG_PATH.read_text(encoding="utf-8")))
    validate(d)
    mutate(d)
    with pytest.raises(W1ConfigError):
        validate(d)
