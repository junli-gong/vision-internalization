"""DEV-0059：configs/w2.json（src/w2/config.py）。配置里的数字与规则由这里钉住；改坏任一处都报错。"""

from __future__ import annotations

import copy
import json

import pytest

from src.w2.config import (CANDIDATE_SETTINGS, MAX_THINKING_BUDGET, SETTING_TASKS, SETTINGS, W2_CONFIG_PATH, W2ConfigError,
                           load_w2_config, validate)

CFG = load_w2_config()


def test_unfrozen_draft_and_d113_numbers():
    assert CFG.data["frozen"] is False  # 初稿：devtune800 上定 K 与生成器之后再冻结
    s = CFG.select_protocol()
    assert s.thinking is True and s.thinking_budget == 2048 == MAX_THINKING_BUDGET and s.answer_budget == 64
    assert "2,048" in CFG.data["basis"] and "两轮" in CFG.data["basis"] and "1 号候选" in CFG.data["basis"]
    assert CFG.propose_mode == "box" and CFG.generators[0] == {"type": "grabcut_default"} and CFG.k == len(CFG.generators) >= 2
    assert CFG.layout in ("grid", "separate") and isinstance(CFG.style, dict)
    assert CFG.random_salt == "w2_random_pick_v1"
    assert CFG.batching() == {"batch_size": 16, "chunk_size": 64, "max_batch_tokens": 48000}
    t = CFG.select_protocol(tiny=True)
    assert t.thinking == s.thinking and t.thinking_budget < s.thinking_budget and t.answer_budget < s.answer_budget
    assert CFG.batching(tiny=True)["batch_size"] == 2 and CFG.batching(tiny=True)["max_batch_tokens"] == 48000


def test_settings_and_routing():
    assert SETTINGS == ("w2a", "random_pick", "oracle_pick", "w1f", "ownbox", "baseline_e1")
    assert SETTING_TASKS["w2a"] == ("refseg", "refexp_box", "count") and SETTING_TASKS["baseline_e1"] == SETTING_TASKS["w2a"]
    assert all(SETTING_TASKS[s] == ("refseg",) for s in ("random_pick", "oracle_pick", "w1f", "ownbox"))
    assert CANDIDATE_SETTINGS == ("w2a", "random_pick", "oracle_pick", "w1f")


def test_select_prompt_is_task_agnostic():
    p = CFG.select_prompt(6, "the dog on the left")
    assert "6 numbered candidate" in p and "the dog on the left" in p and "1 to 6" in p and "none" in p
    low = CFG.data["select"]["prompt"].lower()
    for word in ("grabcut", "superpixel", "slic", "watershed", "edge", "boundary", "first,", "step", "pixel", "tight"):
        assert word not in low, word  # 只说有 K 个候选、问哪个最好地覆盖，不写解题方法
    assert CFG.data["select"]["placeholders"] == ["k", "expression"] and CFG.select_system is None
    bp = CFG.box_points_prompt("the cat")
    assert "the cat" in bp and "bbox_2d" in bp and "point_2d" in bp and "at most 3" in bp


def test_splits_rules_pinned():
    d2 = CFG.splits["devcheck2"]
    assert (d2["population"], d2["salt"], d2["purpose"], d2["n"], d2["exclude_images_of"]) == \
        ("refcoco_unc_val", "w2_dev_v1", "w2_devcheck", 500, "devtune800")
    assert CFG.splits["devtune800"]["w1_splits"] == ["devtune", "devcheck"]
    w1 = json.loads((W2_CONFIG_PATH.parent / "w1.json").read_text())
    assert CFG.splits["test"] == w1["splits"]["test"]
    assert CFG.data["baselines"]["e1_task"] == w1["baselines"]["e1_task"]


def _bad(mut):
    d = copy.deepcopy(CFG.data)
    mut(d)
    with pytest.raises(W2ConfigError):
        validate(d)


def test_validation_rejects_broken_configs():
    _bad(lambda d: d["select"].__setitem__("thinking_budget", 4096))  # D113：至多 2,048
    _bad(lambda d: d["select"].__setitem__("thinking", "yes"))
    _bad(lambda d: d["select"].__setitem__("prompt", "Which of {k} covers {expr}?"))
    _bad(lambda d: d["select"].__setitem__("placeholders", ["k"]))
    _bad(lambda d: d["select"]["parse"].__setitem__("none_words", ["None"]))
    _bad(lambda d: d["select"]["parse"].__setitem__("strip_chars", "x."))
    _bad(lambda d: d["candidates"].__setitem__("generators", [{"type": "box"}, {"type": "grabcut_default"}]))  # 1 号必须是 W1-F
    _bad(lambda d: d["candidates"].__setitem__("generators", [{"type": "grabcut_default"}]))
    _bad(lambda d: d["propose"].__setitem__("mode", "points"))
    _bad(lambda d: d["render"].__setitem__("layout", "mosaic"))
    _bad(lambda d: d["splits"]["devcheck2"].__setitem__("salt", "w1_dev_v1"))
    _bad(lambda d: d["splits"]["devcheck2"].__setitem__("n", 400))
    _bad(lambda d: d.__setitem__("frozen", "no"))
    _bad(lambda d: d["tiny"]["select"].__setitem__("max_rounds", 2))
    validate(copy.deepcopy(CFG.data))


def test_candidates_digest_tracks_the_generator_list():
    d = copy.deepcopy(CFG.data)
    d["candidates"]["generators"] = [{"type": "grabcut_default"}, {"type": "box", "x": 1}]
    from src.w2.config import W2Config

    other = W2Config(d, CFG.path, "x")
    assert other.candidates_digest() != CFG.candidates_digest() and other.k == 2
    d["candidates"]["status"] = "改了说明"
    assert W2Config(d, CFG.path, "y").candidates_digest() == other.candidates_digest()  # 只认清单本身
