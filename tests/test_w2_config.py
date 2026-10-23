"""DEV-0059：configs/w2.json（src/w2/config.py）。配置里的数字与规则由这里钉住；改坏任一处都报错。

DEV-0062：configs/w2.json 会在 W2-a 调参之后冻结成选中的臂（scripts/mgr/w2_tune.py freeze）。断言草稿取值（frozen = false、思考开、box、
K = 8……）的读草稿夹具（tests/_w2_draft.W2_DRAFT）；其余（提示、划分、校验）对冻结前后的仓库配置都成立，照读仓库配置。末尾一节是与冻结
无关的仓库配置检查：草稿夹具钉住 sha256；未冻结时两份仓库配置与草稿逐字节相同；冻结后满足冻结的不变量（1 号 = grabcut_default、
K 在 2–8、思考预算 ≤ 2,048、status 写明 D 编号与臂的 sha256、configs/w2c_train.json 写死的 sha256 = configs/w2.json 的、冻结保护放行）。
"""

from __future__ import annotations

import copy
import importlib.util
import json
import re
from pathlib import Path

import pytest

from src.w2.config import (CANDIDATE_SETTINGS, MAX_THINKING_BUDGET, SETTING_TASKS, SETTINGS, W2_CONFIG_PATH, W2ConfigError,
                           file_sha256, load_w2_config, validate)
from src.w2.w2c_train import W2C_CONFIG_PATH, W2CConfigError, load_w2c_config, resolve_w2_config
from tests._w2_draft import W2_DRAFT, W2_DRAFT_PATH, W2_DRAFT_SHA256, W2C_TRAIN_DRAFT_PATH, W2C_TRAIN_DRAFT_SHA256

REPO = Path(__file__).resolve().parents[1]
CFG = load_w2_config()  # 仓库配置（冻结前是草稿，冻结后是选中的臂）


def test_unfrozen_draft_and_d113_numbers():
    D = W2_DRAFT  # 草稿夹具（冻结前的 configs/w2.json 原样）
    assert D.data["frozen"] is False  # 初稿：devtune800 上定 K 与生成器之后再冻结
    s = D.select_protocol()
    assert s.thinking is True and s.thinking_budget == 2048 == MAX_THINKING_BUDGET and s.answer_budget == 64
    assert "2,048" in D.data["basis"] and "两轮" in D.data["basis"] and "1 号候选" in D.data["basis"]
    assert D.propose_mode == "box" and D.generators[0] == {"type": "grabcut_default"} and D.k == len(D.generators) == 8
    assert D.layout == "grid" and isinstance(D.style, dict)
    assert D.random_salt == "w2_random_pick_v1"
    assert D.batching() == {"batch_size": 16, "chunk_size": 64, "max_batch_tokens": 48000}
    t = D.select_protocol(tiny=True)
    assert t.thinking == s.thinking and t.thinking_budget < s.thinking_budget and t.answer_budget < s.answer_budget
    assert D.batching(tiny=True)["batch_size"] == 2 and D.batching(tiny=True)["max_batch_tokens"] == 48000


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


# ---------------------------------------------------------------------------
# 仓库配置：与冻结无关的检查（DEV-0062）。冻结前后都成立；冻结由 scripts/mgr/w2_tune.py freeze 完成（只改 name、status、frozen，
# 并填 configs/w2c_train.json 的 w2_config.sha256）
# ---------------------------------------------------------------------------

W2C = load_w2c_config()  # 仓库的 configs/w2c_train.json
FROZEN = CFG.data["frozen"] is True
GUARDED = [("refseg", "devcheck2", "w2a"), ("refseg", "devcheck2", "w1f"), ("refexp_box", "devcheck2", "baseline_e1"),
           ("refseg", "test_refcoco_unc_testA", "w2a"), ("count", "test_countbench", "w2a")]
ARM_ONLY_KEYS = ("name", "status", "frozen")  # freeze 只改这三项


def _run_w2():
    spec = importlib.util.spec_from_file_location("_run_w2_for_freeze", REPO / "scripts" / "run_w2.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


RUN_W2 = _run_w2()


def _guard_args(task, split, setting, *extra):
    return ["--model", "qwen3.5-9b", "--task", task, "--split", split, "--setting", setting, "--out", "/nonexistent/0999_w2_freeze",
            "--data-root", "/nonexistent", *extra]


def test_draft_fixtures_are_pinned():
    """草稿夹具是冻结前的两份仓库配置原样（sha256 钉住）：未冻结、w2_config.sha256 是占位、指向 configs/w2.json。"""
    assert file_sha256(W2_DRAFT_PATH) == W2_DRAFT.sha256 == W2_DRAFT_SHA256
    assert file_sha256(W2C_TRAIN_DRAFT_PATH) == W2C_TRAIN_DRAFT_SHA256
    d = load_w2c_config(W2C_TRAIN_DRAFT_PATH)  # 结构合格
    assert W2_DRAFT.frozen is False and d.doc["w2_config"]["sha256"] is None and d.doc["w2_config"]["path"] == "configs/w2.json"


def test_repo_configs_before_freeze_are_the_drafts():
    """未冻结（frozen = false）：两份仓库配置与草稿夹具逐字节相同（冻结前改了草稿，就同步更新夹具与 tests/_w2_draft.py 的 sha256）；
    w2_config.sha256 还是占位，非 tiny 的 W2-c 训练拒跑。冻结后：两份都已离开草稿。"""
    if not FROZEN:
        assert CFG.data["frozen"] is False
        assert W2_CONFIG_PATH.read_bytes() == W2_DRAFT_PATH.read_bytes() and CFG.sha256 == W2_DRAFT_SHA256
        assert W2C_CONFIG_PATH.read_bytes() == W2C_TRAIN_DRAFT_PATH.read_bytes() and W2C.doc["w2_config"]["sha256"] is None
        with pytest.raises(W2CConfigError, match="占位"):
            resolve_w2_config(W2C, tiny=False)
    else:
        assert CFG.sha256 != W2_DRAFT_SHA256 and W2C.sha256 != W2C_TRAIN_DRAFT_SHA256


def test_repo_config_freeze_invariants():
    """冻结后（frozen = true）：1 号生成器是 grabcut_default、K 在 2–8；思考预算 ≤ 2,048；status 写明 D 编号与臂的配置 sha256（臂的文件
    在仓库里时，再核对它的 sha256 与内容：除 name、status、frozen 外与仓库配置相同）；configs/w2c_train.json 写死的 sha256 = configs/w2.json
    的 sha256，非 tiny 的 W2-c 训练放行。未冻结时 frozen 恰是 false（上一项核对与草稿相同）。"""
    if not FROZEN:
        assert CFG.data["frozen"] is False
        return
    assert CFG.generators[0] == {"type": "grabcut_default"} and 2 <= CFG.k <= 8
    assert CFG.data["select"]["thinking_budget"] <= MAX_THINKING_BUDGET == 2048
    status = CFG.data["status"]
    assert re.search(r"已冻结（D[0-9]+）", status), status
    m = re.search(r"（([^（），]+)，sha256 ([0-9a-f]{64})）", status)
    assert m, status
    arm_path, arm_sha = Path(m.group(1)), m.group(2)
    arm_path = arm_path if arm_path.is_absolute() else REPO / arm_path
    if arm_path.is_file():  # 臂的配置可能只在服务器的运行目录里；在仓库里就逐项核对
        assert file_sha256(arm_path) == arm_sha
        arm, frozen = (json.loads(p.read_text(encoding="utf-8")) for p in (arm_path, W2_CONFIG_PATH))
        assert list(arm) == list(frozen) and arm["frozen"] is False
        assert {k: v for k, v in arm.items() if k not in ARM_ONLY_KEYS} == {k: v for k, v in frozen.items() if k not in ARM_ONLY_KEYS}
    assert W2C.doc["w2_config"]["path"] == "configs/w2.json"
    assert W2C.doc["w2_config"]["sha256"] == file_sha256(W2_CONFIG_PATH) == CFG.sha256
    w2, info = resolve_w2_config(W2C, tiny=False)  # 写死的 sha256 对得上、已冻结：正式训练放行
    assert w2.frozen and info["frozen"] is True and info["pinned_sha256"] == CFG.sha256


@pytest.mark.parametrize("task, split, setting", GUARDED)
def test_repo_config_and_the_frozen_guard(task, split, setting, capsys):
    """非 tiny 的 run_w2.py 参数检查（默认 --w2-config = 仓库配置）：冻结后 devcheck2 与 test_* 放行；未冻结时退出码 2（冻结保护）。
    草稿夹具总是被拒。"""
    assert RUN_W2.needs_frozen(split)
    if FROZEN:
        a = RUN_W2.parse_args(_guard_args(task, split, setting))
        assert (a.split, a.w2_config, a.tiny) == (split, str(W2_CONFIG_PATH), False)
    else:
        with pytest.raises(SystemExit) as ex:
            RUN_W2.parse_args(_guard_args(task, split, setting))
        assert ex.value.code == 2 and "协议未冻结" in capsys.readouterr().err
    with pytest.raises(SystemExit) as ex:
        RUN_W2.parse_args(_guard_args(task, split, setting, "--w2-config", str(W2_DRAFT_PATH)))
    assert ex.value.code == 2 and "协议未冻结" in capsys.readouterr().err
