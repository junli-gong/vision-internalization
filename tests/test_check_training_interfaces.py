"""DEV-0021 第 1 条：scripts/check_training_interfaces.py（受限 logits 一致率、LoRA 合并一致率；评测前的合并核对仪器）。

- 彩排（--tiny，CPU fp32，tests/fixtures/e1 的小数据）：三个家族各跑一遍，两项检查都 pass（CPU fp32 的判据是逐个相同），
  随机 LoRA 的前提（输出明显偏离 base）也满足；输出 JSON 的字段齐全。
- --checkpoint：评测前的标准核对（load_trained(merge=False) 对比 load_trained(merge=True)），带查询 token 的检查点也能跑。
- 仪器的灵敏度：合并时把 LoRA 丢掉（unload 而不 merge）、受限 logits 偏向某个 token，都判不过。
- 判据的逻辑、allowed 集合、批配置读配置（D43：64，不写死）、钉住的常量。
- DEV-0032（D56-5、D65-4(a)）：带网格 token 的检查点，restricted 用检查点自己的网格 token 与 E3b 单轮评测的设置（满长 1057 个
  token 的受限解码），另比较合并前后的受限 logits（across_merge）；坏检查点（合并前后网格 token 的 logits 分家）判不过，而合并后
  的模型内部自洽（by_thinking 照样通过）。不带网格 token 的检查点与 base：输出与 69884b8 的脚本（tests/fixtures/cti 存档）逐字节相同。
  过拟合小检查点（DEV-0025 的流程）的 restricted 通过，见 tests/test_e3_eval.py。
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from src.adapters import checkpoint as CK
from src.adapters import training
from src.adapters.factory import make_adapter
from src.e1.prompts import load_prompt_config

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts/check_training_interfaces.py"
FAMILY_KEYS = ("qwen3.5-0.8b", "gemma-4-e2b-it", "internvl3.5-2b-hf")
FIXTURE_N = 3  # tests/fixtures/e1 的三个 RefCOCO 开发子集各 1 句


def _load(path: Path, name: str):
    saved = list(sys.path)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.path[:] = saved
    return mod


@pytest.fixture(scope="module")
def cti():
    return _load(SCRIPT, "_check_training_interfaces_under_test")


def _run(cti, tmp_path, *args):
    out = tmp_path / "check.json"
    code = cti.main(["--tiny", "--out", str(out), *args])
    return code, json.loads(out.read_text())


# ---------------------------------------------------------------------------
# 常量与命令行
# ---------------------------------------------------------------------------


def test_pinned_constants(cti):
    run_e1 = _load(REPO_ROOT / "scripts/run_e1.py", "_run_e1_for_cti")
    assert cti.TINY_INIT_STD == run_e1.TINY_INIT_STD  # 注释说"与 run_e1 相同"
    assert (cti.MAX_NEW_TOKENS, cti.THINKING, cti.DEFAULT_N) == (128, (False, True), 200)  # 任务书
    assert (cti.LORA_RANK, cti.LORA_ALPHA) == (32, 64)  # 简报 §8.4 的 rank；env_report §7 的 LoraConfig
    # 判据 v1（D50-3）
    assert cti.CRITERIA_VERSION == "v1"
    assert (cti.RESTRICTED_MIN_TF_AGREEMENT, cti.RESTRICTED_MIN_EXACT_ROWS_OFF, cti.RESTRICTED_MAX_GAP) == (0.999, 0.95, 2.0)
    assert (cti.MERGE_MIN_TF_AGREEMENT, cti.MERGE_MAX_MEDIAN_GAP) == (0.98, 0.5)
    assert (cti.EFFECT_MAX_EXACT_ROWS, cti.PREMISE_MODES) == (0.5, ("thinking_on",))
    assert not hasattr(cti, "MIN_AGREEMENT") and not hasattr(cti, "EFFECT_MAX_ARGMAX_AGREEMENT")  # 提议值（D42-4）已撤掉
    assert (cti.TASK, cti.SUBSET) == ("grid16", "dev")


def test_parse_args(cti):
    a = cti.parse_args(["--tiny", "--model", "qwen3.5-0.8b", "--out", "x.json"])
    assert (a.device, a.dtype, a.batch_size, a.checks, a.n, a.max_new_tokens) == (
        "cpu", "fp32", cti.TINY_BATCH_SIZE, ["restricted", "merge"], 200, 128)
    a = cti.parse_args(["--model", "qwen3.5-9b", "--data-root", "/d", "--out", "x.json", "--checks", "merge", "restricted"])
    assert (a.device, a.dtype, a.batch_size, a.checks) == ("cuda", "bf16", None, ["restricted", "merge"])
    for bad in (["--model", "qwen3.5-0.8b", "--out", "x.json"],  # 真模型缺 --data-root
                ["--tiny", "--model", "nope", "--out", "x.json"],
                ["--tiny", "--model", "qwen3.5-0.8b", "--out", "x.json", "--n", "0"],
                ["--tiny", "--model", "qwen3.5-0.8b", "--out", "x.json", "--lora-rel-scale", "0"],
                ["--tiny", "--model", "qwen3.5-0.8b", "--out", "x.json", "--checkpoint", "/no/such/dir"]):
        with pytest.raises(SystemExit):
            cti.parse_args(bad)


def test_batch_size_read_from_config(cti, monkeypatch):
    """批大小读 configs/e1_prompts.json 冻结的 batch_protocol（D43 起 64），不写死；真模型换值要显式放开。"""
    monkeypatch.delenv("VISINT_DATA_ROOT", raising=False)
    prompts = load_prompt_config()
    frozen = prompts.batch_protocol()["batch_size"]
    assert frozen == 64  # D43
    real = cti.parse_args(["--model", "qwen3.5-0.8b", "--data-root", "/d", "--out", "x.json"])
    assert cti.resolve_batch_size(real, prompts) == (frozen, True)
    other = cti.parse_args(["--model", "qwen3.5-0.8b", "--data-root", "/d", "--out", "x.json", "--batch-size", "32"])
    with pytest.raises(ValueError, match="冻结"):
        cti.resolve_batch_size(other, prompts)
    other.allow_unfrozen_batch = True
    assert cti.resolve_batch_size(other, prompts) == (32, False)


@pytest.mark.parametrize("key", FAMILY_KEYS)
def test_grid_allowed_tokens(cti, key):
    ad = make_adapter(key, device="cpu", dtype=torch.float32, tiny=True, init_std=cti.TINY_INIT_STD)
    ad.add_grid_tokens([f"<g{i}>" for i in range(4)])
    al = cti.grid_allowed(ad)
    tok = ad.tokenizer
    want = {tok("0", add_special_tokens=False)["input_ids"][0], tok("1", add_special_tokens=False)["input_ids"][0],
            tok("\n", add_special_tokens=False)["input_ids"][0]}
    assert set(al["probe"]["distinct_ids"]) == want and al["probe"]["n_tokens"] == 16 * 16 + 15
    assert al["ids"] == sorted(want | set(ad.stop_token_ids())) and al["stop_ids"] == list(ad.stop_token_ids())
    assert not set(al["ids"]) & set(ad.grid_token_ids.values())  # 不用新增的 grid token
    assert [r["role"] for r in al["tokens"]][:3] == ["digit_0", "digit_1", "newline"]
    json.dumps(al)


def test_randomize_lora_scale_and_seed(cti):
    def lora_state(seed, scale):
        ad = make_adapter("internvl3.5-2b-hf", device="cpu", dtype=torch.float32, tiny=True, init_std=cti.TINY_INIT_STD)
        ad.inject_lora(r=cti.LORA_RANK, lora_alpha=cti.LORA_ALPHA, init_lora_weights=False)
        info = cti.randomize_lora(ad, rel_scale=scale, seed=seed)
        return ad, info, {n: p.detach().clone() for n, p in ad.model.named_parameters() if "lora_" in n}

    ad, info, s0 = lora_state(0, 0.05)
    assert info["n_modules"] == len(ad.lora_target_modules())
    r = info["realized_rel_frobenius"]
    assert 0.05 * (1 - 1e-4) <= r["min"] <= r["max"] <= 0.05 * (1 + 1e-4)
    mods = dict(ad.model.named_modules())
    name = ad.lora_target_modules()[0]
    m = mods[name]
    delta = m.scaling["default"] * (m.lora_B["default"].weight @ m.lora_A["default"].weight)
    assert torch.isclose(delta.norm() / m.base_layer.weight.norm(), torch.tensor(0.05), rtol=1e-4)
    _, _, s0b = lora_state(0, 0.05)
    _, _, s1 = lora_state(1, 0.05)
    assert all(torch.equal(s0[k], s0b[k]) for k in s0)  # 同一种子逐位相同
    assert any(not torch.equal(s0[k], s1[k]) for k in s0)


def test_summaries(cti):
    g = cti.gap_summary([0.01, None, 0.2, 0.05])
    assert (g["n"], g["max"], g["frac_lt_threshold"], g["threshold"]) == (3, 0.2, 1.0, cti.RESTRICTED_MAX_GAP)
    g = cti.gap_summary([0.01, None, 2.0, 2.5], threshold=2.0)
    assert g["frac_lt_threshold"] == pytest.approx(1 / 3) and g["median"] == 2.0
    assert cti.gap_summary([None])["n"] == 0
    p = cti.position_summary([None, 0, 15, 16, 127], 128)
    assert (p["n"], p["min"], p["max"], p["histogram"]["0-15"], p["histogram"]["16-31"], p["histogram"]["112-127"]) == (
        4, 0, 127, 2, 1, 1)


def _pair(agree=1.0, exact=1.0, gap=None, n_dis=0, key="agreement", no_gap=0, gen_gap=None, median=None, tok=None):
    """(生成的读数, 教师强制的读数)。gap = 教师强制不一致处的最大差（median 默认同 gap），gen_gap = 生成分歧处的最大差。"""
    gs = {"n": 0 if gap is None else 1, "max": gap, "median": gap if median is None else median}
    gg = {"n": 0 if gen_gap is None else 1, "max": gen_gap, "median": gen_gap}
    gen = {"exact_rows": exact, "token_agreement": agree if tok is None else tok, "n_divergence_without_gap": no_gap, "gap": gg}
    tf = {key: agree, "n_disagreements": n_dis, "gap": gs}
    return gen, tf


def test_judge_rules_v1(cti):
    """D50-3 的 v1：每条判据单独触发；只报不判的项不影响通过。"""
    gpu, cpu = cti.criteria_block(False), cti.criteria_block(True)
    assert gpu["version"] == cpu["version"] == "v1" and gpu["exact"] is False and cpu["exact"] is True
    R = lambda mode, *a, **k: cti._judge_restricted("x", mode, *_pair(*a, **k), gpu)  # noqa: E731
    off, on = "thinking_off", "thinking_on"
    # restricted：教师强制 ≥ 0.999；关思考 exact_rows ≥ 0.95；不一致处差的最大值 < 2.0（教师强制与解码分歧都算）
    assert R(off, 0.9993, 0.96, 1.49, 12) == [] and R(on, 0.9993, 0.76, 0.41, 3, gen_gap=0.41, tok=0.76) == []
    assert R(off, 0.9989, 1.0, 0.1, 3)  # 教师强制一致率不够
    assert R(on, 0.9989, 1.0, 0.1, 3)  # 两种开关都判
    assert R(off, 0.9995, 0.94, 0.1, 1)  # 关思考 exact_rows 不够
    assert R(on, 0.9995, 0.10, 0.1, 1, tok=0.2) == []  # 开思考 exact_rows、token_agreement 只报不判
    assert R(off, 0.9995, 1.0, 2.0, 1)  # 教师强制的差不是 < 2.0
    assert R(on, 0.9995, 0.5, 0.1, 1, gen_gap=2.5)  # 解码分歧处的差不是 < 2.0
    assert R(on, 0.9995, 0.5, 0.1, 1, gen_gap=0.1, no_gap=1)  # 分歧算不出 gap
    # merge：教师强制 ≥ 0.98；不一致处差的中位数 ≤ 0.5；生成只报不判；最大差只报不判
    M = lambda *a, **k: cti._judge_merge("x", *_pair(*a, key="argmax_agreement", **k), gpu)  # noqa: E731
    assert M(0.9899, 0.295, 3.87, 50, median=0.097, gen_gap=1.47, tok=0.57) == []
    assert M(0.979, 1.0, 0.1, 5)
    assert M(0.99, 1.0, 0.6, 5, median=0.51)
    # CPU fp32：逐个相同
    assert cti._judge_restricted("x", off, *_pair(), cpu) == [] and cti._judge_merge("x", *_pair(key="argmax_agreement"), cpu) == []
    assert cti._judge_restricted("x", on, *_pair(1.0, 1.0, 0.0, 1), cpu)  # 一个不同的位置也不行
    assert cti._judge_merge("x", *_pair(1.0, 0.5, key="argmax_agreement"), cpu)
    # 前提：只在开思考时判 exact_rows ≤ 0.5；教师强制 argmax 与 base 的一致率只报不判
    gen, tf = _pair(key="argmax_agreement")
    eff = {"thinking_off": {"generate": {"exact_rows": 1.0}, "teacher_forcing": {"argmax_agreement": 1.0}},
           "thinking_on": {"generate": {"exact_rows": 0.01}, "teacher_forcing": {"argmax_agreement": 0.999}}}
    checks = {"merge": {"by_thinking": {m: {"generate": gen, "teacher_forcing": tf} for m in (off, on)}, "effect_vs_base": eff}}
    assert cti.judge(checks, cpu)["merge"] == [] and cti.judge(checks, gpu)["merge"] == []
    eff["thinking_on"]["generate"]["exact_rows"] = 0.6
    assert any("太弱" in m and "thinking_on" in m for m in cti.judge(checks, gpu)["merge"])
    assert any("太弱" in m for m in cti.judge(checks, cpu)["merge"])


RUN0021 = REPO_ROOT / "experiments/E3/0021_train_iface_check"
RUN0021_MODELS = ("gemma-4-e2b-it", "internvl3.5-2b-hf", "qwen3.5-0.8b", "qwen3.5-9b")


def test_v1_rejudges_run0021_all_pass(cti):
    """D50-3：RUN-0021 的 4 份 JSON（当时按提议值判，全部不过）按 v1 重判，4/4 通过。"""
    for key in RUN0021_MODELS:
        doc = json.loads((RUN0021 / f"{key}.json").read_text())
        assert doc["model"]["key"] == key and doc["pass"] is False and doc["criteria"]["exact"] is False
        r = cti.rejudge_document(doc)
        assert r["criteria_version"] == "v1" and r["pass"] is True, (key, r["failures"])
        assert set(r["failures"]) == {"restricted", "merge"}
    bad = json.loads((RUN0021 / "gemma-4-e2b-it.json").read_text())
    bad["checks"]["restricted"]["by_thinking"]["thinking_off"]["teacher_forcing"]["agreement"] = 0.998
    assert cti.rejudge_document(bad)["pass"] is False
    with pytest.raises(ValueError, match="format"):
        cti.rejudge_document({"format": "x"})


# ---------------------------------------------------------------------------
# 彩排：三个家族端到端（默认参数：128 个新 token、两种思考开关、两项检查）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", FAMILY_KEYS)
def test_tiny_rehearsal_passes(cti, key, tmp_path):
    code, doc = _run(cti, tmp_path, "--model", key)
    assert doc["error"] is None, doc["error"]
    assert code == 0 and doc["pass"] is True, doc["failures"]
    assert set(doc) >= {"format", "argv", "written_utc", "code", "versions", "gpu", "model", "checkpoint", "data",
                        "generation", "batch_protocol", "allowed", "criteria_version", "criteria", "checks", "failures", "pass",
                        "timing_s"}
    assert doc["criteria"]["exact"] is True and doc["model"]["revision"] and doc["model"]["key"] == key
    assert doc["criteria_version"] == doc["criteria"]["version"] == cti.CRITERIA_VERSION
    assert doc["code"]["commit"] is None or len(doc["code"]["commit"]) == 40
    assert doc["data"]["n"] == FIXTURE_N and doc["batch_protocol"]["batches"] == [[0, 2], [2, 3]]
    assert doc["batch_protocol"]["frozen_batch_size"] == load_prompt_config().batch_protocol()["batch_size"]
    assert doc["generation"]["max_new_tokens"] == 128
    r, m = doc["checks"]["restricted"], doc["checks"]["merge"]
    assert r["pass"] and m["pass"] and m["mode"] == "random_lora"
    for mode in ("thinking_off", "thinking_on"):
        dv, tf = r["by_thinking"][mode]["decode_vs_generate_batch"], r["by_thinking"][mode]["teacher_forcing"]
        assert dv["n_rows"] == FIXTURE_N and dv["exact_rows"] == 1.0 and tf["agreement"] == 1.0 and tf["n_tokens"] >= FIXTURE_N
        g, t = m["by_thinking"][mode]["generate"], m["by_thinking"][mode]["teacher_forcing"]
        assert g["exact_rows"] == 1.0 and t["argmax_agreement"] == 1.0 and t["n_tokens"] == g["n_tokens"]
        assert t["max_abs_logit"] < 1e-3  # fp32：只差合并带来的求和顺序
        e = m["effect_vs_base"][mode]
        assert e["generate"]["exact_rows"] <= cti.EFFECT_MAX_EXACT_ROWS  # 随机 LoRA 让输出明显偏离 base（两种开关都偏离）
        assert e["teacher_forcing"]["argmax_agreement"] <= 0.99  # randomize_lora 说明里小模型上的标定（只报不判，这里钉住）
        assert e["teacher_forcing"]["max_abs_logit"] > 1000 * t["max_abs_logit"]
    assert m["lora"]["r"] == 32 and m["lora"]["rel_scale"] == cti.LORA_REL_SCALE


# ---------------------------------------------------------------------------
# --checkpoint：评测前的标准核对
# ---------------------------------------------------------------------------


def test_checkpoint_mode(cti, tmp_path):
    """训练出来的检查点（随机 LoRA、训练过的行、带查询 token）：load_trained(merge=False) 对比 merge=True。"""
    key = "qwen3.5-0.8b"
    ad = make_adapter(key, device="cpu", dtype=torch.float32, tiny=True, init_std=cti.TINY_INIT_STD)
    grid = ad.add_grid_tokens([f"<g{i}>" for i in range(6)])
    ad.inject_query_tokens(4, "grid2d")
    ad.inject_lora(r=8, lora_alpha=16, init_lora_weights=False)
    cti.randomize_lora(ad, rel_scale=cti.LORA_REL_SCALE, seed=3)
    (w,) = ad.trainable_token_rows(["<g0>", "<g1>"])
    with torch.no_grad():
        w[[grid["<g0>"], grid["<g1>"]]] += 0.01
    qe = torch.randn(4, ad.hidden_size(), generator=torch.Generator().manual_seed(0)) * 0.05
    ck = ad.save_trained(tmp_path / "ck", query_embeds=qe)
    code, doc = _run(cti, tmp_path, "--model", key, "--checkpoint", str(ck), "--max-new-tokens", "16", "--checks", "merge")
    assert doc["error"] is None, doc["error"]
    assert code == 0 and doc["pass"] is True, doc["failures"]
    m = doc["checks"]["merge"]
    assert m["mode"] == "checkpoint" and "effect_vs_base" not in m
    assert doc["checkpoint"]["lora"]["r"] == 8 and doc["checkpoint"]["query"]["n"] == 4
    assert doc["checkpoint"]["trainable_token_rows"] == sorted([grid["<g0>"], grid["<g1>"]])
    for mode in ("thinking_off", "thinking_on"):
        assert m["by_thinking"][mode]["generate"]["exact_rows"] == 1.0
    # DEV-0032：检查点登记了网格 token、却不是 E3b 的词表：restricted 没有定义好的 allowed 集合，报错（不悄悄退回 E1 的数字）
    code, doc = _run(cti, tmp_path, "--model", key, "--checkpoint", str(ck), "--checks", "restricted")
    assert code == 2 and "E3b 词表" in doc["error"]["repr"] and "restricted" not in doc["checks"]
    code, doc = _run(cti, tmp_path, "--model", "qwen3.5-2b", "--checkpoint", str(ck), "--checks", "merge")
    assert code == 2 and "构造方式" in doc["error"]["repr"]  # --model 与检查点不符


# ---------------------------------------------------------------------------
# 仪器的灵敏度：坏掉的合并、偏了的受限 logits 都判不过
# ---------------------------------------------------------------------------


def test_detects_merge_that_drops_lora(cti, tmp_path, monkeypatch):
    def broken_merge(self):  # unload 而不 merge：增量被丢掉，"合并后"的模型就是 base
        self._peft.unload()
        self._peft = None
        self._lora_state = "merged"

    monkeypatch.setattr(CK.CheckpointSupport, "merge_lora", broken_merge)
    code, doc = _run(cti, tmp_path, "--model", "gemma-4-e2b-it", "--checks", "merge", "--max-new-tokens", "16")
    assert doc["error"] is None, doc["error"]
    assert code == 1 and doc["pass"] is False and doc["checks"]["merge"]["pass"] is False
    assert any("生成不逐个相同" in x for x in doc["failures"]["merge"])
    assert any("教师强制" in x for x in doc["failures"]["merge"])
    _check_gap_records(doc["checks"]["merge"]["by_thinking"], "generate")


def _check_gap_records(by_thinking, gen_key):
    """分歧行都算出了 gap；每条记录 gap ≥ top2_gap ≥ 0（gap 是两个分歧 token 之差，不小于前两名之差）。"""
    n_rows = n_dis = 0
    for r in by_thinking.values():
        g, tf = r[gen_key], r["teacher_forcing"]
        n_rows += g["n_divergent_rows"]
        n_dis += tf["n_disagreements"]
        assert g["n_divergence_without_gap"] == 0 and g["gap"]["n"] == g["n_divergent_rows"]
        assert tf["gap"]["n"] == tf["n_disagreements"]
        for rec in g["divergent_rows"] + tf["disagreements"]:
            assert rec["gap"] >= rec["top2_gap"] - 1e-6 and rec["top2_gap"] >= 0
    assert n_rows > 0


def test_detects_biased_restricted_logits(cti, tmp_path, monkeypatch):
    orig = training.TrainingSupport.restricted_logits

    def biased(self, hidden, allowed_ids):  # 给 allowed 里 id 最小的一列加 1：受限解码偏离 generate_batch
        z = orig(self, hidden, allowed_ids)
        return z + torch.nn.functional.one_hot(torch.tensor(0), z.shape[-1]).to(z)

    monkeypatch.setattr(training.TrainingSupport, "restricted_logits", biased)
    code, doc = _run(cti, tmp_path, "--model", "gemma-4-e2b-it", "--checks", "restricted", "--max-new-tokens", "16")
    assert doc["error"] is None, doc["error"]
    assert code == 1 and doc["checks"]["restricted"]["pass"] is False
    assert any("生成不逐个相同" in x for x in doc["failures"]["restricted"])
    _check_gap_records(doc["checks"]["restricted"]["by_thinking"], "decode_vs_generate_batch")


def test_error_writes_json(cti, tmp_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("故意失败")

    monkeypatch.setattr(cti, "grid_allowed", boom)
    code, doc = _run(cti, tmp_path, "--model", "internvl3.5-2b-hf")
    assert code == 2 and doc["pass"] is False and "故意失败" in doc["error"]["repr"]
    assert np.isfinite(doc["timing_s"].get("total_s", 0.0))


# ---------------------------------------------------------------------------
# DEV-0032：带网格 token 的检查点的 restricted；不带网格 token 时输出与 69884b8 逐字节相同
# ---------------------------------------------------------------------------

SNAPSHOT = REPO_ROOT / "tests/fixtures/cti/check_training_interfaces_69884b8.py"  # 69884b8 的 scripts/check_training_interfaces.py 原样
SNAPSHOT_SHA256 = "085d57d1320f5787cda65c03452c62dbf7aa72dda1e62528f191fad5f7380aa1"
VOLATILE = ("written_utc", "timing_s", "argv", "code")  # 墙钟、时间、输出路径、脚本所在目录的 git 信息


def _normalized(doc: dict) -> str:
    return json.dumps({k: v for k, v in doc.items() if k not in VOLATILE}, sort_keys=True, ensure_ascii=False)


def _no_grid_checkpoint(cti, root: Path) -> Path:
    """C3 那样的检查点：随机 LoRA，没有网格 token。"""
    ad = make_adapter("qwen3.5-0.8b", device="cpu", dtype=torch.float32, tiny=True, init_std=cti.TINY_INIT_STD)
    ad.inject_lora(r=8, lora_alpha=16, init_lora_weights=False)
    cti.randomize_lora(ad, rel_scale=cti.LORA_REL_SCALE, seed=5)
    return ad.save_trained(root)


def test_non_grid_outputs_identical_to_69884b8(cti, tmp_path):
    """不给 --checkpoint 的 base（两项检查，随机 LoRA）与不带网格 token 的检查点（C3 那样）：新脚本与 69884b8 的脚本在同一进程、
    同一组参数下的输出，除墙钟、时间、输出路径与 git 信息外逐字节相同（单线程，排除多线程求和顺序的抖动）。"""
    assert hashlib.sha256(SNAPSHOT.read_bytes()).hexdigest() == SNAPSHOT_SHA256
    old = _load(SNAPSHOT, "_check_training_interfaces_69884b8")
    ck = _no_grid_checkpoint(cti, tmp_path / "c3_like")
    cases = {"base": ["--model", "qwen3.5-0.8b", "--max-new-tokens", "16"],
             "no_grid_checkpoint": ["--model", "qwen3.5-0.8b", "--max-new-tokens", "16", "--checkpoint", str(ck)],
             "no_grid_checkpoint_restricted": ["--model", "qwen3.5-0.8b", "--max-new-tokens", "16", "--checkpoint", str(ck),
                                               "--checks", "restricted"]}
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for name, args in cases.items():
            docs = []
            for k, mod in enumerate((old, cti)):
                out = tmp_path / f"{name}_{k}.json"
                code = mod.main(["--tiny", "--out", str(out), *args])
                docs.append((code, json.loads(out.read_text())))
            (c_old, d_old), (c_new, d_new) = docs
            assert c_old == c_new == 0 and d_new["error"] is None, (name, d_new["error"])
            assert _normalized(d_old) == _normalized(d_new), name
            assert "restricted_inputs" not in d_new and "restricted_grid" not in d_new["criteria"]
            assert "across_merge" not in d_new["checks"]["restricted"]
            assert list(d_new["checks"]["restricted"]["by_thinking"]) == ["thinking_off", "thinking_on"]
    finally:
        torch.set_num_threads(n_threads)


def _grid_checkpoint(cti, root: Path, *, pathological: bool, query: bool = False) -> tuple[Path, object]:
    """带 E3b 词表的检查点（Qwen 小模型，E3b --tiny 的图像口径）：网格 token 的行取互不相同的随机值（受限解码写满 1057 个 token）。
    pathological = 病态的 LoRA（同 tests/test_e3_eval.bad_checkpoint：A 的两行几乎相同、B 的两列相反且放大 1e7；真实的 ΔW 很小，
    但合并与不合并两条路径的中间量都很大、舍入各不相同，合并前后网格 token 的 logits 分家）；否则是 randomize_lora 的随机 LoRA。"""
    from src.adapters.image_policy import parse_policy
    from src.e3.config import load_config
    from src.e3.tokens import add_vocab

    cfg = load_config()
    pol = parse_policy(cfg.doc["tiny"]["image_policies"]["qwen3.5"])
    ad = make_adapter("qwen3.5-0.8b", device="cpu", dtype=torch.float32, image_policy=pol, tiny=True, init_std=cti.TINY_INIT_STD)
    tok = add_vocab(ad, cfg.vocab)
    if query:
        ad.inject_query_tokens(4, "grid2d")
    g = torch.Generator().manual_seed(0)
    if pathological:
        ad.inject_lora(r=2, lora_alpha=2, init_lora_weights=False)
        mods = dict(ad.model.named_modules())
        with torch.no_grad():
            for name in ad.lora_target_modules():
                m = mods[name]
                a_w, b_w = m.lora_A["default"].weight, m.lora_B["default"].weight
                u = torch.randn(a_w.shape[1], generator=g)
                a_w.copy_(torch.stack([u, u + 1e-6 * torch.randn(a_w.shape[1], generator=g)]))
                v = torch.randn(b_w.shape[0], generator=g)
                b_w.copy_(torch.stack([v, -v], dim=1) * 1e7)
    else:
        ad.inject_lora(r=8, lora_alpha=16, init_lora_weights=False)
        cti.randomize_lora(ad, rel_scale=cti.LORA_REL_SCALE, seed=3)
    ids = torch.tensor(sorted(tok.grid_ids))
    with torch.no_grad():
        for w in ad.trainable_token_rows(list(cfg.vocab.names)):
            w[ids] = torch.randn(len(ids), w.shape[1], generator=g)
    qe = torch.randn(4, ad.hidden_size(), generator=g) * 0.05 if query else None
    return ad.save_trained(root, query_embeds=qe), tok


def test_grid_checkpoint_restricted_uses_own_grid_tokens(cti, tmp_path):
    """带网格 token 的检查点（带查询 token）：restricted 的 allowed 集合 = tokens.allowed_ids(refseg, task)，提示与长度同 E3b 的
    单轮评测，只有关思考；by_thinking 与 across_merge 都逐个相同（CPU fp32），受限解码写满 1057 个 token；merge 照旧。"""
    from src.e3.config import load_config
    from src.e3.evaluate import make_context
    from src.e3.targets import seq_len

    cfg = load_config()
    ck, tok = _grid_checkpoint(cti, tmp_path / "ck", pathological=False, query=True)
    code, doc = _run(cti, tmp_path, "--model", "qwen3.5-0.8b", "--checkpoint", str(ck), "--checks", "restricted", "merge")
    assert doc["error"] is None, doc["error"]
    assert code == 0 and doc["pass"] is True, doc["failures"]
    ctx = make_context(cfg, tok, "refseg")
    assert doc["allowed"]["ids"] == tok.allowed_ids("refseg", "task") == list(ctx.allowed)
    assert doc["allowed"]["stop_ids"] == [tok.stop_id] and doc["allowed"]["layout"] == tok.layout()
    ri = doc["restricted_inputs"]
    assert (ri["task"], ri["max_new_tokens"], ri["thinking_modes"], ri["order"]) == ("refseg", seq_len(32), [False],
                                                                                        "instruction_first")
    assert ri["instruction_template"] == cfg.doc["tasks"]["refseg"]["grid_template"] and ri["e3b_config"]["sha256"] == cfg.sha256
    r = doc["checks"]["restricted"]
    assert list(r["by_thinking"]) == ["thinking_off"] and list(r["across_merge"]["by_thinking"]) == ["thinking_off"]
    dv, tf = r["by_thinking"]["thinking_off"]["decode_vs_generate_batch"], r["by_thinking"]["thinking_off"]["teacher_forcing"]
    assert dv["n_rows"] == FIXTURE_N and dv["exact_rows"] == 1.0 and tf["agreement"] == 1.0
    assert tf["n_tokens"] == FIXTURE_N * seq_len(32)  # 网格 token 互不相同：每行都写满（不提前停）
    am = r["across_merge"]["by_thinking"]["thinking_off"]
    assert am["n_rows"] == FIXTURE_N and am["teacher_forcing"]["n_tokens"] == tf["n_tokens"]
    assert am["teacher_forcing"]["argmax_agreement"] == 1.0 and am["teacher_forcing"]["max_abs_logit"] < 1e-3
    assert doc["criteria"]["restricted_grid"]["across_merge"]["min_tf_agreement"] == cti.MERGE_MIN_TF_AGREEMENT
    assert doc["checks"]["merge"]["mode"] == "checkpoint" and doc["checks"]["merge"]["pass"] is True
    assert set(doc["timing_s"]) >= {"restricted_s", "restricted_across_merge_s"}
    rj = cti.rejudge_document(doc)  # 按当前判据重判，结果相同
    assert rj["pass"] is True and "restricted_grid" in rj["criteria"]


def test_grid_checkpoint_split_across_merge_fails_restricted(cti, tmp_path):
    """坏检查点：合并前后网格 token 的 logits 分家。合并后的模型内部自洽（by_thinking 逐个相同），只有 across_merge 查得出；
    restricted 判不过（退出码 1）。"""
    ck, _ = _grid_checkpoint(cti, tmp_path / "bad", pathological=True)
    code, doc = _run(cti, tmp_path, "--model", "qwen3.5-0.8b", "--checkpoint", str(ck), "--checks", "restricted")
    assert doc["error"] is None, doc["error"]
    assert code == 1 and doc["pass"] is False and doc["checks"]["restricted"]["pass"] is False
    bad = doc["failures"]["restricted"]
    assert bad and all(m.startswith("restricted/across_merge/thinking_off") for m in bad)
    r = doc["checks"]["restricted"]
    assert r["by_thinking"]["thinking_off"]["teacher_forcing"]["agreement"] == 1.0
    am = r["across_merge"]["by_thinking"]["thinking_off"]["teacher_forcing"]
    assert am["n_disagreements"] > 0 and am["argmax_agreement"] < 1.0
    for rec in am["disagreements"]:
        assert rec["gap"] >= rec["top2_gap"] - 1e-6 and rec["merged"] != rec["unmerged"]


def test_judge_across_merge_rules(cti):
    """across_merge：CPU fp32 时一个不同的位置也不行；GPU 照 v1 merge 一节（argmax 一致率 ≥ 0.98、gap 中位数 ≤ 0.5）。"""
    gpu, cpu = cti.criteria_block(False), cti.criteria_block(True)
    ok = {"argmax_agreement": 1.0, "n_disagreements": 0, "gap": {"n": 0, "median": None, "max": None}}
    assert cti._judge_across_merge("x", ok, cpu) == [] and cti._judge_across_merge("x", ok, gpu) == []
    one = {"argmax_agreement": 0.999, "n_disagreements": 1, "gap": {"n": 1, "median": 0.1, "max": 0.1}}
    assert cti._judge_across_merge("x", one, cpu) and cti._judge_across_merge("x", one, gpu) == []
    low = {**one, "argmax_agreement": 0.97}
    assert any("argmax_agreement" in m for m in cti._judge_across_merge("x", low, gpu))
    wide = {**one, "gap": {"n": 3, "median": 0.6, "max": 3.0}}
    assert any("中位数" in m for m in cti._judge_across_merge("x", wide, gpu))
    # judge：across_merge 的失败记在 restricted 名下；没有 across_merge（base、C3）时与原来相同
    gen = {"exact_rows": 1.0, "token_agreement": 1.0, "n_divergence_without_gap": 0, "gap": {"n": 0, "max": None, "median": None}}
    tf = {"agreement": 1.0, "n_disagreements": 0, "gap": {"n": 0, "max": None, "median": None}}
    checks = {"restricted": {"by_thinking": {"thinking_off": {"decode_vs_generate_batch": gen, "teacher_forcing": tf}}}}
    assert cti.judge(checks, cpu) == {"restricted": []}
    checks["restricted"]["across_merge"] = {"by_thinking": {"thinking_off": {"teacher_forcing": low}}}
    assert cti.judge(checks, gpu)["restricted"] and cti.judge(checks, cpu)["restricted"]
