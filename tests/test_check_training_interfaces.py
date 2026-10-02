"""DEV-0021 第 1 条：scripts/check_training_interfaces.py（受限 logits 一致率、LoRA 合并一致率；评测前的合并核对仪器）。

- 彩排（--tiny，CPU fp32，tests/fixtures/e1 的小数据）：三个家族各跑一遍，两项检查都 pass（CPU fp32 的判据是逐个相同），
  随机 LoRA 的前提（输出明显偏离 base）也满足；输出 JSON 的字段齐全。
- --checkpoint：评测前的标准核对（load_trained(merge=False) 对比 load_trained(merge=True)），带查询 token 的检查点也能跑。
- 仪器的灵敏度：合并时把 LoRA 丢掉（unload 而不 merge）、受限 logits 偏向某个 token，都判不过。
- 判据的逻辑、allowed 集合、批配置读配置（D43：64，不写死）、钉住的常量。
"""

from __future__ import annotations

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
    assert (cti.MIN_AGREEMENT, cti.MAX_GAP) == (0.995, 0.125)  # D42-4 的提议值
    assert cti.MAX_GAP == 2.0 ** -3 == torch.finfo(torch.bfloat16).eps * 16  # bf16 在 [16, 32) 的 ulp
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
    assert al["ids"] == sorted(want | set(ad._stop_ids())) and al["stop_ids"] == ad._stop_ids()
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
    assert (g["n"], g["max"], g["frac_lt_threshold"]) == (3, 0.2, pytest.approx(2 / 3))
    assert cti.gap_summary([None])["n"] == 0
    p = cti.position_summary([None, 0, 15, 16, 127], 128)
    assert (p["n"], p["min"], p["max"], p["histogram"]["0-15"], p["histogram"]["16-31"], p["histogram"]["112-127"]) == (
        4, 0, 127, 2, 1, 1)


def _pair(agree=1.0, exact=1.0, gap=None, n_dis=0, key="agreement", no_gap=0):
    gs = {"n": 0 if gap is None else 1, "max": gap}
    gen = {"exact_rows": exact, "token_agreement": agree, "n_divergence_without_gap": no_gap, "gap": gs}
    tf = {key: agree, "n_disagreements": n_dis, "gap": gs}
    return gen, tf


def test_judge_rules(cti):
    gpu, cpu = cti.criteria_block(False), cti.criteria_block(True)
    assert (gpu["min_agreement"], gpu["max_gap"], cpu["min_agreement"]) == (0.995, 0.125, 1.0)
    assert cti._judge_pair("x", *_pair(0.996, 0.9, 0.1, 3), "agreement", gpu) == []
    assert cti._judge_pair("x", *_pair(0.994, 0.9, 0.1, 3), "agreement", gpu)  # 一致率不够
    assert cti._judge_pair("x", *_pair(0.999, 0.9, 0.125, 1), "agreement", gpu)  # gap 不是 < 0.125
    assert cti._judge_pair("x", *_pair(0.999, 0.9, 0.1, 1, no_gap=1), "agreement", gpu)  # 分歧算不出 gap
    assert cti._judge_pair("x", *_pair(), "agreement", cpu) == []
    assert cti._judge_pair("x", *_pair(1.0, 1.0, 0.0, 1), "agreement", cpu)  # CPU fp32：一个不同的位置也不行
    assert cti._judge_pair("x", *_pair(0.999, 0.5), "agreement", cpu)
    eff = {"thinking_off": {"generate": {"exact_rows": 0.6}, "teacher_forcing": {"argmax_agreement": 0.5}}}
    gen, tf = _pair(key="argmax_agreement")
    checks = {"merge": {"by_thinking": {"thinking_off": {"generate": gen, "teacher_forcing": tf}}, "effect_vs_base": eff}}
    assert any("太弱" in m for m in cti.judge(checks, cpu)["merge"])
    eff["thinking_off"]["generate"]["exact_rows"] = 0.0
    assert cti.judge(checks, cpu)["merge"] == []


# ---------------------------------------------------------------------------
# 彩排：三个家族端到端（默认参数：128 个新 token、两种思考开关、两项检查）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", FAMILY_KEYS)
def test_tiny_rehearsal_passes(cti, key, tmp_path):
    code, doc = _run(cti, tmp_path, "--model", key)
    assert doc["error"] is None, doc["error"]
    assert code == 0 and doc["pass"] is True, doc["failures"]
    assert set(doc) >= {"format", "argv", "written_utc", "code", "versions", "gpu", "model", "checkpoint", "data",
                        "generation", "batch_protocol", "allowed", "criteria", "checks", "failures", "pass", "timing_s"}
    assert doc["criteria"]["exact"] is True and doc["model"]["revision"] and doc["model"]["key"] == key
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
        assert e["generate"]["exact_rows"] <= cti.EFFECT_MAX_EXACT_ROWS  # 随机 LoRA 让输出明显偏离 base
        assert e["teacher_forcing"]["argmax_agreement"] <= cti.EFFECT_MAX_ARGMAX_AGREEMENT
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
    code, doc = _run(cti, tmp_path, "--model", key, "--checkpoint", str(ck), "--max-new-tokens", "16")
    assert doc["error"] is None, doc["error"]
    assert code == 0 and doc["pass"] is True, doc["failures"]
    m = doc["checks"]["merge"]
    assert m["mode"] == "checkpoint" and "effect_vs_base" not in m
    assert doc["checkpoint"]["lora"]["r"] == 8 and doc["checkpoint"]["query"]["n"] == 4
    assert doc["checkpoint"]["trainable_token_rows"] == sorted([grid["<g0>"], grid["<g1>"]])
    assert doc["checks"]["restricted"]["on_model"].startswith("checkpoint")
    for mode in ("thinking_off", "thinking_on"):
        assert m["by_thinking"][mode]["generate"]["exact_rows"] == 1.0
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
