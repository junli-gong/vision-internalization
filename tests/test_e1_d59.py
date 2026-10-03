"""DEV-0027：E1 的思考子集改口径（D59-1）与验收 v3 的 δ 按组（D59-3）。

- 硬性要求：eval、dev、free、extra 的提示渲染、生成参数与解析，与 sha256 5de1aec7… 的旧配置（RUN-0027 主评测用的那一份，存档在
  tests/fixtures/e1/e1_prompts_5de1aec7.json）逐字节相同：用旧、新配置分别渲染 fixture 里这几个子集的全部样本（含全部开发样本），
  指令文本逐字节相同，三个家族的小模型 build_inputs 得到的 input_ids 逐个相同，生成参数（config.yaml 的 generation 一节）与解析函数
  相同；另核对两份配置除了 D59 改的几处以外完全相同（差异清单钉在 CHANGED_PATHS）。
- 思考子集：预算强制收尾（4096 + 1024），走 generate_batch_budgeted；predictions 记各段 token 数与补的 token，metrics 按结束方式
  分开报并给条数，compute 的 prompt_tokens 含补的 token。think8192 已删除。
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch
import yaml

from src.adapters import registry
from src.adapters.factory import make_adapter
from src.e1 import acceptance as A
from src.e1 import prompts as P
from src.e1 import runner as R
from src.e1.tasks import TASK_SUBSETS, DataPaths, select
from tests.test_e1_runner import FakeAdapter, leaf, spec

REPO = Path(__file__).resolve().parents[1]
OLD_PATH = REPO / "tests" / "fixtures" / "e1" / "e1_prompts_5de1aec7.json"
OLD_SHA256 = "5de1aec711fad0140db0cf27094d66e460cf2999ffcd5456f1964bc5b9851158"
NEW = P.load_prompt_config()
SAME_SUBSETS = ("eval", "dev", "free", "extra")
# 新旧配置之间允许不同的地方（D59-1、D59-3）；其余一律逐项相同
CHANGED_PATHS = {
    ("generation", "enable_thinking", "think8192"),
    ("generation", "think_max_new_tokens"),
    ("generation", "think_budget"),
    ("generation", "think_note"),
    ("letter_readout", "free_form_diagnostic"),
    ("acceptance", "rule"),
    ("acceptance", "delta"),
    ("acceptance", "delta_rule"),
    ("acceptance", "basis"),
    *(("acceptance", "groups", k, "delta") for k in range(5)),
}


def _old() -> P.PromptConfig:
    data = json.loads(OLD_PATH.read_text(encoding="utf-8"))
    return P.PromptConfig(data=data, path=str(OLD_PATH), sha256=P.file_sha256(OLD_PATH))


@pytest.fixture(scope="module")
def paths(tmp_path_factory):
    return DataPaths.fixtures(nyu_cache=tmp_path_factory.mktemp("nyu"))


def _diff(a, b, path=()):
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            out += _diff(a.get(k, "<missing>"), b.get(k, "<missing>"), path + (k,))
        return out
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [d for k, (x, y) in enumerate(zip(a, b)) for d in _diff(x, y, path + (k,))]
    return [] if a == b else [path]


def test_old_config_is_the_one_run0027_uses_and_new_sha_differs():
    assert hashlib.sha256(OLD_PATH.read_bytes()).hexdigest() == OLD_SHA256
    assert NEW.sha256 != OLD_SHA256
    with pytest.raises(P.PromptConfigError, match="think_max_new_tokens|think_budget|delta"):
        P.validate(_old().data)  # 旧配置按新规则不合格（它有 think8192、think_max_new_tokens、按类的 δ）


def test_only_d59_sections_changed():
    diff = set(_diff(_old().data, NEW.data))
    assert diff == CHANGED_PATHS, sorted(diff ^ CHANGED_PATHS)


def test_eval_dev_free_extra_render_and_parse_byte_identical(paths):
    old = _old()
    n = 0
    for task in P.TASKS:
        for subset in [s for s in SAME_SUBSETS if s in TASK_SUBSETS[task]]:
            sel = select(task, subset, paths)
            for fam in registry.FAMILIES:
                assert old.max_new_tokens(task, fam, subset) == NEW.max_new_tokens(task, fam, subset)
                assert old.parser(task, fam, subset) == NEW.parser(task, fam, subset)
                assert old.readout(task, subset) == NEW.readout(task, subset)
                assert old.entry(task, fam) == NEW.entry(task, fam)
                for s in sel.samples:
                    a, b = old.render(task, fam, s), NEW.render(task, fam, s)
                    assert a.text.encode("utf-8") == b.text.encode("utf-8") and a.extras == b.extras
                    n += 1
    assert n > 100
    for subset in SAME_SUBSETS:
        assert old.enable_thinking(subset) is NEW.enable_thinking(subset) is False
        assert not NEW.uses_think_budget(subset)
    assert old.decoding() == NEW.decoding() and old.order == NEW.order and old.letter_spellings() == NEW.letter_spellings()
    assert old.batch_protocol() == NEW.batch_protocol()


def test_generation_section_of_config_yaml_unchanged(paths):
    """config.yaml 的 generation 一节（运行身份的一部分）在 eval、dev、free、extra 上与旧配置逐项相同；思考子集另加 think_budget。"""
    old = _old()
    for task in P.TASKS:
        for subset in [s for s in SAME_SUBSETS if s in TASK_SUBSETS[task]]:
            sp = spec(task, subset)
            ro = {"kind": NEW.readout(task, subset), "parse": NEW.parser(task, sp.family, subset)}
            want = {"enable_thinking": old.enable_thinking(subset), "max_new_tokens": old.max_new_tokens(task, sp.family, subset),
                    "decoding": old.decoding(), "order": old.order, "readout": ro}
            got = R.generation_record(NEW, sp, ro)
            assert yaml.safe_dump(got, allow_unicode=True) == yaml.safe_dump(want, allow_unicode=True)
    th = R.generation_record(NEW, spec("count", "think"), {"kind": "text", "parse": "count"})
    assert th["think_budget"]["thinking_budget"] == 4096 and th["think_budget"]["answer_budget"] == 1024
    assert th["max_new_tokens"] == 5120 and th["enable_thinking"] is True


@pytest.mark.parametrize("key", ["qwen3.5-0.8b", "gemma-4-e2b-it", "internvl3.5-2b-hf"])
def test_model_inputs_byte_identical_with_old_and_new_config(key, paths):
    """三个家族的小模型：旧、新配置渲染出的指令送进 build_inputs，input_ids 逐个相同（fixture 里 eval、dev、free、extra 的全部样本）。"""
    ad = make_adapter(key, device="cpu", dtype=torch.float32, tiny=True)
    fam = registry.get_model(key).family
    old = _old()
    seen = set()
    for task in P.TASKS:
        for subset in [s for s in SAME_SUBSETS if s in TASK_SUBSETS[task]]:
            for s in select(task, subset, paths).samples:
                a, b = old.render(task, fam, s).text, NEW.render(task, fam, s).text
                if (a, s.images) in seen:
                    continue
                seen.add((a, s.images))
                ims = R.load_images(s)
                x = ad.build_inputs(ims, a, old.order).tensors["input_ids"]
                y = ad.build_inputs(ims, b, NEW.order).tensors["input_ids"]
                assert x.numpy().tobytes() == y.numpy().tobytes()
    assert len(seen) > 30


def test_think_subset_protocol():
    assert P.SUBSET_KINDS == ("eval", "dev", "think", "extra", "free") and P.THINK_SUBSETS == ("think",)
    assert all("think8192" not in subs for subs in TASK_SUBSETS.values())
    gen = NEW.data["generation"]
    assert "think_max_new_tokens" not in gen and NEW.think_budget() == (4096, 1024)
    assert gen["think_budget"]["method"] == P.THINK_METHOD == "generate_batch_budgeted" and "D59-1" in gen["think_budget"]["basis"]
    assert "D59-1" in gen["think_note"]
    bad = copy.deepcopy(dict(NEW.data))
    bad["generation"]["think_budget"]["answer_budget"] = 0
    with pytest.raises(P.PromptConfigError, match="think_budget"):
        P.validate(bad)
    bad = copy.deepcopy(dict(NEW.data))
    bad["acceptance"]["groups"][2]["delta"] = 1.5
    with pytest.raises(P.PromptConfigError, match="delta"):
        P.validate(bad)
    with pytest.raises(ValueError, match="delta"):
        A.group_delta({"name": "x"})


def test_runner_think_subset_uses_budgeted_generation_and_reports_by_kind(paths, tmp_path):
    sel = select("count", "think", paths)
    fake = FakeAdapter()
    m = R.run_task(fake, spec("count", "think"), sel, NEW, paths, tmp_path, log=lambda _: None)
    assert {w for w, _, _ in fake.calls} == {"generate_batch_budgeted"} and set(fake.max_new) == {4096 + 1024}
    lf = leaf(tmp_path, "count", "think")
    recs = R.read_jsonl_tolerant(lf / "predictions.jsonl", repair=False)
    for r in recs:
        assert r["thinking_end"] in ("natural", "forced", "not_opened")
        assert r["n_thinking_tokens"] + r["n_answer_tokens"] == r["n_generated_tokens"]
        if r["forced_close"]:
            assert r["inserted_text"] == "</think>\n\n" and r["n_inserted_tokens"] == 2 and r["n_thinking_tokens"] == 4096
            assert r["raw_text"][r["inserted_span"][0]: r["inserted_span"][1]] == r["inserted_text"]
        else:
            assert r["inserted_text"] is None and r["n_inserted_tokens"] == 0
    te = m["thinking_end"]
    assert sum(te["counts"].values()) == len(recs) and set(te["counts"]) == {"natural", "forced", "not_opened", "unclosed"}
    for kind, e in te["by_kind"].items():
        assert e["n"] == te["counts"][kind] and e["strict"][0]["name"] == m["headline"][0]["name"]
    comp = json.loads((lf / "compute.json").read_text())
    assert comp["prompt_tokens"] == sum(r["n_prompt_tokens"] + r["n_inserted_tokens"] for r in recs)
    assert comp["generated_tokens"] == sum(r["n_generated_tokens"] for r in recs)
    cfg = yaml.safe_load((lf / "config.yaml").read_text())
    assert cfg["generation"]["think_budget"]["thinking_budget"] == 4096
    readme = (lf / "README.md").read_text()
    assert "思考段的结束方式" in readme and "预算强制收尾" in readme
    # 别的子集：记录与指标里没有这些键（与原来逐字节相同）
    m2 = R.run_task(FakeAdapter(), spec("count", "eval"), select("count", "eval", paths), NEW, paths, tmp_path / "e",
                    log=lambda _: None)
    assert "thinking_end" not in m2
    r2 = R.read_jsonl_tolerant(leaf(tmp_path / "e", "count") / "predictions.jsonl", repair=False)
    assert all("thinking_end" not in r and "answer_text" not in r for r in r2)


def test_generate_rows_budget_guard():
    with pytest.raises(ValueError, match="预算强制收尾"):
        R.generate_rows(FakeAdapter(), [], batch_size=2, enable_thinking=False, max_new_tokens=5120,
                        decoding={"do_sample": False}, think_budget=(4096, 1024))
    with pytest.raises(ValueError, match="预算强制收尾"):
        R.generate_rows(FakeAdapter(), [], batch_size=2, enable_thinking=True, max_new_tokens=4096,
                        decoding={"do_sample": False}, think_budget=(4096, 1024))
