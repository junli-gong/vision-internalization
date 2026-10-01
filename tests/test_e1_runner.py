"""DEV-0014：E1 运行器（src/e1/runner.py）与估算（src/e1/estimate.py），用脚本化的假适配器在 fixture 上端到端跑通：
输出文件齐全、断点续传后逐字节相同、配置不同拒绝续跑、半行修复、记账与逐条记录一致。

假适配器只替代模型（生成文本由一个确定的应答函数给出），解析用 Qwen3.5 的原生框解析（模块级函数，不需要模型），
所以评分路径与真实运行相同。真实小模型的端到端在 tests/test_e1_cli.py。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from src.adapters import qwen35, registry
from src.adapters.base import GenerationResult
from src.e1 import estimate as EST
from src.e1 import runner as R
from src.e1.prompts import TASKS, load_prompt_config
from src.e1.tasks import DataPaths, select

PROMPTS = load_prompt_config()
FILES = ("predictions.jsonl", "metrics.json", "compute.json", "config.yaml", "README.md", "meta.json", "logs/timing.jsonl")


@dataclass
class FakeInputs:
    instruction: str
    size: tuple[int, int]


class Interrupt(RuntimeError):
    pass


def respond(instruction: str, size) -> tuple[str, bool]:
    """确定的应答：按指令的类型给格式正确（或故意坏掉）的回答；第二个值是 stopped。"""
    h = int(hashlib.sha256(f"{instruction}|{size}".encode()).hexdigest(), 16)
    if "every instance" in instruction:
        return '```json\n[{"bbox_2d": [0, 0, 500, 500], "label": "person"}, {"bbox_2d": [10, 10, 20, 20], "label": "unicorn"}]\n```', True
    if instruction.startswith("Locate"):
        return ('[{"bbox_2d": [100, 125, 750, 750]}]', True) if h % 3 else ('[{"bbox_2d": [100, 125, 750', False)
    if instruction.startswith("Outline"):
        return "[[0, 0], [500, 0], [500, 1000], [0, 1000]]", True
    if "16x16" in instruction:
        return "0101\n1010", True  # fixture 的图只有 12×9，小于 16 格，回填会拒绝；这里走解析失败的路径
    if instruction.startswith("How many"):
        return str(2 + h % 2), True
    return ("(A)", True) if h % 2 else ("B is closer" if "closer" in instruction else "B", True)


class FakeAdapter:
    family = registry.FAMILY_QWEN35
    model_key = "fake(qwen3.5-0.8b)"
    device = "cpu"
    parse_boxes = staticmethod(qwen35.parse_boxes)
    parse_one_box = staticmethod(qwen35.parse_one_box)

    def __init__(self, fail_after_batches: int | None = None):
        self.fail_after = fail_after_batches
        self.n_batches = 0
        self.calls: list[tuple[str, int, bool]] = []

    def build_inputs(self, images, instruction, order):
        assert order == "image_first"
        return FakeInputs(instruction, images[0].size)

    def warmup(self, images=(), *, batch_size=1, **kw):
        return 0.0

    def _rows(self, inputs, enable_thinking, max_new_tokens, where):
        if self.fail_after is not None and self.n_batches >= self.fail_after:
            raise Interrupt("模拟中断")
        self.n_batches += 1
        self.calls.append((where, len(inputs), enable_thinking))
        out = []
        for inp in inputs:
            text, stopped = respond(inp.instruction, inp.size)
            n = max_new_tokens if not stopped else 1 + len(text) % 7
            out.append(GenerationResult(text=text, token_ids=tuple(range(n)), n_generated_tokens=n, n_prompt_tokens=40,
                                        n_visual_tokens=12, wall_time_s=0.5 / len(inputs), raw_text=text + "<|im_end|>",
                                        stopped=stopped, n_image_pixels=12 * 1024, batch_size=len(inputs),
                                        batch_wall_time_s=0.5))
        return out

    def generate(self, inputs, *, enable_thinking, max_new_tokens, **kw):
        assert kw["do_sample"] is False and kw["num_beams"] == 1
        return self._rows([inputs], enable_thinking, max_new_tokens, "generate")[0]

    def generate_batch(self, inputs_list, *, enable_thinking, max_new_tokens, **kw):
        return self._rows(list(inputs_list), enable_thinking, max_new_tokens, "generate_batch")


def spec(task, subset="eval", bs=2, limit=None):
    e = registry.get_model("qwen3.5-0.8b")
    return R.RunSpec(model_key=e.key, family=e.family, repo=e.repo, revision=e.revision, adapter_model_key="fake",
                     task=task, subset=subset, policy_tag="default", policy={"type": "PixelBudget", "pixel_budget_tokens": 300},
                     batch_size=bs, limit=limit, tiny=True, map_ci_resamples=3)


@pytest.fixture(scope="module")
def paths(tmp_path_factory):
    return DataPaths.fixtures(nyu_cache=tmp_path_factory.mktemp("nyu"))


def leaf(out, task, subset="eval"):
    return R.leaf_dir(out, "qwen3.5-0.8b", task, subset, "default")


@pytest.mark.parametrize("task,subset", [(t, "eval") for t in TASKS] + [("ovd", "dev"), ("count", "think")])
def test_end_to_end_files_and_resume_bytes(task, subset, paths, tmp_path):
    sel = select(task, subset, paths)
    assert sel.samples
    full = tmp_path / "full"
    m = R.run_task(FakeAdapter(), spec(task, subset), sel, PROMPTS, paths, full, log=lambda _: None)
    lf = leaf(full, task, subset)
    for f in FILES:
        assert (lf / f).exists(), f
    recs = R.read_jsonl_tolerant(lf / "predictions.jsonl", repair=False)
    assert [r["id"] for r in recs] == [s.id for s in sel.samples] and m["n"] == len(recs) and m["complete"]
    for k in ("id", "raw_text", "text", "failure", "parsed", "truncated", "thinking_unclosed", "n_generated_tokens",
              "n_visual_tokens", "n_image_pixels", "instruction"):
        assert all(k in r for r in recs), k
    assert m["prompt_config_sha256"] == PROMPTS.sha256
    assert m["generation"]["enable_thinking"] is (subset == "think")
    assert m["generation"]["max_new_tokens"] == PROMPTS.max_new_tokens(task, "qwen3.5", subset)
    comp = json.loads((lf / "compute.json").read_text())
    assert comp["n_samples"] == len(recs) and comp["generated_tokens"] == sum(r["n_generated_tokens"] for r in recs)
    assert comp["image_pixels"] == sum(r["n_image_pixels"] for r in recs)
    if len(sel.samples) <= 2:
        return  # 只有一批，没有"中途"可断
    # 第 1 批之后中断，再续跑：结果与一次跑完逐字节相同
    part = tmp_path / "part"
    with pytest.raises(Interrupt):
        R.run_task(FakeAdapter(fail_after_batches=1), spec(task, subset), sel, PROMPTS, paths, part, log=lambda _: None)
    lp = leaf(part, task, subset)
    assert len(R.read_jsonl_tolerant(lp / "predictions.jsonl", repair=False)) == 2
    with pytest.raises(FileExistsError):  # 不加 --resume 不覆盖
        R.run_task(FakeAdapter(), spec(task, subset), sel, PROMPTS, paths, part, log=lambda _: None)
    fake = FakeAdapter()
    R.run_task(fake, spec(task, subset), sel, PROMPTS, paths, part, resume=True, log=lambda _: None)
    assert sum(n for _, n, _ in fake.calls) == len(sel.samples) - 2  # 跳过已完成的
    for f in ("predictions.jsonl", "metrics.json", "config.yaml", "README.md"):
        assert (lp / f).read_bytes() == (lf / f).read_bytes(), f
    assert len(list((lp / "logs").glob("run_*.json"))) == 2


def test_resume_repairs_partial_line_and_allows_other_batch_size(paths, tmp_path):
    sel = select("refexp_box", "eval", paths)
    R.run_task(FakeAdapter(), spec("refexp_box"), sel, PROMPTS, paths, tmp_path / "a", log=lambda _: None)
    ref = (leaf(tmp_path / "a", "refexp_box") / "predictions.jsonl").read_bytes()
    out = tmp_path / "b"
    R.run_task(FakeAdapter(), spec("refexp_box"), sel, PROMPTS, paths, out, log=lambda _: None)
    p = leaf(out, "refexp_box") / "predictions.jsonl"
    lines = p.read_bytes().splitlines(keepends=True)
    p.write_bytes(b"".join(lines[:7]) + lines[7][:25])  # 7 条完整 + 半行
    fake = FakeAdapter()
    R.run_task(fake, spec("refexp_box", bs=3), sel, PROMPTS, paths, out, resume=True, log=lambda _: None)
    assert p.read_bytes() == ref and sum(n for _, n, _ in fake.calls) == len(sel.samples) - 7
    assert {n for _, n, _ in fake.calls} <= {1, 2, 3}


def test_resume_refuses_changed_identity(paths, tmp_path):
    sel = select("count", "eval", paths)
    R.run_task(FakeAdapter(), spec("count"), sel, PROMPTS, paths, tmp_path, log=lambda _: None)
    other = R.RunSpec(**{**spec("count").__dict__, "map_ci_resamples": 7})
    with pytest.raises(ValueError, match="map_ci_resamples"):
        R.run_task(FakeAdapter(), other, sel, PROMPTS, paths, tmp_path, resume=True, log=lambda _: None)
    sel.samples = sel.samples[:1]
    with pytest.raises(ValueError, match="data"):
        R.run_task(FakeAdapter(), spec("count"), sel, PROMPTS, paths, tmp_path, resume=True, log=lambda _: None)


def test_batch_size_one_uses_generate(paths, tmp_path):
    sel = select("spatial_embspatial", "eval", paths)
    fake = FakeAdapter()
    R.run_task(fake, spec("spatial_embspatial", bs=1), sel, PROMPTS, paths, tmp_path, log=lambda _: None)
    assert {w for w, _, _ in fake.calls} == {"generate"}
    fake2 = FakeAdapter()
    R.run_task(fake2, spec("spatial_embspatial", bs=4), sel, PROMPTS, paths, tmp_path / "x", log=lambda _: None)
    assert {w for w, _, _ in fake2.calls} == {"generate_batch"}


def test_estimate_task_with_fake_adapter(paths):
    out = EST.estimate_task(FakeAdapter(), "count", "qwen3.5", PROMPTS, paths, batch_sizes=[1, 2], n_dev=4, think=True,
                            n_think=1, policy_tag="default", log=lambda _: None)
    assert [r["batch_size"] for r in out["by_batch_size"]] == [1, 2] and all(r["ok"] for r in out["by_batch_size"])
    n_eval = len(select("count", "eval", paths).samples)
    assert out["n"]["eval"] == n_eval and out["best_batch_size"] in (1, 2)
    sps = out["samples_per_s"]
    assert out["gpu_hours"]["eval"] == pytest.approx(n_eval / sps / 3600)
    t = out["think_measurement"]
    assert t["enable_thinking"] is True and t["max_new_tokens"] == 4096
    assert out["gpu_hours"]["think8192"] == pytest.approx(out["n"]["think8192"] / t["samples_per_s"] / 3600 * 2)
    md = EST.summary_markdown("m", "default", [out], {"x": 1})
    assert "| count |" in md and "合计" in md
    assert EST.best_batch([{"ok": True, "mem_ok": False, "samples_per_s": 9, "batch_size": 8},
                           {"ok": True, "mem_ok": True, "samples_per_s": 3, "batch_size": 4},
                           {"ok": False, "oom": True, "batch_size": 16}])["batch_size"] == 4
