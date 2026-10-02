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


class FakeTokenizer:
    """只认单个字母与"空格 + 字母"两种写法的分词器（受限字母读数的字母表用 encode / decode；DEV-0017）：
    "A" → 1065、" A" → 2065；其余写法拆成逐字符的多个 token。"""

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        if len(text) == 1 and text.isalpha():
            return [1000 + ord(text)]
        if len(text) == 2 and text[0] == " " and text[1].isalpha():
            return [2000 + ord(text[1])]
        return [ord(c) for c in text]

    def decode(self, ids):
        (i,) = ids
        return chr(i - 1000) if i < 2000 else " " + chr(i - 2000)


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
        self.allowed: list[list[int] | None] = []  # 每批的 allowed_token_ids
        self.max_new: list[int] = []
        self.tokenizer = FakeTokenizer()

    def build_inputs(self, images, instruction, order):
        assert order == "image_first"
        return FakeInputs(instruction, images[0].size)

    def warmup(self, images=(), *, batch_size=1, **kw):
        return 0.0

    def _rows(self, inputs, enable_thinking, max_new_tokens, where, allowed=None):
        if self.fail_after is not None and self.n_batches >= self.fail_after:
            raise Interrupt("模拟中断")
        self.n_batches += 1
        self.calls.append((where, len(inputs), enable_thinking))
        self.allowed.append(None if allowed is None else list(allowed))
        self.max_new.append(max_new_tokens)
        out = []
        for inp in inputs:
            if allowed is not None:  # 受限字母读数：在允许的 token 里确定地挑一个，1 个 token、没有停止 token
                h = int(hashlib.sha256(f"{inp.instruction}|{inp.size}".encode()).hexdigest(), 16)
                tid = sorted(allowed)[h % len(allowed)]
                text = self.tokenizer.decode([tid])
                out.append(GenerationResult(text=text, token_ids=(tid,), n_generated_tokens=1, n_prompt_tokens=40,
                                            n_visual_tokens=12, wall_time_s=0.5 / len(inputs), raw_text=text, stopped=False,
                                            n_image_pixels=12 * 1024, batch_size=len(inputs), batch_wall_time_s=0.5))
                continue
            text, stopped = respond(inp.instruction, inp.size)
            n = max_new_tokens if not stopped else 1 + len(text) % 7
            out.append(GenerationResult(text=text, token_ids=tuple(range(n)), n_generated_tokens=n, n_prompt_tokens=40,
                                        n_visual_tokens=12, wall_time_s=0.5 / len(inputs), raw_text=text + "<|im_end|>",
                                        stopped=stopped, n_image_pixels=12 * 1024, batch_size=len(inputs),
                                        batch_wall_time_s=0.5))
        return out

    def generate(self, inputs, *, enable_thinking, max_new_tokens, allowed_token_ids=None, **kw):
        assert kw["do_sample"] is False and kw["num_beams"] == 1
        return self._rows([inputs], enable_thinking, max_new_tokens, "generate", allowed_token_ids)[0]

    def generate_batch(self, inputs_list, *, enable_thinking, max_new_tokens, allowed_token_ids=None, **kw):
        return self._rows(list(inputs_list), enable_thinking, max_new_tokens, "generate_batch", allowed_token_ids)


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


def test_resume_repairs_partial_line_trims_to_batch_boundary_and_refuses_other_batch_size(paths, tmp_path):
    """DEV-0017：批大小是运行身份的一部分（换了就拒绝续跑）；已完成的条数不在批的边界上时，退回到那一批的开头整批重跑，
    批组成与一次跑完相同（逐字节相同）。"""
    sel = select("refexp_box", "eval", paths)
    R.run_task(FakeAdapter(), spec("refexp_box"), sel, PROMPTS, paths, tmp_path / "a", log=lambda _: None)
    ref = (leaf(tmp_path / "a", "refexp_box") / "predictions.jsonl").read_bytes()
    out = tmp_path / "b"
    R.run_task(FakeAdapter(), spec("refexp_box"), sel, PROMPTS, paths, out, log=lambda _: None)
    p = leaf(out, "refexp_box") / "predictions.jsonl"
    lines = p.read_bytes().splitlines(keepends=True)
    p.write_bytes(b"".join(lines[:7]) + lines[7][:25])  # 7 条完整 + 半行；批大小 2 的边界是 6
    with pytest.raises(ValueError, match="batch_size"):
        R.run_task(FakeAdapter(), spec("refexp_box", bs=3), sel, PROMPTS, paths, out, resume=True, log=lambda _: None)
    fake = FakeAdapter()
    R.run_task(fake, spec("refexp_box"), sel, PROMPTS, paths, out, resume=True, log=lambda _: None)
    assert p.read_bytes() == ref and sum(n for _, n, _ in fake.calls) == len(sel.samples) - 6
    assert {n for _, n, _ in fake.calls} <= {1, 2}
    runs = sorted((leaf(out, "refexp_box") / "logs").glob("run_*.json"))
    assert json.loads(runs[-1].read_text())["trimmed_to_batch_boundary"] == {"from": 7, "to": 6}


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


def test_count_with_countbenchqa_questions(paths, tmp_path):
    """DEV-0015 §4：--count-questions countbenchqa 时指令用人工问句，问句与来源进 predictions、config 与 metrics 分档；
    问句来源与样本不一致时拒绝运行；两种来源的输出不能混着续跑。"""
    from src.e1.prompts import COUNTBENCHQA_FIELD, DEFAULT_COUNT_QUESTIONS

    assert DEFAULT_COUNT_QUESTIONS == "countbenchqa" and spec("count").count_questions == "countbenchqa"  # D36-2
    sel = select("count", "eval", paths, count_questions="countbenchqa")
    sp = R.RunSpec(**{**spec("count").__dict__, "count_questions": "countbenchqa"})
    heur = R.RunSpec(**{**spec("count").__dict__, "count_questions": "heuristic"})
    plain_sel = select("count", "eval", paths, count_questions="heuristic")
    m = R.run_task(FakeAdapter(), sp, sel, PROMPTS, paths, tmp_path / "qa", log=lambda _: None)
    lf = leaf(tmp_path / "qa", "count")
    recs = R.read_jsonl_tolerant(lf / "predictions.jsonl", repair=False)
    for s, r in zip(sel.samples, recs):
        q = s.fields[COUNTBENCHQA_FIELD]
        assert r["instruction"] == q + " Answer with a single number." and r["extras"]["question"] == q
    assert {r["extras"]["object_rule"] for r in recs} <= {"countbenchqa", "countbenchqa_free_form"}
    assert set(m["breakdown"]["object_rule"]) == {r["extras"]["object_rule"] for r in recs}
    cfg = (lf / "config.yaml").read_text()
    assert "count_questions: countbenchqa" in cfg and sel.stats["count_questions"]["sha256"] in cfg
    assert m["count_questions"] == {"source": "countbenchqa", "note": PROMPTS.countbenchqa_note}  # D36-2：注明出处
    assert PROMPTS.countbenchqa_note in (lf / "README.md").read_text()
    with pytest.raises(ValueError, match="问句来源不一致"):  # 样本带人工问句，RunSpec 却是 heuristic
        R.run_task(FakeAdapter(), heur, sel, PROMPTS, paths, tmp_path / "x", log=lambda _: None)
    with pytest.raises(ValueError, match="问句来源不一致"):
        R.run_task(FakeAdapter(), sp, plain_sel, PROMPTS, paths, tmp_path / "y", log=lambda _: None)
    with pytest.raises(ValueError, match="续跑的配置与已有的不同"):
        R.run_task(FakeAdapter(), heur, plain_sel, PROMPTS, paths, tmp_path / "qa", resume=True, log=lambda _: None)
    plain = R.run_config(heur, plain_sel, PROMPTS, paths)
    assert plain["prompt"]["count_questions"] == "heuristic"
    hm = R.run_task(FakeAdapter(), heur, plain_sel, PROMPTS, paths, tmp_path / "h", log=lambda _: None)
    assert hm["count_questions"] == {"source": "heuristic", "note": None}
    assert "count_questions" not in R.run_config(spec("grid16"), select("grid16", "eval", paths), PROMPTS, paths)["prompt"]


def test_metrics_and_readme_carry_failure_diagnostics(paths, tmp_path):
    sel = select("refexp_box", "eval", paths)
    m = R.run_task(FakeAdapter(), spec("refexp_box"), sel, PROMPTS, paths, tmp_path, log=lambda _: None)
    d = m["diagnostics"]
    assert d["failure_detail"]["rule_version"] == 1 and d["think_marker_when_off"]["n"] == 0
    readme = (leaf(tmp_path, "refexp_box") / "README.md").read_text()
    assert "## 失败的诊断" in readme and "关思考时自发写出思考标记：0 条" in readme
    t = R.run_task(FakeAdapter(), spec("count", "think"), select("count", "think", paths), PROMPTS, paths, tmp_path / "t",
                   log=lambda _: None)
    assert t["diagnostics"]["think_marker_when_off"] is None  # 开思考：不适用


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
