"""DEV-0028（D59-5）：思维链的预算强制收尾在 E5 运行器里的接法，用实现了固定签名的假适配器（真实路径由 DEV-0027 实现，
合入后由管理员合并验证）：

    adapter.generate_batch_budgeted(inputs_list, *, thinking_budget, answer_budget, allowed_token_ids=None)
    GenerationResult 另有 forced_close、n_thinking_tokens、n_answer_tokens、n_inserted_tokens、answer_text

- 调用参数取自提示配置（思考 4096、作答 1024，allowed_token_ids = None）；--tiny 的上限同时截两段；
- 解析只看 answer_text（思考段里的草稿网格不看），作答段写满判截断；
- predictions.jsonl 记下五个新字段；metrics 按自然收尾与强制收尾分开汇总并给条数；config 与 README 写明口径；
- 零样本仍走 generate_batch；适配器没有 generate_batch_budgeted 时思维链明确报错；续跑逐字节相同。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pytest
import yaml

from src.e5 import runner as RN
from src.e5.prompts import load_prompt_config
from src.grid.codec import relief_grid_to_text
from src.synth.relief import RELIEF_TARGET_KEY


@dataclass
class _Res:
    text: str
    raw_text: str
    stopped: bool
    n_generated_tokens: int
    token_ids: tuple = ()
    n_prompt_tokens: int = 50
    n_visual_tokens: int = 16
    n_image_pixels: int = 307200
    thinking_unclosed: bool = False
    wall_time_s: float = 0.01
    batch_wall_time_s: float = 0.02
    batch_size: int = 1


@dataclass
class _BudgetRes(_Res):
    forced_close: bool = False
    n_thinking_tokens: int = 0
    n_answer_tokens: int = 0
    n_inserted_tokens: int = 0
    answer_text: str = ""


class _Fake:
    """按行号轮流给出四种情形：0 强制收尾 + 完美网格；1 自然收尾 + 完美网格；2 强制收尾 + 作答段写满；3 自然收尾 + 没有网格。
    思考段里总放一张草稿网格：若解析看了全文，就会判 MULTIPLE_BLOCKS。"""

    def __init__(self, answers: dict[str, str]):
        self.answers = answers  # 提示原文 → 完美的起伏网格文本
        self.calls: list[dict] = []
        self.row = 0

    def warmup(self, images, batch_size):
        return 0.0

    def build_inputs(self, images, text, order):
        return {"text": text, "n_images": len(images), "order": order}

    def generate_batch_budgeted(self, inputs_list, *, thinking_budget: int, answer_budget: int, allowed_token_ids=None):
        self.calls.append({"n": len(inputs_list), "thinking_budget": thinking_budget, "answer_budget": answer_budget,
                           "allowed_token_ids": allowed_token_ids})
        out = []
        for inp in inputs_list:
            k = self.row % 4
            self.row += 1
            perfect = self.answers[inp["text"]]
            draft = relief_grid_to_text(np.full((32, 32), 5, np.uint8))
            answer = {0: perfect, 1: perfect, 2: perfect[:40], 3: "I am not sure."}[k]
            forced = k in (0, 2)
            n_th = thinking_budget if forced else 100 + k
            n_an = answer_budget if k == 2 else 300
            ins = 2 if forced else 0
            out.append(_BudgetRes(text=answer, raw_text=f"<think>{draft}</think>{answer}", stopped=k != 2,
                                  n_generated_tokens=n_th + n_an, token_ids=tuple(range(n_th + n_an)), forced_close=forced,
                                  n_thinking_tokens=n_th, n_answer_tokens=n_an, n_inserted_tokens=ins, answer_text=answer))
        return out


class _PlainFake:
    def __init__(self):
        self.calls = []

    def warmup(self, images, batch_size):
        return 0.0

    def build_inputs(self, images, text, order):
        return {"text": text}

    def generate_batch(self, inputs, *, enable_thinking, max_new_tokens, **decoding):
        self.calls.append({"enable_thinking": enable_thinking, "max_new_tokens": max_new_tokens, **decoding})
        return [_Res(text="nope", raw_text="nope", stopped=True, n_generated_tokens=1, token_ids=(1,)) for _ in inputs]


def _spec(system, task, **kw):
    return RN.RunSpec(system=system, model_key="fake-model", task=task, subset="dev", n_per_bin=2, n_resamples=20,
                      family="qwen", repo="x/y", revision="r", adapter_model_key="fake", policy={"type": "Fixed"},
                      batch_size=3, **kw)


def _answers(spec, prompts):
    out = {}
    for it in RN.spec_items(spec):
        s = RN.make_sample(it)
        txt = prompts.render(spec.system, spec.task, s.instruction, s.meta["target_description"])
        out[txt] = relief_grid_to_text(s.extras[RELIEF_TARGET_KEY])
    return out


@pytest.mark.parametrize("task", ["rotation3d", "viewpoint"])
def test_budgeted_cot_end_to_end(task, tmp_path):
    prompts = load_prompt_config()
    spec = _spec("cot", task)
    fake = _Fake(_answers(spec, prompts))
    m = RN.run_model_task(fake, spec, prompts, tmp_path, warmup=True)
    assert fake.calls and all(c["thinking_budget"] == 4096 and c["answer_budget"] == 1024 and c["allowed_token_ids"] is None
                              for c in fake.calls)
    leaf = RN.leaf_dir(tmp_path, spec)
    recs = [json.loads(x) for x in (leaf / "predictions.jsonl").read_text().splitlines()]
    assert len(recs) == 8
    for i, r in enumerate(recs):
        k = i % 4
        assert r["forced_close"] is (k in (0, 2)) and r["n_inserted_tokens"] == (2 if k in (0, 2) else 0)
        assert r["answer_text"] and "<think>" in r["raw_text"] and r["thinking_unclosed"] is False
        if k in (0, 1):  # 完美网格：只解析作答段，思考段里的草稿不影响
            assert r["valid"] and r["score"]["iou_grid"] == 1.0 and r["score"]["relief"]["reason"] in (None, "flat_target")
        elif k == 2:
            assert r["failure"] == "truncated" and r["truncated"] is True
        else:
            assert r["failure"] == "no_grid"
    b = m["budget"]
    assert (b["n"], b["n_forced"], b["n_natural"]) == (8, 4, 4) and b["forced_rate"] == 0.5
    assert b["natural"]["n"] == 4 and b["forced"]["n"] == 4
    assert b["natural"]["thinking_tokens"]["max"] == 103 and b["forced"]["thinking_tokens"]["max"] == 4096
    assert b["forced"]["overall"]["mask_iou_grid"]["strict"] == pytest.approx(0.5)  # 强制收尾：一条完美、一条截断
    assert b["natural"]["overall"]["mask_iou_grid"]["strict"] == pytest.approx(0.5)  # 自然收尾：一条完美、一条没有网格
    assert b["inserted_tokens"]["total"] == 8
    assert m["parse"]["failures"] == {"no_grid": 2, "truncated": 2}
    assert m["gate"]["judgment"]["status"] == "evaluated"  # 思维链给起伏：a、b 的判定条件适用（D59-4）
    cfg = yaml.safe_load((leaf / "config.yaml").read_text())
    assert cfg["generation"]["mode"] == "budgeted" and cfg["generation"]["thinking_budget"] == 4096
    assert cfg["generation"]["answer_budget"] == 1024 and "max_new_tokens" not in cfg["generation"]
    assert cfg["batching"]["path"] == "generate_batch_budgeted"
    assert cfg["prompt"]["format"] == {"rotation3d": "relief_image_cot", "viewpoint": "relief_image_view_cot"}[task]
    assert cfg["gate"]["convex_prior"]["amplitude"] == {"rotation3d": 2, "viewpoint": 1}[task]
    readme = (leaf / "README.md").read_text()
    assert "预算强制收尾" in readme and "自然收尾 4 条，强制收尾 4 条" in readme and "已作废" in readme
    # 续跑：删掉最后两条与半行，--resume 后逐字节相同
    pred, metrics = (leaf / "predictions.jsonl").read_bytes(), (leaf / "metrics.json").read_bytes()
    lines = pred.splitlines(keepends=True)
    (leaf / "predictions.jsonl").write_bytes(b"".join(lines[:6]) + lines[6][:10])
    fake2 = _Fake(fake.answers)
    fake2.row = 6
    RN.run_model_task(fake2, spec, prompts, tmp_path, resume=True)
    assert (leaf / "predictions.jsonl").read_bytes() == pred and (leaf / "metrics.json").read_bytes() == metrics


def test_tiny_override_caps_both_budgets_and_plain_path_unchanged(tmp_path):
    prompts = load_prompt_config()
    spec = _spec("cot", "occlusion", tiny=True, max_new_tokens_override=48, limit=2)
    assert RN.budgets(spec, prompts) == (48, 48)
    g = RN.generation_config(spec, prompts)
    assert g["mode"] == "budgeted" and (g["thinking_budget"], g["answer_budget"]) == (48, 48)
    # 零样本：generate_batch，关思考，max_new_tokens 照旧
    z = _spec("zeroshot", "rotation3d", limit=2)
    fake = _PlainFake()
    m = RN.run_model_task(fake, z, prompts, tmp_path)
    assert fake.calls[0]["enable_thinking"] is False and fake.calls[0]["max_new_tokens"] == 2048
    assert "budget" not in m
    cfg = yaml.safe_load((RN.leaf_dir(tmp_path, z) / "config.yaml").read_text())
    assert cfg["generation"]["mode"] == "plain" and cfg["batching"]["path"] == "generate_batch"
    # 适配器没有 generate_batch_budgeted：思维链明确报错（不静默退回 generate_batch）
    with pytest.raises(RuntimeError, match="generate_batch_budgeted"):
        RN.run_model_task(_PlainFake(), _spec("cot", "occlusion", limit=2), prompts, tmp_path / "x")


def test_budgeted_result_must_carry_the_new_fields(tmp_path):
    prompts = load_prompt_config()
    spec = _spec("cot", "occlusion", limit=1)

    class _Old(_Fake):
        def generate_batch_budgeted(self, inputs_list, **kw):
            return [_Res(text="x", raw_text="x", stopped=True, n_generated_tokens=0) for _ in inputs_list]

    with pytest.raises(RuntimeError, match="forced_close"):
        RN.run_model_task(_Old({}), spec, prompts, tmp_path)
