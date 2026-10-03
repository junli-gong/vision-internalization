"""DEV-0028（D59-5）：思维链的预算强制收尾在 E5 运行器里的接法；DEV-0029：按思考段的结束方式 thinking_end 四类分开报。

    adapter.generate_batch_budgeted(inputs_list, *, thinking_budget, answer_budget, allowed_token_ids=None)
    GenerationResult 另有 thinking_end、forced_close、n_thinking_tokens、n_answer_tokens、n_inserted_tokens、answer_text

- 假适配器：调用参数取自提示配置（思考 4096、作答 1024，allowed_token_ids = None）；--tiny 的上限同时截两段；
  解析只看 answer_text（思考段里的草稿网格不看），作答段写满判截断，unclosed 判 thinking_unclosed；predictions.jsonl 记下新字段；
  metrics 与 README 按 thinking_end 四类分开报条数与读数；config 写明口径；零样本仍走 generate_batch；适配器没有
  generate_batch_budgeted 时思维链明确报错；续跑逐字节相同。
- 真实小模型（三个家族）：复用 DEV-0027 的前向钩子（tests/test_adapter_budget.py 的 _Steer）在同一次运行里造出全部结束方式
  （Qwen 的思考开头在提示里，没有 not_opened；Gemma、InternVL 四类都有），核对记录、解析与按类汇总。
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from src.adapters import registry
from src.adapters.base import THINKING_ENDS
from src.adapters.factory import default_image_policy, make_adapter
from src.e5 import runner as RN
from src.e5.prompts import load_prompt_config
from src.grid.codec import relief_grid_to_text
from src.synth.relief import RELIEF_TARGET_KEY
from tests.test_adapter_budget import INIT_STD, STOP_AFTER, _plen, steering


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
    thinking_end: str = "natural"
    forced_close: bool = False
    n_thinking_tokens: int = 0
    n_answer_tokens: int = 0
    n_inserted_tokens: int = 0
    answer_text: str = ""


# 按行号轮流的六种情形：(thinking_end, 作答段)
CASES = [("forced", "perfect"), ("natural", "perfect"), ("forced", "truncated"), ("natural", "no_grid"),
         ("not_opened", "perfect"), ("unclosed", None)]


class _Fake:
    """按行号轮流给出 CASES 的六种情形（四种结束方式都有）。思考段里总放一张草稿网格：若解析看了全文，就会判 MULTIPLE_BLOCKS。
    unclosed 的行与适配器一样：作答段为空、stopped = True、适配器的 thinking_unclosed = True。"""

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
            k = self.row % len(CASES)
            self.row += 1
            end, ans = CASES[k]
            perfect = self.answers[inp["text"]]
            draft = relief_grid_to_text(np.full((32, 32), 5, np.uint8))
            answer = {"perfect": perfect, "truncated": perfect[:40], "no_grid": "I am not sure.", None: ""}[ans]
            forced = end == "forced"
            n_th = {"forced": thinking_budget, "not_opened": 0, "unclosed": 7}.get(end, 100 + k)
            n_an = {"truncated": answer_budget, None: 0}.get(ans, 300)
            ins = 2 if forced else 0
            raw = answer if end == "not_opened" else f"<think>{draft}</think>{answer}" if end != "unclosed" else f"<think>{draft}"
            out.append(_BudgetRes(text="" if end == "unclosed" else answer, raw_text=raw, stopped=ans != "truncated",
                                  thinking_unclosed=end == "unclosed", n_generated_tokens=n_th + n_an,
                                  token_ids=tuple(range(n_th + n_an)), thinking_end=end, forced_close=forced,
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
        end, ans = CASES[i % len(CASES)]
        assert r["thinking_end"] == end and r["forced_close"] is (end == "forced")
        assert r["n_inserted_tokens"] == (2 if end == "forced" else 0)
        assert r["thinking_unclosed"] is (end == "unclosed") and ("<think>" in r["raw_text"]) is (end != "not_opened")
        if ans == "perfect":  # 完美网格：只解析作答段，思考段里的草稿不影响
            assert r["valid"] and r["score"]["iou_grid"] == 1.0 and r["score"]["relief"]["reason"] in (None, "flat_target")
        elif ans == "truncated":
            assert r["failure"] == "truncated" and r["truncated"] is True
        elif ans == "no_grid":
            assert r["failure"] == "no_grid"
        else:  # unclosed：思考段里就停了，没有作答段 → thinking_unclosed（不算截断）
            assert r["failure"] == "thinking_unclosed" and r["truncated"] is False and r["answer_text"] == ""
    assert "budget" not in m
    te = m["thinking_end"]
    assert te["n"] == 8 and te["counts"] == {"natural": 3, "forced": 3, "not_opened": 1, "unclosed": 1}
    assert list(te["counts"]) == ["natural", "forced", "not_opened", "unclosed"] and set(te["by_kind"]) == set(te["counts"])
    bk = te["by_kind"]
    assert all(bk[k]["n"] == n for k, n in te["counts"].items())
    assert bk["natural"]["thinking_tokens"]["max"] == 103 and bk["forced"]["thinking_tokens"]["max"] == 4096
    assert bk["not_opened"]["thinking_tokens"]["max"] == 0 and bk["unclosed"]["answer_tokens"]["max"] == 0
    assert bk["forced"]["overall"]["mask_iou_grid"]["strict"] == pytest.approx(2 / 3)  # 两条完美、一条截断
    assert bk["natural"]["overall"]["mask_iou_grid"]["strict"] == pytest.approx(2 / 3)  # 两条完美、一条没有网格
    assert bk["not_opened"]["overall"]["mask_iou_grid"]["strict"] == 1.0
    assert bk["unclosed"]["overall"]["mask_iou_grid"]["strict"] == 0.0 and bk["unclosed"]["overall"]["mask_iou_grid"]["valid_rate"] == 0.0
    assert bk["forced"]["parse"]["failures"] == {"truncated": 1} and bk["forced"]["n_truncated"] == 1
    assert bk["unclosed"]["parse"]["failures"] == {"thinking_unclosed": 1} and bk["unclosed"]["n_truncated"] == 0
    assert bk["natural"]["parse_failure_rate"] == pytest.approx(1 / 3) and bk["not_opened"]["parse_failure_rate"] == 0.0
    assert te["inserted_tokens"]["total"] == 6
    assert m["parse"]["failures"] == {"no_grid": 1, "thinking_unclosed": 1, "truncated": 1}
    assert m["thinking_unclosed"]["n"] == 1 and m["truncation"]["n"] == 1
    assert m["gate"]["judgment"]["status"] == "evaluated"  # 思维链给起伏：a、b 的判定条件适用（D59-4）
    cfg = yaml.safe_load((leaf / "config.yaml").read_text())
    assert cfg["generation"]["mode"] == "budgeted" and cfg["generation"]["thinking_budget"] == 4096
    assert cfg["generation"]["answer_budget"] == 1024 and "max_new_tokens" not in cfg["generation"]
    assert cfg["batching"]["path"] == "generate_batch_budgeted"
    assert cfg["prompt"]["format"] == {"rotation3d": "relief_image_cot", "viewpoint": "relief_image_view_cot"}[task]
    assert cfg["gate"]["convex_prior"]["amplitude"] == {"rotation3d": 2, "viewpoint": 1}[task]
    readme = (leaf / "README.md").read_text()
    assert "预算强制收尾" in readme and "已作废" in readme
    assert "自然收尾 3 条，强制收尾 3 条，没写思考段（直接作答） 1 条，思考段里就停了 1 条（共 8 条）" in readme
    assert "| 思考段里就停了 | 1 | " in readme and "| 没写思考段（直接作答） | 1 | " in readme
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
    assert "budget" not in m and "thinking_end" not in m
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
    with pytest.raises(RuntimeError, match="thinking_end"):  # thinking_end 不在四类里
        RN.check_budget_fields(_BudgetRes(text="", raw_text="", stopped=True, n_generated_tokens=0, thinking_end="closed"))
    with pytest.raises(RuntimeError, match="不符"):  # forced_close 与 thinking_end 不一致
        RN.check_budget_fields(_BudgetRes(text="", raw_text="", stopped=True, n_generated_tokens=0, thinking_end="natural",
                                          forced_close=True))


# ---------------------------------------------------------------------------
# 真实小模型：四种结束方式都在同一次运行里出现（DEV-0029）
# ---------------------------------------------------------------------------

TINY_BUDGET = 8  # max_new_tokens_override：思考、作答两段都截到 8（钩子的自然收尾在思考段第 off + 4 个 token，要 ≤ 8）
# 每种行的计划 (思考段怎么结束, 作答段怎么结束)，按提示长度轮流分配（规则见 tests/test_adapter_budget._Steer；同一长度的行计划相同）
PLAN_PROMPT_OPEN = [("natural", "stop"), ("forced", "stop"), ("unclosed", None), ("natural", "run"), ("forced", "run")]  # Qwen
PLAN_OUTPUT_OPEN = [("natural", "stop"), ("forced", "stop"), ("unclosed", None), ("not_opened", "stop"), ("forced", "run")]


@pytest.mark.parametrize("key", ["qwen3.5-0.8b", "gemma-4-e2b-it", "internvl3.5-2b-hf"])
def test_real_tiny_model_reports_each_thinking_end(key, tmp_path):
    """真实小模型（与 run_e5.py --tiny 相同的图像口径与重取参数）走 generate_batch_budgeted；DEV-0027 的前向钩子按行造出本家族的
    全部结束方式（Gemma、InternVL 四类；Qwen 的思考开头在提示里，没有 not_opened）。核对：记录的 thinking_end 与计划相同，解析按
    thinking_end 判（unclosed → thinking_unclosed，作答段写满 → truncated），metrics 的 counts / by_kind 与 README 按四类分开报。"""
    entry = registry.get_model(key)
    policy = default_image_policy(entry.family)
    ad = make_adapter(key, device="cpu", dtype=torch.float32, image_policy=policy, tiny=True, init_std=INIT_STD)
    prompts = load_prompt_config()
    task = "rotation3d"  # 起伏任务：按类的读数里也有起伏误差
    spec = RN.RunSpec(system="cot", model_key=key, task=task, subset="dev", n_per_bin=2, n_resamples=20, tiny=True,
                      family=entry.family, repo=entry.repo, revision=entry.revision, adapter_model_key=ad.model_key,
                      policy={"type": type(policy).__name__, **policy.as_dict()}, batch_size=3,
                      max_new_tokens_override=TINY_BUDGET)
    assert RN.budgets(spec, prompts) == (TINY_BUDGET, TINY_BUDGET)
    samples = [RN.make_sample(it) for it in RN.spec_items(spec)]
    inputs = [ad.build_inputs([Image.fromarray(s.image)], prompts.render("cot", task, s.instruction, s.meta["target_description"]),
                              prompts.order) for s in samples]
    fmt = ad.think_format()
    kinds = PLAN_PROMPT_OPEN if fmt.open_id is None else PLAN_OUTPUT_OPEN
    lens = [_plen(ad, i) for i in inputs]
    plan = {p: kinds[k % len(kinds)] for k, p in enumerate(sorted(set(lens)))}
    rows = [plan[p] for p in lens]
    family_ends = {"natural", "forced", "unclosed"} | ({"not_opened"} if fmt.open_id is not None else set())
    assert {e for e, _ in rows} == family_ends, "计划没有覆盖本家族的全部结束方式（样本的提示长度种类不够）"
    with steering(ad, plan):
        m = RN.run_model_task(ad, spec, prompts, tmp_path, warmup=False)
    leaf = RN.leaf_dir(tmp_path, spec)
    recs = [json.loads(x) for x in (leaf / "predictions.jsonl").read_text().splitlines()]
    assert len(recs) == len(rows) == 8
    off = 0 if fmt.open_id is None else 1
    for r, (end, ans), p in zip(recs, rows, lens):
        assert r["n_prompt_tokens"] == p and r["thinking_end"] == end, (r["id"], end, r["thinking_end"])
        assert r["forced_close"] is (end == "forced") and r["thinking_unclosed"] is (end == "unclosed")
        assert r["n_inserted_tokens"] == (len(fmt.insert_ids) if end == "forced" else 0)
        assert r["n_thinking_tokens"] == {"natural": off + 4, "forced": TINY_BUDGET, "unclosed": off + 3, "not_opened": 0}[end]
        assert r["n_thinking_tokens"] + r["n_answer_tokens"] == r["n_generated_tokens"]
        if end == "unclosed":  # 思考段里写出停止 token：没有作答段，判 thinking_unclosed，不算截断
            assert r["n_answer_tokens"] == 0 and r["answer_text"] == "" and r["stopped"] is True and r["truncated"] is False
            assert r["failure"] == "thinking_unclosed"
        elif ans == "run":  # 作答段写满：判截断
            assert r["n_answer_tokens"] == TINY_BUDGET and r["stopped"] is False and r["truncated"] is True
            assert r["failure"] == "truncated"
        else:  # 作答段写出停止 token：随机小模型解析不出网格，但不是截断、也不是思考段未闭合
            assert r["n_answer_tokens"] == STOP_AFTER + 1 and r["stopped"] is True and r["truncated"] is False
            assert r["failure"] not in (None, "truncated", "thinking_unclosed")
    want = Counter(e for e, _ in rows)
    te = m["thinking_end"]
    assert te["n"] == 8 and te["counts"] == {k: want.get(k, 0) for k in THINKING_ENDS}
    assert set(te["by_kind"]) == family_ends and all(te["by_kind"][k]["n"] == want[k] for k in family_ends)
    assert te["by_kind"]["unclosed"]["parse"]["failures"] == {"thinking_unclosed": want["unclosed"]}
    for k in family_ends:
        e = te["by_kind"][k]
        assert e["n_truncated"] == sum(1 for end, ans in rows if end == k and ans == "run")
        assert e["overall"]["mask_iou_grid"]["n"] == want[k] and "relief_error" in e["overall"]
    assert te["inserted_tokens"]["total"] == want["forced"] * len(fmt.insert_ids)
    assert m["thinking_unclosed"]["n"] == want["unclosed"] and m["truncation"]["n"] == sum(1 for _, a in rows if a == "run")
    readme = (leaf / "README.md").read_text()
    for k in THINKING_ENDS:
        assert f"{RN.THINKING_END_LABELS[k]} {want.get(k, 0)} 条" in readme
    if fmt.open_id is None:
        assert "| 没写思考段（直接作答） | 0 | — |" in readme
