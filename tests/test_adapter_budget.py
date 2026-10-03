"""DEV-0027：预算强制收尾的思考生成（generate_batch_budgeted；D59-1、D59-5）。三家族，CPU 小模型（fp32，init_std 0.05）。

小模型的 greedy 输出近似随机，自己几乎不会写出思考结束标记或停止 token。测试用一个前向钩子（_Steer）按"这一行自己的提示长度"
决定在第几步把哪个 token 的 logit 抬到最高（或压到最低）：规则只看这一行自己（提示的非填充长度与这一行已追加的 token），所以批量与
逐条的行为相同；HF generate 与预算循环都调外层模型的 forward，钩子对两者一样生效。于是同一批里可以混合"自然收尾""写满被强制"
"思考段里就停了""没写思考段（Gemma、InternVL）"几种行，作答段又分"写出停止 token"与"写满作答预算"。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.adapters import gemma4, internvl35, qwen35, registry
from src.adapters.base import THINKING_ENDS, GenerationResult
from src.adapters.budget import BudgetError, ThinkFormat, make_think_format
from src.adapters.factory import make_adapter
from src.adapters.image_policy import PixelBudget, SoftTokenBudget, TilePolicy
from src.adapters.training import TrainingModeError, project

REPO = Path(__file__).resolve().parents[1]
FAMILY_KEYS = ("qwen3.5-0.8b", "gemma-4-e2b-it", "internvl3.5-2b-hf")
POLICIES = {
    "qwen3.5-0.8b": PixelBudget(64),
    "gemma-4-e2b-it": SoftTokenBudget(70),
    "internvl3.5-2b-hf": TilePolicy(crop_to_patches=True, max_patches=4),
}
INIT_STD = 0.05
TB, AB = 8, 5  # 思考预算、作答预算（小）
STOP_AFTER = 2  # 作答段"写出停止 token"的行：作答段的第 STOP_AFTER + 1 个 token 是停止 token
MODULES = {registry.FAMILY_QWEN35: qwen35, registry.FAMILY_GEMMA4: gemma4, registry.FAMILY_INTERNVL35: internvl35}


@pytest.fixture(scope="module", params=FAMILY_KEYS)
def ad(request):
    key = request.param
    return make_adapter(key, device="cpu", dtype=torch.float32, image_policy=POLICIES[key], tiny=True, init_std=INIT_STD)


def _img(rng, h, w):
    a = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    a[h // 4: h // 2, w // 4: w // 2, 1] = 230
    return Image.fromarray(a)


@pytest.fixture(scope="module")
def samples():
    rng = np.random.default_rng(20261003)
    land, port, square, wide = _img(rng, 100, 150), _img(rng, 150, 100), _img(rng, 120, 120), _img(rng, 64, 256)
    return [
        ([land], "Describe the cat.", "image_first"),
        ([port], "What is in this picture? Please answer in detail.", "image_first"),
        ([square, wide], "Compare.", "instruction_first"),
        ([], "Say hi.", "instruction_first"),
        ([wide], "Count the objects in the image and output a number.", "image_first"),
        ([land], "x", "image_first"),
        ([square], "Which part of the image is the brightest one, top or bottom?", "image_first"),
        ([port, land], "Which image is brighter?", "instruction_first"),
    ]


@pytest.fixture(scope="module")
def inputs(ad, samples):
    return [ad.build_inputs(ims, ins, order) for ims, ins, order in samples]


def _plen(ad, inp) -> int:
    return len(ad._prompt_row(inp, True).ids)


# 每种行的计划：(思考段怎么结束, 作答段怎么结束)
KINDS_PROMPT_OPEN = [("natural", "stop"), ("forced", "stop"), ("unclosed", None), ("forced", "run"), ("natural", "run")]
KINDS_OUTPUT_OPEN = [("natural", "stop"), ("forced", "stop"), ("unclosed", None), ("not_opened", "stop"), ("forced", "run"),
                     ("natural", "run"), ("not_opened", "run")]


def _plan(ad, inputs, kinds=None) -> dict[int, tuple[str, str | None]]:
    """提示长度 → 计划。按不同的提示长度轮流分配（同一长度的行计划相同；规则只看行自己）。"""
    fmt = ad.think_format()
    kinds = kinds or (KINDS_PROMPT_OPEN if fmt.open_id is None else KINDS_OUTPUT_OPEN)
    lens = sorted({_plen(ad, i) for i in inputs})
    return {p: kinds[k % len(kinds)] for k, p in enumerate(lens)}


class _Steer:
    """外层模型的前向钩子：按行的计划改最后一个位置的 logits（见模块说明）。"""

    def __init__(self, ad, plan):
        self.fmt: ThinkFormat = ad.think_format()
        self.stop = ad.end_of_turn_token_id()
        self.plan = plan
        self.hist: list[list[int]] = []
        self.plen: list[int] = []

    def target(self, b: int):
        p = self.plen[b]
        gen = self.hist[b][p:]
        plan = self.plan.get(p)
        if plan is None:
            return None
        think, answer = plan
        fmt, n = self.fmt, len(gen)
        if fmt.open_id is not None and n == 0:
            return fmt.open_id, ("suppress" if think == "not_opened" else "boost")
        if think == "not_opened":
            return (self.stop, "boost") if answer == "stop" and n == STOP_AFTER else None
        if fmt.close_id in gen:
            j = gen.index(fmt.close_id)
            start = j + (len(fmt.insert_ids) if think == "forced" else 1)
            return (self.stop, "boost") if answer == "stop" and n - start == STOP_AFTER else None
        off = 0 if fmt.open_id is None else 1
        if think == "natural" and n == off + 3:
            return fmt.close_id, "boost"
        if think == "unclosed" and n == off + 2:
            return self.stop, "boost"
        if think == "forced":  # 写满预算之前不许自己收尾
            return fmt.close_id, "suppress"
        return None

    def __call__(self, module, args, kwargs, output):
        ids, mask = kwargs["input_ids"], kwargs["attention_mask"]
        if ids.shape[1] > 1:  # 预填
            self.hist = [ids[b][mask[b, -ids.shape[1]:].bool()].tolist() for b in range(ids.shape[0])]
            self.plen = [len(h) for h in self.hist]
        else:
            for b in range(ids.shape[0]):
                self.hist[b].append(int(ids[b, 0]))
        logits = output.logits
        for b in range(ids.shape[0]):
            t = self.target(b)
            if t is None:
                continue
            tok, mode = t
            row = logits[b, -1]
            row[tok] = (row.max() + 50.0) if mode == "boost" else (row.min() - 50.0)
        return output


class steering:
    def __init__(self, ad, plan):
        self.ad, self.steer = ad, _Steer(ad, plan)

    def __enter__(self):
        self.h = self.ad.model.register_forward_hook(self.steer, with_kwargs=True)
        return self.steer

    def __exit__(self, *exc):
        self.h.remove()


def _full(r: GenerationResult) -> list[int]:
    ids = list(r.token_ids)
    if r.inserted_at is None:
        return ids
    return ids[: r.inserted_at] + list(r.inserted_token_ids) + ids[r.inserted_at:]


def _check_row(ad, r: GenerationResult, plan, inp):
    fmt = ad.think_format()
    stops = set(ad.stop_token_ids())
    off = 0 if fmt.open_id is None else 1
    think, answer = plan
    assert r.thinking_end == think and r.forced_close == (think == "forced") and r.natural_close == (think == "natural")
    assert r.n_thinking_tokens + r.n_answer_tokens == r.n_generated_tokens == len(r.token_ids)
    assert r.n_prompt_tokens == _plen(ad, inp)
    th, an = r.token_ids[: r.n_thinking_tokens], r.token_ids[r.n_thinking_tokens:]
    if fmt.open_id is not None and think != "not_opened":
        assert th[0] == fmt.open_id
    if think == "natural":
        assert r.n_thinking_tokens == off + 4 and th[-1] == fmt.close_id and fmt.close_id not in th[:-1]
        assert r.n_inserted_tokens == 0 and r.inserted_token_ids == () and r.inserted_at is None and r.inserted_span is None
    elif think == "forced":
        assert r.n_thinking_tokens == TB and fmt.close_id not in th and not (set(th) & stops)
        assert r.inserted_token_ids == fmt.insert_ids and r.n_inserted_tokens == len(fmt.insert_ids)
        assert r.inserted_at == TB and r.raw_text[r.inserted_span[0]: r.inserted_span[1]] == fmt.insert_text
        assert r.raw_text.startswith(ad.tokenizer.decode(list(th), skip_special_tokens=False))
    elif think == "unclosed":
        assert r.n_thinking_tokens == off + 3 and th[-1] in stops and r.n_answer_tokens == 0
        assert r.stopped is True and r.answer_text == "" and r.text == "" and r.thinking_unclosed
    else:
        assert think == "not_opened" and r.n_thinking_tokens == 0 and r.token_ids[0] != fmt.open_id
    if think == "unclosed":
        return
    if answer == "stop":
        assert r.n_answer_tokens == STOP_AFTER + 1 and an[-1] in stops and r.stopped is True
    else:
        assert r.n_answer_tokens == AB and not (set(an) & stops) and r.stopped is False
    want = list(an[:-1]) if an and an[-1] in stops else list(an)
    assert r.answer_text == ad.tokenizer.decode(want, skip_special_tokens=False).strip("\n")
    assert r.text == r.answer_text and not r.thinking_unclosed
    text, raw, unc = ad.split_answer(_full(r), enable_thinking=True)
    assert (text, unc) == (r.text, r.thinking_unclosed)
    if not r.forced_close:
        assert r.raw_text == raw


# ---------------------------------------------------------------------------
# 思考段格式：补的标记与本家族的格式一致
# ---------------------------------------------------------------------------


def test_think_format_matches_chat_template_and_observed_outputs(ad):
    fmt = ad.think_format()
    mod = MODULES[ad.family]
    tok = ad.tokenizer
    assert fmt.insert_ids == tuple(ad._encode(fmt.insert_text)) and fmt.insert_ids[0] == fmt.close_id
    records = [json.loads(x) for x in (REPO / "experiments/P0/0005_thinking_probe/records.jsonl").read_text().splitlines()]
    fam_models = {r["model"] for r in records if registry.get_model(r["model"]).family == ad.family}
    closed = [r for r in records if r["model"] in fam_models and r["condition"] == "think_greedy" and r["think_close"]]
    opened = [r for r in records if r["model"] in fam_models and r["condition"] == "think_greedy" and r["think_open"]]
    assert closed and opened
    if ad.family == registry.FAMILY_QWEN35:
        # 关思考的助手头 `<think>\n\n</think>\n\n` 以补的 token 结尾；开思考的助手头以 `<think>\n` 结尾（开头在提示里）
        assert fmt.open_id is None and fmt.insert_text == "</think>\n\n"
        assert ad.assistant_header_ids(False)[-len(fmt.insert_ids):] == fmt.insert_ids
        assert tok.decode(list(ad.assistant_header_ids(True))).endswith("<think>\n")
        hist = tok.apply_chat_template([{"role": "user", "content": "q"},
                                        {"role": "assistant", "content": "ANSWER", "reasoning_content": "REASON"}],
                                       chat_template=ad.chat_template, tokenize=False)
        assert "REASON\n" + fmt.insert_text + "ANSWER" in hist
    elif ad.family == registry.FAMILY_GEMMA4:
        assert fmt.open_text == mod.CHANNEL_OPEN and fmt.insert_text == mod.CHANNEL_CLOSE == "<channel|>"
        rt = tok.response_template["fields"]["thinking"]
        assert rt["open"].startswith(fmt.open_text) and rt["close"] == fmt.insert_text
        hist = tok.apply_chat_template([{"role": "user", "content": "q"},
                                        {"role": "assistant", "content": "ANSWER", "reasoning": "REASON"}],
                                       chat_template=ad._template, tokenize=False, enable_thinking=True)
        assert "<|channel>thought\nREASON\n" + fmt.insert_text + "ANSWER" in hist
    else:
        assert fmt.open_text == mod.THINK_START and fmt.insert_text == "</think>\n\n"
        assert "<think> and </think>" in mod.R1_SYSTEM_PROMPT and "with a newline" in mod.R1_SYSTEM_PROMPT
    # RUN-0005 的开思考输出：写了思考段的都以开头标记开始（Qwen 的开头在提示里），收尾之后紧跟固定分隔
    for r in closed:
        assert fmt.insert_text in r["raw_text"], r["model"]
    if fmt.open_text is not None:
        assert all(r["raw_text"].startswith(fmt.open_text) for r in opened)


def test_make_think_format_rejects_merging_separator():
    enc = {"</t>": [5], "\n": [7], "</t>\n": [9]}.get
    with pytest.raises(ValueError, match="整段分词"):
        make_think_format(lambda s: enc(s), open_text=None, close_text="</t>", separator_text="\n")
    with pytest.raises(ValueError, match="不是单个 token"):
        make_think_format(lambda s: [1, 2], open_text=None, close_text="</t>", separator_text="")


# ---------------------------------------------------------------------------
# 逐行规则、记账、批量与逐条
# ---------------------------------------------------------------------------


def test_mixed_batch_rows_and_accounting(ad, inputs):
    plan = _plan(ad, inputs)
    with steering(ad, plan):
        res = ad.generate_batch_budgeted(inputs, thinking_budget=TB, answer_budget=AB)
    kinds = {plan[_plen(ad, i)][0] for i in inputs}
    want = {"natural", "forced", "unclosed"} | ({"not_opened"} if ad.think_format().open_id is not None else set())
    assert kinds == want, "计划没有覆盖全部的行"
    assert len({r.token_ids[:3] for r in res}) >= 4, "小模型的输出与输入无关"
    for inp, r in zip(inputs, res):
        _check_row(ad, r, plan[_plen(ad, inp)], inp)
        assert r.batch_size == len(inputs) and math.isclose(r.wall_time_s * len(inputs), r.batch_wall_time_s, rel_tol=1e-9)
        assert r.n_visual_tokens == sum(len(i.span) for i in inp.images)
        assert r.n_image_pixels == sum(i.n_pixels for i in inp.images)


def _same_budget_row(s: GenerationResult, b: GenerationResult) -> None:
    keys = ("text", "token_ids", "raw_text", "thinking_unclosed", "stopped", "n_prompt_tokens", "n_visual_tokens",
            "n_image_pixels", "thinking_end", "forced_close", "n_thinking_tokens", "n_answer_tokens", "n_inserted_tokens",
            "inserted_token_ids", "inserted_at", "inserted_span", "answer_text")
    assert {k: getattr(s, k) for k in keys} == {k: getattr(b, k) for k in keys}


def test_batch_matches_single(ad, inputs):
    plan = _plan(ad, inputs)
    with steering(ad, plan):
        batch = ad.generate_batch_budgeted(inputs, thinking_budget=TB, answer_budget=AB)
        single = [ad.generate_batch_budgeted([i], thinking_budget=TB, answer_budget=AB)[0] for i in inputs]
    for s, b in zip(single, batch):
        _same_budget_row(s, b)
        assert s.batch_size == 1


def test_matches_generate_batch_greedy_token_for_token(ad, inputs):
    """预算循环与 HF generate 的 greedy 逐个相同：没有补 token 的行（自然收尾、思考段里就停了、没写思考段）整条与
    generate_batch(enable_thinking=True) 相同（写满作答预算的行是它的前缀）；强制收尾的行思考段与它的前 TB 个 token 相同。"""
    plan = _plan(ad, inputs)
    with steering(ad, plan):
        res = ad.generate_batch_budgeted(inputs, thinking_budget=TB, answer_budget=AB)
        ref = ad.generate_batch(inputs, enable_thinking=True, max_new_tokens=TB + AB + 4)
    for r, g in zip(res, ref):
        if r.forced_close:
            assert r.token_ids[:TB] == g.token_ids[:TB]
        elif r.stopped:
            assert r.token_ids == g.token_ids and g.stopped
        else:
            assert r.token_ids == g.token_ids[: len(r.token_ids)]


def test_answer_after_forced_close_matches_teacher_forcing(ad, inputs):
    """补 token 之后续用 KV cache 的作答段：每个模型生成的 token 都等于"把完整序列（含补的 token）教师强制送进去"时该位置
    logits 的 argmax（全词表；给了 allowed 时在 allowed 里）。只跳过钩子改过 logits 的位置（第一个 token 的开头标记）。"""
    fmt = ad.think_format()
    plan = _plan(ad, inputs, kinds=[("forced", "run")])
    allowed = sorted(set(ad._encode("0 1 2 3 4 5 6 7 8 9")))
    for allow in (None, allowed):
        with steering(ad, plan):
            res = ad.generate_batch_budgeted(inputs, thinking_budget=TB, answer_budget=AB, allowed_token_ids=allow)
        assert all(r.forced_close for r in res)
        with torch.no_grad():
            tf = ad.training_forward(inputs, [_full(r) for r in res], enable_thinking=True)
            logits = project(tf.target_hidden, ad.lm_head_weight(), softcap=ad._logit_softcap()).cpu()
        for b, r in enumerate(res):
            lg = logits[tf.target_offsets[b]: tf.target_offsets[b + 1]]
            full = _full(r)
            for k, t in enumerate(full):
                if TB <= k < TB + r.n_inserted_tokens or (k == 0 and fmt.open_id is not None):
                    continue
                if allow is not None and k >= TB + r.n_inserted_tokens:
                    pick = allow[int(lg[k][allow].argmax())]
                else:
                    pick = int(lg[k].argmax())
                assert pick == t, (b, k)


def test_allowed_token_ids_only_in_answer_segment(ad, inputs):
    """allowed（数字，不含停止 token）只作用于作答段：作答段的 token 都在 allowed 里、写满作答预算；思考段不受限制（有 allowed
    以外的 token），思考段里照样能写出停止 token（unclosed 的行）。没写思考段的行第一步在 allowed ∪ {开头标记} 里取。"""
    fmt = ad.think_format()
    allowed = sorted(set(ad._encode("0 1 2 3 4 5 6 7 8 9")))
    kinds = [("natural", "run"), ("forced", "run"), ("unclosed", None)] + (
        [("not_opened", "run")] if fmt.open_id is not None else [])
    plan = _plan(ad, inputs, kinds=kinds)
    with steering(ad, plan):
        res = ad.generate_batch_budgeted(inputs, thinking_budget=TB, answer_budget=AB, allowed_token_ids=allowed)
        single = [ad.generate_batch_budgeted([i], thinking_budget=TB, answer_budget=AB, allowed_token_ids=allowed)[0]
                  for i in inputs]
    outside = 0
    for inp, r, s in zip(inputs, res, single):
        _same_budget_row(s, r)
        _check_row(ad, r, plan[_plen(ad, inp)], inp)
        th, an = r.token_ids[: r.n_thinking_tokens], r.token_ids[r.n_thinking_tokens:]
        assert set(an) <= set(allowed)
        outside += len(set(th) - set(allowed) - {fmt.close_id, fmt.open_id})
    assert outside > 0, "思考段也被限制了"


def test_gemma_sliding_window_long_thinking_matches_generate_batch(samples):
    """Gemma 小模型把滑窗改成 16：思考段跨过窗口，预算循环（不传 cache_position）与 generate_batch 仍逐个相同，批量 = 逐条。"""
    from src.adapters.factory import reinit_parameters

    comps = gemma4.load_pinned_components()
    model = gemma4.build_tiny_model(comps.config, sliding_window=16, generation_config=comps.generation_config)
    reinit_parameters(model, std=INIT_STD, seed=0)
    g = gemma4.make_tiny_adapter(comps, model=model)
    items = [g.build_inputs(ims, ins, order) for ims, ins, order in samples]
    plan = _plan(g, items, kinds=[("forced", "run")])
    tb, ab = 24, 6
    with steering(g, plan):
        res = g.generate_batch_budgeted(items, thinking_budget=tb, answer_budget=ab)
        ref = g.generate_batch(items, enable_thinking=True, max_new_tokens=tb)
        one = g.generate_batch_budgeted(items[:1], thinking_budget=tb, answer_budget=ab)[0]
    _same_budget_row(one, res[0])
    for r, x in zip(res, ref):
        assert r.forced_close and r.token_ids[:tb] == x.token_ids


# ---------------------------------------------------------------------------
# 入参与结果字段的核对
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tb,ab", [(0, 4), (4, 0), (-1, 4), (True, 4), (4, 1.5), ("4", 4)])
def test_bad_budgets_rejected(ad, inputs, tb, ab):
    with pytest.raises(BudgetError):
        ad.generate_batch_budgeted(inputs[:1], thinking_budget=tb, answer_budget=ab)


def test_other_bad_arguments(ad, inputs):
    with pytest.raises(ValueError, match="allowed_ids 为空"):
        ad.generate_batch_budgeted(inputs[:1], thinking_budget=2, answer_budget=2, allowed_token_ids=[])
    with pytest.raises(TypeError):
        ad.generate_batch_budgeted(inputs[0], thinking_budget=2, answer_budget=2)
    with pytest.raises(TypeError):
        ad.generate_batch_budgeted(inputs[:1], 2, 2)  # 预算只能按关键字给
    ad.train_mode()
    try:
        with pytest.raises(TrainingModeError):
            ad.generate_batch_budgeted(inputs[:1], thinking_budget=2, answer_budget=2)
    finally:
        ad.eval_mode()


def test_generation_config_with_greedy_processors_is_rejected(ad, inputs):
    """预算循环不经过 HF 的 logits 处理器：generation_config 里有会改 greedy logits 的项（如 repetition_penalty）就报错；
    三家族的固定 revision（小模型取同样的 generation_config）都没有。"""
    from src.adapters.budget import check_greedy_generation_config

    gc = ad.model.generation_config
    check_greedy_generation_config(gc)
    old = gc.repetition_penalty
    gc.repetition_penalty = 1.3
    try:
        with pytest.raises(RuntimeError, match="repetition_penalty"):
            ad.generate_batch_budgeted(inputs[:1], thinking_budget=2, answer_budget=2)
    finally:
        gc.repetition_penalty = old
    gc.repetition_penalty = 1.0  # 中性值可以
    try:
        ad.generate_batch_budgeted(inputs[:1], thinking_budget=2, answer_budget=2)
    finally:
        gc.repetition_penalty = old


def test_existing_generate_batch_results_have_no_budget_fields(ad, inputs):
    r = ad.generate_batch(inputs[:2], enable_thinking=True, max_new_tokens=3)[0]
    assert r.thinking_end is None and r.answer_text is None and r.n_thinking_tokens is None and not r.forced_close


def _gr(**kw):
    base = dict(text="a", token_ids=(1, 2, 3), n_generated_tokens=3, n_prompt_tokens=5, n_visual_tokens=0, wall_time_s=0.0,
                raw_text="xyz", stopped=True)
    base.update(kw)
    return GenerationResult(**base)


def test_generation_result_budget_invariants():
    ok = _gr(thinking_end="natural", n_thinking_tokens=2, n_answer_tokens=1, answer_text="a")
    assert ok.natural_close and not ok.forced_close
    forced = _gr(thinking_end="forced", forced_close=True, n_thinking_tokens=2, n_answer_tokens=1, n_inserted_tokens=1,
                 inserted_token_ids=(9,), inserted_at=2, inserted_span=(1, 2), answer_text="a")
    assert forced.forced_close and not forced.natural_close
    assert set(THINKING_ENDS) == {"natural", "forced", "not_opened", "unclosed"}
    bad = [
        dict(answer_text="a"),  # 不是预算生成却给了字段
        dict(thinking_end="weird", n_thinking_tokens=2, n_answer_tokens=1, answer_text="a"),
        dict(thinking_end="natural", n_thinking_tokens=1, n_answer_tokens=1, answer_text="a"),  # 和不对
        dict(thinking_end="forced", n_thinking_tokens=2, n_answer_tokens=1, answer_text="a"),  # forced_close 不符
        dict(thinking_end="forced", forced_close=True, n_thinking_tokens=2, n_answer_tokens=1, answer_text="a"),  # 没补 token
        dict(thinking_end="natural", n_thinking_tokens=2, n_answer_tokens=1, n_inserted_tokens=1, inserted_token_ids=(9,),
             inserted_at=2, inserted_span=(0, 1), answer_text="a"),
        dict(thinking_end="not_opened", n_thinking_tokens=1, n_answer_tokens=2, answer_text="a"),
        dict(thinking_end="unclosed", n_thinking_tokens=2, n_answer_tokens=1, answer_text=""),
        dict(thinking_end="forced", forced_close=True, n_thinking_tokens=2, n_answer_tokens=1, n_inserted_tokens=1,
             inserted_token_ids=(9,), inserted_at=2, inserted_span=(2, 9), answer_text="a"),  # 区间越界
    ]
    for kw in bad:
        with pytest.raises((ValueError, TypeError)):
            _gr(**kw)
