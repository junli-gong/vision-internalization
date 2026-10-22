"""DEV-0049：多轮对话接口 chat_turn_batch / chat_turn（W1，D103；接口约定 docs/w1_plan.md §1）。

Qwen3.5 小模型（CPU，fp32，init_std 0.05；同 tests/test_adapter_budget.py）。小模型的 greedy 输出近似随机，测试用外层模型上的前向
钩子（_Script）按"这一行自己的提示长度"决定每一步把哪个 token 抬到最高（或压到最低）：思考段在第几步自然收尾、写满预算被强制收尾、
思考段里就停；作答段逐个写出给定的 token（例如一个工具调用）再停，或者一直不停直到写满作答预算。规则只看这一行自己（提示的非填充
长度与这一行已送进去的 token），所以批量与逐条的行为相同；没被钩子指定的步仍由模型自己选。

覆盖：每种收尾 / 截断 / 工具调用的 ChatTurnResult 字段；渲染与官方处理器逐 token 相同（system、user 带图、assistant 带工具调用、
tool 带文字与图、user）；M-RoPE 位置；批内与逐条逐 token 相同；与 generate_batch_budgeted 的口径一致；parse_tool_calls；入参校验；
非 Qwen 家族抛 NotImplementedError。
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch
from PIL import Image

from src.adapters import qwen35
from src.adapters.base import ModelAdapter
from src.adapters.budget import BudgetError
from src.adapters.chat import (
    ChatMessageError,
    ChatTurnResult,
    ToolCall,
    ToolSpec,
    parse_tool_calls,
    template_messages,
)
from src.adapters.factory import make_adapter
from src.adapters.gemma4 import Gemma4Adapter
from src.adapters.image_policy import FixedSize, PixelBudget
from src.adapters.internvl35 import InternVL35Adapter
from src.adapters.training import TrainingModeError

KEY = "qwen3.5-0.8b"
POLICY = PixelBudget(64)
INIT_STD = 0.05
TB, AB = 8, 72  # 思考预算、作答预算（小；作答段要放得下 CALL_TEXT）

ZOOM = ToolSpec("zoom_in", "Show an enlarged view of a region of the original image.",
                {"type": "object",
                 "properties": {"box": {"type": "string", "description": "x1, y1, x2, y2 in 0-1000 of the original image"}},
                 "required": ["box"]})
RUN = ToolSpec("run_python", "Run Python code in the sandbox.",
               {"type": "object", "properties": {"code": {"type": "string"}, "timeout": {"type": "integer"}},
                "required": ["code"]})
SYSTEM = "You are a careful visual assistant. Use the tools when they help."
CALL_TEXT = "I will zoom in.\n\n<tool_call>\n<function=zoom_in>\n<parameter=box>\n100, 200, 300, 400\n</parameter>\n</function>\n</tool_call>"


@pytest.fixture(scope="module")
def ad():
    return make_adapter(KEY, device="cpu", dtype=torch.float32, image_policy=POLICY, tiny=True, init_std=INIT_STD)


@pytest.fixture(scope="module")
def ref(ad):
    return qwen35.reference_processor(ad.tokenizer, ad.image_processor, ad.chat_template)


def _img(rng, h, w):
    a = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    a[h // 4: h // 2, w // 4: w // 2, 1] = 230
    return Image.fromarray(a)


@pytest.fixture(scope="module")
def imgs():
    rng = np.random.default_rng(20261020)
    return {"land": _img(rng, 100, 150), "port": _img(rng, 150, 100), "square": _img(rng, 120, 120),
            "wide": _img(rng, 64, 256)}


def _t(s):
    return {"type": "text", "text": s}


def _i(im):
    return {"type": "image", "image": im}


def _history_call():
    return ToolCall("zoom_in", {"box": "100, 200, 300, 400"}, raw="")


@pytest.fixture(scope="module")
def convs(imgs):
    """3 条长度、图数都不同的对话；b 有工具调用的历史（第二轮，以 tool 消息结尾）。"""
    a = [{"role": "system", "content": [_t(SYSTEM)]},
         {"role": "user", "content": [_i(imgs["land"]), _t("Find the green square and report its box.")]}]
    b = [{"role": "system", "content": [_t(SYSTEM)]},
         {"role": "user", "content": [_i(imgs["port"]), _t("Segment the green square.")]},
         {"role": "assistant", "content": [_t("Let me look closer.")], "reasoning_content": "The square is small.",
          "tool_calls": [_history_call()]},
         {"role": "tool", "content": [_t("View of [100, 200, 300, 400]:"), _i(imgs["square"])]}]
    c = [{"role": "user", "content": [_t("How many corners does a triangle have? Answer with a number.")]}]
    return {"a": a, "b": b, "c": c}


TOOLS = {"a": [ZOOM, RUN], "b": [ZOOM], "c": None}  # 每条对话的工具可以不同（c 没有工具）


# ---------------------------------------------------------------------------
# 钩子
# ---------------------------------------------------------------------------


class _Script:
    """外层模型的前向钩子。plan：提示长度 → (思考段怎么结束, 作答段逐个写出的 token 或 None, 写完之后是否停)。
    思考段：natural 在第 4 个 token 写出 </think>；unclosed 在第 3 个 token 写出停止 token；forced 一直压住 </think>（写满预算由
    适配器补）。没指定的步压住 </think> 与停止 token，只让模型自己在其余 token 里选。作答段：先逐个写出 script，再停（stop）
    或一直压住停止 token（写满作答预算）。"""

    def __init__(self, ad, plan):
        self.fmt = ad.think_format()
        self.stops = list(ad.stop_token_ids())
        self.plan = plan
        self.hist: list[list[int]] = []
        self.plen: list[int] = []

    def targets(self, b):
        plan = self.plan.get(self.plen[b])
        if plan is None:
            return []
        think, script, stop = plan
        gen = self.hist[b][self.plen[b]:]
        fmt, n = self.fmt, len(gen)
        if fmt.close_id in gen:
            j = gen.index(fmt.close_id)
            k = n - (j + (len(fmt.insert_ids) if think == "forced" else 1))
            if k < 0:
                return []  # 正在补 token
            if script is not None and k < len(script):
                return [(script[k], "boost")]
            return [(self.stops[0], "boost")] if stop else [(s, "suppress") for s in self.stops]
        if think == "natural" and n == 3:
            return [(fmt.close_id, "boost")]
        if think == "unclosed" and n == 2:
            return [(self.stops[0], "boost")]
        return [(t, "suppress") for t in [fmt.close_id, *self.stops]]

    def __call__(self, module, args, kwargs, output):
        ids, mask = kwargs["input_ids"], kwargs["attention_mask"]
        if ids.shape[1] > 1:  # 预填
            self.hist = [ids[b][mask[b, -ids.shape[1]:].bool()].tolist() for b in range(ids.shape[0])]
            self.plen = [len(h) for h in self.hist]
        else:
            for b in range(ids.shape[0]):
                self.hist[b].append(int(ids[b, 0]))
        for b in range(ids.shape[0]):
            row = output.logits[b, -1]
            for tok, mode in self.targets(b):
                row[tok] = (row.max() + 50.0) if mode == "boost" else (row.min() - 50.0)
        return output


class scripted:
    def __init__(self, ad, plan):
        self.ad, self.hook = ad, _Script(ad, plan)

    def __enter__(self):
        self.h = self.ad.model.register_forward_hook(self.hook, with_kwargs=True)
        return self.hook

    def __exit__(self, *exc):
        self.h.remove()


def _plen(ad, conv, tools):
    return len(ad.chat_inputs(conv, tools=tools).row.ids)


def _enc(ad, s):
    return ad._encode(s)


def _kinds(ad):
    answer = _enc(ad, "The square is near the top left.")
    return {
        "natural": ("natural", answer, True),
        "forced": ("forced", answer, True),
        "truncated": ("natural", None, False),
        "tool_call": ("natural", _enc(ad, "\n\n" + CALL_TEXT), True),
        "forced_tool_call": ("forced", _enc(ad, CALL_TEXT), True),
        "unclosed": ("unclosed", None, True),
    }


def _decode(ad, ids):
    return ad.tokenizer.decode(list(ids), skip_special_tokens=False)


def _check_result(ad, r: ChatTurnResult, kind, plan, conv, tools):
    think, script, stop = plan
    inp = ad.chat_inputs(conv, tools=tools)
    stops = set(ad.stop_token_ids())
    fmt = ad.think_format()
    assert r.n_prompt_tokens == len(inp.row.ids)
    assert r.n_images == len(inp.images) == sum(1 for m in conv for it in m["content"] if it["type"] == "image")
    assert r.n_visual_tokens == sum(len(i.span) for i in inp.images) == sum(h * w for i in inp.images for h, w in [i.grid_hw])
    assert r.n_image_pixels == sum(i.n_pixels for i in inp.images)
    assert r.n_generated_tokens == len(r.token_ids)
    th, an = r.token_ids[: r.n_thinking_tokens], r.token_ids[r.n_thinking_tokens:]
    assert r.thinking_end == think
    if think == "natural":
        assert r.n_thinking_tokens == 4 and th[-1] == fmt.close_id and r.n_inserted_tokens == 0
        assert r.thinking == _decode(ad, th[:-1])
    elif think == "forced":
        assert r.n_thinking_tokens == TB and fmt.close_id not in th and not set(th) & stops
        assert r.n_inserted_tokens == len(fmt.insert_ids) == 2  # </think> + "\n\n"
        assert r.thinking == _decode(ad, th)
    else:
        assert r.n_thinking_tokens == 3 and th[-1] in stops and not an and r.stopped is True
        assert r.thinking == _decode(ad, th[:-1]) and r.text == "" and r.tool_calls == [] and r.tool_call_errors == []
        return
    if stop:
        assert r.stopped is True and an[-1] in stops and list(an[:-1]) == script
        assert r.text == _decode(ad, script).strip("\n")
    else:
        assert r.stopped is False and len(an) == AB and not set(an) & stops
        assert r.text == _decode(ad, an).strip("\n")
    if kind in ("tool_call", "forced_tool_call"):
        assert r.tool_call_errors == []
        assert r.tool_calls == [ToolCall("zoom_in", {"box": "100, 200, 300, 400"},
                                         raw=CALL_TEXT[CALL_TEXT.index("<tool_call>"):])]
        assert r.text == CALL_TEXT
    elif kind == "natural" or kind == "forced":
        assert r.tool_calls == [] and r.tool_call_errors == []


# ---------------------------------------------------------------------------
# 每种情况的字段
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["natural", "forced", "truncated", "tool_call", "forced_tool_call", "unclosed"])
def test_turn_fields(ad, convs, kind):
    conv, tools = convs["b"], TOOLS["b"]
    plan = _kinds(ad)[kind]
    with scripted(ad, {_plen(ad, conv, tools): plan}):
        r = ad.chat_turn(conv, tools=tools, thinking_budget=TB, answer_budget=AB)
    assert isinstance(r, ChatTurnResult) and r.batch_size == 1 and r.wall_time_s == r.batch_wall_time_s > 0
    _check_result(ad, r, kind, plan, conv, tools)


def test_truncated_tool_call_is_an_error_not_an_exception(ad, convs):
    """作答段写到一半被截断（工具调用没写完）：stopped = False，没有调用，错误列表里记"缺 </tool_call>"。"""
    conv, tools = convs["a"], TOOLS["a"]
    script = _enc(ad, "\n\n" + CALL_TEXT)[:-3]
    with scripted(ad, {_plen(ad, conv, tools): ("natural", script, False)}):
        r = ad.chat_turn(conv, tools=tools, thinking_budget=TB, answer_budget=len(script))
    assert r.stopped is False and r.tool_calls == [] and r.tool_call_errors == ["tool call 1: missing </tool_call>"]


def test_matches_generate_batch_budgeted_for_a_single_user_turn(ad, imgs):
    """只有一条 user（图 + 文字）的对话：提示与 build_inputs(image_first) + 开思考的助手头逐 token 相同，结果与
    generate_batch_budgeted 的口径一致（同一个循环、同一套记账）。"""
    instr = "Describe the image."
    conv = [{"role": "user", "content": [_i(imgs["land"]), _t(instr)]}]
    inp = ad.build_inputs([imgs["land"]], instr, "image_first")
    row = ad._prompt_row(inp, True)
    assert ad.chat_inputs(conv, tools=None).row == row
    for kind in ("natural", "forced", "truncated", "unclosed"):
        plan = {len(row.ids): _kinds(ad)[kind]}
        with scripted(ad, plan):
            r = ad.chat_turn(conv, tools=None, thinking_budget=TB, answer_budget=AB)
        with scripted(ad, plan):
            (g,) = ad.generate_batch_budgeted([inp], thinking_budget=TB, answer_budget=AB)
        assert r.token_ids == g.token_ids and r.text == g.answer_text and r.thinking_end == g.thinking_end
        assert (r.n_prompt_tokens, r.n_generated_tokens, r.n_inserted_tokens, r.n_visual_tokens, r.n_image_pixels,
                r.n_thinking_tokens, r.stopped) == (g.n_prompt_tokens, g.n_generated_tokens, g.n_inserted_tokens,
                                                    g.n_visual_tokens, g.n_image_pixels, g.n_thinking_tokens, g.stopped)


# ---------------------------------------------------------------------------
# 渲染：与官方处理器逐 token 相同
# ---------------------------------------------------------------------------


def test_render_matches_official_processor(ad, ref, imgs):
    """system、user（图）、assistant（带工具调用）、tool（文字加图）、user：送进模型的 token 序列、mm_token_type_ids、image_grid_thw、
    pixel_values 与官方 Qwen3VLProcessor（同样的 min_pixels = max_pixels）对手写的模板消息直接渲染的结果逐个相同。"""
    call = ToolCall("run_python", {"code": "import numpy as np\nprint(np.zeros(3))", "timeout": "5"}, raw="")
    conv = [{"role": "system", "content": [_t(SYSTEM)]},
            {"role": "user", "content": [_i(imgs["wide"]), _t("Outline the green square.")]},
            {"role": "assistant", "content": [_t("Checking.")], "reasoning_content": "Old reasoning.",
             "tool_calls": [_history_call(), call]},
            {"role": "tool", "content": [_t("View:"), _i(imgs["port"])]},
            {"role": "tool", "content": [_t("[0. 0. 0.]")]},
            {"role": "user", "content": [_t("Now submit the mask.")]}]
    hand = [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Outline the green square."}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Checking."}], "reasoning_content": "Old reasoning.",
             "tool_calls": [{"name": "zoom_in", "arguments": {"box": "100, 200, 300, 400"}},
                            {"name": "run_python", "arguments": {"code": "import numpy as np\nprint(np.zeros(3))",
                                                                 "timeout": "5"}}]},
            {"role": "tool", "content": [{"type": "text", "text": "View:"}, {"type": "image"}]},
            {"role": "tool", "content": [{"type": "text", "text": "[0. 0. 0.]"}]},
            {"role": "user", "content": [{"type": "text", "text": "Now submit the mask."}]}]
    tools = [ZOOM, RUN]
    tool_dicts = [{"type": "function", "function": {"name": s.name, "description": s.description,
                                                    "parameters": s.parameters}} for s in tools]
    assert template_messages(conv)[0] == hand
    text = ref.apply_chat_template(hand, tools=tool_dicts, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    px = ad.processor_pixels()
    enc = ref(text=[text], images=[imgs["wide"], imgs["port"]], min_pixels=px, max_pixels=px, return_tensors="pt")
    inp = ad.chat_inputs(conv, tools=tools)
    t = inp.tensors
    assert inp.text == text
    assert torch.equal(enc["input_ids"], t["input_ids"]) and list(inp.row.ids) == enc["input_ids"][0].tolist()
    assert torch.equal(enc["attention_mask"], t["attention_mask"])
    assert torch.equal(enc["mm_token_type_ids"], t["mm_token_type_ids"])
    assert list(inp.row.mm_token_type_ids) == enc["mm_token_type_ids"][0].tolist()
    assert torch.equal(enc["image_grid_thw"], t["image_grid_thw"]) and torch.equal(enc["pixel_values"], t["pixel_values"])
    # 图像数与视觉 token 数
    assert len(inp.images) == 2 and inp.images[0].grid_hw != inp.images[1].grid_hw
    want = [(int(g[1]) // 2, int(g[2]) // 2) for g in enc["image_grid_thw"]]
    assert [i.grid_hw for i in inp.images] == want
    assert inp.row.n_visual_tokens == sum(h * w for h, w in want) == int((enc["input_ids"] == ad.token_id("<|image_pad|>")).sum())
    assert inp.row.n_image_pixels == sum(h * 32 * w * 32 for h, w in want)
    # 工具段在 system 里，工具调用历史按原生格式写出，tool 消息合进一条 user 轮
    assert text.startswith("<|im_start|>system\n# Tools") and SYSTEM in text
    assert ("<tool_call>\n<function=run_python>\n<parameter=code>\nimport numpy as np\nprint(np.zeros(3))\n</parameter>\n"
            "<parameter=timeout>\n5\n</parameter>\n</function>\n</tool_call>") in text
    assert text.count("<|im_start|>user") == 3 and text.count("<tool_response>") == 2
    assert "Old reasoning." not in text  # 最后一个真实 user 之前的思考段被模板去掉
    assert text.endswith("<|im_start|>assistant\n<think>\n")
    assert tuple(inp.row.ids[-len(ad.assistant_header_ids(True)):]) == ad.assistant_header_ids(True)


def test_render_fixed_size_policy_and_numpy_images(imgs):
    """固定尺寸口径（FixedSize）：每张图（含 tool 消息里的 numpy 图）先缩放到 (H, W)，与官方处理器对缩放后的图逐 token 相同。"""
    fx = make_adapter(KEY, device="cpu", dtype=torch.float32, image_policy=FixedSize(64, 96), tiny=True, init_std=INIT_STD)
    ref = qwen35.reference_processor(fx.tokenizer, fx.image_processor, fx.chat_template)
    arr = np.asarray(imgs["port"])
    conv = [{"role": "user", "content": [_i(imgs["land"]), _t("Find the square.")]},
            {"role": "assistant", "content": [_t("Zooming.")], "tool_calls": [_history_call()]},
            {"role": "tool", "content": [_i(arr)]}]
    inp = fx.chat_inputs(conv, tools=[ZOOM])
    text = ref.apply_chat_template(template_messages(conv)[0], tools=[ZOOM.as_template()], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=True)
    px = fx.processor_pixels()
    enc = ref(text=[text], images=[fx.resize_image(imgs["land"]), fx.resize_image(arr)], min_pixels=px, max_pixels=px,
              return_tensors="pt")
    assert inp.text == text and torch.equal(enc["input_ids"], inp.tensors["input_ids"])
    assert torch.equal(enc["pixel_values"], inp.tensors["pixel_values"])
    assert torch.equal(enc["image_grid_thw"], inp.tensors["image_grid_thw"])
    assert [i.grid_hw for i in inp.images] == [(2, 3), (2, 3)] and inp.row.n_visual_tokens == 12


def test_reasoning_kept_after_last_user_and_tools_differ_per_row(ad, convs):
    """以 tool 消息结尾的对话（同一轮里的多步调用）：assistant 的思考段在最后一个 user 之后，模板保留它；没有工具的对话没有
    tools 段。"""
    b = ad.chat_inputs(convs["b"], tools=TOOLS["b"]).text
    assert "<think>\nThe square is small.\n</think>\n\nLet me look closer.\n\n<tool_call>" in b
    assert '"name": "zoom_in"' in b and '"name": "run_python"' not in b
    c = ad.chat_inputs(convs["c"], tools=TOOLS["c"]).text
    assert "# Tools" not in c and c == ad.chat_inputs(convs["c"], tools=[]).text


def _capture_positions(ad, fn):
    lm = ad.model.model.language_model
    seen = []

    def hook(mod, args, kwargs):
        seen.append(kwargs["position_ids"].detach().cpu().clone())

    h = lm.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        out = fn()
    finally:
        h.remove()
    return out, seen


def test_mrope_positions_follow_template_order(ad, convs):
    """多张图分布在多条消息里：预填实际送进语言模型的 (T, H, W) 与本适配器自己的规则（mrope_rows，图像段取各图的网格）逐个相等，
    解码步从提示末尾接着加 1；左填充的行同样。"""
    items = [ad.chat_inputs(convs[k], tools=TOOLS[k]) for k in "abc"]
    keys = list("abc")
    with scripted(ad, {}):
        res, seen = _capture_positions(ad, lambda: ad.chat_turn_batch([convs[k] for k in keys],
                                                                     tools=[TOOLS[k] for k in keys],
                                                                     thinking_budget=3, answer_budget=2))
    width = seen[0].shape[-1]
    assert width == max(len(i.row.ids) for i in items)
    for b, inp in enumerate(items):
        n = len(inp.row.ids)
        own = qwen35.mrope_rows(n, [(i.span.start, i.span.end, i.grid_hw) for i in inp.images])
        assert torch.equal(seen[0][1:, b, width - n:], own) and torch.equal(seen[0][0, b, width - n:], torch.arange(n))
        for t in range(1, len(seen)):
            assert seen[t][1:, b, -1].tolist() == [int(own[0, -1]) + t] * 3
    assert sum(len(i.images) for i in items) == 3 and len(items[1].images) == 2


def test_runtime_position_check_also_runs_without_padding(ad, convs, monkeypatch):
    """chat_turn 的位置核对（get_rope_index 对本适配器的规则）在没有填充的单条上也做：get_rope_index 错位就报错。"""
    real = ad.model.model.get_rope_index

    def shifted(input_ids, **kw):
        pos, delta = real(input_ids, **kw)
        return pos + 1, delta

    monkeypatch.setattr(ad.model.model, "get_rope_index", shifted)
    with pytest.raises(RuntimeError, match="M-RoPE"):
        ad.chat_turn(convs["b"], tools=TOOLS["b"], thinking_budget=2, answer_budget=2)


# ---------------------------------------------------------------------------
# 批一致性
# ---------------------------------------------------------------------------

_FIELDS = ("text", "thinking", "thinking_end", "tool_calls", "tool_call_errors", "stopped", "n_prompt_tokens",
           "n_generated_tokens", "n_inserted_tokens", "n_visual_tokens", "n_images", "token_ids", "n_thinking_tokens",
           "n_image_pixels")


@pytest.mark.parametrize("assign", [("tool_call", "forced", "truncated"), ("forced_tool_call", "unclosed", "natural")])
def test_batch_matches_single(ad, convs, assign):
    """3 条长度、图数都不同的对话（b 有工具调用历史，c 没有图、没有工具）成批生成，与逐条生成逐 token 相同，各字段相同
    （wall_time_s 与批的字段除外）。每行的收尾方式不同，没被钩子指定的步由模型自己选。"""
    keys = list("abc")
    kinds = _kinds(ad)
    lens = {k: _plen(ad, convs[k], TOOLS[k]) for k in keys}
    assert len(set(lens.values())) == 3
    plan = {lens[k]: kinds[a] for k, a in zip(keys, assign)}
    with scripted(ad, plan):
        batch = ad.chat_turn_batch([convs[k] for k in keys], tools=[TOOLS[k] for k in keys], thinking_budget=TB,
                                   answer_budget=AB)
        single = [ad.chat_turn(convs[k], tools=TOOLS[k], thinking_budget=TB, answer_budget=AB) for k in keys]
    for k, a, bt, s in zip(keys, assign, batch, single):
        assert {f: getattr(bt, f) for f in _FIELDS} == {f: getattr(s, f) for f in _FIELDS}, k
        assert bt.batch_size == 3 and s.batch_size == 1 and bt.wall_time_s * 3 == pytest.approx(bt.batch_wall_time_s)
        _check_result(ad, bt, a, kinds[a], convs[k], TOOLS[k])
    assert len({r.token_ids[:3] for r in batch}) >= 2, "小模型的输出与输入无关"  # 思考段开头由模型自己选


def test_two_round_rehearsal_round_synchronous(ad, convs, imgs):
    """按轮同步的两轮彩排（DEV-0051 的用法）：第 1 轮 3 条成批；a 写出工具调用，按 chat 模块说明的写法回填（content = 作答段原文，
    reasoning_content = 思考段），再追加一条带图的 tool 消息；b、c 结束、退出批。第 2 轮只剩 a：提示里保留第 1 轮的思考段与原样的
    工具调用，图多一张；字段照常。"""
    keys = list("abc")
    kinds = _kinds(ad)
    plan1 = {_plen(ad, convs[k], TOOLS[k]): kinds[a] for k, a in zip(keys, ("tool_call", "natural", "forced"))}
    with scripted(ad, plan1):
        r1 = ad.chat_turn_batch([convs[k] for k in keys], tools=[TOOLS[k] for k in keys], thinking_budget=TB,
                                answer_budget=AB)
    assert [len(r.tool_calls) for r in r1] == [1, 0, 0] and r1[0].tool_calls[0].name == "zoom_in"
    box = r1[0].tool_calls[0].arguments["box"]
    a2 = list(convs["a"]) + [
        {"role": "assistant", "content": [_t(r1[0].text)], "reasoning_content": r1[0].thinking},
        {"role": "tool", "content": [_t(f"View of [{box}]:"), _i(imgs["wide"])]}]
    inp2 = ad.chat_inputs(a2, tools=TOOLS["a"])
    assert len(inp2.images) == 2
    assert "<think>\n" + r1[0].thinking.strip() + "\n</think>\n\n" + CALL_TEXT + "<|im_end|>\n<|im_start|>user\n<tool_response>" \
        in inp2.text
    assert len(inp2.row.ids) > r1[0].n_prompt_tokens + r1[0].n_thinking_tokens
    with scripted(ad, {len(inp2.row.ids): kinds["natural"]}):
        (r2,) = ad.chat_turn_batch([a2], tools=[TOOLS["a"]], thinking_budget=TB, answer_budget=AB)
    _check_result(ad, r2, "natural", kinds["natural"], a2, TOOLS["a"])
    assert r2.n_images == 2 and r2.batch_size == 1


# ---------------------------------------------------------------------------
# DEV-0055：W1 的提示作为单独的 user 消息（tool → user → 生成）
# ---------------------------------------------------------------------------

NOTICE = "[Turn 2 of 6 done; tool calls used: 2 of 8; turns left: 4.]"
THINK1, THINK2 = "First I should zoom in on the square.", "The zoomed view shows the square; look again."
VIEW = "View of [100, 200, 300, 400]:"
PAD = "<|vision_start|><|image_pad|><|vision_end|>"


def _w1_two_rounds(imgs):
    """W1 回路回填的两轮历史（D104 的写法：content = 作答原文（含 <tool_call> 块），reasoning_content = 思考段，不给 tool_calls），
    每轮一个调用、结果是文字加图。"""
    return [{"role": "system", "content": [_t(SYSTEM)]},
            {"role": "user", "content": [_i(imgs["land"]), _t("Segment the green square.")]},
            {"role": "assistant", "content": [_t(CALL_TEXT)], "reasoning_content": THINK1},
            {"role": "tool", "content": [_t(VIEW), _i(imgs["square"])]},
            {"role": "assistant", "content": [_t(CALL_TEXT)], "reasoning_content": THINK2},
            {"role": "tool", "content": [_t(VIEW), _i(imgs["wide"])]}]


def _notice_variants(imgs):
    """同一段历史，提示的两种走法：tool（现状：作为一个文字项追加在最后一条 tool 消息末尾）与 user（单独一条 user 消息）。"""
    hist = _w1_two_rounds(imgs)
    as_tool = hist[:-1] + [{"role": "tool", "content": list(hist[-1]["content"]) + [_t(NOTICE)]}]
    as_user = hist + [{"role": "user", "content": [_t(NOTICE)]}]
    return as_tool, as_user


def test_notice_as_user_message_renders_after_tool_results(ad, ref, imgs):
    """check_messages / chat_inputs 接受 assistant → tool → user → 生成；渲染与官方处理器逐 token 相同；user 消息渲染为
    </tool_response> 之后单独的 user 轮、生成提示之前。Qwen3.5 的模板只保留最后一条真实 user 之后的思考段：走 tool 时两轮的思考段
    都在，走 user 时两轮都被去掉（assistant 只剩作答原文）。提示的 token 数随之变化（视觉 token 不变）。"""
    as_tool, as_user = _notice_variants(imgs)
    px = ad.processor_pixels()
    inp = {}
    for name, conv in (("tool", as_tool), ("user", as_user)):
        i = ad.chat_inputs(conv, tools=[ZOOM])
        text = ref.apply_chat_template(template_messages(conv)[0], tools=[ZOOM.as_template()], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=True)
        enc = ref(text=[text], images=[imgs["land"], imgs["square"], imgs["wide"]], min_pixels=px, max_pixels=px,
                  return_tensors="pt")
        assert i.text == text and list(i.row.ids) == enc["input_ids"][0].tolist(), name
        assert torch.equal(enc["pixel_values"], i.tensors["pixel_values"])
        inp[name] = i
    t, u = inp["tool"].text, inp["user"].text
    gen = "<|im_start|>assistant\n<think>\n"
    # tool：提示在 <tool_response> 里，与工具输出之间没有分隔
    assert t.endswith(f"<tool_response>\n{VIEW}{PAD}{NOTICE}\n</tool_response><|im_end|>\n{gen}")
    # user：</tool_response> 之后单独一条 user 轮，紧接着是生成提示
    assert u.endswith(f"<tool_response>\n{VIEW}{PAD}\n</tool_response><|im_end|>\n<|im_start|>user\n{NOTICE}<|im_end|>\n{gen}")
    assert t.count("<|im_start|>user") == 3 and u.count("<|im_start|>user") == 4
    # 思考段：tool 时两轮都在（在最后一条真实 user 之后）；user 时都被模板去掉
    for th in (THINK1, THINK2):
        assert f"<|im_start|>assistant\n<think>\n{th}\n</think>\n\n{CALL_TEXT}<|im_end|>" in t
        assert th not in u
    assert u.count(f"<|im_start|>assistant\n{CALL_TEXT}<|im_end|>") == 2 and "</think>" not in u
    # token 数：视觉 token 相同；user 时去掉的两段思考比多出的 user 轮长，提示变短
    assert inp["tool"].row.n_visual_tokens == inp["user"].row.n_visual_tokens > 0
    dropped = sum(len(ad._encode(f"<think>\n{th}\n</think>\n\n")) for th in (THINK1, THINK2))
    added = len(ad._encode(f"<|im_start|>user\n{NOTICE}<|im_end|>\n")) - len(ad._encode(NOTICE))
    assert len(inp["user"].row.ids) - len(inp["tool"].row.ids) == added - dropped < 0


def test_notice_as_user_message_generates_like_any_turn(ad, imgs):
    """chat_turn_batch 接受以 user 提示结尾的对话（与以 tool 结尾的同批）：字段照常，批内与逐条逐 token 相同。"""
    as_tool, as_user = _notice_variants(imgs)
    kinds = _kinds(ad)
    lt, lu = _plen(ad, as_tool, [ZOOM]), _plen(ad, as_user, [ZOOM])
    assert lt != lu
    plan = {lt: kinds["tool_call"], lu: kinds["natural"]}
    with scripted(ad, plan):
        batch = ad.chat_turn_batch([as_tool, as_user], tools=[[ZOOM], [ZOOM]], thinking_budget=TB, answer_budget=AB)
        single = [ad.chat_turn(c, tools=[ZOOM], thinking_budget=TB, answer_budget=AB) for c in (as_tool, as_user)]
    for bt, s in zip(batch, single):
        assert {f: getattr(bt, f) for f in _FIELDS} == {f: getattr(s, f) for f in _FIELDS}
    _check_result(ad, batch[0], "tool_call", kinds["tool_call"], as_tool, [ZOOM])
    _check_result(ad, batch[1], "natural", kinds["natural"], as_user, [ZOOM])
    assert batch[1].n_prompt_tokens == lu and batch[1].n_images == 3


# ---------------------------------------------------------------------------
# parse_tool_calls
# ---------------------------------------------------------------------------


def _block(name, **params):
    body = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in params.items())
    return f"<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>"


def test_parse_single_call():
    raw = _block("zoom_in", box="1, 2, 3, 4")
    calls, errors = parse_tool_calls("Let me zoom.\n\n" + raw)
    assert errors == [] and calls == [ToolCall("zoom_in", {"box": "1, 2, 3, 4"}, raw=raw)]


def test_parse_multiple_calls_in_order():
    r1, r2, r3 = _block("zoom_in", box="1, 2, 3, 4"), _block("get_pixel", x="10", y="20"), _block("list_masks")
    calls, errors = parse_tool_calls(f"{r1}\n{r2}\n{r3}")
    assert errors == []
    assert [(c.name, c.arguments, c.raw) for c in calls] == [("zoom_in", {"box": "1, 2, 3, 4"}, r1),
                                                             ("get_pixel", {"x": "10", "y": "20"}, r2),
                                                             ("list_masks", {}, r3)]


def test_parse_multiline_and_json_values():
    code = "import numpy as np\n\nm = np.zeros((4, 4))\nprint(m.sum())\n"
    js = '{"points": [[1, 2], [3, 4]], "label": "a <b> c"}'
    raw = _block("run_python", code=code, spec=js)
    (call,), errors = parse_tool_calls(raw)
    assert errors == [] and call.arguments == {"code": code, "spec": js}  # 只去掉紧挨标记的一个换行


def test_parse_round_trips_through_the_template(ad):
    """模板写出的工具调用（历史回填）再解析回来，与原来的 ToolCall 相同。"""
    calls = [ToolCall("zoom_in", {"box": "1, 2, 3, 4"}, raw=""),
             ToolCall("run_python", {"code": "x = 1\nprint(x)", "spec": '{"a": [1, 2]}'}, raw="")]
    conv = [{"role": "user", "content": [_t("q")]}, {"role": "assistant", "content": [_t("Calling.")], "tool_calls": calls},
            {"role": "user", "content": [_t("again")]}]
    text = ad.chat_inputs(conv, tools=None).text
    seg = text.split("<|im_start|>assistant\n", 1)[1].split("<|im_end|>", 1)[0]
    got, errors = parse_tool_calls(seg)
    assert errors == [] and [(c.name, c.arguments) for c in got] == [(c.name, c.arguments) for c in calls]


def test_parse_missing_close_and_empty_name_are_errors():
    good = _block("zoom_in", box="1, 2, 3, 4")
    unclosed = "<tool_call>\n<function=zoom_in>\n<parameter=box>\n1, 2, 3, 4\n</parameter>\n</function>\n"
    calls, errors = parse_tool_calls(unclosed)
    assert calls == [] and errors == ["tool call 1: missing </tool_call>"]
    calls, errors = parse_tool_calls(unclosed + good)  # 下一个 <tool_call> 先出现：第一个缺收尾，第二个照常
    assert [c.raw for c in calls] == [good] and errors == ["tool call 1: missing </tool_call>"]
    calls, errors = parse_tool_calls(_block("", box="1") + good)
    assert [c.raw for c in calls] == [good] and errors == ["tool call 1: empty function name"]
    calls, errors = parse_tool_calls("<tool_call>\n<function=   >\n</function>\n</tool_call>")
    assert calls == [] and errors == ["tool call 1: empty function name"]


@pytest.mark.parametrize("text,err", [
    ("<tool_call>\nzoom_in(box=1)\n</tool_call>", "tool call 1: expected <function=NAME> right after <tool_call>"),
    ("<tool_call>\n<function=zoom_in>\n<parameter=box>\n1\n</parameter>\n</tool_call>", "tool call 1: missing </function>"),
    ("<tool_call>\n<function=zoom_in>\n<parameter=box>\n1\n</function>\n</tool_call>",
     "tool call 1: missing </parameter> for parameter 'box'"),
    ("<tool_call>\n<function=zoom_in>\n<parameter=a>\n1\n<parameter=b>\n2\n</parameter>\n</function>\n</tool_call>",
     "tool call 1: missing </parameter> for parameter 'a'"),
    ("<tool_call>\n<function=zoom_in>\n<parameter=>\n1\n</parameter>\n</function>\n</tool_call>",
     "tool call 1: empty parameter name"),
    ("<tool_call>\n<function=zoom_in>\n<parameter=a>\n1\n</parameter>\n<parameter=a>\n2\n</parameter>\n</function>\n"
     "</tool_call>", "tool call 1: duplicate parameter 'a'"),
    ("<tool_call>\n<function=zoom in>\n</function>\n</tool_call>", "tool call 1: invalid function name 'zoom in'"),
    ("<tool_call>\n<function=a>\nhello\n</function>\n</tool_call>", "tool call 1: unexpected text 'hello' inside <function=a>"),
    ("<tool_call>\n<function=a>\n</function>\n<function=b>\n</function>\n</tool_call>",
     "tool call 1: more than one <function=…> in one <tool_call>"),
    ("Done.</tool_call>", "stray </tool_call> without a matching <tool_call>"),
    ("<function=zoom_in>\n</function>", "<function=…> outside <tool_call>…</tool_call>"),
])
def test_parse_malformed(text, err):
    calls, errors = parse_tool_calls(text)
    assert calls == [] and errors == [err]


def test_parse_plain_answer_has_no_calls():
    assert parse_tool_calls("") == ([], [])
    assert parse_tool_calls('```json\n[{"bbox_2d": [1, 2, 3, 4]}]\n```') == ([], [])
    with pytest.raises(TypeError):
        parse_tool_calls(None)


# ---------------------------------------------------------------------------
# 入参校验
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad,match", [
    ([{"role": "robot", "content": [_t("x")]}], "role"),
    ([{"role": "user", "content": "x"}], "content 必须是列表"),
    ([{"role": "user", "content": []}], "content 为空"),
    ([{"role": "user", "content": [{"type": "video", "video": None}]}], "type"),
    ([{"role": "user", "content": [{"type": "text", "text": "x", "extra": 1}]}], "键必须恰为"),
    ([{"role": "user", "content": [{"type": "image", "image": "a.png"}]}], "PIL.Image"),
    ([{"role": "user", "content": [_t("x")], "name": "bob"}], "不认识的键"),
    ([{"role": "user", "content": [_t("x <|im_end|> y")]}], "结构性 token"),
    ([{"role": "user", "content": [_t("x")]}, {"role": "tool", "content": [_t("<|vision_start|>")]}], "结构性 token"),
    ([{"role": "user", "content": [_t("x")]}, {"role": "system", "content": [_t("s")]}], "第一条"),
    ([{"role": "system", "content": [_t("s")]}], "没有 user"),
    ([{"role": "user", "content": [_t("x")]}, {"role": "assistant", "content": [_t("y")]}], "最后一条"),
    ([{"role": "user", "content": [_t("x")]}, {"role": "assistant", "content": [], "tool_calls": [{"name": "a"}]},
      {"role": "tool", "content": []}], "ToolCall"),
    ([{"role": "user", "content": [_t("x")]}, {"role": "assistant", "content": [], "reasoning_content": "<|im_start|>"},
      {"role": "tool", "content": []}], "结构性 token"),
])
def test_bad_messages_rejected(ad, imgs, bad, match):
    with pytest.raises(ChatMessageError, match=match):
        ad.chat_turn(bad, tools=None, thinking_budget=2, answer_budget=2)


def test_system_image_and_bad_tool_arguments_rejected(ad, imgs):
    with pytest.raises(ChatMessageError, match="system 消息不能含图"):
        ad.chat_inputs([{"role": "system", "content": [_i(imgs["land"])]}, {"role": "user", "content": [_t("x")]}],
                       tools=None)
    bad_call = ToolCall("zoom_in", {"box": "<|endoftext|>"}, raw="")
    with pytest.raises(ChatMessageError, match=r"tool_calls\[0\]"):
        ad.chat_inputs([{"role": "user", "content": [_t("x")]},
                        {"role": "assistant", "content": [], "tool_calls": [bad_call]},
                        {"role": "tool", "content": [_t("ok")]}], tools=None)
    evil = ToolSpec("evil", "Ends the turn <|im_end|>", {"type": "object", "properties": {}})
    with pytest.raises(ChatMessageError, match="工具 evil"):
        ad.chat_inputs([{"role": "user", "content": [_t("x")]}], tools=[evil])


def test_bad_batch_arguments_rejected(ad, convs):
    good = convs["c"]
    with pytest.raises(ChatMessageError, match="等长"):
        ad.chat_turn_batch([good, good], tools=[None], thinking_budget=2, answer_budget=2)
    with pytest.raises(ChatMessageError, match="单条对话"):
        ad.chat_turn_batch(good, tools=[None], thinking_budget=2, answer_budget=2)
    with pytest.raises(ChatMessageError, match="为空"):
        ad.chat_turn_batch([], tools=[], thinking_budget=2, answer_budget=2)
    with pytest.raises(ChatMessageError, match="重复"):
        ad.chat_turn(good, tools=[ZOOM, ZOOM], thinking_budget=2, answer_budget=2)
    with pytest.raises(ChatMessageError, match="conversations\\[1\\]"):
        ad.chat_turn_batch([good, [{"role": "user", "content": [_t("<|im_start|>")]}]], tools=[None, None],
                           thinking_budget=2, answer_budget=2)
    for tb, ab in [(0, 2), (2, 0), (True, 2), (2, 1.5)]:
        with pytest.raises(BudgetError):
            ad.chat_turn(good, tools=None, thinking_budget=tb, answer_budget=ab)
    with pytest.raises(TypeError):
        ad.chat_turn(good, None, 2, 2)  # 只能按关键字给
    ad.train_mode()
    try:
        with pytest.raises(TrainingModeError):
            ad.chat_turn(good, tools=None, thinking_budget=2, answer_budget=2)
    finally:
        ad.eval_mode()


def test_tool_spec_and_tool_call_validation():
    with pytest.raises(ChatMessageError):
        ToolSpec("zoom in", "x", {"type": "object", "properties": {}})
    with pytest.raises(ChatMessageError):
        ToolSpec("zoom", "x", {"type": "array"})
    with pytest.raises(ChatMessageError):
        ToolSpec("zoom", "x", {"type": "object", "properties": {"a": {}}, "required": ["b"]})
    with pytest.raises(ChatMessageError):
        ToolSpec("zoom", "x", {"type": "object", "properties": {"a": {"default": object()}}})
    with pytest.raises(ChatMessageError):
        ToolCall("zoom", {"a": 1}, raw="")
    assert ZOOM.as_template()["function"]["name"] == "zoom_in"


def test_chat_turn_result_invariants():
    base = dict(text="a", thinking="t", thinking_end="natural", tool_calls=[], tool_call_errors=[], stopped=True,
                n_prompt_tokens=10, n_generated_tokens=3, n_inserted_tokens=0, n_visual_tokens=0, n_images=0, wall_time_s=0.1)
    ChatTurnResult(**base)  # §1 的字段就能构造
    for kw in [dict(thinking_end="weird"), dict(n_inserted_tokens=2), dict(thinking_end="forced"),
               dict(thinking_end="unclosed"), dict(n_images=1), dict(token_ids=(1, 2)), dict(wall_time_s=-1.0),
               dict(batch_size=2, batch_wall_time_s=1.0), dict(stopped=None)]:
        with pytest.raises((ValueError, TypeError)):
            ChatTurnResult(**{**base, **kw})


# ---------------------------------------------------------------------------
# 其他家族与接口
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("make", [lambda: Gemma4Adapter(device="cpu", max_soft_tokens=70),
                                  lambda: InternVL35Adapter(device="cpu", crop_to_patches=False, max_patches=1)])
def test_other_families_raise_not_implemented(make):
    other = make()
    conv = [{"role": "user", "content": [_t("hi")]}]
    with pytest.raises(NotImplementedError):
        other.chat_turn(conv, tools=None, thinking_budget=2, answer_budget=2)
    with pytest.raises(NotImplementedError):
        other.chat_turn_batch([conv], tools=[None], thinking_budget=2, answer_budget=2)


def test_chat_budgets_are_required_keyword_arguments():
    for meth in ("chat_turn_batch", "chat_turn"):
        sig = inspect.signature(getattr(qwen35.Qwen35Adapter, meth))
        for p in ("tools", "thinking_budget", "answer_budget"):
            assert sig.parameters[p].kind is inspect.Parameter.KEYWORD_ONLY and sig.parameters[p].default is inspect._empty
    with pytest.raises(TypeError, match="thinking_budget"):
        class _Bad(qwen35.Qwen35Adapter):
            def chat_turn_batch(self, conversations, *, tools, thinking_budget=2048, answer_budget):
                raise AssertionError
    assert ModelAdapter.chat_turn_batch is not qwen35.Qwen35Adapter.chat_turn_batch
