"""DEV-0020 第一部分：训练前向（教师强制版的 generate）、受限解码、训练 / 评测模式与梯度检查点。CPU fp32 小模型，三个家族。

- 拼接：training_forward 送进模型的每一行 = generate 时 HF 实际的前缀（提示）+ 生成部分（用 spy 截下 model.generate 的输出比对）。
- 对齐：位置 t 的隐藏状态预测第 t+1 个 token；第一个目标位置 = 提示的最后一个 token（各家族的助手头末尾，钉在 HEADER_TAIL）。
- 一致性（D21-2、简报 §12）：generate_restricted 与 generate_batch(allowed_token_ids) 的 token 序列逐个相同；把解码出的序列教师
  强制送回 training_forward，每个位置 argmax(restricted_logits) 等于解码出的 token。GPU bf16 的判据是一致率
  （training.restricted_consistency），在这里 CPU fp32 上两项都必须是 1.0。
- 受限解码只经过 restricted_logits：lm_head 一次也不被调用。
- 模式：eval 下 dropout 关闭、两次前向逐位相同；train_mode 下 dropout 生效、视觉塔保持 eval；生成类方法在训练模式下报错。
- 梯度检查点（use_reentrant=False）：梯度与不开时相同，decoder 层确实被重算。
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch
from PIL import Image

from src.adapters import training
from src.adapters.base import ModelAdapter, Span
from src.adapters.factory import make_adapter
from src.adapters.image_policy import PixelBudget, SoftTokenBudget, TilePolicy
from src.adapters.training import HiddenRequest, TrainingInputError, TrainingModeError

FAMILY_KEYS = ("qwen3.5-0.8b", "gemma-4-e2b-it", "internvl3.5-2b-hf")
POLICIES = {
    "qwen3.5-0.8b": PixelBudget(64),
    "gemma-4-e2b-it": SoftTokenBudget(70),
    "internvl3.5-2b-hf": TilePolicy(crop_to_patches=False, max_patches=1),
}
INIT_STD = 0.1  # 同 test_adapter_batching 的理由：HF 默认初始化下小模型的输出几乎与输入无关
# 各家族提示的最后几个 token（= 第一个目标位置之前的助手头末尾；固定 revision 的 tokenizer）：(关思考, 开思考)
HEADER_TAIL = {
    "qwen3.5-0.8b": (["<think>", "\n\n", "</think>", "\n\n"], ["assistant", "\n", "<think>", "\n"]),
    "gemma-4-e2b-it": (["<|turn>", "model", "\n"], ["<|turn>", "model", "\n"]),
    "internvl3.5-2b-hf": (["<|im_start|>", "assistant", "\n"], ["<|im_start|>", "assistant", "\n"]),
}
ALLOWED_BASE = list(range(1000, 1060))


def _key(ad) -> str:
    return ad.model_key.removeprefix("tiny(").removesuffix(")")


def _make(key, **kw):
    try:
        return make_adapter(key, device="cpu", dtype=torch.float32, image_policy=POLICIES[key], tiny=True,
                            init_std=INIT_STD, **kw)
    except OSError as e:
        pytest.fail(f"取不到 {key} 的 config/tokenizer/处理器小文件：{e!r}（需要联网下载一次）", pytrace=False)


@pytest.fixture(scope="module", params=FAMILY_KEYS)
def ad(request):
    return _make(request.param)


@pytest.fixture
def fresh():
    return _make


def _img(rng, h, w):
    a = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    a[h // 4: h // 2, w // 4: w // 2, 1] = 230
    return Image.fromarray(a)


@pytest.fixture(scope="module")
def samples():
    """(图, 指令, 顺序)：提示长短不一（左填充各不相同），图横、竖、两张、没有。"""
    rng = np.random.default_rng(20261020)
    land, port, wide = _img(rng, 100, 150), _img(rng, 150, 100), _img(rng, 64, 256)
    return [
        ([land], "Describe the image in one short sentence.", "instruction_first"),
        ([], "Hi", "instruction_first"),
        ([port, wide], "Which image is brighter?", "image_first"),
        ([wide], "Count the green squares and answer with a number only, please.", "instruction_first"),
    ]


@pytest.fixture(scope="module")
def inputs(ad, samples):
    return [ad.build_inputs(ims, ins, order) for ims, ins, order in samples]


def _allowed(ad) -> list[int]:
    return sorted(set(ALLOWED_BASE + list(ad.stop_token_ids())))


# ---------------------------------------------------------------------------
# 拼接与对齐
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("thinking", [False, True])
def test_training_rows_equal_generate_prefix_plus_generated(ad, inputs, thinking, monkeypatch):
    """training_forward 的每一行 = HF generate 实际送进去的前缀 + 生成部分（spy 截下 model.generate 的输入与输出）。"""
    seen = {}
    orig = ad.model.generate

    def spy(**kw):
        out = orig(**kw)
        seen["in"], seen["out"] = kw["input_ids"].cpu(), out.cpu()
        return out

    monkeypatch.setattr(ad.model, "generate", spy)
    gen = ad.generate_batch(inputs, enable_thinking=thinking, max_new_tokens=6)
    monkeypatch.undo()
    with torch.no_grad():
        tf = ad.training_forward(inputs, [list(g.token_ids) for g in gen], enable_thinking=thinking)
    width = seen["in"].shape[1]
    pads = [int((seen["in"][b] == ad._pad_id()).cumprod(0).sum()) for b in range(len(inputs))]
    assert len(set(pads)) > 1  # 前提：有填充，且各行填充长度不同
    for b, g in enumerate(gen):
        hf_row = seen["out"][b, pads[b]: width + len(g.token_ids)].tolist()
        assert tf.row_ids(b) == hf_row, b
        assert tf.prompt_lengths[b] == g.n_prompt_tokens and tf.target_lengths[b] == g.n_generated_tokens
    # 整批左填充：mask 是 0…0 1…1，填充 id 与 generate 相同
    for b in range(len(inputs)):
        p = tf.n_padding[b]
        assert tf.attention_mask[b, :p].sum() == 0 and bool(tf.attention_mask[b, p:].all())
        assert bool((tf.input_ids[b, :p] == ad._pad_id()).all())


@pytest.mark.parametrize("thinking", [False, True])
def test_target_alignment_and_first_position(ad, inputs, thinking):
    targets = [[1001, 1002, 1003], [], [1004], [1005, 1006]]
    with torch.no_grad():
        tf = ad.training_forward(inputs, targets, enable_thinking=thinking)
    assert tf.target_ids.tolist() == [t for row in targets for t in row]
    assert tf.target_offsets == (0, 3, 3, 4, 6) and tf.target_lengths == (3, 0, 1, 2)
    assert tuple(tf.target_hidden.shape) == (6, ad.hidden_size())
    tail = HEADER_TAIL[_key(ad)][int(thinking)]
    for b, row in enumerate(targets):
        first = tf.n_padding[b] + tf.prompt_lengths[b] - 1  # 第一个目标位置 = 提示的最后一个 token
        last_prompt = tf.input_ids[b, first - len(tail) + 1: first + 1].tolist()
        assert [ad.tokenizer.decode([i]) for i in last_prompt] == tail
        sl = slice(tf.target_offsets[b], tf.target_offsets[b + 1])
        assert tf.target_positions[sl].tolist() == list(range(first, first + len(row)))
        assert bool((tf.target_rows[sl] == b).all())
        # 位置 t 的隐藏状态预测第 t+1 个 token
        assert tf.input_ids[b, tf.target_positions[sl] + 1].tolist() == row


def test_batch_matches_single_rows(ad, inputs):
    targets = [[1001, 1002], [1003], [1004, 1005, 1006], [1007]]
    with torch.no_grad():
        batch = ad.training_forward(inputs, targets, enable_thinking=False)
        for b, (inp, t) in enumerate(zip(inputs, targets)):
            single = ad.training_forward([inp], [t], enable_thinking=False)
            assert single.n_padding == (0,)
            torch.testing.assert_close(batch.row_target_hidden(b), single.target_hidden, rtol=0, atol=2e-5)


def test_target_hidden_matches_sequence_hidden_states(ad, inputs):
    """单行、关思考：目标位置的最后一层隐藏状态与"整行直接前向"的最后一层（层号 num_layers，最终归一化之后）相同。"""
    inp = inputs[0]
    with torch.no_grad():
        tf = ad.training_forward([inp], [[1001, 1002, 1003]], enable_thinking=False,
                                 return_hidden=HiddenRequest(layers=(ad.num_layers(),), regions=("image",)))
        seq = ad.sequence_hidden_states(inp, [ad.num_layers()])[ad.num_layers()]
    img = torch.cat([torch.arange(i.span.start, i.span.end) for i in inp.images])
    torch.testing.assert_close(tf.hidden["image"][ad.num_layers()][0], seq[img], rtol=0, atol=2e-5)


# ---------------------------------------------------------------------------
# 受限解码：与 generate_batch(allowed_token_ids) 相同；教师强制的 argmax 等于解码出的 token
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("thinking", [False, True])
def test_generate_restricted_equals_generate_batch(ad, inputs, thinking):
    allowed = _allowed(ad)
    gen = ad.generate_batch(inputs, enable_thinking=thinking, max_new_tokens=12, allowed_token_ids=allowed)
    res = ad.generate_restricted(inputs, allowed, max_new_tokens=12, stop_ids=list(ad.stop_token_ids()), enable_thinking=thinking)
    assert [r.token_ids for r in res] == [g.token_ids for g in gen]
    assert len({r.token_ids for r in res}) >= 3  # 前提：各行的输出大多不同（小模型的输出随输入变化）
    for r, g in zip(res, gen):
        assert (r.stopped, r.text, r.raw_text, r.thinking_unclosed) == (g.stopped, g.text, g.raw_text, g.thinking_unclosed)
        assert (r.n_prompt_tokens, r.n_visual_tokens, r.n_image_pixels, r.batch_size) == (
            g.n_prompt_tokens, g.n_visual_tokens, g.n_image_pixels, g.batch_size)
        assert set(r.token_ids) <= set(allowed)


def test_generate_restricted_stop_and_order(ad, inputs):
    """allowed 的顺序不影响结果（内部升序，平局取 id 小者）；没有停止 token 时一律生成满 max_new_tokens。"""
    allowed = _allowed(ad)
    a = ad.generate_restricted(inputs, allowed, max_new_tokens=5, stop_ids=list(ad.stop_token_ids()), enable_thinking=False)
    b = ad.generate_restricted(inputs, allowed[::-1], max_new_tokens=5, stop_ids=list(ad.stop_token_ids()), enable_thinking=False)
    assert [x.token_ids for x in a] == [x.token_ids for x in b]
    c = ad.generate_restricted(inputs, ALLOWED_BASE, max_new_tokens=5, stop_ids=[], enable_thinking=False)
    assert all(x.n_generated_tokens == 5 and x.stopped is False for x in c)
    g = ad.generate_batch(inputs, enable_thinking=False, max_new_tokens=5, allowed_token_ids=ALLOWED_BASE)
    assert [x.token_ids for x in c] == [x.token_ids for x in g]


@pytest.mark.parametrize("thinking", [False, True])
def test_teacher_forcing_argmax_equals_restricted_decode(ad, inputs, thinking):
    rep = training.restricted_consistency(ad, inputs, _allowed(ad), max_new_tokens=12, stop_ids=list(ad.stop_token_ids()),
                                          enable_thinking=thinking)
    assert rep["decode_vs_generate_batch"]["exact_rows"] == 1.0
    assert rep["decode_vs_generate_batch"]["token_agreement"] == 1.0
    assert rep["teacher_forcing"]["agreement"] == 1.0 and rep["teacher_forcing"]["disagreements"] == []
    assert rep["teacher_forcing"]["n_tokens"] >= 4 * 2


def test_restricted_decode_only_uses_restricted_logits(ad, inputs, monkeypatch):
    """每一步的 logits 都由 restricted_logits 算；lm_head（全词表）一次也不被调用。"""
    calls = {"restricted": 0, "lm_head": 0}
    orig = ad.restricted_logits

    def counting(hidden, allowed_ids):
        calls["restricted"] += 1
        return orig(hidden, allowed_ids)

    monkeypatch.setattr(ad, "restricted_logits", counting)
    h = ad.model.get_output_embeddings().register_forward_hook(lambda *a: calls.__setitem__("lm_head", calls["lm_head"] + 1))
    try:
        res = ad.generate_restricted(inputs, ALLOWED_BASE, max_new_tokens=4, stop_ids=[], enable_thinking=False)
    finally:
        h.remove()
    assert calls == {"restricted": 4, "lm_head": 0} and all(r.n_generated_tokens == 4 for r in res)


def test_internvl_multi_tile_restricted_decode():
    """InternVL 多 tile（每行 tile 数不同）也与 generate_batch 一致。"""
    ad = make_adapter("internvl3.5-2b-hf", device="cpu", dtype=torch.float32, tiny=True, init_std=INIT_STD,
                      image_policy=TilePolicy(crop_to_patches=True, max_patches=4))
    rng = np.random.default_rng(5)
    inputs = [ad.build_inputs([_img(rng, 448, 900)], "Describe.", "instruction_first"),
              ad.build_inputs([_img(rng, 300, 300)], "What is it?", "image_first")]
    assert len({len(i.images[0].span) for i in inputs}) == 2  # 前提：两行的 tile 数不同
    rep = training.restricted_consistency(ad, inputs, _allowed(ad), max_new_tokens=8, stop_ids=list(ad.stop_token_ids()),
                                          enable_thinking=False)
    assert rep["decode_vs_generate_batch"]["exact_rows"] == 1.0 and rep["teacher_forcing"]["agreement"] == 1.0


# ---------------------------------------------------------------------------
# 另外返回的隐藏状态（E3a 的逐层加权混合）
# ---------------------------------------------------------------------------


def test_return_hidden_image_and_spans(ad, inputs):
    layers = (0, 1, ad.num_layers())
    spans = [{"instr": inp.instruction_span} for inp in inputs]
    with torch.no_grad():
        tf = ad.training_forward(inputs, [[1001]] * len(inputs), enable_thinking=True,
                                 return_hidden=HiddenRequest(layers=layers, regions=("image",), spans=spans))
    assert set(tf.hidden) == {"image", "instr"} and set(tf.hidden["image"]) == set(layers)
    for b, inp in enumerate(inputs):
        n_img = sum(len(i.span) for i in inp.images)
        assert tuple(tf.hidden["image"][0][b].shape) == (n_img, ad.hidden_size())
        assert tuple(tf.hidden["instr"][1][b].shape) == (len(inp.instruction_span), ad.hidden_size())
        if inp.images:
            with torch.no_grad():
                vf = ad.vision_features(inp)
            torch.testing.assert_close(tf.hidden["image"][0][b], vf, rtol=0, atol=1e-6)  # 第 0 层图像位置 = 视觉特征
    # 指令区间第 0 层 = 指令 token 的嵌入查表（Gemma 含 embed_scale）：下标映射（开思考时的前缀偏移）对得上
    inp = inputs[1]
    ids = inp.tensors["input_ids"][0, inp.instruction_span.start: inp.instruction_span.end]
    with torch.no_grad():
        want = ad.model.get_input_embeddings()(ids)
    torch.testing.assert_close(tf.hidden["instr"][0][1], want, rtol=0, atol=0)


def test_return_hidden_carries_grad(fresh):
    ad = fresh("qwen3.5-0.8b")
    inp = ad.build_inputs([], "Hi there", "instruction_first")
    w = ad.model.get_input_embeddings().weight
    tf = ad.training_forward([inp], [[1001]], enable_thinking=False,
                             return_hidden=HiddenRequest(layers=(2,), spans=({"x": inp.instruction_span},)))
    tf.hidden["x"][2][0].pow(2).sum().backward()
    assert w.grad is not None and w.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# 查询 token：必须显式给 query_embeds（D21-2）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", FAMILY_KEYS)
def test_query_embeds_required_and_carry_grad(fresh, key):
    ad = fresh(key)
    ad.inject_query_tokens(4, "seq1d")
    rng = np.random.default_rng(1)
    inputs = [ad.build_inputs([_img(rng, 100, 150)], "Segment the cat.", "instruction_first"),
              ad.build_inputs([], "Hi", "instruction_first")]
    with pytest.raises(TrainingInputError, match="query_embeds"):
        ad.training_forward(inputs, [[1001], [1002]], enable_thinking=False)
    with pytest.raises(ValueError, match="query_embeds"):  # 现有的 hidden_states 也一样（不再静默用占位符的行）
        ad.hidden_states(inputs[0], [0])
    with pytest.raises(ValueError, match="query_embeds"):
        ad.sequence_hidden_states(inputs[0], [0])
    qe = (torch.randn(4, ad.hidden_size()) * 0.05).requires_grad_(True)
    tf = ad.training_forward(inputs, [[1001], [1002]], enable_thinking=True, query_embeds=qe,
                             return_hidden=HiddenRequest(layers=(0, ad.num_layers()), regions=("query",)))
    for b in range(2):
        assert torch.equal(tf.hidden["query"][0][b], qe.detach())  # 第 0 层的查询位置就是 query_embeds
    ad.chunked_lm_loss(tf.target_hidden, tf.target_ids, chunk_size=1).backward()
    assert qe.grad is not None and qe.grad.abs().sum() > 0  # 目标 token 看得到查询，梯度流回 query_embeds
    # 与现有 hidden_states 的查询读数一致（单行、关思考）
    with torch.no_grad():
        hs = ad.hidden_states(inputs[0], [ad.num_layers()], query_embeds=qe.detach())
        tf1 = ad.training_forward(inputs[:1], [[]], enable_thinking=False, query_embeds=qe.detach(),
                                  return_hidden=HiddenRequest(layers=(ad.num_layers(),), regions=("query",)))
    torch.testing.assert_close(tf1.hidden["query"][ad.num_layers()][0], hs.query[ad.num_layers()], rtol=0, atol=2e-5)
    ad.clear_query_tokens()
    plain = ad.build_inputs([], "Hi", "instruction_first")
    with pytest.raises(TrainingInputError, match="没有查询"):
        ad.training_forward([plain], [[1]], enable_thinking=False, query_embeds=qe.detach())


def test_qwen_grid2d_query_positions_in_training_forward(fresh):
    """Qwen grid2d：训练前向的位置与 position_ids()（单行）逐项相同，查询块之后的目标接着编号。"""
    ad = fresh("qwen3.5-0.8b")
    ad.inject_query_tokens(9, "grid2d")
    rng = np.random.default_rng(2)
    inputs = [ad.build_inputs([_img(rng, 100, 150)], "Segment the cat.", "instruction_first"),
              ad.build_inputs([], "Hi", "instruction_first")]
    qe = torch.randn(9, ad.hidden_size()) * 0.05
    captured = {}

    def grab(mod, args, kwargs):
        captured["pos"] = kwargs["position_ids"]  # 返回 None：不改输入

    h = ad.model.model.language_model.register_forward_pre_hook(grab, with_kwargs=True)
    try:
        with torch.no_grad():
            tf = ad.training_forward(inputs, [[1001, 1002], [1003]], enable_thinking=False, query_embeds=qe)
    finally:
        h.remove()
    pos = captured["pos"]
    for b, inp in enumerate(inputs):
        p = tf.n_padding[b]
        own = ad.position_ids(inp)[:, 0]  # (4, L_user)
        n_user = own.shape[1]
        assert torch.equal(pos[:, b, p: p + n_user].cpu(), own.cpu())
        nxt = pos[1:, b, p + n_user:].cpu()  # 之后（助手头 + 目标）都是文本：三维相同、逐个加 1
        start = int(own[1:, -1].max()) + 1
        assert torch.equal(nxt, torch.arange(start, start + nxt.shape[1]).expand(3, -1))


# ---------------------------------------------------------------------------
# 训练 / 评测模式与梯度检查点
# ---------------------------------------------------------------------------


def _set_attention_dropout(ad, p):
    n = 0
    for m in ad.model.model.language_model.modules():
        if hasattr(m, "attention_dropout"):
            m.attention_dropout = p
            n += 1
    assert n > 0


@pytest.mark.parametrize("key", FAMILY_KEYS)
def test_eval_mode_deterministic_train_mode_dropout(fresh, key):
    ad = fresh(key)
    rng = np.random.default_rng(3)
    inputs = [ad.build_inputs([_img(rng, 100, 150)], "Describe the image.", "instruction_first")]
    _set_attention_dropout(ad, 0.5)
    ad.train_mode()
    assert ad.training and ad.model.model.language_model.training
    assert all(not m.training for v in ad._vision_modules() for m in v.modules())  # 视觉塔默认冻结：保持 eval
    with torch.no_grad():
        a = ad.training_forward(inputs, [[1001, 1002]], enable_thinking=False).target_hidden
        b = ad.training_forward(inputs, [[1001, 1002]], enable_thinking=False).target_hidden
    assert not torch.equal(a, b)  # 训练模式：dropout 生效
    with pytest.raises(TrainingModeError, match="eval_mode"):
        ad.generate_batch(inputs, enable_thinking=False, max_new_tokens=1)
    with pytest.raises(TrainingModeError, match="eval_mode"):
        ad.generate(inputs[0], enable_thinking=False, max_new_tokens=1)
    with pytest.raises(TrainingModeError, match="eval_mode"):
        ad.option_logits(inputs, [["A", "B"]], enable_thinking=False)
    with pytest.raises(TrainingModeError, match="eval_mode"):
        ad.generate_restricted(inputs, ALLOWED_BASE, max_new_tokens=1, stop_ids=[], enable_thinking=False)
    ad.eval_mode()
    assert not ad.training and not any(m.training for m in ad.model.modules())
    with torch.no_grad():
        c = ad.training_forward(inputs, [[1001, 1002]], enable_thinking=False).target_hidden
        d = ad.training_forward(inputs, [[1001, 1002]], enable_thinking=False).target_hidden
    assert torch.equal(c, d)  # eval：dropout 关闭，两次前向逐位相同
    ad.generate_batch(inputs, enable_thinking=False, max_new_tokens=1)  # eval 下照常生成


@pytest.mark.parametrize("key", FAMILY_KEYS)
def test_gradient_checkpointing_same_grads(fresh, key):
    ad = fresh(key)
    rng = np.random.default_rng(4)
    inputs = [ad.build_inputs([_img(rng, 100, 150)], "Describe the image.", "instruction_first"),
              ad.build_inputs([], "Hi", "image_first")]
    targets = [[1001, 1002, 1003], [1004]]
    params = dict(ad.model.model.language_model.layers[0].named_parameters())
    layer = ad.model.model.language_model.layers[0]

    def run():
        ad.model.zero_grad(set_to_none=True)
        n = {"calls": 0}
        # 前向 pre-hook：非重入检查点的重算会提前停（early stop），后置 hook 不一定跑到；pre-hook 每次进入都触发
        h = layer.register_forward_pre_hook(lambda *a: n.__setitem__("calls", n["calls"] + 1))
        try:
            tf = ad.training_forward(inputs, targets, enable_thinking=False)
            loss = ad.chunked_lm_loss(tf.target_hidden, tf.target_ids, chunk_size=2)
            loss.backward()
        finally:
            h.remove()
        return float(loss.detach()), {k: p.grad.clone() for k, p in params.items() if p.grad is not None}, n["calls"]

    ad.train_mode()
    loss0, g0, calls0 = run()
    ad.set_gradient_checkpointing(True)
    assert ad.gradient_checkpointing
    # 检查点函数包了一层：重算（反向时）也在 _forward_scope() 里（Qwen 在 CPU 上走 GDN 的 torch 实现；server-parity 条件下钉住）
    assert getattr(layer._gradient_checkpointing_func, "scoped", False)
    loss1, g1, calls1 = run()
    ad.set_gradient_checkpointing(False)
    ad.eval_mode()
    assert calls0 == 1 and calls1 == 2  # 开了检查点：反向时重算一次
    assert loss0 == loss1 and set(g0) == set(g1) and g0
    for k in g0:
        torch.testing.assert_close(g1[k], g0[k], rtol=1e-6, atol=1e-8)


# ---------------------------------------------------------------------------
# 入参检查与接口
# ---------------------------------------------------------------------------


def test_training_forward_validation(ad, inputs):
    tf = ad.training_forward
    with pytest.raises(TrainingInputError, match="行"):
        tf(inputs, [[1]], enable_thinking=False)
    with pytest.raises(TrainingInputError, match="越界"):
        tf(inputs[:1], [[10 ** 9]], enable_thinking=False)
    with pytest.raises(TrainingInputError, match="整数"):
        tf(inputs[:1], [[1.5]], enable_thinking=False)
    with pytest.raises(TrainingInputError):
        tf(inputs[0], [[1]], enable_thinking=False)
    with pytest.raises(TypeError):
        tf(inputs[:1], [[1]], enable_thinking=None)
    with pytest.raises(TypeError):
        tf(inputs[:1], [[1]])  # enable_thinking 必填（D1）
    with pytest.raises(ValueError, match="越界"):
        tf(inputs[:1], [[1]], enable_thinking=False, return_hidden=HiddenRequest(layers=(ad.num_layers() + 1,),
                                                                                 regions=("image",)))
    with pytest.raises(TrainingInputError, match="query"):
        tf(inputs[:1], [[1]], enable_thinking=False, return_hidden=HiddenRequest(layers=(0,), regions=("query",)))
    with pytest.raises(TrainingInputError, match="spans"):
        tf(inputs[:2], [[1], [2]], enable_thinking=False,
           return_hidden=HiddenRequest(layers=(0,), spans=({"a": Span(1, 2)},)))
    with pytest.raises(TrainingInputError, match="超出"):
        tf(inputs[:1], [[1]], enable_thinking=False, return_hidden=HiddenRequest(layers=(0,), spans=({"a": Span(1, 10 ** 6)},)))
    for bad in (dict(layers=()), dict(layers=(0,)), dict(layers=(0,), regions=("bogus",)),
                dict(layers=(0,), spans=({"image": Span(1, 2)},)), dict(layers=(0,), spans=({"a": Span(0, 1)}, {"b": Span(0, 1)}))):
        with pytest.raises(TrainingInputError):
            HiddenRequest(**bad)


def test_generate_restricted_validation(ad, inputs):
    gr = ad.generate_restricted
    for bad in ([], [1, 1], [-1], [10 ** 9]):
        with pytest.raises(TrainingInputError):
            gr(inputs[:1], bad, max_new_tokens=1, stop_ids=[], enable_thinking=False)
    with pytest.raises(TrainingInputError):
        gr(inputs[:1], [1], max_new_tokens=1, stop_ids="x", enable_thinking=False)
    with pytest.raises(ValueError):
        gr(inputs[:1], [1], max_new_tokens=0, stop_ids=[], enable_thinking=False)
    with pytest.raises(TypeError):
        gr(inputs[:1], [1], max_new_tokens=1, enable_thinking=False)  # stop_ids 必填


def test_signatures_and_base_stubs():
    from src.adapters.gemma4 import Gemma4Adapter
    from src.adapters.internvl35 import InternVL35Adapter
    from src.adapters.qwen35 import Qwen35Adapter

    for cls in (Qwen35Adapter, Gemma4Adapter, InternVL35Adapter):
        for meth, params in (("training_forward", ("enable_thinking",)),
                             ("generate_restricted", ("enable_thinking", "max_new_tokens", "stop_ids"))):
            sig = inspect.signature(getattr(cls, meth))
            for p in params:
                par = sig.parameters[p]
                assert par.kind is inspect.Parameter.KEYWORD_ONLY and par.default is inspect.Parameter.empty, (cls, meth, p)
            assert getattr(cls, meth) is not getattr(ModelAdapter, meth)
    with pytest.raises(TypeError, match="enable_thinking"):

        class _Bad(training.TrainingSupport, ModelAdapter):  # noqa: F841
            def training_forward(self, inputs_list, target_ids_list, *, enable_thinking=False, return_hidden=None,
                                 query_embeds=None):
                pass
