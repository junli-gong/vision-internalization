"""DEV-0020 第一部分：唯一的受限 logits 与分块交叉熵（CPU fp32 小模型，三个家族）。

- restricted_logits = 全词表 project 在 allowed 列上的值（只对 allowed 的行做矩阵乘），fp32；Gemma 含 final_logit_softcapping，
  与 HF 前向（lm_head + softcap）在 fp32 下一致；lm_head 是否与嵌入共享照实取（get_output_embeddings().weight）。
- chunked_lm_loss 与不分块的 F.cross_entropy(project(h, W)) 在 fp32 下相同：|Δloss| ≤ 1e-5 · max(1, |loss|)（只差求和顺序；
  实测 ≤ 4e-7），梯度 rtol 1e-5 / atol 5e-6（实测 ≤ 9e-7）；各种 chunk_size（1、不整除、等于 N、大于 N）都对，含 softcap。
- 受限网格损失 = restricted_logits + 普通交叉熵（restricted_targets 把 token id 换成 allowed 里的下标）。
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from src.adapters import training
from src.adapters.factory import make_adapter
from src.adapters.image_policy import PixelBudget, SoftTokenBudget, TilePolicy
from src.adapters.training import TrainingInputError

FAMILY_KEYS = ("qwen3.5-0.8b", "gemma-4-e2b-it", "internvl3.5-2b-hf")
POLICIES = {
    "qwen3.5-0.8b": PixelBudget(64),
    "gemma-4-e2b-it": SoftTokenBudget(70),
    "internvl3.5-2b-hf": TilePolicy(crop_to_patches=False, max_patches=1),
}
# lm_head 是否与嵌入共享（小模型按 factory 的默认建：Qwen 共享、InternVL 不共享；Gemma 一律共享）与 softcap
TIED = {"qwen3.5-0.8b": True, "gemma-4-e2b-it": True, "internvl3.5-2b-hf": False}
SOFTCAP = {"qwen3.5-0.8b": None, "gemma-4-e2b-it": 30.0, "internvl3.5-2b-hf": None}
LOSS_RTOL = 1e-5
GRAD_ATOL = 5e-6  # 梯度只差 fp32 求和顺序（实测 ≤ 9e-7；元素量级 1e-3–1e-2）


@pytest.fixture(scope="module", params=FAMILY_KEYS)
def ad(request):
    key = request.param
    try:
        return make_adapter(key, device="cpu", dtype=torch.float32, image_policy=POLICIES[key], tiny=True, init_std=0.1)
    except OSError as e:
        pytest.fail(f"取不到 {key} 的小文件：{e!r}", pytrace=False)


def _key(ad):
    return ad.model_key.removeprefix("tiny(").removesuffix(")")


def _hidden(ad, n, seed=0, scale=3.0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, ad.hidden_size(), generator=g) * scale


def test_tied_and_softcap_facts(ad):
    w = ad.lm_head_weight()
    assert (w is ad.model.get_input_embeddings().weight) == TIED[_key(ad)]
    assert w is ad.model.get_output_embeddings().weight
    assert ad._logit_softcap() == SOFTCAP[_key(ad)]


def test_restricted_logits_equals_full_projection_columns(ad):
    h = _hidden(ad, 7)
    allowed = [5, 1000, 3, 248000 % ad.lm_head_weight().shape[0], 77]
    r = ad.restricted_logits(h, allowed)
    assert r.dtype == torch.float32 and tuple(r.shape) == (7, 5)
    full = training.project(h, ad.lm_head_weight(), softcap=ad._logit_softcap())
    torch.testing.assert_close(r, full[:, allowed], rtol=0, atol=1e-5)
    # 列的顺序 = allowed 的顺序；张量形式的 allowed 也可以；批维度任意
    # （行的顺序不同，矩阵乘的分块就不同：只差 fp32 求和顺序，≤ 1e-6）
    torch.testing.assert_close(ad.restricted_logits(h, torch.tensor(allowed[::-1])), r.flip(-1), rtol=0, atol=1e-5)
    torch.testing.assert_close(ad.restricted_logits(h.view(7, 1, -1), allowed)[:, 0], r, rtol=0, atol=1e-5)


def test_restricted_logits_match_hf_forward_logits(ad):
    """与 HF 前向的 logits（lm_head，Gemma 再 softcap）在 fp32 下一致：同一组隐藏状态，HF 的 logits 取 allowed 列。"""
    inp = ad.build_inputs([], "Say something about cats.", "instruction_first")
    rows, kw = ad._batch_kw([inp], False)
    with ad._forward_scope(), torch.no_grad():
        kw2 = dict(kw)
        kw2["position_ids"] = ad._decode_positions(kw)
        out = ad.model(**kw2, use_cache=False, output_hidden_states=True, logits_to_keep=0, return_dict=True)
    last = out.hidden_states[-1][0]
    allowed = list(range(0, 50)) + [1000, 2000]
    torch.testing.assert_close(ad.restricted_logits(last, allowed), out.logits[0][:, allowed].float(), rtol=0, atol=2e-5)


def test_softcap_is_applied(ad):
    h = _hidden(ad, 5, scale=200.0)  # 大幅度的隐藏状态：没有 softcap 时 |logit| 远超 30
    r = ad.restricted_logits(h, list(range(100)))
    raw = F.linear(h, ad.lm_head_weight()[:100])
    cap = ad._logit_softcap()
    if cap is None:
        torch.testing.assert_close(r, raw, rtol=0, atol=1e-4)
    else:
        assert float(raw.abs().max()) > 2 * cap and float(r.abs().max()) <= cap
        torch.testing.assert_close(r, torch.tanh(raw / cap) * cap, rtol=0, atol=1e-5)


@pytest.mark.parametrize("chunk_size", [1, 3, 7, 37, 100])
def test_chunked_lm_loss_equals_unchunked(ad, chunk_size):
    n = 37
    h0 = _hidden(ad, n, seed=1)
    t = torch.randint(0, ad.lm_head_weight().shape[0], (n,), generator=torch.Generator().manual_seed(2))
    h_ref = h0.clone().requires_grad_(True)
    ref = F.cross_entropy(training.project(h_ref, ad.lm_head_weight(), softcap=ad._logit_softcap()), t)
    ref.backward()
    h = h0.clone().requires_grad_(True)
    loss = ad.chunked_lm_loss(h, t, chunk_size=chunk_size)
    loss.backward()
    assert loss.dtype == torch.float32
    loss, ref = loss.detach(), ref.detach()
    assert abs(float(loss) - float(ref)) <= LOSS_RTOL * max(1.0, abs(float(ref)))
    torch.testing.assert_close(h.grad, h_ref.grad, rtol=1e-5, atol=GRAD_ATOL)
    s = ad.chunked_lm_loss(h0, t, chunk_size=chunk_size, reduction="sum").detach()
    assert abs(float(s) - float(ref) * n) <= LOSS_RTOL * max(1.0, abs(float(ref)) * n)


def test_chunked_lm_loss_no_grad_path_and_validation(ad):
    h, t = _hidden(ad, 4), torch.tensor([1, 2, 3, 4])
    with torch.no_grad():
        a = ad.chunked_lm_loss(h, t, chunk_size=2)
    b = ad.chunked_lm_loss(h, t, chunk_size=4)
    assert abs(float(a) - float(b)) <= LOSS_RTOL * max(1.0, abs(float(b)))
    for bad in (dict(chunk_size=0), dict(chunk_size=True), dict(chunk_size=2, reduction="none")):
        with pytest.raises(TrainingInputError):
            ad.chunked_lm_loss(h, t, **bad)
    with pytest.raises(TrainingInputError):
        ad.chunked_lm_loss(h, t[:3], chunk_size=2)
    with pytest.raises(TrainingInputError):
        ad.chunked_lm_loss(h, t.int(), chunk_size=2)
    with pytest.raises(TrainingInputError, match="越界"):
        ad.chunked_lm_loss(h, torch.tensor([1, 2, 3, 10 ** 9]), chunk_size=2)
    with pytest.raises(TrainingInputError):
        ad.chunked_lm_loss(h[:0], t[:0], chunk_size=2)


def test_restricted_grid_loss(ad):
    allowed = [1000, 1003, 1001, 1002]
    h = _hidden(ad, 6).requires_grad_(True)
    tgt = torch.tensor([1003, 1000, 1000, 1002, 1001, 1003])
    idx = training.restricted_targets(tgt, allowed)
    assert idx.tolist() == [1, 0, 0, 3, 2, 1]
    loss = F.cross_entropy(ad.restricted_logits(h, allowed), idx)
    full = training.project(h, ad.lm_head_weight(), softcap=ad._logit_softcap())[:, allowed]
    torch.testing.assert_close(loss, F.cross_entropy(full, idx), rtol=1e-6, atol=1e-6)
    loss.backward()
    assert h.grad is not None
    with pytest.raises(TrainingInputError, match="不在"):
        training.restricted_targets([1004], allowed)


def test_restricted_logits_validation(ad):
    h = _hidden(ad, 2)
    for bad in ([], [1, 1], [-1], [10 ** 9], [1.0], "ab", torch.tensor([[1]]), torch.tensor([1.0])):
        with pytest.raises(TrainingInputError):
            ad.restricted_logits(h, bad)
    with pytest.raises(TrainingInputError, match="最后一维"):
        ad.restricted_logits(torch.zeros(2, ad.hidden_size() + 1), [1])
    with pytest.raises(TrainingInputError):
        ad.restricted_logits(torch.zeros(2, ad.hidden_size(), dtype=torch.long), [1])
    with pytest.raises(TrainingInputError, match="meta"):  # 入口断言：hidden 与 lm_head 不在同一个设备
        ad.restricted_logits(torch.zeros(2, ad.hidden_size(), device="meta"), [1])
