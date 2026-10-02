"""DEV-0023：探针的结构、初始化与调度；特征来源的入口断言；post_proj 的核对；pre_proj 参考线（维度、网格与 LLM 层不同）。"""

from __future__ import annotations

import math

import pytest
import torch

from src.adapters.factory import make_adapter
from src.e2.config import PRE_PROJ, load_config, resolve_task
from src.e2.data import E2Paths
from src.e2.features import ChannelStats, FeatureCheckError, FeatureExtractor
from src.e2.probes import ConvProbe, LinearProbe, ProbeBank, lr_factor, make_probe, n_params, probe_seed
from src.e2.runner import RunSpec, TaskRun, run_task


def test_probe_structure_and_param_counts():
    lin = make_probe("linear", 64, 150, {}, seed=1)
    conv = make_probe("conv", 64, 150, {"hidden": 256, "kernel": 3, "padding": 1}, seed=1)
    assert isinstance(lin, LinearProbe) and isinstance(conv, ConvProbe)
    assert n_params(lin) == 64 * 150 + 150
    assert n_params(conv) == (64 * 9 * 256 + 256) + (256 * 9 * 150 + 150)
    x = torch.randn(1, 64, 15, 20)
    assert lin(x).shape == conv(x).shape == (1, 150, 15, 20)
    assert isinstance(conv.act, torch.nn.GELU) and conv.conv1.kernel_size == (3, 3) and lin.proj.kernel_size == (1, 1)
    with pytest.raises(ValueError):
        make_probe("mlp", 64, 1, {}, seed=1)


def test_probe_init_is_seeded_and_independent_of_global_rng():
    a = make_probe("conv", 8, 3, {"hidden": 4, "kernel": 3, "padding": 1}, seed=probe_seed(7, "depth", 3, "conv"))
    torch.manual_seed(123)
    _ = torch.randn(100)
    b = make_probe("conv", 8, 3, {"hidden": 4, "kernel": 3, "padding": 1}, seed=probe_seed(7, "depth", 3, "conv"))
    assert all(torch.equal(p, q) for p, q in zip(a.state_dict().values(), b.state_dict().values()))
    seeds = {probe_seed(7, t, k, kind) for t in ("depth", "semseg") for k in (0, 1, PRE_PROJ) for kind in ("linear", "conv")}
    assert len(seeds) == 12
    st = torch.random.get_rng_state()
    make_probe("linear", 8, 3, {}, seed=5)
    assert torch.equal(st, torch.random.get_rng_state())


def test_lr_schedule_warmup_then_cosine():
    f = lambda s: lr_factor(s, steps=100, warmup_steps=10, min_lr_ratio=0.0)  # noqa: E731
    assert [f(s) for s in (0, 4, 9)] == pytest.approx([0.1, 0.5, 1.0])
    assert f(10) == pytest.approx(1.0) and f(55) == pytest.approx(0.5)
    assert 0 < f(99) < 0.001 and f(100) == pytest.approx(0.0)
    assert lr_factor(0, steps=4, warmup_steps=0, min_lr_ratio=0.1) == pytest.approx(1.0)


def _bank(dims, task="semseg"):
    tc = resolve_task(load_config(), task, tiny=True)
    return ProbeBank(tc, dims, device="cpu", base_seed=1)


def test_bank_chunks_by_grid_and_state_round_trip():
    bank = _bank({PRE_PROJ: 16, 0: 8, 1: 8})
    assert bank.names == ["pre_proj/linear", "pre_proj/conv", "L0/linear", "L0/conv", "L1/linear", "L1/conv"]
    chunks = bank.chunks({PRE_PROJ: (30, 40), 0: (15, 20), 1: (15, 20)}, 3)
    assert chunks == [["pre_proj/linear", "pre_proj/conv"], ["L0/linear", "L0/conv", "L1/linear"], ["L1/conv"]]
    feats = {PRE_PROJ: torch.randn(16, 30, 40), 0: torch.randn(8, 15, 20), 1: torch.randn(8, 15, 20)}
    out = bank.predict(chunks[1], feats)
    assert out.shape == (3, 150, 15, 20)
    out.sum().backward()
    bank.step()
    st = bank.state_dict()
    other = _bank({PRE_PROJ: 16, 0: 8, 1: 8})
    other.load_state_dict(st)
    for n in bank.names:
        for p, q in zip(bank.entries[n].module.parameters(), other.entries[n].module.parameters()):
            assert torch.equal(p, q)
        assert other.entries[n].sched.last_epoch == bank.entries[n].sched.last_epoch == 1
    with pytest.raises(ValueError):
        _bank({0: 8}).load_state_dict(st)


def test_chunk_size_is_capped_by_upsampled_elements():
    """一块的探针数：min(probe_chunk, probe_chunk_max_elements // (C·H·W))，按真实数据的典型尺寸钉住（配置注释里的数）。"""
    from types import SimpleNamespace

    cfg = load_config()
    want = {("semseg", (512, 683)): 2, ("depth", (480, 640)): 8, ("normals", (480, 640)): 8, ("refseg", (480, 640)): 8,
            ("semseg", (2048, 1536)): 1}
    for (task, hw), size in want.items():
        tc = resolve_task(cfg, task, tiny=False)
        bank = _bank({0: 4, 1: 4, 2: 4, 3: 4, 4: 4}, task=task)
        run = SimpleNamespace(task=SimpleNamespace(out_channels=tc["out_channels"]), tc=tc, bank=bank)
        chunks = TaskRun._chunks(run, {k: torch.zeros(4, 15, 20) for k in range(5)}, hw)
        assert max(len(c) for c in chunks) == min(size, 10), (task, hw)


def test_chunked_losses_give_each_probe_its_own_gradient():
    """各探针的损失相加后一起反传：每个探针的梯度与单独反传时相同（梯度互不相干）。"""
    from src.e2.tasks import make_task, upsample

    tc = resolve_task(load_config(), "refseg", tiny=True)
    task = make_task(tc)
    from src.e2.tasks import Target

    m = (torch.rand(9, 12, generator=torch.Generator().manual_seed(0)) > 0.5).float()
    tgt = Target((9, 12), {"mask": m}, {"mask": m.bool().numpy()}, 108)
    feats = {0: torch.randn(8, 3, 4), 1: torch.randn(8, 3, 4)}
    a, b = _bank({0: 8, 1: 8}, task="refseg"), _bank({0: 8, 1: 8}, task="refseg")
    names = a.names
    task.loss(upsample(a.predict(names, feats), (9, 12), tc["upsample"]), tgt).sum().backward()
    for n in names:
        task.loss(upsample(b.predict([n], feats), (9, 12), tc["upsample"]), tgt).sum().backward()
    for n in names:
        for p, q in zip(a.entries[n].module.parameters(), b.entries[n].module.parameters()):
            assert torch.allclose(p.grad, q.grad, rtol=1e-6, atol=1e-8)


def test_channel_stats_chan_merge_matches_two_pass():
    torch.manual_seed(0)
    x = torch.randn(1000, 5, dtype=torch.float64) * 1e3 + 1e4  # 大激活通道：均值远大于标准差
    s = ChannelStats()
    for part in torch.split(x, [1, 300, 499, 200]):
        s.update(part)
    assert s.n == 1000
    assert torch.allclose(s.mean, x.mean(0), rtol=1e-12)
    assert torch.allclose(s.var(), x.var(0, unbiased=False), rtol=1e-10)
    r = ChannelStats.from_state(s.state())
    assert torch.equal(r.mean, s.mean) and torch.equal(r.var(), s.var())


# ---------------------------------------------------------------------------
# 特征来源
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def qwen():
    return make_adapter("qwen3.5-0.8b", device="cpu", dtype=torch.float32, tiny=True, init_std=0.05)


def test_entry_assertion_catches_hidden_states_on_the_wrong_device(qwen, monkeypatch):
    """开发机上不可能出现的设备错配：用 meta 设备做代理（dev.md），纯 Python 的断言照样触发。"""
    from src.data.train_loaders import load_ade20k

    s = load_ade20k(E2Paths.fixtures().train_root, "val").records[0]
    ex = FeatureExtractor(qwen, load_config(), "semseg", [0, 1], [])
    real = qwen.hidden_states

    def on_meta(inputs, layers, **kw):
        hs = real(inputs, layers, **kw)
        return type(hs)(layers=hs.layers, image={l: v.to("meta") for l, v in hs.image.items()}, query=None)

    monkeypatch.setattr(qwen, "hidden_states", on_meta)
    with pytest.raises(FeatureCheckError, match="meta"):
        ex.raw(s, phase="test")


def test_model_must_be_frozen_and_audit_is_recorded(qwen):
    from src.data.train_loaders import load_ade20k

    s = load_ade20k(E2Paths.fixtures().train_root, "val").records[0]
    ex = FeatureExtractor(qwen, load_config(), "semseg", [0, 1], [])
    ex.raw(s, phase="test")
    assert ex.audit["hidden_dtype"] == "torch.float32" and ex.audit["hidden_device"] == "cpu"
    assert ex.audit["hidden_requires_grad"] is False and ex.audit["grid_hw"] == [15, 20]
    qwen.model.train()
    try:
        with pytest.raises(FeatureCheckError, match="train 模式"):
            ex.raw(s, phase="test")
    finally:
        qwen.model.eval()


def test_post_proj_check_raises_when_layer0_differs(qwen, monkeypatch):
    from src.data.train_loaders import load_ade20k

    s = load_ade20k(E2Paths.fixtures().train_root, "val").records[0]
    ex = FeatureExtractor(qwen, load_config(), "semseg", [0, 1], ["post_proj"])
    ex.raw(s, phase="a")
    assert ex.post_proj_check("a")["equal"] is True
    ex.raw(s, phase="b")
    assert ex.post_proj_check("b") is None  # 只在第一次前向（以及被要求时）核对
    real = qwen.vision_features
    monkeypatch.setattr(qwen, "vision_features", lambda inputs: real(inputs) + 1e-3)
    ex.request_post_proj_check()
    with pytest.raises(FeatureCheckError, match="D21-3"):
        ex.raw(s, phase="c")
    with pytest.raises(ValueError, match="第 0 层"):
        FeatureExtractor(qwen, load_config(), "semseg", [1, 2], ["post_proj"])


def test_pre_proj_requires_adapter_interface(qwen):
    assert not hasattr(qwen, "pre_projection_features")
    with pytest.raises(NotImplementedError, match="pre_projection_features"):
        FeatureExtractor(qwen, load_config(), "depth", [0], ["pre_proj"])


def test_pre_proj_reference_with_different_dim_and_grid_end_to_end(qwen, monkeypatch, tmp_path):
    """给小模型临时挂一个 pre_projection_features（维度 24、网格是 LLM 网格的 2 倍）：探针按各自的维度构造，
    上采样按各自的网格，整条流程跑通，metrics 里有 pre_proj 的两个探针。"""
    from src.e2.grid import single_image

    proj = torch.randn(qwen.hidden_size(), 24, generator=torch.Generator().manual_seed(0))

    def pre(inputs):
        h, w = single_image(inputs).grid_hw
        v = qwen.vision_features(inputs) @ proj  # [h·w, 24]
        g = v.reshape(h, w, 24).repeat_interleave(2, 0).repeat_interleave(2, 1)
        return g.reshape(-1, 24), (2 * h, 2 * w)

    monkeypatch.setattr(qwen, "pre_projection_features", pre, raising=False)
    cfg = load_config()
    tc = resolve_task(cfg, "refseg", tiny=True)
    spec = RunSpec("qwen3.5-0.8b", "qwen3.5", "Qwen/Qwen3.5-0.8B", "0" * 40, "tiny", True, {"type": "PixelBudget"}, "fp32",
                   "refseg", "every 2", (0, 2, 4), ("post_proj", "pre_proj"), qwen.num_layers(), qwen.hidden_size())
    m = run_task(qwen, cfg, spec, tc, E2Paths.fixtures(), tmp_path, log=lambda _: None)
    assert m["feature_keys"] == ["pre_proj", "L0", "L2", "L4"]
    assert m["probes"]["pre_proj/linear"]["n_params"] == 24 * 1 + 1
    assert m["probes"]["L0/linear"]["n_params"] == qwen.hidden_size() + 1
    assert m["post_proj_check"]["equal"] is True and m["post_proj_check"]["sample"].startswith("refcoco_unc/val/")
    for n in ("pre_proj/linear", "pre_proj/conv"):
        assert 0.0 <= m["probes"][n]["eval"]["giou"] <= 1.0


def test_task_run_stats_cover_every_feature_key(qwen):
    cfg = load_config()
    tc = resolve_task(cfg, "depth", tiny=True)
    spec = RunSpec("qwen3.5-0.8b", "qwen3.5", "r", "0" * 40, "tiny", True, {}, "fp32", "depth", "all", (0, 4), (),
                   qwen.num_layers(), qwen.hidden_size())
    run = TaskRun(qwen, cfg, spec, tc, E2Paths.fixtures(), log=lambda _: None)
    run.load_data()
    stats = run.compute_stats(run.split.stats)
    assert set(stats) == {0, 4} and all(s.n == len(run.split.stats) * 300 for s in stats.values())
    run.set_stats(stats)
    assert run.bank.names == ["L0/linear", "L0/conv", "L4/linear", "L4/conv"]
    feats = run.features(run.split.val[0], "test")
    for k, v in feats.items():  # 标准化之后在标准化样本上接近零均值
        assert v.shape == (qwen.hidden_size(), 15, 20) and math.isfinite(float(v.abs().max()))
    sums, used = run.train_step(0, run.split.train)
    assert used == tc["batch_size"] and set(sums) == set(run.bank.names)
    assert run.counters.by_phase["train"]["n_forward"] == tc["batch_size"]
    assert run.counters.by_phase["stats"]["visual_tokens"] == 300 * len(run.split.stats)
