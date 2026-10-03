"""DEV-0031：E2 的双线性上采样改为可分离矩阵乘（src/e2/bilinear.py；D63-3）。

- 插值矩阵与 PyTorch 的 CPU 核（F.interpolate 的权重，用单位脉冲取出）一致：误差不超过源坐标的 1 ulp（开发机 ARM 上逐位相同）；
- 前向与对输入的梯度与 F.interpolate 等价：float64 最大绝对差 ≤ TOL_F64（1e-12），fp32 相对差 ≤ TOL_F32_REL（1e-5）；
  形状含上采样、下采样、恒等、奇数尺寸、1 像素宽，以及 E2 实际的网格与 GT 尺寸；align_corners 两种；
- 精度：作用域内矩阵乘是 IEEE fp32（不用 TF32），与全局设置无关。CPU 上没有 TF32（cuBLAS 才有），"全局 TF32 打开后结果不变"
  在 CPU 上恒成立、证明不了什么；所以另有两条腿：① 把矩阵乘换成探针，核对前向与反传的每次矩阵乘执行时精度旗标都是 ieee
  （任何机器上都能观测）；② 运行期审计 fp32_audit 在真设备上故意打开 TF32、与 float64 参照比较，这里用一个模拟 TF32 舍入的
  假矩阵乘核对它确实会报错。有 CUDA 时，"全局 TF32 打开后结果不变"在 GPU 上真跑一遍。
"""

from __future__ import annotations

import json
import platform
import sys

import pytest
import torch
import torch.nn.functional as F

from src.e2 import bilinear as B
from src.e2 import tasks as T
from src.e2.config import load_config, resolve_task

E2_SHAPES = [  # E2 实际出现的 token 网格 → GT 尺寸
    ((16, 21), (512, 683)),  # ADE 的典型尺寸（RUN-0026 微基准）
    ((15, 20), (480, 640)),  # Qwen，NYU 640×480
    ((14, 19), (480, 640)),  # Gemma，NYU
    ((16, 16), (480, 640)),  # InternVL，NYU
    ((14, 20), (427, 640)),  # RefCOCO 的常见尺寸
]
OTHER_SHAPES = [
    ((5, 7), (13, 11)),  # 奇数尺寸上采样
    ((7, 9), (7, 9)),  # 恒等
    ((32, 40), (15, 20)),  # 下采样
    ((480, 640), (15, 20)),  # 大幅下采样
    ((4, 40), (16, 10)),  # 一维放大、一维缩小
    ((1, 1), (5, 7)),  # 1 像素输入
    ((3, 1), (9, 1)),  # 1 像素宽
    ((4, 6), (8, 1)),  # 输出 1 像素宽
    ((1, 5), (1, 17)),  # 1 像素高
    ((2, 2), (1, 1)),  # 输出 1 个像素
]
ALL_SHAPES = E2_SHAPES + OTHER_SHAPES


def _pair(hw, big, dtype, ac, seed=0, pc=(2, 3)):
    """同一组输入与上游梯度，分别走可分离矩阵乘与 F.interpolate，返回 (前向, 对输入的梯度) 两对。"""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(*pc, *hw, generator=g, dtype=dtype)
    gy = torch.randn(*pc, *big, generator=g, dtype=dtype)
    xa, xb = x.clone().requires_grad_(True), x.clone().requires_grad_(True)
    ya = B.separable_bilinear(xa, big, align_corners=ac)
    yb = F.interpolate(xb, size=big, mode="bilinear", align_corners=ac)
    ya.backward(gy)
    yb.backward(gy)
    return (ya.detach(), yb.detach()), (xa.grad, xb.grad)


def _rel(a, b):
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-300))


# ---------------------------------------------------------------------------
# 插值矩阵
# ---------------------------------------------------------------------------


def _torch_weights(n_in, n_out, ac, dtype):
    """F.interpolate 的权重矩阵 [n_out, n_in]：把 n_in 个单位脉冲当作 n_in 个通道，沿宽度插值（高度 1 → 1 是恒等）。"""
    eye = torch.eye(n_in, dtype=dtype).reshape(1, n_in, 1, n_in)
    return F.interpolate(eye, size=(1, n_out), mode="bilinear", align_corners=ac)[0, :, 0, :].T


# 开发机（macOS arm64，clang）上 PyTorch 的 CPU 核把 scale·(o + 0.5) − 0.5 收缩成 FMA，与 bilinear 的构造逐位相同；
# 其他平台（如 x86 的服务器 CPU）是否收缩取决于编译，只要求差不超过源坐标的 1 ulp
EXACT_PLATFORM = sys.platform == "darwin" and platform.machine() == "arm64"


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("ac", [False, True])
def test_interp_matrix_matches_pytorch_weights(dtype, ac):
    """逐个尺寸对比 PyTorch 核的权重：差不超过源坐标的 1 ulp（n_in·eps；FMA 收缩与否只差这么多），开发机上逐位相同；
    并核对结构：每行至多两个非零、在相邻两列（上边界时并成一列）、和为 1。"""
    eps = torch.finfo(dtype).eps
    for n_in in list(range(1, 42)) + [64, 480, 640]:
        for n_out in (1, 2, 3, 7, 13, 64, 427, 480, 512, 640, 683):
            ours = B.interp_matrix(n_in, n_out, ac, torch.device("cpu"), dtype)
            ref = _torch_weights(n_in, n_out, ac, dtype)
            assert ours.shape == (n_out, n_in) and ours.dtype == dtype
            assert float((ours - ref).abs().max()) <= n_in * eps, (n_in, n_out, ac)
            if EXACT_PLATFORM:
                assert torch.equal(ours, ref), (n_in, n_out, ac)
            nz = ours != 0
            assert int(nz.sum(dim=1).max()) <= 2
            cols = [torch.nonzero(r).flatten().tolist() for r in nz]
            assert all(len(c) == 1 or c[1] == c[0] + 1 for c in cols)
            assert float((ours.sum(dim=1) - 1).abs().max()) <= 2 * eps


def test_interp_weights_follow_the_source_rules():
    """源码里的几条规则：同尺寸直接拷贝；align_corners=False 的负源坐标截到 0；上边界 i1 = i0；align_corners 时 n_out = 1 取第 0 格。"""
    i0, i1, l0, l1 = B.interp_weights(5, 5, False)
    assert i0.tolist() == i1.tolist() == list(range(5)) and l0.tolist() == [1.0] * 5 and l1.tolist() == [0.0] * 5
    i0, i1, l0, l1 = B.interp_weights(2, 4, False)  # src = 0.5·(o + 0.5) − 0.5 = [−0.25, 0.25, 0.75, 1.25]
    assert i0.tolist() == [0, 0, 0, 1] and i1.tolist() == [1, 1, 1, 1]
    assert l1.tolist() == [0.0, 0.25, 0.75, 0.25] and l0.tolist() == [1.0, 0.75, 0.25, 0.75]
    i0, i1, l0, l1 = B.interp_weights(6, 1, True)
    assert i0.tolist() == [0] and l1.tolist() == [0.0]
    with pytest.raises(ValueError):
        B.interp_weights(0, 3, False)


def test_interp_matrix_is_cached_per_key():
    """按 (n_in, n_out, align_corners, device, dtype) 缓存：同一组参数返回同一个张量，任一项不同就是另一条。"""
    cpu = torch.device("cpu")
    a = B.interp_matrix(16, 512, False, cpu, torch.float32)
    assert B.interp_matrix(16, 512, False, cpu, torch.float32) is a
    others = [B.interp_matrix(16, 512, True, cpu, torch.float32), B.interp_matrix(16, 512, False, cpu, torch.float64),
              B.interp_matrix(17, 512, False, cpu, torch.float32), B.interp_matrix(16, 513, False, cpu, torch.float32)]
    assert all(o is not a for o in others)
    assert B.interp_matrix.cache_info().maxsize == B.MATRIX_CACHE_SIZE
    with pytest.raises(ValueError):
        B.interp_matrix(4, 8, False, cpu, torch.bfloat16)


# ---------------------------------------------------------------------------
# 与 F.interpolate 等价（前向与对输入的梯度）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ac", [False, True])
@pytest.mark.parametrize("hw,big", ALL_SHAPES)
def test_equivalent_to_interpolate_float64(hw, big, ac):
    (ya, yb), (ga, gb) = _pair(hw, big, torch.float64, ac)
    assert ya.shape == yb.shape and ga.shape == gb.shape
    assert float((ya - yb).abs().max()) <= B.TOL_F64
    assert float((ga - gb).abs().max()) <= B.TOL_F64


@pytest.mark.parametrize("ac", [False, True])
@pytest.mark.parametrize("hw,big", ALL_SHAPES)
def test_equivalent_to_interpolate_fp32(hw, big, ac):
    (ya, yb), (ga, gb) = _pair(hw, big, torch.float32, ac)
    assert ya.dtype == torch.float32 and ga.dtype == torch.float32
    assert _rel(ya, yb) <= B.TOL_F32_REL
    assert _rel(ga, gb) <= B.TOL_F32_REL


def test_semseg_chunk_shape_fp32():
    """E2 semseg 实际的一块：2 个探针 × 150 类，16×21 → 512×683（RUN-0026 微基准的形状）。"""
    (ya, yb), (ga, gb) = _pair((16, 21), (512, 683), torch.float32, False, pc=(2, 150))
    assert ya.is_contiguous() and ya.shape == (2, 150, 512, 683)
    assert _rel(ya, yb) <= B.TOL_F32_REL and _rel(ga, gb) <= B.TOL_F32_REL


def test_gradcheck_float64():
    """反传是前向的伴随（A_hᵀ·G·A_w）：autograd 的数值梯度检查。"""
    for hw, big, ac in [((3, 4), (7, 5), False), ((2, 5), (6, 3), True), ((1, 3), (4, 1), False)]:
        x = torch.randn(2, 2, *hw, dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(lambda t: B.separable_bilinear(t, big, align_corners=ac), (x,))


def test_nonfinite_input_spreads_within_its_own_slice_only():
    """文档里写明的差别：矩阵乘里 0·NaN = NaN，一个 NaN 会污染这个 (p, c) 的整张输出（F.interpolate 只影响相邻像素）；
    别的 (p, c) 不受影响。有限输入不受这条影响。"""
    x = torch.randn(2, 3, 4, 5)
    x[1, 2, 0, 0] = float("nan")
    y = B.separable_bilinear(x, (9, 11), align_corners=False)
    ref = F.interpolate(x, size=(9, 11), mode="bilinear", align_corners=False)
    assert torch.isnan(y[1, 2]).all() and not torch.isnan(ref[1, 2]).all()
    keep = torch.ones(2, 3, dtype=torch.bool)
    keep[1, 2] = False
    assert torch.isfinite(y[keep]).all() and _rel(y[keep], ref[keep]) <= B.TOL_F32_REL


def test_rejects_bad_inputs():
    with pytest.raises(ValueError, match="P, C, h, w"):
        B.separable_bilinear(torch.zeros(3, 4, 5), (8, 8), align_corners=False)
    with pytest.raises(ValueError, match="只接受"):
        B.separable_bilinear(torch.zeros(1, 1, 4, 5, dtype=torch.bfloat16), (8, 8), align_corners=False)


# ---------------------------------------------------------------------------
# tasks.upsample 与配置
# ---------------------------------------------------------------------------


def test_config_selects_separable_matmul_and_pins_the_tolerances():
    cfg = load_config()
    up = cfg.doc["upsample"]
    assert up["impl"] == "separable_matmul" and up["mode"] == "bilinear" and up["align_corners"] is False
    assert B.IMPLS == ("separable_matmul", "interpolate") == T.UPSAMPLE_IMPLS
    # 配置注释里的数与代码、测试用的同一组（dev.md：配置注释里的数字要有测试钉住）
    assert B.TOL_F64 == 1e-12 and B.TOL_F32_REL == 1e-5
    for s in ("≤ 1e-12", "≤ 1e-5", "tests/test_e2_bilinear.py", "D63-3", "RUN-0026", "122 ms", "0.8 ms", "interpolate"):
        assert s in up["basis"], s
    for task in ("depth", "semseg", "normals", "refseg"):  # 进运行身份（config.yaml 的 training.upsample）
        assert resolve_task(cfg, task, tiny=False)["upsample"] == {"mode": "bilinear", "align_corners": False,
                                                                   "impl": "separable_matmul"}


def test_upsample_dispatches_on_impl(monkeypatch):
    """训练与评测共用 tasks.upsample：separable_matmul 走两次矩阵乘（bilinear._mm），interpolate 与 F.interpolate 逐位相同。"""
    B.fp32_audit("cpu")  # 先把本进程的审计做掉（缓存），下面只数上采样本身的矩阵乘
    calls = []
    real = B._mm
    monkeypatch.setattr(B, "_mm", lambda a, b: calls.append(1) or real(a, b))
    tc = resolve_task(load_config(), "semseg", tiny=False)
    x = torch.randn(2, 4, 6, 7)
    y = T.upsample(x, (13, 17), tc["upsample"])
    assert len(calls) == 2 and y.shape == (2, 4, 13, 17)
    ref = F.interpolate(x, size=(13, 17), mode="bilinear", align_corners=False)
    assert _rel(y, ref) <= B.TOL_F32_REL
    calls.clear()
    y2 = T.upsample(x, (13, 17), {**tc["upsample"], "impl": "interpolate"})
    assert not calls and torch.equal(y2, ref)


@pytest.mark.parametrize("spec,msg", [
    ({"mode": "bilinear", "align_corners": False}, "impl"),
    ({"mode": "bilinear", "align_corners": False, "impl": "fft"}, "impl"),
    ({"mode": "nearest", "align_corners": False, "impl": "separable_matmul"}, "双线性"),
    ({"mode": "bilinear", "align_corners": 0, "impl": "separable_matmul"}, "布尔"),
])
def test_upsample_spec_is_checked_before_the_run(spec, msg):
    """配置的 upsample 在构造任务时就查（不设默认实现）。"""
    tc = resolve_task(load_config(), "depth", tiny=True)
    with pytest.raises(ValueError, match=msg):
        T.make_task({**tc, "upsample": spec})
    with pytest.raises(ValueError, match=msg):
        T.upsample(torch.zeros(1, 1, 2, 2), (4, 4), spec)


# ---------------------------------------------------------------------------
# fp32 精度（不用 TF32）
# ---------------------------------------------------------------------------


def _defaults():
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.fp32_precision = "none"
    torch.backends.mkldnn.matmul.fp32_precision = "none"
    torch.backends.fp32_precision = "none"


@pytest.fixture
def restore_flags():
    yield
    _defaults()


def _enable_global_tf32():
    torch.backends.cuda.matmul.allow_tf32 = True  # 旧接口（最常见的写法）
    torch.set_float32_matmul_precision("high")  # 旧接口；同时把 mkldnn 的 matmul 设成 tf32


def test_every_matmul_runs_with_ieee_flags_even_if_global_tf32_is_on(monkeypatch, restore_flags):
    """开发机上可观测的代理：全局 TF32 打开，前向与反传的每次矩阵乘执行时，新旧接口都读出 IEEE fp32；退出后全局设置原样恢复。"""
    seen = []
    real = B._mm
    monkeypatch.setattr(B, "_mm", lambda a, b: seen.append(B.precision_state()) or real(a, b))
    _enable_global_tf32()
    before = B.precision_state()
    assert before == {"cuda.matmul": "tf32", "mkldnn.matmul": "tf32", "legacy": "high"}
    x = torch.randn(2, 3, 5, 6, requires_grad=True)
    B.separable_bilinear(x, (11, 13), align_corners=False, audit=False).square().sum().backward()
    assert len(seen) == 4  # 前向 2 次、反传 2 次
    assert all(s == {"cuda.matmul": "ieee", "mkldnn.matmul": "ieee", "legacy": "highest"} for s in seen), seen
    assert B.precision_state() == before and torch.backends.cuda.matmul.allow_tf32 is True


@pytest.mark.parametrize("setup", [
    "pass",
    "torch.backends.cuda.matmul.allow_tf32 = True",
    "torch.set_float32_matmul_precision('high')",
    "torch.set_float32_matmul_precision('medium')",
    "torch.backends.cuda.matmul.fp32_precision = 'tf32'",  # 只用新接口：新旧接口不一致（读旧接口会报错）
    "torch.backends.fp32_precision = 'tf32'",
])
def test_guard_restores_every_flag_even_on_error(setup, restore_flags):
    exec(setup)
    g = torch._C._get_fp32_precision_getter
    pairs = [(b, o) for b in ("generic", "cuda", "mkldnn") for o in ("all", "matmul", "conv", "rnn") if not (b == "generic" and o != "all")]
    snap = lambda: ({p: g(*p) for p in pairs}, B.precision_state(), torch.backends.cudnn.allow_tf32)  # noqa: E731
    before = snap()
    with B.ieee_fp32_matmul():
        inside = B.precision_state()
    assert inside["cuda.matmul"] == inside["mkldnn.matmul"] == "ieee" and inside["legacy"] in ("highest", None)
    assert snap() == before
    with pytest.raises(KeyError):
        with B.ieee_fp32_matmul():
            raise KeyError("x")
    assert snap() == before


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="没有 CUDA：CPU 上没有 TF32，这条的含义由上面的探针测试与运行期审计承担"))])
def test_global_tf32_does_not_change_the_result(device, restore_flags):
    """故意打开全局 TF32：结果与关闭时逐位相同（GPU 上是真考验；CPU 上没有 TF32，恒成立）。"""
    g = torch.Generator().manual_seed(1)
    c = 150 if device == "cuda" else 8  # GPU 上用 semseg 一块的真实通道数；CPU 上只为跑通，取小
    x = torch.randn(2, c, 16, 21, generator=g).to(device)
    gy = torch.randn(2, c, 512, 683, generator=g).to(device)

    def run():
        xx = x.clone().requires_grad_(True)
        y = B.separable_bilinear(xx, (512, 683), align_corners=False)
        y.backward(gy)
        return y.detach(), xx.grad

    _defaults()
    y0, g0 = run()
    _enable_global_tf32()
    y1, g1 = run()
    assert torch.equal(y0, y1) and torch.equal(g0, g1)


def test_fp32_audit_passes_and_is_cached(monkeypatch):
    monkeypatch.setattr(B, "_AUDITS", {})
    a = B.fp32_audit("cpu")
    assert a["ok"] and a["guarded_fwd_rel"] <= B.TOL_F32_REL and a["guarded_bwd_rel"] <= B.TOL_F32_REL
    assert a["global_state_during_audit"]["cuda.matmul"] == "tf32"  # 审计时全局 TF32 确实是开着的
    assert a["tf32_effective"] is False  # CPU 上没有 TF32
    assert B.fp32_audit(torch.device("cpu")) is a and set(B.audits()) == {"cpu"}
    json.dumps(B.audits())  # 能进运行记录


def test_fp32_audit_catches_reduced_precision_matmul(monkeypatch):
    """审计有区分力：把矩阵乘换成模拟 TF32 的版本（fp32 输入截到 10 位尾数），受保护的误差超出 fp32 的范围，审计报错。"""
    real = B._mm

    def tf32_like(a, b):
        if a.dtype != torch.float32:
            return real(a, b)
        cut = lambda t: (t.contiguous().view(torch.int32) & ~((1 << 13) - 1)).view(torch.float32)  # noqa: E731
        return real(cut(a), cut(b))

    monkeypatch.setattr(B, "_AUDITS", {})
    monkeypatch.setattr(B, "_mm", tf32_like)
    with pytest.raises(B.Fp32PrecisionError, match="不是 IEEE fp32"):
        B.fp32_audit("cpu")
    assert B.audits() == {}


def test_upsample_runs_the_audit_once_per_device(monkeypatch):
    monkeypatch.setattr(B, "_AUDITS", {})
    calls = []
    real = B.fp32_audit
    monkeypatch.setattr(B, "fp32_audit", lambda d: calls.append(str(d)) or real(d))
    tc = resolve_task(load_config(), "refseg", tiny=True)
    for _ in range(3):
        T.upsample(torch.randn(1, 1, 3, 4), (6, 8), tc["upsample"])
    assert calls == ["cpu"] * 3 and set(B.audits()) == {"cpu"}  # 每次调用都经过审计入口（第一次真审计，之后取缓存）
