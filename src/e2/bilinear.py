"""E2 的双线性上采样：可分离矩阵乘实现（DEV-0031；D63-3，RUN-0026 的 B200 微基准）。

为什么：F.interpolate(mode="bilinear") 的反传用原子加，ADE 的 ADE_NUM_CLASSES 类、全分辨率下极慢（B200 上一块 2 个探针 × ADE_NUM_CLASSES 类、16×21 → 512×683
前向加反传 122 ms）。同一个双线性插值写成两次矩阵乘只要 0.8 ms。损失与指标的定义不变，只换实现。

定义：out = A_h · X · A_wᵀ，对 [P, C, h, w] 的每个 (p, c) 做；A_h [H, h]、A_w [W, w] 是行、列两个插值矩阵，每行至多两个非零权重。
插值矩阵按 PyTorch 的源码构造（aten/src/ATen/native/UpSample.h），与 F.interpolate(mode="bilinear", antialias=False) 逐项对应：
- scale = area_pixel_compute_scale：align_corners 时 (n_in − 1)/(n_out − 1)（n_out = 1 时为 0）；否则 compute_scales_value，
  F.interpolate 给 size、不给 scale_factor 时就是 n_in / n_out；
- 源坐标 = area_pixel_compute_source_index(cubic=False)：align_corners 时 scale·o；否则 scale·(o + 0.5) − 0.5，负值截到 0；
- guard_index_and_lambda：i0 = min(floorf(src), n_in − 1)，λ1 = min(max(src − i0, 0), 1)；
- compute_source_index_and_lambda：n_out = n_in 时直接拷贝（i0 = i1 = o，λ0 = 1，λ1 = 0）；否则 i1 = i0 + (i0 < n_in − 1 ? 1 : 0)，
  λ0 = 1 − λ1。i0 = i1 时（上边界）两项权重加在同一格。
- 坐标与权重在 opmath 精度里算：fp32 输入用 float32、float64 输入用 float64（CPU 核的 opmath_t；CUDA 核的 accscalar_t 对 float
  也是 float）。floorf 照源码先转 float32 再取整。scale·(o + 0.5) − 0.5 按"乘加只舍入一次"（编译器收缩成 FMA）算：开发机（ARM）上
  PyTorch 的 CPU 核就是这样，插值矩阵与它的权重逐位相同（fp32 与 float64；tests/test_e2_bilinear.py 在 macOS arm64 上逐位核对，
  其余平台核对差不超过源坐标的 1 ulp）；不收缩的平台上 λ 差 1 ulp，只影响舍入。

与 F.interpolate 的差别只在舍入：F.interpolate 先沿 W 再沿 H 插值（λh0·(λw0·x00 + λw1·x01) + …），这里先沿 H（小张量）再沿 W
（大输出，一次 GEMM 直接写出连续的 [P, C, H, W]），累加顺序不同。等价性测试（tests/test_e2_bilinear.py）：float64 前向与对输入的
梯度最大绝对差 ≤ TOL_F64，fp32 相对差（max|差| / max|参照|）≤ TOL_F32_REL。另一处差别：矩阵乘里 0·NaN = NaN、0·inf = NaN，
输入里一个非有限值会污染这个 (p, c) 的整张输出（F.interpolate 只影响相邻像素），别的 (p, c) 不受影响；有限输入不受这条影响。
探针输出出现非有限值时训练本已失败。

反传：自定义 autograd.Function，grad_X = A_hᵀ · G · A_w，同样是两次 GEMM，没有原子加；前向只存两个插值矩阵。

fp32 的保证（不用 TF32）：前向与反传的矩阵乘都在 ieee_fp32_matmul() 里做，不依赖进程的全局设置。机制（torch 2.13 / 2.14 的 API）：
- 新接口 torch.backends.cuda.matmul.fp32_precision 与 torch.backends.mkldnn.matmul.fp32_precision 临时设成 "ieee"
  （cuBLAS / cuBLASLt 选 fp32 计算类型时读的就是前者：ATen/cuda/detail/CublasLtUtils.h 的 float32Precision(CUDA, MATMUL)）；
- 旧接口 torch.set_float32_matmul_precision 能读出时同时设成 "highest"，作用域内新旧接口一致（否则读 allow_tf32 会报"新旧接口混用"）；
- 设完读回核对（入口断言，不符就抛 Fp32PrecisionError），退出时（含异常）把新旧接口恢复成进入前的值。
这是进程级的全局状态，在作用域内临时改写：E2 的训练单线程同步（backward 期间主线程阻塞），作用域内不会有别的矩阵乘。
反传在 autograd 的设备线程上执行，backward 里再进一次同一个作用域。
运行期审计 fp32_audit(device)：每个进程、每个设备第一次上采样时，故意把全局 TF32 打开，跑一遍受保护的前向与反传，与 float64 参照
比较（相对差必须 ≤ TOL_F32_REL，否则抛 Fp32PrecisionError），同时记下不受保护的矩阵乘在 TF32 下的误差（证明这台设备上 TF32 确实
生效、审计有区分力）。结果进运行记录（runner 的 logs/run_k.json 的 audit.upsample，--estimate 的 estimate.json）。
"""

from __future__ import annotations

import contextlib
import functools
from fractions import Fraction
from typing import Any, Iterator

import torch
from torch.autograd.function import once_differentiable

IMPLS: tuple[str, ...] = ("separable_matmul", "interpolate")  # configs/e2_probes.json 的 upsample.impl
TOL_F64 = 1e-12  # float64：与 F.interpolate 的前向、对输入的梯度最大绝对差（tests 钉住）
TOL_F32_REL = 1e-5  # fp32：相对差 max|差| / max|参照|（tests 与运行期审计共用）
MATRIX_CACHE_SIZE = 1024  # 插值矩阵的 LRU 缓存条数（ADE 的图尺寸各异；一条约 n_out·n_in·4 字节）

_FLOAT_DTYPES = (torch.float32, torch.float64)


class Fp32PrecisionError(RuntimeError):
    """矩阵乘没有按 IEEE fp32 执行（精度旗标读回不符，或运行期审计的误差超出 fp32 的范围）。"""


# ---------------------------------------------------------------------------
# 插值矩阵
# ---------------------------------------------------------------------------


def _opmath(dtype: torch.dtype) -> torch.dtype:
    return torch.float64 if dtype == torch.float64 else torch.float32


def _fma_source_index(scale: torch.Tensor, n_out: int, op: torch.dtype) -> torch.Tensor:
    """align_corners=False 的源坐标 scale·(o + 0.5) − 0.5，按"乘加只舍入一次"（FMA 收缩）算到 op 精度。

    编译器会把这个式子收缩成 FMA（开发机 ARM/clang 上的 CPU 核实测如此：tests 核对插值矩阵与 F.interpolate 的权重逐位相同；
    nvcc 默认 --fmad=true，CUDA 核同理）。fp32：两个 fp32 的积在 float64 里精确（48 位），再减 0.5 仍精确，最后舍入一次到 fp32；
    float64：用有理数精确算，再舍入一次（float(Fraction) 是正确舍入）。不收缩的平台上与这里差 1 ulp（λ 差约 n_in·2⁻²⁴），只影响舍入。"""
    if op == torch.float32:
        o64 = torch.arange(n_out, dtype=torch.float64)
        return (scale.to(torch.float64) * (o64 + 0.5) - 0.5).to(torch.float32)
    s, half = Fraction(float(scale)), Fraction(1, 2)
    return torch.tensor([float(s * (k + half) - half) for k in range(n_out)], dtype=torch.float64)


def interp_weights(n_in: int, n_out: int, align_corners: bool, dtype: torch.dtype = torch.float32
                   ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """一个维度上的 (i0, i1, λ0, λ1)，各 [n_out]（CPU）。逐项照 PyTorch 源码（模块说明），坐标在 opmath(dtype) 里算。"""
    n_in, n_out = int(n_in), int(n_out)
    if n_in < 1 or n_out < 1:
        raise ValueError(f"插值的输入、输出长度必须 ≥ 1，收到 {n_in} → {n_out}")
    op = _opmath(dtype)
    o = torch.arange(n_out, dtype=torch.int64)
    if n_out == n_in:  # compute_source_index_and_lambda：scale_factor = 1, simply copy
        return o, o.clone(), torch.ones(n_out, dtype=op), torch.zeros(n_out, dtype=op)
    if align_corners:  # area_pixel_compute_scale
        scale = torch.tensor(n_in - 1, dtype=op) / (n_out - 1) if n_out > 1 else torch.tensor(0, dtype=op)
        src = scale * o.to(op)  # area_pixel_compute_source_index：scale * dst_index
    else:
        scale = torch.tensor(n_in, dtype=op) / n_out  # compute_scales_value：input_size / output_size
        src = _fma_source_index(scale, n_out, op)  # scale * (dst_index + 0.5) - 0.5
        src = torch.where(src < 0, torch.zeros((), dtype=op), src)  # !cubic && src_idx < 0 → 0
    # guard_index_and_lambda：floorf（先转 float32）、不超过 n_in − 1；λ1 截到 [0, 1]
    i0 = torch.floor(src.to(torch.float32)).to(torch.int64).clamp_max(n_in - 1)
    l1 = (src - i0.to(op)).clamp(0, 1)
    i1 = i0 + (i0 < n_in - 1).to(torch.int64)
    l0 = 1 - l1
    return i0, i1, l0, l1


@functools.lru_cache(maxsize=MATRIX_CACHE_SIZE)
def interp_matrix(n_in: int, n_out: int, align_corners: bool, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """[n_out, n_in] 的插值矩阵（第 o 行在 i0、i1 两列放 λ0、λ1；i0 = i1 时相加），dtype 与输入相同，放在 device 上。

    按 (n_in, n_out, align_corners, device, dtype) 缓存（LRU，MATRIX_CACHE_SIZE 条）；返回的张量只读，调用方不得原地修改。"""
    if dtype not in _FLOAT_DTYPES:
        raise ValueError(f"插值矩阵只支持 {_FLOAT_DTYPES}，收到 {dtype}")
    i0, i1, l0, l1 = interp_weights(n_in, n_out, align_corners, dtype)
    o = torch.arange(int(n_out), dtype=torch.int64)
    a = torch.zeros(int(n_out), int(n_in), dtype=_opmath(dtype))
    a.index_put_((o, i0), l0, accumulate=True)
    a.index_put_((o, i1), l1, accumulate=True)
    return a.to(dtype=dtype, device=device)


# ---------------------------------------------------------------------------
# fp32 精度（不用 TF32）
# ---------------------------------------------------------------------------

_LEGACY = {"ieee": "highest", "tf32": "high"}


def _new_api_knobs() -> tuple[Any, Any]:
    return torch.backends.cuda.matmul, torch.backends.mkldnn.matmul


def _read_legacy() -> str | None:
    """旧接口的值；新旧接口已经不一致（读它会报"混用"）时返回 None，此时只动新接口。"""
    try:
        return torch.get_float32_matmul_precision()
    except RuntimeError:
        return None


def precision_state() -> dict[str, str | None]:
    """当前的矩阵乘精度旗标（记录与测试用）：新接口 cuda / mkldnn 的 matmul，旧接口（不一致时为 None）。"""
    cu, mk = _new_api_knobs()
    return {"cuda.matmul": cu.fp32_precision, "mkldnn.matmul": mk.fp32_precision, "legacy": _read_legacy()}


@contextlib.contextmanager
def matmul_precision(prec: str) -> Iterator[None]:
    """作用域内把 fp32 矩阵乘的精度设成 prec（"ieee" 或 "tf32"），退出时（含异常）恢复进入前的新旧接口。"""
    if prec not in _LEGACY:
        raise ValueError(f"prec 只能是 {sorted(_LEGACY)}，收到 {prec!r}")
    cu, mk = _new_api_knobs()
    saved_new = (cu.fp32_precision, mk.fp32_precision)
    saved_legacy = _read_legacy()
    try:
        if saved_legacy is not None:
            torch.set_float32_matmul_precision(_LEGACY[prec])
        cu.fp32_precision = prec
        mk.fp32_precision = prec
        got = (cu.fp32_precision, mk.fp32_precision)
        if got != (prec, prec):
            raise Fp32PrecisionError(f"矩阵乘精度设成 {prec!r} 后读回 cuda.matmul={got[0]!r}、mkldnn.matmul={got[1]!r}")
        yield
    finally:
        if saved_legacy is not None:
            torch.set_float32_matmul_precision(saved_legacy)
        cu.fp32_precision, mk.fp32_precision = saved_new


def ieee_fp32_matmul() -> contextlib.AbstractContextManager[None]:
    """作用域内的 fp32 矩阵乘是 IEEE fp32（不用 TF32 / bf16），与全局设置无关。"""
    return matmul_precision("ieee")


# ---------------------------------------------------------------------------
# 可分离的双线性插值
# ---------------------------------------------------------------------------


def _mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """全部矩阵乘都经过这里（测试替换它，核对调用时的精度旗标）。"""
    return torch.mm(a, b)


def _apply(x: torch.Tensor, ah: torch.Tensor, aw: torch.Tensor) -> torch.Tensor:
    """x [N, h, w] → A_h · x[n] · A_wᵀ，[N, H, W]（连续）。先沿 H（小张量），再沿 W（一次 GEMM 写出整个输出）。"""
    n, h, w = x.shape
    big_h, big_w = ah.shape[0], aw.shape[0]
    z = _mm(ah, x.transpose(0, 1).reshape(h, n * w))  # [H, N·w]
    z = z.reshape(big_h, n, w).transpose(0, 1).reshape(n * big_h, w)  # [N·H, w]
    return _mm(z, aw.t()).reshape(n, big_h, big_w)


def _apply_t(g: torch.Tensor, ah: torch.Tensor, aw: torch.Tensor) -> torch.Tensor:
    """伴随：g [N, H, W] → A_hᵀ · g[n] · A_w，[N, h, w]。先沿 W（读一遍大梯度），再沿 H。"""
    n, big_h, big_w = g.shape
    h, w = ah.shape[1], aw.shape[1]
    z = _mm(g.reshape(n * big_h, big_w), aw)  # [N·H, w]
    z = z.reshape(n, big_h, w).transpose(0, 1).reshape(big_h, n * w)  # [H, N·w]
    return _mm(ah.t(), z).reshape(h, n, w).transpose(0, 1).contiguous()


class _SeparableBilinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, ah: torch.Tensor, aw: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(ah, aw)
        with ieee_fp32_matmul():
            return _apply(x, ah, aw)

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, g: torch.Tensor) -> tuple[torch.Tensor | None, None, None]:
        ah, aw = ctx.saved_tensors
        if not ctx.needs_input_grad[0]:
            return None, None, None
        with ieee_fp32_matmul():
            return _apply_t(g, ah, aw), None, None


def separable_bilinear(x: torch.Tensor, hw: tuple[int, int], *, align_corners: bool, audit: bool = True) -> torch.Tensor:
    """[P, C, h, w] → [P, C, H, W]：与 F.interpolate(x, size=hw, mode="bilinear", align_corners=align_corners) 同一个插值
    （模块说明），矩阵乘是 IEEE fp32 / float64。audit=True 时本进程在这个设备上第一次调用先做一次 fp32_audit。"""
    if not isinstance(x, torch.Tensor) or x.dim() != 4:
        raise ValueError(f"上采样的输入必须是 [P, C, h, w]，收到 {getattr(x, 'shape', type(x))}")
    if x.dtype not in _FLOAT_DTYPES:
        raise ValueError(f"可分离矩阵乘的上采样只接受 {_FLOAT_DTYPES}（探针输出是 fp32），收到 {x.dtype}")
    big_h, big_w = int(hw[0]), int(hw[1])
    if audit:
        fp32_audit(x.device)
    p, c, h, w = x.shape
    ah = interp_matrix(h, big_h, bool(align_corners), x.device, x.dtype)
    aw = interp_matrix(w, big_w, bool(align_corners), x.device, x.dtype)
    return _SeparableBilinear.apply(x.reshape(p * c, h, w), ah, aw).reshape(p, c, big_h, big_w)


# ---------------------------------------------------------------------------
# 运行期审计
# ---------------------------------------------------------------------------

_AUDITS: dict[str, dict[str, Any]] = {}
AUDIT_SHAPE = ((1, 4, 16, 21), (512, 683))  # E2 semseg 一块的网格与 GT 尺寸（通道数取小）


def _rel(a: torch.Tensor, ref: torch.Tensor) -> float:
    a64, r64 = a.detach().to("cpu", torch.float64), ref.detach().to("cpu", torch.float64)
    return float((a64 - r64).abs().max() / r64.abs().max().clamp_min(1e-300))


def fp32_audit(device: Any) -> dict[str, Any]:
    """本进程在 device 上的 fp32 审计（每个设备只做一次，结果缓存）：全局 TF32 打开时，受保护的前向与反传与 float64 参照的相对差
    必须 ≤ TOL_F32_REL，否则抛 Fp32PrecisionError。另记不受保护的同一组矩阵乘在 TF32 下的相对差（tf32_effective：这台设备上
    TF32 是否真的改变了结果；CPU 上没有 TF32，为 False）。"""
    dev = torch.device(device)
    key = str(dev)
    if key in _AUDITS:
        return _AUDITS[key]
    (p, c, h, w), (big_h, big_w) = AUDIT_SHAPE
    g = torch.Generator().manual_seed(20261003)
    x64 = torch.randn(p * c, h, w, generator=g, dtype=torch.float64)
    g64 = torch.randn(p * c, big_h, big_w, generator=g, dtype=torch.float64)
    ah64 = interp_matrix(h, big_h, False, torch.device("cpu"), torch.float64)
    aw64 = interp_matrix(w, big_w, False, torch.device("cpu"), torch.float64)
    ref_y, ref_gx = _apply(x64, ah64, aw64), _apply_t(g64, ah64, aw64)
    ah = interp_matrix(h, big_h, False, dev, torch.float32)
    aw = interp_matrix(w, big_w, False, dev, torch.float32)
    x = x64.to(dev, torch.float32).requires_grad_(True)
    gy = g64.to(dev, torch.float32)
    with torch.enable_grad(), matmul_precision("tf32"):  # 故意打开全局 TF32（调用方可能在 no_grad 里，例如评测）
        state_global = precision_state()
        y = _SeparableBilinear.apply(x, ah, aw)
        y.backward(gy)
        raw_y = torch.mm(torch.mm(ah, x.detach().transpose(0, 1).reshape(h, -1)).reshape(big_h, p * c, w)
                         .transpose(0, 1).reshape(-1, w), aw.t()).reshape(p * c, big_h, big_w)
    res = {"device": key, "torch": torch.__version__, "shape": [list(AUDIT_SHAPE[0]), list(AUDIT_SHAPE[1])],
           "global_state_during_audit": state_global, "guarded_fwd_rel": _rel(y, ref_y),
           "guarded_bwd_rel": _rel(x.grad, ref_gx), "unguarded_tf32_fwd_rel": _rel(raw_y, ref_y), "tol_rel": TOL_F32_REL}
    res["tf32_effective"] = res["unguarded_tf32_fwd_rel"] > TOL_F32_REL
    res["ok"] = max(res["guarded_fwd_rel"], res["guarded_bwd_rel"]) <= TOL_F32_REL
    if not res["ok"]:
        raise Fp32PrecisionError(f"上采样的矩阵乘在 {key} 上不是 IEEE fp32：全局 TF32 打开时受保护的相对差 "
                                 f"前向 {res['guarded_fwd_rel']:.3g}、反传 {res['guarded_bwd_rel']:.3g} > {TOL_F32_REL}")
    _AUDITS[key] = res
    return res


def audits() -> dict[str, dict[str, Any]]:
    """本进程做过的审计（按设备）。"""
    return {k: dict(v) for k, v in _AUDITS.items()}


__all__ = [
    "AUDIT_SHAPE",
    "Fp32PrecisionError",
    "IMPLS",
    "MATRIX_CACHE_SIZE",
    "TOL_F32_REL",
    "TOL_F64",
    "audits",
    "fp32_audit",
    "ieee_fp32_matmul",
    "interp_matrix",
    "interp_weights",
    "matmul_precision",
    "precision_state",
    "separable_bilinear",
]
