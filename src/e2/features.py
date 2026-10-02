"""E2 的特征：冻结模型一次前向 → 各特征来源（LLM 层与参考线）在图像位置的 token 网格，再做固定的按通道标准化（DEV-0023）。

- 一次前向：adapter.hidden_states(inputs, layers)（torch.no_grad；模型本身是 bf16），每层取图像位置的 [n, D]，按
  src/e2/grid.py 排成 [h, w, D]。同一次前向的结果喂给这个样本上的全部探针；特征不落盘（简报 §4 第 8 条）。
- L{N}_pre（DEV-0024，D53-3 的 I1；pre_norm_last=True 时）：最后一个 decoder 层的原始输出（最终归一化之前），同一次前向里向适配器
  多要一个层键 base.PRE_NORM_LAST；特征键是 config.pre_norm_key(N) 这个字符串，排在整数层之后，探针与标准化统计量各自一组。
- 参考线：
  - post_proj：就是第 0 层（D21-3）。不另算一组特征；每次进程第一次前向时核对 adapter.vision_features(inputs) 与第 0 层逐位相同，
    不相同就报错（与 D21-3 矛盾，G2 的解读要先查清）。
  - pre_proj：视觉塔在投影或合并之前的输出，要求适配器有 pre_projection_features(inputs) -> (张量 [n, D_v], (h, w))，行主序。
    现有三个适配器都没有这个方法，给了 --reference pre_proj 直接报错（接口建议见 DEV-0023 完成总结）。维度与网格都可以与 LLM
    层不同，探针按各自的输入维度构造。
- 标准化（configs/e2_probes.json 的 feature_norm）：每个特征来源一组逐通道统计量，训练前在训练集里按哈希排序取的 n_samples 条上
  一次算好（float64，Chan 的并行合并；不用 E[x²] − E[x]²，避免大激活通道上的相消误差），之后冻结：x̂ = (x − μ) / sqrt(σ² + eps)。
- 入口断言（dev.md：开发机上不可能出差异的属性要在代码里查）：隐藏状态必须在适配器的 device 上、是浮点、形状 [n, D] 与网格一致。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import torch
from PIL import Image

from src.adapters.base import PRE_NORM_LAST
from src.data.records import Sample
from src.e2.config import PRE_PROJ, E2Config, pre_norm_key
from src.e2.grid import grid_to_chw, single_image, tokens_to_grid

FeatureKey = int | str


class FeatureCheckError(RuntimeError):
    """特征的入口检查失败（设备、形状、post_proj ≠ 第 0 层……）。"""


def load_image(sample: Sample) -> Image.Image:
    if len(sample.images) != 1:
        raise ValueError(f"{sample.id}：E2 的样本只有一张图，收到 {len(sample.images)} 张")
    with Image.open(sample.images[0]) as im:
        return im.convert("RGB")


# ---------------------------------------------------------------------------
# 逐通道统计量
# ---------------------------------------------------------------------------


class ChannelStats:
    """逐通道均值与方差（总体方差），float64，按批用 Chan et al. 的并行公式合并（顺序固定时逐位可复现）。"""

    def __init__(self) -> None:
        self.n = 0
        self.mean: torch.Tensor | None = None
        self.m2: torch.Tensor | None = None

    def update(self, x: torch.Tensor) -> None:
        """x：[n_tok, D]（任意浮点 dtype、任意设备），累计到 CPU float64。"""
        if x.dim() != 2 or x.shape[0] == 0:
            raise ValueError(f"ChannelStats.update 要 [n>0, D]，收到 {tuple(x.shape)}")
        xb = x.detach().to(device="cpu", dtype=torch.float64)
        nb = xb.shape[0]
        mb = xb.mean(dim=0)
        m2b = ((xb - mb) ** 2).sum(dim=0)
        if self.mean is None:
            self.n, self.mean, self.m2 = nb, mb, m2b
            return
        if self.mean.shape != mb.shape:
            raise ValueError(f"通道数变了：{tuple(self.mean.shape)} → {tuple(mb.shape)}")
        n = self.n + nb
        delta = mb - self.mean
        self.mean = self.mean + delta * (nb / n)
        self.m2 = self.m2 + m2b + delta * delta * (self.n * nb / n)
        self.n = n

    def var(self) -> torch.Tensor:
        if self.mean is None or self.n == 0:
            raise ValueError("ChannelStats 还没有数据")
        return self.m2 / self.n

    def state(self) -> dict[str, Any]:
        return {"n": self.n, "mean": self.mean, "m2": self.m2}

    @classmethod
    def from_state(cls, st: Mapping[str, Any]) -> "ChannelStats":
        s = cls()
        s.n, s.mean, s.m2 = int(st["n"]), st["mean"], st["m2"]
        return s


@dataclass
class Normalizer:
    """冻结的标准化：每个特征来源一组 (μ, 1/sqrt(σ² + eps))，fp32，放在探针的设备上。"""

    shift: dict[FeatureKey, torch.Tensor]
    scale: dict[FeatureKey, torch.Tensor]

    @classmethod
    def from_stats(cls, stats: Mapping[FeatureKey, ChannelStats], *, eps: float, device: Any) -> "Normalizer":
        shift, scale = {}, {}
        for k, s in stats.items():
            shift[k] = s.mean.to(torch.float32).to(device)
            scale[k] = torch.rsqrt(s.var() + float(eps)).to(torch.float32).to(device)
        return cls(shift, scale)

    def apply(self, key: FeatureKey, grid: torch.Tensor) -> torch.Tensor:
        """[h, w, D]（任意浮点）→ 标准化后的 [D, h, w] fp32。"""
        g = grid.to(torch.float32)
        if g.shape[-1] != self.shift[key].shape[0]:
            raise FeatureCheckError(f"{key}：特征维度 {g.shape[-1]} 与统计量的 {self.shift[key].shape[0]} 不一致")
        return grid_to_chw((g - self.shift[key]) * self.scale[key])


# ---------------------------------------------------------------------------
# 前向
# ---------------------------------------------------------------------------


@dataclass
class ForwardCounters:
    """记账：前向次数、前向耗时（CUDA 上前后同步）、视觉 token 数与像素数。按阶段（stats/train/val/eval/estimate）分开记。"""

    by_phase: dict[str, dict[str, float]] = field(default_factory=dict)

    def add(self, phase: str, *, n_forward: int = 0, n_vision_extra: int = 0, forward_s: float = 0.0,
            visual_tokens: int = 0, image_pixels: int = 0) -> None:
        d = self.by_phase.setdefault(phase, {"n_forward": 0, "n_vision_extra": 0, "forward_s": 0.0,
                                             "visual_tokens": 0, "image_pixels": 0})
        d["n_forward"] += n_forward
        d["n_vision_extra"] += n_vision_extra
        d["forward_s"] += forward_s
        d["visual_tokens"] += visual_tokens
        d["image_pixels"] += image_pixels


def _sync(device: Any) -> None:
    dev = torch.device(device)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


class FeatureExtractor:
    """一个任务的特征提取：样本 → ModelInputs（按任务的顺序与指令）→ 一次前向 → {特征键: [h, w, D]}。"""

    def __init__(self, adapter: Any, cfg: E2Config, task: str, layers: Sequence[int], references: Sequence[str],
                 counters: ForwardCounters | None = None, *, pre_norm_last: bool = False) -> None:
        self.adapter = adapter
        self.cfg = cfg
        self.task = task
        self.order = cfg.order(task)
        self.layers = list(layers)
        self.references = list(references)
        if "post_proj" in self.references and 0 not in self.layers:
            raise ValueError("post_proj 就是第 0 层：层集合里必须有 0（runner 会自动补上）")
        self.use_pre_proj = PRE_PROJ in self.references
        if self.use_pre_proj and not callable(getattr(adapter, "pre_projection_features", None)):
            raise NotImplementedError(
                f"{type(adapter).__name__} 没有 pre_projection_features(inputs)：pre_proj 参考线（视觉塔在投影 / 合并之前的输出）"
                "需要适配器提供这个接口（DEV-0023 完成总结），本任务不改适配器")
        # L{N}_pre：最后一个 decoder 层的原始输出（最终归一化之前），向适配器要层键 PRE_NORM_LAST
        self.pre_norm: str | None = pre_norm_key(int(adapter.num_layers())) if pre_norm_last else None
        self.keys: list[FeatureKey] = (([PRE_PROJ] if self.use_pre_proj else []) + self.layers
                                       + ([self.pre_norm] if self.pre_norm else []))
        self.counters = counters or ForwardCounters()
        # post_proj 核对：进程里第一次前向时做一次（尽早失败）；调用方可以再要求一次（runner 在评测集第一条上再核一次，
        # 这一条的记录与是否续跑无关，写进 metrics.json）
        self.post_proj_checks: list[dict[str, Any]] = []
        self._pp_pending = "post_proj" in self.references
        self.audit: dict[str, Any] | None = None  # 第一次前向时记下（进运行记录）：隐藏状态的 dtype / 设备、模型是否 eval

    def _check_frozen(self) -> int | None:
        """模型必须是 eval 模式（每次前向都查）；第一次前向时另数一下 requires_grad 的参数个数（只记录：前向在 no_grad 下，
        探针的优化器里只有探针参数，模型参数的 requires_grad 不影响冻结）。"""
        model = getattr(self.adapter, "model", None)
        if bool(getattr(model, "training", False)):
            raise FeatureCheckError("模型处于 train 模式：E2 要求模型冻结（eval、no_grad）")
        if self.audit is not None or model is None or not hasattr(model, "parameters"):
            return None
        return int(sum(p.numel() for p in model.parameters() if p.requires_grad))

    def request_post_proj_check(self) -> None:
        """下一次前向时再核对一次 post_proj（vision_features）与第 0 层逐位相同。"""
        self._pp_pending = "post_proj" in self.references

    def post_proj_check(self, phase: str) -> dict[str, Any] | None:
        """phase 阶段做过的最后一次核对（没有就是 None）。"""
        hits = [c for c in self.post_proj_checks if c["phase"] == phase]
        return hits[-1] if hits else None

    def instruction(self, sample: Sample) -> str:
        return self.cfg.instruction(self.task, sample.fields["expression"] if self.task == "refseg" else None)

    def build(self, sample: Sample) -> Any:
        return self.adapter.build_inputs([load_image(sample)], self.instruction(sample), self.order)

    def raw(self, sample: Sample, *, phase: str) -> tuple[dict[FeatureKey, torch.Tensor], Any]:
        """一次前向，返回 ({特征键: [h, w, D]（模型 dtype，在模型设备上）}, ModelInputs)。"""
        inputs = self.build(sample)
        info = single_image(inputs)
        dev = self.adapter.device
        n_req_grad = self._check_frozen()
        _sync(dev)
        t0 = time.perf_counter()
        with torch.no_grad():
            hs = self.adapter.hidden_states(inputs, self.layers + ([PRE_NORM_LAST] if self.pre_norm else []))
            pre = self.adapter.pre_projection_features(inputs) if self.use_pre_proj else None
        _sync(dev)
        dt = time.perf_counter() - t0
        out: dict[FeatureKey, torch.Tensor] = {}
        for l in self.layers:
            x = hs.image[l]
            self._check_tensor(x, f"第 {l} 层", n=len(info.span), dim=self.adapter.hidden_size())
            out[l] = tokens_to_grid(x, info.grid_hw)
        if self.pre_norm:
            x = hs.image[PRE_NORM_LAST]
            self._check_tensor(x, self.pre_norm, n=len(info.span), dim=self.adapter.hidden_size())
            out[self.pre_norm] = tokens_to_grid(x, info.grid_hw)
        if pre is not None:
            x, hw = pre
            hw = (int(hw[0]), int(hw[1]))
            self._check_tensor(x, PRE_PROJ, n=hw[0] * hw[1], dim=None)
            out[PRE_PROJ] = tokens_to_grid(x, hw)
        self.counters.add(phase, n_forward=1, n_vision_extra=int(pre is not None), forward_s=dt,
                          visual_tokens=len(info.span), image_pixels=int(info.n_pixels or 0))
        if self.audit is None:
            x0 = hs.image[self.layers[0]]
            self.audit = {"hidden_dtype": str(x0.dtype), "hidden_device": str(x0.device), "adapter_device": str(dev),
                          "hidden_requires_grad": bool(x0.requires_grad), "model_params_requires_grad": n_req_grad,
                          "grid_hw": list(info.grid_hw), "n_image_pixels": info.n_pixels, "sample": sample.id}
        if hs.image[self.layers[0]].requires_grad:
            raise FeatureCheckError("隐藏状态带梯度：前向没有在 no_grad 下进行")
        if self._pp_pending:
            self._pp_pending = False
            self._check_post_proj(inputs, hs.image[0], sample.id, phase)
        return out, inputs

    def _check_tensor(self, x: Any, what: str, *, n: int, dim: int | None) -> None:
        if not isinstance(x, torch.Tensor) or not x.is_floating_point():
            raise FeatureCheckError(f"{what}：隐藏状态不是浮点张量（{type(x).__name__}）")
        want = torch.device(self.adapter.device)
        if x.device.type != want.type:
            raise FeatureCheckError(f"{what}：隐藏状态在 {x.device}，适配器在 {want}")
        if x.dim() != 2 or x.shape[0] != n or (dim is not None and x.shape[1] != dim):
            raise FeatureCheckError(f"{what}：形状 {tuple(x.shape)}，期望 [{n}, {dim if dim is not None else 'D'}]")

    def _check_post_proj(self, inputs: Any, layer0: torch.Tensor, sample_id: str, phase: str) -> None:
        dev = self.adapter.device
        _sync(dev)
        t0 = time.perf_counter()
        with torch.no_grad():
            vf = self.adapter.vision_features(inputs)
        _sync(dev)
        self.counters.add(phase, n_vision_extra=1, forward_s=time.perf_counter() - t0)
        equal = bool(vf.shape == layer0.shape and torch.equal(vf, layer0))
        diff = float((vf.float() - layer0.float()).abs().max()) if vf.shape == layer0.shape else None
        self.post_proj_checks.append({"phase": phase, "sample": sample_id, "equal": equal, "max_abs_diff": diff,
                                      "note": "post_proj = vision_features；D21-3：与第 0 层在图像位置上逐位相同"})
        if not equal:
            raise FeatureCheckError(f"post_proj（vision_features）与第 0 层不逐位相同（最大差 {diff}），与 D21-3 矛盾；先查清再跑")


def compute_stats(extractor: FeatureExtractor, samples: Sequence[Sample], *,
                  log: Callable[[str], None] | None = None) -> dict[FeatureKey, ChannelStats]:
    """标准化统计量：samples 逐条前向，按特征来源累计（顺序即 samples 的顺序）。"""
    stats = {k: ChannelStats() for k in extractor.keys}
    for i, s in enumerate(samples):
        raw, _ = extractor.raw(s, phase="stats")
        for k, g in raw.items():
            stats[k].update(g.reshape(-1, g.shape[-1]))
        if log and (i + 1 == len(samples) or (i + 1) % 64 == 0):
            log(f"[stats] {i + 1}/{len(samples)}")
    return stats


__all__ = [
    "ChannelStats",
    "FeatureCheckError",
    "FeatureExtractor",
    "FeatureKey",
    "ForwardCounters",
    "Normalizer",
    "compute_stats",
    "load_image",
]
