"""E3a 的读出结构（简报 §8.4；硬约束第 3 条）：逐层加权混合、共享薄解码器、开放词汇的类别头与 1K 以内的线性适配；参数量断言与
像素通路的结构检查。

**输入**：读出模块（Readout）的 forward 只收三样（READOUT_INPUTS）：
- query_hidden [S, B, Q, H]：S 个来源的隐藏状态。full / c1 是全部 LLM 层（层号 0..num_layers()，D19）在查询位置的隐藏状态；
  c2 是 S = 1：vision_features 的图像 token 网格缩放到 16 × 16（features.py）。Q = grid²，按行主序排成 grid × grid。
- class_hidden [S, B, T, H] 与 class_pool [B, C, T]（只在语义分割时给）：类别名所在那一段 token 的隐藏状态（同样的 S 个来源）与
  每个类别名的平均矩阵（第 c 行在第 c 个类别名的 token 上取 1/长度，其余 0）。
没有别的输入：不收像素、不收视觉编码器的输出（c2 例外，它的 query_hidden 就是视觉编码器的输出，是对照）。check_no_pixel_path 在构造时
核对；tests/test_readout_model.py 另用梯度检查钉住。

**逐层加权混合**（LayerMix）：每个来源先做无参数的 LayerNorm（逐 token，fp32；不同层的隐藏状态幅度差几个数量级，不归一化时权重
反映的是幅度而不是贡献），再按 softmax(logits) 加权求和、乘标量 gamma（Peters et al. 2018 的 scalar mix）。logits 初始化为 0
（均匀），gamma 为 1。类别名与查询共用同一组权重（简报 §8.4"同一逐层混合"）。c2 只有一个来源：权重固定为 1，没有可训练参数。

**共享薄解码器**（ThinDecoder，结构写在配置的 decoder 节）：LayerNorm(H) → Linear(H → widths[0])，排成 [widths[0], 16, 16]；
16 × 16 上 res_blocks[0] 个残差块；之后每级：固定的 2× 双线性上采样（Upsample2x，无参数，与 F.interpolate(scale 2, bilinear,
align_corners=False) 相同的插值；只用切片与逐元素运算，反传没有原子加）→ 3×3 卷积到 widths[i] → GroupNorm → GELU → res_blocks[i] 个
残差块；最后 1×1 卷积到 embed_dim（64），即 128 × 128 × 64 的逐像素嵌入场。所有任务共享。

**输出头**：semseg = 像素嵌入与类别嵌入的点积（类别嵌入 = class_proj(类别名 token 的混合隐藏状态的平均)，class_proj = LayerNorm(H) +
Linear(H → embed_dim)）；refseg、depth、normals 各一个 1×1 卷积（线性适配，64 → 1 / 1 / 3：65、65、195 个参数）。

**参数量**（param_report）：mix + decoder + class_proj 合计不超过 DECODER_MAX_PARAMS（简报 §4 第 3 条的 50M；class_proj 只用于类别头，
这里保守地计入解码器的预算，见完成总结的待裁项）；每个线性头不超过 HEAD_MAX_PARAMS（1K）。构造时断言，超了抛 ParamBudgetError。
查询嵌入（256 × H）是 LLM 的输入，不计入解码器，单独报告。
"""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

# 简报 §4 第 3 条：解码器默认参数量上限、任务专用输出适配的上限（硬约束，不在配置里调）
DECODER_MAX_PARAMS = 50_000_000
HEAD_MAX_PARAMS = 1_000

TASKS: tuple[str, ...] = ("refseg", "depth", "semseg", "normals")  # 前两个与 E3b 的 GRID_TASKS 同序（data.TASK_INDEX）
LINEAR_HEAD_TASKS: tuple[str, ...] = ("refseg", "depth", "normals")
VARIANTS: tuple[str, ...] = ("full", "c1", "c2")
READOUT_INPUTS: tuple[str, ...] = ("query_hidden", "class_hidden", "class_pool")

# 各变体的读出输入来源（构造时登记、check_no_pixel_path 核对）
SOURCE_LLM_QUERY = "llm_hidden:query"
SOURCE_LLM_CLASS = "llm_hidden:class_names"
SOURCE_VISION = "vision_features"
SOURCE_EMBED_CLASS = "llm_input_embeddings:class_names"
LLM_ONLY_SOURCES = frozenset({SOURCE_LLM_QUERY, SOURCE_LLM_CLASS})
SOURCES: Mapping[str, tuple[str, ...]] = {
    "full": (SOURCE_LLM_QUERY, SOURCE_LLM_CLASS),
    "c1": (SOURCE_LLM_QUERY, SOURCE_LLM_CLASS),
    "c2": (SOURCE_VISION, SOURCE_EMBED_CLASS),
}


class PixelPathError(RuntimeError):
    """读出模块可能看到像素：输入来源、forward 的参数、子模块类型、钩子或与 LLM 共享的参数不合规（硬约束第 3 条）。"""


class ParamBudgetError(ValueError):
    """解码器超过 DECODER_MAX_PARAMS，或某个线性头超过 HEAD_MAX_PARAMS（硬约束第 3 条）。"""


# ---------------------------------------------------------------------------
# 结构的说明（由配置展开，进运行身份与检查点）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReadoutSpec:
    variant: str
    hidden_size: int
    n_sources: int
    grid: int
    widths: tuple[int, ...]
    res_blocks: tuple[int, ...]
    embed_dim: int
    out_size: int
    head_channels: Mapping[str, int] = field(default_factory=lambda: {"refseg": 1, "depth": 1, "normals": 3})
    norm_eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.variant not in VARIANTS:
            raise ValueError(f"variant 必须是 {VARIANTS} 之一，收到 {self.variant!r}")
        object.__setattr__(self, "widths", tuple(int(w) for w in self.widths))
        object.__setattr__(self, "res_blocks", tuple(int(r) for r in self.res_blocks))
        object.__setattr__(self, "head_channels", dict(sorted((str(k), int(v)) for k, v in self.head_channels.items())))
        for name in ("hidden_size", "n_sources", "grid", "embed_dim", "out_size"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise ValueError(f"{name} 必须是正整数，收到 {v!r}")
        if len(self.widths) < 1 or len(self.widths) != len(self.res_blocks) or min(self.widths) < 1 or min(self.res_blocks) < 0:
            raise ValueError(f"widths {self.widths} 与 res_blocks {self.res_blocks} 必须等长、为正 / 非负")
        if self.grid * 2 ** (len(self.widths) - 1) != self.out_size:
            raise ValueError(f"grid {self.grid} 经 {len(self.widths) - 1} 次 2× 上采样得不到 out_size {self.out_size}")
        if set(self.head_channels) != set(LINEAR_HEAD_TASKS):
            raise ValueError(f"head_channels 必须恰好给 {LINEAR_HEAD_TASKS}，收到 {sorted(self.head_channels)}")
        if self.variant == "c2" and self.n_sources != 1:
            raise ValueError("c2 只有一个来源（视觉特征）")

    def as_dict(self) -> dict[str, Any]:
        return {"variant": self.variant, "hidden_size": self.hidden_size, "n_sources": self.n_sources, "grid": self.grid,
                "widths": list(self.widths), "res_blocks": list(self.res_blocks), "embed_dim": self.embed_dim,
                "out_size": self.out_size, "head_channels": dict(self.head_channels), "norm_eps": self.norm_eps}


# ---------------------------------------------------------------------------
# 组件
# ---------------------------------------------------------------------------


def _groups(c: int) -> int:
    """GroupNorm 的组数：gcd(c, 32)（256、128、64 → 32；小模型彩排的窄通道照样整除）。"""
    return math.gcd(int(c), 32)


def _up_axis(x: torch.Tensor, dim: int) -> torch.Tensor:
    """沿 dim 做 2× 双线性（align_corners=False）：out[2i] = 0.25·x[i−1] + 0.75·x[i]，out[2i+1] = 0.75·x[i] + 0.25·x[i+1]，
    越界的邻居取边上的值（与 F.interpolate 的夹紧相同）。只有切片、拼接与逐元素运算。"""
    n = x.shape[dim]
    prev = torch.cat([x.narrow(dim, 0, 1), x.narrow(dim, 0, n - 1)], dim=dim)
    nxt = torch.cat([x.narrow(dim, 1, n - 1), x.narrow(dim, n - 1, 1)], dim=dim)
    even = 0.75 * x + 0.25 * prev
    odd = 0.75 * x + 0.25 * nxt
    shape = list(x.shape)
    shape[dim] = 2 * n
    return torch.stack([even, odd], dim=dim + 1).reshape(shape)


class Upsample2x(nn.Module):
    """固定的 2× 双线性上采样（无参数）：[B, C, h, w] → [B, C, 2h, 2w]，先 H 后 W。"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Upsample2x 要 [B, C, h, w]，收到 {tuple(x.shape)}")
        return _up_axis(_up_axis(x, 2), 3)


class ResBlock(nn.Module):
    """3×3 卷积 → GN → GELU → 3×3 卷积 → GN，加上输入，再 GELU。"""

    def __init__(self, c: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1)
        self.norm1 = nn.GroupNorm(_groups(c), c)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1)
        self.norm2 = nn.GroupNorm(_groups(c), c)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(self.conv1(x)))
        return self.act(x + self.norm2(self.conv2(h)))


class UpBlock(nn.Module):
    """2× 上采样 → 3×3 卷积（cin → cout）→ GN → GELU → n_res 个残差块。"""

    def __init__(self, cin: int, cout: int, n_res: int) -> None:
        super().__init__()
        self.up = Upsample2x()
        self.conv = nn.Conv2d(cin, cout, 3, padding=1)
        self.norm = nn.GroupNorm(_groups(cout), cout)
        self.act = nn.GELU()
        self.res = nn.Sequential(*[ResBlock(cout) for _ in range(int(n_res))])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res(self.act(self.norm(self.conv(self.up(x)))))


class LayerMix(nn.Module):
    """逐层加权混合（模块说明）。h [S, ..., H] → [..., H]（fp32）。"""

    def __init__(self, n_sources: int, *, learnable: bool, eps: float = 1e-6) -> None:
        super().__init__()
        self.n_sources = int(n_sources)
        self.learnable = bool(learnable)
        self.eps = float(eps)
        if learnable:
            self.logits = nn.Parameter(torch.zeros(self.n_sources))
            self.gamma = nn.Parameter(torch.ones(()))
        else:
            if self.n_sources != 1:
                raise ValueError("不可训练的混合只用于单一来源（c2）")
            self.register_buffer("logits", torch.zeros(1), persistent=False)
            self.register_buffer("gamma", torch.ones(()), persistent=False)

    def weights(self) -> torch.Tensor:
        return torch.softmax(self.logits.float(), dim=0)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if h.shape[0] != self.n_sources:
            raise ValueError(f"混合要 {self.n_sources} 个来源，收到 {h.shape[0]}")
        w = self.weights()
        acc = None
        for s in range(self.n_sources):
            x = h[s].float()
            x = F.layer_norm(x, (x.shape[-1],), eps=self.eps)
            t = w[s] * x
            acc = t if acc is None else acc + t
        return self.gamma.float() * acc

    def snapshot(self) -> dict[str, Any]:
        """当前的混合权重（softmax 之后）与 gamma，写进日志。"""
        with torch.no_grad():
            return {"weights": [float(v) for v in self.weights().cpu()], "gamma": float(self.gamma.float().cpu())}


class ThinDecoder(nn.Module):
    """[B, grid², H] → [B, embed_dim, out_size, out_size]（结构见模块说明）。"""

    def __init__(self, spec: ReadoutSpec) -> None:
        super().__init__()
        self.grid = spec.grid
        w, r = spec.widths, spec.res_blocks
        self.in_norm = nn.LayerNorm(spec.hidden_size)
        self.in_proj = nn.Linear(spec.hidden_size, w[0])
        self.stem = nn.Sequential(*[ResBlock(w[0]) for _ in range(r[0])])
        self.ups = nn.ModuleList([UpBlock(w[i - 1], w[i], r[i]) for i in range(1, len(w))])
        self.out = nn.Conv2d(w[-1], spec.embed_dim, 1)

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        b, n, _ = q.shape
        if n != self.grid * self.grid:
            raise ValueError(f"解码器要 {self.grid}×{self.grid} = {self.grid ** 2} 个查询，收到 {n}")
        x = self.in_proj(self.in_norm(q))  # [B, Q, C]
        x = x.transpose(1, 2).reshape(b, -1, self.grid, self.grid)
        x = self.stem(x)
        for up in self.ups:
            x = up(x)
        return self.out(x)


class ClassProj(nn.Module):
    """类别名的表示（[.., H]）→ 类别嵌入（[.., embed_dim]）：LayerNorm + 线性投影。"""

    def __init__(self, hidden_size: int, embed_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.proj = nn.Linear(hidden_size, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(x))


@dataclass
class ReadoutOutput:
    embed: torch.Tensor  # [B, E, R, R]
    class_emb: torch.Tensor | None = None  # [B, C, E]


# 读出模块里允许出现的子模块类型（check_no_pixel_path）：不允许任何别的模块（例如视觉塔、适配器的子模块）挂进来
ALLOWED_MODULE_TYPES: tuple[type, ...] = (
    nn.Conv2d, nn.Linear, nn.GroupNorm, nn.LayerNorm, nn.GELU, nn.Sequential, nn.ModuleList, nn.ModuleDict,
    Upsample2x, ResBlock, UpBlock, LayerMix, ThinDecoder, ClassProj,
)


class Readout(nn.Module):
    """E3a 的读出：混合 → 解码器 → 各头（模块说明）。构造时断言参数量并做像素通路的结构检查。"""

    def __init__(self, spec: ReadoutSpec) -> None:
        super().__init__()
        self.spec = spec
        self.sources: tuple[str, ...] = SOURCES[spec.variant]
        self.mix = LayerMix(spec.n_sources, learnable=spec.variant != "c2", eps=spec.norm_eps)
        self.decoder = ThinDecoder(spec)
        self.class_proj = ClassProj(spec.hidden_size, spec.embed_dim)
        self.heads = nn.ModuleDict({t: nn.Conv2d(spec.embed_dim, spec.head_channels[t], 1) for t in LINEAR_HEAD_TASKS})
        self.params = param_report(self)
        check_param_budget(self.params)
        check_no_pixel_path(self)

    def forward(self, query_hidden: torch.Tensor, class_hidden: torch.Tensor | None = None,
                class_pool: torch.Tensor | None = None) -> ReadoutOutput:
        embed = self.decoder(self.mix(query_hidden))
        cls = None
        if class_hidden is not None:
            if class_pool is None:
                raise ValueError("给了 class_hidden 就要给 class_pool")
            c = self.mix(class_hidden)  # [B, T, H]
            rep = torch.bmm(class_pool.to(c.dtype), c)  # [B, C, H]
            cls = self.class_proj(rep)
        elif class_pool is not None:
            raise ValueError("给了 class_pool 却没有 class_hidden")
        return ReadoutOutput(embed=embed, class_emb=cls)

    def task_output(self, task: str, embed: torch.Tensor, class_emb: torch.Tensor | None = None) -> torch.Tensor:
        """一批样本在某个任务上的输出 [B, C_t, R, R]：semseg 是点积 logits（C_t = 类别数），其余是线性头（1 / 1 / 3）。"""
        if task == "semseg":
            if class_emb is None:
                raise ValueError("semseg 要类别嵌入")
            return torch.einsum("bchw,bkc->bkhw", embed, class_emb)
        if task not in self.heads:
            raise ValueError(f"未知任务 {task!r}（有：{TASKS}）")
        return self.heads[task](embed)


# ---------------------------------------------------------------------------
# 参数量
# ---------------------------------------------------------------------------


def _numel(m: nn.Module) -> int:
    return int(sum(p.numel() for p in m.parameters()))


def param_report(readout: nn.Module) -> dict[str, Any]:
    """实际参数量（模块说明的口径）：mix、decoder、class_proj 与三者之和（计入 DECODER_MAX_PARAMS），各线性头。"""
    mix, dec, cls = _numel(readout.mix), _numel(readout.decoder), _numel(readout.class_proj)
    heads = {t: _numel(readout.heads[t]) for t in LINEAR_HEAD_TASKS}
    return {"mix": mix, "decoder": dec, "class_proj": cls, "decoder_budget_total": mix + dec + cls,
            "heads": heads, "readout_total": mix + dec + cls + sum(heads.values()),
            "limits": {"decoder_budget_total": DECODER_MAX_PARAMS, "each_head": HEAD_MAX_PARAMS},
            "basis": "decoder_budget_total = mix + decoder + class_proj（class_proj 只用于类别头，保守地计入解码器的预算）；"
                     "每个线性头单独 ≤ each_head"}


def check_param_budget(report: Mapping[str, Any]) -> None:
    if int(report["decoder_budget_total"]) > DECODER_MAX_PARAMS:
        raise ParamBudgetError(f"解码器（含混合与类别投影）{report['decoder_budget_total']} 个参数，超过上限 {DECODER_MAX_PARAMS}"
                               "（简报 §4 第 3 条）")
    bad = {t: n for t, n in report["heads"].items() if int(n) > HEAD_MAX_PARAMS}
    if bad:
        raise ParamBudgetError(f"线性头 {bad} 超过每个 {HEAD_MAX_PARAMS} 个参数的上限（简报 §4 第 3 条）")


def count_params(spec: ReadoutSpec) -> dict[str, Any]:
    """不分配内存地数参数（在 meta 设备上构造）：真模型的 hidden（1024–4096）上报告与断言用。"""
    with torch.device("meta"):
        r = Readout(spec)
    return r.params


# ---------------------------------------------------------------------------
# 像素通路的结构检查（硬约束第 3 条）
# ---------------------------------------------------------------------------

_MODULE_INTERNALS = frozenset({
    "training", "_parameters", "_buffers", "_non_persistent_buffers_set", "_backward_pre_hooks", "_backward_hooks",
    "_is_full_backward_hook", "_forward_hooks", "_forward_hooks_with_kwargs", "_forward_hooks_always_called",
    "_forward_pre_hooks", "_forward_pre_hooks_with_kwargs", "_state_dict_hooks", "_state_dict_pre_hooks",
    "_load_state_dict_pre_hooks", "_load_state_dict_post_hooks", "_modules",
})
_ADAPTER_MARKERS = ("vision_features", "build_inputs", "hidden_states", "training_forward", "pixel_values")


def check_no_pixel_path(readout: nn.Module, *, adapter_model: Any = None) -> dict[str, Any]:
    """读出模块的输入来源检查（构造时调用；训练开始时再带上适配器的模型调一次）。不合规抛 PixelPathError。

    1. 登记的输入来源：full / c1 只能是 LLM 的隐藏状态（LLM_ONLY_SOURCES）；c2 是声明过的对照（视觉特征）。
    2. forward 的参数恰好是 READOUT_INPUTS（没有像素、图像或适配器参数）。
    3. 子模块只能是 ALLOWED_MODULE_TYPES 里的类型；没有前向 / 反向钩子（钩子能把别的张量混进来）。
    4. 子模块的属性里没有未登记的模块、适配器样的对象（有 vision_features / build_inputs 等方法）或张量引用。
    5. 给了 adapter_model 时：读出模块的参数与它不共享（同一个对象或同一块存储），读出模块的子模块也不在它里面。
    返回检查摘要（写进运行记录）。"""
    variant = getattr(getattr(readout, "spec", None), "variant", None)
    sources = tuple(getattr(readout, "sources", ()))
    if variant in ("full", "c1"):
        if not sources or set(sources) - LLM_ONLY_SOURCES:
            raise PixelPathError(f"{variant} 的读出输入来源只能是 LLM 的隐藏状态 {sorted(LLM_ONLY_SOURCES)}，登记的是 {sources}")
    elif variant == "c2":
        if sources != SOURCES["c2"]:
            raise PixelPathError(f"c2 的输入来源应为 {SOURCES['c2']}，登记的是 {sources}")
    else:
        raise PixelPathError(f"未知变体 {variant!r}")
    params = [p for p in inspect.signature(type(readout).forward).parameters if p != "self"]
    if tuple(params) != READOUT_INPUTS:
        raise PixelPathError(f"读出模块 forward 的参数必须恰好是 {READOUT_INPUTS}，实际 {tuple(params)}")
    n_mod = 0
    for name, m in readout.named_modules():
        n_mod += 1
        if m is not readout and not isinstance(m, ALLOWED_MODULE_TYPES):
            raise PixelPathError(f"读出模块里有不允许的子模块 {name or '<root>'}：{type(m).__name__}")
        hooks = [h for h in ("_forward_hooks", "_forward_pre_hooks", "_backward_hooks", "_backward_pre_hooks")
                 if getattr(m, h, None)]
        if hooks:
            raise PixelPathError(f"读出模块的子模块 {name or '<root>'} 挂了钩子 {hooks}")
        for k, v in vars(m).items():
            if k in _MODULE_INTERNALS:
                continue
            if isinstance(v, nn.Module) or torch.is_tensor(v):
                raise PixelPathError(f"子模块 {name or '<root>'} 的属性 {k} 引用了未登记的模块或张量（{type(v).__name__}）")
            if any(hasattr(v, a) for a in _ADAPTER_MARKERS) and not isinstance(v, type):
                raise PixelPathError(f"子模块 {name or '<root>'} 的属性 {k} 引用了适配器样的对象（{type(v).__name__}）")
    shared = 0
    if adapter_model is not None:
        mine = {id(p) for p in readout.parameters()}
        ptrs = {p.data_ptr() for p in readout.parameters() if p.device.type != "meta" and p.numel()}
        for n, p in adapter_model.named_parameters():
            if id(p) in mine or (p.device.type != "meta" and p.numel() and p.data_ptr() in ptrs):
                raise PixelPathError(f"读出模块的参数与适配器模型的 {n} 共享")
        theirs = {id(m) for m in adapter_model.modules()}
        for name, m in readout.named_modules():
            if id(m) in theirs:
                raise PixelPathError(f"读出模块的子模块 {name or '<root>'} 是适配器模型里的模块")
        shared = 0
    return {"variant": variant, "sources": list(sources), "forward_inputs": list(params), "n_modules": n_mod,
            "checked_against_adapter_model": adapter_model is not None, "n_shared_with_adapter": shared}


__all__ = [
    "ALLOWED_MODULE_TYPES",
    "DECODER_MAX_PARAMS",
    "HEAD_MAX_PARAMS",
    "LINEAR_HEAD_TASKS",
    "LLM_ONLY_SOURCES",
    "LayerMix",
    "ParamBudgetError",
    "PixelPathError",
    "READOUT_INPUTS",
    "Readout",
    "ReadoutOutput",
    "ReadoutSpec",
    "SOURCES",
    "TASKS",
    "ThinDecoder",
    "Upsample2x",
    "VARIANTS",
    "check_no_pixel_path",
    "check_param_budget",
    "count_params",
    "param_report",
]
