"""E2 的探针（DEV-0023；简报 §8.3）：每个（特征来源, 种类）一个探针、一个优化器、一个学习率调度器。

- linear：1×1 卷积，D → C（带偏置）。
- conv：3×3 卷积 D → hidden（256）、GELU、3×3 卷积 → C；padding = 1，输出与输入同一个 token 网格。
- 初始化：PyTorch 的默认初始化，每个探针在自己的种子下构造（seed = sha256(基础种子|任务|特征键|种类) 的前 8 字节），所以加减层
  不改变其他探针的初值；在 CPU 上构造再搬到设备上，与设备无关。
- 优化：AdamW（configs/e2_probes.json 的 optim，lr 按种类），学习率 = lr × λ(s)：前 warmup_steps 步线性升到 lr
  （λ(s) = (s+1)/warmup），之后按 (s − warmup)/(steps − warmup) 余弦降向 min_lr_ratio·lr（最后一次更新仍略高于下限）。
  全部模型、层共用一份。
- 全部探针参数是 fp32；输入是标准化后的 fp32 特征。
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from src.e2.config import PROBE_KINDS, layer_label

FeatureKey = int | str


class LinearProbe(nn.Module):
    def __init__(self, in_dim: int, out_ch: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_dim, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class ConvProbe(nn.Module):
    def __init__(self, in_dim: int, out_ch: int, *, hidden: int, kernel: int, padding: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_dim, hidden, kernel_size=kernel, padding=padding)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(hidden, out_ch, kernel_size=kernel, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.act(self.conv1(x)))


def probe_seed(base_seed: int, task: str, key: FeatureKey, kind: str) -> int:
    h = hashlib.sha256(f"{int(base_seed)}|{task}|{layer_label(key)}|{kind}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") & ((1 << 63) - 1)


def make_probe(kind: str, in_dim: int, out_ch: int, spec: Mapping[str, Any], *, seed: int) -> nn.Module:
    """按种类构造探针（CPU，fp32），初值由 seed 决定，不动全局随机状态。"""
    if kind not in PROBE_KINDS:
        raise ValueError(f"未知探针种类 {kind!r}，可用：{PROBE_KINDS}")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        if kind == "linear":
            return LinearProbe(int(in_dim), int(out_ch))
        return ConvProbe(int(in_dim), int(out_ch), hidden=int(spec["hidden"]), kernel=int(spec["kernel"]),
                         padding=int(spec["padding"]))


def lr_factor(step: int, *, steps: int, warmup_steps: int, min_lr_ratio: float) -> float:
    """第 step 次更新（0 起）的学习率倍数 λ(step)。"""
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    span = max(1, steps - warmup_steps)
    progress = min(1.0, (step - warmup_steps) / span)
    return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def n_params(m: nn.Module) -> int:
    return int(sum(p.numel() for p in m.parameters()))


@dataclass
class ProbeEntry:
    name: str
    key: FeatureKey
    kind: str
    module: nn.Module
    opt: torch.optim.Optimizer
    sched: torch.optim.lr_scheduler.LambdaLR


def probe_name(key: FeatureKey, kind: str) -> str:
    return f"{layer_label(key)}/{kind}"


class ProbeBank:
    """一个任务的全部探针。顺序：特征键（参考线在前，层号升序）× 种类（linear、conv）。"""

    def __init__(self, task_cfg: Mapping[str, Any], dims: Mapping[FeatureKey, int], *, device: Any, base_seed: int) -> None:
        self.task = task_cfg["task"]
        self.out_ch = int(task_cfg["out_channels"])
        optim = task_cfg["optim"]
        if optim["optimizer"] != "adamw" or optim["schedule"] != "warmup_cosine":
            raise ValueError(f"只实现了 adamw + warmup_cosine，配置是 {optim['optimizer']} + {optim['schedule']}")
        self.entries: dict[str, ProbeEntry] = {}
        steps = int(task_cfg["steps"])
        for key, dim in dims.items():
            for kind in PROBE_KINDS:
                name = probe_name(key, kind)
                m = make_probe(kind, dim, self.out_ch, task_cfg["probes"][kind],
                               seed=probe_seed(base_seed, self.task, key, kind)).to(device)
                opt = torch.optim.AdamW(m.parameters(), lr=float(optim["lr"][kind]), betas=tuple(optim["betas"]),
                                        eps=float(optim["eps"]), weight_decay=float(optim["weight_decay"]))
                w, mr = int(optim["warmup_steps"]), float(optim["min_lr_ratio"])
                sched = torch.optim.lr_scheduler.LambdaLR(
                    opt, lambda s, _w=w, _mr=mr: lr_factor(s, steps=steps, warmup_steps=_w, min_lr_ratio=_mr))
                self.entries[name] = ProbeEntry(name, key, kind, m, opt, sched)

    @property
    def names(self) -> list[str]:
        return list(self.entries)

    def chunks(self, grids: Mapping[FeatureKey, tuple[int, int]], size: int) -> list[list[str]]:
        """按输出网格形状分组（同组才能拼成一批上采样），组内按 bank 顺序每 size 个一块。"""
        groups: dict[tuple[int, int], list[str]] = {}
        for name, e in self.entries.items():
            groups.setdefault(tuple(grids[e.key]), []).append(name)
        out = []
        for names in groups.values():
            out += [names[i:i + size] for i in range(0, len(names), size)]
        return out

    def train(self) -> None:
        for e in self.entries.values():
            e.module.train()

    def eval(self) -> None:
        for e in self.entries.values():
            e.module.eval()

    def zero_grad(self) -> None:
        for e in self.entries.values():
            e.opt.zero_grad(set_to_none=True)

    def step(self) -> None:
        for e in self.entries.values():
            e.opt.step()
            e.sched.step()

    def lrs(self) -> dict[str, float]:
        return {n: float(e.opt.param_groups[0]["lr"]) for n, e in self.entries.items()}

    def param_counts(self) -> dict[str, int]:
        return {n: n_params(e.module) for n, e in self.entries.items()}

    def state_dict(self) -> dict[str, Any]:
        return {n: {"module": e.module.state_dict(), "opt": e.opt.state_dict(), "sched": e.sched.state_dict()}
                for n, e in self.entries.items()}

    def module_state(self) -> dict[str, Any]:
        return {n: e.module.state_dict() for n, e in self.entries.items()}

    def load_state_dict(self, st: Mapping[str, Any]) -> None:
        if set(st) != set(self.entries):
            raise ValueError(f"检查点的探针 {sorted(st)} 与本次的 {sorted(self.entries)} 不一致")
        for n, e in self.entries.items():
            e.module.load_state_dict(st[n]["module"])
            e.opt.load_state_dict(st[n]["opt"])
            e.sched.load_state_dict(st[n]["sched"])

    def predict(self, names: Sequence[str], feats: Mapping[FeatureKey, torch.Tensor]) -> torch.Tensor:
        """一块探针在一个样本上的输出：[P, C, h, w]（feats[key] 是 [D, h, w]）。"""
        return torch.cat([self.entries[n].module(feats[self.entries[n].key][None]) for n in names], dim=0)


__all__ = [
    "ConvProbe",
    "LinearProbe",
    "ProbeBank",
    "ProbeEntry",
    "lr_factor",
    "make_probe",
    "n_params",
    "probe_name",
    "probe_seed",
]
