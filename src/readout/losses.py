"""E3a 的四种损失（DEV-0034 §2），在 128 × 128 上逐样本算，定义同 E2（src/e2/tasks.py；参数取 configs/e3a_train.json 的
tasks.<任务>.loss，tests 钉住与 configs/e2_probes.json 相同，且与 E2 的实现在同一组输入上数值相同）：

- refseg：BCE（logit，逐像素平均）· bce_weight + Dice（sigmoid 概率，1 − (2Σpg + s)/(Σp + Σg + s)）· dice_weight；
- depth：SILog = sqrt(mean(d²) − λ·mean(d)²) · scale，d = 预测的 log 深度 − GT 的 log 深度，只在有效格子上；
- semseg：交叉熵，忽略 ADE_IGNORE_INDEX，按这条样本的有效像素数平均；
- normals：预测逐像素单位化后 1 − cos，在有效格子上平均。
每个函数返回 [B] 的逐样本损失（fp32）。一步的损失怎样合成见 train.py（loss.task_weights、sample_mean）。
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn.functional as F

from src.constants import ADE_IGNORE_INDEX


def refseg_loss(out: torch.Tensor, mask: torch.Tensor, cfg: Mapping[str, Any]) -> torch.Tensor:
    """out [B, 1, R, R]（logit），mask [B, R, R]（0/1）。"""
    b = out.shape[0]
    x = out[:, 0].float().reshape(b, -1)
    y = mask.float().reshape(b, -1)
    bce = F.binary_cross_entropy_with_logits(x, y, reduction="none").mean(dim=1)
    p = torch.sigmoid(x)
    s = float(cfg["dice_smooth"])
    dice = 1.0 - (2.0 * (p * y).sum(dim=1) + s) / (p.sum(dim=1) + y.sum(dim=1) + s)
    return float(cfg["bce_weight"]) * bce + float(cfg["dice_weight"]) * dice


def depth_loss(out: torch.Tensor, log_gt: torch.Tensor, valid: torch.Tensor, cfg: Mapping[str, Any]) -> torch.Tensor:
    """out [B, 1, R, R]（log 深度），log_gt [B, R, R]，valid [B, R, R] bool（每条至少一个有效格子）。"""
    b = out.shape[0]
    m = valid.reshape(b, -1).float()
    n = m.sum(dim=1)
    if bool((n <= 0).any()):
        raise ValueError("深度损失：有样本没有有效格子")
    d = (out[:, 0].float() - log_gt.float()).reshape(b, -1) * m
    v = (d * d).sum(dim=1) / n - float(cfg["variance_focus"]) * (d.sum(dim=1) / n) ** 2
    return torch.sqrt(v.clamp_min(1e-12)) * float(cfg["scale"])


def semseg_loss(logits: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    """logits [B, C, R, R]，label [B, R, R]（long，忽略值 ADE_IGNORE_INDEX）。"""
    b = logits.shape[0]
    ce = F.cross_entropy(logits.float(), label.long(), ignore_index=ADE_IGNORE_INDEX, reduction="none").reshape(b, -1)
    n = (label != ADE_IGNORE_INDEX).reshape(b, -1).sum(dim=1).float()
    if bool((n <= 0).any()):
        raise ValueError("语义分割损失：有样本全是忽略值")
    return ce.sum(dim=1) / n


def normals_loss(out: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """out [B, 3, R, R]，gt [B, 3, R, R]（单位向量），valid [B, R, R] bool。"""
    b = out.shape[0]
    u = F.normalize(out.float(), dim=1, eps=1e-12)
    cos = (u * gt.float()).sum(dim=1).reshape(b, -1)
    m = valid.reshape(b, -1).float()
    n = m.sum(dim=1)
    if bool((n <= 0).any()):
        raise ValueError("法向损失：有样本没有有效格子")
    return ((1.0 - cos) * m).sum(dim=1) / n


__all__ = ["depth_loss", "normals_loss", "refseg_loss", "semseg_loss"]
