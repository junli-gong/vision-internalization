"""E2 的吞吐与算力估算（--estimate；DEV-0023 任务书 §1"吞吐与算力估算"），用于 E2 预注册。

每个任务在真实的训练与评测样本上实测（与正式运行同一条代码路径：TaskRun.train_step / TaskRun.collect）：
- 标准化前向：在 2 条标准化样本上计时（只用来构造探针，统计量不代表正式运行）；
- 训练步：先跑 1 步预热（不计），再计时 configs/e2_probes.json 的 estimate.train_steps 步（每步 = batch_size 个样本的前向 +
  全部探针的前向、上采样、损失与反传 + 优化器）；
- 评测：计时 estimate.eval_samples 条评测样本（前向 + 全部探针的上采样与逐样本指标）。
外推（秒）= n_norm × t_前向 + steps × t_步 + (曲线点数 × n_val + n_val + 评测集条数) × t_评测一条。
bootstrap（CPU）不计入。墙钟在 CUDA 上前后同步；显存峰值取 torch.cuda.max_memory_allocated（CPU 上为 null）。
实测用的训练与评测样本取各自列表的开头，图像尺寸未必代表全集（ADE 的图大小不一），外推只是估计。
"""

from __future__ import annotations

import time
from typing import Any, Callable, Mapping

import torch

from src.e2.config import E2Config, curve_steps
from src.e2.data import E2Paths
from src.e2.runner import RunSpec, TaskRun

N_STATS_SAMPLES = 2


def _sync(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def estimate_task(adapter: Any, cfg: E2Config, spec: RunSpec, task_cfg: Mapping[str, Any], paths: E2Paths, *,
                  train_steps: int, eval_samples: int, log: Callable[[str], None]) -> dict[str, Any]:
    run = TaskRun(adapter, cfg, spec, task_cfg, paths, log=log)
    run.load_data()
    dev = torch.device(adapter.device)
    train, ev = run.split.train, run.eval_set.records
    _sync(dev)
    t0 = time.perf_counter()
    stats = run.compute_stats(run.split.stats[:N_STATS_SAMPLES])
    _sync(dev)
    n_st = min(N_STATS_SAMPLES, len(run.split.stats))
    t_fwd = (time.perf_counter() - t0) / max(1, n_st)
    run.set_stats(stats)
    run.train_step(0, train)  # 预热
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    _sync(dev)
    t0 = time.perf_counter()
    for k in range(train_steps):
        run.train_step(1 + k, train)
    _sync(dev)
    t_step = (time.perf_counter() - t0) / max(1, train_steps)
    sub = ev[:eval_samples]
    _sync(dev)
    t0 = time.perf_counter()
    run.collect(sub, "estimate")
    _sync(dev)
    t_eval = (time.perf_counter() - t0) / max(1, len(sub))
    steps, n_val = int(task_cfg["steps"]), len(run.split.val)
    n_curve = len(curve_steps(steps, int(task_cfg["eval_every"])))
    n_norm = int(task_cfg["feature_norm"]["n_samples"])
    parts = {"norm_s": n_norm * t_fwd, "train_s": steps * t_step, "val_curves_s": (n_curve + 1) * n_val * t_eval,
             "eval_s": len(ev) * t_eval}
    total = sum(parts.values())
    return {"task": spec.task, "n_probes": len(run.bank.names), "n_feature_keys": len(run.extractor.keys),
            "batch_size": int(task_cfg["batch_size"]), "steps": steps, "n_curve_points": n_curve, "n_val": n_val,
            "n_eval": len(ev), "n_train": len(train), "n_norm": n_norm,
            "measured": {"forward_s": t_fwd, "train_step_s": t_step, "eval_sample_s": t_eval, "train_steps": train_steps,
                         "eval_samples": len(sub)},
            "seconds": parts, "gpu_hours": total / 3600.0,
            "forwards": {"norm": n_norm, "train": steps * int(task_cfg["batch_size"]), "val": (n_curve + 1) * n_val,
                         "eval": len(ev)},
            "peak_mem_bytes": int(torch.cuda.max_memory_allocated(dev)) if dev.type == "cuda" else None}


def summary_markdown(model_key: str, rows: list[dict[str, Any]], header: Mapping[str, Any]) -> str:
    lines = [f"# E2 估算 · {model_key}", "", f"- 设备：{header.get('gpu') or header.get('device')}；dtype {header.get('dtype')}；"
             f"层 {header.get('layer_spec')}（{header.get('n_layers')} 个）；参考线 {header.get('references') or '无'}", "",
             "| 任务 | 探针数 | 步 × 批 | 前向 s | 一步 s | 评测一条 s | 评测集 | GPU 小时 | 显存峰值 GB |",
             "|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        m = r["measured"]
        pk = r["peak_mem_bytes"]
        lines.append(f"| {r['task']} | {r['n_probes']} | {r['steps']} × {r['batch_size']} | {m['forward_s']:.3f} | "
                     f"{m['train_step_s']:.3f} | {m['eval_sample_s']:.3f} | {r['n_eval']} | {r['gpu_hours']:.2f} | "
                     f"{'—' if pk is None else f'{pk / 1e9:.1f}'} |")
    total = sum(r["gpu_hours"] for r in rows)
    lines += ["", f"合计 {total:.2f} GPU 小时（不含 bootstrap 的 CPU 时间与模型加载）。", ""]
    return "\n".join(lines)


__all__ = ["estimate_task", "summary_markdown"]
