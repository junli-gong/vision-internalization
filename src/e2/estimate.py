"""E2 的吞吐与算力估算（--estimate；DEV-0023 任务书 §1"吞吐与算力估算"），用于 E2 预注册。

每个任务在真实的训练与评测样本上实测（与正式运行同一条代码路径：TaskRun.train_step / TaskRun.collect）：
- 标准化前向：在 2 条标准化样本上计时（只用来构造探针，统计量不代表正式运行）；
- 训练步：先跑 1 步预热（不计），再计时 configs/e2_probes.json 的 estimate.train_steps 步（每步 = batch_size 个样本的前向 +
  全部探针的前向、上采样、损失与反传 + 优化器）；
- 评测：计时 estimate.eval_samples 条评测样本（前向 + 全部探针的上采样与逐样本指标）。
外推（秒）= n_norm × t_前向 + steps × t_步 + (曲线点数 × n_val + n_val + 评测集条数) × t_评测一条。
bootstrap（CPU）不计入。墙钟在 CUDA 上前后同步；显存峰值取 torch.cuda.max_memory_allocated（CPU 上为 null）。
实测用的训练与评测样本取各自列表的开头，图像尺寸未必代表全集（ADE 的图大小不一），外推只是估计。

覆盖（DEV-0031）：--task all 含法向（D59-6 许可已确认，D63-4 撤掉许可门）。给了 --reference pre_proj 时，pre_proj 那组探针与
正式运行一样在探针库里，所以已在实测里：前向开销（每次前向另调一次视觉塔的 pre_projection_features，计时在 t_前向 / t_步 /
t_评测一条 之内；forwards.vision_extra 是外推的额外视觉塔前向次数）、探针数（n_probes 含 n_probes_pre_proj）、步时间（这组探针的
上采样、损失与反传）。每行记下上采样实现（upsample_impl，configs/e2_probes.json 的 upsample.impl）；表头写明层、参考线、是否含法向。
"""

from __future__ import annotations

import time
from typing import Any, Callable, Mapping

import torch

from src.e2.config import PRE_PROJ, E2Config, curve_steps, layer_label
from src.e2.data import E2Paths
from src.e2.runner import RunSpec, TaskRun
from src.e2.tasks import PositionAccumulator

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
    run.collect(sub, "estimate", positions=PositionAccumulator(run.task, run.bank.names))  # 与正式评测同一条路径（含按位置分开）
    _sync(dev)
    t_eval = (time.perf_counter() - t0) / max(1, len(sub))
    steps, n_val = int(task_cfg["steps"]), len(run.split.val)
    n_curve = len(curve_steps(steps, int(task_cfg["eval_every"])))
    n_norm = int(task_cfg["feature_norm"]["n_samples"])
    parts = {"norm_s": n_norm * t_fwd, "train_s": steps * t_step, "val_curves_s": (n_curve + 1) * n_val * t_eval,
             "eval_s": len(ev) * t_eval}
    total = sum(parts.values())
    forwards = {"norm": n_norm, "train": steps * int(task_cfg["batch_size"]), "val": (n_curve + 1) * n_val, "eval": len(ev)}
    tr = run.counters.by_phase.get("train", {})
    extra_per_fwd = float(tr.get("n_vision_extra", 0)) / max(1, int(tr.get("n_forward", 0)))  # pre_proj：每次前向 1 次
    return {"task": spec.task, "upsample_impl": str(task_cfg["upsample"]["impl"]),
            "n_probes": len(run.bank.names), "n_feature_keys": len(run.extractor.keys),
            "feature_keys": [layer_label(k) for k in run.extractor.keys], "references": list(spec.references),
            "n_probes_pre_proj": sum(1 for e in run.bank.entries.values() if e.key == PRE_PROJ),
            "batch_size": int(task_cfg["batch_size"]), "steps": steps, "n_curve_points": n_curve, "n_val": n_val,
            "n_eval": len(ev), "n_train": len(train), "n_norm": n_norm,
            "measured": {"forward_s": t_fwd, "train_step_s": t_step, "eval_sample_s": t_eval, "train_steps": train_steps,
                         "eval_samples": len(sub), "vision_extra_per_forward": extra_per_fwd},
            "seconds": parts, "gpu_hours": total / 3600.0,
            "forwards": {**forwards, "vision_extra": round(extra_per_fwd * sum(forwards.values()))},
            "peak_mem_bytes": int(torch.cuda.max_memory_allocated(dev)) if dev.type == "cuda" else None}


def summary_markdown(model_key: str, rows: list[dict[str, Any]], header: Mapping[str, Any]) -> str:
    refs = list(header.get("references") or [])
    pre_norm = header.get("pre_norm_last")
    layers = f"{header.get('layer_spec')}（{header.get('n_layers')} 个" + (f"，另加 {pre_norm}" if pre_norm else "") + "）"
    lines = [f"# E2 估算 · {model_key}", "",
             f"- 设备：{header.get('gpu') or header.get('device')}；dtype {header.get('dtype')}",
             f"- 层：{layers}",
             f"- 参考线：{'、'.join(refs) if refs else '无'}"
             + ("（pre_proj 那组探针与视觉塔的额外前向都在实测里）" if PRE_PROJ in refs else ""),
             f"- 含法向：{'是' if header.get('includes_normals') else '否'}；任务：{'、'.join(header.get('tasks') or [])}", "",
             "| 任务 | 上采样实现 | 探针数（pre_proj） | 步 × 批 | 前向 s | 一步 s | 评测一条 s | 评测集 | GPU 小时 | 显存峰值 GB |",
             "|---|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        m = r["measured"]
        pk = r["peak_mem_bytes"]
        lines.append(f"| {r['task']} | {r['upsample_impl']} | {r['n_probes']}（{r['n_probes_pre_proj']}） | "
                     f"{r['steps']} × {r['batch_size']} | {m['forward_s']:.3f} | "
                     f"{m['train_step_s']:.3f} | {m['eval_sample_s']:.3f} | {r['n_eval']} | {r['gpu_hours']:.2f} | "
                     f"{'—' if pk is None else f'{pk / 1e9:.1f}'} |")
    total = sum(r["gpu_hours"] for r in rows)
    lines += ["", f"合计 {total:.2f} GPU 小时（不含 bootstrap 的 CPU 时间与模型加载）。", ""]
    return "\n".join(lines)


__all__ = ["estimate_task", "summary_markdown"]
