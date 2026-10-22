"""W1 的运行器：一个模型 × 一个任务 × 一种设置 × 一个划分 → 输出目录（DEV-0051；docs/w1_plan.md §3）。

输出（<out>/<模型>/<任务>/<设置>/<划分>/；docs/w1_plan.md §3 的布局末尾加了一层划分，同一次运行可以放几个划分）：
- predictions.jsonl：每条样本一行（回路的设置：EpisodeRecord.to_json() + score；基线：见 src/w1/baselines.py）。不含墙钟，逐字节可复现。
- metrics.json：只由 predictions.jsonl 算出（src/w1/scoring.aggregate_w1；E1 的基线用 E1 的 aggregate），含 bootstrap 95% CI。
- compute.json：src/accounting.py（每轮记一次调用；墙钟取 logs/timing.jsonl，同一 id 取最后一条）。
- config.yaml：运行身份（模型与 revision、协议（含 DEV-0053 的开关）、提示走哪种消息（DEV-0055 的 status_role、last_round_role）、
  W1 配置文件（--w1-config）的路径与 sha256、划分清单与这个划分的 sha256、样本 id 指纹、工作台选项与工具清单的 sha256、批大小与
  组大小；不含时间戳与机器相关的路径）；meta.json（src/runlog.write_run_metadata）；README.md。
- logs/：timing.jsonl（每条样本每轮的生成墙钟与工具墙钟）、workbench.jsonl（Workbench.log()：每次调用的耗时、沙箱代码原文……）、
  run_<k>.json。

续跑（--resume）：目录里已有内容时必须给 --resume，且 config.yaml 的身份完全相同（同 E1）；末尾半行截掉；已完成的 id 必须是样本顺序的
前缀；样本按 chunk_size 分组，没写完的那一组整组重跑（组内的批组成因而与一次跑完相同），所以续跑与一次跑完的 predictions.jsonl 逐字节相同。
"""

from __future__ import annotations

import datetime as dt
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from src.accounting import Compute, merge, write_compute_json
from src.data.records import Sample
from src.e1.posthoc_runner import _finish_run_log, _prepare_leaf, _resume_point, _start_run_log
from src.e1.runner import (
    COMPUTE,
    CONFIG,
    METRICS,
    PREDICTIONS,
    README,
    TIMING,
    _append,
    _jsonl_line,
    batch_plan,
    ids_digest,
    load_images,
    read_jsonl_tolerant,
)
from src.e1.tasks import DataPaths
from src.w1.config import W1Config
from src.w1.data import W1_SPLITS_PATH, splits_sha256
from src.w1.loop import Protocol, run_episodes, tools_digest
from src.w1.scoring import aggregate_w1, score_episode

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKBENCH_LOG = "logs/workbench.jsonl"


@dataclass(frozen=True)
class W1RunSpec:
    model_key: str
    family: str
    repo: str
    revision: str
    adapter_model_key: str
    task: str
    split: str
    setting: str
    tiny: bool
    limit: int | None
    image_policy: Mapping[str, Any]


def w1_leaf(out: str | Path, model_key: str, task: str, setting: str, split: str) -> Path:
    return Path(out) / model_key / task / setting / split


def _rel(p: str | Path) -> str:
    q = Path(p).resolve()
    return str(q.relative_to(REPO_ROOT)) if q.is_relative_to(REPO_ROOT) else str(p)


def model_record(spec: W1RunSpec) -> dict[str, Any]:
    return {"key": spec.model_key, "family": spec.family, "repo": spec.repo, "revision": spec.revision,
            "adapter_model_key": spec.adapter_model_key, "tiny": spec.tiny}


def w1_record(spec: W1RunSpec, splits_doc: Mapping[str, Any], samples: Sequence[Sample]) -> dict[str, Any]:
    """运行身份里的 W1 一节（基线的 E1 格式目录也把它放进 selection_stats.w1）：任务、划分、设置、划分清单与这个划分的指纹。"""
    e = splits_doc["splits"][spec.split]
    return {"task": spec.task, "split": spec.split, "setting": spec.setting,
            "splits_config": "tiny（tests/fixtures/e1 上现算）" if splits_doc.get("tiny") else _rel(W1_SPLITS_PATH),
            "splits_sha256": splits_sha256(splits_doc), "split_sha256": e["sha256"], "split_n": e["n"],
            "n_samples": len(samples), "limit": spec.limit, "sample_ids_sha256": ids_digest([s.id for s in samples])}


def probe_tools(factory: Callable[..., Any], sample: Sample, protocol: Protocol,
                options: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """工作台工具清单（运行身份用）：在第一条样本的原图上构造一个工作台（与回路相同的关键字参数：disabled_groups 与
    W1Config.workbench_options()），取 specs() 的名字与 sha256（tools_digest）。"""
    wb = factory(load_images(sample)[0].copy(), disabled_groups=protocol.disabled_groups, **dict(options or {}))
    specs = list(wb.specs())
    return {"tools_sha256": tools_digest(specs), "tools": [s.name for s in specs]}


def loop_identity(spec: W1RunSpec, cfg: W1Config, protocol: Protocol, splits_doc: Mapping[str, Any], samples: Sequence[Sample],
                  tools: Mapping[str, Any], workbench_name: str, batch_size: int, chunk_size: int,
                  paths: DataPaths, max_batch_tokens: int | None = None) -> dict[str, Any]:
    import hashlib

    prompts = cfg.loop_prompts(protocol)
    return {
        "experiment": "W1",
        "framework": "src/w1（DEV-0051；docs/w1_plan.md）",
        "w1": w1_record(spec, splits_doc, samples),
        "model": model_record(spec),
        "image_policy": dict(spec.image_policy),
        "protocol": protocol.as_dict(),
        "prompts": {"config": _rel(cfg.path), "config_sha256": cfg.sha256, "frozen": bool(cfg.data["frozen"]),
                    "system_sha256": hashlib.sha256(prompts.system.encode("utf-8")).hexdigest(),
                    "task_template": prompts.tasks[spec.task], "last_round": prompts.last_round, "status": prompts.status,
                    "final_round_refused": prompts.final_round_refused, "status_role": prompts.status_role,
                    "last_round_role": prompts.last_round_role},
        "workbench": {"factory": workbench_name, "options": dict(cfg.data["workbench"]), **dict(tools)},
        "batching": {"batch_size": batch_size, "chunk_size": chunk_size, "max_batch_tokens": max_batch_tokens,
                     "config_default": cfg.batching(tiny=spec.tiny)["batch_size"] == batch_size},
        "data": {"config_sha256": {Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths.config_files()}},
    }


def _timing_by_id(leaf: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for t in read_jsonl_tolerant(leaf / TIMING, repair=False):
        out[t["id"]] = t
    return out


def loop_compute(recs: Sequence[Mapping[str, Any]], timing: Mapping[str, Mapping[str, Any]]) -> Compute:
    """由 predictions.jsonl 的每轮 token 数与 logs/timing.jsonl 的墙钟重建 compute（与回路里的 EpisodeRecord.compute 同一算法）。"""
    out = []
    for r in recs:
        t = timing.get(r["id"], {})
        walls = list(t.get("round_walls") or [0.0] * len(r["rounds"]))
        calls = [Compute.call(generated_tokens=rd["n_generated_tokens"], prompt_tokens=rd["n_prompt_tokens"] + rd["n_inserted_tokens"],
                              visual_tokens=rd["n_visual_tokens"], image_pixels=rd["n_image_pixels"], call_time_s=float(w),
                              new_round=True) for rd, w in zip(r["rounds"], walls)]
        out.append(merge(calls).as_sample(wall_time_s=float(t.get("wall_time_s", 0.0))))
    return merge(out)


def _fmt(v: Any) -> str:
    return "null" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))


def _ci(ci: Mapping[str, Any] | None) -> str:
    return f"[{_fmt(ci['low'])}, {_fmt(ci['high'])}]" if ci else "—"


def loop_readme(spec: W1RunSpec, cfg: Mapping[str, Any], m: Mapping[str, Any]) -> str:
    pr, w = cfg["protocol"], cfg["w1"]
    lines = [f"# W1 · {spec.model_key} · {spec.task} · {spec.setting} · {spec.split}", "",
             f"- 模型：{spec.repo}@{spec.revision}（适配器 {spec.adapter_model_key}）",
             f"- 协议：至多 {pr['max_rounds']} 轮、{pr['max_tool_calls']} 次计数的工具调用（提交不计数：{pr['submit_exempt_from_cap']}），"
             f"每轮思考 {pr['thinking_budget']}、作答 {pr['answer_budget']} token；关掉的工具组 {pr['disabled_groups'] or '无'}",
             f"- 协议开关：最后一轮的工具 {pr['final_round_tools']}；满额之后的工具 {pr['capped_tools']}；每轮状态提示 "
             f"{'有' if cfg['prompts']['status'] is not None else '无'}；提示走的消息 status → {cfg['prompts'].get('status_role', 'tool')}、"
             f"last_round → {cfg['prompts'].get('last_round_role', 'tool')}；工作台选项 {cfg['workbench']['options']}",
             f"- 提示：{cfg['prompts']['config']} sha256 {cfg['prompts']['config_sha256'][:16]}…（frozen = {cfg['prompts']['frozen']}）；"
             f"工具清单 sha256 {cfg['workbench']['tools_sha256'][:16]}…（{len(cfg['workbench']['tools'])} 个工具）",
             f"- 划分 {w['split']}：n = {m['n']}（划分清单 {w['split_n']} 条，指纹 {w['split_sha256'][:16]}…）；批 {cfg['batching']['batch_size']}"
             f"（行数 × 提示宽度 ≤ {cfg['batching']['max_batch_tokens']}），组 {cfg['batching']['chunk_size']}；"
             f"预检拒收 {m['ended_by'].get('input_rejected', 0)} 条", "", "## 标题指标", ""]
    for h in m["headline"]:
        lines.append(f"- {h['name']}：{_fmt(h['value'])}，95% CI {_ci(h.get('ci'))}，n = {h['n']}")
    lines += ["", "## 回路", "",
              f"- 失败：{m['failures']}；结束方式：{m['ended_by']}；答案来源：{m['answer_source']}",
              f"- 轮数：{m['rounds']}", f"- 计数的工具调用：{m['tool_calls']['counted']}",
              f"- 被拒的调用 {m['tool_calls']['n_refused']}（{m['tool_usage']['refused_by_reason']}）；解析失败的调用 "
              f"{m['tool_calls']['n_tool_call_errors']}；工具报错 {m['tool_usage']['n_tool_errors']}",
              f"- 各工具执行次数：{m['tool_usage']['executed_by_name']}", f"- 思考段的结束方式（逐轮）：{m['thinking_end']}", "",
              "## 文件", "", "- predictions.jsonl：逐条（每轮原文、工具调用与结果摘要、最终答案、得分）；metrics.json；compute.json；config.yaml；"
              "meta.json；logs/（timing.jsonl、workbench.jsonl、run_*.json）。", ""]
    return "\n".join(lines)


def finalize_loop(leaf: Path, spec: W1RunSpec, cfg: Mapping[str, Any], samples: Sequence[Sample]) -> dict[str, Any]:
    recs = read_jsonl_tolerant(leaf / PREDICTIONS, repair=False)
    ids = [s.id for s in samples]
    if [r["id"] for r in recs] != ids[: len(recs)]:
        raise RuntimeError(f"{leaf / PREDICTIONS} 的 id 不是样本顺序的前缀")
    body = aggregate_w1(spec.task, recs)
    metrics = {"experiment": "W1", "w1": cfg["w1"], "model": spec.model_key, "revision": spec.revision,
               "protocol": cfg["protocol"], "batching": cfg["batching"], "prompt_config_sha256": cfg["prompts"]["config_sha256"],
               "tools_sha256": cfg["workbench"]["tools_sha256"], **body, "complete": len(recs) == len(ids), "n_expected": len(ids)}
    (leaf / METRICS).write_text(json.dumps(metrics, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    write_compute_json(leaf / COMPUTE, loop_compute(recs, _timing_by_id(leaf)))
    (leaf / README).write_text(loop_readme(spec, cfg, metrics), encoding="utf-8")
    return metrics


def run_loop(adapter: Any, spec: W1RunSpec, cfg: W1Config, protocol: Protocol, samples: Sequence[Sample], paths: DataPaths,
             out: str | Path, *, splits_doc: Mapping[str, Any], batch_size: int, chunk_size: int, tool_workers: int,
             workbench_factory: Callable[..., Any], workbench_name: str, max_batch_tokens: int | None = None,
             resume: bool = False, argv: Sequence[str] = (),
             log: Callable[[str], None] | None = None, warmup: bool = True) -> dict[str, Any]:
    """回路的一种设置（full、no_eye、no_check、no_snap）跑一个划分，返回 metrics。"""
    log = log or (lambda m: print(m, file=sys.stderr, flush=True))
    if not samples:
        raise ValueError("没有样本")
    leaf = w1_leaf(out, spec.model_key, spec.task, spec.setting, spec.split)
    wb_options = cfg.workbench_options()
    tools = probe_tools(workbench_factory, samples[0], protocol, wb_options)
    cfg_doc = loop_identity(spec, cfg, protocol, splits_doc, samples, tools, workbench_name, batch_size, chunk_size, paths,
                            max_batch_tokens=max_batch_tokens)
    files = [cfg.path, *([] if splits_doc.get("tiny") else [W1_SPLITS_PATH]), *paths.config_files()]
    _prepare_leaf(leaf, cfg_doc, files, resume=resume)
    ids = [s.id for s in samples]
    plan = batch_plan([None] * len(ids), chunk_size)
    what = f"W1 {spec.model_key} {spec.task}/{spec.setting}/{spec.split}"
    keep, trimmed = _resume_point(leaf / PREDICTIONS, ids, lambda r: r["id"], plan, log, what)
    todo = [(a, b) for a, b in plan if a >= keep]
    run_k, run_path, run_log = _start_run_log(leaf, argv, n_done_before=keep, n_remaining=len(ids) - keep, batch_size=batch_size,
                                              chunk_size=chunk_size, tool_workers=tool_workers, trimmed_to_batch_boundary=trimmed)
    t_all = time.perf_counter()
    if todo and warmup and hasattr(adapter, "warmup"):
        run_log["warmup_wall_s"] = adapter.warmup(load_images(samples[keep]), batch_size=min(batch_size, len(ids) - keep))
    prompts = cfg.loop_prompts(protocol)
    n_chunks = 0
    for a, b in todo:
        chunk = list(samples[a:b])
        recs = run_episodes(adapter, chunk, spec.task, protocol, prompts, batch_size=batch_size, max_batch_tokens=max_batch_tokens,
                            workbench_factory=workbench_factory, workbench_options=wb_options, tool_workers=tool_workers,
                            expected_tools_sha256=tools["tools_sha256"], log=log)
        lines, tlines, wlines = [], [], []
        for s, rec in zip(chunk, recs):
            row = {"id": s.id, "source": s.source, "split": s.split, "w1_split": spec.split, "setting": spec.setting,
                   **{k: v for k, v in rec.to_json().items() if k != "id"}, "score": score_episode(spec.task, s, rec)}
            lines.append(_jsonl_line(row))
            tlines.append(_jsonl_line({**rec.timing(), "run": run_k, "chunk_index": n_chunks, "chunk_rows": len(chunk)}))
            wlines.append(json.dumps({"id": s.id, "run": run_k, "log": rec.workbench_log}, ensure_ascii=False, default=str) + "\n")
        _append(leaf / TIMING, tlines)
        _append(leaf / WORKBENCH_LOG, wlines)
        _append(leaf / PREDICTIONS, lines)
        n_chunks += 1
        log(f"[{what}] {b}/{len(ids)}")
    metrics = finalize_loop(leaf, spec, cfg_doc, samples)
    _finish_run_log(run_path, run_log, n_batches=n_chunks, t_all=t_all)
    return metrics


def read_identity(leaf: str | Path) -> dict[str, Any]:
    """一个 W1 输出目录的身份摘要（w1_gate 用）：模型、revision、任务、设置、划分，以及 predictions 是否写完。三种目录格式都认：
    W1 自己的（config.yaml 的 w1 一节）与 E1 格式的基线（config.yaml 的 data.selection_stats.w1）。"""
    import yaml

    leaf = Path(leaf)
    cfg = yaml.safe_load((leaf / CONFIG).read_text(encoding="utf-8"))
    w1 = cfg.get("w1") or ((cfg.get("data") or {}).get("selection_stats") or {}).get("w1")
    if not w1:
        raise ValueError(f"{leaf}：config.yaml 里没有 W1 一节（不是 W1 的输出目录）")
    m = json.loads((leaf / METRICS).read_text(encoding="utf-8")) if (leaf / METRICS).exists() else {}
    return {"leaf": str(leaf), "model": cfg["model"]["key"], "revision": cfg["model"]["revision"], "tiny": cfg["model"].get("tiny"),
            "task": w1["task"], "setting": w1["setting"], "split": w1["split"], "split_sha256": w1["split_sha256"],
            "n_samples": w1["n_samples"], "limit": w1.get("limit"), "complete": m.get("complete")}


__all__ = [
    "W1RunSpec",
    "WORKBENCH_LOG",
    "finalize_loop",
    "loop_compute",
    "loop_identity",
    "model_record",
    "probe_tools",
    "read_identity",
    "run_loop",
    "w1_leaf",
    "w1_record",
]
