"""吞吐与算力估算（DEV-0014 任务书 §4；D28-10，预注册草稿 §9；协议 v2 见 DEV-0017）。

在开发子集上，按每个任务的真实提示、读法与生成上限，测每个 batch 大小的样本/秒与 token/秒（墙钟只含 model.generate，预热不计，
与 scripts/validate_batching.py 同一口径），再按各子集的 n（由 src/e1/tasks.select 实际数出）外推 GPU 小时。

- 关思考（eval、extra 子集）：每个批大小测一遍；GPU 上 OOM 记为不可行，峰值显存（torch.cuda.max_memory_allocated）超过
  总显存的 MEM_HEADROOM 也记为不可行；可行的里取样本/秒最高的作为推荐批大小。开发机（CPU）上只测速度，不判显存。
  外推用冻结的批大小（batch_protocol.batch_size）的实测；它没测或不可行时退回推荐批大小，用的是哪个写在 extrapolated_with。
- 选择题与点对任务：eval 是受限字母读数（1 个 token，按字母集合分批，与 runner 同一个 batch_plan）；free 子集（自由作答诊断，
  关思考、上限 1024）在少量开发样本上按自由作答另测一遍，外推 free。
- 开思考（think，可选 --estimate-think）：与正式运行同一个生成方式——预算强制收尾（D59-1：generate_batch_budgeted，思考与作答的
  预算取配置的 generation.think_budget），用 min(推荐批大小, 样本数) 在少量开发样本上测一遍（批比正式运行小，吞吐偏保守），
  另记各结束方式的条数。think8192 子集与它的上界估计由 D59-1 取消。
- 附加对照（extra）只在 --policy 与该子集的口径一致时外推（qwen:1024、internvl:native），否则记 null。
开发子集的输出长度未必代表评测集（例如检测的开发图换成了 RefCOCO 的图），外推只是估计，实际用量超过估算 20% 就停（D28-10）。
"""

from __future__ import annotations

import time
from typing import Any, Mapping, Sequence

from src.e1.letters import build_letter_table
from src.e1.prompts import DEFAULT_COUNT_QUESTIONS, FREE_SUBSET, PromptConfig
from src.e1.runner import batch_keys, batch_plan, generate_rows, load_images
from src.e1.tasks import DataPaths, TASK_SUBSETS, select

MEM_HEADROOM = 0.9
THINK_BUDGET_NOTE = "think = 预算强制收尾（D59-1）的实测外推；开发样本的思考长度未必代表评测集"
SUBSET_TOTALS: tuple[str, ...] = ("eval", "think", "extra", FREE_SUBSET)


def _cuda(adapter: Any):
    import torch

    dev = getattr(adapter, "device", None)
    dev = torch.device(dev) if dev is not None else torch.device("cpu")
    return (torch, dev) if dev.type == "cuda" else (None, dev)


def measure(adapter: Any, task: str, family: str, samples: Sequence[Any], prompts: PromptConfig, *, batch_size: int,
            enable_thinking: bool, max_new_tokens: int, subset: str = "eval") -> dict[str, Any]:
    """一个批大小上的吞吐（读法与分批按 subset 取，与 runner 相同）。OOM 时 ok=False、oom=True（不抛出）。"""
    torch, dev = _cuda(adapter)
    if torch is not None:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(dev)
    kind = prompts.readout(task, subset)
    think_budget = prompts.think_budget() if prompts.uses_think_budget(subset) else None
    rendered = [prompts.render(task, family, s) for s in samples]
    table = None
    if kind == "letter" and rendered:
        table = build_letter_table(adapter.tokenizer, {lt for r in rendered for lt in r.extras["letters"]},
                                   prompts.letter_spellings())
    n_rows = n_tok = n_trunc = n_ins = 0
    ends: dict[str, int] = {}
    gen_wall = build_wall = 0.0
    out: dict[str, Any] = {"batch_size": batch_size, "subset": subset, "readout": kind, "enable_thinking": enable_thinking,
                           "max_new_tokens": max_new_tokens}
    if think_budget is not None:
        out["think_budget"] = {"thinking_budget": think_budget[0], "answer_budget": think_budget[1]}
    try:
        for a, b in batch_plan(batch_keys(kind, rendered), batch_size):
            t0 = time.perf_counter()
            inputs = [adapter.build_inputs(load_images(s), r.text, prompts.order)
                      for s, r in zip(samples[a:b], rendered[a:b])]
            build_wall += time.perf_counter() - t0
            allowed = table.allowed_ids(rendered[a].extras["letters"]) if table is not None else None
            res = generate_rows(adapter, inputs, batch_size=batch_size, enable_thinking=enable_thinking,
                                max_new_tokens=max_new_tokens, decoding=prompts.decoding(), allowed_token_ids=allowed,
                                think_budget=think_budget)
            gen_wall += res[0].batch_wall_time_s if res[0].batch_wall_time_s is not None else sum(r.wall_time_s for r in res)
            n_rows += len(res)
            n_tok += sum(r.n_generated_tokens for r in res)
            n_trunc += sum(1 for r in res if r.stopped is False and kind != "letter")
            if think_budget is not None:
                n_ins += sum(r.n_inserted_tokens for r in res)
                for r in res:
                    ends[r.thinking_end] = ends.get(r.thinking_end, 0) + 1
    except Exception as e:  # noqa: BLE001
        if torch is not None and isinstance(e, torch.cuda.OutOfMemoryError):
            torch.cuda.empty_cache()
            out.update(ok=False, oom=True, error="CUDA OOM")
            return out
        raise
    out.update(ok=True, oom=False, n_rows=n_rows, generated_tokens=n_tok, gen_wall_s=gen_wall, build_inputs_s=build_wall,
               samples_per_s=n_rows / gen_wall if gen_wall > 0 else None,
               tokens_per_s=n_tok / gen_wall if gen_wall > 0 else None,
               mean_generated_tokens=n_tok / n_rows if n_rows else None,
               truncation_rate=n_trunc / n_rows if n_rows else None)
    if think_budget is not None:
        out.update(inserted_tokens=n_ins, thinking_end_counts=dict(sorted(ends.items())))
    if torch is not None:
        peak = int(torch.cuda.max_memory_allocated(dev))
        total = int(torch.cuda.get_device_properties(dev).total_memory)
        out.update(peak_mem_bytes=peak, total_mem_bytes=total, mem_ok=peak <= MEM_HEADROOM * total)
    else:
        out.update(peak_mem_bytes=None, total_mem_bytes=None, mem_ok=None)
    return out


def _feasible(r: Mapping[str, Any]) -> bool:
    return bool(r.get("ok") and r.get("mem_ok") is not False and r.get("samples_per_s"))


def best_batch(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    ok = [r for r in rows if _feasible(r)]
    return max(ok, key=lambda r: (r["samples_per_s"], -r["batch_size"])) if ok else None


def _hours(n: int, sps: float | None) -> float | None:
    return None if not sps else n / sps / 3600.0


def estimate_task(adapter: Any, task: str, family: str, prompts: PromptConfig, paths: DataPaths, *,
                  batch_sizes: Sequence[int], n_dev: int, think: bool, n_think: int, policy_tag: str,
                  count_questions: str = DEFAULT_COUNT_QUESTIONS, log=print) -> dict[str, Any]:
    dev = select(task, "dev", paths, count_questions=count_questions).samples[:n_dev]
    max_new = prompts.max_new_tokens(task, family, "eval")
    frozen = prompts.batch_protocol()["batch_size"]
    out: dict[str, Any] = {"task": task, "n_dev_used": len(dev), "readout": prompts.readout(task, "eval"),
                           "max_new_tokens": max_new, "frozen_batch_size": frozen, "by_batch_size": []}
    if dev:  # 预热用最小的批大小：最大的那个可能 OOM，留给 measure 去判
        adapter.warmup(load_images(dev[0]), batch_size=min(batch_sizes))
    for bs in batch_sizes:
        if not dev:
            break
        r = measure(adapter, task, family, dev, prompts, batch_size=bs, enable_thinking=False, max_new_tokens=max_new)
        log(f"[estimate {task}] bs={bs} {r.get('samples_per_s')} samples/s ok={r['ok']}")
        out["by_batch_size"].append(r)
    best = best_batch(out["by_batch_size"])
    out["best_batch_size"] = None if best is None else best["batch_size"]
    at_frozen = next((r for r in out["by_batch_size"] if r["batch_size"] == frozen and _feasible(r)), None)
    use = at_frozen or best
    out["extrapolated_with"] = None if use is None else {"batch_size": use["batch_size"],
                                                         "why": "冻结的批大小" if use is at_frozen else "推荐批大小（冻结的没测或不可行）"}
    sps = None if use is None else use["samples_per_s"]
    out["samples_per_s"] = sps
    n = {s: len(select(task, s, paths, count_questions=count_questions).samples) for s in TASK_SUBSETS[task] if s != "dev"}
    out["n"] = n
    # 主口径的子集只按主口径的实测外推；附加对照（extra）只按附加口径（qwen:1024 / internvl:native）的实测外推
    gh: dict[str, Any] = {"eval": _hours(n["eval"], sps) if policy_tag == "default" else None}
    if "extra" in n:
        gh["extra"] = _hours(n["extra"], sps) if policy_tag != "default" else None
    if FREE_SUBSET in n:  # 自由作答诊断：另测一遍自由作答（关思考，上限取任务条目的 max_new_tokens）
        free_row = None
        if dev and policy_tag == "default":
            k = max(1, n_think)
            free_row = measure(adapter, task, family, dev[:k], prompts, batch_size=min(frozen, k), enable_thinking=False,
                               max_new_tokens=prompts.max_new_tokens(task, family, FREE_SUBSET), subset=FREE_SUBSET)
            out["free_measurement"] = free_row
        f_sps = free_row.get("samples_per_s") if free_row and free_row.get("ok") else None
        gh[FREE_SUBSET] = _hours(n[FREE_SUBSET], f_sps) if policy_tag == "default" else None
    think_row = None
    if think and dev and best is not None and policy_tag == "default":
        bs = best["batch_size"]
        think_row = measure(adapter, task, family, dev[:n_think], prompts, batch_size=min(bs, max(1, n_think)),
                            enable_thinking=True, max_new_tokens=prompts.max_new_tokens(task, family, "think"), subset="think")
        out["think_measurement"] = think_row
    t_sps = think_row.get("samples_per_s") if think_row and think_row.get("ok") else None
    gh["think"] = _hours(n["think"], t_sps) if policy_tag == "default" else None
    out["gpu_hours"] = gh
    out["notes"] = [THINK_BUDGET_NOTE] if think else ["思考子集未测（加 --estimate-think）"]
    return out


def summary_markdown(model_key: str, policy_tag: str, rows: Sequence[Mapping[str, Any]], header: Mapping[str, Any]) -> str:
    def f(v, d=3):
        return "—" if v is None else (f"{v:.{d}f}" if isinstance(v, float) else str(v))

    lines = [f"# 吞吐与算力估算 · {model_key} · 口径 {policy_tag}", "", f"- {dict(header)}", "",
             "| 任务 | 读法 | 开发样本 | max_new_tokens | 推荐批大小 | 外推用的批大小 | 样本/秒 | n(eval) | GPU 小时(eval) | n(think) | "
             "GPU 小时(think，预算强制收尾) | n(extra) | GPU 小时(extra) | n(free) | GPU 小时(free) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    tot: dict[str, float | None] = {k: None for k in SUBSET_TOTALS}
    for r in rows:
        g, n = r["gpu_hours"], r["n"]
        for k in tot:
            if g.get(k) is not None:
                tot[k] = (tot[k] or 0.0) + g[k]
        ex = r.get("extrapolated_with") or {}
        lines.append(f"| {r['task']} | {r.get('readout', '—')} | {r['n_dev_used']} | {r['max_new_tokens']} | {f(r['best_batch_size'])} | "
                     f"{f(ex.get('batch_size'))} | {f(r['samples_per_s'])} | {n.get('eval')} | {f(g.get('eval'))} | {n.get('think')} | "
                     f"{f(g.get('think'))} | {n.get('extra', '—')} | "
                     f"{f(g.get('extra'))} | {n.get(FREE_SUBSET, '—')} | {f(g.get(FREE_SUBSET))} |")
    lines.append(f"| 合计 | | | | | | | | {f(tot['eval'])} | | {f(tot['think'])} | | "
                 f"{f(tot['extra'])} | | {f(tot[FREE_SUBSET])} |")
    lines += ["", "逐批大小的明细（样本/秒、token/秒、峰值显存、截断率）见 estimate.json。"]
    return "\n".join(lines) + "\n"


__all__ = ["MEM_HEADROOM", "SUBSET_TOTALS", "THINK_BUDGET_NOTE", "best_batch", "estimate_task", "measure", "summary_markdown"]
