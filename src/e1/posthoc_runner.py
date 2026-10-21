"""E1 事后读数的运行器（DEV-0033；D66；E1 预注册 §10）：去偏字母读数与格式强制诊断，一个模型 × 一个任务 × 一种读法 → 输出目录。

与主评测的运行器（src/e1/runner.py）同一套约定，只是读法不同；主评测的代码路径一行不改：
- 样本：src/e1/tasks.select 的 eval / dev（去偏读数）或 think 的 id（格式强制，关思考），顺序固定。
- 批：runner.batch_plan，批大小取冻结的 batch_protocol（64），左填充；去偏读数按 (顺序, 字母集合) 分批（order_major，见
  src/e1/debiased.reading_rows），格式强制按样本顺序连续成批。
- 每批写完才落盘（先 logs/timing.jsonl，再数据文件，flush + fsync）；续跑（--resume）要求 config.yaml 完全相同，末尾半行截掉，
  没写完的那一批整批重跑，所以续跑与一次跑完逐字节相同。没有 --resume 而目录非空：拒绝覆盖。
- config.yaml（运行身份）= 主评测的 run_config（取样本的那个 E1 子集）+ subset 换成事后子集 + generation 换成这种读法 + posthoc 一节
  （事后配置 configs/e1_posthoc.json 的 sha256、读法、取样本的子集）。meta.json 另记事后配置的 sha256。

输出目录（<out>/<模型>/<任务>__<子集>）：
- 去偏读数（--subset debiased | debiased_dev）：<任务>__debiased[_dev]/
  - readings.jsonl：每行一次读数（样本 × 顺序，order_major 的顺序）：perm、该顺序的指令原文、字母的 logit 与概率、argmax；
  - predictions.jsonl：每条样本一行（选择顺序）：k 种顺序的字母概率、映射回选项内容后的平均概率、答案与是否正确、原字母读数
    （s = 0）；由 readings.jsonl 算出（finalize 时整份重写，逐字节可复现）；
  - metrics.json：debiased.debiased_metrics（标题 = 去偏读数的严格读数与 CI；original_letter、paired_diff、per_order、random_level）；
  - compute.json：每次读数记一次调用（生成 1 个 token，D37），同一条样本的 k 次调用合成一个样本（第一次记新的一轮）。
- 格式强制（--subset forced）：<任务>__forced_<读法>/（refexp_poly__forced_prefill、grid16__forced_restricted、grid16__forced_structured），
  文件同主评测（predictions.jsonl、metrics.json、compute.json、README.md、config.yaml、meta.json、logs/）；记录另有 forced 一节
  （读法、是否固定长度、预填与它的 token）。

训练后的检查点（DEV-0035；D66-3、D68-4(a)）：去偏读数可以在 load_trained(merge=True) 的模型上跑（scripts/run_e1.py --subset debiased
--checkpoint …，先过合并核对），读法与输出同上，config.yaml 另记 checkpoint、README 另记一行，目录在 <out>/<标签>/ 之下。格式强制只用于
base 的 Qwen（D68-3），不配检查点。base 运行的输出逐字节不变。

DEV-0036（D72-2、D72-3；事后）另加两种读法，上面两种的代码路径一行不改：
- 对调标记的去偏读数（--subset letter_swap | letter_swap_dev，只有 NYU 点对）：<任务>__letter_swap[_dev]/，文件同去偏读数。
  每条样本读两次（original、labels_swapped 两张图，提示不变），order_major 分批（original 那一段与主评测逐批相同），labels_swapped 的图
  在运行时从原图渲染，并核对它与 original 只差标签框里的字形（src/e1/letter_swap.check_swapped）。
- 结构强制的多边形（--subset forced_poly，只做 Qwen）：<任务>__<变体>/（refexp_poly__poly_structured、refexp_poly__poly_structured_min8），
  文件同格式强制；逐位置的语法处理器（src/e1/poly_forced.PolyStructureProcessor），入口断言见 poly_forced.check_poly_output。
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Hashable, Mapping, Sequence

import yaml

from src.accounting import Compute, merge, write_compute_json
from src.constants import BOOTSTRAP_SEED
from src.e1 import letter_swap as LS
from src.e1.debiased import combine_sample, debiased_metrics, ordered_prompts, reading_record, reading_rows
from src.e1.forced import GridStructureProcessor, check_fixed_output, grid_format_ids, parse_view
from src.e1.letters import LetterTable, build_letter_table
from src.e1.poly_forced import (
    PolyGrammar,
    PolyStructureProcessor,
    check_poly_output,
    near_rect_summary,
    near_rectangle,
    poly_format_ids,
)
from src.e1.posthoc import (
    DEBIASED_READOUT,
    DEBIASED_SUBSETS,
    FORCED_POLY_SUBSET,
    FORCED_SUBSET,
    LETTER_SWAP_READOUT,
    LETTER_SWAP_SUBSETS,
    LETTER_SWAP_VIEWS,
    PosthocConfig,
)
from src.e1.prompts import PromptConfig
from src.e1.runner import (
    COMPUTE,
    CONFIG,
    METRICS,
    PREDICTIONS,
    README,
    TIMING,
    RunSpec,
    _append,
    _common_record,
    _identity,
    _jsonl_line,
    _metrics_doc,
    _truncate_lines,
    batch_plan,
    grid_token_diagnostic,
    leaf_dir,
    load_images,
    read_jsonl_tolerant,
    readme_text,
    run_config,
)
from src.e1.scoring import aggregate, distribution, is_valid, score_generation
from src.e1.tasks import DataPaths, Selection
from src.runlog import write_run_metadata

REPO_ROOT = Path(__file__).resolve().parents[2]
READINGS = "readings.jsonl"


def _rel(path: str | Path) -> str:
    p = Path(path)
    return os.path.relpath(p, REPO_ROOT) if p.resolve().is_relative_to(REPO_ROOT) else str(p)


def _log_default(m: str) -> None:
    print(m, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# 公共：输出目录、续跑、运行记录
# ---------------------------------------------------------------------------


def _prepare_leaf(leaf: Path, cfg: Mapping[str, Any], files: Sequence[str | Path], *, resume: bool) -> None:
    """新目录：写 config.yaml 与 meta.json；已有内容：必须 --resume 且 config.yaml 完全相同（同 runner.run_task）。"""
    if leaf.exists() and any(leaf.iterdir()):
        if not resume:
            raise FileExistsError(f"{leaf} 已存在且非空；续跑请加 --resume")
        if not (leaf / CONFIG).exists():
            raise FileExistsError(f"{leaf} 里没有 {CONFIG}，不能续跑")
        old = yaml.safe_load((leaf / CONFIG).read_text(encoding="utf-8"))
        a, b = _identity(old), _identity(cfg)
        if a != b:
            diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
            raise ValueError(f"{leaf}：续跑的配置与已有的不同（{diff}），拒绝混写")
    else:
        leaf.mkdir(parents=True, exist_ok=True)
        (leaf / "logs").mkdir(exist_ok=True)
        write_run_metadata(leaf, config=cfg, seed=BOOTSTRAP_SEED, data_files=list(files), repo_root=REPO_ROOT)
    (leaf / "logs").mkdir(exist_ok=True)


def _resume_point(path: Path, expected: Sequence[Hashable], key_of: Callable[[Mapping[str, Any]], Hashable],
                  plan: Sequence[tuple[int, int]], log: Callable[[str], None], what: str) -> tuple[int, dict | None]:
    """已完成的行必须是预期序列的前缀；不在批的边界上就截回那一批的开头（整批重跑）。返回 (保留的行数, 截断记录)。"""
    done = read_jsonl_tolerant(path, repair=True)
    if [key_of(r) for r in done] != list(expected[: len(done)]):
        raise RuntimeError(f"{path} 里已完成的行不是预期顺序的前缀，不能续跑")
    bounds = [a for a, _ in plan] + [len(expected)]
    keep = max(b for b in bounds if b <= len(done))
    trimmed = None
    if keep < len(done):
        _truncate_lines(path, keep)
        trimmed = {"from": len(done), "to": keep}
        log(f"[{what}] 已完成 {len(done)} 行不在批的边界上，退回到 {keep} 行整批重跑")
    return keep, trimmed


def _start_run_log(leaf: Path, argv: Sequence[str], **fields: Any) -> tuple[int, Path, dict[str, Any]]:
    run_k = len(list((leaf / "logs").glob("run_*.json"))) + 1
    run_path = leaf / "logs" / f"run_{run_k:03d}.json"
    run_log: dict[str, Any] = {"argv": list(argv), "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                               **fields, "finished_utc": None}
    run_path.write_text(json.dumps(run_log, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return run_k, run_path, run_log


def _finish_run_log(run_path: Path, run_log: dict[str, Any], *, n_batches: int, t_all: float) -> None:
    run_log.update(finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), n_batches=n_batches,
                   wall_s=round(time.perf_counter() - t_all, 3))
    run_path.write_text(json.dumps(run_log, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def _posthoc_record(posthoc: PosthocConfig, kind: str, base_subset: str, **extra: Any) -> dict[str, Any]:
    return {"kind": kind, "config": _rel(posthoc.path), "config_sha256": posthoc.sha256, "base_subset": base_subset,
            "note": "事后读数（D66；E1 预注册 §10，写于运行之前）：报告里注明\"事后\"", **extra}


def _base_config(spec: RunSpec, sel: Selection, prompts: PromptConfig, paths: DataPaths, *, readout: Mapping[str, Any],
                 generation: Mapping[str, Any], posthoc_info: Mapping[str, Any]) -> dict[str, Any]:
    """config.yaml：主评测的 run_config（按 eval 口径取提示条目、数据、批配置），subset、generation、framework 换成事后读数的，
    末尾加 posthoc 一节。"""
    cfg = run_config(dataclasses.replace(spec, subset="eval"), sel, prompts, paths, readout=readout)
    cfg["framework"] = "src/e1 事后读数（DEV-0033；E1 预注册 §10，D66）"
    cfg["subset"] = spec.subset
    cfg["generation"] = dict(generation)
    cfg["posthoc"] = dict(posthoc_info)
    return cfg


# ---------------------------------------------------------------------------
# 去偏字母读数（§10.1）
# ---------------------------------------------------------------------------


def check_spellings(scores: Sequence[Any], table: LetterTable, letters: Sequence[str], where: str) -> None:
    """option_logits 用的写法必须与本次运行的字母表（主评测同一种构造）的 token 逐个相同，否则两种读数不是同一个机制。"""
    for sc in scores:
        for lt, sp in zip(letters, sc.spellings):
            got = sorted(s.token_id for s in sp)
            want = sorted(tid for _, tid, _ in table.entries[lt])
            if got != want:
                raise RuntimeError(f"{where}：字母 {lt!r} 在 option_logits 里的写法 token {got} 与字母表 {want} 不同")


def run_debiased_task(adapter: Any, spec: RunSpec, sel: Selection, prompts: PromptConfig, posthoc: PosthocConfig,
                      paths: DataPaths, out_dir: str | Path, *, resume: bool = False, argv: Sequence[str] = (),
                      log: Callable[[str], None] | None = None, warmup: bool = True,
                      run_extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """去偏字母读数（规则见 src/e1/debiased.py 与 configs/e1_posthoc.json 的 debiased_letter）。返回 metrics。
    训练后的检查点（DEV-0035，D68-4(a)；scripts/run_e1.py --subset debiased --checkpoint）：adapter 是 load_trained(merge=True) 的结果，
    spec.checkpoint 经 runner.run_config 进 config.yaml 的 checkpoint（base 没有这个键，输出不变）；run_extra（合并核对的摘要与检查点路径）
    并进 logs/run_<k>.json。"""
    log = log or _log_default
    if spec.subset not in DEBIASED_SUBSETS or spec.task not in posthoc.tasks_for(spec.subset):
        raise ValueError(f"去偏字母读数只用于 {posthoc.debiased['tasks']} 的 {DEBIASED_SUBSETS}，收到 {spec.task}/{spec.subset}")
    base = posthoc.base_subset(spec.subset)
    if sel.subset != base:
        raise ValueError(f"样本取自 {sel.subset!r}，{spec.subset} 应取 {base!r}")
    enable_thinking = prompts.enable_thinking(base)
    pairs = [ordered_prompts(prompts, posthoc, spec.task, spec.family, s) for s in sel.samples]  # 每条都断言原顺序逐字节相同
    rendered = [r for r, _ in pairs]
    orders = [o for _, o in pairs]
    letters = [tuple(r.extras["letters"]) for r in rendered]
    union = sorted({lt for ls in letters for lt in ls})
    table = build_letter_table(adapter.tokenizer, union, prompts.letter_spellings()) if union else None
    rows, keys = reading_rows([len(ls) for ls in letters], letters)
    plan = batch_plan(keys, spec.batch_size)
    readout = {"kind": DEBIASED_READOUT, "parse": "letter_token", "spellings": list(prompts.letter_spellings()),
               "table": table.as_dict() if table is not None else None, "orders": posthoc.debiased["orders"],
               "method": "option_logits"}
    generation = {"enable_thinking": enable_thinking, "max_new_tokens": 1,
                  "decoding": "option_logits：一次前向，取第一个生成位置的原始 logits（不采样；D37）", "order": prompts.order,
                  "readout": readout}
    info = _posthoc_record(posthoc, "debiased_letter", base, version=posthoc.debiased["version"],
                           layout=posthoc.debiased["batch_layout"]["layout"], n_readings=len(rows),
                           split_how=posthoc.split_how(spec.task), line_format=posthoc.line_format)
    cfg = _base_config(spec, sel, prompts, paths, readout=readout, generation=generation, posthoc_info=info)
    leaf = leaf_dir(out_dir, spec.model_key, spec.task, spec.subset, spec.policy_tag)
    _prepare_leaf(leaf, cfg, [prompts.path, posthoc.path, *paths.config_files()], resume=resume)
    what = f"{spec.model_key} {spec.task}/{spec.subset}"
    expected = [(sel.samples[j].id, s) for j, s in rows]
    keep, trimmed = _resume_point(leaf / READINGS, expected, lambda r: (r["id"], int(r["order"])), plan, log, what)
    todo = [(a, b) for a, b in plan if a >= keep]
    run_k, run_path, run_log = _start_run_log(leaf, argv, n_done_before=keep, n_remaining=len(rows) - keep,
                                              batch_size=spec.batch_size, trimmed_to_batch_boundary=trimmed,
                                              **dict(run_extra or {}))
    t_all = time.perf_counter()
    if todo and warmup:
        run_log["warmup_wall_s"] = adapter.warmup(load_images(sel.samples[rows[keep][0]]),
                                                  batch_size=min(spec.batch_size, len(rows) - keep))
    n_batches = 0
    for a, b in todo:
        batch = rows[a:b]
        lts = letters[batch[0][0]]
        if any(letters[j] != lts or s != batch[0][1] for j, s in batch):
            raise RuntimeError("同一批的顺序或字母集合不同（batch_plan 的分批键有误）")
        t0 = time.perf_counter()
        imgs = [load_images(sel.samples[j]) for j, _ in batch]
        inputs = [adapter.build_inputs(im, orders[j][s].text, prompts.order) for im, (j, s) in zip(imgs, batch)]
        build_s = time.perf_counter() - t0
        scores = adapter.option_logits(inputs, [list(lts)] * len(batch), enable_thinking=enable_thinking)
        if len(scores) != len(batch):
            raise RuntimeError("option_logits 返回的行数与输入不同")
        check_spellings(scores, table, lts, what)
        lines, tlines = [], []
        for (j, s), im, sc in zip(batch, imgs, scores):
            rec = reading_record(sel.samples[j], orders[j][s], lts, sc)
            rec["image_size"] = list(im[0].size)
            lines.append(_jsonl_line(rec))
            tlines.append(_jsonl_line({"id": sel.samples[j].id, "order": s, "run": run_k, "batch_index": n_batches,
                                       "batch_rows": len(batch), "wall_time_s": sc.wall_time_s,
                                       "batch_wall_time_s": sc.batch_wall_time_s, "build_inputs_s": build_s / len(batch)}))
        _append(leaf / TIMING, tlines)
        _append(leaf / READINGS, lines)
        n_batches += 1
        if n_batches == 1 or n_batches % 20 == 0 or b == len(rows):
            log(f"[{what}] {b}/{len(rows)} 次读数")
    metrics = finalize_debiased(leaf, spec, sel, cfg, prompts, rendered, rows)
    _finish_run_log(run_path, run_log, n_batches=n_batches, t_all=t_all)
    return metrics


def finalize_debiased(leaf: Path, spec: RunSpec, sel: Selection, cfg: Mapping[str, Any], prompts: PromptConfig,
                      rendered: Sequence[Any], rows: Sequence[tuple[int, int]]) -> dict[str, Any]:
    """由 readings.jsonl 与 logs/timing.jsonl 写 predictions.jsonl、metrics.json、compute.json、README.md（只读文件，逐字节可复现）。"""
    readings = read_jsonl_tolerant(leaf / READINGS, repair=False)
    expected = [(sel.samples[j].id, s) for j, s in rows]
    if [(r["id"], int(r["order"])) for r in readings] != expected:
        raise RuntimeError(f"{leaf / READINGS} 的读数不全或顺序不对（{len(readings)}/{len(expected)}）")
    by_id: dict[str, list[dict[str, Any]]] = {}
    for r in readings:
        by_id.setdefault(r["id"], []).append(r)
    recs = []
    for s, ren in zip(sel.samples, rendered):
        rs = sorted(by_id[s.id], key=lambda r: int(r["order"]))
        recs.append(combine_sample(spec.task, s, ren, rs[0]["image_size"], rs))
    (leaf / PREDICTIONS).write_text("".join(_jsonl_line(r) for r in recs), encoding="utf-8")
    body = debiased_metrics(spec.task, sel.samples, recs, has_gt=sel.has_gt)
    body["complete"] = True
    body["n_expected"] = len(sel.samples)
    metrics = _metrics_doc(spec, cfg, body, prompts)
    metrics["posthoc"] = cfg["posthoc"]
    (leaf / METRICS).write_text(json.dumps(metrics, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    timing: dict[tuple[str, int], dict[str, Any]] = {}
    for t in read_jsonl_tolerant(leaf / TIMING, repair=False):
        timing[(t["id"], int(t["order"]))] = t
    comp = []
    for s in sel.samples:
        calls, wall = [], 0.0
        for r in sorted(by_id[s.id], key=lambda r: int(r["order"])):
            w = float(timing.get((s.id, int(r["order"])), {}).get("wall_time_s", 0.0))
            wall += w
            calls.append(Compute.call(generated_tokens=1, prompt_tokens=int(r["n_prompt_tokens"]),
                                      visual_tokens=int(r["n_visual_tokens"]), image_pixels=int(r["n_image_pixels"]),
                                      call_time_s=w, new_round=int(r["order"]) == 0))
        comp.append(merge(calls).as_sample(wall_time_s=wall))
    write_compute_json(leaf / COMPUTE, merge(comp))
    (leaf / README).write_text(debiased_readme(spec, cfg, metrics), encoding="utf-8")
    return metrics


def _f(v: Any) -> str:
    return "null" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))


def _ci(ci: Mapping[str, Any] | None) -> str:
    return f"[{_f(ci['low'])}, {_f(ci['high'])}]" if ci else "—"


def debiased_readme(spec: RunSpec, cfg: Mapping[str, Any], metrics: Mapping[str, Any]) -> str:
    d = metrics["debiased"]
    ph = cfg["posthoc"]
    lines = [f"# E1 事后读数 · 去偏字母读数 · {spec.model_key} · {spec.task} · {spec.subset}", "",
             "事后修订（D66-3，N22 → (b)；E1 预注册 §10.1，写于运行之前）。规则见 configs/e1_posthoc.json 的 debiased_letter。", "",
             f"- 模型：{spec.repo}@{spec.revision}（{spec.family}）；图像口径：{cfg['image_policy']}",
             f"- 样本：E1 的 {ph['base_subset']} 子集，n = {metrics['n']}；读数 {ph['n_readings']} 次（k 种循环移位，order_major 分批，"
             f"批大小 {cfg['batch_protocol']['batch_size']}）",
             f"- 提示配置 sha256：{cfg['prompt']['config_sha256']}；事后配置 sha256：{ph['config_sha256']}",
             f"- 随机水平（逐题 1/k 的平均）：{_f(d['random_level'])}；k 的分布：{d['k_counts']}", "",
             "## 标题：去偏读数（严格读数）", ""]
    if cfg.get("checkpoint"):  # 训练后的检查点（DEV-0035）；base 的 README 不变
        ck = cfg["checkpoint"]
        lines[5:5] = [f"- 训练后的检查点：标签 {ck.get('label')}，manifest sha256 {ck.get('manifest_sha256')}，"
                      f"load_trained(merge=True)（合并核对 {ck.get('merge_check')}）"]
    for h in metrics["headline"]:
        lines.append(f"- {h['name']}（{h['scope']}）：{_f(h['value'])}，95% CI {_ci(h.get('ci'))}，n = {h['n']}")
    if "original_letter" in d:
        lines += ["", "## 对照：原字母读数（同一次运行的 s = 0，批组成同主评测）", ""]
        for h in d["original_letter"]["headline"]:
            lines.append(f"- {h['name']}（{h['scope']}）：{_f(h['value'])}，95% CI {_ci(h.get('ci'))}，n = {h['n']}")
        pdiff = d["paired_diff"]
        lines += [f"- 去偏 − 原字母：{_f(pdiff['value'])}，配对 95% CI {_ci(pdiff['ci'])}；答案变了 {pdiff['n_answer_changed']} 条"
                  f"（错→对 {pdiff['n_wrong_to_right']}，对→错 {pdiff['n_right_to_wrong']}）",
                  f"- 正确选项的平均概率：去偏 {_f(d['mean_p_gold']['debiased'])}，原字母 {_f(d['mean_p_gold']['original'])}",
                  "", "## 每种顺序单独的读数", "", "| 顺序 s | n | 准确率 | 95% CI | 最常见字母 | 占比 |", "|---|---|---|---|---|---|"]
        for e in d["per_order"]:
            lines.append(f"| {e['order']} | {e['n']} | {_f(e['accuracy']['value'])} | {_ci(e['accuracy']['ci'])} | "
                         f"{e['most_common_letter']} | {_f(e['most_common_letter_share'])} |")
    lines += ["", "## 文件", "",
              "- readings.jsonl：每次读数（样本 × 顺序）；predictions.jsonl：每条样本（由 readings.jsonl 算出）；metrics.json：全部指标；",
              "- compute.json：计算记账（每次读数记一次调用、生成 1 个 token）；config.yaml：运行身份；meta.json；logs/。", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 格式强制诊断（§10.2）
# ---------------------------------------------------------------------------


def forced_subset_name(variant: str) -> str:
    """输出目录用的子集名：forced_prefill、forced_restricted、forced_structured。"""
    return f"{FORCED_SUBSET}_{variant}"


def run_forced_task(adapter: Any, spec: RunSpec, sel: Selection, prompts: PromptConfig, posthoc: PosthocConfig,
                    paths: DataPaths, out_dir: str | Path, *, variant: str, resume: bool = False, argv: Sequence[str] = (),
                    log: Callable[[str], None] | None = None, warmup: bool = True) -> dict[str, Any]:
    """格式强制的一种读法（规则见 src/e1/forced.py 与 configs/e1_posthoc.json 的 format_forced）。返回 metrics。"""
    log = log or _log_default
    if spec.subset != FORCED_SUBSET:
        raise ValueError(f"格式强制的 subset 是 {FORCED_SUBSET}，收到 {spec.subset!r}")
    v = posthoc.forced["variants"][variant]
    if v["task"] != spec.task:
        raise ValueError(f"读法 {variant} 用于 {v['task']}，收到 {spec.task}")
    if spec.family not in posthoc.forced["families"]:
        raise ValueError(f"格式强制只做 {posthoc.forced['families']}（预注册 §10.2：Qwen 四个尺寸），{spec.model_key} 是 {spec.family}")
    base = posthoc.base_subset(spec.subset)
    if sel.subset != base:
        raise ValueError(f"样本取自 {sel.subset!r}，格式强制应取 {base!r} 的 id")
    enable_thinking = bool(posthoc.forced["enable_thinking"])
    fixed = bool(v["fixed_length"])
    fmt = grid_format_ids(adapter.tokenizer) if fixed else None
    if fixed:
        max_new = int(v["n_tokens"])  # 固定长度：不允许停止 token，恰好生成这么多
    elif v["max_new_tokens"] == "entry":
        max_new = prompts.max_new_tokens(spec.task, spec.family, "eval")  # 上限同 E1 主评测（不含预填）
    else:
        raise ValueError(f"读法 {variant} 的 max_new_tokens 只能是 entry，收到 {v['max_new_tokens']!r}")
    parser = prompts.parser(spec.task, spec.family)
    rendered = [prompts.render(spec.task, spec.family, s) for s in sel.samples]
    plan = batch_plan([None] * len(rendered), spec.batch_size)
    readout = {"kind": forced_subset_name(variant), "parse": parser}
    forced_info = {"variant": variant, "fixed_length": fixed, "answer_prefix": v.get("answer_prefix"),
                   "allowed_texts": v.get("allowed"), "format_ids": fmt, "n_tokens": v.get("n_tokens"),
                   "newline_period": v.get("newline_period"), "stop_tokens_allowed": v.get("stop_tokens_allowed"),
                   "rule": v["rule"], "fixed_length_rule": posthoc.forced["fixed_length_rule"] if fixed else None}
    generation = {"enable_thinking": enable_thinking, "max_new_tokens": max_new, "decoding": prompts.decoding(),
                  "order": prompts.order, "readout": readout, "forced": forced_info}
    info = _posthoc_record(posthoc, "format_forced", base, version=posthoc.forced["version"], variant=variant)
    cfg = _base_config(spec, sel, prompts, paths, readout=readout, generation=generation, posthoc_info=info)
    leaf = leaf_dir(out_dir, spec.model_key, spec.task, forced_subset_name(variant), spec.policy_tag)
    _prepare_leaf(leaf, cfg, [prompts.path, posthoc.path, *paths.config_files()], resume=resume)
    what = f"{spec.model_key} {spec.task}/{forced_subset_name(variant)}"
    ids = [s.id for s in sel.samples]
    keep, trimmed = _resume_point(leaf / PREDICTIONS, ids, lambda r: r["id"], plan, log, what)
    todo = [(a, b) for a, b in plan if a >= keep]
    run_k, run_path, run_log = _start_run_log(leaf, argv, n_done_before=keep, n_remaining=len(ids) - keep,
                                              batch_size=spec.batch_size, trimmed_to_batch_boundary=trimmed)
    t_all = time.perf_counter()
    if todo and warmup:
        run_log["warmup_wall_s"] = adapter.warmup(load_images(sel.samples[keep]), batch_size=min(spec.batch_size, len(ids) - keep))
    n_batches = 0
    for a, b in todo:
        batch, ren = sel.samples[a:b], rendered[a:b]
        t0 = time.perf_counter()
        imgs = [load_images(s) for s in batch]
        inputs = [adapter.build_inputs(im, r.text, prompts.order) for im, r in zip(imgs, ren)]
        build_s = time.perf_counter() - t0
        kw: dict[str, Any] = dict(prompts.decoding())
        if variant == "prefill":
            kw["answer_prefix"] = v["answer_prefix"]
        else:
            kw["allowed_token_ids"] = list(fmt["ids"])
            if variant == "structured":
                kw["logits_processor"] = [GridStructureProcessor(fmt["digit_ids"], fmt["newline_id"], period=v["newline_period"])]
        results = adapter.generate_batch(inputs, enable_thinking=enable_thinking, max_new_tokens=max_new, **kw)
        if len(results) != len(batch):
            raise RuntimeError("generate_batch 返回的行数与输入不同")
        lines, tlines = [], []
        for s, r, im, res in zip(batch, ren, imgs, results):
            if fixed:
                check_fixed_output(res, variant=variant, fmt=fmt, n_tokens=max_new, where=f"{what} {s.id}")
            elif tuple(res.answer_prefix_ids) == ():
                raise RuntimeError(f"{what} {s.id}：预填读法的结果没有 answer_prefix_ids（适配器没有预填）")
            size = im[0].size
            rec = _common_record(s, r, size, res, readout["kind"])
            if fixed:
                rec["truncated"] = False  # 固定长度是设计如此，不是截断（fixed_length_rule）
            rec["forced"] = {"variant": variant, "fixed_length": fixed, "answer_prefix": v.get("answer_prefix"),
                             "answer_prefix_ids": list(res.answer_prefix_ids)}
            rec.update(score_generation(spec.task, parser, s, parse_view(res, fixed_length=fixed), adapter=adapter,
                                        image_size=size, extras=r.extras))
            lines.append(_jsonl_line(rec))
            tlines.append(_jsonl_line({"id": s.id, "run": run_k, "batch_index": n_batches, "batch_rows": len(batch),
                                       "wall_time_s": res.wall_time_s, "batch_wall_time_s": res.batch_wall_time_s,
                                       "build_inputs_s": build_s / len(batch)}))
        _append(leaf / TIMING, tlines)
        _append(leaf / PREDICTIONS, lines)
        n_batches += 1
        if n_batches == 1 or n_batches % 20 == 0 or b == len(ids):
            log(f"[{what}] {b}/{len(ids)}")
    metrics = finalize_forced(leaf, spec, sel, cfg, prompts)
    _finish_run_log(run_path, run_log, n_batches=n_batches, t_all=t_all)
    return metrics


def finalize_forced(leaf: Path, spec: RunSpec, sel: Selection, cfg: Mapping[str, Any], prompts: PromptConfig) -> dict[str, Any]:
    """由 predictions.jsonl 与 logs/timing.jsonl 写 metrics.json、compute.json、README.md（同 runner.finalize，另加 posthoc 一节）。"""
    recs = read_jsonl_tolerant(leaf / PREDICTIONS, repair=False)
    ids = [s.id for s in sel.samples]
    if [r["id"] for r in recs] != ids[: len(recs)]:
        raise RuntimeError(f"{leaf / PREDICTIONS} 的 id 不是样本顺序的前缀")
    samples = sel.samples[: len(recs)]
    body = aggregate(spec.task, samples, recs, has_gt=sel.has_gt, enable_thinking=bool(cfg["generation"]["enable_thinking"]))
    gt_diag = grid_token_diagnostic(recs)
    if gt_diag is not None:
        body["diagnostics"]["grid_tokens_in_output"] = gt_diag
    gen = cfg["generation"]
    body["forced"] = {"variant": gen["forced"]["variant"], "fixed_length": gen["forced"]["fixed_length"],
                      "n_fixed_length": sum(1 for r in recs if r["forced"]["fixed_length"]),
                      "n_stopped": sum(1 for r in recs if r["stopped"]),
                      "answer_prefix": gen["forced"]["answer_prefix"]}
    body["complete"] = len(recs) == len(ids)
    body["n_expected"] = len(ids)
    metrics = _metrics_doc(spec, cfg, body, prompts)
    metrics["posthoc"] = cfg["posthoc"]
    (leaf / METRICS).write_text(json.dumps(metrics, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    timing: dict[str, dict[str, Any]] = {}
    for t in read_jsonl_tolerant(leaf / TIMING, repair=False):
        timing[t["id"]] = t
    comp = []
    for r in recs:
        wall = float(timing.get(r["id"], {}).get("wall_time_s", 0.0))
        comp.append(Compute.call(generated_tokens=r["n_generated_tokens"], prompt_tokens=r["n_prompt_tokens"],
                                 visual_tokens=r["n_visual_tokens"], image_pixels=r["n_image_pixels"], call_time_s=wall,
                                 new_round=True).as_sample(wall_time_s=wall))
    write_compute_json(leaf / COMPUTE, merge(comp))
    head = [f"> 事后诊断（D66-2；E1 预注册 §10.2，写于运行之前）：格式强制，读法 {gen['forced']['variant']}"
            f"（{'固定长度 ' + str(gen['max_new_tokens']) + ' 个 token，不判截断' if gen['forced']['fixed_length'] else '预填 ' + repr(gen['forced']['answer_prefix'])}）；"
            f"样本取 E1 的 {cfg['posthoc']['base_subset']} 子集的 id，关思考。规则见 configs/e1_posthoc.json 的 format_forced。", ""]
    (leaf / README).write_text("\n".join(head) + "\n" + readme_text(spec, cfg, metrics), encoding="utf-8")
    return metrics


# ---------------------------------------------------------------------------
# 对调标记的去偏读数（DEV-0036；D72-2）
# ---------------------------------------------------------------------------


def _view_images(sample: Any, view: int) -> tuple[list[Any], str]:
    """第 view 张图（0 = original：主评测的渲染图；1 = labels_swapped：运行时从原图渲染，先核对与 original 只差标签框里的字形）。
    返回 (送进模型的图, 这张图的像素 sha256)。"""
    import numpy as np
    from PIL import Image

    from src.data.nyu_pairs import pixels_sha256

    orig = load_images(sample)
    if len(orig) != 1:
        raise LS.LetterSwapError(f"{sample.id}：NYU 点对应只有一张图，收到 {len(orig)} 张")
    o = np.asarray(orig[0])
    if view == 0:
        return orig, pixels_sha256(o)
    sw = LS.swapped_image(sample)
    LS.check_swapped(o, sw, sample)
    return [Image.fromarray(sw)], pixels_sha256(sw)


def run_letter_swap_task(adapter: Any, spec: RunSpec, sel: Selection, prompts: PromptConfig, posthoc: PosthocConfig,
                         paths: DataPaths, out_dir: str | Path, *, resume: bool = False, argv: Sequence[str] = (),
                         log: Callable[[str], None] | None = None, warmup: bool = True,
                         run_extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """对调标记的去偏读数（规则见 src/e1/letter_swap.py 与 configs/e1_posthoc.json 的 letter_swap）。返回 metrics。
    批、续跑、落盘与 run_debiased_task 相同（order_major，分批键 = (图, 字母集合)）；检查点的用法同 run_debiased_task。"""
    log = log or _log_default
    ls = posthoc.letter_swap
    if spec.subset not in LETTER_SWAP_SUBSETS or spec.task not in posthoc.tasks_for(spec.subset):
        raise ValueError(f"对调标记的去偏读数只用于 {ls['tasks']} 的 {LETTER_SWAP_SUBSETS}，收到 {spec.task}/{spec.subset}")
    base = posthoc.base_subset(spec.subset)
    if sel.subset != base:
        raise ValueError(f"样本取自 {sel.subset!r}，{spec.subset} 应取 {base!r}")
    enable_thinking = prompts.enable_thinking(base)
    rendered = [prompts.render(spec.task, spec.family, s) for s in sel.samples]
    views = [LS.view_prompts(r) for r in rendered]
    letters = [tuple(r.extras["letters"]) for r in rendered]
    union = sorted({lt for ls_ in letters for lt in ls_})
    table = build_letter_table(adapter.tokenizer, union, prompts.letter_spellings()) if union else None
    rows, keys = reading_rows([len(v) for v in views], letters)
    plan = batch_plan(keys, spec.batch_size)
    readout = {"kind": LETTER_SWAP_READOUT, "parse": "letter_token", "spellings": list(prompts.letter_spellings()),
               "table": table.as_dict() if table is not None else None, "views": list(LETTER_SWAP_VIEWS),
               "method": "option_logits"}
    generation = {"enable_thinking": enable_thinking, "max_new_tokens": 1,
                  "decoding": "option_logits：一次前向，取第一个生成位置的原始 logits（不采样；D37）", "order": prompts.order,
                  "readout": readout}
    info = _posthoc_record(posthoc, "letter_swap", base, version=ls["version"], layout=ls["batch_layout"]["layout"],
                           n_readings=len(rows), views=list(LETTER_SWAP_VIEWS),
                           render="src/data/nyu_pairs.render_pair(raw_image, A, B, swap_labels=True)")
    cfg = _base_config(spec, sel, prompts, paths, readout=readout, generation=generation, posthoc_info=info)
    leaf = leaf_dir(out_dir, spec.model_key, spec.task, spec.subset, spec.policy_tag)
    _prepare_leaf(leaf, cfg, [prompts.path, posthoc.path, *paths.config_files()], resume=resume)
    what = f"{spec.model_key} {spec.task}/{spec.subset}"
    expected = [(sel.samples[j].id, v) for j, v in rows]
    keep, trimmed = _resume_point(leaf / READINGS, expected, lambda r: (r["id"], int(r["order"])), plan, log, what)
    todo = [(a, b) for a, b in plan if a >= keep]
    run_k, run_path, run_log = _start_run_log(leaf, argv, n_done_before=keep, n_remaining=len(rows) - keep,
                                              batch_size=spec.batch_size, trimmed_to_batch_boundary=trimmed,
                                              **dict(run_extra or {}))
    t_all = time.perf_counter()
    if todo and warmup:
        run_log["warmup_wall_s"] = adapter.warmup(load_images(sel.samples[rows[keep][0]]),
                                                  batch_size=min(spec.batch_size, len(rows) - keep))
    n_batches = 0
    for a, b in todo:
        batch = rows[a:b]
        lts = letters[batch[0][0]]
        if any(letters[j] != lts or v != batch[0][1] for j, v in batch):
            raise RuntimeError("同一批的图或字母集合不同（batch_plan 的分批键有误）")
        t0 = time.perf_counter()
        got = [_view_images(sel.samples[j], v) for j, v in batch]
        imgs = [g[0] for g in got]
        inputs = [adapter.build_inputs(im, views[j][v].text, prompts.order) for im, (j, v) in zip(imgs, batch)]
        build_s = time.perf_counter() - t0
        scores = adapter.option_logits(inputs, [list(lts)] * len(batch), enable_thinking=enable_thinking)
        if len(scores) != len(batch):
            raise RuntimeError("option_logits 返回的行数与输入不同")
        check_spellings(scores, table, lts, what)
        lines, tlines = [], []
        for (j, v), im, (_, sha), sc in zip(batch, imgs, got, scores):
            rec = reading_record(sel.samples[j], views[j][v], lts, sc)
            rec["view"] = LETTER_SWAP_VIEWS[v]
            rec["image_sha256"] = sha
            rec["image_size"] = list(im[0].size)
            lines.append(_jsonl_line(rec))
            tlines.append(_jsonl_line({"id": sel.samples[j].id, "order": v, "run": run_k, "batch_index": n_batches,
                                       "batch_rows": len(batch), "wall_time_s": sc.wall_time_s,
                                       "batch_wall_time_s": sc.batch_wall_time_s, "build_inputs_s": build_s / len(batch)}))
        _append(leaf / TIMING, tlines)
        _append(leaf / READINGS, lines)
        n_batches += 1
        if n_batches == 1 or n_batches % 20 == 0 or b == len(rows):
            log(f"[{what}] {b}/{len(rows)} 次读数")
    metrics = finalize_letter_swap(leaf, spec, sel, cfg, prompts, rendered, rows)
    _finish_run_log(run_path, run_log, n_batches=n_batches, t_all=t_all)
    return metrics


def finalize_letter_swap(leaf: Path, spec: RunSpec, sel: Selection, cfg: Mapping[str, Any], prompts: PromptConfig,
                         rendered: Sequence[Any], rows: Sequence[tuple[int, int]]) -> dict[str, Any]:
    """由 readings.jsonl 与 logs/timing.jsonl 写 predictions.jsonl、metrics.json、compute.json、README.md（同 finalize_debiased）。"""
    readings = read_jsonl_tolerant(leaf / READINGS, repair=False)
    expected = [(sel.samples[j].id, v) for j, v in rows]
    if [(r["id"], int(r["order"])) for r in readings] != expected:
        raise RuntimeError(f"{leaf / READINGS} 的读数不全或顺序不对（{len(readings)}/{len(expected)}）")
    by_id: dict[str, list[dict[str, Any]]] = {}
    for r in readings:
        by_id.setdefault(r["id"], []).append(r)
    recs = []
    for s, ren in zip(sel.samples, rendered):
        rs = sorted(by_id[s.id], key=lambda r: int(r["order"]))
        recs.append(LS.combine_sample(spec.task, s, ren, rs[0]["image_size"], rs))
    (leaf / PREDICTIONS).write_text("".join(_jsonl_line(r) for r in recs), encoding="utf-8")
    body = LS.letter_swap_metrics(spec.task, sel.samples, recs, has_gt=sel.has_gt)
    body["complete"] = True
    body["n_expected"] = len(sel.samples)
    metrics = _metrics_doc(spec, cfg, body, prompts)
    metrics["posthoc"] = cfg["posthoc"]
    (leaf / METRICS).write_text(json.dumps(metrics, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    timing: dict[tuple[str, int], dict[str, Any]] = {}
    for t in read_jsonl_tolerant(leaf / TIMING, repair=False):
        timing[(t["id"], int(t["order"]))] = t
    comp = []
    for s in sel.samples:
        calls, wall = [], 0.0
        for r in sorted(by_id[s.id], key=lambda r: int(r["order"])):
            w = float(timing.get((s.id, int(r["order"])), {}).get("wall_time_s", 0.0))
            wall += w
            calls.append(Compute.call(generated_tokens=1, prompt_tokens=int(r["n_prompt_tokens"]),
                                      visual_tokens=int(r["n_visual_tokens"]), image_pixels=int(r["n_image_pixels"]),
                                      call_time_s=w, new_round=int(r["order"]) == 0))
        comp.append(merge(calls).as_sample(wall_time_s=wall))
    write_compute_json(leaf / COMPUTE, merge(comp))
    (leaf / README).write_text(letter_swap_readme(spec, cfg, metrics), encoding="utf-8")
    return metrics


def letter_swap_readme(spec: RunSpec, cfg: Mapping[str, Any], metrics: Mapping[str, Any]) -> str:
    d = metrics["letter_swap"]
    ph = cfg["posthoc"]
    lines = [f"# E1 事后读数 · 对调标记的去偏读数 · {spec.model_key} · {spec.task} · {spec.subset}", "",
             "事后修订（D72-2，N26 → 问题 1 选 (a)；DEV-0036）。规则见 configs/e1_posthoc.json 的 letter_swap：每题读两张图（original、"
             "labels_swapped：图上 A、B 两个标签的字形互换，位置与环不变；提示不变），字母概率按物理点取平均。", "",
             f"- 模型：{spec.repo}@{spec.revision}（{spec.family}）；图像口径：{cfg['image_policy']}",
             f"- 样本：E1 的 {ph['base_subset']} 子集，n = {metrics['n']}；读数 {ph['n_readings']} 次（两张图，order_major 分批，"
             f"批大小 {cfg['batch_protocol']['batch_size']}）",
             f"- 提示配置 sha256：{cfg['prompt']['config_sha256']}；事后配置 sha256：{ph['config_sha256']}",
             f"- 随机水平：{_f(d['random_level'])}", "",
             "## 标题：对调标记的去偏读数（严格读数）", ""]
    if cfg.get("checkpoint"):
        ck = cfg["checkpoint"]
        lines[5:5] = [f"- 训练后的检查点：标签 {ck.get('label')}，manifest sha256 {ck.get('manifest_sha256')}，"
                      f"load_trained(merge=True)（合并核对 {ck.get('merge_check')}）"]
    for h in metrics["headline"]:
        lines.append(f"- {h['name']}（{h['scope']}）：{_f(h['value'])}，95% CI {_ci(h.get('ci'))}，n = {h['n']}")
    if "original_letter" in d:
        lines += ["", "## 对照：原字母读数（同一次运行 original 那张图，批组成同主评测）", ""]
        for h in d["original_letter"]["headline"]:
            lines.append(f"- {h['name']}（{h['scope']}）：{_f(h['value'])}，95% CI {_ci(h.get('ci'))}，n = {h['n']}")
        pdiff = d["paired_diff"]
        lines += [f"- 对调去偏 − 原字母：{_f(pdiff['value'])}，配对 95% CI {_ci(pdiff['ci'])}；答案变了 {pdiff['n_answer_changed']} 条"
                  f"（错→对 {pdiff['n_wrong_to_right']}，对→错 {pdiff['n_right_to_wrong']}）",
                  "", "## 两张图各自的读数", "", "| 图 | n | 准确率 | 95% CI | 最常见字母 | 占比 |", "|---|---|---|---|---|---|"]
        for e in d["per_view"]:
            lines.append(f"| {e['view']} | {e['n']} | {_f(e['accuracy']['value'])} | {_ci(e['accuracy']['ci'])} | "
                         f"{e['most_common_letter']} | {_f(e['most_common_letter_share'])} |")
    lines += ["", "## 文件", "",
              "- readings.jsonl：每次读数（样本 × 图；另记图名与像素 sha256）；predictions.jsonl：每条样本；metrics.json：全部指标；",
              "- compute.json：计算记账（每次读数记一次调用、生成 1 个 token）；config.yaml：运行身份；meta.json；logs/。", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 结构强制的多边形（DEV-0036；D72-3）
# ---------------------------------------------------------------------------


def run_forced_poly_task(adapter: Any, spec: RunSpec, sel: Selection, prompts: PromptConfig, posthoc: PosthocConfig,
                         paths: DataPaths, out_dir: str | Path, *, variant: str, resume: bool = False,
                         argv: Sequence[str] = (), log: Callable[[str], None] | None = None,
                         warmup: bool = True, sample_subset: str | None = None,
                         leaf: str | Path | None = None) -> dict[str, Any]:
    """结构强制的多边形的一个变体（规则见 src/e1/poly_forced.py 与 configs/e1_posthoc.json 的 format_forced_poly）。返回 metrics。

    DEV-0051（W1 的对照，D103 §三）：sample_subset 给了时，样本可以取自 E1 思考子集以外的清单（W1 的划分，sel.subset 必须等于它），
    它代替 format_forced_poly.sample_subset 记进 config.yaml 的 posthoc.base_subset；leaf 给了就把输出写进这个目录（W1 的布局）。
    两者都不给时只认思考子集，路径与输出与原来逐字节相同（tests/test_w1_baselines.py 的快照核对）。"""
    log = log or _log_default
    fp = posthoc.forced_poly
    if spec.subset != FORCED_POLY_SUBSET:
        raise ValueError(f"结构强制的多边形的 subset 是 {FORCED_POLY_SUBSET}，收到 {spec.subset!r}")
    if spec.task != fp["task"]:
        raise ValueError(f"结构强制的多边形只用于 {fp['task']}，收到 {spec.task}")
    if variant not in fp["variants"]:
        raise ValueError(f"未知的变体 {variant!r}；可用 {list(fp['variants'])}")
    if spec.family not in fp["families"]:
        raise ValueError(f"结构强制的多边形只做 {fp['families']}（D72-3：Qwen 四个尺寸），{spec.model_key} 是 {spec.family}")
    base = posthoc.base_subset(spec.subset)
    if sample_subset is not None:
        if sample_subset == base or not str(sample_subset).strip():
            raise ValueError(f"sample_subset 只用于 E1 思考子集以外的清单，收到 {sample_subset!r}")
        base = sample_subset
    if sel.subset != base:
        raise ValueError(f"样本取自 {sel.subset!r}，结构强制的多边形应取 {base!r} 的 id")
    v = fp["variants"][variant]
    enable_thinking = bool(fp["enable_thinking"])
    if fp["max_new_tokens"] != "entry":
        raise ValueError(f"max_new_tokens 只能是 entry，收到 {fp['max_new_tokens']!r}")
    max_new = prompts.max_new_tokens(spec.task, spec.family, "eval")  # 上限同 E1 主评测
    fmt = poly_format_ids(adapter.tokenizer, fp["pieces"])
    stop_ids = [int(s) for s in adapter.stop_token_ids()]
    grammar = PolyGrammar(fmt, min_points=int(v["min_points"]), max_points=int(fp["max_points"]), stop_ids=stop_ids,
                          coord_max=int(fp["coord_range"][1]))
    if grammar.max_tokens() > max_new:
        raise ValueError(f"最长的生成 {grammar.max_tokens()} 个 token 超过上限 {max_new}：按本家族的切分，max_points 太大")
    nr = posthoc.near_rect_thresholds()
    parser = prompts.parser(spec.task, spec.family)
    rendered = [prompts.render(spec.task, spec.family, s) for s in sel.samples]
    plan = batch_plan([None] * len(rendered), spec.batch_size)
    readout = {"kind": variant, "parse": parser}
    forced_info = {"variant": variant, "grammar": grammar.as_dict(), "format_ids": fmt, "rule": v["rule"],
                   "grammar_rule": fp["grammar"], "near_rectangle": nr}
    generation = {"enable_thinking": enable_thinking, "max_new_tokens": max_new, "decoding": prompts.decoding(),
                  "order": prompts.order, "readout": readout, "forced_poly": forced_info}
    info = _posthoc_record(posthoc, "format_forced_poly", base, version=fp["version"], variant=variant)
    cfg = _base_config(spec, sel, prompts, paths, readout=readout, generation=generation, posthoc_info=info)
    leaf = Path(leaf) if leaf is not None else leaf_dir(out_dir, spec.model_key, spec.task, variant, spec.policy_tag)
    _prepare_leaf(leaf, cfg, [prompts.path, posthoc.path, *paths.config_files()], resume=resume)
    what = f"{spec.model_key} {spec.task}/{variant}"
    ids = [s.id for s in sel.samples]
    keep, trimmed = _resume_point(leaf / PREDICTIONS, ids, lambda r: r["id"], plan, log, what)
    todo = [(a, b) for a, b in plan if a >= keep]
    run_k, run_path, run_log = _start_run_log(leaf, argv, n_done_before=keep, n_remaining=len(ids) - keep,
                                              batch_size=spec.batch_size, trimmed_to_batch_boundary=trimmed)
    t_all = time.perf_counter()
    if todo and warmup:
        run_log["warmup_wall_s"] = adapter.warmup(load_images(sel.samples[keep]), batch_size=min(spec.batch_size, len(ids) - keep))
    n_batches = 0
    for a, b in todo:
        batch, ren = sel.samples[a:b], rendered[a:b]
        t0 = time.perf_counter()
        imgs = [load_images(s) for s in batch]
        inputs = [adapter.build_inputs(im, r.text, prompts.order) for im, r in zip(imgs, ren)]
        build_s = time.perf_counter() - t0
        proc = PolyStructureProcessor(grammar)  # 每批新建（一个实例只用于一次 generate）
        results = adapter.generate_batch(inputs, enable_thinking=enable_thinking, max_new_tokens=max_new,
                                         logits_processor=[proc], **dict(prompts.decoding()))
        if len(results) != len(batch):
            raise RuntimeError("generate_batch 返回的行数与输入不同")
        lines, tlines = [], []
        for s, r, im, res in zip(batch, ren, imgs, results):
            n_pts = check_poly_output(res, grammar, where=f"{what} {s.id}")
            size = im[0].size
            rec = _common_record(s, r, size, res, readout["kind"])
            rec.update(score_generation(spec.task, parser, s, res, adapter=adapter, image_size=size, extras=r.extras))
            nrect = near_rectangle(rec["parsed"]["polygons"] if rec["parsed"] else None, **nr)
            rec["forced_poly"] = {"variant": variant, "n_points": n_pts, "min_points": grammar.min_points,
                                  "max_points": grammar.max_points, "near_rect": nrect["near_rect"],
                                  "area_ratio": nrect["area_ratio"]}
            lines.append(_jsonl_line(rec))
            tlines.append(_jsonl_line({"id": s.id, "run": run_k, "batch_index": n_batches, "batch_rows": len(batch),
                                       "wall_time_s": res.wall_time_s, "batch_wall_time_s": res.batch_wall_time_s,
                                       "build_inputs_s": build_s / len(batch)}))
        _append(leaf / TIMING, tlines)
        _append(leaf / PREDICTIONS, lines)
        n_batches += 1
        if n_batches == 1 or n_batches % 20 == 0 or b == len(ids):
            log(f"[{what}] {b}/{len(ids)}")
    metrics = finalize_forced_poly(leaf, spec, sel, cfg, prompts)
    _finish_run_log(run_path, run_log, n_batches=n_batches, t_all=t_all)
    return metrics


def finalize_forced_poly(leaf: Path, spec: RunSpec, sel: Selection, cfg: Mapping[str, Any],
                         prompts: PromptConfig) -> dict[str, Any]:
    """由 predictions.jsonl 与 logs/timing.jsonl 写 metrics.json、compute.json、README.md（同 finalize_forced，另加 forced_poly 一节）。"""
    recs = read_jsonl_tolerant(leaf / PREDICTIONS, repair=False)
    ids = [s.id for s in sel.samples]
    if [r["id"] for r in recs] != ids[: len(recs)]:
        raise RuntimeError(f"{leaf / PREDICTIONS} 的 id 不是样本顺序的前缀")
    samples = sel.samples[: len(recs)]
    body = aggregate(spec.task, samples, recs, has_gt=sel.has_gt, enable_thinking=bool(cfg["generation"]["enable_thinking"]))
    fpg = cfg["generation"]["forced_poly"]
    body["forced_poly"] = {"variant": fpg["variant"], "min_points": fpg["grammar"]["min_points"],
                           "max_points": fpg["grammar"]["max_points"],
                           "n_points": distribution(r["forced_poly"]["n_points"] for r in recs),
                           "n_at_max_points": sum(1 for r in recs if r["forced_poly"]["n_points"] == fpg["grammar"]["max_points"]),
                           "n_stopped": sum(1 for r in recs if r["stopped"]),
                           "near_rectangle": {**fpg["near_rectangle"],
                                              **near_rect_summary([r["forced_poly"] for r in recs],
                                                                  [is_valid(spec.task, r) for r in recs])}}
    body["complete"] = len(recs) == len(ids)
    body["n_expected"] = len(ids)
    metrics = _metrics_doc(spec, cfg, body, prompts)
    metrics["posthoc"] = cfg["posthoc"]
    (leaf / METRICS).write_text(json.dumps(metrics, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    timing: dict[str, dict[str, Any]] = {}
    for t in read_jsonl_tolerant(leaf / TIMING, repair=False):
        timing[t["id"]] = t
    comp = []
    for r in recs:
        wall = float(timing.get(r["id"], {}).get("wall_time_s", 0.0))
        comp.append(Compute.call(generated_tokens=r["n_generated_tokens"], prompt_tokens=r["n_prompt_tokens"],
                                 visual_tokens=r["n_visual_tokens"], image_pixels=r["n_image_pixels"], call_time_s=wall,
                                 new_round=True).as_sample(wall_time_s=wall))
    write_compute_json(leaf / COMPUTE, merge(comp))
    fp = body["forced_poly"]
    nrs = fp["near_rectangle"]
    head = [f"> 事后诊断（D72-3；DEV-0036）：结构强制的多边形，变体 {fp['variant']}（至少 {fp['min_points']} 个点、至多 "
            f"{fp['max_points']} 个点，每个点恰好两个 0–1000 的整数，语法见 configs/e1_posthoc.json 的 format_forced_poly）；样本取 E1 的 "
            f"{cfg['posthoc']['base_subset']} 子集的 id，关思考。", "",
            f"- 点数分布：{fp['n_points']}；写满上限的 {fp['n_at_max_points']} 条",
            f"- 近矩形（恰好 {nrs['n_points']} 个点、面积 / 外接框面积 ≥ {nrs['min_area_ratio']}）：{nrs['n_near_rect']}/{nrs['n']}"
            f"（占全部 {_f(nrs['share_of_all'])}；占有效 {_f(nrs['share_of_valid'])}）", ""]
    (leaf / README).write_text("\n".join(head) + "\n" + readme_text(spec, cfg, metrics), encoding="utf-8")
    return metrics


__all__ = [
    "READINGS",
    "check_spellings",
    "debiased_readme",
    "finalize_debiased",
    "finalize_forced",
    "finalize_forced_poly",
    "finalize_letter_swap",
    "forced_subset_name",
    "letter_swap_readme",
    "run_debiased_task",
    "run_forced_poly_task",
    "run_forced_task",
    "run_letter_swap_task",
]
