"""E5 的运行器：一个系统 × 一个模型（或基线 / 外部系统）× 一个任务 × 一个子集 → 输出目录（DEV-0018；结构参照 src/e1/runner.py）。

输出 <out>/<系统>/<模型>/<任务>[__dev]/：
- predictions.jsonl：每条样本一行（id、档、分列维度、是否计入标题、提示原文、raw_text、text、解析失败原因、预测的网格 / 起伏 /
  多边形、逐条得分；模型系统另有停止原因、生成 token 数、视觉 token 数与像素数；预算强制收尾的系统（思维链，DEV-0028）另有
  forced_close、思考 / 作答 / 补入的 token 数与 answer_text）。不含墙钟，续跑后逐字节可复现。
- metrics.json：只由 predictions.jsonl 与生成器重新生成的样本算出（scoring.aggregate + 两级门槛 scoring.gate；门槛里的基线
  用同一个评分函数现算），续跑后再算一遍逐字节相同。
- compute.json：src/accounting.py；模型系统的墙钟来自 logs/timing.jsonl；基线与外部预测记 0（外部系统的算力不在这里）。
- config.yaml（运行身份：系统、模型、任务、子集、每档条数、样本 id 指纹、生成器的任务版本、提示配置 sha256 与模板原文、生成参数、
  批配置、外部预测目录的标记文件 sha256……；不含时间戳）与 meta.json（src/runlog.write_run_metadata）。
- README.md：由 metrics.json 生成。logs/：timing.jsonl、每次调用一份 run_<k>.json。

断点续传（--resume）：同 E1：目录非空时必须给 --resume，config.yaml 必须与本次完全相同（批大小也算：批配置是协议的一部分，
D36-1），末尾不完整的一行截掉，已完成的 id 必须是样本顺序的前缀，从下一条接着跑。
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml
from PIL import Image

from src.accounting import Compute, merge, write_compute_json
from src.constants import BOOTSTRAP_SEED
from src.e5 import external as EXT
from src.e5 import scoring as SC
from src.e5.prompts import PROMPTS_PATH, PromptConfig
from src.e5.protocol import (
    MODEL_SYSTEMS,
    SUBSETS,
    TASK_LETTER,
    EvalItem,
    eval_items,
    needs_canonical_platform,
    on_canonical_platform,
)
from src.runlog import write_run_metadata
from src.synth import generate
from src.synth.constants import SYNTH_GENERATOR_VERSION, TASK_GENERATOR_VERSIONS
from src.synth.relief import CONVEX_BASELINE, CONVEX_SOURCE_BASELINE, RELIEF_TASKS, convex_amplitude
from src.synth.sample import Sample
from src.synth.seeding import canonical_task

REPO_ROOT = Path(__file__).resolve().parents[2]
PREDICTIONS = "predictions.jsonl"
METRICS = "metrics.json"
COMPUTE = "compute.json"
CONFIG = "config.yaml"
README = "README.md"
TIMING = "logs/timing.jsonl"


@dataclass(frozen=True)
class RunSpec:
    system: str  # zeroshot / cot / baseline / external
    model_key: str  # 注册键、基线名或外部系统名
    task: str
    subset: str  # eval / dev
    n_per_bin: int
    n_resamples: int
    tiny: bool = False
    family: str | None = None
    repo: str | None = None
    revision: str | None = None
    adapter_model_key: str | None = None
    policy: Mapping[str, Any] | None = None
    batch_size: int | None = None
    max_new_tokens_override: int | None = None  # 只给 --tiny 彩排用
    external_dir: str | None = None
    limit: int | None = None  # 只取评测清单的前 N 条（彩排、调试）；记进配置


def spec_items(spec: RunSpec) -> list[EvalItem]:
    items = eval_items(spec.task, spec.subset, spec.n_per_bin)
    return items if spec.limit is None else items[: spec.limit]


def leaf_dir(out: str | Path, spec: RunSpec) -> Path:
    name = canonical_task(spec.task) + ("" if spec.subset == "eval" else f"__{spec.subset}")
    return Path(out) / spec.system / spec.model_key / name


def ids_digest(ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def _jsonl_line(rec: Mapping[str, Any]) -> str:
    return json.dumps(rec, ensure_ascii=False) + "\n"


def read_jsonl_tolerant(path: Path, *, repair: bool) -> list[dict[str, Any]]:
    """读 jsonl；末尾不完整的一行丢掉，repair=True 时把文件截到最后一个完整行（同 src/e1/runner.py）。"""
    if not path.exists():
        return []
    raw = path.read_bytes()
    keep = raw.rfind(b"\n") + 1
    recs = [json.loads(line) for line in raw[:keep].decode("utf-8").splitlines() if line.strip()]
    if repair and keep != len(raw):
        with open(path, "r+b") as f:
            f.truncate(keep)
    return recs


def _append(path: Path, lines: Sequence[str]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write("".join(lines))
        f.flush()
        os.fsync(f.fileno())


def make_sample(item: EvalItem) -> Sample:
    return generate(item.task, item.split, item.index, bin=item.bin)


def max_new_tokens(spec: RunSpec, prompts: PromptConfig) -> int:
    n = prompts.max_new_tokens(spec.system)
    return n if spec.max_new_tokens_override is None else int(spec.max_new_tokens_override)


def budgets(spec: RunSpec, prompts: PromptConfig) -> tuple[int, int]:
    """预算强制收尾的 (thinking_budget, answer_budget)；--tiny 的 max_new_tokens_override 同时截两段（只用于彩排）。"""
    th, an = prompts.budgets(spec.system)
    o = spec.max_new_tokens_override
    return (th, an) if o is None else (min(th, int(o)), min(an, int(o)))


def generation_config(spec: RunSpec, prompts: PromptConfig) -> dict[str, Any]:
    """运行身份里的生成参数：plain 记 max_new_tokens；budgeted 记两段预算（D59-5）。"""
    g: dict[str, Any] = {"enable_thinking": prompts.enable_thinking(spec.system), "mode": prompts.generation(spec.system)}
    if g["mode"] == "plain":
        g["max_new_tokens"] = max_new_tokens(spec, prompts)
    else:
        g["thinking_budget"], g["answer_budget"] = budgets(spec, prompts)
        g["allowed_token_ids"] = None
    g.update(max_new_tokens_override=spec.max_new_tokens_override, decoding=prompts.decoding(), order=prompts.order)
    return g


def convex_prior_config(task: str) -> dict[str, Any] | None:
    """凸起先验基线（D59-4）的冻结参数，记进运行身份；没有起伏的任务为 None。"""
    t = canonical_task(task)
    if t not in RELIEF_TASKS:
        return None
    return {"baseline": CONVEX_BASELINE, "source_baseline": CONVEX_SOURCE_BASELINE, "amplitude": convex_amplitude(t),
            "definition": "src/synth/relief.py convex_prior_codes（D59-4，DEV-0028）"}


def run_config(spec: RunSpec, items: Sequence[EvalItem], prompts: PromptConfig | None) -> dict[str, Any]:
    t = canonical_task(spec.task)
    ids = [it.id for it in items]
    cfg: dict[str, Any] = {
        "experiment": "E5",
        "framework": "src/e5（DEV-0018；DEV-0028）",
        "system": spec.system,
        "task": t,
        "task_letter": TASK_LETTER[t],
        "subset": spec.subset,
        "data": {"generator": "src.synth.generate", "split": SUBSETS[spec.subset], "n_per_bin": spec.n_per_bin,
                 "n_samples": len(ids), "sample_ids_sha256": ids_digest(ids),
                 "index_rule": "序号 0..B·n−1，档 = 序号 mod B（src/e5/protocol.py）",
                 "limit": spec.limit,
                 "task_generator_version": TASK_GENERATOR_VERSIONS[t], "synth_generator_version": SYNTH_GENERATOR_VERSION,
                 "platform": {"machine": platform.machine(), "system": platform.system()}},
        "n_resamples": spec.n_resamples,
        "tiny": spec.tiny,
        "gate": {"rule": "D59-4：配对 bootstrap 95% CI，判定条件同时胜过 " + "、".join(SC.JUDGMENT_BASELINES),
                 "convex_prior": convex_prior_config(t)},
    }
    if spec.system in MODEL_SYSTEMS:
        assert prompts is not None
        cfg["model"] = {"key": spec.model_key, "family": spec.family, "repo": spec.repo, "revision": spec.revision,
                        "adapter_model_key": spec.adapter_model_key}
        cfg["image_policy"] = dict(spec.policy or {})
        cfg["generation"] = generation_config(spec, prompts)
        path = "generate_batch" if prompts.generation(spec.system) == "plain" else "generate_batch_budgeted"
        cfg["batching"] = {"batch_size": spec.batch_size, "padding": prompts.data["batching"]["padding"],
                           "order": prompts.data["batching"]["order"], "path": path}
        cfg["prompt"] = {"config": os.path.relpath(prompts.path, REPO_ROOT) if Path(prompts.path).is_relative_to(REPO_ROOT)
                         else prompts.path, "config_sha256": prompts.sha256,
                         "format": prompts.format_of(spec.system, t), "template": prompts.template(spec.system, t)}
    elif spec.system == "baseline":
        cfg["baseline"] = spec.model_key
    elif spec.system == "external":
        root = Path(spec.external_dir)
        cfg["external"] = {"system": spec.model_key, "dir": str(root),
                           "marker_sha256": hashlib.sha256((root / EXT.EXTERNAL_MARKER).read_bytes()).hexdigest()}
    else:
        raise ValueError(f"未知系统 {spec.system!r}")
    return cfg


def _identity(cfg: Mapping[str, Any]) -> dict[str, Any]:
    return yaml.safe_load(yaml.safe_dump(dict(cfg), sort_keys=False, allow_unicode=True))


def _prepare(leaf: Path, cfg: Mapping[str, Any], resume: bool) -> None:
    if leaf.exists() and any(leaf.iterdir()):
        if not resume:
            raise FileExistsError(f"{leaf} 已存在且非空；续跑请加 --resume")
        if not (leaf / CONFIG).exists():
            raise FileExistsError(f"{leaf} 里没有 {CONFIG}，不能续跑")
        old = _identity(yaml.safe_load((leaf / CONFIG).read_text(encoding="utf-8")))
        new = _identity(cfg)
        if old != new:
            diff = sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k))
            raise ValueError(f"{leaf}：续跑的配置与已有的不同（{diff}），拒绝混写")
    else:
        leaf.mkdir(parents=True, exist_ok=True)
        (leaf / "logs").mkdir(exist_ok=True)
        files = [PROMPTS_PATH]
        write_run_metadata(leaf, config=cfg, seed=BOOTSTRAP_SEED, data_files=files, repo_root=REPO_ROOT)
    (leaf / "logs").mkdir(exist_ok=True)


def check_platform(task: str, tiny: bool) -> None:
    """d 的数据只在规范平台上生成（D24）：非规范平台上只允许 --tiny 彩排。运行与导出输入（run_e5.py --export-inputs）共用。"""
    if needs_canonical_platform(task) and not on_canonical_platform() and not tiny:
        raise RuntimeError(f"{task}：d 的数据只在规范平台上生成（D24）；本机 {platform.machine()} {platform.system()} "
                           "只允许 --tiny 彩排")


def _check_platform(spec: RunSpec) -> None:
    check_platform(spec.task, spec.tiny)


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------


def _fmt(v: Any) -> str:
    return "null" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))


def _ci_text(ci: Mapping[str, Any] | None) -> str:
    return f"[{_fmt(ci['low'])}, {_fmt(ci['high'])}]" if ci else "—"


def _block_lines(blk: Mapping[str, Any]) -> list[str]:
    out = []
    for key, label in (("relief_error", "起伏误差（越低越好）"), ("mask_iou_grid", "mask IoU（网格）"),
                       ("mask_iou_full", "mask IoU（全分辨率）"),
                       ("mask_iou_current_grid", "对当前 mask 的 IoU（网格，D59-5）"),
                       ("mask_iou_current_full", "对当前 mask 的 IoU（全分辨率，D59-5）")):
        b = blk.get(key)
        if b is None:
            continue
        out.append(f"- {label}：严格 {_fmt(b['strict'])}（95% CI {_ci_text(b.get('strict_ci'))}），有效率 {_fmt(b['valid_rate'])}，"
                   f"条件 {_fmt(b['conditional'])}（中位数 {_fmt(b['conditional_median'])}），n = {b['n']}")
    return out


def readme_text(spec: RunSpec, cfg: Mapping[str, Any], m: Mapping[str, Any]) -> str:
    lines = [f"# E5 · {spec.system} · {spec.model_key} · {m['task']}（{TASK_LETTER[m['task']]}）· {spec.subset}", "",
             f"- 样本：{cfg['data']['split']} 划分，每档 {spec.n_per_bin} 条，n = {m['n']}（id 指纹 "
             f"{cfg['data']['sample_ids_sha256'][:16]}…）；生成器任务版本 {cfg['data']['task_generator_version']}",
             f"- 标题口径：{m['headline_scope']}"]
    if "generation" in cfg:
        g = cfg["generation"]
        if g["mode"] == "plain":
            lines.append(f"- 生成：enable_thinking={g['enable_thinking']}，max_new_tokens={g['max_new_tokens']}，greedy，"
                         f"generate_batch，批大小 {cfg['batching']['batch_size']}")
        else:
            lines.append(f"- 生成：预算强制收尾（D59-5），enable_thinking={g['enable_thinking']}，thinking_budget={g['thinking_budget']}，"
                         f"answer_budget={g['answer_budget']}，greedy，generate_batch_budgeted，批大小 {cfg['batching']['batch_size']}；"
                         "解析只看作答段")
        lines.append(f"- 提示配置 sha256：{cfg['prompt']['config_sha256']}；格式 {cfg['prompt']['format']}")
    cp = (cfg.get("gate") or {}).get("convex_prior")
    if cp:
        lines.append(f"- 凸起先验基线 {cp['baseline']}：A = {cp['amplitude']} 档（val 上选定后冻结，{cp['definition']}）")
    lines += ["", "## 标题（严格 / 有效率 / 条件读数）", ""]
    lines += _block_lines(m["headline"])
    bud = m.get("budget")
    if bud:
        lines += ["", "## 预算强制收尾：自然收尾与强制收尾分开（D59-5）", "",
                  f"- 自然收尾 {bud['n_natural']} 条，强制收尾 {bud['n_forced']} 条（强制率 {_fmt(bud['forced_rate'])}）"]
        for name, label in (("natural", "自然收尾"), ("forced", "强制收尾")):
            sub = bud[name]
            lines.append(f"- {label}（标题口径 {sub['n_headline']} 条）：")
            lines += ["  " + x for x in _block_lines(sub["headline"])]
    g = m.get("gate") or {}
    pre, jud = g.get("prerequisite"), g.get("judgment")
    lines += ["", "## 两级门槛（D59-4：逐样本配对 bootstrap 95% CI）", ""]
    if pre:
        lines.append(f"- 先决条件（{g['iou_key']}）：系统 {_fmt(pre['system'])}（中位数 {_fmt(pre['system_median'])}）vs 最强非真值基线 "
                     f"{pre['strongest']} {_fmt(pre['value'])}（中位数 {_fmt(pre['median'])}）；差 {_fmt(pre['diff'])}，"
                     f"CI {_ci_text(pre['diff_ci'])}，逐样本差的中位数 {_fmt(pre['diff_median'])} → pass={pre['pass']}")
    if jud:
        lines.append(f"- 判定条件：{jud['status']}" + (f"，系统起伏误差 {_fmt(jud.get('system'))}（中位数 "
                     f"{_fmt(jud.get('system_median'))}），n = {jud.get('n')} → pass={jud.get('pass')}" if jud["status"] == "evaluated" else ""))
        for name, c in (jud.get("baselines") or {}).items():
            lines.append(f"  - vs {name}：基线 {_fmt(c['value'])}（中位数 {_fmt(c['median'])}）；差 {_fmt(c['diff'])}，"
                         f"CI {_ci_text(c['diff_ci'])}，逐样本差的中位数 {_fmt(c['diff_median'])}，系统更低的比例 "
                         f"{_fmt(c['system_better_rate'])} → 胜过={c['beats']}")
    lines.append(f"- imagining：{g.get('imagining')}")
    lines.append("- 注：D28-13 照字面的判法（只比真值仿射、按点估计判）与 DEV-0018 的 pass_vs_flat 已作废（D59-4），不再输出。")
    p = m["parse"]
    lines += ["", "## 解析", "", f"- 解析失败率：{_fmt(p['failure_rate'])}（{p['n_failed']}/{p['n']}）；分原因：{p['failures']}"]
    if "truncation" in m:
        lines.append(f"- 截断率：{_fmt(m['truncation']['rate'])}；思考段未闭合：{_fmt(m['thinking_unclosed']['rate'])}；"
                     f"拒答（诊断）：{m['refusal']['n']} 条")
    lines += ["", "## 文件", "", "- predictions.jsonl、metrics.json（含每档、分列与门槛）、compute.json、config.yaml、meta.json、logs/", ""]
    return "\n".join(lines)


def finalize(leaf: Path, spec: RunSpec, items: Sequence[EvalItem], cfg: Mapping[str, Any]) -> dict[str, Any]:
    recs = read_jsonl_tolerant(leaf / PREDICTIONS, repair=False)
    ids = [it.id for it in items]
    if [r["id"] for r in recs] != ids[: len(recs)]:
        raise RuntimeError(f"{leaf / PREDICTIONS} 的 id 不是样本顺序的前缀")
    samples = [make_sample(it) for it in items[: len(recs)]]
    enable_thinking = cfg.get("generation", {}).get("enable_thinking")
    body = SC.aggregate(spec.task, recs, n_resamples=spec.n_resamples, enable_thinking=enable_thinking)
    base_recs, roles = SC.baseline_records(samples)
    body["gate"] = SC.gate(spec.task, recs, base_recs, roles, n_resamples=spec.n_resamples)
    body["baselines_headline"] = {n: SC.blocks([r for r in v if r["headline"]], 0) for n, v in sorted(base_recs.items())}
    body["baseline_roles"] = dict(sorted(roles.items()))
    body["complete"] = len(recs) == len(ids)
    body["n_expected"] = len(ids)
    metrics = {"system": spec.system, "model": spec.model_key, "subset": spec.subset, **body}
    (leaf / METRICS).write_text(json.dumps(metrics, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    timing: dict[str, dict[str, Any]] = {t["id"]: t for t in read_jsonl_tolerant(leaf / TIMING, repair=False)}
    comp = []
    for r in recs:
        wall = float(timing.get(r["id"], {}).get("wall_time_s", 0.0))
        if "n_generated_tokens" in r:
            c = Compute.call(generated_tokens=r["n_generated_tokens"], prompt_tokens=r["n_prompt_tokens"],
                             visual_tokens=r["n_visual_tokens"], image_pixels=r["n_image_pixels"], call_time_s=wall,
                             new_round=True)
        else:
            c = Compute()
        comp.append(c.as_sample(wall_time_s=wall))
    write_compute_json(leaf / COMPUTE, merge(comp))
    (leaf / README).write_text(readme_text(spec, cfg, metrics), encoding="utf-8")
    return metrics


# ---------------------------------------------------------------------------
# 运行
# ---------------------------------------------------------------------------


def _run_log_start(leaf: Path, argv: Sequence[str], n_done: int, n_remaining: int, batch_size: int | None) -> tuple[Path, dict]:
    run_k = len(list((leaf / "logs").glob("run_*.json"))) + 1
    path = leaf / "logs" / f"run_{run_k:03d}.json"
    log = {"argv": list(argv), "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "n_done_before": n_done, "n_remaining": n_remaining, "batch_size": batch_size, "finished_utc": None, "run": run_k}
    path.write_text(json.dumps(log, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return path, log


def _run_log_finish(path: Path, log: dict, t0: float, **extra: Any) -> None:
    log.update(finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
               wall_s=round(time.perf_counter() - t0, 3), **extra)
    path.write_text(json.dumps(log, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


BUDGET_FIELDS = ("forced_close", "n_thinking_tokens", "n_answer_tokens", "n_inserted_tokens", "answer_text")


def check_budget_fields(res: Any) -> None:
    """generate_batch_budgeted 的结果必须带 DEV-0027 固定的五个新字段（解析之前就查，缺了直接报错）。"""
    missing = [f for f in BUDGET_FIELDS if not hasattr(res, f)]
    if missing:
        raise RuntimeError(f"generate_batch_budgeted 的结果缺字段 {missing}（签名见 DEV-0027）")


def model_record(sample: Sample, prompt: str, res: Any, pred: SC.Pred, *, relief_expected: bool,
                 budgeted: bool = False) -> dict[str, Any]:
    if res.stopped is None or res.n_image_pixels is None:
        raise RuntimeError("适配器没有填 stopped / n_image_pixels（D29：三个适配器一律填写）")
    rec = SC.sample_fields(sample)
    rec.update({"prompt": prompt, "raw_text": res.raw_text, "text": res.text, "stopped": res.stopped,
                "truncated": res.stopped is False, "thinking_unclosed": res.thinking_unclosed,
                "n_generated_tokens": res.n_generated_tokens, "n_prompt_tokens": res.n_prompt_tokens,
                "n_visual_tokens": res.n_visual_tokens, "n_image_pixels": res.n_image_pixels})
    if budgeted:
        check_budget_fields(res)
        rec.update({f: getattr(res, f) for f in BUDGET_FIELDS})
        rec["forced_close"] = bool(rec["forced_close"])
        # 预算强制收尾：思考段由适配器闭合，"截断"只指作答段写满 answer_budget；思考段未闭合不会出现
        rec["thinking_unclosed"] = False
    rec.update({"valid": pred.valid, "failure": pred.failure, "pred": SC.pred_record(pred, store_rle=pred.polygons is not None),
                "score": SC.score(sample, pred, relief_expected=relief_expected)})
    return rec


def run_model_task(adapter: Any, spec: RunSpec, prompts: PromptConfig, out_dir: str | Path, *, resume: bool = False,
                   argv: Sequence[str] = (), log: Callable[[str], None] | None = None, warmup: bool = True) -> dict[str, Any]:
    """零样本或思维链：逐批生成 → 解析 → 评分 → 落盘。零样本走 generate_batch；思维链走 generate_batch_budgeted
    （预算强制收尾，D59-5；签名由 DEV-0027 固定），解析只看作答段 answer_text。"""
    log = log or (lambda m: print(m, file=sys.stderr, flush=True))
    if spec.system not in MODEL_SYSTEMS:
        raise ValueError(f"{spec.system} 不是模型系统")
    if isinstance(spec.batch_size, bool) or not isinstance(spec.batch_size, int) or spec.batch_size < 1:
        raise ValueError(f"batch_size 必须是正整数，收到 {spec.batch_size!r}")
    _check_platform(spec)
    t = canonical_task(spec.task)
    items = spec_items(spec)
    leaf = leaf_dir(out_dir, spec)
    cfg = run_config(spec, items, prompts)
    _prepare(leaf, cfg, resume)
    pred_path = leaf / PREDICTIONS
    done = read_jsonl_tolerant(pred_path, repair=True)
    ids = [it.id for it in items]
    if [r["id"] for r in done] != ids[: len(done)]:
        raise RuntimeError(f"{pred_path} 里已完成的 id 不是样本顺序的前缀，不能续跑")
    remaining = items[len(done):]
    budgeted = prompts.generation(spec.system) == "budgeted"
    if budgeted and not callable(getattr(adapter, "generate_batch_budgeted", None)):
        raise RuntimeError(f"{spec.system} 用预算强制收尾（D59-5），适配器 {type(adapter).__name__} 没有 generate_batch_budgeted"
                           "（接口由 DEV-0027 实现）")
    run_path, run_log = _run_log_start(leaf, argv, len(done), len(remaining), spec.batch_size)
    enable_thinking = prompts.enable_thinking(spec.system)
    kind = prompts.kind_of(spec.system, t)
    relief_expected = kind == "relief_grid"
    if budgeted:
        thinking_budget, answer_budget = budgets(spec, prompts)

        def generate_rows(inputs):
            return adapter.generate_batch_budgeted(inputs, thinking_budget=thinking_budget, answer_budget=answer_budget,
                                                   allowed_token_ids=None)
    else:
        max_new = max_new_tokens(spec, prompts)

        def generate_rows(inputs):
            return adapter.generate_batch(inputs, enable_thinking=enable_thinking, max_new_tokens=max_new, **prompts.decoding())
    t_all = time.perf_counter()
    if remaining and warmup:
        s0 = make_sample(remaining[0])
        imgs = [] if s0.image is None else [Image.fromarray(s0.image)]
        run_log["warmup_wall_s"] = adapter.warmup(imgs, batch_size=min(spec.batch_size, len(remaining)))
    n_batches = 0
    for start in range(0, len(remaining), spec.batch_size):
        batch = [make_sample(it) for it in remaining[start: start + spec.batch_size]]
        t0 = time.perf_counter()
        texts = [prompts.render(spec.system, t, s.instruction, s.meta["target_description"]) for s in batch]
        inputs = [adapter.build_inputs([] if s.image is None else [Image.fromarray(s.image)], txt, prompts.order)
                  for s, txt in zip(batch, texts)]
        build_s = time.perf_counter() - t0
        results = generate_rows(inputs)
        if len(results) != len(batch):
            raise RuntimeError("generate_batch 返回的行数与输入不同")
        lines, tlines = [], []
        for s, txt, res in zip(batch, texts, results):
            h, w = s.target_mask.shape
            if budgeted:
                check_budget_fields(res)
            pred = SC.parse_answer(kind, SC.budget_view(res) if budgeted else res, h, w)
            rec = model_record(s, txt, res, pred, relief_expected=relief_expected, budgeted=budgeted)
            lines.append(_jsonl_line(rec))
            tlines.append(_jsonl_line({"id": rec["id"], "run": run_log["run"], "batch_index": n_batches, "batch_rows": len(batch),
                                       "wall_time_s": res.wall_time_s, "batch_wall_time_s": res.batch_wall_time_s,
                                       "build_inputs_s": build_s / len(batch)}))
        _append(leaf / TIMING, tlines)
        _append(pred_path, lines)
        n_batches += 1
        n_now = len(done) + start + len(batch)
        if n_batches == 1 or n_batches % 20 == 0 or n_now == len(ids):
            log(f"[E5 {spec.system} {spec.model_key} {t}/{spec.subset}] {n_now}/{len(ids)}")
    metrics = finalize(leaf, spec, items, cfg)
    _run_log_finish(run_path, run_log, t_all, n_batches=n_batches)
    return metrics


def _run_offline(spec: RunSpec, out_dir: str | Path, make_records: Callable[[Sequence[EvalItem]], list[dict[str, Any]]], *,
                 resume: bool, argv: Sequence[str]) -> dict[str, Any]:
    _check_platform(spec)
    items = spec_items(spec)
    leaf = leaf_dir(out_dir, spec)
    cfg = run_config(spec, items, None)
    _prepare(leaf, cfg, resume)
    done = read_jsonl_tolerant(leaf / PREDICTIONS, repair=True)
    if [r["id"] for r in done] != [it.id for it in items][: len(done)]:
        raise RuntimeError(f"{leaf / PREDICTIONS} 里已完成的 id 不是样本顺序的前缀，不能续跑")
    run_path, run_log = _run_log_start(leaf, argv, len(done), len(items) - len(done), None)
    t0 = time.perf_counter()
    recs = make_records(items[len(done):])
    _append(leaf / PREDICTIONS, [_jsonl_line(r) for r in recs])
    metrics = finalize(leaf, spec, items, cfg)
    _run_log_finish(run_path, run_log, t0)
    return metrics


def run_baseline_task(spec: RunSpec, out_dir: str | Path, *, resume: bool = False, argv: Sequence[str] = ()) -> dict[str, Any]:
    """平凡基线（或参照线）：生成器给出的预测，直接评分。spec.model_key = 基线名。"""
    if spec.system != "baseline":
        raise ValueError("run_baseline_task 只跑 system=baseline")

    def make(items: Sequence[EvalItem]) -> list[dict[str, Any]]:
        out = []
        for it in items:
            s = make_sample(it)
            if spec.model_key not in SC.baseline_names(s):
                raise KeyError(f"{it.task} 没有基线 {spec.model_key!r}；有 {SC.baseline_names(s)}")
            p = SC.baseline_pred(s, spec.model_key)
            out.append({**SC.sample_fields(s), "role": SC.baseline_roles(s)[spec.model_key], "valid": True, "failure": None,
                        "pred": SC.pred_record(p, store_rle=False),
                        "score": SC.score(s, p, relief_expected=p.relief is not None)})
        return out

    return _run_offline(spec, out_dir, make, resume=resume, argv=argv)


def run_external_task(spec: RunSpec, out_dir: str | Path, *, resume: bool = False, argv: Sequence[str] = ()) -> dict[str, Any]:
    """外部预测（LISA 式对照）：读外部预测目录里的 mask，只做评分（src/e5/external.py 的格式）。"""
    if spec.system != "external" or not spec.external_dir:
        raise ValueError("run_external_task 要 system=external 与 external_dir")
    manifest = EXT.read_manifest(spec.external_dir)
    if manifest["system"] != spec.model_key:
        raise ValueError(f"外部预测目录的 system={manifest['system']!r} 与 --model {spec.model_key!r} 不同")

    def make(items: Sequence[EvalItem]) -> list[dict[str, Any]]:
        samples = [make_sample(it) for it in items]
        if not samples:
            return []
        got, _stats = EXT.load_task(spec.external_dir, canonical_task(spec.task), [it.id for it in items],
                                    tuple(samples[0].target_mask.shape))
        out = []
        for it, s in zip(items, samples):
            v = got[it.id]
            if isinstance(v, str):
                code = SC.E5Failure.EXTERNAL_MISSING if v == EXT.MISSING else SC.E5Failure.EXTERNAL_FAILURE
                p = SC.failed(code.value)
            else:
                p = SC.from_mask(v)
            out.append({**SC.sample_fields(s), "valid": p.valid, "failure": p.failure,
                        "external_failure": None if p.valid else v,
                        "pred": SC.pred_record(p, store_rle=True),
                        "score": SC.score(s, p, relief_expected=False, current=True)})  # 另算对当前 mask 的 IoU（D59-5）
        return out

    return _run_offline(spec, out_dir, make, resume=resume, argv=argv)


__all__ = [
    "COMPUTE", "CONFIG", "METRICS", "PREDICTIONS", "README", "RunSpec", "TIMING", "check_platform", "finalize", "leaf_dir", "make_sample",
    "read_jsonl_tolerant", "run_baseline_task", "run_config", "run_external_task", "run_model_task", "spec_items",
]
