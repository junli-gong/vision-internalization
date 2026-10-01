"""E1 的运行器：一个模型 × 一个任务 × 一个子集 → 输出目录（DEV-0014 任务书 §2；简报 §11.1）。

流程：加载器选样本（src/e1/tasks.select）→ 按提示配置组装输入（build_inputs，图在前）→ generate（batch = 1）或 generate_batch
→ 解析 → 评分（src/e1/scoring）。每批写完才落盘：先写 logs/timing.jsonl，再写 predictions.jsonl（flush + fsync）。

输出（<out>/<模型>/<任务>[__<子集>][__<口径>]/）：
- predictions.jsonl：每条样本一行（id、instruction、raw_text、text、解析结果、失败原因、是否截断、思考段是否未闭合、生成 token 数、
  视觉 token 数与像素数、逐条得分、坐标诊断）。不含墙钟，所以与续跑无关、逐字节可复现。
- metrics.json：只由 predictions.jsonl 与样本 GT 算出（scoring.aggregate），续跑后再算一遍逐字节相同。
- compute.json：src/accounting.py（schema 3）；墙钟来自 logs/timing.jsonl（同一 id 取最后一条）。
- config.yaml（运行的身份：模型、revision、口径、任务、子集、提示配置 sha256 与本家族的提示原文、样本 id 指纹……；不含时间戳与
  git 状态）与 meta.json（src/runlog.write_run_metadata：git、主机、包版本、数据清单的 sha256、时间）。
- README.md：由 metrics.json 生成。logs/：timing.jsonl、每次调用一份 run_<k>.json。

断点续传（--resume）：目录里已有 predictions.jsonl 时必须给 --resume，并且 config.yaml 除 batch_size 外与本次完全相同，否则报错；
末尾不完整的一行截掉；已完成的 id 必须是样本顺序的前缀，从下一条接着跑。没有 --resume 而目录非空：拒绝覆盖。
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml
from PIL import Image

from src.accounting import Compute, merge, write_compute_json
from src.constants import BOOTSTRAP_SEED
from src.data.records import Sample
from src.e1.prompts import COUNTBENCHQA_FIELD, DEFAULT_COUNT_QUESTIONS, PromptConfig
from src.e1.scoring import aggregate, score_generation
from src.e1.tasks import DataPaths, Selection
from src.runlog import write_run_metadata

REPO_ROOT = Path(__file__).resolve().parents[2]
PREDICTIONS = "predictions.jsonl"
METRICS = "metrics.json"
COMPUTE = "compute.json"
CONFIG = "config.yaml"
README = "README.md"
TIMING = "logs/timing.jsonl"
NOT_IDENTITY_KEYS = ("batch_size",)  # 续跑时允许不同的配置键（只影响速度；GPU 上可能影响 bf16 的末位，记在 timing 里）


@dataclass(frozen=True)
class RunSpec:
    model_key: str
    family: str
    repo: str
    revision: str
    adapter_model_key: str
    task: str
    subset: str
    policy_tag: str
    policy: Mapping[str, Any]
    batch_size: int
    limit: int | None
    tiny: bool
    map_ci_resamples: int
    count_questions: str = DEFAULT_COUNT_QUESTIONS  # 计数的问句来源（DEV-0015 §4）；别的任务不用


def policy_dirname(tag: str) -> str:
    return tag.replace(":", "-")


def leaf_dir(out: str | Path, model_key: str, task: str, subset: str, policy_tag: str) -> Path:
    """<out>/<模型>/<任务>，非主口径时加 __<子集>、__<口径>（如 refexp_box__extra__qwen-1024）。"""
    name = task
    if subset != "eval":
        name += f"__{subset}"
    if policy_tag != "default":
        name += f"__{policy_dirname(policy_tag)}"
    return Path(out) / model_key / name


def _jsonl_line(rec: Mapping[str, Any]) -> str:
    return json.dumps(rec, ensure_ascii=False) + "\n"


def read_jsonl_tolerant(path: Path, *, repair: bool) -> list[dict[str, Any]]:
    """读 jsonl；末尾不完整的一行（中断时写了一半）丢掉，repair=True 时把文件截到最后一个完整行。"""
    if not path.exists():
        return []
    raw = path.read_bytes()
    keep = raw.rfind(b"\n") + 1
    recs = []
    for line in raw[:keep].decode("utf-8").splitlines():
        if line.strip():
            recs.append(json.loads(line))
    if repair and keep != len(raw):
        with open(path, "r+b") as f:
            f.truncate(keep)
    return recs


def _append(path: Path, lines: Sequence[str]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write("".join(lines))
        f.flush()
        os.fsync(f.fileno())


def ids_digest(ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def run_config(spec: RunSpec, sel: Selection, prompts: PromptConfig, paths: DataPaths) -> dict[str, Any]:
    """config.yaml 的内容：运行的身份（不含时间戳、git 状态与机器相关的路径）。"""
    e = prompts.entry(spec.task, spec.family)
    ids = [s.id for s in sel.samples]
    return {
        "experiment": "E1",
        "framework": "src/e1（DEV-0014）",
        "model": {"key": spec.model_key, "family": spec.family, "repo": spec.repo, "revision": spec.revision,
                  "adapter_model_key": spec.adapter_model_key, "tiny": spec.tiny},
        "image_policy": {"tag": spec.policy_tag, **dict(spec.policy)},
        "task": spec.task,
        "subset": spec.subset,
        "generation": {"enable_thinking": prompts.enable_thinking(spec.subset),
                       "max_new_tokens": prompts.max_new_tokens(spec.task, spec.family, spec.subset),
                       "decoding": prompts.decoding(), "order": prompts.order},
        "prompt": {"config": os.path.relpath(prompts.path, REPO_ROOT) if Path(prompts.path).is_relative_to(REPO_ROOT)
                   else prompts.path, "config_sha256": prompts.sha256,
                   "entry": {k: e[k] for k in ("template", "native", "provenance", "source_url", "parse",
                                               "max_new_tokens") if k in e},
                   **({"count_questions": spec.count_questions} if spec.task == "count" else {})},
        "data": {"n_samples": len(ids), "sample_ids_sha256": ids_digest(ids), "has_gt": sel.has_gt,
                 "selection_stats": json.loads(json.dumps(sel.stats, ensure_ascii=False)),
                 "config_sha256": {Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                                   for p in paths.config_files()}},
        "limit": spec.limit,
        "map_ci_resamples": spec.map_ci_resamples,
        "batch_size": spec.batch_size,
    }


def _identity(cfg: Mapping[str, Any]) -> dict[str, Any]:
    norm = yaml.safe_load(yaml.safe_dump(dict(cfg), sort_keys=False, allow_unicode=True))
    return {k: v for k, v in norm.items() if k not in NOT_IDENTITY_KEYS}


def load_images(sample: Sample) -> list[Image.Image]:
    out = []
    for p in sample.images:
        with Image.open(p) as im:
            out.append(im.convert("RGB"))
    return out


def _metrics_doc(spec: RunSpec, cfg: Mapping[str, Any], body: Mapping[str, Any]) -> dict[str, Any]:
    return {"task": spec.task, "subset": spec.subset, "model": spec.model_key, "family": spec.family,
            "revision": spec.revision, "image_policy": cfg["image_policy"], "generation": cfg["generation"],
            "prompt_config_sha256": cfg["prompt"]["config_sha256"], **body}


def _fmt(v: Any) -> str:
    return "null" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))


def _think_marker_line(tm: Mapping[str, Any] | None) -> str:
    if tm is None:
        return "不适用（开思考）"
    return f"{tm['n']} 条（其中思考段未闭合 {tm['n_thinking_unclosed']} 条）"


def readme_text(spec: RunSpec, cfg: Mapping[str, Any], metrics: Mapping[str, Any]) -> str:
    lines = [f"# E1 · {spec.model_key} · {spec.task} · {spec.subset}", "",
             f"- 模型：{spec.repo}@{spec.revision}（{spec.family}；适配器 {spec.adapter_model_key}）",
             f"- 图像口径：{cfg['image_policy']}",
             f"- 生成：enable_thinking={cfg['generation']['enable_thinking']}，max_new_tokens="
             f"{cfg['generation']['max_new_tokens']}，greedy，{cfg['generation']['order']}",
             f"- 提示配置 sha256：{cfg['prompt']['config_sha256']}；模板：`{cfg['prompt']['entry']['template']!r}`",
             f"- 样本数 n = {metrics['n']}（id 指纹 {cfg['data']['sample_ids_sha256'][:16]}…）；有 GT：{metrics['has_gt']}", "",
             "## 标题指标", ""]
    if not metrics["headline"]:
        lines.append("（没有 GT 或空集合，不算指标）")
    for h in metrics["headline"]:
        ci = h.get("ci")
        ci_s = f"[{_fmt(ci['low'])}, {_fmt(ci['high'])}]" if ci else "—"
        lines.append(f"- {h['name']}（{h['scope']}）：{_fmt(h['value'])}，95% CI {ci_s}，n = {h['n']}")
    p = metrics["parse"]
    lines += ["", "## 解析与截断", "",
              f"- 解析失败率：{_fmt(p['failure_rate'])}（{p['n_failed']}/{p['n']}）；分原因：{p['failures']}",
              f"- 截断率：{_fmt(metrics['truncation']['rate'])}；思考段未闭合：{_fmt(metrics['thinking_unclosed']['rate'])}",
              "", "## 失败的诊断（只作诊断，失败原因不变；规则见 src/e1/scoring.failure_detail）", "",
              f"- 失败的样子：{metrics['diagnostics']['failure_detail']['counts'] or '（无）'}",
              f"- 关思考时自发写出思考标记：{_think_marker_line(metrics['diagnostics']['think_marker_when_off'])}",
              "", "## 文件", "",
              "- predictions.jsonl：逐条输出、解析与得分；metrics.json：全部指标（由 predictions.jsonl 算出）；",
              "- compute.json：计算记账（src/accounting.py）；config.yaml：运行身份；meta.json：git、主机、包版本；logs/：墙钟与每次调用的记录。",
              ""]
    return "\n".join(lines)


def finalize(leaf: Path, spec: RunSpec, sel: Selection, cfg: Mapping[str, Any]) -> dict[str, Any]:
    """由 predictions.jsonl 与 logs/timing.jsonl 写 metrics.json、compute.json、README.md。返回 metrics。"""
    recs = read_jsonl_tolerant(leaf / PREDICTIONS, repair=False)
    ids = [s.id for s in sel.samples]
    if [r["id"] for r in recs] != ids[: len(recs)]:
        raise RuntimeError(f"{leaf / PREDICTIONS} 的 id 不是样本顺序的前缀")
    samples = sel.samples[: len(recs)]
    body = aggregate(spec.task, samples, recs, has_gt=sel.has_gt, map_ci_resamples=spec.map_ci_resamples,
                     enable_thinking=bool(cfg["generation"]["enable_thinking"]))
    body["complete"] = len(recs) == len(ids)
    body["n_expected"] = len(ids)
    metrics = _metrics_doc(spec, cfg, body)
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
    (leaf / README).write_text(readme_text(spec, cfg, metrics), encoding="utf-8")
    return metrics


def _common_record(s: Sample, rendered: Any, image_size: tuple[int, int], res: Any) -> dict[str, Any]:
    if res.stopped is None or res.n_image_pixels is None:
        raise RuntimeError(f"{s.id}：适配器没有填 stopped / n_image_pixels（D29：三个适配器一律填写）")
    return {"id": s.id, "source": s.source, "split": s.split, "instruction": rendered.text, "extras": rendered.extras,
            "image_size": list(image_size), "raw_text": res.raw_text, "text": res.text, "stopped": res.stopped,
            "truncated": res.stopped is False, "thinking_unclosed": res.thinking_unclosed,
            "n_generated_tokens": res.n_generated_tokens, "n_prompt_tokens": res.n_prompt_tokens,
            "n_visual_tokens": res.n_visual_tokens, "n_image_pixels": res.n_image_pixels}


def generate_rows(adapter: Any, inputs: Sequence[Any], *, batch_size: int, enable_thinking: bool, max_new_tokens: int,
                  decoding: Mapping[str, Any]) -> list[Any]:
    """batch_size = 1 走 generate（逐条 greedy 的参照路径）；否则 generate_batch（即使最后一批只剩一行）。"""
    if batch_size == 1:
        if len(inputs) != 1:
            raise ValueError("batch_size = 1 时每批只能有一条")
        return [adapter.generate(inputs[0], enable_thinking=enable_thinking, max_new_tokens=max_new_tokens,
                                 **dict(decoding))]
    return adapter.generate_batch(list(inputs), enable_thinking=enable_thinking, max_new_tokens=max_new_tokens,
                                  **dict(decoding))


def run_task(adapter: Any, spec: RunSpec, sel: Selection, prompts: PromptConfig, paths: DataPaths, out_dir: str | Path,
             *, resume: bool = False, argv: Sequence[str] = (), log: Callable[[str], None] | None = None,
             warmup: bool = True) -> dict[str, Any]:
    """跑一个任务，返回 metrics（规则见模块说明）。sel 的样本已按 limit 截好。"""
    log = log or (lambda m: print(m, file=sys.stderr, flush=True))
    if isinstance(spec.batch_size, bool) or not isinstance(spec.batch_size, int) or spec.batch_size < 1:
        raise ValueError(f"batch_size 必须是正整数，收到 {spec.batch_size!r}")
    if spec.task == "count":  # 问句来源由样本决定（tasks.select），这里核对与 RunSpec 一致，防止静默混用
        has = [COUNTBENCHQA_FIELD in s.fields for s in sel.samples]
        mixed = any(has) and not all(has)
        if has and (mixed or any(has) != (spec.count_questions == "countbenchqa")):
            raise ValueError(f"count_questions={spec.count_questions!r}，而样本里带人工问句的有 {sum(has)}/{len(has)} 条："
                             "select 与 RunSpec 的问句来源不一致")
    leaf = leaf_dir(out_dir, spec.model_key, spec.task, spec.subset, spec.policy_tag)
    cfg = run_config(spec, sel, prompts, paths)
    pred_path = leaf / PREDICTIONS
    if leaf.exists() and any(leaf.iterdir()):
        if not resume:
            raise FileExistsError(f"{leaf} 已存在且非空；续跑请加 --resume")
        if not (leaf / CONFIG).exists():
            raise FileExistsError(f"{leaf} 里没有 {CONFIG}，不能续跑")
        old = yaml.safe_load((leaf / CONFIG).read_text(encoding="utf-8"))
        if _identity(old) != _identity(cfg):
            diff = sorted(k for k in set(_identity(old)) | set(_identity(cfg)) if _identity(old).get(k) != _identity(cfg).get(k))
            raise ValueError(f"{leaf}：续跑的配置与已有的不同（{diff}），拒绝混写")
    else:
        leaf.mkdir(parents=True, exist_ok=True)
        (leaf / "logs").mkdir(exist_ok=True)
        files = [prompts.path, *paths.config_files()]
        write_run_metadata(leaf, config=cfg, seed=BOOTSTRAP_SEED, data_files=files, repo_root=REPO_ROOT)
    (leaf / "logs").mkdir(exist_ok=True)
    done = read_jsonl_tolerant(pred_path, repair=True)
    ids = [s.id for s in sel.samples]
    if [r["id"] for r in done] != ids[: len(done)]:
        raise RuntimeError(f"{pred_path} 里已完成的 id 不是样本顺序的前缀，不能续跑")
    remaining = sel.samples[len(done):]
    run_k = len(list((leaf / "logs").glob("run_*.json"))) + 1
    run_path = leaf / "logs" / f"run_{run_k:03d}.json"
    run_log: dict[str, Any] = {"argv": list(argv), "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                               "n_done_before": len(done), "n_remaining": len(remaining), "batch_size": spec.batch_size,
                               "finished_utc": None}
    run_path.write_text(json.dumps(run_log, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")  # 先占位：中断也留下记录
    enable_thinking = prompts.enable_thinking(spec.subset)
    max_new = prompts.max_new_tokens(spec.task, spec.family, spec.subset)
    decoding = prompts.decoding()
    parser = prompts.parser(spec.task, spec.family)
    t_all = time.perf_counter()
    if remaining and warmup:
        run_log["warmup_wall_s"] = adapter.warmup(load_images(remaining[0]), batch_size=min(spec.batch_size, len(remaining)))
    n_batches = 0
    for start in range(0, len(remaining), spec.batch_size):
        batch = remaining[start: start + spec.batch_size]
        rendered = [prompts.render(spec.task, spec.family, s) for s in batch]
        t0 = time.perf_counter()
        imgs = [load_images(s) for s in batch]
        inputs = [adapter.build_inputs(im, r.text, prompts.order) for im, r in zip(imgs, rendered)]
        build_s = time.perf_counter() - t0
        results = generate_rows(adapter, inputs, batch_size=spec.batch_size, enable_thinking=enable_thinking,
                                max_new_tokens=max_new, decoding=decoding)
        if len(results) != len(batch):
            raise RuntimeError("generate_batch 返回的行数与输入不同")
        lines, tlines = [], []
        for s, r, im, res in zip(batch, rendered, imgs, results):
            size = im[0].size
            rec = _common_record(s, r, size, res)
            rec.update(score_generation(spec.task, parser, s, res, adapter=adapter, image_size=size, extras=r.extras))
            lines.append(_jsonl_line(rec))
            tlines.append(_jsonl_line({"id": s.id, "run": run_k, "batch_index": n_batches, "batch_rows": len(batch),
                                       "wall_time_s": res.wall_time_s, "batch_wall_time_s": res.batch_wall_time_s,
                                       "build_inputs_s": build_s / len(batch)}))
        _append(leaf / TIMING, tlines)
        _append(pred_path, lines)
        n_batches += 1
        n_now = len(done) + start + len(batch)
        if n_batches == 1 or n_batches % 20 == 0 or n_now == len(ids):
            log(f"[{spec.model_key} {spec.task}/{spec.subset}] {n_now}/{len(ids)}")
    metrics = finalize(leaf, spec, sel, cfg)
    run_log.update(finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), n_batches=n_batches,
                   wall_s=round(time.perf_counter() - t_all, 3))
    run_path.write_text(json.dumps(run_log, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return metrics


__all__ = [
    "COMPUTE",
    "CONFIG",
    "METRICS",
    "PREDICTIONS",
    "README",
    "RunSpec",
    "TIMING",
    "finalize",
    "generate_rows",
    "ids_digest",
    "leaf_dir",
    "load_images",
    "policy_dirname",
    "read_jsonl_tolerant",
    "run_config",
    "run_task",
]
