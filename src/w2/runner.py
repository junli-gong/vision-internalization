"""W2 的运行器：一个模型 × 一个任务 × 一种设置 × 一个划分 → 输出目录（DEV-0059；docs/w2_plan.md §3）。

布局（同 W1）：<out>/<模型>/<任务>/<设置>/<划分>/。共用的中间结果也放在这棵树里：
- <out>/<模型>/refexp_box/baseline_e1/<划分>：E1 refexp_box（"提出"的框，mode = box；也是 ownbox 与 w1f 的框）。E1 的运行器原样，
  config.yaml 的 data.selection_stats.w2 记 W2 的划分；没跑完就先按 E1 的协议跑完（ensure_e1_run，同 W1 的 ensure_box_run）。
- <out>/<模型>/refseg/propose_box_points/<划分>：mode = box_points 的提出（另一个提示的单轮生成，同 E1 的口径）。
- <out>/<模型>/refseg/candidates/<划分>（mode = box_points 时是 candidates_box_points）：候选缓存 candidates.jsonl（每条样本一行：提出、
  K 个候选的 COCO RLE、名字、ok、info（参数、耗时、失败原因））；logs/timing.jsonl 是每条的生成墙钟。w2a、random_pick、oracle_pick、
  w1f 共用；w1f 总是用 mode = box 的那份（1 号候选 = W1-F：E1 协议的框交给默认 GrabCut）。只生成一次：已有且身份相同就续上。
设置：
- w2a（refseg）：提出 → 候选 → 挑选（src/w2/loop.run_selection）；w2a（refexp_box、count）：按输出类型路由，就是同一个划分上的
  baseline_e1 的结果（没跑完先跑），目录里原样复制 predictions.jsonl、compute.json，config.yaml 记路由与来源的 sha256。
- random_pick、oracle_pick、w1f（refseg）：同一份候选，不跑模型。
- ownbox（refseg）：E1 refexp_box 的框光栅化（同 W1 的"自己的框"）；baseline_e1：E1 零样本（refseg → refexp_poly）。
- w2c（refseg；W2-c，DEV-0061，src/w2/w2c.py）：训练过的检查点做提出（<out>/<模型>/refexp_box/propose_w2c/<划分>）与挑选（关思考），
  候选缓存另放 <out>/<模型>/refseg/candidates_w2c/<划分>（ensure_candidates 的 cand_setting）；config.yaml 另记 checkpoint。
输出：predictions.jsonl（逐条，不含墙钟，逐字节可复现）、metrics.json（只由 predictions.jsonl 算：cIoU、gIoU 与 bootstrap 95% CI；
挑中最优候选、挑中 1 号、"都不对"、解析失败的比例；失败、各候选的 cIoU……）、compute.json（每次模型调用一条：提出一次，w2a 加挑选
一次；墙钟 = 提出 + 候选生成 + 挑选）、config.yaml（运行身份）、meta.json、README.md、logs/（timing.jsonl、run_<k>.json）。
续跑（--resume）：同 W1，按 chunk_size 分组，没写完的那一组整组重跑，与一次跑完的 predictions.jsonl 逐字节相同。
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import yaml

from src.accounting import Compute, merge, write_compute_json
from src.data.records import Sample
from src.e1.posthoc import POSTHOC_PATH, load_posthoc_config
from src.e1.posthoc_runner import _finish_run_log, _prepare_leaf, _resume_point, _start_run_log
from src.e1.prompts import PromptConfig
from src.e1.runner import (
    COMPUTE,
    CONFIG,
    METRICS,
    PREDICTIONS,
    README,
    TIMING,
    _append,
    _common_record,
    _jsonl_line,
    batch_plan,
    generate_rows,
    ids_digest,
    load_images,
    read_jsonl_tolerant,
    run_task,
)
from src.e1.scoring import distribution, gt_mask, is_valid
from src.e1.tasks import DataPaths, Selection
from src.w1 import baselines as B
from src.w1.runner import model_record
from src.w1.scoring import headline
from src.w2.config import CANDIDATE_SETTINGS, MODEL_PICK_SETTINGS, OUTPUT_KIND, PROPOSE_BOX, PROPOSE_BOX_POINTS, W2C_SETTING, W2Config
from src.w2.data import splits_path, splits_sha256
from src.w2.loop import (
    FALLBACK_NONE,
    FALLBACK_PARSE,
    SelectItem,
    candidate_record,
    candidates_digest_rows,
    candidates_row,
    parse_box_points_generation,
    proposal_of,
    run_selection,
    selection_candidates,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CANDIDATES = "candidates.jsonl"
FRAMEWORK = "src/w2（DEV-0059；docs/w2_plan.md）"
BOX_TASK = "refexp_box"
E1_SETTING = "baseline_e1"
POINTS_SETTING = "propose_box_points"
DEFAULT_CANDIDATES_MODULE = "src.w2.candidates"
DEFAULT_RENDER_MODULE = "src.w2.render"


@dataclass(frozen=True)
class W2RunSpec:
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
    unseal: Mapping[str, Any] | None = None  # 密封划分（devcheck3）放行的记录（门名；DEV-0067，src/data/explore_splits.require_unsealed）
    test_unlock: Mapping[str, Any] | None = None  # test 的放行记录（批准编号、gate_explore.json 的路径与 sha256；DEV-0072，scripts/run_x2.py）


def w2_leaf(out: str | Path, model_key: str, task: str, setting: str, split: str) -> Path:
    return Path(out) / model_key / task / setting / split


def candidates_setting(mode: str) -> str:
    return "candidates" if mode == PROPOSE_BOX else f"candidates_{mode}"


def w2_record(spec: W2RunSpec, splits_doc: Mapping[str, Any], samples: Sequence[Sample]) -> dict[str, Any]:
    """运行身份里的 W2 一节（E1 格式的目录放在 data.selection_stats.w2）：任务、划分、设置、划分清单与这个划分的指纹。"""
    e = splits_doc["splits"][spec.split]
    return {"task": spec.task, "split": spec.split, "setting": spec.setting,
            "splits_config": "tiny（tests/fixtures 上现算）" if splits_doc.get("tiny") else B._rel(splits_path(splits_doc)),
            "splits_sha256": splits_sha256(splits_doc), "split_sha256": e["sha256"], "split_n": e["n"],
            "n_samples": len(samples), "limit": spec.limit, "sample_ids_sha256": ids_digest([s.id for s in samples]),
            **({"unseal": dict(spec.unseal)} if spec.unseal is not None else {}),
            **({"test_unlock": json.loads(json.dumps(dict(spec.test_unlock)))} if spec.test_unlock is not None else {})}


def base_model_record(spec: W2RunSpec) -> dict[str, Any]:
    """共用的中间结果（提出、候选缓存）的模型身份：不含挑选一步的适配器键（换挑选的适配器不影响它们）。"""
    return {k: v for k, v in model_record(spec).items() if k != "adapter_model_key"}


def _splits_files(splits_doc: Mapping[str, Any]) -> list[Path]:
    return [] if splits_doc.get("tiny") else [splits_path(splits_doc)]


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _run_ref(out: str | Path, leaf: Path) -> dict[str, Any]:
    """另一个运行目录的引用（运行身份用）：<out>/… 路径、predictions 与 config.yaml 的 sha256。"""
    return {"leaf": B._under_out(out, leaf), "predictions_sha256": _sha(leaf / PREDICTIONS), "config_sha256": _sha(leaf / CONFIG)}


def _counts(values) -> dict[str, int]:
    c = Counter("ok" if v is None else str(v) for v in values)
    return {k: c[k] for k in sorted(c)}


def _fmt(v: Any) -> str:
    return "null" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))


def _timing_by_id(leaf: Path) -> dict[str, dict[str, Any]]:
    return {t["id"]: t for t in read_jsonl_tolerant(leaf / TIMING, repair=False)}


# ---------------------------------------------------------------------------
# 候选与渲染的实现（DEV-0058；测试与彩排注入假实现）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Impl:
    """候选生成器与渲染：模块名（记进运行身份）与 docs/w2_plan.md §1、§2 约定的几个名字。render 只有 w2a 要。"""

    candidates_module: str
    generate: Callable[..., Any]
    Candidate: type
    CandidateSpec: type
    render_module: str | None = None
    render: Callable[..., Any] | None = None


def load_impl(candidates_module: str = DEFAULT_CANDIDATES_MODULE, render_module: str | None = None) -> Impl:
    cm = importlib.import_module(candidates_module)
    rm = importlib.import_module(render_module) if render_module else None
    return Impl(candidates_module, cm.generate_candidates, cm.Candidate, cm.CandidateSpec, render_module,
                None if rm is None else rm.render_selection)


# ---------------------------------------------------------------------------
# E1 的运行（baseline_e1；"提出"的框）
# ---------------------------------------------------------------------------


def e1_selection(e1_task: str, samples: Sequence[Sample], w2: Mapping[str, Any]) -> Selection:
    return Selection(e1_task, "eval", list(samples), True, {"w2": dict(w2), "n": len(samples)})


def run_baseline_e1(adapter: Any, entry: Any, policy: Any, spec: W2RunSpec, cfg: W2Config, samples: Sequence[Sample],
                    paths: DataPaths, out: str | Path, *, splits_doc: Mapping[str, Any], batch_size: int, resume: bool,
                    prompts: PromptConfig, argv: Sequence[str] = (), warmup: bool = True) -> dict[str, Any]:
    """E1 零样本（refseg → refexp_poly、refexp_box、count）在 W2 的划分上跑，E1 的运行器原样，返回 E1 的 metrics。"""
    e1_task = cfg.e1_task(spec.task)
    rs = B.e1_spec(entry, adapter, policy, e1_task, "eval", batch_size, tiny=spec.tiny, limit=spec.limit)
    sel = e1_selection(e1_task, samples, w2_record(spec, splits_doc, samples))
    leaf = w2_leaf(out, spec.model_key, spec.task, spec.setting, spec.split)
    return run_task(adapter, rs, sel, prompts, paths, out, resume=resume, argv=argv, leaf=leaf, warmup=warmup)


def e1_run_complete(leaf: Path, samples: Sequence[Sample], spec: W2RunSpec) -> bool:
    """E1 格式的运行已经写完、是同一个模型、同一批样本、同一个 W2 划分。"""
    if not (leaf / CONFIG).exists() or not (leaf / PREDICTIONS).exists():
        return False
    recs = read_jsonl_tolerant(leaf / PREDICTIONS, repair=False)
    if [r["id"] for r in recs] != [s.id for s in samples]:
        return False
    c = yaml.safe_load((leaf / CONFIG).read_text(encoding="utf-8"))
    w2 = ((c.get("data") or {}).get("selection_stats") or {}).get("w2") or {}
    return (c["model"]["key"], c["model"]["revision"], bool(c["model"].get("tiny")), w2.get("split")) == \
        (spec.model_key, spec.revision, spec.tiny, spec.split)


def ensure_e1_run(get_model: Callable[[], tuple[Any, Any, Any]], spec: W2RunSpec, task: str, cfg: W2Config,
                  samples: Sequence[Sample], paths: DataPaths, out: str | Path, *, splits_doc: Mapping[str, Any], batch_size: int,
                  prompts: PromptConfig, argv: Sequence[str] = (), log: Callable[[str], None] | None = None,
                  warmup: bool = True) -> tuple[Path, list[dict[str, Any]]]:
    """同一个划分上的 baseline_e1（task）：已写完就直接读（不加载模型）；否则 get_model() 取 (entry, policy, 适配器) 按 E1 的协议跑完。"""
    leaf = w2_leaf(out, spec.model_key, task, E1_SETTING, spec.split)
    if not e1_run_complete(leaf, samples, spec):
        entry, policy, ad = get_model()
        bspec = dataclasses.replace(spec, task=task, setting=E1_SETTING)
        if log:
            log(f"[W2 {spec.model_key}] 先跑 E1 的 {task}：{leaf}")
        run_baseline_e1(ad, entry, policy, bspec, cfg, samples, paths, out, splits_doc=splits_doc, batch_size=batch_size, resume=True,
                        prompts=prompts, argv=argv, warmup=warmup)
        if not e1_run_complete(leaf, samples, spec):
            raise RuntimeError(f"{leaf}：E1 的运行跑完之后仍不完整")
    return leaf, read_jsonl_tolerant(leaf / PREDICTIONS, repair=False)


# ---------------------------------------------------------------------------
# box_points 的提出
# ---------------------------------------------------------------------------


def points_identity(spec: W2RunSpec, cfg: W2Config, splits_doc: Mapping[str, Any], samples: Sequence[Sample], prompts: PromptConfig,
                    paths: DataPaths, batch_size: int) -> dict[str, Any]:
    bp = cfg.box_points
    return {"experiment": "W2", "framework": FRAMEWORK, "w2": w2_record(spec, splits_doc, samples), "model": base_model_record(spec),
            "image_policy": dict(spec.image_policy),
            "propose": {"mode": PROPOSE_BOX_POINTS, "prompt": bp["prompt"], "max_points": int(bp["max_points"]),
                        "max_new_tokens": int(bp["max_new_tokens"]), "w2_config": cfg.identity()},
            "generation": {"enable_thinking": False, "decoding": prompts.decoding(), "order": prompts.order,
                           "e1_prompts_sha256": prompts.sha256},
            "batch_size": batch_size,
            "data": {"config_sha256": {Path(p).name: _sha(Path(p)) for p in paths.config_files()}}}


def run_points_proposal(adapter: Any, spec: W2RunSpec, cfg: W2Config, samples: Sequence[Sample], paths: DataPaths, out: str | Path,
                        *, splits_doc: Mapping[str, Any], batch_size: int, prompts: PromptConfig, resume: bool,
                        argv: Sequence[str] = (), log: Callable[[str], None] | None = None, warmup: bool = True) -> Path:
    """mode = box_points 的提出：单轮生成（关思考、贪心、E1 的解码参数、图在前），适配器的 parse_box_points 解析。返回目录。"""
    log = log or (lambda m: print(m, file=sys.stderr, flush=True))
    bp = cfg.box_points
    if spec.family not in bp["families"]:
        raise ValueError(f"propose.mode = box_points 只给 {bp['families']} 用（点是 Qwen 的原生格式），收到 {spec.family}")
    leaf = w2_leaf(out, spec.model_key, "refseg", POINTS_SETTING, spec.split)
    pspec = dataclasses.replace(spec, task="refseg", setting=POINTS_SETTING)
    cfg_doc = points_identity(pspec, cfg, splits_doc, samples, prompts, paths, batch_size)
    _prepare_leaf(leaf, cfg_doc, [cfg.path, *_splits_files(splits_doc), *paths.config_files()], resume=resume)
    ids = [s.id for s in samples]
    plan = batch_plan([None] * len(ids), batch_size)
    what = f"W2 {spec.model_key} propose_box_points/{spec.split}"
    keep, trimmed = _resume_point(leaf / PREDICTIONS, ids, lambda r: r["id"], plan, log, what)
    run_k, run_path, run_log = _start_run_log(leaf, argv, n_done_before=keep, n_remaining=len(ids) - keep, batch_size=batch_size,
                                              trimmed_to_batch_boundary=trimmed)
    t_all = time.perf_counter()
    todo = [(a, b) for a, b in plan if a >= keep]
    if todo and warmup and hasattr(adapter, "warmup"):
        run_log["warmup_wall_s"] = adapter.warmup(load_images(samples[keep]), batch_size=min(batch_size, len(ids) - keep))
    n_batches = 0
    for a, b in todo:
        batch = samples[a:b]
        imgs = [load_images(s) for s in batch]
        texts = [cfg.box_points_prompt(s.fields["expression"]) for s in batch]
        inputs = [adapter.build_inputs(im, t, prompts.order) for im, t in zip(imgs, texts)]
        results = generate_rows(adapter, inputs, batch_size=batch_size, enable_thinking=False,
                                max_new_tokens=int(bp["max_new_tokens"]), decoding=prompts.decoding())
        if len(results) != len(batch):
            raise RuntimeError("generate_batch 返回的行数与输入不同")
        lines, tlines = [], []
        for s, t, im, res in zip(batch, texts, imgs, results):
            w, h = im[0].size
            rec = _common_record(s, SimpleNamespace(text=t, extras={}), (w, h), res, "text")
            p = parse_box_points_generation(res, adapter, width=w, height=h, max_points=int(bp["max_points"]))
            rec.update(failure=None if p.failure is None else p.failure.value,
                       parsed=None if p.failure else {"box": list(p.box), "fg_points": [list(v) for v in p.fg_points],
                                                      "bg_points": [list(v) for v in p.bg_points]},
                       n_points_dropped=p.n_points_dropped)
            lines.append(_jsonl_line(rec))
            tlines.append(_jsonl_line({"id": s.id, "run": run_k, "batch_index": n_batches, "batch_rows": len(batch),
                                       "wall_time_s": res.wall_time_s}))
        _append(leaf / TIMING, tlines)
        _append(leaf / PREDICTIONS, lines)
        n_batches += 1
        log(f"[{what}] {b}/{len(ids)}")
    recs = read_jsonl_tolerant(leaf / PREDICTIONS, repair=False)
    timing = _timing_by_id(leaf)
    m = {"experiment": "W2", "w2": cfg_doc["w2"], "model": spec.model_key, "revision": spec.revision, "n": len(recs),
         "failures": _counts(r["failure"] for r in recs),
         "fg_points": distribution(len((r["parsed"] or {}).get("fg_points") or []) for r in recs),
         "bg_points": distribution(len((r["parsed"] or {}).get("bg_points") or []) for r in recs),
         "n_points_dropped": int(sum(r["n_points_dropped"] or 0 for r in recs)), "complete": len(recs) == len(ids),
         "n_expected": len(ids)}
    (leaf / METRICS).write_text(json.dumps(m, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    write_compute_json(leaf / COMPUTE, merge(Compute.call(generated_tokens=r["n_generated_tokens"], prompt_tokens=r["n_prompt_tokens"],
                                                          visual_tokens=r["n_visual_tokens"], image_pixels=r["n_image_pixels"],
                                                          call_time_s=float(timing.get(r["id"], {}).get("wall_time_s", 0.0)),
                                                          new_round=True)
                                             .as_sample(wall_time_s=float(timing.get(r["id"], {}).get("wall_time_s", 0.0)))
                                             for r in recs))
    (leaf / README).write_text("\n".join([
        f"# W2 · {spec.model_key} · 提出（box_points）· {spec.split}", "",
        f"- 提示：{bp['prompt'][:80]}…（至多 {bp['max_points']} 个前景点、{bp['max_points']} 个背景点）；关思考、贪心，批 {batch_size}",
        f"- n = {m['n']}；失败 {m['failures']}；丢掉的点 {m['n_points_dropped']}", ""]), encoding="utf-8")
    _finish_run_log(run_path, run_log, n_batches=n_batches, t_all=t_all)
    return leaf


def ensure_points_run(get_model: Callable[[], tuple[Any, Any, Any]], spec: W2RunSpec, cfg: W2Config, samples: Sequence[Sample],
                      paths: DataPaths, out: str | Path, *, splits_doc: Mapping[str, Any], batch_size: int, prompts: PromptConfig,
                      argv: Sequence[str] = (), log: Callable[[str], None] | None = None) -> tuple[Path, list[dict[str, Any]]]:
    leaf = w2_leaf(out, spec.model_key, "refseg", POINTS_SETTING, spec.split)
    recs = read_jsonl_tolerant(leaf / PREDICTIONS, repair=False) if (leaf / PREDICTIONS).exists() else []
    if [r["id"] for r in recs] != [s.id for s in samples] or not (leaf / METRICS).exists():
        _, _, ad = get_model()
        run_points_proposal(ad, spec, cfg, samples, paths, out, splits_doc=splits_doc, batch_size=batch_size, prompts=prompts,
                            resume=True, argv=argv, log=log)
        recs = read_jsonl_tolerant(leaf / PREDICTIONS, repair=False)
    else:  # 已写完：身份必须与这次的配置相同（不同就拒绝混用）
        old = yaml.safe_load((leaf / CONFIG).read_text(encoding="utf-8"))
        want = points_identity(dataclasses.replace(spec, task="refseg", setting=POINTS_SETTING), cfg, splits_doc, samples, prompts,
                               paths, batch_size)
        if old["propose"] != want["propose"] or old["model"]["key"] != spec.model_key or old["w2"]["split"] != spec.split:
            raise ValueError(f"{leaf}：已有的提出与这次的配置不同，拒绝混用")
    return leaf, recs


# ---------------------------------------------------------------------------
# 候选缓存
# ---------------------------------------------------------------------------


def _proposals_by_id(mode: str, samples: Sequence[Sample], prop_recs: Sequence[Mapping[str, Any]], prompts: PromptConfig,
                     spec: W2RunSpec, cfg: W2Config) -> dict[str, Mapping[str, Any]]:
    """按 id 对上提出的记录并核对同一句话：mode = box 用 W1 的 matched_boxes（E1 的提示原文、原图尺寸）；box_points 核对提示原文。"""
    if mode == PROPOSE_BOX:
        return B.matched_boxes(samples, prop_recs, prompts, spec.family, {s.id: gt_mask(s) for s in samples})
    by = {r["id"]: r for r in prop_recs}
    bad = [s.id for s in samples if s.id not in by or by[s.id]["instruction"] != cfg.box_points_prompt(s.fields["expression"])]
    if bad:
        raise ValueError(f"提出的记录与样本对不上：{len(bad)} 条（例如 {bad[:3]}）")
    return {s.id: by[s.id] for s in samples}


def candidates_identity(spec: W2RunSpec, cfg: W2Config, splits_doc: Mapping[str, Any], samples: Sequence[Sample], out: str | Path,
                        *, mode: str, prop_dir: Path, impl: Impl, chunk_size: int, cand_setting: str | None = None) -> dict[str, Any]:
    cspec = dataclasses.replace(spec, task="refseg", setting=cand_setting or candidates_setting(mode))
    return {"experiment": "W2", "framework": FRAMEWORK, "w2": w2_record(cspec, splits_doc, samples), "model": base_model_record(spec),
            "propose": {"mode": mode, "run": _run_ref(out, prop_dir)},
            "candidates": {"module": impl.candidates_module, "generators": list(cfg.generators), "k": cfg.k,
                           "generators_sha256": cfg.candidates_digest()},
            "chunk_size": chunk_size}


# 候选生成的并行（管理员审查 DEV-0059 的补充 C）：--candidate-workers N（默认 1，不进运行身份）。N > 1 时按组用 spawn 的进程池，
# 子进程里 OpenCV 与 numpy / BLAS 都限成 1 个线程；N = 1 时在本进程里跑，OpenCV 同样临时限成 1 个线程（两条路径的线程数一致）。
# 结果按样本顺序写回（pool.map），candidates.jsonl 除 info 里的耗时外与单进程逐字节相同；续跑粒度仍是组。
THREAD_ENV_VARS: tuple[str, ...] = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                                    "NUMEXPR_NUM_THREADS")


def _set_cv2_threads(n: int | None) -> int | None:
    """设 OpenCV 的线程数，返回原来的值（没有 OpenCV 时什么都不做，返回 None）。"""
    try:
        import cv2
    except ImportError:
        return None
    old = int(cv2.getNumThreads())
    if n is not None:
        cv2.setNumThreads(int(n))
    return old


def _init_candidate_worker() -> None:
    """进程池子进程的初始化：OpenCV 1 个线程（环境变量里的 BLAS 线程数在建池时已设为 1，子进程启动时继承）。"""
    _set_cv2_threads(1)


def candidate_job(job: tuple[str, tuple[dict[str, Any], ...], int, Sample, dict[str, Any]]) -> tuple[dict[str, Any], float]:
    """一条样本的候选（单进程与进程池共用同一个函数）：(候选模块名, 生成器清单, K, 样本, 提出) → (candidates.jsonl 的一行, 生成墙钟)。"""
    module, generators, k, sample, prop = job
    cm = importlib.import_module(module)
    im = load_images(sample)[0]
    cands, wall = None, 0.0
    if prop["box"] is not None:
        t0 = time.perf_counter()
        cands = cm.generate_candidates(np.asarray(im.convert("RGB")), prop["box"], fg_points_abs=prop["fg_points"] or None,
                                       bg_points_abs=prop["bg_points"] or None, spec=cm.CandidateSpec(generators=tuple(generators)))
        wall = time.perf_counter() - t0
    return candidates_row(sample.id, im.size, prop, cands, k), wall


class _CandidateRunner:
    """workers = 1：本进程（OpenCV 临时 1 个线程）；workers > 1：spawn 的进程池（BLAS 环境变量在建池期间设为 1，结束后还原）。"""

    def __init__(self, workers: int) -> None:
        if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
            raise ValueError(f"candidate workers 必须是正整数，收到 {workers!r}")
        self.workers = workers
        self.pool = None
        self._env: dict[str, str | None] = {}
        self._cv2_old: int | None = None

    def __enter__(self) -> "_CandidateRunner":
        if self.workers == 1:
            self._cv2_old = _set_cv2_threads(1)
            return self
        import multiprocessing
        import os
        from concurrent.futures import ProcessPoolExecutor

        self._env = {k: os.environ.get(k) for k in THREAD_ENV_VARS}
        os.environ.update({k: "1" for k in THREAD_ENV_VARS})
        self.pool = ProcessPoolExecutor(max_workers=self.workers, mp_context=multiprocessing.get_context("spawn"),
                                        initializer=_init_candidate_worker)
        return self

    def map(self, jobs: Sequence[Any]) -> list[tuple[dict[str, Any], float]]:
        if self.pool is None:
            return [candidate_job(j) for j in jobs]
        return list(self.pool.map(candidate_job, jobs))  # 按输入顺序返回

    def __exit__(self, *exc: Any) -> None:
        if self.pool is not None:
            import os

            self.pool.shutdown(wait=True)
            for k, v in self._env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        elif self._cv2_old is not None:
            _set_cv2_threads(self._cv2_old)


def ensure_candidates(spec: W2RunSpec, cfg: W2Config, samples: Sequence[Sample], out: str | Path, *, splits_doc: Mapping[str, Any],
                      mode: str, prop_dir: Path, prop_recs: Sequence[Mapping[str, Any]], impl: Impl, chunk_size: int,
                      prompts: PromptConfig, argv: Sequence[str] = (), log: Callable[[str], None] | None = None,
                      workers: int = 1, cand_setting: str | None = None) -> tuple[Path, dict[str, dict[str, Any]]]:
    """候选缓存（规则见模块说明）：没有就生成，生成到一半就续上（身份必须相同）。返回 (目录, {id: 行})。
    workers：候选生成的进程数（不进运行身份；内容与单进程相同，见上面的说明）。cand_setting：缓存目录的设置名（默认
    candidates_setting(mode)；W2-c 用 candidates_w2c，与 base 的候选分开）。"""
    log = log or (lambda m: print(m, file=sys.stderr, flush=True))
    cname = cand_setting or candidates_setting(mode)
    leaf = w2_leaf(out, spec.model_key, "refseg", cname, spec.split)
    cfg_doc = candidates_identity(spec, cfg, splits_doc, samples, out, mode=mode, prop_dir=prop_dir, impl=impl, chunk_size=chunk_size,
                                  cand_setting=cand_setting)
    _prepare_leaf(leaf, cfg_doc, [cfg.path, *_splits_files(splits_doc)], resume=True)
    ids = [s.id for s in samples]
    by = _proposals_by_id(mode, samples, prop_recs, prompts, spec, cfg)
    plan = batch_plan([None] * len(ids), chunk_size)
    what = f"W2 {spec.model_key} {cname}/{spec.split}"
    keep, trimmed = _resume_point(leaf / CANDIDATES, ids, lambda r: r["id"], plan, log, what)
    todo = [(a, b) for a, b in plan if a >= keep]
    if todo:
        run_k, run_path, run_log = _start_run_log(leaf, argv, n_done_before=keep, n_remaining=len(ids) - keep, chunk_size=chunk_size,
                                                  candidate_workers=workers, trimmed_to_batch_boundary=trimmed)
        t_all = time.perf_counter()
        with _CandidateRunner(workers) as runner:
            for n_chunk, (a, b) in enumerate(todo):
                chunk = list(samples[a:b])
                jobs = [(impl.candidates_module, cfg.generators, cfg.k, s, proposal_of(mode, by[s.id])) for s in chunk]
                results = runner.map(jobs)
                lines, tlines = [], []
                for s, (row, wall) in zip(chunk, results):
                    if row["id"] != s.id:
                        raise RuntimeError(f"候选生成返回的顺序不对：{row['id']} ≠ {s.id}")
                    lines.append(json.dumps(row, ensure_ascii=False) + "\n")
                    tlines.append(_jsonl_line({"id": s.id, "run": run_k, "chunk_index": n_chunk, "wall_time_s": wall}))
                _append(leaf / TIMING, tlines)
                _append(leaf / CANDIDATES, lines)
                log(f"[{what}] {b}/{len(ids)}")
        _finish_run_log(run_path, run_log, n_batches=len(todo), t_all=t_all)
    rows = read_jsonl_tolerant(leaf / CANDIDATES, repair=False)
    if [r["id"] for r in rows] != ids:
        raise RuntimeError(f"{leaf / CANDIDATES} 的 id 与样本不同")
    if not (leaf / METRICS).exists() or todo:
        _candidates_summary(leaf, spec, cfg_doc, rows)
    return leaf, {r["id"]: r for r in rows}


def _candidates_summary(leaf: Path, spec: W2RunSpec, cfg_doc: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> None:
    with_c = [r for r in rows if r["candidates"] is not None]
    names = [c["name"] for c in (with_c[0]["candidates"] if with_c else [])]
    not_ok = Counter(c["index"] for r in with_c for c in r["candidates"] if not c["ok"])
    dup = 0
    for r in with_c:  # 与编号更小的某个候选完全相同（RLE 相同）的候选数
        seen: set[str] = set()
        for c in r["candidates"]:
            key = json.dumps(c["mask"]["rle"], sort_keys=True)
            dup += key in seen
            seen.add(key)
    timing = _timing_by_id(leaf)
    m = {"experiment": "W2", "w2": cfg_doc["w2"], "model": spec.model_key, "n": len(rows), "n_with_candidates": len(with_c),
         "k": cfg_doc["candidates"]["k"], "names": names, "not_ok_by_index": {str(k): not_ok[k] for k in sorted(not_ok)},
         "n_duplicates": dup, "duplicate_rate": (dup / (len(with_c) * cfg_doc["candidates"]["k"])) if with_c else None,
         "content_sha256": candidates_digest_rows(rows),
         "wall_s": distribution(float(timing.get(r["id"], {}).get("wall_time_s", 0.0)) for r in with_c)}
    (leaf / METRICS).write_text(json.dumps(m, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (leaf / README).write_text("\n".join([
        f"# W2 · {spec.model_key} · 候选缓存 · {cfg_doc['w2']['setting']} · {spec.split}", "",
        f"- 提出：{cfg_doc['propose']['run']['leaf']}（mode = {cfg_doc['propose']['mode']}）",
        f"- 生成器（{cfg_doc['candidates']['module']}，K = {m['k']}）：{names}",
        f"- n = {m['n']}，有候选 {m['n_with_candidates']}；失败（ok = false，用框代替）按编号 {m['not_ok_by_index']}；与更小编号完全相同的候选 "
        f"{m['n_duplicates']}（{_fmt(m['duplicate_rate'])}）",
        "- candidates.jsonl 的 info 含耗时，不逐字节可复现；下游设置的运行身份用 content_sha256（不含 info）。", ""]), encoding="utf-8")


# ---------------------------------------------------------------------------
# 用候选的设置：w2a、random_pick、oracle_pick、w1f
# ---------------------------------------------------------------------------


def setting_identity(spec: W2RunSpec, cfg: W2Config, splits_doc: Mapping[str, Any], samples: Sequence[Sample], out: str | Path, *,
                     mode: str, prop_dir: Path, cand_leaf: Path, cand_rows: Sequence[Mapping[str, Any]], impl: Impl, batching: Mapping[str, Any],
                     paths: DataPaths, tiny: bool, checkpoint: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """运行身份。checkpoint（只有 w2c）：训练过的检查点的记录（src/w2/w2c.checkpoint_record），写进 config.yaml 的 checkpoint
    （scripts/w2_gate.py 据此核对 W2-c 带检查点）；其余设置没有这个键（与原来逐字节相同）。"""
    doc: dict[str, Any] = {
        "experiment": "W2", "framework": FRAMEWORK, "w2": w2_record(spec, splits_doc, samples), "model": model_record(spec),
        "image_policy": dict(spec.image_policy), "w2_config": cfg.identity(),
        "propose": {"mode": mode, "run": _run_ref(out, prop_dir)},
        "candidates": {"leaf": B._under_out(out, cand_leaf), "module": impl.candidates_module, "generators": list(cfg.generators),
                       "k": cfg.k, "content_sha256": candidates_digest_rows(cand_rows)},
        "batching": dict(batching),
        "data": {"config_sha256": {Path(p).name: _sha(Path(p)) for p in paths.config_files()}},
    }
    if spec.setting in MODEL_PICK_SETTINGS:
        sp = cfg.select_protocol_for(spec.setting, tiny=tiny)
        doc["select"] = {**sp.as_dict(), "system": cfg.select_system, "prompt": cfg.data["select"]["prompt"],
                         "method": "chat_turn_batch" if sp.thinking else "generate_batch"}
        doc["render"] = {"module": impl.render_module, "layout": cfg.layout, "style": cfg.style}
    if spec.setting == "random_pick":
        doc["random_pick"] = {"salt": cfg.random_salt}
    if checkpoint is not None:
        doc["checkpoint"] = json.loads(json.dumps(dict(checkpoint), ensure_ascii=False))
    return doc


def sample_compute(rec: Mapping[str, Any], t: Mapping[str, Any]) -> Compute:
    """一条样本的计算量：predictions 的 token 数（提出 + 挑选）与 logs/timing.jsonl 的墙钟。"""
    tk = rec["tokens"]
    k = int(rec["n_model_calls"])
    return Compute(n_calls=k, n_rounds=k, generated_tokens=int(tk["generated"]), prompt_tokens=int(tk["prompt"]) + int(tk["inserted"]),
                   visual_tokens=int(tk["visual"]), image_pixels=int(tk["image_pixels"]),
                   call_time_s=float(t.get("proposal_wall_s", 0.0)) + float(t.get("select_wall_s", 0.0))
                   ).as_sample(wall_time_s=float(t.get("wall_time_s", 0.0)))


def _rate(n: int, d: int) -> dict[str, Any]:
    return {"n": n, "of": d, "rate": (n / d) if d else None}


def aggregate_w2(setting: str, recs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """predictions.jsonl 的记录 → metrics.json 的主体（只读记录；续跑后再算一遍逐字节相同）。"""
    from src.metrics.seg import ciou as _ciou

    n = len(recs)
    m: dict[str, Any] = {"n": n, "headline": headline("refseg", recs) if recs else []}
    m["failures"] = _counts(r["failure"] for r in recs)
    wc = [r for r in recs if r["n_candidates"]]
    d = len(wc)
    m["n_with_candidates"] = d
    m["pick"] = {"final_is_best": _rate(sum(1 for r in wc if r["final_is_best"]), d),
                 "final_is_1": _rate(sum(1 for r in wc if r["final_index"] == 1), d),
                 "none": _rate(sum(1 for r in wc if r["fallback"] == FALLBACK_NONE), d),
                 "parse_failure": _rate(sum(1 for r in wc if r["fallback"] == FALLBACK_PARSE), d),
                 "rule": "分母 = 有候选的样本（提出失败的没有候选）；final_is_best = 交的候选的 IoU 等于这条样本候选里的最高 IoU（并列都算）；"
                         "none、parse_failure 都交 1 号候选"}
    m["final_index"] = {str(k): v for k, v in sorted(Counter(r["final_index"] for r in wc).items())}
    m["best_index"] = {str(k): v for k, v in sorted(Counter(r["best_index"] for r in wc).items())}
    k = max((r["n_candidates"] for r in wc), default=0)
    per = {}
    for i in range(k):
        st = [(r["candidates"][i]["inter"], r["candidates"][i]["union"], False) for r in wc]
        per[str(i + 1)] = {"name": wc[0]["candidates"][i]["name"], "cIoU_on_samples_with_candidates": _ciou(st),
                           "n_not_ok": sum(1 for r in wc if not r["candidates"][i]["ok"])}
    m["candidates"] = per
    m["oracle_cIoU_on_samples_with_candidates"] = _ciou([(r["candidates"][r["best_index"] - 1]["inter"],
                                                          r["candidates"][r["best_index"] - 1]["union"], False) for r in wc])
    for key in ("prompt", "generated", "inserted", "visual"):
        m.setdefault("tokens", {})[key] = distribution(r["tokens"][key] for r in recs)
    m["n_model_calls"] = distribution(r["n_model_calls"] for r in recs)
    m["n_tool_calls"] = distribution(r["n_tool_calls"] for r in recs)
    if setting in MODEL_PICK_SETTINGS:
        sel = [r["select"] for r in wc]
        m["select"] = {"method": _counts(s["method"] for s in sel), "thinking_end": {k: v for k, v in sorted(Counter(str(s.get("thinking_end")) for s in sel).items())},
                       "parse_failure": _counts(s["parse_failure"] for s in sel if s["parse_failure"] is not None),
                       "answer": {k: v for k, v in sorted(Counter("parse_failure" if s["answer"] is None else str(s["answer"])
                                                                  for s in sel).items())},
                       "stopped": {k: v for k, v in sorted(Counter(str(s.get("stopped")) for s in sel).items())}}
    return m


def _readme(spec: W2RunSpec, cfg_doc: Mapping[str, Any], m: Mapping[str, Any]) -> str:
    p = m["pick"]
    lines = [f"# W2 · {spec.model_key} · refseg · {spec.setting} · {spec.split}", "",
             f"- 模型：{spec.repo}@{spec.revision}（适配器 {spec.adapter_model_key}）",
             f"- 配置：{cfg_doc['w2_config']['config']} sha256 {cfg_doc['w2_config']['config_sha256'][:16]}…（frozen = "
             f"{cfg_doc['w2_config']['frozen']}）；提出 mode = {cfg_doc['propose']['mode']}（{cfg_doc['propose']['run']['leaf']}）",
             f"- 候选：{cfg_doc['candidates']['leaf']}（{cfg_doc['candidates']['module']}，K = {cfg_doc['candidates']['k']}）"]
    if cfg_doc.get("checkpoint"):
        ck = cfg_doc["checkpoint"]
        lines.append(f"- 检查点：{ck.get('path')}（manifest sha256 {str(ck.get('manifest_sha256'))[:16]}…；训练 {ck.get('training')}）")
    if "select" in cfg_doc:
        s = cfg_doc["select"]
        lines.append(f"- 挑选：{s['method']}，思考 {s['thinking']}（预算 {s['thinking_budget']}），作答 {s['answer_budget']}；渲染 "
                     f"{cfg_doc['render']['module']}（layout = {cfg_doc['render']['layout']}）；批 {cfg_doc['batching']}")
    lines += ["", "## 标题指标", ""]
    for h in m["headline"]:
        ci = h.get("ci")
        lines.append(f"- {h['name']}：{_fmt(h['value'])}，95% CI {'—' if not ci else '[' + _fmt(ci['low']) + ', ' + _fmt(ci['high']) + ']'}"
                     f"，n = {h['n']}")
    lines += ["", "## 挑选", "",
              f"- 有候选 {m['n_with_candidates']} / {m['n']}；失败 {m['failures']}",
              f"- 交的是最优候选 {_fmt(p['final_is_best']['rate'])}；交 1 号 {_fmt(p['final_is_1']['rate'])}；\"都不对\" "
              f"{_fmt(p['none']['rate'])}；解析失败 {_fmt(p['parse_failure']['rate'])}",
              f"- 最优候选（oracle）在有候选的样本上的 cIoU：{_fmt(m['oracle_cIoU_on_samples_with_candidates'])}",
              f"- 各候选的 cIoU：{ {k: (v['name'], _fmt(v['cIoU_on_samples_with_candidates'])) for k, v in m['candidates'].items()} }", "",
              "## 文件", "", "- predictions.jsonl（逐条：提出、各候选的 IoU、挑选原文、挑中的编号、fallback、最终 mask 的 RLE、计分）；metrics.json；"
              "compute.json；config.yaml；meta.json；logs/（timing.jsonl、run_*.json）。", ""]
    return "\n".join(lines)


def run_candidate_setting(adapter: Any, spec: W2RunSpec, cfg: W2Config, samples: Sequence[Sample], paths: DataPaths,
                          out: str | Path, *, splits_doc: Mapping[str, Any], mode: str, prop_dir: Path,
                          prop_recs: Sequence[Mapping[str, Any]], cand_leaf: Path, cand_rows: Mapping[str, Mapping[str, Any]],
                          impl: Impl, prompts: PromptConfig, batch_size: int, chunk_size: int, max_batch_tokens: int | None,
                          resume: bool, argv: Sequence[str] = (), log: Callable[[str], None] | None = None,
                          warmup: bool = True, checkpoint: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """w2a、random_pick、oracle_pick、w1f（refseg）在一个划分上跑，返回 metrics。adapter 只有 w2a 用。
    w2c（W2-c，DEV-0061）：同 w2a，只是挑选关思考（W2Config.select_protocol_for），adapter 是加载了检查点的模型，checkpoint 必须给
    （记进 config.yaml）；其余设置不能给 checkpoint。"""
    log = log or (lambda m: print(m, file=sys.stderr, flush=True))
    if spec.task != "refseg" or spec.setting not in CANDIDATE_SETTINGS + (W2C_SETTING,):
        raise ValueError(f"{spec.task}/{spec.setting} 不是用候选的设置")
    if (spec.setting == W2C_SETTING) != (checkpoint is not None):
        raise ValueError(f"{spec.setting}：w2c 必须带训练过的检查点，其余设置不能带（收到 checkpoint = {checkpoint is not None}）")
    if spec.setting == W2C_SETTING and cfg.select_system is not None:
        raise ValueError("w2c 的挑选关思考，走 build_inputs + generate_batch，没有 system 轮：configs/w2.json 的 select.system 必须是 null")
    tiny = spec.tiny
    ids = [s.id for s in samples]
    rows = [cand_rows[i] for i in ids]
    batching = {"batch_size": batch_size, "chunk_size": chunk_size, "max_batch_tokens": max_batch_tokens}
    cfg_doc = setting_identity(spec, cfg, splits_doc, samples, out, mode=mode, prop_dir=prop_dir, cand_leaf=cand_leaf, cand_rows=rows,
                               impl=impl, batching=batching if spec.setting in MODEL_PICK_SETTINGS else {"chunk_size": chunk_size},
                               paths=paths, tiny=tiny, checkpoint=checkpoint)
    leaf = w2_leaf(out, spec.model_key, spec.task, spec.setting, spec.split)
    _prepare_leaf(leaf, cfg_doc, [cfg.path, *_splits_files(splits_doc), *paths.config_files()], resume=resume)
    props = _proposals_by_id(mode, samples, prop_recs, prompts, spec, cfg)
    prop_t = _timing_by_id(prop_dir)
    cand_t = _timing_by_id(cand_leaf)
    plan = batch_plan([None] * len(ids), chunk_size)
    what = f"W2 {spec.model_key} refseg/{spec.setting}/{spec.split}"
    keep, trimmed = _resume_point(leaf / PREDICTIONS, ids, lambda r: r["id"], plan, log, what)
    todo = [(a, b) for a, b in plan if a >= keep]
    run_k, run_path, run_log = _start_run_log(leaf, argv, n_done_before=keep, n_remaining=len(ids) - keep, **batching,
                                              trimmed_to_batch_boundary=trimmed)
    t_all = time.perf_counter()
    protocol = cfg.select_protocol_for(spec.setting, tiny=tiny) if spec.setting in MODEL_PICK_SETTINGS else None
    if spec.setting in MODEL_PICK_SETTINGS:
        if impl.render is None:
            raise ValueError(f"{spec.setting} 要渲染（render_selection）")
        if todo and warmup and hasattr(adapter, "warmup"):
            run_log["warmup_wall_s"] = adapter.warmup(load_images(samples[keep]), batch_size=min(batch_size, len(ids) - keep))
    for n_chunk, (a, b) in enumerate(todo):
        chunk = list(samples[a:b])
        images = {s.id: load_images(s)[0] for s in chunk}
        outcomes: dict[str, Any] = {}
        render_s: dict[str, float] = {}
        if spec.setting in MODEL_PICK_SETTINGS:
            items = []
            for s in chunk:
                row = cand_rows[s.id]
                if row["candidates"] is None:
                    continue
                t0 = time.perf_counter()
                cands = selection_candidates(row, impl.Candidate)  # 1 号候选失败时画空叠加（裁定 A）
                imgs = impl.render(images[s.id], cands, layout=cfg.layout, style=cfg.style)
                if not isinstance(imgs, list) or not imgs or any(not hasattr(im, "size") for im in imgs):
                    raise RuntimeError(f"{s.id}：render_selection 应返回非空的 PIL 图列表，收到 {type(imgs).__name__}")
                render_s[s.id] = time.perf_counter() - t0
                items.append(SelectItem(s.id, tuple(imgs), cfg.select_prompt(len(cands), s.fields["expression"]), len(cands)))
            outcomes = {o.id: o for o in run_selection(adapter, items, protocol, system=cfg.select_system, batch_size=batch_size,
                                                       max_batch_tokens=max_batch_tokens, decoding=prompts.decoding(),
                                                       order=prompts.order)}
        lines, tlines = [], []
        for s in chunk:
            pw = float(prop_t.get(s.id, {}).get("wall_time_s", 0.0))
            o = outcomes.get(s.id)
            rec, _ = candidate_record(spec.setting, s, gt_mask(s), cand_rows[s.id], salt=cfg.random_salt, proposal_rec=props[s.id],
                                      proposal_wall_s=pw, outcome=o)
            rec = {"id": rec["id"], "w2_split": spec.split, **{k: v for k, v in rec.items() if k != "id"}}
            cw = float(cand_t.get(s.id, {}).get("wall_time_s", 0.0))
            sw = 0.0 if o is None else float(o.wall_time_s)
            lines.append(_jsonl_line(rec))
            tlines.append(_jsonl_line({"id": s.id, "run": run_k, "chunk_index": n_chunk, "proposal_wall_s": pw, "candidates_wall_s": cw,
                                       "render_s": render_s.get(s.id, 0.0), "select_wall_s": sw,
                                       "wall_time_s": pw + cw + render_s.get(s.id, 0.0) + sw}))
        _append(leaf / TIMING, tlines)
        _append(leaf / PREDICTIONS, lines)
        log(f"[{what}] {b}/{len(ids)}")
    recs = read_jsonl_tolerant(leaf / PREDICTIONS, repair=False)
    if [r["id"] for r in recs] != ids[: len(recs)]:
        raise RuntimeError(f"{leaf / PREDICTIONS} 的 id 不是样本顺序的前缀")
    body = aggregate_w2(spec.setting, recs)
    metrics = {"experiment": "W2", "w2": cfg_doc["w2"], "model": spec.model_key, "revision": spec.revision,
               "w2_config_sha256": cfg.sha256, "propose_mode": mode, **body, "complete": len(recs) == len(ids), "n_expected": len(ids)}
    (leaf / METRICS).write_text(json.dumps(metrics, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    timing = _timing_by_id(leaf)
    write_compute_json(leaf / COMPUTE, merge(sample_compute(r, timing.get(r["id"], {})) for r in recs))
    (leaf / README).write_text(_readme(spec, cfg_doc, metrics), encoding="utf-8")
    _finish_run_log(run_path, run_log, n_batches=len(todo), t_all=t_all)
    return metrics


# ---------------------------------------------------------------------------
# ownbox；按输出类型路由的 w2a（refexp_box、count）
# ---------------------------------------------------------------------------


def run_ownbox(spec: W2RunSpec, cfg: W2Config, samples: Sequence[Sample], out: str | Path, *, splits_doc: Mapping[str, Any],
               box_dir: Path, box_recs: Sequence[Mapping[str, Any]], prompts: PromptConfig, resume: bool,
               argv: Sequence[str] = ()) -> dict[str, Any]:
    """"自己的框"：E1 refexp_box 的框光栅化（W1 的 own_box_record 原样），写 W2 布局的目录，返回 metrics。"""
    if spec.task != "refseg":
        raise ValueError("ownbox 只对 refseg")
    posthoc = load_posthoc_config(prompts, POSTHOC_PATH)
    gts = {s.id: gt_mask(s) for s in samples}
    matched = B.matched_boxes(samples, box_recs, prompts, spec.family, gts)
    w2 = w2_record(spec, splits_doc, samples)
    cfg_doc = {"experiment": "W2", "framework": FRAMEWORK, "w2": w2, "model": model_record(spec), "box_run": _run_ref(out, box_dir),
               "raster_rule": posthoc.own_box["raster_rule"], "posthoc_config_sha256": posthoc.sha256}
    leaf = w2_leaf(out, spec.model_key, spec.task, spec.setting, spec.split)
    B._prepare_derived(leaf, cfg_doc, [posthoc.path, *_splits_files(splits_doc)], resume=resume)
    recs = []
    for s in samples:
        b = matched[s.id]
        r = B.own_box_record(b, gts[s.id])
        r.update(instruction=b["instruction"], image_size=b["image_size"], box_valid=is_valid(BOX_TASK, b))
        recs.append(r)
    (leaf / PREDICTIONS).write_text("".join(_jsonl_line(r) for r in recs), encoding="utf-8")
    m = {"experiment": "W2", "w2": w2, "model": spec.model_key, "revision": spec.revision, "n": len(recs),
         "headline": headline("refseg", recs), "n_box_failed": sum(1 for r in recs if r["failure"] is not None),
         "n_box_valid": sum(1 for r in recs if r["box_valid"]), "box_run": cfg_doc["box_run"], "complete": True,
         "n_expected": len(samples)}
    (leaf / METRICS).write_text(json.dumps(m, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    if (box_dir / COMPUTE).exists():  # 计算量就是框的那次运行
        (leaf / COMPUTE).write_bytes((box_dir / COMPUTE).read_bytes())
    (leaf / README).write_text("\n".join([
        f"# W2 · {spec.model_key} · refseg · ownbox · {spec.split}", "",
        f"- \"自己的框\"：{cfg_doc['box_run']['leaf']} 的框（E1 refexp_box）光栅化；n = {m['n']}，框解析失败 {m['n_box_failed']} 条（空 mask）",
        *[f"- {h['name']}：{_fmt(h['value'])}" for h in m["headline"]], ""]), encoding="utf-8")
    (leaf / "logs" / "derive.json").write_text(json.dumps({"argv": list(argv), "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                                                                          time.gmtime())},
                                                          indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return m


def run_routed(spec: W2RunSpec, cfg: W2Config, samples: Sequence[Sample], out: str | Path, *, splits_doc: Mapping[str, Any],
               src_dir: Path, resume: bool, argv: Sequence[str] = ()) -> dict[str, Any]:
    """w2a 遇到要框或数字的任务（refexp_box、count）：按输出类型路由到 E1 单轮，就是同一个划分上 baseline_e1 的结果（原样复制）。"""
    kind = OUTPUT_KIND[spec.task]
    if kind == "mask":
        raise ValueError("要 mask 的任务不路由")
    w2 = w2_record(spec, splits_doc, samples)
    cfg_doc = {"experiment": "W2", "framework": FRAMEWORK, "w2": w2, "model": model_record(spec), "w2_config": cfg.identity(),
               "routing": {"output_kind": kind, "rule": "按输出类型路由（D113）：要框或数字的任务按 E1 协议单轮作答，不用工具；"
                           "结果就是同一个划分上 baseline_e1 的那一份", "source": _run_ref(out, src_dir)}}
    leaf = w2_leaf(out, spec.model_key, spec.task, spec.setting, spec.split)
    B._prepare_derived(leaf, cfg_doc, [cfg.path, *_splits_files(splits_doc)], resume=resume)
    (leaf / PREDICTIONS).write_bytes((src_dir / PREDICTIONS).read_bytes())
    if (src_dir / COMPUTE).exists():
        (leaf / COMPUTE).write_bytes((src_dir / COMPUTE).read_bytes())
    e1m = json.loads((src_dir / METRICS).read_text(encoding="utf-8"))
    n = len(read_jsonl_tolerant(leaf / PREDICTIONS, repair=False))
    m = {"experiment": "W2", "w2": w2, "model": spec.model_key, "revision": spec.revision, "n": n, "headline": e1m.get("headline", []),
         "routing": cfg_doc["routing"], "e1_metrics": e1m, "complete": n == len(samples), "n_expected": len(samples)}
    (leaf / METRICS).write_text(json.dumps(m, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (leaf / README).write_text("\n".join([
        f"# W2 · {spec.model_key} · {spec.task} · w2a（按输出类型路由到 E1 单轮）· {spec.split}", "",
        f"- 结果 = {cfg_doc['routing']['source']['leaf']}（baseline_e1）原样；n = {n}",
        *[f"- {h['name']}：{_fmt(h.get('value'))}" for h in m["headline"]], ""]), encoding="utf-8")
    return m


# ---------------------------------------------------------------------------
# 身份摘要（w2_gate 用）
# ---------------------------------------------------------------------------


def read_identity(leaf: str | Path) -> dict[str, Any]:
    """一个 W2 输出目录的身份摘要：模型、revision、检查点、任务、设置、划分、W2 配置（是否冻结）、提出方式，以及是否写完。"""
    leaf = Path(leaf)
    cfg = yaml.safe_load((leaf / CONFIG).read_text(encoding="utf-8"))
    w2 = cfg.get("w2") or ((cfg.get("data") or {}).get("selection_stats") or {}).get("w2")
    if not w2:
        raise ValueError(f"{leaf}：config.yaml 里没有 W2 一节（不是 W2 的输出目录）")
    m = json.loads((leaf / METRICS).read_text(encoding="utf-8")) if (leaf / METRICS).exists() else {}
    return {"leaf": str(leaf), "model": cfg["model"]["key"], "revision": cfg["model"]["revision"], "tiny": cfg["model"].get("tiny"),
            "checkpoint": cfg.get("checkpoint"), "task": w2["task"], "setting": w2["setting"], "split": w2["split"],
            "split_sha256": w2["split_sha256"], "n_samples": w2["n_samples"], "limit": w2.get("limit"),
            "w2_config": cfg.get("w2_config"), "propose_mode": (cfg.get("propose") or {}).get("mode"),
            "candidates": cfg.get("candidates"), "complete": m.get("complete")}


__all__ = [
    "CANDIDATES",
    "DEFAULT_CANDIDATES_MODULE",
    "DEFAULT_RENDER_MODULE",
    "Impl",
    "W2RunSpec",
    "aggregate_w2",
    "candidates_identity",
    "candidates_setting",
    "e1_run_complete",
    "ensure_candidates",
    "ensure_e1_run",
    "ensure_points_run",
    "load_impl",
    "read_identity",
    "run_baseline_e1",
    "run_candidate_setting",
    "run_ownbox",
    "run_points_proposal",
    "run_routed",
    "sample_compute",
    "setting_identity",
    "w2_leaf",
    "w2_record",
]
