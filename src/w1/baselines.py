"""W1 的对照与基线（DEV-0051；D103 §二.6、§三）：同一个模型、同一批句子、E1 的协议（关思考、贪心、E1 冻结的批大小）。

输出布局与 W1 的回路相同（<out>/<模型>/<任务>/<设置>/<划分>/，src/w1/runner.w1_leaf），所以能按 id 配对：
- baseline_e1（W1-0）：E1 零样本。W1 的任务 → E1 的任务：refseg → refexp_poly（多边形），refexp_box → refexp_box，count → count。
  走 E1 的运行器原样（src/e1/runner.run_task(leaf=…)；提示、解析、评分、批配置、续跑都是 E1 的），目录是 E1 的格式
  （predictions.jsonl、metrics.json、compute.json、config.yaml……），config.yaml 的 data.selection_stats.w1 记 W1 的划分与设置。
- baseline_ownbox（"自己的框"，只对 refseg）：同一个模型在同一个划分上的 W1-0 refexp_box 预测（<out>/<模型>/refexp_box/baseline_e1/<划分>，
  没写完就先按 E1 的协议跑完），按 configs/e1_posthoc.json 的 own_box.raster_rule 光栅化：四个角当多边形，经 E1 多边形的同一个光栅化
  与 GT 算 inter / union（scripts/summarize_e1_posthoc.own_box_record 原样；按 id 对上并核对同一句话用同一脚本的 match_boxes）。
  只读框的运行，不跑模型。
- baseline_poly_structured（只对 refseg）：E1 结构强制的多边形，变体 poly_structured（configs/e1_posthoc.json 的 format_forced_poly），
  src/e1/posthoc_runner.run_forced_poly_task(sample_subset="w1:<划分>", leaf=…)，目录是 E1 事后读数的格式。
- w1f（W1-F，只对 refseg）：固定流程，没有 LLM。"自己的框"（同上，原图绝对像素 xyxy）直接交给工作台的同一个 GrabCut 函数
  src.w1.workbench.grabcut_mask（框初始化；迭代次数取工作台的 GRABCUT_ITERATIONS，随机数种子固定，D106），返回的 mask 就是答案。
  框解析失败记那个失败原因、GrabCut 报错（框里或框外没有像素等）记 tool_error，都按空 mask 计分。
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import yaml

from src.accounting import Compute, merge, write_compute_json
from src.constants import BOOTSTRAP_SEED
from src.data.records import Sample
from src.e1.posthoc import FORCED_POLY_SUBSET, POSTHOC_PATH, load_posthoc_config
from src.e1.posthoc_runner import _finish_run_log, _prepare_leaf, _resume_point, _start_run_log, run_forced_poly_task
from src.e1.prompts import DEFAULT_COUNT_QUESTIONS, PROMPTS_PATH, PromptConfig, load_prompt_config
from src.e1.runner import (
    COMPUTE,
    CONFIG,
    METRICS,
    PREDICTIONS,
    README,
    TIMING,
    RunSpec,
    _append,
    _identity,
    _jsonl_line,
    batch_plan,
    load_images,
    read_jsonl_tolerant,
    run_task,
)
from src.e1.scoring import gt_mask, is_valid, metric_of
from src.e1.tasks import DataPaths, Selection
from src.executors.raster import mask_to_rle
from src.metrics.seg import mask_inter_union, sample_iou
from src.runlog import write_run_metadata
from src.w1.config import W1Config
from src.w1.data import W1_SPLITS_PATH
from src.w1.runner import W1RunSpec, _rel, model_record, w1_leaf, w1_record
from src.w1.scoring import headline

REPO_ROOT = Path(__file__).resolve().parents[2]
SUMMARY_SCRIPT = REPO_ROOT / "scripts" / "summarize_e1_posthoc.py"
FAILURE_TOOL_ERROR = "tool_error"


@functools.lru_cache(maxsize=1)
def _summary():
    """scripts/summarize_e1_posthoc.py（own_box_record、match_boxes 的唯一定义；按文件加载，sys.path 原样恢复）。"""
    saved = list(sys.path)
    spec = importlib.util.spec_from_file_location("_w1_summarize_e1_posthoc", SUMMARY_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.path[:] = saved
    return mod


def own_box_record(box_rec: Mapping[str, Any], gt: np.ndarray) -> dict[str, Any]:
    """一条"自己的框"（多边形对照）：scripts/summarize_e1_posthoc.own_box_record("refexp_poly", …) 原样。"""
    return _summary().own_box_record("refexp_poly", box_rec, gt)


def e1_spec(entry: Any, adapter: Any, policy: Any, task: str, subset: str, batch_size: int, *, tiny: bool,
            limit: int | None) -> RunSpec:
    return RunSpec(model_key=entry.key, family=entry.family, repo=entry.repo, revision=entry.revision,
                   adapter_model_key=str(getattr(adapter, "model_key", entry.key)), task=task, subset=subset, policy_tag="default",
                   policy={"type": type(policy).__name__, **policy.as_dict()}, batch_size=batch_size, limit=limit, tiny=tiny,
                   map_ci_resamples=0, count_questions=DEFAULT_COUNT_QUESTIONS)


def w1_selection(e1_task: str, subset: str, samples: Sequence[Sample], w1: Mapping[str, Any]) -> Selection:
    """W1 的划分包成 E1 的 Selection：selection_stats.w1 记 W1 的划分与设置（进 config.yaml，续跑核对）。"""
    return Selection(e1_task, subset, list(samples), True, {"w1": dict(w1), "n": len(samples)})


# ---------------------------------------------------------------------------
# W1-0：E1 零样本；结构强制的多边形
# ---------------------------------------------------------------------------


def run_baseline_e1(adapter: Any, entry: Any, policy: Any, spec: W1RunSpec, cfg: W1Config, samples: Sequence[Sample],
                    paths: DataPaths, out: str | Path, *, splits_doc: Mapping[str, Any], batch_size: int, resume: bool,
                    argv: Sequence[str] = (), prompts: PromptConfig | None = None, warmup: bool = True) -> dict[str, Any]:
    """W1-0（E1 零样本）在 W1 的划分上跑，返回 E1 的 metrics。"""
    prompts = prompts or load_prompt_config(PROMPTS_PATH)
    e1_task = cfg.e1_task(spec.task)
    rs = e1_spec(entry, adapter, policy, e1_task, "eval", batch_size, tiny=spec.tiny, limit=spec.limit)
    sel = w1_selection(e1_task, "eval", samples, w1_record(spec, splits_doc, samples))
    leaf = w1_leaf(out, spec.model_key, spec.task, spec.setting, spec.split)
    return run_task(adapter, rs, sel, prompts, paths, out, resume=resume, argv=argv, leaf=leaf, warmup=warmup)


def run_baseline_poly_structured(adapter: Any, entry: Any, policy: Any, spec: W1RunSpec, cfg: W1Config,
                                 samples: Sequence[Sample], paths: DataPaths, out: str | Path, *,
                                 splits_doc: Mapping[str, Any], batch_size: int, resume: bool, argv: Sequence[str] = (),
                                 prompts: PromptConfig | None = None, warmup: bool = True) -> dict[str, Any]:
    """E1 结构强制的多边形（poly_structured）在 W1 的划分上跑，返回 E1 事后读数的 metrics。"""
    if spec.task != "refseg":
        raise ValueError("结构强制的多边形只对 refseg")
    prompts = prompts or load_prompt_config(PROMPTS_PATH)
    posthoc = load_posthoc_config(prompts, POSTHOC_PATH)
    rs = e1_spec(entry, adapter, policy, "refexp_poly", FORCED_POLY_SUBSET, batch_size, tiny=spec.tiny, limit=spec.limit)
    base = f"w1:{spec.split}"
    sel = w1_selection("refexp_poly", base, samples, w1_record(spec, splits_doc, samples))
    leaf = w1_leaf(out, spec.model_key, spec.task, spec.setting, spec.split)
    return run_forced_poly_task(adapter, rs, sel, prompts, posthoc, paths, out, variant=cfg.baselines["poly_structured"]["variant"],
                                resume=resume, argv=argv, warmup=warmup, sample_subset=base, leaf=leaf)


# ---------------------------------------------------------------------------
# "自己的框"
# ---------------------------------------------------------------------------


def box_leaf(out: str | Path, spec: W1RunSpec, cfg: W1Config) -> Path:
    ob = cfg.baselines["ownbox"]
    return w1_leaf(out, spec.model_key, ob["box_task"], ob["box_setting"], spec.split)


def box_run_complete(leaf: Path, samples: Sequence[Sample], spec: W1RunSpec) -> bool:
    """框的运行（W1-0 refexp_box）已经写完、是同一个模型、同一批样本：predictions 的 id 恰是样本顺序，config.yaml 的模型与 W1 一节相符。"""
    if not (leaf / CONFIG).exists() or not (leaf / PREDICTIONS).exists():
        return False
    recs = read_jsonl_tolerant(leaf / PREDICTIONS, repair=False)
    if [r["id"] for r in recs] != [s.id for s in samples]:
        return False
    c = yaml.safe_load((leaf / CONFIG).read_text(encoding="utf-8"))
    w1 = (c.get("data") or {}).get("selection_stats", {}).get("w1") or {}
    return (c["model"]["key"], c["model"]["revision"], bool(c["model"].get("tiny")), w1.get("split")) == \
        (spec.model_key, spec.revision, spec.tiny, spec.split)


def ensure_box_run(get_adapter: Callable[[], tuple[Any, Any, Any]], spec: W1RunSpec, cfg: W1Config, samples: Sequence[Sample],
                   paths: DataPaths, out: str | Path, *, splits_doc: Mapping[str, Any], batch_size: int,
                   argv: Sequence[str] = (), prompts: PromptConfig | None = None, log: Callable[[str], None] | None = None,
                   warmup: bool = True) -> tuple[Path, list[dict[str, Any]]]:
    """"自己的框"要的框：W1-0 refexp_box 在同一个划分上的运行。已写完就直接读（不加载模型）；否则 get_adapter() 取 (entry, policy, 适配器)，
    按 E1 的协议跑完（续跑语义同 E1：身份必须相同）。返回 (框的目录, 框的记录)。"""
    ob = cfg.baselines["ownbox"]
    leaf = box_leaf(out, spec, cfg)
    if not box_run_complete(leaf, samples, spec):
        entry, policy, ad = get_adapter()
        bspec = W1RunSpec(**{**spec.__dict__, "task": ob["box_task"], "setting": ob["box_setting"]})
        if log:
            log(f"[W1 {spec.model_key}] 先跑框：{leaf}")
        run_baseline_e1(ad, entry, policy, bspec, cfg, samples, paths, out, splits_doc=splits_doc, batch_size=batch_size,
                        resume=True, argv=argv, prompts=prompts, warmup=warmup)
        if not box_run_complete(leaf, samples, spec):
            raise RuntimeError(f"{leaf}：框的运行跑完之后仍不完整")
    return leaf, read_jsonl_tolerant(leaf / PREDICTIONS, repair=False)


def matched_boxes(samples: Sequence[Sample], box_recs: Sequence[Mapping[str, Any]], prompts: PromptConfig, family: str,
                  gts: Mapping[str, np.ndarray]) -> dict[str, Mapping[str, Any]]:
    """按 id 对上框记录并核对同一句话（scripts/summarize_e1_posthoc.match_boxes）；对不上就报错（同一次运行的框，不该有）。"""
    ok, bad = _summary().match_boxes(samples, box_recs, prompts, family, gts)
    n_bad = {k: len(v) for k, v in bad.items() if v}
    if n_bad:
        raise ValueError(f"框记录与样本对不上：{n_bad}（例如 {[v[:3] for v in bad.values() if v]}）")
    return ok


def _under_out(out: str | Path, p: str | Path) -> str:
    """输出目录下的路径记成 <out>/…（运行身份与机器、--out 的写法无关）。"""
    try:
        return "<out>/" + str(Path(p).resolve().relative_to(Path(out).resolve()))
    except ValueError:
        return str(p)


def _derived_identity(spec: W1RunSpec, w1: Mapping[str, Any], out: str | Path, box_dir: Path,
                      extra: Mapping[str, Any]) -> dict[str, Any]:
    pred = box_dir / PREDICTIONS
    return {"experiment": "W1", "framework": "src/w1/baselines（DEV-0051）", "w1": dict(w1), "model": model_record(spec),
            "box_run": {"leaf": _under_out(out, box_dir), "predictions_sha256": hashlib.sha256(pred.read_bytes()).hexdigest(),
                        "config_sha256": hashlib.sha256((box_dir / CONFIG).read_bytes()).hexdigest()},
            **dict(extra)}


def _prepare_derived(leaf: Path, cfg_doc: Mapping[str, Any], files: Sequence[str | Path], *, resume: bool) -> None:
    """只由别的运行导出的目录（"自己的框"）：新目录写 config.yaml 与 meta.json；已有内容要 --resume 且身份完全相同，整份重写。"""
    if leaf.exists() and any(leaf.iterdir()):
        if not resume:
            raise FileExistsError(f"{leaf} 已存在且非空；续跑请加 --resume")
        old = yaml.safe_load((leaf / CONFIG).read_text(encoding="utf-8"))
        if _identity(old) != _identity(cfg_doc):
            raise ValueError(f"{leaf}：配置与已有的不同，拒绝混写")
    else:
        leaf.mkdir(parents=True, exist_ok=True)
        write_run_metadata(leaf, config=cfg_doc, seed=BOOTSTRAP_SEED, data_files=list(files), repo_root=REPO_ROOT)
    (leaf / "logs").mkdir(exist_ok=True)


def _fmt(v: Any) -> str:
    return "null" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))


def _headline_lines(m: Mapping[str, Any]) -> list[str]:
    out = []
    for h in m["headline"]:
        ci = h.get("ci")
        cs = "—" if not ci else "[" + _fmt(ci["low"]) + ", " + _fmt(ci["high"]) + "]"
        out.append(f"- {h['name']}：{_fmt(h['value'])}，95% CI {cs}，n = {h['n']}")
    return out


def run_baseline_ownbox(spec: W1RunSpec, cfg: W1Config, samples: Sequence[Sample], paths: DataPaths, out: str | Path, *,
                        splits_doc: Mapping[str, Any], box_dir: Path, box_recs: Sequence[Mapping[str, Any]], resume: bool,
                        argv: Sequence[str] = (), prompts: PromptConfig | None = None) -> dict[str, Any]:
    """"自己的框"：框光栅化后与 GT 算 cIoU、gIoU（规则见模块说明），写 W1 布局的目录，返回 metrics。"""
    if spec.task != "refseg":
        raise ValueError("\"自己的框\"只对 refseg")
    prompts = prompts or load_prompt_config(PROMPTS_PATH)
    posthoc = load_posthoc_config(prompts, POSTHOC_PATH)
    gts = {s.id: gt_mask(s) for s in samples}
    matched = matched_boxes(samples, box_recs, prompts, spec.family, gts)
    w1 = w1_record(spec, splits_doc, samples)
    cfg_doc = _derived_identity(spec, w1, out, box_dir, {"raster_rule": posthoc.own_box["raster_rule"],
                                                    "posthoc_config_sha256": posthoc.sha256})
    leaf = w1_leaf(out, spec.model_key, spec.task, spec.setting, spec.split)
    _prepare_derived(leaf, cfg_doc, [posthoc.path, *([] if splits_doc.get("tiny") else [W1_SPLITS_PATH])], resume=resume)
    recs = []
    for s in samples:
        b = matched[s.id]
        r = own_box_record(b, gts[s.id])
        r.update(instruction=b["instruction"], image_size=b["image_size"], box_valid=is_valid("refexp_box", b))
        recs.append(r)
    (leaf / PREDICTIONS).write_text("".join(_jsonl_line(r) for r in recs), encoding="utf-8")
    m = {"experiment": "W1", "w1": w1, "model": spec.model_key, "revision": spec.revision, "n": len(recs),
         "headline": headline("refseg", recs), "n_box_failed": sum(1 for r in recs if r["failure"] is not None),
         "n_box_valid": sum(1 for r in recs if r["box_valid"]), "box_run": cfg_doc["box_run"], "complete": True,
         "n_expected": len(samples)}
    (leaf / METRICS).write_text(json.dumps(m, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    box_compute = box_dir / COMPUTE
    if box_compute.exists():  # 计算量就是框的那次运行（这里不跑模型）
        (leaf / COMPUTE).write_bytes(box_compute.read_bytes())
    (leaf / README).write_text("\n".join([
        f"# W1 · {spec.model_key} · refseg · baseline_ownbox · {spec.split}", "",
        f"- \"自己的框\"：{_under_out(out, box_dir)} 的框（W1-0 refexp_box，E1 的协议）光栅化（{posthoc.own_box['raster_rule'][:60]}…）",
        f"- n = {len(recs)}；框解析失败 {m['n_box_failed']} 条（空 mask）；框有效 {m['n_box_valid']} 条", "", "## 标题指标", "",
        *_headline_lines(m), "", "- compute.json 是框那次运行的计算量（这里没有模型调用）。", ""]), encoding="utf-8")
    run_log = {"argv": list(argv), "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (leaf / "logs" / "derive.json").write_text(json.dumps(run_log, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return m


# ---------------------------------------------------------------------------
# W1-F：自己的框 → GrabCut（没有 LLM）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GrabCut:
    """W1-F 用的 GrabCut：函数（签名同 src.w1.workbench.grabcut_mask：image_rgb, *, box_abs, iterations）、迭代次数与记进运行身份的名字。"""

    fn: Callable[..., np.ndarray]
    iterations: int
    name: str


def workbench_grabcut() -> GrabCut:
    """工作台的同一个 GrabCut（DEV-0050，D106）：src.w1.workbench.grabcut_mask，迭代次数 GRABCUT_ITERATIONS。按需 import。"""
    from src.w1 import workbench as WB

    return GrabCut(WB.grabcut_mask, int(WB.GRABCUT_ITERATIONS), "src.w1.workbench.grabcut_mask")


def _grabcut_errors() -> tuple[type[BaseException], ...]:
    import cv2

    return (ValueError, cv2.error)  # grabcut_mask 的约定：参数不合法、某一类没有像素抛 ValueError；OpenCV 内部失败抛 cv2.error


def w1f_episode(sample: Sample, box_rec: Mapping[str, Any], image: Any, grabcut: GrabCut) -> tuple[dict[str, Any], dict[str, Any]]:
    """一条 W1-F：返回 (predictions 的记录, 计时)。"""
    w, h = image.size
    box = (box_rec.get("parsed") or {}).get("box")
    rec: dict[str, Any] = {"id": sample.id, "source": sample.source, "split": sample.split, "box": box,
                           "box_failure": box_rec["failure"], "failure": None, "error": None, "answer": None}
    times: list[float] = []
    mask = None
    if box is None:
        rec["failure"] = box_rec["failure"]
    else:
        t0 = time.perf_counter()
        try:
            mask = grabcut.fn(np.asarray(image.convert("RGB")), box_abs=[float(v) for v in box], iterations=grabcut.iterations)
        except _grabcut_errors() as e:
            rec.update(failure=FAILURE_TOOL_ERROR, error=f"{type(e).__name__}: {e}")
        times.append(time.perf_counter() - t0)
        if mask is not None:
            mask = np.asarray(mask)
            if mask.dtype != bool or mask.shape != (h, w):
                raise RuntimeError(f"{sample.id}：GrabCut 应返回 (H, W) = {(h, w)} 的 bool 数组，收到 {mask.dtype} {mask.shape}")
            rec["answer"] = {"source": grabcut.name, "kind": "mask",
                             "mask": {"rle": mask_to_rle(mask), "area": int(np.count_nonzero(mask))}}
    g = gt_mask(sample)
    if g.shape != (h, w):
        raise ValueError(f"{sample.id}：GT mask 形状 {g.shape} 与原图 (H, W) = {(h, w)} 不符")
    st = mask_inter_union(mask if rec["failure"] is None else None, g)
    rec["score"] = {"inter": st.inter, "union": st.union, "iou": sample_iou(*st)}
    return rec, {"id": sample.id, "tool_times": times, "wall_time_s": float(sum(times))}


def run_w1f(spec: W1RunSpec, cfg: W1Config, samples: Sequence[Sample], paths: DataPaths, out: str | Path, *,
            splits_doc: Mapping[str, Any], box_dir: Path, box_recs: Sequence[Mapping[str, Any]], chunk_size: int, resume: bool,
            argv: Sequence[str] = (), prompts: PromptConfig | None = None, grabcut: GrabCut | None = None,
            log: Callable[[str], None] | None = None) -> dict[str, Any]:
    """W1-F（规则见模块说明）在一个划分上跑，写 W1 布局的目录，返回 metrics。按 chunk_size 分组落盘，续跑同回路。
    grabcut 默认是工作台的同一个函数（workbench_grabcut）；测试可以换成假的。"""
    log = log or (lambda m: print(m, file=sys.stderr, flush=True))
    if spec.task != "refseg":
        raise ValueError("W1-F 只对 refseg")
    grabcut = grabcut or workbench_grabcut()
    prompts = prompts or load_prompt_config(PROMPTS_PATH)
    gts = {s.id: gt_mask(s) for s in samples}
    matched = matched_boxes(samples, box_recs, prompts, spec.family, gts)
    w1f = cfg.baselines["w1f"]
    w1 = w1_record(spec, splits_doc, samples)
    cfg_doc = _derived_identity(spec, w1, out, box_dir, {
        "w1f": {"function": grabcut.name, "iterations": grabcut.iterations, "rule": w1f["rule"]},
        "prompts": {"config": _rel(cfg.path), "config_sha256": cfg.sha256}, "chunk_size": chunk_size})
    leaf = w1_leaf(out, spec.model_key, spec.task, spec.setting, spec.split)
    _prepare_leaf(leaf, cfg_doc, [cfg.path, *([] if splits_doc.get("tiny") else [W1_SPLITS_PATH])], resume=resume)
    ids = [s.id for s in samples]
    plan = batch_plan([None] * len(ids), chunk_size)
    what = f"W1-F {spec.model_key} {spec.split}"
    keep, trimmed = _resume_point(leaf / PREDICTIONS, ids, lambda r: r["id"], plan, log, what)
    run_k, run_path, run_log = _start_run_log(leaf, argv, n_done_before=keep, n_remaining=len(ids) - keep, chunk_size=chunk_size,
                                              trimmed_to_batch_boundary=trimmed)
    t_all = time.perf_counter()
    n_chunks = 0
    for a, b in [(a, b) for a, b in plan if a >= keep]:
        lines, tlines = [], []
        for s in samples[a:b]:
            rec, t = w1f_episode(s, matched[s.id], load_images(s)[0], grabcut)
            lines.append(_jsonl_line(rec))
            tlines.append(_jsonl_line({**t, "run": run_k, "chunk_index": n_chunks}))
        _append(leaf / TIMING, tlines)
        _append(leaf / PREDICTIONS, lines)
        n_chunks += 1
        log(f"[{what}] {b}/{len(ids)}")
    recs = read_jsonl_tolerant(leaf / PREDICTIONS, repair=False)
    if [r["id"] for r in recs] != ids:
        raise RuntimeError(f"{leaf / PREDICTIONS} 的 id 与样本不同")
    fails: dict[str, int] = {}
    for r in recs:
        k = "ok" if r["failure"] is None else r["failure"]
        fails[k] = fails.get(k, 0) + 1
    m = {"experiment": "W1", "w1": w1, "model": spec.model_key, "revision": spec.revision, "n": len(recs),
         "headline": headline("refseg", recs), "failures": dict(sorted(fails.items())), "box_run": cfg_doc["box_run"],
         "w1f": cfg_doc["w1f"], "complete": True, "n_expected": len(ids)}
    (leaf / METRICS).write_text(json.dumps(m, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    timing = {t["id"]: t for t in read_jsonl_tolerant(leaf / TIMING, repair=False)}
    write_compute_json(leaf / COMPUTE, merge(Compute().as_sample(wall_time_s=float(timing.get(i, {}).get("wall_time_s", 0.0)))
                                             for i in ids))
    (leaf / README).write_text("\n".join([
        f"# W1 · {spec.model_key} · refseg · w1f · {spec.split}", "",
        f"- W1-F：{_under_out(out, box_dir)} 的框 → {grabcut.name}（迭代 {grabcut.iterations} 次）；没有 LLM（compute.json 只有 "
        "GrabCut 的墙钟）",
        f"- n = {len(recs)}；失败 {m['failures']}", "", "## 标题指标", "", *_headline_lines(m), ""]), encoding="utf-8")
    _finish_run_log(run_path, run_log, n_batches=n_chunks, t_all=t_all)
    return m


__all__ = [
    "FAILURE_TOOL_ERROR",
    "box_leaf",
    "box_run_complete",
    "GrabCut",
    "e1_spec",
    "ensure_box_run",
    "matched_boxes",
    "own_box_record",
    "run_baseline_e1",
    "run_baseline_ownbox",
    "run_baseline_poly_structured",
    "run_w1f",
    "w1_selection",
    "w1f_episode",
    "workbench_grabcut",
]
