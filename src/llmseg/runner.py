"""LLMSeg 外部基线的评测（DEV-0066；D125-4）：W2 的 5 个 test 划分各取 300 句，跑官方管线（本地 Qwen3.5 提供方），按 W2 的口径计分。

子集（configs/w2_llmseg_subsets.json，build_subsets / check_subsets）：每个 test 划分的 1,000 句（configs/w2_splits.json）按
sha256(salt|purpose|sent_id) 升序（src/data/subsets.hash_rank，salt = w2_llmseg_v1，purpose = 划分名）取前 300 句，按 id 自然顺序列出；
清单记每个划分的 id、条数、sha256（population_digest）与来源划分的 sha256。W2-c 在同样的 300 句上配对比较（compare_runs）。

一条样本（run_item）：原图拷进 <叶目录>/work/<样本键>/image.<后缀>（同官方 ReasonSeg 评测的做法；官方管线的每次调用都带这个目录里的
图，本地提供方据此认出样本），官方 pipeline.run(原图拷贝, 指代句, 模型名, 工作目录, T, 主配置的其余参数) 在本进程里跑（官方评测是每条一个
子进程；这里为了合批改成线程，管线代码一样）。之后按官方评测（evals/common.run_pipeline + load_pred）取 mask：
- 修正前（官方 seg_*）：seg_mask.png（阶段 2 之后）；没有就是空 mask；
- 修正后（官方 rec_*，主配置 T = 8，标题读数）：rec_mask.png，没有就退回 seg_mask.png，再没有就是空 mask。
管线自己失败（返回 1，或抛出 Python 异常 = 官方的子进程崩溃）也照官方的口径按它留下的 mask 计分，status 记 failed。
计分与 W2 相同：同一份 GT（src/e1/scoring.gt_mask）、同一个 mask_inter_union / sample_iou；标题指标 = src/w1/scoring.headline（cIoU、
gIoU 与 bootstrap 95% CI）。两组读数（seg、rec）都给，各附计算量（按多轮记账：修正前只算阶段 1、2 的两次调用）。

输出（<out>/<模型>/refseg/<llmseg|llmseg_think>/<划分>/，同 W2 的布局）：predictions.jsonl（逐条，不含墙钟：两份 mask 的 RLE 与计分、
各阶段的调用数与 token、截断、格式失败、坐标统计；逐字节可复现）、metrics.json（只由 predictions 算）、compute.json（修正后，全部调用）、
compute_stage2.json（修正前，阶段 1、2）、config.yaml（运行身份：官方 commit、PROVENANCE 的 sha256、模型、迭代数、提供方、合批）、
meta.json、README.md、logs/（timing.jsonl 每条的墙钟；calls.jsonl 每次调用的记账与合批信息；batches.jsonl 每个子批；gpu_summary.json；
run_<k>.json）、work/<样本键>/（官方管线的 JSON 记录、run.log、calls.jsonl 含回复原文；前 --save 条另留 PNG）。
续跑（--resume）：同 W2，按 chunk_size 分组，没写完的那一组整组重跑（工作目录先清空），与一次跑完的 predictions.jsonl 逐字节相同。
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
import time
import traceback
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import yaml
from PIL import Image

from src.accounting import Compute, merge, write_compute_json
from src.data.records import Sample
from src.data.subsets import dump_config, hash_rank, population_digest, sort_ids
from src.e1.posthoc_runner import _finish_run_log, _prepare_leaf, _resume_point, _start_run_log
from src.e1.runner import COMPUTE, CONFIG, METRICS, PREDICTIONS, README, TIMING, _append, _jsonl_line, batch_plan, ids_digest, read_jsonl_tolerant
from src.e1.scoring import ci_dict, distribution, gt_mask
from src.e1.tasks import _sent_id
from src.executors.raster import mask_to_rle
from src.llmseg import provider as PV
from src.llmseg import vendor as V
from src.metrics.seg import mask_inter_union, sample_iou
from src.stats import paired_bootstrap_ci
from src.w1.data import poly_test_split
from src.w1.scoring import headline, paired_ciou
from src.w2.config import load_w2_config
from src.w2.data import W2_SPLITS_PATH, splits_sha256
from src.w2.runner import w2_leaf

REPO_ROOT = Path(__file__).resolve().parents[2]
SUBSETS_PATH = REPO_ROOT / "configs" / "w2_llmseg_subsets.json"
SUBSETS_VERSION = 1
SUBSET_SALT = "w2_llmseg_v1"
SUBSET_N = 300
EXPERIMENT = "W2-ext-LLMSeg"
FRAMEWORK = "src/llmseg（DEV-0066；D125-4）"
TASK = "refseg"
SETTING = "llmseg"
SETTING_THINK = "llmseg_think"
COMPUTE_STAGE2 = "compute_stage2.json"
CALLS_LOG = "logs/calls.jsonl"
BATCHES_LOG = "logs/batches.jsonl"
GPU_SUMMARY = "logs/gpu_summary.json"
WORK = "work"
KEEP_SUFFIXES = (".json", ".jsonl", ".log")  # 不在 --save 里的样本只留这些（同官方 run_benchmark 的 prune：只留 JSON）

# 官方主配置（README / EVALUATION.md：--iterations 8，--topology、--refine-box、--no-postprocess 都关；N_max = pipeline.MAX_INSTANCES）。
# tests/test_llmseg_vendor.py 核对它们与官方代码的默认值相同（evals/common.DEFAULT_ITERATIONS、pipeline.main 的 argparse 默认）。
MAIN_ITERATIONS = 8
MAIN_SWITCHES: Mapping[str, bool] = {"topology": False, "refine_box": False, "postprocess": True}
TARGET_N_ITEMS = 5 * SUBSET_N  # 用时外推的目标条数


# ---------------------------------------------------------------------------
# 子集
# ---------------------------------------------------------------------------


def test_splits() -> list[str]:
    """W2 的 5 个 test 划分（configs/w2.json 的 splits.test，名字同 src/w2/data.build_splits）。"""
    return [poly_test_split(k) for k in load_w2_config().splits["test"]]


def subset_rule(n: int = SUBSET_N, salt: str = SUBSET_SALT) -> str:
    return (f"每个 W2 test 划分（configs/w2_splits.json，各 1,000 句）按 sha256({salt}|<划分名>|sent_id) 升序（src/data/subsets.hash_rank，"
            f"purpose = 划分名）取前 {n} 句，按 id 自然顺序列出；sha256 = population_digest（自然顺序、换行连接后的 sha256）")


def select_subset(ids: Sequence[str], split: str, *, n: int = SUBSET_N, salt: str = SUBSET_SALT) -> list[str]:
    return sort_ids(hash_rank([str(i) for i in ids], split, salt)[:n])


def build_subsets(w2_doc: Mapping[str, Any], *, splits: Sequence[str], n: int = SUBSET_N, salt: str = SUBSET_SALT,
                  source: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in splits:
        e = w2_doc["splits"][name]
        ids = [str(i) for i in e["ids"]]
        if len(ids) != e["n"] or population_digest(ids) != e["sha256"]:
            raise ValueError(f"W2 清单的 {name}：{len(ids)} 条、指纹与记的 {e['n']} 条 {e['sha256'][:12]}… 不符")
        sel = select_subset(ids, name, n=n, salt=salt)
        out[name] = {"dataset": e.get("dataset"), "split_by": e.get("split_by"), "split": e.get("split"), "unit": e.get("unit"),
                     "source_n": e["n"], "source_sha256": e["sha256"], "purpose": name, "n": len(sel),
                     "sha256": population_digest(sel), "ids": sel}
    return {"version": SUBSETS_VERSION, "name": "LLMSeg 外部基线的 W2 test 子集（DEV-0066；D125-4）", "salt": salt, "n_per_split": n,
            "rule": subset_rule(n, salt), "source": dict(source), "splits": out}


def subsets_source(path: str | Path = W2_SPLITS_PATH) -> dict[str, Any]:
    p = Path(path)
    rel = str(p.resolve().relative_to(REPO_ROOT)) if p.resolve().is_relative_to(REPO_ROOT) else str(p)
    return {"path": rel, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}


def load_subsets(path: str | Path = SUBSETS_PATH) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check_subsets(path: str | Path = SUBSETS_PATH, w2_path: str | Path = W2_SPLITS_PATH) -> list[str]:
    """入库的清单必须与按规则重算的逐字节相同（来源清单的 sha256 也要对上）。返回问题列表（空 = 通过）。"""
    text = Path(path).read_text(encoding="utf-8")
    w2_doc = json.loads(Path(w2_path).read_text(encoding="utf-8"))
    ref = dump_config(build_subsets(w2_doc, splits=test_splits(), source=subsets_source(w2_path)))
    return [] if ref == text else [f"{path} 与按规则（{w2_path}）重算的不同"]


def item_key(sample_id: str) -> str:
    """样本 id（如 refcoco_unc/testA/106）→ 工作目录名（refcoco_unc__testA__106）。"""
    k = str(sample_id).replace("/", "__")
    if not re.fullmatch(r"[A-Za-z0-9+._-]+", k):
        raise ValueError(f"样本 id {sample_id!r} 换不成安全的目录名")
    return k


def subset_samples(samples: Sequence[Sample], ids: Sequence[str]) -> list[Sample]:
    """W2 划分的样本（加载器顺序）里取子集的句子（清单的 id 是 sent_id，同 configs/w2_splits.json），保持 W2 的顺序；子集里有找不到的
    sent_id 就报错。"""
    want = {str(i) for i in ids}
    out = [s for s in samples if _sent_id(s) in want]
    if len(out) != len(want):
        missing = sorted(want - {_sent_id(s) for s in out})
        raise ValueError(f"子集里有 {len(missing)} 句不在划分的样本里，例如 {missing[:3]}")
    keys = [item_key(s.id) for s in out]
    if len(set(keys)) != len(keys):
        raise ValueError("子集的样本键有重复")
    return out


# ---------------------------------------------------------------------------
# 运行身份
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PipelineConfig:
    """官方管线的参数（主配置 = MAIN_ITERATIONS 加 MAIN_SWITCHES；max_instances 取官方 pipeline.MAX_INSTANCES）。"""

    iterations: int = MAIN_ITERATIONS
    topology: bool = MAIN_SWITCHES["topology"]
    refine_box: bool = MAIN_SWITCHES["refine_box"]
    postprocess: bool = MAIN_SWITCHES["postprocess"]
    max_instances: int | None = None  # None = 官方默认

    def identity(self, pipeline: Any) -> dict[str, Any]:
        mi = int(pipeline.MAX_INSTANCES if self.max_instances is None else self.max_instances)
        main = (self.iterations == MAIN_ITERATIONS and all(getattr(self, k) == v for k, v in MAIN_SWITCHES.items())
                and mi == int(pipeline.MAX_INSTANCES))
        return {"iterations": self.iterations, "topology": self.topology, "refine_box": self.refine_box,
                "postprocess": self.postprocess, "max_instances": mi, "main_config": main,
                "converge": {"eps": float(pipeline.CONVERGE_EPS), "patience": int(pipeline.CONVERGE_PATIENCE)},
                "calls_bound": 2 * self.iterations + 2 + int(self.refine_box)}


@dataclass(frozen=True)
class RunSpec:
    model_key: str
    family: str
    repo: str
    revision: str
    adapter_model_key: str
    split: str
    setting: str
    tiny: bool
    limit: int | None
    image_policy: Mapping[str, Any]


def identity(spec: RunSpec, *, code: Mapping[str, Any], conv: Mapping[str, Any], pcfg: PipelineConfig, pipeline: Any,
             gen: PV.GenerationConfig, slots: int, chunk_size: int, splits_doc: Mapping[str, Any], subsets_doc: Mapping[str, Any],
             samples: Sequence[Sample], data_files: Sequence[Path]) -> dict[str, Any]:
    e = splits_doc["splits"][spec.split]
    sub = subsets_doc["splits"][spec.split]
    return {
        "experiment": EXPERIMENT, "framework": FRAMEWORK,
        "w2": {"task": TASK, "split": spec.split, "setting": spec.setting,
               "splits_config": "tiny（tests/fixtures 上现算）" if splits_doc.get("tiny") else "configs/w2_splits.json",
               "splits_sha256": splits_sha256(splits_doc), "split_sha256": e["sha256"], "split_n": e["n"],
               "subset": {"config": "tiny（同一规则现算）" if splits_doc.get("tiny") else "configs/w2_llmseg_subsets.json",
                          "rule": subsets_doc["rule"], "salt": subsets_doc["salt"], "n": sub["n"], "sha256": sub["sha256"]},
               "n_samples": len(samples), "limit": spec.limit, "sample_ids_sha256": ids_digest([s.id for s in samples])},
        "model": {"key": spec.model_key, "family": spec.family, "repo": spec.repo, "revision": spec.revision,
                  "adapter_model_key": spec.adapter_model_key, "tiny": spec.tiny},
        "image_policy": dict(spec.image_policy),
        "llmseg": {"code": dict(code), "paper": V.PAPER, "coordinates": dict(conv), "pipeline": pcfg.identity(pipeline),
                   "harness": "本进程内调用官方 pipeline.run（官方评测是每条一个子进程）；mask 的取法同官方 evals/common.run_pipeline："
                              "修正后 = rec_mask.png（没有退回 seg_mask.png），修正前 = seg_mask.png，缺失按空 mask"},
        "provider": {"name": PV.LOCAL_PROVIDER, "injected": ["llm._complete", *[f"llm.{n}（禁用）" for n in PV.PATCHED_PROVIDER_CALLS],
                                                          "llm.configure(provider='local')", "llm/pipeline/geometry.log"],
                     **gen.identity()},
        "batching": {"scheduler": "lockstep（全部在跑的管线都在等模型时成一轮；按 (阶段, 上限) 分组、组内 plan_sub_batches）",
                     "slots": slots, "batch_size": gen.batch_size, "max_batch_tokens": gen.max_batch_tokens, "chunk_size": chunk_size},
        "data": {"config_sha256": {Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in data_files}},
    }


# ---------------------------------------------------------------------------
# 一条样本
# ---------------------------------------------------------------------------


def _read_json(p: Path) -> dict[str, Any] | None:
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def load_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    """官方 load_pred 的读法（L 通道 > 0）；尺寸必须与 GT 相同（不缩放，对不上就报错）。"""
    with Image.open(path) as im:
        m = np.array(im.convert("L")) > 0
    if m.shape != tuple(shape):
        raise RuntimeError(f"{path}：mask 尺寸 {m.shape} 与 GT {tuple(shape)} 不同")
    return m


def final_masks(work: Path, shape: tuple[int, int], iterations: int) -> tuple[tuple[str, np.ndarray], tuple[str, np.ndarray]]:
    """(修正前, 修正后) 各自的 (来源, mask)，规则见模块说明。"""
    seg_p, rec_p = work / "seg_mask.png", work / "rec_mask.png"
    empty = np.zeros(shape, dtype=bool)
    seg = ("seg_mask", load_mask(seg_p, shape)) if seg_p.exists() else ("empty", empty)
    if iterations > 0 and rec_p.exists():
        rec = ("rec_mask", load_mask(rec_p, shape))
    else:
        rec = seg
    return seg, rec


def _score(mask: np.ndarray, gt: np.ndarray) -> dict[str, Any]:
    st = mask_inter_union(mask, gt)
    return {"inter": st.inter, "union": st.union, "iou": sample_iou(*st)}


def _mask_doc(src: str, mask: np.ndarray) -> dict[str, Any]:
    return {"source": src, "rle": mask_to_rle(mask), "area": int(np.count_nonzero(mask))}


def _calls_block(calls: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    s = PV.calls_summary(calls)
    s["n_rounds"] = sum(1 for c in calls if c["stage"] in PV.NEW_ROUND_STAGES and "input_rejected" not in c)
    return s


@dataclass
class ItemRun:
    calls: list[dict[str, Any]]
    rc: int | None
    error: str | None
    traceback: str | None
    wall_s: float
    log: tuple[str, ...] = ()


_PIPELINE_EVENT = re.compile(r"^\[(detect|segment|refine)\] (error|aborted|truncated reply|malformed reply|pass \d+: unparseable)")


def pipeline_events(log: Sequence[str]) -> list[str]:
    """官方管线日志里的出错与抢救事件（只取标签，如 "detect error"、"refine pass unparseable"、"segment truncated reply"）。"""
    out = []
    for line in log:
        m = _PIPELINE_EVENT.match(line)
        if m:
            out.append(f"{m.group(1)} {re.sub(r'pass [0-9]+: ', 'pass ', m.group(2))}")
    return out


class ItemContext:
    """一个划分的运行里每条样本共用的东西：官方模块、提供方、日志分流、管线参数、工作目录。"""

    def __init__(self, mods: Any, provider: PV.LocalProvider, router: PV.LogRouter, pcfg: PipelineConfig, work_root: Path):
        self.mods = mods
        self.provider = provider
        self.router = router
        self.pcfg = pcfg
        self.work_root = work_root

    def run(self, index: int, sample: Sample) -> ItemRun:
        """官方管线跑一条（在调度器的槽位线程里）。LocalProviderError 与 BaseException（含 ExternalCallForbidden）往外抛，中止运行；
        其余 Python 异常 = 官方的子进程崩溃，记下来照常计分。"""
        work = self.work_root / item_key(sample.id)
        if work.exists():
            shutil.rmtree(work)
        work.mkdir(parents=True)
        src = Path(sample.images[0])
        img = work / f"image{src.suffix.lower() or '.jpg'}"
        shutil.copyfile(src, img)
        p = self.pcfg
        buf: list[str] = []
        rc = err = tb = None
        self.provider.register(work, index)
        t0 = time.perf_counter()
        try:
            with self.router.capture(buf):
                rc = self.mods.pipeline.run(img, sample.fields["expression"], self.provider.model_name, work, p.iterations,
                                            max_instances=int(self.mods.pipeline.MAX_INSTANCES if p.max_instances is None else p.max_instances),
                                            topology=p.topology, refine_box=p.refine_box, postprocess=p.postprocess)
        except PV.LocalProviderError:
            raise
        except Exception as e:  # noqa: BLE001 — 官方子进程崩溃的口径
            err, tb = f"{type(e).__name__}: {e}"[:500], traceback.format_exc()
        finally:
            calls = self.provider.release(work)
        wall = time.perf_counter() - t0
        (work / "run.log").write_text("\n".join(buf) + ("\n" + tb if tb else "") + "\n", encoding="utf-8")
        with open(work / "calls.jsonl", "w", encoding="utf-8") as f:
            for c in calls:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        return ItemRun(calls=calls, rc=rc, error=err, traceback=tb, wall_s=wall, log=tuple(buf))


def item_record(spec: RunSpec, sample: Sample, run: ItemRun, work: Path, iterations: int) -> dict[str, Any]:
    """predictions.jsonl 的一行（不含墙钟）。"""
    gt = gt_mask(sample)
    h, w = gt.shape
    (seg_src, seg_m), (rec_src, rec_m) = final_masks(work, (h, w), iterations)
    det = _read_json(work / "detect.json")
    seg = _read_json(work / "seg.json")
    rec = _read_json(work / "recseg.json") if iterations > 0 else None
    calls = run.calls
    s2 = [c for c in calls if c["stage"] in PV.STAGE2_STAGES]
    allc = _calls_block(calls)
    st2 = _calls_block(s2)
    status = "ok" if run.rc == 0 and run.error is None else "failed"
    return {
        "id": sample.id, "w2_split": spec.split, "source": sample.source, "split": sample.split, "setting": spec.setting, "task": TASK,
        "instruction": sample.fields["expression"], "image_size": [int(w), int(h)],
        "status": status, "pipeline_rc": run.rc, "error": run.error, "pipeline_events": pipeline_events(run.log),
        "failure": None,  # W2 的键：mask 一律按官方口径从管线留下的文件取（见 answer.mask.source），不记作"没有预测"
        "n_detections": None if det is None else len(det.get("detections") or []),
        "n_instances": None if seg is None else len(seg.get("instances") or []),
        "degree": None if seg is None else seg.get("degree"),
        "refine": None if rec is None else {"iterations": rec.get("iterations"), "converged": rec.get("converged"),
                                            "complete": rec.get("complete")},
        "stage2": {"mask": _mask_doc(seg_src, seg_m), "score": _score(seg_m, gt), "n_model_calls": st2["n_model_calls"],
                   "n_rounds": st2["n_rounds"], "n_input_rejected": st2["n_input_rejected"], "tokens": st2["tokens"]},
        "answer": {"kind": "mask", "mask": _mask_doc(rec_src, rec_m)},
        "score": _score(rec_m, gt),
        **{k: allc[k] for k in ("n_model_calls", "n_rounds", "calls_by_stage", "tokens", "tokens_by_stage", "truncated_by_stage",
                                "format_fail_by_stage", "n_empty_reply", "n_input_rejected", "coords")},
        "n_tool_calls": 0,
    }


def timing_record(sample: Sample, run: ItemRun, run_k: int, chunk: int) -> dict[str, Any]:
    ok = [c for c in run.calls if "input_rejected" not in c]
    s2 = [c for c in ok if c["stage"] in PV.STAGE2_STAGES]
    return {"id": sample.id, "run": run_k, "chunk_index": chunk, "wall_time_s": run.wall_s,
            "call_time_s": sum(c["wall_s"] for c in ok), "call_time_stage2_s": sum(c["wall_s"] for c in s2),
            "stage2_wall_s": max((c["t_end_s"] for c in s2), default=0.0), "queue_s": sum(c.get("queue_s", 0.0) for c in ok)}


def prune(work: Path, keep_all: bool) -> None:
    """不在 --save 里的样本：只留 JSON、jsonl 与 run.log（同官方 run_benchmark 的 prune）。"""
    if keep_all:
        return
    for p in work.iterdir():
        if p.is_file() and p.suffix.lower() not in KEEP_SUFFIXES:
            p.unlink()


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------


def _rate(n: int, d: int) -> dict[str, Any]:
    return {"n": int(n), "of": int(d), "rate": (n / d) if d else None}


def _sum_by_stage(recs: Sequence[Mapping[str, Any]], key: str) -> dict[str, int]:
    return {s: int(sum(r[key][s] for r in recs)) for s in PV.STAGES}


def _per_image_compute(recs: Sequence[Mapping[str, Any]], block: Callable[[Mapping[str, Any]], Mapping[str, Any]]) -> dict[str, Any]:
    if not recs:
        return {}
    b = [block(r) for r in recs]
    return {"n_model_calls": float(np.mean([x["n_model_calls"] for x in b])), "n_rounds": float(np.mean([x["n_rounds"] for x in b])),
            **{f"{k}_tokens": float(np.mean([x["tokens"][k] for x in b])) for k in ("prompt", "generated", "visual")},
            "rule": "每图均值；墙钟见 compute.json / compute_stage2.json（logs/timing.jsonl）"}


def aggregate(recs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """predictions.jsonl 的记录 → metrics.json 的主体（只读记录）。"""
    n = len(recs)
    seg_view = [{"score": r["stage2"]["score"]} for r in recs]
    calls = _sum_by_stage(recs, "calls_by_stage")
    rej = int(sum(r["n_input_rejected"] for r in recs))
    answered = {s: calls[s] for s in PV.STAGES}
    trunc = _sum_by_stage(recs, "truncated_by_stage")
    fmt = _sum_by_stage(recs, "format_fail_by_stage")
    n_calls = int(sum(calls.values()))
    m: dict[str, Any] = {
        "n": n,
        "headline": headline(TASK, recs) if recs else [],
        "readouts": {
            "rec": {"what": "修正后（官方 rec_*；主配置 T 轮修正，标题读数）", "headline": headline(TASK, recs) if recs else [],
                    "compute_per_image": _per_image_compute(recs, lambda r: r)},
            "seg": {"what": "修正前（官方 seg_*：阶段 2 之后，阶段 1、2 两次调用）", "headline": headline(TASK, seg_view) if recs else [],
                    "compute_per_image": _per_image_compute(recs, lambda r: r["stage2"])},
        },
        "status": {k: v for k, v in sorted(Counter(r["status"] for r in recs).items())},
        "mask_source": {"rec": {k: v for k, v in sorted(Counter(r["answer"]["mask"]["source"] for r in recs).items())},
                        "seg": {k: v for k, v in sorted(Counter(r["stage2"]["mask"]["source"] for r in recs).items())}},
        "calls_per_image": distribution(r["n_model_calls"] for r in recs),
        "calls_by_stage": calls,
        "truncation": {**{s: _rate(trunc[s], answered[s]) for s in PV.STAGES}, "all": _rate(sum(trunc.values()), n_calls - rej),
                       "rule": "截断 = 生成到这一阶段的上限仍没有停止 token（stopped = False）；分母 = 送进模型的调用"},
        "format_failure": {**{s: _rate(fmt[s], answered[s]) for s in PV.STAGES}, "all": _rate(sum(fmt.values()), n_calls - rej),
                           "rule": "格式失败 = 回复不能被官方 extract_json 严格解析（含空回复；官方之后仍会尝试修复与抢救）"},
        "coords": {"n_values": int(sum(r["coords"]["n_values"] for r in recs)), "n_gt_1000": int(sum(r["coords"]["n_gt_1000"] for r in recs)),
                   "n_calls_gt_1000": int(sum(r["coords"]["n_calls_gt_1000"] for r in recs)),
                   "n_calls_unparsed": int(sum(r["coords"]["n_calls_unparsed"] for r in recs)),
                   "max_abs": max((r["coords"]["max_abs"] for r in recs if r["coords"]["max_abs"] is not None), default=None),
                   "rule": "回复里框与控制点的坐标值（官方要求 0–1000 网格）；> 1000 的那一组官方会当像素、不换算（只统计，不改官方代码）"},
        "n_empty_reply": int(sum(r["n_empty_reply"] for r in recs)), "n_input_rejected": rej,
        "pipeline_events": {k: v for k, v in sorted(Counter(e for r in recs for e in r["pipeline_events"]).items())},
        "n_detections": distribution(r["n_detections"] for r in recs if r["n_detections"] is not None),
        "n_instances": distribution(r["n_instances"] for r in recs if r["n_instances"] is not None),
        "refine": {"iterations": distribution(r["refine"]["iterations"] for r in recs if r["refine"]),
                   "converged": _rate(sum(1 for r in recs if r["refine"] and r["refine"]["converged"]), sum(1 for r in recs if r["refine"]))},
    }
    for key in ("prompt", "generated", "inserted", "visual"):
        m.setdefault("tokens", {})[key] = distribution(r["tokens"][key] for r in recs)
    if m["coords"]["n_values"]:
        m["coords"]["rate_gt_1000"] = m["coords"]["n_gt_1000"] / m["coords"]["n_values"]
    return m


def compute_of(recs: Sequence[Mapping[str, Any]], timing: Mapping[str, Mapping[str, Any]], *, stage2: bool) -> Compute:
    out = []
    for r in recs:
        b = r["stage2"] if stage2 else r
        t = timing.get(r["id"], {})
        tk = b["tokens"]
        k = int(b["n_model_calls"]) - int(b["n_input_rejected"])  # 送进模型的调用（拒收的不算）
        out.append(Compute(n_calls=k, n_rounds=int(b["n_rounds"]), generated_tokens=int(tk["generated"]),
                           prompt_tokens=int(tk["prompt"]) + int(tk["inserted"]), visual_tokens=int(tk["visual"]),
                           image_pixels=int(tk["image_pixels"]),
                           call_time_s=float(t.get("call_time_stage2_s" if stage2 else "call_time_s", 0.0)))
                   .as_sample(wall_time_s=float(t.get("stage2_wall_s" if stage2 else "wall_time_s", 0.0))))
    return merge(out)


def gpu_summary(batches: Sequence[Mapping[str, Any]], n_items: int, *, target: int = TARGET_N_ITEMS) -> dict[str, Any]:
    """logs/batches.jsonl（每个子批一行）→ 生成的总墙钟、按阶段的子批统计、每条样本的生成墙钟与外推到 target 条的 GPU 小时。
    生成墙钟只含 model.generate（不含 CPU 上的几何与 build_inputs），所以外推是下界；run_<k>.json 的 wall_s 是整次运行的墙钟。"""
    tot = float(sum(b["wall_s"] for b in batches))
    by: dict[str, Any] = {}
    for s in PV.STAGES:
        bs = [b for b in batches if b["stage"] == s]
        if bs:
            by[s] = {"n_sub_batches": len(bs), "rows": int(sum(b["rows"] for b in bs)), "wall_s": float(sum(b["wall_s"] for b in bs)),
                     "mean_rows": float(np.mean([b["rows"] for b in bs])), "mean_max_generated": float(np.mean([b["max_generated"] for b in bs])),
                     "n_truncated": int(sum(b["n_truncated"] for b in bs)), "mean_max_prompt": float(np.mean([b["max_prompt"] for b in bs]))}
    per = tot / n_items if n_items else None
    return {"n_sub_batches": len(batches), "generate_wall_s": tot, "n_items": n_items, "generate_wall_per_item_s": per, "by_stage": by,
            "extrapolate": {"target_items": target, "gpu_hours": None if per is None else per * target / 3600.0,
                            "rule": "每条样本的生成墙钟 × 目标条数；同一配置（slots、batch_size、max_batch_tokens）下才可比"}}


def _fmt(v: Any) -> str:
    return "null" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))


def _readme(spec: RunSpec, cfg_doc: Mapping[str, Any], m: Mapping[str, Any]) -> str:
    lines = [f"# W2 外部基线 LLMSeg · {spec.model_key} · {spec.setting} · {spec.split}", "",
             f"- 官方代码：{cfg_doc['llmseg']['code']['upstream']}@{cfg_doc['llmseg']['code']['commit'][:12]}（third_party/llmseg，未改动；"
             f"PROVENANCE sha256 {cfg_doc['llmseg']['code']['provenance_sha256'][:16]}…）",
             f"- 管线：{cfg_doc['llmseg']['pipeline']}",
             f"- 模型：{spec.repo}@{spec.revision}（适配器 {spec.adapter_model_key}；官方模型名 {cfg_doc['llmseg']['coordinates']['model_name']}）；"
             f"思考 {cfg_doc['provider']['enable_thinking']}；{cfg_doc['provider']['method']}",
             f"- 子集：{cfg_doc['w2']['subset']['n']} 句（sha256 {cfg_doc['w2']['subset']['sha256'][:16]}…）；合批 {cfg_doc['batching']}", "",
             "## 读数", ""]
    for key, r in m["readouts"].items():
        for h in r["headline"]:
            ci = h.get("ci")
            lines.append(f"- {key}（{r['what']}）{h['name']}：{_fmt(h['value'])}，95% CI "
                         f"{'—' if not ci else '[' + _fmt(ci['low']) + ', ' + _fmt(ci['high']) + ']'}，n = {h['n']}")
        lines.append(f"  - 每图计算量：{r['compute_per_image']}")
    lines += ["", "## 调用与格式", "",
              f"- status {m['status']}；每图调用 {m['calls_per_image']}",
              f"- 截断 {m['truncation']['all']}；格式失败 {m['format_failure']['all']}；坐标 > 1000 {m['coords']['n_gt_1000']} / "
              f"{m['coords']['n_values']}（{m['coords']['n_calls_gt_1000']} 次调用）", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 一个划分
# ---------------------------------------------------------------------------


def run_split(adapter: Any, spec: RunSpec, samples: Sequence[Sample], out: str | Path, *, gen: PV.GenerationConfig,
              pcfg: PipelineConfig, slots: int, chunk_size: int, splits_doc: Mapping[str, Any], subsets_doc: Mapping[str, Any],
              data_files: Sequence[Path], resume: bool, save: int, argv: Sequence[str] = (),
              log: Callable[[str], None] | None = None) -> dict[str, Any]:
    """一个划分（子集）上跑官方管线并写出全部结果，返回 metrics。"""
    log = log or (lambda msg: print(msg, file=sys.stderr, flush=True))
    code = V.check_vendor()
    mods = V.load()
    model_name = spec.model_key
    conv = PV.check_coordinate_convention(mods.llm, model_name)
    leaf = w2_leaf(out, spec.model_key, TASK, spec.setting, spec.split)
    cfg_doc = identity(spec, code=code, conv=conv, pcfg=pcfg, pipeline=mods.pipeline, gen=gen, slots=slots, chunk_size=chunk_size,
                       splits_doc=splits_doc, subsets_doc=subsets_doc, samples=samples, data_files=data_files)
    _prepare_leaf(leaf, cfg_doc, [V.PROVENANCE_PATH, *data_files], resume=resume)
    ids = [s.id for s in samples]
    plan = batch_plan([None] * len(ids), chunk_size)
    what = f"LLMSeg {spec.model_key} {spec.setting}/{spec.split}"
    keep, trimmed = _resume_point(leaf / PREDICTIONS, ids, lambda r: r["id"], plan, log, what)
    todo = [(a, b) for a, b in plan if a >= keep]
    run_k, run_path, run_log = _start_run_log(leaf, argv, n_done_before=keep, n_remaining=len(ids) - keep, slots=slots,
                                              batch_size=gen.batch_size, max_batch_tokens=gen.max_batch_tokens, chunk_size=chunk_size,
                                              trimmed_to_batch_boundary=trimmed)
    t_all = time.perf_counter()
    work_root = leaf / WORK
    work_root.mkdir(exist_ok=True)
    save_ids = set(ids[: max(0, int(save))])
    batches: list[dict[str, Any]] = []
    sched = PV.LockstepScheduler(PV.AdapterBackend(adapter, gen), slots=slots, on_batch=batches.append)
    provider = PV.LocalProvider(sched, llm=mods.llm, model_name=model_name)
    router = PV.LogRouter()
    inst = PV.install(mods, provider, router)
    ctx = ItemContext(mods, provider, router, pcfg, work_root)
    try:
        for n_chunk, (a, b) in enumerate(todo):
            chunk = list(samples[a:b])
            batches.clear()
            runs = sched.run(chunk, lambda i, s: ctx.run(a + i, s))
            lines, tlines, clines = [], [], []
            for s, r in zip(chunk, runs):
                work = work_root / item_key(s.id)
                lines.append(_jsonl_line(item_record(spec, s, r, work, pcfg.iterations)))
                tlines.append(_jsonl_line(timing_record(s, r, run_k, n_chunk)))
                clines += [_jsonl_line({"id": s.id, "run": run_k, "chunk_index": n_chunk, **{k: v for k, v in c.items() if k != "text"}})
                           for c in r.calls]
                prune(work, s.id in save_ids)
            _append(leaf / BATCHES_LOG, [_jsonl_line({"run": run_k, "chunk_index": n_chunk, **x}) for x in batches])
            _append(leaf / CALLS_LOG, clines)
            _append(leaf / TIMING, tlines)
            _append(leaf / PREDICTIONS, lines)
            log(f"[{what}] {b}/{len(ids)}（{len(batches)} 个子批）")
    finally:
        inst.restore()
    recs = read_jsonl_tolerant(leaf / PREDICTIONS, repair=False)
    if [r["id"] for r in recs] != ids[: len(recs)]:
        raise RuntimeError(f"{leaf / PREDICTIONS} 的 id 不是样本顺序的前缀")
    body = aggregate(recs)
    metrics = {"experiment": EXPERIMENT, "w2": cfg_doc["w2"], "model": spec.model_key, "revision": spec.revision,
               "llmseg_commit": code["commit"], "setting": spec.setting, **body, "complete": len(recs) == len(ids), "n_expected": len(ids)}
    (leaf / METRICS).write_text(json.dumps(metrics, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    timing = {t["id"]: t for t in read_jsonl_tolerant(leaf / TIMING, repair=False)}
    write_compute_json(leaf / COMPUTE, compute_of(recs, timing, stage2=False))
    write_compute_json(leaf / COMPUTE_STAGE2, compute_of(recs, timing, stage2=True))
    all_batches = read_jsonl_tolerant(leaf / BATCHES_LOG, repair=False)
    (leaf / GPU_SUMMARY).write_text(json.dumps(gpu_summary(all_batches, len(recs)), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (leaf / README).write_text(_readme(spec, cfg_doc, metrics), encoding="utf-8")
    _finish_run_log(run_path, run_log, n_batches=len(todo), t_all=t_all)
    return metrics


# ---------------------------------------------------------------------------
# 与 W2-c 的配对比较
# ---------------------------------------------------------------------------


def read_leaf(leaf: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    leaf = Path(leaf)
    cfg = yaml.safe_load((leaf / CONFIG).read_text(encoding="utf-8"))
    return cfg, read_jsonl_tolerant(leaf / PREDICTIONS, repair=False)


def _view(recs: Sequence[Mapping[str, Any]], readout: str) -> list[dict[str, Any]]:
    return [{"id": r["id"], "score": r["stage2"]["score"] if readout == "seg" else r["score"]} for r in recs]


def paired(llm_view: Sequence[Mapping[str, Any]], w2c_view: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """同一批 id 上 LLMSeg − W2-c 的 cIoU 差（配对 bootstrap，ratio_of_sums）与 gIoU 差（配对 bootstrap，均值）。"""
    c = paired_ciou(llm_view, w2c_view)
    by = {r["id"]: r for r in w2c_view}
    ia = np.array([r["score"]["iou"] for r in llm_view], dtype=np.float64)
    ib = np.array([by[r["id"]]["score"]["iou"] for r in llm_view], dtype=np.float64)
    g = paired_bootstrap_ci(ia, ib)
    gi = {"n": len(ia), "a": float(ia.mean()) if len(ia) else None, "b": float(ib.mean()) if len(ib) else None,
          "diff": None if g is None else g.estimate, "ci": ci_dict(g)}
    return {"cIoU": c, "gIoU": gi}


def compare_runs(pairs: Sequence[tuple[str, str | Path, str | Path]]) -> dict[str, Any]:
    """[(划分, LLMSeg 的叶目录, W2-c 的叶目录)] → 各划分与合并（全部划分的句子放在一起）的配对差，两组读数（seg、rec）都比。
    W2-c 的记录先取 LLMSeg 的那批 id（W2-c 是全部 1,000 句）；缺 id 就报错。a = LLMSeg，b = W2-c，差 = a − b。"""
    per: dict[str, Any] = {}
    pooled: dict[str, tuple[list, list]] = {"rec": ([], []), "seg": ([], [])}
    ids_info: dict[str, Any] = {}
    for split, l_leaf, w_leaf in pairs:
        lc, lr = read_leaf(l_leaf)
        wc, wr = read_leaf(w_leaf)
        if (lc.get("w2") or {}).get("split") != split or (wc.get("w2") or {}).get("split") != split:
            raise ValueError(f"{split}：两个目录记的划分是 {lc.get('w2', {}).get('split')!r} 与 {wc.get('w2', {}).get('split')!r}")
        if (wc.get("w2") or {}).get("setting") != "w2c" or not wc.get("checkpoint"):
            raise ValueError(f"{w_leaf}：不是 W2-c 的目录（setting = {(wc.get('w2') or {}).get('setting')!r}，checkpoint 缺）")
        want = {r["id"] for r in lr}
        wsub = [r for r in wr if r["id"] in want]
        if len(wsub) != len(want):
            raise ValueError(f"{split}：W2-c 缺 {len(want) - len(wsub)} 句 LLMSeg 的句子")
        ids_info[split] = {"n": len(lr), "ids_sha256": population_digest([r["id"] for r in lr]), "llmseg_model": lc["model"]["key"],
                           "w2c_model": wc["model"]["key"], "llmseg_complete_n": len(lr), "w2c_checkpoint": wc["checkpoint"].get("path")}
        per[split] = {}
        for ro in ("rec", "seg"):
            lv = _view(lr, ro)
            wv = _view(wsub, "rec")  # W2-c 只有一组读数
            per[split][ro] = paired(lv, wv)
            pooled[ro][0].extend(lv)
            pooled[ro][1].extend(wv)
    out = {"what": "LLMSeg − W2-c 的配对差（a = LLMSeg，b = W2-c）；rec = LLMSeg 修正后（标题），seg = LLMSeg 修正前；W2-c 是它的标题读数",
           "splits": per, "pooled": {ro: paired(*pooled[ro]) for ro in ("rec", "seg")}, "ids": ids_info}
    rec = out["pooled"]["rec"]["cIoU"]
    out["flag_llmseg_not_lower"] = {"rule": "D125-4：LLMSeg（9B）不低于 W2-c 就立即报告；这里给点估计与 CI 两种口径，不作判定",
                                    "point_estimate_ge_0": None if rec["diff"] is None else rec["diff"] >= 0,
                                    "ci_high_ge_0": None if not rec["ci"] else rec["ci"]["high"] >= 0}
    return out


def compare_markdown(res: Mapping[str, Any]) -> str:
    def row(name: str, d: Mapping[str, Any]) -> list[str]:
        out = []
        for ro in ("rec", "seg"):
            for met in ("cIoU", "gIoU"):
                x = d[ro][met]
                ci = x["ci"]
                out.append(f"| {name} | {ro} | {met} | {x['n']} | {_fmt(x['a'])} | {_fmt(x['b'])} | {_fmt(x['diff'])} | "
                           f"{'—' if not ci else '[' + _fmt(ci['low']) + ', ' + _fmt(ci['high']) + ']'} |")
        return out

    lines = ["# LLMSeg 与 W2-c 的配对比较", "", res["what"], "",
             "| 划分 | LLMSeg 读数 | 指标 | n | LLMSeg | W2-c | 差 | 95% CI |", "|---|---|---|---|---|---|---|---|"]
    for split, d in res["splits"].items():
        lines += row(split, d)
    lines += row("合并", res["pooled"])
    lines += ["", f"- {res['flag_llmseg_not_lower']}", ""]
    return "\n".join(lines)


__all__ = [
    "EXPERIMENT",
    "ItemContext",
    "ItemRun",
    "MAIN_ITERATIONS",
    "MAIN_SWITCHES",
    "PipelineConfig",
    "RunSpec",
    "SETTING",
    "SETTING_THINK",
    "SUBSETS_PATH",
    "SUBSET_N",
    "SUBSET_SALT",
    "aggregate",
    "build_subsets",
    "check_subsets",
    "compare_markdown",
    "compare_runs",
    "compute_of",
    "final_masks",
    "gpu_summary",
    "item_key",
    "item_record",
    "load_subsets",
    "paired",
    "run_split",
    "select_subset",
    "subset_samples",
    "subsets_source",
    "test_splits",
]
