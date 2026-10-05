"""E3a 的评测（DEV-0034 §3；简报 §8.4、§12）：一个检查点 × 一个任务 → <out>/eval/e3a/<变体>/<模型>/seed<k>/<任务>/。

**加载**（调用方：scripts/eval_e3a.py）：full 先做评测前的合并核对（src/e3/merge_check：check_training_interfaces --checkpoint 的
merge 一项，作用于 <检查点>/model，判据 v1 = D50-3；不通过就不评测）；c1、c2 没有 LoRA，跳过。再 factory.load_trained(<检查点>/model,
merge=True)，full、c1 重新登记查询 token（个数与位置模式取 readout.json），读出与查询嵌入从 <检查点>/readout 载入（与 model/ 的配对
由 manifest 的 sha256 核对）。

**前向**（与训练同一个函数：features.llm_inputs / vision_inputs → Readout → 任务头；no_grad）：按样本顺序连续成批（chunk_bounds：
最后一批只有 1 条时并进前一批，打乱检查要求每批至少 2 条），每条样本得到 128 × 128 的输出：refseg 1 通道 logit、depth 1 通道 log 深度、
normals 3 通道向量、semseg ADE_NUM_CLASSES 通道的 logits。

**上采样到原图与逐样本指标**（score_payload，进程池；src/readout/targets.to_original）：guided（默认，以原图为引导图）与 bilinear 两种：
- refseg：logit 上采样，> 0 记前景；交、并、IoU（src/metrics/seg.sample_iou）、Boundary IoU；
- depth：log 深度上采样再 exp；depth_sample_metrics（Eigen crop，align = none 与 scale_shift）、depth_sample_rmse；没有有效像素记 None；
- semseg：逐类 logits 上采样（按 semseg_channel_chunk 分块取 argmax，与一次算全部通道逐位相同），intersect_and_union、pixel_correct_total
  （reduce_zero 同 E2）；
- normals：上采样后逐像素单位化（normal_angular_errors），normal_image_summary（sum、n、within、hist）。
数据集级指标与 bootstrap CI 用 E2 的同一套聚合（src/e2/tasks 的 DepthTask / SemsegTask / NormalsTask.aggregate；BOOTSTRAP_* 取
src/constants）；refseg：cIoU（ratio_of_sums 的 CI）、gIoU、Boundary IoU（逐样本均值的 CI），按被指代物体是不是 person 分列。

**泄漏检查**（简报 §8.4，必须做；结果在 metrics.json 的 leak_checks，只用默认上采样）：
- shuffle：同一批里每条样本的读出输入换成批内循环顺序里下一条与它不同图的样本的（shuffle_perm；全批同图时退回下一条，整个评测集
  只有 1 条时不打乱，都计入 n_same_image_fallback），类别名的表示与 GT 不动；报告打乱后的指标与配对差（正常 − 打乱）的 bootstrap CI；
- no_instruction（eval.leak_checks.no_instruction.tasks，默认只有 refseg）：指令换成空串（user 轮里只剩图像与查询 token）再前向一次；
  报告指标与配对差。C2 的解码器看不到指令，这一项对它恒为零差（照报）。
崩溃与否的判定规则留给 E3 预注册（配置的 leak_checks.rule_basis）。

**输出**（叶目录）：scores.jsonl（逐样本、逐模式、逐上采样的统计；每行一个样本；semseg 的逐类交 / 并写成稀疏的 {类号: 计数}）、
arrays/chunk_<k>.npz（normals 的逐样本角误差直方图，只用于中位角误差的 CI；*.npz 按 .gitignore 不入库，配对比较不需要它）、metrics.json（只由 scores 与 arrays 算出，逐字节可复现）、compute.json 与
compute_no_instruction.json（src/accounting；含墙钟）、config.yaml（运行身份：检查点两份 manifest 的 sha256、模型、任务、评测集 id 指纹、
指令、批配置、上采样与泄漏检查的设置、E3a 配置的 sha256）、meta.json、README.md、logs/timing.jsonl、logs/run_<k>.json。
配对比较（完整方法对 C1、C2）：paired_compare(叶目录 a, 叶目录 b)，按样本 id 对齐，标题指标的配对差与 bootstrap CI。
**续跑**（--resume）：config.yaml 必须与本次完全相同；scores.jsonl 与 arrays 按批追加，中断在一批中间时退回到这一批的开头整批重跑。
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import multiprocessing
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import yaml

from src.accounting import Compute, merge, write_compute_json
from src.constants import (
    ADE_NUM_CLASSES,
    ADE_REDUCE_ZERO_LABEL,
    BOOTSTRAP_N_RESAMPLES,
    BOOTSTRAP_SEED,
    GUIDED_FILTER_EPS,
    GUIDED_FILTER_RADIUS,
    NYU_EVAL_HW,
)
from src.data import loaders as L
from src.data import train_loaders as TL
from src.data.records import LoadResult, Sample
from src.data.subsets import DEV_SUBSETS_PATH
from src.e1.scoring import PERSON_CATEGORY, ci_dict, gt_mask
from src.e3.data import load_rgb
from src.readout.config import E3aConfig
from src.readout.features import ClassSpans, ReadoutInputs, c2_class_side, llm_inputs, vision_inputs
from src.readout.model import TASKS, Readout, ReadoutSpec
from src.readout.targets import to_original
from src.readout.train import (
    MODEL_SUB,
    READOUT_SUB,
    layers_for,
    load_readout,
    read_readout_manifest,
)
from src.metrics.depth import depth_sample_metrics, depth_sample_rmse, nyu_eval_mask
from src.metrics.normal import NORMAL_ANGLE_THRESHOLDS, normal_angular_errors, normal_image_summary
from src.metrics.seg import boundary_iou, intersect_and_union, mask_inter_union, pixel_correct_total, sample_iou
from src.runlog import sha256_file, write_run_metadata
from src.stats import bootstrap_ci, miou_statistic, paired_bootstrap_ci, ratio_of_sums

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_REFSEG_ROOT = REPO_ROOT / "tests" / "fixtures" / "e3" / "eval_root"
FIXTURE_REFSEG_DEV = REPO_ROOT / "tests" / "fixtures" / "e3" / "eval_dev_subsets.json"
FIXTURE_DEPTH_ROOT = REPO_ROOT / "tests" / "fixtures" / "e1" / "root"
FIXTURE_DENSE_ROOT = REPO_ROOT / "tests" / "fixtures" / "e2" / "root"
REFSEG_DATASET = "refcoco"

EVAL_SUB = "eval"
EXPERIMENT = "e3a"
SCORES = "scores.jsonl"
ARRAYS_DIR = "arrays"
METRICS = "metrics.json"
COMPUTE = "compute.json"
COMPUTE_NO_INSTRUCTION = "compute_no_instruction.json"
CONFIG = "config.yaml"
README = "README.md"
TIMING = "logs/timing.jsonl"
MERGE_CHECK = "merge_check.json"
MODES: tuple[str, ...] = ("normal", "shuffle", "no_instruction")
GROUPS = ("person", "non_person", "unknown")
# 数组形式的逐样本统计：法向的角误差直方图（0.1° 一箱，共 1800 箱）存进 arrays/ 的 npz（只在算中位角误差的 CI 时用；*.npz 按 .gitignore
# 不入库，配对比较不需要它）；语义分割的逐类交 / 并以稀疏的 {类号: 计数}（只列并 > 0 的类）写进 scores.jsonl，配对比较的 mIoU 用它
ARRAY_FIELDS: Mapping[str, tuple[str, ...]] = {"normals": ("hist",), "semseg": (), "refseg": (), "depth": ()}
SPARSE_FIELDS: Mapping[str, tuple[str, ...]] = {"semseg": ("inter", "union"), "normals": (), "refseg": (), "depth": ()}
# E2 的任务对象只用来复用它的 aggregate（上采样不经过它）；构造函数要一个合法的上采样说明
_E2_UPSAMPLE_STUB = {"mode": "bilinear", "impl": "interpolate", "align_corners": False}


# ---------------------------------------------------------------------------
# 路径、评测集、设置
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalPaths:
    """refseg_root / depth_root：E1 的落地根（refcoco、nyuv2）；semseg_root / normals_root：DEV-0022 的落地根（ade20k/val、
    nyu_normals/test）；dev_subsets：E1 的开发子集清单。"""

    refseg_root: Path
    depth_root: Path
    semseg_root: Path
    normals_root: Path
    dev_subsets: Path = DEV_SUBSETS_PATH
    strict: bool = True

    @classmethod
    def roots(cls, e1_root: str | Path, train_root: str | Path | None = None,
              dev_subsets: str | Path | None = None) -> "EvalPaths":
        e1 = Path(e1_root)
        tr = Path(train_root) if train_root else e1
        return cls(refseg_root=e1, depth_root=e1, semseg_root=tr, normals_root=tr,
                   dev_subsets=Path(dev_subsets or DEV_SUBSETS_PATH))

    @classmethod
    def fixtures(cls) -> "EvalPaths":
        return cls(refseg_root=FIXTURE_REFSEG_ROOT, depth_root=FIXTURE_DEPTH_ROOT, semseg_root=FIXTURE_DENSE_ROOT,
                   normals_root=FIXTURE_DENSE_ROOT, dev_subsets=FIXTURE_REFSEG_DEV)

    def as_dict(self) -> dict[str, Any]:
        return {"refseg_root": str(self.refseg_root), "depth_root": str(self.depth_root), "semseg_root": str(self.semseg_root),
                "normals_root": str(self.normals_root), "dev_subsets": str(self.dev_subsets), "strict": self.strict}


def load_eval_set(task: str, paths: EvalPaths) -> LoadResult:
    """评测集（配置的 eval.sets）。"""
    if task == "refseg":
        return L.load_refcoco(paths.refseg_root, REFSEG_DATASET, "val", subset="eval", dev_config=paths.dev_subsets,
                              strict=paths.strict)
    if task == "depth":
        from src.e2.data import load_nyu_test

        return load_nyu_test(paths.depth_root, strict=paths.strict)
    if task == "semseg":
        return TL.load_ade20k(paths.semseg_root, "val", strict=paths.strict)
    if task == "normals":
        return TL.load_nyu_normals(paths.normals_root, "test", strict=paths.strict)
    raise ValueError(f"未知任务 {task!r}（有：{TASKS}）")


def resolve_eval(cfg: E3aConfig, *, tiny: bool, batch_size: int | None = None, score_workers: int | None = None,
                 merge_check_n: int | None = None) -> dict[str, Any]:
    """配置的 eval 节 → 本次生效的设置（--tiny 时按 eval.tiny 覆盖规模；命令行可再覆盖批大小、评分进程数、合并核对的 n）。"""
    e = cfg.doc["eval"]
    t = e["tiny"] if tiny else {}
    lc = e["leak_checks"]
    ni_tasks = [str(x) for x in lc["no_instruction"]["tasks"]]
    if "semseg" in ni_tasks:
        raise ValueError("no_instruction 不能用于 semseg（去掉指令就没有类别名）")
    mc = e["merge_check"]
    n_mc = merge_check_n if merge_check_n is not None else (int(t["merge_check_n"]) if tiny else int(mc["n"]))
    out = {
        "version": e["version"],
        "batch_size": int(batch_size if batch_size is not None else t.get("batch_size", e["batch_size"])),
        "batch_size_config": int(e["batch_size"]),
        "upsample": {"primary": e["upsample"]["primary"], "methods": list(e["upsample"]["methods"]),
                     "guided_filter": {"radius": GUIDED_FILTER_RADIUS, "eps": GUIDED_FILTER_EPS}},
        "semseg_channel_chunk": int(e["semseg_channel_chunk"]),
        "leak_checks": {"shuffle": {"upsample": lc["shuffle"]["upsample"]},
                        "no_instruction": {"tasks": ni_tasks, "instruction": str(lc["no_instruction"]["instruction"]),
                                           "upsample": lc["no_instruction"]["upsample"]}},
        "merge_check": {"n": int(n_mc), "checks": [str(c) for c in mc["checks"]]},
        "score_workers": int(score_workers if score_workers is not None else t.get("score_workers", e["score_workers"])),
        "bootstrap": {"n_resamples": BOOTSTRAP_N_RESAMPLES, "seed": BOOTSTRAP_SEED},
    }
    if out["batch_size"] < 2 or out["score_workers"] < 0 or out["merge_check"]["n"] < 1 or out["semseg_channel_chunk"] < 1:
        raise ValueError(f"批大小必须 ≥ 2（打乱检查要求每批至少 2 条），合并核对的 n、分块 ≥ 1，评分进程数 ≥ 0：{out}")
    for k in ("shuffle", "no_instruction"):
        if out["leak_checks"][k]["upsample"] not in out["upsample"]["methods"]:
            raise ValueError(f"leak_checks.{k}.upsample 必须在 upsample.methods 里")
    return out


def identity_settings(settings: Mapping[str, Any]) -> dict[str, Any]:
    """进运行身份的设置：评分进程数不改变任何结果，不进；合并核对另记在 merge_check.json。"""
    return {k: v for k, v in settings.items() if k not in ("score_workers", "merge_check")}


# ---------------------------------------------------------------------------
# 模型：检查点 → 适配器 + 读出
# ---------------------------------------------------------------------------


@dataclass
class EvalModel:
    """合并后的适配器、读出模块与查询嵌入；features / outputs 与训练同一套函数（no_grad）。"""

    ad: Any
    readout: Readout
    cfg: E3aConfig
    variant: str
    query_embeds: torch.Tensor | None
    layers: list[int] | None
    class_spans: ClassSpans
    c2_class: tuple[torch.Tensor, torch.Tensor] | None = None
    manifests: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_checkpoint(cls, ckpt: str | Path, cfg: E3aConfig, *, device: str, dtype: torch.dtype | None = None) -> "EvalModel":
        from src.adapters.factory import load_trained

        ck = Path(ckpt)
        rman = read_readout_manifest(ck / READOUT_SUB)
        ad, _ = load_trained(ck / MODEL_SUB, device=device, dtype=dtype, merge=True)
        spec = ReadoutSpec(**{**rman["spec"], "widths": tuple(rman["spec"]["widths"]), "res_blocks": tuple(rman["spec"]["res_blocks"])})
        if spec.hidden_size != ad.hidden_size():
            raise ValueError(f"读出的 hidden {spec.hidden_size} 与模型的 {ad.hidden_size()} 不同")
        q = rman.get("query")
        if (q is None) != (spec.variant == "c2"):
            raise ValueError("readout.json 的查询登记与变体不符")
        if q is not None:
            ad.inject_query_tokens(int(q["n"]), str(q["position_mode"]))
        readout = Readout(spec).to(ad.device)
        qe, _ = load_readout(ck / READOUT_SUB, readout, model_dir=ck / MODEL_SUB)
        readout.eval()
        ad.eval_mode()
        layers = None
        c2 = None
        if spec.variant != "c2":
            layers = layers_for(ad)
            if rman.get("layers") != layers or spec.n_sources != len(layers):
                raise ValueError(f"readout.json 的层 {rman.get('layers')} 与模型的 {layers} 不同")
        else:
            c2 = c2_class_side(ad, cfg.instruction("semseg"), cfg.class_names, order=cfg.order("semseg"))
        return cls(ad=ad, readout=readout, cfg=cfg, variant=spec.variant,
                   query_embeds=None if qe is None else qe.to(ad.device, torch.float32), layers=layers,
                   class_spans=ClassSpans(cfg.class_names), c2_class=c2,
                   manifests={"model_manifest_sha256": sha256_file(ck / MODEL_SUB / "manifest.json"),
                              "readout_manifest_sha256": sha256_file(ck / READOUT_SUB / "readout.json")})

    def features(self, inputs: Sequence[Any], *, semseg: bool) -> ReadoutInputs:
        with torch.no_grad():
            if self.variant == "c2":
                return vision_inputs(self.ad, inputs, grid=self.cfg.query_grid, class_side=self.c2_class if semseg else None)
            return llm_inputs(self.ad, inputs, self.query_embeds, layers=self.layers,
                              class_spans=self.class_spans if semseg else None, enable_thinking=self.cfg.enable_thinking)

    def outputs(self, task: str, feats: ReadoutInputs) -> np.ndarray:
        """[B, C_t, R, R] float32（CPU numpy）。"""
        with torch.no_grad():
            ro = self.readout(*feats.args())
            return self.readout.task_output(task, ro.embed, ro.class_emb).float().cpu().numpy()


# ---------------------------------------------------------------------------
# 分批、打乱
# ---------------------------------------------------------------------------


def chunk_bounds(n: int, batch_size: int) -> list[tuple[int, int]]:
    """[0, n) 按顺序切成至多 batch_size 条的批（续跑的边界）；最后一批只有 1 条且 n ≥ 2 时并进前一批（打乱检查要求每批至少 2 条）。"""
    if batch_size < 2:
        raise ValueError("batch_size 必须 ≥ 2")
    out = [(lo, min(lo + batch_size, n)) for lo in range(0, n, batch_size)]
    if len(out) >= 2 and out[-1][1] - out[-1][0] == 1:
        out[-2:] = [(out[-2][0], out[-1][1])]
    return out


def shuffle_perm(keys: Sequence[str]) -> tuple[list[int], int]:
    """批内的打乱：第 i 条取循环顺序里第一个与它不同图（image key 不同）的样本；找不到时取下一条（计数）。返回 (下标, 退回的条数)。
    只有 1 条时返回 [0]（不打乱，计 1 条退回）。"""
    n = len(keys)
    perm, fb = [], 0
    for i in range(n):
        j = next((((i + d) % n) for d in range(1, n) if keys[(i + d) % n] != keys[i]), None)
        if j is None:
            j = (i + 1) % n
            fb += 1
        perm.append(j)
    return perm, fb


# ---------------------------------------------------------------------------
# 逐样本评分（进程池的入口）
# ---------------------------------------------------------------------------


def read_gt(task: str, sample: Sample) -> Any:
    """评测样本的原分辨率 GT：refseg = E1 的 gt_mask；depth = depth_raw（米，0 = 无效）；semseg = ADE 原始标签（0 = 忽略）；
    normals = (单位法向, 有效掩码)。"""
    if task == "refseg":
        return gt_mask(sample)
    if task == "depth":
        g = sample.gt
        if "depth_npz" in g:
            with np.load(g["depth_npz"]) as z:
                return np.asarray(z[g["key"]], dtype=np.float32)
        return TL.read_depth_m(g["depth"])
    if task == "semseg":
        return TL.read_ade_label(sample.gt["label"])
    if task == "normals":
        return TL.read_normals(sample.gt["normal"])
    raise ValueError(f"未知任务 {task!r}")


def semseg_argmax(logits: np.ndarray, image: np.ndarray, method: str, chunk: int) -> np.ndarray:
    """[C, R, R] 的逐类 logits → 原图上的 argmax（int64），按 chunk 个通道分块上采样（各通道互相独立；跨块用严格大于，平票取靠前的类，
    与一次算全部通道的 np.argmax 相同）。"""
    best = arg = None
    for c0 in range(0, logits.shape[0], int(chunk)):
        up = to_original(np.ascontiguousarray(logits[c0: c0 + chunk].transpose(1, 2, 0)), image, method)
        m, a = up.max(axis=2), up.argmax(axis=2) + c0
        if best is None:
            best, arg = m, a
        else:
            upd = m > best
            best = np.where(upd, m, best)
            arg = np.where(upd, a, arg)
    return arg.astype(np.int64)


def sample_stat(task: str, out: np.ndarray, image: np.ndarray, gt: Any, method: str, chunk: int) -> dict[str, Any] | None:
    """一条样本在一种上采样下的统计（规则见模块说明）。out 是这条样本的 [C_t, R, R]。"""
    if task == "refseg":
        pred = to_original(out[0], image, method) > 0.0
        st = mask_inter_union(pred, gt)
        return {"inter": int(st.inter), "union": int(st.union), "iou": sample_iou(st.inter, st.union),
                "biou": boundary_iou(pred, gt)}
    if task == "depth":
        pred = np.exp(to_original(out[0], image, method))
        a = depth_sample_metrics(pred, gt, align="none")
        if a is None:
            return None
        b = depth_sample_metrics(pred, gt, align="scale_shift")
        return {"abs_rel": a["abs_rel"], "d1": a["d1"], "rmse": depth_sample_rmse(pred, gt, align="none"),
                "abs_rel_ss": b["abs_rel"], "d1_ss": b["d1"]}
    if task == "semseg":
        pred = semseg_argmax(out, image, method, chunk)
        inter, union = intersect_and_union(pred, gt, reduce_zero=ADE_REDUCE_ZERO_LABEL)
        c, t = pixel_correct_total(pred, gt, reduce_zero=ADE_REDUCE_ZERO_LABEL)
        return {"inter": inter.astype(np.int64), "union": union.astype(np.int64), "correct": int(c), "total": int(t)}
    if task == "normals":
        n, valid = gt
        up = to_original(np.ascontiguousarray(out.transpose(1, 2, 0)), image, method)
        s = normal_image_summary(normal_angular_errors(up, n, valid))
        return {"sum": float(s["sum"]), "n": int(s["n"]), "within": [int(x) for x in s["within"]], "hist": s["hist"]}
    raise ValueError(f"未知任务 {task!r}")


def gt_info(task: str, gt: Any) -> dict[str, Any]:
    if task == "refseg":
        return {"area": int(np.count_nonzero(gt)), "hw": list(gt.shape)}
    if task == "depth":
        return {"n_eval_px": int(nyu_eval_mask(gt).sum()) if gt.shape == tuple(NYU_EVAL_HW) else 0, "hw": list(gt.shape)}
    if task == "semseg":
        return {"n_labeled": int(np.count_nonzero(gt)), "hw": list(gt.shape)}
    return {"n_valid": int(np.count_nonzero(gt[1])), "hw": list(gt[1].shape)}


def score_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """一个样本的全部评分。payload：task、sample、outs（模式 → [C_t, R, R] 或 None）、methods（模式 → 上采样列表）、chunk。"""
    task, s = payload["task"], payload["sample"]
    image = load_rgb(s.images[0])
    gt = read_gt(task, s)
    stats: dict[str, dict[str, Any]] = {}
    for mode, out in payload["outs"].items():
        if out is None:
            continue
        stats[mode] = {m: sample_stat(task, out, image, gt, m, int(payload["chunk"])) for m in payload["methods"][mode]}
    return {"id": s.id, "gt": gt_info(task, gt), "stats": stats}


def split_arrays(task: str, stats: Mapping[str, Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """逐样本统计 → (进 scores.jsonl 的部分（SPARSE_FIELDS 写成稀疏的 {类号: 计数}），数组部分 {"模式.上采样.字段": 数组}（进 arrays/）)。"""
    scal: dict[str, Any] = {}
    arrs: dict[str, np.ndarray] = {}
    for mode, by in stats.items():
        scal[mode] = {}
        for m, st in by.items():
            if st is None:
                scal[mode][m] = None
                continue
            d = {k: v for k, v in st.items() if k not in ARRAY_FIELDS[task] and k not in SPARSE_FIELDS[task]}
            if SPARSE_FIELDS[task]:
                keep = np.flatnonzero(np.asarray(st["union"]))
                for k in SPARSE_FIELDS[task]:
                    a = np.asarray(st[k])
                    d[k] = {str(int(c)): int(a[c]) for c in keep if a[c]}
            scal[mode][m] = d
            for k in ARRAY_FIELDS[task]:
                arrs[f"{mode}.{m}.{k}"] = np.asarray(st[k])
    return scal, arrs


def merge_arrays(task: str, scal: Mapping[str, Any], arrs: Mapping[str, np.ndarray] | None, b: int) -> dict[str, dict[str, Any]]:
    """split_arrays 的逆：第 b 条样本的完整统计（稀疏字段还原成长度 ADE_NUM_CLASSES 的数组）。arrs 为 None 时数组字段记 None。"""
    out: dict[str, dict[str, Any]] = {}
    for mode, by in scal.items():
        out[mode] = {}
        for m, st in by.items():
            if st is None:
                out[mode][m] = None
                continue
            d = dict(st)
            for k in SPARSE_FIELDS[task]:
                a = np.zeros(ADE_NUM_CLASSES, dtype=np.int64)
                for c, v in st[k].items():
                    a[int(c)] = int(v)
                d[k] = a
            for k in ARRAY_FIELDS[task]:
                d[k] = None if arrs is None else np.asarray(arrs[f"{mode}.{m}.{k}"][b]).astype(np.int64)
            out[mode][m] = d
    return out


# ---------------------------------------------------------------------------
# 聚合与配对比较
# ---------------------------------------------------------------------------


def e2_task(task: str, cfg: E3aConfig) -> Any:
    """E2 的任务对象（只用它的 aggregate：同一套指标定义与 bootstrap）。"""
    from src.e2.tasks import make_task

    ch = {"depth": 1, "semseg": ADE_NUM_CLASSES, "normals": 3, "refseg": 1}[task]
    return make_task({"task": task, "out_channels": ch, "loss": dict(cfg.task(task)["loss"]), "upsample": dict(_E2_UPSAMPLE_STUB)})


def _mean_ci(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"value": None, "ci": None, "n": 0}
    ci = bootstrap_ci(np.asarray(values, dtype=np.float64))
    return {"value": ci.estimate, "ci": ci_dict(ci), "n": len(values)}


def _ratio_ci(pairs: Sequence[Sequence[float]]) -> dict[str, Any]:
    if not pairs or sum(u for _, u in pairs) == 0:
        return {"value": None, "ci": None, "n": len(pairs)}
    arr = np.asarray(pairs, dtype=np.float64)
    try:
        ci = bootstrap_ci(arr, statistic=ratio_of_sums)
    except ValueError:  # 某次重采样的分母为 0（极小样本）
        return {"value": ratio_of_sums(arr), "ci": None, "n": len(pairs)}
    return {"value": ci.estimate, "ci": ci_dict(ci), "n": len(pairs)}


def refseg_block(stats: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {"cIoU": _ratio_ci([(s["inter"], s["union"]) for s in stats]), "gIoU": _mean_ci([s["iou"] for s in stats]),
            "BIoU": _mean_ci([s["biou"] for s in stats]), "n": len(stats)}


def aggregate_stats(task: str, stats: Sequence[Any], cfg: E3aConfig, groups: Sequence[str | None] | None = None) -> dict[str, Any]:
    """一种模式、一种上采样下的数据集级指标（带 bootstrap CI）。"""
    if task == "refseg":
        out = refseg_block(stats)
        if groups is not None:
            out["by_group"] = {g: refseg_block([s for s, gg in zip(stats, groups) if gg == g]) for g in GROUPS if g in groups}
        return out
    return e2_task(task, cfg).aggregate(list(stats), n_resamples=BOOTSTRAP_N_RESAMPLES, seed=BOOTSTRAP_SEED)


HEADLINE: Mapping[str, tuple[str, ...]] = {"refseg": ("cIoU", "gIoU", "BIoU"), "depth": ("abs_rel", "d1"),
                                           "semseg": ("miou", "pixel_acc"), "normals": ("mean", "median", "within_11.25")}


def headline(task: str, agg: Mapping[str, Any]) -> dict[str, Any]:
    if task == "refseg":
        return {k: agg[k] for k in HEADLINE[task]}
    return {k: {"value": agg.get(k), "ci": agg.get(f"{k}_ci")} for k in HEADLINE[task]}


def _paired(a: Sequence[Any], b: Sequence[Any], statistic: Any = None) -> dict[str, Any] | None:
    if len(a) == 0:
        return None
    try:
        ci = paired_bootstrap_ci(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64), statistic=statistic)
    except ValueError:
        return None
    return {"estimate": ci.estimate, **ci_dict(ci), "n": len(a)}


def paired_diff(task: str, a: Sequence[Any], b: Sequence[Any]) -> dict[str, Any]:
    """按样本对齐的两组逐样本统计 → 标题指标的配对差（a − b）与 bootstrap CI（同一组重采样下标）。"""
    if len(a) != len(b):
        raise ValueError("配对比较的两边样本数不同")
    if task == "refseg":
        return {"gIoU": _paired([x["iou"] for x in a], [x["iou"] for x in b]),
                "cIoU": _paired([[x["inter"], x["union"]] for x in a], [[x["inter"], x["union"]] for x in b],
                                statistic=ratio_of_sums),
                "BIoU": _paired([x["biou"] for x in a], [x["biou"] for x in b])}
    if task == "depth":
        both = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
        return {"abs_rel": _paired([x["abs_rel"] for x, _ in both], [y["abs_rel"] for _, y in both]),
                "d1": _paired([x["d1"] for x, _ in both], [y["d1"] for _, y in both]), "n_both_valid": len(both)}
    if task == "semseg":
        ia = np.stack([np.stack([x["inter"], x["union"]]) for x in a]) if a else np.zeros((0, 2, ADE_NUM_CLASSES))
        ib = np.stack([np.stack([x["inter"], x["union"]]) for x in b]) if b else np.zeros((0, 2, ADE_NUM_CLASSES))
        return {"miou": _paired(ia, ib, statistic=miou_statistic),
                "pixel_acc": _paired([[x["correct"], x["total"]] for x in a], [[x["correct"], x["total"]] for x in b],
                                     statistic=ratio_of_sums)}
    if task == "normals":
        k = NORMAL_ANGLE_THRESHOLDS.index(11.25)
        return {"mean": _paired([[x["sum"], x["n"]] for x in a], [[x["sum"], x["n"]] for x in b], statistic=ratio_of_sums),
                "within_11.25": _paired([[x["within"][k], x["n"]] for x in a], [[x["within"][k], x["n"]] for x in b],
                                        statistic=ratio_of_sums)}
    raise ValueError(task)


def group_of(task: str, s: Sample) -> str | None:
    if task != "refseg":
        return None
    cat = s.fields.get("category")
    return "unknown" if cat is None else ("person" if cat == PERSON_CATEGORY else "non_person")


# ---------------------------------------------------------------------------
# 运行身份、README
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalSpec:
    model_key: str
    family: str
    repo: str
    revision: str
    adapter_model_key: str
    tiny: bool
    dtype: str
    variant: str
    seed: int
    checkpoint: Mapping[str, Any]
    limit: int | None = None


def model_dir(out: str | Path, spec: EvalSpec) -> Path:
    return Path(out) / EVAL_SUB / EXPERIMENT / spec.variant / spec.model_key / f"seed{int(spec.seed)}"


def leaf_dir(out: str | Path, spec: EvalSpec, task: str) -> Path:
    return model_dir(out, spec) / task


def ids_digest(ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def run_identity(spec: EvalSpec, cfg: E3aConfig, settings: Mapping[str, Any], task: str, sel: LoadResult,
                 samples: Sequence[Sample], model: EvalModel) -> dict[str, Any]:
    ids = [s.id for s in samples]
    tc = cfg.task(task)
    instr = {"order": tc["order"], "instruction_template": tc["instruction_template"]}
    if task == "semseg":
        instr["class_names_sha256"] = hashlib.sha256("\n".join(cfg.class_names).encode("utf-8")).hexdigest()
    return {
        "experiment": "E3a-eval",
        "framework": "src/readout/evaluate.py（DEV-0034）",
        "variant": spec.variant,
        "seed": int(spec.seed),
        "task": task,
        "model": {"key": spec.model_key, "family": spec.family, "repo": spec.repo, "revision": spec.revision,
                  "adapter_model_key": spec.adapter_model_key, "tiny": spec.tiny, "dtype": spec.dtype},
        "checkpoint": dict(spec.checkpoint),
        "readout": {"spec": model.readout.spec.as_dict(), "sources": list(model.readout.sources), "layers": model.layers},
        "instruction": instr,
        "eval": identity_settings(settings),
        "config": {"path": cfg.path, "sha256": cfg.sha256, "version": cfg.doc["version"]},
        "data": {"n": len(ids), "ids_sha256": ids_digest(ids), "limit": spec.limit,
                 "stats": json.loads(json.dumps(sel.stats, ensure_ascii=False, default=str))},
    }


def _identity(d: Mapping[str, Any]) -> dict[str, Any]:
    return yaml.safe_load(yaml.safe_dump(dict(d), sort_keys=False, allow_unicode=True))


def _fmt(v: Any) -> str:
    return "null" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))


def _ci_s(x: Mapping[str, Any] | None) -> str:
    if not x or x.get("value") is None:
        return "—"
    ci = x.get("ci")
    return f"{_fmt(x['value'])}" + (f" [{_fmt(ci['low'])}, {_fmt(ci['high'])}]" if ci else "")


def _pd_s(x: Mapping[str, Any] | None) -> str:
    if not x:
        return "—"
    return f"{_fmt(x['estimate'])} [{_fmt(x['low'])}, {_fmt(x['high'])}]"


def readme_text(spec: EvalSpec, ident: Mapping[str, Any], metrics: Mapping[str, Any]) -> str:
    task = ident["task"]
    prim = metrics["headline"]["upsample"]
    lines = [f"# E3a 评测 · {spec.variant} · {spec.model_key} · seed {spec.seed} · {task}", "",
             f"- 检查点：model/manifest sha256 {spec.checkpoint.get('model_manifest_sha256')}，readout.json sha256 "
             f"{spec.checkpoint.get('readout_manifest_sha256')}",
             f"- 模型：{spec.repo}@{spec.revision}（{spec.family}；{spec.dtype}；tiny = {spec.tiny}）",
             f"- 评测集：n = {ident['data']['n']}（id 指纹 {ident['data']['ids_sha256'][:16]}…）；批大小 {ident['eval']['batch_size']}",
             f"- 上采样：默认 {prim}；另报 {', '.join(m for m in ident['eval']['upsample']['methods'] if m != prim)}", "",
             "## 指标（95% CI）", "", "| 上采样 | " + " | ".join(HEADLINE[task]) + " |", "|---|" + "---|" * len(HEADLINE[task])]
    for m, agg in metrics["by_upsample"].items():
        h = headline(task, agg)
        lines.append(f"| {m} | " + " | ".join(_ci_s(h[k]) for k in HEADLINE[task]) + " |")
    lines += ["", f"## 泄漏检查（{prim}；配对差 = 正常 − 检查，95% CI）", "",
              "| 检查 | " + " | ".join(HEADLINE[task]) + " | 配对差 |", "|---|" + "---|" * (len(HEADLINE[task]) + 1)]
    for name, lc in metrics["leak_checks"].items():
        h = headline(task, lc["metrics"])
        pd = "; ".join(f"{k} {_pd_s(v)}" for k, v in lc["paired_normal_minus_check"].items() if isinstance(v, dict))
        lines.append(f"| {name} | " + " | ".join(_ci_s(h[k]) for k in HEADLINE[task]) + f" | {pd} |")
    lines += ["", "- 判定规则留给 E3 预注册；这里只给读数（规则见 src/readout/evaluate.py 的模块说明）", "", "## 文件", "",
              "- scores.jsonl、arrays/：逐样本统计（配对比较用 src/readout/evaluate.paired_compare）；metrics.json：全部汇总；",
              "- compute*.json：记账（含墙钟）；config.yaml：运行身份；meta.json；logs/：墙钟与每次调用的记录。", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def read_jsonl_tolerant(path: Path, *, repair: bool) -> list[dict[str, Any]]:
    from src.e1.runner import read_jsonl_tolerant as _r

    return _r(path, repair=repair)


def _truncate_lines(path: Path, k: int) -> None:
    raw = path.read_bytes()
    pos = 0
    for _ in range(k):
        pos = raw.index(b"\n", pos) + 1
    with open(path, "r+b") as f:
        f.truncate(pos)


def _append(path: Path, lines: Sequence[str]) -> None:
    if not lines:
        return
    with open(path, "a", encoding="utf-8") as f:
        f.write("".join(lines))
        f.flush()
        os.fsync(f.fileno())


def _jsonl(rec: Mapping[str, Any]) -> str:
    return json.dumps(rec, ensure_ascii=False) + "\n"


def _chunk_file(leaf: Path, ci: int) -> Path:
    return leaf / ARRAYS_DIR / f"chunk_{ci:05d}.npz"


def score_all(payloads: Sequence[Mapping[str, Any]], workers: int) -> list[dict[str, Any]]:
    if workers <= 0 or len(payloads) <= 1:
        return [score_payload(p) for p in payloads]
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as ex:
        return list(ex.map(score_payload, payloads))


def run_chunk(model: EvalModel, task: str, samples: Sequence[Sample], settings: Mapping[str, Any], cfg: E3aConfig
              ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """一批：正常、打乱、（可选）去掉指令三种读出，返回 (每条的评分 payload, 计数与墙钟)。"""
    t0 = time.perf_counter()
    images = [load_rgb(s.images[0]) for s in samples]
    order = cfg.order(task)
    instr = [cfg.instruction(task, s.fields.get("expression") if task == "refseg" else None) for s in samples]
    inputs = [model.ad.build_inputs([im], ins, order) for im, ins in zip(images, instr)]
    t1 = time.perf_counter()
    feats = model.features(inputs, semseg=task == "semseg")
    out_n = model.outputs(task, feats)
    keys = [s.image_keys[0] if s.image_keys else s.id for s in samples]
    perm, fb = shuffle_perm(keys)
    out_s = model.outputs(task, feats.permuted(perm))
    t2 = time.perf_counter()
    out_i = None
    t3 = t2
    lens_i = None
    ni = settings["leak_checks"]["no_instruction"]
    if task in ni["tasks"]:
        inputs_i = [model.ad.build_inputs([im], ni["instruction"], order) for im in images]
        feats_i = model.features(inputs_i, semseg=False)
        out_i = model.outputs(task, feats_i)
        lens_i = feats_i.prompt_lengths
        t3 = time.perf_counter()
    prim = settings["upsample"]["primary"]
    methods = {"normal": list(settings["upsample"]["methods"]), "shuffle": [settings["leak_checks"]["shuffle"]["upsample"]],
               "no_instruction": [ni["upsample"]]}
    payloads = [{"task": task, "sample": s, "chunk": settings["semseg_channel_chunk"],
                 "outs": {"normal": out_n[b], "shuffle": out_s[b], "no_instruction": None if out_i is None else out_i[b]},
                 "methods": methods} for b, s in enumerate(samples)]
    info = {"perm": perm, "n_same_image_fallback": fb, "prompt_lengths": list(feats.prompt_lengths),
            "prompt_lengths_no_instruction": None if lens_i is None else list(lens_i),
            "visual_tokens": [sum(len(i.span) for i in inp.images) for inp in inputs],
            "image_pixels": [sum(int(i.n_pixels or 0) for i in inp.images) for inp in inputs],
            "prep_s": t1 - t0, "fwd_s": t2 - t1, "fwd_no_instruction_s": t3 - t2, "primary": prim}
    return payloads, info


def finalize(leaf: Path, spec: EvalSpec, ident: Mapping[str, Any], task: str, cfg: E3aConfig, settings: Mapping[str, Any],
             samples: Sequence[Sample]) -> dict[str, Any]:
    """由 scores.jsonl、arrays/ 与 logs/timing.jsonl 写 metrics.json、compute*.json、README.md。返回 metrics。"""
    rows, bounds = load_rows(leaf, task, [s.id for s in samples])
    prim = settings["upsample"]["primary"]
    groups = [group_of(task, s) for s in samples]
    by_up = {m: aggregate_stats(task, [r["stats"]["normal"][m] for r in rows], cfg, groups if task == "refseg" else None)
             for m in settings["upsample"]["methods"]}
    leak: dict[str, Any] = {}
    sh_m = settings["leak_checks"]["shuffle"]["upsample"]
    normal_s = [r["stats"]["normal"][sh_m] for r in rows]
    shuffled = [r["stats"]["shuffle"][sh_m] for r in rows]
    leak["shuffle"] = {"upsample": sh_m, "metrics": aggregate_stats(task, shuffled, cfg),
                       "paired_normal_minus_check": paired_diff(task, normal_s, shuffled),
                       "n_same_image_fallback": int(sum(r["n_same_image_fallback"] for r in bounds)),
                       "definition": "同一批里每条样本的读出输入换成批内循环顺序里下一条不同图的样本的；类别名的表示与 GT 不动"}
    ni = settings["leak_checks"]["no_instruction"]
    if task in ni["tasks"]:
        m = ni["upsample"]
        a = [r["stats"]["normal"][m] for r in rows]
        b = [r["stats"]["no_instruction"][m] for r in rows]
        leak["no_instruction"] = {"upsample": m, "instruction": ni["instruction"], "metrics": aggregate_stats(task, b, cfg),
                                  "paired_normal_minus_check": paired_diff(task, a, b),
                                  "definition": "指令换成空串（user 轮里只剩图像与查询 token）再前向一次"}
    timing = {t["id"]: t for t in read_jsonl_tolerant(leaf / TIMING, repair=False)}
    comp_n, comp_i = [], []
    for r in rows:
        t = timing.get(r["id"], {})
        c = Compute.call(generated_tokens=0, prompt_tokens=int(r["prompt_tokens"]), visual_tokens=int(r["visual_tokens"]),
                         image_pixels=int(r["image_pixels"]), call_time_s=float(t.get("fwd_s", 0.0)), new_round=True)
        comp_n.append(c.as_sample(wall_time_s=float(t.get("fwd_s", 0.0)) + float(t.get("prep_s", 0.0)) + float(t.get("score_s", 0.0))))
        if r.get("prompt_tokens_no_instruction") is not None:
            ci = Compute.call(generated_tokens=0, prompt_tokens=int(r["prompt_tokens_no_instruction"]),
                              visual_tokens=int(r["visual_tokens"]), image_pixels=int(r["image_pixels"]),
                              call_time_s=float(t.get("fwd_no_instruction_s", 0.0)), new_round=True)
            comp_i.append(ci.as_sample(wall_time_s=float(t.get("fwd_no_instruction_s", 0.0))))
    write_compute_json(leaf / COMPUTE, merge(comp_n))
    if comp_i:
        write_compute_json(leaf / COMPUTE_NO_INSTRUCTION, merge(comp_i))
    counts = {k: int(sum(int(r[k]) for r in rows)) for k in ("prompt_tokens", "visual_tokens", "image_pixels")}
    metrics = {"task": task, "model": spec.model_key, "variant": spec.variant, "seed": int(spec.seed),
               "checkpoint": dict(spec.checkpoint), "n": len(rows), "complete": True, "config_sha256": ident["config"]["sha256"],
               "headline": {"upsample": prim, **headline(task, by_up[prim])},
               "by_upsample": by_up, "leak_checks": leak,
               "groups": dict(sorted(Counter(g for g in groups if g is not None).items())),
               "compute": {"n_calls": len(rows), **counts, "n_calls_no_instruction": len(comp_i),
                           "note": "一条样本一次前向（没有生成）；打乱检查复用同一次前向的隐藏状态，只多跑一遍读出；去掉指令另跑一次前向"
                                   "（compute_no_instruction.json）。墙钟在 compute*.json"}}
    (leaf / METRICS).write_text(json.dumps(metrics, indent=1, ensure_ascii=False, default=_json_default) + "\n", encoding="utf-8")
    (leaf / README).write_text(readme_text(spec, ident, metrics), encoding="utf-8")
    return metrics


def _json_default(o: Any) -> Any:
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"不能写成 JSON：{type(o).__name__}")


def load_rows(leaf: Path, task: str, ids: Sequence[str] | None = None, *, arrays: bool = True
              ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """scores.jsonl + arrays/ → 每条样本的完整统计（数组已并回）与每批的记录（n_same_image_fallback）。ids 给了时核对完整与顺序。
    arrays=False 时不读 arrays/（法向的直方图记 None；配对比较只用 scores.jsonl，入库的文件就够）。"""
    recs = read_jsonl_tolerant(leaf / SCORES, repair=False)
    if ids is not None and [r["id"] for r in recs] != list(ids):
        raise RuntimeError(f"{leaf / SCORES} 不完整或顺序不对（{len(recs)}/{len(ids)} 条）")
    by_chunk: dict[int, list[int]] = {}
    for i, r in enumerate(recs):
        by_chunk.setdefault(int(r["chunk"]), []).append(i)
    rows: list[dict[str, Any]] = [dict(r) for r in recs]
    bounds = []
    for ci, idx in sorted(by_chunk.items()):
        arrs: dict[str, np.ndarray] | None = {}
        if ARRAY_FIELDS[task] and arrays:
            with np.load(_chunk_file(leaf, ci)) as z:
                arrs = {k: z[k] for k in z.files}
        elif ARRAY_FIELDS[task]:
            arrs = None
        for b, i in enumerate(idx):
            rows[i]["stats"] = merge_arrays(task, recs[i]["stats"], arrs, b)
        bounds.append({"chunk": ci, "n": len(idx), "n_same_image_fallback": int(recs[idx[0]]["chunk_same_image_fallback"])})
    return rows, bounds


def run_eval_task(model: EvalModel, cfg: E3aConfig, settings: Mapping[str, Any], spec: EvalSpec, task: str, sel: LoadResult,
                  out_dir: str | Path, *, resume: bool = False, argv: Sequence[str] = (),
                  run_extra: Mapping[str, Any] | None = None, log: Callable[[str], None] | None = None) -> dict[str, Any]:
    """一个任务的评测（规则见模块说明）。返回 metrics。"""
    log = log or (lambda m: print(m, file=sys.stderr, flush=True))
    samples = list(sel.records)[: spec.limit] if spec.limit else list(sel.records)
    if not samples:
        raise ValueError(f"{task} 的评测集是空的")
    leaf = leaf_dir(out_dir, spec, task)
    ident = run_identity(spec, cfg, settings, task, sel, samples, model)
    if leaf.exists() and any(leaf.iterdir()):
        if not resume:
            raise FileExistsError(f"{leaf} 已存在且非空；续跑请加 --resume")
        if not (leaf / CONFIG).exists():
            raise FileExistsError(f"{leaf} 里没有 {CONFIG}，不能续跑")
        old, new = _identity(yaml.safe_load((leaf / CONFIG).read_text(encoding="utf-8"))), _identity(ident)
        if old != new:
            diff = sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k))
            raise ValueError(f"{leaf}：续跑的运行身份与已有的不同（{diff}），拒绝混写")
    else:
        leaf.mkdir(parents=True, exist_ok=True)
        files = [REPO_ROOT / cfg.path if not Path(cfg.path).is_absolute() else Path(cfg.path)]
        write_run_metadata(leaf, config=ident, seed=int(spec.seed), data_files=files, repo_root=REPO_ROOT)
    (leaf / "logs").mkdir(exist_ok=True)
    (leaf / ARRAYS_DIR).mkdir(exist_ok=True)
    run_k = len(list((leaf / "logs").glob("run_*.json"))) + 1
    run_path = leaf / "logs" / f"run_{run_k:03d}.json"
    run_log: dict[str, Any] = {"argv": list(argv), "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                               "device": str(getattr(model.ad, "device", None)), "score_workers": settings["score_workers"],
                               **dict(run_extra or {}), "finished_utc": None}
    run_path.write_text(json.dumps(run_log, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    t_all = time.perf_counter()
    ids = [s.id for s in samples]
    bounds = chunk_bounds(len(samples), int(settings["batch_size"]))
    done = read_jsonl_tolerant(leaf / SCORES, repair=True)
    if [r["id"] for r in done] != ids[: len(done)]:
        raise RuntimeError(f"{leaf / SCORES} 里已完成的 id 不是样本顺序的前缀，不能续跑")
    keep = max(lo for lo, _ in bounds + [(len(ids), len(ids))] if lo <= len(done))
    if keep < len(done):
        _truncate_lines(leaf / SCORES, keep)
        log(f"[e3a {task}] 已完成 {len(done)} 条不在批的边界上，退回到 {keep} 条整批重跑")
    for ci, (lo, _hi) in enumerate(bounds):
        if lo >= keep and _chunk_file(leaf, ci).exists():
            _chunk_file(leaf, ci).unlink()
    n_run = 0
    for ci, (lo, hi) in enumerate(bounds):
        if lo < keep:
            continue
        chunk = samples[lo:hi]
        payloads, info = run_chunk(model, task, chunk, settings, cfg)
        t_s = time.perf_counter()
        scored = score_all(payloads, int(settings["score_workers"]))
        score_s = time.perf_counter() - t_s
        lines, tlines, arrays = [], [], {}
        n = len(chunk)
        for b, (s, sc) in enumerate(zip(chunk, scored)):
            if sc["id"] != s.id:
                raise RuntimeError("评分结果与样本顺序不符")
            scal, arrs = split_arrays(task, sc["stats"])
            for k, v in arrs.items():
                arrays.setdefault(k, []).append(v)
            rec = {"id": s.id, "chunk": ci, "group": group_of(task, s), "image_key": s.image_keys[0] if s.image_keys else s.id,
                   "shuffle_source": chunk[info["perm"][b]].id, "chunk_same_image_fallback": info["n_same_image_fallback"],
                   "gt": sc["gt"], "prompt_tokens": int(info["prompt_lengths"][b]),
                   "prompt_tokens_no_instruction": None if info["prompt_lengths_no_instruction"] is None
                   else int(info["prompt_lengths_no_instruction"][b]),
                   "visual_tokens": int(info["visual_tokens"][b]), "image_pixels": int(info["image_pixels"][b]),
                   "stats": scal}
            lines.append(json.dumps(rec, ensure_ascii=False, default=_json_default) + "\n")
            tlines.append(_jsonl({"id": s.id, "chunk": ci, "run": run_k, "batch_rows": n, "prep_s": info["prep_s"] / n,
                                  "fwd_s": info["fwd_s"] / n, "fwd_no_instruction_s": info["fwd_no_instruction_s"] / n,
                                  "score_s": score_s / n}))
        if arrays:
            np.savez_compressed(_chunk_file(leaf, ci), **{k: np.stack(v).astype(np.uint32) for k, v in arrays.items()})
        _append(leaf / TIMING, tlines)
        _append(leaf / SCORES, lines)
        n_run += 1
        if n_run == 1 or n_run % 20 == 0 or hi == len(ids):
            log(f"[e3a {spec.variant} {task}] {hi}/{len(ids)}")
    t_fin = time.perf_counter()
    metrics = finalize(leaf, spec, ident, task, cfg, settings, samples)
    run_log.update(finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), chunks_run=n_run,
                   wall_s=round(time.perf_counter() - t_all, 3), finalize_wall_s=round(time.perf_counter() - t_fin, 3))
    run_path.write_text(json.dumps(run_log, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return metrics


def paired_compare(leaf_a: str | Path, leaf_b: str | Path, *, upsample: str | None = None) -> dict[str, Any]:
    """两个评测叶目录（同一任务、同一评测集，例如完整方法对 C1 / C2）按样本 id 对齐，正常模式下标题指标的配对差（a − b）与
    bootstrap CI。upsample 不给时取 a 的默认上采样。"""
    a, b = Path(leaf_a), Path(leaf_b)
    ma, mb = (json.loads((x / METRICS).read_text(encoding="utf-8")) for x in (a, b))
    if ma["task"] != mb["task"]:
        raise ValueError(f"任务不同：{ma['task']} 与 {mb['task']}")
    task = ma["task"]
    ra, _ = load_rows(a, task, arrays=False)
    rb, _ = load_rows(b, task, arrays=False)
    if [r["id"] for r in ra] != [r["id"] for r in rb]:
        raise ValueError("两边的样本 id 不同或顺序不同，不能配对")
    m = upsample or ma["headline"]["upsample"]
    return {"task": task, "upsample": m, "a": {"variant": ma["variant"], "model": ma["model"], "seed": ma["seed"]},
            "b": {"variant": mb["variant"], "model": mb["model"], "seed": mb["seed"]}, "n": len(ra),
            "a_minus_b": paired_diff(task, [r["stats"]["normal"][m] for r in ra], [r["stats"]["normal"][m] for r in rb])}


__all__ = [
    "EvalModel",
    "EvalPaths",
    "EvalSpec",
    "HEADLINE",
    "MERGE_CHECK",
    "aggregate_stats",
    "chunk_bounds",
    "finalize",
    "leaf_dir",
    "load_eval_set",
    "load_rows",
    "model_dir",
    "paired_compare",
    "paired_diff",
    "resolve_eval",
    "run_eval_task",
    "sample_stat",
    "score_payload",
    "semseg_argmax",
    "shuffle_perm",
]
