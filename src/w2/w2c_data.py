"""W2-c 的训练数据（DEV-0060；D113 第二节第 4 条；格式见 docs/w2_plan.md §5）：RefCOCO 系列 train 上，base 模型按 E1 协议出框，
W2-a 冻结的候选生成器出 K 个候选，真值 IoU 定挑选的标签。数据由真值得到，不涉及任何学习型视觉模型。

**总体与去重**（数据目录的 manifest.json，即 <out>/w2c_data/manifest.json，逐步记条数与 sha256；不含机器相关的路径，所以逐字节可复现）：
1. 总体 = E3 的训练池（src/e3/data.load_pool("refseg")，同 C3）：RefCOCO（unc）、RefCOCO+（unc）、RefCOCOg（umd）train 的全部句子，
   训练加载器自动剔除 E3 训练池的剔除清单（<训练数据根>/dedupe/train_exclusions.json，硬约束 7：没有就报错），指代句只有空白的剔除；
   开跑前同 E3 核对剔除清单比较过 configs/e3b_train.json 的 dedupe.required_eval_sets.refseg（check_dedupe）。
2. 按 COCO 图像 id 排除 W2 的全部评测图：configs/w2_splits.json 里 refseg 的全部划分——devtune800、devcheck2、5 个 test 子集
   （eval_split_names 核对恰是这 7 个、devcheck2 已生成），每个划分的句子所在的图（Sample.image_keys，coco:<image_id>）。
   这些图上的训练句子全部去掉（真实数据在落地时已按 D11-1 去掉评测图，预期去掉 0 句；照样核对并记数）。
3. 抽样：剩下的句子按 sha256(salt|purpose|id)（src/data/subsets.hash_rank，salt = w2c_train_v1，purpose = w2c_train；id 是样本 id，
   如 refcoco_unc/train/123，三个数据集的 sent_id 各自编号，带数据集前缀才唯一）升序取前 N 句。**处理与写出都按这个哈希顺序**，
   所以前 k 片恰是 N = k × 1,000 时的抽样。
4. 核对：选中的句子没有一句在评测图上（check_manifest，不要数据也能核对）。

**出框**（GPU）：E1 的运行器原样（src/e1/runner.run_task，leaf = <out>/propose）：refexp_box 的提示、关思考、贪心、E1 冻结的批大小，
输出与续跑照 E1（predictions.jsonl 逐字节可复现；config.yaml 的 data.selection_stats.w2c 记清单的 sha256）。

**候选与标签**（CPU）：每句 = src/w2/runner.candidate_job（提出的框 → generate_candidates，--w2-config 的生成器清单；--candidate-workers N
用 spawn 的进程池，子进程单线程，同 W2）→ src/w2/loop.effective_masks（1 号候选 ok = False 时当空 mask，裁定 A；其余失败的用框）→
各候选与 GT mask 的 IoU → label = oracle_index（真值 IoU 最高的编号，整数精确比较，并列取编号小的）。提出失败时 candidates、label 为 null。

**输出**（<out>/）：
- propose/（E1 格式的运行目录）；
- w2c_data/manifest.json（总体、去重、抽样各步；选中的 id 按处理顺序与它们的图；docs/w2_plan.md §5 的"数据目录的 manifest.json"）；
- w2c_data/<00000>.jsonl…：每片 SHARD_SIZE（1,000）句，格式 docs/w2_plan.md §5（data_row）；先写 .tmp、fsync 之后改名，写完一片才算数；
- config.yaml（运行身份：模型与 revision、w2 配置的路径与 sha256、候选清单与它的 sha256、提出的运行、抽样规则、去重清单的 sha256、片大小）
  与 meta.json；metrics.json（只由各片算，逐字节可复现）、README.md；logs/（timing.jsonl：每句候选生成的墙钟；run_<k>.json）。
续跑：已写完的片核对 id 后跳过，没写完的片整片重算；除耗时（logs/、meta.json）外与一次跑完逐字节相同。
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np
import yaml

from src.constants import BOOTSTRAP_SEED
from src.data import train_dedupe as TD
from src.data.land.refcoco_train import DATASETS as TRAIN_DATASETS
from src.data.records import Sample
from src.data.subsets import dump_config, hash_rank, population_digest, sort_ids
from src.e1.posthoc_runner import _finish_run_log, _start_run_log
from src.e1.runner import (
    CONFIG,
    METRICS,
    PREDICTIONS,
    README,
    _append,
    _identity,
    _jsonl_line,
    ids_digest,
    read_jsonl_tolerant,
    run_task,
)
from src.e1.scoring import distribution, gt_mask
from src.e1.tasks import DataPaths, Selection
from src.executors.raster import mask_to_rle
from src.metrics.seg import mask_inter_union, sample_iou
from src.runlog import write_run_metadata
from src.w1.baselines import e1_spec
from src.w2 import runner as R
from src.w2.config import PROPOSE_BOX, W2Config
from src.w2.data import DEVCHECK2, DEVTUNE, STATUS_PENDING, W2_SPLITS_PATH, load_splits, split_names, tiny_paths, tiny_splits, w2_items
from src.w2.loop import FIRST_NOT_OK_EMPTY, effective_masks, first_as_empty, oracle_index, proposal_of

REPO_ROOT = Path(__file__).resolve().parents[2]
E3B_TRAIN_PATH = REPO_ROOT / "configs" / "e3b_train.json"  # E3 训练池的去重要求（dedupe.required_eval_sets.refseg）
W2C_FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "w2c" / "root"  # --tiny 的训练数据根（tests/fixtures/w2c/build_fixture.py）
FRAMEWORK = "src/w2/w2c_data（DEV-0060；docs/w2_plan.md §5）"
SALT = "w2c_train_v1"
PURPOSE = "w2c_train"
DEFAULT_N = 16_000
SHARD_SIZE = 1_000
TINY_N = 10
TINY_SHARD_SIZE = 4
MANIFEST = "manifest.json"  # 放在数据目录里：<out>/w2c_data/manifest.json
MANIFEST_SCHEMA = 1
PROPOSE_DIR = "propose"
DATA_DIR = "w2c_data"
SHARD_SUFFIX = ".jsonl"
TIMING = "logs/timing.jsonl"
E1_TASK = "refexp_box"
E1_SUBSET = "eval"  # E1 的主口径（提示、关思考、解码与读法都按 eval 子集取）
EVAL_TASK = "refseg"  # W2 的评测划分里要排除其图的那些（refseg 与 refexp_box 共用同一批句子）
ROW_KEYS: tuple[str, ...] = ("id", "source", "split", "image", "image_size", "expression", "gt_box", "gt_rle", "proposal", "candidates",
                             "label", "spec_sha256", "w2_config_sha256")  # docs/w2_plan.md §5 的键，按这个顺序写出
CANDIDATE_KEYS: tuple[str, ...] = ("index", "name", "ok", "rle", "iou")
PROPOSAL_KEYS: tuple[str, ...] = ("box", "failure", "raw_text")
SAMPLING_RULE = (f"总体按图像排除 W2 评测图之后，按 sha256({SALT}|{PURPOSE}|样本 id) 升序（src/data/subsets.hash_rank）取前 n 句；"
                 "处理与写出都按这个顺序")
LABEL_RULE = ("label = 真值 IoU 最高的候选编号（1 起；IoU 按整数 inter / union 精确比较，src/w2/loop.oracle_index，同 oracle_pick），"
              "并列取编号小的；1 号候选 ok = False 时按空 mask 算（裁定 A）；提出失败时 candidates 与 label 为 null")
PROPOSE_PROTOCOL = ("E1 refexp_box（configs/e1_prompts.json 的提示、关思考、贪心、E1 冻结的批大小；src/e1/runner.run_task 原样，"
                    "leaf = <out>/propose）")


class W2CDataError(ValueError):
    """W2-c 的数据不能生成，或与已有的输出对不上。"""


# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class W2CPaths:
    """train_root：训练数据的落地根（含 refcoco_train/ 与 dedupe/train_exclusions.json；服务器上 $VI/data/train_data）；
    eval：W2 评测划分的数据（E1 的落地，服务器上 $VI/b200/data），只用来取评测句子所在的图。"""

    train_root: Path
    eval: DataPaths
    exclusions: str | Path = "auto"
    require_dedupe: bool = True

    @classmethod
    def tiny(cls) -> "W2CPaths":
        return cls(train_root=W2C_FIXTURE_ROOT, eval=tiny_paths())


def shard_name(k: int) -> str:
    return f"{int(k):05d}{SHARD_SUFFIX}"


def shard_plan(n: int, shard_size: int) -> list[tuple[int, int]]:
    """[0, n) 按 shard_size 连续切片：[(起, 止)]，止不含。"""
    if isinstance(shard_size, bool) or not isinstance(shard_size, int) or shard_size < 1:
        raise ValueError(f"shard_size 必须是正整数，收到 {shard_size!r}")
    return [(a, min(a + shard_size, n)) for a in range(0, n, shard_size)]


def shard_paths(out: str | Path) -> list[Path]:
    """<out>/w2c_data/ 下已写完的片（不含 .tmp），按片号升序。"""
    d = Path(out) / DATA_DIR
    return sorted(p for p in d.glob(f"*{SHARD_SUFFIX}") if p.name[: -len(SHARD_SUFFIX)].isdigit()) if d.exists() else []


def read_rows(out: str | Path) -> Iterator[dict[str, Any]]:
    """按片号、片内顺序逐行读 W2-c 的数据（DEV-0061 用）。"""
    for p in shard_paths(out):
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def manifest_path(out: str | Path) -> Path:
    return Path(out) / DATA_DIR / MANIFEST


def load_manifest(out: str | Path) -> dict[str, Any]:
    return json.loads(manifest_path(out).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 规则（纯函数）
# ---------------------------------------------------------------------------


def _sha_list(items: Iterable[str]) -> str:
    """有序列表的指纹（换行连接后的 sha256，同 src/e1/runner.ids_digest）。"""
    return ids_digest([str(i) for i in items])


def select_sentences(population: Sequence[tuple[str, str]], eval_images: Iterable[str], *, n: int, salt: str = SALT,
                     purpose: str = PURPOSE, allow_fewer: bool = False) -> dict[str, Any]:
    """去重第 2 步与抽样（模块说明）。population = 总体的 (样本 id, 图像键)；eval_images = 评测图的图像键。
    返回 ids（哈希顺序，即处理顺序）、image_of、各步的条数与 sha256。allow_fewer = False 时剩下的不够 n 句就报错（--tiny 取全部）。"""
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError(f"n 必须是正整数，收到 {n!r}")
    ids = [str(i) for i, _ in population]
    if len(set(ids)) != len(ids):
        raise W2CDataError("总体里的样本 id 有重复")
    img = {str(i): str(k) for i, k in population}
    bad = {str(k) for k in eval_images}
    removed = [i for i in ids if img[i] in bad]
    kept = [i for i in ids if img[i] not in bad]
    if len(kept) < n and not allow_fewer:
        raise W2CDataError(f"按图像排除评测图之后只剩 {len(kept)} 句，不够 n = {n} 句")
    chosen = hash_rank(kept, purpose, salt)[:n]
    return {
        "ids": chosen,
        "image_of": {i: img[i] for i in chosen},
        "steps": {
            "w2_eval_exclusion": {"n_before": len(ids), "sha256_before": population_digest(ids), "n_removed": len(removed),
                                  "removed_sha256": population_digest(removed), "n_images_hit": len({img[i] for i in removed}),
                                  "n_after": len(kept), "sha256_after": population_digest(kept)},
            "selected": {"salt": salt, "purpose": purpose, "n_requested": n, "n": len(chosen), "sha256": population_digest(chosen),
                         "order": "hash_rank 升序（处理与写出的顺序）", "ordered_sha256": _sha_list(chosen),
                         "images_n": len(set(img[i] for i in chosen))},
        },
    }


def check_manifest(doc: Mapping[str, Any]) -> list[str]:
    """manifest.json 的自洽（不要数据）：选中的句子与评测图不重叠、条数与 sha256、顺序是哈希顺序。返回问题列表（空 = 通过）。"""
    bad: list[str] = []
    sel = doc["selected"]
    ids = [str(i) for i in sel["ids"]]
    st = doc["steps"]["selected"]
    if len(set(ids)) != len(ids):
        bad.append("选中的 id 有重复")
    if len(ids) != st["n"] or population_digest(ids) != st["sha256"] or _sha_list(ids) != st["ordered_sha256"]:
        bad.append("选中的 id 与记的条数或 sha256 不符")
    if len(set(ids)) == len(ids) and ids != hash_rank(ids, st["purpose"], st["salt"]):
        bad.append("选中的 id 不是哈希顺序")
    img = sel["image_of"]
    if set(img) != set(ids):
        bad.append("image_of 与选中的 id 对不上")
    ev = doc["steps"]["w2_eval_images"]
    images = set(ev["images"])
    if len(images) != ev["n_images"] or population_digest(sorted(images)) != ev["images_sha256"]:
        bad.append("评测图的清单与它记的条数或 sha256 不符")
    hit = sorted(i for i in ids if img.get(i) in images)
    if hit:
        bad.append(f"{len(hit)} 句选中的句子在 W2 的评测图上，例如 {hit[:3]}")
    union = set()
    for name, e in ev["splits"].items():
        union |= set(e["images"])
        if len(e["images"]) != e["n_images"] or population_digest(e["images"]) != e["images_sha256"]:
            bad.append(f"评测划分 {name} 的图与它记的条数或 sha256 不符")
    if union != images:
        bad.append("各评测划分的图的并集不等于评测图的清单")
    return bad


def score_candidates(row: Mapping[str, Any], gt: np.ndarray) -> tuple[list[dict[str, Any]] | None, int | None]:
    """candidates.jsonl 格式的一行（src/w2/loop.candidates_row）与 GT mask → (§5 的 candidates, label)。
    mask 用 effective_masks（1 号候选 ok = False 时为空 mask，裁定 A；其余失败的仍是框），RLE 随之（1 号当空时重编码空 mask，其余原样）；
    iou = src/metrics/seg.sample_iou；label = oracle_index（整数精确比较，并列取编号小的）。提出失败（没有候选）返回 (None, None)。"""
    if row["candidates"] is None:
        return None, None
    w, h = row["image_size"]
    if gt.shape != (h, w):
        raise W2CDataError(f"{row['id']}：GT mask 形状 {gt.shape} 与原图 (H, W) = {(h, w)} 不符")
    masks = effective_masks(row)
    empty1 = first_as_empty(row)
    pairs: list[tuple[int, int]] = []
    out: list[dict[str, Any]] = []
    for c, m in zip(row["candidates"], masks):
        st = mask_inter_union(m, gt)
        pairs.append((int(st.inter), int(st.union)))
        rle = mask_to_rle(m) if (c["index"] == 1 and empty1) else c["mask"]["rle"]
        out.append({"index": int(c["index"]), "name": c["name"], "ok": bool(c["ok"]), "rle": rle,
                    "iou": sample_iou(int(st.inter), int(st.union))})
    return out, oracle_index(pairs)


def w2c_job(job: tuple[str, tuple[dict[str, Any], ...], int, Sample, dict[str, Any]]) -> tuple[dict[str, Any], float]:
    """一句的候选与标签（单进程与进程池共用；在子进程里跑，所以要能 pickle）：src/w2/runner.candidate_job 原样生成候选，
    再与 GT 算 IoU、定标签。返回 ({id, image_size, gt_rle, candidates, label}, 候选生成的墙钟)。"""
    sample = job[3]
    row, wall = R.candidate_job(job)
    gt = gt_mask(sample)
    w, h = row["image_size"]
    if gt.shape != (h, w):
        raise W2CDataError(f"{sample.id}：GT mask 形状 {gt.shape} 与原图 (H, W) = {(h, w)} 不符")
    cands, label = score_candidates(row, gt)
    return {"id": sample.id, "image_size": [int(w), int(h)], "gt_rle": mask_to_rle(gt), "candidates": cands, "label": label}, wall


def data_row(sample: Sample, image_rel: str, proposal_rec: Mapping[str, Any], res: Mapping[str, Any], *, spec_sha256: str,
             w2_config_sha256: str) -> dict[str, Any]:
    """docs/w2_plan.md §5 的一行（键序 = ROW_KEYS）。proposal_rec 是 E1 refexp_box 的预测记录，res 是 w2c_job 的结果。"""
    if res["id"] != sample.id or proposal_rec["id"] != sample.id:
        raise W2CDataError(f"{sample.id}：候选或提出的记录对不上（{res['id']}、{proposal_rec['id']}）")
    prop = proposal_of(PROPOSE_BOX, proposal_rec)
    if (prop["box"] is None) != (res["candidates"] is None):
        raise W2CDataError(f"{sample.id}：提出的框与候选对不上（box = {prop['box']!r}，有候选 = {res['candidates'] is not None}）")
    row = {"id": sample.id, "source": sample.source, "split": sample.split, "image": image_rel, "image_size": list(res["image_size"]),
           "expression": sample.fields["expression"], "gt_box": [float(v) for v in sample.gt["box"]], "gt_rle": res["gt_rle"],
           "proposal": {"box": prop["box"], "failure": prop["failure"], "raw_text": proposal_rec["raw_text"]},
           "candidates": res["candidates"], "label": res["label"], "spec_sha256": spec_sha256, "w2_config_sha256": w2_config_sha256}
    assert tuple(row) == ROW_KEYS
    return row


def check_row(row: Mapping[str, Any], *, k: int | None = None) -> list[str]:
    """一行是否合 §5 的格式（键、类型、编号、label 的规则）。返回问题列表（空 = 合格）。"""
    bad: list[str] = []
    if tuple(row) != ROW_KEYS:
        bad.append(f"键 {list(row)} ≠ {list(ROW_KEYS)}")
        return bad
    p = row["proposal"]
    if tuple(p) != PROPOSAL_KEYS:
        bad.append(f"proposal 的键 {list(p)}")
    if (p["box"] is None) == (p["failure"] is None):
        bad.append("proposal 要么有框、要么有失败原因")
    c = row["candidates"]
    if p["failure"] is not None:
        if c is not None or row["label"] is not None:
            bad.append("提出失败时 candidates、label 必须为 null")
        return bad
    if not isinstance(c, list) or not c or (k is not None and len(c) != k):
        bad.append(f"candidates 应是 K = {k} 个候选的列表")
        return bad
    if any(tuple(x) != CANDIDATE_KEYS for x in c) or [x["index"] for x in c] != list(range(1, len(c) + 1)):
        bad.append("candidates 的键或编号不对")
        return bad
    best = max(x["iou"] for x in c)
    if row["label"] != next(x["index"] for x in c if x["iou"] == best) and best != 1.0:  # 1.0 的例外见 data_metrics
        bad.append(f"label {row['label']} 不是 IoU 最高、编号最小的候选")
    if c[0]["ok"] is False and c[0]["rle"] != mask_to_rle(np.zeros(tuple(c[0]["rle"]["size"]), dtype=bool)):
        bad.append("1 号候选 ok = False 时应为空 mask")
    return bad


# ---------------------------------------------------------------------------
# 统计（metrics.json：只由各片的行算）
# ---------------------------------------------------------------------------


def _mean(v: Sequence[float]) -> float | None:
    return float(np.mean(np.asarray(v, dtype=np.float64))) if v else None


def _rate(k: int, n: int) -> float | None:
    return (k / n) if n else None


def data_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """出框的失败率、候选各编号作为最优的次数（并列都算）、标签的分布、最优候选与 1 号（W1-F）的平均 IoU、各编号的平均 IoU 与失败数、
    重复率。label 与 IoU 的自洽在这里再核一遍：label 必须是 iou 最高的编号里最小的那个。唯一合法的例外是 GT 为空、候选也为空
    （并集为 0）：sample_iou 记 1.0（LISA 的口径），oracle_index 按 0 比（同 W2 的 oracle_pick），两者可以不同——这种句子计数
    （n_label_union0_exception），其余不一致就报错。"""
    n = len(rows)
    fails = Counter("ok" if r["proposal"]["failure"] is None else str(r["proposal"]["failure"]) for r in rows)
    wc = [r for r in rows if r["candidates"] is not None]
    d = len(wc)
    k = len(wc[0]["candidates"]) if wc else 0
    names = [c["name"] for c in wc[0]["candidates"]] if wc else []
    labels: Counter = Counter()
    ties: Counter = Counter()
    n_tie_rows = n_union0 = 0
    best, first, dup = [], [], 0
    not_ok: Counter = Counter()
    per: dict[int, list[float]] = {i: [] for i in range(1, k + 1)}
    for r in wc:
        c = r["candidates"]
        if len(c) != k or [x["name"] for x in c] != names:
            raise W2CDataError(f"{r['id']}：候选的个数或名字与第一行不同")
        ious = [float(x["iou"]) for x in c]
        m = max(ious)
        winners = [x["index"] for x in c if x["iou"] == m]
        if r["label"] != winners[0]:
            if not (m == 1.0 and all(v in (0.0, 1.0) for v in ious) and r["label"] == 1):  # GT 为空：oracle 全按 0 比，取 1 号
                raise W2CDataError(f"{r['id']}：label {r['label']} 不是 IoU 最高、编号最小的候选 {winners[0]}")
            n_union0 += 1
        labels[r["label"]] += 1
        ties.update(winners)
        n_tie_rows += len(winners) > 1
        best.append(m)
        first.append(ious[0])
        seen: set[str] = set()
        for x in c:
            per[x["index"]].append(float(x["iou"]))
            not_ok[x["index"]] += x["ok"] is False
            key = json.dumps(x["rle"], sort_keys=True)
            dup += key in seen
            seen.add(key)
    lab = {str(i): labels[i] for i in range(1, k + 1)}
    lab["null"] = n - d
    return {
        "n": n,
        "proposal": {"n_failed": n - fails["ok"], "failure_rate": _rate(n - fails["ok"], n),
                     "failures": {kk: fails[kk] for kk in sorted(fails) if kk != "ok"}},
        "n_with_candidates": d, "k": k, "names": names,
        "label": {"counts": lab, "rate_of_with_candidates": {str(i): _rate(labels[i], d) for i in range(1, k + 1)}, "rule": LABEL_RULE,
                  "n_label_union0_exception": n_union0},
        "best_with_ties": {"counts": {str(i): ties[i] for i in range(1, k + 1)}, "n_rows_with_ties": n_tie_rows,
                           "rule": "每句里 IoU 等于最高 IoU 的每个编号各记一次（并列都算）；label 只取其中编号最小的"},
        "best_iou": {"mean_on_rows_with_candidates": _mean(best), "distribution": distribution(best)},
        "first_iou": {"mean_on_rows_with_candidates": _mean(first),
                      "mean_on_all_rows_failure_as_0": _mean(first + [0.0] * (n - d)) if n else None,
                      "distribution": distribution(first),
                      "note": "1 号候选 = W1-F（E1 的框交给默认 GrabCut；GrabCut 失败按空 mask）；提出失败的句子 W1-F 交空，按 0 计"},
        "best_minus_first_mean": (_mean(best) - _mean(first)) if d else None,
        "per_index": {str(i): {"name": names[i - 1], "mean_iou": _mean(per[i]), "n_not_ok": not_ok[i], "n_label": labels[i],
                               "n_best_with_ties": ties[i]} for i in range(1, k + 1)},
        "duplicates": {"n": dup, "rate": _rate(dup, d * k),
                       "rule": "与编号更小的某个候选 RLE 完全相同的候选数（不去重，照常编号）"},
    }


# ---------------------------------------------------------------------------
# 总体、去重与抽样（要数据）
# ---------------------------------------------------------------------------


def required_eval_sets() -> list[str]:
    """E3 训练池的去重要求：configs/e3b_train.json 的 dedupe.required_eval_sets.refseg（E3 的 check_dedupe 同一份名单）。"""
    return list(json.loads(E3B_TRAIN_PATH.read_text(encoding="utf-8"))["dedupe"]["required_eval_sets"]["refseg"])


def load_train_pool(paths: W2CPaths, *, real_model: bool) -> tuple[list[Sample], dict[str, Any]]:
    """去重第 1 步：E3 的训练池（load_pool("refseg")，三个数据集的 train 句子，剔除 E3 的剔除清单与空句）与剔除清单的核对。
    返回 (样本, manifest 的 population 与 e3_exclusions 两步)。"""
    from src.e3.data import E3Paths, check_dedupe, load_pool  # 按需 import（E3 的模块较重）

    e3p = E3Paths(refseg_root=Path(paths.train_root), depth_root=Path(paths.train_root), exclusions=paths.exclusions,
                  require_dedupe=paths.require_dedupe)
    res = load_pool("refseg", e3p, {"datasets": list(TRAIN_DATASETS)})
    dedupe_file = res.stats.get("dedupe_file")
    chk = check_dedupe("refseg", dedupe_file, required_eval_sets(), real_model=real_model)
    for s in res.records:
        if len(s.image_keys) != 1 or not str(s.image_keys[0]).startswith("coco:"):
            raise W2CDataError(f"{s.id}：RefCOCO 的训练句子应恰有一个 coco: 图像键，收到 {s.image_keys}")
    ids = [s.id for s in res.records]
    per = {ds: {k: v for k, v in st.items() if k in ("n_total", "n_missing", "n_dedupe_excluded", "n_kept")}
           for ds, st in res.stats["datasets"].items()}
    excl: dict[str, Any] = {"file": None, "sha256": None}
    if dedupe_file is not None:
        p = Path(dedupe_file)
        entries = sorted(TD.load_exclusions(p).get("refcoco_train/train", {}))
        excl = {"file": _rel_to(p, Path(paths.train_root)), "sha256": chk["file_sha256"], "schema": chk["schema"], "max_hamming": chk["max_hamming"],
                "eval_sets": chk["eval_sets"], "required_eval_sets": chk["required_eval_sets"],
                "required_from": "configs/e3b_train.json 的 dedupe.required_eval_sets.refseg",
                "n_refcoco_train_entries": len(entries), "refcoco_train_entries_sha256": population_digest(entries)}
    steps = {
        "population": {"source": "src/e3/data.load_pool('refseg')（E3 / C3 的训练池）", "datasets": per,
                       "n_empty_expression": int(res.stats.get("n_empty_expression", 0)),
                       "n": len(ids), "sha256": population_digest(ids),
                       "rule": "三个数据集 train 的全部句子；训练加载器剔除 E3 训练池的剔除清单（按图）；指代句去首尾空白后为空的剔除"},
        "e3_exclusions": {**excl, "n_sentences_removed": int(sum(v.get("n_dedupe_excluded", 0) for v in per.values()))},
    }
    return list(res.records), steps


def eval_split_names(cfg: W2Config, doc: Mapping[str, Any]) -> list[str]:
    """W2 的全部评测划分（要排除其图）：devtune800、devcheck2 与配置的 5 个 test 子集；核对清单里 refseg 的划分恰是这些、devcheck2 已生成。"""
    want = [DEVTUNE, DEVCHECK2] + [f"test_{k}" for k in cfg.splits["test"]]
    got = split_names(EVAL_TASK, doc)
    if sorted(got) != sorted(want):
        raise W2CDataError(f"W2 清单里 {EVAL_TASK} 的划分 {got} 与应排除的 {want} 不同")
    if doc["splits"][DEVCHECK2].get("status") == STATUS_PENDING:
        raise W2CDataError("devcheck2 还没生成（status = pending），排除不了它的图")
    return want


def eval_images(cfg: W2Config, doc: Mapping[str, Any], eval_paths: DataPaths) -> dict[str, Any]:
    """去重第 2 步的评测图：每个评测划分的句子（w2_items，清单的 id 与 sha256 都核对）所在的图。devcheck2 另核对清单的 image_of。"""
    splits: dict[str, Any] = {}
    union: set[str] = set()
    for name in eval_split_names(cfg, doc):
        samples = w2_items(EVAL_TASK, name, eval_paths, splits=doc)
        keys = []
        for s in samples:
            if len(s.image_keys) != 1:
                raise W2CDataError(f"{s.id}：RefCOCO 应是单图，收到 {len(s.image_keys)} 个图像键")
            keys.append(str(s.image_keys[0]))
        if name == DEVCHECK2:
            want = doc["splits"][DEVCHECK2]["image_of"]
            got = {str(s.fields["sent_id"]): str(s.image_keys[0]) for s in samples}
            if got != {str(k): str(v) for k, v in want.items()}:
                raise W2CDataError("devcheck2 的句子所在的图与清单的 image_of 不同")
        imgs = sort_ids(set(keys))
        e = doc["splits"][name]
        splits[name] = {"n_sentences": len(samples), "sentences_sha256": e["sha256"], "n_images": len(imgs),
                        "images_sha256": population_digest(imgs), "images": imgs}
        union |= set(imgs)
    images = sort_ids(union)
    return {"splits": splits, "n_images": len(images), "images_sha256": population_digest(images), "images": images}


def build_manifest(pool_steps: Mapping[str, Any], ev: Mapping[str, Any], sel: Mapping[str, Any], *, tiny: bool,
                   splits_source: Mapping[str, Any]) -> dict[str, Any]:
    steps = dict(pool_steps)
    steps["w2_eval_images"] = {"source": dict(splits_source), "task": EVAL_TASK, **ev,
                               "rule": "W2 的全部评测划分（devtune800、devcheck2、5 个 test 子集）的句子所在的 COCO 图"}
    steps.update(sel["steps"])
    ev_set = set(ev["images"])
    doc = {"schema": MANIFEST_SCHEMA, "kind": "w2c_train_selection", "framework": FRAMEWORK, "tiny": tiny,
           "rule": "总体 = E3 的训练池（剔除 E3 的剔除清单与空句）→ 按 COCO 图像 id 排除 W2 的全部评测图 → " + SAMPLING_RULE,
           "steps": steps,
           "checks": {"selected_on_w2_eval_images": sum(1 for i in sel["ids"] if sel["image_of"][i] in ev_set)},
           "selected": {"ids": list(sel["ids"]), "image_of": dict(sel["image_of"])}}
    problems = check_manifest(doc)
    if problems or doc["checks"]["selected_on_w2_eval_images"]:
        raise W2CDataError("去重核对不过：" + "；".join(problems or ["选中的句子在评测图上"]))
    if pool_steps["population"]["sha256"] != sel["steps"]["w2_eval_exclusion"]["sha256_before"]:
        raise W2CDataError("去重第 2 步的输入不是第 1 步的总体")
    return doc


def prepare(cfg: W2Config, paths: W2CPaths, *, n: int, tiny: bool) -> tuple[dict[str, Any], list[Sample]]:
    """总体 → 去重 → 抽样。返回 (manifest, 选中的样本（处理顺序）)。"""
    pool, pool_steps = load_train_pool(paths, real_model=not tiny)
    doc = tiny_splits(cfg) if tiny else load_splits(W2_SPLITS_PATH)
    src = ({"path": "tiny（src/w2/data.tiny_splits，tests/fixtures 上现算）"} if tiny else
           {"path": "configs/w2_splits.json", "sha256": hashlib.sha256(W2_SPLITS_PATH.read_bytes()).hexdigest()})
    ev = eval_images(cfg, doc, paths.eval)
    sel = select_sentences([(s.id, s.image_keys[0]) for s in pool], ev["images"], n=n, allow_fewer=tiny)
    manifest = build_manifest(pool_steps, ev, sel, tiny=tiny, splits_source=src)
    by = {s.id: s for s in pool}
    return manifest, [by[i] for i in sel["ids"]]


def manifest_text(doc: Mapping[str, Any]) -> str:
    return dump_config(doc)


def ensure_manifest(out: str | Path, doc: Mapping[str, Any]) -> str:
    """<out>/w2c_data/manifest.json：没有就写；已有必须逐字节相同（数据、清单或规则变了就拒绝混用）。返回它的 sha256。"""
    text = manifest_text(doc)
    p = manifest_path(out)
    if p.exists():
        if p.read_text(encoding="utf-8") != text:
            raise W2CDataError(f"{p} 与这次重算的不同（数据、去重清单或抽样规则变了），拒绝混用；换一个 --out")
    else:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# 出框（E1 的运行器原样）
# ---------------------------------------------------------------------------


def propose_leaf(out: str | Path) -> Path:
    return Path(out) / PROPOSE_DIR


def propose_selection(samples: Sequence[Sample], manifest_sha256: str) -> Selection:
    """E1 的 Selection：selection_stats.w2c 记清单的 sha256 与抽样规则（进 config.yaml，续跑核对）。"""
    w2c = {"manifest_sha256": manifest_sha256, "n": len(samples), "salt": SALT, "purpose": PURPOSE, "order": "hash_rank 升序"}
    return Selection(E1_TASK, E1_SUBSET, list(samples), True, {"w2c": w2c, "n": len(samples)})


def run_propose(adapter: Any, entry: Any, policy: Any, samples: Sequence[Sample], out: str | Path, *, manifest_sha256: str,
                paths: DataPaths, prompts: Any, batch_size: int, tiny: bool, argv: Sequence[str] = (), warmup: bool = True,
                log: Callable[[str], None] | None = None) -> dict[str, Any]:
    """E1 refexp_box 在选中的句子上跑（src/e1/runner.run_task 原样，续跑语义同 E1），返回 E1 的 metrics。"""
    rs = e1_spec(entry, adapter, policy, E1_TASK, E1_SUBSET, batch_size, tiny=tiny, limit=None)
    return run_task(adapter, rs, propose_selection(samples, manifest_sha256), prompts, paths, out, resume=True, argv=argv,
                    leaf=propose_leaf(out), warmup=warmup, log=log)


def propose_complete(out: str | Path, samples: Sequence[Sample], *, model_key: str, revision: str, tiny: bool,
                     manifest_sha256: str) -> bool:
    """出框已经跑完：predictions 的 id 恰是选中的句子（处理顺序），metrics.json 已写，config.yaml 的模型与清单的 sha256 相符。"""
    leaf = propose_leaf(out)
    if not all((leaf / f).exists() for f in (CONFIG, PREDICTIONS, METRICS)):
        return False
    recs = read_jsonl_tolerant(leaf / PREDICTIONS, repair=False)
    if [r["id"] for r in recs] != [s.id for s in samples]:
        return False
    c = yaml.safe_load((leaf / CONFIG).read_text(encoding="utf-8"))
    w2c = ((c.get("data") or {}).get("selection_stats") or {}).get("w2c") or {}
    return (c["model"]["key"], c["model"]["revision"], bool(c["model"].get("tiny")), c["task"], w2c.get("manifest_sha256")) == \
        (model_key, revision, tiny, E1_TASK, manifest_sha256)


def matched_proposals(samples: Sequence[Sample], recs: Sequence[Mapping[str, Any]], prompts: Any, family: str) -> dict[str, Mapping[str, Any]]:
    """按 id 对上提出的记录并核对同一句话（E1 提示原文）与同一张图（原图尺寸）；对不上就报错。"""
    by = {r["id"]: r for r in recs}
    bad: dict[str, list[str]] = {"missing_id": [], "instruction_differs": [], "image_size_differs": []}
    for s in samples:
        r = by.get(s.id)
        if r is None:
            bad["missing_id"].append(s.id)
        elif r["instruction"] != prompts.render(E1_TASK, family, s).text:
            bad["instruction_differs"].append(s.id)
        elif [int(v) for v in r["image_size"]] != [int(v) for v in s.fields["image_size"]]:
            bad["image_size_differs"].append(s.id)
    n_bad = {k: len(v) for k, v in bad.items() if v}
    if n_bad:
        raise W2CDataError(f"提出的记录与样本对不上：{n_bad}（例如 {[v[:3] for v in bad.values() if v]}）")
    return {s.id: by[s.id] for s in samples}


# ---------------------------------------------------------------------------
# 候选、标签与分片
# ---------------------------------------------------------------------------


class _W2CRunner(R._CandidateRunner):
    """W2 的候选进程池（spawn、子进程单线程；workers = 1 时本进程、OpenCV 临时 1 个线程），每句跑 w2c_job（候选 + IoU + 标签）。"""

    def map(self, jobs: Sequence[Any]) -> list[tuple[dict[str, Any], float]]:
        if self.pool is None:
            return [w2c_job(j) for j in jobs]
        return list(self.pool.map(w2c_job, jobs))  # 按输入顺序返回


def model_record(entry: Any, *, tiny: bool) -> dict[str, Any]:
    return {"key": entry.key, "family": entry.family, "repo": entry.repo, "revision": entry.revision, "tiny": bool(tiny)}


def data_identity(cfg: W2Config, entry: Any, manifest: Mapping[str, Any], manifest_sha256: str, out: str | Path, *, tiny: bool,
                  candidates_module: str, shard_size: int) -> dict[str, Any]:
    """<out>/config.yaml：W2-c 数据的运行身份（不含时间与机器相关的路径）。"""
    st = manifest["steps"]
    return {
        "experiment": "W2", "kind": "w2c_train_data", "framework": FRAMEWORK, "format": "docs/w2_plan.md §5",
        "model": model_record(entry, tiny=tiny),
        "w2_config": {**cfg.identity(), "propose_mode_in_config": cfg.propose_mode},
        "candidates": {"module": candidates_module, "generators": list(cfg.generators), "k": cfg.k, "spec_sha256": cfg.candidates_digest(),
                       "first_not_ok": FIRST_NOT_OK_EMPTY, "rule": "src/w2/runner.candidate_job（同 W2-a 的候选缓存）"},
        "propose": {"mode": PROPOSE_BOX, "protocol": PROPOSE_PROTOCOL, "run": R._run_ref(out, propose_leaf(out))},
        "sampling": {"salt": SALT, "purpose": PURPOSE, "n_requested": st["selected"]["n_requested"], "n": st["selected"]["n"],
                     "ordered_sha256": st["selected"]["ordered_sha256"], "rule": SAMPLING_RULE},
        "dedupe": {"manifest": f"{DATA_DIR}/{MANIFEST}", "manifest_sha256": manifest_sha256,
                   "e3_exclusions_sha256": st["e3_exclusions"].get("sha256"),
                   "w2_splits": st["w2_eval_images"]["source"], "w2_eval_images_sha256": st["w2_eval_images"]["images_sha256"]},
        "label_rule": LABEL_RULE,
        "shard_size": shard_size,
        "image_root": "image 是相对训练数据根（--train-data-root）的路径",
    }


def _prepare_root(out: Path, cfg_doc: Mapping[str, Any], files: Sequence[str | Path]) -> None:
    """<out>/config.yaml：没有就写（连同 meta.json）；已有必须身份相同（同 E1 的续跑规则），否则拒绝混写。"""
    p = out / CONFIG
    if p.exists():
        old = yaml.safe_load(p.read_text(encoding="utf-8"))
        a, b = _identity(old), _identity(cfg_doc)
        if a != b:
            diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
            raise W2CDataError(f"{out}：已有数据的身份与这次的不同（{diff}），拒绝混写；换一个 --out")
    else:
        if shard_paths(out):
            raise W2CDataError(f"{out / DATA_DIR} 已有内容却没有 {CONFIG}，不能续跑")
        write_run_metadata(out, config=dict(cfg_doc), seed=BOOTSTRAP_SEED, data_files=list(files), repo_root=REPO_ROOT)
    (out / "logs").mkdir(parents=True, exist_ok=True)


def _rel_to(p: str | Path, root: str | Path) -> str:
    """p 相对 root 的路径（正斜杠；两者都由同一个根拼出，通常直接 relative_to）。"""
    try:
        return Path(p).relative_to(Path(root)).as_posix()
    except ValueError:
        return Path(os.path.relpath(Path(p).resolve(), Path(root).resolve())).as_posix()


def _image_rel(sample: Sample, train_root: Path) -> str:
    return _rel_to(sample.images[0], train_root)


def _write_shard(path: Path, lines: Sequence[str]) -> None:
    """先写 <片>.tmp，flush + fsync 之后改名：改名之后这一片才算写完。"""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("".join(lines))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _check_done_shard(path: Path, want_ids: Sequence[str]) -> None:
    got = [json.loads(x)["id"] for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if got != list(want_ids):
        raise W2CDataError(f"{path}：已写完的片的 id 与选中的句子对不上（{len(got)} 行，应 {len(want_ids)} 行），拒绝续跑")


def run_candidates(cfg: W2Config, entry: Any, samples: Sequence[Sample], out: str | Path, *, manifest: Mapping[str, Any],
                   manifest_sha256: str, train_root: str | Path, prompts: Any, tiny: bool, shard_size: int, workers: int = 1,
                   candidates_module: str = R.DEFAULT_CANDIDATES_MODULE, eval_files: Sequence[str | Path] = (),
                   stop_after_shards: int | None = None, argv: Sequence[str] = (),
                   log: Callable[[str], None] | None = None) -> dict[str, Any]:
    """候选与标签，按片写出（规则见模块说明），返回 metrics。出框必须已跑完（propose_complete）。"""
    log = log or (lambda m: print(m, file=sys.stderr, flush=True))
    out = Path(out)
    if not propose_complete(out, samples, model_key=entry.key, revision=entry.revision, tiny=tiny, manifest_sha256=manifest_sha256):
        raise W2CDataError(f"{propose_leaf(out)}：出框还没跑完（或不是这个模型、这份清单）；先跑 --stage propose")
    recs = matched_proposals(samples, read_jsonl_tolerant(propose_leaf(out) / PREDICTIONS, repair=False), prompts, entry.family)
    cfg_doc = data_identity(cfg, entry, manifest, manifest_sha256, out, tiny=tiny, candidates_module=candidates_module,
                            shard_size=shard_size)
    files = [cfg.path, *([] if tiny else [W2_SPLITS_PATH]), E3B_TRAIN_PATH, *eval_files]
    _prepare_root(out, cfg_doc, files)
    data_dir = out / DATA_DIR
    data_dir.mkdir(exist_ok=True)
    plan = shard_plan(len(samples), shard_size)
    names = {shard_name(k) for k in range(len(plan))}
    extra = sorted(p.name for p in data_dir.iterdir() if p.name not in names | {MANIFEST} and not p.name.endswith(".tmp"))
    if extra:
        raise W2CDataError(f"{data_dir} 里有不属于这份清单的文件：{extra[:5]}")
    done = []
    for k, (a, b) in enumerate(plan):
        p = data_dir / shard_name(k)
        if p.exists():
            _check_done_shard(p, [s.id for s in samples[a:b]])
            done.append(k)
    todo = [k for k in range(len(plan)) if k not in done]
    if stop_after_shards is not None:
        todo = todo[: max(int(stop_after_shards), 0)]
    spec_sha, cfg_sha = cfg.candidates_digest(), cfg.sha256
    gens, kk = cfg.generators, cfg.k
    if todo:
        run_k, run_path, run_log = _start_run_log(out, argv, stage="candidates", n_shards=len(plan), n_shards_done_before=len(done),
                                                  shards_this_run=todo, shard_size=shard_size, candidate_workers=workers,
                                                  stop_after_shards=stop_after_shards)
        t_all = time.perf_counter()
        walls: list[float] = []
        with _W2CRunner(workers) as runner:
            for k in todo:
                a, b = plan[k]
                chunk = list(samples[a:b])
                for t in data_dir.glob(shard_name(k) + ".tmp"):
                    t.unlink()
                jobs = [(candidates_module, gens, kk, s, proposal_of(PROPOSE_BOX, recs[s.id])) for s in chunk]
                t0 = time.perf_counter()
                results = runner.map(jobs)
                lines, tlines = [], []
                for s, (res, wall) in zip(chunk, results):
                    row = data_row(s, _image_rel(s, Path(train_root)), recs[s.id], res, spec_sha256=spec_sha, w2_config_sha256=cfg_sha)
                    lines.append(json.dumps(row, ensure_ascii=False) + "\n")
                    tlines.append(_jsonl_line({"id": s.id, "run": run_k, "shard": k, "wall_time_s": wall}))
                    walls.append(wall)
                _write_shard(data_dir / shard_name(k), lines)
                _append(out / TIMING, tlines)
                log(f"[W2-c {entry.key}] 片 {k + 1}/{len(plan)}（{b}/{len(samples)} 句）写完，{time.perf_counter() - t0:.1f} s")
        run_log["candidate_wall_s"] = distribution(walls)
        _finish_run_log(run_path, run_log, n_batches=len(todo), t_all=t_all)
    return finalize(out, entry, cfg, manifest_sha256, n_expected=len(samples), n_shards=len(plan))


def finalize(out: Path, entry: Any, cfg: W2Config, manifest_sha256: str, *, n_expected: int, n_shards: int) -> dict[str, Any]:
    """metrics.json 与 README.md：只由已写完的各片算（逐字节可复现）。"""
    rows = list(read_rows(out))
    body = data_metrics(rows)
    n_done = len(shard_paths(out))
    m = {"experiment": "W2", "kind": "w2c_train_data", "model": entry.key, "revision": entry.revision,
         "w2_config_sha256": cfg.sha256, "spec_sha256": cfg.candidates_digest(), "manifest_sha256": manifest_sha256,
         **body, "n_expected": n_expected, "n_shards": {"done": n_done, "expected": n_shards},
         "complete": len(rows) == n_expected and n_done == n_shards}
    (out / METRICS).write_text(json.dumps(m, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / README).write_text(readme_text(m), encoding="utf-8")
    return m


def _fmt(v: Any) -> str:
    return "null" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))


def readme_text(m: Mapping[str, Any]) -> str:
    lab = m["label"]["counts"]
    return "\n".join([
        f"# W2-c 训练数据 · {m['model']}", "",
        f"- 格式：docs/w2_plan.md §5；w2c_data/ 下 {m['n_shards']['done']}/{m['n_shards']['expected']} 片，{m['n']}/{m['n_expected']} 句"
        f"（complete = {m['complete']}）",
        f"- w2 配置 sha256 {m['w2_config_sha256'][:16]}…；候选清单 sha256 {m['spec_sha256'][:16]}…（K = {m['k']}：{m['names']}）；"
        f"manifest.json sha256 {m['manifest_sha256'][:16]}…",
        f"- 出框失败 {m['proposal']['n_failed']}（{_fmt(m['proposal']['failure_rate'])}）：{m['proposal']['failures']}",
        f"- 最优候选的平均 IoU {_fmt(m['best_iou']['mean_on_rows_with_candidates'])}；1 号（W1-F）的平均 IoU "
        f"{_fmt(m['first_iou']['mean_on_rows_with_candidates'])}（有候选的句子）",
        f"- 标签的分布：{lab}；并列的句子 {m['best_with_ties']['n_rows_with_ties']}",
        f"- 各编号作为最优的次数（并列都算）：{m['best_with_ties']['counts']}", "",
        "## 文件", "",
        "- w2c_data/manifest.json：总体、去重（E3 的剔除清单、W2 的评测图）、抽样各步的条数与 sha256，选中的 id（处理顺序）与它们的图；",
        "- propose/：E1 refexp_box 的运行（E1 格式）；w2c_data/*.jsonl：数据；config.yaml：运行身份；metrics.json：由数据算出；"
        "logs/：每句候选生成的墙钟与每次调用的记录。", ""])


__all__ = [
    "CANDIDATE_KEYS",
    "DATA_DIR",
    "DEFAULT_N",
    "LABEL_RULE",
    "MANIFEST",
    "PROPOSE_DIR",
    "PURPOSE",
    "ROW_KEYS",
    "SALT",
    "SHARD_SIZE",
    "TINY_N",
    "TINY_SHARD_SIZE",
    "W2CDataError",
    "W2CPaths",
    "W2C_FIXTURE_ROOT",
    "build_manifest",
    "check_manifest",
    "check_row",
    "data_identity",
    "data_metrics",
    "data_row",
    "ensure_manifest",
    "eval_images",
    "eval_split_names",
    "finalize",
    "load_manifest",
    "manifest_path",
    "load_train_pool",
    "matched_proposals",
    "prepare",
    "propose_complete",
    "propose_leaf",
    "propose_selection",
    "read_rows",
    "required_eval_sets",
    "run_candidates",
    "run_propose",
    "score_candidates",
    "select_sentences",
    "shard_name",
    "shard_paths",
    "shard_plan",
    "w2c_job",
]
