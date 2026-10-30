"""统一主表的两只"手"（DEV-0074；预注册 U-1、U-2；D154）：9B 零样本的 E1 结构强制多边形、9B 的 E3b 像素 token 网格，在 W2 test 的
1,500 句子集（configs/w2_llmseg_subsets.json，与 LLMSeg 的 RUN-0078 同一份）上运行，按 W2 的口径计分，写 W2 的叶目录。

**推理照搬，一行不改**（scripts/run_u_hands.py 调用）：
- e1_poly_structured：src/e1/posthoc_runner.run_forced_poly_task 原样（变体 poly_structured = 至少 3 个点，关思考、贪心、E1 冻结的
  图像口径与批大小，D72-3；配置 configs/e1_posthoc.json 的 format_forced_poly），样本取子集（sample_subset = "w2_llmseg_subsets:<划分>"，
  同 W1 的 baseline_poly_structured 用的 DEV-0051 的放宽），输出写进 <叶目录>/raw/（E1 事后读数的格式）。
- e3b_grid：src/e3/evaluate.run_eval_task 原样（单轮 k_values = [1]、结构强制解码 decode = structured、guided 上采样，D83-3；同 E3
  表的 refseg 读法与 RUN-0056），检查点先过评测前的合并核对（同 scripts/eval_e3b.py，判据 v1），输出写进
  <叶目录>/raw/eval/<变体>/<模型>/seed<k>/refseg__structured/（E3b 评测的格式）。

**W2 的计分与叶目录**（本模块；<out>/<模型>/refseg/<设置名>/<划分>/，设置名 u_e1_poly_structured、u_e3b_grid）：
- 逐句的 mask：E1 = 解析出的多边形经 src/executors/raster.polygons_to_mask（E1 计分的同一个光栅化）；E3b = 网格经
  src/e3/eval_scoring.seg_mask（主上采样 guided、阈值取 configs/e3b_train.json 的 eval.upsample）。解析失败（E1 解析不出多边形、E3b 解析
  不出网格）没有 mask，按空 mask 计（metrics.seg 的约定：交 0、并 |GT|、IoU 0），失败原因进 metrics.json 的 failures。
- 逐句计分 = W2 的同一组函数：GT = src/e1/scoring.gt_mask，mask_inter_union + sample_iou（src/metrics/seg）；与 raw 里原运行器自己的
  逐句交并核对，必须逐个相同（入口断言：不同就报错，不写结果）。标题 = src/w1/scoring.headline（cIoU、gIoU 与 bootstrap 95% CI，同
  W2 的 aggregate_w2 与 LLMSeg）。
- 文件：predictions.jsonl（每句一行：id、w2_split、指代句、失败、mask 的 RLE、score 的 inter / union / iou、调用数与 token；不含墙钟，
  逐字节可复现）、metrics.json（只由 predictions 算；n、complete、headline、failures、空 mask 数、每句的计算量、raw 的出处与读数）、
  compute.json（raw 那次运行的 compute.json 原样：单轮，每句一次调用）、config.yaml（运行身份，含 w2.test_unlock = 批准编号 D154）、
  meta.json、README.md、logs/derive_<k>.json。scripts/mgr/w2_test_table.preds 直接读。
- 续跑：W2 的身份必须相同（同 W2）；raw 按原运行器的续跑规则整批续上；raw 跑完之后整份重新导出（逐字节相同）。
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.data.records import Sample
from src.e1.posthoc_runner import _prepare_leaf
from src.e1.runner import COMPUTE, CONFIG, METRICS, PREDICTIONS, README, _jsonl_line, read_jsonl_tolerant
from src.e1.scoring import distribution, gt_mask
from src.executors.raster import mask_to_rle, polygons_to_mask
from src.metrics.seg import mask_inter_union, sample_iou
from src.w1.runner import model_record
from src.w1.scoring import headline
from src.w2.runner import W2RunSpec, w2_leaf, w2_record

EXPERIMENT = "U"
FRAMEWORK = "src/unified/hands.py（DEV-0074；预注册 U-2，D154）"
TASK = "refseg"
SYSTEM_E1 = "e1_poly_structured"
SYSTEM_E3B = "e3b_grid"
SYSTEMS: tuple[str, ...] = (SYSTEM_E1, SYSTEM_E3B)
SETTINGS: Mapping[str, str] = {SYSTEM_E1: "u_e1_poly_structured", SYSTEM_E3B: "u_e3b_grid"}  # W2 叶目录的设置名（任务书）
E1_TASK = "refexp_poly"
E1_VARIANT = "poly_structured"  # configs/e1_posthoc.json 的 format_forced_poly.variants：至少 3 个点（D72-3；tests 钉住 min_points）
E3B_TASK = "refseg"
E3B_K_VALUES: tuple[int, ...] = (1,)  # 单轮（E3 表的主读数；D72-4 (c) 的 --k-values 1）
E3B_DECODE = "structured"  # 结构强制解码（D83）
RAW = "raw"  # 原运行器的输出放在 <叶目录>/raw/
SUBSET_PREFIX = "w2_llmseg_subsets:"  # E1 事后读数的 sample_subset 名（不是思考子集；DEV-0051 的放宽要求另起名字）

# test 子集的放行（D154）：依据 DECISIONS.md D154-2(2)（项目负责人 2026-10-30 指示，原文 docs/lead_answers/2026-10-30-lang-table-and-freeze.md）：
#   统一主表新跑三项（9B 零样本的 E1 结构强制多边形、9B 的 E3b 像素 token 网格、LISA-7B）在 test 子集上运行的批准编号是 D154，写进运行器的
#   放行检查，同 D147（DEV-0072）。预注册 U-2"新跑三项的共同要求"。别的编号一律不认。
TEST_APPROVAL = "D154"
TEST_UNLOCK_LOG = "test_unlock.jsonl"
UNLOCK_RULE = ("D154-2(2)：统一主表的新跑基线在 W2 test 的 5 × 300 句子集（configs/w2_llmseg_subsets.json）上运行；预注册 U-2")


class TestApprovalError(ValueError):
    """test 子集的放行不合格（D154）。"""


def check_approval(approval: str | None) -> dict[str, Any]:
    """--test-approval 必须恰是 D154；返回进运行身份的放行记录（w2.test_unlock）。不合格抛 TestApprovalError。"""
    if approval is None:
        raise TestApprovalError(f"这份子集全是 W2 的 test 句子：要给 --test-approval {TEST_APPROVAL}（负责人的批准编号）")
    if approval != TEST_APPROVAL:
        raise TestApprovalError(f"--test-approval 必须恰是 {TEST_APPROVAL}（统一主表新跑基线上 test 的裁定编号），收到 {approval!r}")
    return {"approval": TEST_APPROVAL, "rule": UNLOCK_RULE}


def unlock_line(ident: Mapping[str, Any], *, system: str, split: str, model: str, argv: Sequence[str],
                checkpoint_manifest_sha256: str | None = None) -> dict[str, Any]:
    """<out>/<模型>/test_unlock.jsonl 的一行（每个划分一行；时间与 argv 只在这里，不进运行身份）。"""
    return {**dict(ident), "system": system, "setting": SETTINGS[system], "task": TASK, "split": split, "model": model,
            "checkpoint_manifest_sha256": checkpoint_manifest_sha256,
            "unlocked_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "argv": list(argv)}


def append_unlock(out: str | Path, model: str, lines: Sequence[Mapping[str, Any]]) -> Path:
    log = Path(out) / model / TEST_UNLOCK_LOG
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a", encoding="utf-8") as f:
        for rec in lines:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return log


# ---------------------------------------------------------------------------
# 路径与运行身份
# ---------------------------------------------------------------------------


def leaf_of(out: str | Path, model_key: str, system: str, split: str) -> Path:
    return w2_leaf(out, model_key, TASK, SETTINGS[system], split)


def subset_name(split: str) -> str:
    return f"{SUBSET_PREFIX}{split}"


def subset_record(subsets_doc: Mapping[str, Any], split: str, *, tiny: bool) -> dict[str, Any]:
    """运行身份 w2.subset（同 LLMSeg 的 src/llmseg/runner.identity）。"""
    sub = subsets_doc["splits"][split]
    return {"config": "tiny（同一规则现算）" if tiny else "configs/w2_llmseg_subsets.json", "rule": subsets_doc["rule"],
            "salt": subsets_doc["salt"], "n": sub["n"], "sha256": sub["sha256"]}


def w2_block(spec: W2RunSpec, splits_doc: Mapping[str, Any], subsets_doc: Mapping[str, Any], samples: Sequence[Sample]) -> dict[str, Any]:
    """W2 一节 = src/w2/runner.w2_record（含 test_unlock）加子集的记录。raw 的 selection_stats / stats 也记这一节。"""
    return {**w2_record(spec, splits_doc, samples), "subset": subset_record(subsets_doc, spec.split, tiny=bool(splits_doc.get("tiny")))}


def scoring_rule(system: str, *, upsample: str | None = None, threshold: float | None = None) -> dict[str, Any]:
    mask = ("解析出的多边形经 src/executors/raster.polygons_to_mask（E1 计分的同一个光栅化）" if system == SYSTEM_E1 else
            f"网格经 src/e3/eval_scoring.seg_mask(method = {upsample}, threshold = {threshold})（E3b 评测的主上采样）")
    return {"mask": mask,
            "per_sentence": "W2 的计分：GT = src/e1/scoring.gt_mask；src/metrics/seg.mask_inter_union + sample_iou；解析失败按空 mask"
                            "（交 0、并 |GT|、IoU 0）；与 raw 的逐句交并逐个核对",
            "headline": "src/w1/scoring.headline（cIoU = Σinter / Σunion、gIoU = 逐句 IoU 均值，bootstrap 95% CI）"}


def identity(spec: W2RunSpec, *, system: str, w2: Mapping[str, Any], system_block: Mapping[str, Any], raw_dir: str,
             scoring: Mapping[str, Any]) -> dict[str, Any]:
    """W2 叶目录的 config.yaml（运行身份；不含时间、argv、机器相关的路径）。"""
    return {"experiment": EXPERIMENT, "framework": FRAMEWORK, "system": system, "w2": json.loads(json.dumps(dict(w2), ensure_ascii=False)),
            "model": model_record(spec), "image_policy": dict(spec.image_policy), system: json.loads(json.dumps(dict(system_block),
                                                                                                                ensure_ascii=False)),
            "raw": {"dir": raw_dir, "rule": "原运行器的输出原样（格式见它自己的 README）；本目录由它导出"}, "scoring": dict(scoring)}


def prepare(leaf: Path, cfg_doc: Mapping[str, Any], files: Sequence[str | Path], *, resume: bool) -> None:
    """同 W2 的续跑规则（src/e1/posthoc_runner._prepare_leaf）：新目录写 config.yaml 与 meta.json；已有内容要 --resume 且身份相同。"""
    _prepare_leaf(leaf, cfg_doc, files, resume=resume)


# ---------------------------------------------------------------------------
# 导出：raw → W2 的记录
# ---------------------------------------------------------------------------


def _score(mask: np.ndarray | None, gt: np.ndarray) -> dict[str, Any]:
    st = mask_inter_union(mask, gt)
    return {"inter": st.inter, "union": st.union, "iou": sample_iou(*st)}


def _same_score(a: Mapping[str, Any], b: Mapping[str, Any], where: str) -> None:
    if (int(a["inter"]), int(a["union"])) != (int(b["inter"]), int(b["union"])) or float(a["iou"]) != float(b["iou"]):
        raise RuntimeError(f"{where}：W2 计分 {dict(a)} 与原运行器的 {dict(b)} 不同（mask 或 GT 取错了）")


def _tokens(raw: Mapping[str, Any]) -> dict[str, int]:
    return {"prompt": int(raw["n_prompt_tokens"]), "generated": int(raw["n_generated_tokens"]), "inserted": 0,
            "visual": int(raw["n_visual_tokens"]), "image_pixels": int(raw["n_image_pixels"] or 0)}


def _base_record(spec: W2RunSpec, system: str, s: Sample, gt: np.ndarray, failure: str | None, mask: np.ndarray | None,
                 answer_source: str, raw: Mapping[str, Any]) -> dict[str, Any]:
    h, w = gt.shape
    if failure is None and mask is None:
        raise RuntimeError(f"{s.id}：没有失败却没有 mask")
    if mask is not None and np.asarray(mask).shape != (h, w):
        raise RuntimeError(f"{s.id}：mask 形状 {np.asarray(mask).shape} 与 GT {(h, w)} 不同")
    m = None if failure is not None else np.asarray(mask, dtype=bool)
    return {"id": s.id, "w2_split": spec.split, "source": s.source, "split": s.split, "setting": spec.setting, "system": system,
            "task": TASK, "instruction": s.fields["expression"], "image_size": [int(w), int(h)], "failure": failure,
            "answer": None if m is None else {"kind": "mask", "source": answer_source,
                                              "mask": {"rle": mask_to_rle(m), "area": int(np.count_nonzero(m))}},
            "score": _score(m, gt), "n_model_calls": 1, "n_rounds": 1, "n_tool_calls": 0, "tokens": _tokens(raw)}


def e1_records(spec: W2RunSpec, samples: Sequence[Sample], raw_recs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """E1 结构强制多边形的 raw 记录（src/e1/posthoc_runner.run_forced_poly_task 的 predictions.jsonl）→ W2 的记录。"""
    if [r["id"] for r in raw_recs] != [s.id for s in samples]:
        raise RuntimeError("raw 的 predictions.jsonl 与样本的 id 不同（没跑完或顺序不对）")
    out = []
    for s, r in zip(samples, raw_recs):
        gt = gt_mask(s)
        h, w = gt.shape
        if list(r["image_size"]) != [w, h]:
            raise RuntimeError(f"{s.id}：raw 记的原图尺寸 {r['image_size']} 与 GT {(w, h)} 不符")
        failure = r["failure"]
        mask = None if failure is not None else polygons_to_mask(r["parsed"]["polygons"], h, w)
        rec = _base_record(spec, SYSTEM_E1, s, gt, failure, mask, "e1_poly_structured:polygons_to_mask", r)
        _same_score(rec["score"], r["score"], f"{s.id}（E1）")
        fp = r["forced_poly"]
        rec["raw"] = {"n_points": fp["n_points"], "near_rect": fp["near_rect"], "area_ratio": fp["area_ratio"],
                      "n_polygons": None if r["parsed"] is None else len(r["parsed"]["polygons"]), "stopped": r["stopped"],
                      "truncated": r["truncated"]}
        out.append(rec)
    return out


def e3b_records(spec: W2RunSpec, samples: Sequence[Sample], raw_recs: Sequence[Mapping[str, Any]],
                raw_scores: Sequence[Mapping[str, Any]], *, n: int, upsample: str, threshold: float) -> list[dict[str, Any]]:
    """E3b 单轮的 raw 记录（src/e3/evaluate 的 predictions.jsonl = 第 1 轮，与 scores.jsonl）→ W2 的记录。mask 用同一个 seg_mask 重算，
    与 scores.jsonl 里单轮、主上采样的交并核对。"""
    from src.e3 import eval_scoring as ES

    ids = [s.id for s in samples]
    if [r["id"] for r in raw_recs] != ids or [r["id"] for r in raw_scores] != ids:
        raise RuntimeError("raw 的 predictions.jsonl / scores.jsonl 与样本的 id 不同（没跑完或顺序不对）")
    out = []
    for s, r, sc in zip(samples, raw_recs, raw_scores):
        if not r.get("called") or r.get("stage") != "round1":
            raise RuntimeError(f"{s.id}：单轮的记录应是调用过的第 1 轮，收到 stage {r.get('stage')!r}、called {r.get('called')!r}")
        gt = gt_mask(s)
        code = r["grid"]
        if code is None:
            failure, mask = (r["failure"] or "no_grid"), None
        else:
            if r["failure"] is not None:
                raise RuntimeError(f"{s.id}：有网格却记了失败 {r['failure']!r}")
            failure = None
            mask = ES.seg_mask(ES.decode_grid(TASK, code, n), ES.load_image(s.images[0]), upsample, threshold)
        rec = _base_record(spec, SYSTEM_E3B, s, gt, failure, mask, f"e3b_grid:seg_mask({upsample})", r)
        p = sc["procs"]["single"]
        ref = p[upsample]
        _same_score(rec["score"], {"inter": ref["inter"], "union": ref["union"],
                                   "iou": sample_iou(int(ref["inter"]), int(ref["union"]), not p["valid"])}, f"{s.id}（E3b）")
        if float(ref["iou"]) != rec["score"]["iou"]:
            raise RuntimeError(f"{s.id}：IoU {rec['score']['iou']} 与 scores.jsonl 的 {ref['iou']} 不同")
        rec["raw"] = {"grid": code, "stopped": r.get("stopped"), "biou": ref.get("biou"), "cell_iou": (p.get("grid") or {}).get("cell_iou")}
        out.append(rec)
    return out


# ---------------------------------------------------------------------------
# 汇总与写出
# ---------------------------------------------------------------------------


def _counts(values) -> dict[str, int]:
    c = Counter("ok" if v is None else str(v) for v in values)
    return {k: c[k] for k in sorted(c)}


def aggregate(recs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """predictions.jsonl 的记录 → metrics.json 的主体（只读记录；重导出逐字节相同）。"""
    n = len(recs)
    m: dict[str, Any] = {"n": n, "headline": headline(TASK, recs) if recs else []}
    m["failures"] = _counts(r["failure"] for r in recs)
    m["n_failed"] = sum(1 for r in recs if r["failure"] is not None)
    m["empty_mask"] = {"n": sum(1 for r in recs if r["failure"] is None and r["answer"]["mask"]["area"] == 0),
                       "rule": "解析成功但 mask 面积为 0（多边形退化、网格全 0 或上采样后不过阈值）的句数；不算失败，计分同空 mask"}
    m["mask_area"] = distribution(r["answer"]["mask"]["area"] for r in recs if r["answer"] is not None)
    m["compute_per_sentence"] = {
        "n_model_calls": float(np.mean([r["n_model_calls"] for r in recs])) if recs else None,
        **{f"{k}_tokens": (float(np.mean([r["tokens"][k] for r in recs])) if recs else None)
           for k in ("prompt", "generated", "visual")},
        "rule": "每句均值（由 predictions 算）；墙钟见 compute.json（raw 那次运行的记账原样）"}
    for k in ("prompt", "generated", "visual"):
        m.setdefault("tokens", {})[k] = distribution(r["tokens"][k] for r in recs)
    return m


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _fmt(v: Any) -> str:
    return "null" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))


def readme(spec: W2RunSpec, system: str, cfg_doc: Mapping[str, Any], m: Mapping[str, Any]) -> str:
    what = {SYSTEM_E1: "E1 结构强制多边形（9B 零样本；configs/e1_posthoc.json 的 format_forced_poly，变体 poly_structured：至少 3 个点；"
                       "关思考、贪心、E1 冻结的图像口径与批大小；D72-3）",
            SYSTEM_E3B: "E3b 像素 token 网格（检查点的单轮、结构强制解码、guided 上采样；D83-3；同 E3 表的 refseg 读法）"}[system]
    lines = [f"# 统一主表的\"手\" · {spec.model_key} · refseg · {spec.setting} · {spec.split}", "",
             f"- 系统：{what}",
             f"- 句子：W2 test 子集 {cfg_doc['w2']['subset']['n']} 句（{cfg_doc['w2']['subset']['config']}，sha256 "
             f"{str(cfg_doc['w2']['subset']['sha256'])[:16]}…）；本次 n = {m['n']}（limit = {cfg_doc['w2']['limit']}）",
             f"- 放行：{cfg_doc['w2']['test_unlock']['approval']}（{cfg_doc['w2']['test_unlock']['rule']}）",
             f"- 原运行器的输出：{cfg_doc['raw']['dir']}/（{m['raw']['format']}）"]
    if system == SYSTEM_E3B:
        ck = cfg_doc[system]["checkpoint"]
        lines.append(f"- 检查点：manifest sha256 {ck.get('manifest_sha256')}（变体 {cfg_doc[system]['variant']}，seed {cfg_doc[system]['seed']}）")
    lines += ["", "## 标题指标（W2 的计分）", ""]
    for h in m["headline"]:
        ci = h.get("ci")
        lines.append(f"- {h['name']}：{_fmt(h['value'])}，95% CI {'—' if not ci else '[' + _fmt(ci['low']) + ', ' + _fmt(ci['high']) + ']'}"
                     f"，n = {h['n']}")
    lines += ["", f"- 失败（按空 mask 计）：{m['failures']}；解析成功但 mask 为空：{m['empty_mask']['n']}",
              f"- 每句计算量：{ {k: v for k, v in m['compute_per_sentence'].items() if k != 'rule'} }（compute.json 是 raw 那次运行的记账）",
              "", "## 文件", "",
              "- predictions.jsonl（逐句：失败、mask 的 RLE、score、调用与 token）；metrics.json；compute.json；config.yaml；meta.json；"
              "logs/derive_<k>.json；raw/（原运行器的输出原样）。", ""]
    return "\n".join(lines)


def write_leaf(leaf: Path, spec: W2RunSpec, system: str, cfg_doc: Mapping[str, Any], recs: Sequence[Mapping[str, Any]], *,
               n_expected: int, raw_leaf: Path, raw_format: str, raw_headline: Any, extra: Mapping[str, Any] | None = None,
               argv: Sequence[str] = (), run_info: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """写 predictions.jsonl、metrics.json、compute.json（raw 原样）、README.md 与 logs/derive_<k>.json，返回 metrics。"""
    (leaf / PREDICTIONS).write_text("".join(_jsonl_line(r) for r in recs), encoding="utf-8")
    back = read_jsonl_tolerant(leaf / PREDICTIONS, repair=False)
    body = aggregate(back)
    raw = {"leaf": str(raw_leaf.relative_to(leaf)), "format": raw_format, "predictions_sha256": _sha(raw_leaf / PREDICTIONS),
           "config_sha256": _sha(raw_leaf / CONFIG), "headline": json.loads(json.dumps(raw_headline, ensure_ascii=False))}
    metrics = {"experiment": EXPERIMENT, "system": system, "w2": cfg_doc["w2"], "model": spec.model_key, "revision": spec.revision,
               "setting": spec.setting, **body, **({system: dict(extra)} if extra else {}), "raw": raw,
               "complete": len(back) == n_expected, "n_expected": n_expected}
    (leaf / METRICS).write_text(json.dumps(metrics, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (leaf / COMPUTE).write_bytes((raw_leaf / COMPUTE).read_bytes())  # 计算量就是 raw 那次运行（单轮，每句一次调用）
    (leaf / README).write_text(readme(spec, system, cfg_doc, metrics), encoding="utf-8")
    logs = leaf / "logs"
    logs.mkdir(exist_ok=True)
    k = len(list(logs.glob("derive_*.json"))) + 1
    (logs / f"derive_{k:03d}.json").write_text(json.dumps({"argv": list(argv), **dict(run_info or {}),
                                                           "written_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")},
                                                          indent=1, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    return metrics


__all__ = [
    "E1_TASK",
    "E1_VARIANT",
    "E3B_DECODE",
    "E3B_K_VALUES",
    "EXPERIMENT",
    "RAW",
    "SETTINGS",
    "SYSTEMS",
    "SYSTEM_E1",
    "SYSTEM_E3B",
    "TEST_APPROVAL",
    "TEST_UNLOCK_LOG",
    "TestApprovalError",
    "aggregate",
    "append_unlock",
    "check_approval",
    "e1_records",
    "e3b_records",
    "identity",
    "leaf_of",
    "prepare",
    "subset_name",
    "unlock_line",
    "w2_block",
    "write_leaf",
]
