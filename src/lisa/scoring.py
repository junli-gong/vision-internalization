"""LISA-7B 外部对照的计分（DEV-0075；D154）：在我们的环境里读 LISA 环境写的原始输出，按 W2 的口径计分，写 W2 的叶目录。

硬约束第 1 条：这里是 LISA 输出唯一的读者（加上出表的 scripts/mgr/）。计分结果只进主表，绝不进入训练数据、标签、奖励或任何挑选。

- 句子：重新按 src/lisa/data.split_samples（= LLMSeg 的取样）取每个划分的样本，必须与清单逐 id、逐顺序相同。
- 计分（与 RUN-0078 的 src/llmseg/runner._score 同一组函数）：GT = src/e1/scoring.gt_mask（W2 的 GT），
  st = src/metrics/seg.mask_inter_union，iou = sample_iou(*st)；标题指标 = src/w1/scoring.headline（cIoU、gIoU 与 bootstrap 95% CI）。
  没有 [SEG]、没有 mask、推理异常：按空 mask 计（mask_inter_union(None, GT) = (0, |GT|)，同 W1/W2 的失败口径），failure 记原因，
  metrics.json 的 failures 计数。mask 全空但有输出的不算失败（照实记 n_empty_mask）。
- 叶目录 <out>/lisa-7b-v1/refseg/lisa_7b/<划分>/：predictions.jsonl（逐句，不含墙钟：mask 的 RLE、score.inter/union/iou、
  failure、LISA 的 [SEG] 个数与前向次数、token 数）、metrics.json（只由 predictions 算）、compute.json、config.yaml、meta.json、
  README.md、logs/timing.jsonl。scripts/mgr/w2_test_table.py 的 preds 能直接读。原始输出写到一半时照样计分已写出的前缀，
  complete 为假（w2_test_table 不读没写完的目录）。
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.accounting import Compute, merge, write_compute_json
from src.constants import BOOTSTRAP_SEED
from src.data.records import Sample
from src.e1.runner import ids_digest
from src.e1.scoring import distribution, gt_mask
from src.executors.raster import mask_to_rle, rle_to_mask
from src.lisa import common as C
from src.lisa import data as D
from src.metrics.seg import mask_inter_union, sample_iou
from src.runlog import write_run_metadata
from src.w1.scoring import headline
from src.w2.data import splits_sha256

FRAMEWORK = "src/lisa（DEV-0075；D154）"
PREDICTIONS, METRICS, COMPUTE, CONFIG, META, README, TIMING = (
    "predictions.jsonl", "metrics.json", "compute.json", "config.yaml", "meta.json", "README.md", "logs/timing.jsonl")
LEAF_FILES = (PREDICTIONS, METRICS, COMPUTE, CONFIG, META, README, TIMING)
ISOLATION_NOTE = ("硬约束第 1 条：LISA 把 SAM 接在 LLM 旁边（外接）。它只作对照单独运行，输出只用来计分与出表，绝不进入我们的训练数据、"
                  "标签、奖励或任何挑选（D154；简报 §8.8；预注册 U-2）。")
SCORING = {"gt": "src/e1/scoring.gt_mask（W2 的 GT）",
           "iou": "src/metrics/seg.mask_inter_union + sample_iou（同 RUN-0078 的 src/llmseg/runner._score）",
           "headline": "src/w1/scoring.headline（cIoU = Σinter / Σunion，gIoU = 逐句 IoU 均值，bootstrap 95% CI）",
           "failure": "没有 [SEG]、没有 mask、推理异常：按空 mask 计（mask_inter_union(None, GT) = (0, |GT|)），failure 记原因"}


class LisaScoreError(C.LisaError):
    pass


def _score(mask: np.ndarray | None, gt: np.ndarray) -> dict[str, Any]:
    st = mask_inter_union(mask, gt)
    return {"inter": st.inter, "union": st.union, "iou": sample_iou(*st)}


def read_run_config(out: str | Path) -> tuple[dict[str, Any], str]:
    p = C.raw_dir(out) / C.RUN_CONFIG
    if not p.is_file():
        raise LisaScoreError(f"{p} 不存在：LISA 还没有在 {out} 上运行（scripts/run_lisa.py）")
    return json.loads(p.read_text(encoding="utf-8")), C.sha256_file(p)


def check_run(out: str | Path, ident: Mapping[str, Any], manifest_sha: Mapping[str, str], *, allow_stand_in: bool) -> dict[str, Any]:
    """原始输出的身份：批准编号 D154、清单的 sha256、官方代码与现在的 third_party/lisa 相同；替身或非默认权重清单要 allow_stand_in。"""
    if ident.get("approval") != C.TEST_APPROVAL:
        raise LisaScoreError(f"原始输出的批准编号是 {ident.get('approval')!r}，不是 {C.TEST_APPROVAL}")
    steps = {r.get("step") for r in C.read_unlock(out) if r.get("approval") == C.TEST_APPROVAL}
    if not {"export", "run"} <= steps:
        raise LisaScoreError(f"{C.system_dir(out) / C.TEST_UNLOCK_LOG} 里没有导出与运行两步的放行记录（有 {sorted(steps)}）")
    if dict(ident.get("manifest", {}).get("files_sha256", {})) != dict(manifest_sha):
        raise LisaScoreError("原始输出记的清单 sha256 与现在的清单不同")
    code = C.check_vendor()
    if C.normalized(code) != ident.get("code"):
        raise LisaScoreError("原始输出记的官方代码身份与现在的 third_party/lisa 不同")
    stand_in = (ident.get("runner") or {}).get("kind") != "lisa" or not ((ident.get("weights") or {}).get("manifest") or {}).get("default")
    if stand_in and not allow_stand_in:
        raise LisaScoreError("原始输出来自替身运行器或非默认的权重清单（不是 LISA-7B-v1）：只有 --allow-stand-in 才计分")
    return {"code": code, "stand_in": stand_in}


def item_record(sample: Sample, split: str, row: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """一句的 predictions.jsonl 行（不含墙钟）与 timing 行。"""
    gt = gt_mask(sample)
    h, w = gt.shape
    failure = row.get("failure")
    mask = None
    if failure is None:
        rle = row.get("mask_rle")
        if rle is None:
            raise LisaScoreError(f"{sample.id}：没有失败却没有 mask_rle")
        mask = rle_to_mask(rle)
        if mask.shape != (h, w):
            raise LisaScoreError(f"{sample.id}：mask 尺寸 {mask.shape} 与 GT {(h, w)} 不同")
    shown = mask if mask is not None else np.zeros((h, w), dtype=bool)
    forwarded = "forwards" in row
    n_text = int(row["n_input_ids"]) - 1 if forwarded else 0  # 图像占位（IMAGE_TOKEN_INDEX）换成 256 个视觉 token
    n_vis = int(row["n_visual_tokens"]) if forwarded else 0
    rec = {"id": sample.id, "w2_split": split, "source": sample.source, "split": sample.split, "setting": C.SETTING, "task": C.TASK,
           "instruction": sample.fields["expression"], "image_size": [int(w), int(h)], "status": row["status"], "failure": failure,
           "answer": {"kind": "mask", "mask": {"source": "lisa" if mask is not None else "empty", "rle": mask_to_rle(shown),
                                               "area": int(np.count_nonzero(shown))}},
           "score": _score(mask, gt),
           "lisa": {k: row.get(k) for k in ("n_seg", "n_masks", "seg_source", "generated_text", "n_input_ids", "n_visual_tokens",
                                            "forwards", "image_hw_decoded", "image_hw_mismatch", "exif_orientation", "sam_resize",
                                            "error")},
           "n_model_calls": int(forwarded), "n_rounds": int(forwarded), "n_tool_calls": 0,
           "tokens": {"prompt": n_text, "generated": 0, "inserted": 0, "visual": n_vis, "image_pixels": C.CLIP_PIXELS if n_vis else 0}}
    timing = {"id": sample.id, "wall_s": float(row.get("wall_s", 0.0)), "forward_s": float(row.get("forward_s", 0.0))}
    return rec, timing


def aggregate(recs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    fails = Counter("ok" if r["failure"] is None else str(r["failure"]) for r in recs)
    fw = [r["lisa"]["forwards"] for r in recs if r["lisa"]["forwards"]]
    return {
        "n": len(recs), "headline": headline(C.TASK, recs) if recs else [],
        "failures": {k: fails[k] for k in sorted(fails)}, "n_failed": sum(1 for r in recs if r["failure"] is not None),
        "n_empty_mask": sum(1 for r in recs if r["failure"] is None and r["answer"]["mask"]["area"] == 0),
        "n_image_hw_mismatch": sum(1 for r in recs if r["lisa"]["image_hw_mismatch"]),
        "n_exif_rotated": sum(1 for r in recs if r["lisa"]["exif_orientation"] not in (None, 1)),
        "n_seg": distribution(r["lisa"]["n_seg"] for r in recs),
        "forwards_per_item": {k: (float(np.mean([f[k] for f in fw])) if fw else None) for k in ("llm", "sam_image_encoder", "sam_mask_decoder")},
        "forwards_total": {k: int(sum(f[k] for f in fw)) for k in ("llm", "sam_image_encoder", "sam_mask_decoder")},
        "tokens": {k: distribution(r["tokens"][k] for r in recs) for k in ("prompt", "generated", "visual")},
        "rule": {"failures": SCORING["failure"], "n_empty_mask": "有输出、mask 全空（不算失败）",
                 "n_exif_rotated": "图像带 EXIF 方向（≠ 1）：官方 cv2.imread 按它旋转，我们的 GT 是原始像素方向；照官方口径不处理，只计数",
                 "forwards": "每句：LLM 一次前向（teacher forcing，不生成）、SAM 图像编码一次、SAM 解码 = [SEG] 个数"},
    }


def compute_of(recs: Sequence[Mapping[str, Any]], timing: Mapping[str, Mapping[str, Any]]) -> Compute:
    out = []
    for r in recs:
        t = timing.get(r["id"], {})
        tk = r["tokens"]
        out.append(Compute(n_calls=r["n_model_calls"], n_rounds=r["n_rounds"], generated_tokens=0, prompt_tokens=int(tk["prompt"]),
                           visual_tokens=int(tk["visual"]), image_pixels=int(tk["image_pixels"]),
                           call_time_s=float(t.get("forward_s", 0.0))).as_sample(wall_time_s=float(t.get("wall_s", 0.0))))
    return merge(out)


def _fmt(v: Any) -> str:
    return "null" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))


def _readme(split: str, cfg: Mapping[str, Any], m: Mapping[str, Any], stand_in: bool) -> str:
    lines = [f"# W2 外部对照 LISA-7B · {C.SYSTEM} · {C.SETTING} · {split}", ""]
    if stand_in:
        lines += ["**替身运行器的输出（开发机彩排），不是 LISA-7B-v1 的读数。**", ""]
    lines += [f"- {ISOLATION_NOTE}",
              f"- 官方代码：{C.UPSTREAM_URL}@{C.UPSTREAM_COMMIT[:12]}（third_party/lisa，未改动；PROVENANCE sha256 "
              f"{cfg['lisa']['code']['provenance_sha256'][:16]}…）；论文 {C.PAPER}",
              f"- 权重：{cfg['model']['repo']}@{str(cfg['model']['revision'])[:12]}；视觉塔 {cfg['model']['clip']}",
              f"- 协议：{C.PROTOCOL['what']}；精度 {C.PROTOCOL['precision']}",
              f"- 句子：{cfg['w2']['subset']['n']} 句（{cfg['w2']['subset']['config']}，sha256 {cfg['w2']['subset']['sha256'][:16]}…），"
              f"与 LLMSeg 相同；批准 {C.TEST_APPROVAL}", "", "## 读数", ""]
    for h in m["headline"]:
        ci = h.get("ci")
        lines.append(f"- {h['name']}：{_fmt(h['value'])}，95% CI {'—' if not ci else '[' + _fmt(ci['low']) + ', ' + _fmt(ci['high']) + ']'}"
                     f"，n = {h['n']}")
    lines += ["", f"- 失败 {m['failures']}（按空 mask 计）；mask 全空 {m['n_empty_mask']}；读图尺寸与标注不符 {m['n_image_hw_mismatch']}；"
              f"带 EXIF 方向 {m['n_exif_rotated']}",
              f"- 每句前向 {m['forwards_per_item']}；complete = {m['complete']}（{m['n']} / {m['n_expected']}）", "",
              "## 文件", "", "- predictions.jsonl（逐句：mask 的 RLE、score、failure、LISA 的 [SEG] 与前向次数）；metrics.json；compute.json；"
              "config.yaml；meta.json；logs/timing.jsonl。原始输出在 ../../../raw/。", ""]
    return "\n".join(lines)


def _clear_leaf(leaf: Path, rescore: bool) -> None:
    if leaf.exists() and any(leaf.iterdir()):
        if not rescore:
            raise LisaScoreError(f"{leaf} 已存在且非空；重新计分请加 --rescore")
        extra = sorted(p.relative_to(leaf).as_posix() for p in leaf.rglob("*") if p.is_file() and p.relative_to(leaf).as_posix() not in LEAF_FILES)
        if extra:
            raise LisaScoreError(f"{leaf} 里有不是计分写的文件 {extra[:5]}，不动它")
        for f in LEAF_FILES:
            if (leaf / f).exists():
                (leaf / f).unlink()
    (leaf / "logs").mkdir(parents=True, exist_ok=True)


def score_split(out: str | Path, split: str, samples: Sequence[Sample], rows: Sequence[Mapping[str, Any]], *, ident: Mapping[str, Any],
                run_config_sha: str, checked: Mapping[str, Any], meta: Mapping[str, Any], manifest_sha: Mapping[str, str],
                splits_doc: Mapping[str, Any], subsets_doc: Mapping[str, Any], data_files: Sequence[Path], rescore: bool) -> dict[str, Any]:
    ids = [s.id for s in samples]
    if ids != [r["id"] for r in rows]:
        raise LisaScoreError(f"{split}：清单的 id 与按 W2（LLMSeg）的取样重算的不同")
    raw_p = C.raw_rows_path(out, split)
    raw = C.read_raw_prefix(raw_p, ids, repair=False)
    recs, timing = [], []
    for s, row in zip(samples, raw):
        r, t = item_record(s, split, row)
        recs.append(r)
        timing.append(t)
    e = splits_doc["splits"][split]
    sub = subsets_doc["splits"][split]
    tiny = bool(meta.get("tiny"))
    wts = ident.get("weights") or {}
    wdoc = C.load_weights_manifest()
    cfg = {
        "experiment": C.EXPERIMENT, "framework": FRAMEWORK, "isolation": ISOLATION_NOTE,
        "w2": {"task": C.TASK, "split": split, "setting": C.SETTING,
               "splits_config": "tiny（tests/fixtures 上现算）" if tiny else "configs/w2_splits.json", "splits_sha256": splits_sha256(splits_doc),
               "split_sha256": e["sha256"], "split_n": e["n"],
               "subset": {"config": meta["subsets"]["config"], "rule": subsets_doc["rule"], "salt": subsets_doc["salt"], "n": sub["n"],
                          "sha256": sub["sha256"]},
               "n_samples": len(samples), "limit": meta.get("limit"), "sample_ids_sha256": ids_digest(ids),
               "test_unlock": {"approval": C.TEST_APPROVAL, "log": f"{C.SYSTEM}/{C.TEST_UNLOCK_LOG}"}},
        "model": {"key": C.SYSTEM, "family": "external（LISA，外接 SAM）", "repo": wdoc["models"]["lisa"]["repo"],
                  "revision": wdoc["models"]["lisa"]["revision"],
                  "clip": f"{wdoc['models']['clip']['repo']}@{wdoc['models']['clip']['revision']}", "tiny": tiny,
                  "stand_in": bool(checked["stand_in"]), "weights_verified": {k: v for k, v in wts.items() if k != "manifest"} or None},
        "lisa": {"code": dict(checked["code"]), "paper": C.PAPER, "protocol": dict(ident["protocol"]), "citations": "third_party/lisa/PROVENANCE.md",
                 "runner": dict(ident["runner"]), "env": dict(ident.get("env") or {}),
                 "raw": {"run_config_sha256": run_config_sha, "rows": f"{C.SYSTEM}/{C.RAW_DIR}/{split}.jsonl", "n_rows": len(raw),
                         "rows_sha256": C.sha256_bytes(b"".join((json.dumps(r, ensure_ascii=False) + "\n").encode("utf-8") for r in raw))},
                 "manifest_files_sha256": dict(manifest_sha)},
        "scoring": dict(SCORING),
    }
    leaf = C.leaf_dir(out, split)
    _clear_leaf(leaf, rescore)
    write_run_metadata(leaf, config=cfg, seed=BOOTSTRAP_SEED, data_files=[*data_files, C.PROVENANCE_PATH, C.WEIGHTS_MANIFEST_PATH],
                       repo_root=C.REPO_ROOT)
    (leaf / PREDICTIONS).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs), encoding="utf-8")
    (leaf / TIMING).write_text("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in timing), encoding="utf-8")
    body = aggregate(recs)
    m = {"experiment": C.EXPERIMENT, "w2": cfg["w2"], "model": C.SYSTEM, "revision": cfg["model"]["revision"], "setting": C.SETTING,
         "lisa_commit": C.UPSTREAM_COMMIT, "protocol": C.PROTOCOL_NAME, "stand_in": bool(checked["stand_in"]), **body,
         "complete": len(recs) == len(samples), "n_expected": len(samples)}
    C.write_json(leaf / METRICS, m)
    write_compute_json(leaf / COMPUTE, compute_of(recs, {t["id"]: t for t in timing}))
    (leaf / README).write_text(_readme(split, cfg, m, bool(checked["stand_in"])), encoding="utf-8")
    return m


def score(out: str | Path, *, data_root: str | Path | None, allow_stand_in: bool, rescore: bool) -> dict[str, dict[str, Any]]:
    meta, rows_by_split, manifest_sha = C.read_manifest(out)
    ident, run_config_sha = read_run_config(out)
    checked = check_run(out, ident, manifest_sha, allow_stand_in=allow_stand_in)
    tiny = bool(meta.get("tiny"))
    splits_doc, subsets_doc, data_files = D.load_selection(tiny=tiny)
    paths = D.data_paths(data_root, tiny=tiny)
    res = {}
    for split, rows in rows_by_split.items():
        samples = D.split_samples(split, paths, splits_doc, subsets_doc, meta.get("limit"))
        res[split] = score_split(out, split, samples, rows, ident=ident, run_config_sha=run_config_sha, checked=checked, meta=meta,
                                 manifest_sha=manifest_sha, splits_doc=splits_doc, subsets_doc=subsets_doc,
                                 data_files=[*data_files, *([] if tiny else paths.config_files())], rescore=rescore)
    return res


__all__ = ["FRAMEWORK", "ISOLATION_NOTE", "LisaScoreError", "aggregate", "check_run", "item_record", "score", "score_split"]
