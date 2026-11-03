"""X-M 的训练与判定（DEV-0081）测试用的假数据（DEV-0079 的数据根假数据在 tests/_xm_fakes.py，这里不重复）：

- 训练目标：DEV-0079 的行格式（src/xm/targets.pair_targets 写的那些键；过 targets.row_problems），manifest 用 targets.finalize 写。
  图像取 E3 训练池的 fixture（tests/fixtures/e3/root 的 4 张 128×96 合成图，与 X-2 的假数据同一个数据根），每张图几个"类别"，每对给定
  框数（含 > 12、截到 30 的对，测按 12 个框切段），框是网格上的确定矩形、按 targets.reading_order 排，样条是框内的椭圆（夹进框）；
  另有带 crowd 的对（D167-4：两条流都剔除）与样条目标不可用的对（只进列框流）。
- 判定的输入（scripts/gate_xm.py）：X-M 的检查点（训练叶目录的布局，config.yaml 记 config、seed、xm.prompts.version）、DEV-0080 格式的
  叶目录（接口的假设，见 gate_xm 的模块说明）、run_x2 格式（W2）的指代分割叶目录、lang_battery 的运行目录（lang_battery.json +
  predictions.jsonl，逐题结果手造，scripts/lang_battery.load_run 能读）与 --compare 的输出（诊断）。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml
from PIL import Image

from src.explore.x2_tools import ellipse_ctrl_q
from src.xm import data as XD
from src.xm import metrics as XMM
from src.xm import prompts as XMP
from src.xm import runner as XR
from src.xm import targets as XT
from tests import _w2c_fakes as FK

ROOT = FK.ROOT
CATEGORIES = ("person", "dog", "traffic light", "cup", "kite")
N_BOXES = (1, 2, 33, 13, 25, 3, 4, 1, 2, 12, 5, 1, 3, 2, 6, 1, 24, 2, 1, 7)  # 4 张图 × 5 类 = 20 对；33 个实例的截到 30
CROWD = {5: 2, 14: 1}  # 第 5、14 对：该类在图里带 crowd 标注（D167-4：整对剔除）
NO_SPLINE = (10,)  # 第 10 对：样条目标不可用（一个实例拟合失败），只进列框流


def fixture_images() -> list[tuple[str, int, int]]:
    out = set()
    for s in FK.train_pool():
        rel = str(Path(s.images[0]).resolve().relative_to(Path(ROOT).resolve()))
        with Image.open(Path(ROOT) / rel) as im:
            out.add((rel, im.size[0], im.size[1]))
    return sorted(out)


def grid_boxes(n: int, seed: int) -> list[list[int]]:
    """n 个确定的网格框（未排序）。"""
    out = []
    for k in range(n):
        x1 = (97 * k + 31 * seed) % 800
        y1 = (61 * k + 17 * seed) % 800
        out.append([x1, y1, x1 + 60 + (k % 4) * 30, y1 + 80 + (k % 3) * 40])
    return out


def target_row(i: int, n: int, *, n_crowd: int = 0, spline_ok: bool = True) -> dict[str, Any]:
    """DEV-0079 格式的一行（同 targets.pair_targets 的键）。"""
    imgs = fixture_images()
    rel, w, h = imgs[i % len(imgs)]
    cat = i // len(imgs)
    raw = grid_boxes(n, i)
    order = XT.reading_order(raw)
    kept = [raw[k] for k in order][: XT.MAX_BOXES]
    ctrls = [XT.clamp_ctrl(ellipse_ctrl_q(b, 1000, 1000, XT.SPLINE_N), b) for b in kept]
    fits: list[Any] = [round(0.8 + 0.01 * (k % 10), 3) for k in range(len(kept))]
    fails = []
    if not spline_ok:
        fits[0] = None
        fails = [{"ann_id": 100 * i, "reason": XT.FAIL_FIT}]
    return {"id": f"coco2017/train2017/{9000 + i % len(imgs)}/{cat + 1}", "kind": XT.KIND, "source": XD.SOURCE_COCO, "split": XD.TRAIN,
            "image": rel, "image_size": [w, h], "category_id": cat + 1, "category": CATEGORIES[cat % len(CATEGORIES)], "n_instances": n,
            "n_kept": len(kept), "truncated": n > XT.MAX_BOXES, "n_crowd": int(n_crowd), "ann_ids": [100 * i + k for k in range(len(kept))],
            "boxes_q": kept, "box_text": XT.box_list_text(kept), "n": XT.SPLINE_N, "spline_ok": spline_ok, "spline_fail": fails,
            "ctrl_q": ctrls if spline_ok else None, "spline_text": XT.spline_list_text(ctrls) if spline_ok else None, "fit_iou": fits}


def target_rows(n_boxes: Sequence[int] = N_BOXES) -> list[dict[str, Any]]:
    return [target_row(i, n, n_crowd=CROWD.get(i, 0), spline_ok=i not in NO_SPLINE) for i, n in enumerate(n_boxes)]


def write_targets(out: str | Path, rows: Sequence[Mapping[str, Any]], *, shard_size: int = 6) -> Path:
    """<out>/xm_targets/targets_<k>.jsonl + manifest.json（targets.finalize：参数 = targets.DEFAULT_PARAMS）。"""
    out = Path(out)
    d = out / XT.DATA_DIR
    d.mkdir(parents=True, exist_ok=True)
    n = 0
    for k, i in enumerate(range(0, len(rows), shard_size)):
        (d / XT.shard_name(k)).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows[i: i + shard_size]), encoding="utf-8")
        n = k + 1
    XT.finalize(out, n_shards=n, params=XT.check_params(XT.DEFAULT_PARAMS), source={"fake": "tests/_xm_train_fakes（DEV-0081）"})
    return out


# ---------------------------------------------------------------------------
# 判定的输入
# ---------------------------------------------------------------------------


def decision_pairs(n_images: int = 500) -> list[XD.XMPair]:
    """n_images 张图，每张 1–3 个类别（按图号轮换），id = coco2017/val2017/<image_id>/<category_id>。"""
    out = []
    for i in range(n_images):
        for c in range(1 + i % 3):
            out.append(XD.XMPair(id=f"coco2017/val2017/{100000 + i}/{c + 1}", source=XD.SOURCE_COCO, split=XD.XM_DECISION,
                                 image=f"coco2017/images/val2017/{100000 + i:012d}.jpg", width=640, height=480, category_id=c + 1,
                                 category=CATEGORIES[c], instances=(), crowd_rles=()))
    return out


def n_gt(pair_id: str) -> int:
    return 1 + int(pair_id.split("/")[2]) % 4


def counts(pair_id: str, quality: str) -> dict[str, int]:
    """确定的计数：good = 全对；mid = 真值多于 1 个时少一个 TP、多一个 FP；bad = 只对一半、多一个 FP。"""
    g = n_gt(pair_id)
    if quality == "good":
        return {"tp": g, "fp": 0, "fn": 0}
    if quality == "mid":
        tp = g - 1 if g > 1 else g
        return {"tp": tp, "fp": g - tp, "fn": g - tp}
    tp = g // 2
    return {"tp": tp, "fp": 1, "fn": g - tp}


def eval_counts(pair_id: str, quality: str) -> dict[str, Any]:
    """一个对在全部 IoU 阈值上的计数（src/xm/metrics.eval_pair 的形状；各阈值取同样的 counts）。"""
    c = counts(pair_id, quality)
    t = len(XMM.IOU_THRESHOLDS)
    return {"tp": [c["tp"]] * t, "fp": [c["fp"]] * t, "fn": [c["fn"]] * t, "ignored": [0] * t, "matched_iou": [0.8] * c["tp"]}


def checkpoint(manifest: str, *, config_sha256: str, seed: int = 0) -> dict[str, Any]:
    return {"path": f"/ck/{manifest[:8]}", "manifest_sha256": manifest, "training": {"config_sha256": config_sha256, "seed": seed}}


PROMPT_VERSION = XMP.PROMPT_VERSION  # 冻结的 xm-prompts-v1（D168）


def write_xm_checkpoint(root: Path, *, config_sha256: str, seed: int = 0, prompt_version: str | None = PROMPT_VERSION,
                        version: str = "xm-train-v1") -> tuple[Path, str]:
    """<root>/w2c/qwen3.5-9b/seed<k>/ckpt/final/model/manifest.json + 叶目录的 config.yaml（运行身份：config、seed、xm.prompts.version）。
    返回 (检查点目录, manifest 的 sha256)；manifest 的内容随 root 变，不同 root 是不同的检查点。"""
    leaf = root / "w2c" / "qwen3.5-9b" / f"seed{seed}"
    ck = leaf / "ckpt" / "final" / "model"
    ck.mkdir(parents=True, exist_ok=True)
    (ck / "manifest.json").write_text(json.dumps({"format": "fake", "root": str(root)}) + "\n", encoding="utf-8")
    ident = {"variant": "w2c", "seed": seed, "config": {"path": "configs/xm_train.json", "sha256": config_sha256, "version": version}}
    if prompt_version is not None:
        ident["xm"] = {"prompts": {"version": prompt_version}}
    (leaf / "config.yaml").write_text(yaml.safe_dump(ident, allow_unicode=True), encoding="utf-8")
    return ck, hashlib.sha256((ck / "manifest.json").read_bytes()).hexdigest()


def write_xm_leaf(leaf: Path, setting: str, pairs: Sequence[XD.XMPair], *, det: str | None, seg: str | None, ck: Mapping | None,
                  host: str = "hpcgpu06", tiny: bool = False, limit: int | None = None, unseal: str | None = XD.GATE,
                  split: str = XD.XM_DECISION, model: str = "qwen3.5-9b", complete: bool = True,
                  prompt_version: str | None = PROMPT_VERSION, prompts_frozen: bool = True) -> Path:
    """src/xm/runner.derive_settings 的叶目录格式（predictions.jsonl 的行：id、image_key、category_id、det / seg = 逐 IoU 阈值的计数，
    同 src/xm/metrics.eval_pair；config.yaml 的 model、checkpoint、xm、prompts 几节）。det / seg：counts 的 quality，None = 这一项是 null。
    真跑出来的叶目录见 tests/test_gate_xm.py 的端到端测试（src/xm/runner.run_split + DEV-0080 的假适配器）。"""
    leaf.mkdir(parents=True, exist_ok=True)
    recs = []
    for p in pairs:
        recs.append({"id": p.id, "image_key": XR.image_key(p), "category_id": p.category_id, "setting": setting,
                     "det": None if det is None else eval_counts(p.id, det), "seg": None if seg is None else eval_counts(p.id, seg)})
    (leaf / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    (leaf / "metrics.json").write_text(json.dumps({"complete": complete, "n": len(recs)}), encoding="utf-8")
    cfg = {"experiment": "X-M", "model": {"key": model, "revision": "r", "tiny": tiny}, "checkpoint": None if ck is None else dict(ck),
           "xm": {"task": "xm", "split": split, "system": "base" if ck is None else "xm", "limit": limit, "setting": setting,
                  **({} if unseal is None else {"unseal": {"gate": unseal, "decision": XD.SEAL_DECISION}})},
           "prompts": {"version": prompt_version, "frozen": prompts_frozen, "templates_sha256": "t" * 64}}
    (leaf / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    (leaf / "meta.json").write_text(json.dumps({"hostname": host}), encoding="utf-8")
    return leaf


def refseg_ids(split: str, n: int) -> list[str]:
    return [f"{split}/{k:04d}" for k in range(n)]


def write_refseg_leaf(leaf: Path, split: str, n: int, *, ck: Mapping | None, host: str = "hpcgpu07", shrink: float = 0.0,
                      model: str = "qwen3.5-9b", tiny: bool = False, limit: int | None = None, setting: str = "x2_spline_trimap") -> Path:
    """run_x2 格式（W2）的叶目录：每句 inter = round(union × (0.6 + 0.3 × (k mod 7) / 6) × (1 − shrink))。"""
    leaf.mkdir(parents=True, exist_ok=True)
    recs = []
    for k, i in enumerate(refseg_ids(split, n)):
        union = 1000 + 37 * (k % 11)
        inter = int(round(union * (0.6 + 0.3 * (k % 7) / 6) * (1 - shrink)))
        recs.append({"id": i, "score": {"inter": inter, "union": union, "iou": inter / union}})
    (leaf / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    (leaf / "metrics.json").write_text(json.dumps({"complete": True, "n": n}), encoding="utf-8")
    cfg = {"model": {"key": model, "revision": "r", "tiny": tiny}, "checkpoint": None if ck is None else dict(ck),
           "w2": {"task": "refseg", "setting": setting, "split": split, "split_sha256": "0" * 64, "n_samples": n, "limit": limit},
           "w2_config": {"frozen": True}, "propose": {"mode": "box"}, "candidates": {}}
    (leaf / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    (leaf / "meta.json").write_text(json.dumps({"hostname": host}), encoding="utf-8")
    return leaf


def write_lb_compare(path: Path, *, base_ck: str, other_ck: str, mmlu_ci: Sequence[float] = (-1.4, 1.2), ifeval_ci: Sequence[float] = (-1.9, 3.0),
                     hosts: Sequence[str] = ("hpcgpu07", "hpcgpu07"), partial: bool = False, n: Sequence[int] = (2000, 541)) -> Path:
    """lang_battery --compare 的输出（只有 gate_xm 读的键）与两边运行目录的 meta.json。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    sides = {}
    for side, ck, host in (("base", base_ck, hosts[0]), ("other", other_ck, hosts[1])):
        d = path.parent / f"lb_{side}"
        d.mkdir(exist_ok=True)
        (d / "meta.json").write_text(json.dumps({"hostname": host}), encoding="utf-8")
        sides[side] = {"label": side, "path": str(d / "lang_battery.json"), "model": "qwen3.5-9b",
                       "checkpoint": {"dir": "/ck", "manifest_sha256": ck}, "limit": None}
    rows = [{"reading": "mmlu_pro_letter", "metric": "accuracy", "headline": True, "n": n[0], "base_pct": 60.0, "other_pct": 59.9,
             "delta_pp": round((mmlu_ci[0] + mmlu_ci[1]) / 2, 1), "ci95_pp": list(mmlu_ci)},
            {"reading": "ifeval", "metric": "prompt_level_strict", "headline": True, "n": n[1], "base_pct": 70.0, "other_pct": 70.5,
             "delta_pp": round((ifeval_ci[0] + ifeval_ci[1]) / 2, 1), "ci95_pp": list(ifeval_ci)},
            {"reading": "ifeval", "metric": "instruction_level_strict", "headline": False, "n": 834, "base_pct": 78.0, "other_pct": 78.0,
             "delta_pp": 0.0, "ci95_pp": [-1.0, 1.0]}]
    doc = {"script": "scripts/lang_battery.py --compare", "battery_version": 1, **sides, "partial_run": partial, "rows": rows,
           "statistics": {"bootstrap": {"n_resamples": 1000, "level": 0.95, "seed": 20260929}}}
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def write_lb_run(d: Path, *, ck: str | None, mmlu: Sequence[bool], ifeval: Sequence[bool], host: str | None = "hpcgpu07",
                 meta_json: bool = True, limit: int | None = None, model: str = "qwen3.5-9b", mode: str = "real", ok: bool = True,
                 battery_version: int = 1) -> Path:
    """一次 lang_battery 运行的目录（scripts/lang_battery.load_run 读的那些键）：逐题结果手造——MMLU-Pro 的 correct、IFEval 的
    strict.follow_all；节点写在 meta.json（meta_json）与 lang_battery.json 的 run.meta.hostname。"""
    d.mkdir(parents=True, exist_ok=True)
    lines = [{"reading": "mmlu_pro_letter", "id": f"mmlu-{k:04d}", "correct": bool(c), "p_gold": 0.5, "meta": {"category": f"cat{k % 14}"}}
             for k, c in enumerate(mmlu)]
    lines += [{"reading": "ifeval", "id": f"ifeval-{k:04d}", "strict": {"follow_all": bool(c), "follow_list": [bool(c)]},
               "loose": {"follow_all": bool(c), "follow_list": [bool(c)]}, "meta": {"in_health_check": False}} for k, c in enumerate(ifeval)]
    pred = d / "predictions.jsonl"
    pred.write_text("".join(json.dumps(r) + "\n" for r in lines), encoding="utf-8")
    doc = {"script": "scripts/lang_battery.py", "battery_version": battery_version, "ok": ok, "mode": mode, "limit": limit,
           "label": f"fake @ {ck}", "model": {"key": model}, "checkpoint": None if ck is None else {"dir": "/ck", "manifest_sha256": ck},
           "data": {"mmlu_pro": {"sha256": "m" * 64, "ids_sha256": "i" * 64, "sampling": {"n": len(mmlu)}},
                    "ifeval": {"sha256": "f" * 64, "ids_sha256": "j" * 64}},
           "run": {"meta": {"hostname": host}}, "predictions": {"path": "predictions.jsonl", "sha256": hashlib.sha256(pred.read_bytes()).hexdigest()}}
    (d / "lang_battery.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    if meta_json:
        (d / "meta.json").write_text(json.dumps({"hostname": host}), encoding="utf-8")
    return d
