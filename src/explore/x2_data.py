"""X-2 的训练数据（DEV-0069；预注册 X）：在 CPU 上从 W2-c 的训练数据行（DEV-0060 的 w2c_data，RUN-0067/0069 的 16k 句）算描样条与
指修正框的目标，不跑模型。监督只来自真值（gt_rle）与 W2 冻结的候选（数据行里存好的 RLE），工具只用 src/explore/guidance。

**输入**：<w2c_data>/*.jsonl（docs/w2_plan.md §5 的行：id、source、split = train、image、image_size、expression、gt_box、gt_rle、proposal.box、
candidates[].index / name / ok / rle / iou……；按文件名升序）与同目录的 manifest.json。图像相对 --data-root（同 W2-c 的训练）。

**样条行**（spline_row；参数 spline.n、spline.min_box_iou）：
- 不用的行（按原因计数）：提出失败（proposal.box 为 null）；提出的框与真值框（gt_box）的 IoU < min_box_iou（geometry.box_iou）；
  真值 mask 为空；拟合不出轮廓。
- 目标：gt_rle 的 mask 按 oracle_sim.oracle_spline(gt, n, parts = 1, holes = False) 拟合——面积最大的 8 连通域填洞后的外轮廓，
  n 个控制点的闭合周期均匀三次 B 样条（guidance.fit_mask_bspline），夹到图内后取整到 0–1000 网格（同 DEV-0068 的 parts = 1，无噪声）。
- 框：数据行里提出的框（基座的 E1 框，原图像素）与它的网格整数 box_q（x2_prompts.box_q）。
- fit_iou：目标按推理时的同一套光栅化（x2_prompts.spline_mask：采样点夹在提出的框内）与真值的 IoU，即这一行描样条的上限。

**修正行**（correct_rows；参数 correct.m、correct.min_box_iou、start_salt、round2_ratio、round2_salt、round2_apply）：
- 不用的行：提出失败；min_box_iou 不是 null 时框 IoU 低于它（仓库配置是 null：修正行不按框过滤，见完成总结）；真值 mask 为空。
- 起点（start_choice，确定性的轮换）：4 个候选按记下的真值 IoU 从高到低排（并列取编号小的）得到名次 0..K−1；
  h = int(sha256(start_salt|id), 16)，档 = ("good", "mid", "bad")[h mod 3]：good 取名次 0（最好的）、bad 取名次 K − 1（最差的）、
  mid 取名次 1 + (h // 3) mod (K − 2)（中间的）。三档各约 1/3。起点 mask = 该候选的 RLE（DEV-0060 存的就是回路里用的 mask：1 号失败时为空）。
- 目标（correction_targets；同 oracle_sim.diff_components 的分块与排序）：起点与真值之差的 8 连通块，先与提出的框求交（框外的改不了），
  漏掉的（真值 ∖ 起点）记 "+"、多出的（起点 ∖ 真值）记 "-"，按面积从大到小（并列先 "+"、再按块里第一个像素的行优先序号）逐个看：
  外接框 [列₀, 行₀, 列₁ + 1, 行₁ + 1] 取整到 0–1000 网格，换回原图像素、与提出的框求交（空了就丢，计 n_outside），再过 effect gate
  （guidance.effect_gate，门槛 EFFECT_MIN_PX、EFFECT_MIN_FRAC 同 guidance 的缺省；第 1 轮的承诺场为 0，第 2 轮用第 1 轮施加后的承诺场），
  过不了计 n_gated；过了的留下，留满 m 个为止。一个都没留下时目标是 []。这几步与推理时 correction_round 对模型输出的处理一致，所以
  目标里的每个框在同一个起点上都不会被约束框或 effect gate 去掉。
- 第 2 轮（round2_ratio 是第 2 轮行数与第 1 轮行数之比的期望）：第 1 轮目标非空、且 int(sha256(round2_salt|id), 16) / 2²⁵⁶ < round2_ratio
  的句子另造一行：把第 1 轮的目标（网格框换回原图像素）用 guidance.correction_round 施加到起点上（rule = round2_apply.rule = grabcut、
  soft_lock = round2_apply.soft_lock = true、effect gate 与约束框 = 提出的框，其余取 guidance 的缺省），得到第 2 轮的起点与承诺场，
  再按同样的规则算目标。第 1 轮目标为 [] 的句子不造第 2 轮（起点不变，会是重复的一行），计数。
- 每行存起点的 RLE（训练时现画叠加图，x2_render.render_overlay）与起点的真值 IoU。

**输出**（<out>/）：
- x2_data/spline_<分片名>.jsonl、x2_data/correct_<分片名>.jsonl：一个输入分片对应一对输出分片（按输入的行序）；先写 .tmp、fsync 之后改名，
  两个都改名之后这一片才算写完；
- x2_data/manifest.json：参数、输入（w2c_data 的 manifest 与各分片的 sha256）、过滤与各类的条数、各输出文件的条数与 sha256、
  fit_iou 的分布、目标为 [] 的比例、每个目标的框数、起点的档与 IoU（只由输出的行算，不含耗时与路径，逐字节可复现）；
- config.yaml（运行身份：参数、输入的指纹）、logs/timing.jsonl（每句的耗时）、logs/run_<k>.json。
续跑：config.yaml 相同时，已写完的片核对 id 后跳过，没写完的片整片重算；除 logs/ 外与一次跑完逐字节相同。--workers N：一片之内按
句子用 spawn 的进程池（子进程里 OpenCV 与 BLAS 都是 1 个线程，同 W2 的候选生成），结果按顺序写回，与单进程逐字节相同。

**读取**（load_x2_data，训练用）：manifest 与每个文件的 sha256 对得上；参数等于训练配置的；来源的 w2c_data 与训练读的那一份是同一份
（manifest 与分片的 sha256）；每行的 target_text 等于按 x2_prompts 重新写出的文本、且能被严格解析回同样的目标。
只读一部分类（kinds，DEV-0070 的 X-3：只训描样条）：只读、只核对这几类的文件与行，参数只比这几类的那几节（manifest 的其余类不看）；
kinds 缺省 = params 里有的类，两类都有时与以前逐字节相同。

只 import 不含学习的模块（tests/test_x2_data.py 在干净的子进程里核对）。
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from src.executors.geometry import box_iou
from src.executors.raster import mask_to_rle, rle_to_mask
from src.explore import guidance as G
from src.explore import oracle_sim as OS
from src.explore import x2_prompts as XP
from src.explore.x2_tools import ToolPool, load_rgb
from src.metrics.seg import mask_inter_union, sample_iou

SCHEMA = 1
FRAMEWORK = "src/explore/x2_data.py（DEV-0069；探索 X-2）"
DATA_DIR = "x2_data"
MANIFEST = "manifest.json"
CONFIG = "config.yaml"
TIMING = "logs/timing.jsonl"
W2C_MANIFEST = "manifest.json"
KINDS: tuple[str, ...] = ("spline", "correct")
TIERS: tuple[str, ...] = ("good", "mid", "bad")
ROUND2_APPLY_RULES = G.CORRECTION_RULES
INPUT_KEYS = ("id", "source", "split", "image", "image_size", "expression", "gt_box", "gt_rle", "proposal", "candidates")
DEFAULT_PARAMS: Mapping[str, Any] = {
    "spline": {"n": XP.SPLINE_N_DEFAULT, "min_box_iou": 0.5},
    "correct": {"m": XP.CORRECT_M_DEFAULT, "min_box_iou": None, "start_salt": "x2_correct_start_v1", "round2_ratio": 0.5,
                "round2_salt": "x2_correct_round2_v1", "round2_apply": {"rule": "grabcut", "soft_lock": True}},
}


class X2DataError(ValueError):
    """X-2 的数据不能生成、不合格，或与训练配置 / W2-c 的数据对不上。"""


# ---------------------------------------------------------------------------
# 参数
# ---------------------------------------------------------------------------


def _num01(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v)) and 0.0 <= float(v) <= 1.0


def check_kinds(kinds: Sequence[str]) -> tuple[str, ...]:
    """kinds（DEV-0070）：KINDS 的非空子集，按 KINDS 的先后、不重复。"""
    t = tuple(kinds)
    if not t or len(set(t)) != len(t) or set(t) - set(KINDS) or list(t) != [k for k in KINDS if k in t]:
        raise X2DataError(f"kinds 必须是 {list(KINDS)} 的非空子集（按这个先后），收到 {list(t)}")
    return t


def check_params(doc: Mapping[str, Any], *, kinds: Sequence[str] = KINDS) -> dict[str, Any]:
    """数据参数（configs/x2_train.json 的 spline、correct 两节里与数据有关的键）→ 规范化的 {"spline": {...}, "correct": {...}}。
    缺的键报错（不补缺省）；多余的键忽略（训练用的键也在这两节里）。kinds（DEV-0070）：只核对、只返回这几类（缺省两类，与以前相同）。"""
    kinds = check_kinds(kinds)
    bad: list[str] = []
    sp, co = doc.get("spline"), doc.get("correct")
    if kinds != KINDS:
        if any(not isinstance(doc.get(k), Mapping) for k in kinds):
            raise X2DataError(f"参数必须有 {'、'.join(kinds)} 一节")
    elif not isinstance(sp, Mapping) or not isinstance(co, Mapping):
        raise X2DataError("参数必须有 spline 与 correct 两节")
    out: dict[str, Any] = {}
    if "spline" in kinds:
        out_sp: dict[str, Any] = {}
        try:
            out_sp["n"] = XP.check_n(sp.get("n"))
        except ValueError as e:
            bad.append(f"spline.n：{e}")
        if not _num01(sp.get("min_box_iou")):
            bad.append(f"spline.min_box_iou 必须是 [0, 1] 内的数，收到 {sp.get('min_box_iou')!r}")
        else:
            out_sp["min_box_iou"] = float(sp["min_box_iou"])
        out["spline"] = out_sp
    if "correct" in kinds:
        out["correct"] = _check_correct(co, bad)
    if bad:
        raise X2DataError("X-2 的数据参数不合格：\n- " + "\n- ".join(bad))
    return out


def _check_correct(co: Mapping[str, Any], bad: list[str]) -> dict[str, Any]:
    """correct 一节的数据参数（check_params 的后半；问题追加进 bad）。"""
    out_co: dict[str, Any] = {}
    try:
        out_co["m"] = XP.check_m(co.get("m"))
    except ValueError as e:
        bad.append(f"correct.m：{e}")
    mb = co.get("min_box_iou", "缺")
    if mb is not None and not _num01(mb):
        bad.append(f"correct.min_box_iou 必须是 null 或 [0, 1] 内的数，收到 {mb!r}")
    else:
        out_co["min_box_iou"] = None if mb is None else float(mb)
    for k in ("start_salt", "round2_salt"):
        if not isinstance(co.get(k), str) or not co[k]:
            bad.append(f"correct.{k} 必须是非空字符串")
        else:
            out_co[k] = co[k]
    if co.get("start_salt") is not None and co.get("start_salt") == co.get("round2_salt"):
        bad.append("correct.start_salt 与 round2_salt 不能相同")
    if not _num01(co.get("round2_ratio")):
        bad.append(f"correct.round2_ratio 必须是 [0, 1] 内的数，收到 {co.get('round2_ratio')!r}")
    else:
        out_co["round2_ratio"] = float(co["round2_ratio"])
    ap = co.get("round2_apply")
    if not isinstance(ap, Mapping) or set(ap) != {"rule", "soft_lock"} or ap["rule"] not in ROUND2_APPLY_RULES \
            or not isinstance(ap["soft_lock"], bool):
        bad.append(f"correct.round2_apply 必须恰是 {{rule: {list(ROUND2_APPLY_RULES)} 之一, soft_lock: bool}}，收到 {ap!r}")
    else:
        out_co["round2_apply"] = {"rule": ap["rule"], "soft_lock": bool(ap["soft_lock"])}
    return out_co


def tool_constants() -> dict[str, Any]:
    """数据规则里用到的 guidance 缺省（进 manifest 与运行身份）。"""
    return {"effect_min_px": G.EFFECT_MIN_PX, "effect_min_frac": G.EFFECT_MIN_FRAC, "lock_alpha": G.LOCK_ALPHA,
            "context_pct": G.CONTEXT_PCT_DEFAULT, "grabcut_iterations": G.GRABCUT_ITERATIONS, "prompt_version": XP.PROMPT_VERSION,
            "text_format": OS.TEXT_FORMAT_VERSION}


# ---------------------------------------------------------------------------
# 规则（纯函数）
# ---------------------------------------------------------------------------


def _h(salt: str, sample_id: str) -> int:
    return int(hashlib.sha256(f"{salt}|{sample_id}".encode("utf-8")).hexdigest(), 16)


def iou_ranking(cands: Sequence[Mapping[str, Any]]) -> list[int]:
    """候选按记下的真值 IoU 从高到低的下标（0 起），并列取编号小的。"""
    return sorted(range(len(cands)), key=lambda i: (-float(cands[i]["iou"]), int(cands[i]["index"])))


def start_choice(sample_id: str, cands: Sequence[Mapping[str, Any]], salt: str) -> dict[str, Any]:
    """修正的起点（模块说明）：{"tier", "rank", "index"（候选编号）, "name", "iou"}。"""
    k = len(cands)
    if k < 3:
        raise X2DataError(f"{sample_id}：起点的三档轮换要 K ≥ 3 个候选，收到 {k}")
    h = _h(salt, sample_id)
    tier = TIERS[h % 3]
    rank = 0 if tier == "good" else (k - 1 if tier == "bad" else 1 + (h // 3) % (k - 2))
    c = cands[iou_ranking(cands)[rank]]
    return {"tier": tier, "rank": int(rank), "index": int(c["index"]), "name": str(c["name"]), "iou": float(c["iou"])}


def wants_round2(sample_id: str, salt: str, ratio: float) -> bool:
    return _h(salt, sample_id) / float(2 ** 256) < float(ratio)


def correction_targets(start: np.ndarray, gt: np.ndarray, lock: np.ndarray | None, box: Sequence[float], m: int) -> tuple[list[dict], dict]:
    """一轮修正的目标（模块说明）：([{"sign", "box"（网格整数）}], {"n_components", "n_outside", "n_gated", "areas"})。"""
    h, w = gt.shape
    lk = np.zeros((h, w), np.float64) if lock is None else lock
    comps = OS.diff_components(start, gt, confine_box=box)
    kept, areas = [], []
    n_out = n_gated = 0
    for d in comps:
        if len(kept) >= m:
            break
        q = OS.quantize_box(d["box"], w, h)
        b = G.intersect_box(OS.dequantize_box(q, w, h), box)
        if b is None:
            n_out += 1
            continue
        if not G.effect_gate(d["sign"], b, start, lk):
            n_gated += 1
            continue
        kept.append({"sign": d["sign"], "box": q})
        areas.append(int(d["area"]))
    return kept, {"n_components": len(comps), "n_outside": n_out, "n_gated": n_gated, "areas": areas}


def _iou(a: np.ndarray, gt: np.ndarray) -> float:
    st = mask_inter_union(a, gt)
    return float(sample_iou(st.inter, st.union))


def _base(row: Mapping[str, Any], box: Sequence[float], biou: float | None) -> dict[str, Any]:
    w, h = row["image_size"]
    return {"source": row["source"], "image": row["image"], "image_size": [int(w), int(h)], "expression": row["expression"],
            "box": [float(v) for v in box], "box_q": XP.box_q(box, w, h), "box_iou": biou}


def spline_row(row: Mapping[str, Any], gt: np.ndarray, box: Sequence[float], biou: float, n: int) -> tuple[dict | None, str | None]:
    """一句的样条行（模块说明）；不用时返回 (None, 原因)。"""
    sp = OS.oracle_spline(gt, n, parts=1, holes=False)
    if not sp["loops_q"]:
        return None, "fit_failed"
    ctrl_q = [[int(x), int(y)] for x, y in sp["loops_q"][0]["ctrl"]]
    fit = XP.spline_mask(ctrl_q, box, gt.shape)
    return {"id": row["id"], "kind": "spline", **_base(row, box, biou), "n": int(n), "ctrl_q": ctrl_q,
            "target_text": XP.spline_text(ctrl_q), "fit_iou": _iou(fit, gt)}, None


def _correct(row: Mapping[str, Any], box: Sequence[float], biou: float | None, *, rnd: int, start: np.ndarray, start_info: Mapping[str, Any],
             gt: np.ndarray, lock: np.ndarray | None, m: int, applied: Mapping[str, Any] | None = None) -> dict[str, Any]:
    tgt, info = correction_targets(start, gt, lock, box, m)
    out = {"id": f"{row['id']}#r{rnd}", "sample_id": row["id"], "kind": "correct", "round": int(rnd), **_base(row, box, biou), "m": int(m),
           "start": {**dict(start_info), "start_iou": _iou(start, gt)}, "start_rle": mask_to_rle(start), "target": tgt,
           "target_text": XP.corrections_text(tgt), "target_areas": info["areas"],
           "n_components": info["n_components"], "n_outside": info["n_outside"], "n_gated": info["n_gated"]}
    if applied is not None:
        out["applied"] = dict(applied)
    return out


def load_image(root: str | Path, rel: str, size: Sequence[int]) -> np.ndarray:
    """原图 (H, W, 3) uint8 RGB（x2_tools.load_rgb：PIL 打开后 convert("RGB")，同 W2 的候选生成）；尺寸必须等于 image_size。"""
    arr = load_rgb(str(Path(root) / rel))
    if [arr.shape[1], arr.shape[0]] != [int(size[0]), int(size[1])]:
        raise X2DataError(f"{rel}：图像 {arr.shape[1]}×{arr.shape[0]} 与 image_size {list(size)} 不符")
    return arr


def input_problems(row: Mapping[str, Any]) -> list[str]:
    bad = [f"缺少 {k}" for k in INPUT_KEYS if k not in row]
    if bad:
        return bad
    if row["split"] != "train":
        bad.append(f"split = {row['split']!r}（只读训练集）")
    sz = row["image_size"]
    if not (isinstance(sz, list) and len(sz) == 2 and all(isinstance(v, int) and v > 0 for v in sz)):
        return bad + [f"image_size 不对：{sz!r}"]
    w, h = sz
    if list((row["gt_rle"] or {}).get("size") or []) != [h, w]:
        bad.append("gt_rle 的尺寸不是 [H, W]")
    p = row["proposal"]
    if not isinstance(p, Mapping) or (p.get("box") is None) != (row["candidates"] is None):
        bad.append("proposal.box 为 null ⟺ candidates 为 null")
    for c in row["candidates"] or []:
        if list((c.get("rle") or {}).get("size") or []) != [h, w] or not isinstance(c.get("iou"), (int, float)):
            bad.append(f"候选 {c.get('index')} 的 rle 尺寸或 iou 不对")
    return bad


def sentence_job(job: tuple[Mapping[str, Any], str, Mapping[str, Any]]) -> tuple[dict[str, Any], float]:
    """一句的全部输出（单进程与进程池共用）：(w2c 的行, 图像根, 参数) → ({"id", "spline", "spline_skip", "correct", "correct_skip",
    "round2_skip"}, 墙钟)。"""
    row, root, params = job
    t0 = time.perf_counter()
    sp_p, co_p = params["spline"], params["correct"]
    res: dict[str, Any] = {"id": row["id"], "spline": None, "spline_skip": None, "correct": [], "correct_skip": None, "round2_skip": None}
    box = (row["proposal"] or {}).get("box")
    if box is None:
        res["spline_skip"] = res["correct_skip"] = "proposal_failed"
        return res, time.perf_counter() - t0
    gt = rle_to_mask(row["gt_rle"]).astype(bool)
    biou = float(box_iou(box, row["gt_box"])) if (row["gt_box"][2] > row["gt_box"][0] and row["gt_box"][3] > row["gt_box"][1]) else 0.0
    if not gt.any():
        res["spline_skip"] = res["correct_skip"] = "gt_empty"
        return res, time.perf_counter() - t0
    if biou < sp_p["min_box_iou"]:
        res["spline_skip"] = "box_iou_low"
    else:
        res["spline"], res["spline_skip"] = spline_row(row, gt, box, biou, sp_p["n"])
    if co_p["min_box_iou"] is not None and biou < co_p["min_box_iou"]:
        res["correct_skip"] = "box_iou_low"
        return res, time.perf_counter() - t0
    m = co_p["m"]
    st = start_choice(row["id"], row["candidates"], co_p["start_salt"])
    start = rle_to_mask(row["candidates"][st["index"] - 1]["rle"]).astype(bool)
    r1 = _correct(row, box, biou, rnd=1, start=start, start_info=st, gt=gt, lock=None, m=m)
    res["correct"].append(r1)
    if not wants_round2(row["id"], co_p["round2_salt"], co_p["round2_ratio"]):
        res["round2_skip"] = "not_drawn"
    elif not r1["target"]:
        res["round2_skip"] = "round1_empty"
    else:
        ap = co_p["round2_apply"]
        img = load_image(root, row["image"], row["image_size"])
        w, h = row["image_size"]
        m2, lk2, info = G.correction_round(img, start, None, XP.proposals_abs(r1["target"], w, h), rule=ap["rule"],
                                           soft_lock=ap["soft_lock"], gate=True, confine_box=box)
        applied = {"rule": ap["rule"], "soft_lock": ap["soft_lock"], **{k: int(v) for k, v in info.items()}}
        res["correct"].append(_correct(row, box, biou, rnd=2, start=m2, start_info=st, gt=gt, lock=lk2 if ap["soft_lock"] else None,
                                       m=m, applied=applied))
    return res, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# 输入与输出的文件
# ---------------------------------------------------------------------------


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def input_shards(w2c_dir: str | Path) -> list[Path]:
    d = Path(w2c_dir)
    shards = sorted(d.glob("*.jsonl"))
    if not shards:
        raise X2DataError(f"{d} 里没有 *.jsonl 分片")
    return shards


def source_identity(w2c_dir: str | Path, *, require_manifest: bool) -> dict[str, Any]:
    """来源（w2c_data）的指纹：各分片的文件名、条数、sha256 与 manifest 的 sha256（不含路径）。"""
    d = Path(w2c_dir)
    man = d / W2C_MANIFEST
    if require_manifest and not man.is_file():
        raise X2DataError(f"{d} 里没有 {W2C_MANIFEST}（DEV-0060 的去重与抽样记录）")
    shards = []
    for p in input_shards(d):
        raw = p.read_bytes()
        shards.append({"file": p.name, "n": sum(1 for ln in raw.decode("utf-8").splitlines() if ln.strip()),
                       "sha256": hashlib.sha256(raw).hexdigest()})
    return {"format": "docs/w2_plan.md §5（DEV-0060 的 w2c_data）", "shards": shards,
            "manifest_sha256": _sha(man) if man.is_file() else None}


def out_name(kind: str, shard: str) -> str:
    return f"{kind}_{Path(shard).stem}.jsonl"


def read_jsonl(p: Path) -> list[dict[str, Any]]:
    return [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _line(doc: Mapping[str, Any]) -> str:
    return json.dumps(doc, ensure_ascii=False) + "\n"


def _write_pair(d: Path, shard: str, lines: Mapping[str, Sequence[str]]) -> None:
    tmps = {}
    for kind in KINDS:
        tmp = d / (out_name(kind, shard) + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write("".join(lines[kind]))
            f.flush()
            os.fsync(f.fileno())
        tmps[kind] = tmp
    for kind in KINDS:  # 两个都改名之后这一片才算写完（shard_done 要两个都在）
        os.replace(tmps[kind], d / out_name(kind, shard))


def shard_done(d: Path, shard: str) -> bool:
    return all((d / out_name(kind, shard)).is_file() for kind in KINDS)


def _check_done(d: Path, shard: str, ids: Sequence[str]) -> None:
    pos = {i: k for k, i in enumerate(ids)}
    for kind in KINDS:
        rows = read_jsonl(d / out_name(kind, shard))
        got = [r["id"] if kind == "spline" else r["sample_id"] for r in rows]
        if any(g not in pos for g in got) or [pos[g] for g in got] != sorted(pos[g] for g in got):
            raise X2DataError(f"{d / out_name(kind, shard)}：已写完的片与输入分片 {shard} 的句子对不上，拒绝续跑")


# ---------------------------------------------------------------------------
# 生成
# ---------------------------------------------------------------------------


def build_identity(params: Mapping[str, Any], source: Mapping[str, Any], *, tiny: bool) -> dict[str, Any]:
    return {"experiment": "X-2", "framework": FRAMEWORK, "schema": SCHEMA, "tiny": bool(tiny), "params": json.loads(json.dumps(params)),
            "tools": tool_constants(), "source": dict(source)}


def _dump_yaml(doc: Mapping[str, Any]) -> str:
    import yaml

    return yaml.safe_dump(json.loads(json.dumps(doc)), allow_unicode=True, sort_keys=False)


def _prepare(out: Path, ident: Mapping[str, Any], *, resume: bool) -> None:
    cfg = out / CONFIG
    if cfg.exists():
        import yaml

        old = yaml.safe_load(cfg.read_text(encoding="utf-8"))
        if old != json.loads(json.dumps(ident)):
            raise X2DataError(f"{out}：已有的 {CONFIG} 与本次的参数或输入不同，拒绝混写；换一个 --out")
        if not resume:
            raise X2DataError(f"{out} 已存在；续跑请加 --resume")
    else:
        if out.exists() and any(out.iterdir()):
            raise X2DataError(f"{out} 非空、却没有 {CONFIG}，拒绝写入")
        out.mkdir(parents=True, exist_ok=True)
        (out / CONFIG).write_text(_dump_yaml(ident), encoding="utf-8")
    (out / DATA_DIR).mkdir(exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)


def build(w2c_dir: str | Path, out: str | Path, *, data_root: str | Path, params: Mapping[str, Any], tiny: bool = False,
          workers: int = 1, resume: bool = False, stop_after_shards: int | None = None, argv: Sequence[str] = (),
          log: Callable[[str], None] | None = None) -> dict[str, Any] | None:
    """生成 X-2 的数据（规则见模块说明）。返回 manifest（stop_after_shards 到了、还没写完时返回 None）。"""
    log = log or (lambda m: print(m, file=sys.stderr, flush=True))
    out = Path(out)
    params = check_params(params)
    source = source_identity(w2c_dir, require_manifest=not tiny)
    ident = build_identity(params, source, tiny=tiny)
    _prepare(out, ident, resume=resume)
    d = out / DATA_DIR
    shards = input_shards(w2c_dir)
    names = {out_name(k, s.name) for s in shards for k in KINDS} | {MANIFEST}
    extra = sorted(p.name for p in d.iterdir() if p.name not in names and not p.name.endswith(".tmp"))
    if extra:
        raise X2DataError(f"{d} 里有不属于这份输入的文件：{extra[:5]}")
    runs = sorted((out / "logs").glob("run_*.json"))
    run_k = len(runs) + 1
    run_path = out / "logs" / f"run_{run_k:03d}.json"
    run_log = {"argv": list(argv), "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "workers": workers,
               "shards_total": len(shards), "finished_utc": None}
    todo = []
    for s in shards:
        rows = read_jsonl(s)
        if shard_done(d, s.name):
            _check_done(d, s.name, [r["id"] for r in rows])
            continue
        todo.append((s, rows))
    run_log["shards_todo"] = [s.name for s, _ in todo]
    if stop_after_shards is not None:
        todo = todo[: max(int(stop_after_shards), 0)]
    run_log["shards_this_run"] = [s.name for s, _ in todo]
    run_path.write_text(json.dumps(run_log, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    t_all = time.perf_counter()
    with ToolPool(workers) as pool:
        for s, rows in todo:
            bad = [(r.get("id"), p) for r in rows for p in input_problems(r)]
            if bad:
                raise X2DataError(f"{s.name}：{len(bad)} 处输入不合格，例如 {bad[:3]}")
            for t in d.glob(out_name("*", s.name) + ".tmp"):
                t.unlink()
            t0 = time.perf_counter()
            results = pool.map(sentence_job, [(r, str(data_root), params) for r in rows], chunksize=4)
            lines: dict[str, list[str]] = {k: [] for k in KINDS}
            tl = []
            for r, (res, wall) in zip(rows, results):
                if res["id"] != r["id"]:
                    raise RuntimeError(f"返回的顺序不对：{res['id']} ≠ {r['id']}")
                if res["spline"] is not None:
                    lines["spline"].append(_line(res["spline"]))
                lines["correct"].extend(_line(c) for c in res["correct"])
                tl.append(_line({"id": r["id"], "run": run_k, "shard": s.name, "wall_time_s": wall, "spline_skip": res["spline_skip"],
                                 "correct_skip": res["correct_skip"], "round2_skip": res["round2_skip"]}))
            _write_pair(d, s.name, lines)
            with open(out / TIMING, "a", encoding="utf-8") as f:
                f.write("".join(tl))
            log(f"[X-2 数据] {s.name}：{len(rows)} 句 → 样条 {len(lines['spline'])} 行、修正 {len(lines['correct'])} 行，"
                f"{time.perf_counter() - t0:.1f} s")
    complete = all(shard_done(d, s.name) for s in shards)
    run_log.update(finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                   wall_s=round(time.perf_counter() - t_all, 3), complete=complete)
    run_path.write_text(json.dumps(run_log, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    if not complete:
        return None
    man = finalize(out, w2c_dir, params, source)
    return man


def _dist(values: Sequence[float]) -> dict[str, Any] | None:
    v = [float(x) for x in values if x is not None]
    if not v:
        return None
    a = np.asarray(v, dtype=np.float64)
    q = np.percentile(a, [5, 25, 50, 75, 95])
    return {"n": len(v), "mean": float(a.mean()), "min": float(a.min()), "p05": float(q[0]), "p25": float(q[1]), "p50": float(q[2]),
            "p75": float(q[3]), "p95": float(q[4]), "max": float(a.max())}


def _counts(values) -> dict[str, int]:
    c = Counter(str(v) for v in values)
    return {k: c[k] for k in sorted(c)}


def _skip_reasons(w2c_dir: str | Path, out: Path, params: Mapping[str, Any]) -> dict[str, Any]:
    """过滤的计数：由输入行与输出行重算（不读 timing），所以逐字节可复现。"""
    n_rows = 0
    sp_ids: set[str] = set()
    co_r1: set[str] = set()
    co_r2: set[str] = set()
    d = out / DATA_DIR
    for s in input_shards(w2c_dir):
        sp_ids |= {r["id"] for r in read_jsonl(d / out_name("spline", s.name))}
        for r in read_jsonl(d / out_name("correct", s.name)):
            (co_r1 if r["round"] == 1 else co_r2).add(r["sample_id"])
    reasons_sp, reasons_co, reasons_r2 = Counter(), Counter(), Counter()
    for s in input_shards(w2c_dir):
        for r in read_jsonl(s):
            n_rows += 1
            box = (r["proposal"] or {}).get("box")
            gt_empty = not rle_to_mask(r["gt_rle"]).any()
            gb = r["gt_box"]
            biou = float(box_iou(box, gb)) if (box is not None and gb[2] > gb[0] and gb[3] > gb[1]) else 0.0
            pre = "proposal_failed" if box is None else ("gt_empty" if gt_empty else None)
            if r["id"] not in sp_ids:
                reasons_sp[pre or ("box_iou_low" if biou < params["spline"]["min_box_iou"] else "fit_failed")] += 1
            if r["id"] not in co_r1:
                reasons_co[pre or "box_iou_low"] += 1
            elif r["id"] not in co_r2:
                reasons_r2["not_drawn" if not wants_round2(r["id"], params["correct"]["round2_salt"], params["correct"]["round2_ratio"])
                           else "round1_empty"] += 1
    return {"n_input_rows": n_rows, "spline_skipped": dict(sorted(reasons_sp.items())), "correct_skipped": dict(sorted(reasons_co.items())),
            "round2_not_made": dict(sorted(reasons_r2.items()))}


def finalize(out: Path, w2c_dir: str | Path, params: Mapping[str, Any], source: Mapping[str, Any]) -> dict[str, Any]:
    """manifest.json：只由输入与输出的行算（不含耗时与路径），逐字节可复现。"""
    d = out / DATA_DIR
    files, spl, cor = [], [], []
    for s in input_shards(w2c_dir):
        for kind in KINDS:
            p = d / out_name(kind, s.name)
            rows = read_jsonl(p)
            files.append({"file": p.name, "kind": kind, "n": len(rows), "sha256": _sha(p)})
            (spl if kind == "spline" else cor).extend(rows)
    r1 = [r for r in cor if r["round"] == 1]
    r2 = [r for r in cor if r["round"] == 2]

    def empty_rate(rs):
        return {"n": len(rs), "n_empty": sum(1 for r in rs if not r["target"]),
                "rate": (sum(1 for r in rs if not r["target"]) / len(rs)) if rs else None}

    man = {
        "schema": SCHEMA, "framework": FRAMEWORK, "params": params, "tools": tool_constants(), "source": dict(source),
        "filters": _skip_reasons(w2c_dir, out, params),
        "counts": {"spline": len(spl), "correct": {"round1": len(r1), "round2": len(r2), "total": len(cor)},
                   "spline_by_source": _counts(r["source"] for r in spl), "correct_by_source": _counts(r["source"] for r in cor),
                   "correct_by_tier": {f"round{k}": _counts(r["start"]["tier"] for r in rs) for k, rs in ((1, r1), (2, r2))}},
        "files": files,
        "spline": {"fit_iou": _dist([r["fit_iou"] for r in spl]), "box_iou": _dist([r["box_iou"] for r in spl]),
                   "fit_iou_lt_0.8": sum(1 for r in spl if r["fit_iou"] < 0.8)},
        "correct": {"target_empty": {"round1": empty_rate(r1), "round2": empty_rate(r2), "all": empty_rate(cor)},
                    "n_boxes": {f"round{k}": _counts(len(r["target"]) for r in rs) for k, rs in ((1, r1), (2, r2))},
                    "signs": _counts(t["sign"] for r in cor for t in r["target"]),
                    "start_iou": {f"round{k}": _dist([r["start"]["start_iou"] for r in rs]) for k, rs in ((1, r1), (2, r2))},
                    "start_iou_by_tier": {t: _dist([r["start"]["start_iou"] for r in r1 if r["start"]["tier"] == t]) for t in TIERS},
                    "n_gated": int(sum(r["n_gated"] for r in cor)), "n_outside": int(sum(r["n_outside"] for r in cor))},
    }
    (d / MANIFEST).write_text(json.dumps(man, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return man


# ---------------------------------------------------------------------------
# 读取（训练用）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class X2Data:
    """X-2 的数据：rows（读进来的类的行：spline 与 correct，或只有 spline（DEV-0070），按 manifest 的文件顺序）；pools：kind → rows 的下标
    （键 = 读进来的类）。"""

    run_dir: Path
    rows: tuple[Mapping[str, Any], ...]
    pools: Mapping[str, tuple[int, ...]]
    manifest_sha256: str
    manifest: Mapping[str, Any]

    def pool(self, kind: str) -> tuple[int, ...]:
        return self.pools[kind]

    @property
    def kinds(self) -> tuple[str, ...]:
        return tuple(self.pools)

    @property
    def params(self) -> dict[str, Any]:
        """读进来的类的数据参数（manifest 的 params 里的这几节；两类都读时就是 manifest 的 params，load_x2_data 核对过）。"""
        return {k: v for k, v in self.manifest["params"].items() if k in self.pools}

    def file_paths(self) -> list[Path]:
        """读进来的类的数据文件与 manifest（两类都读时 = manifest 的全部文件）。"""
        d = self.run_dir / DATA_DIR
        return [d / f["file"] for f in self.manifest["files"] if f["kind"] in self.pools] + [d / MANIFEST]

    def identity(self) -> dict[str, Any]:
        """运行身份里的 X-2 数据一节：manifest 的 sha256、参数（读进来的类的）、来源、池的条数与 id 指纹；不含路径。"""
        return {"format": FRAMEWORK, "manifest_sha256": self.manifest_sha256, "params": self.params,
                "tools": dict(self.manifest["tools"]), "source_manifest_sha256": self.manifest["source"]["manifest_sha256"],
                "pools": {k: {"n": len(p), "ids_sha256": hashlib.sha256("\n".join(str(self.rows[i]["id"]) for i in p).encode("utf-8"))
                              .hexdigest()} for k, p in self.pools.items()},
                "counts": dict(self.manifest["counts"])}


def row_problems(r: Mapping[str, Any], kind: str, params: Mapping[str, Any]) -> list[str]:
    """一行的问题（空 = 合格）：目标文本等于重新写出的、能严格解析回同样的目标、参数一致。"""
    bad: list[str] = []
    try:
        w, h = r["image_size"]
        if r["box_q"] != XP.box_q(r["box"], w, h):
            bad.append("box_q 与 box 对不上")
        if kind == "spline":
            n = params["spline"]["n"]
            if r["n"] != n or r["target_text"] != XP.spline_text(r["ctrl_q"]) or XP.parse_spline(r["target_text"], n=n).value != r["ctrl_q"]:
                bad.append("样条的目标与 n、文本或解析对不上")
        else:
            m = params["correct"]["m"]
            p = XP.parse_corrections(r["target_text"], m=m)
            if r["m"] != m or r["target_text"] != XP.corrections_text(r["target"]) or p.failure is not None or p.value != r["target"]:
                bad.append("修正的目标与 m、文本或解析对不上")
            if list(r["start_rle"]["size"]) != [h, w]:
                bad.append("start_rle 的尺寸不是 [H, W]")
    except (KeyError, TypeError, ValueError) as e:
        bad.append(f"结构不对：{type(e).__name__}: {e}")
    return bad


def load_x2_data(run_dir: str | Path, *, params: Mapping[str, Any], source: Mapping[str, Any] | None,
                 data_root: str | Path | None = None, kinds: Sequence[str] | None = None) -> X2Data:
    """读 build 的输出并核对（模块说明）。params：训练配置里的数据参数（check_params 之后）；source：训练读的 w2c_data 的指纹
    （source_identity；None = 不核对，只给 --tiny 的彩排）；data_root 给了就核对图像在。
    kinds（DEV-0070）：读哪几类（缺省 = params 里有的类）。不是两类时只读、只核对这几类的文件与行，参数只比这几节。"""
    kinds = check_kinds([k for k in KINDS if k in params] if kinds is None else kinds)
    rd = Path(run_dir)
    d = rd / DATA_DIR
    mp = d / MANIFEST
    if not mp.is_file():
        raise X2DataError(f"{mp} 不在（数据没生成完）")
    man = json.loads(mp.read_text(encoding="utf-8"))
    want = check_params(params, kinds=kinds)
    if kinds == KINDS:
        if man.get("params") != want:
            raise X2DataError(f"X-2 数据的参数 {man.get('params')} 与训练配置的 {want} 不同")
    else:
        got = man.get("params") if isinstance(man.get("params"), Mapping) else {}
        if {k: got.get(k) for k in kinds} != want:
            raise X2DataError(f"X-2 数据的参数 {man.get('params')} 的 {list(kinds)} 与训练配置的 {want} 不同")
    if source is not None and man["source"] != dict(source):
        raise X2DataError("X-2 数据来自另一份 w2c_data（manifest 或分片的 sha256 不同）：与训练读的提出、挑选数据必须是同一份")
    rows: list[Mapping[str, Any]] = []
    pools: dict[str, list[int]] = {k: [] for k in KINDS}
    problems: list[str] = []
    seen: set[str] = set()
    for f in man["files"]:
        if f["kind"] in KINDS and f["kind"] not in kinds:  # DEV-0070：没选的类不读、不核对
            continue
        p = d / f["file"]
        if not p.is_file() or _sha(p) != f["sha256"]:
            raise X2DataError(f"{p}：文件不在或 sha256 与 manifest 不符")
        for r in read_jsonl(p):
            kind = f["kind"]
            if r.get("kind") != kind or r["id"] in seen:
                problems.append(f"{f['file']}（{r.get('id')}）：kind 不对或 id 重复")
                continue
            bad = row_problems(r, kind, want)
            if not bad and data_root is not None and not (Path(data_root) / r["image"]).is_file():
                bad.append(f"图像不在：{Path(data_root) / r['image']}")
            if bad:
                problems.append(f"{f['file']}（{r['id']}）：{'；'.join(bad)}")
                continue
            seen.add(r["id"])
            pools[kind].append(len(rows))
            rows.append(r)
    if problems:
        raise X2DataError(f"X-2 的数据有 {len(problems)} 行不合格：\n- " + "\n- ".join(problems[:10]))
    for k in kinds:
        if not pools[k]:
            raise X2DataError(f"{rd}：{k} 的池是空的")
    return X2Data(run_dir=rd, rows=tuple(rows), pools={k: tuple(pools[k]) for k in kinds}, manifest_sha256=_sha(mp), manifest=man)


__all__ = [
    "DATA_DIR", "DEFAULT_PARAMS", "KINDS", "MANIFEST", "TIERS", "X2Data", "X2DataError", "build", "check_kinds", "check_params",
    "correction_targets", "finalize", "input_problems", "iou_ranking", "load_image", "load_x2_data", "out_name", "row_problems", "sentence_job",
    "source_identity", "spline_row", "start_choice", "tool_constants", "wants_round2",
]
