"""Rex-Omni 外部参照的运行器（DEV-0083；D165 第四节、D177）：一个划分 × 一个设置，只做检测。

**两个设置**（SETTINGS；进运行身份，同一次运行可以都跑）：
- rexomni_det_joint（query_mode joint，**标题读数**，Rex-Omni 自己的协议）：每张图调用一次，提示里用 ", " 列出这张图在本划分里的全部
  正类别（category_id 升序；lvis_sub 是 DEV-0082 每图至多 3 类之后的那几类，即本划分里这张图的对），输出按类名分给各个对
  （protocol.select_joint），名字不在所问类别里的框丢掉并计数；
- rexomni_det（query_mode per_category，诊断，与 X-M 同协议）：每个（图, 类别）对调用一次，提示里只有这一个类别。

**输出**（两个设置的叶目录格式相同，都是每对一行，可用同一套 src/xm/metrics.table / paired_bootstrap 与 X-M 配对比较）：
    <out>/rex-omni/xm/<设置>/<划分>/predictions.jsonl   每对一行：id、image_key、category_id、det（src/xm/metrics.eval_pair）、seg = null、
                                                     原始输出、解析计数、框（原图像素与网格整数）、这次调用的 token 数与 call（joint 时
                                                     同一张图的各对共用一次调用：index、n_pairs、first）
    <out>/rex-omni/xm/<设置>/<划分>/timing.jsonl        每对的墙钟（本批墙钟 / 调用数 / 本次调用的对数；不进 predictions）
    <out>/rex-omni/xm/<设置>/<划分>/metrics.json、config.yaml（运行身份）、meta.json（src/runlog）、README.md、logs/run_NNN.json
    <out>/rex-omni/xm_gate_unlock.jsonl              xm_final、lvis_sub 的放行记录（src/xm/gate.UNLOCK_LOG）

**流程**：对按划分的顺序（src/xm/data.load_pairs：image_id、category_id 升序）分成调用单位（joint：一张图的全部对，必须在输入里相邻；
per_category：一个对），每 batch_size 个单位一批：读图（RGB，尺寸必须等于对的 width、height）→ 官方提示 → 生成
（src/rexomni/model.Engine）→ 官方解析（src/rexomni/protocol）→ 各对类别下的框，输出顺序、不去重、最多 100 个 → eval_pair（框 IoU；
crowd 区域上的预测忽略；LVIS 的非穷尽类别没配上的预测忽略）。predictions.jsonl 的行序就是输入的对序。每批写完 predictions.jsonl 与
timing.jsonl 再进下一批；续跑（resume）只在 config.yaml 的身份完全相同时进行，已完成的行退回到批的边界。

**冒烟线**（管理员，DEV-0083 审阅意见第 2 条）：xm_tune 上 joint 的 f1_rexomni_exact（复现官方计分代码的那一列）低于 SMOKE_THRESHOLD
（0.60）就先停，不跑 xm_final。metrics.json 的 smoke_line 记 {statistic, threshold, value, pass}；scripts/run_rexomni.py 不合格时以退出码
3 结束（--tiny 只记不拦）。

**记账**：解析失败（没有任何 <|object_ref_start|>…<|box_start|> 组）、截断（到生成上限还没停）、停止串、未问类名的框、每次调用的生成
token 数与墙钟都单列进 metrics.json 的 summary。

image_key 与非穷尽类别的判定照 src/xm/runner.image_key / not_exhaustive（那个模块 import X-M 的提示与后处理，这里不 import；
tests/test_rexomni.py 核对两者逐个相同）。
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from src.rexomni import protocol as P
from src.xm import data as XD
from src.xm import metrics as XMM

REPO_ROOT = Path(__file__).resolve().parents[2]
FRAMEWORK = "src/rexomni/runner.py（DEV-0083；Rex-Omni 外部参照）"
EXPERIMENT = "X-M-ext-RexOmni"
APPROVAL = "D165 第四节（外部参照，仅检测）；D177-c；CLAUDE.md 硬约束第 1 条的单独运行的对照系统"
SYSTEM = "rex-omni"  # 叶目录里"模型"那一级
TASK = "xm"
SETTING_JOINT, SETTING_PER_CATEGORY = "rexomni_det_joint", "rexomni_det"
SETTINGS: Mapping[str, Mapping[str, str]] = {
    SETTING_JOINT: {"query_mode": P.QUERY_JOINT,
                    "role": "标题读数：Rex-Omni 自己的协议（每图一次，列出该图本划分里的全部正类别，category_id 升序）"},
    SETTING_PER_CATEGORY: {"query_mode": P.QUERY_PER_CATEGORY, "role": "诊断：与 X-M 同协议（每个（图, 类别）对一次，只问这一个类别）"},
}
SPLITS: tuple[str, ...] = (XD.XM_TUNE, XD.XM_FINAL, XD.LVIS_SUB)
REFUSED_SPLITS: Mapping[str, str] = {
    XD.XM_DECISION: "xm_decision 是密封的判定集，G-XM 已经判完（RUN-0108、D177），Rex-Omni 不跑",
    XD.REASONSEG_VAL: "reasonseg_val 是指代分割；Rex-Omni 外部参照只做检测（D165 第四节）",
    XD.TRAIN: "train 不评测",
}
PREDICTIONS, TIMING, METRICS, CONFIG, META, README = "predictions.jsonl", "timing.jsonl", "metrics.json", "config.yaml", "meta.json", "README.md"
RUN_DIR_TAG = "_rexomni"  # --out 的末级目录名必须以四位运行编号开头且含它（隔离扫描靠它认出结果目录）
NOT_EXHAUSTIVE_KEY = "not_exhaustive"  # 同 src/xm/runner.NOT_EXHAUSTIVE_KEY（DEV-0079 的 XMPair.meta）
FAIL_NO_GROUP = "no_group"  # 有输出，但没有任何 <|object_ref_start|>…<|box_start|> 组
FAIL_EMPTY = "empty_output"  # 输出为空（只有 <|im_end|>，或停止串之前什么都没有）
SMOKE_SPLIT, SMOKE_SETTING, SMOKE_STATISTIC = XD.XM_TUNE, SETTING_JOINT, "f1_rexomni_exact"
SMOKE_THRESHOLD = 0.60  # 管理员（DEV-0083 审阅意见第 2 条）：低于它就先停，不跑 xm_final


class RexOmniRunError(ValueError):
    """运行器的输入不合格。"""


def check_run_dir(out: str | Path) -> None:
    name = Path(out).name
    if not XD.RUN_DIR_RE.fullmatch(name) or RUN_DIR_TAG not in name:
        raise RexOmniRunError(f"--out 的末级目录名必须以四位运行编号开头且含 {RUN_DIR_TAG}（形如 0110_rexomni_tune），收到 {name!r}")


def check_split(split: str) -> None:
    if split in REFUSED_SPLITS:
        raise RexOmniRunError(REFUSED_SPLITS[split])
    if split not in SPLITS:
        raise RexOmniRunError(f"--split 只能是 {list(SPLITS)}，收到 {split!r}")


def check_setting(setting: str) -> str:
    if setting not in SETTINGS:
        raise RexOmniRunError(f"设置只能是 {list(SETTINGS)}，收到 {setting!r}")
    return SETTINGS[setting]["query_mode"]


def leaf_dir(out: str | Path, split: str, setting: str) -> Path:
    return Path(out) / SYSTEM / TASK / setting / split


def image_key(pair: XD.XMPair) -> str:
    """同 src/xm/runner.image_key：<source>:<相对路径>（按图像重抽与配对的键）。"""
    return f"{pair.source}:{pair.image}"


def not_exhaustive(pair: XD.XMPair) -> bool:
    """同 src/xm/runner.not_exhaustive：LVIS 的非穷尽类别（没配上的预测不记 FP）。"""
    meta = getattr(pair, "meta", None) or {}
    if meta.get(NOT_EXHAUSTIVE_KEY) is not None:
        return bool(meta[NOT_EXHAUSTIVE_KEY])
    return pair.category_id in {int(c) for c in meta.get("not_exhaustive_category_ids") or ()}


def pairs_digest(pairs: Sequence[XD.XMPair]) -> str:
    return hashlib.sha256("\n".join(p.id for p in pairs).encode("utf-8")).hexdigest()


def det_eval(pair: XD.XMPair, boxes: Sequence[Sequence[float]]) -> dict[str, Any]:
    """同 src/xm/runner.det_eval：框 IoU（crowd 区域取外接框）→ eval_pair（输出顺序，至多 100 个）。"""
    ious, crowd = XMM.box_ious(boxes, [i.box_xyxy for i in pair.instances], pair.crowd_rles)
    return XMM.eval_pair(ious, crowd, len(pair.instances), not_exhaustive=not_exhaustive(pair))


def load_image(path: Path, pair: XD.XMPair) -> Any:
    from PIL import Image

    with Image.open(path) as im:
        img = im.convert(P.IMAGE_MODE)  # inference_text_prompt.py:592
    if img.size != (pair.width, pair.height):
        raise RexOmniRunError(f"{pair.id}：图像 {path} 的尺寸 {img.size} 与对的 (width, height) = {(pair.width, pair.height)} 不符")
    return img


# ---------------------------------------------------------------------------
# 调用单位
# ---------------------------------------------------------------------------


def call_units(pairs: Sequence[XD.XMPair], query_mode: str) -> list[tuple[list[XD.XMPair], list[str]]]:
    """[(这次调用的对（输入顺序）, 提示里的类别（按顺序）)]。per_category：每对一个单位；joint：一张图的全部对一个单位，类别按
    category_id 升序（protocol.JOINT_ORDER）。joint 要求同一张图的对在输入里相邻（load_pairs 的顺序就是这样）。"""
    if query_mode == P.QUERY_PER_CATEGORY:
        return [([p], [p.category]) for p in pairs]
    if query_mode != P.QUERY_JOINT:
        raise RexOmniRunError(f"未知的 query_mode {query_mode!r}")
    units: list[list[XD.XMPair]] = []
    seen: set[str] = set()
    for p in pairs:
        k = image_key(p)
        if units and image_key(units[-1][0]) == k:
            units[-1].append(p)
            continue
        if k in seen:
            raise RexOmniRunError(f"joint：同一张图的对在输入里不相邻（{k}）")
        seen.add(k)
        units.append([p])
    out = []
    for u in units:
        ids = [p.category_id for p in u]
        if len(set(ids)) != len(ids):
            raise RexOmniRunError(f"joint：图 {image_key(u[0])} 里同一个类别有两个对 {ids}")
        out.append((u, [p.category for p in sorted(u, key=lambda q: q.category_id)]))
    return out


# ---------------------------------------------------------------------------
# 一对的记录
# ---------------------------------------------------------------------------


def unit_records(unit: Sequence[XD.XMPair], categories: Sequence[str], gen: Mapping[str, Any], *, setting: str,
                 call_index: int) -> list[dict[str, Any]]:
    """一次调用的各对在 predictions.jsonl 里的行（输入顺序；不含墙钟）。gen：Engine.generate 的一条。"""
    query_mode = check_setting(setting)
    text = P.truncate_at_stop(gen["raw_text"])[0]
    extracted, pinfo = P.parse_prediction(text)
    w, h = unit[0].width, unit[0].height
    if query_mode == P.QUERY_JOINT:
        sel, n_unasked = P.select_joint(extracted, categories, w, h, max_dets=XMM.MAX_DETS)
    else:
        sel, n_unasked = {unit[0].category: P.select_boxes(extracted, unit[0].category, w, h, max_dets=XMM.MAX_DETS)}, None
    failure = None if pinfo["n_groups"] else (FAIL_EMPTY if not text.split(P.EOS_TEXT)[0].strip() else FAIL_NO_GROUP)
    tokens = {"prompt": int(gen["n_prompt_tokens"]), "generated": int(gen["n_generated_tokens"]), "visual": int(gen["n_visual_tokens"]),
              "max_new_tokens": int(gen["max_new_tokens"])}
    rows = []
    for k, pair in enumerate(unit):
        boxes, bins, sinfo = sel[pair.category]
        if query_mode == P.QUERY_JOINT:
            sinfo = {key: v for key, v in sinfo.items() if key != "n_other_category"}  # joint：别的所问类别的框属于别的对
        rows.append({"id": pair.id, "image_key": image_key(pair), "source": pair.source, "split": pair.split, "setting": setting,
                     "query_mode": query_mode, "task": TASK, "category_id": pair.category_id, "category": pair.category,
                     "image_size": [pair.width, pair.height], "n_gt": len(pair.instances), "n_crowd": len(pair.crowd_rles),
                     "not_exhaustive": not_exhaustive(pair), "prompt": gen["prompt"], "asked_categories": list(categories),
                     "raw": gen["raw_text"], "stop_reason": gen["stop_reason"], "stop_string": gen["stop_string"],
                     "truncated": gen["stop_reason"] == "max_new_tokens", "failure": failure,
                     "parse": {**pinfo, **sinfo, "n_unasked_category": n_unasked},
                     "instances": [{"box": b, "bins": q} for b, q in zip(boxes, bins)], "det": det_eval(pair, boxes), "seg": None,
                     "tokens": dict(tokens), "call": {"index": call_index, "n_pairs": len(unit), "first": k == 0},
                     "image_grid_thw": list(gen["image_grid_thw"]), "n_model_calls": 1})
    return rows


def pair_record(pair: XD.XMPair, gen: Mapping[str, Any]) -> dict[str, Any]:
    """per_category 的一行（unit_records 的特例；测试与单对调试用）。"""
    return unit_records([pair], [pair.category], gen, setting=SETTING_PER_CATEGORY, call_index=0)[0]


# ---------------------------------------------------------------------------
# 叶目录、身份与续跑
# ---------------------------------------------------------------------------


def run_identity(*, split: str, setting: str, pairs: Sequence[XD.XMPair], n_calls: int, engine_info: Mapping[str, Any], batch_size: int,
                 limit: int | None, xm_gate: Mapping[str, Any] | None, files: Mapping[str, str], tiny: bool) -> dict[str, Any]:
    query_mode = check_setting(setting)
    src = Counter(p.source for p in pairs)
    return {"experiment": EXPERIMENT, "framework": FRAMEWORK, "approval": APPROVAL, "system": SYSTEM, "task": TASK, "setting": setting,
            "query_mode": query_mode, "role": SETTINGS[setting]["role"], "split": split, "tiny": bool(tiny),
            "data": {"n_pairs": len(pairs), "pair_ids_sha256": pairs_digest(pairs), "n_images": len({image_key(p) for p in pairs}),
                     "n_instances": int(sum(len(p.instances) for p in pairs)), "n_calls": int(n_calls),
                     "sources": {k: src[k] for k in sorted(src)}, "limit": limit, "files_sha256": dict(sorted(files.items()))},
            "xm_gate": None if xm_gate is None else dict(xm_gate),
            "model": json.loads(json.dumps(dict(engine_info), ensure_ascii=False, default=str)),
            "protocol": P.identity(query_mode), "metrics": XMM.identity(),
            "smoke_line": {"split": SMOKE_SPLIT, "setting": SMOKE_SETTING, "statistic": SMOKE_STATISTIC, "threshold": SMOKE_THRESHOLD},
            "batching": {"batch_size": int(batch_size), "unit": "一张图" if query_mode == P.QUERY_JOINT else "一个对",
                         "order": "划分的对序（load_pairs：image_id、category_id 升序），每 batch_size 个调用单位一批"}}


def _norm(doc: Any) -> Any:
    return json.loads(json.dumps(doc, ensure_ascii=False, default=str))


def prepare_leaf(leaf: Path, ident: Mapping[str, Any], data_files: Sequence[str | Path], *, resume: bool) -> None:
    """新目录：写 config.yaml 与 meta.json（src/runlog.write_run_metadata）；已有内容：必须 resume 且 config.yaml 与本次身份完全相同。"""
    import yaml

    from src.runlog import write_run_metadata

    if leaf.exists() and any(leaf.iterdir()):
        if not resume:
            raise FileExistsError(f"{leaf} 已存在且非空；续跑请加 --resume")
        if not (leaf / CONFIG).is_file():
            raise FileExistsError(f"{leaf} 里没有 {CONFIG}，不能续跑")
        old = _norm(yaml.safe_load((leaf / CONFIG).read_text(encoding="utf-8")))
        new = _norm(ident)
        if old != new:
            diff = sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k))
            raise RexOmniRunError(f"{leaf}：续跑的身份与已有的不同（{diff}），拒绝混写；换一个 --out")
    else:
        leaf.mkdir(parents=True, exist_ok=True)
        write_run_metadata(leaf, config=_norm(ident), seed=int(XMM.BOOTSTRAP_SEED), data_files=list(data_files), repo_root=REPO_ROOT)
    (leaf / "logs").mkdir(exist_ok=True)


def read_jsonl_prefix(path: Path, *, repair: bool) -> list[dict[str, Any]]:
    """已完成的行（末尾不完整的一行丢掉；repair 时把文件截掉）。"""
    if not path.exists():
        return []
    raw = path.read_bytes()
    keep = raw.rfind(b"\n") + 1
    if repair and keep != len(raw):
        with open(path, "r+b") as f:
            f.truncate(keep)
    return [json.loads(x) for x in raw[:keep].decode("utf-8").splitlines() if x.strip()]


def append_jsonl(path: Path, recs: Sequence[Mapping[str, Any]]) -> None:
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs)
    if not text:
        return
    with open(path, "a", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())


def _rewrite_jsonl(path: Path, recs: Sequence[Mapping[str, Any]]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs), encoding="utf-8")


def batch_bounds(units: Sequence[tuple[Sequence[Any], Any]], batch_size: int) -> list[tuple[int, int, int]]:
    """[(第一个单位, 最后一个单位 + 1, 这一批结束时累计的行数)]。"""
    out, rows = [], 0
    for a in range(0, len(units), batch_size):
        b = min(a + batch_size, len(units))
        rows += sum(len(units[i][0]) for i in range(a, b))
        out.append((a, b, rows))
    return out


def resume_point(leaf: Path, ids: Sequence[str], bounds: Sequence[tuple[int, int, int]], log: Callable[[str], None]
                 ) -> tuple[int, int, dict[str, Any] | None]:
    """已完成的行必须是对序的前缀；退回到批的边界（整批重跑），timing.jsonl 只留已完成的对。返回 (保留的行数, 从第几批起跑, 截断记录)。"""
    done = read_jsonl_prefix(leaf / PREDICTIONS, repair=True)
    got = [r["id"] for r in done]
    if got != list(ids[: len(got)]):
        raise RexOmniRunError(f"{leaf / PREDICTIONS} 里已完成的行不是对序的前缀，不能续跑")
    n_batches = 0
    keep = 0
    for k, (_, _, rows) in enumerate(bounds):
        if rows <= len(done):
            n_batches, keep = k + 1, rows
    trimmed = None
    if keep < len(done):
        _rewrite_jsonl(leaf / PREDICTIONS, done[:keep])
        trimmed = {"from": len(done), "to": keep}
        log(f"[Rex-Omni] 已完成 {len(done)} 行不在批的边界上，退回到 {keep} 行整批重跑")
    kept = set(ids[:keep])
    tm = [t for t in read_jsonl_prefix(leaf / TIMING, repair=True) if t.get("id") in kept]
    _rewrite_jsonl(leaf / TIMING, tm)
    return keep, n_batches, trimmed


# ---------------------------------------------------------------------------
# 读数
# ---------------------------------------------------------------------------


def distribution(values: Sequence[float]) -> dict[str, Any]:
    v = np.asarray(list(values), np.float64)
    if v.size == 0:
        return {"n": 0}
    return {"n": int(v.size), "mean": float(v.mean()), "min": float(v.min()), "p50": float(np.percentile(v, 50)),
            "p90": float(np.percentile(v, 90)), "max": float(v.max()), "sum": float(v.sum())}


def _cnt(vals: Sequence[Any]) -> dict[str, int]:
    c = Counter("ok" if v is None else str(v) for v in vals)
    return {k: c[k] for k in sorted(c)}


def summarize(rows: Sequence[Mapping[str, Any]], timing: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """按对的计数（框、解析）与按调用的计数（失败、截断、停止、token、墙钟、未问类名的框；joint 时一次调用对应一张图的几对）。"""
    parse_keys = ("n_coord_lists", "n_other_category", "n_not_box", "n_boxes_parsed", "n_capped", "n_out_of_grid", "n_inverted")
    calls = [r for r in rows if r["call"]["first"]]
    return {"n_calls": len(calls), "pairs_per_call": distribution([r["call"]["n_pairs"] for r in calls]),
            "failure": _cnt([r["failure"] for r in calls]), "n_parse_failures": int(sum(1 for r in calls if r["failure"] is not None)),
            "stop_reason": _cnt([r["stop_reason"] for r in calls]), "n_truncated": int(sum(1 for r in calls if r["truncated"])),
            "n_incomplete_group": int(sum(1 for r in calls if r["parse"]["incomplete"])),
            "n_unasked_category": int(sum(r["parse"]["n_unasked_category"] or 0 for r in calls)),
            "parse_counts": {k: int(sum(r["parse"].get(k) or 0 for r in rows)) for k in parse_keys},
            "n_pred_boxes": distribution([len(r["instances"]) for r in rows]),
            "generated_tokens": distribution([r["tokens"]["generated"] for r in calls]),
            "prompt_tokens": distribution([r["tokens"]["prompt"] for r in calls]),
            "visual_tokens": distribution([r["tokens"]["visual"] for r in calls]),
            "call_wall_s": distribution([float(timing[r["id"]]["call_wall_s"]) for r in calls if r["id"] in timing]),
            "pair_wall_s": distribution([float(timing[r["id"]]["wall_s"]) for r in rows if r["id"] in timing]),
            "n_timing_missing": int(sum(1 for r in rows if r["id"] not in timing))}


def smoke_line(metrics: Mapping[str, Any]) -> dict[str, Any] | None:
    """冒烟线（模块说明）：只对 xm_tune 上的 joint。返回 {statistic, threshold, value, pass} 或 None（别的划分与设置）。"""
    if metrics["split"] != SMOKE_SPLIT or metrics["setting"] != SMOKE_SETTING:
        return None
    v = metrics["det"][SMOKE_STATISTIC]["value"]
    return {"split": SMOKE_SPLIT, "setting": SMOKE_SETTING, "statistic": SMOKE_STATISTIC, "threshold": SMOKE_THRESHOLD,
            "value": v, "pass": v is not None and float(v) >= SMOKE_THRESHOLD,
            "rule": f"xm_tune 上 joint 的 {SMOKE_STATISTIC} 低于 {SMOKE_THRESHOLD} 就先停，不跑 xm_final（管理员，DEV-0083 审阅意见第 2 条）"}


def _fmt(v: Any) -> str:
    return "null" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))


def readme(split: str, setting: str, metrics: Mapping[str, Any]) -> str:
    sec = metrics["det"]
    qm = SETTINGS[setting]["query_mode"]
    proto = ("每张图一次调用，提示里列出该图本划分里的全部正类别（category_id 升序；官方 COCO.jsonl 的顺序不是升序，见 config.yaml 的 "
             "protocol.deviations）。这是 Rex-Omni 自己的协议，作论文里外部参照的标题读数；划分是 X-M 的子集，不是论文的完整 val2017。"
             if qm == P.QUERY_JOINT else
             "每个（图像, 类别）一次调用，提示里只有这一个类别。**不是**官方 COCO/LVIS 评测的每图一次多类别提示，读数不能当作 Rex-Omni "
             "论文数字的复现；只作与 X-M 同协议的诊断。")
    lines = [f"# Rex-Omni（外部参照）· {setting} · {split}", "",
             f"- 批准：{APPROVAL}。学习型检测器，单独运行；输出不进入我们的任何数据、标签、奖励或挑选。",
             f"- 设置：{SETTINGS[setting]['role']}（query_mode {qm}）。{proto}",
             f"- 官方代码 {P.UPSTREAM_URL} @ {P.UPSTREAM_COMMIT[:8]}；权重 {P.WEIGHTS_REPO} @ {P.WEIGHTS_REVISION[:8]}（{P.UPSTREAM_LICENSE}）。", "",
             f"## det（框 IoU；{sec['n_pairs']} 个对、{sec['n_images']} 张图、{sec['n_categories']} 类）"]
    for st in (*XMM.STATISTICS, *XMM.EXACT_STATISTICS):
        e = sec[st]
        ci = e.get("ci")
        lines.append(f"- {st}：{_fmt(e['value'])}" + (f"（95% CI [{_fmt(ci['low'])}, {_fmt(ci['high'])}]）" if ci else "")
                     + (f"；P {_fmt(e.get('precision'))}、R {_fmt(e.get('recall'))}" if "precision" in e else ""))
    s = metrics["summary"]
    lines += ["", f"- 调用 {s['n_calls']} 次；解析失败 {s['n_parse_failures']}（{json.dumps(s['failure'], ensure_ascii=False)}）；截断 "
                  f"{s['n_truncated']}；停止 {json.dumps(s['stop_reason'], ensure_ascii=False)}；未问类名的坐标串 {s['n_unasked_category']}",
              f"- 解析计数：{json.dumps(s['parse_counts'], ensure_ascii=False)}",
              f"- 每次调用生成 token：均值 {_fmt(s['generated_tokens'].get('mean'))}、最大 {_fmt(s['generated_tokens'].get('max'))}；"
              f"每次调用墙钟均值 {_fmt(s['call_wall_s'].get('mean'))} s"]
    if metrics.get("smoke_line"):
        sl = metrics["smoke_line"]
        lines.append(f"- 冒烟线：{sl['statistic']} = {_fmt(sl['value'])}，门槛 {sl['threshold']}：{'通过' if sl['pass'] else '不通过，先停，不跑 xm_final'}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# 一个划分 × 一个设置
# ---------------------------------------------------------------------------


def run_split(engine: Any, pairs: Sequence[XD.XMPair], out: str | Path, *, split: str, setting: str, data_root: str | Path,
              batch_size: int = 1, resume: bool = False, limit: int | None = None, xm_gate: Mapping[str, Any] | None = None,
              data_files: Sequence[str | Path] = (), tiny: bool = False, argv: Sequence[str] = (),
              n_resamples: int | None = XMM.BOOTSTRAP_N_RESAMPLES, log: Callable[[str], None] | None = None) -> dict[str, Any]:
    """模块说明的流程。engine：src/rexomni/model.Engine（或同接口的假模型：.info 与 .generate([(图, 类别列表)])）。返回 metrics.json 的内容。"""
    log = log or (lambda m: print(m, file=sys.stderr, flush=True))
    check_split(split)
    query_mode = check_setting(setting)
    check_run_dir(out)
    if batch_size < 1:
        raise RexOmniRunError("batch_size 必须是正整数")
    wrong = sorted({p.split for p in pairs} - {split})
    if wrong:
        raise RexOmniRunError(f"对的 split {wrong} 不是 {split}")
    XD.pair_by_id(pairs)  # id 不能重复
    units = call_units(pairs, query_mode)
    files = {str(f): XD.file_sha256(f) for f in data_files}
    ident = run_identity(split=split, setting=setting, pairs=pairs, n_calls=len(units), engine_info=engine.info, batch_size=batch_size,
                         limit=limit, xm_gate=xm_gate, files=files, tiny=tiny)
    leaf = leaf_dir(out, split, setting)
    prepare_leaf(leaf, ident, data_files, resume=resume)
    ids = [p.id for p in pairs]
    bounds = batch_bounds(units, batch_size)
    keep, start, trimmed = resume_point(leaf, ids, bounds, log)
    run_k = len(list((leaf / "logs").glob("run_*.json"))) + 1
    run_path = leaf / "logs" / f"run_{run_k:03d}.json"
    run_log = {"argv": list(argv), "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "n_done_before": keep,
               "n_remaining": len(ids) - keep, "batch_size": batch_size, "trimmed_to_batch_boundary": trimmed, "finished_utc": None}
    run_path.write_text(json.dumps(run_log, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    root = Path(data_root)
    t_all = time.perf_counter()
    for b in range(start, len(bounds)):
        a, z, n_rows = bounds[b]
        part = units[a:z]
        gens = engine.generate([(load_image(root / u[0].image, u[0]), list(cats)) for u, cats in part])
        if len(gens) != len(part):
            raise RuntimeError(f"generate 返回 {len(gens)} 条，输入 {len(part)} 条")
        recs, tms = [], []
        for i, ((u, cats), g) in enumerate(zip(part, gens)):
            recs += unit_records(u, cats, g, setting=setting, call_index=a + i)
            tms += [{"id": p.id, "run": run_k, "batch_index": b, "call_index": a + i, "wall_s": g["wall_s"] / len(u), "call_wall_s": g["wall_s"],
                     "batch_wall_s": g["batch_wall_s"], "batch_size": g["batch_size"]} for p in u]
        append_jsonl(leaf / TIMING, tms)
        append_jsonl(leaf / PREDICTIONS, recs)
        log(f"[Rex-Omni {setting} {split}] {n_rows}/{len(pairs)}")
    run_log.update(finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), wall_s=time.perf_counter() - t_all)
    run_path.write_text(json.dumps(run_log, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    rows = read_jsonl_prefix(leaf / PREDICTIONS, repair=False)
    if [r["id"] for r in rows] != ids:
        raise RuntimeError(f"{leaf / PREDICTIONS} 的 id 与对不同")
    timing = {t["id"]: t for t in read_jsonl_prefix(leaf / TIMING, repair=False)}
    metrics = {"experiment": EXPERIMENT, "system": SYSTEM, "setting": setting, "query_mode": query_mode, "role": SETTINGS[setting]["role"],
               "split": split, "approval": APPROVAL, "weights": {"repo": P.WEIGHTS_REPO, "revision": P.WEIGHTS_REVISION}, "tiny": bool(tiny),
               "det": XMM.readout(rows, "det", n_resamples=n_resamples), "seg": None,
               "det_note": "检测：框 IoU，src/xm/metrics 的口径（主读数 f1_rexomni，XM-3a）；Rex-Omni 不出 mask，seg 为 null",
               "summary": summarize(rows, timing), "complete": len(rows) == len(pairs), "n": len(rows), "n_expected": len(pairs)}
    metrics["smoke_line"] = smoke_line(metrics)
    (leaf / METRICS).write_text(json.dumps(metrics, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (leaf / README).write_text(readme(split, setting, metrics), encoding="utf-8")
    return metrics


__all__ = [
    "APPROVAL", "EXPERIMENT", "FAIL_EMPTY", "FAIL_NO_GROUP", "PREDICTIONS", "REFUSED_SPLITS", "RUN_DIR_TAG", "RexOmniRunError", "SETTINGS",
    "SETTING_JOINT", "SETTING_PER_CATEGORY", "SMOKE_THRESHOLD", "SPLITS", "SYSTEM", "TASK", "TIMING", "batch_bounds", "call_units",
    "check_run_dir", "check_setting", "check_split", "det_eval", "image_key", "leaf_dir", "not_exhaustive", "pair_record", "run_identity",
    "run_split", "smoke_line", "summarize", "unit_records",
]
