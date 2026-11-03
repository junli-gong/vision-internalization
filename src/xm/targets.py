"""X-M 的训练目标（DEV-0079；D165 指令第二、三节）：COCO train2017（剔除去重清单之后）的每个 XMPair 生成两个文本目标，写成分片
jsonl 与 manifest，供 DEV-0081 的训练器读。只用真值与不含学习的工具（src/explore 的拟合与光栅化，同 X-2），不跑模型。

**框列表**（调用 1 的目标）：该类全部实例的框，0–1000 网格的整数 [x1, y1, x2, y2]（x2_prompts.box_q，同 X-2 的网格），按阅读顺序排：
- reading_order：先按框中心的 y 分行——两个框中心 y 之差小于两者高度较小者的一半，算同一行。具体做法：框按 (中心 y, 中心 x, 原序号)
  升序逐个看，与当前行里的**每一个**框都满足"同一行"就加入当前行，否则另起一行（不做链式合并）；行按建立的先后（即行首的中心 y）
  从上到下，行内按 (中心 x, 中心 y, 原序号) 从左到右。在网格整数框上算（目标文本里的框就是它们，所以目标的顺序可由目标本身复核）。
- 最多 MAX_BOXES = 30 个：多出的只保留阅读顺序的前 30 个（两个目标都只含这 30 个），记数（truncated、n_dropped_by_cap）。
- 文本：JSON [[x1, y1, x2, y2], …]，分隔符 (", ", ": ")（同 X-2 与 E1 的统一格式）。

**每实例 B 样条**（调用 2 的目标）：对每个保留下来的框，按 X-2 的同一个拟合（src/explore/x2_data.spline_row 用的
oracle_sim.oracle_spline(gt, n = 16, parts = 1, holes = False)：面积最大的 8 连通域填洞后的外轮廓、16 个控制点的闭合周期均匀三次
B 样条，夹到图内后取整到网格）从真值 mask 拟合，再把控制点逐坐标夹进各自的网格框（指令第三节"控制点限制在各自的框内"）。
- 文本：与框一一对应的列表 [[[x, y], …16 个], …]，每一条与 X-2 的样条目标逐字相同（x2_prompts.spline_text），外面再套一层列表
  （"[" + ", ".join(各条) + "]"；只有一个实例时也是三层括号）。
- fit_iou：按推理时的同一套光栅化（x2_prompts.spline_mask：采样点夹在网格框换回的原图像素框里）与真值 mask 的 IoU，即描样条的上限。
- 真值 mask 为空或拟合不出轮廓的实例：这一对的样条目标不可用（spline_ok = false，spline_text = null，原因逐个记），框列表照常；
  训练器只用框列表。记数。

**输出**（<out>/）：xm_targets/targets_<k>.jsonl（按对的先后每 SHARD_PAIRS 对一片；先写 .tmp、fsync 之后改名）、xm_targets/manifest.json
（参数、来源、各文件的条数与 sha256、计数与分布；只由输出的行算，逐字节可复现）、config.yaml（运行身份）、logs/。
续跑：config.yaml 相同时已写完的片核对 id 后跳过。--workers N：spawn 的进程池（src/explore/x2_tools.ToolPool），与单进程逐字节相同。

**读取**（load_targets，训练用）：manifest 与每个文件的 sha256 对得上，参数相同；每行的两个目标文本等于由 boxes_q / ctrl_q 重新写出的，
框的顺序等于 reading_order 的结果，控制点都在各自的框内。
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

from src.executors.raster import rle_to_mask
from src.explore import oracle_sim as OS
from src.explore import x2_prompts as XP
from src.metrics.seg import mask_inter_union, sample_iou
from src.xm import data as D

SCHEMA = 1
FRAMEWORK = "src/xm/targets.py（DEV-0079；D165）"
TEXT_FORMAT = "xm-targets-v1"
DATA_DIR = "xm_targets"
MANIFEST = "manifest.json"
CONFIG = "config.yaml"
TIMING = "logs/timing.jsonl"
MAX_BOXES = 30
SPLINE_N = XP.SPLINE_N_DEFAULT  # 16
SHARD_PAIRS = 20_000
KIND = "xm_pair"
FAIL_MASK_EMPTY = "mask_empty"
FAIL_FIT = "fit_failed"
DEFAULT_PARAMS: Mapping[str, Any] = {"n": SPLINE_N, "max_boxes": MAX_BOXES, "text_format": TEXT_FORMAT}


class XMTargetsError(ValueError):
    """X-M 的训练目标不能生成、不合格，或与训练配置对不上。"""


def check_params(p: Mapping[str, Any]) -> dict[str, Any]:
    bad = []
    try:
        n = XP.check_n(p.get("n"))
    except ValueError as e:
        bad.append(f"n：{e}")
        n = None
    mb = p.get("max_boxes")
    if isinstance(mb, bool) or not isinstance(mb, int) or mb < 1:
        bad.append(f"max_boxes 必须是正整数，收到 {mb!r}")
    if p.get("text_format") != TEXT_FORMAT:
        bad.append(f"text_format 必须是 {TEXT_FORMAT}，收到 {p.get('text_format')!r}")
    if bad:
        raise XMTargetsError("X-M 目标的参数不合格：" + "；".join(bad))
    return {"n": n, "max_boxes": int(mb), "text_format": TEXT_FORMAT}


# ---------------------------------------------------------------------------
# 规则（纯函数）
# ---------------------------------------------------------------------------


def reading_order(boxes: Sequence[Sequence[float]]) -> list[int]:
    """框（[x1, y1, x2, y2]）→ 阅读顺序的下标（规则见模块说明）。"""
    items = []
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = (float(v) for v in b)
        items.append(((y1 + y2) / 2.0, (x1 + x2) / 2.0, y2 - y1, i))
    rows: list[list[tuple[float, float, float, int]]] = []
    for it in sorted(items, key=lambda t: (t[0], t[1], t[3])):
        if rows and all(abs(it[0] - o[0]) < min(it[2], o[2]) / 2.0 for o in rows[-1]):
            rows[-1].append(it)
        else:
            rows.append([it])
    return [t[3] for r in rows for t in sorted(r, key=lambda t: (t[1], t[0], t[3]))]


def box_list_text(boxes_q: Sequence[Sequence[int]]) -> str:
    """网格框 → [[x1, y1, x2, y2], …]（零个写 []）。"""
    return json.dumps([[int(v) for v in b] for b in boxes_q], separators=XP.JSON_SEPARATORS)


def spline_list_text(ctrls_q: Sequence[Sequence[Sequence[int]]]) -> str:
    """每实例的控制点 → [[[x, y], …], …]；每一条与 X-2 的 x2_prompts.spline_text 逐字相同。"""
    return "[" + ", ".join(XP.spline_text(c) for c in ctrls_q) + "]"


def parse_box_list(text: str) -> list[list[int]]:
    """box_list_text 的严格逆（只给目标的复核用；推理时的解析在 DEV-0080 的 src/xm/prompts.py）。"""
    doc = json.loads(text)
    if not isinstance(doc, list) or any(not isinstance(b, list) or len(b) != 4 or not all(isinstance(v, int) and not isinstance(v, bool)
                                                                                         for v in b) for b in doc):
        raise ValueError("不是 [[x1, y1, x2, y2], …] 的整数列表")
    return [[int(v) for v in b] for b in doc]


def parse_spline_list(text: str, *, n: int = SPLINE_N) -> list[list[list[int]]]:
    """spline_list_text 的严格逆：外层是列表，每一条按 X-2 的 parse_spline 严格解析。"""
    doc = json.loads(text)
    if not isinstance(doc, list):
        raise ValueError("外层不是列表")
    out = []
    for c in doc:
        p = XP.parse_spline(json.dumps(c, separators=XP.JSON_SEPARATORS), n=n)
        if p.failure is not None:
            raise ValueError(f"某一条样条不合格：{XP.failure_name(p.failure)}")
        out.append(p.value)
    return out


def clamp_ctrl(ctrl_q: Sequence[Sequence[int]], box_q: Sequence[int]) -> list[list[int]]:
    """网格控制点逐坐标夹进网格框 [x1, y1, x2, y2]。"""
    x1, y1, x2, y2 = (int(v) for v in box_q)
    return [[min(max(int(x), x1), x2), min(max(int(y), y1), y2)] for x, y in ctrl_q]


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    st = mask_inter_union(a, b)
    return float(sample_iou(st.inter, st.union))


def instance_spline(inst: D.XMInstance, box_q: Sequence[int], width: int, height: int, n: int) -> tuple[list[list[int]] | None, float | None, str | None]:
    """一个实例的样条目标（模块说明）：(网格控制点（已夹进框）, fit_iou, 失败原因)。"""
    gt = rle_to_mask(inst.rle).astype(bool)
    if gt.shape != (int(height), int(width)):
        raise XMTargetsError(f"实例 {inst.ann_id} 的 RLE 尺寸 {gt.shape} 与图像 {(height, width)} 不符")
    if not gt.any():
        return None, None, FAIL_MASK_EMPTY
    sp = OS.oracle_spline(gt, n, parts=1, holes=False)
    if not sp["loops_q"]:
        return None, None, FAIL_FIT
    ctrl = clamp_ctrl(sp["loops_q"][0]["ctrl"], box_q)
    fit = XP.spline_mask(ctrl, XP.box_abs(box_q, width, height), gt.shape)
    return ctrl, _iou(fit, gt), None


def pair_targets(pair: D.XMPair, *, n: int = SPLINE_N, max_boxes: int = MAX_BOXES) -> dict[str, Any]:
    """一对的训练行（模块说明）。"""
    w, h = int(pair.width), int(pair.height)
    if not pair.instances:
        raise XMTargetsError(f"{pair.id}：没有实例（只为出现的类别建对）")
    qs = [XP.box_q(i.box_xyxy, w, h) for i in pair.instances]
    order = reading_order(qs)
    kept = order[:max_boxes]
    boxes_q = [qs[k] for k in kept]
    ctrls, fits, fails = [], [], []
    for k in kept:
        c, iou, why = instance_spline(pair.instances[k], qs[k], w, h, n)
        ctrls.append(c)
        fits.append(iou)
        if why is not None:
            fails.append({"ann_id": int(pair.instances[k].ann_id), "reason": why})
    ok = not fails
    return {"id": pair.id, "kind": KIND, "source": pair.source, "split": pair.split, "image": pair.image, "image_size": [w, h],
            "category_id": int(pair.category_id), "category": pair.category, "n_instances": len(pair.instances), "n_kept": len(kept),
            "truncated": len(pair.instances) > max_boxes, "n_crowd": len(pair.crowd_rles),
            "ann_ids": [int(pair.instances[k].ann_id) for k in kept], "boxes_q": boxes_q, "box_text": box_list_text(boxes_q), "n": int(n),
            "spline_ok": ok, "spline_fail": fails, "ctrl_q": ctrls if ok else None, "spline_text": spline_list_text(ctrls) if ok else None,
            "fit_iou": fits}


def pair_job(job: tuple[D.XMPair, int, int]) -> tuple[dict[str, Any], float]:
    pair, n, mb = job
    t0 = time.perf_counter()
    row = pair_targets(pair, n=n, max_boxes=mb)
    return row, time.perf_counter() - t0


def row_problems(r: Mapping[str, Any], params: Mapping[str, Any]) -> list[str]:
    """一行的问题（空 = 合格）：文本等于重新写出的、能严格解析回来、框是阅读顺序、控制点在框内、参数一致。"""
    bad: list[str] = []
    try:
        n, mb = params["n"], params["max_boxes"]
        bq = r["boxes_q"]
        if r["kind"] != KIND or r["n"] != n or len(bq) != r["n_kept"] or r["n_kept"] != min(r["n_instances"], mb) \
                or r["truncated"] != (r["n_instances"] > mb):
            bad.append("kind、n、条数或截断与参数对不上")
        if r["box_text"] != box_list_text(bq) or parse_box_list(r["box_text"]) != bq:
            bad.append("框列表的文本与 boxes_q 对不上")
        # 截断过的行不复核：保留的 30 个在全部框上是阅读顺序，单拿出来重排时，被截掉的那一行的行首可能并进上一行
        if not r["truncated"] and reading_order(bq) != list(range(len(bq))):
            bad.append("框不是阅读顺序")
        if r["spline_ok"]:
            c = r["ctrl_q"]
            if len(c) != len(bq) or r["spline_text"] != spline_list_text(c) or parse_spline_list(r["spline_text"], n=n) != c:
                bad.append("样条的文本与 ctrl_q 对不上")
            elif any(not (b[0] <= x <= b[2] and b[1] <= y <= b[3]) for cc, b in zip(c, bq) for x, y in cc):
                bad.append("控制点不在各自的框内")
        elif r["spline_text"] is not None or r["ctrl_q"] is not None or not r["spline_fail"]:
            bad.append("spline_ok = false 时 spline_text、ctrl_q 必须是 null，且要记原因")
    except (KeyError, TypeError, ValueError) as e:
        bad.append(f"结构不对：{type(e).__name__}: {e}")
    return bad


# ---------------------------------------------------------------------------
# 生成
# ---------------------------------------------------------------------------


def shard_name(k: int) -> str:
    return f"targets_{int(k):03d}.jsonl"


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _line(doc: Mapping[str, Any]) -> str:
    return json.dumps(doc, ensure_ascii=False) + "\n"


def read_jsonl(p: Path) -> list[dict[str, Any]]:
    return [json.loads(ln) for ln in Path(p).read_text(encoding="utf-8").splitlines() if ln.strip()]


def _dump_yaml(doc: Mapping[str, Any]) -> str:
    import yaml

    return yaml.safe_dump(json.loads(json.dumps(doc)), allow_unicode=True, sort_keys=False)


def source_identity(pairs: Sequence[D.XMPair], *, splits_path: str | Path, exclusions_path: str | Path, train_json_sha256: str | None) -> dict[str, Any]:
    return {"xm_splits_sha256": D.file_sha256(splits_path), "xm_train_exclusions_sha256": D.file_sha256(exclusions_path),
            "coco_train2017_sha256": train_json_sha256, "n_pairs": len(pairs),
            "pair_ids_sha256": hashlib.sha256("\n".join(p.id for p in pairs).encode("utf-8")).hexdigest()}


def _prepare(out: Path, ident: Mapping[str, Any], *, resume: bool) -> None:
    cfg = out / CONFIG
    if cfg.exists():
        import yaml

        if yaml.safe_load(cfg.read_text(encoding="utf-8")) != json.loads(json.dumps(ident)):
            raise XMTargetsError(f"{out}：已有的 {CONFIG} 与本次的参数或输入不同，拒绝混写；换一个 --out")
        if not resume:
            raise XMTargetsError(f"{out} 已存在；续跑请加 --resume")
    else:
        if out.exists() and any(out.iterdir()):
            raise XMTargetsError(f"{out} 非空、却没有 {CONFIG}，拒绝写入")
        out.mkdir(parents=True, exist_ok=True)
        cfg.write_text(_dump_yaml(ident), encoding="utf-8")
    (out / DATA_DIR).mkdir(exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)


def build(pairs: Sequence[D.XMPair], out: str | Path, *, params: Mapping[str, Any], source: Mapping[str, Any], tiny: bool = False,
          workers: int = 1, resume: bool = False, shard_pairs: int = SHARD_PAIRS, stop_after_shards: int | None = None,
          argv: Sequence[str] = (), log: Callable[[str], None] | None = None) -> dict[str, Any] | None:
    """生成训练目标（模块说明）。返回 manifest（stop_after_shards 到了、还没写完时返回 None）。"""
    from src.explore.x2_tools import ToolPool

    log = log or (lambda m: print(m, file=sys.stderr, flush=True))
    out = Path(out)
    params = check_params(params)
    ident = {"experiment": "X-M", "framework": FRAMEWORK, "schema": SCHEMA, "tiny": bool(tiny), "params": params,
             "shard_pairs": int(shard_pairs), "tools": {"spline_fit": "oracle_sim.oracle_spline(parts=1, holes=False)",
                                                        "raster": "x2_prompts.spline_mask", "prompt_version": XP.PROMPT_VERSION},
             "source": dict(source)}
    _prepare(out, ident, resume=resume)
    d = out / DATA_DIR
    chunks = [list(pairs[i: i + shard_pairs]) for i in range(0, len(pairs), shard_pairs)]
    run_k = len(list((out / "logs").glob("run_*.json"))) + 1
    run_path = out / "logs" / f"run_{run_k:03d}.json"
    run_log: dict[str, Any] = {"argv": list(argv), "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                               "workers": workers, "shards_total": len(chunks), "finished_utc": None}
    todo = []
    for k, ch in enumerate(chunks):
        p = d / shard_name(k)
        if p.is_file():
            if [r["id"] for r in read_jsonl(p)] != [q.id for q in ch]:
                raise XMTargetsError(f"{p}：已写完的片与这一片的对对不上，拒绝续跑")
            continue
        todo.append(k)
    if stop_after_shards is not None:
        todo = todo[: max(int(stop_after_shards), 0)]
    run_log["shards_this_run"] = [shard_name(k) for k in todo]
    run_path.write_text(json.dumps(run_log, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    t_all = time.perf_counter()
    with ToolPool(workers) as pool:
        for k in todo:
            t0 = time.perf_counter()
            res = pool.map(pair_job, [(q, params["n"], params["max_boxes"]) for q in chunks[k]], chunksize=16)
            tmp = d / (shard_name(k) + ".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                for q, (row, _) in zip(chunks[k], res):
                    if row["id"] != q.id:
                        raise RuntimeError(f"返回的顺序不对：{row['id']} ≠ {q.id}")
                    f.write(_line(row))
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, d / shard_name(k))
            with open(out / TIMING, "a", encoding="utf-8") as f:
                f.write("".join(_line({"id": row["id"], "run": run_k, "shard": shard_name(k), "wall_time_s": wall}) for row, wall in res))
            log(f"[X-M 目标] {shard_name(k)}：{len(chunks[k])} 对，{time.perf_counter() - t0:.1f} s")
    complete = all((d / shard_name(k)).is_file() for k in range(len(chunks)))
    run_log.update(finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), wall_s=round(time.perf_counter() - t_all, 3),
                   complete=complete)
    run_path.write_text(json.dumps(run_log, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    if not complete:
        return None
    return finalize(out, n_shards=len(chunks), params=params, source=source)


def dist(values: Sequence[float]) -> dict[str, Any] | None:
    """分布：n、均值、最小、p50、p90、p99、最大（numpy.percentile 的 linear）。"""
    v = [float(x) for x in values if x is not None]
    if not v:
        return None
    a = np.asarray(v, dtype=np.float64)
    q = np.percentile(a, [5, 25, 50, 75, 90, 95, 99])
    return {"n": len(v), "mean": float(a.mean()), "min": float(a.min()), "p05": float(q[0]), "p25": float(q[1]), "p50": float(q[2]),
            "p75": float(q[3]), "p90": float(q[4]), "p95": float(q[5]), "p99": float(q[6]), "max": float(a.max())}


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """训练行的计数与分布（manifest 与 scripts/xm/data_stats.py 共用）。"""
    fails = Counter(f["reason"] for r in rows for f in r["spline_fail"])
    fits = [v for r in rows for v in r["fit_iou"] if v is not None]
    return {"n_pairs": len(rows), "n_images": len({r["image"] for r in rows}),
            "n_instances": int(sum(r["n_instances"] for r in rows)), "n_instances_kept": int(sum(r["n_kept"] for r in rows)),
            "n_truncated_pairs": int(sum(1 for r in rows if r["truncated"])),
            "truncated_rate": (sum(1 for r in rows if r["truncated"]) / len(rows)) if rows else None,
            "n_dropped_by_cap": int(sum(r["n_instances"] - r["n_kept"] for r in rows)),
            "n_pairs_with_crowd": int(sum(1 for r in rows if r["n_crowd"])),
            "n_spline_ok": int(sum(1 for r in rows if r["spline_ok"])), "spline_fail_instances": {k: fails[k] for k in sorted(fails)},
            "instances_per_pair": dist([r["n_instances"] for r in rows]), "fit_iou": dist(fits),
            "fit_iou_lt_0.8": int(sum(1 for v in fits if v < 0.8)),
            "by_category": {k: v for k, v in sorted(Counter(r["category"] for r in rows).items())}}


def finalize(out: Path, *, n_shards: int, params: Mapping[str, Any], source: Mapping[str, Any]) -> dict[str, Any]:
    d = out / DATA_DIR
    files, rows = [], []
    for k in range(n_shards):
        p = d / shard_name(k)
        rs = read_jsonl(p)
        files.append({"file": p.name, "n": len(rs), "sha256": _sha(p)})
        rows.extend(rs)
    man = {"schema": SCHEMA, "framework": FRAMEWORK, "params": dict(params), "source": dict(source), "files": files,
           "counts": summarize_rows(rows)}
    (d / MANIFEST).write_text(json.dumps(man, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return man


def load_targets(run_dir: str | Path, *, params: Mapping[str, Any] | None = None, data_root: str | Path | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """读 build 的输出并逐行核对（模块说明）。返回 (行, manifest)。params 给了就核对与 manifest 的相同；data_root 给了就核对图像在。"""
    d = Path(run_dir) / DATA_DIR
    mp = d / MANIFEST
    if not mp.is_file():
        raise XMTargetsError(f"{mp} 不在（目标没生成完）")
    man = json.loads(mp.read_text(encoding="utf-8"))
    want = check_params(params if params is not None else man["params"])
    if man["params"] != want:
        raise XMTargetsError(f"X-M 目标的参数 {man['params']} 与训练配置的 {want} 不同")
    rows, problems, seen = [], [], set()
    for f in man["files"]:
        p = d / f["file"]
        if not p.is_file() or _sha(p) != f["sha256"]:
            raise XMTargetsError(f"{p}：文件不在或 sha256 与 manifest 不符")
        for r in read_jsonl(p):
            bad = row_problems(r, want)
            if r.get("id") in seen:
                bad.append("id 重复")
            if not bad and data_root is not None and not (Path(data_root) / r["image"]).is_file():
                bad.append(f"图像不在：{Path(data_root) / r['image']}")
            if bad:
                problems.append(f"{f['file']}（{r.get('id')}）：{'；'.join(bad)}")
                continue
            seen.add(r["id"])
            rows.append(r)
    if problems:
        raise XMTargetsError(f"X-M 的目标有 {len(problems)} 行不合格：\n- " + "\n- ".join(problems[:10]))
    return rows, man


__all__ = ["DEFAULT_PARAMS", "MAX_BOXES", "SPLINE_N", "XMTargetsError", "box_list_text", "build", "check_params", "clamp_ctrl", "dist",
           "instance_spline", "load_targets", "pair_targets", "parse_box_list", "parse_spline_list", "reading_order", "row_problems",
           "source_identity", "spline_list_text", "summarize_rows"]
