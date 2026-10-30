#!/usr/bin/env python
"""定性图（DEV-0076；预注册 U-5；D154 (3)）：只用已有输出（RUN-0092 的 X-2 与椭圆对照、x2_trace 里 LLM 描的 B 样条），不跑模型，
不改任何结果目录，只写 --out。

**抽样（U-5 原文，select_examples；只读 predictions.jsonl，开发机上也能跑）**：
- 池：每个划分在它的 300 句子集（configs/w2_llmseg_subsets.json，划分的序号 k = 它在 splits 里的位置，0–4）里不放回抽 40 句：
  numpy.random.default_rng(20260929 + k).choice(子集的 sent_id 列表（清单里的顺序，即 id 自然顺序）, 40, replace=False)；
  抽中顺序 = choice 返回的顺序；
- 随机例：池里抽中顺序的前 3 句；
- 失败例：池里 X-2（x2_spline_trimap）逐句 IoU（predictions.jsonl 的 score.iou）最低的 2 句，同分按 id（样本 id，如
  refcoco_unc/testA/1090）的字典序；与随机例重合时照规则照样入选（selection.json 的 also 记下重合）；
- 共 25 条（5 × 5），全部进附录；正文取每个划分的第 1 个随机例与第 1 个失败例，共 10 条。
输出 <out>/selection.json：规则与常量、numpy 版本、输入文件的 sha256、每个划分的池（抽中顺序与 X-2 的 IoU）、25 条的 id、划分、
类型、名次、各系统的 IoU（X-2、椭圆对照；给了 --w2c-run 再加 W2-c）。--expect-selection 给一份已有的 selection.json 时逐条核对
相同（例如开发机上先出的那份），不同就退出码 1、什么图都不出。

**每条四格**（标题是指代表达式；每格下标该句的 IoU）：
1. 原图 + LLM 描的 B 样条：控制点（白点）、控制多边形（细虚线）、曲线（eval_closed_bspline，采样数同管线的 bspline_samples），
   提出的框（虚线；管线光栅化前把采样点夹在框内）。IoU = 样条直接光栅化（trace 的 raster_rle）对真值；解析失败时写失败原因；
2. X-2 的最终 mask（x2_spline_trimap 的 answer，即贴边之后）；
3. 椭圆对照（x2_ellipse_trimap 的 answer）；
4. 真值（src/e1/scoring.gt_mask，同 W2 的计分）。
2、3 的 IoU 取 predictions.jsonl 的 score.iou，并用解码的 mask 与真值重算交并核对逐位相同（对不上就报错，说明数据或输出不是同一份）。

**输出**：附录 5 张（每个划分一张，5 条）fig_qual_appendix_<k>_<划分>；正文 1 张 fig_qual_main（5 行 = 划分，每行随机例 | 失败例）；
每张 PDF（矢量，TrueType）与 PNG（300 dpi），配色、字体、尺寸取 scripts/figures/make_paper_figures.py（X-2 = 分类槽 7 的紫）；
每张一个 <名>_caption.tex（抽样规则与数据来源）；figures.json（每条的 IoU、样条失败、核对结果、图像的 sha256、输出文件的 sha256；
不含 mask 与图像）。

**trace.jsonl**（每个约 20 MB，只在 HPC 上）：读之前核对字节数与 sha256 与 <x2-run>/LARGE_FILES.manifest.tsv 逐字相同，不符就退出。

**模式**：
    python scripts/figures/make_qualitative.py --out <目录> --select-only                 # 只出 selection.json（开发机）
    python scripts/figures/make_qualitative.py --out <目录> --data-root $VI/b200/data     # 全流程（要 matplotlib）
    python scripts/figures/make_qualitative.py --out <目录> --data-root … --collect-only  # 不画图：<out>/bundle/（每条的 mask RLE、
        样条与原图拷贝），给没有 matplotlib 的机器用；之后在有 matplotlib 的机器上：
    python scripts/figures/make_qualitative.py --out <目录> --render-from <out>/bundle
--x2-run 缺省 RUN-0092 的 C1（scripts/mgr/x_test_table.CANDS），--w2c-run 缺省 RUN-0072（x_test_table.REF；给 none 不读）。
--tiny：tests/fixtures 的小数据（划分与子集按同一套规则现算，池的大小取 min(40, 子集句数)），只配开发机彩排与测试。
--out 的最后一级目录名必须以四位运行编号开头（如 0098_qualitative），否则退出码 2，什么都不写。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
import textwrap
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from src.data.subsets import population_digest  # noqa: E402
from src.llmseg.runner import SUBSETS_PATH, build_subsets, check_subsets, load_subsets, test_splits  # noqa: E402
from src.w2.data import W2_SPLITS_PATH  # noqa: E402


def _load_script(alias: str, rel: str):
    spec = importlib.util.spec_from_file_location(alias, REPO_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


XT = _load_script("_qual_x_test_table", "scripts/mgr/x_test_table.py")  # 候选的运行目录与设置名（CANDS、MAIN、ABL、M、REF）

RUN_DIR_NAME = re.compile(r"[0-9]{4}_[A-Za-z0-9][A-Za-z0-9_.-]*")  # 同 scripts/run_w2.py
# U-5 的常量（PREREGISTRATION.md U-5 原文；tests/test_qualitative.py 钉住）。种子基数与 src/constants.BOOTSTRAP_SEED 数值相同，但是
# U-5 自己写的数，不是 bootstrap 的约定，所以单列。
POOL_SEED_BASE = 20260929
POOL_N = 40
N_RANDOM = 3
N_FAILURE = 2
MODEL = XT.M
MAIN_SETTING, ELLIPSE_SETTING = XT.MAIN, XT.ABL  # x2_spline_trimap、x2_ellipse_trimap
W2C_SETTING = "w2c"
TASK = "refseg"
TRACE_SETTING = "x2_trace"  # src/explore/x2_runner.TRACE_SETTING（测试核对）
TRACE = "trace.jsonl"
MANIFEST = "LARGE_FILES.manifest.tsv"
PREDICTIONS = "predictions.jsonl"
SELECTION = "selection.json"
FIGURES_JSON = "figures.json"
BUNDLE = "bundle"
RANDOM, FAILURE = "random", "failure"
DRAW_RULE = ("numpy.random.default_rng(POOL_SEED_BASE + k).choice(子集的 sent_id 列表（configs/w2_llmseg_subsets.json 里的顺序）, "
             "POOL_N, replace=False)；抽中顺序 = choice 返回的顺序")
FAILURE_RULE = "池里按 (X-2 的 score.iou, 样本 id) 升序取前 N_FAILURE 句（同分按 id 字典序）；与随机例重合也照样入选"


class QualError(ValueError):
    """输入不合格（子集、预测、trace 的 sha256、数据对不上……）。"""


def _sha(p: str | Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _rel(p: str | Path) -> str:
    q = Path(p).resolve()
    return str(q.relative_to(REPO_ROOT)) if q.is_relative_to(REPO_ROOT) else str(p)


def read_jsonl(p: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(p).read_text(encoding="utf-8").splitlines() if line.strip()]


def leaf(run: str | Path, setting: str, split: str) -> Path:
    return Path(run) / MODEL / TASK / setting / split


# ---------------------------------------------------------------------------
# 抽样（U-5；纯函数）
# ---------------------------------------------------------------------------


def draw_pool(ids: Sequence[str], k: int, n: int = POOL_N) -> list[str]:
    """池：default_rng(POOL_SEED_BASE + k).choice(ids, n, replace=False)，按抽中顺序。"""
    if len(set(ids)) != len(ids):
        raise QualError("子集的 id 有重复")
    if len(ids) < n:
        raise QualError(f"子集只有 {len(ids)} 句，不够抽 {n} 句")
    rng = np.random.default_rng(POOL_SEED_BASE + int(k))
    return [str(v) for v in rng.choice(np.asarray([str(i) for i in ids], dtype=object), size=n, replace=False)]


def pick(pool: Sequence[str], iou: Mapping[str, float], full_id: Mapping[str, str], *, n_random: int = N_RANDOM,
         n_failure: int = N_FAILURE) -> tuple[list[str], list[str]]:
    """(随机例, 失败例)：池的前 n_random 句；按 (iou, 样本 id) 升序的前 n_failure 句。都返回 sent_id。"""
    rnd = list(pool[:n_random])
    fail = sorted(pool, key=lambda s: (float(iou[s]), str(full_id[s])))[:n_failure]
    return rnd, fail


def _by_sent(recs: Sequence[Mapping[str, Any]], split: str, what: str) -> dict[str, Mapping[str, Any]]:
    out: dict[str, Mapping[str, Any]] = {}
    for r in recs:
        if r.get("w2_split", split) != split:
            raise QualError(f"{what} 里有别的划分的行：{r['id']}（{r.get('w2_split')}）")
        sid = str(r["id"]).rsplit("/", 1)[-1]
        if sid in out:
            raise QualError(f"{what} 的 sent_id {sid} 重复")
        out[sid] = r
    return out


def _iou_block(rec: Mapping[str, Any]) -> dict[str, Any]:
    return {"iou": float(rec["score"]["iou"]), "inter": int(rec["score"]["inter"]), "union": int(rec["score"]["union"]),
            "failure": rec.get("failure"), "fallback": rec.get("fallback")}


def select_examples(subsets_doc: Mapping[str, Any], preds: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]], *,
                    pool_n: int | None = POOL_N) -> dict[str, Any]:
    """U-5 的抽样。preds = {设置: {划分: predictions 的行}}，必须有 MAIN_SETTING 与 ELLIPSE_SETTING，可以有 W2C_SETTING。
    pool_n = None：池取 min(POOL_N, 子集句数)（只配 --tiny）。返回 selection 的主体（不含输入文件记录）。"""
    splits_out, entries = [], []
    for k, (split, sub) in enumerate(subsets_doc["splits"].items()):
        ids = [str(i) for i in sub["ids"]]
        if len(ids) != int(sub["n"]) or population_digest(ids) != sub["sha256"]:
            raise QualError(f"子集 {split} 的 {len(ids)} 句与它记的条数或 sha256 不符")
        by = {st: _by_sent(preds[st][split], split, f"{st}/{split}") for st in preds}
        for st, d in by.items():
            missing = [i for i in ids if i not in d]
            if missing:
                raise QualError(f"{st}/{split} 的 predictions 里没有子集的 {len(missing)} 句，例如 {missing[:3]}")
        x2 = by[MAIN_SETTING]
        full = {s: str(x2[s]["id"]) for s in ids}
        for st, d in by.items():
            bad = [s for s in ids if str(d[s]["id"]) != full[s]]
            if bad:
                raise QualError(f"{st}/{split} 与 {MAIN_SETTING} 的样本 id 不同，例如 sent_id {bad[0]}")
        n = min(POOL_N, len(ids)) if pool_n is None else int(pool_n)
        pool = draw_pool(ids, k, n)
        iou = {s: float(x2[s]["score"]["iou"]) for s in pool}
        rnd, fail = pick(pool, iou, full)
        splits_out.append({"split": split, "k": k, "dataset": sub.get("dataset"), "n_subset": len(ids), "pool_n": n,
                           "pool": [{"order": i, "sent_id": s, "id": full[s], "x2_iou": iou[s]} for i, s in enumerate(pool)],
                           "random": [full[s] for s in rnd], "failure": [full[s] for s in fail]})
        for typ, lst, other, olst in ((RANDOM, rnd, FAILURE, fail), (FAILURE, fail, RANDOM, rnd)):
            for rank, s in enumerate(lst, 1):
                entries.append({"key": f"k{k}_{typ}{rank}", "split": split, "k": k, "type": typ, "rank": rank, "id": full[s], "sent_id": s,
                                "pool_order": pool.index(s), "instruction": str(x2[s].get("instruction")), "main": rank == 1,
                                "also": f"{other}{olst.index(s) + 1}" if s in olst else None,
                                "iou": {st: _iou_block(by[st][s]) for st in preds}})
    return {"splits": splits_out, "entries": entries, "main": [e["key"] for e in entries if e["main"]]}


def selection_doc(sel: Mapping[str, Any], inputs: Mapping[str, Any], *, tiny: bool) -> dict[str, Any]:
    return {"experiment": "U-5 定性图的抽样（DEV-0076）", "prereg": "PREREGISTRATION.md U-5", "framework": _rel(__file__),
            "constants": {"POOL_SEED_BASE": POOL_SEED_BASE, "POOL_N": POOL_N, "N_RANDOM": N_RANDOM, "N_FAILURE": N_FAILURE},
            "rules": {"pool": DRAW_RULE, "random": "池里抽中顺序的前 N_RANDOM 句", "failure": FAILURE_RULE,
                      "main": "每个划分的第 1 个随机例与第 1 个失败例", "iou": f"{MAIN_SETTING} / {ELLIPSE_SETTING} / {W2C_SETTING} 的 "
                      "predictions.jsonl 的 score（W2 的计分；失败按空 mask 计）"},
            "numpy": np.__version__, "tiny": bool(tiny), "inputs": dict(inputs), **dict(sel)}


def check_expected(sel: Mapping[str, Any], expect: Mapping[str, Any]) -> list[str]:
    """两份 selection 的池与 25 条（键、id、类型、名次、IoU）是否相同。返回不同之处。"""
    keys = ("key", "id", "split", "type", "rank", "main", "also", "iou")
    bad = []
    a = [{k: e[k] for k in keys} for e in sel["entries"]]
    b = [{k: e.get(k) for k in keys} for e in expect.get("entries", [])]
    if a != b:
        bad.append("entries 不同")
    if [s["pool"] for s in sel["splits"]] != [s.get("pool") for s in expect.get("splits", [])]:
        bad.append("池不同")
    return bad


# ---------------------------------------------------------------------------
# 输入
# ---------------------------------------------------------------------------


def load_preds(x2_run: Path, w2c_run: Path | None, splits: Sequence[str]) -> tuple[dict[str, dict[str, list]], dict[str, Any]]:
    runs = [(MAIN_SETTING, x2_run), (ELLIPSE_SETTING, x2_run)] + ([(W2C_SETTING, w2c_run)] if w2c_run is not None else [])
    preds: dict[str, dict[str, list]] = {}
    files: dict[str, Any] = {}
    for st, run in runs:
        for split in splits:
            p = leaf(run, st, split) / PREDICTIONS
            if not p.is_file():
                raise QualError(f"没有 {p}")
            preds.setdefault(st, {})[split] = read_jsonl(p)
            files.setdefault(st, {})[split] = {"path": _rel(p), "sha256": _sha(p)}
    return preds, files


def manifest_entry(run: Path, rel: str) -> dict[str, Any]:
    p = Path(run) / MANIFEST
    if not p.is_file():
        raise QualError(f"没有 {p}")
    lines = [ln.split("\t") for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    head = lines[0]
    rows = [dict(zip(head, ln)) for ln in lines[1:]]
    hit = [r for r in rows if r.get("path") == rel]
    if len(hit) != 1:
        raise QualError(f"{p} 里 {rel} 有 {len(hit)} 行（应为 1 行）")
    return hit[0]


def verified_trace(run: Path, split: str) -> dict[str, Any]:
    """trace.jsonl 的路径与 manifest 的核对（字节数、sha256 逐字相同）；不符就报错，不读。"""
    rel = f"{MODEL}/{TASK}/{TRACE_SETTING}/{split}/{TRACE}"
    m = manifest_entry(run, rel)
    p = Path(run) / rel
    if not p.is_file():
        raise QualError(f"没有 {p}（trace 只在 HPC 上）")
    size, sha = p.stat().st_size, _sha(p)
    if str(size) != m["bytes"] or sha != m["sha256"]:
        raise QualError(f"{p} 与 {MANIFEST} 不符：{size} B / {sha}，manifest 记 {m['bytes']} B / {m['sha256']}")
    return {"path": p, "rel": rel, "bytes": size, "sha256": sha}


def trace_rows(path: Path, ids: Sequence[str]) -> dict[str, dict[str, Any]]:
    want = set(ids)
    out: dict[str, dict[str, Any]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r["id"] in want:
                out[r["id"]] = r
    missing = sorted(want - set(out))
    if missing:
        raise QualError(f"{path} 里没有 {missing}")
    return out


# ---------------------------------------------------------------------------
# 每条的数据（collect）
# ---------------------------------------------------------------------------


def _decode(rec: Mapping[str, Any]) -> np.ndarray | None:
    from src.executors.raster import rle_to_mask

    ans = rec.get("answer")
    if rec.get("failure") is not None or not ans:
        return None
    return rle_to_mask(ans["mask"]["rle"]).astype(bool)


def collect(selection: Mapping[str, Any], preds: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]], samples: Mapping[str, Any],
            traces: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> list[dict[str, Any]]:
    """25 条的记录（JSON 可写）：mask 的 RLE、样条、IoU 与核对。samples = {样本 id: Sample}；traces = {划分: {样本 id: trace 行}}。"""
    from src.e1.scoring import gt_mask
    from src.executors.raster import mask_to_rle, rle_to_mask
    from src.metrics.seg import mask_inter_union, sample_iou

    by_id = {st: {str(r["id"]): r for split in d.values() for r in split} for st, d in preds.items()}
    out = []
    for e in selection["entries"]:
        s = samples[e["id"]]
        gt = gt_mask(s)
        h, w = gt.shape
        expr = str(s.fields["expression"])
        if expr != e["instruction"]:
            raise QualError(f"{e['id']}：数据里的指代句 {expr!r} 与预测里的 {e['instruction']!r} 不同")
        masks: dict[str, Any] = {}
        for st in (MAIN_SETTING, ELLIPSE_SETTING):
            rec = by_id[st][e["id"]]
            m = _decode(rec)
            got = mask_inter_union(m, gt)
            if (int(got.inter), int(got.union)) != (int(rec["score"]["inter"]), int(rec["score"]["union"])):
                raise QualError(f"{e['id']} {st}：解码的 mask 对真值的交并 {got.inter}/{got.union} 与预测记的 "
                                f"{rec['score']['inter']}/{rec['score']['union']} 不同")
            masks[st] = {"rle": None if m is None else mask_to_rle(m), "iou": float(rec["score"]["iou"]), "failure": rec.get("failure"),
                         "fallback": rec.get("fallback"), "recomputed_inter_union": [int(got.inter), int(got.union)]}
        t = traces[e["split"]][e["id"]]
        sp = t.get("spline") or {}
        call_failure = (sp.get("call") or {}).get("failure")
        raster_iou = None
        if sp.get("raster_rle"):
            st_ = mask_inter_union(rle_to_mask(sp["raster_rle"]).astype(bool), gt)
            raster_iou = sample_iou(*st_)
        spline = {"box": t.get("box"), "ctrl_q": sp.get("ctrl_q"), "proposal_failure": t.get("failure"),
                  "failure": call_failure if sp else (t.get("failure") or "no_spline"), "raster_iou": raster_iou}
        img = Path(s.images[0])
        out.append({**{k: e[k] for k in ("key", "id", "split", "k", "type", "rank", "main", "also")}, "expression": expr,
                    "image": {"path": str(img), "file": img.name, "sha256": _sha(img), "width": int(w), "height": int(h)},
                    "gt_rle": mask_to_rle(gt), "masks": masks, "spline": spline})
    return out


def public_record(r: Mapping[str, Any]) -> dict[str, Any]:
    """figures.json 的一条：去掉 mask 与图像的路径（只留文件名与 sha256）。"""
    return {**{k: v for k, v in r.items() if k not in ("gt_rle", "masks", "image")},
            "image": {k: r["image"][k] for k in ("file", "sha256", "width", "height")},
            "iou": {"spline_raster": r["spline"]["raster_iou"], **{st: m["iou"] for st, m in r["masks"].items()}},
            "masks": {st: {k: m[k] for k in ("failure", "fallback", "recomputed_inter_union")} for st, m in r["masks"].items()}}


def write_bundle(records: Sequence[Mapping[str, Any]], out: Path) -> Path:
    b = out / BUNDLE
    (b / "images").mkdir(parents=True, exist_ok=True)
    rows = []
    for r in records:
        dst = b / "images" / r["image"]["file"]
        if not dst.exists():
            shutil.copyfile(r["image"]["path"], dst)
        rows.append({**r, "image": {k: v for k, v in r["image"].items() if k != "path"}})
    (b / "entries.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return b


def read_bundle(b: Path) -> list[dict[str, Any]]:
    rows = json.loads((b / "entries.json").read_text(encoding="utf-8"))
    for r in rows:
        p = b / "images" / r["image"]["file"]
        if _sha(p) != r["image"]["sha256"]:
            raise QualError(f"{p} 的 sha256 与 entries.json 记的不同")
        r["image"]["path"] = str(p)
    return rows


# ---------------------------------------------------------------------------
# 画图
# ---------------------------------------------------------------------------


def materialize(r: Mapping[str, Any]) -> dict[str, Any]:
    """一条记录 → 画图用的数组（图像、mask、样条的控制点与曲线，原图像素坐标，像素 (c, r) 的中心在 (c + 0.5, r + 0.5)）。"""
    from src.executors.raster import rle_to_mask
    from src.explore import guidance as G
    from src.explore import x2_prompts as XP
    from src.explore.x2_tools import load_rgb

    img = load_rgb(r["image"]["path"])
    h, w = img.shape[:2]
    if (w, h) != (r["image"]["width"], r["image"]["height"]):
        raise QualError(f"{r['id']}：图像尺寸 {(w, h)} 与记录的 {(r['image']['width'], r['image']['height'])} 不同")
    sp = r["spline"]
    ctrl = curve = None
    if sp["ctrl_q"] is not None and sp["box"] is not None:
        ctrl = XP.points_abs(sp["ctrl_q"], w, h)
        curve = G.eval_closed_bspline(ctrl, G.bspline_samples(sp["box"]))
    return {"img": img, "gt": rle_to_mask(r["gt_rle"]).astype(bool),
            "masks": {st: (None if m["rle"] is None else rle_to_mask(m["rle"]).astype(bool)) for st, m in r["masks"].items()},
            "ctrl": ctrl, "curve": curve, "box": sp["box"]}


def _mpf():
    return _load_script("_qual_paper_figures", "scripts/figures/make_paper_figures.py")


def _fmt(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.3f}"


def _panel_labels(r: Mapping[str, Any]) -> list[tuple[str, str]]:
    sp = r["spline"]
    if sp["raster_iou"] is not None:
        s1 = f"raster IoU {_fmt(sp['raster_iou'])}"
    elif sp["proposal_failure"]:
        s1 = f"no proposal\n({sp['proposal_failure']})"
    else:
        s1 = f"spline not parsed\n({sp['failure']})"
    m = r["masks"]
    s2 = f"IoU {_fmt(m[MAIN_SETTING]['iou'])}" + ("\n(fallback: candidate 1)" if m[MAIN_SETTING]["fallback"] else "")
    return [("LLM B-spline", s1), ("X-2 (snapped)", s2), ("Ellipse control", f"IoU {_fmt(m[ELLIPSE_SETTING]['iou'])}"),
            ("Ground truth", "reference")]


def _draw_mask(ax, img, mask, color, alpha=0.45):
    import matplotlib.colors as mcolors

    h, w = img.shape[:2]
    ax.imshow(img, extent=(0, w, h, 0), interpolation="antialiased")
    if mask is None:
        ax.text(w / 2, h / 2, "no mask", ha="center", va="center", fontsize=5.5, color="white",
                bbox={"facecolor": "black", "alpha": 0.5, "lw": 0, "pad": 1})
        return
    if mask.any():
        rgba = np.zeros((h, w, 4), dtype=np.float32)
        rgba[mask] = (*mcolors.to_rgb(color), alpha)
        ax.imshow(rgba, extent=(0, w, h, 0), interpolation="nearest")
        ax.contour(np.arange(w) + 0.5, np.arange(h) + 0.5, mask.astype(np.uint8), levels=[0.5], colors=[color], linewidths=0.5)


def _draw_spline(ax, img, item, color):
    import matplotlib.patheffects as pe
    from matplotlib.patches import Rectangle

    h, w = img.shape[:2]
    ax.imshow(img, extent=(0, w, h, 0), interpolation="antialiased")
    if item["box"] is not None:
        x0, y0, x1, y1 = (float(v) for v in item["box"])
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="white", lw=0.5, ls=(0, (2, 1.5))))
    if item["curve"] is not None:
        c = np.vstack([item["curve"], item["curve"][:1]])
        p = np.vstack([item["ctrl"], item["ctrl"][:1]])
        halo = [pe.withStroke(linewidth=1.6, foreground="white")]
        ax.plot(p[:, 0], p[:, 1], color=color, lw=0.35, ls=(0, (1, 1)), alpha=0.9)
        ax.plot(c[:, 0], c[:, 1], color=color, lw=0.8, path_effects=halo)
        ax.plot(item["ctrl"][:, 0], item["ctrl"][:, 1], "o", ms=1.6, mfc="white", mec=color, mew=0.4)


def _group(sf, r: Mapping[str, Any], *, colors: Mapping[str, str], wrap: int, headers: bool, title_prefix: str):
    item = materialize(r)
    axs = sf.subplots(1, 4)
    labels = _panel_labels(r)
    for i, ax in enumerate(axs):
        if i == 0:
            _draw_spline(ax, item["img"], item, colors["spline"])
        elif i == 1:
            _draw_mask(ax, item["img"], item["masks"][MAIN_SETTING], colors["x2"])
        elif i == 2:
            _draw_mask(ax, item["img"], item["masks"][ELLIPSE_SETTING], colors["ellipse"])
        else:
            _draw_mask(ax, item["img"], item["gt"], colors["gt"])
        h, w = item["img"].shape[:2]
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        if headers:
            ax.set_title(labels[i][0], fontsize=5.5, pad=2)
        ax.set_xlabel(labels[i][1], fontsize=5.2, labelpad=1.5)
    sf.suptitle(textwrap.fill(f'{title_prefix}"{r["expression"]}"', width=wrap), fontsize=6, ha="center")


def _tag(r: Mapping[str, Any], main: bool) -> str:
    if main:
        return "Random: " if r["type"] == RANDOM else "Failure: "
    return f"{'Random' if r['type'] == RANDOM else 'Failure'} {r['rank']}: "


def render(records: Sequence[Mapping[str, Any]], selection: Mapping[str, Any], out: Path) -> dict[str, Any]:
    """附录 5 张 + 正文 1 张，PDF 与 PNG，各配一个 caption .tex。返回 {图名: {files, caption, keys}}。"""
    MPF = _mpf()
    import matplotlib.pyplot as plt

    plt.rcParams.update(MPF.STYLE)
    colors = {"spline": MPF.SERIES_7, "x2": MPF.SERIES_7, "ellipse": MPF.SERIES_2, "gt": MPF.SERIES_3}
    by_key = {r["key"]: r for r in records}
    splits = [s["split"] for s in selection["splits"]]
    figs: dict[str, Any] = {}
    for s in selection["splits"]:
        keys = [e["key"] for e in selection["entries"] if e["split"] == s["split"]]
        name = f"fig_qual_appendix_{s['k']}_{s['split'].removeprefix('test_').replace('+', 'plus')}"
        fig = plt.figure(figsize=(MPF.DOUBLE_COL_IN, 1.55 * len(keys)), layout="constrained")
        sfs = np.atleast_1d(fig.subfigures(len(keys), 1, hspace=0.02))
        for i, (sf, key) in enumerate(zip(sfs, keys)):
            _group(sf, by_key[key], colors=colors, wrap=120, headers=True, title_prefix=_tag(by_key[key], False))
        fig.suptitle(f"{MPF.split_label(s['split'])} (k = {s['k']})", fontsize=7, fontweight="bold")
        files = MPF._save(fig, out, name)
        figs[name] = {"files": files, "keys": keys, "caption": caption_appendix(s, selection, MPF)}
    rows = [[e["key"] for e in selection["entries"] if e["split"] == sp and e["main"]] for sp in splits]
    fig = plt.figure(figsize=(MPF.DOUBLE_COL_IN, 1.05 * len(rows)), layout="constrained")
    rsfs = np.atleast_1d(fig.subfigures(len(rows), 1, hspace=0.02))
    for i, (rsf, keys, sp) in enumerate(zip(rsfs, rows, splits)):
        gsfs = np.atleast_1d(rsf.subfigures(1, len(keys), wspace=0.03))
        for gsf, key in zip(gsfs, keys):
            _group(gsf, by_key[key], colors=colors, wrap=62, headers=i == 0, title_prefix=_tag(by_key[key], True))
        rsf.supylabel(MPF.split_label(sp), fontsize=6.5)
    files = MPF._save(fig, out, "fig_qual_main")
    figs["fig_qual_main"] = {"files": files, "keys": [k for r in rows for k in r], "caption": caption_main(selection, MPF)}
    for name, f in figs.items():
        p = out / f"{name}_caption.tex"
        p.write_text(f"% Generated by scripts/figures/make_qualitative.py; do not edit.\n\\caption{{{MPF._tex_escape(f['caption'])}}}\n",
                     encoding="utf-8")
        f["files"] = [*f["files"], p]
    return figs


def _run_label(path: str) -> str:
    """结果文件的路径 → "RUN-0092"（运行目录名的四位编号；没有就用目录名）。"""
    name = Path(path.split(f"/{MODEL}/")[0]).name
    return f"RUN-{name[:4]}" if RUN_DIR_NAME.fullmatch(name) else name


def _source_sentence(selection: Mapping[str, Any]) -> str:
    runs = sorted({_run_label(v["path"]) for st in (MAIN_SETTING, ELLIPSE_SETTING) for v in selection["inputs"]["predictions"][st].values()})
    return (f"Data: existing per-expression outputs of {', '.join(runs)} (X-2 and the ellipse control; the spline control points from "
            "the same run's trace, sha256-checked); ground truth from the RefCOCO annotations. No model was run for this figure.")


COLUMNS = ("Columns: (1) image with the B-spline written by the LLM (white dots: control points; dotted: control polygon; solid: the "
           "closed uniform cubic B-spline; dashed: the proposed box, inside which the curve is clamped before rasterization), labelled with "
           "the IoU of the spline's direct rasterization; (2) X-2's final mask after snapping (spline-seeded GrabCut; when the spline "
           "cannot be parsed, X-2 falls back to the first candidate mask, as marked); (3) the ellipse "
           "control (the inscribed ellipse of the same proposed box, same snapping); (4) ground truth. IoU is per expression, scored as in "
           "W2 (a failed pipeline counts as an empty mask).")


def _rule_sentence() -> str:
    return (f"Sampling (pre-registered, U-5): in each split, {POOL_N} expressions are drawn without replacement from its 300-expression "
            f"subset with numpy.random.default_rng({POOL_SEED_BASE} + k), k = the split's index (0–4); the random examples are the first "
            f"{N_RANDOM} drawn, the failure examples the {N_FAILURE} with the lowest X-2 IoU in the pool (ties broken by id). "
            "No example was hand-picked or replaced.")


def caption_main(selection: Mapping[str, Any], MPF) -> str:
    return (f"Qualitative results on the W2 test (Qwen3.5-9B). One row per split; left: the first random example, right: the "
            f"lowest-IoU example of the pool. " + COLUMNS + " " + _rule_sentence() + " All 25 examples are in the appendix. "
            + _source_sentence(selection))


def caption_appendix(s: Mapping[str, Any], selection: Mapping[str, Any], MPF) -> str:
    return (f"Qualitative results on {MPF.split_label(s['split'])} (W2 test, Qwen3.5-9B): random examples 1–{N_RANDOM} and failure "
            f"examples 1–{N_FAILURE} of the pre-registered sample. " + COLUMNS + " " + _rule_sentence() + " " + _source_sentence(selection))


# ---------------------------------------------------------------------------
# 命令行
# ---------------------------------------------------------------------------


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="输出目录（最后一级以四位运行编号开头，如 experiments/U/0098_qualitative）")
    p.add_argument("--x2-run", default=XT.CANDS["C1 (X-2)"], help="X-2 的运行目录（默认 RUN-0092 的 C1）")
    p.add_argument("--w2c-run", default=XT.REF, help="W2-c 的运行目录（默认 RUN-0072；给 none 不读）")
    p.add_argument("--subsets", default=str(SUBSETS_PATH), help="300 句子集（默认 configs/w2_llmseg_subsets.json）")
    p.add_argument("--data-root", help="落地目录（默认环境变量 VISINT_DATA_ROOT）；画图与 --collect-only 要")
    p.add_argument("--splits-file", default=str(W2_SPLITS_PATH), help="W2 的划分清单（默认 configs/w2_splits.json）")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--select-only", action="store_true", help="只出 selection.json（只读 predictions.jsonl）")
    g.add_argument("--collect-only", action="store_true", help="不画图，写 <out>/bundle（给没有 matplotlib 的机器）")
    g.add_argument("--render-from", metavar="BUNDLE", help="从 --collect-only 的 bundle 画图（要 matplotlib，不要数据与 trace）")
    p.add_argument("--expect-selection", help="已有的 selection.json：本次抽样必须与它逐条相同")
    p.add_argument("--tiny", action="store_true", help="tests/fixtures 的小数据（开发机彩排与测试）")
    a = p.parse_args(argv)
    if not RUN_DIR_NAME.fullmatch(Path(a.out).name):
        p.error(f"--out 的目录名必须以四位运行编号开头，形如 0098_qualitative，收到 {a.out!r}")
    a.w2c_run = None if str(a.w2c_run).lower() == "none" else a.w2c_run
    needs_data = not (a.select_only or a.render_from)
    if needs_data and not a.tiny and not (a.data_root or os.environ.get("VISINT_DATA_ROOT")):
        p.error("画图与 --collect-only 要 --data-root（或设 VISINT_DATA_ROOT）")
    if needs_data and not a.collect_only and importlib.util.find_spec("matplotlib") is None:
        p.error("这个环境没有 matplotlib：先用 --collect-only 出 bundle，再在有 matplotlib 的机器上 --render-from")
    if a.render_from and importlib.util.find_spec("matplotlib") is None:
        p.error("--render-from 要 matplotlib")
    return a


def _subsets(a) -> tuple[dict[str, Any], dict[str, Any]]:
    if a.tiny:
        from src.w2.config import load_w2_config
        from src.w2.data import tiny_splits

        doc = build_subsets(tiny_splits(load_w2_config()), splits=test_splits(), source={"path": "tiny（fixture 上现算）"})
        return doc, {"path": "tiny（src/llmseg/runner.build_subsets 在 src/w2/data.tiny_splits 上现算）"}
    problems = check_subsets(a.subsets, a.splits_file)
    if problems:
        raise QualError("；".join(problems))
    return load_subsets(a.subsets), {"path": _rel(a.subsets), "sha256": _sha(a.subsets)}


def _samples(a, splits: Sequence[str], ids: set[str]) -> dict[str, Any]:
    from src.e1.tasks import DataPaths
    from src.w2.config import load_w2_config
    from src.w2.data import load_splits, tiny_paths, tiny_splits, w2_items

    if a.tiny:
        paths, doc = (DataPaths(root=Path(a.data_root)) if a.data_root else tiny_paths()), tiny_splits(load_w2_config())
    else:
        paths, doc = DataPaths(root=Path(a.data_root or os.environ["VISINT_DATA_ROOT"])), load_splits(a.splits_file)
    out = {}
    for split in splits:
        for s in w2_items(TASK, split, paths, splits=doc):
            if s.id in ids:
                out[s.id] = s
    missing = sorted(ids - set(out))
    if missing:
        raise QualError(f"数据里找不到 {missing[:3]}")
    return out


def _write_json(p: Path, doc: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def _figures_doc(figs: Mapping[str, Any], records: Sequence[Mapping[str, Any]], extra: Mapping[str, Any]) -> dict[str, Any]:
    return {"experiment": "U-5 定性图（DEV-0076）", "framework": _rel(__file__), "script_sha256": _sha(__file__), **dict(extra),
            "figures": {n: {"keys": f["keys"], "files": {Path(x).name: _sha(x) for x in f["files"]}} for n, f in figs.items()},
            "entries": [public_record(r) for r in records]}


def main(argv=None) -> int:
    a = parse_args(argv)
    out = Path(a.out)
    try:
        if a.render_from:
            records = read_bundle(Path(a.render_from))
            selection = json.loads((Path(a.render_from).parent / SELECTION).read_text(encoding="utf-8"))
            out.mkdir(parents=True, exist_ok=True)
            figs = render(records, selection, out)
            _write_json(out / FIGURES_JSON, _figures_doc(figs, records, {"from_bundle": _rel(a.render_from)}))
            print(f"写出 {len(figs)} 张图（{out}）")
            return 0
        subsets_doc, sub_src = _subsets(a)
        splits = list(subsets_doc["splits"])
        preds, pred_files = load_preds(Path(a.x2_run), Path(a.w2c_run) if a.w2c_run else None, splits)
        sel = select_examples(subsets_doc, preds, pool_n=None if a.tiny else POOL_N)
        selection = selection_doc(sel, {"subsets": sub_src, "predictions": pred_files}, tiny=a.tiny)
        if a.expect_selection:
            bad = check_expected(selection, json.loads(Path(a.expect_selection).read_text(encoding="utf-8")))
            if bad:
                print(f"抽样与 {a.expect_selection} 不同：{bad}；什么都不写", file=sys.stderr)
                return 1
        _write_json(out / SELECTION, selection)
        print(f"写出 {out / SELECTION}（{len(selection['entries'])} 条；正文 {len(selection['main'])} 条）")
        for e in selection["entries"]:
            print(f"  {e['key']:<14} {e['id']:<28} X-2 IoU {e['iou'][MAIN_SETTING]['iou']:.4f}" + (f"（也是 {e['also']}）" if e["also"] else ""))
        if a.select_only:
            return 0
        traces, trace_files = {}, {}
        for s in selection["splits"]:
            v = verified_trace(Path(a.x2_run), s["split"])
            trace_files[s["split"]] = {"path": _rel(v["path"]), "bytes": v["bytes"], "sha256": v["sha256"], "manifest": "same"}
            ids = [e["id"] for e in selection["entries"] if e["split"] == s["split"]]
            traces[s["split"]] = trace_rows(v["path"], ids)
        samples = _samples(a, splits, {e["id"] for e in selection["entries"]})
        records = collect(selection, preds, samples, traces)
        extra = {"selection_sha256": _sha(out / SELECTION), "traces": trace_files, "tiny": a.tiny}
        if a.collect_only:
            b = write_bundle(records, out)
            _write_json(out / FIGURES_JSON, _figures_doc({}, records, {**extra, "bundle": str(b)}))
            print(f"写出 {b}（{len(records)} 条；图像拷贝 {len(list((b / 'images').iterdir()))} 张）")
            return 0
        figs = render(records, selection, out)
        _write_json(out / FIGURES_JSON, _figures_doc(figs, records, extra))
        print(f"写出 {len(figs)} 张图与图注（{out}）")
        return 0
    except (QualError, OSError, KeyError) as e:
        print(f"失败：{e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
