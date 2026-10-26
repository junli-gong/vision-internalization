#!/usr/bin/env python
"""探索 X-1：各种"LLM 输出通道"的上限（真值模拟，CPU；DEV-0068；预注册 X；docs/exploration/log.md）。

不跑任何模型。每句话 × 每种框（gt：真值框；base：基座 9B 的 E1 框；w2c：W2-c 的框），用真值模拟的"理想 LLM"（src/explore/oracle_sim）
给出点、B 样条、带符号的局部修正框，交给不含学习的工具（src/explore/guidance）出 mask，再加生成器库 v2 的最优候选；格点与噪声档在
configs/explore_ceiling.json（grids.<--grid>，noise_levels），逐句的计算在 src/explore/ceiling.evaluate_job。

运行（服务器 CPU；--out 是本次运行的目录 experiments/X/<运行编号>_ceiling_<节点>，每个划分写进 <out>/<划分>/）：
    export HF_HOME=$ROOT/.hf HF_HUB_CACHE=<9B 的 tokenizer 所在的缓存> HF_HUB_OFFLINE=1 TMPDIR=$PROJ/tmp PIP_NO_CACHE_DIR=1
    python scripts/explore_ceiling.py --splits devtune800 devcheck2 --data-root $PROJ/data --grid first --workers 16 \\
        --out experiments/X/00NN_ceiling_<节点> [--resume]
  跑完 first 再用同一条命令、--grid full、另一个 --out 跑全网格。中断后加 --resume 续跑（身份必须相同；没写完的那一组整组重跑）。
划分：devtune800、devcheck2（自 G-W2 起只作开发集，D126-3）；以后 explore_val 生成了，用 --splits-file 指向它的清单、--box-run 给框。
  名字以 test 开头或含 devcheck3 的划分一律拒绝（退出码 2，什么都不写）；configs/explore_ceiling.json 的 allowed_splits 之外的也拒绝。
框：configs/explore_ceiling.json 的 box_sources（按划分；RUN-0069、RUN-0070 的目录），可用 --box-run 种类=目录 覆盖；--boxes 选种类。
  修正通道的起点另有 W2-c 的最终 mask（start_masks；--start-mask 名=种类:目录 覆盖）。
输出（<out>/<划分>/）：
- predictions.jsonl：每句 × 每种框一行（id、box_kind、box、box_iou、cells：{格点: [交, 并, IoU, token 数]}、info），不存 mask；
  除耗时外逐字节可复现（进程数、续跑都不影响）；
- metrics.json：只由 predictions.jsonl 算：每种框、每个格点的 cIoU 与 95% CI（全部、框 IoU ≥ 0.9、< 0.9 各一次）、gIoU、token 均值；
- summary.md：汇总表；compute.json：CPU 墙钟、各通道的 CPU 秒、进程池子进程里有没有学习型模块（应为空）；
- config.yaml（运行身份）、meta.json、logs/（timing.jsonl、run_<k>.json）。
进程池：--workers N（spawn；子进程里 OpenCV 与 BLAS 都是 1 个线程，同 W2 的候选生成）。开发机彩排：--tiny（tests/fixtures 的小数据，
框由真值框确定地抖动而成，tokenizer 用 tiny_model）。--bench：在 640×480 的 scikit-image 自带照片上测每句每格点的 CPU 时间并估墙钟。

用时估计（--bench；开发机 Apple M4，单进程、单线程，2026-10-25，机器同时有别的任务；scikit-image 自带的 astronaut、chelsea、coffee、
rocket 缩放到 640×480，每张配中、小、大三个框（图面积的 25%、9%、72%，同 W2 的 BENCH_BOXES），真值是框里的椭圆，按 w2c 框计（有 W2-c
起点，格点最多）；共 12 个 job，每个 job = 一句话 × 一种框；时间是 CPU 秒）。--tiny 的夹具图只有 48×36（每 job 约 0.2 s），不能代表
COCO 的 640×480，所以没用它估：
- grid first（w2c 框 60 个格点）：每 job 均值 36.1 s、中位 32.0 s（小框 10–12 s，大框 36–90 s）；其中生成器库 16.0、点 12.0、
  样条作种子 4.5、修正（局部 GrabCut）3.4、修正（direct）0.15、样条直接光栅化 0.03。
- grid full（w2c 框 322 个格点）：每 job 均值 117.7 s、中位 97.5 s；其中点 58.7、样条作种子 23.4、修正（局部 GrabCut）20.2、
  生成器库 14.1、修正（direct）1.2、光栅化 0.15。
- 1,300 句 = devtune800（800 句 × gt、base 两种框）+ devcheck2（500 句 × gt、base、w2c 三种框）= 3,100 个 job。按上面的均值
  （上界：gt、base 框没有 W2-c 起点，修正的格点少一半）：first 约 31 CPU 小时 → 8 核约 3.9 小时、16 核约 1.9 小时；full 约 101 CPU
  小时 → 8 核约 12.7 小时、16 核约 6.3 小时。服务器的单核通常比 M4 慢（倍数没有实测；docs/w2_tools.md §6 同样提醒先在服务器上复测），
  进程池在每组末尾还有空等，所以排期时留余量；可先在服务器上跑 --bench --bench-grids first（单进程约 8 分钟）换算。
  先跑 first（不含噪声档），看过再定 full 跑不跑、跑哪几个通道。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# 顶层只 import 不含 torch 的模块：spawn 的子进程会把本脚本当 __mp_main__ 重新 import 一遍
from src.explore import ceiling as X  # noqa: E402
from src.explore import oracle_sim as O  # noqa: E402

CONFIG_PATH = REPO_ROOT / "configs" / "explore_ceiling.json"
PREDICTIONS, METRICS, SUMMARY, COMPUTE, CONFIG, TIMING = "predictions.jsonl", "metrics.json", "summary.md", "compute.json", "config.yaml", "timing.jsonl"
FRAMEWORK = "scripts/explore_ceiling.py（DEV-0068；src/explore）"
TINY_INPUTS = "_tiny_inputs"
TINY_JITTER = 0.1  # --tiny 的假框：真值框每个坐标加 ±10% 框宽 / 框高以内的确定抖动


class SplitRefused(ValueError):
    """划分不允许（test_*、devcheck3、不在 allowed_splits 里）。"""


def check_split(name: str, cfg: Mapping[str, Any]) -> None:
    low = str(name).lower()
    if low.startswith("test") or "devcheck3" in low:
        raise SplitRefused(f"划分 {name!r}：test 与 devcheck3 在判定前不碰（预注册 X §1），拒绝")
    if name not in cfg["allowed_splits"]:
        raise SplitRefused(f"划分 {name!r} 不在 configs/explore_ceiling.json 的 allowed_splits {cfg['allowed_splits']} 里")


def load_config(path: str | Path = CONFIG_PATH) -> tuple[dict[str, Any], str]:
    raw = Path(path).read_bytes()
    cfg = json.loads(raw)
    X.check_grid(cfg["grids"]["full"], cfg["generators"]["order"], cfg["noise_levels"])
    for name, g in cfg["grids"].items():
        X.check_grid(g, cfg["generators"]["order"], cfg["noise_levels"])
    for k, v in cfg["noise_levels"].items():
        O.NoiseLevel.from_dict(k, v)
    return cfg, hashlib.sha256(raw).hexdigest()


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _rel(p: str | Path) -> str:
    q = Path(p).resolve()
    return str(q.relative_to(REPO_ROOT)) if q.is_relative_to(REPO_ROOT) else str(p)


def _where(p: str | Path, out: str | Path) -> str:
    """运行身份里的路径：--out 之下的记成 <out>/…（与机器、--out 的写法无关），其余同 _rel。"""
    try:
        return "<out>/" + str(Path(p).resolve().relative_to(Path(out).resolve()))
    except ValueError:
        return _rel(p)


def _abs(p: str | Path) -> Path:
    q = Path(p)
    return q if q.is_absolute() else REPO_ROOT / q


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(ln) for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()]


def _jsonl_line(doc: Mapping[str, Any]) -> str:
    return json.dumps(doc, ensure_ascii=False, separators=(",", ":")) + "\n"


# ---------------------------------------------------------------------------
# 样本、框与起点
# ---------------------------------------------------------------------------


def load_samples(split: str, *, tiny: bool, data_root: str | None, splits_file: str | None) -> tuple[list[Any], dict[str, Any], Any]:
    """(样本, 清单的出处, DataPaths)。tiny：W2 的 tiny 划分（tests/fixtures 上现算）。"""
    from src.e1.tasks import DataPaths
    from src.w2.data import W2_SPLITS_PATH, load_splits, tiny_paths, tiny_splits, w2_items

    if tiny:
        doc, paths, src = tiny_splits(), tiny_paths(), {"path": "tiny（src/w2/data.tiny_splits）"}
    else:
        if not data_root:
            raise SystemExit("真数据要给 --data-root（或设 VISINT_DATA_ROOT）")
        sf = Path(splits_file) if splits_file else W2_SPLITS_PATH
        doc, paths, src = load_splits(sf), DataPaths(root=Path(data_root)), {"path": _rel(sf), "sha256": _sha(sf)}
    return w2_items("refseg", split, paths, splits=doc), src, paths


def _box_records(leaf: Path) -> list[dict[str, Any]]:
    p = leaf / "predictions.jsonl"
    if not p.is_file():
        raise FileNotFoundError(f"{p} 不存在")
    return read_jsonl(p)


def matched_proposals(samples: Sequence[Any], recs: Sequence[Mapping[str, Any]], gts: Mapping[str, Any], family: str) -> dict[str, Any]:
    """按 W2 的提出读法对上框记录（src/w1/baselines.matched_boxes：同一句提示、同一原图尺寸），返回 {id: 原图像素 xyxy 或 None}。"""
    from src.e1.prompts import PROMPTS_PATH, load_prompt_config
    from src.w1 import baselines as B
    from src.w2.config import PROPOSE_BOX
    from src.w2.loop import proposal_of

    by = B.matched_boxes(samples, recs, load_prompt_config(PROMPTS_PATH), family, gts)
    return {s.id: proposal_of(PROPOSE_BOX, by[s.id])["box"] for s in samples}


def start_masks(leaf: Path, samples: Sequence[Any], boxes: Mapping[str, Any]) -> dict[str, Any]:
    """W2-c 最终 mask 的 RLE（refseg/w2c/<划分> 的 answer.mask）；核对它的 proposal.box 与对应种类的框相同。框是 None 的句子给 None。"""
    by = {r["id"]: r for r in _box_records(leaf)}
    out: dict[str, Any] = {}
    for s in samples:
        r = by.get(s.id)
        if r is None:
            raise ValueError(f"{leaf}：缺 {s.id}")
        pb, want = (r.get("proposal") or {}).get("box"), boxes[s.id]
        if (pb is None) != (want is None) or (pb is not None and max(abs(float(a) - float(b)) for a, b in zip(pb, want)) > 1e-6):
            raise ValueError(f"{leaf}：{s.id} 的 proposal.box {pb} 与框 {want} 不同")
        ans = r.get("answer")
        if want is not None and (ans is None or ans.get("kind") != "mask"):
            raise ValueError(f"{leaf}：{s.id} 有框却没有 mask 答案")
        out[s.id] = None if want is None else ans["mask"]["rle"]
    return out


def tiny_box_inputs(out_split: Path, samples: Sequence[Any], gts: Mapping[str, Any], family: str, kinds: Sequence[str],
                    split: str) -> tuple[dict[str, Path], dict[str, Path]]:
    """--tiny：写出 E1 格式的假框记录（真值框确定地抖动；base 的最后一句记解析失败）与 W2 格式的假 W2-c 最终 mask，
    走与真运行相同的读法。返回 ({种类: 目录}, {起点: 目录})。"""
    import numpy as np

    from src.e1.prompts import PROMPTS_PATH, load_prompt_config
    from src.executors.raster import box_to_mask, mask_to_rle

    prompts = load_prompt_config(PROMPTS_PATH)
    leaves: dict[str, Path] = {}
    for kind in kinds:
        if kind == "gt":
            continue
        leaf = out_split / TINY_INPUTS / kind
        leaf.mkdir(parents=True, exist_ok=True)
        lines = []
        for i, s in enumerate(samples):
            h, w = gts[s.id].shape
            x0, y0, x1, y1 = (float(v) for v in s.gt["box"])
            rng = O.rng_for("tiny", kind, s.id)
            d = rng.uniform(-TINY_JITTER, TINY_JITTER, 4) * np.array([x1 - x0, y1 - y0, x1 - x0, y1 - y0])
            b = [min(max(x0 + d[0], 0.0), w - 1.0), min(max(y0 + d[1], 0.0), h - 1.0), 0.0, 0.0]
            b[2], b[3] = max(min(x1 + d[2], float(w)), b[0] + 1.0), max(min(y1 + d[3], float(h)), b[1] + 1.0)
            fail = kind == "base" and i == len(samples) - 1
            lines.append({"id": s.id, "instruction": prompts.render("refexp_box", family, s).text, "image_size": [w, h],
                          "failure": "no_box" if fail else None, "parsed": None if fail else {"box": b}})
        (leaf / "predictions.jsonl").write_text("".join(_jsonl_line(r) for r in lines), encoding="utf-8")
        leaves[kind] = leaf
    starts: dict[str, Path] = {}
    if "w2c" in leaves and split == "devcheck2":
        leaf = out_split / TINY_INPUTS / "w2c_final"
        leaf.mkdir(parents=True, exist_ok=True)
        lines = []
        for r in _box_records(leaves["w2c"]):
            g = gts[r["id"]]
            b = (r.get("parsed") or {}).get("box")
            ans = None if b is None else {"kind": "mask", "mask": {"rle": mask_to_rle(box_to_mask(b, *g.shape) & np.roll(g, 1, axis=1))}}
            lines.append({"id": r["id"], "proposal": {"box": b}, "answer": ans})
        (leaf / "predictions.jsonl").write_text("".join(_jsonl_line(r) for r in lines), encoding="utf-8")
        starts["w2c_final"] = leaf
    return leaves, starts


# ---------------------------------------------------------------------------
# 运行
# ---------------------------------------------------------------------------


def build_plan(cfg: Mapping[str, Any], grid_name: str) -> dict[str, Any]:
    if grid_name not in cfg["grids"]:
        raise SystemExit(f"没有格点 {grid_name!r}；可选 {sorted(cfg['grids'])}")
    return {"grid": cfg["grids"][grid_name], "tools": cfg["tools"], "generator_order": list(cfg["generators"]["order"]),
            "noise_levels": cfg["noise_levels"], "salt": cfg["salt"]}


def _identity_norm(d: Mapping[str, Any]) -> Any:
    return json.loads(json.dumps(d, sort_keys=True, ensure_ascii=False))


def resume_point(path: Path, keys: Sequence[tuple[str, str]], bounds: Sequence[int], log: Callable[[str], None]) -> int:
    """已写的行里与 job 顺序一致的前缀，退回到最近的组边界，并把文件截到那里。返回保留的行数。"""
    if not path.exists():
        return 0
    raw = path.read_bytes()
    good, pos, cut = 0, 0, 0
    while good < len(keys):
        nl = raw.find(b"\n", pos)
        if nl < 0:
            break
        try:
            doc = json.loads(raw[pos:nl])
        except json.JSONDecodeError:
            break
        if (doc.get("id"), doc.get("box_kind")) != tuple(keys[good]):
            raise RuntimeError(f"{path}：第 {good + 1} 行是 {(doc.get('id'), doc.get('box_kind'))}，应为 {tuple(keys[good])}；拒绝续跑")
        good += 1
        pos = nl + 1
    keep = max(b for b in bounds if b <= good)
    if keep:
        cut = 0
        for _ in range(keep):
            cut = raw.index(b"\n", cut) + 1
    with open(path, "r+b") as f:
        f.truncate(cut)
    if keep != len(raw.splitlines()):
        log(f"{path}：保留 {keep} 行（已写 {good} 行完整），从第 {keep + 1} 个 job 续跑")
    return keep


def _machine() -> dict[str, Any]:
    import cv2
    import numpy as np
    import skimage

    return {"platform": platform.platform(), "processor": platform.processor() or platform.machine(), "cpu_count": os.cpu_count(),
            "python": platform.python_version(), "opencv": cv2.__version__, "skimage": skimage.__version__, "numpy": np.__version__}


def make_line(res: Mapping[str, Any], job: Mapping[str, Any], tok: Any) -> dict[str, Any]:
    """worker 的结果 → predictions.jsonl 的一行（token 在主进程数）。"""
    names = list(res["cells"])
    flat, owner = [], []
    for n in names:
        for t in res["texts"].get(n, []):
            flat.append(t)
            owner.append(n)
    counts = O.count_tokens(tok, flat)
    ntok = {n: 0 for n in names}
    for n, c in zip(owner, counts):
        ntok[n] += c
    cells = {}
    for n in names:
        i, u = res["cells"][n]
        cells[n] = [int(i), int(u), round(i / u, 6) if u else 1.0, int(ntok[n])]
    return {"id": job["id"], "box_kind": job["box_kind"], "box": job["box"], "box_iou": job["box_iou"],
            "failure": job["box"] is None, "n_cells": len(names), "cells": cells, "info": res["info"]}


def run_split(split: str, a: argparse.Namespace, cfg: Mapping[str, Any], cfg_sha: str, tok: Any, tok_rec: Mapping[str, Any],
              log: Callable[[str], None]) -> Path:
    import numpy as np

    from src.adapters import registry
    from src.data.subsets import population_digest
    from src.e1.scoring import gt_mask
    from src.executors.raster import mask_to_rle
    from src.runlog import write_run_metadata
    from src.constants import BOOTSTRAP_CI_LEVEL, BOOTSTRAP_N_RESAMPLES, BOOTSTRAP_SEED

    check_split(split, cfg)
    out_split = Path(a.out) / split
    samples, splits_src, _ = load_samples(split, tiny=a.tiny, data_root=a.data_root, splits_file=a.splits_file)
    if a.limit is not None:
        samples = samples[: a.limit]
    gts = {s.id: gt_mask(s) for s in samples}
    family = registry.get_model(cfg["tokenizer"]["model"]).family
    kinds = [k for k in (a.boxes or cfg["box_kinds"])]
    out_split.mkdir(parents=True, exist_ok=True)
    srcs = dict(cfg["box_sources"].get(split, {}))
    starts_cfg = {k: dict(v) for k, v in cfg["start_masks"].get(split, {}).items()}
    for item in a.box_run or []:
        k, _, p = item.partition("=")
        srcs[k] = p
    for item in a.start_mask or []:
        nm, _, rest = item.partition("=")
        kind, _, p = rest.partition(":")
        starts_cfg[nm] = {"box_kind": kind, "leaf": p}
    if a.tiny:  # 与真运行同样的种类（devtune800 没有 W2-c 的框），框换成假的
        kinds = [k for k in kinds if k == "gt" or k in srcs]
        leaves, sleaves = tiny_box_inputs(out_split, samples, gts, family, kinds, split)
        srcs = {k: str(v) for k, v in leaves.items()}
        starts_cfg = {nm: {"box_kind": "w2c", "leaf": str(p)} for nm, p in sleaves.items()}
    kinds = [k for k in kinds if k == "gt" or k in srcs]
    boxes: dict[str, dict[str, Any]] = {}
    ident_boxes: dict[str, Any] = {}
    for k in kinds:
        if k == "gt":
            boxes[k] = {s.id: O.dequantize_box(O.quantize_box(s.gt["box"], *gts[s.id].shape[::-1]), *gts[s.id].shape[::-1])
                        for s in samples}
            ident_boxes[k] = {"rule": cfg["box_rule"].split("；")[0]}
            continue
        leaf = _abs(srcs[k])
        boxes[k] = matched_proposals(samples, _box_records(leaf), gts, family)
        ident_boxes[k] = {"leaf": _where(leaf, a.out), "predictions_sha256": _sha(leaf / "predictions.jsonl")}
    starts: dict[str, dict[str, Any]] = {k: {} for k in kinds}
    ident_starts: dict[str, Any] = {}
    for nm, sc in sorted(starts_cfg.items()):
        if sc["box_kind"] not in boxes:
            continue
        leaf = _abs(sc["leaf"])
        rles = start_masks(leaf, samples, boxes[sc["box_kind"]])
        starts[sc["box_kind"]][nm] = rles
        ident_starts[nm] = {"box_kind": sc["box_kind"], "leaf": _where(leaf, a.out),
                            "predictions_sha256": _sha(leaf / "predictions.jsonl")}
    plan = build_plan(cfg, a.grid)
    jobs = []
    for s in samples:
        for k in kinds:
            b = boxes[k][s.id]
            jobs.append({"id": s.id, "box_kind": k, "image": str(s.images[0]), "gt_rle": mask_to_rle(gts[s.id]),
                         "gt_box": [float(v) for v in s.gt["box"]], "box": None if b is None else [float(v) for v in b],
                         "box_iou": X.box_iou_or_none(b, s.gt["box"]),
                         "starts": {nm: rles[s.id] for nm, rles in starts[k].items()}, "plan": plan})
    cells_by_kind = {k: X.cell_names(plan["grid"], plan["generator_order"], [X.START_GC_DEFAULT] + sorted(starts[k])) for k in kinds}
    ids = [s.id for s in samples]
    identity = {"experiment": "X-1", "framework": FRAMEWORK, "split": split, "n": len(ids), "ids_sha256": population_digest(ids),
                "splits": splits_src, "limit": a.limit, "tiny": bool(a.tiny),
                "config": {"path": _rel(a.config), "sha256": cfg_sha}, "grid": a.grid,
                "cells_sha256": {k: hashlib.sha256("\n".join(v).encode()).hexdigest() for k, v in cells_by_kind.items()},
                "n_cells": {k: len(v) for k, v in cells_by_kind.items()},
                "box_kinds": kinds, "boxes": ident_boxes, "start_masks": ident_starts, "tokenizer": dict(tok_rec),
                "text_format": O.TEXT_FORMAT_VERSION, "chunk_size": int(cfg["chunk_size"]), "seed": BOOTSTRAP_SEED}
    cfg_path = out_split / CONFIG
    if cfg_path.exists():
        import yaml

        if not a.resume:
            raise SystemExit(f"{out_split} 已有结果；续跑请加 --resume")
        old = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        if _identity_norm(old) != _identity_norm(identity):
            raise SystemExit(f"{cfg_path}：运行身份与已有的不同，拒绝混写")
    else:
        files = [_abs(a.config)] + ([] if a.tiny else [_abs(splits_src["path"])])
        write_run_metadata(out_split, config=identity, seed=BOOTSTRAP_SEED, data_files=files, repo_root=REPO_ROOT)
    (out_split / "logs").mkdir(exist_ok=True)
    keys = [(j["id"], j["box_kind"]) for j in jobs]
    chunk = int(cfg["chunk_size"])
    bounds = list(range(0, len(jobs), chunk)) + [len(jobs)]
    pred = out_split / PREDICTIONS
    keep = resume_point(pred, keys, bounds, log)
    todo = [(lo, min(lo + chunk, len(jobs))) for lo in range(keep, len(jobs), chunk)]
    if a.max_chunks is not None:
        todo = todo[: a.max_chunks]
    run_k = len(list((out_split / "logs").glob("run_*.json"))) + 1
    t_all = time.time()
    with X.JobRunner(a.workers) as runner:
        for n_chunk, (lo, hi) in enumerate(todo):
            res = runner.map(jobs[lo:hi])
            lines, tl = [], []
            for j, r in zip(jobs[lo:hi], res):
                if tuple(r["key"]) != (j["id"], j["box_kind"]):
                    raise RuntimeError(f"进程池返回的顺序不对：{r['key']} ≠ {(j['id'], j['box_kind'])}")
                lines.append(_jsonl_line(make_line(r, j, tok)))
                tl.append(_jsonl_line({"id": j["id"], "box_kind": j["box_kind"], "run": run_k, "cpu_s": r["cpu_s"],
                                       "runtime": r["runtime"]}))
            with open(pred, "a", encoding="utf-8") as f:
                f.writelines(lines)
            with open(out_split / "logs" / TIMING, "a", encoding="utf-8") as f:
                f.writelines(tl)
            log(f"[X-1 {split}] {hi}/{len(jobs)} job（{time.time() - t_all:.0f} s）")
    wall = time.time() - t_all
    (out_split / "logs" / f"run_{run_k}.json").write_text(json.dumps(
        {"run": run_k, "argv": sys.argv, "workers": a.workers, "n_jobs_before": keep, "n_jobs_run": sum(h - l for l, h in todo),
         "wall_s": wall, "machine": _machine()}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    lines_all = read_jsonl(pred)
    if len(lines_all) == len(jobs):
        met = X.aggregate(lines_all, good_iou=cfg["metrics"]["good_box_iou"], seed=BOOTSTRAP_SEED, n_resamples=BOOTSTRAP_N_RESAMPLES,
                          level=BOOTSTRAP_CI_LEVEL)
        doc = {"experiment": "X-1", "split": split, "grid": a.grid, "n_sentences": len(ids), "good_box_iou": cfg["metrics"]["good_box_iou"],
               "bootstrap": {"n_resamples": BOOTSTRAP_N_RESAMPLES, "level": BOOTSTRAP_CI_LEVEL, "seed": BOOTSTRAP_SEED},
               "tokenizer": dict(tok_rec), "text_format": O.TEXT_FORMAT_VERSION, "box_kinds": met}
        (out_split / METRICS).write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        (out_split / SUMMARY).write_text(render_summary(doc), encoding="utf-8")
        write_compute(out_split, len(jobs))
        log(f"[X-1 {split}] 写出 {out_split / METRICS}、{SUMMARY}、{COMPUTE}")
    else:
        log(f"[X-1 {split}] 只写了 {len(lines_all)}/{len(jobs)} 个 job（--max-chunks），续跑请加 --resume")
    return out_split


def write_compute(out_split: Path, n_jobs: int) -> None:
    tl = read_jsonl(out_split / "logs" / TIMING)
    last: dict[tuple[str, str], Mapping[str, Any]] = {}
    for r in tl:
        last[(r["id"], r["box_kind"])] = r
    cpu: dict[str, float] = {}
    learned: set[str] = set()
    pids: set[int] = set()
    for r in last.values():
        for k, v in r["cpu_s"].items():
            cpu[k] = cpu.get(k, 0.0) + float(v)
        learned.update(r["runtime"]["learned_modules"])
        pids.add(int(r["runtime"]["pid"]))
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((out_split / "logs").glob("run_*.json"))]
    doc = {"n_jobs": n_jobs, "wall_s_total": sum(r["wall_s"] for r in runs),
           "runs": [{k: r[k] for k in ("run", "workers", "n_jobs_before", "n_jobs_run", "wall_s")} for r in runs],
           "cpu_s_total": cpu, "cpu_s_per_job_mean": {k: v / max(len(last), 1) for k, v in cpu.items()},
           "n_processes": len(pids), "learned_modules_in_workers": sorted(learned), "machine": runs[-1]["machine"] if runs else None,
           "note": "墙钟与 CPU 秒不在可复现之内；learned_modules_in_workers 是各 job 所在进程里已经 import 的学习型模块（--workers > 1 时应为空；"
                   "--workers 1 时是主进程，主进程为了 tokenizer 与读框会 import transformers / torch）。"}
    (out_split / COMPUTE).write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def _fmt(e: Mapping[str, Any] | None, ci: bool = True) -> str:
    if e is None:
        return "—"
    s = f"{e['ciou']:.3f}"
    if ci and e.get("ci"):
        s += f" [{e['ci'][0]:.3f}, {e['ci'][1]:.3f}]"
    return s


def render_summary(doc: Mapping[str, Any]) -> str:
    """metrics.json → markdown：每种框一张表（格点、cIoU [95% CI]、好框 / 坏框的 cIoU、gIoU、token 均值）。"""
    out = [f"# X-1 上限（{doc['split']}，格点 {doc['grid']}）", "",
           f"- {doc['n_sentences']} 句；框 IoU ≥ {doc['good_box_iou']} 记好框；CI 为逐句 bootstrap {doc['bootstrap']['level']:.0%}"
           f"（{doc['bootstrap']['n_resamples']} 次，种子 {doc['bootstrap']['seed']}）。",
           f"- token：{doc['tokenizer']['repo']}@{doc['tokenizer']['revision'][:12]}，格式 {doc['text_format']}（只数这个通道的输出，框不计）。", ""]
    out += ["## 要点：每种框上的参照与各通道、各噪声档里 cIoU 最高的格点", "",
            "| 框 | 行 | 格点 | cIoU [95% CI] | 好框 cIoU | 坏框 cIoU | token 均值 |", "|---|---|---|---|---|---|---|"]
    for kind, m in doc["box_kinds"].items():
        cells = m["cells"]
        refs = [n for n in ("box", "gen.gc_default") if n in cells] + [n for n in cells if n.startswith(("genK.", "ref."))]
        rows = [("参照", n) for n in refs]
        for ch in ("pts", "spl", "cor"):
            noises = list(dict.fromkeys(n.rsplit(".", 1)[1] for n in cells if n.startswith(ch + ".")))  # 格点里的先后
            for nz in noises:
                cand = [n for n in cells if n.startswith(ch + ".") and n.endswith("." + nz) and cells[n]["all"]]
                if cand:
                    rows.append((f"{ch} 最高（{nz}）", max(cand, key=lambda n: (cells[n]["all"]["ciou"], -cand.index(n)))))
        for label, n in rows:
            e = cells[n]
            t = e["tokens_mean"]
            out.append(f"| {kind} | {label} | `{n}` | {_fmt(e['all'])} | {_fmt(e['good'], False)} | {_fmt(e['bad'], False)} | "
                       f"{'—' if t is None else f'{t:.1f}'} |")
    out.append("")
    for kind, m in doc["box_kinds"].items():
        out += [f"## 框：{kind}", "",
                f"n = {m['n']}；好框 {m['n_good_box']}、坏框 {m['n_bad_box']}（其中解析失败 {m['n_box_failure']}）；框 Acc@0.5 = {m['box_acc50']:.3f}", "",
                "| 格点 | cIoU [95% CI] | 好框 cIoU | 坏框 cIoU | gIoU | token 均值 |", "|---|---|---|---|---|---|"]
        for name, e in m["cells"].items():
            g = e["all"]["giou"] if e["all"] else None
            t = e["tokens_mean"]
            out.append(f"| `{name}` | {_fmt(e['all'])} | {_fmt(e['good'], False)} | {_fmt(e['bad'], False)} | "
                       f"{'—' if g is None else f'{g:.3f}'} | {'—' if t is None else f'{t:.1f}'} |")
        out.append("")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# --bench：用时估计
# ---------------------------------------------------------------------------

BENCH_IMAGES = ("astronaut", "chelsea", "coffee", "rocket")
BENCH_BOXES = ((0.25, 0.25, 0.75, 0.75), (0.40, 0.35, 0.65, 0.70), (0.05, 0.10, 0.90, 0.95))  # 同 src/w2/candidates.BENCH_BOXES
EST_SENTENCES = {"devtune800": (800, 2), "devcheck2": (500, 3)}  # 句数、框的种类数（devtune800 没有 W2-c 的框）


def bench(a: argparse.Namespace, cfg: Mapping[str, Any], log: Callable[[str], None]) -> int:
    import numpy as np
    from PIL import Image
    from skimage import data as skdata

    from src.executors.raster import box_to_mask, mask_to_rle

    outd = Path(a.out)
    (outd / "bench_images").mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"machine": _machine(), "grids": {}}
    for grid_name in a.bench_grids:
        plan = build_plan(cfg, grid_name)
        cpu: list[dict[str, float]] = []
        n_cells: list[int] = []
        with X.JobRunner(1) as runner:
            for nm in BENCH_IMAGES:
                p = outd / "bench_images" / f"{nm}.png"
                if not p.exists():
                    Image.fromarray(getattr(skdata, nm)()).convert("RGB").resize((640, 480)).save(p)
                h, w = 480, 640
                for fx0, fy0, fx1, fy1 in BENCH_BOXES:
                    box = [fx0 * w, fy0 * h, fx1 * w, fy1 * h]
                    yy, xx = np.mgrid[0:h, 0:w]
                    cx, cy, rx, ry = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2, 0.45 * (box[2] - box[0]), 0.45 * (box[3] - box[1])
                    gt = ((xx + 0.5 - cx) / rx) ** 2 + ((yy + 0.5 - cy) / ry) ** 2 <= 1.0
                    start = box_to_mask(box, h, w) & np.roll(gt, 3, axis=1)
                    job = {"id": f"bench/{nm}/{fx0}", "box_kind": "w2c", "image": str(p), "gt_rle": mask_to_rle(gt), "gt_box": box,
                           "box": box, "box_iou": 1.0, "starts": {"w2c_final": mask_to_rle(start)}, "plan": plan}
                    r = runner.map([job])[0]
                    cpu.append(r["cpu_s"])
                    n_cells.append(len(r["cells"]))
                    log(f"[bench {grid_name}] {nm} {fx0}: {r['cpu_s']['total']:.1f} s CPU，{len(r['cells'])} 个格点")
        keys = sorted({k for c in cpu for k in c})
        mean = {k: float(np.mean([c.get(k, 0.0) for c in cpu])) for k in keys}
        per_job = mean["total"]
        jobs = sum(n * k for n, k in EST_SENTENCES.values())
        total = jobs * per_job
        report["grids"][grid_name] = {"n_jobs_measured": len(cpu), "n_cells_w2c_box": int(np.max(n_cells)),
                                      "cpu_s_per_job_mean": mean, "cpu_s_per_job_median": float(np.median([c["total"] for c in cpu])),
                                      "cpu_ms_per_cell_mean": 1000.0 * per_job / float(np.mean(n_cells)),
                                      "estimate": {"sentences": sum(n for n, _ in EST_SENTENCES.values()), "jobs": jobs,
                                                   "cpu_h": total / 3600.0, "wall_h_8_cores": total / 8 / 3600.0,
                                                   "wall_h_16_cores": total / 16 / 3600.0}}
    (outd / "bench.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0


# ---------------------------------------------------------------------------
# 命令行
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="探索 X-1：LLM 输出通道的上限（真值模拟，CPU）")
    ap.add_argument("--splits", nargs="+", default=["devtune800", "devcheck2"])
    ap.add_argument("--out", required=True, help="本次运行的目录 experiments/X/<运行编号>_ceiling_<节点>")
    ap.add_argument("--data-root", default=os.environ.get("VISINT_DATA_ROOT"))
    ap.add_argument("--splits-file", default=None, help="划分清单（默认 configs/w2_splits.json；explore_val 用 DEV-0067 的清单）")
    ap.add_argument("--config", default=str(CONFIG_PATH))
    ap.add_argument("--grid", default="full", help="configs/explore_ceiling.json 的 grids 之一（full、first；--tiny 缺省用 tiny）")
    ap.add_argument("--boxes", nargs="+", default=None, help="框的种类（默认 box_kinds：gt base w2c，没有来源的跳过）")
    ap.add_argument("--box-run", action="append", help="种类=E1 refexp_box 的叶目录（覆盖 box_sources）")
    ap.add_argument("--start-mask", action="append", help="名字=框的种类:W2 refseg 的叶目录（覆盖 start_masks）")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tiny", action="store_true", help="tests/fixtures 的小数据（开发机彩排）")
    ap.add_argument("--max-chunks", type=int, default=None, help=argparse.SUPPRESS)  # 测试续跑用：跑这么多组就停
    ap.add_argument("--bench", action="store_true", help="在 640×480 的照片上测每句每格点的 CPU 时间")
    ap.add_argument("--bench-grids", nargs="+", default=["first", "full"])
    a = ap.parse_args(argv)
    log = lambda m: print(m, file=sys.stderr, flush=True)  # noqa: E731
    cfg, cfg_sha = load_config(a.config)
    if a.tiny and a.grid == "full" and "--grid" not in (argv if argv is not None else sys.argv):
        a.grid = "tiny"
    if a.bench:
        return bench(a, cfg, log)
    try:
        for sp in a.splits:
            check_split(sp, cfg)
    except SplitRefused as e:
        print(f"拒绝：{e}", file=sys.stderr)
        return 2
    model = cfg["tokenizer"]["tiny_model" if a.tiny else "model"]
    tok, tok_rec = O.load_tokenizer(model)
    for sp in a.splits:
        run_split(sp, a, cfg, cfg_sha, tok, tok_rec, log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
