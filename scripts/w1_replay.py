#!/usr/bin/env python
"""W1 的错误分析（DEV-0057；D113-4）：按记录重放 W1 回路的每条样本，得到每一步的 mask 与 IoU，再给失败样本分类。只用 CPU。

规则（重放、一致性核对、每条样本的量、四类与先后、阈值的边界）写在 src/w1/replay.py 的模块说明里，这里只管读写。

输入（--run 指向一次 W1 运行的目录，如 RUN-0063 的 experiments/W1/0063_w1_devcheck_hpcgpu07）：
- 回路：<run>/<模型>/refseg/<--setting，默认 full>/<--split，默认 devcheck>/{predictions.jsonl, config.yaml}
  （运行身份 config.yaml 给出工作台类、关掉的组与工作台选项；<run>/LARGE_FILES.manifest.tsv 里列了这个 predictions.jsonl 时核对 sha256）；
- 基线：同一个 <run> 下的 refseg/w1f、refseg/baseline_ownbox、refexp_box/baseline_e1（自己的框）的 predictions.jsonl；
- 图像与真值：--data-root 下的 RefCOCO（同 run_w1.py；划分清单 configs/w1_splits.json，指纹与运行身份核对）。
  运行身份是 --tiny 的（开发机彩排）且不给 --data-root 时，用 tests/fixtures 的小数据（同 run_w1.py 的 data_paths）。
输出（--out）：
- replay.jsonl：每条样本一行（src/w1/replay.replay_episode；不含墙钟，逐字节可复现）；
- replay_meta.json：输入文件与 sha256、运行身份摘要、条数、不一致条数、库版本、墙钟；logs/replay_timing.jsonl：每条的墙钟；
- 加 --classify：error_analysis.json、error_analysis.md（src/w1/replay.classify / render_md）。
只给 --classify 与 --out、不给 --run：只用 <out>/replay.jsonl 重新分类（例如换阈值；开发机上就能跑，不要数据）。

服务器上怎么跑（ITSC HPC；只用 CPU、不占 GPU，经 srun --overlap 进本节点任务的资源范围；铁律：不取消、不停止 Slurm 任务 157296、157297）：

    VI=/home/mail/chenyuma/visint
    # 工作树 $VI/wt/wtNNNN 建在含本脚本的 origin/main 上；RUN-0063 的结果在 $VI/repo/experiments/W1/0063_w1_devcheck_hpcgpu07
    ssh -n hpcgpu07 "srun --jobid=157297 --overlap -n1 -c 8 env CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 \\
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TMPDIR=$VI/tmp PYTHONHASHSEED=0 \\
        bash -c 'cd $VI/wt/wtNNNN && $VI/env/bin/python scripts/w1_replay.py \\
          --run $VI/repo/experiments/W1/0063_w1_devcheck_hpcgpu07 --data-root $VI/b200/data \\
          --out $VI/repo/experiments/W1/NNNN_w1_error_analysis --workers 8 --classify' \\
        > $VI/logs/replay_NNNN.log 2>&1"

  - 也可以在 hpcmgt 上直接跑（同样只用 CPU），去掉 srun 那一层即可。
  - CUDA_VISIBLE_DEVICES 置空：import 链里有 torch（E1 的加载器），这样保证不碰 GPU。
  - --workers：并行的进程数（spawn；每个进程逐条重放，结果按样本顺序写出，与 --workers 1 逐字节相同）。在别的任务正在训练的节点上，
    取不超过空闲核数的值（8 足够）。srun 的 -c 8 给这一步 8 个核：不给时 Slurm 可能只给 1 个核，--workers 8 就等于串行；
    srun 拒绝 -c 8 时去掉它、改用 --workers 1。
  - run_python：RUN-0063 的 8 次 run_python（5 条样本）全部被沙箱的 4 GB 内存上限杀掉，都没有写出 mask。重放在同一个环境里应得到
    同样的报错；若沙箱这次表现不同，这几条会因返回文字不同标为不一致（不进分类），条数与 id 在 replay_meta.json 里。
- 估计用时：RUN-0063 的 logs/workbench.jsonl 里 500 条样本工具调用的墙钟合计 1,060 s（grabcut 424 次 694 s、run_python 8 次 291 s、
  superpixels 175 次 46 s，其余合计 < 30 s；最慢的一条 255 s，主要是 run_python）。重放多出读图、解 RLE 与逐步算 IoU（每步每个 mask
  一次交并，640×480 级，< 1 ms），每个进程启动约 3 s（import torch 与 OpenCV）。
  --workers 1 约 20–25 分钟；--workers 8 约 5–8 分钟（下限是最慢的那一条，约 4–5 分钟）。节点在训练时 CPU 偏忙，按 2 倍留余量。
- 产出约 2–5 MB（replay.jsonl 每次调用记下所有 mask 的 IoU 与面积）；分类只读 replay.jsonl，几秒。

退出码：0 = 完成（重放不一致的条数打印在最后一行，也写进 replay_meta.json，不改变退出码）；2 = 出错（参数、输入不全、工具清单不符等）。
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import hashlib
import json
import multiprocessing as mp
import os
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.w1.replay import ReplayError, Thresholds, classify, render_md  # noqa: E402

TASK = "refseg"
REPLAY = "replay.jsonl"
META = "replay_meta.json"
TIMING = "logs/replay_timing.jsonl"
ANALYSIS_JSON = "error_analysis.json"
ANALYSIS_MD = "error_analysis.md"
MANIFEST = "LARGE_FILES.manifest.tsv"
BASELINES = {"w1f": (TASK, "w1f"), "ownbox": (TASK, "baseline_ownbox"), "box": ("refexp_box", "baseline_e1")}


def parse_args(argv=None):
    d = Thresholds()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", help="W1 运行的目录（如 experiments/W1/0063_w1_devcheck_hpcgpu07）；不给时只做 --classify")
    p.add_argument("--out", required=True, help="输出目录")
    p.add_argument("--model", help="模型键（<run> 下的子目录；不给时 <run> 下必须恰有一个模型目录）")
    p.add_argument("--setting", default="full", help="回路设置（默认 full，即 W1-1）")
    p.add_argument("--split", default="devcheck", help="划分（默认 devcheck）")
    p.add_argument("--data-root", help="落地目录（默认环境变量 VISINT_DATA_ROOT；运行身份是 tiny 时可不给，用 tests/fixtures）")
    p.add_argument("--workers", type=int, default=1, help="并行的进程数（默认 1；结果与 1 逐字节相同）")
    p.add_argument("--limit", type=int, help="只重放前 N 条（调试；分类照常做，报告里注明）")
    p.add_argument("--force", action="store_true", help="覆盖 --out 里已有的 replay.jsonl")
    p.add_argument("--classify", action="store_true", help="分类，写 error_analysis.{json,md}（不给 --run 时只读 <out>/replay.jsonl）")
    p.add_argument("--f1-iou", type=float, default=d.f1_iou, help=f"F1：final_iou < 这个值（默认 {d.f1_iou:g}）")
    p.add_argument("--f2-margin", type=float, default=d.f2_margin, help=f"F2：final_iou < w1f_iou − 这个值（默认 {d.f2_margin:g}）")
    p.add_argument("--degrade", type=float, default=d.degrade, help=f"修改把好的结果改坏：best_iou − final_iou ≥ 这个值（默认 {d.degrade:g}）")
    p.add_argument("--seed-iou", type=float, default=d.seed_iou, help=f"种子点放错：seed_box_iou_gt < 这个值（默认 {d.seed_iou:g}）")
    p.add_argument("--delta-eps", type=float, default=d.delta_eps, help=f"IoU 不变：|Δ| ≤ 这个值（默认 {d.delta_eps:g}）")
    a = p.parse_args(argv)
    if not a.run and not a.classify:
        p.error("要么给 --run（重放），要么给 --classify（只用已有的 replay.jsonl 分类）")
    if a.workers < 1 or (a.limit is not None and a.limit < 1):
        p.error("--workers 与 --limit 必须是正整数")
    try:
        a.thresholds = Thresholds(f1_iou=a.f1_iou, f2_margin=a.f2_margin, degrade=a.degrade, seed_iou=a.seed_iou, delta_eps=a.delta_eps)
    except ValueError as e:
        p.error(str(e))
    return a


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(p: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def by_id(recs: Sequence[Mapping[str, Any]], what: str) -> dict[str, Mapping[str, Any]]:
    out: dict[str, Mapping[str, Any]] = {}
    for r in recs:
        if r["id"] in out:
            raise ReplayError(f"{what}：id {r['id']} 重复")
        out[r["id"]] = r
    return out


def _rel(p: Path, root: Path) -> str:
    try:
        return str(p.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(p)


def model_dir(run: Path, model: str | None) -> str:
    if model:
        if not (run / model).is_dir():
            raise ReplayError(f"{run} 下没有模型目录 {model}")
        return model
    ms = sorted(x.name for x in run.iterdir() if x.is_dir() and (x / TASK).is_dir())
    if len(ms) != 1:
        raise ReplayError(f"{run} 下的模型目录有 {ms}，请用 --model 指定")
    return ms[0]


def check_manifest(run: Path, pred: Path, sha: str) -> str | None:
    """<run>/LARGE_FILES.manifest.tsv 里列了这个文件时核对 sha256（大文件不入 git，服务器上是原件）。返回清单里的 sha256 或 None。"""
    mf = run / MANIFEST
    if not mf.exists():
        return None
    rel = _rel(pred, run)
    for line in mf.read_text(encoding="utf-8").splitlines()[1:]:
        parts = line.split("\t")
        if parts and parts[0] == rel:
            if parts[2] != sha:
                raise ReplayError(f"{pred} 的 sha256 {sha[:12]}… 与 {mf} 里的 {parts[2][:12]}… 不同")
            return parts[2]
    return None


def load_samples(cfg: Mapping[str, Any], split: str, data_root: str | None):
    """运行身份 → (样本, 划分清单的指纹)：划分清单与运行身份里的 splits_sha256 必须相同。"""
    from src.e1.tasks import DataPaths
    from src.w1.config import load_w1_config
    from src.w1.data import W1_SPLITS_PATH, load_splits, splits_sha256, tiny_splits, w1_items

    tiny = bool(cfg["model"].get("tiny"))
    doc = tiny_splits(load_w1_config()) if tiny else load_splits(W1_SPLITS_PATH)
    if splits_sha256(doc) != cfg["w1"]["splits_sha256"]:
        raise ReplayError(f"划分清单的指纹 {splits_sha256(doc)[:12]}… 与运行身份里的 {cfg['w1']['splits_sha256'][:12]}… 不同")
    root = data_root or os.environ.get("VISINT_DATA_ROOT")
    if root:
        paths = DataPaths(root=Path(root))
    elif tiny:  # 同 scripts/run_w1.py 的 data_paths：tiny 的 RefCOCO 用放大到 48×36 的事后读数 fixture
        paths = DataPaths.posthoc_fixtures()
    else:
        raise ReplayError("要给 --data-root（或设 VISINT_DATA_ROOT）")
    return w1_items(TASK, split, paths, splits=doc), splits_sha256(doc)


def build_jobs(a, run: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import yaml

    from src.w1.replay import workbench_kwargs

    model = model_dir(run, a.model)
    leaf = run / model / TASK / a.setting / a.split
    pred = leaf / "predictions.jsonl"
    for f in (pred, leaf / "config.yaml"):
        if not f.exists():
            raise ReplayError(f"缺少 {f}")
    cfg = yaml.safe_load((leaf / "config.yaml").read_text(encoding="utf-8"))
    w1 = cfg.get("w1") or {}
    if (w1.get("task"), w1.get("setting"), w1.get("split")) != (TASK, a.setting, a.split):
        raise ReplayError(f"{leaf}：运行身份是 {w1.get('task')}/{w1.get('setting')}/{w1.get('split')}，不是 {TASK}/{a.setting}/{a.split}")
    pred_sha = sha256_file(pred)
    manifest_sha = check_manifest(run, pred, pred_sha)
    recs = read_jsonl(pred)
    samples, split_sha = load_samples(cfg, a.split, a.data_root)
    ids = [s.id for s in samples]
    if [r["id"] for r in recs] != ids:
        raise ReplayError(f"{pred} 的 id（{len(recs)} 条）与划分 {a.split} 的样本（{len(ids)} 条）不是同一列")
    inputs: dict[str, Any] = {"predictions": {"path": _rel(pred, run), "sha256": pred_sha, "manifest_checked": manifest_sha is not None},
                              "config": {"path": _rel(leaf / "config.yaml", run), "sha256": sha256_file(leaf / "config.yaml")}}
    base: dict[str, dict[str, Mapping[str, Any]]] = {}
    for key, (task, setting) in BASELINES.items():
        bp = run / model / task / setting / a.split / "predictions.jsonl"
        if not bp.exists():
            raise ReplayError(f"缺少基线 {key}：{bp}")
        brecs = by_id(read_jsonl(bp), str(bp))
        missing = [i for i in ids if i not in brecs]
        if missing:
            raise ReplayError(f"基线 {key}（{bp}）缺 {len(missing)} 条，如 {missing[:3]}")
        base[key] = brecs
        inputs[key] = {"path": _rel(bp, run), "sha256": sha256_file(bp)}
    factory = cfg["workbench"]["factory"]
    wkw = workbench_kwargs(cfg)
    n = len(recs) if a.limit is None else min(a.limit, len(recs))
    jobs = []
    for r, s in list(zip(recs, samples))[:n]:
        box = base["box"][s.id]
        jobs.append({"record": r, "sample": s, "factory": factory, "workbench_kw": wkw,
                     "own_box": (box.get("parsed") or {}).get("box"), "own_box_iou_gt": (box.get("score") or {}).get("iou"),
                     "w1f_score": base["w1f"][s.id]["score"], "ownbox_score": base["ownbox"][s.id]["score"]})
    source = {"run": str(run), "model": model, "task": TASK, "setting": a.setting, "split": a.split, "tiny": bool(cfg["model"].get("tiny")),
              "revision": cfg["model"].get("revision"), "tools_sha256": cfg["workbench"]["tools_sha256"], "workbench_factory": factory,
              "workbench_kwargs": {k: (sorted(v) if isinstance(v, frozenset) else v) for k, v in wkw.items()},
              "split_sha256": w1.get("split_sha256"), "splits_sha256": split_sha, "predictions_sha256": pred_sha,
              "n_records": len(recs), "n_replayed": n, "limit": a.limit, "inputs": inputs}
    return jobs, source


def run_jobs(jobs: Sequence[Mapping[str, Any]], workers: int, log) -> list[tuple[dict[str, Any], float]]:
    from src.w1.replay import replay_job

    out: list[tuple[dict[str, Any], float]] = []
    if workers == 1:
        it = map(replay_job, jobs)
        pool = None
    else:
        pool = cf.ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"))
        it = pool.map(replay_job, jobs, chunksize=1)
    try:
        t0 = time.perf_counter()
        for i, res in enumerate(it, start=1):
            out.append(res)
            if i % 25 == 0 or i == len(jobs):
                log(f"[w1_replay] {i}/{len(jobs)}（{time.perf_counter() - t0:.0f} s）")
    finally:
        if pool is not None:
            pool.shutdown(wait=True)
    return out


def _lib_versions() -> dict[str, str]:
    out = {"python": platform.python_version(), "platform": platform.platform(), "host": platform.node()}
    for mod in ("numpy", "cv2", "skimage", "scipy", "pycocotools"):
        try:
            m = __import__(mod)
            out[mod] = str(getattr(m, "__version__", "?"))
        except ImportError:
            out[mod] = "missing"
    return out


def _write(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(p)


def do_replay(a, out: Path, log) -> dict[str, Any]:
    if (out / REPLAY).exists() and not a.force:
        raise ReplayError(f"{out / REPLAY} 已存在；覆盖请加 --force（只重新分类请只给 --classify --out）")
    run = Path(a.run)
    jobs, source = build_jobs(a, run)
    log(f"[w1_replay] {source['model']} {TASK}/{a.setting}/{a.split}：重放 {len(jobs)} 条，--workers {a.workers}")
    t0 = time.perf_counter()
    results = run_jobs(jobs, a.workers, log)
    wall = time.perf_counter() - t0
    rows = [r for r, _ in results]
    _write(out / REPLAY, "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows))
    _write(out / TIMING, "".join(json.dumps({"id": r["id"], "wall_s": t}) + "\n" for r, t in results))
    bad = [r for r in rows if r["replay"]["mismatch"]]
    meta = {"source": source, "n_rows": len(rows), "n_replay_mismatch": len(bad),
            "replay_mismatch": [{"id": r["id"], "reasons": r["replay"]["reasons"]} for r in bad],
            "n_image_mismatch_calls": sum(r["replay"]["n_image_mismatch"] for r in rows),
            "replay_sha256": sha256_file(out / REPLAY), "workers": a.workers, "wall_s": wall,
            "sum_sample_wall_s": float(sum(t for _, t in results)), "libs": _lib_versions(),
            "written_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "argv": sys.argv[1:]}
    _write(out / META, json.dumps(meta, indent=1, ensure_ascii=False) + "\n")
    log(f"[w1_replay] 写出 {out / REPLAY}：{len(rows)} 条，重放不一致 {len(bad)} 条，墙钟 {wall:.0f} s")
    return meta


def do_classify(a, out: Path, log) -> dict[str, Any]:
    p = out / REPLAY
    if not p.exists():
        raise ReplayError(f"缺少 {p}（先重放：给 --run）")
    rows = read_jsonl(p)
    source: dict[str, Any] = {}
    if (out / META).exists():
        meta = json.loads((out / META).read_text(encoding="utf-8"))
        source = {k: v for k, v in meta["source"].items() if k != "inputs"}
        if meta.get("replay_sha256") not in (None, sha256_file(p)):
            raise ReplayError(f"{p} 与 {out / META} 记的 sha256 不同（replay.jsonl 被改过）")
    source["replay_sha256"] = sha256_file(p)
    doc = classify(rows, a.thresholds, source=source)
    _write(out / ANALYSIS_JSON, json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    _write(out / ANALYSIS_MD, render_md(doc))
    f1, f2 = doc["failure_defs"]["F1"], doc["failure_defs"]["F2"]
    log(f"[w1_replay] 分类：进入分析 {doc['n_analyzed']} 条；F1 失败 {f1['n_failed']} 条 "
        f"{ {c: x['n'] for c, x in f1['classes'].items()} }；F2 失败 {f2['n_failed']} 条 {({c: x['n'] for c, x in f2['classes'].items()})}")
    return doc


def main(argv=None) -> int:
    a = parse_args(argv)
    out = Path(a.out)
    log = lambda m: print(m, file=sys.stderr, flush=True)  # noqa: E731
    try:
        meta = do_replay(a, out, log) if a.run else None
        if a.classify:
            do_classify(a, out, log)
    except (ReplayError, OSError, ValueError, KeyError) as e:
        print(f"失败：{type(e).__name__}: {e}", file=sys.stderr)
        return 2
    except Exception:  # noqa: BLE001 — 意外的错误：留下完整的 traceback 给服务器的报告
        traceback.print_exc()
        return 2
    if meta is not None:
        print(f"w1_replay n={meta['n_rows']} replay_mismatch={meta['n_replay_mismatch']} wall_s={meta['wall_s']:.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
