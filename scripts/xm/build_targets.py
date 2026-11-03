#!/usr/bin/env python
"""X-M 的训练目标（DEV-0079）：COCO train2017（剔除去重清单之后）的每个"图像 × 类别"生成框列表与每实例 B 样条两个文本目标。CPU，不跑模型。
规则见 src/xm/targets.py；对由 src/xm/data.load_pairs("train") 给出（按去重清单自动剔除）。

服务器（HPC；在 06 或 07 的分配里用 CPU：srun --jobid=… --overlap -n1 …；--out 是本次运行的目录）：
    python scripts/xm/build_targets.py --data-root $VI/data --splits configs/xm_splits.json --exclusions configs/xm_train_exclusions.json \\
        --out experiments/XM/<NNNN>_xm_targets --workers 14 [--resume] [--stop-after-shards N] [--limit N]
    # 输出 <out>/xm_targets/targets_<k>.jsonl（每片 20,000 对）与 manifest.json、<out>/config.yaml、<out>/logs/。
    # 用时：读 instances_train2017（448 MB）并把约 86 万个标注转 RLE 约 2–4 分钟、内存约 6–8 GB；每个实例一次真值拟合（X-2 的
    #   oracle_spline，640×480 的整图）+ 一次框约束的光栅化，开发机单进程约 5–10 ms / 实例；约 34 万对、约 84 万个保留的实例，单进程约
    #   1.5–2.5 CPU 小时，--workers 14 约 8–15 分钟。先 --limit 2000 --out experiments/XM/<NNNN>_xm_targets_probe 量一次再外推。
    #   输出约 34 万行，约 1–1.5 GB。可续跑：--resume 跳过已写完的片（核对 id），与一次跑完逐字节相同（logs/ 除外）。
开发机彩排（先 scripts/xm/build_configs.py --tiny --out D1）：
    python scripts/xm/build_targets.py --tiny --data-root D1/fake_root --splits D1/xm_splits.json --exclusions D1/xm_train_exclusions.json --out <dir>
--out 的最后一级目录名必须以四位运行编号开头（--tiny 不查）。退出码：0 = 写完或按 --stop-after-shards 停下；2 = 用法或数据错误。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.xm import data as D  # noqa: E402
from src.xm import exclusions as X  # noqa: E402
from src.xm import targets as T  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", required=True, help="RUN-0102 的落地（$VI/data）")
    p.add_argument("--splits", default=str(D.XM_SPLITS_PATH), help="划分清单（默认 configs/xm_splits.json）")
    p.add_argument("--exclusions", default=str(D.XM_EXCLUSIONS_PATH), help="去重清单（默认 configs/xm_train_exclusions.json）")
    p.add_argument("--out", required=True, help="本次运行的目录（experiments/XM/<NNNN_slug>）")
    p.add_argument("--workers", type=int, default=1, help="进程数（默认 1；>1 时 spawn 的进程池，结果与单进程相同）")
    p.add_argument("--resume", action="store_true", help="续跑：跳过已写完的分片（参数与输入必须相同）")
    p.add_argument("--stop-after-shards", type=int, help="本次只做 N 片（下次 --resume 接着做）")
    p.add_argument("--limit", type=int, help="只做按 id 顺序的前 N 对（量吞吐用；进运行身份）")
    p.add_argument("--shard-pairs", type=int, default=T.SHARD_PAIRS, help=f"每片的对数（默认 {T.SHARD_PAIRS}）")
    p.add_argument("--tiny", action="store_true", help="开发机彩排（不查 --out 的运行编号）")
    a = p.parse_args(argv)
    for k in ("workers", "stop_after_shards", "limit", "shard_pairs"):
        v = getattr(a, k)
        if v is not None and v < 1:
            p.error(f"--{k.replace('_', '-')} 必须是正整数")
    if not a.tiny:
        try:
            D.check_run_dir(a.out)
        except D.XMDataError as e:
            p.error(str(e))
    return a


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    a = parse_args(argv)
    try:
        ex = X.load_exclusions(a.exclusions, splits_path=a.splits)
        pairs = D.load_pairs(D.TRAIN, a.data_root, splits_path=a.splits, exclusions_path=a.exclusions)
        if a.limit is not None:
            pairs = pairs[: a.limit]
        src = T.source_identity(pairs, splits_path=a.splits, exclusions_path=a.exclusions,
                                train_json_sha256=ex["sources"]["coco_train2017"]["sha256"])
        src["limit"] = a.limit
        man = T.build(pairs, a.out, params=T.DEFAULT_PARAMS, source=src, tiny=a.tiny, workers=a.workers, resume=a.resume,
                      shard_pairs=a.shard_pairs, stop_after_shards=a.stop_after_shards, argv=argv)
    except (D.XMDataError, T.XMTargetsError, OSError, ValueError) as e:
        print(f"失败：{type(e).__name__}: {e}", file=sys.stderr)
        return 2
    if man is None:
        print("按 --stop-after-shards 停下；加 --resume 接着跑")
        return 0
    c = man["counts"]
    fi = c["fit_iou"] or {}
    print(f"X-M 目标：{c['n_pairs']} 对、{c['n_images']} 张图、{c['n_instances']} 个实例（保留 {c['n_instances_kept']}）；超过 "
          f"{T.MAX_BOXES} 个的对 {c['n_truncated_pairs']}（{100 * (c['truncated_rate'] or 0):.2f}%）；样条可用 {c['n_spline_ok']} 对，"
          f"失败的实例 {c['spline_fail_instances']}；fit_iou 中位数 {fi.get('p50')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
