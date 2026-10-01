#!/usr/bin/env python
"""E1 数据落地（DEV-0012 任务书 §1）。每个数据源一个子命令：

    python scripts/data/land_e1.py <源> --root $PROJ/data [--tmpdir $TMPDIR] [--budget-gb 10] [--min-free-gb 3]
                                   [--workers 8] [--limit N] [--dry-run]

源：refcoco、coco_val2017、odinw13、countbench、nyuv2、blink_reldepth、cvbench、embspatial、all（按此顺序逐个）。
另有可选的源 countbenchqa_questions（DEV-0015：big_vision 的 CountBench 人工问句，31 KB；E1 默认不用它），只能单独点名，
不在 all 里。

- 可断点续传、幂等：已落地且 sha256 与上次清单相符的文件跳过；下载先写 .part，续传用 Range。
- 每个源写 <root>/<源>/MANIFEST.json（来源 URL 与 revision、每个文件的 sha256 与字节数、许可原文、落地时间、脚本的
  git commit、开跑前的估算、每一步的 df、不可达的 URL）。
- 开跑前估算体积：超过 --budget-gb 或会让根盘剩余低于 --min-free-gb（D6 红线 3 GB）就拒绝；运行中跌破红线立即停止。
- 大文件只进 TMPDIR（服务器上是 /dev/shm 下的项目目录），用完即删；zip 一律用 Range 只取需要的成员。
- 并发 ≤ 8，失败按指数退避重试，重试用尽的 URL 逐条记进清单 failures，退出码 2。
- --limit N：开发机小样本（每个源只落 N 条），NYUv2 另加 --nyu-remote 用 Range 读远端 .mat，不下载。
- cvbench 放在 coco_val2017 与 nyuv2 之后跑，它的清单会记与 COCO 500 子集、NYU 官方 test 的重合。

退出码：0 成功；2 有不可达的 URL 或核对不通过（清单已写）；3 预算拒绝；4 跌破磁盘红线。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.data.land import coco, countbenchqa, hfbench, nyuv2, odinw, refcoco  # noqa: E402
from src.data.land.common import DryRun, LandContext  # noqa: E402
from src.data.land.manifest import BudgetExceeded, DiskRedLine, read_manifest  # noqa: E402
from src.data.land.net import MAX_CONCURRENCY  # noqa: E402

LANDERS = {
    "refcoco": refcoco.land,
    "coco_val2017": coco.land,
    "odinw13": odinw.land,
    "countbench": hfbench.land_countbench,
    "nyuv2": nyuv2.land,
    "blink_reldepth": hfbench.land_blink,
    "cvbench": hfbench.land_cvbench,
    "embspatial": hfbench.land_embspatial,
}
# 可选的源：只能单独点名，不在 all 里（DEV-0015 §4：CountBench 人工问句默认不用，项目负责人批准后再定）
OPTIONAL_LANDERS = {
    "countbenchqa_questions": countbenchqa.land,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", choices=[*LANDERS, *OPTIONAL_LANDERS, "all"])
    ap.add_argument("--root", required=True, type=Path, help="落地根目录（服务器：$PROJ/data）")
    ap.add_argument("--tmpdir", type=Path, default=Path(os.environ.get("TMPDIR", "/tmp")), help="大文件与压缩包的临时目录（默认 $TMPDIR）")
    ap.add_argument("--budget-gb", type=float, default=10.0, help="单个源的落地体积上限（GB，1e9），超过就拒绝开跑")
    ap.add_argument("--min-free-gb", type=float, default=3.0, help="根盘剩余红线（D6），开跑前与运行中都检查")
    ap.add_argument("--workers", type=int, default=MAX_CONCURRENCY, help=f"并发数（1..{MAX_CONCURRENCY}）")
    ap.add_argument("--limit", type=int, default=None, help="开发机小样本：每个源只落 N 条")
    ap.add_argument("--retries", type=int, default=5)
    ap.add_argument("--dry-run", action="store_true", help="只做估算与预算检查")
    ap.add_argument("--refcoco-datasets", nargs="+", choices=list(refcoco.SPEC["datasets"]), default=None)
    ap.add_argument("--skip-mirror-check", action="store_true", help="RefCOCO：跳过与 HF 镜像的逐条比对（只在开发机小样本时用）")
    ap.add_argument("--odinw-configs", nargs="+", choices=list(odinw.CONFIGS), default=None)
    ap.add_argument("--nyu-local-dir", type=Path, default=None, help=f"本地已有 .mat 的目录（默认 {nyuv2.SPEC['local_dir']}）")
    ap.add_argument("--nyu-verify", choices=["stream", "none"], default="stream")
    ap.add_argument("--nyu-remote", action="store_true", help="开发机：Range 读远端 .mat（配合 --limit）")
    args = ap.parse_args(argv)
    if not 1 <= args.workers <= MAX_CONCURRENCY:
        ap.error(f"--workers 必须在 1..{MAX_CONCURRENCY}")
    return args


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    ctx = LandContext(
        root=a.root,
        tmpdir=a.tmpdir,
        budget_gb=a.budget_gb,
        min_free_gb=a.min_free_gb,
        workers=a.workers,
        limit=a.limit,
        retries=a.retries,
        dry_run=a.dry_run,
        options={
            "refcoco_datasets": a.refcoco_datasets,
            "skip_mirror_check": a.skip_mirror_check,
            "odinw_configs": a.odinw_configs,
            "nyu_local_dir": a.nyu_local_dir,
            "nyu_verify": a.nyu_verify,
            "nyu_remote": a.nyu_remote,
        },
    )
    sources = list(LANDERS) if a.source == "all" else [a.source]
    code = 0
    for s in sources:
        print(f"== {s}", file=sys.stderr)
        try:
            {**LANDERS, **OPTIONAL_LANDERS}[s](ctx)
        except DryRun:
            print(f"[dry-run] {s}：估算通过，未下载", file=sys.stderr)
            continue
        except BudgetExceeded as e:
            print(f"[拒绝] {s}：{e}", file=sys.stderr)
            return 3
        except DiskRedLine as e:
            print(f"[红线] {s}：{e}", file=sys.stderr)
            return 4
        man = read_manifest(ctx.source_dir(s))
        bad = [v for v in man["verification"] if not v["ok"]]
        print(
            f"[完成] {s}：{man['n_files']} 个文件，{man['total_bytes'] / 1e9:.3f} GB；失败 {len(man['failures'])}；核对不通过 {len(bad)}",
            file=sys.stderr,
        )
        if man["failures"] or bad:
            code = 2
    return code


if __name__ == "__main__":
    sys.exit(main())
