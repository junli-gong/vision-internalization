#!/usr/bin/env python
"""E2/E3 训练数据落地（DEV-0022 任务书 §1；D11）。每个数据源一个子命令，约定沿用 scripts/data/land_e1.py：

    python scripts/data/land_train.py <源> --root <数据根> [--tmpdir $TMPDIR] [--budget-gb 10] [--min-free-gb 3]
                                      [--workers 8] [--limit N] [--dry-run] [--local-zip PATH] [--refer-zip-dir DIR]
                                      [--hf-base URL] [--bts-list PATH]

源（all = 按此顺序逐个；--local-zip 只能配单个源）：
- refcoco_train：D11-1 的 RefCOCO 训练图（COCO train2014.zip 的成员，24,407 张，4.15 GB）与三套 refs 的 train 标注；
- nyu_depth_train：D11-2 的 BTS 训练对（HF aradhye/nyu_depth_v2 的成员，22,848 对，3.96 GB）；
- nyu_normals：NYUv2 法向 GT（Bae 2021 / GeoNet，train 795 + test 654，1.01 GB）；
- ade20k：ADEChallengeData2016 的 train 20,210 + val 2,000（0.97 GB）。

- 可断点续传、幂等：zip 成员已在盘上且大小与 CRC32 都与中央目录相同就跳过；文件先写 .part 再改名。
- 每个源写 <root>/<源>/MANIFEST.json（来源与 revision、zip 中央目录的 sha256、逐文件 sha256 与字节数、图像的宽高与
  感知哈希、许可原文、脚本的 git commit、开跑前的估算、每一步的 df、失败的成员）。
- 开跑前估算体积：超过 --budget-gb 或会让剩余低于 --min-free-gb 就拒绝；运行中跌破红线立即停止。
- zip 一律用 Range 只取需要的成员（相邻的合并成一个请求），并发 ≤ 8，失败按指数退避重试，失败的成员逐条记进清单。
- 国内网络的备选：--local-zip（本地 zip，同样核对中央目录 sha256 与 CRC32）、--refer-zip-dir（refer 的三个 zip）、
  --hf-base（默认取 $HF_ENDPOINT，如 https://hf-mirror.com）、--bts-list（BTS 清单的本地副本，sha256 必须相符；默认先用仓库里的 P0 副本）。

退出码：0 成功；2 有失败的成员或核对不通过（清单已写）；3 预算拒绝；4 跌破磁盘红线。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.data.land import ade20k, nyu_depth_train, nyu_normals, refcoco_train  # noqa: E402
from src.data.land.common import DryRun, LandContext  # noqa: E402
from src.data.land.manifest import BudgetExceeded, DiskRedLine, read_manifest  # noqa: E402
from src.data.land.net import MAX_CONCURRENCY  # noqa: E402

LANDERS = {
    "refcoco_train": refcoco_train.land,
    "nyu_depth_train": nyu_depth_train.land,
    "nyu_normals": nyu_normals.land,
    "ade20k": ade20k.land,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", choices=[*LANDERS, "all"])
    ap.add_argument("--root", required=True, type=Path, help="落地根目录（A800 上的数据盘）")
    ap.add_argument("--tmpdir", type=Path, default=Path(os.environ.get("TMPDIR", "/tmp")), help="临时目录（默认 $TMPDIR；refcoco_train 要约 160 MB）")
    ap.add_argument("--budget-gb", type=float, default=10.0, help="单个源的落地体积上限（GB，1e9），超过就拒绝开跑")
    ap.add_argument("--min-free-gb", type=float, default=3.0, help="剩余空间红线，开跑前与运行中都检查")
    ap.add_argument("--workers", type=int, default=MAX_CONCURRENCY, help=f"并发数（1..{MAX_CONCURRENCY}）")
    ap.add_argument("--limit", type=int, default=None, help="开发机小样本：每个源只落 N 条")
    ap.add_argument("--retries", type=int, default=5)
    ap.add_argument("--dry-run", action="store_true", help="只做估算与预算检查（不联网）")
    ap.add_argument("--local-zip", type=Path, default=None, help="用本地已有的 zip 代替远端（train2014.zip / nyu_depth_v2.zip / nyu_dataset.zip / ADEChallengeData2016.zip）")
    ap.add_argument("--refer-zip-dir", type=Path, default=None, help="refcoco_train：本地放着 refcoco.zip、refcoco+.zip、refcocog.zip 的目录")
    ap.add_argument("--hf-base", default=os.environ.get("HF_ENDPOINT") or None, help="nyu_depth_train：HF 的根地址（默认取 $HF_ENDPOINT，再没有就是 https://huggingface.co；国内可用 https://hf-mirror.com）")
    ap.add_argument("--bts-list", type=Path, default=None, help="nyu_depth_train：BTS 训练清单的本地副本（sha256 必须相符）")
    args = ap.parse_args(argv)
    if not 1 <= args.workers <= MAX_CONCURRENCY:
        ap.error(f"--workers 必须在 1..{MAX_CONCURRENCY}")
    if args.local_zip is not None and args.source == "all":
        ap.error("--local-zip 只能配单个源")
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
        options={"local_zip": a.local_zip, "refer_zip_dir": a.refer_zip_dir, "hf_base": a.hf_base, "bts_list": a.bts_list},
    )
    sources = list(LANDERS) if a.source == "all" else [a.source]
    code = 0
    for s in sources:
        print(f"== {s}", file=sys.stderr)
        try:
            LANDERS[s](ctx)
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
