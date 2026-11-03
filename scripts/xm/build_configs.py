#!/usr/bin/env python
"""X-M 的两份清单（DEV-0079；D165 指令第二节）：评测划分 xm_splits.json 与训练去重清单 xm_train_exclusions.json。只读真值，CPU，不跑模型。
规则见 src/xm/data.py（划分、密封、LVIS 子集、ReasonSeg）与 src/xm/exclusions.py（去重）。

服务器（HPC；--data-root 是 RUN-0102 落地的 $VI/data，--refcoco-root 是 E1 的落地 $VI/b200/data；--out 是本次运行的目录）：
    python scripts/xm/build_configs.py --data-root $VI/data --refcoco-root $VI/b200/data \\
        --out experiments/XM/<NNNN>_xm_configs --workers 12
    # 输出 <out>/xm_splits.json、<out>/xm_train_exclusions.json（dump_config 的规范文本）与 <out>/build_log.json（耗时、argv）。
    # 管理员审过后原样拷到 configs/xm_splits.json 与 configs/xm_train_exclusions.json（两者互相钉住：去重清单记划分文件的 sha256）。
    # 用时：划分约 1–2 分钟（instances_val2017 20 MB、lvis_v1_val 约 200 MB 的 JSON，1,000 张 LVIS 图的多边形转 RLE）；去重的大头是
    #   train2017 的 118,287 张图（约 19 GB）逐张 sha256，Lustre 上 --workers 12 约 5–15 分钟；instances_train2017（448 MB）读入约 1 分钟、
    #   内存约 6 GB。合计约 10–20 分钟。
开发机彩排（--tiny：数据根缺省用 tests/_xm_fakes 写到 --out 下的 fake_root，RefCOCO 用 tests/fixtures/explore/root）：
    python scripts/xm/build_configs.py --tiny --out <dir>
--out 的最后一级目录名必须以四位运行编号开头（--tiny 不查）；输出文件已存在时拒绝。退出码：0 = 写完；2 = 用法或数据错误。
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.xm import data as D  # noqa: E402
from src.xm import exclusions as X  # noqa: E402

SPLITS_FILE = "xm_splits.json"
EXCLUSIONS_FILE = "xm_train_exclusions.json"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", help="RUN-0102 的落地（含 coco2017/、lvis/、reasonseg/）；--tiny 时缺省在 --out 下现造")
    p.add_argument("--refcoco-root", help="E1 的落地（含 refcoco/<数据集>_<splitBy>.json；HPC 上 $VI/b200/data）；--tiny 时缺省 tests/fixtures/explore/root")
    p.add_argument("--out", required=True, help="本次运行的目录（experiments/XM/<NNNN_slug>）")
    p.add_argument("--workers", type=int, default=8, help="算图像 sha256 的线程数（默认 8）")
    p.add_argument("--tiny", action="store_true", help="开发机彩排：假数据与 fixture 的 RefCOCO，不核对 P0 计数")
    a = p.parse_args(argv)
    if a.workers < 1:
        p.error("--workers 必须是正整数")
    if not a.tiny:
        if not a.data_root or not a.refcoco_root:
            p.error("要给 --data-root 与 --refcoco-root")
        try:
            D.check_run_dir(a.out)
        except D.XMDataError as e:
            p.error(str(e))
    return a


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    a = parse_args(argv)
    out = Path(a.out)
    for f in (SPLITS_FILE, EXCLUSIONS_FILE):
        if (out / f).exists():
            print(f"{out / f} 已存在，不覆盖；换一个 --out", file=sys.stderr)
            return 2
    t0 = time.perf_counter()
    started = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")

    def log(m: str) -> None:
        print(m, file=sys.stderr, flush=True)

    try:
        out.mkdir(parents=True, exist_ok=True)
        root = Path(a.data_root) if a.data_root else None
        if root is None:
            from tests._xm_fakes import build_fake_root

            root = out / "fake_root"
            build_fake_root(root)
        doc = D.build_splits_doc(root, tiny=a.tiny, log=log)
        t_split = time.perf_counter() - t0
        sp = out / SPLITS_FILE
        sp.write_text(D.dump_config(doc), encoding="utf-8")
        ex = X.build_exclusions(root, splits_path=sp, refcoco_root=a.refcoco_root, tiny=a.tiny, workers=a.workers, log=log)
        (out / EXCLUSIONS_FILE).write_text(D.dump_config(ex), encoding="utf-8")
    except (D.XMDataError, OSError, ValueError, KeyError) as e:
        print(f"失败：{type(e).__name__}: {e}", file=sys.stderr)
        return 2
    wall = time.perf_counter() - t0
    (out / "build_log.json").write_text(json.dumps({"argv": argv, "started_utc": started, "wall_s": round(wall, 1),
                                                    "splits_wall_s": round(t_split, 1), "tiny": a.tiny,
                                                    "xm_splits_sha256": D.file_sha256(out / SPLITS_FILE),
                                                    "xm_train_exclusions_sha256": D.file_sha256(out / EXCLUSIONS_FILE)},
                                                   indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    for s, e in doc["splits"].items():
        print(f"{s}：{e['n_images']} 张图、{e['n_pairs']} 对、{e['n_instances']} 个实例（{e['sha256'][:16]}…）")
    for k, v in ex["by_category"].items():
        print(f"去重 {k}：train2017 里剔除 {v['n_train2017_images']} 张")
    rc = ex["categories"]["refcoco_eval"]
    print(f"RefCOCO 评测划分 {rc['n_images']} 张图（在 train2017 {rc['n_in_train2017']} 张）；子集核对 {len(rc['subsets_covered'])} 个，全部在内")
    print(f"去重清单共 {ex['n']} 张（{ex['sha256'][:16]}…）；用时 {wall:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
