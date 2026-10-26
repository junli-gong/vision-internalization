#!/usr/bin/env python
"""X-2 的训练数据（DEV-0069；探索 X-2；预注册 X）：CPU 上从 W2-c 的训练数据行（DEV-0060 的 w2c_data）算描样条与指修正框的目标，
不跑模型、不用 GPU。规则见 src/explore/x2_data.py；参数取 --config（默认 configs/x2_train.json）的 spline、correct 两节。

服务器（CPU；--out 是本次运行的目录 experiments/X/<NNNN_slug>；结果 <out>/x2_data/ 与 <out>/config.yaml）：
    python scripts/build_x2_data.py --w2c-data experiments/W2/0067_w2c_data_hpcgpu06/w2c_data --data-root $VI/data/train_data \\
        --out experiments/X/<NNNN>_x2_data --workers 12 [--resume] [--stop-after-shards N]
    # --w2c-data：DEV-0060 的 <out>/w2c_data（16 个分片 + manifest.json，RUN-0067/0069 训 W2-c 用的那一份）；--data-root 同 W2-c 的训练
    #   （数据行里 image 的相对根；第 2 轮的修正要读原图跑 GrabCut）。
    # 用时（开发机 Apple M 系列单进程、640×480 的合成样本实测）：只有第 1 轮的句子约 0.012 s；另造第 2 轮的句子（施加第 1 轮的真值修正
    #   要 1–2 次局部 GrabCut）约 0.21 s。16k 句里约一半造第 2 轮，单进程约 30 分钟，--workers 12 约 3–5 分钟。输出约 15k 样条行 + 约 23k
    #   修正行（每行带起点的 RLE），几十 MB。
    # 可续跑：一个输入分片对应一对输出分片，写完才改名；--resume 跳过已写完的（核对 id），没写完的整片重算；除 logs/ 外与一次跑完逐字节相同。
    # 完成后 <out>/x2_data/manifest.json 有：过滤（提出失败、框 IoU < 0.5、真值为空）与各类条数、各文件的 sha256、fit_iou 的分布（描样条
    #   的上限）、目标为 [] 的比例、每个目标的框数、起点的档与 IoU。

开发机彩排（--tiny：不要求 w2c_data 有 manifest.json；--data-root 缺省取 tests/fixtures/e3/root）：
    python -m tests._x2_fakes <dir>/w2c_data            # 造带 gt_rle 的 §5 格式假数据
    python scripts/build_x2_data.py --tiny --w2c-data <dir>/w2c_data --out <dir>/x2
退出码：0 = 跑完或按 --stop-after-shards 停下；2 = 出错（argparse 用法错误也是 2）。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.explore import x2_data as XD  # noqa: E402

DEFAULT_CONFIG = REPO_ROOT / "configs" / "x2_train.json"
TINY_DATA_ROOT = REPO_ROOT / "tests" / "fixtures" / "e3" / "root"  # tests/_w2c_fakes 的图像根（src/e3/data.FIXTURE_REFSEG_ROOT）
RUN_DIR_NAME = re.compile(r"[0-9]{4}_[A-Za-z0-9][A-Za-z0-9_.-]*")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--w2c-data", required=True, help="DEV-0060 的 w2c_data 目录（*.jsonl 分片 + manifest.json）")
    p.add_argument("--data-root", help="数据行里 image 的相对根（默认环境变量 VISINT_DATA_ROOT；--tiny 时取 tests/fixtures/e3/root）")
    p.add_argument("--out", required=True, help="本次运行的目录（experiments/X/<NNNN_slug>）")
    p.add_argument("--config", default=str(DEFAULT_CONFIG), help="参数取它的 spline、correct 两节（默认 configs/x2_train.json）")
    p.add_argument("--workers", type=int, default=1, help="进程数（默认 1；>1 时 spawn 进程池、子进程单线程，结果与单进程相同）")
    p.add_argument("--resume", action="store_true", help="续跑：跳过已写完的分片（参数与输入必须相同）")
    p.add_argument("--stop-after-shards", type=int, help="本次只做 N 个分片（下次 --resume 接着做）")
    p.add_argument("--tiny", action="store_true", help="开发机彩排：不要求 manifest.json，图像根缺省取 fixture")
    a = p.parse_args(argv)
    if a.workers < 1:
        p.error("--workers 必须是正整数")
    if a.stop_after_shards is not None and a.stop_after_shards < 1:
        p.error("--stop-after-shards 必须是正整数")
    if not a.tiny and not (a.data_root or os.environ.get("VISINT_DATA_ROOT")):
        p.error("要给 --data-root（或设 VISINT_DATA_ROOT）")
    if not a.tiny and not RUN_DIR_NAME.fullmatch(Path(a.out).name):
        p.error(f"--out 的目录名必须带运行编号，形如 0080_x2_data，收到 {a.out!r}")
    return a


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    a = parse_args(argv)
    try:
        doc = json.loads(Path(a.config).read_text(encoding="utf-8"))
        params = XD.check_params({k: doc.get(k) for k in ("spline", "correct")})
        root = Path(a.data_root) if a.data_root else (TINY_DATA_ROOT if a.tiny else Path(os.environ["VISINT_DATA_ROOT"]))
        man = XD.build(a.w2c_data, a.out, data_root=root, params=params, tiny=a.tiny, workers=a.workers, resume=a.resume,
                       stop_after_shards=a.stop_after_shards, argv=argv)
    except (XD.X2DataError, OSError, ValueError) as e:
        print(f"失败：{e}", file=sys.stderr)
        return 2
    if man is None:
        print("按 --stop-after-shards 停下；加 --resume 接着跑")
        return 0
    c = man["counts"]
    print(f"X-2 数据：样条 {c['spline']} 行，修正 {c['correct']['total']} 行（第 1 轮 {c['correct']['round1']}、第 2 轮 {c['correct']['round2']}）；"
          f"fit_iou 均值 {man['spline']['fit_iou']['mean'] if man['spline']['fit_iou'] else None}；"
          f"目标为 [] {man['correct']['target_empty']['all']['rate']}；过滤 {man['filters']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
