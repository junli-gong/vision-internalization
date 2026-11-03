#!/usr/bin/env python
"""X-M 的 COCO 训练样本（DEV-0081；D165 第三节；预注册 XM-4；D167）：CPU 上从 DEV-0079 的训练目标（src/xm/targets.py：COCO train2017
去重之后，每个"图像 × 类别"一行：阅读顺序的框列表 + 每实例 16 控制点 B 样条）按 hash 抽对，写出列框（coco_det）与每实例样条（coco_inst，
每条至多 12 个框，按阅读顺序切开）两条流的样本行。这一类在这张图里带 crowd 标注的对整对剔除（D167-4）；样条目标不可用的对只进列框流。
不跑模型、不用 GPU。规则见 src/xm/train_data.py；参数取 --config（默认 configs/xm_train.json）的 xm_data 一节（pair_salt、margin）与
coco_inst.max_boxes，要抽多少对按配置的 steps 与 coco_det、coco_inst 的 per_step 算（--steps 可改）。

服务器（CPU；--out 的末级目录名以四位运行编号开头，如 experiments/XM/0104_xm_data；结果 <out>/xm_data/ 与 <out>/config.yaml）：
    python scripts/build_xm_data.py --targets <scripts/xm/build_targets.py 的 --out> --data-root $VI/data \\
        --out experiments/XM/<NNNN>_xm_data [--resume] [--steps N]
    # --targets：DEV-0079 的训练目标（<targets>/xm_targets/targets_<k>.jsonl + manifest.json）；逐片核对 sha256 与每一行（targets.row_problems）。
    # 抽对：去掉带 crowd 的对之后，按 hash_rank(…, "xm_train_pairs", pair_salt) 依次取，直到两条流都够 ceil(2,000 × 6 × 1.2) = 14,400 条
    #   （一个对给 1 条列框、样条可用时 ceil(框数 / 12) 条样条），即约 14,400 个对；manifest 记抽样、剔除的 crowd 对数、只进列框流的对数、
    #   各流条数、框数与 fit_iou 的分布、选中对的 id 指纹。
    # --data-root：样本行里 image 的相对根（DEV-0079 的数据根）；给了就核对选中的图都在，训练时 train_w2c.py --xm-data-root 给同一个。
    # 用时：逐片读两遍训练目标（约 34 万对、几百 MB 的 jsonl；第一遍逐行核对），不看图，约 5–10 分钟；输出约 14,400 + 约 15,000 行，二三十 MB。
    # 可续跑：一个输入分片对应一对输出分片，写完才改名；--resume 跳过已写完的（与重算的逐字节相同才跳过），没写完的重算。

开发机彩排（--tiny：抽对按 tiny.steps × tiny_per_step，不够就全取）：
    python scripts/build_xm_data.py --tiny --targets <scripts/xm/build_targets.py --tiny 的 --out> --out <dir>/xm
退出码：0 = 跑完；2 = 出错（argparse 用法错误也是 2）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.xm import data as XMD  # noqa: E402
from src.xm import train_data as XTD  # noqa: E402

DEFAULT_CONFIG = REPO_ROOT / "configs" / "xm_train.json"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--targets", required=True, help="DEV-0079 的训练目标（scripts/xm/build_targets.py 的 --out，里面有 xm_targets/）")
    p.add_argument("--out", required=True, help="本次运行的目录（末级目录名以四位运行编号开头，如 experiments/XM/0104_xm_data）")
    p.add_argument("--config", default=str(DEFAULT_CONFIG), help="参数取它的 xm_data、coco_det、coco_inst 三节（默认 configs/xm_train.json）")
    p.add_argument("--data-root", help="样本行里 image 的相对根；给了就核对选中的图都在")
    p.add_argument("--steps", type=int, help="按多少步抽（默认配置的 steps；--tiny 时 tiny.steps）")
    p.add_argument("--resume", action="store_true", help="续跑：跳过已写完的分片（参数与输入必须相同）")
    p.add_argument("--tiny", action="store_true", help="开发机彩排：按 tiny 节的规模抽，不够就全取")
    a = p.parse_args(argv)
    if a.steps is not None and a.steps < 1:
        p.error("--steps 必须是正整数")
    if not a.tiny:
        try:
            XMD.check_run_dir(a.out)  # 运行编号的规则只在 src/xm/data.py 一处（DEV-0079）
        except XMD.XMDataError as e:
            p.error(str(e))
    return a


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    a = parse_args(argv)
    try:
        doc = json.loads(Path(a.config).read_text(encoding="utf-8"))
        params = XTD.params_from_config(doc)
        size = XTD.sizing_from_config(doc, tiny=a.tiny, steps=a.steps)
        man = XTD.build(a.targets, a.out, params=params, size=size, tiny=a.tiny, resume=a.resume, data_root=a.data_root, argv=argv)
    except (XTD.XMDataError, OSError, ValueError, KeyError) as e:
        print(f"失败：{e!r}", file=sys.stderr)
        return 2
    c, s = man["counts"], man["selection"]
    print(f"X-M 数据：{s['n_selected_pairs']}/{s['n_eligible_pairs']} 对（共 {s['n_available_pairs']} 对，剔除带 crowd 的 "
          f"{s['n_crowd_dropped']} 对；{c['images']} 张图）→ 列框 {c['coco_det']} 条、每实例样条 {c['coco_inst']} 条（切开的对 "
          f"{c['pairs_split']}，只进列框流的对 {c['pairs_det_only']}）；要 {s['need']}，够了 = {s['satisfied']}；"
          f"每对框数 {man['distributions']['boxes_per_pair']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
