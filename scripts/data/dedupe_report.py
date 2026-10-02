#!/usr/bin/env python
"""训练集与评测集之间的去重报告（硬约束 7；D11-7；DEV-0022 任务书 §2）。实现在 src/data/train_dedupe.py。

两步，跨机器：
1. 各自导出"图像键 + 感知哈希"清单（小文件，几 MB）：
       # B200（评测数据，E1 的全部源）
       python scripts/data/dedupe_report.py --export-hashes $PROJ/data --out eval_hashes_b200.json
       # A800（训练数据）
       python scripts/data/dedupe_report.py --export-hashes $DATA --out train_hashes_a800.json
2. 在任意一台机器上比较（--train-root 可换成 --train-hashes <导出的清单>；--eval-manifest 收导出的清单、单个
   MANIFEST.json 或数据根，可给多个）：
       python scripts/data/dedupe_report.py --train-root $DATA --eval-manifest eval_hashes_b200.json --out $DATA/dedupe

输出到 --out 目录：dedupe_report.json、dedupe_report.md、train_exclusions.json。训练加载器（src/data/train_loaders.py）
默认读 <训练数据根>/dedupe/train_exclusions.json，自动排除其中的样本；所以 --out 请用 <训练数据根>/dedupe。
D20 的探针用图与 COCO val2017 全部 5,000 个 id 作为内置评测集自动加入。
train_exclusions.json 是 schema 2（DEV-0024；格式见 src/data/train_dedupe.py 的模块说明）：记 max_hamming、train_sets、
eval_sets（参与比较的评测集）与 inputs（各输入清单的 sha256、主机、集合与张数）；参与比较的训练集都有键（没剔除的为空列表）。
训练加载器遇到缺键的训练集报错；E2 开跑前断言当前任务的评测集在 eval_sets 里。所以要训练的集合都要一次给进来比较。

退出码：0 成功；1 输入有问题。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.data.constants import PHASH_MAX_HAMMING  # noqa: E402
from src.data.train_dedupe import build_report, export_root, load_input, write_outputs  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export-hashes", type=Path, default=None, metavar="ROOT", help="只导出 ROOT 下各源的图像键与感知哈希到 --out（一个 JSON 文件）")
    ap.add_argument("--train-root", type=Path, action="append", default=[], help="训练数据根（A800），直接读各源的 MANIFEST.json；可给多个")
    ap.add_argument("--train-hashes", type=Path, action="append", default=[], help="训练侧导出的哈希清单；可给多个")
    ap.add_argument("--eval-manifest", type=Path, nargs="+", default=[], help="评测侧：导出的哈希清单、单个 MANIFEST.json 或数据根")
    ap.add_argument("--out", type=Path, required=True, help="--export-hashes 时是输出文件；否则是输出目录（建议 <训练数据根>/dedupe）")
    ap.add_argument("--max-hamming", type=int, default=PHASH_MAX_HAMMING, help=f"感知哈希判为重复的汉明距离上限（默认 {PHASH_MAX_HAMMING}）")
    a = ap.parse_args(argv)
    if a.export_hashes is None and not (a.train_root or a.train_hashes):
        ap.error("要么 --export-hashes ROOT，要么给 --train-root / --train-hashes")
    if a.export_hashes is None and not a.eval_manifest:
        ap.error("比较时要给 --eval-manifest")
    return a


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    if a.export_hashes is not None:
        doc = export_root(a.export_hashes)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(doc, ensure_ascii=False, separators=(",", ":")) + "\n")
        n = sum(len(s["items"]) for s in doc["sets"].values())
        print(f"[导出] {a.export_hashes} → {a.out}：{len(doc['sets'])} 个集合，{n} 张图", file=sys.stderr)
        return 0
    inputs = [(f"train-root {p}", load_input(p)) for p in a.train_root]
    inputs += [(f"train-hashes {p}", load_input(p)) for p in a.train_hashes]
    inputs += [(f"eval {p}", load_input(p)) for p in a.eval_manifest]
    report, excl = build_report(inputs, max_hamming=a.max_hamming)
    paths = write_outputs(report, excl, a.out)
    for k, v in report["exclusions_summary"].items():
        print(f"[剔除] {k}：{v} 条", file=sys.stderr)
    print(f"[标定] {report['calibration']['verdict']}", file=sys.stderr)
    print(f"[输出] {', '.join(str(p) for p in paths.values())}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
