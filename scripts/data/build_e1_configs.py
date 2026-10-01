#!/usr/bin/env python
"""生成或核对 E1 的两个协议文件（DEV-0012 任务书 §2、§3）：

    # 开发子集：从落地目录读各基准的总体 id，按 src/data/subsets.py 的规则抽，写 configs/e1_dev_subsets.json
    python scripts/data/build_e1_configs.py dev-subsets --root $PROJ/data [--check]
    # NYUv2 点对：从落地的 NYUv2（Eigen test 654 + 开发子集 50 张 train）由 GT 构造，写 configs/e1_nyu_pairs.json
    python scripts/data/build_e1_configs.py nyu-pairs --root $PROJ/data [--check]
    # 评测子集（多边形、16×16 网格、思考、8192 敏感性、附加对照）：从剔除开发子集与 D20 后的评测总体里抽，
    # 写 configs/e1_eval_subsets.json；需要全部源落地，且开发子集与点对两个配置已就绪
    python scripts/data/build_e1_configs.py eval-subsets --root $PROJ/data [--check]

--check：不写文件，重算后与入库的文件比较；不一致时打印差异、退出码 1。
dev-subsets --check 比较每个子集的 ids、k、n_population、population_sha256（不比较出处说明文字）。
nyu-pairs 需要 NYUv2 全部落地（654 + 50 张）；--check 逐字节比较。
eval-subsets --check 比较每个总体的 n 与指纹、每个子集的 ids、k、n_population、population_sha256。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.data.subsets import DEV_SUBSETS_PATH, build_dev_subsets, dump_config  # noqa: E402


def cmd_dev_subsets(a: argparse.Namespace) -> int:
    from src.data.loaders import populations

    pops, prov = populations(a.root, a.only)
    cfg = build_dev_subsets(pops, prov)
    out = Path(a.out or DEV_SUBSETS_PATH)
    if a.check:
        old = json.loads(out.read_text())
        bad = []
        for name, sub in cfg["subsets"].items():
            o = old["subsets"].get(name)
            if o is None:
                bad.append(f"{name}：入库文件里没有")
                continue
            for k in ("ids", "k", "n_population", "population_sha256"):
                if o[k] != sub[k]:
                    bad.append(f"{name}.{k}：入库 {str(o[k])[:80]} ≠ 重算 {str(sub[k])[:80]}")
        for line in bad:
            print(line)
        print("一致" if not bad else f"不一致：{len(bad)} 处")
        return 1 if bad else 0
    if a.only:
        old = json.loads(out.read_text()) if out.exists() else {"subsets": {}}
        merged = dict(cfg)
        merged["subsets"] = {**old.get("subsets", {}), **cfg["subsets"]}
        merged["subsets"] = dict(sorted(merged["subsets"].items()))
        cfg = merged
    out.write_text(dump_config(cfg))
    print(f"已写 {out}：{ {k: v['k'] for k, v in cfg['subsets'].items()} }")
    return 0


def cmd_nyu_pairs(a: argparse.Namespace) -> int:
    import numpy as np
    from PIL import Image

    from src.data.land.manifest import read_manifest
    from src.data.nyu_pairs import PAIRS_PATH, build_pairs, dump_pairs

    src = Path(a.root) / "nyuv2"
    index = json.loads((src / "index.json").read_text())["images"]
    if not a.allow_partial:
        n_test = sum(1 for r in index if r["split"] == "test")
        if n_test != 654:
            print(f"NYUv2 只落了 {n_test}/654 张测试图；点对要在完整落地后生成（--allow-partial 只供开发机试跑）")
            return 2
    images, rgbs = [], {}
    for r in index:
        with np.load(src / r["depth"]) as d:
            from src.data.constants import NYU_PAIR_DEPTH_KEY

            images.append((r["idx"], r["split"], d[NYU_PAIR_DEPTH_KEY]))
        rgbs[r["idx"]] = np.asarray(Image.open(src / r["rgb"]).convert("RGB"))
    cfg = build_pairs(images, render=rgbs)
    man = read_manifest(src)
    cfg["source"] = {
        "nyuv2_index_sha256": man["files"]["index.json"]["sha256"],
        "mat": {k: v for k, v in man["upstream"][0].items() if k in ("url", "bytes", "sha256", "official_check")},
    }
    text = dump_pairs(cfg)
    out = Path(a.out or PAIRS_PATH)
    if a.check:
        same = out.read_text() == text
        print("一致" if same else "不一致")
        return 0 if same else 1
    out.write_text(text)
    print(f"已写 {out}：{json.dumps(cfg['stats'], ensure_ascii=False)}")
    return 0


def cmd_eval_subsets(a: argparse.Namespace) -> int:
    from src.data.eval_subsets import EVAL_SUBSETS_PATH, build_eval_subsets, compare, eval_populations

    pops = eval_populations(a.root, dev_config=a.dev_config, pairs_config=a.pairs_config)
    cfg = build_eval_subsets(pops)
    out = Path(a.out or EVAL_SUBSETS_PATH)
    if a.check:
        bad = compare(cfg, json.loads(out.read_text()))
        for line in bad:
            print(line)
        print("一致" if not bad else f"不一致：{len(bad)} 处")
        return 1 if bad else 0
    out.write_text(dump_config(cfg))
    print(f"已写 {out}：{ {k: v['k'] for k, v in cfg['subsets'].items()} }")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dev-subsets")
    d.add_argument("--root", required=True, type=Path)
    d.add_argument("--out", type=Path, default=None)
    d.add_argument("--check", action="store_true")
    d.add_argument("--only", nargs="+", default=None, help="只重算这些子集（其余沿用入库文件）")
    n = sub.add_parser("nyu-pairs")
    n.add_argument("--root", required=True, type=Path)
    n.add_argument("--out", type=Path, default=None)
    n.add_argument("--check", action="store_true")
    n.add_argument("--allow-partial", action="store_true")
    e = sub.add_parser("eval-subsets")
    e.add_argument("--root", required=True, type=Path)
    e.add_argument("--out", type=Path, default=None)
    e.add_argument("--check", action="store_true")
    e.add_argument("--dev-config", type=Path, default=None, help="默认 configs/e1_dev_subsets.json")
    e.add_argument("--pairs-config", type=Path, default=None, help="默认 configs/e1_nyu_pairs.json")
    a = ap.parse_args(argv)
    return {"dev-subsets": cmd_dev_subsets, "nyu-pairs": cmd_nyu_pairs, "eval-subsets": cmd_eval_subsets}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
