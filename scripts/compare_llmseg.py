#!/usr/bin/env python
"""LLMSeg 与 W2-c 的配对比较（DEV-0066；D125-4）：同样的 300 句上 LLMSeg − W2-c 的 cIoU 差与 gIoU 差（配对 bootstrap 95% CI），
各划分与合并（5 个划分的句子放在一起）；LLMSeg 的两组读数都比：rec（修正后，主配置 T = 8，标题）与 seg（修正前，阶段 2 之后）。

    python scripts/compare_llmseg.py \\
        --llmseg experiments/W2/0074_llmseg_hpcgpu06/qwen3.5-9b/refseg/llmseg \\
        --w2c experiments/W2/0072_w2_test_hpcgpu06/qwen3.5-9b/refseg/w2c \\
        --out experiments/W2/0074_llmseg_hpcgpu06/compare_w2c [--splits test_refcoco_unc_testA ...]

--llmseg、--w2c 是"设置"一级的目录，下面每个划分一个子目录（W2 的布局）；默认 5 个 test 划分都要有。W2-c 的读数取 RUN-0072 的结果目录
（config.yaml 的 setting 必须是 w2c 且带 checkpoint）。写出 <out>/compare.json 与 compare.md；退出码 0 = 写完，2 = 出错。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.llmseg import runner as R  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--llmseg", required=True, help="LLMSeg 的设置目录（<out>/<模型>/refseg/llmseg）")
    p.add_argument("--w2c", required=True, help="W2-c 的设置目录（RUN-0072 的 <out>/<模型>/refseg/w2c）")
    p.add_argument("--out", required=True, help="输出目录")
    p.add_argument("--splits", nargs="+", help="默认 W2 的 5 个 test 划分")
    a = p.parse_args(argv)
    splits = a.splits or R.test_splits()
    try:
        pairs = []
        for s in splits:
            lf, wf = Path(a.llmseg) / s, Path(a.w2c) / s
            for d in (lf, wf):
                if not (d / "predictions.jsonl").exists():
                    raise FileNotFoundError(f"{d / 'predictions.jsonl'} 不存在")
            pairs.append((s, lf, wf))
        res = R.compare_runs(pairs)
        res["inputs"] = {"llmseg": str(a.llmseg), "w2c": str(a.w2c), "splits": list(splits)}
    except (OSError, ValueError, KeyError) as e:
        print(f"失败：{e!r}", file=sys.stderr)
        return 2
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "compare.json").write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    md = R.compare_markdown(res)
    (out / "compare.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
