#!/usr/bin/env python
"""LISA-7B 外部对照（DEV-0075；D154）在我们环境里的两步，加权重的下载与核对。LISA 自己的推理在单独的 venv 里（scripts/run_lisa.py）。

硬约束第 1 条：LISA 的输出只经 score 计分、出表，绝不进入我们的训练数据、标签、奖励或任何挑选。

    # 权重（HPC 登录节点；HF_HUB_CACHE=$VI/hf/hub 是本项目自己的缓存；hf-mirror 时另设 HF_ENDPOINT 与 HF_HUB_DISABLE_XET=1）
    python scripts/lisa_w2.py weights --download        # 按 configs/lisa_weights.json 钉住的 revision 下载并逐文件核对
    python scripts/lisa_w2.py weights --verify          # 只核对（离线），打印两个快照目录
    # 1. 导出清单：W2 test 的 5 × 300 句（configs/w2_llmseg_subsets.json，与 LLMSeg 相同），要批准编号
    python scripts/lisa_w2.py export --out experiments/W2/<NNNN>_lisa_hpcgpu06 --data-root $DATA --test-approval D154
    # 2. （LISA 的 venv）scripts/run_lisa.py --out <同上> --test-approval D154 --lisa-weights … --clip-weights …
    # 3. 计分：W2 的口径，写 <out>/lisa-7b-v1/refseg/lisa_7b/<划分>/
    python scripts/lisa_w2.py score --out <同上> --data-root $DATA [--rescore]
- export：--split <W2 的 test 划分名>|all（默认 all），--limit N（每个划分只取前 N 句，试跑；进清单），--tiny（tests/fixtures 的小数据，
  子集按同一规则现算）。放行：--test-approval 必须恰是 D154，否则退出码 2、什么都不写；放行记录追加进 <out>/lisa-7b-v1/test_unlock.jsonl。
- score：核对原始输出的批准编号、两步的放行记录、清单与官方代码的 sha256；替身运行器的输出要 --allow-stand-in。已计过分要 --rescore。
--out 的最后一级必须是四位运行编号加 _lisa（如 0099_lisa_hpcgpu06）。退出码：0 = 完成；1 = weights 核对不通过；2 = 用法或核对不合格。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.lisa import common as C  # noqa: E402

ALL = "all"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("weights", help="下载或核对 configs/lisa_weights.json 的权重")
    g = w.add_mutually_exclusive_group(required=True)
    g.add_argument("--download", action="store_true")
    g.add_argument("--verify", action="store_true")
    w.add_argument("--manifest", default=str(C.WEIGHTS_MANIFEST_PATH))
    e = sub.add_parser("export", help="导出清单（W2 的取样，句子同 LLMSeg）")
    s = sub.add_parser("score", help="按 W2 的口径计分")
    for q in (e, s):
        q.add_argument("--out", required=True, help=f"本次运行的目录（{C.RUN_DIR_EXAMPLE}）")
        q.add_argument("--data-root", help="落地目录（默认环境变量 VISINT_DATA_ROOT；--tiny 的清单不要）")
    e.add_argument("--test-approval", metavar="ID", help=f"W2 test 子集的批准编号，必须恰是 {C.TEST_APPROVAL}")
    e.add_argument("--split", default=ALL, help=f"W2 的 test 划分名，或 {ALL}")
    e.add_argument("--limit", type=int, help="每个划分只取前 N 句（试跑）")
    e.add_argument("--tiny", action="store_true", help="tests/fixtures 的小数据（开发机彩排）")
    s.add_argument("--allow-stand-in", action="store_true", help="也给替身运行器的输出计分（彩排）")
    s.add_argument("--rescore", action="store_true", help="叶目录已存在时重写")
    a = p.parse_args(argv)
    if a.cmd in ("export", "score"):
        try:
            C.check_run_dir(a.out)
            if a.cmd == "export":
                C.check_approval(a.test_approval)
        except C.LisaError as err:
            p.error(str(err))
    if a.cmd == "export":
        from src.llmseg import runner as LR

        if a.split != ALL and a.split not in LR.test_splits():
            p.error(f"--split 只能是 {ALL} 或 {LR.test_splits()}")
        if a.limit is not None and a.limit < 1:
            p.error("--limit 必须是正整数")
        if not a.tiny and not (a.data_root or os.environ.get("VISINT_DATA_ROOT")):
            p.error("真数据要给 --data-root（或设 VISINT_DATA_ROOT）")
    return a


def weights_cmd(a) -> int:
    from huggingface_hub import snapshot_download

    doc = C.load_weights_manifest(a.manifest)
    res, ok = {}, True
    for key, m in doc["models"].items():
        snap = snapshot_download(m["repo"], revision=m["revision"], allow_patterns=[f["path"] for f in m["files"]],
                                 local_files_only=not a.download)
        try:
            res[key] = {**C.verify_weights(snap, key, doc), "ok": True}
        except C.WeightsError as err:
            res[key] = {"dir": snap, "ok": False, "error": str(err)}
            ok = False
    total = C.weights_total_bytes(doc)
    print(json.dumps({"ok": ok, "models": res, "total_bytes": total, "total_gb": round(sum(total.values()) / 1e9, 3)},
                     indent=1, ensure_ascii=False))
    return 0 if ok else 1


def export_cmd(a, argv) -> int:
    from src.lisa import data as D
    from src.llmseg import runner as LR

    splits = LR.test_splits() if a.split == ALL else [a.split]
    paths = D.data_paths(a.data_root or os.environ.get("VISINT_DATA_ROOT"), tiny=a.tiny)
    meta = D.export(a.out, paths=paths, splits=splits, limit=a.limit, tiny=a.tiny)
    C.append_unlock(a.out, {"step": "export", "script": "scripts/lisa_w2.py", "splits": splits, "limit": a.limit, "tiny": a.tiny,
                            "manifest_rows_sha256": meta["rows_sha256"], "argv": list(argv)})
    print("清单：" + "；".join(f"{k} {v['n']} 句" for k, v in meta["splits"].items()) + f"（{C.inputs_dir(a.out)}）")
    return 0


def score_cmd(a) -> int:
    from src.lisa import scoring as S

    res = S.score(a.out, data_root=a.data_root or os.environ.get("VISINT_DATA_ROOT"), allow_stand_in=a.allow_stand_in, rescore=a.rescore)
    for split, m in res.items():
        head = "; ".join(f"{h['name']}={h.get('value')}" for h in m["headline"])
        print(f"LISA {split} n={m['n']}/{m['n_expected']} complete={m['complete']} {head} failures={m['failures']}")
    return 0


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    a = parse_args(argv)
    try:
        if a.cmd == "weights":
            return weights_cmd(a)
        if a.cmd == "export":
            return export_cmd(a, argv)
        return score_cmd(a)
    except (C.LisaError, OSError, ValueError, KeyError) as err:
        print(f"失败：{err!r}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
