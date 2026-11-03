#!/usr/bin/env python
"""X-M 的数据统计（DEV-0079；给负责人的可行性报告）：train 与三个 COCO 评测划分（另附 lvis_sub 作参考）上的
- 对数、图数、每对实例数的分布（均值、中位数、p90、p99、最大值）；
- 两个目标（框列表、每实例 B 样条，src/xm/targets）的 token 数分布，用 Qwen3.5 的分词器（--tokenizer-model，缺省 qwen3.5-9b；
  oracle_sim.count_tokens：add_special_tokens = False，不含结束符与提示），以及两者之和（一对的两次调用的输出总长）；
- 截到 30 个的比例（实例数 > 30 的对）与被截掉的实例数；样条不可用的对；fit_iou（描样条的上限）。
只读真值，不跑任何系统：评测划分用 src/xm/data.stats_pairs（不过密封守卫，同生成清单），目标按与训练相同的 pair_targets 现算。
train 读 scripts/xm/build_targets.py 的输出（--targets，load_targets 逐行核对）。

服务器（HPC 登录节点或分配里的 CPU；要能离线读到 Qwen3.5 的分词器：HF_HOME=$VI/hf HF_HUB_OFFLINE=1）：
    python scripts/xm/data_stats.py --data-root $VI/data --targets experiments/XM/<NNNN>_xm_targets \\
        --out experiments/XM/<NNNN>_xm_data_stats --workers 12
    # 输出 <out>/data_stats.json 与 <out>/data_stats.md。用时：train 约 34 万行读入与核对约 2–4 分钟、分词约 70 万段文本约 2–5 分钟；
    #   评测划分约 1,800 + 1,000 张图的目标现算（--workers 12）约 1–2 分钟。合计约 10 分钟。
开发机彩排：--tiny（--tokenizer-model 缺省换成 qwen3.5-0.8b；数据与清单照 build_targets 的彩排给）。
--out 的最后一级目录名必须以四位运行编号开头（--tiny 不查）。退出码：0 = 写完；2 = 用法或数据错误。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.xm import data as D  # noqa: E402
from src.xm import targets as T  # noqa: E402

EVAL_SPLITS: tuple[str, ...] = (*D.COCO_EVAL_SPLITS, D.LVIS_SUB)
TOKEN_BATCH = 2048


def count_texts(tokenizer: Any, texts: Sequence[str]) -> list[int]:
    from src.explore import oracle_sim as OS

    out: list[int] = []
    for i in range(0, len(texts), TOKEN_BATCH):
        out += OS.count_tokens(tokenizer, list(texts[i: i + TOKEN_BATCH]))
    return out


def split_stats(rows: Sequence[Mapping[str, Any]], tokenizer: Any) -> dict[str, Any]:
    """一个划分的统计（模块说明）。"""
    s = T.summarize_rows(rows)
    s.pop("by_category", None)
    box_tok = count_texts(tokenizer, [r["box_text"] for r in rows])
    ok = [k for k, r in enumerate(rows) if r["spline_ok"]]
    spl_tok = count_texts(tokenizer, [rows[k]["spline_text"] for k in ok])
    total = [box_tok[k] + t for k, t in zip(ok, spl_tok)]
    per_inst = [t / rows[k]["n_kept"] for k, t in zip(ok, spl_tok)]
    s["tokens"] = {"box_text": T.dist(box_tok), "spline_text": T.dist(spl_tok), "both": T.dist(total), "spline_per_instance": T.dist(per_inst)}
    s["n_spline_not_ok"] = len(rows) - len(ok)
    return s


def eval_rows(split: str, data_root: str | Path, *, splits_path: str | Path, workers: int) -> list[dict[str, Any]]:
    from src.explore.x2_tools import ToolPool

    pairs = D.stats_pairs(split, data_root, splits_path=splits_path)
    with ToolPool(workers) as pool:
        res = pool.map(T.pair_job, [(p, T.SPLINE_N, T.MAX_BOXES) for p in pairs], chunksize=8)
    return [r for r, _ in res]


def _f(v: Any, nd: int = 1) -> str:
    return "—" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def render_md(doc: Mapping[str, Any]) -> str:
    lines = ["# X-M 数据统计（DEV-0079）", "", f"分词器：{doc['tokenizer']}；截断上限 {doc['max_boxes']} 个；样条 {doc['n']} 个控制点。", "",
             "| 划分 | 对 | 图 | 实例 | 每对实例 中位/p90/p99/最大 | >30 的对 | 截掉的实例 | 样条不可用的对 | 框列表 token 中位/p90/p99/最大 | 样条 token 中位/p90/p99/最大 | 两者之和 p50/p99/最大 | fit_iou 中位 |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, s in doc["splits"].items():
        ip, tk = s["instances_per_pair"] or {}, s["tokens"]
        b, sp, both = tk["box_text"] or {}, tk["spline_text"] or {}, tk["both"] or {}
        lines.append(
            f"| {name} | {s['n_pairs']} | {s['n_images']} | {s['n_instances']} | {_f(ip.get('p50'))}/{_f(ip.get('p90'))}/{_f(ip.get('p99'))}/{_f(ip.get('max'), 0)} "
            f"| {s['n_truncated_pairs']}（{100 * (s['truncated_rate'] or 0):.2f}%） | {s['n_dropped_by_cap']} | {s['n_spline_not_ok']} "
            f"| {_f(b.get('p50'), 0)}/{_f(b.get('p90'), 0)}/{_f(b.get('p99'), 0)}/{_f(b.get('max'), 0)} "
            f"| {_f(sp.get('p50'), 0)}/{_f(sp.get('p90'), 0)}/{_f(sp.get('p99'), 0)}/{_f(sp.get('max'), 0)} "
            f"| {_f(both.get('p50'), 0)}/{_f(both.get('p99'), 0)}/{_f(both.get('max'), 0)} | {_f((s['fit_iou'] or {}).get('p50'), 3)} |")
    lines += ["", "token 数只算模型的输出（框列表一次、样条一次），不含提示与图像 token；lvis_sub 只作参考（任务书要的是 train 与三个 COCO 划分）。",
              "分位数是 numpy.percentile 的 linear 插值。", ""]
    return "\n".join(lines)


def build(*, data_root: str | Path, targets_dir: str | Path, splits_path: str | Path, exclusions_path: str | Path, tokenizer: Any,
          tokenizer_record: Mapping[str, Any], workers: int = 1, log: Callable[[str], None] | None = None) -> dict[str, Any]:
    log = log or (lambda m: print(m, file=sys.stderr, flush=True))
    from src.xm import exclusions as X

    X.load_exclusions(exclusions_path, splits_path=splits_path)
    t0 = time.perf_counter()
    rows, man = T.load_targets(targets_dir)
    if man["source"]["xm_splits_sha256"] != D.file_sha256(splits_path) or man["source"]["xm_train_exclusions_sha256"] != D.file_sha256(exclusions_path):
        raise D.XMDataError("--targets 是按另一份划分或去重清单生成的")
    if man["source"].get("limit") is not None:
        log(f"[X-M 统计] 注意：--targets 是 --limit {man['source']['limit']} 的部分运行")
    splits: dict[str, Any] = {"train": split_stats(rows, tokenizer)}
    log(f"[X-M 统计] train：{len(rows)} 对，{time.perf_counter() - t0:.0f} s")
    for s in EVAL_SPLITS:
        t1 = time.perf_counter()
        splits[s] = split_stats(eval_rows(s, data_root, splits_path=splits_path, workers=workers), tokenizer)
        log(f"[X-M 统计] {s}：{splits[s]['n_pairs']} 对，{time.perf_counter() - t1:.0f} s")
    return {"framework": "scripts/xm/data_stats.py（DEV-0079）", "tokenizer": dict(tokenizer_record), "n": T.SPLINE_N, "max_boxes": T.MAX_BOXES,
            "sources": {"xm_splits_sha256": D.file_sha256(splits_path), "xm_train_exclusions_sha256": D.file_sha256(exclusions_path),
                        "targets_manifest_sha256": D.file_sha256(Path(targets_dir) / T.DATA_DIR / T.MANIFEST), "targets_limit": man["source"].get("limit")},
            "splits": splits}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", required=True, help="RUN-0102 的落地（$VI/data）")
    p.add_argument("--targets", required=True, help="scripts/xm/build_targets.py 的 --out")
    p.add_argument("--splits", default=str(D.XM_SPLITS_PATH), help="划分清单（默认 configs/xm_splits.json）")
    p.add_argument("--exclusions", default=str(D.XM_EXCLUSIONS_PATH), help="去重清单（默认 configs/xm_train_exclusions.json）")
    p.add_argument("--out", required=True, help="本次运行的目录（experiments/XM/<NNNN_slug>）")
    p.add_argument("--tokenizer-model", help="注册键（缺省 qwen3.5-9b；--tiny 缺省 qwen3.5-0.8b）")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--tiny", action="store_true")
    a = p.parse_args(argv)
    if a.workers < 1:
        p.error("--workers 必须是正整数")
    if not a.tiny:
        try:
            D.check_run_dir(a.out)
        except D.XMDataError as e:
            p.error(str(e))
    a.tokenizer_model = a.tokenizer_model or ("qwen3.5-0.8b" if a.tiny else "qwen3.5-9b")
    return a


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    a = parse_args(argv)
    out = Path(a.out)
    if (out / "data_stats.json").exists():
        print(f"{out / 'data_stats.json'} 已存在，不覆盖", file=sys.stderr)
        return 2
    try:
        from src.explore import oracle_sim as OS

        tok, rec = OS.load_tokenizer(a.tokenizer_model)
        doc = build(data_root=a.data_root, targets_dir=a.targets, splits_path=a.splits, exclusions_path=a.exclusions, tokenizer=tok,
                    tokenizer_record=rec, workers=a.workers)
    except (D.XMDataError, T.XMTargetsError, OSError, ValueError) as e:
        print(f"失败：{type(e).__name__}: {e}", file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)
    (out / "data_stats.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    md = render_md(doc)
    (out / "data_stats.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
