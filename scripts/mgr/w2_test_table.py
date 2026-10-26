"""W2 的 test 主实验汇总（W2 预注册 §11 的读数；管理员的分析脚本，只读结果目录，不跑模型）。

读 RUN-0072（9B 全部设置）与 RUN-0073（2B、4B 的自己的框、W1-F、W2-a）的叶目录 <run>/<模型>/refseg/<设置>/<划分>/：
- 每个划分、每个设置：cIoU、gIoU 与 95% CI（取 metrics.json 的 headline，即运行时算的那一份）、n、complete；
- 配对差（§11）：W2-c − W1-F、W2-a − W1-F（以及 W2-c − W2-a），按句配对 bootstrap，ratio-of-sums，95%，重抽 10,000 次，
  种子 src/constants.BOOTSTRAP_SEED；每个划分一份，另把 5 个划分的句子合并成 5,000 句再算一份（合并读数是附加的描述，§11 按划分报）；
- 挑中最优的比例（W2-a、W2-c 的 final_is_best；W2-c 的最优是它自己的框上的候选）。
逐句数据（predictions.jsonl）缺的格子写"缺逐句数据"：同步循环不推 > 5 MB 的文件（W2-a 的 predictions.jsonl 约 10 MB），
这种格子在 HPC 上跑本脚本补齐（同一条命令，--root 指 $VI/repo）。只描述，不判定（test 上不判定，§11）。

用法：python scripts/mgr/w2_test_table.py [--root .] [--out docs/analysis/w2_test_table]  → <out>.json 与 <out>.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src import constants as C  # noqa: E402
from src import stats as S  # noqa: E402
from src.stats import paired_bootstrap_ci, ratio_of_sums  # noqa: E402

SPLITS = ["test_refcoco_unc_testA", "test_refcoco_unc_testB", "test_refcoco+_unc_testA", "test_refcoco+_unc_testB",
          "test_refcocog_umd_test"]
RUNS = {"qwen3.5-9b": "experiments/W2/0072_w2_test_hpcgpu06", "qwen3.5-4b": "experiments/W2/0073_w2_test_hpcgpu07",
        "qwen3.5-2b": "experiments/W2/0073_w2_test_hpcgpu07"}
SETTINGS = {"qwen3.5-9b": ["ownbox", "w1f", "random_pick", "oracle_pick", "w2a", "w2c", "baseline_e1"],
            "qwen3.5-4b": ["ownbox", "w1f", "w2a"], "qwen3.5-2b": ["ownbox", "w1f", "w2a"]}
PAIRS = {"qwen3.5-9b": [("w2c", "w1f"), ("w2a", "w1f"), ("w2c", "w2a")], "qwen3.5-4b": [("w2a", "w1f")], "qwen3.5-2b": [("w2a", "w1f")]}
LEVEL, N_RES = 0.95, 10_000


def leaf(root: Path, model: str, setting: str, split: str) -> Path:
    return root / RUNS[model] / model / "refseg" / setting / split


def metrics(lf: Path) -> dict | None:
    p = lf / "metrics.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    h = {x["name"]: x for x in d.get("headline", []) if x.get("scope") == "all"}
    def g(k):
        x = h.get(k)
        return None if x is None else {"value": x["value"], "ci": [x["ci"]["low"], x["ci"]["high"]] if x.get("ci") else None}
    return {"n": d.get("n"), "complete": d.get("complete"), "cIoU": g("cIoU"), "gIoU": g("gIoU")}


def preds(lf: Path) -> dict[str, dict] | None:
    p = lf / "predictions.jsonl"
    if not p.exists():
        return None
    out = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["id"]] = {"inter": int(r["score"]["inter"]), "union": int(r["score"]["union"]),
                            "best": r.get("final_is_best")}
    return out


def paired(a: dict, b: dict) -> dict:
    ids = sorted(a)
    if set(ids) != set(b):
        raise SystemExit("配对的两边句子集合不同")
    A = np.array([[a[i]["inter"], a[i]["union"]] for i in ids], dtype=np.float64)
    B = np.array([[b[i]["inter"], b[i]["union"]] for i in ids], dtype=np.float64)
    ci = fast_paired_ratio_ci(A, B)
    return {"n": len(ids), "diff": ci.estimate, "ci": [ci.low, ci.high]}


def fast_paired_ratio_ci(A: np.ndarray, B: np.ndarray):
    """与 src.stats.paired_bootstrap_ci(A, B, statistic=ratio_of_sums, …) 逐位相同（同一组重抽下标、同一个分位数），只是向量化：
    每次重抽的 Σinter / Σunion 用下标一次求和。--check 时与参考实现比较。"""
    idx = S._resample_indices(A.shape[0], N_RES, C.BOOTSTRAP_SEED)
    boot = A[idx, 0].sum(1) / A[idx, 1].sum(1) - B[idx, 0].sum(1) / B[idx, 1].sum(1)
    return S._percentile_ci(float(ratio_of_sums(A) - ratio_of_sums(B)), boot, LEVEL, N_RES)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--out", default="docs/analysis/w2_test_table")
    ap.add_argument("--check", action="store_true", help="向量化的 CI 与 src.stats 的参考实现在随机数据上逐位比较后退出")
    a = ap.parse_args(argv)
    if a.check:
        rng = np.random.default_rng(1)
        u = rng.integers(100, 5000, size=300).astype(np.float64)
        A = np.stack([np.floor(u * rng.random(300)), u], 1); B = np.stack([np.floor(u * rng.random(300)), u], 1)
        ref = paired_bootstrap_ci(A, B, statistic=ratio_of_sums, n_resamples=N_RES, level=LEVEL, seed=C.BOOTSTRAP_SEED)
        got = fast_paired_ratio_ci(A, B)
        same = (ref.estimate, ref.low, ref.high) == (got.estimate, got.low, got.high)
        print("check", "same" if same else f"DIFF ref={ref} got={got}")
        return 0 if same else 1
    root = Path(a.root)
    doc: dict = {"level": LEVEL, "n_resamples": N_RES, "seed": C.BOOTSTRAP_SEED, "models": {}}
    for model in ("qwen3.5-9b", "qwen3.5-4b", "qwen3.5-2b"):
        m: dict = {"per_split": {}, "pairs": {}, "pairs_pooled": {}, "pick_best": {}}
        P: dict = {}
        for sp in SPLITS:
            m["per_split"][sp] = {s: metrics(leaf(root, model, s, sp)) for s in SETTINGS[model]}
            P[sp] = {s: preds(leaf(root, model, s, sp)) for s in SETTINGS[model]}
            for s in ("w2a", "w2c"):
                if s in SETTINGS[model] and P[sp][s] is not None:
                    v = [r["best"] for r in P[sp][s].values() if r["best"] is not None]
                    m["pick_best"].setdefault(s, {})[sp] = (sum(bool(x) for x in v) / len(v)) if v else None
        for x, y in PAIRS[model]:
            key = f"{x}-{y}"
            m["pairs"][key] = {}
            pool_x, pool_y, ok = {}, {}, True
            for sp in SPLITS:
                px, py = P[sp].get(x), P[sp].get(y)
                if px is None or py is None:
                    m["pairs"][key][sp] = "缺逐句数据"; ok = False; continue
                m["pairs"][key][sp] = paired(px, py)
                pool_x.update(px); pool_y.update(py)
            m["pairs_pooled"][key] = paired(pool_x, pool_y) if ok else "缺逐句数据（至少一个划分）"
        doc["models"][model] = m
    out = Path(a.out) if Path(a.out).is_absolute() else ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    def f(x, p=3):
        return "—" if x is None else f"{x:.{p}f}"
    L = ["# W2 test 汇总（W2 预注册 §11；`scripts/mgr/w2_test_table.py` 生成；只描述，不判定）", "",
         f"配对 bootstrap，ratio-of-sums，{int(LEVEL * 100)}%，{N_RES:,} 次，种子 {C.BOOTSTRAP_SEED}。合并 = 5 个划分的句子合在一起。", ""]
    for model, m in doc["models"].items():
        L += [f"## {model}", "", "| 划分 | " + " | ".join(SETTINGS[model]) + " |", "|---|" + "---|" * len(SETTINGS[model])]
        for sp in SPLITS:
            cells = []
            for s in SETTINGS[model]:
                r = m["per_split"][sp][s]
                cells.append("—" if r is None or r["cIoU"] is None else
                             f"{f(r['cIoU']['value'])}" + (" ⚠未完成" if r.get("complete") is not True else ""))
            L.append(f"| {sp} | " + " | ".join(cells) + " |")
        L += ["", "配对差（cIoU）：", "", "| 对比 | " + " | ".join(SPLITS) + " | 合并 |", "|---|" + "---|" * (len(SPLITS) + 1)]
        for key, per in m["pairs"].items():
            cells = [(v if isinstance(v, str) else f"{v['diff']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]") for v in per.values()]
            pv = m["pairs_pooled"][key]
            cells.append(pv if isinstance(pv, str) else f"{pv['diff']:+.3f} [{pv['ci'][0]:+.3f}, {pv['ci'][1]:+.3f}]（n {pv['n']}）")
            L.append(f"| {key} | " + " | ".join(cells) + " |")
        if m["pick_best"]:
            L += ["", "挑中最优的比例：" + "；".join(f"{s} " + "、".join(f(v) for v in d.values()) for s, d in m["pick_best"].items())]
        L.append("")
    out.with_suffix(".md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(out.with_suffix(".md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
