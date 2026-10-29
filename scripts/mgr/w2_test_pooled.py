"""W2 test（9B）各系统合并 5,000 句的 cIoU、gIoU 与 95% CI（D152）。管理员的分析脚本，只读结果目录，不跑模型。

- 系统：RUN-0072 的自己的框、W1-F、随机挑选、最优候选、W2-c、E1 零样本多边形；W2-a 取 RUN-0095 从 HPC 抽回的逐句交并
  （experiments/W2/0095_w2a_scores，没有时跳过并注明）；W2-c+回放取 RUN-0076 补充 2；探索候选 C1、C2 与 C1 的椭圆消融取 RUN-0092。
- 口径与 scripts/mgr/x_test_table.py 相同：cIoU = Σinter / Σunion，gIoU = 逐句 IoU 的均值；重抽 10,000 次，
  种子 src/constants.BOOTSTRAP_SEED；5 个划分逐 id 合并。只描述，不做新的检验。

用法：python scripts/mgr/w2_test_pooled.py [--out docs/analysis/w2_test_pooled]  → <out>.json 与 <out>.md
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src import constants as C  # noqa: E402

_spec = importlib.util.spec_from_file_location("xtt", ROOT / "scripts/mgr/x_test_table.py")
XT = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(XT)
W2T, SPLITS = XT.W2T, XT.SPLITS

REF = "experiments/W2/0072_w2_test_hpcgpu06"
REPLAY = "experiments/W2/0076_w2c_replay_test_hpcgpu06"
W2A = ROOT / "experiments/W2/0095_w2a_scores"
# (显示名, 结果目录, 设置)
SYSTEMS = [("自己的框", REF, "ownbox"), ("W1-F", REF, "w1f"), ("随机挑选", REF, "random_pick"),
           ("最优候选", REF, "oracle_pick"), ("W2-a", None, "w2a"), ("W2-c", REF, "w2c"),
           ("W2-c+回放", REPLAY, "w2c"), ("E1 零样本多边形", REF, "baseline_e1"),
           ("C1 = X-2（主方法）", "experiments/X/0092_x2_test_hpcgpu06", XT.MAIN),
           ("C1 的椭圆消融", "experiments/X/0092_x2_test_hpcgpu06", XT.ABL),
           ("C2 = X-3（消融）", "experiments/X/0092_x3_test_hpcgpu06", XT.MAIN)]


def w2a_preds(sp: str) -> dict | None:
    p = W2A / f"{sp}.jsonl"
    if not p.exists():
        return None
    return {r["id"]: {"inter": int(r["inter"]), "union": int(r["union"])}
            for r in map(json.loads, p.read_text(encoding="utf-8").splitlines()) if r}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="docs/analysis/w2_test_pooled")
    a = ap.parse_args(argv)
    doc: dict = {"n_resamples": XT.N_RES, "level": XT.LEVEL, "seed": C.BOOTSTRAP_SEED, "systems": {}}
    ids0 = None
    for name, run, setting in SYSTEMS:
        per = {sp: (w2a_preds(sp) if run is None else W2T.preds(XT.leaf(run, setting, sp))) for sp in SPLITS}
        if not all(per.values()):
            doc["systems"][name] = {"status": "缺", "missing": [sp for sp in SPLITS if not per[sp]]}
            continue
        pool = {k: v for sp in SPLITS for k, v in per[sp].items()}
        if len(pool) != sum(len(per[sp]) for sp in SPLITS):
            raise SystemExit(f"{name}：划分之间 id 重复")
        ids0 = ids0 or set(pool)
        if set(pool) != ids0:
            raise SystemExit(f"{name}：句子集合与其他系统不同")
        src = str(W2A.relative_to(ROOT)) if run is None else f"{run}/qwen3.5-9b/refseg/{setting}"
        doc["systems"][name] = {"status": "完成", "source": src, "cIoU": XT.single_ci(pool), "gIoU": XT.single_g(pool)}
    out = Path(a.out) if Path(a.out).is_absolute() else ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    L = ["# W2 test 各系统的合并读数（9B，5 个划分共 5,000 句；`scripts/mgr/w2_test_pooled.py` 生成；只描述）", "",
         f"cIoU = Σinter / Σunion，gIoU = 逐句 IoU 的均值；95% CI，bootstrap {XT.N_RES:,} 次，种子 {C.BOOTSTRAP_SEED}（D152）。", "",
         "| 系统 | n | cIoU [95% CI] | gIoU [95% CI] | 来源 |", "|---|---|---|---|---|"]
    for name, e in doc["systems"].items():
        if e["status"] != "完成":
            L.append(f"| {name} | — | 缺：{', '.join(e['missing'])} | | |")
            continue
        c, g = e["cIoU"], e["gIoU"]
        L.append(f"| {name} | {c['n']:,} | {c['ciou']:.3f} [{c['ci'][0]:.3f}, {c['ci'][1]:.3f}] | "
                 f"{g['giou']:.3f} [{g['ci'][0]:.3f}, {g['ci'][1]:.3f}] | `{e['source']}` |")
    out.with_suffix(".md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(out.with_suffix(".md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
