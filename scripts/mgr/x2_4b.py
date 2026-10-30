"""补充实验：4B 上的 X-2（预注册 X-7；D157；RUN-0098）。管理员的分析脚本，只读结果目录，不跑模型。

- 句子：explore_val 三层、devtune800、devcheck2，合并 2,300 句（同 scripts/mgr/seed_variance.py）。
- 读数：4B 的 x2_spline_trimap、x2_ellipse_trimap、x2_select、x2_select5 的合并 cIoU 与 95% CI；
  样条 − 椭圆、样条 − 只挑工具候选的配对差（95%，10,000 次，种子 src/constants.BOOTSTRAP_SEED）；
  与 9B 三个种子（docs/analysis/seed_variance.json）并列；健康检查（4B 基座对 4B X-2，07）。只描述。

用法：python scripts/mgr/x2_4b.py [--out docs/analysis/x2_4b]
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

_spec = importlib.util.spec_from_file_location("sv", ROOT / "scripts/mgr/seed_variance.py")
SV = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(SV)
XT, W2T, SPLITS, SETTINGS = SV.XT, SV.W2T, SV.SPLITS, SV.SETTINGS
RUN = "experiments/X/0098_x2_4b_hpcgpu07/0098_eval"
HEALTH = "experiments/X/0098_x2_4b_hpcgpu07/0098_eval/health/health_compare.json"
M4 = "qwen3.5-4b"


def preds(setting: str, sp: str):
    return W2T.preds(ROOT / RUN / M4 / "refseg" / setting / sp)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="docs/analysis/x2_4b")
    a = ap.parse_args(argv)
    per = {st: {sp: preds(st, sp) for sp in SPLITS} for st in SETTINGS}
    missing = [(st, sp) for st in SETTINGS for sp in SPLITS if per[st][sp] is None]
    if missing:
        raise SystemExit(f"4B 还缺：{missing}")
    pool = {st: {k: v for sp in SPLITS for k, v in per[st][sp].items()} for st in SETTINGS}
    sv = json.loads((ROOT / "docs/analysis/seed_variance.json").read_text(encoding="utf-8"))
    doc: dict = {"run": RUN, "n_resamples": XT.N_RES, "level": XT.LEVEL, "seed": C.BOOTSTRAP_SEED, "splits": SPLITS,
                 "settings": {st: {"per_split": {sp: XT.single_ci(per[st][sp])["ciou"] for sp in SPLITS},
                                   "pooled": XT.single_ci(pool[st])} for st in SETTINGS},
                 "spline_minus_ellipse": XT.paired(pool["x2_spline_trimap"], pool["x2_ellipse_trimap"]),
                 "spline_minus_select": XT.paired(pool["x2_spline_trimap"], pool["x2_select"]),
                 "nine_b": {st: sv["across_seeds"][st] for st in SETTINGS}}
    h = ROOT / HEALTH
    if h.exists():
        hj = json.loads(h.read_text(encoding="utf-8"))
        doc["health"] = {"any_flag": hj.get("any_flag"),
                         "rows": {f"{r['reading']}.{r['metric']}": {"base_pct": r.get("base_pct"), "delta_pp": r.get("delta_pp"),
                                                                     "flag": r.get("flag")} for r in hj.get("rows", []) if r.get("headline")}}
    out = Path(a.out) if Path(a.out).is_absolute() else ROOT / a.out
    out.with_suffix(".json").write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    def d(x):
        return f"{x['diff']:+.3f} [{x['ci'][0]:+.3f}, {x['ci'][1]:+.3f}]"
    L = ["# 补充实验：4B 上的 X-2（预注册 X-7；`scripts/mgr/x2_4b.py` 生成；只描述）", "",
         f"开发数据 2,300 句（explore_val 三层、devtune800、devcheck2）；cIoU = Σinter / Σunion；配对差 bootstrap {XT.N_RES:,} 次，种子 {C.BOOTSTRAP_SEED}。"
         "RUN-0098（07）；9B 为 C1 的三个种子（RUN-0080、0093、0094）。", "",
         "| 设置 | " + " | ".join(SPLITS) + " | 4B 合并 [95% CI] | 9B 三个种子的合并 |", "|---|" + "---|" * (len(SPLITS) + 2)]
    for st in SETTINGS:
        e = doc["settings"][st]
        nb = doc["nine_b"][st]
        L.append(f"| {st} | " + " | ".join(f"{e['per_split'][sp]:.3f}" for sp in SPLITS) +
                 f" | {e['pooled']['ciou']:.3f} [{e['pooled']['ci'][0]:.3f}, {e['pooled']['ci'][1]:.3f}] | " +
                 " / ".join(f"{v:.3f}" for v in nb["pooled"]) + " |")
    L += ["", f"4B 样条 − 椭圆：{d(doc['spline_minus_ellipse'])}；4B 样条 − 只挑工具候选：{d(doc['spline_minus_select'])}。"]
    if "health" in doc:
        L += ["", "健康检查（4B 基座对 4B X-2，07）：" + "；".join(
            f"{k} 基座 {v['base_pct']} 差 {v['delta_pp']:+.1f}{' 标红' if v['flag'] else ''}" for k, v in doc["health"]["rows"].items())]
    out.with_suffix(".md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(out.with_suffix(".md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
