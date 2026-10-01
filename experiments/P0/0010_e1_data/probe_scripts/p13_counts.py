#!/usr/bin/env python
"""RUN-0010 §4.4：用统一加载器把每个评测集的 n 读一遍（subset=eval 与 subset=dev），并读出点对的剔除统计，写 counts.json。
只调用 src.data.loaders 的公开函数；点对统计取自本运行生成的 e1_nyu_pairs.json。
usage (在 $REPO 下): python experiments/P0/0010_e1_data/probe_scripts/p13_counts.py <root> <pairs_config> <out.json>
"""
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

from src.data import loaders as L  # noqa: E402
from src.data.land.odinw import CONFIGS as ODINW_CONFIGS  # noqa: E402

root, pairs_cfg, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
res = {"root": root, "pairs_config": pairs_cfg, "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "sets": {}}


def rec(name, fn, **kw):
    t0 = time.time()
    try:
        r = fn(**kw)
        res["sets"][name] = {"n": len(r), "stats": r.stats, "load_s": round(time.time() - t0, 2)}
    except Exception as e:  # noqa: BLE001
        res["sets"][name] = {"error": repr(e)}
    print(name, res["sets"][name].get("n", res["sets"][name].get("error")), flush=True)


for ds, splits in (("refcoco", ("val", "testA", "testB")), ("refcoco+", ("val", "testA", "testB")), ("refcocog", ("val", "test"))):
    for sp in splits:
        for sub in ("eval", "dev"):
            if sub == "dev" and sp != ("val"):
                continue  # 开发子集只在 val 划分上定义（refcoco_dev_name）
            rec(f"{ds}/{sp}/{sub}", L.load_refcoco, root=root, dataset=ds, split=sp, subset=sub)
rec("coco_ovd/all", L.load_coco_ovd, root=root)
for cfg in ODINW_CONFIGS:
    rec(f"odinw13/{cfg}/all", L.load_odinw, root=root, config=cfg)
for name, fn in (("countbench", L.load_countbench), ("blink_reldepth", L.load_blink_reldepth), ("cvbench", L.load_cvbench), ("embspatial", L.load_embspatial)):
    for sub in ("eval", "dev"):
        rec(f"{name}/{sub}", fn, root=root, subset=sub)
for sub in ("eval", "dev"):
    rec(f"nyu_pairs/{sub}", L.load_nyu_pairs, root=root, subset=sub, pairs_config=pairs_cfg)
pc = json.loads(Path(pairs_cfg).read_text())
res["nyu_pairs_config_stats"] = pc["stats"]
res["nyu_pairs_constraints"] = pc["constraints"]
res["odinw13_total_all"] = sum(v["n"] for k, v in res["sets"].items() if k.startswith("odinw13/") and "n" in v)
Path(out_path).write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str))
