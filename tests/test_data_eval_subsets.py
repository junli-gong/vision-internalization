"""E1 评测子集清单（src/data/eval_subsets.py、build_e1_configs.py eval-subsets；DEV-0012 审查第 1 轮）。

- fixture 上的 --check：tests/fixtures/e1/eval_subsets.json 由同一命令在 fixture 上生成，重算必须一致（逐字节）。
- 总体：剔除开发子集与 D20 之后的评测总体；子集是总体（或父子集）的子集；不足目标数全取并标明。
- 每类子集一个独立的盐：同一总体上的多边形子集与 16×16 子集互不嵌套。
- 合成大总体上钉死配置文本的 sha256（固定盐下逐字节复现）。
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from src.data import constants as C
from src.data.eval_subsets import POLY_SPLITS, THINK_TASKS, build_eval_subsets, compare, eval_populations
from src.data.subsets import dump_config, sort_ids

REPO = Path(__file__).resolve().parents[1]
FX = Path(__file__).resolve().parent / "fixtures" / "e1"
ARGS = ["--root", str(FX / "root"), "--dev-config", str(FX / "dev_subsets.json"), "--pairs-config", str(FX / "nyu_pairs.json")]


def script():
    spec = importlib.util.spec_from_file_location("_build_e1_configs_eval", REPO / "scripts/data/build_e1_configs.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_fixture_check_and_bytewise_regeneration(tmp_path):
    mod = script()
    assert mod.main(["eval-subsets", *ARGS, "--out", str(FX / "eval_subsets.json"), "--check"]) == 0
    out = tmp_path / "e.json"
    assert mod.main(["eval-subsets", *ARGS, "--out", str(out)]) == 0
    assert out.read_text() == (FX / "eval_subsets.json").read_text()
    cfg = json.loads(out.read_text())
    cfg["subsets"]["think:count"]["ids"] = ["3", "0"]
    out.write_text(json.dumps(cfg))
    assert mod.main(["eval-subsets", *ARGS, "--out", str(out), "--check"]) == 1


def test_fixture_subsets_respect_exclusions_and_nesting():
    pops = eval_populations(FX / "root", dev_config=FX / "dev_subsets.json", pairs_config=FX / "nyu_pairs.json")
    cfg = json.loads((FX / "eval_subsets.json").read_text())
    dev = {k: set(v["ids"]) for k, v in json.loads((FX / "dev_subsets.json").read_text())["subsets"].items()}
    # 总体已剔除开发子集与 D20
    assert "101" not in pops["refcoco_unc_val"]["ids"] and pops["refcoco_unc_val"]["stats"]["excluded_by_reason"] == {"开发子集": 1}
    assert "39769" not in pops["coco_val2017"]["ids"] and pops["coco_val2017"]["stats"]["excluded_by_reason"] == {"D20 探针用图": 1}
    assert "0" not in pops["cvbench"]["ids"] and not dev["cvbench"] & set(pops["cvbench"]["ids"])
    assert all(i.startswith("nyu0001_") for i in pops["nyu_pairs"]["ids"])  # 只取 test 划分的点对
    for name, sub in cfg["subsets"].items():
        parent = cfg["subsets"][sub["population"]]["ids"] if sub["population"] in cfg["subsets"] else pops[sub["population"]]["ids"]
        assert set(sub["ids"]) <= set(parent), name
        assert sub["k"] == min(sub["k_target"], len(parent)) and sub["all_taken"] == (sub["k"] == len(parent))
        assert sub["ids"] == sort_ids(sub["ids"])
    assert {n for n in cfg["subsets"] if n.startswith("poly:")} == {f"poly:{d}_{b}_{s}" for d, b, s in POLY_SPLITS}
    assert {n.split(":", 1)[1] for n in cfg["subsets"] if n.startswith("think:")} == {t for t, _ in THINK_TASKS}
    assert cfg["subsets"]["think:refexp_poly"]["population"] == "poly:refcoco_unc_val"
    assert cfg["subsets"]["think:grid16"]["population"] == "grid16:refcoco_unc_val"
    assert all(cfg["subsets"][f"think8192:{t}"]["population"] == f"think:{t}" for t, _ in THINK_TASKS)
    assert "全取" in cfg["subsets"]["think:rel_depth_blink"]["note"]


def synthetic_pops() -> dict:
    def p(ids):
        return {"ids": ids, "unit": "x", "provenance": "synthetic", "stats": {"n_eval": len(ids)}}

    pops = {f"{d}_{b}_{s}": p([str(i) for i in range(5000 + k)]) for k, (d, b, s) in enumerate(POLY_SPLITS)}
    pops.update(
        countbench=p([str(i) for i in range(442)]),
        blink_reldepth=p([f"val_Relative_Depth_{i}" for i in range(1, 113)]),
        cvbench=p([str(i) for i in range(2588)]),
        embspatial=p([f"mp3d_{i}" for i in range(3590)]),
        coco_val2017=p([str(i) for i in range(500)]),
        nyu_pairs=p([f"nyu{i:04d}_{k}" for i in range(1, 655) for k in range(5)]),
    )
    return pops


def test_sizes_independent_salts_and_pinned_bytes():
    cfg = build_eval_subsets(synthetic_pops())
    s = cfg["subsets"]
    assert all(s[f"poly:{d}_{b}_{sp}"]["k"] == C.EVAL_POLY_PER_SPLIT == 1000 for d, b, sp in POLY_SPLITS)
    assert s["grid16:refcoco_unc_val"]["k"] == C.EVAL_GRID16_N == 500
    assert all(s[f"think:{t}"]["k"] == 200 for t, _ in THINK_TASKS if t != "rel_depth_blink")
    assert s["think:rel_depth_blink"]["k"] == 112 and s["think:rel_depth_blink"]["all_taken"]
    assert all(s[f"think8192:{t}"]["k"] == C.EVAL_THINK8192_N == 50 for t, _ in THINK_TASKS)
    assert s["extra:refcoco_unc_val"]["k"] == 500 and s["extra:coco_val2017"]["k"] == 100
    assert len(set(C.EVAL_SUBSET_SALTS.values())) == len(C.EVAL_SUBSET_SALTS)
    poly, grid = set(s["poly:refcoco_unc_val"]["ids"]), set(s["grid16:refcoco_unc_val"]["ids"])
    assert not grid <= poly and 0 < len(poly & grid) < len(grid)  # 独立抽，允许重叠
    extra = set(s["extra:refcoco_unc_val"]["ids"])
    assert extra != grid and not extra <= poly
    text = dump_config(cfg)
    assert text == dump_config(build_eval_subsets(synthetic_pops()))
    assert hashlib.sha256(text.encode()).hexdigest() == PINNED


PINNED = "3dadfef8eb5f3bc4df9257082ebf51a7890abdac59d1ea16fe79c2e49793b9a5"


def test_compare_reports_population_changes():
    a = build_eval_subsets(synthetic_pops())
    pops = synthetic_pops()
    pops["countbench"]["ids"] = pops["countbench"]["ids"][:-1]
    b = build_eval_subsets(pops)
    bad = compare(b, a)
    assert any(x.startswith("总体 countbench") for x in bad) and any("think:count" in x for x in bad)
    assert compare(a, a) == []
    with pytest.raises(KeyError):
        from src.data.eval_subsets import subset_ids

        subset_ids("nope", FX / "eval_subsets.json")
