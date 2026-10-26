"""DEV-0068：X-1 上限测量（scripts/explore_ceiling.py、src/explore/ceiling.py）。

钉住：拒绝 test_* 与 devcheck3（什么都不写）；--tiny 端到端（真实的读框路径：E1 格式的框记录、W2 格式的 W2-c 最终 mask、解析失败的框）；
续跑（中途停下、最后一行写了一半）与一次跑完的 predictions.jsonl、metrics.json、summary.md 逐字节相同；--workers 2（spawn）逐字节相同，
且子进程里没有学习型模块；运行身份不同拒绝续跑；格点的个数；CI 与 src.stats.bootstrap_ci 逐位相同；按框 IoU 分档。
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.explore import ceiling as X

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "explore_ceiling.py"
CFG = json.loads((REPO / "configs" / "explore_ceiling.json").read_text(encoding="utf-8"))
OUT_FILES = ("predictions.jsonl", "metrics.json", "summary.md", "config.yaml")


def _script():
    spec = importlib.util.spec_from_file_location("explore_ceiling_script", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S = _script()


def _small_config(tmp: Path) -> Path:
    cfg = dict(CFG, chunk_size=4)
    p = tmp / "explore_ceiling_small.json"
    p.write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")
    return p


@pytest.fixture(scope="module")
def oneshot(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("x1")
    cfgp = _small_config(tmp)
    out = tmp / "a"
    assert S.main(["--tiny", "--out", str(out), "--config", str(cfgp)]) == 0
    return out, cfgp


def _bytes(d: Path, split: str) -> dict[str, bytes]:
    return {f: (d / split / f).read_bytes() for f in OUT_FILES}


# ---------------------------------------------------------------------------
# 划分
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("split", ["test_refcoco_unc_testA", "TEST_x", "devcheck3", "explore_devcheck3", "devtune_countbench"])
def test_refuses_test_devcheck3_and_unlisted(split, tmp_path):
    out = tmp_path / "o"
    assert S.main(["--tiny", "--splits", "devtune800", split, "--out", str(out)]) == 2
    assert not out.exists()  # 什么都不写（先核对全部划分）
    with pytest.raises(S.SplitRefused):
        S.check_split(split, CFG)


def test_allowed_splits_listed():
    for sp in ("devtune800", "devcheck2", "explore_val"):
        S.check_split(sp, CFG)
    assert not any(s.startswith("test") or "devcheck3" in s for s in CFG["allowed_splits"])


# ---------------------------------------------------------------------------
# 端到端
# ---------------------------------------------------------------------------


def test_tiny_end_to_end(oneshot):
    out, _ = oneshot
    for split, n_sent, kinds in (("devtune800", 5, ["gt", "base"]), ("devcheck2", 3, ["gt", "base", "w2c"])):
        d = out / split
        for f in (*OUT_FILES, "compute.json", "meta.json"):
            assert (d / f).is_file(), (split, f)
        lines = [json.loads(x) for x in (d / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
        assert len(lines) == n_sent * len(kinds)
        assert [ln["box_kind"] for ln in lines[: len(kinds)]] == kinds
        plan = S.build_plan(CFG, "tiny")
        for ln in lines:
            starts = [X.START_GC_DEFAULT] + (["w2c_final"] if (split, ln["box_kind"]) == ("devcheck2", "w2c") else [])
            assert list(ln["cells"]) == X.cell_names(plan["grid"], plan["generator_order"], starts)
            for name, (i, u, iou, tok) in ln["cells"].items():
                assert 0 <= i <= u and (iou == round(i / u, 6) if u else iou == 1.0)
                if not ln["failure"] and name.startswith(("pts.", "spl.", "cor.")):
                    assert tok > 0, name
        fails = [ln for ln in lines if ln["failure"]]
        assert [(f["box_kind"], f["box"]) for f in fails] == [("base", None)]  # 假框里 base 的最后一句是解析失败
        assert all(c[0] == 0 and c[3] == 0 for c in fails[0]["cells"].values())
        gt_lines = [ln for ln in lines if ln["box_kind"] == "gt"]
        assert all(ln["box_iou"] > 0.98 for ln in gt_lines)
        met = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
        assert sorted(met["box_kinds"]) == sorted(kinds)
        m = met["box_kinds"]["gt"]["cells"]["spl.n16.raster.none"]
        assert m["all"]["ciou"] > 0.8 and m["bad"] is None and m["tokens_mean"] > 0
        assert "| `genK.16` |" in (d / "summary.md").read_text(encoding="utf-8")
    ref = json.loads((out / "devcheck2" / "predictions.jsonl").read_text().splitlines()[2])
    assert ref["box_kind"] == "w2c" and "ref.w2c_final" in ref["cells"] and "cor.w2c_final.m1.lock.grabcut.r1.none" in ref["cells"]


def test_resume_is_byte_identical(oneshot, tmp_path):
    out_a, cfgp = oneshot
    out_b = tmp_path / "b"
    assert S.main(["--tiny", "--out", str(out_b), "--config", str(cfgp), "--splits", "devcheck2", "--max-chunks", "1"]) == 0
    pred = out_b / "devcheck2" / "predictions.jsonl"
    assert len(pred.read_text().splitlines()) == 4 and not (out_b / "devcheck2" / "metrics.json").exists()
    with open(pred, "a", encoding="utf-8") as f:
        f.write('{"id": "refcoco_unc/val/1')  # 中断时写了一半的一行
    assert S.main(["--tiny", "--out", str(out_b), "--config", str(cfgp), "--splits", "devcheck2", "--resume", "--max-chunks", "1"]) == 0
    assert S.main(["--tiny", "--out", str(out_b), "--config", str(cfgp), "--splits", "devcheck2", "--resume"]) == 0
    assert _bytes(out_b, "devcheck2") == _bytes(out_a, "devcheck2")
    comp = json.loads((out_b / "devcheck2" / "compute.json").read_text())
    assert len(comp["runs"]) == 3 and comp["n_jobs"] == 9


def test_resume_refuses_other_identity_and_overwrite(oneshot, tmp_path):
    out_a, cfgp = oneshot
    with pytest.raises(SystemExit):
        S.main(["--tiny", "--out", str(out_a), "--config", str(cfgp), "--splits", "devtune800"])  # 已有结果、没加 --resume
    with pytest.raises(SystemExit):
        S.main(["--tiny", "--out", str(out_a), "--config", str(cfgp), "--splits", "devtune800", "--resume", "--grid", "first"])


def test_workers_spawn_identical_and_learned_free(oneshot, tmp_path):
    out_a, cfgp = oneshot
    out_c = tmp_path / "c"
    env = dict(os.environ, HF_HUB_OFFLINE="1")
    r = subprocess.run([sys.executable, str(SCRIPT), "--tiny", "--out", str(out_c), "--config", str(cfgp), "--splits", "devcheck2",
                        "--workers", "2"], capture_output=True, text=True, cwd=REPO, env=env)
    assert r.returncode == 0, r.stderr[-2000:]
    assert _bytes(out_c, "devcheck2") == _bytes(out_a, "devcheck2")
    comp = json.loads((out_c / "devcheck2" / "compute.json").read_text())
    assert comp["learned_modules_in_workers"] == []  # spawn 的子进程里没有 torch / transformers 等
    assert comp["n_processes"] >= 1 and comp["runs"][0]["workers"] == 2


# ---------------------------------------------------------------------------
# 格点与汇总
# ---------------------------------------------------------------------------


def test_cell_counts_full_grid():
    g, order = CFG["grids"]["full"], CFG["generators"]["order"]
    base = 1 + 16 + 4 + 4 * 2 * 2 * 3 + 4 * 3 * 3
    cor_per_start = 3 * 3 * 2 * 2 * 3
    assert len(X.cell_names(g, order, ["gc_default"])) == base + cor_per_start
    assert len(X.cell_names(g, order, ["gc_default", "w2c_final"])) == base + 1 + 2 * cor_per_start
    for name in CFG["grids"]:
        X.check_grid(CFG["grids"][name], order, CFG["noise_levels"])


def test_config_tool_values_equal_library_defaults():
    from src.explore import guidance as G

    t = CFG["tools"]
    assert t["points"]["radius_pct"] == G.SEED_RADIUS_PCT_DEFAULT and t["points"]["radius_min_px"] == G.SEED_RADIUS_MIN_PX
    assert t["spline"]["band_pct"] == G.BAND_PCT_DEFAULT and t["spline"]["band_min_px"] == G.BAND_MIN_PX
    c = t["corrections"]
    assert (c["alpha"], c["gate_min_px"], c["gate_min_frac"]) == (G.LOCK_ALPHA, G.EFFECT_MIN_PX, G.EFFECT_MIN_FRAC)
    for sec in ("points", "spline", "corrections"):
        assert t[sec]["iterations"] == G.GRABCUT_ITERATIONS and t[sec]["context_pct"] == G.CONTEXT_PCT_DEFAULT, sec
    assert set(CFG["box_kinds"]) == {"gt", "base", "w2c"}
    assert CFG["metrics"]["good_box_iou"] == 0.9  # X-0 的分档
    for g in ("full", "first"):  # first 是 full 的子网格
        for sec, keys in CFG["grids"][g].items():
            for k, v in keys.items():
                if isinstance(v, list):
                    assert set(v) <= set(CFG["grids"]["full"][sec][k]), (g, sec, k)


def test_check_grid_rejects():
    order, nz = CFG["generators"]["order"], CFG["noise_levels"]
    bad = json.loads(json.dumps(CFG["grids"]["first"]))
    bad["points"]["noise"] = ["loud"]
    with pytest.raises(ValueError):
        X.check_grid(bad, order, nz)
    bad = json.loads(json.dumps(CFG["grids"]["first"]))
    bad["generators"]["K"] = [17]
    with pytest.raises(ValueError):
        X.check_grid(bad, order, nz)
    bad = json.loads(json.dumps(CFG["grids"]["first"]))
    bad["corrections"]["starts"] = ["w2c_final"]
    with pytest.raises(ValueError):
        X.check_grid(bad, order, nz)


def test_ciou_ci_equals_bootstrap_ci():
    from src.constants import BOOTSTRAP_CI_LEVEL, BOOTSTRAP_N_RESAMPLES, BOOTSTRAP_SEED
    from src.stats import bootstrap_ci, ratio_of_sums

    rng = np.random.default_rng(1)
    u = rng.integers(50, 5000, size=37)
    pairs = np.stack([rng.integers(0, u), u], axis=1)
    ref = bootstrap_ci(pairs, statistic=ratio_of_sums, n_resamples=BOOTSTRAP_N_RESAMPLES, level=BOOTSTRAP_CI_LEVEL, seed=BOOTSTRAP_SEED)
    got = X.ciou_ci(pairs, seed=BOOTSTRAP_SEED, n_resamples=BOOTSTRAP_N_RESAMPLES, level=BOOTSTRAP_CI_LEVEL)
    assert got["ciou"] == ref.estimate and got["ci"] == [ref.low, ref.high]
    assert X.ciou_ci(np.zeros((0, 2)), seed=1, n_resamples=10, level=0.95) is None


def test_aggregate_bins_by_box_iou():
    def line(i, kind, biou, cells):
        return {"id": str(i), "box_kind": kind, "box": None if biou is None else [0, 0, 1, 1], "box_iou": biou, "cells": cells}

    lines = [line(0, "gt", 0.95, {"a": [9, 10, 0.9, 3]}), line(1, "gt", 0.5, {"a": [1, 10, 0.1, 5]}),
             line(2, "gt", None, {"a": [0, 10, 0.0, 0]})]
    m = X.aggregate(lines, good_iou=0.9, seed=0, n_resamples=50, level=0.95)["gt"]
    assert m["n_good_box"] == 1 and m["n_bad_box"] == 2 and m["n_box_failure"] == 1
    a = m["cells"]["a"]
    assert a["good"]["ciou"] == 0.9 and a["bad"]["ciou"] == 1 / 20 and a["all"]["ciou"] == 10 / 30
    assert a["tokens_mean"] == pytest.approx(4.0)  # 框解析失败的句子不进 token 均值
    with pytest.raises(ValueError):
        X.aggregate(lines + [line(3, "gt", 1.0, {"b": [1, 1, 1.0, 0]})], good_iou=0.9, seed=0, n_resamples=10, level=0.95)
