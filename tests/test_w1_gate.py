"""DEV-0051：scripts/w1_gate.py（D103 §三的去留判定，规则写死）。构造的三组读数分别得出"继续"、"放弃（条件 1 不过）"、
"放弃（条件 2 不过）"；恰好高 0.2 算满足（精确比较）；三个运行的 id 集合、模型、划分、设置不对就报错；不是判定点时结论标为彩排。"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def gate():
    saved = list(sys.path)
    spec = importlib.util.spec_from_file_location("_w1_gate_under_test", REPO / "scripts" / "w1_gate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.path[:] = saved
    return mod


def rows(ious, n=500, seed=0, noise=0.0, prefix="refcoco_unc/val"):
    rng = np.random.default_rng(seed)
    union = rng.integers(500, 1500, n)
    frac = np.clip(ious + rng.normal(0, noise, n), 0, 1) if noise else np.full(n, ious)
    return [{"id": f"{prefix}/{k}", "score": {"inter": int(f * u), "union": int(u)}} for k, (f, u) in enumerate(zip(frac, union))]


def test_continue(gate):
    d = gate.decide(rows(0.7, noise=0.1, seed=1), rows(0.5, noise=0.1, seed=2), rows(0.3, noise=0.1, seed=3))
    assert d["condition_1"]["pass"] and d["condition_2"]["pass"] and d["decision"] == gate.CONTINUE == "继续"
    assert d["condition_1"]["ci_low"] > 0 and d["condition_2"]["diff"] > 0.35
    assert d["condition_1"]["n_resamples"] == 1000 and d["condition_1"]["level"] == 0.95 and d["n"] == 500


def test_abandon_condition_1(gate):
    # W1-1 与自己的框一样好（逐条噪声大），比结构强制多边形高很多：条件 2 满足，条件 1 不满足
    w = rows(0.5, noise=0.3, seed=4)
    o = rows(0.5, noise=0.3, seed=5)
    d = gate.decide(w, o, rows(0.1, seed=6))
    assert not d["condition_1"]["pass"] and d["condition_2"]["pass"] and d["decision"] == gate.ABANDON == "放弃"
    assert d["condition_1"]["ci_low"] <= 0 <= d["condition_1"]["ci_high"]


def test_abandon_condition_2(gate):
    d = gate.decide(rows(0.7, noise=0.05, seed=7), rows(0.5, noise=0.05, seed=8), rows(0.6, noise=0.05, seed=9))
    assert d["condition_1"]["pass"] and not d["condition_2"]["pass"] and d["decision"] == "放弃"
    assert 0.05 < d["condition_2"]["diff"] < 0.15


def test_exactly_point_two_counts(gate):
    w = [{"id": f"x/{k}", "score": {"inter": 7, "union": 10}} for k in range(50)]
    o = [{"id": f"x/{k}", "score": {"inter": 1, "union": 10}} for k in range(50)]
    p = [{"id": f"x/{k}", "score": {"inter": 5, "union": 10}} for k in range(50)]
    d = gate.decide(w, o, p)
    assert 0.7 - 0.5 < 0.2 and d["condition_2"]["diff_exact"] == "1/5" and d["condition_2"]["pass"]


def test_id_sets_must_match(gate):
    with pytest.raises(ValueError, match="id 集合不同"):
        gate.decide(rows(0.7), rows(0.5, prefix="other"), rows(0.3))
    with pytest.raises(ValueError, match="id 集合不同"):
        gate.decide(rows(0.7), rows(0.5), rows(0.3, n=499))


def _leaf(root: Path, setting: str, recs, *, model="qwen3.5-9b", split="devcheck", tiny=False, e1_format=False,
          complete=True, limit=None) -> Path:
    leaf = root / setting
    leaf.mkdir(parents=True)
    w1 = {"task": "refseg", "split": split, "setting": setting, "split_sha256": "ab" * 32, "n_samples": len(recs), "limit": limit}
    cfg = {"model": {"key": model, "revision": "f" * 40, "tiny": tiny}}
    cfg.update({"data": {"selection_stats": {"w1": w1}}} if e1_format else {"w1": w1})
    (leaf / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True))
    (leaf / "metrics.json").write_text(json.dumps({"complete": complete}))
    (leaf / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    return leaf


def test_evaluate_reads_runs_and_flags_gate_point(gate, tmp_path):
    a = _leaf(tmp_path / "ok", "full", rows(0.7, noise=0.1, seed=1))
    b = _leaf(tmp_path / "ok", "baseline_ownbox", rows(0.5, noise=0.1, seed=2))
    c = _leaf(tmp_path / "ok", "baseline_poly_structured", rows(0.3, noise=0.1, seed=3), e1_format=True)
    doc = gate.evaluate(a, b, c)
    assert doc["gate_point"]["is_gate_point"] and doc["verdict"] == "继续"
    out_json, out_md = tmp_path / "g.json", tmp_path / "g.md"
    assert gate.main(["--w1", str(a), "--ownbox", str(b), "--poly", str(c), "--out-json", str(out_json), "--out-md", str(out_md)]) == 0
    assert json.loads(out_json.read_text())["decision"] == "继续" and "结论：继续" in out_md.read_text()
    # 不是判定点（小模型、少于 500 句）：照算，结论标为彩排
    r = tmp_path / "tiny"
    t = [_leaf(r, s, rows(v, n=20, seed=k), model="qwen3.5-0.8b", tiny=True)
         for k, (s, v) in enumerate((("full", 0.7), ("baseline_ownbox", 0.5), ("baseline_poly_structured", 0.3)))]
    doc = gate.evaluate(*t)
    assert not doc["gate_point"]["is_gate_point"] and doc["verdict"].startswith("彩排，不作数")
    assert doc["gate_point"]["checks"]["n_is_500"] is False and doc["gate_point"]["checks"]["not_tiny"] is False


@pytest.mark.parametrize("break_it, match", [
    ("model", "model 不同"),
    ("split", "split 不同"),
    ("setting", "应是 refseg/baseline_ownbox"),
    ("ids", "id 集合不同"),
    ("incomplete", "没有写完"),
])
def test_evaluate_rejects_mismatched_runs(gate, tmp_path, break_it, match):
    a = _leaf(tmp_path, "full", rows(0.7))
    kw = {"model": "qwen3.5-4b"} if break_it == "model" else {"split": "devtune"} if break_it == "split" else {}
    if break_it == "incomplete":
        kw["complete"] = False
    b_setting = "full2" if break_it == "setting" else "baseline_ownbox"
    b_recs = rows(0.5, prefix="other") if break_it == "ids" else rows(0.5)
    b = _leaf(tmp_path, b_setting, b_recs, **kw)
    c = _leaf(tmp_path, "baseline_poly_structured", rows(0.3))
    with pytest.raises(gate.GateInputError, match=match):
        gate.evaluate(a, b, c)
    assert gate.main(["--w1", str(a), "--ownbox", str(b), "--poly", str(c)]) == 2
