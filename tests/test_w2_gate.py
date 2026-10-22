"""DEV-0059：scripts/w2_gate.py（D113 第六条的去留判定，规则写死）。

- 构造的读数分别得出"通过"与"不通过"；
- 0.975 的水平与 10,000 次重抽样确实生效：结果与 src/stats 用这两个数现算的逐位相同；有一组数据 95% CI 的下界 > 0、97.5% 的 ≤ 0，
  判为不通过；与 1,000 次的结果不同；
- 判定点核对（9B、devcheck2、n = 500、非 tiny、不加 --limit、配置已冻结且同一份）：任一条不满足时结论标为"彩排，不作数"；
- 输入核对：设置、W1-F 的提出方式、检查点、模型、划分、id 集合、没写完 → 退出码 2。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from src import constants as C
from src.stats import paired_bootstrap_ci, ratio_of_sums

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def gate():
    saved = list(sys.path)
    spec = importlib.util.spec_from_file_location("_w2_gate_under_test", REPO / "scripts" / "w2_gate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.path[:] = saved
    return mod


def rows(ious, n=500, seed=0, noise=0.0, prefix="refcoco_unc/val"):
    rng = np.random.default_rng(seed)
    union = rng.integers(500, 1500, n)
    frac = np.clip(ious + rng.normal(0, noise, n), 0, 1) if noise else np.full(n, ious)
    return [{"id": f"{prefix}/{k}", "score": {"inter": int(f * u), "union": int(u)}} for k, (f, u) in enumerate(zip(frac, union))]


def test_rule_constants(gate):
    assert (gate.LEVEL, gate.N_RESAMPLES, gate.GATE_MODEL, gate.GATE_SPLIT, gate.GATE_N) == (0.975, 10_000, "qwen3.5-9b", "devcheck2", 500)
    assert (gate.MAIN_SETTING, gate.BASE_SETTING, gate.BASE_PROPOSE_MODE) == ("w2a", "w1f", "box")


def test_pass_and_fail(gate):
    d = gate.decide(rows(0.62, noise=0.1, seed=1), rows(0.57, noise=0.1, seed=2))
    assert d["pass"] and d["decision"] == gate.PASS == "通过" and d["ci_low"] > 0 and d["n"] == 500
    assert (d["level"], d["n_resamples"], d["seed"]) == (0.975, 10_000, C.BOOTSTRAP_SEED)
    d = gate.decide(rows(0.57, noise=0.2, seed=3), rows(0.57, noise=0.2, seed=4))
    assert not d["pass"] and d["decision"] == "不通过" and d["ci_low"] <= 0 <= d["ci_high"]
    d = gate.decide(rows(0.50, noise=0.05, seed=5), rows(0.57, noise=0.05, seed=6))  # 主配置更差
    assert not d["pass"] and d["ci_high"] < 0


def _pairs(a, b):
    ia = np.array([[r["score"]["inter"], r["score"]["union"]] for r in a], dtype=np.float64)
    ib = np.array([[r["score"]["inter"], r["score"]["union"]] for r in b], dtype=np.float64)
    return ia, ib


def test_level_and_resamples_take_effect(gate):
    """找一组 95% 下界 > 0、97.5% 下界 ≤ 0 的数据：规则判不通过（水平确实是 0.975）；读数与 src/stats 现算的逐位相同。"""
    seed = 65  # 在 range(200) 里按顺序找到的第一个（构造数据的种子）
    a, b = rows(0.60, n=500, seed=seed, noise=0.25), rows(0.60, n=500, seed=10_000 + seed, noise=0.25)
    ia, ib = _pairs(a, b)
    c95 = paired_bootstrap_ci(ia, ib, statistic=ratio_of_sums, n_resamples=10_000, level=0.95, seed=C.BOOTSTRAP_SEED)
    c975 = paired_bootstrap_ci(ia, ib, statistic=ratio_of_sums, n_resamples=10_000, level=0.975, seed=C.BOOTSTRAP_SEED)
    assert c95.low > 0 >= c975.low  # 95% 会判通过，97.5% 不会
    d = gate.decide(a, b)
    assert not d["pass"] and (d["ci_low"], d["ci_high"], d["diff"]) == (c975.low, c975.high, c975.estimate)
    ia, ib = _pairs(a, b)
    c1k = paired_bootstrap_ci(ia, ib, statistic=ratio_of_sums, n_resamples=1000, level=0.975, seed=C.BOOTSTRAP_SEED)
    assert (c1k.low, c1k.high) != (d["ci_low"], d["ci_high"])  # 重抽样次数确实是 10,000


# ---------------------------------------------------------------------------
# 运行目录（构造的 config.yaml / metrics.json / predictions.jsonl）
# ---------------------------------------------------------------------------


def make_leaf(root: Path, setting: str, recs, *, model="qwen3.5-9b", split="devcheck2", tiny=False, limit=None, frozen=True,
              cfg_sha="a" * 64, mode="box", checkpoint=None, complete=True, task="refseg") -> Path:
    leaf = root / model / task / setting / split
    leaf.mkdir(parents=True)
    cfg = {"experiment": "W2", "w2": {"task": task, "split": split, "setting": setting, "split_sha256": "s" * 64, "n_samples": len(recs),
                                      "limit": limit},
           "model": {"key": model, "revision": "r" * 40, "tiny": tiny}, "w2_config": {"config": "configs/w2.json", "config_sha256": cfg_sha,
                                                                                    "frozen": frozen},
           "propose": {"mode": mode}}
    if checkpoint is not None:
        cfg["checkpoint"] = checkpoint
    (leaf / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    (leaf / "metrics.json").write_text(json.dumps({"complete": complete}), encoding="utf-8")
    (leaf / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    return leaf


def test_gate_point_checks(gate, tmp_path):
    good_m, good_b = rows(0.62, noise=0.1, seed=1), rows(0.57, noise=0.1, seed=2)

    def ev(name, m_kw=None, b_kw=None, main_name="W2-a", m=good_m, b=good_b):
        root = tmp_path / name
        lm = make_leaf(root / "m", "w2a", m, **(m_kw or {}))
        lb = make_leaf(root / "b", "w1f", b, **(b_kw or {}))
        return gate.evaluate(lm, lb, main_name)

    d = ev("ok")
    assert d["gate_point"]["is_gate_point"] and d["verdict"] == "通过"
    for name, kw, key in (("m4b", {"model": "qwen3.5-4b"}, "model_is_qwen3.5-9b"), ("tune", {"split": "devtune800"}, "split_is_devcheck2"),
                          ("tiny", {"tiny": True}, "not_tiny"), ("lim", {"limit": 500}, "no_limit"),
                          ("unfrozen", {"frozen": False}, "w2_config_frozen"), ("othercfg", {"cfg_sha": "b" * 64}, "same_w2_config")):
        b_kw = {k: v for k, v in kw.items() if k not in ("frozen", "cfg_sha")}
        d = ev(name, kw, b_kw)
        assert not d["gate_point"]["checks"][key] and not d["gate_point"]["is_gate_point"], name
        assert d["verdict"].startswith("彩排，不作数"), name
    d = ev("n400", m=good_m[:400], b=good_b[:400])
    assert not d["gate_point"]["checks"]["n_is_500"] and d["verdict"].startswith("彩排")


def test_gate_input_errors(gate, tmp_path):
    m, b = rows(0.62, noise=0.1, seed=1), rows(0.57, noise=0.1, seed=2)

    def bad(name, msg, *, m_setting="w2a", b_setting="w1f", m_kw=None, b_kw=None, mm=m, bb=b, main_name="W2-a"):
        root = tmp_path / name
        lm = make_leaf(root / "m", m_setting, mm, **(m_kw or {}))
        lb = make_leaf(root / "b", b_setting, bb, **(b_kw or {}))
        with pytest.raises(gate.GateInputError, match=msg):
            gate.evaluate(lm, lb, main_name)

    bad("set", "w2a", m_setting="random_pick")
    bad("base", "w1f", b_setting="oracle_pick")
    bad("mode", "propose mode", b_kw={"mode": "box_points"})
    bad("ckb", "检查点", b_kw={"checkpoint": {"sha256": "x"}})
    bad("w2c", "W2-c", main_name="W2-c")
    bad("w2a_ck", "W2-a", m_kw={"checkpoint": {"sha256": "x"}})
    bad("model", "model", b_kw={"model": "qwen3.5-4b"})
    bad("split", "split", b_kw={"split": "devtune800"})
    bad("ids", "id 集合", bb=rows(0.57, seed=2, prefix="other"))
    bad("incomplete", "没有写完", m_kw={"complete": False})
    root = tmp_path / "w2c_ok"
    d = gate.evaluate(make_leaf(root / "m", "w2a", m, checkpoint={"sha256": "c"}), make_leaf(root / "b", "w1f", b), "W2-c")
    assert d["main_name"] == "W2-c" and d["runs"]["main"]["checkpoint"] == {"sha256": "c"}


def test_gate_cli_writes_json_and_md(gate, tmp_path):
    lm = make_leaf(tmp_path / "m", "w2a", rows(0.62, noise=0.1, seed=1))
    lb = make_leaf(tmp_path / "b", "w1f", rows(0.57, noise=0.1, seed=2))
    assert gate.main(["--main-name", "W2-a", "--main", str(lm), "--w1f", str(lb), "--out-dir", str(tmp_path / "g")]) == 0
    doc = json.loads((tmp_path / "g" / "gate.json").read_text())
    md = (tmp_path / "g" / "gate.md").read_text()
    assert doc["verdict"] == "通过" and "97.5% CI" in md and "10000 次" in md and "至少一个通过" in md
    assert gate.main(["--main-name", "W2-a", "--main", str(lb), "--w1f", str(lm)]) == 2
