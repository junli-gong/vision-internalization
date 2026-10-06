"""DEV-0038 §3：scripts/compare_train_logs.py（跨机器训练一致性，D72-4）。

- 同一份日志 → 一致；人为加偏（单步超 2%、整体超中位数 0.5%、最后 10 步平均超 1%）→ 不一致；NaN / Inf → 不一致；
- 每步的数据字段不同（喂的不是同一批数据）→ 不一致；grad_norm 只报不判；
- 运行身份除机器字段外不同、种子不同、步数不够 → 不可比（退出码 2）；机器字段与 --ignore-key 的键忽略并照记；
- 真的 --tiny 训练日志（scripts/train_e5.py 跑两遍）→ 一致。
"""

from __future__ import annotations

import copy
import importlib.util
import json
import math
import shutil
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
_SCRIPTS: dict[str, object] = {}


def _script(name: str):
    if name not in _SCRIPTS:
        saved = list(sys.path)
        spec = importlib.util.spec_from_file_location(f"_{name}_under_test_ctl", REPO / "scripts" / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        sys.path[:] = saved
        _SCRIPTS[name] = mod
    return _SCRIPTS[name]


CT = _script("compare_train_logs")
IDENT = {"experiment": "E3b", "seed": 0, "model": {"key": "qwen3.5-2b", "dtype": "bf16"},
         "training": {"steps": 4000, "optim": {"lora": {"lr": 0.0002}}}, "config": {"path": "configs/e3b_train.json", "sha256": "ab"}}


def _rec(step: int) -> dict:
    loss = 3.0 * math.exp(-step / 40.0) + 0.5 + 0.05 * math.sin(step)
    return {"step": step, "lr": {"lora": 2e-4 * min(1.0, (step + 1) / 120)}, "loss": loss,
            "loss_by_task": {"depth": loss * 1.3, "refseg": loss * 0.7}, "n_supervised_by_task": {"depth": 2050, "refseg": 2114},
            "n_samples": 16, "n_by_kind": {"grid": 16}, "n_supervised_tokens": 16000 + step, "n_prompt_tokens": 3000 + 7 * step,
            "n_visual_tokens": 2000 + step, "n_image_pixels": 10 ** 6 + step, "grad_norm": 5.0 + 0.1 * step,
            "grid_token_acc": 0.5, "n_skipped": 0, "skipped": []}


def _leaf(root: Path, name: str, recs=None, ident=None) -> Path:
    d = root / name
    d.mkdir(parents=True)
    (d / "config.yaml").write_text(yaml.safe_dump(ident or IDENT, allow_unicode=True, sort_keys=False))
    (d / "train_log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in (recs if recs is not None else [_rec(s) for s in range(50)])))
    return d


def _scaled(f) -> list:
    out = []
    for s in range(50):
        r = _rec(s)
        r["loss"] *= f(s)
        out.append(r)
    return out


def test_identical_is_consistent(tmp_path):
    a, b = _leaf(tmp_path, "a"), _leaf(tmp_path, "b")
    rep = CT.compare(a, b)
    assert rep["verdict"] == "一致" and rep["loss_rel"]["max"] == 0.0 and all(c["pass"] for c in rep["checks"].values())
    assert CT.main([str(a), str(b / "train_log.jsonl"), "--out", str(tmp_path / "r.json"), "--md", str(tmp_path / "r.md")]) == 0
    assert json.loads((tmp_path / "r.json").read_text())["verdict"] == "一致" and "| 49 |" in (tmp_path / "r.md").read_text()
    # bf16 噪声量级的小差（每步 ±0.1%）也算一致
    small = _leaf(tmp_path, "small", _scaled(lambda s: 1 + 0.001 * (-1) ** s))
    assert CT.compare(a, small)["verdict"] == "一致"
    # grad_norm 只报不判
    g = [_rec(s) for s in range(50)]
    for r in g:
        r["grad_norm"] *= 1.2
    rep = CT.compare(a, _leaf(tmp_path, "g", g))
    assert rep["verdict"] == "一致" and rep["grad_norm_rel"]["max"] == pytest.approx(0.2 / 1.1)


@pytest.mark.parametrize("name,f,failing", [
    ("spike", lambda s: 1.03 if s == 17 else 1.0, "per_step_max"),  # 单步 3% > 2%
    ("drift", lambda s: 1.008, "per_step_median"),  # 每步 0.8%：最大不超 2%，中位数超 0.5%
    ("tail", lambda s: 1.015 if s >= 40 else 1.0, "last_k_mean"),  # 最后 10 步 1.5%：单步不超 2%，中位数 0
])
def test_biased_is_inconsistent(tmp_path, name, f, failing):
    a, b = _leaf(tmp_path, "a"), _leaf(tmp_path, name, _scaled(f))
    rep = CT.compare(a, b)
    assert rep["verdict"] == "不一致"
    assert [k for k, c in rep["checks"].items() if not c["pass"]] == [failing]
    assert CT.main([str(a), str(b)]) == 1


def test_nonfinite_and_data_mismatch(tmp_path):
    a = _leaf(tmp_path, "a")
    nan = [_rec(s) for s in range(50)]
    nan[30]["grad_norm"] = float("nan")
    rep = CT.compare(a, _leaf(tmp_path, "nan", nan))
    assert rep["verdict"] == "不一致" and not rep["checks"]["finite"]["pass"] and rep["checks"]["finite"]["nonfinite"][0]["step"] == 30
    inf = [_rec(s) for s in range(60)]
    inf[55]["loss_by_task"]["depth"] = float("inf")  # 窗口之外的步也查
    assert not CT.compare(a, _leaf(tmp_path, "inf", inf))["checks"]["finite"]["pass"]
    other = [_rec(s) for s in range(50)]
    other[3]["n_prompt_tokens"] += 1  # 喂的不是同一批数据
    rep = CT.compare(a, _leaf(tmp_path, "data", other))
    assert rep["verdict"] == "不一致" and rep["checks"]["data_identical"]["mismatches"] == [{"step": 3, "keys": ["n_prompt_tokens"]}]


def test_not_comparable(tmp_path):
    a = _leaf(tmp_path, "a")
    lr = copy.deepcopy(IDENT)
    lr["training"]["optim"]["lora"]["lr"] = 1e-4
    with pytest.raises(CT.NotComparable, match="training.optim.lora.lr"):
        CT.compare(a, _leaf(tmp_path, "lr", ident=lr))
    assert CT.main([str(a), str(tmp_path / "lr"), "--out", str(tmp_path / "r.json")]) == 2
    assert json.loads((tmp_path / "r.json").read_text())["verdict"] == "不可比"
    seed = {**IDENT, "seed": 1}
    with pytest.raises(CT.NotComparable, match="seed"):
        CT.compare(a, _leaf(tmp_path, "seed", ident=seed))
    with pytest.raises(CT.NotComparable, match="第 0"):
        CT.compare(a, _leaf(tmp_path, "short", [_rec(s) for s in range(30)]))
    # 机器字段（主机、GPU、设备、本机路径）忽略并照记；--ignore-key 追加
    mach = {**IDENT, "model": {**IDENT["model"], "device": "cuda:1"}, "paths": {"data_root": "/other"}}
    rep = CT.compare(a, _leaf(tmp_path, "mach", ident=mach))
    assert rep["verdict"] == "一致" and rep["identity"]["ignored_machine_keys"] == ["model.device", "paths.data_root"]
    extra = {**IDENT, "config": {**IDENT["config"], "path": "/abs/configs/e3b_train.json"}}
    with pytest.raises(CT.NotComparable):
        CT.compare(a, _leaf(tmp_path, "extra", ident=extra))
    assert CT.compare(a, tmp_path / "extra", ignore_keys=["config.path"])["identity"]["ignored_machine_keys"] == ["config.path"]


def test_real_tiny_logs(tmp_path):
    """两份真的 --tiny 训练日志（同一个种子、同一批数据，跑两遍）：一致；运行身份逐键相同。"""
    tr = _script("train_e5")
    for name in ("x", "y"):
        assert tr.main(["--tiny", "--init", "base", "--model", "qwen3.5-0.8b", "--n-train", "20", "--seed", "0",
                        "--out", str(tmp_path / name)]) == 0
    leaf = Path("train") / "base" / "qwen3.5-0.8b" / "n20" / "seed0"
    rep = CT.compare(tmp_path / "x" / leaf, tmp_path / "y" / leaf, n_steps=5)
    assert rep["verdict"] == "一致" and rep["checks"]["last_k_mean"]["k"] == 5 and rep["identity"]["ignored_machine_keys"] == []
    shutil.rmtree(tmp_path / "x")
