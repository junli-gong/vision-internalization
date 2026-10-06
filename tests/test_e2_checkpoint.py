"""DEV-0038 §2（D77-2(b)）：scripts/run_e2.py --checkpoint 与指定层 --layers L… （CPU fp32 小模型，fixture 小数据）。

- 指定层：只训练给定层上的探针；base 上与原来的 run_e2（全部层）在同一层上逐字节相同（探针的初值、批、标准化统计量都只由层自己决定）；
- 给 base（检查点的 LoRA 为零，合并后与 base 逐位相同）：--checkpoint 的输出（curves、metrics、eval_samples）与 base 的同一次运行逐字节
  相同，config.yaml 只多 checkpoint 一个键；
- 小检查点（E3b 网格监督、E3a 的 full）上跑通，运行身份记 manifest 的 sha256，输出目录带标签；E3a 只用 model/（LoRA），不用查询与读出；
- C1、C2（E3a 冻结 LLM）、没有 LoRA 也没有网格 token 的检查点、E5 检查点：拒绝，不建任何目录；
- 合并核对不过（病态 LoRA）：拒绝（退出码 1），不建任务目录；
- 命令行：--checkpoint 要配合法的 --checkpoint-label，不能配 --estimate；标签与 run_e1 同一个正则。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from src.adapters.factory import make_adapter
from src.e2 import checkpoint as E2CK
from src.e2.config import expand_layers, explicit_pre_norm, is_explicit, parse_layer_spec

REPO = Path(__file__).resolve().parents[1]
KEY = "qwen3.5-0.8b"
_SCRIPTS: dict[str, object] = {}


def _script(name: str):
    if name not in _SCRIPTS:
        saved = list(sys.path)
        spec = importlib.util.spec_from_file_location(f"_{name}_under_test_e2ck", REPO / "scripts" / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        sys.path[:] = saved
        _SCRIPTS[name] = mod
    return _SCRIPTS[name]


def _e2(*args: str) -> int:
    return _script("run_e2").main(["--tiny", "--model", KEY, *args])


def _zero_lora(path: Path) -> Path:
    """与 run_e2 --tiny 同一个 base，挂一份默认初始化的 LoRA（B = 0）：合并后与 base 逐位相同。"""
    ad = make_adapter(KEY, device="cpu", dtype=torch.float32, tiny=True, init_std=_script("run_e2").TINY_INIT_STD)
    ad.inject_lora(r=2, lora_alpha=2)
    return ad.save_trained(path)


@pytest.fixture(scope="module")
def ckpts(tmp_path_factory):
    root = tmp_path_factory.mktemp("e2ck")
    assert _script("train_e3b").main(["--tiny", "--model", KEY, "--supervision", "grid", "--seed", "0", "--out", str(root / "e3")]) == 0
    for v in ("full", "c1", "c2"):
        assert _script("train_e3a").main(["--tiny", "--model", KEY, "--variant", v, "--seed", "0", "--out", str(root / "e3a")]) == 0
    fin = Path("seed0") / "ckpt" / "final"
    return {"e3b": root / "e3" / "e3b" / KEY / fin / "model", **{v: root / "e3a" / "e3a" / v / KEY / fin for v in ("full", "c1", "c2")},
            "zero": _zero_lora(root / "zero")}


# ---------------------------------------------------------------------------
# 指定层
# ---------------------------------------------------------------------------


def test_explicit_layer_specs():
    assert parse_layer_spec(["L24", "L8"]) == "L8 L24" and is_explicit("L8 L24") and not is_explicit("all")
    assert parse_layer_spec(["L4_pre", "L1"]) == "L1 L4_pre" and expand_layers("L1 L4_pre", 4) == [1]
    assert explicit_pre_norm("L1 L4_pre", 4) is True and explicit_pre_norm("L1 L4", 4) is False
    assert explicit_pre_norm("all", 4) is None and explicit_pre_norm("every 2", 4) is None  # 原写法照旧由配置决定
    assert expand_layers("L8 L24", 24, always_include_last=True) == [8, 24] and expand_layers("L3", 32) == [3]  # 不补最后一层
    for bad in (["L8", "L8"], ["L08"], ["L-1"], ["L4_pre"], ["L8", "all"], ["8"]):
        with pytest.raises(ValueError):
            parse_layer_spec(bad)
    for spec in ("L5", "L1 L3_pre"):
        with pytest.raises(ValueError):
            expand_layers(spec, 4)
    # 原来的写法不变
    assert parse_layer_spec(["all"]) == "all" and parse_layer_spec(["every", "2"]) == "every 2" and expand_layers("all", 4) == [0, 1, 2, 3, 4]


def _probe_parts(leaf: Path, names: list[str]) -> dict:
    m = json.loads((leaf / "metrics.json").read_text())
    curves = [l for l in (leaf / "curves.jsonl").read_text().splitlines() if json.loads(l)["probe"] in names]
    npz = np.load(leaf / "eval_samples.npz")
    want = {n.replace("/", "__") for n in names}  # eval_samples.npz 的键：<特征>__<种类>__<字段>
    arrays = {k: npz[k] for k in npz.files if "__".join(k.split("__")[:2]) in want}
    w = torch.load(leaf / "ckpt" / "probes_final.pt", map_location="cpu", weights_only=False)["probes"]
    return {"probes": {n: json.dumps(m["probes"][n], sort_keys=True) for n in names}, "curves": curves, "arrays": arrays,
            "weights": {n: w[n] for n in names}, "n_eval": m["n_eval"], "n_val": m["n_val"]}


def _same(a: dict, b: dict) -> None:
    assert a["probes"] == b["probes"] and a["curves"] == b["curves"] and a["n_eval"] == b["n_eval"] and a["n_val"] == b["n_val"]
    assert a["arrays"].keys() == b["arrays"].keys() and all(np.array_equal(a["arrays"][k], b["arrays"][k]) for k in a["arrays"])
    assert all(torch.equal(a["weights"][n][k], b["weights"][n][k]) for n in a["weights"] for k in a["weights"][n])


def test_base_explicit_layers_equal_full_run(tmp_path):
    """base 上只训指定层，与原来的 run_e2（全部层 + L4_pre）在同一层上逐字节相同——前提是探针所在的块一样大。
    探针按 probe_chunk 个一块做上采样与损失（tiny：3 个）；批内的矩阵乘与求和对块的大小 P 不是逐位不变的（与块里是哪几个探针、
    排第几无关）。L1 L2 L3 是 6 个探针、两块各 3 个，与全部层时每块 3 个一致 → 逐字节相同；L1 L3 是 4 个探针，L3/conv 单独一块
    （全部层时它在 3 个一块里）→ 只差 fp32 舍入。"""
    full, sub, two = tmp_path / "full", tmp_path / "sub", tmp_path / "two"
    assert _e2("--task", "depth", "refseg", "--out", str(full)) == 0
    assert _e2("--task", "depth", "refseg", "--layers", "L3", "L1", "L2", "--out", str(sub)) == 0
    assert _e2("--task", "depth", "--layers", "L3", "L1", "--out", str(two)) == 0
    names = [f"L{l}/{k}" for l in (1, 2, 3) for k in ("linear", "conv")]
    for task in ("depth", "refseg"):
        ms = json.loads((sub / KEY / task / "metrics.json").read_text())
        assert sorted(ms["probes"]) == sorted(names) and ms["layers"] == [1, 2, 3] and ms["pre_norm_last"] is None  # 只训这几层
        assert yaml.safe_load((sub / KEY / task / "config.yaml").read_text())["layers"]["spec"] == "L1 L2 L3"
        assert "checkpoint" not in yaml.safe_load((full / KEY / task / "config.yaml").read_text())  # base 的运行身份没有这个键
        _same(_probe_parts(full / KEY / task, names), _probe_parts(sub / KEY / task, names))
    same = ["L1/linear", "L1/conv", "L3/linear"]  # 块大小与全部层时相同
    _same(_probe_parts(full / KEY / "depth", same), _probe_parts(two / KEY / "depth", same))
    mf = json.loads((full / KEY / "depth" / "metrics.json").read_text())["probes"]["L3/conv"]["eval"]["abs_rel"]
    mt = json.loads((two / KEY / "depth" / "metrics.json").read_text())["probes"]["L3/conv"]["eval"]["abs_rel"]
    assert mt == pytest.approx(mf, rel=1e-4)  # 块大小不同：只差 fp32 舍入（记作偏离，见完成总结）


def test_zero_lora_checkpoint_equals_base(ckpts, tmp_path):
    """给 base（零 LoRA 的检查点）：输出与 base 的同一次运行逐字节相同，config.yaml 只多 checkpoint 一个键。"""
    base, ck = tmp_path / "base", tmp_path / "ck"
    args = ["--task", "depth", "refseg", "--layers", "L1", "L4", "L4_pre"]
    assert _e2(*args, "--out", str(base)) == 0
    assert _e2(*args, "--checkpoint", str(ckpts["zero"]), "--checkpoint-label", "zero/seed0", "--out", str(ck)) == 0
    for task in ("depth", "refseg"):
        lb, lc = base / KEY / task, ck / "zero" / "seed0" / KEY / task
        for f in ("curves.jsonl", "metrics.json", "eval_samples.npz"):
            assert (lb / f).read_bytes() == (lc / f).read_bytes(), (task, f)
        cb, cc = yaml.safe_load((lb / "config.yaml").read_text()), yaml.safe_load((lc / "config.yaml").read_text())
        rec = cc.pop("checkpoint")
        assert cb == cc and rec["manifest_sha256"] == E2CK.file_sha256(ckpts["zero"] / "manifest.json")
        assert rec["label"] == "zero/seed0" and rec["kind"] == "unknown" and rec["merge_check"]["checks"] == ["merge"]
        assert json.loads((lc / "metrics.json").read_text())["single_keys"] == ["L4_pre"]
    mc = json.loads((ck / "zero" / "seed0" / KEY / "merge_check.json").read_text())
    assert mc["pass"] is True


def test_small_checkpoints_run(ckpts, tmp_path):
    out = tmp_path / "o"
    assert _e2("--task", "refseg", "--layers", "L2", "L4", "--checkpoint", str(ckpts["e3b"]), "--checkpoint-label", "e3b/seed0",
               "--out", str(out)) == 0
    leaf = out / "e3b" / "seed0" / KEY / "refseg"
    rec = yaml.safe_load((leaf / "config.yaml").read_text())["checkpoint"]
    assert rec["kind"] == "e3b" and rec["experiment"] == "E3b" and rec["train_seed"] == 0 and rec["n_grid_tokens"] == 67
    assert rec["manifest_sha256"] == E2CK.file_sha256(ckpts["e3b"] / "manifest.json")
    m = json.loads((leaf / "metrics.json").read_text())
    assert m["complete"] and sorted(m["probes"]) == ["L2/conv", "L2/linear", "L4/conv", "L4/linear"]
    assert "检查点：e3b/seed0" in (leaf / "README.md").read_text()
    run = json.loads(sorted((leaf / "logs").glob("run_*.json"))[-1].read_text())
    assert run["checkpoint_dir"] == str(ckpts["e3b"]) and run["merge_check"]["pass"] is True  # 路径只进运行记录
    # E3a：给 final（model/ + readout/）；只用 model/ 的 LoRA，readout.json 只记 sha256
    assert _e2("--task", "depth", "--layers", "L1", "--checkpoint", str(ckpts["full"]), "--checkpoint-label", "e3a/full/seed0",
               "--out", str(out)) == 0
    rec = yaml.safe_load((out / "e3a" / "full" / "seed0" / KEY / "depth" / "config.yaml").read_text())["checkpoint"]
    assert rec["kind"] == "e3a_full" and rec["lora"] is not None and rec["readout_manifest_sha256"]
    assert rec["manifest_sha256"] == E2CK.file_sha256(ckpts["full"] / "model" / "manifest.json")
    ref = E2CK.resolve_checkpoint(ckpts["full"] / "model")  # 给 final/model 也认得出是 E3a
    assert ref.kind == "e3a_full" and ref.model_dir == ckpts["full"] / "model"
    ad = E2CK.load_for_probes(ref, device="cpu", dtype=torch.float32)
    assert ad.lora_state == "merged" and ad.token_table()["query"]["n"] is None  # 不登记查询 token


def test_frozen_llm_and_e5_refused(ckpts, tmp_path):
    for v in ("c1", "c2"):
        out = tmp_path / v
        assert _e2("--task", "depth", "--checkpoint", str(ckpts[v]), "--checkpoint-label", v, "--out", str(out)) == 2
        (err,) = (out / KEY / "errors").glob("error_*.json")
        assert "base 读数" in json.loads(err.read_text())["error"]
        assert not (out / v).exists()  # 不建任何输出目录（含合并核对）
        with pytest.raises(E2CK.RefusedCheckpoint, match="C1|C2"):
            E2CK.resolve_checkpoint(ckpts[v] / "model")
    bare = make_adapter(KEY, device="cpu", dtype=torch.float32, tiny=True).save_trained(tmp_path / "bare")
    with pytest.raises(E2CK.RefusedCheckpoint, match="base 读数"):
        E2CK.resolve_checkpoint(bare)
    fake_e5 = tmp_path / "e5final"
    fake_e5.mkdir()
    (fake_e5 / "e5_stack.json").write_text("{}")
    with pytest.raises(E2CK.RefusedCheckpoint, match="E5"):
        E2CK.resolve_checkpoint(fake_e5)


def test_merge_check_failure_refused(tmp_path):
    from tests.test_e5_train import _pathological_lora

    ad = make_adapter(KEY, device="cpu", dtype=torch.float32, tiny=True, init_std=_script("run_e2").TINY_INIT_STD)
    ad.inject_lora(r=2, lora_alpha=2, init_lora_weights=False)
    _pathological_lora(ad)
    bad = ad.save_trained(tmp_path / "bad")
    out = tmp_path / "o"
    assert _e2("--task", "depth", "--checkpoint", str(bad), "--checkpoint-label", "bad", "--out", str(out)) == 1
    mdir = out / "bad" / KEY
    assert sorted(p.name for p in mdir.iterdir()) == ["merge_check.json"]  # 没有任何任务目录
    doc = json.loads((mdir / "merge_check.json").read_text())
    assert doc["pass"] is False and doc["failures"]["merge"]


def test_cli_rules(ckpts, tmp_path):
    from src.adapters import registry  # noqa: F401  (run_e2 的 parse_args 要它)

    p = _script("run_e2").parse_args
    ck = str(ckpts["zero"])
    for bad in (["--checkpoint", ck], ["--checkpoint", ck, "--checkpoint-label", "../x"], ["--checkpoint-label", "x"],
                ["--checkpoint", ck, "--checkpoint-label", "x", "--estimate"], ["--merge-check-n", "3"],
                ["--checkpoint", str(tmp_path / "missing"), "--checkpoint-label", "x"]):
        with pytest.raises(SystemExit):
            p(["--tiny", "--model", KEY, "--task", "depth", "--out", str(tmp_path), *bad])
    a = p(["--tiny", "--model", KEY, "--task", "depth", "--out", str(tmp_path), "--layers", "L4", "L2"])
    assert a.layers == "L2 L4" and a.checkpoint is None
    assert _e2("--task", "depth", "--layers", "L9", "--out", str(tmp_path / "o")) == 2  # 小模型只有 4 层
    assert E2CK.CHECKPOINT_LABEL.pattern == _script("run_e1").CHECKPOINT_LABEL.pattern  # 与 run_e1 同一个标签规则
