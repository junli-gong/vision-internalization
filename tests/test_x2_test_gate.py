"""DEV-0072（D147）：scripts/run_x2.py 在 test 划分上只经通过 G-EX 的候选与负责人的批准编号放行。

- gate_explore.json 用 scripts/gate_explore.py 真写出一份（构造的 devcheck3、候选 = 本次的检查点、设置 x2_spline_trimap），
  键名与真的判定一致；判定点核对的键与 run_x2 写死的 GATE_POINT_CHECKS 相同；
- 放行：两个参数都给且合格 → 跑完；运行身份（config.yaml 的 w2.test_unlock，提出、轨迹、各设置都有）与 <out>/<模型>/test_unlock.jsonl
  有记录；只导出 --settings 给的两个设置；
- 拒绝（退出码 2，什么都不写）：不通过、彩排、判定点核对有假、manifest 不符、设置不符或没给 --settings、导出未经判定的设置、批准编号不对、
  缺任一参数、模型不同、配 --splits-file、非 test 划分给了这两个参数；两个参数都不给时与以前逐字相同；
- test 的句子：与 run_w2 的 test 划分同一份清单（configs/w2_splits.json）、同一个取样函数，逐 id 相同（tiny 现算；真清单 5 × 1,000 句）；
- 其余划分不变：W2RunSpec 不带放行时 w2 一节的键与以前相同（tests/test_x2_runner 的端到端另钉 devcheck2、devcheck3 的输出）。
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from src.data import explore_splits as EX
from src.w2 import runner as R
from src.w2.config import load_w2_config
from src.w2.data import W2_SPLITS_PATH, tiny_paths, tiny_splits, w2_items
from tests.test_gate_explore import scaled, scores, side, toy_manifest
from tests.test_x2_runner import FAKES, QWEN, RUN_X2, jsonl, run, x2_ckpt  # noqa: F401 — x2_ckpt 是 fixture

REPO = Path(__file__).resolve().parents[1]
W2 = load_w2_config()
TEST_SPLIT = "test_refcoco_unc_testA"
TEST_SPLITS = ["test_refcoco_unc_testA", "test_refcoco_unc_testB", "test_refcoco+_unc_testA", "test_refcoco+_unc_testB", "test_refcocog_umd_test"]
SETTINGS = ["x2_spline_trimap", "x2_ellipse_trimap"]
OLD_REFUSAL = f"X-2 是探索：不碰 test（预注册 X §1），拒绝 --split {TEST_SPLIT}"  # DEV-0069 起的原文


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def rx():
    return _load("_run_x2_under_test", RUN_X2)


@pytest.fixture(scope="module")
def gate_mod():
    return _load("_gate_explore_for_test_gate", REPO / "scripts" / "gate_explore.py")


def write_gate(gate_mod, tmp: Path, ck_sha: str, *, setting: str = "x2_spline_trimap", model: str = QWEN, host: str = "hpcgpu07") -> Path:
    """scripts/gate_explore.py 在构造的 devcheck3 上真判一次（候选比 W2-c 好 15%，规则 (i) 通过；没给健康检查）→ gate_explore.json。"""
    tmp.mkdir(parents=True, exist_ok=True)
    path, doc = toy_manifest(tmp)
    ids = EX.group_ids(doc, EX.DEVCHECK3)
    ref = scores(ids, seed=1)
    refs = side(tmp / "ref", doc, ref, checkpoint={"manifest_sha256": "c" * 64}, model=model, host=host)
    cands = side(tmp / "cand", doc, scaled(ref, 1.15), setting=setting, checkpoint={"manifest_sha256": ck_sha}, model=model, host=host)
    out = tmp / "gate_C1"
    args = ["--candidate-name", "C1", "--unseal-devcheck3", "G-EX", "--splits-file", str(path), "--reference", *map(str, refs),
            "--candidate", *map(str, cands), "--out-dir", str(out)]
    assert gate_mod.main(args) == 0
    g = out / "gate_explore.json"
    d = json.loads(g.read_text(encoding="utf-8"))
    assert d["verdict"] == "通过" and d["gate_point"]["is_gate_point"] and d["rules"]["i"]["pass"]
    return g


@pytest.fixture(scope="module")
def fake_ck(tmp_path_factory):
    """parse_args 只看检查点目录与它的 manifest.json：拒绝的用例不用训练。"""
    d = tmp_path_factory.mktemp("fakeck") / "model"
    d.mkdir()
    (d / "manifest.json").write_text(json.dumps({"build_spec": {"key": QWEN, "tiny": True}}), encoding="utf-8")
    return d


@pytest.fixture(scope="module")
def fake_gate(gate_mod, fake_ck, tmp_path_factory):
    sha = hashlib.sha256((fake_ck / "manifest.json").read_bytes()).hexdigest()
    return write_gate(gate_mod, tmp_path_factory.mktemp("gfake"), sha)


def _edited(src: Path, dst: Path, fn) -> Path:
    d = json.loads(src.read_text(encoding="utf-8"))
    fn(d)
    dst.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    return dst


# ---------------------------------------------------------------------------
# 常量与 gate_explore.json 的键
# ---------------------------------------------------------------------------


def test_constants_and_gate_keys(rx, fake_gate):
    assert rx.TEST_APPROVAL == "D147" and rx.TEST_ABLATIONS == ("x2_ellipse_trimap",) and rx.TEST_UNLOCK_LOG == "test_unlock.jsonl"
    d = json.loads(fake_gate.read_text(encoding="utf-8"))
    assert tuple(d["gate_point"]["checks"]) == rx.GATE_POINT_CHECKS  # scripts/gate_explore.py 写的键，一个不少
    assert rx.GATE_PASS == "通过" and rx.GATE_REHEARSAL in "彩排，不作数（规则给出：通过）"
    assert {"verdict", "decision", "pass", "identity", "runs", "candidate_name"} <= set(d)
    assert all({"setting", "checkpoint"} <= set(c) for c in d["runs"]["candidate"])
    assert all(rx.XR.setting_defs(3)[s]["ablation"] == "ellipse" for s in rx.TEST_ABLATIONS)


# ---------------------------------------------------------------------------
# 拒绝（parse_args：用法错误，退出码 2，什么都不写）
# ---------------------------------------------------------------------------


def _args(ck, out, *extra, split=TEST_SPLIT):
    return ["--tiny", "--model", QWEN, "--checkpoint", str(ck), "--split", split, "--out", str(out), *FAKES, *map(str, extra)]


def _refused(rx, capsys, argv) -> str:
    with pytest.raises(SystemExit) as e:
        rx.parse_args(argv)
    assert e.value.code == 2
    return capsys.readouterr().err


def test_refusals(rx, fake_ck, fake_gate, tmp_path, capsys):
    out = tmp_path / "o"
    ok = ["--settings", *SETTINGS, "--test-gate", fake_gate, "--test-approval", "D147"]
    a = rx.parse_args(_args(fake_ck, out, *ok))  # 合格：放行
    assert a.split == TEST_SPLIT and a.test_approval == "D147"
    cases = [
        ("不通过", lambda d: d.update(verdict="不通过", decision="不通过", **{"pass": False})),
        ("彩排", lambda d: (d.update(verdict="彩排，不作数（规则给出：通过）"), d["gate_point"]["checks"].update(same_host=False),
                           d["gate_point"].update(is_gate_point=False))),
        ("判定点", lambda d: d["gate_point"]["checks"].update(n_is_500=False)),  # 结论照写"通过"但核对有假（手改）也不认
        ("判定点", lambda d: d["gate_point"]["checks"].pop("same_host")),
        ("不是本次的 --checkpoint", lambda d: [c["checkpoint"].update(manifest_sha256="d" * 64) for c in d["runs"]["candidate"]]),
        ("没有判定过的设置 x2_select", lambda d: [c.update(setting="x2_select") for c in d["runs"]["candidate"]]),
        ("判的是模型", lambda d: d["identity"].update(model="qwen3.5-9b")),
        ("不是 X-2 的一个设置", lambda d: d["runs"]["candidate"][0].update(setting="x2_spline_raster")),
    ]
    for i, (msg, fn) in enumerate(cases):
        g = _edited(fake_gate, tmp_path / f"g{i}.json", fn)
        err = _refused(rx, capsys, _args(fake_ck, out, "--settings", *SETTINGS, "--test-gate", g, "--test-approval", "D147"))
        assert msg in err, (msg, err[-400:])
    g = _edited(fake_gate, tmp_path / "gcheck.json", lambda d: d["gate_point"]["checks"].update(n_is_500=False))
    assert "n_is_500" in _refused(rx, capsys, _args(fake_ck, out, *ok[:-4], "--test-gate", g, "--test-approval", "D147"))
    gate_args = ["--test-gate", fake_gate, "--test-approval", "D147"]
    for argv, msg in [
        (_args(fake_ck, out, "--settings", *SETTINGS, "--test-gate", fake_gate, "--test-approval", "D146"), "必须恰是 D147"),
        (_args(fake_ck, out, "--settings", *SETTINGS, "--test-gate", fake_gate, "--test-approval", "d147"), "必须恰是 D147"),
        (_args(fake_ck, out, "--settings", *SETTINGS, "--test-gate", fake_gate), "要一起给"),
        (_args(fake_ck, out, "--settings", *SETTINGS, "--test-approval", "D147"), "要一起给"),
        (_args(fake_ck, out, *gate_args), "要用 --settings"),
        (_args(fake_ck, out, "--settings", "x2_ellipse_trimap", *gate_args), "没有判定过的设置 x2_spline_trimap"),
        (_args(fake_ck, out, "--settings", *SETTINGS, "x2_select", *gate_args), "没有经过判定"),
        (_args(fake_ck, out, "--settings", *SETTINGS, "x2_oracle4", *gate_args), "没有经过判定"),
        (_args(fake_ck, out, "--settings", *SETTINGS, "--test-gate", tmp_path / "nowhere.json", "--test-approval", "D147"), "读不了"),
        (_args(fake_ck, out, "--settings", *SETTINGS, "--splits-file", REPO / "configs" / "explore_splits.json", *gate_args),
         "configs/w2_splits.json"),
        (_args(fake_ck, out, "--settings", *SETTINGS, *gate_args, split="devcheck2"), "只配 test 划分"),
        (_args(fake_ck, out, "--test-approval", "D147", split="devtune800"), "只配 test 划分"),
    ]:
        assert msg in _refused(rx, capsys, argv), msg
    # 两个参数都不给：与以前逐字相同（在其余检查之前，同原来的位置）
    err = _refused(rx, capsys, _args(fake_ck, out, "--settings", *SETTINGS))
    assert err.rstrip("\n").splitlines()[-1] == f"{Path(sys.argv[0]).name}: error: {OLD_REFUSAL}"
    assert not out.exists()


def test_refusals_cli_write_nothing(fake_ck, fake_gate, tmp_path):
    out = tmp_path / "t"
    p = run(RUN_X2, _args(fake_ck, out), tmp_path, ok=False)  # 两个参数都不给
    assert p.returncode == 2 and p.stderr.rstrip("\n").splitlines()[-1] == f"run_x2.py: error: {OLD_REFUSAL}" and not out.exists()
    g = _edited(fake_gate, tmp_path / "other_ck.json",
                lambda d: [c["checkpoint"].update(manifest_sha256="e" * 64) for c in d["runs"]["candidate"]])
    p = run(RUN_X2, _args(fake_ck, out, "--settings", *SETTINGS, "--test-gate", g, "--test-approval", "D147"), tmp_path, ok=False)
    assert p.returncode == 2 and "不是本次的 --checkpoint" in p.stderr and not out.exists()


# ---------------------------------------------------------------------------
# test 的句子：与 run_w2 同一份清单、同一个取样函数
# ---------------------------------------------------------------------------


def _run_w2_ids(split: str, *, tiny: bool) -> tuple[list[str], dict]:
    rw2 = _load("_run_w2_for_test_gate", REPO / "scripts" / "run_w2.py")
    ns = argparse.Namespace(splits_file=str(W2_SPLITS_PATH), tiny=tiny, data_root=None, task="refseg")
    doc = rw2.load_splits_doc(ns, W2)
    if not tiny:
        return list(doc["splits"][split]["ids"]), doc
    return [s.id for s in w2_items("refseg", split, rw2.data_paths(ns, doc), splits=doc)], doc


def test_test_samples_are_run_w2s(rx):
    rw2 = _load("_run_w2_for_test_gate2", REPO / "scripts" / "run_w2.py")
    for fn in ("load_splits_doc", "data_paths", "is_default_splits"):  # run_x2 用的是 run_w2 的同一个函数（按路径加载，不复制）
        assert getattr(rx.RW2, fn).__code__.co_code == getattr(rw2, fn).__code__.co_code, fn
    assert rx.w2_items is w2_items
    want = json.loads(W2_SPLITS_PATH.read_text(encoding="utf-8"))
    for sp in TEST_SPLITS:  # 真清单：5 个 test 划分，各 1,000 句，就是 configs/w2_splits.json 的 test_*（W2-c 的 RUN-0072 用的那份）
        ns = argparse.Namespace(splits_file=str(W2_SPLITS_PATH), tiny=False, data_root="/x")
        doc = rx.RW2.load_splits_doc(ns, W2)
        ids, _ = _run_w2_ids(sp, tiny=False)
        assert doc["splits"][sp]["ids"] == ids == want["splits"][sp]["ids"] and len(ids) == 1000 and "refseg" in doc["splits"][sp]["tasks"]


# ---------------------------------------------------------------------------
# 放行：端到端（tiny 的 X-2 检查点）
# ---------------------------------------------------------------------------


def test_unlock_end_to_end(x2_ckpt, gate_mod, tmp_path):  # noqa: F811
    sha = hashlib.sha256((x2_ckpt / "manifest.json").read_bytes()).hexdigest()
    g = write_gate(gate_mod, tmp_path / "g", sha)
    g_sha = hashlib.sha256(g.read_bytes()).hexdigest()
    out = tmp_path / "ev"
    argv = _args(x2_ckpt, out, "--settings", *SETTINGS, "--test-gate", g, "--test-approval", "D147")
    p = run(RUN_X2, argv, tmp_path)
    assert "x2_spline_trimap=" in p.stdout and "x2_ellipse_trimap=" in p.stdout
    m = out / QWEN
    want_unlock = {"approval": "D147", "gate_explore": {"path": str(g), "sha256": g_sha}, "candidate_name": "C1",
                   "candidate_setting": "x2_spline_trimap"}
    ids, doc = _run_w2_ids(TEST_SPLIT, tiny=True)
    assert len(ids) == 2
    for name in SETTINGS:
        leaf = m / "refseg" / name / TEST_SPLIT
        cfg = yaml.safe_load((leaf / "config.yaml").read_text(encoding="utf-8"))
        assert cfg["w2"]["test_unlock"] == want_unlock and cfg["w2"]["split"] == TEST_SPLIT and cfg["checkpoint"]["manifest_sha256"] == sha
        assert cfg["w2"]["split_sha256"] == doc["splits"][TEST_SPLIT]["sha256"] and "unseal" not in cfg["w2"]
        recs = jsonl(leaf / "predictions.jsonl")
        assert [r["id"] for r in recs] == ids  # 与 run_w2 的 test 划分逐 id 相同（同一顺序）
        assert cfg["w2"]["sample_ids_sha256"] == R.ids_digest(ids)
    tcfg = yaml.safe_load((m / "refseg" / "x2_trace" / TEST_SPLIT / "config.yaml").read_text(encoding="utf-8"))
    assert tcfg["w2"]["test_unlock"] == want_unlock
    pcfg = yaml.safe_load((m / "refexp_box" / "propose_w2c" / TEST_SPLIT / "config.yaml").read_text(encoding="utf-8"))
    assert pcfg["data"]["selection_stats"]["w2"]["test_unlock"] == want_unlock
    exported = {d.name for d in (m / "refseg").iterdir()}
    assert exported == {"candidates_w2c", "x2_trace", "x2_summary", *SETTINGS}  # 只导出判定过的设置与批准的消融
    summ = json.loads((m / "refseg" / "x2_summary" / f"{TEST_SPLIT}.json").read_text(encoding="utf-8"))
    assert set(summ["settings"]) == set(SETTINGS)
    log = jsonl(m / "test_unlock.jsonl")
    assert len(log) == 1 and {k: log[0][k] for k in want_unlock} == want_unlock
    assert log[0]["split"] == TEST_SPLIT and log[0]["settings"] == SETTINGS and log[0]["checkpoint_manifest_sha256"] == sha
    assert log[0]["gate"]["verdict"] == "通过" and log[0]["argv"] == [str(x) for x in argv]
    assert not (m / "unseal_devcheck3.jsonl").exists()
    # 续跑：同一个放行，身份相同；追加一行放行记录
    before = (m / "refseg" / "x2_spline_trimap" / TEST_SPLIT / "predictions.jsonl").read_bytes()
    run(RUN_X2, argv + ["--resume"], tmp_path)
    assert (m / "refseg" / "x2_spline_trimap" / TEST_SPLIT / "predictions.jsonl").read_bytes() == before
    assert len(jsonl(m / "test_unlock.jsonl")) == 2


# ---------------------------------------------------------------------------
# 其余划分不变
# ---------------------------------------------------------------------------


def test_w2_record_unchanged_without_unlock():
    """W2RunSpec 不带放行时，w2 一节的键与 DEV-0072 之前相同（run_w2 的各设置、run_x2 的非 test 划分）；带 unseal 时同以前。"""
    doc = tiny_splits(W2)
    samples = w2_items("refseg", "devcheck2", tiny_paths(), splits=doc)
    spec = R.W2RunSpec(model_key=QWEN, family="qwen3.5", repo="r", revision="v", adapter_model_key=QWEN, task="refseg", split="devcheck2",
                       setting="w2c", tiny=True, limit=None, image_policy={})
    base = {"task", "split", "setting", "splits_config", "splits_sha256", "split_sha256", "split_n", "n_samples", "limit", "sample_ids_sha256"}
    assert set(R.w2_record(spec, doc, samples)) == base
    import dataclasses

    u = dataclasses.replace(spec, unseal={"gate": "G-EX", "decision": "x"})
    assert set(R.w2_record(u, doc, samples)) == base | {"unseal"}
    t = dataclasses.replace(spec, test_unlock={"approval": "D147"})
    assert R.w2_record(t, doc, samples) == {**R.w2_record(spec, doc, samples), "test_unlock": {"approval": "D147"}}
    assert [f.name for f in dataclasses.fields(R.W2RunSpec)][-2:] == ["unseal", "test_unlock"]  # 新字段在末尾、有默认值


def test_non_test_splits_take_no_unlock(rx, fake_ck, tmp_path):
    a = rx.parse_args(_args(fake_ck, tmp_path / "o", split="devcheck2"))
    assert a.test_gate is None and a.test_approval is None
    ex = ["--splits-file", REPO / "configs" / "explore_splits.json"]
    a = rx.parse_args(_args(fake_ck, tmp_path / "o", *ex, split="explore_val_refcoco_unc_val"))
    assert a.test_gate is None and not hasattr(a, "test_unlock")
