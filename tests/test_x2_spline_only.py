"""DEV-0070：探索 X-3——只训描样条（configs/x3_train.json = configs/x2_train.json 去掉 correct 一节，样条每步 6 → 10 条）。
CPU fp32 小型随机 Qwen，测试造的数据（同 tests/test_x2_train.py：tests/_x2_fakes 的 w2c 假行 → build_x2_data 的规则；回放数据同
tests/test_w2c_replay）。

- 配置：x3 除 version、status、spline 的条数与说明文字（x2_data.rule、spline.basis、x2_streams_basis）外与 x2 逐项相同、键序相同；
  只有 spline 一节时 x2_spec 只开样条流（流号 4、微批 4 + 4 + 2）；只有 correct 一节仍然报错；每步 8 + 8 + 10 + 3 + 2 = 31；
  文档字符串里的用时估计与配置说明里的 epoch 数按 RUN-0080 的产物重算（产物在仓库里时）；
- 数据（src/explore/x2_data 的 kinds）：只读、只核对样条的行与文件，参数只比 spline 一节；两类都读时与 DEV-0069 的读法逐项相同；
- **两节都在时逐字节不变**：configs/x2_train.json 用 51dffb25（DEV-0069 合入）的 w2c_train、train_w2c、x2_data 原文件与现在的各跑一遍
  （分段 + 续跑），确定性的输出、检查点、运行身份逐字节相同；
- tiny 上只有样条一条 X-2 流（加两条回放流）：中途崩溃后续跑与一次跑完逐位相同；运行身份、README、meta.json 只有样条；数据读的类与配置
  的流不一致时拒绝；
- 运行器：有修正流的检查点上轨迹与运行身份同 51dffb25 的 x2_runner；没有修正流时轨迹不调修正，命令行端到端（train_w2c --config
  configs/x3_train.json → run_x2）修正类设置跳过、不写目录，x2_summary 写明"检查点没有修正流"，其余设置照常（W2 格式、gate_explore 能读）。
"""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import statistics
from pathlib import Path

import pytest
import torch
import yaml

from src.e3.train import leaf_dir
from src.explore import x2_data as XD
from src.explore import x2_prompts as XP
from src.explore import x2_runner as XRN
from src.explore import x2_tools as XT
from src.w2 import runner as R
from src.w2 import w2c_train as T
from src.w2.data import tiny_splits
from tests import _x2_fakes as XF
from tests.test_x2_runner import FAKE_IMPL, FAKES, PROMPTS, RC, RUN_X2, _cand_rows, _inject_boxes, _load_gate_explore, _samples
from tests.test_x2_train import (  # noqa: F401 — w2c_dir、data、x2_run、replay_run 是 fixture
    DET, IMPL, QWEN, TRAIN, W2, _load_path, _spec, _tensors, _tiny, data, jsonl, load_rdata, replay_run, run, w2c_dir, x2_run,
)

REPO = Path(__file__).resolve().parents[1]
CFG_X = T.load_w2c_config(T.X2_CONFIG_PATH)
CFG_3 = T.load_w2c_config(T.X3_CONFIG_PATH)
SNAP_DIR = REPO / "tests" / "fixtures" / "x2"
SNAP_51 = {  # 51dffb25（DEV-0069 合入）的原文件与 sha256
    "w2c_train": ("w2c_train_51dffb25.py", "686b254267db577677e70ee26b17d3e624bda14d9fabe1899a619bc11f982885"),
    "train_w2c": ("train_w2c_51dffb25.py", "6af6d26829afe6fa9be250a670d12b18dc9b1eced9000c83599d6ea7c81043fd"),
    "x2_data": ("x2_data_51dffb25.py", "b7b88f108544aa358837080fa4e0693651aa04e61065c0be9742b9146d1d3a06"),
    "x2_runner": ("x2_runner_51dffb25.py", "10c78a8caa440f9d168acc70daed183ed43dca6e9b36c09ec5e9ed8d5a105a3b"),
}
R80 = REPO / "experiments" / "X" / "0080_x2_hpcgpu07"
R80_LEAF = R80 / "0080_train" / "w2c" / "qwen3.5-9b" / "seed0"
R80_MANIFEST = R80 / "0080_x2_data" / "x2_data" / "manifest.json"
PARAMS_X = XD.check_params(XD.DEFAULT_PARAMS)
PARAMS_S = XD.check_params(XD.DEFAULT_PARAMS, kinds=("spline",))


def _snap(name: str):
    f, sha = SNAP_51[name]
    assert hashlib.sha256((SNAP_DIR / f).read_bytes()).hexdigest() == sha, f
    return _load_path(SNAP_DIR / f, f"_{name}_51dffb25")


def _snapshot_train():
    old_xd = _snap("x2_data")
    old_t = _snap("w2c_train")
    # 快照放在 tests/fixtures/ 下，按自己的位置推出的仓库根不对；x2_data 也换成同一时刻的原文件
    old_t.REPO_ROOT, old_t.W2C_CONFIG_PATH = T.REPO_ROOT, T.W2C_CONFIG_PATH
    old_t.W2C_REPLAY_CONFIG_PATH, old_t.X2_CONFIG_PATH = T.W2C_REPLAY_CONFIG_PATH, T.X2_CONFIG_PATH
    old_t.XD = old_xd
    old_s = _snap("train_w2c")
    old_s.T, old_s.XD = old_t, old_xd
    return old_t, old_s, old_xd


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------


def test_x3_config_is_x2_without_correct():
    a, b = CFG_3.doc, CFG_X.doc
    assert list(a) == [k for k in b if k != "correct"]  # 去掉 correct 一节，键序不变
    assert {k for k in a if a[k] != b[k]} == {"version", "status", "x2_data", "spline", "x2_streams_basis"}
    assert list(a["x2_data"]) == list(b["x2_data"]) and {k for k in a["x2_data"] if a["x2_data"][k] != b["x2_data"][k]} == {"rule"}
    assert list(a["spline"]) == list(b["spline"]) and {k for k in a["spline"] if a["spline"][k] != b["spline"][k]} == {"per_step", "basis"}
    assert (a["spline"]["per_step"], a["spline"]["tiny_per_step"], b["spline"]["per_step"]) == (10, 2, 6)
    assert a["version"] == "x3-train-v1" and "DEV-0070" in a["status"] and "X-2 一节" in a["status"] and a["x2_data"]["path"] is None
    rc, rx = T.resolve_w2c_run(CFG_3, tiny=False), T.resolve_w2c_run(CFG_X, tiny=False)
    x = rc["x2"]
    assert x["order"] == ["spline"] and x["per_step_total"] == 10
    assert {s: (v["per_step"], v["loss_weight"], v["index"], v["micro_batches"], v["order"]) for s, v in x["streams"].items()} == \
        {"spline": (10, 1.0, 4, 3, "image_first")}  # 微批 4：4 + 4 + 2
    assert x["data_params"] == PARAMS_S == {"spline": PARAMS_X["spline"]}
    assert x["streams"]["spline"] == {**rx["x2"]["streams"]["spline"], "per_step": 10, "micro_batches": 3}
    assert {k: v for k, v in rc.items() if k != "x2"} == {k: v for k, v in rx.items() if k != "x2"}  # 其余训练参数逐项相同
    assert T.stream_names(rc) == ("propose", "select", "spline", "replay_general", "replay_count") and T.stream_order(rc) == list(T.stream_names(rc))
    assert T.stream_weights(rc) == {"propose": 1.0, "select": 1.0, "spline": 1.0, "replay_general": 0.5, "replay_count": 0.5}
    assert rc["batch_size"] + x["per_step_total"] + rc["replay"]["per_step_total"] == 31 and rc["steps"] == 2000
    assert "31 条" in a["x2_streams_basis"] and "4 + 4 + 2" in a["x2_streams_basis"]
    rt = T.resolve_w2c_run(CFG_3, tiny=True)
    assert {s: v["per_step"] for s, v in rt["x2"]["streams"].items()} == {"spline": 2}
    # 配置说明里的 epoch 数：2,000 步 × 10 条 / 13,987 行
    assert f"{2000 * 10 / 13987:.2f}" == "1.43" and "13,987" in a["spline"]["basis"] and "1.43" in a["spline"]["basis"]


@pytest.mark.skipif(not (R80_LEAF / "train_log.jsonl").is_file() or not R80_MANIFEST.is_file(), reason="RUN-0080 的产物不在仓库里")
def test_estimates_recomputed_from_run_0080():
    """scripts/train_w2c.py 的 X-3 用时估计与配置里的样条池行数，按 RUN-0080 的产物重算。"""
    assert json.loads(R80_MANIFEST.read_text())["counts"]["spline"] == 13987
    log = jsonl(R80_LEAF / "train_log.jsonl")
    tok = {s: sum(r["by_stream"][s]["n_prompt_tokens"] + r["by_stream"][s]["n_target_tokens"] for r in log) / len(log) for s in log[0]["by_stream"]}
    assert {s: round(v) for s, v in tok.items()} == {"propose": 3122, "select": 13328, "spline": 3446, "correct": 3195, "replay_general": 990,
                                                     "replay_count": 685}
    assert {s: log[0]["by_stream"][s]["n_samples"] for s in tok} == {"propose": 8, "select": 8, "spline": 6, "correct": 6, "replay_general": 3,
                                                                     "replay_count": 2}
    xs = [json.loads(x)["step_s"] for f in sorted((R80_LEAF / "logs").glob("timing_run_*.jsonl")) for x in f.read_text().splitlines()
          if x.strip() and json.loads(x)["step"] >= 5]
    med = statistics.median(xs)
    tok = {s: round(v) for s, v in tok.items()}  # 文档字符串里写的是各流取整后的数
    tot = sum(tok.values())
    x3 = tot - tok["correct"] + tok["spline"] * (10 - 6) / 6  # 去掉修正，样条 6 → 10 条
    assert round(med, 1) == 17.5 and round(tot) == 24766 and round(x3) == 23868
    lo, hi = 17.5 * 31 / 33, 17.5 * x3 / tot  # 按任务书：每步 17.5 s 折算
    assert (round(lo, 1), round(hi, 1)) == (16.4, 16.9)
    assert (round(2000 * lo / 3600, 1), round(2000 * hi / 3600, 1)) == (9.1, 9.4)
    doc = (REPO / "scripts" / "train_w2c.py").read_text(encoding="utf-8")
    for s in ("16.4–16.9 s/步", "9.1–9.4 GPU 小时", "24,766", "23,868", "17.5 / 33 × 31 ≈ 16.4", "≈ 16.9"):
        assert s in doc, s
    assert "16.4–16.9 s/步" in CFG_3.doc["x2_streams_basis"] and "9.1–9.4 GPU 小时" in CFG_3.doc["x2_streams_basis"]


@pytest.mark.parametrize("edit,msg", [
    (lambda d: (d.pop("spline"), d.__setitem__("correct", copy.deepcopy(CFG_X.doc["correct"]))), "不能单独有"),  # 只有 correct：仍然报错
    (lambda d: d.pop("x2_data"), "x2_data"),
    (lambda d: d["spline"].update(per_step=0), "per_step"),
    (lambda d: d["spline"].update(tiny_per_step=1.5), "tiny_per_step"),
    (lambda d: d["spline"].update(loss_weight=0), "loss_weight"),
    (lambda d: d["spline"].update(order="instruction_first"), "image_order"),
    (lambda d: d["spline"].update(n=2), "spline.n"),
    (lambda d: d["spline"].pop("min_box_iou"), "min_box_iou"),
    (lambda d: d["spline"].update(m=2), "未知的键"),
])
def test_x3_config_validation(tmp_path, edit, msg):
    d = copy.deepcopy(dict(CFG_3.doc))
    edit(d)
    p = tmp_path / "c.json"
    p.write_text(json.dumps(d, ensure_ascii=False))
    with pytest.raises(T.W2CConfigError, match=msg):
        T.load_w2c_config(p)


def test_check_params_kinds():
    assert XD.check_params(XD.DEFAULT_PARAMS, kinds=XD.KINDS) == PARAMS_X and list(PARAMS_X) == ["spline", "correct"]
    assert XD.check_params({"spline": XD.DEFAULT_PARAMS["spline"]}, kinds=("spline",)) == PARAMS_S
    with pytest.raises(XD.X2DataError, match="spline 与 correct 两节"):  # 缺省两类：缺一节照旧报错
        XD.check_params({"spline": XD.DEFAULT_PARAMS["spline"]})
    with pytest.raises(XD.X2DataError, match="spline 一节"):
        XD.check_params({"correct": XD.DEFAULT_PARAMS["correct"]}, kinds=("spline",))
    with pytest.raises(XD.X2DataError, match="spline.n"):
        XD.check_params({"spline": {"n": 3, "min_box_iou": 0.5}}, kinds=("spline",))
    for bad in ((), ("correct", "spline"), ("spline", "spline"), ("ellipse",)):
        with pytest.raises(XD.X2DataError, match="kinds"):
            XD.check_params(XD.DEFAULT_PARAMS, kinds=bad)


# ---------------------------------------------------------------------------
# 数据：只读样条
# ---------------------------------------------------------------------------


def test_load_spline_only(x2_run, w2c_dir, tmp_path):
    src = XD.source_identity(w2c_dir, require_manifest=True)
    full = XD.load_x2_data(x2_run, params=PARAMS_X, source=src, data_root=XF.ROOT)
    sp = XD.load_x2_data(x2_run, params=PARAMS_S, source=src, data_root=XF.ROOT)  # kinds 缺省 = params 里有的类
    assert sp == XD.load_x2_data(x2_run, params=PARAMS_S, source=src, data_root=XF.ROOT, kinds=("spline",))
    assert (full.kinds, sp.kinds) == (XD.KINDS, ("spline",))
    assert sp.rows == tuple(full.rows[i] for i in full.pool("spline")) and all(r["kind"] == "spline" for r in sp.rows)
    assert sp.pool("spline") == tuple(range(len(sp.rows)))
    assert full.params == full.manifest["params"] == PARAMS_X and sp.params == PARAMS_S
    fi, si = full.identity(), sp.identity()
    assert si["params"] == PARAMS_S and set(si["pools"]) == {"spline"} and si["pools"]["spline"] == fi["pools"]["spline"]
    assert {k: v for k, v in si.items() if k not in ("params", "pools")} == {k: v for k, v in fi.items() if k not in ("params", "pools")}
    d = x2_run / XD.DATA_DIR
    man = json.loads((d / XD.MANIFEST).read_text())
    assert full.file_paths() == [d / f["file"] for f in man["files"]] + [d / XD.MANIFEST]
    assert sp.file_paths() == [d / f["file"] for f in man["files"] if f["kind"] == "spline"] + [d / XD.MANIFEST]
    # 两类都读：与 51dffb25 的读法逐项相同
    old_xd = _snap("x2_data")
    old = old_xd.load_x2_data(x2_run, params=PARAMS_X, source=src, data_root=XF.ROOT)
    assert (old.rows, old.pools, old.manifest_sha256, old.manifest, old.identity(), old.file_paths()) == \
        (full.rows, full.pools, full.manifest_sha256, full.manifest, full.identity(), full.file_paths())
    # 修正一类的文件坏了、参数不同：只读样条时照样能读，两类一起读就拒绝
    bad = tmp_path / "bad"
    shutil.copytree(x2_run, bad)
    cf = bad / XD.DATA_DIR / XD.out_name("correct", "shard_0000.jsonl")
    cf.write_text(cf.read_text() + "garbage\n")
    assert XD.load_x2_data(bad, params=PARAMS_S, source=src).rows == sp.rows
    with pytest.raises(XD.X2DataError, match="sha256"):
        XD.load_x2_data(bad, params=PARAMS_X, source=src)
    p2 = json.loads(json.dumps(PARAMS_X))
    p2["correct"]["round2_ratio"] = 1.0
    with pytest.raises(XD.X2DataError, match="参数"):
        XD.load_x2_data(x2_run, params=p2, source=src)
    p3 = json.loads(json.dumps(PARAMS_S))
    p3["spline"]["n"] = 12
    with pytest.raises(XD.X2DataError, match="参数"):  # 样条的参数照样核对
        XD.load_x2_data(x2_run, params=p3, source=src)
    with pytest.raises(XD.X2DataError, match="另一份 w2c_data"):  # 来源照样核对
        XD.load_x2_data(x2_run, params=PARAMS_S, source=dict(src, manifest_sha256="0" * 64))
    sf = bad / XD.DATA_DIR / XD.out_name("spline", "shard_0000.jsonl")
    sf.write_text(sf.read_text() + "garbage\n")
    with pytest.raises(XD.X2DataError, match="sha256"):  # 样条文件照样核对
        XD.load_x2_data(bad, params=PARAMS_S, source=src)


# ---------------------------------------------------------------------------
# 两节都在时逐字节不变（51dffb25 的快照）
# ---------------------------------------------------------------------------


def test_x2_config_bytewise_identical_to_dev0069(w2c_dir, x2_run, replay_run, tmp_path, capsys):
    old_t, old_s, old_xd = _snapshot_train()
    new_s = _load_path(TRAIN, "_train_w2c_dev0070")
    oc = old_t.load_w2c_config(REPO / CFG_X.path)
    assert (oc.sha256, oc.path, oc.doc) == (CFG_X.sha256, CFG_X.path, CFG_X.doc)
    for tiny in (False, True):
        assert old_t.resolve_w2c_run(oc, tiny=tiny) == T.resolve_w2c_run(CFG_X, tiny=tiny)
        assert old_t.x2_spec(oc.doc, tiny=tiny) == T.x2_spec(CFG_X.doc, tiny=tiny)
    for c in (T.load_w2c_config(), T.load_w2c_config(T.W2C_REPLAY_CONFIG_PATH)):  # 没有 X-2 两节的配置：仍是 None
        assert T.x2_spec(c.doc, tiny=False) is None and old_t.x2_spec(c.doc, tiny=False) is None
    extra = ["--config", str(REPO / CFG_X.path), "--x2-data", str(x2_run), "--replay-data", str(replay_run)]
    outs, prints = [], []
    for k, mod in enumerate((old_s, new_s)):
        out = tmp_path / f"o{k}"
        base = ["--tiny", "--model", QWEN, "--w2c-data", str(w2c_dir), "--out", str(out), *extra]
        assert mod.main(base + ["--stop-after-steps", "3"]) == 0
        assert mod.main(base + ["--resume"]) == 0
        prints.append(capsys.readouterr().out)
        outs.append(leaf_dir(out, "w2c", QWEN, 0))
    assert prints[0] == prints[1] and "steps=4" in prints[1] and "'correct'" in prints[1]
    a, b = outs
    for f in DET:  # train_log、config.yaml（运行身份）、README、train_summary
        assert (a / f).read_bytes() == (b / f).read_bytes(), f
    for f in ("manifest.json", "tokens.json"):
        assert (a / "ckpt/final/model" / f).read_bytes() == (b / "ckpt/final/model" / f).read_bytes(), f
    ta, tb = _tensors(a / "ckpt/final/model"), _tensors(b / "ckpt/final/model")
    assert set(ta) == set(tb) and all(torch.equal(ta[x], tb[x]) for x in ta)
    acc = [json.loads((x / "accounting.json").read_text()) for x in outs]
    assert list(acc[0]) == list(acc[1]) and acc[0]["by_stream"] == acc[1]["by_stream"]
    for x in ("run_001.json", "run_002.json"):
        ra, rb = (json.loads((o / "logs" / x).read_text()) for o in outs)
        assert list(ra) == list(rb) and ra["paths"] == rb["paths"] and ra["audit"] == rb["audit"]
    ma, mb = (json.loads((o / "meta.json").read_text()) for o in outs)
    assert ma["data_sha256"] == mb["data_sha256"] and any("correct_" in k for k in mb["data_sha256"])
    assert sorted(p.relative_to(a).as_posix() for p in a.rglob("*")) == sorted(p.relative_to(b).as_posix() for p in b.rglob("*"))
    x = yaml.safe_load((b / "config.yaml").read_text())["x2"]
    assert set(x["params"]) == {"spline", "correct"} and "render" in x and set(x["streams"]) == {"spline", "correct"}


# ---------------------------------------------------------------------------
# tiny 训练：只有样条
# ---------------------------------------------------------------------------


def _rc3(tiny=True):
    return T.resolve_w2c_run(CFG_3, tiny=tiny)


@pytest.fixture(scope="module")
def xdata3(x2_run, w2c_dir):
    rc = _rc3()
    return XD.load_x2_data(x2_run, params=rc["x2"]["data_params"], source=XD.source_identity(w2c_dir, require_manifest=True),
                           data_root=XF.ROOT, kinds=tuple(rc["x2"]["order"]))


def _train3(ad, data, xdata, rdata, out, *, cfg=CFG_3, **kw):
    rc = T.resolve_w2c_run(cfg, tiny=True)
    w2_info = T.resolve_w2_config(cfg, tiny=True)[1]
    return T.run_w2c_training(ad, cfg, _spec(ad), rc, data, w2=W2, w2_info=w2_info, impl=IMPL, data_root=XF.ROOT, out_dir=out,
                              log=lambda m: None, replay=rdata, x2=xdata, **kw)


def test_spline_only_schedule(data, xdata3, replay_run):
    rc = _rc3(tiny=False)
    rdata = load_rdata(replay_run, rc)
    total = rc["x2"]["per_step_total"] + rc["replay"]["per_step_total"]
    sched = T.W2CSchedule(data, seed=0, per_step=rc["streams"]["per_step"], order=rc["streams"]["order"], extra_per_step=total)
    xs = T.X2Schedule(xdata3, seed=0, spec=rc["x2"], base_batch=sched.batch_size, step_size=sched.step_size)
    rs = T.ReplaySchedule(rdata, seed=0, spec=rc["replay"], base_batch=sched.batch_size + rc["x2"]["per_step_total"])
    assert sched.step_size == 31 and xs.batch_size == 26 and rs.batch_size == 31
    for step in (0, 3):
        plans = sched.step_plans(step) + xs.step_plans(step) + rs.step_plans(step)
        assert [p.k for p in plans] == list(range(step * 31, step * 31 + 31))
        assert [p.stream for p in plans] == ["propose"] * 8 + ["select"] * 8 + ["spline"] * 10 + ["replay_general"] * 3 + ["replay_count"] * 2
    n = len(xdata3.pool("spline"))
    assert sorted(xs.plan("spline", j).row for j in range(n)) == sorted(xdata3.pool("spline"))
    # 同一个种子、同一个流号：样条的样本顺序（按 id）与两类都读时相同
    full = T.X2Schedule(XD.load_x2_data(xdata3.run_dir, params=PARAMS_X, source=None), seed=0,
                        spec=T.resolve_w2c_run(CFG_X, tiny=False)["x2"], base_batch=16, step_size=33)
    assert [xs.plan("spline", j).sample_id for j in range(2 * n)] == [full.plan("spline", j).sample_id for j in range(2 * n)]


def test_spline_only_training_resume_identity(data, xdata3, replay_run, x2_run, w2c_dir, tmp_path):
    rdata = load_rdata(replay_run, _rc3())
    a, b = tmp_path / "A", tmp_path / "B"
    sa = _train3(_tiny(), data, xdata3, rdata, a)
    assert sa["complete"] and sa["steps"] == 4
    assert sa["x2"] == {"manifest_sha256": xdata3.manifest_sha256, "pools": {"spline": len(xdata3.pool("spline"))}, "params": PARAMS_S}
    with pytest.raises(RuntimeError, match="测试注入的中断"):
        _train3(_tiny(), data, xdata3, rdata, b, _fail_after_step=3)
    sb = _train3(_tiny(), data, xdata3, rdata, b, resume=True)
    la, lb = leaf_dir(a, "w2c", QWEN, 0), leaf_dir(b, "w2c", QWEN, 0)
    for f in DET:
        assert (la / f).read_bytes() == (lb / f).read_bytes(), f
    ta, tb = _tensors(la / "ckpt" / "final" / "model"), _tensors(lb / "ckpt" / "final" / "model")
    assert set(ta) == set(tb) and all(torch.equal(ta[k], tb[k]) for k in ta)
    assert sa == sb
    log = jsonl(la / "train_log.jsonl")
    assert [r["step"] for r in log] == [0, 1, 2, 3]
    assert all(r["n_by_stream"] == {"propose": 2, "select": 2, "spline": 2, "replay_general": 2, "replay_count": 1} for r in log)
    lb_ = lambda r, s: r["loss_by_stream"][s]  # noqa: E731
    assert all(r["loss"] == pytest.approx(sum(lb_(r, s) for s in ("propose", "select", "spline"))
                                          + 0.5 * lb_(r, "replay_general") + 0.5 * lb_(r, "replay_count")) for r in log)
    ident = yaml.safe_load((la / "config.yaml").read_text())
    assert ident["variant"] == "w2c" and ident["config"]["sha256"] == CFG_3.sha256 and ident["config"]["version"] == "x3-train-v1"
    x = ident["x2"]
    assert x["params"] == PARAMS_S and set(x["pools"]) == {"spline"} and set(x["streams"]) == {"spline"} and "render" not in x
    assert x["prompts"] == {"version": XP.PROMPT_VERSION, "spline": XP.SPLINE_TEMPLATE} and "没有修正流" in x["sample"]
    assert x["manifest_sha256"] == xdata3.manifest_sha256 and x["streams"]["spline"]["index"] == 4
    assert ident["training"]["x2"]["order"] == ["spline"] and ident["training"]["x2"]["per_step_total"] == 2
    run1 = json.loads((la / "logs" / "run_001.json").read_text())
    assert {e["stream"] for e in run1["audit"]["first_step_examples"]} == {"propose", "select", "spline", "replay_general", "replay_count"}
    assert "没有修正流" in (la / "README.md").read_text() and "spline 2 条" in (la / "README.md").read_text()
    meta = json.loads((la / "meta.json").read_text())
    xfiles = [k for k in meta["data_sha256"] if str(x2_run) in k]
    assert sorted(xfiles) == sorted(str(p) for p in xdata3.file_paths()) and not any("correct_" in k for k in xfiles)
    acc = json.loads((la / "accounting.json").read_text())
    assert set(acc["by_stream"]) == {"propose", "select", "spline", "replay_general", "replay_count"}
    # 数据读的类与配置的流不一致：拒绝
    full = XD.load_x2_data(x2_run, params=PARAMS_X, source=None, data_root=XF.ROOT)
    with pytest.raises(ValueError, match="参数"):
        _train3(_tiny(), data, full, rdata, tmp_path / "C")
    with pytest.raises(ValueError, match="参数"):
        _train3(_tiny(), data, xdata3, load_rdata(replay_run), tmp_path / "D", cfg=CFG_X)
    with pytest.raises(ValueError, match="同时有"):
        _train3(_tiny(), data, None, rdata, tmp_path / "E")


# ---------------------------------------------------------------------------
# 运行器
# ---------------------------------------------------------------------------


def _ctx(mod, ad, pool, *, m):
    bat = RC.batching(tiny=True)
    return mod.ChunkContext(adapter=ad, w2=W2, rc=RC, impl=FAKE_IMPL, prompts=PROMPTS, n=16, m=m, batch_size=bat["batch_size"],
                            max_batch_tokens=bat["max_batch_tokens"], tiny=True, pool=pool)


def test_runner_trace_with_and_without_correction():
    old = _snap("x2_runner")
    samples = _samples()
    rows = _cand_rows(samples)
    with XT.ToolPool(1) as pool:
        ref, _ = old.process_chunk(_ctx(old, XF.ScriptedX2Adapter(select="2"), pool, m=2), samples, rows)
        new, _ = XRN.process_chunk(_ctx(XRN, XF.ScriptedX2Adapter(select="2"), pool, m=2), samples, rows)
        ad = XF.ScriptedX2Adapter(select="2")
        sp, tms = XRN.process_chunk(_ctx(XRN, ad, pool, m=None), samples, rows)
    assert new == ref  # 有修正流：与 51dffb25 逐项相同
    assert all(t["cor"] and t["randbox"] for t in ref)
    assert sp == [{k: v for k, v in t.items() if k not in ("cor", "randbox")} for t in ref]  # 没有修正流：只少了修正与随机框
    assert not any("current mask" in c["prompt"] for c in ad.calls)  # 不调修正
    assert all(not any(k.startswith(("cor_", "randbox_")) for k in {**tm["model_s"], **tm["tool_s"]}) for tm in tms)


def test_runner_identity_and_checkpoint_params(tmp_path):
    old = _snap("x2_runner")
    samples = _samples()
    rows = _cand_rows(samples)
    out = tmp_path / "out"
    prop = out / "p"
    prop.mkdir(parents=True)
    (prop / "predictions.jsonl").write_text("{}\n")
    (prop / "config.yaml").write_text("a: 1\n")
    spec = R.W2RunSpec(model_key=QWEN, family="qwen3.5", repo="r", revision="v", adapter_model_key=QWEN, task="refseg", split="devcheck2",
                       setting="x2", tiny=True, limit=None, image_policy={"tag": "checkpoint"})
    kw = dict(prop_dir=prop, cand_leaf=out / "c", cand_rows=list(rows.values()), impl=FAKE_IMPL, checkpoint={"manifest_sha256": "0" * 64},
              batching=RC.batching(tiny=True))
    sd = tiny_splits(W2)
    x2f = {"n": 16, "m": 2, "params": PARAMS_X, "manifest_sha256": "a" * 64}
    assert XRN.trace_identity(spec, W2, RC, sd, samples, out, x2=x2f, **kw) == old.trace_identity(spec, W2, RC, sd, samples, out, x2=x2f, **kw)
    x2s = {"n": 16, "m": None, "params": PARAMS_S, "manifest_sha256": "a" * 64}
    a = XRN.trace_identity(spec, W2, RC, sd, samples, out, x2=x2f, **kw)
    b = XRN.trace_identity(spec, W2, RC, sd, samples, out, x2=x2s, **kw)
    assert {k: v for k, v in b.items() if k != "x2"} == {k: v for k, v in a.items() if k != "x2"}
    assert b["x2"] == {**a["x2"], "m": None, "train_params": PARAMS_S, "correction": b["x2"]["correction"]}
    assert b["x2"]["correction"].startswith(XRN.NO_CORRECTION) and "correction" not in a["x2"]
    # 检查点的 x2 一节：有 correct → m；没有 → m = None、不核对叠加图版本；提示版本照样核对
    leaf = tmp_path / "w2c" / QWEN / "seed0"
    ck = leaf / "ckpt" / "final" / "model"
    ck.mkdir(parents=True)
    from src.explore import x2_render as XR_

    full = {"params": PARAMS_X, "prompts": {"version": XP.PROMPT_VERSION}, "render": {"version": XR_.RENDER_VERSION}, "manifest_sha256": "m"}
    (leaf / "config.yaml").write_text(yaml.safe_dump({"variant": "w2c", "x2": full}))
    assert XRN.checkpoint_x2_params(ck) == old.checkpoint_x2_params(ck) and XRN.checkpoint_x2_params(ck)["m"] == 2
    (leaf / "config.yaml").write_text(yaml.safe_dump({"variant": "w2c", "x2": {"params": PARAMS_S, "prompts": {"version": XP.PROMPT_VERSION},
                                                                                "manifest_sha256": "m"}}))
    got = XRN.checkpoint_x2_params(ck)
    assert got == {"n": 16, "m": None, "params": PARAMS_S, "manifest_sha256": "m"} and not XRN.correction_trained(got)
    (leaf / "config.yaml").write_text(yaml.safe_dump({"variant": "w2c", "x2": {"params": PARAMS_S, "prompts": {"version": "old"}}}))
    with pytest.raises(XRN.X2RunError, match="版本"):
        XRN.checkpoint_x2_params(ck)
    if (R80_LEAF / "config.yaml").is_file():  # RUN-0080 的真检查点的运行身份：m = 2，与 51dffb25 的读法相同
        real = tmp_path / "r80" / "w2c" / "qwen3.5-9b" / "seed0"
        (real / "ckpt" / "final" / "model").mkdir(parents=True)
        shutil.copy(R80_LEAF / "config.yaml", real / "config.yaml")
        rk = real / "ckpt" / "final" / "model"
        assert XRN.checkpoint_x2_params(rk) == old.checkpoint_x2_params(rk) and XRN.checkpoint_x2_params(rk)["m"] == 2


@pytest.fixture(scope="module")
def x3_ckpt(w2c_dir, x2_run, replay_run, tmp_path_factory):
    """tiny 的 X-3 检查点（命令行：train_w2c --config configs/x3_train.json --x2-data <X-2 的同一份数据>）。"""
    tmp = tmp_path_factory.mktemp("x3ck")
    base = ["--tiny", "--model", QWEN, "--config", T.X3_CONFIG_PATH, "--w2c-data", w2c_dir, "--replay-data", replay_run]
    p = run(TRAIN, base + ["--out", tmp / "n"], tmp, ok=False)
    assert p.returncode == 2 and "x2_data.path" in p.stderr and "占位" in p.stderr
    p = run(TRAIN, base + ["--x2-data", x2_run, "--out", tmp / "train"], tmp)
    assert "steps=4" in p.stdout and "'spline'" in p.stdout and "'correct'" not in p.stdout
    return tmp / "train" / "w2c" / QWEN / "seed0" / "ckpt" / "final" / "model"


def test_cli_run_x2_on_spline_only_checkpoint(x3_ckpt, tmp_path):
    tmp = tmp_path
    out = tmp / "ev"
    base = ["--tiny", "--model", QWEN, "--checkpoint", x3_ckpt, "--split", "devcheck2", "--out", out, *FAKES]
    run(RUN_X2, base, tmp)
    m = out / QWEN
    defs = XRN.setting_defs(3)
    _inject_boxes(m / "refexp_box" / "propose_w2c" / "devcheck2")
    for d in ["candidates_w2c", XRN.TRACE_SETTING, XRN.SUMMARY_DIR, *defs]:
        shutil.rmtree(m / "refseg" / d, ignore_errors=True)
    p = run(RUN_X2, base + ["--tool-workers", "2"], tmp)
    cor = [n for n, d in defs.items() if d["kind"] in XRN.CORRECTION_KINDS]
    keep = [n for n in defs if n not in cor]
    assert len(cor) == 12 and all("_cor" in n for n in cor) and all("_cor" not in n for n in keep)
    assert keep == ["x2_select", "x2_spline_raster", "x2_spline_trimap", "x2_select5", "x2_oracle4", "x2_oracle5", "x2_ellipse_raster",
                    "x2_ellipse_trimap"]
    assert "跳过 12 个设置：检查点没有修正流" in p.stdout and "x2_select=" in p.stdout
    ck_sha = hashlib.sha256((x3_ckpt / "manifest.json").read_bytes()).hexdigest()
    ge = _load_gate_explore()
    for name in cor:
        assert not (m / "refseg" / name).exists()
    for name in keep:
        leaf = m / "refseg" / name / "devcheck2"
        recs = jsonl(leaf / "predictions.jsonl")
        assert len(recs) == 3 and recs[0]["failure"] is not None and all(r["failure"] is None for r in recs[1:])
        cfg = yaml.safe_load((leaf / "config.yaml").read_text())
        assert cfg["w2"]["setting"] == name and cfg["checkpoint"]["manifest_sha256"] == ck_sha
        assert cfg["x2"]["setting"] == name and cfg["x2"]["m"] is None and cfg["x2"]["correction"].startswith(XRN.NO_CORRECTION)
        assert cfg["x2"]["train_params"] == PARAMS_S
        met = json.loads((leaf / "metrics.json").read_text())
        assert met["complete"] is True and met["n"] == 3 and [h["name"] for h in met["headline"]] == ["cIoU", "gIoU"]
        assert json.loads((leaf / "compute.json").read_text())["n_calls"] == sum(r["n_model_calls"] for r in recs)
        g = ge.load_dir(leaf, "候选")  # scripts/gate_explore.py 的读法
        assert g["identity"]["setting"] == name and len(g["records"]) == 3
    summ = json.loads((m / "refseg" / XRN.SUMMARY_DIR / "devcheck2.json").read_text())
    assert list(summ["settings"]) == keep and summ["skipped"] == {"reason": "检查点没有修正流", "settings": cor}
    assert summ["compute_pairing"] == {}  # 多轮设置都跳过了
    tr = jsonl(m / "refseg" / XRN.TRACE_SETTING / "devcheck2" / "trace.jsonl")
    assert len(tr) == 3 and not any("cor" in t or "randbox" in t for t in tr)
    # --settings 只要修正类：全部跳过并记录
    out2 = tmp / "ev2"
    shutil.copytree(out, out2)
    for d in [XRN.SUMMARY_DIR, *keep]:
        shutil.rmtree(out2 / QWEN / "refseg" / d, ignore_errors=True)
    run(RUN_X2, ["--tiny", "--model", QWEN, "--checkpoint", x3_ckpt, "--split", "devcheck2", "--out", out2, *FAKES, "--resume",
                 "--settings", "x2_select", "x2_select_cor1"], tmp)
    s2 = json.loads((out2 / QWEN / "refseg" / XRN.SUMMARY_DIR / "devcheck2.json").read_text())
    assert list(s2["settings"]) == ["x2_select"] and s2["skipped"]["settings"] == ["x2_select_cor1"]
    assert not (out2 / QWEN / "refseg" / "x2_select_cor1").exists()
