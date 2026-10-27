"""DEV-0069：X-2 的训练（src/w2/w2c_train.py 的 spline、correct 两条流，scripts/train_w2c.py --x2-data，configs/x2_train.json）。
CPU fp32 小型随机 Qwen，测试造的数据（tests/_x2_fakes 的 w2c 假行 → scripts/build_x2_data.py 的规则；回放数据同 tests/test_w2c_replay）。

- 配置：configs/x2_train.json 除 version、status 与新增的 x2_data、spline、correct、x2_streams_basis 外与 configs/w2c_replay_train.json
  逐项相同；每步 8 + 8 + 6 + 6 + 3 + 2 = 33，权重 1 : 1 : 1 : 1 : 0.5 : 0.5；流号 spline = 4、correct = 5；结构核对；
- **缺省配置逐字节不变**（7f53f240 的快照，同 DEV-0065）；回放配置也逐字节不变（DEV-0065 合入后的 9749cf1b 的快照）；
- 日程无状态、覆盖池、一步里的全局编号连续；样本（描样条：原图 + 提示 → 控制点文本；指修正框：叠加图逐位等于 x2_render 的输出）；
- 损失：六条流各自除以本流的监督 token 数再乘权重（含不满的微批）；
- tiny 上四流加两条回放流一起跑：中途崩溃后续跑、分段运行与一次跑完逐位相同；运行身份、记账、README；X-2 数据换了就拒绝续跑；
- 命令行：--x2-data 的用法；检查点能被 run_w2.py --setting w2c 加载（scripts/run_x2.py 的加载见 tests/test_x2_runner.py）。
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from src.adapters import registry
from src.adapters.checkpoint import policy_spec
from src.adapters.factory import make_adapter
from src.adapters.image_policy import parse_policy
from src.e3.train import leaf_dir
from src.e3.tokens import family_stop_id
from src.executors.raster import rle_to_mask
from src.explore import x2_data as XD
from src.explore import x2_prompts as XP
from src.explore import x2_render as XR
from src.w2 import runner as R
from src.w2 import w2c_replay as RP
from src.w2 import w2c_train as T
from src.w2.config import load_w2_config
from tests import _x2_fakes as XF

REPO = Path(__file__).resolve().parents[1]
TRAIN = REPO / "scripts" / "train_w2c.py"
BUILD_RP = REPO / "scripts" / "build_w2c_replay.py"
RUN_W2 = REPO / "scripts" / "run_w2.py"
QWEN = "qwen3.5-0.8b"
CFG = T.load_w2c_config()
CFG_R = T.load_w2c_config(T.W2C_REPLAY_CONFIG_PATH)
CFG_X = T.load_w2c_config(T.X2_CONFIG_PATH)
W2 = load_w2_config()
IMPL = R.load_impl(R.DEFAULT_CANDIDATES_MODULE, R.DEFAULT_RENDER_MODULE)
SNAP = {  # 快照：(目录, w2c_train 文件, train_w2c 文件, 两个文件的 sha256)
    "7f53f240": (REPO / "tests" / "fixtures" / "w2c", "w2c_train_7f53f240.py", "train_w2c_7f53f240.py",
                 ("ce3450d240e5bf4ae822ccea15cd1a9efffce743e5f969211df7ded3f49a3d12",
                  "51d015b76f243ec2acd6e23fca8cc2f9712d71407035cc63ef8aa2e57af0b256")),
    "9749cf1b": (REPO / "tests" / "fixtures" / "x2", "w2c_train_9749cf1b.py", "train_w2c_9749cf1b.py",
                 ("96dc8db7a64b2b7a5531401db4482783e3b503dd62bd995ece75693292ca2c5b",
                  "354ec7ef5258a99077bded2f85e898bdcaf01ba2ba32118ccd20dc0b09d4f346")),
}
DET = ("train_log.jsonl", "config.yaml", "README.md", "train_summary.json")
X2_ONLY = ("version", "status", "x2_data", "spline", "correct", "x2_streams_basis")


def run(script: Path, args, tmp: Path, *, ok: bool = True) -> subprocess.CompletedProcess:
    env = {**os.environ, "TMPDIR": str(tmp), "HF_HUB_OFFLINE": "1"}
    env.pop("VISINT_DATA_ROOT", None)
    p = subprocess.run([sys.executable, str(script), *map(str, args)], capture_output=True, text=True, cwd=REPO, env=env, timeout=1800)
    if ok:
        assert p.returncode == 0, p.stderr[-3000:]
    return p


def jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def _tiny():
    fam = registry.get_model(QWEN).family
    t = CFG_X.doc["tiny"]
    return make_adapter(QWEN, device="cpu", dtype=torch.float32, image_policy=parse_policy(t["image_policies"][fam]), tiny=True,
                        init_std=float(t["init_std"]))


def _spec(ad, seed=0):
    e = registry.get_model(QWEN)
    return T.W2CRunSpec(model_key=e.key, family=e.family, repo=e.repo, revision=e.revision, adapter_model_key=ad.model_key, tiny=True,
                        image_policy=policy_spec(ad.image_policy), model_dtype="fp32", seed=seed)


def _tensors(d: Path) -> dict:
    from safetensors.torch import load_file

    out = {}
    for f in sorted(d.rglob("*.safetensors")):
        for k, v in load_file(str(f)).items():
            out[f"{f.relative_to(d)}:{k}"] = v
    return out


@pytest.fixture(scope="module")
def w2c_dir(tmp_path_factory):
    return XF.write_fake_w2c_data(tmp_path_factory.mktemp("x2t") / "w2c_data")


@pytest.fixture(scope="module")
def data(w2c_dir):
    return T.load_w2c_data(w2c_dir, w2=W2, sources=CFG.doc["data"]["sources"], data_root=XF.ROOT, require_manifest=True)


@pytest.fixture(scope="module")
def x2_run(w2c_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp("x2d") / "x2"
    XD.build(w2c_dir, out, data_root=XF.ROOT, params=T.resolve_w2c_run(CFG_X, tiny=True)["x2"]["data_params"], tiny=True,
             log=lambda m: None)
    return out


def load_xdata(run_dir: Path, w2c_dir: Path, rc=None):
    rc = rc or T.resolve_w2c_run(CFG_X, tiny=True)
    return XD.load_x2_data(run_dir, params=rc["x2"]["data_params"], source=XD.source_identity(w2c_dir, require_manifest=True),
                           data_root=XF.ROOT)


@pytest.fixture(scope="module")
def xdata(x2_run, w2c_dir):
    return load_xdata(x2_run, w2c_dir)


@pytest.fixture(scope="module")
def replay_run(w2c_dir, tmp_path_factory):
    """回放数据（同 tests/test_w2c_replay：命令行 --tiny 生成，回答换成以停止 token 结尾的短回答）。"""
    from tests.test_w2c_replay import inject_stops

    tmp = tmp_path_factory.mktemp("x2rp")
    out = tmp / "rp"
    run(BUILD_RP, ["--tiny", "--w2c-data", w2c_dir, "--out", out], tmp)
    dst = tmp / "rpi"
    shutil.copytree(out, dst)
    inject_stops(dst)
    return dst


def load_rdata(run_dir: Path, rc=None):
    rc = rc or T.resolve_w2c_run(CFG_X, tiny=True)
    e = registry.get_model(QWEN)
    return RP.load_replay_data(run_dir, streams={s: v["kinds"] for s, v in rc["replay"]["streams"].items()}, model_key=e.key,
                               revision=e.revision, tiny=True, data_root=XF.ROOT)


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------


def test_config_is_replay_config_plus_x2_sections():
    a = {k: v for k, v in CFG_X.doc.items() if k not in X2_ONLY}
    b = {k: v for k, v in CFG_R.doc.items() if k not in ("version", "status")}
    assert a == b and list(a) == list(b)  # 其余逐项相同（含回放、LoRA、优化器、步数、W2 配置的 sha256）
    assert CFG_X.doc["version"] == "x2-train-v1" and CFG_X.doc["x2_data"]["path"] is None
    rc = T.resolve_w2c_run(CFG_X, tiny=False)
    x = rc["x2"]
    assert x["order"] == ["spline", "correct"] and x["per_step_total"] == 12
    assert {s: (v["per_step"], v["loss_weight"], v["index"], v["micro_batches"], v["order"]) for s, v in x["streams"].items()} == \
        {"spline": (6, 1.0, 4, 2, "image_first"), "correct": (6, 1.0, 5, 2, "image_first")}
    assert x["data_params"] == XD.check_params(XD.DEFAULT_PARAMS)
    assert T.stream_names(rc) == ("propose", "select", "spline", "correct", "replay_general", "replay_count")
    assert T.stream_order(rc) == list(T.stream_names(rc))
    assert T.stream_weights(rc) == {"propose": 1.0, "select": 1.0, "spline": 1.0, "correct": 1.0, "replay_general": 0.5, "replay_count": 0.5}
    assert rc["batch_size"] + x["per_step_total"] + rc["replay"]["per_step_total"] == 33 and rc["steps"] == 2000
    assert (rc["lora"], rc["optim"]) == (T.resolve_w2c_run(CFG, tiny=False)["lora"], T.resolve_w2c_run(CFG, tiny=False)["optim"])
    rt = T.resolve_w2c_run(CFG_X, tiny=True)
    assert {s: v["per_step"] for s, v in rt["x2"]["streams"].items()} == {"spline": 2, "correct": 2}
    for c in (CFG, CFG_R):  # 没有 spline、correct 两节：没有 x2 键
        assert "x2" not in T.resolve_w2c_run(c, tiny=False) and T.x2_spec(c.doc, tiny=False) is None
    assert T.stream_names(T.resolve_w2c_run(CFG, tiny=False)) == T.STREAMS


@pytest.mark.parametrize("edit,msg", [
    (lambda d: d.pop("spline"), "不能单独有"),  # DEV-0070：可以只有 spline（tests/test_x2_spline_only.py），只有 correct 仍然报错
    (lambda d: d.pop("x2_data"), "x2_data"),
    (lambda d: d["x2_data"].update(path=""), "x2_data"),
    (lambda d: d["spline"].update(per_step=0), "per_step"),
    (lambda d: d["correct"].update(tiny_per_step=1.5), "tiny_per_step"),
    (lambda d: d["spline"].update(loss_weight=0), "loss_weight"),
    (lambda d: d["correct"].update(loss_weight=True), "loss_weight"),
    (lambda d: d["spline"].update(order="instruction_first"), "image_order"),
    (lambda d: d["spline"].update(extra=1), "未知的键"),
    (lambda d: d["spline"].update(n=2), "spline.n"),
    (lambda d: d["correct"].update(round2_ratio=2), "round2_ratio"),
    (lambda d: d["correct"].pop("round2_salt"), "round2_salt"),
])
def test_x2_config_validation(tmp_path, edit, msg):
    d = copy.deepcopy(dict(CFG_X.doc))
    edit(d)
    p = tmp_path / "c.json"
    p.write_text(json.dumps(d, ensure_ascii=False))
    with pytest.raises(T.W2CConfigError, match=msg):
        T.load_w2c_config(p)


# ---------------------------------------------------------------------------
# 缺省配置、回放配置逐字节不变（快照对照）
# ---------------------------------------------------------------------------


def _load_path(path: Path, name: str):
    saved = list(sys.path)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    sys.path[:] = saved
    return mod


def _snapshot(tag: str):
    d, ft, fs, shas = SNAP[tag]
    for f, sha in zip((ft, fs), shas):
        assert hashlib.sha256((d / f).read_bytes()).hexdigest() == sha, f
    old_t = _load_path(d / ft, f"_w2c_train_{tag}")
    old_t.REPO_ROOT, old_t.W2C_CONFIG_PATH = T.REPO_ROOT, T.W2C_CONFIG_PATH  # 快照放在 tests/fixtures/ 下，按自己的位置推出的仓库根不对
    if hasattr(old_t, "W2C_REPLAY_CONFIG_PATH"):
        old_t.W2C_REPLAY_CONFIG_PATH = T.W2C_REPLAY_CONFIG_PATH
    old_s = _load_path(d / fs, f"_train_w2c_{tag}")
    old_s.T = old_t
    return old_t, old_s


@pytest.mark.parametrize("tag,cfg", [("7f53f240", CFG), ("9749cf1b", CFG_R)])
def test_configs_without_x2_are_bytewise_identical_to_snapshots(tag, cfg, w2c_dir, replay_run, tmp_path, capsys):
    old_t, old_s = _snapshot(tag)
    new_s = _load_path(TRAIN, f"_train_w2c_dev0069_{tag}")
    oc = old_t.load_w2c_config(REPO / cfg.path)
    assert (oc.sha256, oc.path, oc.doc) == (cfg.sha256, cfg.path, cfg.doc)
    for tiny in (False, True):
        assert old_t.resolve_w2c_run(oc, tiny=tiny) == T.resolve_w2c_run(cfg, tiny=tiny)
        assert old_t.resolve_w2c_run(oc, tiny=tiny, steps=7) == T.resolve_w2c_run(cfg, tiny=tiny, steps=7)
    extra = [] if cfg is CFG else ["--config", str(REPO / cfg.path), "--replay-data", str(replay_run)]
    outs, prints = [], []
    for k, mod in enumerate((old_s, new_s)):
        out = tmp_path / f"o{k}"
        base = ["--tiny", "--model", QWEN, "--w2c-data", str(w2c_dir), "--out", str(out), *extra]
        assert mod.main(base + ["--stop-after-steps", "3"]) == 0
        assert mod.main(base + ["--resume"]) == 0
        prints.append(capsys.readouterr().out)
        outs.append(leaf_dir(out, "w2c", QWEN, 0))
    assert prints[0] == prints[1] and "steps=4" in prints[1]
    a, b = outs
    for f in DET:
        assert (a / f).read_bytes() == (b / f).read_bytes(), f
    for f in ("manifest.json", "tokens.json"):
        assert (a / "ckpt/final/model" / f).read_bytes() == (b / "ckpt/final/model" / f).read_bytes(), f
    ta, tb = _tensors(a / "ckpt/final/model"), _tensors(b / "ckpt/final/model")
    assert set(ta) == set(tb) and all(torch.equal(ta[x], tb[x]) for x in ta)
    acc = [json.loads((x / "accounting.json").read_text()) for x in outs]
    assert list(acc[0]) == list(acc[1]) and acc[0]["by_stream"] == acc[1]["by_stream"]
    for x in ("run_001.json", "run_002.json"):
        ra, rb = (json.loads((o / "logs" / x).read_text()) for o in outs)
        assert list(ra) == list(rb) and ra["paths"].keys() == rb["paths"].keys() and ra["audit"] == rb["audit"]
    assert sorted(p.relative_to(a).as_posix() for p in a.rglob("*")) == sorted(p.relative_to(b).as_posix() for p in b.rglob("*"))
    assert "x2" not in yaml.safe_load((b / "config.yaml").read_text())


# ---------------------------------------------------------------------------
# 日程与样本
# ---------------------------------------------------------------------------


def test_schedule_stateless_covers_pools_and_numbers_contiguously(data, xdata, replay_run):
    rc = T.resolve_w2c_run(CFG_X, tiny=False)
    rdata = load_rdata(replay_run, rc)
    total = rc["x2"]["per_step_total"] + rc["replay"]["per_step_total"]
    sched = T.W2CSchedule(data, seed=0, per_step=rc["streams"]["per_step"], order=rc["streams"]["order"], extra_per_step=total)
    xs = T.X2Schedule(xdata, seed=0, spec=rc["x2"], base_batch=sched.batch_size, step_size=sched.step_size)
    rs = T.ReplaySchedule(rdata, seed=0, spec=rc["replay"], base_batch=sched.batch_size + rc["x2"]["per_step_total"])
    assert sched.step_size == 33 and xs.batch_size == 28 and rs.batch_size == 33
    for step in (0, 3):
        plans = sched.step_plans(step) + xs.step_plans(step) + rs.step_plans(step)
        assert [p.k for p in plans] == list(range(step * 33, step * 33 + 33))
        assert [p.stream for p in plans] == ["propose"] * 8 + ["select"] * 8 + ["spline"] * 6 + ["correct"] * 6 + \
            ["replay_general"] * 3 + ["replay_count"] * 2
    for st in ("spline", "correct"):
        n = len(xdata.pool(st))
        rows = [xs.plan(st, j).row for j in range(n)]
        assert sorted(rows) == sorted(xdata.pool(st))  # 一个 epoch 恰好覆盖本流的池一次
        assert all(xdata.rows[r]["kind"] == st for r in rows)
    again = T.X2Schedule(xdata, seed=0, spec=rc["x2"], base_batch=16, step_size=33)
    assert [p.sample_id for p in again.step_plans(5)] == [p.sample_id for p in xs.step_plans(5)]
    other = T.X2Schedule(xdata, seed=1, spec=rc["x2"], base_batch=16, step_size=33)
    assert [other.plan("correct", j).row for j in range(20)] != [xs.plan("correct", j).row for j in range(20)]
    # 流号不与提出、挑选、回放重复
    assert T.pool_position(3, 50, seed=0, stream="spline", index=4) != T.pool_position(3, 50, seed=0, stream="x", index=2)


def test_x2_examples(xdata):
    ad = _tiny()
    enc = lambda s: list(ad.tokenizer(s, add_special_tokens=False)["input_ids"])  # noqa: E731
    rc = T.resolve_w2c_run(CFG_X, tiny=True)
    b = T.X2Builder(xdata, w2=W2, data_root=XF.ROOT, stop_id=family_stop_id(ad), encode=enc, spec=rc["x2"])
    xs = T.X2Schedule(xdata, seed=0, spec=rc["x2"], base_batch=4, step_size=11)
    seen = set()
    for step in range(6):
        for p in xs.step_plans(step):
            e = b.build(p)
            row = xdata.rows[p.row]
            seen.add((row["kind"], row.get("round")))
            with Image.open(XF.ROOT / row["image"]) as im:
                orig = im.convert("RGB")
            assert len(e.images) == 1 and e.order == "image_first" and e.n_supervised == len(e.target_ids)
            assert e.target_ids == enc(row["target_text"]) + [family_stop_id(ad)] and e.target_text == row["target_text"]
            if p.stream == "spline":
                assert np.array_equal(np.asarray(e.images[0]), np.asarray(orig))
                assert e.instruction == XP.spline_prompt(row["expression"], row["box_q"], 16)
                assert XP.parse_spline(e.target_text, n=16).value == row["ctrl_q"]
            else:
                ov = XR.render_overlay(orig, rle_to_mask(row["start_rle"]).astype(bool), row["box"], style=W2.style)
                assert np.array_equal(np.asarray(e.images[0]), np.asarray(ov)) and e.images[0].size == ov.size
                assert e.instruction == XP.correct_prompt(row["expression"], row["box_q"], 2)
                assert e.info["round"] == row["round"] and e.info["n_boxes"] == len(row["target"])
    assert {("spline", None), ("correct", 1), ("correct", 2)} <= seen
    bad = T.W2CPlan(stream="spline", k=0, step=0, j=0, index=0, row=xdata.pool("correct")[0], epoch=0,
                    sample_id=str(xdata.rows[xdata.pool("correct")[0]]["id"]))
    with pytest.raises(RuntimeError, match="对不上"):
        b.build(bad)


# ---------------------------------------------------------------------------
# 损失
# ---------------------------------------------------------------------------


def test_six_streams_loss_with_underfilled_microbatches(data, xdata, replay_run, monkeypatch, w2c_dir):
    d = copy.deepcopy(dict(CFG_X.doc))
    d["spline"]["tiny_per_step"] = 3  # 微批 2：2 + 1，最后一个不满
    cfg = T.W2CConfig(path="x", sha256="0" * 64, doc=d)
    rc = T.resolve_w2c_run(cfg, tiny=True)
    rdata = load_rdata(replay_run, rc)
    ad = _tiny()
    tr = T.W2CTrainer(ad, cfg, rc, seed=0, number_ids=None)
    tr.setup_fresh()
    enc = lambda s: list(ad.tokenizer(s, add_special_tokens=False)["input_ids"])  # noqa: E731
    b = T.W2CBuilder(cfg, data, w2=W2, impl=IMPL, data_root=XF.ROOT, stop_id=family_stop_id(ad), encode=enc, family="qwen3.5",
                     format_boxes=ad.format_boxes)
    xb = T.X2Builder(xdata, w2=W2, data_root=XF.ROOT, stop_id=family_stop_id(ad), encode=enc, spec=rc["x2"])
    rb = T.ReplayBuilder(rdata, data_root=XF.ROOT, stop_ids=ad.stop_token_ids())
    sched = T.W2CSchedule(data, seed=0, per_step=rc["streams"]["per_step"], order=rc["streams"]["order"],
                          extra_per_step=rc["x2"]["per_step_total"] + rc["replay"]["per_step_total"])
    xs = T.X2Schedule(xdata, seed=0, spec=rc["x2"], base_batch=sched.batch_size, step_size=sched.step_size)
    rs = T.ReplaySchedule(rdata, seed=0, spec=rc["replay"], base_batch=sched.batch_size + rc["x2"]["per_step_total"])
    plans = sched.step_plans(1) + xs.step_plans(1) + rs.step_plans(1)
    ex = [xb.build(p) if p.stream in rc["x2"]["streams"] else (rb.build(p) if p.stream in rc["replay"]["streams"] else b.build(p))
          for p in plans]
    sums, scaled = [], []
    orig_loss, orig_bwd = ad.chunked_lm_loss, torch.Tensor.backward

    def loss(*a, **k):
        out = orig_loss(*a, **k)
        sums.append(float(out.detach()))
        return out

    def bwd(self, *a, **k):
        scaled.append(float(self.detach()))
        return orig_bwd(self, *a, **k)

    monkeypatch.setattr(ad, "chunked_lm_loss", loss)
    monkeypatch.setattr(torch.Tensor, "backward", bwd)
    rec, _, _ = tr.optimize(ex)
    monkeypatch.undo()
    names = ["propose", "select", "spline", "correct", "replay_general", "replay_count"]
    assert rec["n_by_stream"] == {"propose": 2, "select": 2, "spline": 3, "correct": 2, "replay_general": 2, "replay_count": 1}
    chunks = ["propose", "select", "spline", "spline", "correct", "replay_general", "replay_count"]  # 微批的先后
    assert len(sums) == len(chunks)
    n, w = rec["n_supervised_by_stream"], T.stream_weights(rc)
    for st, v, sc in zip(chunks, sums, scaled):
        assert sc == pytest.approx(w[st] * v / n[st])
    assert rec["loss_by_stream"]["spline"] == pytest.approx((sums[2] + sums[3]) / n["spline"])
    assert rec["loss"] == pytest.approx(sum(w[s] * rec["loss_by_stream"][s] for s in names))
    assert n["spline"] == sum(len(e.target_ids) for e in ex if e.plan.stream == "spline") and n["spline"] > 3 * 100  # 16 个点约 160 个 token


# ---------------------------------------------------------------------------
# tiny 训练
# ---------------------------------------------------------------------------


def _train_x(ad, data, xdata, rdata, out, *, cfg=CFG_X, rc=None, **kw):
    rc = rc or T.resolve_w2c_run(cfg, tiny=True)
    w2_info = T.resolve_w2_config(cfg, tiny=True)[1]
    return T.run_w2c_training(ad, cfg, _spec(ad), rc, data, w2=W2, w2_info=w2_info, impl=IMPL, data_root=XF.ROOT, out_dir=out,
                              log=lambda m: None, replay=rdata, x2=xdata, **kw)


def test_x2_training_resume_segments_identity(data, xdata, replay_run, w2c_dir, x2_run, tmp_path):
    rdata = load_rdata(replay_run)
    a, b, c = tmp_path / "A", tmp_path / "B", tmp_path / "C"
    sa = _train_x(_tiny(), data, xdata, rdata, a)
    assert sa["complete"] and sa["steps"] == 4 and sa["x2"]["pools"] == {k: len(v) for k, v in xdata.pools.items()}
    with pytest.raises(RuntimeError, match="测试注入的中断"):
        _train_x(_tiny(), data, xdata, rdata, b, _fail_after_step=3)
    sb = _train_x(_tiny(), data, xdata, rdata, b, resume=True)
    assert _train_x(_tiny(), data, xdata, rdata, c, stop_after_steps=1) is None
    assert _train_x(_tiny(), data, xdata, rdata, c, stop_after_steps=2, resume=True) is None
    sc = _train_x(_tiny(), data, xdata, rdata, c, resume=True)
    la, lb, lc = (leaf_dir(x, "w2c", QWEN, 0) for x in (a, b, c))
    for other in (lb, lc):
        for f in DET:
            assert (la / f).read_bytes() == (other / f).read_bytes(), f
        ta, tb = _tensors(la / "ckpt" / "final" / "model"), _tensors(other / "ckpt" / "final" / "model")
        assert set(ta) == set(tb) and all(torch.equal(ta[k], tb[k]) for k in ta)
    assert sa == sb == sc
    log = jsonl(la / "train_log.jsonl")
    assert [r["step"] for r in log] == [0, 1, 2, 3] and all(math.isfinite(r["loss"]) for r in log)
    assert all(r["n_by_stream"] == {"propose": 2, "select": 2, "spline": 2, "correct": 2, "replay_general": 2, "replay_count": 1}
               for r in log)
    lb_ = lambda r, s: r["loss_by_stream"][s]  # noqa: E731
    assert all(r["loss"] == pytest.approx(sum(lb_(r, s) for s in ("propose", "select", "spline", "correct"))
                                          + 0.5 * lb_(r, "replay_general") + 0.5 * lb_(r, "replay_count")) for r in log)
    ident = yaml.safe_load((la / "config.yaml").read_text())
    assert ident["variant"] == "w2c" and ident["config"]["sha256"] == CFG_X.sha256 and ident["config"]["version"] == "x2-train-v1"
    x = ident["x2"]
    assert x["manifest_sha256"] == xdata.manifest_sha256 and x["params"] == xdata.manifest["params"]
    assert x["prompts"]["spline"] == XP.SPLINE_TEMPLATE and x["prompts"]["correct"] == XP.CORRECT_TEMPLATE
    assert x["streams"]["spline"]["index"] == 4 and ident["training"]["x2"]["per_step_total"] == 4
    assert str(x2_run) not in (la / "config.yaml").read_text()
    run1 = json.loads((la / "logs" / "run_001.json").read_text())
    assert {e["stream"] for e in run1["audit"]["first_step_examples"]} == {"propose", "select", "spline", "correct", "replay_general",
                                                                           "replay_count"}
    assert run1["paths"]["x2_data"] == str(x2_run)
    acc = json.loads((lc / "accounting.json").read_text())
    assert acc["by_stream"]["spline"]["n_samples"] == 8 and acc["by_stream"]["correct"]["n_samples"] == 8 and acc["n_runs"] == 3
    readme = (la / "README.md").read_text()
    assert "X-2（DEV-0069" in readme and "spline 2 条" in readme and "回放" in readme
    man = json.loads((la / "ckpt" / "final" / "model" / "manifest.json").read_text())
    assert man["format"] == "visint-e3-checkpoint/1" and man["trainable_token_rows"] is None  # 与 W2-c、C3 同格式
    meta = json.loads((la / "meta.json").read_text())
    assert meta["data_sha256"][str(x2_run / XD.DATA_DIR / XD.MANIFEST)] == xdata.manifest_sha256  # X-2 数据的文件进 meta.json
    # X-2 数据变了：运行身份不同，拒绝续跑
    p2 = json.loads(json.dumps(xdata.manifest["params"]))
    p2["correct"]["round2_ratio"] = 1.0
    alt = tmp_path / "alt"
    XD.build(w2c_dir, alt, data_root=XF.ROOT, params=p2, tiny=True, log=lambda m: None)
    d = copy.deepcopy(dict(CFG_X.doc))
    d["correct"]["round2_ratio"] = 1.0
    cfg2 = T.W2CConfig(path=CFG_X.path, sha256=CFG_X.sha256, doc=d)
    rc2 = T.resolve_w2c_run(cfg2, tiny=True)
    _train_x(_tiny(), data, xdata, rdata, tmp_path / "D", stop_after_steps=1)
    with pytest.raises(ValueError, match="运行身份"):
        _train_x(_tiny(), data, load_xdata(alt, w2c_dir, rc2), rdata, tmp_path / "D", cfg=cfg2, rc=rc2, resume=True)
    # 参数与 run_cfg 不同、数据与 x2 一节必须同时有
    with pytest.raises(ValueError, match="参数"):
        _train_x(_tiny(), data, load_xdata(alt, w2c_dir, rc2), rdata, tmp_path / "E")
    with pytest.raises(ValueError, match="同时有"):
        _train_x(_tiny(), data, None, rdata, tmp_path / "F")
    with pytest.raises(ValueError, match="同时有"):
        T.run_w2c_training(_tiny(), CFG_R, _spec(_tiny()), T.resolve_w2c_run(CFG_R, tiny=True), data, w2=W2,
                           w2_info=T.resolve_w2_config(CFG_R, tiny=True)[1], impl=IMPL, data_root=XF.ROOT, out_dir=tmp_path / "G",
                           replay=rdata, x2=xdata)


# ---------------------------------------------------------------------------
# 命令行
# ---------------------------------------------------------------------------


def test_cli_x2_data_usage_train_and_w2c_eval(w2c_dir, x2_run, replay_run, tmp_path):
    tmp = tmp_path
    base = ["--tiny", "--model", QWEN, "--w2c-data", w2c_dir]
    p = run(TRAIN, base + ["--x2-data", x2_run, "--out", tmp / "n1"], tmp, ok=False)
    assert p.returncode == 2 and "不收 --x2-data" in p.stderr
    p = run(TRAIN, base + ["--config", T.X2_CONFIG_PATH, "--replay-data", replay_run, "--out", tmp / "n2"], tmp, ok=False)
    assert p.returncode == 2 and "x2_data.path" in p.stderr and "占位" in p.stderr
    out = tmp / "train"
    p = run(TRAIN, base + ["--config", T.X2_CONFIG_PATH, "--x2-data", x2_run, "--replay-data", replay_run, "--out", out], tmp)
    assert "steps=4" in p.stdout and "'spline'" in p.stdout and "'correct'" in p.stdout
    ck = out / "w2c" / QWEN / "seed0" / "ckpt" / "final" / "model"
    man_sha = hashlib.sha256((ck / "manifest.json").read_bytes()).hexdigest()
    # run_w2 --setting w2c 能加载（W2 配置与训练时相同：仓库的 configs/w2.json；候选与渲染用测试的假实现）
    ev = tmp / "eval"
    p = run(RUN_W2, ["--tiny", "--model", QWEN, "--task", "refseg", "--split", "devcheck2", "--setting", "w2c", "--out", ev,
                     "--checkpoint", ck, "--candidates-module", "tests._w2_fakes", "--render-module", "tests._w2_fakes"], tmp, ok=False)
    assert p.returncode == 0, p.stderr[-2000:]
    c = yaml.safe_load((ev / QWEN / "refseg" / "w2c" / "devcheck2" / "config.yaml").read_text())
    assert c["checkpoint"]["manifest_sha256"] == man_sha and c["checkpoint"]["training"]["config_sha256"] == CFG_X.sha256
    assert json.loads((ev / QWEN / "refseg" / "w2c" / "devcheck2" / "metrics.json").read_text())["complete"] is True
