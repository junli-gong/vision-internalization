"""DEV-0081：X-M 的训练（src/w2/w2c_train.py 的 coco_det、coco_inst 两条流，scripts/train_w2c.py --xm-data，configs/xm_train.json）。
CPU fp32 小型随机 Qwen，测试造的数据（X-2 的假数据与回放同 tests/test_x2_train；COCO 的训练目标 tests/_xm_train_fakes（DEV-0079 的行格式）→ build_xm_data）。

- 配置：configs/xm_train.json 除 version、status 与新增的 xm_data、coco_det、coco_inst、xm_streams_basis 外与 configs/x2_train.json
  逐项相同；run_cfg 去掉 xm 一节后与 X-2 的逐项相同；每步 45 条；coco_inst 的微批 2（tiny 1）；
- **X-2、X-3 的配置逐字节不变**（7e1008d6 的快照：w2c_train、train_w2c 原文件与现在的各跑一遍，比 train_log、config.yaml、README、
  train_summary、检查点、记账、逐步计时的键）；configs/x2_train.json、x3_train.json 的 sha256 不变；
- X-M 里 X-2 的六条流：每一步取到的样本与同种子的 X-2 训练逐个相同，样本（图、提示、目标 token）也相同；
- COCO 两流的日程与样本；损失按流归一（含单独的微批）；
- tiny 上用 configs/xm_train.json 训 3 步不报错，逐步计时有按流的用时；分段续跑与一次跑完逐位相同；检查点能被 run_x2 的读法加载；
- 命令行：--xm-data 的用法。
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from src.e3.tokens import family_stop_id
from src.e3.train import leaf_dir
from src.explore import x2_runner as XRUN
from src.w2 import w2c_train as T
from src.xm import prompts as XMP
from src.xm import runner as XMR
from src.xm import train_data as XTD
from tests import _xm_train_fakes as XMF
from tests.test_x2_train import (  # noqa: F401  （fixture 与工具：X-2 的假数据、回放、tiny 模型）
    DET,
    IMPL,
    QWEN,
    TRAIN,
    W2,
    XF,
    _spec,
    _tensors,
    _tiny,
    data,
    jsonl,
    load_rdata,
    replay_run,
    run,
    w2c_dir,
    x2_run,
    xdata,
)

REPO = Path(__file__).resolve().parents[1]
CFG = T.load_w2c_config()
CFG_X = T.load_w2c_config(T.X2_CONFIG_PATH)
CFG_X3 = T.load_w2c_config(T.X3_CONFIG_PATH)
CFG_M = T.load_w2c_config(T.XM_CONFIG_PATH)
XM_ONLY = ("version", "status", "xm_data", "coco_det", "coco_inst", "xm_streams_basis")
SNAP_DIR = REPO / "tests" / "fixtures" / "xm"
SNAP = {"w2c_train": ("w2c_train_7e1008d6.py", "da240f65aaa95920e982aef507da2236c37e5c7057948dde1c217062f822d010"),
        "train_w2c": ("train_w2c_7e1008d6.py", "37167567f085f6cf27178ca6dcef408f5311785cdbc9d1e709e90ddc3a92f6f7")}
CONFIG_SHA = {"x2_train.json": "c80a4b0b39ac8275dafd6be7af83ec05dc0d5f4fbf21335d5e67e2ab829b5228",  # 7e1008d6（RUN-0080 训 X-2 的那一份）
              "x3_train.json": "bae8942ad281527a3907061d6577c1feef218df0c89a15c88b670d304a390f17"}
ALL8 = ("propose", "select", "spline", "correct", "coco_det", "coco_inst", "replay_general", "replay_count")


@pytest.fixture(scope="module")
def xm_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("xmd")
    tg = XMF.write_targets(tmp / "targets", XMF.target_rows())
    out = tmp / "xm"
    XTD.build(tg, out, params=XTD.params_from_config(CFG_M.doc), size=XTD.sizing_from_config(CFG_M.doc, tiny=True), tiny=True,
              data_root=XMF.ROOT, log=lambda m: None)
    return out


@pytest.fixture(scope="module")
def mdata(xm_run):
    return XTD.load_xm_data(xm_run, params=T.resolve_w2c_run(CFG_M, tiny=True)["xm"]["data_params"], data_root=XMF.ROOT)


def _load_path(path: Path, name: str):
    saved = list(sys.path)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    sys.path[:] = saved
    return mod


def _snapshot():
    for f, sha in SNAP.values():
        assert hashlib.sha256((SNAP_DIR / f).read_bytes()).hexdigest() == sha, f
    old_t = _load_path(SNAP_DIR / SNAP["w2c_train"][0], "_w2c_train_7e1008d6")
    old_t.REPO_ROOT, old_t.W2C_CONFIG_PATH, old_t.W2C_REPLAY_CONFIG_PATH = T.REPO_ROOT, T.W2C_CONFIG_PATH, T.W2C_REPLAY_CONFIG_PATH
    old_t.X2_CONFIG_PATH, old_t.X3_CONFIG_PATH = T.X2_CONFIG_PATH, T.X3_CONFIG_PATH
    old_s = _load_path(SNAP_DIR / SNAP["train_w2c"][0], "_train_w2c_7e1008d6")
    old_s.T = old_t
    return old_t, old_s


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------


def test_config_is_x2_config_plus_xm_sections():
    for name, sha in CONFIG_SHA.items():  # X-2、X-3 的配置文件逐字节不变
        assert hashlib.sha256((REPO / "configs" / name).read_bytes()).hexdigest() == sha, name
    a = {k: v for k, v in CFG_M.doc.items() if k not in XM_ONLY}
    b = {k: v for k, v in CFG_X.doc.items() if k not in ("version", "status")}
    assert a == b and list(a) == list(b)
    assert CFG_M.doc["version"] == "xm-train-v1" and CFG_M.doc["xm_data"]["path"] is None
    for tiny in (False, True):
        rm, rx = T.resolve_w2c_run(CFG_M, tiny=tiny), T.resolve_w2c_run(CFG_X, tiny=tiny)
        assert {k: v for k, v in rm.items() if k != "xm"} == rx  # X-2 的流、条数、权重、LoRA、优化器、步数逐项相同
        assert T.stream_names(rm) == ALL8 and T.stream_order(rm) == list(ALL8)
        assert {s: w for s, w in T.stream_weights(rm).items() if s not in T.XM_STREAMS} == T.stream_weights(rx)
        assert {s: m for s, m in T.stream_micro(rm).items() if s not in T.XM_STREAMS} == T.stream_micro(rx)
    rc = T.resolve_w2c_run(CFG_M, tiny=False)
    x = rc["xm"]
    assert x["order"] == ["coco_det", "coco_inst"] and x["per_step_total"] == 12
    assert {s: (v["per_step"], v["loss_weight"], v["index"], v["micro_batch_size"], v["micro_batches"], v["order"])
            for s, v in x["streams"].items()} == {"coco_det": (6, 1.0, 6, 4, 2, "image_first"), "coco_inst": (6, 1.0, 7, 2, 3, "image_first")}
    assert x["data_params"] == {"pair_salt": "xm_train_pairs_v1", "margin": 0.2, "inst_max_boxes": 12}
    assert rc["batch_size"] + rc["x2"]["per_step_total"] + x["per_step_total"] + rc["replay"]["per_step_total"] == 45
    assert T.stream_weights(rc) == {"propose": 1.0, "select": 1.0, "spline": 1.0, "correct": 1.0, "coco_det": 1.0, "coco_inst": 1.0,
                                    "replay_general": 0.5, "replay_count": 0.5}
    rt = T.resolve_w2c_run(CFG_M, tiny=True)["xm"]["streams"]
    assert {s: (v["per_step"], v["micro_batch_size"]) for s, v in rt.items()} == {"coco_det": (2, 2), "coco_inst": (2, 1)}
    for c in (CFG, CFG_X, CFG_X3):  # 没有 coco_det、coco_inst 两节：没有 xm 键
        assert "xm" not in T.resolve_w2c_run(c, tiny=False) and T.xm_spec(c.doc, tiny=False) is None
    assert min(T.XM_STREAM_INDEX.values()) > max(T.X2_STREAM_INDEX.values())


@pytest.mark.parametrize("edit,msg", [
    (lambda d: d.pop("coco_inst"), "一起有"),
    (lambda d: d.pop("xm_data"), "xm_data"),
    (lambda d: (d.pop("coco_det"), d.pop("coco_inst")), "要配 coco_det"),
    (lambda d: d["xm_data"].update(path=""), "xm_data"),
    (lambda d: d["xm_data"].update(extra=1), "未知的键"),
    (lambda d: d["xm_data"].update(margin=-1), "margin"),
    (lambda d: d["coco_inst"].update(micro_batch_size=0), "micro_batch_size"),
    (lambda d: d["coco_det"].update(per_step=0), "per_step"),
    (lambda d: d["coco_det"].update(loss_weight=0), "loss_weight"),
    (lambda d: d["coco_inst"].update(order="instruction_first"), "image_order"),
    (lambda d: d["coco_inst"].update(max_boxes=31), "inst_max_boxes"),
    (lambda d: d["coco_det"].update(max_boxes=12), "未知的键"),
])
def test_xm_config_validation(tmp_path, edit, msg):
    d = copy.deepcopy(dict(CFG_M.doc))
    edit(d)
    p = tmp_path / "c.json"
    p.write_text(json.dumps(d, ensure_ascii=False))
    with pytest.raises(T.W2CConfigError, match=msg):
        T.load_w2c_config(p)


# ---------------------------------------------------------------------------
# X-2、X-3 的配置逐字节不变（7e1008d6 的快照）
# ---------------------------------------------------------------------------


def _timing_keys(leaf: Path) -> list:
    return [sorted(r) for p in sorted((leaf / "logs").glob("timing_run_*.jsonl")) for r in jsonl(p)]


@pytest.mark.parametrize("cfg", [CFG_X, CFG_X3], ids=["x2", "x3"])
def test_x2_configs_bytewise_identical_to_snapshot(cfg, w2c_dir, x2_run, replay_run, tmp_path, capsys):
    old_t, old_s = _snapshot()
    new_s = _load_path(TRAIN, f"_train_w2c_dev0081_{Path(cfg.path).stem}")
    oc = old_t.load_w2c_config(REPO / cfg.path)
    assert (oc.sha256, oc.path, oc.doc) == (cfg.sha256, cfg.path, cfg.doc)
    for tiny in (False, True):
        assert old_t.resolve_w2c_run(oc, tiny=tiny) == T.resolve_w2c_run(cfg, tiny=tiny)
        assert old_t.resolve_w2c_run(oc, tiny=tiny, steps=7) == T.resolve_w2c_run(cfg, tiny=tiny, steps=7)
    extra = ["--config", str(REPO / cfg.path), "--x2-data", str(x2_run), "--replay-data", str(replay_run)]
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
    assert _timing_keys(a) == _timing_keys(b) and all("by_stream" not in k for k in _timing_keys(b))  # 逐步计时的键不变
    ma, mb = (json.loads((o / "meta.json").read_text()) for o in outs)
    assert ma["data_sha256"] == mb["data_sha256"]
    assert sorted(p.relative_to(a).as_posix() for p in a.rglob("*")) == sorted(p.relative_to(b).as_posix() for p in b.rglob("*"))
    assert "xm" not in yaml.safe_load((b / "config.yaml").read_text())


# ---------------------------------------------------------------------------
# 日程与样本
# ---------------------------------------------------------------------------


def _scheds(rc, data, xdata, rdata, mdata):
    total = rc["x2"]["per_step_total"] + rc["replay"]["per_step_total"] + (rc["xm"]["per_step_total"] if "xm" in rc else 0)
    s = T.W2CSchedule(data, seed=0, per_step=rc["streams"]["per_step"], order=rc["streams"]["order"], extra_per_step=total)
    xs = T.X2Schedule(xdata, seed=0, spec=rc["x2"], base_batch=s.batch_size, step_size=s.step_size)
    mx = rc["xm"]["per_step_total"] if "xm" in rc else 0
    ms = T.XMSchedule(mdata, seed=0, spec=rc["xm"], base_batch=s.batch_size + rc["x2"]["per_step_total"], step_size=s.step_size) \
        if "xm" in rc else None
    rs = T.ReplaySchedule(rdata, seed=0, spec=rc["replay"], base_batch=s.batch_size + rc["x2"]["per_step_total"] + mx)
    return s, xs, ms, rs


def _plans(scheds, step):
    s, xs, ms, rs = scheds
    return s.step_plans(step) + xs.step_plans(step) + ([] if ms is None else ms.step_plans(step)) + rs.step_plans(step)


@pytest.mark.parametrize("tiny", [False, True])
def test_x2_samples_unchanged_inside_xm(tiny, data, xdata, replay_run, mdata):
    rm, rx = T.resolve_w2c_run(CFG_M, tiny=tiny), T.resolve_w2c_run(CFG_X, tiny=tiny)
    rdata = load_rdata(replay_run, rx)
    sm, sx = _scheds(rm, data, xdata, rdata, mdata), _scheds(rx, data, xdata, rdata, None)
    per = {**rm["streams"]["per_step"], **{k: v["per_step"] for k, v in rm["x2"]["streams"].items()},
           **{k: v["per_step"] for k, v in rm["xm"]["streams"].items()}, **{k: v["per_step"] for k, v in rm["replay"]["streams"].items()}}
    n = sum(per.values())
    assert n == (45 if not tiny else 15)
    for step in (0, 1, 7, 33):
        pm, px = _plans(sm, step), _plans(sx, step)
        key = lambda p: (p.stream, p.step, p.j, p.index, p.row, p.epoch, p.sample_id)  # noqa: E731（k 是一步里的全局编号，会后移）
        assert [key(p) for p in pm if p.stream not in T.XM_STREAMS] == [key(p) for p in px]
        assert [p.k for p in pm] == list(range(step * n, step * n + n))  # 一步里的全局编号连续
        assert [p.stream for p in pm] == [s for s in ALL8 for _ in range(per[s])]


def test_x2_examples_unchanged_and_xm_examples(data, xdata, replay_run, mdata):
    ad = _tiny()
    enc = lambda s: list(ad.tokenizer(s, add_special_tokens=False)["input_ids"])  # noqa: E731
    stop = family_stop_id(ad)
    rm, rx = T.resolve_w2c_run(CFG_M, tiny=True), T.resolve_w2c_run(CFG_X, tiny=True)
    rdata = load_rdata(replay_run, rx)
    b = T.W2CBuilder(CFG_M, data, w2=W2, impl=IMPL, data_root=XF.ROOT, stop_id=stop, encode=enc, family="qwen3.5", format_boxes=ad.format_boxes)
    bx = T.W2CBuilder(CFG_X, data, w2=W2, impl=IMPL, data_root=XF.ROOT, stop_id=stop, encode=enc, family="qwen3.5", format_boxes=ad.format_boxes)
    xb = T.X2Builder(xdata, w2=W2, data_root=XF.ROOT, stop_id=stop, encode=enc, spec=rm["x2"])
    rb = T.ReplayBuilder(rdata, data_root=XF.ROOT, stop_ids=ad.stop_token_ids())
    mb = T.XMBuilder(mdata, data_root=XMF.ROOT, stop_id=stop, encode=enc, spec=rm["xm"])
    sm, sx = _scheds(rm, data, xdata, rdata, mdata), _scheds(rx, data, xdata, rdata, None)

    def build(builder, p):
        if p.stream in T.X2_STREAMS:
            return xb.build(p)
        if p.stream in T.XM_STREAMS:
            return mb.build(p)
        return rb.build(p) if p.stream.startswith(T.REPLAY_PREFIX) else builder.build(p)

    seen_chunks = set()
    for step in range(4):
        em = [build(b, p) for p in _plans(sm, step)]
        ex = [build(bx, p) for p in _plans(sx, step)]
        ems = [e for e in em if e.plan.stream not in T.XM_STREAMS]
        assert len(ems) == len(ex)
        for e1, e2 in zip(ems, ex):  # X-2 的样本：图、提示、顺序、目标 token 逐个相同
            assert (e1.plan.sample_id, e1.instruction, e1.order, e1.target_ids, e1.target_text) == \
                (e2.plan.sample_id, e2.instruction, e2.order, e2.target_ids, e2.target_text)
            assert len(e1.images) == len(e2.images) and all(
                np.array_equal(np.asarray(i1), np.asarray(i2)) for i1, i2 in zip(e1.images, e2.images))
        for e in em:
            if e.plan.stream not in T.XM_STREAMS:
                continue
            row = mdata.rows[e.plan.row]
            with Image.open(XMF.ROOT / row["image"]) as im:
                orig = im.convert("RGB")
            assert len(e.images) == 1 and np.array_equal(np.asarray(e.images[0]), np.asarray(orig)) and e.order == "image_first"
            assert e.target_ids == enc(row["target_text"]) + [stop] and e.n_supervised == len(e.target_ids)
            if e.plan.stream == "coco_det":
                assert e.instruction == XMP.call1_prompt(row["category"]) and json.loads(e.target_text) == row["boxes"]
            else:
                assert e.instruction == XMP.call2_prompt(row["category"], row["boxes"]) and len(row["boxes"]) <= 12
                assert json.loads(e.target_text) == row["splines"]
                seen_chunks.add(row["n_chunks"])
    # 一个 epoch 恰好覆盖本流的池一次
    _, _, ms, _ = sm
    for st in T.XM_STREAMS:
        rows = [ms.plan(st, j).row for j in range(len(mdata.pool(st)))]
        assert sorted(rows) == sorted(mdata.pool(st)) and all(mdata.rows[r]["kind"] == st for r in rows)
    bad = T.W2CPlan(stream="coco_det", k=0, step=0, j=0, index=0, row=mdata.pool("coco_inst")[0], epoch=0,
                    sample_id=str(mdata.rows[mdata.pool("coco_inst")[0]]["id"]))
    with pytest.raises(RuntimeError, match="对不上"):
        mb.build(bad)


# ---------------------------------------------------------------------------
# 损失
# ---------------------------------------------------------------------------


def test_eight_streams_loss_and_per_stream_micro(data, xdata, replay_run, mdata, monkeypatch):
    d = copy.deepcopy(dict(CFG_M.doc))
    d["coco_inst"]["tiny_per_step"] = 3  # 微批 1：三个；列框 2 条一个微批
    cfg = T.W2CConfig(path="x", sha256="0" * 64, doc=d)
    rc = T.resolve_w2c_run(cfg, tiny=True)
    rdata = load_rdata(replay_run, rc)
    ad = _tiny()
    tr = T.W2CTrainer(ad, cfg, rc, seed=0, number_ids=None)
    tr.setup_fresh()
    enc = lambda s: list(ad.tokenizer(s, add_special_tokens=False)["input_ids"])  # noqa: E731
    stop = family_stop_id(ad)
    b = T.W2CBuilder(cfg, data, w2=W2, impl=IMPL, data_root=XF.ROOT, stop_id=stop, encode=enc, family="qwen3.5", format_boxes=ad.format_boxes)
    xb = T.X2Builder(xdata, w2=W2, data_root=XF.ROOT, stop_id=stop, encode=enc, spec=rc["x2"])
    rb = T.ReplayBuilder(rdata, data_root=XF.ROOT, stop_ids=ad.stop_token_ids())
    mb = T.XMBuilder(mdata, data_root=XMF.ROOT, stop_id=stop, encode=enc, spec=rc["xm"])
    plans = _plans(_scheds(rc, data, xdata, rdata, mdata), 1)
    ex = [xb.build(p) if p.stream in T.X2_STREAMS else mb.build(p) if p.stream in T.XM_STREAMS else
          (rb.build(p) if p.stream.startswith(T.REPLAY_PREFIX) else b.build(p)) for p in plans]
    sums, scaled, sizes = [], [], []
    orig_loss, orig_bwd, orig_fwd = ad.chunked_lm_loss, torch.Tensor.backward, ad.training_forward

    def fwd(inputs, targets, **k):
        sizes.append(len(inputs))
        return orig_fwd(inputs, targets, **k)

    def loss(*a, **k):
        out = orig_loss(*a, **k)
        sums.append(float(out.detach()))
        return out

    def bwd(self, *a, **k):
        scaled.append(float(self.detach()))
        return orig_bwd(self, *a, **k)

    monkeypatch.setattr(ad, "training_forward", fwd)
    monkeypatch.setattr(ad, "chunked_lm_loss", loss)
    monkeypatch.setattr(torch.Tensor, "backward", bwd)
    rec, times, _ = tr.optimize(ex)
    monkeypatch.undo()
    assert rec["n_by_stream"] == {"propose": 2, "select": 2, "spline": 2, "correct": 2, "coco_det": 2, "coco_inst": 3,
                                  "replay_general": 2, "replay_count": 1}
    chunks = ["propose", "select", "spline", "correct", "coco_det", "coco_inst", "coco_inst", "coco_inst", "replay_general", "replay_count"]
    assert sizes == [2, 2, 2, 2, 2, 1, 1, 1, 2, 1]  # coco_inst 单独的微批 1，其余照 tiny 的 2
    n, w = rec["n_supervised_by_stream"], T.stream_weights(rc)
    for st, v, sc in zip(chunks, sums, scaled):
        assert sc == pytest.approx(w[st] * v / n[st])
    assert rec["loss_by_stream"]["coco_inst"] == pytest.approx(sum(sums[5:8]) / n["coco_inst"])
    assert rec["loss"] == pytest.approx(sum(w[s] * rec["loss_by_stream"][s] for s in ALL8))
    bs = times["by_stream"]  # 按流的用时（CPU 上没有显存峰值）
    assert list(bs) == list(ALL8) and {s: v["n_micro"] for s, v in bs.items()} == {
        "propose": 1, "select": 1, "spline": 1, "correct": 1, "coco_det": 1, "coco_inst": 3, "replay_general": 1, "replay_count": 1}
    assert all(v["s"] > 0 and v["peak_mem_bytes"] is None for v in bs.values())
    assert sum(v["s"] for v in bs.values()) == pytest.approx(times["build_inputs_s"] + times["fwd_s"] + times["bwd_s"], rel=1e-6)


# ---------------------------------------------------------------------------
# tiny 训练
# ---------------------------------------------------------------------------


def _train_m(ad, data, xdata, rdata, mdata, out, **kw):
    rc = T.resolve_w2c_run(CFG_M, tiny=True, steps=3)
    return T.run_w2c_training(ad, CFG_M, _spec(ad), rc, data, w2=W2, w2_info=T.resolve_w2_config(CFG_M, tiny=True)[1], impl=IMPL,
                              data_root=XF.ROOT, out_dir=out, log=lambda m: None, replay=rdata, x2=xdata, xm=mdata, **kw)


def test_xm_training_resume_identity_and_x2_loader(data, xdata, replay_run, mdata, xm_run, tmp_path):
    rdata = load_rdata(replay_run)
    a, c = tmp_path / "A", tmp_path / "C"
    sa = _train_m(_tiny(), data, xdata, rdata, mdata, a)
    assert sa["complete"] and sa["steps"] == 3 and sa["xm"]["pools"] == {k: len(v) for k, v in mdata.pools.items()}
    assert _train_m(_tiny(), data, xdata, rdata, mdata, c, stop_after_steps=1) is None
    sc = _train_m(_tiny(), data, xdata, rdata, mdata, c, resume=True)
    la, lc = leaf_dir(a, "w2c", QWEN, 0), leaf_dir(c, "w2c", QWEN, 0)
    for f in DET:
        assert (la / f).read_bytes() == (lc / f).read_bytes(), f
    ta, tc = _tensors(la / "ckpt/final/model"), _tensors(lc / "ckpt/final/model")
    assert set(ta) == set(tc) and all(torch.equal(ta[k], tc[k]) for k in ta) and sa == sc
    log = jsonl(la / "train_log.jsonl")
    assert [r["step"] for r in log] == [0, 1, 2]
    assert all(r["n_by_stream"] == {"propose": 2, "select": 2, "spline": 2, "correct": 2, "coco_det": 2, "coco_inst": 2,
                                    "replay_general": 2, "replay_count": 1} for r in log)
    timing = jsonl(la / "logs" / "timing_run_001.jsonl")
    assert [r["step"] for r in timing] == [0, 1, 2] and all(list(r["by_stream"]) == list(ALL8) for r in timing)
    assert all(r["by_stream"]["coco_inst"]["n_micro"] == 2 and r["by_stream"]["coco_det"]["s"] > 0 for r in timing)
    ts = T.timing_summary(la, from_step=1)
    assert ts["n_steps"] == 2 and ts["step_s_median"] == pytest.approx(float(np.median([r["step_s"] for r in timing[1:]])))
    assert set(ts["by_stream"]) == set(ALL8) and ts["peak_mem_bytes_max"] is None
    with pytest.raises(ValueError, match="没有第 5 步以后"):
        T.timing_summary(la)
    ident = yaml.safe_load((la / "config.yaml").read_text())
    assert ident["config"]["version"] == "xm-train-v1" and ident["variant"] == "w2c" and "x2" in ident and "replay" in ident
    x = ident["xm"]
    assert x["manifest_sha256"] == mdata.manifest_sha256 and x["params"] == mdata.params
    assert x["prompts"] == T.xm_prompts_identity() and x["prompts"]["call2_example"] == XMP.call2_prompt(*T.XM_PROMPT_EXAMPLE)
    assert (x["prompts"]["version"], x["prompts"]["frozen"], x["n"]) == ("xm-prompts-v1", True, 16)  # 冻结的提示（D168）
    assert x["target_format"]["prompt_version"] == "xm-prompts-v1" and x["target_format"]["prompts_frozen"] is True
    # DEV-0080 的运行器认这个检查点（训练叶目录 config.yaml 的 xm 一节：prompts.version、n）
    xmp = XMR.checkpoint_xm_params(la / "ckpt" / "final" / "model")
    assert (xmp["prompts_version"], xmp["n"]) == ("xm-prompts-v1", 16)
    assert {s: v["index"] for s, v in x["streams"].items()} == {"coco_det": 6, "coco_inst": 7} and str(xm_run) not in (la / "config.yaml").read_text()
    run1 = json.loads((la / "logs" / "run_001.json").read_text())
    assert {e["stream"] for e in run1["audit"]["first_step_examples"]} == set(ALL8) and run1["paths"]["xm_data"] == str(xm_run)
    readme = (la / "README.md").read_text()
    assert "X-M（DEV-0081" in readme and "coco_inst 2 条（微批 1" in readme
    meta = json.loads((la / "meta.json").read_text())
    assert meta["data_sha256"][str(xm_run / XTD.DATA_DIR / XTD.MANIFEST)] == mdata.manifest_sha256
    # 检查点：run_x2 的读法（x2 一节）照常加载，做指代分割用
    xp = XRUN.checkpoint_x2_params(la / "ckpt" / "final" / "model")
    assert xp["n"] == 16 and xp["m"] == 2 and xp["manifest_sha256"] == xdata.manifest_sha256
    # 数据与 xm 一节必须同时有；参数不同拒绝；非 tiny 时池不够一个 epoch 拒绝
    with pytest.raises(ValueError, match="同时有"):
        _train_m(_tiny(), data, xdata, rdata, None, tmp_path / "D")
    with pytest.raises(ValueError, match="同时有"):
        T.run_w2c_training(_tiny(), CFG_X, _spec(_tiny()), T.resolve_w2c_run(CFG_X, tiny=True), data, w2=W2,
                           w2_info=T.resolve_w2_config(CFG_X, tiny=True)[1], impl=IMPL, data_root=XF.ROOT, out_dir=tmp_path / "E",
                           replay=rdata, x2=xdata, xm=mdata)
    bad = XTD.XMData(run_dir=mdata.run_dir, rows=mdata.rows, pools=mdata.pools, manifest_sha256=mdata.manifest_sha256,
                     manifest={**mdata.manifest, "params": {**mdata.params, "pair_salt": "other"}})
    with pytest.raises(ValueError, match="参数"):
        _train_m(_tiny(), data, xdata, rdata, bad, tmp_path / "F")


def test_cli_xm_data_usage_and_three_steps(w2c_dir, x2_run, replay_run, xm_run, tmp_path):
    base = ["--tiny", "--model", QWEN, "--w2c-data", w2c_dir, "--x2-data", x2_run, "--replay-data", replay_run]
    p = run(TRAIN, base + ["--config", T.X2_CONFIG_PATH, "--xm-data", xm_run, "--out", tmp_path / "n1"], tmp_path, ok=False)
    assert p.returncode == 2 and "不收 --xm-data" in p.stderr
    p = run(TRAIN, base + ["--config", T.XM_CONFIG_PATH, "--out", tmp_path / "n2"], tmp_path, ok=False)
    assert p.returncode == 2 and "xm_data.path" in p.stderr and "占位" in p.stderr
    out = tmp_path / "train"
    p = run(TRAIN, base + ["--config", T.XM_CONFIG_PATH, "--xm-data", xm_run, "--xm-data-root", XMF.ROOT, "--steps", "3", "--out", out],
            tmp_path)
    assert "steps=3" in p.stdout and "'coco_det'" in p.stdout and "'coco_inst'" in p.stdout
    leaf = out / "w2c" / QWEN / "seed0"
    assert len(jsonl(leaf / "train_log.jsonl")) == 3
    timing = jsonl(leaf / "logs" / "timing_run_001.jsonl")
    assert all(set(r["by_stream"]) == set(ALL8) and r["step_s"] > 0 for r in timing)
    assert json.loads((leaf / "logs" / "run_001.json").read_text())["paths"]["xm_data_root"] == str(XMF.ROOT)
