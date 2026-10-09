"""DEV-0044：E3a+C3 混合训练（scripts/train_e3a.py --variant full --with-text，变体 e3a_c3；配置的 mixed 节）。

- 配置：mixed 节的数字（每步 8 + 8、8000 步、1 : 1、文本在前）与来历（8000 × 8 = 4000 × 16，两条流各自核对）钉住；load_config 拒绝
  不是 1 : 1 的权重、对不上的步数与条数、文本流不是 C3、两份配置不相容；resolve_run 的 e3a_c3 与 full 只差 variant、steps、warmup_steps
  与 mixed 一节；稠密流每步的微批组成；
- 配置只新增 version_note 与 mixed 节、升版本号（与 v1 原样 28be66e5… 比较）；已有的 full、c1、c2 的 resolve_run 逐项不变；
- 新旧代码对比：改动前（36bcbbf）的 src/readout/{config,data,train}.py 与 scripts/train_e3a.py 原样（tests/fixtures/e3a/readout_36bcbbf/）
  与现在的代码在同一进程里各跑一遍 --tiny：full、c1、c2 的训练日志、最终检查点（model/ 与 readout/ 的每个文件）逐字节相同；同一份旧配置
  下运行身份（config.yaml）、train_summary.json、README.md 也逐字节相同；新配置下运行身份只差 config 一项；
- 逐样本相同：第 t 步的文本 8 条 = C3 同一种子第 8t … 8t + 7 个槽位，稠密 8 条 = E3a full 同一种子第 8t … 8t + 7 个槽位（三个种子 ×
  前 64 步）；文本的微批与 C3 的微批逐个相同；三家族前 4 步的样本连同 build_inputs 的输入张量逐个相同（文本对 C3，稠密对 E3a full）；
- 损失：一步 = 文本的逐 token 平均 + 稠密的逐样本平均（1 : 1），用独立算出的两项核对数值与梯度；每个微批只含一种监督、文本在前；
  文本样本不插查询 token、不经过读出（入口断言有反例）；
- 结构差别：与 C3 相比只多查询占位符那一行（三家族），且不训练；
- 整个运行：两种监督各自的合计与 C3、E3a full 单独训练（同一种子）的合计逐个相同；文本流的运行身份与 C3 的对应项相同；续跑与一次跑完
  逐字节相同；
- 命令行：三家族 --tiny 跑通，检查点与 E3a full 同一种，eval_e3a --variant full 与 run_e1 --checkpoint 能读。
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import yaml

from src.adapters import registry
from src.adapters.checkpoint import policy_spec
from src.e3 import config as E3C
from src.e3 import data as E3D
from src.e3 import train as E3T
from src.readout import data as D
from src.readout import losses as LS
from src.readout.config import (
    E3A_C3,
    MIXED_STREAMS,
    ConfigError,
    load_config,
    micro_batch_groups,
    resolve_run,
    text_stream_config,
)
from src.readout.train import RunSpec, Trainer, leaf_dir, run_training, stack_targets
from tests._e3a_overfit import tiny_adapter

REPO = Path(__file__).resolve().parents[1]
FAMILIES = ("qwen3.5-0.8b", "gemma-4-e2b-it", "internvl3.5-2b-hf")
CFG = load_config()
TEXT_CFG = text_stream_config(CFG.doc)
OLD_CONFIG = REPO / "tests" / "fixtures" / "e3a" / "e3a_train_28be66e5.json"  # DEV-0044 之前的 E3a 配置原样（v1，D71）
OLD_SHA = "28be66e52fd3e9e5b45c5b04e299342bda469c9c2f9d20e3b2de022b0c790067"
SNAP = REPO / "tests" / "fixtures" / "e3a" / "readout_36bcbbf"  # 36bcbbf（DEV-0044 之前）的 src/readout 三个文件与训练脚本原样
SNAP_MODULES = ("config", "data", "train")  # 依赖顺序；src/readout 的其余模块没有改动


@pytest.fixture(scope="module")
def pools():
    paths = D.E3aPaths.fixtures()
    return {t: D.load_pool(t, paths, CFG).records for t in CFG.tasks}


def _rc(tiny=True, **kw):
    return resolve_run(CFG, variant=E3A_C3, tiny=tiny, **kw)


def _mixed_trainer(key, *, rc=None, seed=0):
    ad = tiny_adapter(key)
    rc = rc or _rc()
    tr = Trainer(ad, CFG, rc, variant=E3A_C3, seed=seed, tiny=True, text_cfg=TEXT_CFG)
    tr.setup_fresh()
    return tr


def _builders(tr, pools, key, seed=0):
    dense = D.ExampleBuilder(CFG, pools, out_size=tr.readout.spec.out_size)
    text = D.c3_text_builder(TEXT_CFG, pools, seed=seed, adapter=tr.ad, family=registry.get_model(key).family)
    return D.MixedBuilder(dense, text)


def _c3_builder(ad, pools, key, seed=0):
    """C3 自己的构造（与 src/e3/train.run_training 的参数相同，同 tests/test_e3_mixed.py 的 _builder）。"""
    from src.e3.tokens import family_stop_id

    return E3D.ExampleBuilder(TEXT_CFG, {t: pools[t] for t in TEXT_CFG.tasks}, seed=seed, supervision="text",
                              stop_id=family_stop_id(ad), encode=lambda s: list(ad.tokenizer(s, add_special_tokens=False)["input_ids"]),
                              family=registry.get_model(key).family, format_boxes=ad.format_boxes)


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------


def test_config_mixed_pins():
    m = CFG.doc["mixed"]
    assert m["per_step"] == {"text": 8, "dense": 8} and m["steps"] == 8000 and m["loss_weights"] == {"text": 1, "dense": 1}
    assert m["order"] == ["text", "dense"] and m["tiny"] == {"per_step": {"text": 2, "dense": 2}, "steps": 8}
    assert m["text_stream"] == {"config": "configs/e3b_train.json", "supervision": "text", "variant": "c3"}
    batch = CFG.doc["batch"]["micro_batch_size"] * CFG.doc["batch"]["grad_accum_steps"]
    c3 = E3C.resolve_run(TEXT_CFG, supervision="text", tiny=False)
    assert m["steps"] * m["per_step"]["dense"] == CFG.doc["steps"] * batch == 64000
    assert m["steps"] * m["per_step"]["text"] == c3["steps"] * c3["batch_size"] == 64000
    assert m["steps"] == 2 * CFG.doc["steps"] == 2 * c3["steps"] and sum(m["per_step"].values()) == batch
    full = resolve_run(CFG, variant="full", tiny=False)
    rc = _rc(tiny=False)
    assert rc["variant"] == "e3a_c3" and rc["steps"] == 8000 and rc["warmup_steps"] == 240
    assert rc["batch_size"] == 16 and rc["micro_batch_size"] == 4 and rc["grad_accum_steps"] == 4
    same = {k for k in full if k not in ("variant", "steps", "warmup_steps")}
    assert {k: rc[k] for k in same} == {k: full[k] for k in same}  # LoRA、查询、优化器、学习率、损失权重、混合同 E3a full
    assert set(rc) - set(full) == {"mixed"} and list(rc)[:len(full) - 2] == list(full)[:len(full) - 2]
    mx = rc["mixed"]
    assert mx["per_step"] == {"text": 8, "dense": 8} and mx["loss_weights"] == {"text": 1.0, "dense": 1.0}
    assert mx["schedule_length_ratio"] == {"text": 2.0, "dense": 2.0}
    assert mx["micro_batches"]["text"] == 2
    assert mx["micro_batches"]["dense"]["per_step"] == [["semseg", "semseg"], ["refseg", "depth", "normals", "refseg"],
                                                        ["depth", "normals"]]  # 配置 basis 里写的组成
    assert mx["reference"]["text"] == {"variant": "c3", "config": {"path": "configs/e3b_train.json", "sha256": TEXT_CFG.sha256,
                                                                   "version": TEXT_CFG.doc["version"]},
                                       "steps": 4000, "batch_size": 16, "micro_batch_size": 4, "samples": 64000}
    assert mx["reference"]["dense"] == {"variant": "full", "steps": 4000, "batch_size": 16, "micro_batch_size": 4, "samples": 64000}
    t = _rc(tiny=True)
    assert (t["steps"], t["batch_size"], t["warmup_steps"], t["mixed"]["per_step"]) == (8, 4, 1, {"text": 2, "dense": 2})
    assert t["mixed"]["schedule_length_ratio"] == {"text": 2.0, "dense": 2.0} and t["mixed"]["micro_batches"]["dense"]["per_step"] is None


def _mut(path_keys, value):
    def f(doc):
        d = doc
        for k in path_keys[:-1]:
            d = d[k]
        d[path_keys[-1]] = value
    return f


@pytest.mark.parametrize("mutate,match", [
    (_mut(["mixed", "loss_weights"], {"text": 1, "dense": 2}), "1 : 1"),
    (_mut(["mixed", "loss_weights"], {"text": 0.5, "dense": 0.5}), "1 : 1"),
    (_mut(["mixed", "steps"], 4000), "逐样本相同"),
    (_mut(["mixed", "per_step"], {"text": 6, "dense": 10}), "微批"),
    (_mut(["mixed", "per_step"], {"text": 8, "dense": 4}), "每步条数"),
    (_mut(["mixed", "per_step"], {"text": 16}), "per_step"),
    (_mut(["mixed", "per_step"], {"text": 8, "grid": 8}), "per_step"),
    (_mut(["mixed", "order"], ["text", "text"]), "order"),
    (_mut(["mixed", "tiny", "steps"], 4), "mixed.tiny"),
    (_mut(["mixed", "text_stream", "variant"], "e3b"), "C3"),
    (_mut(["mixed", "text_stream", "supervision"], "grid"), "C3"),
    (_mut(["mixed", "text_stream", "config"], "configs/no_such.json"), "读不了"),
    (_mut(["tasks", "refseg", "datasets"], ["refcoco"]), "同一个训练池"),
    (_mut(["seeds"], [0, 1, 2, 3]), "seeds"),
    (_mut(["enable_thinking"], True), "enable_thinking"),
    (_mut(["dedupe", "required_eval_sets", "depth"], ["ade20k/val"]), "去重名单"),
    (_mut(["mix", "weights"], {"refseg": 1, "depth": 0, "semseg": 1, "normals": 1}), "mix.weights"),
])
def test_config_rejects_bad_mixed(tmp_path, mutate, match):
    doc = copy.deepcopy(dict(CFG.doc))
    mutate(doc)
    p = tmp_path / "c.json"
    p.write_text(json.dumps(doc, ensure_ascii=False))
    with pytest.raises(ConfigError, match=match):
        load_config(p)


def test_config_only_adds_mixed():
    """DEV-0044 只新增 version_note 与 mixed 节（并升版本号），其余条目与 v1（28be66e5…，D71）逐项相同。"""
    assert hashlib.sha256(OLD_CONFIG.read_bytes()).hexdigest() == OLD_SHA
    old = json.loads(OLD_CONFIG.read_text(encoding="utf-8"))
    assert set(CFG.doc) - set(old) == {"version_note", "mixed"} and not set(old) - set(CFG.doc)
    assert {k for k in old if CFG.doc[k] != old[k]} == {"version"}
    assert (old["version"], CFG.doc["version"]) == ("e3a-train-v1", "e3a-train-v2")
    keys = list(CFG.doc)
    assert keys[keys.index("version") + 1] == "version_note" and keys[-2:] == ["mixed", "tiny"]
    assert [k for k in keys if k in old] == list(old)  # 原有条目的顺序不变
    assert CFG.sha256 != OLD_SHA


@pytest.mark.parametrize("tiny", [False, True])
@pytest.mark.parametrize("variant", ["full", "c1", "c2"])
def test_existing_resolve_run_unchanged(variant, tiny):
    old = load_config(OLD_CONFIG)
    assert resolve_run(CFG, variant=variant, tiny=tiny) == resolve_run(old, variant=variant, tiny=tiny)
    with pytest.raises(ConfigError, match="mixed"):
        resolve_run(old, variant=E3A_C3, tiny=tiny)


# ---------------------------------------------------------------------------
# 新旧代码对比（改动前的模块原样，同一进程）
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def old_code():
    """把 36bcbbf 的 src/readout/{config,data,train}.py 临时装成 src.readout.<同名>（__file__ 指向原路径，REPO_ROOT 等不变），载入当时的
    scripts/train_e3a.py；退出时恢复现在的模块。src/readout 的其余模块没有改动，两边共用。"""
    import src.readout as pkg

    saved = {n: (sys.modules.get(f"src.readout.{n}"), getattr(pkg, n, None)) for n in SNAP_MODULES}
    try:
        for n in SNAP_MODULES:
            name = f"src.readout.{n}"
            m = types.ModuleType(name)
            m.__file__ = str(REPO / "src" / "readout" / f"{n}.py")
            m.__package__ = "src.readout"
            sys.modules[name] = m
            setattr(pkg, n, m)
            exec(compile((SNAP / f"{n}.py").read_text(encoding="utf-8"), m.__file__, "exec"), m.__dict__)
        script = types.ModuleType("_train_e3a_36bcbbf")
        script.__file__ = str(REPO / "scripts" / "train_e3a.py")
        path = list(sys.path)
        exec(compile((SNAP / "train_e3a.py").read_text(encoding="utf-8"), script.__file__, "exec"), script.__dict__)
        sys.path[:] = path
        yield script
    finally:
        for n, (mod, attr) in saved.items():
            sys.modules[f"src.readout.{n}"] = mod
            setattr(pkg, n, attr)


def _new_script():
    saved = list(sys.path)
    spec = importlib.util.spec_from_file_location("_train_e3a_under_test", REPO / "scripts" / "train_e3a.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.path[:] = saved
    return mod


def _files(d: Path) -> dict[str, bytes]:
    return {str(p.relative_to(d)): p.read_bytes() for p in sorted(d.rglob("*")) if p.is_file()}


DETERMINISTIC = ("train_log.jsonl", "config.yaml", "train_summary.json", "README.md")


@pytest.mark.parametrize("key,variant", [("qwen3.5-0.8b", "full"), ("gemma-4-e2b-it", "c1"), ("internvl3.5-2b-hf", "c2")])
def test_old_and_new_code_identical(key, variant, tmp_path):
    """改动前的代码 + 旧配置、现在的代码 + 旧配置、现在的代码 + 新配置，各跑一遍 --tiny（同一个进程、同一组参数）。"""
    snap_text = {n: (SNAP / f"{n}.py").read_text(encoding="utf-8") for n in SNAP_MODULES}
    assert "E3A_C3" not in snap_text["config"] and "MixedSchedule" not in snap_text["data"]  # 确实是改动前的版本
    args = ["--tiny", "--model", key, "--variant", variant, "--seed", "0"]
    with old_code() as old:
        assert sys.modules["src.readout.config"].__dict__.get("E3A_C3") is None
        assert old.main(args + ["--out", str(tmp_path / "oo"), "--config", str(OLD_CONFIG)]) == 0
    import src.readout.config as cur

    assert cur.E3A_C3 == "e3a_c3" and sys.modules["src.readout.train"].__dict__.get("text_stream_identity") is not None
    new = _new_script()
    assert new.main(args + ["--out", str(tmp_path / "no"), "--config", str(OLD_CONFIG)]) == 0
    assert new.main(args + ["--out", str(tmp_path / "nn")]) == 0
    leaves = {k: leaf_dir(tmp_path / k, variant, key, 0) for k in ("oo", "no", "nn")}
    ck = {k: _files(v / "ckpt" / "final") for k, v in leaves.items()}
    assert ck["oo"] and ck["oo"] == ck["no"] == ck["nn"]  # model/ 与 readout/ 的每个文件逐字节相同
    for f in DETERMINISTIC:
        assert (leaves["oo"] / f).read_bytes() == (leaves["no"] / f).read_bytes(), f
    assert (leaves["oo"] / "train_log.jsonl").read_bytes() == (leaves["nn"] / "train_log.jsonl").read_bytes()
    assert (leaves["oo"] / "README.md").read_bytes() == (leaves["nn"] / "README.md").read_bytes()
    a = yaml.safe_load((leaves["oo"] / "config.yaml").read_text(encoding="utf-8"))
    b = yaml.safe_load((leaves["nn"] / "config.yaml").read_text(encoding="utf-8"))
    assert list(a) == list(b) and {k for k in a if a[k] != b[k]} == {"config"}
    assert a["config"] == {"path": str(OLD_CONFIG.relative_to(REPO)), "sha256": OLD_SHA, "version": "e3a-train-v1"}
    assert b["config"] == {"path": "configs/e3a_train.json", "sha256": CFG.sha256, "version": "e3a-train-v2"}
    sa = json.loads((leaves["oo"] / "train_summary.json").read_text())
    sb = json.loads((leaves["nn"] / "train_summary.json").read_text())
    assert {k for k in sa if sa[k] != sb[k]} == {"identity_sha256"}
    acc = {k: json.loads((v / "accounting.json").read_text()) for k, v in leaves.items()}
    keys = ("n_steps_logged", "n_samples", "n_prompt_tokens", "n_visual_tokens", "n_image_pixels", "n_query_tokens", "n_class_rows",
            "note")
    assert {k: acc["oo"][k] for k in keys} == {k: acc["no"][k] for k in keys} == {k: acc["nn"][k] for k in keys}
    assert set(acc["oo"]) == set(acc["nn"])


# ---------------------------------------------------------------------------
# 逐样本相同
# ---------------------------------------------------------------------------

E3_PLAN_KEYS = ("k", "task", "j", "index", "epoch", "sample_id", "kind", "text_kind")
RO_PLAN_KEYS = ("k", "task", "j", "index", "epoch", "sample_id")


def _same(a, b, keys) -> bool:
    return type(a) is type(b) and all(getattr(a, f) == getattr(b, f) for f in keys)


def test_streams_are_c3_and_e3a_slot_by_slot(pools):
    rc = _rc(tiny=False)
    per = rc["mixed"]["per_step"]
    c3_pools = {t: E3D.load_pool(t, E3D.E3Paths.fixtures(), TEXT_CFG.task(t)).records for t in TEXT_CFG.tasks}
    assert {t: [s.id for s in v] for t, v in c3_pools.items()} == {t: [s.id for s in pools[t]] for t in c3_pools}  # 同一个训练池
    c3_micro = E3C.resolve_run(TEXT_CFG, supervision="text", tiny=False)["micro_batch_size"]
    for seed in CFG.seeds:
        ms = D.MixedSchedule(pools, CFG, TEXT_CFG, seed=seed, per_step=per, order=rc["mixed"]["order"])
        c3 = E3D.Schedule(c3_pools, TEXT_CFG, seed=seed, supervision="text", batch_size=16)
        full = D.Schedule(pools, CFG, seed=seed, batch_size=16)
        for t in range(64):
            plans = ms.step_plans(t)
            assert len(plans) == 16 and [D.stream_of(p) for p in plans] == ["text"] * 8 + ["dense"] * 8
            text, dense = plans[:8], plans[8:]
            for i in range(8):
                k = 8 * t + i
                assert _same(text[i], c3.plan(k), E3_PLAN_KEYS) and text[i].step == t, (seed, t, i)
                assert _same(dense[i], full.plan(k), RO_PLAN_KEYS) and dense[i].step == t, (seed, t, i)
            # 文本的两个微批（每 4 条）各自恰好是 C3 运行里的一个微批（C3 每步 16 条，微批边界也在 4 的倍数上）
            for g in range(0, 8, rc["micro_batch_size"]):
                ks = [p.k for p in text[g: g + rc["micro_batch_size"]]]
                assert len({(k // 16, (k % 16) // c3_micro) for k in ks}) == 1 and len(ks) == c3_micro
            # 任务内的序号：文本流每步前进 4（两个任务交替），稠密流前进 2（四个任务交替）
            assert sorted(p.j for p in text if p.task == "refseg") == [4 * t + x for x in range(4)]
            assert sorted(p.j for p in dense if p.task == "refseg") == [2 * t, 2 * t + 1]
        # 第 0 步：refseg、depth 各前 2 条以两种监督各出现一次（同一个置换：任务号两边相同）
        s0 = ms.step_plans(0)
        both = {(p.task, p.sample_id, p.epoch) for p in s0[:8]} & {(p.task, p.sample_id, p.epoch) for p in s0[8:]}
        assert both == {(p.task, p.sample_id, p.epoch) for p in s0[8:] if p.task in ("refseg", "depth")} and len(both) == 4
    a = D.MixedSchedule(pools, CFG, TEXT_CFG, seed=0, per_step=per).step_plans(3)
    b = D.MixedSchedule(pools, CFG, TEXT_CFG, seed=1, per_step=per).step_plans(3)
    assert [p.sample_id for p in a] != [p.sample_id for p in b]  # 两条流都跟着运行种子走
    with pytest.raises(ValueError, match="样本流"):
        D.MixedSchedule(pools, CFG, TEXT_CFG, seed=0, per_step={"text": 8})
    with pytest.raises(TypeError):
        D.stream_of(object())


def _same_inputs(x, y) -> None:
    assert set(x.tensors) == set(y.tensors)
    for k in x.tensors:
        vx, vy = x.tensors[k], y.tensors[k]
        assert (torch.equal(vx, vy) if torch.is_tensor(vx) else vx == vy), k
    assert (x.order, x.instruction_span, x.images, x.query_span) == (y.order, y.instruction_span, y.images, y.query_span)


@pytest.mark.parametrize("key", FAMILIES)
def test_examples_and_inputs_identical(key, pools):
    """E3a+C3 的构造（适配器注入了查询 token 与 LoRA）造出的文本样本与 C3 的构造（没有查询 token 的适配器）逐个相同，连同输入张量
    （文本微批临时取消查询登记）；稠密样本与 E3a full 的构造逐个相同，连同输入张量（带查询 token）。正式的每步 8 + 8 条、前 4 步。"""
    rc = _rc(tiny=False)
    rc["gradient_checkpointing"] = False
    tr = _mixed_trainer(key, rc=rc)
    ad_c3 = tiny_adapter(key)  # C3：没有查询 token
    full = Trainer(tiny_adapter(key), CFG, resolve_run(CFG, variant="full", tiny=True), variant="full", seed=0, tiny=True)
    full.setup_fresh()
    mb = _builders(tr, pools, key)
    c3b = _c3_builder(ad_c3, pools, key)
    fb = D.ExampleBuilder(CFG, pools, out_size=full.readout.spec.out_size)
    ms = D.MixedSchedule(pools, CFG, TEXT_CFG, seed=0, per_step=rc["mixed"]["per_step"])
    c3 = E3D.Schedule({t: pools[t] for t in TEXT_CFG.tasks}, TEXT_CFG, seed=0, supervision="text", batch_size=16)
    fs = D.Schedule(pools, CFG, seed=0, batch_size=16)
    seen = set()
    for t in range(4):
        plans = ms.step_plans(t)
        text = [mb.build(p) for p in plans[:8]]
        for i, a in enumerate(text):
            b = c3b.build(c3.plan(8 * t + i))
            assert _same(a.plan, b.plan, E3_PLAN_KEYS)
            assert (a.instruction, a.order, a.target_ids, a.weights, a.target_text) == (b.instruction, b.order, b.target_ids,
                                                                                        b.weights, b.target_text)
            assert len(a.images) == len(b.images) == 1 and np.array_equal(a.images[0], b.images[0])
            assert json.dumps(a.info, sort_keys=True, default=str) == json.dumps(b.info, sort_keys=True, default=str)
            seen.add(a.plan.text_kind)
        for x, y in zip(tr.text_inputs(text), [ad_c3.build_inputs(b.images, b.instruction, b.order)
                                                for b in (c3b.build(c3.plan(8 * t + i)) for i in range(8))]):
            assert x.query_span is None
            _same_inputs(x, y)
        for i, p in enumerate(plans[8:]):
            a, b = mb.build(p), fb.build(fs.plan(8 * t + i))
            assert _same(a.plan, b.plan, RO_PLAN_KEYS) and type(a) is type(b) is D.Example
            assert (a.instruction, a.order, a.image_key) == (b.instruction, b.order, b.image_key)
            assert np.array_equal(a.image, b.image) and np.array_equal(a.target.values, b.target.values)
            assert (a.target.valid is None) == (b.target.valid is None)
            if a.target.valid is not None:
                assert np.array_equal(a.target.valid, b.target.valid)
            x, y = tr.build_inputs(a), full.build_inputs(b)
            assert x.query_span is not None and len(x.query_span) == CFG.query_n
            _same_inputs(x, y)
            seen.add(p.task)
    assert seen >= {"box", "polygon", "pairs", "refseg", "depth", "semseg", "normals"}, seen


# ---------------------------------------------------------------------------
# 损失：两种平均之和（1 : 1）
# ---------------------------------------------------------------------------


def _manual(tr, text, dense):
    """独立算出的两项（同样的微批分组，求和后再除；每项各自 backward，梯度累加）：文本 = 全词表 logits（lm_head 的权重直接相乘）的
    交叉熵之和 ÷ 文本目标 token 数（C3 的 token_mean）；稠密 = Σ 任务权重 × 逐样本损失 ÷ 稠密样本数（E3a 的 sample_mean）。"""
    ad = tr.ad
    micro = int(tr.rc["micro_batch_size"])
    w_head = ad.lm_head_weight().float()
    n_text = sum(len(e.target_ids) for e in text)
    lt = 0.0
    for s in range(0, len(text), micro):
        g = text[s: s + micro]
        out = ad.training_forward(tr.text_inputs(g), [e.target_ids for e in g], enable_thinking=False)
        ls = F.cross_entropy(out.target_hidden.float() @ w_head.T, out.target_ids, reduction="sum")
        (ls / n_text).backward()
        lt += float(ls.detach())
    ld = 0.0
    tw = CFG.doc["loss"]["task_weights"]
    for g in [[dense[i] for i in idx] for idx in micro_batch_groups([e.task for e in dense], micro)]:
        semseg = g[0].task == "semseg"
        feats = tr.features([tr.build_inputs(e) for e in g], semseg=semseg)
        ro = tr.readout(*feats.args())
        total = 0.0
        for b, e in enumerate(g):
            out = tr.readout.task_output(e.task, ro.embed[b: b + 1], None if ro.class_emb is None else ro.class_emb[b: b + 1])
            tgt = stack_targets(e.task, [e], "cpu")
            if e.task == "refseg":
                ls = LS.refseg_loss(out, tgt["mask"], CFG.task("refseg")["loss"])
            elif e.task == "depth":
                ls = LS.depth_loss(out, tgt["log_gt"], tgt["valid"], CFG.task("depth")["loss"])
            elif e.task == "semseg":
                ls = LS.semseg_loss(out, tgt["label"])
            else:
                ls = LS.normals_loss(out, tgt["gt"], tgt["valid"])
            total = total + float(tw[e.task]) * ls.sum()
        (total / len(dense)).backward()
        ld += float(total.detach())
    return lt / n_text, ld / len(dense), n_text


def test_loss_is_sum_of_two_means(pools):
    """合成的一步（文本 4 条、稠密 4 条，四个稠密任务各一条）：记下的 loss = L_text + L_dense，两项与独立算出的相同；梯度 = 这个和的
    梯度（LoRA、查询嵌入、混合、读出）；不是两种合在一起的平均。"""
    key = "qwen3.5-0.8b"
    rc = _rc()
    rc["lora"] = {**rc["lora"], "dropout": 0.0}  # 关掉 dropout：手算与训练器的前向逐位可比
    rc["optim"] = copy.deepcopy(rc["optim"])
    rc["optim"]["max_grad_norm"] = None  # 不裁剪：比较原始梯度
    rc["mixed"] = {**rc["mixed"], "per_step": {"text": 4, "dense": 4}}
    rc["batch_size"] = 8
    tr = _mixed_trainer(key, rc=rc)
    mb = _builders(tr, pools, key)
    ms = D.MixedSchedule(pools, CFG, TEXT_CFG, seed=0, per_step={"text": 4, "dense": 4})
    examples = [mb.build(p) for p in ms.step_plans(0)]
    text = [e for e in examples if D.stream_of(e.plan) == "text"]
    dense = [e for e in examples if D.stream_of(e.plan) == "dense"]
    assert len(text) == len(dense) == 4 and sorted(e.task for e in dense) == ["depth", "normals", "refseg", "semseg"]
    tr.ad.train_mode()
    params = {**{f"llm.{n}": p for n, p in tr.ad.model.named_parameters() if "lora_" in n}, "query": tr.query_embeds,
              **{f"readout.{n}": p for n, p in tr.readout.named_parameters()}}
    for p in params.values():
        p.grad = None
    lt, ld, n_text = _manual(tr, text, dense)
    want = {n: p.grad.detach().clone() for n, p in params.items() if p.grad is not None}
    for p in params.values():
        p.grad = None
    got = {}
    hook = tr.opt.register_step_pre_hook(lambda opt, a, k: got.update({n: p.grad.detach().clone() for n, p in params.items()
                                                                       if p.grad is not None}))
    rec, _, comps = tr.optimize(examples)
    hook.remove()
    assert rec["loss_by_task"]["text"] == pytest.approx(lt, rel=1e-5)
    assert rec["loss_by_task"]["dense"] == pytest.approx(ld, rel=1e-5)
    assert rec["loss"] == pytest.approx(lt + ld, rel=1e-5)
    assert rec["loss"] == pytest.approx(rec["loss_by_task"]["text"] + rec["loss_by_task"]["dense"], rel=1e-12)
    pooled = (lt * n_text + ld * len(dense)) / (n_text + len(dense))
    assert abs(rec["loss"] - pooled) > 1.0  # 不是合在一起的平均
    assert rec["by_supervision"]["text"]["n_supervised_tokens"] == n_text and rec["by_supervision"]["dense"]["n_samples"] == 4
    assert set(rec["loss_by_task"]) == {"text", "dense", "text/refseg", "text/depth", "dense/refseg", "dense/depth", "dense/semseg",
                                        "dense/normals"}
    assert rec["n_by_task"] == {"dense/depth": 1, "dense/normals": 1, "dense/refseg": 1, "dense/semseg": 1, "text/depth": 2,
                                "text/refseg": 2}
    assert sum(rec["n_supervised_by_task"].values()) == n_text and set(rec["n_supervised_by_task"]) == {"text/refseg", "text/depth"}
    assert [c.generated_tokens for c in comps] == [len(e.target_ids) for e in text] + [0] * 4
    assert set(got) == set(want) and {n.split(".")[0] for n in want} == {"llm", "query", "readout"}
    for n in want:
        assert torch.allclose(got[n], want[n], rtol=1e-4, atol=1e-7), n


def test_microbatches_one_supervision_text_first_no_query_no_readout(pools):
    """正式的每步 8 + 8、微批 4：先 2 个文本微批（每个 4 条，没有查询 token，只走语言损失），后 3 个稠密微批（E3a 的规则）；读出只在
    稠密微批上调用；一步之后查询登记照旧。"""
    key = "gemma-4-e2b-it"
    rc = _rc(tiny=False)
    rc["gradient_checkpointing"] = False
    tr = _mixed_trainer(key, rc=rc)
    mb = _builders(tr, pools, key)
    ms = D.MixedSchedule(pools, CFG, TEXT_CFG, seed=0, per_step=rc["mixed"]["per_step"], order=rc["mixed"]["order"])
    examples = [mb.build(p) for p in ms.step_plans(0)]
    calls = []
    real_text, real_feat, real_loss = tr.text_inputs, tr.features, tr.c3._task_loss

    def spy_text(group):
        inputs = real_text(group)
        calls.append(("text", [D.stream_of(e.plan) for e in group], [x.query_span for x in inputs]))
        return inputs

    def spy_feat(inputs, *, semseg):
        calls.append(("dense", semseg, [len(x.query_span) for x in inputs]))
        return real_feat(inputs, semseg=semseg)

    def spy_loss(out, group, *, grid):
        calls.append(("lm_loss", grid, len(group)))
        return real_loss(out, group, grid=grid)

    tr.text_inputs, tr.features, tr.c3._task_loss = spy_text, spy_feat, spy_loss
    tr.ad.train_mode()
    rec, _, comps = tr.optimize(examples)
    q = CFG.query_n
    assert calls == [("text", ["text"] * 4, [None] * 4), ("lm_loss", False, 4)] * 2 + [
        ("dense", True, [q] * 2), ("dense", False, [q] * 4), ("dense", False, [q] * 2)]
    assert len(comps) == 16 and math.isfinite(rec["loss"]) and rec["n_samples"] == 16
    assert rec["by_supervision"]["text"]["n_samples"] == rec["by_supervision"]["dense"]["n_samples"] == 8
    assert rec["n_query_tokens"] == 8 * q and rec["by_supervision"]["dense"]["n_class_rows"] == rec["n_class_rows"] == 2
    img = examples[8].image
    assert tr.ad.build_inputs([img], CFG.instruction("depth"), "instruction_first").query_span is not None  # 登记已恢复


def test_text_inputs_entry_assertion_and_guards(pools):
    """入口断言的反例：取消查询登记失效时（文本输入里混进了查询 token）报错；变体与 run_cfg、文本流配置对不上时拒绝构造。"""
    key = "qwen3.5-0.8b"
    tr = _mixed_trainer(key)
    mb = _builders(tr, pools, key)
    ex = [mb.build(p) for p in D.MixedSchedule(pools, CFG, TEXT_CFG, seed=0, per_step={"text": 2, "dense": 2}).step_plans(0)][:2]
    tr.ad.clear_query_tokens, real = (lambda: None), tr.ad.clear_query_tokens
    with pytest.raises(RuntimeError, match="查询 token"):
        tr.text_inputs(ex)
    tr.ad.clear_query_tokens = real
    assert all(x.query_span is None for x in tr.text_inputs(ex))
    full_rc = resolve_run(CFG, variant="full", tiny=True)
    with pytest.raises(ValueError, match="不符"):
        Trainer(tiny_adapter(key), CFG, full_rc, variant=E3A_C3, seed=0, tiny=True, text_cfg=TEXT_CFG)
    with pytest.raises(ValueError, match="不符"):
        Trainer(tiny_adapter(key), CFG, _rc(), variant="full", seed=0, tiny=True)
    with pytest.raises(ValueError, match="文本流的配置"):
        Trainer(tiny_adapter(key), CFG, _rc(), variant=E3A_C3, seed=0, tiny=True)
    old_e3b = E3C.load_config(REPO / "tests" / "fixtures" / "e3" / "e3b_train_35e69b56.json")
    with pytest.raises(ValueError, match="sha256"):
        Trainer(tiny_adapter(key), CFG, _rc(), variant=E3A_C3, seed=0, tiny=True, text_cfg=old_e3b)
    with pytest.raises(ValueError, match="text_cfg"):
        Trainer(tiny_adapter(key), CFG, full_rc, variant="full", seed=0, tiny=True, text_cfg=TEXT_CFG)


@pytest.mark.parametrize("key", FAMILIES)
def test_only_structural_difference_is_query_row(key, pools):
    """与 C3 的模型结构差别（配置 basis 的披露）：搭建之后，输入嵌入与 lm_head 只有查询占位符那一行与"base + LoRA"（C3 的搭建）不同，
    它是基础词表行的均值；没有新增 token；LoRA 的目标相同。训练两步之后这一行（与 base 的全部权重）逐位不变。"""
    tr = _mixed_trainer(key)
    ad = tr.ad
    c3 = tiny_adapter(key)
    c3.inject_lora(r=32, lora_alpha=64, lora_dropout=0.05)
    assert ad.lora_target_modules() == c3.lora_target_modules() and not ad.grid_token_ids
    qid = ad.query_token_id
    for get in (lambda a: a.model.get_input_embeddings().weight, lambda a: a.lm_head_weight()):
        x, y = get(ad).detach(), get(c3).detach()
        assert x.shape == y.shape
        assert (x != y).any(-1).nonzero().flatten().tolist() == [qid]
    assert len(ad.tokenizer) == len(c3.tokenizer)  # 文本的分词不变（Qwen、InternVL 的占位符不进 tokenizer；Gemma 复用 <unusedN>）
    snap = {n: p.detach().clone() for n, p in ad.model.named_parameters() if "lora_" not in n}
    mb = _builders(tr, pools, key)
    ms = D.MixedSchedule(pools, CFG, TEXT_CFG, seed=0, per_step={"text": 2, "dense": 2})
    ad.train_mode()
    for t in range(2):
        tr.optimize([mb.build(p) for p in ms.step_plans(t)])
    assert all(torch.equal(p, snap[n]) for n, p in ad.model.named_parameters() if "lora_" not in n)
    tr.check_base_unchanged()


# ---------------------------------------------------------------------------
# 整个运行
# ---------------------------------------------------------------------------


def _spec(key, ad, variant):
    e = registry.get_model(key)
    return RunSpec(model_key=key, family=e.family, repo=e.repo, revision=e.revision, adapter_model_key=str(ad.model_key), tiny=True,
                   image_policy=policy_spec(ad.image_policy), model_dtype="fp32", variant=variant, seed=0)


def _e3_spec(key, ad):
    e = registry.get_model(key)
    return E3T.RunSpec(model_key=e.key, family=e.family, repo=e.repo, revision=e.revision, adapter_model_key=ad.model_key, tiny=True,
                       image_policy=policy_spec(ad.image_policy), model_dtype="fp32", supervision="text", seed=0)


def _log(leaf: Path) -> list[dict]:
    return [json.loads(x) for x in (leaf / "train_log.jsonl").read_text().splitlines()]


def _tensors(d: Path) -> dict:
    from safetensors.torch import load_file

    return {f"{f.relative_to(d)}:{k}": v for f in sorted(d.rglob("*.safetensors")) for k, v in load_file(str(f)).items()}


def test_run_totals_match_c3_and_e3a_full_and_resume_identical(tmp_path):
    """tiny：E3a+C3 8 步 × (2 + 2) 条；C3、E3a full 各 4 步 × 4 条。两种监督的合计（样本数、各类 token、分任务的样本数与监督 token）
    与 C3、E3a full 单独训练的合计逐个相同；文本流的运行身份与 C3 的对应项相同。中途崩溃后续跑，与一次跑完逐字节相同。"""
    key = "qwen3.5-0.8b"
    ad = tiny_adapter(key)
    c3_rc = E3C.resolve_run(TEXT_CFG, supervision="text", tiny=True)
    assert E3T.run_training(ad, TEXT_CFG, _e3_spec(key, ad), c3_rc, E3D.E3Paths.fixtures(), tmp_path / "c3", log=lambda m: None)["complete"]
    c3_leaf = E3T.leaf_dir(tmp_path / "c3", "c3", key, 0)
    ad = tiny_adapter(key)
    s = run_training(ad, CFG, _spec(key, ad, "full"), resolve_run(CFG, variant="full", tiny=True), D.E3aPaths.fixtures(),
                     tmp_path / "full", log=lambda m: None)
    assert s["complete"]
    full_leaf = leaf_dir(tmp_path / "full", "full", key, 0)
    ad = tiny_adapter(key)
    s = run_training(ad, CFG, _spec(key, ad, E3A_C3), _rc(), D.E3aPaths.fixtures(), tmp_path / "one", log=lambda m: None)
    assert s["complete"] and s["steps"] == 8 and s["variant"] == "e3a_c3"
    mx = leaf_dir(tmp_path / "one", E3A_C3, key, 0)
    assert mx == tmp_path / "one" / "e3a" / "e3a_c3" / key / "seed0"
    log = _log(mx)
    assert len(log) == 8 and all(r["n_samples"] == 4 and r["by_supervision"]["text"]["n_samples"] == 2 for r in log)
    acc = json.loads((mx / "accounting.json").read_text())
    assert s["by_supervision"] == acc["by_supervision"]
    c3_acc = json.loads((c3_leaf / "accounting.json").read_text())
    full_acc = json.loads((full_leaf / "accounting.json").read_text())
    assert acc["by_supervision"]["text"] == {k: c3_acc[k] for k in acc["by_supervision"]["text"]}
    assert acc["by_supervision"]["dense"] == {k: full_acc[k] for k in acc["by_supervision"]["dense"]}
    assert set(acc["by_supervision"]["text"]) == {"n_samples", "n_supervised_tokens", "n_target_tokens", "n_prompt_tokens",
                                                  "n_visual_tokens", "n_image_pixels"}
    c3_log, full_log = _log(c3_leaf), _log(full_leaf)
    for t in TEXT_CFG.tasks:
        assert sum(r["n_supervised_by_task"].get(f"text/{t}", 0) for r in log) == sum(r["n_supervised_by_task"].get(t, 0) for r in c3_log)
    for t in CFG.tasks:
        assert sum(r["n_by_task"].get(f"dense/{t}", 0) for r in log) == sum(r["n_by_task"].get(t, 0) for r in full_log)
    # 文本流的运行身份 = C3 运行身份的对应项
    ident = yaml.safe_load((mx / "config.yaml").read_text())
    c3_ident = yaml.safe_load((c3_leaf / "config.yaml").read_text())
    ts = ident["text_stream"]
    assert ident["variant"] == "e3a_c3" and ident["seed"] == 0 and ident["training"]["mixed"]["per_step"] == {"text": 2, "dense": 2}
    assert ts["config"] == c3_ident["config"] and ts["instructions"] == c3_ident["instructions"] and ts["c3"] == c3_ident["c3"]
    assert ts["stop_id"] == c3_ident["stop_id"] and (ts["variant"], ts["supervision"]) == (c3_ident["variant"], c3_ident["supervision"])
    assert ts["training"] == {k: c3_ident["training"][k] for k in ts["training"]}
    for t in TEXT_CFG.tasks:
        assert (ident["data"][t]["ids_sha256"], ident["data"][t]["n"]) == (c3_ident["data"][t]["ids_sha256"], c3_ident["data"][t]["n"])
    full_ident = yaml.safe_load((full_leaf / "config.yaml").read_text())
    assert set(ident) - set(full_ident) == {"text_stream"}
    assert {k for k in full_ident if full_ident[k] != ident[k]} == {"variant", "training"}
    readme = (mx / "README.md").read_text()
    assert "日程长度是 C3（4 步）的 2 倍、E3a full（4 步）的 2 倍" in readme and "以两种监督各出现一次" in readme
    meta = json.loads((mx / "meta.json").read_text())
    assert any("e3b_train.json" in json.dumps(v) for v in meta.values())  # 文本流的配置也登记在 meta 的数据文件里
    # 续跑
    ad = tiny_adapter(key)
    with pytest.raises(RuntimeError, match="测试注入的中断"):
        run_training(ad, CFG, _spec(key, ad, E3A_C3), _rc(), D.E3aPaths.fixtures(), tmp_path / "two", log=lambda m: None,
                     _fail_after_step=5)
    ad = tiny_adapter(key)
    assert run_training(ad, CFG, _spec(key, ad, E3A_C3), _rc(), D.E3aPaths.fixtures(), tmp_path / "two", resume=True,
                        stop_after_steps=1, log=lambda m: None) is None
    ad = tiny_adapter(key)
    run_training(ad, CFG, _spec(key, ad, E3A_C3), _rc(), D.E3aPaths.fixtures(), tmp_path / "two", resume=True, log=lambda m: None)
    lb = leaf_dir(tmp_path / "two", E3A_C3, key, 0)
    for f in ("train_log.jsonl", "config.yaml", "README.md", "train_summary.json"):
        assert (mx / f).read_bytes() == (lb / f).read_bytes(), f
    assert _files(mx / "ckpt" / "final") == _files(lb / "ckpt" / "final")
    ta, tb = _tensors(mx / "ckpt" / "final"), _tensors(lb / "ckpt" / "final")
    assert set(ta) == set(tb) and all(torch.equal(ta[k], tb[k]) for k in ta)
    assert json.loads((lb / "accounting.json").read_text())["by_supervision"] == acc["by_supervision"]
    # 运行身份不同（例如换成 full 的 run_cfg）时拒绝续跑；RunSpec 与 run_cfg 的变体对不上时拒绝
    ad = tiny_adapter(key)
    with pytest.raises(ValueError, match="不符"):
        run_training(ad, CFG, _spec(key, ad, E3A_C3), resolve_run(CFG, variant="full", tiny=True), D.E3aPaths.fixtures(),
                     tmp_path / "three", log=lambda m: None)


# ---------------------------------------------------------------------------
# 命令行
# ---------------------------------------------------------------------------


def run_script(name: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(REPO / "scripts" / f"{name}.py"), *args], capture_output=True, text=True, cwd=REPO)


def test_cli_three_families_checkpoint_loads(tmp_path):
    out = tmp_path / "m"
    p = run_script("train_e3a", "--tiny", "--model", *FAMILIES, "--variant", "full", "--with-text", "--seed", "0", "--out", str(out))
    assert p.returncode == 0, p.stderr[-3000:]
    assert p.stdout.count("e3a_c3 seed0 steps=8") == 3
    for key in FAMILIES:
        leaf = out / "e3a" / "e3a_c3" / key / "seed0"
        ck = leaf / "ckpt" / "final"
        assert sorted(str(x.relative_to(ck)) for x in ck.rglob("*") if x.is_file()) == [
            "model/lora/adapter_config.json", "model/lora/adapter_model.safetensors", "model/manifest.json",
            "model/token_rows.safetensors", "model/tokens.json", "readout/decoder.safetensors", "readout/heads.safetensors",
            "readout/layer_mix.safetensors", "readout/query_embeds.safetensors", "readout/readout.json"]  # 与 E3a full 同一种
        assert sorted(x.name for x in (leaf / "ckpt").iterdir()) == ["final"]
        assert json.loads((ck / "readout" / "readout.json").read_text())["spec"]["variant"] == "full"
        assert all(math.isfinite(r["loss"]) for r in _log(leaf))
    ck = out / "e3a" / "e3a_c3" / FAMILIES[0] / "seed0" / "ckpt" / "final"
    p = run_script("eval_e3a", "--tiny", "--checkpoint", str(ck), "--model", FAMILIES[0], "--variant", "full", "--seed", "0",
                   "--task", "refseg", "--limit", "2", "--out", str(tmp_path / "e"))
    assert p.returncode == 0, p.stderr[-3000:]
    mc = json.loads((tmp_path / "e" / "eval" / "e3a" / "full" / FAMILIES[0] / "seed0" / "merge_check.json").read_text())
    assert mc["pass"] is True and list(mc["checks"]) == ["merge"]
    p = run_script("run_e1", "--tiny", "--model", FAMILIES[0], "--task", "refexp_box", "--limit", "1", "--checkpoint", str(ck / "model"),
                   "--checkpoint-label", "t_text/e3a_c3/seed0", "--out", str(tmp_path / "t"))
    assert p.returncode == 0, p.stderr[-3000:]
    assert (tmp_path / "t" / "t_text" / "e3a_c3" / "seed0" / FAMILIES[0] / "refexp_box" / "metrics.json").exists()


def test_cli_rules(tmp_path):
    p = run_script("train_e3a", "--tiny", "--model", FAMILIES[0], "--variant", "c1", "--with-text", "--seed", "0",
                   "--out", str(tmp_path / "x"))
    assert p.returncode == 2 and "--with-text 只配 --variant full" in p.stderr
    p = run_script("train_e3a", "--tiny", "--model", FAMILIES[0], "--variant", "full", "--with-text", "--tasks", "refseg", "depth",
                   "--seed", "0", "--out", str(tmp_path / "x"))
    assert p.returncode == 2 and "--tasks" in p.stderr
    p = run_script("train_e3a", "--model", FAMILIES[0], "--variant", "full", "--with-text", "--seed", "0", "--out", str(tmp_path / "0044_x"))
    assert p.returncode == 2 and "要给 --data-root" in p.stderr  # 真模型要训练池
    assert MIXED_STREAMS == ("text", "dense")
