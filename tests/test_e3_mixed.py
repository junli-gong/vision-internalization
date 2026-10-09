"""DEV-0042 §1：E3b+C3 混合训练（scripts/train_e3b.py --supervision grid+text，变体 e3b_c3；配置的 mixed 节）。

- 配置：mixed 节的数字（每步 8 + 8、8000 步、1 : 1、文本在前）与来历（8000 × 8 = 4000 × 16）钉住；load_config 拒绝不是 1 : 1 的权重、
  对不上的步数与条数；resolve_run 的 grid+text 与 E3b 的网格监督只差 steps、warmup_steps、supervision、variant 与 mixed 一节；
- 配置只新增 mixed 节（与 DEV-0035 的最后一版 35e69b56… 比较），已有的 E3b、C3、C1′、预热的 resolve_run 逐项不变，运行身份
  （config.yaml）除 config 一项（sha256、version、path）以外逐字节不变；
- 逐样本相同：第 t 步的文本 8 条 = C3 同一种子第 8t … 8t + 7 个槽位（样本 id、epoch、text_kind、指令、图、输入 token、目标 token 逐个
  相同；三家族），网格 8 条 = E3b 同一种子的第 8t … 8t + 7 个槽位（含修正对）；同一步里同一张图以两种监督各出现一次；
- 损失：一步 = 文本的逐 token 平均 + 网格的逐 token 平均（1 : 1），用独立算出的两项核对数值与梯度；不是把两种 token 合在一起的平均；
  每个微批只含一种监督、文本在前；
- 整个运行：两种监督各自的样本数与 token 数合计，与 C3、E3b 单独训练（同一种子）的合计逐个相同；续跑与一次跑完逐字节相同；
- 命令行：三家族 --tiny 跑通，检查点与 E3b 同一种，能被 eval_e3b、run_e1 --checkpoint 加载；E5 读叶目录 config.yaml 的 variant 与 seed。
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import yaml

from src.adapters import registry
from src.adapters.checkpoint import policy_spec
from src.adapters.factory import make_adapter
from src.adapters.image_policy import parse_policy
from src.adapters.training import restricted_targets
from src.e3 import data as D
from src.e3.config import MIXED, STREAMS, ConfigError, load_config, resolve_run
from src.e3.train import RunSpec, Trainer, leaf_dir, run_training

REPO = Path(__file__).resolve().parents[1]
FAMILIES = ("qwen3.5-0.8b", "gemma-4-e2b-it", "internvl3.5-2b-hf")
CFG = load_config()
OLD_CONFIG = REPO / "tests" / "fixtures" / "e3" / "e3b_train_35e69b56.json"  # DEV-0035 之后、DEV-0042 之前的 E3b 配置原样（D70）
OLD_SHA = "35e69b56324281356aba22419391fe8d092e3c0f0413ee1af4ed8ea74b60f457"


def _tiny(key):
    fam = registry.get_model(key).family
    t = CFG.doc["tiny"]
    return make_adapter(key, device="cpu", dtype=torch.float32, image_policy=parse_policy(t["image_policies"][fam]), tiny=True,
                        init_std=float(t["init_std"]))


@pytest.fixture(scope="module")
def pools():
    paths = D.E3Paths.fixtures()
    return {t: D.load_pool(t, paths, CFG.task(t)).records for t in CFG.tasks}


def _encode(ad):
    return lambda s: list(ad.tokenizer(s, add_special_tokens=False)["input_ids"])


def _builder(ad, pools, supervision, tokens, key, seed=0):
    from src.e3.tokens import family_stop_id

    return D.ExampleBuilder(CFG, pools, seed=seed, supervision=supervision, stop_id=family_stop_id(ad), tokens=tokens,
                            encode=_encode(ad), family=registry.get_model(key).family, format_boxes=ad.format_boxes)


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------


def test_config_mixed_pins():
    m = CFG.doc["mixed"]
    assert m["per_step"] == {"text": 8, "grid": 8} and m["steps"] == 8000 and m["loss_weights"] == {"text": 1, "grid": 1}
    assert m["order"] == ["text", "grid"] and m["tiny"] == {"per_step": {"text": 2, "grid": 2}, "steps": 8}
    batch = CFG.doc["batch"]["micro_batch_size"] * CFG.doc["batch"]["grad_accum_steps"]
    assert m["steps"] * m["per_step"]["text"] == m["steps"] * m["per_step"]["grid"] == CFG.doc["steps"] * batch == 64000
    assert m["steps"] == 2 * CFG.doc["steps"] and sum(m["per_step"].values()) == batch
    e3b = resolve_run(CFG, supervision="grid", tiny=False)
    rc = resolve_run(CFG, supervision=MIXED, tiny=False)
    assert rc["variant"] == "e3b_c3" and rc["supervision"] == MIXED and rc["steps"] == 8000 and rc["warmup_steps"] == 240
    assert rc["batch_size"] == 16 and rc["micro_batch_size"] == 4 and rc["grad_accum_steps"] == 4
    same = {k for k in e3b if k not in ("supervision", "variant", "steps", "warmup_steps")}
    assert {k: rc[k] for k in same} == {k: e3b[k] for k in same}  # lr、LoRA、优化器、混合、修正对比例同 E3b
    assert set(rc) - set(e3b) == {"mixed"}
    mx = rc["mixed"]
    assert mx["per_step"] == {"text": 8, "grid": 8} and mx["micro_batches"] == {"text": 2, "grid": 2}
    assert mx["loss_weights"] == {"text": 1.0, "grid": 1.0} and mx["schedule_length_ratio"] == 2.0
    assert mx["reference"] == {"steps": 4000, "batch_size": 16, "variants": {"text": "c3", "grid": "e3b"}, "samples_per_stream": 64000}
    t = resolve_run(CFG, supervision=MIXED, tiny=True)
    assert (t["steps"], t["batch_size"], t["warmup_steps"], t["mixed"]["per_step"]) == (8, 4, 1, {"text": 2, "grid": 2})
    assert t["mixed"]["schedule_length_ratio"] == 2.0


@pytest.mark.parametrize("mutate,match", [
    (lambda m: m.__setitem__("loss_weights", {"text": 1, "grid": 2}), "1 : 1"),
    (lambda m: m.__setitem__("loss_weights", {"text": 0.5, "grid": 0.5}), "1 : 1"),
    (lambda m: m.__setitem__("steps", 4000), "逐样本相同"),
    (lambda m: m.__setitem__("per_step", {"text": 6, "grid": 10}), "微批"),
    (lambda m: m.__setitem__("per_step", {"text": 8, "grid": 4}), "每步条数"),
    (lambda m: m.__setitem__("per_step", {"text": 16}), "per_step"),
    (lambda m: m.__setitem__("order", ["text", "text"]), "order"),
    (lambda m: m["tiny"].__setitem__("steps", 4), "mixed.tiny"),
])
def test_config_rejects_bad_mixed(tmp_path, mutate, match):
    doc = copy.deepcopy(dict(CFG.doc))
    mutate(doc["mixed"])
    p = tmp_path / "c.json"
    p.write_text(json.dumps(doc, ensure_ascii=False))
    with pytest.raises(ConfigError, match=match):
        load_config(p)


def test_config_only_adds_mixed():
    """DEV-0042 只新增 mixed 节（并升版本号），其余条目与 DEV-0035 之后的版本（35e69b56…，D70）逐项相同。"""
    assert hashlib.sha256(OLD_CONFIG.read_bytes()).hexdigest() == OLD_SHA
    old = json.loads(OLD_CONFIG.read_text(encoding="utf-8"))
    assert set(CFG.doc) - set(old) == {"mixed"} and not set(old) - set(CFG.doc)
    assert {k for k in old if CFG.doc[k] != old[k]} == {"version", "version_note"}
    assert (old["version"], CFG.doc["version"]) == ("e3b-train-v2", "e3b-train-v3")
    assert list(CFG.doc)[-4:] == ["c1prime", "warmup", "mixed", "tiny"]
    assert CFG.sha256 != OLD_SHA


EXISTING = [("grid", None), ("text", None), ("grid", "c1prime"), ("warmup", None)]


@pytest.mark.parametrize("tiny", [False, True])
@pytest.mark.parametrize("supervision,variant", EXISTING)
def test_existing_resolve_run_unchanged(supervision, variant, tiny):
    old = load_config(OLD_CONFIG)
    assert resolve_run(CFG, supervision=supervision, tiny=tiny, variant=variant) == \
        resolve_run(old, supervision=supervision, tiny=tiny, variant=variant)


def _spec(key, ad, supervision, *, variant=None):
    e = registry.get_model(key)
    return RunSpec(model_key=e.key, family=e.family, repo=e.repo, revision=e.revision, adapter_model_key=ad.model_key, tiny=True,
                   image_policy=policy_spec(ad.image_policy), model_dtype="fp32", supervision=supervision, seed=0,
                   frozen_llm=variant == "c1prime")


@pytest.mark.parametrize("supervision,variant", EXISTING)
def test_existing_run_identity_unchanged(supervision, variant, tmp_path):
    """已有的四个变体：新配置下写出的 config.yaml（运行身份）与旧配置下的，除 config 一项（sha256、version、path）以外逐字节相同；
    第一步的 train_log 也逐字节相同（新配置只多了 mixed 节，不影响它们）。"""
    key = "qwen3.5-0.8b"
    texts, logs = {}, {}
    for name, cfg in (("old", load_config(OLD_CONFIG)), ("new", CFG)):
        rc = resolve_run(cfg, supervision=supervision, tiny=True, variant=variant)
        rc["data_workers"] = 0
        ad = _tiny(key)
        spec = _spec(key, ad, supervision, variant=variant)
        run_training(ad, cfg, spec, rc, D.E3Paths.fixtures(), tmp_path / name, stop_after_steps=1, log=lambda m: None)
        leaf = leaf_dir(tmp_path / name, spec.variant, key, 0)
        ident = yaml.safe_load((leaf / "config.yaml").read_text(encoding="utf-8"))
        texts[name] = ident
        logs[name] = (leaf / "train_log.jsonl").read_bytes()
    a, b = texts["old"], texts["new"]
    assert list(a) == list(b)  # 键与顺序相同
    assert {k for k in a if a[k] != b[k]} == {"config"}
    assert b["config"] == {"path": "configs/e3b_train.json", "sha256": CFG.sha256, "version": "e3b-train-v3"}
    assert a["config"]["sha256"] == OLD_SHA
    strip = lambda d: yaml.safe_dump({k: v for k, v in d.items() if k != "config"}, sort_keys=False, allow_unicode=True)  # noqa: E731
    assert strip(a) == strip(b)
    assert logs["old"] == logs["new"]


# ---------------------------------------------------------------------------
# 逐样本相同
# ---------------------------------------------------------------------------

PLAN_KEYS = ("k", "task", "j", "index", "epoch", "sample_id", "kind", "text_kind")


def _same_plan(a: D.SlotPlan, b: D.SlotPlan) -> bool:
    return all(getattr(a, f) == getattr(b, f) for f in PLAN_KEYS)


def test_streams_are_c3_and_e3b_slot_by_slot(pools):
    rc = resolve_run(CFG, supervision=MIXED, tiny=False)
    per = rc["mixed"]["per_step"]
    for seed in CFG.seeds:
        ms = D.MixedSchedule(pools, CFG, seed=seed, per_step=per, order=rc["mixed"]["order"])
        c3 = D.Schedule(pools, CFG, seed=seed, supervision="text", batch_size=16)
        e3b = D.Schedule(pools, CFG, seed=seed, supervision="grid", batch_size=16)
        kinds = set()
        for t in range(64):
            plans = ms.step_plans(t)
            assert len(plans) == 16 and [D.stream_of(p) for p in plans] == ["text"] * 8 + ["grid"] * 8
            text, grid = plans[:8], plans[8:]
            for i in range(8):
                k = 8 * t + i
                assert _same_plan(text[i], c3.plan(k)) and text[i].step == t, (seed, t, i)
                assert _same_plan(grid[i], e3b.plan(k)) and grid[i].step == t, (seed, t, i)
                # 两条流的同一个槽位是同一条样本（同一张图、同一个 epoch），以两种监督各出现一次
                assert (text[i].task, text[i].sample_id, text[i].epoch) == (grid[i].task, grid[i].sample_id, grid[i].epoch)
                kinds |= {(text[i].task, text[i].text_kind), (grid[i].task, grid[i].kind)}
        assert kinds == {("refseg", "box"), ("refseg", "polygon"), ("depth", "pairs"), ("refseg", "grid"), ("refseg", "correction"),
                         ("depth", "grid"), ("depth", "correction")}
        # 全程：文本流的 64,000 条恰好是 C3 运行的全部槽位（步数 × 每步条数），网格流同理
        assert rc["steps"] * per["text"] == 4000 * 16 == rc["steps"] * per["grid"]
    # 换种子就换样本流（两条流都跟着运行种子走）
    a = D.MixedSchedule(pools, CFG, seed=0, per_step=per).step_plans(3)
    b = D.MixedSchedule(pools, CFG, seed=1, per_step=per).step_plans(3)
    assert [p.sample_id for p in a] != [p.sample_id for p in b]


def _same_inputs(x, y) -> None:
    assert set(x.tensors) == set(y.tensors)
    for k in x.tensors:
        vx, vy = x.tensors[k], y.tensors[k]
        assert (torch.equal(vx, vy) if torch.is_tensor(vx) else vx == vy), k
    assert (x.order, x.instruction_span, x.images) == (y.order, y.instruction_span, y.images)


def _same_example(a: D.Example, b: D.Example) -> None:
    assert _same_plan(a.plan, b.plan)
    assert (a.instruction, a.order, a.target_ids, a.weights, a.target_text) == (b.instruction, b.order, b.target_ids, b.weights,
                                                                                b.target_text)
    assert len(a.images) == len(b.images) and all(np.array_equal(x, y) for x, y in zip(a.images, b.images))
    assert json.dumps(a.info, sort_keys=True, default=str) == json.dumps(b.info, sort_keys=True, default=str)


@pytest.mark.parametrize("key", FAMILIES)
def test_examples_and_inputs_identical(key, pools):
    """E3b+C3 的构造器（适配器加了网格 token）造出的文本样本与 C3 的构造器（没有网格 token 的适配器）逐个相同，连同 build_inputs 的
    输入张量；网格样本与 E3b 的构造器逐个相同。前 4 步 × 每步 8 + 8 条，覆盖框、多边形、点对、网格、修正对。"""
    from src.e3.tokens import add_vocab

    ad_c3 = _tiny(key)  # C3：不加网格 token
    ad_mx = _tiny(key)  # E3b+C3：加网格 token（与 E3b 同一份布局）
    tok = add_vocab(ad_mx, CFG.vocab)
    c3b = _builder(ad_c3, pools, "text", None, key)
    e3bb = _builder(ad_mx, pools, "grid", tok, key)
    mxb = _builder(ad_mx, pools, MIXED, tok, key)
    rc = resolve_run(CFG, supervision=MIXED, tiny=False)
    ms = D.MixedSchedule(pools, CFG, seed=0, per_step=rc["mixed"]["per_step"])
    c3 = D.Schedule(pools, CFG, seed=0, supervision="text", batch_size=16)
    e3b = D.Schedule(pools, CFG, seed=0, supervision="grid", batch_size=16)
    seen = set()
    for t in range(4):
        plans = ms.step_plans(t)
        for i, p in enumerate(plans):
            k = 8 * t + i % 8
            if D.stream_of(p) == "text":
                a, b = mxb.build(p), c3b.build(c3.plan(k))
                _same_inputs(ad_mx.build_inputs(a.images, a.instruction, a.order), ad_c3.build_inputs(b.images, b.instruction, b.order))
                seen.add(p.text_kind)
            else:
                a, b = mxb.build(p), e3bb.build(e3b.plan(k))
                _same_inputs(ad_mx.build_inputs(a.images, a.instruction, a.order), ad_mx.build_inputs(b.images, b.instruction, b.order))
                seen.add((p.task, p.kind))
            _same_example(a, b)
    assert seen >= {"box", "polygon", "pairs", ("refseg", "grid"), ("refseg", "correction"), ("depth", "grid"),
                    ("depth", "correction")}, seen


def test_builder_needs_both_halves(pools):
    ad = _tiny("qwen3.5-0.8b")
    with pytest.raises(ValueError, match="tokens"):
        D.ExampleBuilder(CFG, pools, seed=0, supervision=MIXED, stop_id=1, tokens=None, encode=_encode(ad), family="qwen3.5",
                         format_boxes=ad.format_boxes)
    from src.e3.tokens import add_vocab

    tok = add_vocab(ad, CFG.vocab)
    with pytest.raises(ValueError, match="encode"):
        D.ExampleBuilder(CFG, pools, seed=0, supervision=MIXED, stop_id=1, tokens=tok)
    with pytest.raises(ValueError, match="样本流"):
        D.MixedSchedule(pools, CFG, seed=0, per_step={"text": 8})


# ---------------------------------------------------------------------------
# 损失：两种逐 token 平均之和（1 : 1）
# ---------------------------------------------------------------------------


def _mixed_setup(key, pools, *, rc=None):
    ad = _tiny(key)
    rc = rc or resolve_run(CFG, supervision=MIXED, tiny=True)
    tr = Trainer(ad, CFG, rc, supervision=MIXED, seed=0)
    tr.setup_fresh()
    b = _builder(ad, pools, MIXED, tr.tokens, key)
    s = D.MixedSchedule(pools, CFG, seed=0, per_step=rc["mixed"]["per_step"], order=rc["mixed"]["order"])
    return ad, tr, b, s


def _manual_losses(ad, tr, text, grid, micro):
    """独立算出的两项（按同样的微批分组前向，求和后再除）：文本 = 全词表 logits（lm_head 的权重直接相乘）的交叉熵之和 / 文本目标 token
    数；网格 = 允许集合那几行的 logits 的交叉熵 × 逐 token 权重之和 / 权重和。"""
    w_head = ad.lm_head_weight().float()
    lt, n_text = 0.0, 0
    for s in range(0, len(text), micro):
        g = text[s: s + micro]
        out = ad.training_forward([tr.build_inputs(e) for e in g], [e.target_ids for e in g], enable_thinking=False)
        lt = lt + F.cross_entropy(out.target_hidden.float() @ w_head.T, out.target_ids, reduction="sum")
        n_text += int(out.target_ids.numel())
    lg, wsum = 0.0, 0.0
    for s in range(0, len(grid), micro):
        g = grid[s: s + micro]
        out = ad.training_forward([tr.build_inputs(e) for e in g], [e.target_ids for e in g], enable_thinking=False)
        w = torch.tensor([x for e in g for x in e.weights], dtype=torch.float32)
        for task in ("refseg", "depth"):
            rows = [i for i, e in enumerate(g) if e.plan.task == task]
            if not rows:
                continue
            sel = torch.isin(out.target_rows, torch.tensor(rows)).nonzero().flatten()
            allowed = tr.tokens.allowed_ids(task, CFG.allowed_scope)
            logits = out.target_hidden.index_select(0, sel).float() @ w_head[allowed].T
            ce = F.cross_entropy(logits, restricted_targets(out.target_ids.index_select(0, sel), allowed), reduction="none")
            lg = lg + (ce * w.index_select(0, sel)).sum()
            wsum += float(w.index_select(0, sel).sum())
    return lt / n_text, lg / wsum, n_text, wsum


def test_loss_is_sum_of_two_token_means(pools):
    """合成的一步（文本 4 条、网格 4 条，含深度的不监督格子）：记下的 loss = L_text + L_grid，两项与独立算出的相同；梯度 = 这个和的梯度；
    不是把两种 token 合在一起的平均（那样网格的 1057 个 token 会压倒文本的几十个）。"""
    key = "qwen3.5-0.8b"
    rc = resolve_run(CFG, supervision=MIXED, tiny=True)
    rc["lora"] = {**rc["lora"], "dropout": 0.0}  # 关掉 dropout：手算与训练器的前向逐位可比
    rc["optim"] = copy.deepcopy(rc["optim"])
    rc["optim"]["max_grad_norm"] = None  # 不裁剪：比较原始梯度
    rc["mixed"] = {**rc["mixed"], "per_step": {"text": 4, "grid": 4}}
    rc["batch_size"] = 8
    ad, tr, b, s = _mixed_setup(key, pools, rc=rc)
    s = D.MixedSchedule(pools, CFG, seed=0, per_step={"text": 4, "grid": 4})
    examples = [b.build(p) for p in s.step_plans(0)]
    text = [e for e in examples if D.stream_of(e.plan) == "text"]
    grid = [e for e in examples if D.stream_of(e.plan) == "grid"]
    assert len(text) == len(grid) == 4 and any(0 in e.weights for e in grid if e.plan.task == "depth")
    ad.train_mode()
    lora = {n: p for n, p in ad.model.named_parameters() if "lora_" in n}
    # 独立算的两项与它们之和的梯度（同一个模型状态，optimize 之前）
    for p in lora.values():
        p.grad = None
    lt, lg, n_text, n_grid = _manual_losses(ad, tr, text, grid, int(rc["micro_batch_size"]))
    (lt + lg).backward()
    lt, lg = lt.detach(), lg.detach()
    want = {n: p.grad.detach().clone() for n, p in lora.items()}
    for p in lora.values():
        p.grad = None
    tr.rows.zero_dense_grads()
    got = {}
    hook = tr.opt.register_step_pre_hook(lambda opt, args, kwargs: got.update({n: p.grad.detach().clone() for n, p in lora.items()}))
    rec, _, _ = tr.optimize(examples)
    hook.remove()
    assert rec["loss_by_task"]["text"] == pytest.approx(float(lt), rel=1e-5)
    assert rec["loss_by_task"]["grid"] == pytest.approx(float(lg), rel=1e-5)
    assert rec["loss"] == pytest.approx(float(lt) + float(lg), rel=1e-5)
    assert rec["loss"] == pytest.approx(rec["loss_by_task"]["text"] + rec["loss_by_task"]["grid"], rel=1e-12)
    pooled = (float(lt) * n_text + float(lg) * n_grid) / (n_text + n_grid)
    assert abs(rec["loss"] - pooled) > 1.0  # 不是合在一起的平均
    assert rec["by_supervision"]["text"]["n_supervised_tokens"] == n_text and rec["by_supervision"]["grid"]["n_supervised_tokens"] == n_grid
    assert rec["n_supervised_tokens"] == n_text + int(n_grid)
    assert set(rec["loss_by_task"]) == {"text", "grid", "text/refseg", "text/depth", "grid/refseg", "grid/depth"}
    assert set(rec["n_supervised_by_task"]) == {"text/refseg", "text/depth", "grid/refseg", "grid/depth"}
    assert sum(v for k, v in rec["n_supervised_by_task"].items() if k.startswith("text/")) == n_text
    assert set(got) == set(want)
    for n in want:
        assert torch.allclose(got[n], want[n], rtol=1e-4, atol=1e-7), n


def test_microbatches_hold_one_supervision_text_first(pools):
    key = "gemma-4-e2b-it"
    rc = resolve_run(CFG, supervision=MIXED, tiny=False)  # 正式的每步 8 + 8、微批 4
    rc["gradient_checkpointing"] = False
    ad, tr, b, _ = _mixed_setup(key, pools, rc=rc)
    s = D.MixedSchedule(pools, CFG, seed=0, per_step=rc["mixed"]["per_step"], order=rc["mixed"]["order"])
    examples = [b.build(p) for p in s.step_plans(0)]
    calls = []
    real = tr._task_loss

    def spy(out, group, *, grid):
        calls.append((grid, [D.stream_of(e.plan) for e in group]))
        return real(out, group, grid=grid)

    tr._task_loss = spy
    ad.train_mode()
    rec, _, comps = tr.optimize(examples)
    assert calls == [(False, ["text"] * 4)] * 2 + [(True, ["grid"] * 4)] * 2
    assert len(comps) == 16 and math.isfinite(rec["loss"]) and rec["n_samples"] == 16
    assert rec["by_supervision"]["text"]["n_samples"] == rec["by_supervision"]["grid"]["n_samples"] == 8


def test_single_supervision_loss_path_unchanged(pools):
    """E3b、C3 的训练器不走混合的分支：_loss 仍是同一个函数（E5 的训练器覆盖它），记录里没有 by_supervision。"""
    ad = _tiny("qwen3.5-0.8b")
    rc = resolve_run(CFG, supervision="grid", tiny=True)
    tr = Trainer(ad, CFG, rc, supervision="grid", seed=0)
    tr.setup_fresh()
    assert not tr.mixed and tr.grid_like and tr.with_grid_tokens
    b = D.ExampleBuilder(CFG, pools, seed=0, supervision="grid", stop_id=tr.stop_id, tokens=tr.tokens)
    sch = D.Schedule(pools, CFG, seed=0, supervision="grid", batch_size=4)
    ad.train_mode()
    rec, _, _ = tr.optimize([b.build(p) for p in sch.step_plans(0)])
    assert "by_supervision" not in rec and set(rec["loss_by_task"]) <= {"refseg", "depth"}
    with pytest.raises(ValueError, match="mixed"):
        Trainer(_tiny("qwen3.5-0.8b"), CFG, rc, supervision=MIXED, seed=0)


# ---------------------------------------------------------------------------
# 整个运行
# ---------------------------------------------------------------------------


def _log(leaf: Path) -> list[dict]:
    return [json.loads(x) for x in (leaf / "train_log.jsonl").read_text().splitlines()]


def _tensors(d: Path) -> dict:
    from safetensors.torch import load_file

    out = {}
    for f in sorted(d.rglob("*.safetensors")):
        for k, v in load_file(str(f)).items():
            out[f"{f.relative_to(d)}:{k}"] = v
    return out


def test_run_totals_match_c3_and_e3b_and_resume_identical(tmp_path):
    """tiny：E3b+C3 8 步 × (2 + 2) 条；C3、E3b 各 4 步 × 4 条。两种监督的合计（样本数、各类 token、各类样本、分任务的监督 token）与
    C3、E3b 单独训练的合计逐个相同。中途崩溃后续跑，与一次跑完逐字节相同。"""
    key = "qwen3.5-0.8b"
    paths = D.E3Paths.fixtures()
    runs = {}
    for sup in ("text", "grid", MIXED):
        rc = resolve_run(CFG, supervision=sup, tiny=True)
        ad = _tiny(key)
        spec = _spec(key, ad, sup)
        s = run_training(ad, CFG, spec, rc, paths, tmp_path / "one", log=lambda m: None)
        assert s["complete"] and s["steps"] == rc["steps"]
        runs[sup] = leaf_dir(tmp_path / "one", spec.variant, key, 0)
    mx = runs[MIXED]
    log = _log(mx)
    assert len(log) == 8 and all(r["n_samples"] == 4 and r["by_supervision"]["text"]["n_samples"] == 2 for r in log)
    acc = json.loads((mx / "accounting.json").read_text())
    summ = json.loads((mx / "train_summary.json").read_text())
    assert summ["by_supervision"] == acc["by_supervision"]
    for stream, sup in (("text", "text"), ("grid", "grid")):
        ref = json.loads((runs[sup] / "accounting.json").read_text())
        assert acc["by_supervision"][stream] == {k: ref[k] for k in acc["by_supervision"][stream]}, stream
        ref_log = _log(runs[sup])
        kinds_ref, kinds_mx = {}, {}
        for r in ref_log:
            for k, v in r["n_by_kind"].items():
                kinds_ref[k] = kinds_ref.get(k, 0) + v
        for r in log:
            for k, v in r["n_by_kind"].items():
                if (k == "text") == (stream == "text"):
                    kinds_mx[k] = kinds_mx.get(k, 0) + v
        assert kinds_mx == kinds_ref
        by_task_ref = {t: sum(r["n_supervised_by_task"].get(t, 0) for r in ref_log) for t in ("refseg", "depth")}
        by_task_mx = {t: sum(r["n_supervised_by_task"].get(f"{stream}/{t}", 0) for r in log) for t in ("refseg", "depth")}
        assert by_task_mx == by_task_ref
    ident = yaml.safe_load((mx / "config.yaml").read_text())
    assert ident["variant"] == "e3b_c3" and ident["seed"] == 0 and ident["supervision"] == MIXED
    assert ident["c3"] is not None and ident["noise"] is not None and len(ident["tokens"]["vocab"]) == 67
    assert ident["training"]["mixed"]["schedule_length_ratio"] == 2.0
    readme = (mx / "README.md").read_text()
    assert "日程长度是 c3、e3b（4 步）的 2 倍" in readme and "同一步里同一张图以两种监督各出现一次" in readme
    # 续跑
    rc = resolve_run(CFG, supervision=MIXED, tiny=True)
    ad = _tiny(key)
    with pytest.raises(RuntimeError, match="测试注入的中断"):
        run_training(ad, CFG, _spec(key, ad, MIXED), rc, paths, tmp_path / "two", log=lambda m: None, _fail_after_step=5)
    ad = _tiny(key)
    run_training(ad, CFG, _spec(key, ad, MIXED), rc, paths, tmp_path / "two", resume=True, log=lambda m: None)
    lb = leaf_dir(tmp_path / "two", "e3b_c3", key, 0)
    for f in ("train_log.jsonl", "config.yaml", "README.md", "train_summary.json", "tokens_init.json"):
        assert (mx / f).read_bytes() == (lb / f).read_bytes(), f
    ta, tb = _tensors(mx / "ckpt" / "final" / "model"), _tensors(lb / "ckpt" / "final" / "model")
    assert set(ta) == set(tb) and all(torch.equal(ta[k], tb[k]) for k in ta)
    assert json.loads((lb / "accounting.json").read_text())["by_supervision"] == acc["by_supervision"]


# ---------------------------------------------------------------------------
# 命令行
# ---------------------------------------------------------------------------


def run_script(name: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(REPO / "scripts" / f"{name}.py"), *args], capture_output=True, text=True, cwd=REPO)


def test_cli_three_families_checkpoint_loads(tmp_path):
    out = tmp_path / "m"
    p = run_script("train_e3b", "--tiny", "--model", *FAMILIES, "--supervision", MIXED, "--seed", "0", "--out", str(out))
    assert p.returncode == 0, p.stderr[-3000:]
    assert p.stdout.count("e3b_c3 seed0 steps=8") == 3
    for key in FAMILIES:
        leaf = out / "e3b_c3" / key / "seed0"
        files = sorted(str(x.relative_to(leaf / "ckpt" / "final" / "model")) for x in (leaf / "ckpt" / "final" / "model").rglob("*")
                       if x.is_file())
        assert files == ["lora/adapter_config.json", "lora/adapter_model.safetensors", "manifest.json", "token_rows.safetensors",
                         "tokens.json"]  # 与 E3b 同一种检查点
        assert sorted(x.name for x in (leaf / "ckpt").iterdir()) == ["final"]
        assert all(math.isfinite(r["loss"]) for r in _log(leaf))
    # E5 读起点叶目录的 variant 与 seed（src/e5/stack 的 _leaf_variant / _leaf_seed，同一种格式）
    from src.e5.stack import _leaf_seed, _leaf_variant

    ck = out / "e3b_c3" / FAMILIES[0] / "seed0" / "ckpt" / "final" / "model"
    assert _leaf_variant(ck)[0] == "e3b_c3" and _leaf_seed(ck) == 0
    p = run_script("eval_e3b", "--tiny", "--checkpoint", str(ck), "--model", FAMILIES[0], "--variant", "e3b_c3", "--seed", "0",
                   "--task", "refseg", "--limit", "1", "--k-values", "1", "--out", str(tmp_path / "e"))
    assert p.returncode == 0, p.stderr[-3000:]
    mc = json.loads((tmp_path / "e" / "eval" / "e3b_c3" / FAMILIES[0] / "seed0" / "merge_check.json").read_text())
    assert mc["pass"] is True and list(mc["checks"]) == ["merge", "restricted"]
    p = run_script("run_e1", "--tiny", "--model", FAMILIES[0], "--task", "refexp_box", "--limit", "1", "--checkpoint", str(ck),
                   "--checkpoint-label", "t_text/e3b_c3/seed0", "--out", str(tmp_path / "t"))
    assert p.returncode == 0, p.stderr[-3000:]
    assert (tmp_path / "t" / "t_text" / "e3b_c3" / "seed0" / FAMILIES[0] / "refexp_box" / "metrics.json").exists()


def test_cli_rules(tmp_path):
    p = run_script("train_e3b", "--tiny", "--model", FAMILIES[0], "--supervision", MIXED, "--variant", "e3b", "--seed", "0",
                   "--out", str(tmp_path / "x"))
    assert p.returncode == 2 and "e3b_c3" in p.stderr
    p = run_script("train_e3b", "--model", FAMILIES[0], "--supervision", MIXED, "--seed", "0", "--out", str(tmp_path / "0043_x"))
    assert p.returncode == 2 and "要给 --data-root" in p.stderr  # 真模型要训练池
    assert STREAMS == ("text", "grid")
