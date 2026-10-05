"""DEV-0034 §2、§5：E3a 的训练（src/readout/train.py、data.py、features.py）。CPU fp32 小模型。

- 日程：只给指代分割与深度时，每一步的样本 id 与 E3b 逐个相同（简报 §8.5 的同预算对照）；四个任务的循环；
- 三家族 × full / c1 / c2：一次前向 + 反向的梯度只落在该变体允许的参数上；两步优化后 LLM 的 base 权重逐位不变（c1：整个 LLM 逐位不变），
  LoRA、查询嵌入、读出各自在变；
- 没有像素通路（梯度检查）：full、c1 下解码器输出对像素张量（视觉编码器之前）的梯度存在，而切断读出输入（LLM 的隐藏状态）之后
  不存在；查询位置第 0 层的隐藏状态（= 查询嵌入）与像素无关；反例：读出输入里混进像素，同一检查能发现；
- 续跑：按 --stop-after-steps 分段再 --resume，train_log 与最终检查点逐字节 / 逐位相同；
- 检查点：model/ 里没有查询登记（scripts/run_e1.py --checkpoint 能直接加载、能生成），readout/ 与 model/ 配对，载回后输出与内存里的
  模型相同；c1、c2 没有 LoRA。
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from src.adapters import factory, registry
from src.adapters.checkpoint import read_manifest
from src.e3 import config as E3C
from src.e3 import data as E3D
from src.readout import data as D
from src.readout import evaluate as EV
from src.readout.config import E3aConfig, load_config, resolve_run
from src.readout.features import ReadoutInputs
from src.readout.train import (
    FINAL,
    MODEL_SUB,
    READOUT_SUB,
    CheckpointMismatch,
    RunSpec,
    Trainer,
    leaf_dir,
    load_readout,
    run_training,
    stack_targets,
    task_losses,
)
from tests._e3a_overfit import tiny_adapter

REPO = Path(__file__).resolve().parents[1]
FAMILIES = ("qwen3.5-0.8b", "gemma-4-e2b-it", "internvl3.5-2b-hf")
CFG = load_config()


def _with_mix(cfg: E3aConfig, weights: dict) -> E3aConfig:
    doc = copy.deepcopy(dict(cfg.doc))
    doc["mix"] = {**doc["mix"], "weights": weights}
    return E3aConfig(path=cfg.path, sha256=cfg.sha256, doc=doc)


@pytest.fixture(scope="module")
def pools():
    paths = D.E3aPaths.fixtures()
    return {t: D.load_pool(t, paths, CFG).records for t in CFG.tasks}


def _load_script(name: str):
    saved = list(sys.path)
    spec = importlib.util.spec_from_file_location(f"_{name}_under_test", REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.path[:] = saved
    return mod


# ---------------------------------------------------------------------------
# 日程
# ---------------------------------------------------------------------------


def test_schedule_matches_e3b_on_refseg_depth(pools):
    """只给指代分割与深度时，每个 k 的任务、样本 id、epoch 与 E3b（同一种子、同一批大小）逐个相同。"""
    e3b = E3C.load_config()
    e3b_pools = {t: E3D.load_pool(t, E3D.E3Paths.fixtures(), e3b.task(t)).records for t in e3b.tasks}
    assert [s.id for s in e3b_pools["refseg"]] == [s.id for s in pools["refseg"]]
    assert [s.id for s in e3b_pools["depth"]] == [s.id for s in pools["depth"]]
    cfg2 = _with_mix(CFG, {"refseg": 1, "depth": 1, "semseg": 0, "normals": 0})
    for seed in (0, 1):
        a = D.Schedule({t: pools[t] for t in ("refseg", "depth")}, cfg2, seed=seed, batch_size=16)
        b = E3D.Schedule(e3b_pools, e3b, seed=seed, supervision="grid", batch_size=16)
        for k in range(160):
            pa, pb = a.plan(k), b.plan(k)
            assert (pa.task, pa.sample_id, pa.epoch, pa.step) == (pb.task, pb.sample_id, pb.epoch, pb.step), k
    # 四个任务等权：循环 refseg、depth、semseg、normals
    s4 = D.Schedule(pools, CFG, seed=0, batch_size=4)
    assert s4.cycle == ("refseg", "depth", "semseg", "normals")
    assert [p.task for p in s4.step_plans(3)] == ["refseg", "depth", "semseg", "normals"]
    assert D.mix_cycle({"refseg": 2, "depth": 1, "semseg": 0, "normals": 1}).count("refseg") == 2


def test_train_script_tasks_flag():
    tr = _load_script("train_e3a")
    cfg = tr.with_tasks(CFG, ["refseg", "depth"])
    assert cfg.tasks == ("refseg", "depth") and cfg.sha256 == CFG.sha256
    assert resolve_run(cfg, variant="full", tiny=True)["mix"] == {"refseg": 1, "depth": 1, "semseg": 0, "normals": 0}
    assert tr.with_tasks(CFG, None) is CFG


# ---------------------------------------------------------------------------
# 三家族 × 三个变体：梯度只落在允许的参数上
# ---------------------------------------------------------------------------


def _trainer(key: str, variant: str, *, seed: int = 0) -> Trainer:
    ad = tiny_adapter(key)
    rc = resolve_run(CFG, variant=variant, tiny=True)
    tr = Trainer(ad, CFG, rc, variant=variant, seed=seed, tiny=True)
    tr.setup_fresh()
    return tr


def _examples(pools) -> list[D.Example]:
    b = D.ExampleBuilder(CFG, pools, out_size=128)
    s = D.Schedule(pools, CFG, seed=0, batch_size=4)
    return [b.build(p) for p in s.step_plans(0)]


def _backward(tr: Trainer, examples) -> torch.Tensor:
    """optimize 的前向 + 反向（不 step），返回总损失。"""
    total = None
    for group in tr.micro_batches(examples):
        feats = tr.features([tr.build_inputs(e) for e in group], semseg=group[0].task == "semseg")
        ro = tr.readout(*feats.args())
        for task in dict.fromkeys(e.task for e in group):
            rows = [b for b, e in enumerate(group) if e.task == task]
            idx = torch.tensor(rows)
            out = tr.readout.task_output(task, ro.embed[idx], None if ro.class_emb is None else ro.class_emb[idx])
            ls = task_losses(task, out, stack_targets(task, [group[b] for b in rows], "cpu"), CFG.task(task)["loss"]).sum()
            ls.backward(retain_graph=True)
            total = ls.detach() if total is None else total + ls.detach()
    return total


@pytest.mark.parametrize("key", FAMILIES)
@pytest.mark.parametrize("variant", ["full", "c1", "c2"])
def test_grads_and_frozen_params(key, variant, pools):
    tr = _trainer(key, variant)
    ad = tr.ad
    ex = _examples(pools)
    assert [e.task for e in ex] == ["refseg", "depth", "semseg", "normals"]
    assert tr.audit["pixel_path_check"]["checked_against_adapter_model"] is True
    assert tr.audit["readout_params"]["decoder_budget_total"] == tr.readout.params["decoder_budget_total"]
    ad.train_mode()
    loss = _backward(tr, ex)
    assert torch.isfinite(loss)
    lora = {n for n, _ in ad.model.named_parameters() if "lora_" in n}
    assert bool(lora) == (variant == "full")
    for n, p in ad.model.named_parameters():
        if p.grad is not None and torch.any(p.grad != 0):
            assert n in lora, f"{n} 不是 LoRA，却有梯度"
    if variant == "full":
        assert any(ad.model.get_parameter(n).grad is not None for n in lora)
    if variant == "c2":
        assert tr.query_embeds is None and not list(tr.readout.mix.parameters())
    else:
        assert tr.query_embeds.grad is not None and torch.any(tr.query_embeds.grad != 0)
        assert tr.readout.mix.logits.grad is not None
    assert all(p.grad is not None for p in tr.readout.heads.parameters())
    assert tr.readout.class_proj.proj.weight.grad is not None and torch.any(tr.readout.class_proj.proj.weight.grad != 0)
    tr.opt.zero_grad(set_to_none=True)
    for p in ad.model.parameters():
        p.grad = None

    # 两步正式优化：LLM 的 base 权重逐位不变；c1、c2 下整个 LLM 逐位不变
    snap = {n: p.detach().clone() for n, p in ad.model.named_parameters()}
    q0 = None if tr.query_embeds is None else tr.query_embeds.detach().clone()
    r0 = {n: p.detach().clone() for n, p in tr.readout.named_parameters()}
    for _ in range(2):
        rec, times, comps = tr.optimize(ex)
        assert np.isfinite(rec["loss"]) and rec["n_samples"] == 4 and len(comps) == 4
        assert rec["n_by_task"] == {"depth": 1, "normals": 1, "refseg": 1, "semseg": 1}
        assert (rec["mix"] is None) == (variant == "c2")
    changed_lora = 0
    for n, p in ad.model.named_parameters():
        if n in lora:
            changed_lora += int(not torch.equal(p, snap[n]))
        else:
            assert torch.equal(p, snap[n]), f"{n} 变了"
    assert (changed_lora > 0) == (variant == "full")
    if q0 is not None:
        assert not torch.equal(q0, tr.query_embeds.detach())
    assert sum(int(not torch.equal(p, r0[n])) for n, p in tr.readout.named_parameters()) > 0
    tr.check_base_unchanged()


def test_c1_llm_bitwise_unchanged_over_training(pools):
    """C1：冻结 LLM。多步训练（含梯度检查点）之后 LLM 的全部参数逐位不变（含视觉塔与嵌入），查询嵌入与读出在变。"""
    tr = _trainer("qwen3.5-0.8b", "c1")
    ad = tr.ad
    snap = {n: p.detach().clone() for n, p in ad.model.named_parameters()}
    assert not any(p.requires_grad for p in ad.model.parameters())
    assert ad.gradient_checkpointing
    ex = _examples(pools)
    ad.train_mode()
    q0 = tr.query_embeds.detach().clone()
    for _ in range(3):
        tr.optimize(ex)
    for n, p in ad.model.named_parameters():
        assert torch.equal(p, snap[n]), n
    assert not torch.equal(q0, tr.query_embeds.detach())
    assert ad.lora_state == "none"


# ---------------------------------------------------------------------------
# 没有像素通路（硬约束第 3 条）：梯度检查
# ---------------------------------------------------------------------------


def _pixel_grads(tr: Trainer, *, tap=None):
    """depth 的一条输入：像素张量求梯度，(读出输出对像素的梯度, 查询第 0 层对像素的梯度)。tap 作用在读出输入上（切断或混入）。"""
    ad = tr.ad
    img = (np.random.default_rng(0).random((96, 128, 3)) * 255).astype(np.uint8)
    inp = ad.build_inputs([img], CFG.instruction("depth"), "instruction_first")
    pv = inp.tensors["pixel_values"]
    pv.requires_grad_(True)
    feats = tr.features([inp], semseg=False)
    layer0 = feats.query[0].sum() if tr.variant != "c2" else None
    if tap is not None:
        feats = tap(feats, pv)
    out = tr.readout(*feats.args())
    y = tr.readout.task_output("depth", out.embed).sum() + tr.readout.task_output("normals", out.embed).sum()
    (g,) = torch.autograd.grad(y, pv, allow_unused=True, retain_graph=True)
    g0 = None
    if layer0 is not None and layer0.requires_grad:
        (g0,) = torch.autograd.grad(layer0, pv, allow_unused=True)
    return g, g0


@pytest.mark.parametrize("key", FAMILIES)
@pytest.mark.parametrize("variant", ["full", "c1"])
def test_no_pixel_path_gradient(key, variant):
    tr = _trainer(key, variant)
    tr.ad.eval_mode()
    # 1) 完整的通路：像素 → 视觉编码器 → LLM → 查询位置的隐藏状态 → 读出。梯度存在（LLM 路径是通的，检查有区分力）
    g, g0 = _pixel_grads(tr)
    assert g is not None and float(g.abs().sum()) > 0
    # 查询位置第 0 层（嵌入输出）就是查询嵌入：与像素无关
    assert g0 is None or float(g0.abs().sum()) == 0.0
    # 2) 切断读出输入（LLM 的隐藏状态）：读出输出与像素之间没有任何别的通路
    g_cut, _ = _pixel_grads(tr, tap=lambda f, pv: f.detached())
    assert g_cut is None
    # 3) 反例：读出输入里混进像素（模拟一条跳连），同一检查能发现
    def leak(f: ReadoutInputs, pv: torch.Tensor) -> ReadoutInputs:
        d = f.detached()
        return ReadoutInputs(query=d.query + pv.float().mean(), classes=d.classes, class_pool=d.class_pool,
                             prompt_lengths=d.prompt_lengths)

    g_leak, _ = _pixel_grads(tr, tap=leak)
    assert g_leak is not None and float(g_leak.abs().sum()) > 0


def test_c2_reads_vision_features_only():
    """C2 是对照：它的读出输入就是视觉编码器的输出（登记的来源如此），缩放到 16 × 16；没有查询 token、没有 LoRA。"""
    tr = _trainer("internvl3.5-2b-hf", "c2")
    assert tr.readout.sources == ("vision_features", "llm_input_embeddings:class_names")
    img = (np.random.default_rng(0).random((96, 128, 3)) * 255).astype(np.uint8)
    inp = tr.ad.build_inputs([img], CFG.instruction("depth"), "instruction_first")
    assert inp.query_span is None
    feats = tr.features([inp], semseg=False)
    vf = tr.ad.vision_features(inp)
    assert feats.query.shape == (1, 1, 256, tr.ad.hidden_size())
    assert inp.images[0].grid_hw == (16, 16) and torch.allclose(feats.query[0, 0], vf.float(), atol=0)  # InternVL 单 tile 原样
    emb, pool = tr.c2_class
    assert pool.shape[0] == 150 and emb.shape == (pool.shape[1], tr.ad.hidden_size())


# ---------------------------------------------------------------------------
# 续跑、检查点
# ---------------------------------------------------------------------------


def _spec(key: str, ad, variant: str, seed: int = 0) -> RunSpec:
    from src.adapters.checkpoint import policy_spec

    e = registry.get_model(key)
    return RunSpec(model_key=key, family=e.family, repo=e.repo, revision=e.revision, adapter_model_key=str(ad.model_key),
                   tiny=True, image_policy=policy_spec(ad.image_policy), model_dtype="fp32", variant=variant, seed=seed)


def _run(key: str, variant: str, out: Path, **kw):
    ad = tiny_adapter(key)
    rc = resolve_run(CFG, variant=variant, tiny=True)
    return run_training(ad, CFG, _spec(key, ad, variant), rc, D.E3aPaths.fixtures(), out, log=lambda _m: None, **kw)


def _tensors(d: Path) -> dict:
    from safetensors.torch import load_file

    return {str(p.relative_to(d)): load_file(str(p)) for p in sorted(d.rglob("*.safetensors"))}


@pytest.mark.parametrize("key,variant", [("qwen3.5-0.8b", "full"), ("gemma-4-e2b-it", "c1"), ("internvl3.5-2b-hf", "c2")])
def test_resume_is_bitwise_identical(key, variant, tmp_path):
    s1 = _run(key, variant, tmp_path / "a")
    assert s1["complete"]
    assert _run(key, variant, tmp_path / "b", stop_after_steps=1) is None
    assert _run(key, variant, tmp_path / "b", resume=True, stop_after_steps=2) is None
    s2 = _run(key, variant, tmp_path / "b", resume=True)
    assert s2["complete"] and s2["identity_sha256"] == s1["identity_sha256"]
    la, lb = leaf_dir(tmp_path / "a", variant, key, 0), leaf_dir(tmp_path / "b", variant, key, 0)
    assert (la / "train_log.jsonl").read_bytes() == (lb / "train_log.jsonl").read_bytes()
    assert (la / "config.yaml").read_bytes() == (lb / "config.yaml").read_bytes()
    ta, tb = _tensors(la / "ckpt" / FINAL), _tensors(lb / "ckpt" / FINAL)
    assert ta.keys() == tb.keys()
    for f in ta:
        assert ta[f].keys() == tb[f].keys()
        for k in ta[f]:
            assert torch.equal(ta[f][k], tb[f][k]), (f, k)
    assert len(list((lb / "logs").glob("run_*.json"))) == 3
    summ = json.loads((la / "train_summary.json").read_text())
    assert (summ["layer_mix_init"] is None) == (variant == "c2")
    if variant != "c2":
        assert summ["layer_mix_init"] == json.loads((lb / "train_summary.json").read_text())["layer_mix_init"]
        assert len(summ["layer_mix_final"]["weights"]) == len(summ["layers"])
    # 已完成：再跑直接跳过；目录非空又没有 --resume：拒绝覆盖
    assert _run(key, variant, tmp_path / "b", resume=True)["complete"]
    with pytest.raises(FileExistsError):
        _run(key, variant, tmp_path / "a")


@pytest.fixture(scope="module")
def full_ckpt(tmp_path_factory):
    out = tmp_path_factory.mktemp("e3a_full")
    s = _run("qwen3.5-0.8b", "full", out)
    return leaf_dir(out, "full", "qwen3.5-0.8b", 0) / s["final_checkpoint"]


def test_checkpoint_layout_and_roundtrip(full_ckpt, pools):
    ck = full_ckpt
    assert sorted(p.name for p in ck.iterdir()) == [MODEL_SUB, READOUT_SUB]
    man = read_manifest(ck / MODEL_SUB)
    assert man["lora"]["r"] == 32 and man["query"] == {"n": None, "position_mode": None}
    tok = json.loads((ck / MODEL_SUB / "tokens.json").read_text())
    assert tok["query"]["n"] is None and tok["query"]["injected"] is True and tok["n_grid_tokens"] == 0
    assert sorted(p.name for p in (ck / READOUT_SUB).iterdir()) == [
        "decoder.safetensors", "heads.safetensors", "layer_mix.safetensors", "query_embeds.safetensors", "readout.json"]
    # 载回：同一个检查点载两次，输出逐位相同；查询 token 只在评测侧重新登记之后出现
    ad, _ = factory.load_trained(ck / MODEL_SUB, device="cpu", merge=True)
    img = (np.random.default_rng(3).random((96, 128, 3)) * 255).astype(np.uint8)
    assert ad.build_inputs([img], "Describe.", "instruction_first").query_span is None  # T 文本评测：没有查询 token
    m1 = EV.EvalModel.from_checkpoint(ck, CFG, device="cpu")
    m2 = EV.EvalModel.from_checkpoint(ck, CFG, device="cpu")
    ex = _examples(pools)
    for e in ex:
        f1 = m1.features([m1.ad.build_inputs([e.image], e.instruction, e.order)], semseg=e.task == "semseg")
        f2 = m2.features([m2.ad.build_inputs([e.image], e.instruction, e.order)], semseg=e.task == "semseg")
        assert np.array_equal(m1.outputs(e.task, f1), m2.outputs(e.task, f2))



@pytest.mark.parametrize("key,variant", [("qwen3.5-0.8b", "full"), ("gemma-4-e2b-it", "full"), ("internvl3.5-2b-hf", "c1"),
                                         ("qwen3.5-0.8b", "c2")])
def test_saved_checkpoint_reproduces_in_memory_outputs(key, variant, pools, tmp_path):
    """训练两步后存检查点（model/ + readout/），评测侧载回（load_trained(merge=True) + 重新登记查询 + 读出）的输出与内存里的
    模型（LoRA 未合并）相同（fp32 下只差合并的舍入）。"""
    tr = _trainer(key, variant)
    ex = _examples(pools)
    tr.ad.train_mode()
    for _ in range(2):
        tr.optimize(ex)
    tr.ad.eval_mode()
    tr.readout.eval()
    want = {}
    with torch.no_grad():
        for e in ex:
            f = tr.features([tr.build_inputs(e)], semseg=e.task == "semseg")
            ro = tr.readout(*f.args())
            want[e.plan.k] = tr.readout.task_output(e.task, ro.embed, ro.class_emb).numpy()
    ck = tr.save(tmp_path / "ck")
    m = EV.EvalModel.from_checkpoint(ck, CFG, device="cpu")
    assert m.variant == variant and (m.ad.lora_state == ("merged" if variant == "full" else "none"))
    for e in ex:
        got = m.outputs(e.task, m.features([m.ad.build_inputs([e.image], e.instruction, e.order)], semseg=e.task == "semseg"))
        err = float(np.abs(got - want[e.plan.k]).max())
        assert err <= 1e-3 * max(1.0, float(np.abs(want[e.plan.k]).max())), (e.task, err)  # 合并的舍入；实现错误是 O(1) 的差


def test_readout_must_pair_with_model(full_ckpt, tmp_path):
    """readout/ 只能配它自己那次保存的 model/（readout.json 记了 model/manifest.json 的 sha256）；文件被改动也拒绝。"""
    import shutil

    ck = full_ckpt
    m = EV.EvalModel.from_checkpoint(ck, CFG, device="cpu")
    other = tmp_path / MODEL_SUB
    shutil.copytree(ck / MODEL_SUB, other)
    mf = other / "manifest.json"
    mf.write_text(mf.read_text() + "\n")
    with pytest.raises(CheckpointMismatch, match="不是同一次保存"):
        load_readout(ck / READOUT_SUB, m.readout, model_dir=other)
    bad = tmp_path / READOUT_SUB
    shutil.copytree(ck / READOUT_SUB, bad)
    with open(bad / "heads.safetensors", "ab") as f:
        f.write(b"\0")
    with pytest.raises(CheckpointMismatch, match="sha256"):
        load_readout(bad, m.readout)


def test_t_text_eval_loads_e3a_checkpoint(full_ckpt, tmp_path):
    """任务书 §3：E3a 的检查点能被 run_e1.py --checkpoint 加载（只用合并后的 LoRA，不需要查询 token），合并核对通过、能生成。"""
    r1 = _load_script("run_e1")
    out = tmp_path / "t"
    code = r1.main(["--tiny", "--model", "qwen3.5-0.8b", "--task", "grid16", "--checkpoint", str(full_ckpt / MODEL_SUB),
                    "--checkpoint-label", "t_text/e3a/full/seed0", "--out", str(out), "--limit", "2"])
    assert code == 0
    base = out / "t_text" / "e3a" / "full" / "seed0" / "qwen3.5-0.8b"
    mc = json.loads((base / "merge_check.json").read_text())
    assert mc["pass"] is True and list(mc["checks"]) == ["merge"]
    m = json.loads((base / "grid16" / "metrics.json").read_text())
    assert m["n"] == 2 and m["complete"] is True
    import yaml

    cfg = yaml.safe_load((base / "grid16" / "config.yaml").read_text())
    assert cfg["checkpoint"]["n_grid_tokens"] == 0 and cfg["checkpoint"]["lora"]["r"] == 32
