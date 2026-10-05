"""DEV-0034 §1、§5：E3a 读出结构的单元测试（src/readout/model.py、targets.py、losses.py、config.py）。不加载模型。

- 参数量：8 个登记模型的 hidden / 层数上，解码器（含混合与类别投影）≤ DECODER_MAX_PARAMS、每个线性头 ≤ HEAD_MAX_PARAMS；构造时超了就报错；
- 像素通路的结构检查：forward 的参数、子模块类型、钩子、未登记的引用、与适配器模型共享的参数，各造一个反例；
- 固定的 2× 上采样与 F.interpolate 相同；逐层混合的定义；
- GT 降到 128：最近邻取格中心、面积平均的权重、有效面积比例；128 → 原图（含小于 128 的边）；
- 四种损失与 E2 的实现在同一组输入上相同；配置里与 E2 / E3b / 简报共用的数字。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.constants import ADE_IGNORE_INDEX, ADE_NUM_CLASSES
from src.readout import losses as LS
from src.readout import targets as T
from src.readout.config import load_config, readout_spec, resolve_run
from src.readout.model import (
    DECODER_MAX_PARAMS,
    HEAD_MAX_PARAMS,
    LayerMix,
    ParamBudgetError,
    PixelPathError,
    Readout,
    ReadoutSpec,
    Upsample2x,
    check_no_pixel_path,
    count_params,
)

REPO = Path(__file__).resolve().parents[1]
CFG = load_config()
# 8 个登记模型的 (hidden, num_layers)：docs/model_notes.md §2 的表（LLM 部分）
MODEL_DIMS = {"qwen3.5-0.8b": (1024, 24), "qwen3.5-2b": (2048, 24), "qwen3.5-4b": (2560, 32), "qwen3.5-9b": (4096, 32),
              "gemma-4-e2b-it": (1536, 35), "gemma-4-e4b-it": (2560, 42), "internvl3.5-2b-hf": (2048, 28),
              "internvl3.5-4b-hf": (2560, 36)}


def _tiny_spec(variant: str = "full", hidden: int = 32, n_layers: int = 3) -> ReadoutSpec:
    return readout_spec(CFG.doc, variant=variant, hidden_size=hidden, n_layers=n_layers, tiny=True)


# ---------------------------------------------------------------------------
# 参数量
# ---------------------------------------------------------------------------


def test_constants_pin_brief():
    assert DECODER_MAX_PARAMS == 50_000_000 and HEAD_MAX_PARAMS == 1_000  # 简报 §4 第 3 条
    d = CFG.doc
    assert d["query"]["n"] == 256 and d["query"]["grid"] == 16  # 任务书 §1：16 × 16 = 256
    assert d["decoder"]["out_size"] == 128 and d["decoder"]["embed_dim"] == 64  # 128 × 128 × 64
    assert d["heads"]["channels"] == {"refseg": 1, "depth": 1, "normals": 3}
    assert d["query"]["position_mode"] == {"qwen3.5": "grid2d", "gemma4": "seq1d", "internvl3.5": "seq1d"}


@pytest.mark.parametrize("key", sorted(MODEL_DIMS))
@pytest.mark.parametrize("variant", ["full", "c1", "c2"])
def test_param_budget_real_models(key, variant):
    h, n_layers = MODEL_DIMS[key]
    p = count_params(readout_spec(CFG.doc, variant=variant, hidden_size=h, n_layers=n_layers, tiny=False))
    assert p["decoder_budget_total"] == p["mix"] + p["decoder"] + p["class_proj"] <= DECODER_MAX_PARAMS
    assert p["heads"] == {"refseg": 65, "depth": 65, "normals": 195}
    assert all(v <= HEAD_MAX_PARAMS for v in p["heads"].values())
    assert p["mix"] == (0 if variant == "c2" else n_layers + 2)  # softmax logits（层号 0..L）+ gamma
    assert p["class_proj"] == 66 * h + 64
    assert p["decoder_budget_total"] < 5_000_000  # 配置 basis 里写的量级（约 3.14M + 324·H）
    assert p["decoder_budget_total"] == 3_142_912 + 324 * h + p["mix"]  # 精确：常数部分（卷积、GN、输出层）+ (2 + 256 + 66)·H


def test_param_budget_asserted_at_construction():
    big = ReadoutSpec(variant="full", hidden_size=64, n_sources=2, grid=2, widths=(4096, 4096), res_blocks=(2, 0), embed_dim=64,
                      out_size=4)
    with pytest.raises(ParamBudgetError, match="50000000"):
        count_params(big)
    fat = ReadoutSpec(variant="full", hidden_size=64, n_sources=2, grid=2, widths=(16, 16), res_blocks=(0, 0), embed_dim=64,
                      out_size=4, head_channels={"refseg": 1, "depth": 1, "normals": 16})
    with pytest.raises(ParamBudgetError, match="线性头"):
        count_params(fat)


def test_spec_rules():
    with pytest.raises(ValueError, match="out_size"):
        ReadoutSpec(variant="full", hidden_size=8, n_sources=2, grid=16, widths=(8, 8), res_blocks=(0, 0), embed_dim=4, out_size=128)
    with pytest.raises(ValueError, match="c2"):
        ReadoutSpec(variant="c2", hidden_size=8, n_sources=3, grid=16, widths=(8, 8, 8, 8), res_blocks=(0, 0, 0, 0), embed_dim=4,
                    out_size=128)
    s = readout_spec(CFG.doc, variant="full", hidden_size=1024, n_layers=24, tiny=False)
    assert s.n_sources == 25 and s.widths == (256, 128, 64, 64) and s.out_size == 128
    assert readout_spec(CFG.doc, variant="c2", hidden_size=1024, n_layers=24, tiny=False).n_sources == 1
    t = readout_spec(CFG.doc, variant="full", hidden_size=64, n_layers=4, tiny=True)
    assert t.widths == tuple(CFG.doc["tiny"]["decoder"]["widths"]) and t.embed_dim == 64 and t.out_size == 128


def test_forward_shapes_and_heads():
    torch.manual_seed(0)
    r = Readout(_tiny_spec())
    q = torch.randn(4, 2, 256, 32)
    c = torch.randn(4, 2, 7, 32)
    pool = torch.zeros(2, 3, 7)
    pool[:, 0, 0:2] = 0.5
    pool[:, 1, 3] = 1.0
    pool[:, 2, 5:7] = 0.5
    out = r(q, c, pool)
    assert out.embed.shape == (2, 64, 128, 128) and out.class_emb.shape == (2, 3, 64)
    assert r.task_output("semseg", out.embed, out.class_emb).shape == (2, 3, 128, 128)
    for t, ch in (("refseg", 1), ("depth", 1), ("normals", 3)):
        assert r.task_output(t, out.embed).shape == (2, ch, 128, 128)
    # 开放词汇：类别 logits = 像素嵌入 · 类别嵌入
    lg = r.task_output("semseg", out.embed, out.class_emb)
    assert torch.allclose(lg[1, 2, 5, 7], (out.embed[1, :, 5, 7] * out.class_emb[1, 2]).sum(), atol=1e-5)
    with pytest.raises(ValueError):
        r(q, c, None)


# ---------------------------------------------------------------------------
# 像素通路的结构检查（硬约束第 3 条）
# ---------------------------------------------------------------------------


def test_pixel_path_check_passes_and_records_sources():
    for v in ("full", "c1", "c2"):
        r = Readout(_tiny_spec(v, n_layers=0 if v == "c2" else 3))
        s = check_no_pixel_path(r)
        assert s["forward_inputs"] == ["query_hidden", "class_hidden", "class_pool"]
        assert s["sources"] == (["vision_features", "llm_input_embeddings:class_names"] if v == "c2"
                                else ["llm_hidden:query", "llm_hidden:class_names"])


def test_pixel_path_check_catches_violations():
    # 1) forward 多一个像素参数
    class WithPixels(Readout):
        def forward(self, query_hidden, class_hidden=None, class_pool=None, pixel_values=None):  # noqa: D401
            return super().forward(query_hidden, class_hidden, class_pool)

    with pytest.raises(PixelPathError, match="forward 的参数"):
        WithPixels(_tiny_spec())
    # 2) 登记的来源不是 LLM 的隐藏状态
    r = Readout(_tiny_spec())
    r.sources = ("llm_hidden:query", "vision_features")
    with pytest.raises(PixelPathError, match="来源"):
        check_no_pixel_path(r)
    # 3) 挂进一个不允许的子模块（例如视觉塔里的卷积换成别的类型）
    r = Readout(_tiny_spec())
    r.decoder.extra = nn.Embedding(3, 4)
    with pytest.raises(PixelPathError, match="不允许的子模块"):
        check_no_pixel_path(r)
    # 4) 前向钩子（能把别的张量混进来）
    r = Readout(_tiny_spec())
    r.decoder.out.register_forward_hook(lambda m, a, o: o)
    with pytest.raises(PixelPathError, match="钩子"):
        check_no_pixel_path(r)
    # 5) 属性里藏一个适配器样的对象 / 张量

    class FakeAdapter:
        def vision_features(self, inputs):
            return None

    r = Readout(_tiny_spec())
    object.__setattr__(r.decoder, "side", FakeAdapter())
    with pytest.raises(PixelPathError, match="适配器样"):
        check_no_pixel_path(r)
    r = Readout(_tiny_spec())
    object.__setattr__(r.heads["depth"], "_cached_pixels", torch.zeros(3))
    with pytest.raises(PixelPathError, match="张量"):
        check_no_pixel_path(r)
    # 6) 与适配器的模型共享参数 / 子模块
    r = Readout(_tiny_spec())
    other = nn.Sequential(nn.Linear(2, 2))
    other.add_module("stolen", r.heads["refseg"])
    with pytest.raises(PixelPathError, match="共享"):
        check_no_pixel_path(r, adapter_model=other)
    assert check_no_pixel_path(Readout(_tiny_spec()), adapter_model=nn.Linear(2, 2))["checked_against_adapter_model"]


# ---------------------------------------------------------------------------
# 组件
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("hw", [(1, 1), (1, 5), (3, 4), (16, 16)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_upsample2x_matches_interpolate(hw, dtype):
    x = torch.randn(2, 3, *hw, dtype=dtype, requires_grad=True)
    y = Upsample2x()(x)
    ref = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
    tol = 1e-12 if dtype == torch.float64 else 1e-6
    assert y.shape == ref.shape and torch.allclose(y, ref, atol=tol, rtol=0)
    g = torch.randn_like(y)
    (gx,) = torch.autograd.grad((y * g).sum(), x)
    (gr,) = torch.autograd.grad((ref * g).sum(), x)
    assert torch.allclose(gx, gr, atol=tol * 10, rtol=0)


def test_layer_mix_definition():
    torch.manual_seed(0)
    m = LayerMix(3, learnable=True)
    with torch.no_grad():
        m.logits.copy_(torch.tensor([0.5, -1.0, 2.0]))
        m.gamma.fill_(1.7)
    h = torch.randn(3, 2, 5, 8) * torch.tensor([1.0, 10.0, 100.0])[:, None, None, None]
    w = torch.softmax(torch.tensor([0.5, -1.0, 2.0]), 0)
    want = 1.7 * sum(w[i] * F.layer_norm(h[i], (8,), eps=1e-6) for i in range(3))
    assert torch.allclose(m(h), want, atol=1e-5)
    snap = m.snapshot()
    assert abs(sum(snap["weights"]) - 1.0) < 1e-6 and abs(snap["gamma"] - 1.7) < 1e-6
    fixed = LayerMix(1, learnable=False)
    assert list(fixed.parameters()) == [] and torch.allclose(fixed(h[:1]), F.layer_norm(h[0], (8,), eps=1e-6), atol=1e-6)
    with pytest.raises(ValueError):
        LayerMix(2, learnable=False)


# ---------------------------------------------------------------------------
# GT 与分辨率
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_in", [9, 96, 128, 480, 640, 683])
def test_area_weights_and_nearest(n_in):
    a = T.area_weights(n_in, 128)
    assert a.shape == (128, n_in)
    assert np.allclose(a.sum(axis=1), n_in / 128) and np.allclose(a.sum(axis=0), 1.0)
    idx = T.nearest_indices(n_in, 128)
    centers = (np.arange(128) + 0.5) * n_in / 128
    assert np.array_equal(idx, np.minimum(n_in - 1, np.floor(centers).astype(int)))


def test_area_mean_exact_blocks_and_valid_fraction():
    rng = np.random.default_rng(0)
    blocks = rng.random((128, 128))
    v = np.kron(blocks, np.ones((4, 5)))  # 512 × 640，每个目标格正好 4 × 5 个源像素
    valid = np.ones(v.shape, bool)
    valid[:2, :] = False  # 第一行格子一半无效
    mean, frac = T.area_mean(v, valid, 128)
    assert np.allclose(mean, blocks) and np.allclose(frac[0], 0.5) and np.allclose(frac[1:], 1.0)
    v3 = np.stack([v, 2 * v, -v], -1)
    m3, _ = T.area_mean(v3, valid, 128)
    assert np.allclose(m3[..., 1], 2 * blocks)


def test_task_targets():
    lab = np.zeros((96, 128), np.uint8)
    lab[:, 64:] = 7
    t = T.semseg_target(lab, 128)
    assert t.values.dtype == np.int64 and set(np.unique(t.values)) == {ADE_IGNORE_INDEX, 6}
    with pytest.raises(T.NoSupervision):
        T.semseg_target(np.zeros((10, 10), np.uint8), 128)
    m = np.zeros((96, 128), bool)
    m[48:, :] = True
    r = T.refseg_target(m, 128)
    assert r.values.shape == (128, 128) and r.values[:64].sum() == 0 and r.values[64:].min() == 1.0
    d = np.full((480, 640), 2.0, np.float32)
    d[:, :8] = 0.0  # 无效
    dt = T.depth_target(d, 128, min_valid_fraction=0.5)
    assert np.allclose(dt.values[dt.valid], np.log(2.0)) and not dt.valid[:, 0].any() and dt.valid[:, 2:].all()
    n = np.zeros((480, 640, 3), np.float32)
    n[..., 2] = 1.0
    nv = np.ones((480, 640), bool)
    nt = T.normals_target(n, nv, 128, min_valid_fraction=0.5)
    assert nt.values.shape == (3, 128, 128) and np.allclose(nt.values[2], 1.0) and nt.valid.all()


@pytest.mark.parametrize("hw", [(96, 128), (128, 96), (64, 48), (480, 640)])
@pytest.mark.parametrize("method", ["guided", "bilinear"])
def test_to_original_any_size(hw, method):
    img = (np.random.default_rng(1).random((*hw, 3)) * 255).astype(np.uint8)
    f = np.full((128, 128), 3.0)
    up = T.to_original(f, img, method)
    assert up.shape == hw and np.allclose(up, 3.0, atol=1e-6)
    f3 = np.random.default_rng(2).random((128, 128, 3))
    assert T.to_original(f3, img, method).shape == (*hw, 3)


# ---------------------------------------------------------------------------
# 损失 = E2 的定义
# ---------------------------------------------------------------------------


def _e2(task: str):
    from src.e2.tasks import make_task

    e2 = json.loads((REPO / "configs" / "e2_probes.json").read_text())
    return make_task({**e2["tasks"][task], "task": task, "upsample": e2["upsample"]})


def test_losses_match_e2_and_config_pins():
    e2 = json.loads((REPO / "configs" / "e2_probes.json").read_text())
    for t in ("refseg", "depth", "semseg", "normals"):
        assert {k: v for k, v in CFG.task(t)["loss"].items()} == {k: v for k, v in e2["tasks"][t]["loss"].items() if k != "basis"}
    from src.e2.tasks import Target

    g = torch.Generator().manual_seed(0)
    r = 16
    # refseg
    out = torch.randn(1, 1, r, r, generator=g)
    mask = (torch.rand(r, r, generator=g) > 0.6).float()
    want = _e2("refseg").loss(out, Target((r, r), {"mask": mask}, {}, r * r))
    assert torch.allclose(LS.refseg_loss(out, mask[None], CFG.task("refseg")["loss"]), want, atol=1e-6)
    # depth
    out = torch.randn(1, 1, r, r, generator=g)
    log_gt = torch.randn(r, r, generator=g)
    valid = torch.rand(r, r, generator=g) > 0.3
    idx = valid.flatten().nonzero().flatten()
    want = _e2("depth").loss(out, Target((r, r), {"idx": idx, "log_gt": log_gt.flatten()[idx]}, {}, int(idx.numel())))
    assert torch.allclose(LS.depth_loss(out, log_gt[None], valid[None], CFG.task("depth")["loss"]), want, atol=1e-5)
    # semseg
    lg = torch.randn(1, ADE_NUM_CLASSES, r, r, generator=g)
    lab = torch.randint(0, ADE_NUM_CLASSES, (r, r), generator=g)
    lab[:3] = ADE_IGNORE_INDEX
    want = _e2("semseg").loss(lg, Target((r, r), {"label": lab}, {}, int((lab != ADE_IGNORE_INDEX).sum())))
    assert torch.allclose(LS.semseg_loss(lg, lab[None]), want, atol=1e-5)
    # normals
    out = torch.randn(1, 3, r, r, generator=g)
    gt = F.normalize(torch.randn(3, r, r, generator=g), dim=0)
    valid = torch.rand(r, r, generator=g) > 0.3
    idx = valid.flatten().nonzero().flatten()
    want = _e2("normals").loss(out, Target((r, r), {"idx": idx, "gt": gt.reshape(3, -1)[:, idx]}, {}, int(idx.numel())))
    assert torch.allclose(LS.normals_loss(out, gt[None], valid[None]), want, atol=1e-6)


# ---------------------------------------------------------------------------
# 配置：与 E3b、E2 共用的取值
# ---------------------------------------------------------------------------


def test_config_shared_values_with_e3b():
    e3b = json.loads((REPO / "configs" / "e3b_train.json").read_text())
    d = CFG.doc
    assert {k: d["lora"][k] for k in ("r", "alpha", "dropout")} == {k: e3b["lora"][k] for k in ("r", "alpha", "dropout")}
    assert d["batch"]["micro_batch_size"] == e3b["batch"]["micro_batch_size"]
    assert d["batch"]["grad_accum_steps"] == e3b["batch"]["grad_accum_steps"]
    assert d["steps"] == e3b["steps"] and d["seeds"] == e3b["seeds"]
    assert d["optim"]["groups"]["lora"] == e3b["optim"]["lora"] and d["optim"]["groups"]["query"] == {
        k: e3b["optim"]["token_rows"][k] for k in ("lr", "weight_decay")}
    assert d["optim"]["warmup_ratio"] == e3b["optim"]["warmup_ratio"] and d["optim"]["max_grad_norm"] == e3b["optim"]["max_grad_norm"]
    e2 = json.loads((REPO / "configs" / "e2_probes.json").read_text())
    assert d["optim"]["groups"]["mix"]["lr"] == e2["optim"]["lr"]["linear"]
    assert d["optim"]["groups"]["readout"] == {"lr": e2["optim"]["lr"]["conv"], "weight_decay": e2["optim"]["weight_decay"]}
    # 去重名单：E3b 的名单 + ade20k/val、nyu_normals/test，四个训练池同一份（升序）
    want = sorted(set(e3b["dedupe"]["required_eval_sets"]["refseg"]) | {"ade20k/val", "nyu_normals/test"})
    assert e3b["dedupe"]["required_eval_sets"]["refseg"] == e3b["dedupe"]["required_eval_sets"]["depth"]
    for t in ("refseg", "depth", "semseg", "normals"):
        assert d["dedupe"]["required_eval_sets"][t] == want
    # 剔除清单的 fixture 比较过这些评测集，四个训练集都有键
    fx = json.loads((REPO / "tests" / "fixtures" / "e3a" / "dedupe" / "train_exclusions.json").read_text())
    assert set(want) <= set(fx["eval_sets"]) and set(fx["sets"]) == set(fx["train_sets"]) == {
        "ade20k/train", "nyu_depth_train/train", "nyu_normals/train", "refcoco_train/train"}
    # 指代分割与 E3b 同一个训练池（三个数据集）、同一句的前半句
    assert d["tasks"]["refseg"]["datasets"] == e3b["tasks"]["refseg"]["datasets"]
    assert e3b["tasks"]["refseg"]["grid_template"].startswith(d["tasks"]["refseg"]["instruction_template"])
    assert e3b["tasks"]["depth"]["grid_instruction"].startswith(d["tasks"]["depth"]["instruction_template"])


def test_config_class_names_and_instructions():
    names = CFG.class_names
    assert len(names) == ADE_NUM_CLASSES == 150 and len(set(names)) == 150
    assert names[:5] == ["wall", "building", "sky", "floor", "tree"] and names[-1] == "flag" and names[12] == "person"
    ins = CFG.instruction("semseg")
    assert ins.startswith("Segment the image into these classes: wall, building, sky,") and ins.endswith(", clock, flag.")
    assert CFG.instruction("refseg", "  the man ") == "Segment the man in the image."
    assert CFG.instruction("depth") == "Estimate the depth of the image."
    with pytest.raises(ValueError):
        CFG.instruction("depth", "x")


def test_object_info_parse_and_check(tmp_path):
    from src.readout.data import check_class_names, parse_object_info

    lines = ["Idx\tRatio\tTrain\tVal\tName"] + [f"{i + 1}\t0.1\t1\t1\t{n}, synonym;x" for i, n in enumerate(CFG.class_names)]
    (tmp_path / "ade20k").mkdir()
    (tmp_path / "ade20k" / "objectInfo150.txt").write_text("\n".join(lines) + "\n")
    assert parse_object_info("\n".join(lines)) == CFG.class_names
    assert check_class_names(tmp_path, CFG.class_names)["checked"] is True
    bad = list(CFG.class_names)
    bad[3] = "flooring"
    with pytest.raises(ValueError, match="不一致"):
        check_class_names(tmp_path, bad)
    assert check_class_names(tmp_path / "nope", CFG.class_names) == {"checked": False, "file": None}


def test_resolve_run_groups_and_tiny():
    full = resolve_run(CFG, variant="full", tiny=False)
    assert list(full["optim"]["groups"]) == ["lora", "query", "mix", "readout"] and full["lora"]["r"] == 32
    assert full["batch_size"] == 16 and full["warmup_steps"] == math.ceil(0.03 * 4000) and full["gradient_checkpointing"]
    c1 = resolve_run(CFG, variant="c1", tiny=False)
    assert list(c1["optim"]["groups"]) == ["query", "mix", "readout"] and c1["lora"] is None and c1["gradient_checkpointing"]
    c2 = resolve_run(CFG, variant="c2", tiny=True)
    assert list(c2["optim"]["groups"]) == ["readout"] and c2["query"] is None and not c2["gradient_checkpointing"]
    assert c2["steps"] == CFG.doc["tiny"]["steps"] and c2["decoder"]["widths"] == CFG.doc["tiny"]["decoder"]["widths"]
