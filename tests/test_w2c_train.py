"""DEV-0061：W2-c 的训练（src/w2/w2c_train.py、scripts/train_w2c.py、configs/w2c_train.json）。CPU fp32 小型随机 Qwen，测试造的假数据。

- 配置：LoRA、优化器、批、梯度检查点、分块损失、数据线程与 C3（configs/e3b_train.json）同名项相同；每步 8 + 8、1 : 1、2000 步、种子 0；
  w2_config.sha256 是占位时非 tiny 拒跑；结构核对（1 : 1 以外、每个微批混两种样本、挑选的顺序不是 E1 的 image_order……）。
- 数据（docs/w2_plan.md §5）：只读训练集（split ≠ train、source 不对拒绝）；与 W2 配置对得上（w2_config_sha256、spec_sha256、K、RLE 尺寸）；
  标签规则（iou 最高者里编号最小的，含并列）；提出失败的行只进提出的池；manifest 必须有（非 tiny）。
- 提出的样本与 C3 的框监督逐字相同（同一条 fixture 句子，src/e3/data.ExampleBuilder 的 box 分支），提示也就是 E1 评测时的提示原文；
  挑选的目标是编号的十进制文本 + 回合结束 token。
- 挑选图与 W2-a 评测时逐位相同：同一组候选在 src/w2/runner.run_candidate_setting（w2c，关思考）里交给 build_inputs 的图、提示、顺序，
  与训练样本逐个相同（含 1 号失败画空）。
- 日程无状态；tiny 训练：损失有限、梯度只在 LoRA 上；中途崩溃后续跑、分段运行（stop_after_steps）与一次跑完逐位相同；不读任何评测集；
  命令行彩排与用法错误。
"""

from __future__ import annotations

import copy
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from src.adapters import registry
from src.adapters.checkpoint import policy_spec
from src.adapters.factory import make_adapter
from src.adapters.image_policy import parse_policy
from src.e1.prompts import PROMPTS_PATH, load_prompt_config
from src.e3 import data as D
from src.e3.config import load_config
from src.e3.train import leaf_dir
from src.w2 import runner as R
from src.w2 import w2c_train as T
from src.w2.config import W2_CONFIG_PATH, load_w2_config
from tests import _w2c_fakes as FK

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "train_w2c.py"
QWEN = "qwen3.5-0.8b"
CFG = T.load_w2c_config()
E3 = load_config()
W2 = load_w2_config()
IMPL = R.load_impl(R.DEFAULT_CANDIDATES_MODULE, R.DEFAULT_RENDER_MODULE)


def _tiny():
    fam = registry.get_model(QWEN).family
    t = CFG.doc["tiny"]
    return make_adapter(QWEN, device="cpu", dtype=torch.float32, image_policy=parse_policy(t["image_policies"][fam]), tiny=True,
                        init_std=float(t["init_std"]))


def _spec(ad, seed=0):
    e = registry.get_model(QWEN)
    return T.W2CRunSpec(model_key=e.key, family=e.family, repo=e.repo, revision=e.revision, adapter_model_key=ad.model_key, tiny=True,
                        image_policy=policy_spec(ad.image_policy), model_dtype="fp32", seed=seed)


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    return FK.write_fake_w2c_data(tmp_path_factory.mktemp("w2c") / "w2c_data")


@pytest.fixture(scope="module")
def data(data_dir):
    return T.load_w2c_data(data_dir, w2=W2, sources=CFG.doc["data"]["sources"], data_root=FK.ROOT, require_manifest=True)


def _w2_info():
    return T.resolve_w2_config(CFG, tiny=True)[1]


def _train(ad, data, out, *, rc=None, **kw):
    rc = rc or T.resolve_w2c_run(CFG, tiny=True)
    return T.run_w2c_training(ad, CFG, _spec(ad), rc, data, w2=W2, w2_info=_w2_info(), impl=IMPL, data_root=FK.ROOT, out_dir=out,
                              log=lambda m: None, **kw)


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------


def test_config_matches_c3_and_task_book():
    d, e = CFG.doc, E3.doc
    assert {k: d["lora"][k] for k in ("r", "alpha", "dropout")} == {k: e["lora"][k] for k in ("r", "alpha", "dropout")}
    assert (d["lora"]["r"], d["lora"]["alpha"], d["optim"]["lora"]["lr"]) == (32, 64, 2e-4)  # 任务书：r 32、alpha 64、lr 2e-4
    strip = lambda o: {k: v for k, v in o.items() if k not in ("basis", "token_rows")}  # noqa: E731
    assert strip(d["optim"]) == strip(e["optim"])  # 学习率日程照 C3（warmup_ratio、余弦、min_lr_ratio、裁剪）
    assert {k: d["batch"][k] for k in ("micro_batch_size", "grad_accum_steps")} == \
        {k: e["batch"][k] for k in ("micro_batch_size", "grad_accum_steps")}
    assert {k: d["model"][k] for k in ("dtype", "gradient_checkpointing")} == {k: e["model"][k] for k in ("dtype", "gradient_checkpointing")}
    assert d["loss"]["chunk_size"] == e["c3"]["chunk_size"] and d["data"]["workers"] == e["data"]["workers"]
    assert d["propose"]["box_template_from"] == e["c3"]["box_template_from"]
    assert d["tiny"]["image_policies"] == e["tiny"]["image_policies"] and d["tiny"]["init_std"] == e["tiny"]["init_std"]
    assert d["enable_thinking"] is False and d["streams"]["per_step"] == {"propose": 8, "select": 8}
    assert d["streams"]["loss_weights"] == {"propose": 1, "select": 1} and d["steps"] == 2000 and d["seed"] == 0
    assert d["model"]["default_key"] == "qwen3.5-9b" and d["data"]["sources"] == ["refcoco", "refcoco+", "refcocog"]
    e1 = load_prompt_config(PROMPTS_PATH).order
    assert d["select"]["order"] == e1 and d["propose"]["order"] == e1 == "image_first"  # 训练 = W2-c 评测的 E1 协议（C3 是 instruction_first）
    assert e["tasks"]["refseg"]["order"] == "instruction_first"  # 与 C3 的唯一差别（完成总结上报待裁）
    assert d["w2_config"]["path"] == "configs/w2.json"
    sha = d["w2_config"]["sha256"]
    assert sha is None or (len(sha) == 64 and sha == load_w2_config(REPO / d["w2_config"]["path"]).sha256)
    rc = T.resolve_w2c_run(CFG, tiny=False)
    assert (rc["steps"], rc["batch_size"], rc["warmup_steps"], rc["ckpt_every"]) == (2000, 16, 60, 250)
    assert rc["streams"]["micro_batches"] == {"propose": 2, "select": 2} and rc["streams"]["order"] == ["propose", "select"]
    assert rc["lora"] == {"r": 32, "alpha": 64, "dropout": 0.05} and "token_rows" not in rc["optim"]
    rt = T.resolve_w2c_run(CFG, tiny=True)
    assert (rt["steps"], rt["batch_size"], rt["warmup_steps"]) == (4, 4, 1) and rt["streams"]["per_step"] == {"propose": 2, "select": 2}
    ro = T.resolve_w2c_run(CFG, tiny=False, steps=500)
    assert ro["steps"] == 500 and ro["warmup_steps"] == 15 and ro["steps_override"] == {"config": 2000, "used": 500}


@pytest.mark.parametrize("edit, msg", [
    (lambda d: d["streams"]["loss_weights"].update(select=2), "1 : 1"),
    (lambda d: d["streams"]["per_step"].update(propose=6, select=10), "整数倍"),
    (lambda d: d["streams"]["per_step"].update(propose=8, select=4), "每步条数"),
    (lambda d: d["select"].update(order="instruction_first"), "image_order"),
    (lambda d: d.update(enable_thinking=True), "关思考"),
    (lambda d: d["data"].update(split="val"), "只读训练集"),
    (lambda d: d["optim"].update(token_rows={"lr": 1}), "token_rows"),
    (lambda d: d["w2_config"].update(sha256="xyz"), "十六进制"),
])
def test_config_validation(tmp_path, edit, msg):
    d = copy.deepcopy(dict(CFG.doc))
    edit(d)
    p = tmp_path / "c.json"
    p.write_text(json.dumps(d, ensure_ascii=False))
    with pytest.raises(T.W2CConfigError, match=msg):
        T.load_w2c_config(p)


def test_w2_config_pinning(tmp_path):
    d = copy.deepcopy(dict(CFG.doc))
    d["w2_config"]["sha256"] = None
    (tmp_path / "c.json").write_text(json.dumps(d, ensure_ascii=False))
    c = T.load_w2c_config(tmp_path / "c.json")
    with pytest.raises(T.W2CConfigError, match="占位"):
        T.resolve_w2_config(c, tiny=False)
    w2, info = T.resolve_w2_config(c, tiny=True)  # tiny 不核对
    assert info["pinned_sha256"] is None and info["config_sha256"] == W2.sha256 and info["k"] == W2.k
    d["w2_config"]["sha256"] = W2.sha256  # 写死了，但 configs/w2.json 还没冻结
    (tmp_path / "c.json").write_text(json.dumps(d, ensure_ascii=False))
    with pytest.raises(T.W2CConfigError, match="没有冻结"):
        T.resolve_w2_config(T.load_w2c_config(tmp_path / "c.json"), tiny=False)
    wd = json.loads(W2_CONFIG_PATH.read_text())
    wd["frozen"] = True
    (tmp_path / "w2f.json").write_text(json.dumps(wd, ensure_ascii=False))
    d["w2_config"].update(path=str(tmp_path / "w2f.json"), sha256=load_w2_config(tmp_path / "w2f.json").sha256)
    (tmp_path / "c.json").write_text(json.dumps(d, ensure_ascii=False))
    w2, info = T.resolve_w2_config(T.load_w2c_config(tmp_path / "c.json"), tiny=False)  # 冻结且 sha256 对得上：放行
    assert w2.frozen and info["frozen"] is True
    d["w2_config"]["sha256"] = "0" * 64
    (tmp_path / "c.json").write_text(json.dumps(d, ensure_ascii=False))
    with pytest.raises(T.W2CConfigError, match="写死"):
        T.resolve_w2_config(T.load_w2c_config(tmp_path / "c.json"), tiny=False)
    with pytest.raises(T.W2CConfigError, match="只配 --tiny"):
        T.resolve_w2_config(CFG, tiny=False, override=tmp_path / "w2f.json")
    wd["select"]["system"] = "You pick."
    (tmp_path / "w2s.json").write_text(json.dumps(wd, ensure_ascii=False))
    with pytest.raises(T.W2CConfigError, match="system"):
        T.resolve_w2_config(CFG, tiny=True, override=tmp_path / "w2s.json")


# ---------------------------------------------------------------------------
# 数据
# ---------------------------------------------------------------------------


def test_data_pools_and_identity(data, data_dir):
    rows = data.rows
    assert len(rows) == len(FK.train_pool()) and data.propose == tuple(range(len(rows)))
    failed = [i for i, r in enumerate(rows) if r["candidates"] is None]
    assert failed == [1, 7] and all(r["label"] is None and r["proposal"]["failure"] for r in (rows[i] for i in failed))
    assert set(data.select) == set(range(len(rows))) - set(failed)
    assert rows[2]["candidates"][0]["ok"] is False  # 1 号失败的行照常进挑选的池
    ident = data.identity()
    assert ident["n_rows"] == len(rows) and ident["n_select"] == len(rows) - 2 and ident["n_proposal_failed"] == 2
    assert [s["file"] for s in ident["shards"]] == sorted(p.name for p in data_dir.glob("*.jsonl")) and ident["manifest_sha256"]
    assert sum(ident["label_counts"].values()) == ident["n_select"] and len(ident["label_counts"]) >= 3  # 标签分散在几个编号上
    assert set(ident["by_source"]) == {"refcoco", "refcoco+", "refcocog"}
    assert "data_dir" not in json.dumps(ident) and str(data_dir) not in json.dumps(ident)  # 运行身份不含路径


def test_label_rule_with_ties():
    c = lambda *ious: [{"iou": v} for v in ious]  # noqa: E731
    assert T.check_label(c(0.2, 0.5, 0.5), 2) and not T.check_label(c(0.2, 0.5, 0.5), 3)  # 并列取编号小的
    assert T.check_label(c(0.0, 0.0), 1) and not T.check_label(c(0.0, 0.0), 2)
    assert T.check_label(c(0.9, 0.1), 1) and not T.check_label(c(0.1, 0.9), 1)


def _bad_rows(tmp_path, mutate, name):
    rows = FK.fake_rows(W2, n=6)
    mutate(rows)
    return FK.write_rows(tmp_path / name, rows)


@pytest.mark.parametrize("name, mutate, msg", [
    ("split", lambda rs: rs[0].update(split="val"), "只读训练集"),
    ("source", lambda rs: rs[0].update(source="coco"), "不在"),
    ("w2sha", lambda rs: rs[0].update(w2_config_sha256="0" * 64), "w2_config_sha256"),
    ("spec", lambda rs: rs[0].update(spec_sha256="0" * 64), "spec_sha256"),
    ("label", lambda rs: rs[0].update(label=(rs[0]["label"] % len(rs[0]["candidates"])) + 1), "编号最小"),
    ("k", lambda rs: rs[0]["candidates"].pop(), "恰有 K"),
    ("fail_cands", lambda rs: rs[1].update(candidates=rs[0]["candidates"]), "⟺"),
    ("fail_label", lambda rs: rs[1].update(label=1), "label 也必须为 null"),
    ("img", lambda rs: rs[0].update(image="refcoco_train/images/missing.jpg"), "图像不在"),
    ("abs", lambda rs: rs[0].update(image="/etc/passwd"), "相对数据根"),
    ("dup", lambda rs: rs[3].update(id=rs[0]["id"]), "id 重复"),
    ("rle", lambda rs: rs[0]["candidates"][1]["rle"].update(size=[1, 1]), "rle 的 size"),
    ("index", lambda rs: rs[0]["candidates"][1].update(index=5), "index 必须是 2"),
])
def test_data_rejections(tmp_path, name, mutate, msg):
    d = _bad_rows(tmp_path, mutate, name)
    with pytest.raises(T.W2CDataError, match=msg):
        T.load_w2c_data(d, w2=W2, sources=CFG.doc["data"]["sources"], data_root=FK.ROOT, require_manifest=True)


def test_manifest_required_outside_tiny(tmp_path):
    d = FK.write_rows(tmp_path / "nm", FK.fake_rows(W2, n=4), manifest=False)
    with pytest.raises(T.W2CDataError, match="manifest"):
        T.load_w2c_data(d, w2=W2, sources=CFG.doc["data"]["sources"], data_root=FK.ROOT, require_manifest=True)
    got = T.load_w2c_data(d, w2=W2, sources=CFG.doc["data"]["sources"], data_root=FK.ROOT, require_manifest=False)
    assert got.manifest_sha256 is None and got.files() == sorted(d.glob("*.jsonl"))


# ---------------------------------------------------------------------------
# 日程
# ---------------------------------------------------------------------------


def test_schedule_is_stateless_and_covers_pools(data):
    s = T.W2CSchedule(data, seed=0, per_step={"propose": 8, "select": 8}, order=["propose", "select"])
    plans = [s.step_plans(t) for t in range(6)]
    assert plans == [T.W2CSchedule(data, seed=0, per_step={"propose": 8, "select": 8}, order=["propose", "select"]).step_plans(t)
                     for t in range(6)]
    for t, ps in enumerate(plans):
        assert [p.stream for p in ps] == ["propose"] * 8 + ["select"] * 8 and [p.k for p in ps] == list(range(16 * t, 16 * t + 16))
        assert all(p.step == t for p in ps)
    flat = [p for ps in plans for p in ps]
    prop = [p.row for p in flat if p.stream == "propose"]
    n = len(data.propose)
    assert sorted(prop[:n]) == list(range(n))  # 第一个 epoch 恰好每句一次（置换）
    sel = [p.row for p in flat if p.stream == "select"]
    assert set(sel) <= set(data.select) and sorted(sel[: len(data.select)]) == sorted(data.select)
    assert all(data.rows[p.row]["id"] == p.sample_id for p in flat)
    other = T.W2CSchedule(data, seed=1, per_step={"propose": 8, "select": 8}, order=["propose", "select"])
    assert [p.row for p in other.step_plans(0)] != [p.row for p in plans[0]]  # 种子决定顺序
    rev = T.W2CSchedule(data, seed=0, per_step={"propose": 8, "select": 8}, order=["select", "propose"])
    assert [p.stream for p in rev.step_plans(0)] == ["select"] * 8 + ["propose"] * 8
    assert {(p.stream, p.row, p.epoch) for p in rev.step_plans(2)} == {(p.stream, p.row, p.epoch) for p in plans[2]}


# ---------------------------------------------------------------------------
# 样本
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_ad():
    return _tiny()


def _builder(ad, data, root=FK.ROOT, impl=IMPL, w2=W2):
    from src.e3.tokens import family_stop_id

    return T.W2CBuilder(CFG, data, w2=w2, impl=impl, data_root=root, stop_id=family_stop_id(ad),
                        encode=lambda s: list(ad.tokenizer(s, add_special_tokens=False)["input_ids"]), family="qwen3.5",
                        format_boxes=ad.format_boxes)


def test_propose_example_matches_c3_box_supervision(tiny_ad, data):
    """同一条 fixture 句子：W2-c 的提出样本与 C3 的框样本（src/e3/data.ExampleBuilder，text_kind = box）逐字相同（图、提示、目标 token）；
    提示也就是 E1 refexp_box 评测时的提示原文。顺序是 E1 的 image_first（C3 是 instruction_first，上报待裁）。"""
    from src.e3.tokens import family_stop_id

    ad = tiny_ad
    b = _builder(ad, data)
    pool = D.load_pool("refseg", D.E3Paths.fixtures(), E3.task("refseg")).records
    c3 = D.ExampleBuilder(E3, {"refseg": pool}, seed=0, supervision="text", stop_id=family_stop_id(ad),
                          encode=lambda s: list(ad.tokenizer(s, add_special_tokens=False)["input_ids"]), family="qwen3.5",
                          format_boxes=ad.format_boxes)
    prompts = load_prompt_config(PROMPTS_PATH)
    sched = T.W2CSchedule(data, seed=0, per_step={"propose": 2, "select": 2}, order=["propose", "select"])
    by_id = {s.id: i for i, s in enumerate(pool)}
    checked = 0
    for t in range(4):
        for p in sched.step_plans(t):
            if p.stream != "propose":
                continue
            e = b.build(p)
            idx = by_id[p.sample_id]
            ref = c3.build(D.SlotPlan(k=0, step=0, task="refseg", j=0, index=idx, epoch=0, sample_id=p.sample_id, kind="text",
                                      text_kind="box"))
            assert ref.info["text_kind_used"] == "box"
            assert (e.instruction, e.target_text, e.target_ids, e.weights) == (ref.instruction, ref.target_text, ref.target_ids, ref.weights)
            assert len(e.images) == 1 and np.array_equal(e.images[0], ref.images[0])
            assert e.instruction == prompts.render("refexp_box", "qwen3.5", pool[idx]).text  # = E1 评测时的提示
            assert e.order == "image_first" and ref.order == "instruction_first"
            assert e.target_ids[-1] == family_stop_id(ad) and '"bbox_2d"' in e.target_text
            checked += 1
    assert checked == 8


def test_select_target_is_the_decimal_number(tiny_ad, data):
    from src.e3.tokens import family_stop_id

    ad = tiny_ad
    b = _builder(ad, data)
    enc = lambda s: list(ad.tokenizer(s, add_special_tokens=False)["input_ids"])  # noqa: E731
    seen = set()
    for i in data.select:
        row = data.rows[i]
        plan = T.W2CPlan(stream="select", k=0, step=0, j=0, index=0, row=i, epoch=0, sample_id=row["id"])
        e = b.build(plan)
        assert e.target_text == str(row["label"]) and e.target_ids == enc(str(row["label"])) + [family_stop_id(ad)]
        assert len(e.target_ids) == 2 and e.weights == [1, 1] and e.order == "image_first"
        assert e.instruction == W2.select_prompt(W2.k, row["expression"]) and len(e.images) == (1 if W2.layout == "grid" else W2.k + 1)
        seen.add(row["label"])
    assert len(seen) >= 3
    ids = T.number_token_ids(enc, 8)
    assert ids is not None and len(set(ids)) == 8 and T.number_token_ids(enc, 12) is None  # "10" 不是单 token


def test_select_rendering_is_bitwise_identical_to_w2a_eval(tmp_path, tiny_ad):
    """同一组候选：W2 评测（run_candidate_setting，setting = w2c，关思考的 build_inputs 路径）交给适配器的挑选图、提示、顺序，与训练样本
    逐个相同；1 号失败（ok = False）时两边都画空叠加。"""
    from src.e1.prompts import PROMPTS_PATH as PP
    from src.e1.runner import load_images
    from src.w2.data import tiny_paths, tiny_splits, w2_items
    from src.w2.loop import selection_candidates
    from tests import _w2_fakes as F

    d = json.loads(W2_CONFIG_PATH.read_text())
    d["candidates"]["generators"] = [{"type": "grabcut_default"}, {"type": "box_scale", "percent": 20}, {"type": "box_scale", "percent": -20},
                                     {"type": "box"}]
    (tmp_path / "w2.json").write_text(json.dumps(d, ensure_ascii=False))
    cfg = load_w2_config(tmp_path / "w2.json")
    doc = tiny_splits(cfg)
    paths = tiny_paths()
    samples = w2_items("refseg", "devtune800", paths, splits=doc)
    prompts = load_prompt_config(PP)
    out = tmp_path / "out"
    spec = R.W2RunSpec(model_key=QWEN, family="qwen3.5", repo="Qwen/x", revision="r" * 8, adapter_model_key="fake-select", task="refseg",
                       split="devtune800", setting="w2c", tiny=True, limit=None, image_policy={"tag": "checkpoint"})
    # 提出：E1 refexp_box 格式的记录（合法的框 = GT 框内缩 1 像素）
    from src.e1.runner import _common_record
    from types import SimpleNamespace

    prop_dir = out / QWEN / "refexp_box" / "propose_w2c" / "devtune800"
    prop_dir.mkdir(parents=True)
    recs = []
    for s in samples:
        im = load_images(s)[0]
        x0, y0, x1, y1 = s.gt["box"]
        res = SimpleNamespace(text="x", raw_text="x", thinking_unclosed=False, stopped=True, n_prompt_tokens=10, n_generated_tokens=5,
                              n_visual_tokens=4, n_image_pixels=4096, wall_time_s=0.0, batch_wall_time_s=0.0)
        r = _common_record(s, SimpleNamespace(text=prompts.render("refexp_box", "qwen3.5", s).text, extras={}), im.size, res, "text")
        r.update(failure=None, parsed={"box": [x0 + 1, y0 + 1, x1 - 1, y1]})
        recs.append(r)
    (prop_dir / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    (prop_dir / "config.yaml").write_text("x: 1\n")
    impl = R.load_impl("tests._w2_fakes", "src.w2.render")  # 假的候选生成器（快）+ 真的渲染
    cl, rows = R.ensure_candidates(spec, cfg, samples, out, mode="box", prop_dir=prop_dir, prop_recs=recs, impl=impl, chunk_size=2,
                                   splits_doc=doc, prompts=prompts, cand_setting="candidates_w2c")
    assert cl.parent.name == "candidates_w2c"
    fi = 1
    first = samples[fi].id
    c1, c2 = rows[first]["candidates"][0], rows[first]["candidates"][1]
    assert c2["mask"]["area"] > 0
    c1.update(ok=False, mask=copy.deepcopy(c2["mask"]))  # 1 号失败（生成器按 §1 给了非空的 mask）：评测按空 mask 画

    class Rec(F.ScriptedSelectAdapter):
        def __init__(self):
            super().__init__({}, default="2")
            self.built = []

        def build_inputs(self, images, instruction, order):
            self.built.append((list(images), instruction, order))
            return super().build_inputs(images, instruction, order)

    ad = Rec()
    R.run_candidate_setting(ad, spec, cfg, samples, paths, out, splits_doc=doc, mode="box", prop_dir=prop_dir, prop_recs=recs, cand_leaf=cl,
                            cand_rows=rows, impl=impl, prompts=prompts, batch_size=2, chunk_size=2, max_batch_tokens=None, resume=False,
                            checkpoint={"manifest_sha256": "c" * 64, "path": "/x"})
    assert ad.generate_calls and all(c["enable_thinking"] is False for c in ad.generate_calls)  # w2c 关思考
    built = {}
    for imgs, instr, order in ad.built:
        built[instr] = (imgs, order)
    root = Path(paths.root)
    w2c_rows = [FK.from_candidates_row(rows[s.id], s, cfg, root=root) for s in samples]
    from src.executors.raster import rle_to_mask

    assert w2c_rows[fi]["candidates"][0]["ok"] is False and not np.any(rle_to_mask(w2c_rows[fi]["candidates"][0]["rle"]))
    dd = T.W2CData(data_dir=tmp_path, rows=tuple(w2c_rows), propose=tuple(range(len(w2c_rows))), select=tuple(range(len(w2c_rows))),
                   shards=(), manifest_sha256=None, stats={})
    b = _builder(tiny_ad, dd, root=root, impl=impl, w2=cfg)
    n = 0
    for i, s in enumerate(samples):
        imgs, prompt = b.select_inputs(w2c_rows[i])
        ev_imgs, ev_order = built[prompt]
        assert ev_order == b.orders["select"] == prompts.order
        assert len(imgs) == len(ev_imgs) and all(np.array_equal(np.asarray(a), np.asarray(c)) and a.mode == c.mode
                                                 for a, c in zip(imgs, ev_imgs))
        # 与评测同一条函数链直接算一遍（1 号失败画空）
        ref = impl.render(load_images(s)[0], selection_candidates(rows[s.id], impl.Candidate), layout=cfg.layout, style=cfg.style)
        assert all(np.array_equal(np.asarray(a), np.asarray(c)) for a, c in zip(imgs, ref))
        n += 1
    assert n == len(samples) >= 4
    # 1 号失败那一条：与 1 号照常画的图不同（真的画了空）
    ok_row = copy.deepcopy(rows[first])
    ok_row["candidates"][0]["ok"] = True
    plain = impl.render(load_images(samples[fi])[0], selection_candidates(ok_row, impl.Candidate), layout=cfg.layout, style=cfg.style)
    assert not all(np.array_equal(np.asarray(a), np.asarray(c)) for a, c in zip(b.select_inputs(w2c_rows[fi])[0], plain))


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------


def _tensors(d: Path) -> dict:
    from safetensors.torch import load_file

    out = {}
    for f in sorted(d.rglob("*.safetensors")):
        for k, v in load_file(str(f)).items():
            out[f"{f.relative_to(d)}:{k}"] = v
    return out


def test_one_step_grads_only_on_lora(data):
    ad = _tiny()
    rc = T.resolve_w2c_run(CFG, tiny=True)
    tr = T.W2CTrainer(ad, CFG, rc, seed=0, number_ids=None)
    tr.setup_fresh()
    b = _builder(ad, data)
    sched = T.W2CSchedule(data, seed=0, per_step=rc["streams"]["per_step"], order=rc["streams"]["order"])
    ex = [b.build(p) for p in sched.step_plans(0)]
    frozen = {n: p.detach().clone() for n, p in ad.model.named_parameters() if "lora_" not in n}
    lora0 = {n: p.detach().clone() for n, p in ad.model.named_parameters() if "lora_" in n}
    rec, times, comps = tr.optimize(ex)
    assert math.isfinite(rec["loss"]) and set(rec["loss_by_stream"]) == {"propose", "select"}
    assert rec["loss"] == pytest.approx(rec["loss_by_stream"]["propose"] + rec["loss_by_stream"]["select"])  # 1 : 1
    assert rec["n_by_stream"] == {"propose": 2, "select": 2} and rec["n_supervised_by_stream"]["select"] == 4  # 编号 + 回合结束
    assert rec["select_number_acc"] is None and len(comps) == 4
    assert all(torch.equal(p, frozen[n]) for n, p in ad.model.named_parameters() if "lora_" not in n)  # 只有 LoRA 动
    assert any(not torch.equal(p, lora0[n]) for n, p in ad.model.named_parameters() if "lora_" in n)
    assert ad.grid_token_ids == () or not ad.grid_token_ids
    assert tr.audit["token_row_ids"] is None and tr.audit["lora_dtypes"] == ["torch.float32"]


def test_resume_after_crash_and_segments_are_bitwise_identical(data, tmp_path):
    a, b, c = tmp_path / "A", tmp_path / "B", tmp_path / "C"
    sa = _train(_tiny(), data, a)
    assert sa["complete"] and sa["steps"] == 4 and sa["trainable_token_rows"] is None and sa["lora"]["r"] == 32
    with pytest.raises(RuntimeError, match="测试注入的中断"):
        _train(_tiny(), data, b, _fail_after_step=3)
    lb = leaf_dir(b, "w2c", QWEN, 0)
    assert (lb / "ckpt" / "latest" / "resume_state.pt").exists()
    with pytest.raises(FileExistsError):
        _train(_tiny(), data, b)
    sb = _train(_tiny(), data, b, resume=True)
    assert _train(_tiny(), data, c, stop_after_steps=1) is None  # 分段：1 步 + 1 步 + 2 步
    assert _train(_tiny(), data, c, stop_after_steps=1, resume=True) is None
    sc = _train(_tiny(), data, c, resume=True)
    la, lc = leaf_dir(a, "w2c", QWEN, 0), leaf_dir(c, "w2c", QWEN, 0)
    for other in (lb, lc):
        for f in ("train_log.jsonl", "config.yaml", "README.md", "train_summary.json"):
            assert (la / f).read_bytes() == (other / f).read_bytes(), f
        ta, tb = _tensors(la / "ckpt" / "final" / "model"), _tensors(other / "ckpt" / "final" / "model")
        assert set(ta) == set(tb) and all(torch.equal(ta[k], tb[k]) for k in ta)
        assert not (other / "ckpt" / "latest").exists()
    assert sa == sb == sc
    log = [json.loads(x) for x in (la / "train_log.jsonl").read_text().splitlines()]
    assert [r["step"] for r in log] == [0, 1, 2, 3] and all(math.isfinite(r["loss"]) for r in log)
    assert all(r["n_by_stream"] == {"propose": 2, "select": 2} and r["select_number_acc"] is not None for r in log)
    acc = json.loads((lc / "accounting.json").read_text())
    assert acc["n_runs"] == 3 and acc["n_steps_logged"] == 4 and acc["by_stream"]["select"]["n_samples"] == 8
    comp = json.loads((la / "compute.json").read_text())
    assert comp["n_samples"] == 16 and comp["generated_tokens"] == sum(r["n_target_tokens"] for r in log)
    man = json.loads((la / "ckpt" / "final" / "model" / "manifest.json").read_text())
    assert man["format"] == "visint-e3-checkpoint/1" and man["lora"]["r"] == 32 and man["trainable_token_rows"] is None
    tok = json.loads((la / "ckpt" / "final" / "model" / "tokens.json").read_text())
    assert not tok.get("grid_tokens") and not tok.get("row_ids")  # 与 C3 同格式：只有 LoRA，没有新增 token
    import yaml

    ident = yaml.safe_load((la / "config.yaml").read_text())
    assert ident["variant"] == "w2c" and ident["w2_config"]["config_sha256"] == W2.sha256 and ident["data"]["n_select"] == 37
    assert ident["propose"]["order"] == "image_first" and ident["select"]["target"].startswith("str(label)")
    assert _train(_tiny(), data, b, resume=True) == sb  # 已完成：直接返回
    # 运行身份变了（换了数据）就拒绝续跑
    d2 = FK.write_rows(tmp_path / "d2", FK.fake_rows(W2, n=20))
    data2 = T.load_w2c_data(d2, w2=W2, sources=CFG.doc["data"]["sources"], data_root=FK.ROOT, require_manifest=True)
    _train(_tiny(), data, tmp_path / "D", stop_after_steps=1)
    with pytest.raises(ValueError, match="运行身份"):
        _train(_tiny(), data2, tmp_path / "D", resume=True)


def test_training_reads_no_eval_set(data, tmp_path, monkeypatch):
    """只读训练集：训练期间碰任何评测集的加载器就失败。"""
    import src.data.loaders as L
    import src.e1.tasks as ET
    import src.w2.data as WD

    def boom(*a, **k):
        raise AssertionError("训练不该读评测集")

    for mod, name in ((L, "load_refcoco"), (ET, "select"), (WD, "w2_items"), (D, "load_pool")):
        monkeypatch.setattr(mod, name, boom)
    rc = T.resolve_w2c_run(CFG, tiny=True)
    rc["steps"] = 2
    s = _train(_tiny(), data, tmp_path / "o", rc=rc)
    assert s["complete"] and s["steps"] == 2


# ---------------------------------------------------------------------------
# 命令行
# ---------------------------------------------------------------------------


def run(args, tmp, *, ok=True):
    env = {**os.environ, "TMPDIR": str(tmp), "HF_HUB_OFFLINE": "1"}
    env.pop("VISINT_DATA_ROOT", None)
    p = subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True, cwd=REPO, env=env, timeout=1800)
    if ok:
        assert p.returncode == 0, p.stderr[-3000:]
    return p


def test_cli_tiny_segments_and_rules(data_dir, tmp_path):
    out = tmp_path / "t"
    base = ["--tiny", "--model", QWEN, "--w2c-data", str(data_dir), "--out", str(out)]
    p = run(base + ["--stop-after-steps", "3"], tmp_path)
    assert "停下" in p.stdout
    p = run(base + ["--resume"], tmp_path)
    assert "steps=4" in p.stdout and "by_stream" in p.stdout
    leaf = leaf_dir(out, "w2c", QWEN, 0)
    assert (leaf / "ckpt" / "final" / "model" / "manifest.json").exists() and len(list((leaf / "logs").glob("run_*.json"))) == 2
    bad = run(["--model", QWEN, "--w2c-data", str(data_dir), "--out", str(tmp_path / "norun"), "--data-root", "/x"], tmp_path, ok=False)
    assert bad.returncode == 2 and "运行编号" in bad.stderr
    bad = run(["--model", QWEN, "--w2c-data", str(data_dir), "--out", str(tmp_path / "0001_x"), "--data-root", "/x",
               "--w2-config", str(W2_CONFIG_PATH)], tmp_path, ok=False)
    assert bad.returncode == 2 and "只配 --tiny" in bad.stderr
    bad = run(["--model", QWEN, "--w2c-data", str(data_dir), "--out", str(tmp_path / "0001_x"), "--data-root", str(FK.ROOT)], tmp_path,
              ok=False)  # 非 tiny：w2_config.sha256 还是占位 → 拒跑（报错写进 errors/）
    if CFG.doc["w2_config"]["sha256"] is None:
        assert bad.returncode == 2 and "占位" in bad.stderr
        assert list((tmp_path / "0001_x" / "w2c").rglob("errors/error_*.json"))
    p = run(base[:-2] + ["--out", str(tmp_path / "t2"), "--steps", "2", "--seed", "1"], tmp_path)
    assert "steps=2" in p.stdout and leaf_dir(tmp_path / "t2", "w2c", QWEN, 1).exists()
