"""DEV-0037 §1、§2：E5 的训练数据、像素 token 网格的格式、与零样本同一套评分（不加载模型）。

- 配置里钉住的数字：数据量档、每类条数、步数（固定 epoch）、seed_stride、评测集与批大小、起伏映射的深度档（档几何中心约 1.1–4.6 m）；
- 训练集：小档是大档的前缀（同一个数据种子下嵌套）、各类等量、档内均匀、不同数据种子不相交、a–d 子集与五类全训逐条相同、无状态；
- 训练集与 val、test 不相交：场景种子的区间、物体身份（2D 与 3D 的身份划分表）、样本 id；
- 日程：每个 epoch 每条样本恰好一次、各 epoch 的顺序不同、步数与最后一步；
- 网格 token：值 token 与允许集合、往返、解析失败的原因；提示（结尾句取自 E3b 配置）；
- 评分：同一个网格经零样本的文本路径与训练后的 token 路径，scoring.score 给出相同的分数；
- 标题口径：训练后系统的记录直接用 protocol.in_headline（与零样本同一个函数；b 的范围由 protocol 定，D73），这里不钉具体的档。
"""

from __future__ import annotations

import json
import math
import platform
from pathlib import Path

import numpy as np
import pytest

from src.adapters.base import GenerationResult
from src.constants import DEPTH_NUM_BINS, GRID_SIZE_E3B
from src.e3.targets import seq_len
from src.e3.tokens import GridTokens
from src.e5 import protocol as P
from src.e5 import scoring as SC
from src.e5 import token_grid as TG
from src.e5 import train_data as TD
from src.e5.prompts import load_prompt_config
from src.e5.train_config import E5ConfigError, load_train_config, per_task, resolve_run, steps_for
from src.grid.codec import RELIEF_NUM_LEVELS, bins_to_depth, grid_to_text, relief_grid_to_text
from src.synth import generate
from src.synth.constants import SCENE_SEED_STRIDE, SPLIT_CODES
from src.synth.relief import RELIEF_TARGET_KEY, baseline_relief_key
from src.synth.shapes import identity_split_table
from src.synth.solids import solid_identity_split_table

REPO = Path(__file__).resolve().parents[1]
CFG = load_train_config()


def _fake_tokens() -> GridTokens:
    names = CFG.e3b.vocab.names
    return GridTokens(vocab=CFG.e3b.vocab, ids={n: 1000 + i for i, n in enumerate(names)}, stop_id=7)


def _ts(n=1000, tasks=None, seed=0):
    return TD.TrainSet.build(CFG, n, tasks, seed)


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------


def test_config_pins():
    d = CFG.doc
    assert CFG.tiers(False) == (1000, 4000, 16000, 64000)  # 任务书的数据量档
    assert [per_task(n) for n in CFG.tiers(False)] == [200, 800, 3200, 12800]
    assert CFG.seed_stride == 2**32 and CFG.seed_stride * 2**16 <= SCENE_SEED_STRIDE
    assert CFG.seeds == (0, 1, 2)  # D72-4：3 个种子
    assert d["training"]["epochs"] == 2
    # epochs_basis 里的步数：1K 档 125 步、64K 档 8000 步（批 16）；D77-3（DEV-0038）：1K、4K 另有 1,000 步的下限，16K、64K 不变
    rcs = {n: resolve_run(CFG, n_train=n, n_examples=len(_ts(n)), tiny=False) for n in CFG.tiers(False)}
    assert {n: r["steps_by_epochs"] for n, r in rcs.items()} == {1000: 125, 4000: 500, 16000: 2000, 64000: 8000}
    assert {n: r["steps"] for n, r in rcs.items()} == {1000: 1000, 4000: 1000, 16000: 2000, 64000: 8000}
    rc = rcs[1000]
    assert rc["batch_size"] == 16 and rc["warmup_steps"] == math.ceil(0.03 * 1000)  # 学习率调度按实际步数排
    # LoRA 与优化器的结构同 E3b（configs/e3b_train.json；只换学习率等几项）
    e3 = CFG.e3b.doc
    assert rc["lora"] == {k: e3["lora"][k] for k in ("r", "alpha", "dropout")} and rc["lora"]["r"] == 32
    assert rc["optim"]["token_rows"]["mode"] == "fp32_master_rows" and rc["optim"]["betas"] == e3["optim"]["betas"]
    assert rc["allowed_scope"] == e3["allowed_scope"] and rc["enable_thinking"] is False
    assert "mix" not in rc and "correction_fraction" not in rc and rc["variant"] == "e5"
    assert steps_for(10, 1, 4) == 3  # 最后一步不满
    # 评测集与零样本相同（预注册草稿 §2：每档 100，b 200）；批大小同 E1 冻结的 64；max_new_tokens 1057
    assert d["eval"]["n_per_bin"] == {t: (200 if t == "viewpoint" else P.DEFAULT_N_PER_BIN_EVAL) for t in P.E5_TASKS}
    e1 = json.loads((REPO / "configs" / "e1_prompts.json").read_text())
    assert d["eval"]["batch_size"] == e1["batch_protocol"]["batch_size"] == load_prompt_config().batch_size() == 64
    assert TG.max_new_tokens(CFG.grid_n) == seq_len(GRID_SIZE_E3B) == 1057
    assert "headline_scope" not in d["eval"]  # 标题口径只在 src/e5/protocol 定义一处
    # 起点的规则
    assert {k: d["starts"][k]["grid_tokens"] for k in ("base_warmup", "e3b", "e3a", "base")} == \
        {"base_warmup": "from_checkpoint", "e3b": "from_checkpoint", "e3a": "add", "base": "add"}
    assert d["start_lora"]["mode"] == "merge_then_new_lora"


def test_config_rejects_bad(tmp_path):
    d = json.loads((REPO / "configs" / "e5_train.json").read_text())
    for mutate in (lambda x: x["data"].update(tiers=[1000, 999]), lambda x: x["data"].update(tiers=[4000, 1000]),
                   lambda x: x["training"]["lr"].pop("lora"), lambda x: x["starts"].pop("e3a"),
                   lambda x: x["format"].update(relief_depth_bins="stride2"), lambda x: x["eval"]["merge_check"].pop("batch_size"),
                   lambda x: x["format"].update(template="{instruction}")):
        y = json.loads(json.dumps(d))
        mutate(y)
        p = tmp_path / "c.json"
        p.write_text(json.dumps(y))
        with pytest.raises((E5ConfigError, ValueError, KeyError)):
            load_train_config(p)


def test_relief_mapping_pins():
    """起伏档 k → 深度 token 第 16 + k 档：64 档正中间连续的 32 档；档几何中心约 1.1–4.6 m（配置 format.relief_basis 的数）。"""
    assert TG.RELIEF_DEPTH_OFFSET == (DEPTH_NUM_BINS - RELIEF_NUM_LEVELS) // 2 == 16
    assert TG.RELIEF_DEPTH_BINS == tuple(range(16, 48))
    lo, hi = bins_to_depth(np.array([16, 47]))
    assert round(float(lo), 1) == 1.1 and round(float(hi), 1) == 4.6
    from src.synth.constants import ROT3D_DISTANCE_M, VIEW_DISTANCE_M

    assert ROT3D_DISTANCE_M == (1.8, 2.8) and VIEW_DISTANCE_M == (4.0, 5.0)  # relief_basis 里的场景深度


def test_merge_check_sample_numbers():
    """eval.merge_check_basis 里的数：每类 8 条 → 合计 40 条，二值 24 条、起伏 16 条，各不超过一批；val 划分。"""
    from src.e5.trained_eval import merge_check_items

    mc = CFG.doc["eval"]["merge_check"]
    items = merge_check_items(mc["n_per_task"], P.E5_TASKS)
    assert len(items) == 40 and all(it.split == "val" for it in items)
    n_rel = sum(1 for it in items if TG.task_format(it.task) == "relief")
    assert (len(items) - n_rel, n_rel) == (24, 16) and max(24, 16) <= mc["batch_size"]
    for t in P.E5_TASKS:
        bins = [it.bin for it in items if it.task == t]
        assert len(bins) == 8 and max(np.bincount(bins)) - min(np.bincount(bins, minlength=P.num_bins(t))) <= 1


def test_epoch_basis_numbers():
    """training.epochs_basis 里的数：四档合计约是最大档的 1.33 倍（固定步数是 4 倍）；1 个 epoch 时 1K 档只有 63 步。"""
    tiers = CFG.tiers(False)
    assert round(sum(tiers) / max(tiers), 2) == 1.33
    assert steps_for(1000, 1, 16) == 63


# ---------------------------------------------------------------------------
# 训练集
# ---------------------------------------------------------------------------


def test_nested_tiers_and_equal_tasks():
    small, big = _ts(1000), _ts(4000)
    assert len(small) == 1000 and len(big) == 4000 and small.tag == "n1000"
    for t in P.E5_TASKS:
        a = [small.item_of(t, j) for j in range(small.per_task)]
        b = [big.item_of(t, j) for j in range(big.per_task)]
        assert a == b[: len(a)]  # 小档是大档的前缀
        assert sum(1 for it in small.items() if it.task == t) == 200  # 各类等量
        counts = np.bincount([it.bin for it in a], minlength=P.num_bins(t))
        assert counts.max() - counts.min() <= 1  # 档内均匀
    assert len(set(small.items()) - set(big.items())) == 0
    # 同一条样本：生成器的纯函数（几条抽查摘要）
    for t in ("a", "e"):
        it_s, it_b = small.item_of(t, 3), big.item_of(t, 3)
        assert TD.make_train_sample(it_s).digest() == TD.make_train_sample(it_b).digest()


def test_data_seed_disjoint_and_stateless():
    a, b = _ts(4000, seed=0), _ts(4000, seed=1)
    ia, ib = {(it.task, it.index) for it in a.items()}, {(it.task, it.index) for it in b.items()}
    assert not ia & ib
    assert a.items() == _ts(4000, seed=0).items() and a.digest() == _ts(4000, seed=0).digest()
    assert a.digest() != b.digest()
    with pytest.raises(ValueError):
        _ts(1000, seed=2**16)


def test_task_subset_same_items():
    full, abcd = _ts(1000), _ts(1000, ["a", "b", "c", "d"])
    assert abcd.tag == "n1000_abcd" and len(abcd) == 800 and abcd.per_task == full.per_task
    assert abcd.items() == [it for it in full.items() if it.task != "imagery"]
    assert TD.canonical_tasks(["d", "a"]) == ("rotation3d", "physics")
    assert TD.tasks_tag(["all"]) == "" and TD.tasks_tag(["e"]) == "e"
    with pytest.raises(ValueError):
        TD.canonical_tasks(["a", "rotation3d"])


def _identities(s):
    objs = s.meta["scene"]["objects"]
    return [(o["family"], int(o["variant"]), int(o["appearance"])) for o in objs]


def test_train_disjoint_from_eval_splits():
    """训练集取 train 划分：场景种子在 [0, 2^48)，物体身份都在 train 的身份池；val、test 的样本反之（D23-2）。"""
    t2, t3 = identity_split_table(), solid_identity_split_table()
    ts = _ts(1000, seed=3)
    eval_ids = {it.id for t in P.E5_TASKS for it in P.eval_items(t, "eval", 100)}
    assert not eval_ids & {it.id for it in ts.items()}
    for t in P.E5_TASKS:
        table = t3 if t in ("rotation3d", "viewpoint") else t2
        for j in (0, 1, 2):
            s = TD.make_train_sample(ts.item_of(t, j))
            assert s.split == "train" and s.meta["split"] == "train"
            assert 0 <= s.meta["scene_seed"] < SCENE_SEED_STRIDE
            assert all(table[i] == "train" for i in _identities(s)), (t, _identities(s))
        for split in ("val", "test"):
            s = generate(t, split, 0)
            lo = SPLIT_CODES[split] * SCENE_SEED_STRIDE
            assert lo <= s.meta["scene_seed"] < lo + SCENE_SEED_STRIDE
            assert all(table[i] == split for i in _identities(s))


def test_schedule_epochs():
    ts = TD.TrainSet.build(CFG, 20, None, 0)
    sch = TD.Schedule(ts, seed=1, epochs=2, batch_size=6)
    assert sch.total == 40
    plans = [p for s in range(math.ceil(40 / 6)) for p in sch.step_plans(s)]
    assert len(plans) == 40 and len(sch.step_plans(6)) == 4  # 最后一步不满
    for ep in (0, 1):
        ids = [p.sample_id for p in plans if p.epoch == ep]
        assert sorted(ids) == sorted(it.id for it in ts.items())  # 每个 epoch 每条恰好一次
    assert [p.sample_id for p in plans[:20]] != [p.sample_id for p in plans[20:]]
    again = TD.Schedule(ts, seed=1, epochs=2, batch_size=6)
    assert [again.plan(k) for k in (0, 17, 39)] == [plans[0], plans[17], plans[39]]  # 无状态
    other = TD.Schedule(ts, seed=2, epochs=2, batch_size=6)
    assert [other.plan(k).sample_id for k in range(20)] != [p.sample_id for p in plans[:20]]
    with pytest.raises(IndexError):
        sch.plan(40)


@pytest.mark.skipif((platform.machine(), platform.system()) == ("x86_64", "Linux"), reason="规范平台上 d 可以不加 --tiny")
def test_physics_needs_canonical_platform():
    tok = TG.E5Tokens(_fake_tokens(), "task")
    with pytest.raises(RuntimeError, match="规范平台"):
        TD.ExampleBuilder(_ts(1000), tokens=tok, prompt=CFG.prompt_format(), n=CFG.grid_n, tiny=False)
    TD.ExampleBuilder(_ts(1000), tokens=tok, prompt=CFG.prompt_format(), n=CFG.grid_n, tiny=True)
    TD.ExampleBuilder(_ts(1000, ["a", "b", "c", "e"]), tokens=tok, prompt=CFG.prompt_format(), n=CFG.grid_n, tiny=False)


def test_example_build():
    tok = TG.E5Tokens(_fake_tokens(), "task")
    ts = _ts(1000)
    b = TD.ExampleBuilder(ts, tokens=tok, prompt=CFG.prompt_format(), n=CFG.grid_n, tiny=True)
    sch = TD.Schedule(ts, seed=0, epochs=1, batch_size=16)
    for p in sch.step_plans(0)[:6]:
        e = b.build(p)
        s = TD.make_train_sample(ts.item_of(p.task, p.j))
        assert len(e.target_ids) == 1057 and e.weights == [1] * 1057 and e.order == "instruction_first"
        assert e.instruction == f"{s.instruction} Output the 32x32 {'depth' if p.task in ('rotation3d', 'viewpoint') else 'mask'} grid."
        assert len(e.images) == (0 if p.task == "imagery" else 1)
        g, f = tok.parse(p.task, e.target_ids, 32)
        assert f is None and np.array_equal(g, TG.target_values(s))


# ---------------------------------------------------------------------------
# 网格 token 与提示
# ---------------------------------------------------------------------------


def test_token_mapping_and_allowed():
    gt = _fake_tokens()
    tok = TG.E5Tokens(gt, "task")
    m0, m1 = gt.value_ids("refseg")
    depth = gt.value_ids("depth")
    assert tok.value_ids("occlusion") == (m0, m1)
    assert tok.value_ids("rotation3d") == (m0,) + tuple(depth[16:48]) and len(tok.value_ids("viewpoint")) == 33
    assert tok.allowed_ids("imagery") == sorted({m0, m1, gt.row_id, gt.stop_id})
    assert tok.allowed_ids("rotation3d") == sorted({m0, *depth[16:48], gt.row_id, gt.stop_id})
    assert TG.E5Tokens(gt, "all").allowed_ids("physics") == sorted(set(gt.grid_ids) | {gt.stop_id})
    lay = tok.layout()
    assert lay["relief"]["depth_bins"] == [16, 47] and lay["binary"]["value_ids"] == [m0, m1]
    with pytest.raises(TG.TokenGridError):
        TG.E5Tokens(gt, "some")


def test_token_round_trip_and_failures():
    tok = TG.E5Tokens(_fake_tokens(), "task")
    rng = np.random.default_rng(0)
    for task, hi in (("rotation3d", 33), ("physics", 2)):
        for n in (8, 32):
            g = rng.integers(0, hi, (n, n))
            ids = tok.target_ids(task, g, n)
            assert len(ids) == seq_len(n)
            back, f = tok.parse(task, ids, n)
            assert f is None and np.array_equal(back, g)
    g = rng.integers(0, 33, (8, 8))
    ids = tok.target_ids("rotation3d", g, 8)
    assert tok.parse("rotation3d", ids[:-1], 8)[1] == "truncated"
    assert tok.parse("rotation3d", ids[:9] + ids[10:], 8)[1] == "wrong_length"
    bad = list(ids)
    bad[3] = tok.row_id
    assert tok.parse("rotation3d", bad, 8)[1] == "row_separator"
    bad = list(ids)
    bad[0] = tok.value_ids("physics")[1]  # 二值 token 1 不在起伏网格的值 token 里
    assert tok.parse("rotation3d", bad, 8)[1] == "foreign_token"
    assert set(TG.FAILURE_OF.values()) <= set(SC._FAILURES)  # 都是评分认识的失败原因
    p = tok.pred("rotation3d", bad, 8, 64, 64)
    assert not p.valid and p.failure == "foreign_token"


def test_prompt_format():
    pf = CFG.prompt_format()
    e3 = CFG.e3b.doc["tasks"]
    assert e3["refseg"]["grid_template"].endswith(" " + pf.endings["binary"])
    assert e3["depth"]["grid_instruction"].endswith(" " + pf.endings["relief"])
    s = generate("c", "train", 5)
    assert pf.render(s, 32) == s.instruction.strip() + " Output the 32x32 mask grid."
    assert pf.render(generate("a", "train", 5), 8).endswith("Output the 8x8 depth grid.")
    assert TG.ending_of("A. B {n}.") == "B {n}." and TG.sample_images(generate("e", "train", 0)) == []
    with pytest.raises(TG.TokenGridError):
        TG.ending_of("Segment {expression}.")


# ---------------------------------------------------------------------------
# 评分：与零样本同一套代码
# ---------------------------------------------------------------------------


def _text_result(text: str) -> GenerationResult:
    return GenerationResult(text=text, token_ids=(1,), n_generated_tokens=1, n_prompt_tokens=0, n_visual_tokens=0, wall_time_s=0.0,
                            raw_text=text, stopped=True, n_image_pixels=0)


@pytest.mark.parametrize("task", P.E5_TASKS)
def test_same_scores_as_zero_shot(task):
    """同一个网格：零样本的路径（文本 → scoring.parse_answer）与训练后的路径（网格 token → E5Tokens.pred）给出相同的 Pred 与得分。"""
    prompts = load_prompt_config()
    kind = prompts.kind_of("zeroshot", task)
    tok = TG.E5Tokens(_fake_tokens(), "task")
    rng = np.random.default_rng(1)
    for it in P.eval_items(task, "eval", 1)[:3]:
        s = generate(it.task, it.split, it.index, bin=it.bin)
        h, w = s.target_mask.shape
        grids = [TG.target_values(s)]
        if TG.task_format(task) == "relief":
            grids += [np.asarray(s.extras[baseline_relief_key(b)], np.int64) for b in ("copy", "copy_flat")]
            grids.append(rng.integers(0, 33, (32, 32)))
        else:
            grids += [np.asarray(s.baselines[b].grid, np.int64) for b in sorted(s.baselines)]
            grids.append(rng.integers(0, 2, (32, 32)))
        for g in grids:
            text = relief_grid_to_text(g) if kind == "relief_grid" else grid_to_text(g)
            zs = SC.parse_answer(kind, _text_result(text), h, w)
            tr = tok.pred(task, tok.target_ids(task, g, 32), 32, h, w)
            for a, b in ((zs.grid, tr.grid), (zs.mask, tr.mask), (zs.relief, tr.relief)):
                assert (a is None and b is None) or np.array_equal(a, b)
            relief = kind == "relief_grid"
            assert SC.score(s, zs, relief_expected=relief) == SC.score(s, tr, relief_expected=relief)
    if TG.task_format(task) == "relief":
        assert RELIEF_TARGET_KEY in s.extras


# ---------------------------------------------------------------------------
# 标题口径：与零样本同一个函数（protocol.in_headline；b 的范围由 protocol 定，这里不钉具体的档）
# ---------------------------------------------------------------------------


def test_trained_record_uses_protocol_headline():
    from src.e5.trained_eval import record

    tok = TG.E5Tokens(_fake_tokens(), "task")
    for task in P.E5_TASKS:
        for it in P.eval_items(task, "eval", 2):
            s = generate(it.task, it.split, it.index, bin=it.bin)
            ids = tuple(tok.target_ids(task, TG.target_values(s), 32))
            res = GenerationResult(text="", token_ids=ids, n_generated_tokens=len(ids), n_prompt_tokens=10, n_visual_tokens=0,
                                   wall_time_s=0.0, raw_text="", stopped=True, n_image_pixels=0)
            rec = record(s, "p", res, tok, 32)
            g = P.groups_of(task, s.meta)
            assert rec["headline"] == P.in_headline(task, it.bin, g) == SC.sample_fields(s)["headline"]
            assert rec["valid"] and rec["score"]["iou_grid"] == 1.0 and rec["truncated"] is False
