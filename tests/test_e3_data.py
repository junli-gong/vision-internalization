"""DEV-0025 §2：E3b 的训练样本（src/e3/data.py）。tests/fixtures/e3（RefCOCO 家族）与 tests/fixtures/e2（NYU 深度）的小数据。

- 训练池：三个 RefCOCO 数据集拼接、空指代句剔除并计数；没有去重剔除清单就报错；清单没比较过要求的评测集也报错；
- 样本日程无状态：同种子同结果，换种子换结果，C3 与 E3b 每一步的样本 id 逐个相同（匹配曝光量）；混合循环与任务内序号；
- 修正对：比例、可复现（图像逐字节相同）、换 epoch 换噪声、第二张图就是 feedback_image 的结果（与评测共用同一个函数）；
- C3：框 / 多边形按权重选，RLE 的样本退回框；文本目标以停止 token 结尾。
"""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from src.e3 import data as D
from src.e3 import feedback as FB
from src.e3.config import E3Config, load_config
from src.e3.targets import SEQ_LEN, parse_target, seg_grid
from src.e3.tokens import GridTokens
from src.noising.mask_noise import corrupt, sample_severity

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cfg() -> E3Config:
    return load_config()


@pytest.fixture(scope="module")
def pools(cfg):
    paths = D.E3Paths.fixtures()
    return {t: D.load_pool(t, paths, cfg.task(t)) for t in cfg.tasks}


@pytest.fixture(scope="module")
def tokens(cfg) -> GridTokens:
    return GridTokens(vocab=cfg.vocab, ids={n: 5000 + k for k, n in enumerate(cfg.vocab.names)}, stop_id=11)


def _recs(pools):
    return {t: p.records for t, p in pools.items()}


def _cfg_with(cfg: E3Config, **patch) -> E3Config:
    doc = copy.deepcopy(dict(cfg.doc))
    for path, v in patch.items():
        d = doc
        keys = path.split("__")
        for k in keys[:-1]:
            d = d[k]
        d[keys[-1]] = v
    return E3Config(path=cfg.path, sha256="patched", doc=doc)


def test_pools_from_fixtures(pools):
    rs, dp = pools["refseg"], pools["depth"]
    assert len(rs) == 39 and rs.stats["n_empty_expression"] == 1  # 13 个物体 × (2 + 1 + 1) 句 − 1 句空白
    assert sorted(rs.stats["datasets"]) == ["refcoco", "refcoco+", "refcocog"]
    assert {s.source for s in rs.records} == {"refcoco", "refcoco+", "refcocog"}
    assert len(dp) == 10 and all(s.task == "depth" for s in dp.records)
    assert rs.stats["dedupe_file"].endswith("dedupe/train_exclusions.json")


def test_missing_dedupe_list_is_an_error(tmp_path, cfg):
    root = tmp_path / "root"
    shutil.copytree(D.FIXTURE_REFSEG_ROOT, root)
    shutil.rmtree(root / "dedupe")
    with pytest.raises(FileNotFoundError, match="硬约束 7"):
        D.load_pool("refseg", D.E3Paths(refseg_root=root, depth_root=root), cfg.task("refseg"))
    # 清单在、但没有这个训练集的键：同样报错
    shutil.copytree(D.FIXTURE_REFSEG_ROOT / "dedupe", root / "dedupe")
    doc = json.loads((root / "dedupe" / "train_exclusions.json").read_text())
    doc["train_sets"], doc["sets"] = [], {}
    (root / "dedupe" / "train_exclusions.json").write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="没有这个训练集的键"):
        D.load_pool("refseg", D.E3Paths(refseg_root=root, depth_root=root), cfg.task("refseg"))


def test_check_dedupe_required_eval_sets(pools):
    f = pools["refseg"].stats["dedupe_file"]
    ok = D.check_dedupe("refseg", f, ["refcoco/eval"], real_model=True)
    assert ok["schema"] == 2 and "refcoco/eval" in ok["eval_sets"]
    with pytest.raises(ValueError, match="不在剔除清单"):
        D.check_dedupe("refseg", f, ["refcoco+/eval"], real_model=True)
    with pytest.raises(ValueError, match="硬约束 7"):
        D.check_dedupe("refseg", None, ["refcoco/eval"], real_model=True)
    assert D.check_dedupe("refseg", None, ["refcoco/eval"], real_model=False)["file_sha256"] is None


def test_mix_cycle_and_slot_task():
    assert D.mix_cycle({"refseg": 1, "depth": 1}) == ("refseg", "depth")
    assert D.mix_cycle({"refseg": 2, "depth": 1}) == ("refseg", "depth", "refseg")
    assert D.mix_cycle({"refseg": 1, "depth": 0}) == ("refseg",)
    with pytest.raises(ValueError):
        D.mix_cycle({"semseg": 1})
    cyc = D.mix_cycle({"refseg": 3, "depth": 2})
    seen: dict[str, list[int]] = {}
    for k in range(50):
        t, j = D.slot_task(k, cyc)
        seen.setdefault(t, []).append(j)
    assert seen["refseg"] == list(range(30)) and seen["depth"] == list(range(20))  # 任务内序号连续、不重不漏


def test_pool_position_is_epoch_permutation():
    n = 7
    got = [D.pool_position(j, n, seed=3, task="depth") for j in range(3 * n)]
    for ep in range(3):
        idx = [i for i, e in got[ep * n:(ep + 1) * n]]
        assert sorted(idx) == list(range(n)) and all(e == ep for _, e in got[ep * n:(ep + 1) * n])
    assert got != [D.pool_position(j, n, seed=4, task="depth") for j in range(3 * n)]


def test_schedule_stateless_and_matched_exposure(cfg, pools):
    a = D.Schedule(_recs(pools), cfg, seed=0, supervision="grid", batch_size=4)
    b = D.Schedule(_recs(pools), cfg, seed=0, supervision="grid", batch_size=4)
    c = D.Schedule(_recs(pools), cfg, seed=1, supervision="grid", batch_size=4)
    t = D.Schedule(_recs(pools), cfg, seed=0, supervision="text", batch_size=4)
    pa = [a.plan(k) for k in range(60)]
    assert pa == [b.plan(k) for k in range(60)]
    assert [p.sample_id for p in pa] != [c.plan(k).sample_id for k in range(60)]
    # 倒着算与顺着算相同（无状态）
    assert [a.plan(k) for k in reversed(range(60))] == list(reversed(pa))
    # C3 与 E3b：每一格的任务、样本 id、epoch 逐个相同，只有 kind 不同
    pt = [t.plan(k) for k in range(60)]
    assert [(p.task, p.sample_id, p.epoch) for p in pa] == [(p.task, p.sample_id, p.epoch) for p in pt]
    assert {p.kind for p in pt} == {"text"} and {p.kind for p in pa} == {"grid", "correction"}
    assert a.step_plans(2) == pa[8:12]


def test_correction_fraction(cfg, pools):
    s = D.Schedule(_recs(pools), cfg, seed=0, supervision="grid", batch_size=8)
    kinds = [s.plan(k).kind for k in range(2000)]
    frac = kinds.count("correction") / len(kinds)
    assert abs(frac - cfg.doc["correction"]["fraction"]) < 0.05
    for f, want in ((0.0, {"grid"}), (1.0, {"correction"})):
        s2 = D.Schedule(_recs(pools), _cfg_with(cfg, correction__fraction=f), seed=0, supervision="grid", batch_size=8)
        assert {s2.plan(k).kind for k in range(200)} == want


def _plan(sched: D.Schedule, task: str, kind: str, start: int = 0) -> D.SlotPlan:
    for k in range(start, start + 4000):
        p = sched.plan(k)
        if p.task == task and p.kind == kind:
            return p
    raise AssertionError(f"找不到 {task} {kind}")


@pytest.mark.parametrize("task", ["refseg", "depth"])
def test_correction_examples_reproducible_and_use_feedback_image(cfg, pools, tokens, task, monkeypatch):
    sched = D.Schedule(_recs(pools), cfg, seed=0, supervision="grid", batch_size=4)
    b = D.ExampleBuilder(cfg, _recs(pools), seed=0, supervision="grid", stop_id=tokens.stop_id, tokens=tokens)
    p = _plan(sched, task, "correction")
    e1, e2 = b.build(p), b.build(p)
    assert len(e1.images) == 2 and e1.target_ids == e2.target_ids and len(e1.target_ids) == SEQ_LEN
    assert all(np.array_equal(x, y) for x, y in zip(e1.images, e2.images))
    assert json.dumps(e1.info, sort_keys=True) == json.dumps(e2.info, sort_keys=True)
    assert "second image" in e1.instruction and "corrected complete" in e1.instruction
    # 目标是干净网格：解析回去就是 GT 网格（与从零预测的样本相同）
    g = _plan(sched, task, "grid")
    grid_target = parse_target(task, b.build(D.SlotPlan(**{**g.__dict__, "index": p.index, "sample_id": p.sample_id,
                                                           "epoch": p.epoch})).target_ids, tokens)
    assert np.array_equal(parse_target(task, e1.target_ids, tokens).grid, grid_target.grid)
    # 第二张图 = feedback_image(task, 原图, 带错的网格)：训练与评测共用同一个函数（模块里没有别的渲染入口）
    calls = []
    real = FB.feedback_image

    def spy(t, image, grid):
        calls.append((t, grid.copy()))
        return real(t, image, grid)

    monkeypatch.setattr(D, "feedback_image", spy)
    e3 = b.build(p)
    assert len(calls) == 1 and calls[0][0] == task
    assert np.array_equal(e3.images[1], real(task, e3.images[0], calls[0][1]))
    assert D.feedback_image is spy and real is FB.feedback_image
    # 换 epoch：同一条样本拿到不同的修正对
    p2 = D.SlotPlan(**{**p.__dict__, "epoch": p.epoch + 1})
    assert json.dumps(b.build(p2).info.get("noise"), sort_keys=True) != json.dumps(e1.info.get("noise"), sort_keys=True)


def test_refseg_correction_recipe(cfg, pools, tokens):
    """分割修正对的做法逐步复算：严重度按噪声日程、corrupt 在原图分辨率、再 mask_to_grid。"""
    sched = D.Schedule(_recs(pools), cfg, seed=0, supervision="grid", batch_size=4)
    b = D.ExampleBuilder(cfg, _recs(pools), seed=0, supervision="grid", stop_id=tokens.stop_id, tokens=tokens)
    p = _plan(sched, "refseg", "correction")
    e = b.build(p)
    s = pools["refseg"].records[p.index]
    mask = D.refseg_mask(s)
    sev = sample_severity(D.sample_rng(0, s.id, p.epoch, "severity"))
    noisy, rec = corrupt(mask, sev, D.sample_rng(0, s.id, p.epoch, "corrupt"), others=b._merge_others(s, mask.shape))
    assert e.info["severity"] == sev and json.dumps(e.info["noise"]) == json.dumps(rec)
    assert np.array_equal(e.images[1], FB.feedback_image("refseg", e.images[0], seg_grid(noisy)))
    assert len(b._merge_others(s, mask.shape)) >= 1  # 同一张图上有别的物体可合并


def test_c3_examples(cfg, pools):
    enc = lambda s: [ord(c) for c in s]  # noqa: E731 — 测试用的"分词器"
    t = D.Schedule(_recs(pools), cfg, seed=0, supervision="text", batch_size=4)
    b = D.ExampleBuilder(cfg, _recs(pools), seed=0, supervision="text", stop_id=11, encode=enc, family="qwen3.5")
    kinds = {}
    for k in range(80):
        p = t.plan(k)
        e = b.build(p)
        assert e.target_ids == enc(e.target_text) + [11] and e.n_supervised == len(e.target_ids) and len(e.images) == 1
        kinds.setdefault(p.task, set()).add(e.info.get("text_kind_used", p.text_kind))
        if e.info.get("text_kind_used") == "polygon":
            assert e.instruction == b.polygon_template.format(expression=pools["refseg"].records[p.index].fields["expression"].strip())
    assert kinds["refseg"] == {"box", "polygon"} and kinds["depth"] == {"pairs"}
    # 只要多边形：RLE 的物体退回框
    only_poly = _cfg_with(cfg, c3__seg_kinds={"polygon": 1, "box": 0})
    t2 = D.Schedule(_recs(pools), only_poly, seed=0, supervision="text", batch_size=4)
    b2 = D.ExampleBuilder(only_poly, _recs(pools), seed=0, supervision="text", stop_id=11, encode=enc, family="qwen3.5")
    rle_ids = {s.id for s in pools["refseg"].records if "rle" in s.gt["mask"]}
    assert rle_ids
    for k in range(0, 200, 2):
        p = t2.plan(k)
        assert p.text_kind == "polygon"
        e = b2.build(p)
        assert e.info["text_kind_used"] == ("box" if p.sample_id in rle_ids else "polygon")
