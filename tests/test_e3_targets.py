"""DEV-0025 §1、§2：网格目标与解析（src/e3/targets.py）、叠加反馈图（src/e3/feedback.py）、深度加噪（src/e3/depth_noise.py）、
C3 的文本目标（src/e3/c3.py）。不需要模型：GridTokens 用一份假的 id 布局。"""

from __future__ import annotations

import colorsys
import hashlib
import json

import numpy as np
import pytest

from src.constants import DEPTH_NUM_BINS, GRID_SIZE_E3B, NYU_EVAL_HW
from src.e3 import c3 as C3
from src.e3.config import load_config
from src.e3.depth_noise import OPS, corrupt_depth_grid
from src.e3.feedback import DEPTH_COLORMAP, depth_feedback_image, feedback_image, seg_feedback_image, upsample_cells
from src.e3.targets import (
    ROW_LEN,
    SEQ_LEN,
    GridIdsFailure,
    NoSupervision,
    depth_grid,
    depth_target,
    grid_to_ids,
    parse_grid_ids,
    parse_target,
    seg_target,
    seq_len,
    weights_for,
)
from src.e3.tokens import GridTokens
from src.executors.overlay import render_overlay
from src.executors.raster import polygons_to_mask
from src.grid.codec import bins_to_depth, depth_to_bins, grid_to_mask, mask_to_grid
from src.parsing.spatial import parse_polygons

N = GRID_SIZE_E3B


@pytest.fixture(scope="module")
def tokens() -> GridTokens:
    v = load_config().vocab
    ids = {n: 1000 + k for k, n in enumerate(v.names)}
    return GridTokens(vocab=v, ids=ids, stop_id=7)


# ---------------------------------------------------------------------------
# 序列格式
# ---------------------------------------------------------------------------


def test_sequence_length_constant():
    assert SEQ_LEN == N * (N + 1) + 1 == 1057 == seq_len(N) and ROW_LEN == N + 1


@pytest.mark.parametrize("task,n_values", [("refseg", 2), ("depth", DEPTH_NUM_BINS)])
def test_round_trip_grid_ids_grid(tokens, task, n_values):
    rng = np.random.default_rng(0)
    for _ in range(5):
        g = rng.integers(0, n_values, (N, N))
        ids = grid_to_ids(g, tokens.value_ids(task), tokens.row_id, tokens.stop_id)
        assert len(ids) == SEQ_LEN and ids[-1] == tokens.stop_id
        assert all(ids[r * ROW_LEN + N] == tokens.row_id for r in range(N))  # 每行末尾是行分隔
        assert ids[:N] == [tokens.value_ids(task)[v] for v in g[0]]  # 光栅顺序：第一行在最前、行内自左而右
        p = parse_target(task, ids, tokens)
        assert p.failure is None and np.array_equal(p.grid, g)
        # 再编码回去逐个相同
        assert grid_to_ids(p.grid, tokens.value_ids(task), tokens.row_id, tokens.stop_id) == ids


def test_parse_failures(tokens):
    g = np.zeros((N, N), dtype=np.int64)
    ids = grid_to_ids(g, tokens.value_ids("refseg"), tokens.row_id, tokens.stop_id)
    assert parse_target("refseg", ids[:-1], tokens).failure is GridIdsFailure.TRUNCATED
    assert parse_target("refseg", [], tokens).failure is GridIdsFailure.TRUNCATED
    assert parse_target("refseg", ids[:-2] + [tokens.stop_id], tokens).failure is GridIdsFailure.WRONG_LENGTH
    bad = list(ids)
    bad[N], bad[N + 1] = bad[N + 1], bad[N]  # 行分隔挪了一格
    assert parse_target("refseg", bad, tokens).failure is GridIdsFailure.ROW_SEPARATOR
    bad = list(ids)
    bad[5] = tokens.value_ids("depth")[3]  # 分割网格里出现深度 token
    assert parse_target("refseg", bad, tokens).failure is GridIdsFailure.FOREIGN_TOKEN
    bad = list(ids)
    bad[5] = tokens.stop_id
    assert parse_target("refseg", bad, tokens).failure is GridIdsFailure.FOREIGN_TOKEN
    # 小网格同一套规则（n 是参数）
    small = grid_to_ids(np.eye(4, dtype=np.int64), tokens.value_ids("refseg"), tokens.row_id, tokens.stop_id, n=4)
    assert len(small) == seq_len(4) == 21
    assert np.array_equal(parse_grid_ids(small, tokens.value_ids("refseg"), tokens.row_id, tokens.stop_id, n=4).grid, np.eye(4))


def test_seg_target_is_mask_to_grid(tokens):
    h, w = 96, 128
    m = polygons_to_mask([[10, 10, 90, 12, 70, 80, 15, 60]], h, w)
    t = seg_target(m, tokens)
    assert np.array_equal(t.grid, mask_to_grid(m, N)) and t.n_supervised == SEQ_LEN and len(t.ids) == SEQ_LEN
    assert t.weights == tuple([1] * SEQ_LEN)


def test_depth_grid_invalid_pixels_and_log_mean():
    h, w = NYU_EVAL_HW
    rng = np.random.default_rng(1)
    d = np.exp(rng.uniform(np.log(0.6), np.log(9.0), (h, w))).astype(np.float32)
    d[:60, :] = 0.0  # 上面 60 行无效（BTS 的 0）：480 / 32 = 15 行一格，前 4 行格子全无效
    d[90:105, 200:220] = np.nan  # 一格（第 6 行、第 10 列；格高 15、格宽 20）全无效
    d[150:165, 300:310] = 0.0  # 一格只剩一半有效：恰好 ≥ 0.5，仍可监督
    bins, sup, info = depth_grid(d, min_valid_fraction=0.5)
    assert not sup[:4].any() and sup[4:].sum() == (N - 4) * N - 1
    assert not sup[6, 10] and sup[10, 15]
    # 可监督格子的值 = 有效像素深度的几何平均再分档
    r0, r1, c0, c1 = 150, 165, 300, 320
    cell = d[r0:r1, c0:c1]
    v = cell[(cell > 0) & np.isfinite(cell)].astype(np.float64)
    assert bins[10, 15] == depth_to_bins(np.exp(np.log(v).mean()))
    # 不可监督的格子填最近的可监督格子（第 0..3 行 → 第 4 行同一列）
    assert np.array_equal(bins[0], bins[4]) and np.array_equal(bins[3], bins[4])
    assert info["n_supervised_cells"] == int(sup.sum()) and info["n_filled_cells"] == N * N - int(sup.sum())
    w_ = weights_for(sup)
    assert len(w_) == SEQ_LEN and w_[-1] == 1 and all(w_[r * ROW_LEN + N] == 1 for r in range(N))
    assert sum(w_) == int(sup.sum()) + N + 1
    with pytest.raises(NoSupervision):
        depth_grid(np.zeros((h, w), dtype=np.float32), min_valid_fraction=0.5)


def test_depth_target_round_trip(tokens):
    h, w = NYU_EVAL_HW
    yy = np.mgrid[0:h, 0:w][0].astype(np.float32)
    d = 0.7 + 8.0 * yy / h
    t = depth_target(d, tokens, min_valid_fraction=0.5)
    p = parse_target("depth", t.ids, tokens)
    assert np.array_equal(p.grid, t.grid) and t.n_supervised == SEQ_LEN
    rec = bins_to_depth(p.grid)  # 解码回米：每格的几何中心
    assert np.all(np.abs(np.log(rec) - np.log(d[7::15, 10::20][:N, :N])) < np.log(10 / 0.5) / DEPTH_NUM_BINS + 1e-6)


# ---------------------------------------------------------------------------
# 叠加反馈图（训练与评测共用的唯一入口）
# ---------------------------------------------------------------------------


def test_seg_feedback_is_render_overlay_of_grid_to_mask():
    rng = np.random.default_rng(2)
    img = rng.integers(0, 256, (96, 128, 3), dtype=np.uint8)
    g = (rng.random((N, N)) < 0.3).astype(np.uint8)
    before = img.copy()
    out = feedback_image("refseg", img, g)
    assert np.array_equal(out, render_overlay(img, [grid_to_mask(g, 96, 128)]))
    assert np.array_equal(out, seg_feedback_image(img, g))
    assert np.array_equal(img, before)  # 不改输入


def test_depth_feedback_formula_and_colormap():
    rng = np.random.default_rng(3)
    img = rng.integers(0, 256, (96, 128, 3), dtype=np.uint8)
    g = rng.integers(0, DEPTH_NUM_BINS, (N, N))
    out = feedback_image("depth", img, g)
    col = DEPTH_COLORMAP[upsample_cells(g, 96, 128)].astype(np.int64)
    assert np.array_equal(out, ((img.astype(np.int64) + col + 1) // 2).astype(np.uint8))  # alpha 1/2，.5 进位
    assert np.array_equal(out, depth_feedback_image(img, g))
    # 色表：HSV 色相 240° → 0° 的等距色带，与 colorsys 逐数相同；64 色两两不同，相邻档 L1 色差 ≥ 16
    ref = np.array([[int(np.floor(c * 255 + 0.5)) for c in colorsys.hsv_to_rgb(h / 360, 1, 1)]
                    for h in np.linspace(240, 0, DEPTH_NUM_BINS)])
    assert np.array_equal(DEPTH_COLORMAP, ref) and DEPTH_COLORMAP.shape == (DEPTH_NUM_BINS, 3)
    assert len({tuple(c) for c in DEPTH_COLORMAP}) == DEPTH_NUM_BINS
    assert int(np.abs(np.diff(DEPTH_COLORMAP.astype(np.int64), axis=0)).sum(1).min()) >= 16
    assert hashlib.sha256(DEPTH_COLORMAP.tobytes()).hexdigest() == \
        "d2df31377966688804a4d3397b51a45b3d5f212265a12ca87d940e05225044f9"  # 色表变了就要重新定稿（D14-3 同类项）
    with pytest.raises(ValueError):
        feedback_image("depth", img, g + DEPTH_NUM_BINS)
    with pytest.raises(ValueError):
        feedback_image("semseg", img, g)


# ---------------------------------------------------------------------------
# 深度加噪
# ---------------------------------------------------------------------------


def test_depth_noise_identity_determinism_and_range():
    p = load_config().doc["depth_noise"]
    rng = np.random.default_rng(4)
    g = rng.integers(0, DEPTH_NUM_BINS, (N, N))
    out0, rec0 = corrupt_depth_grid(g, 0.0, np.random.default_rng(9), p)
    assert np.array_equal(out0, g) and rec0["severity"] == 0.0
    a, ra = corrupt_depth_grid(g, 0.7, np.random.default_rng(9), p)
    b, rb = corrupt_depth_grid(g, 0.7, np.random.default_rng(9), p)
    assert np.array_equal(a, b) and json.dumps(ra) == json.dumps(rb)
    assert a.dtype == np.int64 and a.min() >= 0 and a.max() < DEPTH_NUM_BINS
    # 同一个种子下，不同严重度选同一组算子（随机量的抽取与 s 无关）
    for s in (0.2, 1.0):
        _, r = corrupt_depth_grid(g, s, np.random.default_rng(9), p)
        assert [o["op"] for o in r["ops"]] == [o["op"] for o in ra["ops"]]
    seen = set()
    for seed in range(40):
        _, r = corrupt_depth_grid(g, 1.0, np.random.default_rng(seed), p)
        seen.update(o["op"] for o in r["ops"])
    assert seen == set(OPS)
    with pytest.raises(ValueError):
        corrupt_depth_grid(g, 1.5, np.random.default_rng(0), p)


# ---------------------------------------------------------------------------
# C3 的文本目标
# ---------------------------------------------------------------------------


def test_norm1000_and_box_target_checks():
    assert C3.norm1000(64, 128) == 500 and C3.norm1000(-3, 128) == 0 and C3.norm1000(200, 128) == 1000
    assert not hasattr(C3, "box_text")  # DEV-0025 暂行的统一框格式已删除（DEV-0030 §3）
    seen = []

    def fake_format(boxes, *, width, height, labels):
        seen.append((boxes, width, height, labels))
        return "ok"

    assert C3.box_target(fake_format, [12, 30.5, 100, 95.9], 128, 96, "the dog") == "ok"
    assert seen == [([[12.0, 30.5, 100.0, 95.9]], 128.0, 96.0, ["the dog"])]
    with pytest.raises(ValueError):
        C3.box_target(fake_format, [5, 5, 1, 1], 10, 10, "x")


def test_box_template_is_e1_native_prompt():
    from src.e3.config import REPO_ROOT

    cfg = load_config()
    e1 = json.loads((REPO_ROOT / "configs" / "e1_prompts.json").read_text())
    for fam in ("qwen3.5", "gemma4", "internvl3.5"):
        t, src = C3.load_box_template(cfg.doc, fam, REPO_ROOT)
        entry = e1["tasks"]["refexp_box"]["families"][fam]
        assert t == entry["template"] and src["native"] is True and src["task"] == "refexp_box"
        assert C3.render_e1_template(t, "the red cup") == t.format(expression="the red cup").strip()
    assert "box_template" not in cfg.doc["c3"]  # 不另写提示


def test_polygon_text_matches_e1_parser():
    h, w = 96, 128
    poly = [10.0, 10.0, 90.0, 12.0, 70.0, 80.0, 15.0, 60.0]
    t = C3.polygon_text([poly], w, h, max_points=64)
    assert t.startswith("[[") and not t.startswith("[[[")
    p = parse_polygons(t, width=w, height=h)
    assert p.failure is None and len(p.value) == 1
    gt = polygons_to_mask([poly], h, w)
    pred = polygons_to_mask([np.asarray(p.value[0]).reshape(-1)], h, w)
    assert (gt & pred).sum() / (gt | pred).sum() > 0.95
    # 两个部件 → [[[x, y], …], …]；E1 的解析器同样认
    t2 = C3.polygon_text([poly, [100.0, 50.0, 120.0, 50.0, 110.0, 70.0]], w, h, max_points=64)
    assert t2.startswith("[[[") and len(parse_polygons(t2, width=w, height=h).value) == 2
    # 顶点过多时等步长抽稀到 ≤ max_points
    ang = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    circle = np.stack([64 + 40 * np.cos(ang), 48 + 30 * np.sin(ang)], 1).reshape(-1).tolist()
    t3 = C3.polygon_text([circle], w, h, max_points=64)
    pts = json.loads(t3)
    assert 3 <= len(pts) <= 64
    # 退化的部件（取整后不足 3 个点）丢掉；一个都不剩时 None（调用方退回框）
    assert C3.polygon_text([[1.0, 1.0, 1.01, 1.0, 1.0, 1.01]], 1000, 1000, max_points=64) is None


def test_depth_pairs_follow_gt_and_format():
    h, w = NYU_EVAL_HW
    yy = np.mgrid[0:h, 0:w][0].astype(np.float32)
    d = (0.7 + 8.0 * yy / h).astype(np.float32)
    pairs = C3.depth_pairs(d, key="x#e0", n_pairs=5, seed=0)
    assert len(pairs) == 5
    for p in pairs:
        assert (p["closer"] == "A") == (p["depth_A"] < p["depth_B"])
        assert p["closer"] == ("A" if p["A"][1] < p["B"][1] else "B")  # 深度随行号增加
    assert pairs == C3.depth_pairs(d, key="x#e0", n_pairs=5, seed=0)  # 确定
    assert pairs != C3.depth_pairs(d, key="x#e1", n_pairs=5, seed=0)  # 换 epoch 换点
    assert {p["closer"] for s in range(4) for p in C3.depth_pairs(d, key=f"k{s}", n_pairs=5, seed=s)} == {"A", "B"}
    c = load_config().doc["c3"]
    from src.data.constants import NYU_PAIRS_PER_IMAGE

    assert c["depth_pairs_per_image"] == NYU_PAIRS_PER_IMAGE
    prompt = C3.depth_pairs_prompt(c["depth_template"], c["depth_pair_line"], pairs, w, h)
    ans = C3.depth_pairs_answer(c["depth_answer_line"], pairs)
    ax, ay = C3.point1000(pairs[0]["A"], w, h)
    assert f"1. A [{ax}, {ay}], B " in prompt and prompt.count("\n") == 5
    assert ans.splitlines()[0] == f"1. {pairs[0]['closer']} is closer" and len(ans.splitlines()) == 5
    with pytest.raises(ValueError):
        C3.depth_pairs(d[:100], key="x", n_pairs=5, seed=0)


def test_polygon_template_is_e1_unified_prompt():
    from src.e3.config import REPO_ROOT

    cfg = load_config()
    e1 = json.loads((REPO_ROOT / "configs" / "e1_prompts.json").read_text())
    for fam in ("qwen3.5", "gemma4", "internvl3.5"):
        t, src = C3.load_polygon_template(cfg.doc, fam, REPO_ROOT)
        assert t == e1["tasks"]["refexp_poly"]["families"][fam]["template"] and src["provenance"] == "unified"
