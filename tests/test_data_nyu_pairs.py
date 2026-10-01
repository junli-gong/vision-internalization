"""NYUv2 相对深度点对（src/data/nyu_pairs.py；DEV-0012 任务书 §3）：全部约束、A/B 各半、固定种子逐字节复现、渲染。

合成深度：480×640，几块深度不同的平面加一条无效带（深度 0）。真实数据上的点对由服务器在完整落地后生成
（scripts/data/build_e1_configs.py nyu-pairs），入库后由 test_committed_pairs_config 检查结构性约束。
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from src.constants import NYU_DEPTH_MAX, NYU_DEPTH_MIN, NYU_EIGEN_CROP, OVERLAY_PALETTE
from src.data import constants as C
from src.data import nyu_pairs as P

REPO = Path(__file__).resolve().parents[1]


def synth_depth(seed: int) -> np.ndarray:
    yy, xx = np.mgrid[:480, :640].astype(np.float64)
    d = np.where(xx < 200, 1.2 + yy / 300.0, np.where(xx < 420, 2.5 + seed * 0.1 + xx / 900.0, 4.0 + (yy + xx) / 400.0))
    d = d.astype(np.float32)
    d[:, 300:330] = 0.0  # 无效带
    d[100:120, :] = 12.0  # 超出 (0.001, 10) 的无效带
    return d


def synth_rgb(seed: int) -> np.ndarray:
    yy, xx = np.mgrid[:480, :640]
    return np.stack([(xx + seed * 40) % 256, (yy + seed * 20) % 256, (xx // 2 + yy // 2) % 256], -1).astype(np.uint8)


IMAGES = [(1, "test", synth_depth(1)), (2, "test", synth_depth(2)), (9, "test", synth_depth(3)), (4, "train", synth_depth(4)), (6, "train", synth_depth(5))]
RGBS = {idx: synth_rgb(k) for k, (idx, _, _) in enumerate(IMAGES)}


@pytest.fixture(scope="module")
def cfg():
    return P.build_pairs(IMAGES, render=RGBS)


def check_pair_constraints(p: dict, depth: np.ndarray | None) -> None:
    r0, r1, c0, c1 = NYU_EIGEN_CROP
    for k in ("A", "B"):
        x, y = p[k]
        assert r0 <= y < r1 and c0 <= x < c1
        d = p[f"depth_{k}"]
        assert NYU_DEPTH_MIN < d < NYU_DEPTH_MAX
        if depth is not None:
            assert d == float(depth[y, x])
            w = C.NYU_PAIR_LOCAL_WINDOW
            win = depth[y - w : y + w + 1, x - w : x + w + 1].astype(np.float64)
            assert r0 <= y - w and y + w < r1 and c0 <= x - w and x + w < c1  # 窗口不出 crop
            assert win.shape == (2 * w + 1, 2 * w + 1) and (win > NYU_DEPTH_MIN).all() and (win < NYU_DEPTH_MAX).all()
            lim = 1 + C.NYU_PAIR_NEIGHBOR_MAX_REL  # 相邻像素跳变判据：窗口内任意 4 邻接两点
            for a, b in ((win[:, :-1], win[:, 1:]), (win[:-1, :], win[1:, :])):
                assert (np.maximum(a, b) <= lim * np.minimum(a, b)).all()
    dx, dy = p["A"][0] - p["B"][0], p["A"][1] - p["B"][1]
    assert C.NYU_PAIR_MIN_DIST_DEN**2 * (dx * dx + dy * dy) >= C.NYU_PAIR_MIN_DIST_NUM**2 * 480**2
    hi, lo = max(p["depth_A"], p["depth_B"]), min(p["depth_A"], p["depth_B"])
    assert hi >= (1 + C.NYU_PAIR_MIN_REL_DEPTH_DIFF) * lo
    assert (p["closer"] == "A") == (p["depth_A"] < p["depth_B"])
    assert P.label_layout(p["A"], p["B"], 480, 640) is not None


def test_defaults_match_task_book():
    assert C.NYU_PAIRS_PER_IMAGE == 5
    assert (C.NYU_PAIR_MIN_DIST_NUM, C.NYU_PAIR_MIN_DIST_DEN) == (1, 10)
    assert C.NYU_PAIR_MIN_REL_DEPTH_DIFF == 0.15
    # 深度边缘保护打开（审查第 1 轮裁定）：窗口半径 = 标记环外半径
    assert C.NYU_PAIR_LOCAL_WINDOW == C.NYU_MARKER_OUTER_RADIUS == 9 and C.NYU_PAIR_NEIGHBOR_MAX_REL == 0.05
    assert C.NYU_PAIR_DEPTH_KEY == "depth_raw"


def test_all_constraints_hold(cfg):
    depth = {idx: d for idx, _, d in IMAGES}
    assert len(cfg["pairs"]) == 5 * len(IMAGES)
    for p in cfg["pairs"]:
        check_pair_constraints(p, depth[p["idx"]])
    for idx, _, _ in IMAGES:
        pts = [tuple(q[k]) for q in cfg["pairs"] if q["idx"] == idx for k in ("A", "B")]
        assert len(pts) == len(set(pts)) == 10  # 同一张图里的点不重复


def test_ab_balanced_per_split(cfg):
    for split in ("test", "train"):
        ps = [p for p in cfg["pairs"] if p["split"] == split]
        n_a = sum(p["closer"] == "A" for p in ps)
        assert n_a == len(ps) // 2 and len(ps) - n_a == (len(ps) + 1) // 2
        assert cfg["stats"][split]["n_A_closer"] == n_a and cfg["stats"][split]["shortfall_total"] == 0


def test_bytewise_reproducible_and_pinned(cfg):
    again = P.build_pairs(list(reversed(IMAGES)), render=RGBS)
    t1, t2 = P.dump_pairs(cfg), P.dump_pairs(again)
    assert t1 == t2
    assert json.loads(t1) == json.loads(json.dumps(cfg))
    assert hashlib.sha256(t1.encode()).hexdigest() == PINNED_DUMP_SHA256


PINNED_DUMP_SHA256 = "be5bf63e10d78bb19c0367ab34cbd6c566a3aabaa7d1b93be96c2e303116a2a5"


def test_shortfall_recorded_without_relaxing():
    w = C.NYU_PAIR_LOCAL_WINDOW
    d = np.zeros((480, 640), np.float32)
    d[200 : 200 + 2 * w + 1, 100 : 100 + 2 * w + 3] = 2.0  # 一块平面，窗口放得下的中心只有 1×3 个：凑不出远于 48 px 的点对
    pairs, info = P.pairs_for_image(7, d)
    assert pairs == [] and info["shortfall"] == 5 and info["attempts"] == C.NYU_PAIR_MAX_ATTEMPTS
    assert info["n_valid"] == (2 * w + 1) * (2 * w + 3) and info["n_candidates"] == 3
    d = np.zeros((480, 640), np.float32)
    d[200 : 200 + 2 * w + 1, 100 : 100 + 2 * w + 2] = 2.0
    d[300 : 300 + 2 * w + 1, 500 : 500 + 2 * w + 2] = 3.0  # 两块各 2 个候选中心：最多凑出 2 对（点不重复使用）
    pairs, info = P.pairs_for_image(7, d)
    assert len(pairs) == 2 and info["shortfall"] == 3
    for q in P.assign_ab(pairs, "test"):
        check_pair_constraints(q, d)


def test_edge_guard_excludes_points_near_depth_steps():
    w = C.NYU_PAIR_LOCAL_WINDOW
    d = np.full((480, 640), 2.0, np.float32)
    d[:, 320:] = 3.0  # 深度台阶在 x=319|320
    pairs, info = P.pairs_for_image(5, d)
    assert len(pairs) == 5 and info["n_candidates"] < info["n_valid"]
    for p in pairs:
        for x, _y in (p["p"], p["q"]):
            assert x + w < 320 or x - w >= 320  # 窗口不跨台阶


def old_window_ratio_ok(d: np.ndarray, valid: np.ndarray, w: int) -> np.ndarray:
    """第 1 轮的旧判据（整个窗口 max/min ≤ 1.05），只作对照。"""
    from scipy import ndimage

    dd = np.where(valid, d, np.nan).astype(np.float64)
    mx = ndimage.maximum_filter(np.nan_to_num(dd, nan=-np.inf), 2 * w + 1, mode="nearest")
    mn = ndimage.minimum_filter(np.nan_to_num(dd, nan=np.inf), 2 * w + 1, mode="nearest")
    return mx <= 1.05 * mn


def test_edge_guard_keeps_grazing_floor_and_receding_wall():
    """掠射角的地面（深度随行按透视变化）与纵深的墙（深度随列线性变大）：窗口内 max/min 远超 1.05，但相邻像素都平滑，不剔。
    同一张图里放一个前景物体，它的遮挡边界照样剔。"""
    w = C.NYU_PAIR_LOCAL_WINDOW
    yy, xx = np.mgrid[:480, :640].astype(np.float64)
    d = np.empty((480, 640))
    floor = yy >= 240
    d[floor] = 1.5 * 525.0 / (yy[floor] - 175.0)  # 相机高 1.5 m、焦距 525 px、地平线在第 175 行：12.1 m → 2.6 m
    d[~floor] = 1.0035 ** xx[~floor]  # 纵深的墙：每列 +0.35%，19 列的窗口内 max/min ≈ 1.065；1.0 m → 9.4 m
    d[300:360, 280:360] = 1.2  # 前景物体，四周都是遮挡边界
    d = d.astype(np.float32)
    valid = P.valid_mask(d)
    all_valid, no_jump = P.edge_guard_masks(d, valid, w)
    old = old_window_ratio_ok(d, valid, w)
    floor_pts = valid & all_valid & (yy > 400) & (np.abs(xx - 320) > 60)
    wall_pts = valid & all_valid & (yy < 200)
    assert floor_pts.sum() > 1000 and wall_pts.sum() > 1000
    assert (old[floor_pts]).mean() < 0.5 and (old[wall_pts]).mean() < 0.01  # 旧判据剔掉大半地面、几乎全部墙
    assert no_jump[floor_pts].all() and no_jump[wall_pts].all()  # 新判据全部保留
    # 遮挡边界：物体左边缘两侧 w 像素内的点都被剔
    edge = np.zeros_like(valid)
    edge[300 - w : 360 + w, 280 - w : 280 + w] = True
    assert not (no_jump & edge).any()
    pairs, info = P.pairs_for_image(11, d)
    assert len(pairs) == 5
    for p in P.assign_ab(pairs, "test"):
        check_pair_constraints(p, d)


def test_render_only_touches_marker_pixels():
    rgb = synth_rgb(0)
    A, B = (150, 200), (450, 300)
    out = P.render_pair(rgb, A, B)
    m = P.marker_mask(rgb.shape[:2], A, B)
    assert out.shape == rgb.shape and out.dtype == np.uint8
    assert np.array_equal(out[~m], rgb[~m])
    col = np.array(OVERLAY_PALETTE[C.NYU_MARKER_COLOR_INDEX], np.uint8)
    assert (out[m] == col).all(axis=1).mean() > 0.9  # 环与笔画窄，几乎全是实色轮廓
    assert m[A[1], A[0]] == False and m[A[1], A[0] + C.NYU_MARKER_OUTER_RADIUS]  # noqa: E712 — 环心不涂，环上涂
    assert np.array_equal(rgb, synth_rgb(0))  # 不改输入
    lay = P.label_layout(A, B, 480, 640)
    assert set(lay) == {"A", "B"}


def test_label_layout_avoids_border_and_other_marker():
    lay = P.label_layout((630, 10), (100, 400), 480, 640)  # 右上角放不下，换到左下
    g = P.glyph("A")
    x0, y0 = lay["A"]
    assert 0 <= x0 and x0 + g.shape[1] <= 640 and 0 <= y0 and y0 + g.shape[0] <= 480
    assert P.label_layout((5, 5), (100, 100), 30, 30) is None


def test_committed_pairs_config():
    """服务器生成并入库 configs/e1_nyu_pairs.json 之后的结构性检查；还没生成时跳过。"""
    if not P.PAIRS_PATH.exists():
        pytest.skip("configs/e1_nyu_pairs.json 由服务器在 NYUv2 完整落地后生成")
    text = P.PAIRS_PATH.read_text()
    cfg = json.loads(text)
    assert text == P.dump_pairs(cfg)
    assert cfg["constraints"] == P.constraints_record() and cfg["marker"] == P.marker_record()
    for p in cfg["pairs"]:
        check_pair_constraints(p, None)
    for split, st in cfg["stats"].items():
        ps = [p for p in cfg["pairs"] if p["split"] == split]
        assert st["n_pairs"] == len(ps) and st["n_A_closer"] == len(ps) // 2
    assert cfg["stats"]["test"]["n_images"] == 654 and cfg["stats"]["train"]["n_images"] == C.NYU_DEV_IMAGES


def test_build_script_nyu_pairs(tmp_path):
    """build_e1_configs.py nyu-pairs 读落地目录（index.json + depth npz + rgb png）生成配置，--check 逐字节核对。"""
    from PIL import Image

    from src.data.land.nyuv2 import npz_bytes

    src = tmp_path / "root" / "nyuv2"
    (src / "rgb").mkdir(parents=True)
    (src / "depth").mkdir()
    index = []
    for (idx, split, d) in IMAGES:
        Image.fromarray(RGBS[idx]).save(src / f"rgb/{idx:04d}.png")
        (src / f"depth/{idx:04d}.npz").write_bytes(npz_bytes(depth_raw=d, depth_filled=d))
        index.append({"idx": idx, "split": split, "rgb": f"rgb/{idx:04d}.png", "depth": f"depth/{idx:04d}.npz"})
    (src / "index.json").write_text(json.dumps({"images": index}))
    (src / "MANIFEST.json").write_text(json.dumps({"files": {"index.json": {"sha256": "ab" * 32}}, "upstream": [{"url": "u", "sha256": "cd" * 32}]}))
    spec = importlib.util.spec_from_file_location("_b", REPO / "scripts/data/build_e1_configs.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = tmp_path / "pairs.json"
    assert mod.main(["nyu-pairs", "--root", str(tmp_path / "root"), "--out", str(out)]) == 2  # 不足 654 张时拒绝
    assert mod.main(["nyu-pairs", "--root", str(tmp_path / "root"), "--out", str(out), "--allow-partial"]) == 0
    got = json.loads(out.read_text())
    assert got["pairs"] == P.build_pairs(IMAGES, render=RGBS)["pairs"] and got["source"]["mat"]["sha256"] == "cd" * 32
    assert mod.main(["nyu-pairs", "--root", str(tmp_path / "root"), "--out", str(out), "--allow-partial", "--check"]) == 0


def test_fixture_nyu_pairs_check():
    """tests/fixtures/e1/nyu_pairs.json 由 nyu-pairs 在 fixture 落地目录上生成；重算逐字节一致。
    fixture 的深度文件是 npz 格式、扩展名 .depth.zip（仓库 .gitignore 忽略 *.npz）。"""
    fx = Path(__file__).resolve().parent / "fixtures" / "e1"
    spec = importlib.util.spec_from_file_location("_b2", REPO / "scripts/data/build_e1_configs.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main(["nyu-pairs", "--root", str(fx / "root"), "--out", str(fx / "nyu_pairs.json"), "--allow-partial", "--check"]) == 0
    cfg = json.loads((fx / "nyu_pairs.json").read_text())
    assert cfg["constraints"] == P.constraints_record() and cfg["constraints"]["neighbor_jump_max_rel"] == 0.05
