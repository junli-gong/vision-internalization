"""DEV-0068：真值模拟的"理想 LLM"（src/explore/oracle_sim.py）。

钉住：确定性（同样的输入与种子逐位相同）；点取自距离变换的极大值、落在该在的区域里；噪声档（错点率、抖动的大小随档位上升、
配置里的档都能解析、未知键与越界的概率报错）；B 样条与修正框的规则；0–1000 网格取整；文本格式与 token 数；tokenizer 的配置。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.executors.raster import box_to_mask
from src.explore import guidance as G
from src.explore import oracle_sim as O

REPO = Path(__file__).resolve().parents[1]
CFG = json.loads((REPO / "configs" / "explore_ceiling.json").read_text(encoding="utf-8"))
H, W = 120, 160


def _gt():
    g = np.zeros((H, W), bool)
    g[30:90, 40:120] = True
    g[30:50, 100:120] = False  # 缺一角
    return g


BOX = [36.0, 26.0, 124.0, 94.0]


def test_rng_and_determinism():
    a, b = O.rng_for("s", "id", "gt", "cell").normal(size=5), O.rng_for("s", "id", "gt", "cell").normal(size=5)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, O.rng_for("s", "id", "gt", "cell2").normal(size=5))
    nz = O.NoiseLevel.from_dict("medium", CFG["noise_levels"]["medium"])
    for fn in (lambda r: O.oracle_points(_gt(), BOX, 4, with_bg=True, noise=nz, rng=r),
               lambda r: O.oracle_spline(_gt(), 12, noise=nz, rng=r),
               lambda r: O.oracle_corrections(np.roll(_gt(), 5, axis=1), _gt(), 3, confine_box=BOX, noise=nz, rng=r)):
        x, y = fn(O.rng_for("t", 1)), fn(O.rng_for("t", 1))
        assert json.dumps(x, default=lambda v: np.asarray(v).tolist()) == json.dumps(y, default=lambda v: np.asarray(v).tolist())


def test_dt_peaks():
    reg = np.zeros((50, 50), bool)
    reg[10:40, 10:40] = True
    pts = O.dt_peaks(reg, 3)
    assert pts[0] == (25.5, 25.5) or abs(pts[0][0] - 25) <= 1  # 第一个点在正中（并列取行优先的第一个）
    assert len(set(pts)) == 3 and all(reg[int(y), int(x)] for x, y in pts)
    tiny = np.zeros((10, 10), bool)
    tiny[5, 5] = True
    assert O.dt_peaks(tiny, 4) == [(5.5, 5.5)]  # 取完就停
    assert O.dt_peaks(np.zeros((5, 5), bool), 3) == []


def test_oracle_points_regions_and_grid():
    g = _gt()
    o = O.oracle_points(g, BOX, 4, with_bg=True)
    assert len(o["fg"]) == 4 and len(o["bg"]) == 4 and o["n_wrong"] == 0
    for x, y in o["fg"]:
        assert g[int(y), int(x)]
    inside = box_to_mask(BOX, H, W)
    for x, y in o["bg"]:
        assert inside[int(y), int(x)] and not g[int(y), int(x)]
    assert all(isinstance(v, int) and 0 <= v <= 1000 for p in o["fg_q"] + o["bg_q"] for v in p)
    assert np.allclose(o["fg"], O.dequantize_xy(o["fg_q"], W, H))
    assert len(O.oracle_points(g, BOX, 4, with_bg=False)["bg"]) == 0
    with pytest.raises(ValueError):
        O.oracle_points(g, BOX, 2, with_bg=True, noise=O.NoiseLevel("x", point_wrong_rate=0.5))  # 有噪声要 rng


def test_point_noise_wrong_rate_and_jitter_scale():
    g = _gt()
    wrong = O.NoiseLevel("w", point_wrong_rate=1.0)
    o = O.oracle_points(g, BOX, 4, with_bg=True, noise=wrong, rng=O.rng_for("w"))
    assert o["n_wrong"] == 8
    for x, y in o["fg"]:
        assert not g[int(y), int(x)]  # 前景点全变成了背景上的点
    clean = O.oracle_points(g, BOX, 8, with_bg=True)
    d = {}
    for name in ("light", "medium"):
        nz = O.NoiseLevel.from_dict(name, CFG["noise_levels"][name])
        dev = []
        for s in range(20):
            o = O.oracle_points(g, BOX, 8, with_bg=True, noise=O.NoiseLevel(name, point_jitter_frac=nz.point_jitter_frac),
                                rng=O.rng_for("j", name, s))
            dev.append(np.abs(o["fg"] - clean["fg"]).mean())
        d[name] = float(np.mean(dev))
    assert 0 < d["light"] < d["medium"]


def test_noise_level_parsing():
    for k, v in CFG["noise_levels"].items():
        nz = O.NoiseLevel.from_dict(k, v)
        assert nz.is_zero == (k == "none")
    with pytest.raises(ValueError):
        O.NoiseLevel.from_dict("x", {"point_jiter_frac": 0.1})
    with pytest.raises(ValueError):
        O.NoiseLevel.from_dict("x", {"point_wrong_rate": 1.5})
    with pytest.raises(ValueError):
        O.NoiseLevel.from_dict("x", {"ctrl_jitter_frac": -0.1})
    lt, md = (O.NoiseLevel.from_dict(k, CFG["noise_levels"][k]) for k in ("light", "medium"))
    for f in ("point_jitter_frac", "point_wrong_rate", "ctrl_jitter_frac", "corr_box_jitter_frac", "corr_sign_flip_rate"):
        assert 0 < getattr(lt, f) < getattr(md, f), f


def test_oracle_spline():
    g = _gt()
    o = O.oracle_spline(g, 16)
    m = G.rasterize_loops(o["loops"], g.shape)
    assert (m & g).sum() / (m | g).sum() > 0.93
    assert all(isinstance(v, int) for lp in o["loops_q"] for p in lp["ctrl"] for v in p)
    nz = O.NoiseLevel.from_dict("medium", CFG["noise_levels"]["medium"])
    o2 = O.oracle_spline(g, 16, noise=nz, rng=O.rng_for("s"))
    assert not np.allclose(o2["loops"][0]["ctrl"], o["loops"][0]["ctrl"])


def test_diff_components_and_corrections():
    g = _gt()
    cur = g.copy()
    cur[60:90, 40:60] = False  # 漏掉 20×30 = 600
    cur[5:15, 5:15] = True  # 多出 100（在框外）
    cur[92:94, 60:70] = True  # 多出 20（框内一条）
    comps = O.diff_components(cur, g)
    assert [(c["sign"], c["area"]) for c in comps] == [("+", 600), ("-", 100), ("-", 20)]
    assert comps[0]["box"] == [40.0, 60.0, 60.0, 90.0]
    conf = O.diff_components(cur, g, confine_box=BOX)
    assert [(c["sign"], c["area"]) for c in conf] == [("+", 600), ("-", 20)]
    o = O.oracle_corrections(cur, g, 1, confine_box=BOX)
    assert len(o["proposals"]) == 1 and o["proposals"][0][1] == "+"
    m, _, _ = G.correction_round(np.zeros((H, W, 3), np.uint8), cur, None, o["proposals"], rule="direct", confine_box=BOX)
    assert m[60:90, 40:60].all()  # 直接填补上了漏掉的块
    flip = O.NoiseLevel("f", corr_sign_flip_rate=1.0)
    of = O.oracle_corrections(cur, g, 2, confine_box=BOX, noise=flip, rng=O.rng_for("f"))
    assert [s for _, s in of["proposals"]] == ["-", "+"] and of["n_flipped"] == 2
    assert O.oracle_corrections(g, g, 4)["proposals"] == []


def test_text_formats_exact():
    assert O.points_text([[1, 2], [3, 4]], []) == '{"foreground": [[1, 2], [3, 4]]}'
    assert O.points_text([[1, 2]], [[5, 6]]) == '{"foreground": [[1, 2]], "background": [[5, 6]]}'
    assert O.spline_text([{"role": "outer", "ctrl": [[1, 2], [3, 4], [5, 6], [7, 8]]}]) == "[[1, 2], [3, 4], [5, 6], [7, 8]]"
    two = [{"role": "outer", "ctrl": [[1, 2]]}, {"role": "outer", "ctrl": [[3, 4]]}]
    assert O.spline_text(two) == "[[[1, 2]], [[3, 4]]]"
    assert O.spline_text([{"role": "outer", "ctrl": [[1, 2]]}, {"role": "hole", "ctrl": [[3, 4]]}]) == '{"outer": [[1, 2]], "holes": [[[3, 4]]]}'
    assert O.corrections_text([{"sign": "+", "box": [1, 2, 3, 4]}]) == '[{"sign": "+", "box": [1, 2, 3, 4]}]'
    assert O.corrections_text([]) == "[]" and O.pick_text(12) == "12"
    from src.e3.c3 import polygon_text

    # 外环的写法与 E1 的统一多边形格式（C3 的 polygon_text）相同
    assert O.spline_text([{"role": "outer", "ctrl": [[100, 200], [300, 200], [300, 400]]}]) == polygon_text(
        [[100.0, 200.0, 300.0, 200.0, 300.0, 400.0]], 1000.0, 1000.0, max_points=10)


def test_quantize_roundtrip():
    q = O.quantize_box([10.2, 20.7, 100.0, 119.9], W, H)
    assert all(isinstance(v, int) for v in q)
    b = O.dequantize_box(q, W, H)
    assert max(abs(a - c) for a, c in zip(b, [10.2, 20.7, 100.0, 119.9])) <= max(W, H) / 2000 + 1e-9


def test_tokenizer_config_and_count():
    from src.adapters import qwen35, registry

    assert CFG["tokenizer"]["tiny_model"] == qwen35.TINY_SOURCE_KEY
    assert registry.get_model(CFG["tokenizer"]["model"]).family == registry.FAMILY_QWEN35
    try:
        tok, rec = O.load_tokenizer(CFG["tokenizer"]["tiny_model"])
    except Exception as e:  # 开发机缓存里没有就跳过（HF_HUB_OFFLINE=1）
        pytest.skip(f"tiny tokenizer 不可用：{e}")
    assert rec["repo"] == registry.get_model(CFG["tokenizer"]["tiny_model"]).repo
    n = O.count_tokens(tok, ["[[123, 456]]", "", "7"])
    assert n[1] == 0 and n[2] >= 1 and n[0] > n[2]
    assert O.count_tokens(tok, []) == []
