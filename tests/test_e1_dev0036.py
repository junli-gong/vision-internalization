"""DEV-0036（D72-2、D72-3、D72-4；事后读数，不改 G1）：进程内的单元测试。命令行彩排见 test_e1_dev0036_cli.py。

- 配置：configs/e1_posthoc.json 只在末尾新增四节（letter_swap、headline_d72、format_forced_poly、own_box），已有条目逐字节不动（截掉
  新增部分就是 DEV-0033 的 e02971cd…）；新 sha256 钉住；改坏新节的任一处都报错。
- 对调标记的渲染（src/data/nyu_pairs.render_pair(swap_labels=True)）：默认行为不变；换标记的图只在两个标签框里不同，框里恰是另一个字母
  的字形；随机的点对布局上逐个核对。check_swapped 抓得住框外有差、框里没变。
- 对调标记的合成：两张图的字母概率按物理点平均（第二张图 A ↔ 点 2）；只按标记作答的偏好被抵消；平票取点 1；原字母读数 = 第一张图。
- 结构强制的多边形：Qwen 分词器上的规范切分（数字逐位）；语法的整数规则、点数上下限、最长生成 769 个 token；随机游走的输出全部能被
  E1 的解析接受；处理器在小模型上逐位置生效；入口断言抓得住未停、不合语法、点数不够。近矩形的判定。
- 解码时屏蔽网格 token 的处理器。
- "模型自己的框"：框的光栅化（多边形与 16×16 网格两种）、按 id 对上并核对同一句话、对不上的分原因计数。
- 汇总脚本：models_in 不再把 probe_scripts 当成模型；D72-2 的标题表；估算钉住 769。
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.data import nyu_pairs as P
from src.data.records import Sample
from src.e1 import debiased as DB
from src.e1 import letter_swap as LS
from src.e1 import posthoc as PH
from src.e1.grid_mask import GridTokenMask, mask_record
from src.e1.poly_forced import (
    PolyFormatError,
    PolyGrammar,
    PolyGrammarError,
    PolyStructureProcessor,
    check_poly_output,
    near_rect_summary,
    near_rectangle,
    poly_format_ids,
    polygon_area_ratio,
    polygon_text,
)
from src.e1.prompts import load_prompt_config
from src.e1.tasks import DataPaths, select

REPO = Path(__file__).resolve().parents[1]
POSTHOC_SHA256_DEV0033 = "e02971cd232a57891e193f581eb2e09de5af164440f60409e8ba11383471b8cc"
POSTHOC_SHA256 = "9a5607826bae8b2387cd9b5bbd88cd71d5887ebf86237b786546ab13ac613ce1"  # DEV-0036：只在末尾新增四节
PIECES = {"open": "[[", "xy_sep": ", ", "point_sep": "], [", "close": "]]"}
QWEN = "qwen3.5-0.8b"


@pytest.fixture(scope="module")
def prompts():
    return load_prompt_config()


@pytest.fixture(scope="module")
def posthoc(prompts):
    return PH.load_posthoc_config(prompts)


def _load_script(name: str, path: Path):
    saved = list(sys.path)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.path[:] = saved
    return mod


@pytest.fixture(scope="module")
def summ():
    return _load_script("_summarize_e1_posthoc_dev0036", REPO / "scripts" / "summarize_e1_posthoc.py")


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------


def test_config_append_only_and_pinned(prompts, posthoc):
    raw = (REPO / "configs/e1_posthoc.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == POSTHOC_SHA256 == posthoc.sha256
    cut = raw.index(b',\n "letter_swap": {')
    assert hashlib.sha256(raw[:cut] + b"\n}\n").hexdigest() == POSTHOC_SHA256_DEV0033  # 已有条目逐字节不动
    assert list(posthoc.data)[-4:] == ["letter_swap", "headline_d72", "format_forced_poly", "own_box"]
    assert prompts.sha256.startswith("f3353de0")  # 主评测的配置不动
    assert posthoc.base_subset("letter_swap") == "eval" and posthoc.base_subset("letter_swap_dev") == "dev"
    assert posthoc.base_subset("forced_poly") == "think" and posthoc.tasks_for("forced_poly") == ("refexp_poly",)
    assert posthoc.tasks_for("letter_swap") == ("rel_depth_nyu",)
    assert posthoc.near_rect_thresholds() == {"n_points": 4, "min_area_ratio": 0.95}
    fp = posthoc.forced_poly
    assert fp["max_points"] == 64 and fp["pieces"] == PIECES
    assert {k: v["min_points"] for k, v in fp["variants"].items()} == {"poly_structured": 3, "poly_structured_min8": 8}
    assert "769" in fp["max_points_rule"]  # 配置里的数字由 test_poly_grammar_bound_and_walks 钉住
    assert posthoc.headline["tasks"] == {"rel_depth_nyu": "letter_swap", "rel_depth_blink": "free",
                                         "spatial_cvbench": "letter_debiased", "spatial_embspatial": "letter_debiased"}
    assert set(PH.POSTHOC_SUBSETS) >= {"letter_swap", "letter_swap_dev", "forced_poly"}


@pytest.mark.parametrize("mutate", [
    lambda d: d["letter_swap"].__setitem__("tasks", ["rel_depth_nyu", "rel_depth_blink"]),
    lambda d: d["letter_swap"].__setitem__("views", ["labels_swapped", "original"]),
    lambda d: d["letter_swap"]["batch_layout"].__setitem__("layout", "sample_major"),
    lambda d: d["headline_d72"]["tasks"].__setitem__("rel_depth_blink", "letter_debiased"),
    lambda d: d["format_forced_poly"]["variants"]["poly_structured_min8"].__setitem__("min_points", 7),
    lambda d: d["format_forced_poly"]["pieces"].__setitem__("xy_sep", ","),
    lambda d: d["format_forced_poly"].__setitem__("coord_range", [0, 999]),
    lambda d: d["format_forced_poly"]["near_rectangle"].__setitem__("min_area_ratio", 1.5),
    lambda d: d["format_forced_poly"]["models"].append("gemma-4-e2b-it"),
    lambda d: d["own_box"].__setitem__("targets", ["refexp_poly"]),
    lambda d: d["own_box"].__setitem__("raster_rule", ""),
])
def test_new_sections_validated(prompts, posthoc, mutate):
    d = copy.deepcopy(dict(posthoc.data))
    mutate(d)
    with pytest.raises(PH.PosthocConfigError):
        PH.validate(d, prompts)


# ---------------------------------------------------------------------------
# 对调标记：渲染
# ---------------------------------------------------------------------------


def _rgb(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (480, 640, 3), dtype=np.uint8)


def test_render_default_unchanged_and_swap_only_touches_label_boxes():
    rng = np.random.default_rng(0)
    n_checked = 0
    for k in range(200):
        A = (int(rng.integers(0, 640)), int(rng.integers(0, 480)))
        B = (int(rng.integers(0, 640)), int(rng.integers(0, 480)))
        if P.label_layout(A, B, 480, 640) is None or A == B:
            continue
        rgb = _rgb(k)
        orig = P.render_pair(rgb, A, B)
        assert np.array_equal(orig, P.render_pair(rgb, A, B, swap_labels=False))  # 默认就是 False
        sw = P.render_pair(rgb, A, B, swap_labels=True)
        boxes = P.label_boxes((480, 640), A, B)
        outside = np.ones((480, 640), dtype=bool)
        for x0, y0, x1, y1 in boxes.values():
            outside[y0:y1, x0:x1] = False
        assert np.array_equal(orig[outside], sw[outside]), (A, B)  # 标签框之外逐像素相同（环也不变）
        m0, m1 = P.marker_mask((480, 640), A, B), P.marker_mask((480, 640), A, B, swap_labels=True)
        for k_, other in (("A", "B"), ("B", "A")):  # 框里的字形恰是另一个字母
            x0, y0, x1, y1 = boxes[k_]
            assert np.array_equal(m0[y0:y1, x0:x1], P.glyph(k_)) and np.array_equal(m1[y0:y1, x0:x1], P.glyph(other))
            assert not np.array_equal(orig[y0:y1, x0:x1], sw[y0:y1, x0:x1])
        n_checked += 1
    assert n_checked > 100


def test_committed_nyu_render_hash_unchanged_by_default():
    """点对配置记录的 render_sha256（加载器每次核对）仍由默认渲染给出：fixture 的点对逐个重渲染、比对。"""
    from PIL import Image

    cfg = P.load_pairs(REPO / "tests/fixtures/e1/nyu_pairs.json")
    root = REPO / "tests/fixtures/e1/root/nyuv2"
    index = {r["idx"]: r for r in json.loads((root / "index.json").read_text())["images"]}
    n = 0
    for p in cfg["pairs"]:
        if not p.get("render_sha256"):
            continue
        rgb = np.asarray(Image.open(root / index[p["idx"]]["rgb"]).convert("RGB"))
        assert P.pixels_sha256(P.render_pair(rgb, p["A"], p["B"])) == p["render_sha256"]
        n += 1
    assert n >= 5


@pytest.fixture(scope="module")
def nyu_samples(tmp_path_factory):
    paths = DataPaths.posthoc_fixtures(nyu_cache=tmp_path_factory.mktemp("nyu"))
    return select("rel_depth_nyu", "eval", paths).samples


def test_swapped_image_and_check(nyu_samples):
    from src.e1.runner import load_images

    for s in nyu_samples:
        o = np.asarray(load_images(s)[0])
        sw = LS.swapped_image(s)
        info = LS.check_swapped(o, sw, s)
        assert info["n_changed_pixels"] > 0
        A, B = s.fields["points"]["A"], s.fields["points"]["B"]
        assert np.array_equal(sw, P.render_pair(np.asarray(load_images(SimpleNamespace(images=(s.fields["raw_image"],)))[0]),
                                                A, B, swap_labels=True))
    s = nyu_samples[0]
    o = np.asarray(load_images(s)[0])
    with pytest.raises(LS.LetterSwapError, match="没有变化"):
        LS.check_swapped(o, o.copy(), s)
    bad = LS.swapped_image(s).copy()
    bad[0, 0] = 255 - bad[0, 0]  # 框外改一个像素
    with pytest.raises(LS.LetterSwapError, match="之外"):
        LS.check_swapped(o, bad, s)


# ---------------------------------------------------------------------------
# 对调标记：合成与指标
# ---------------------------------------------------------------------------


def _ls_sample(sid: str, gt: str) -> Sample:
    return Sample(id=sid, task="rel_depth_nyu", images=("x",), image_keys=("k",), fields={}, gt={"closer": gt},
                  source="nyuv2_pairs", split="test")


def _ls_rendered():
    return SimpleNamespace(text="t", extras={"letters": ["A", "B"], "options": ["A is closer", "B is closer"]})


def _ls_readings(sid, probs):
    """probs[v] = 第 v 张图上 (P(A), P(B))。"""
    out = []
    for v, p in enumerate(probs):
        perm = DB.cyclic_perm(2, v)
        i = int(np.argmax(p))
        out.append({"id": sid, "order": v, "k": 2, "perm": list(perm), "instruction": "t", "letter_probs": list(p),
                    "choice_letter": "AB"[i], "choice_option": perm[i], "n_prompt_tokens": 10, "n_visual_tokens": 4,
                    "n_image_pixels": 64, "view": PH.LETTER_SWAP_VIEWS[v]})
    return out


def test_view_prompts_keep_instruction_and_map_letters_to_points():
    vs = LS.view_prompts(_ls_rendered())
    assert [(v.order, v.perm, v.text) for v in vs] == [(0, (0, 1), "t"), (1, (1, 0), "t")]
    with pytest.raises(LS.LetterSwapError):
        LS.view_prompts(SimpleNamespace(text="t", extras={"letters": ["A", "B", "C"]}))


def test_letter_swap_combine_by_physical_point():
    # 只按标记作答（两张图都 0.8 选 A）：按物理点平均后 0.5 / 0.5，平票取点 1（A）
    s = _ls_sample("n/1", "B")
    rec = LS.combine_sample("rel_depth_nyu", s, _ls_rendered(), (8, 8), _ls_readings("n/1", [[0.8, 0.2], [0.8, 0.2]]))
    assert rec["mean_probs"] == [0.5, 0.5] and rec["parsed"]["letter"] == "A" and not rec["score"]["correct"]
    assert rec["readout"] == "letter_swap" and rec["parsed"]["route"] == "letter_swap"
    assert [o["view"] for o in rec["orders"]] == ["original", "labels_swapped"]
    # 真认点 2：原图 P(B) = 0.7；换标记后点 2 标 A，P(A) = 0.6 → 点 2 平均 0.65
    s = _ls_sample("n/2", "B")
    rec = LS.combine_sample("rel_depth_nyu", s, _ls_rendered(), (8, 8), _ls_readings("n/2", [[0.3, 0.7], [0.6, 0.4]]))
    assert rec["mean_probs"] == pytest.approx([0.35, 0.65]) and rec["parsed"]["letter"] == "B" and rec["score"]["correct"]
    assert rec["parsed"]["original_letter"] == "B" and rec["score"]["original_correct"]
    rs = _ls_readings("n/3", [[0.3, 0.7], [0.6, 0.4]])
    rs[1]["view"] = "original"
    with pytest.raises(LS.LetterSwapError):
        LS.combine_sample("rel_depth_nyu", _ls_sample("n/3", "A"), _ls_rendered(), (8, 8), rs)
    rs = _ls_readings("n/4", [[0.3, 0.7], [0.6, 0.4]])
    rs[1]["instruction"] = "换了提示"
    with pytest.raises(LS.LetterSwapError):
        LS.combine_sample("rel_depth_nyu", _ls_sample("n/4", "A"), _ls_rendered(), (8, 8), rs)


def test_letter_swap_metrics_per_view():
    samples, recs = [], []
    for n, (gt, probs) in enumerate([("B", [[0.8, 0.2], [0.9, 0.1]]),   # 总选 A：点 1 = (0.8 + 0.1)/2，点 2 = (0.2 + 0.9)/2 → B（对）
                                     ("A", [[0.7, 0.3], [0.6, 0.4]]),   # 点 1 = 0.55 → A（对）
                                     ("B", [[0.6, 0.4], [0.3, 0.7]])]):  # 点 1 = 0.65 → A（错）
        s = _ls_sample(f"n/{n}", gt)
        samples.append(s)
        recs.append(LS.combine_sample("rel_depth_nyu", s, _ls_rendered(), (8, 8), _ls_readings(s.id, probs)))
    m = LS.letter_swap_metrics("rel_depth_nyu", samples, recs, has_gt=True)
    assert "debiased" not in m and m["readout"] == {"letter_swap": 3}
    d = m["letter_swap"]
    assert m["headline"][0]["value"] == pytest.approx(2 / 3) and d["original_letter"]["headline"][0]["value"] == pytest.approx(1 / 3)
    assert [e["view"] for e in d["per_view"]] == ["original", "labels_swapped"] and "per_order" not in d
    pv = d["per_view"][1]  # 换标记的图：A、A、B → 点 2、点 2、点 1；gt B、A、B → 只有第一条对
    assert pv["accuracy"]["value"] == pytest.approx(1 / 3) and pv["most_common_letter"] == "A"
    assert pv["most_common_letter_share"] == pytest.approx(2 / 3)
    p0 = d["per_view"][0]  # 原图：三条都选 A = 点 1 → gt 为 A 的一条对
    assert p0["accuracy"]["value"] == pytest.approx(1 / 3) and p0["most_common_letter_share"] == 1.0


# ---------------------------------------------------------------------------
# 结构强制的多边形
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def qwen_tok():
    from transformers import AutoTokenizer

    from src.adapters import registry

    e = registry.get_model(QWEN)
    return AutoTokenizer.from_pretrained(e.repo, revision=e.revision)


@pytest.fixture(scope="module")
def qwen_fmt(qwen_tok):
    return poly_format_ids(qwen_tok, PIECES)


STOP = (900001, 900002)


def test_qwen_tokenization_is_per_digit(qwen_fmt, qwen_tok):
    assert len(qwen_fmt["digit_ids"]) == 10 and qwen_fmt["digit_tokens"] == list("0123456789")
    assert [len(qwen_fmt["pieces"][k]["ids"]) for k in ("open", "xy_sep", "point_sep", "close")] == [1, 2, 2, 1]
    text = polygon_text([(1000, 7), (12, 340)], PIECES)
    assert text == "[[1000, 7], [12, 340]]"
    assert qwen_tok.convert_ids_to_tokens(qwen_tok(text, add_special_tokens=False)["input_ids"])[1:5] == ["1", "0", "0", "0"]


def test_poly_format_rejects_non_canonical_tokenizer():
    class MergedDigits:  # 把 "10" 当成一个 token 的分词器：不是逐位
        def __call__(self, s, add_special_tokens=False):
            out, i = [], 0
            while i < len(s):
                if s.startswith("10", i):
                    out.append(100)
                    i += 2
                elif s.startswith("[[", i) or s.startswith("]]", i):
                    out.append(200 + (s[i] == "]"))
                    i += 2
                else:
                    out.append(ord(s[i]))
                    i += 1
            return {"input_ids": out}

        def decode(self, ids):
            return "".join("10" if i == 100 else "[[" if i == 200 else "]]" if i == 201 else chr(i) for i in ids)

        def convert_ids_to_tokens(self, i):
            return str(i)

    with pytest.raises(PolyFormatError, match="规范切分"):
        poly_format_ids(MergedDigits(), PIECES)


def _walk(g: PolyGrammar, choose) -> tuple[list[int], object]:
    st, ids = g.start(), []
    while st.phase != "done":
        tok = choose(st, g.allowed(st))
        ids.append(tok)
        st = g.advance(st, tok)
    return ids, st


def test_poly_grammar_bound_and_walks(qwen_fmt, qwen_tok):
    import random

    from src.parsing import parse_polygons

    for mn in (3, 8):
        g = PolyGrammar(qwen_fmt, min_points=mn, max_points=64, stop_ids=STOP)
        assert g.max_tokens() == 769  # 1 + 64 × 10 + 63 × 2 + 1 + 1（配置 max_points_rule 与汇总脚本的上界）
        rng = random.Random(mn)
        for _ in range(150):
            ids, st = _walk(g, lambda st, a: rng.choice(a))
            assert len(ids) <= g.max_tokens() and ids[-1] in STOP and mn <= st.n_points <= 64
            p = parse_polygons(qwen_tok.decode(ids[:-1]), width=640, height=480)
            assert p.failure is None and len(p.value) == 1 and len(p.value[0]) == st.n_points
    summ = _load_script("_summ_bound", REPO / "scripts" / "summarize_e1_posthoc.py")
    assert summ.POLY_STRUCTURED_MAX_TOKENS == 769


def test_poly_grammar_rules(qwen_fmt):
    g = PolyGrammar(qwen_fmt, min_points=8, max_points=9, stop_ids=STOP)
    d = {c: t for c, t in zip("0123456789", qwen_fmt["digit_ids"])}
    seq = {k: qwen_fmt["pieces"][k]["ids"] for k in PIECES}
    st = g.advance(g.start(), seq["open"][0])
    assert set(g.allowed(st)) == set(d.values())  # 数字开头：任何一位（"0" 也可以），还不能接分隔
    s0 = g.advance(st, d["0"])
    assert g.allowed(s0) == [seq["xy_sep"][0]]  # "0" 之后不能再写数字（无前导零）
    s1 = g.advance(g.advance(g.advance(st, d["1"]), d["0"]), d["0"])
    assert set(g.allowed(s1)) == {d["0"], seq["xy_sep"][0]}  # "100" 之后只能接 0（→ 1000）
    s2 = g.advance(s1, d["0"])
    assert g.allowed(s2) == [seq["xy_sep"][0]]  # 1000 之后不能再写
    with pytest.raises(PolyGrammarError):
        g.advance(s1, d["1"])  # 1001 超范围
    # 点数：第 8 个点之前没有 close，第 9 个点（上限）之后只有 close
    st = g.start()
    pts = []
    for k in range(9):
        if k == 0:
            st = g.advance(st, seq["open"][0])
        st = g.advance(st, d["5"])
        for t in seq["xy_sep"]:
            st = g.advance(st, t)
        st = g.advance(st, d["7"])
        nxt = set(g.allowed(st)) - set(d.values())
        pts.append(nxt)
        if k < 8:
            for t in seq["point_sep"]:
                st = g.advance(st, t)
    assert all(p == {seq["point_sep"][0]} for p in pts[:7])  # 第 1–7 个点之后只能接着写
    assert pts[7] == {seq["point_sep"][0], seq["close"][0]} and pts[8] == {seq["close"][0]}
    st = g.advance(st, seq["close"][0])
    assert st.phase == "end" and st.n_points == 9 and g.allowed(st) == sorted(STOP)
    with pytest.raises(ValueError):
        PolyGrammar(qwen_fmt, min_points=2, max_points=9, stop_ids=STOP)  # 少于 E1 解析的下限 3


def test_check_poly_output(qwen_fmt):
    g = PolyGrammar(qwen_fmt, min_points=3, max_points=64, stop_ids=STOP)
    d = qwen_fmt["digit_ids"]
    seq = {k: qwen_fmt["pieces"][k]["ids"] for k in PIECES}
    pt = [d[1], *seq["xy_sep"], d[2]]
    good = [*seq["open"], *pt, *seq["point_sep"], *pt, *seq["point_sep"], *pt, *seq["close"], STOP[0]]

    def res(ids, stopped=True):
        return SimpleNamespace(token_ids=tuple(ids), stopped=stopped)

    assert check_poly_output(res(good), g) == 3
    with pytest.raises(PolyGrammarError, match="停止"):
        check_poly_output(res(good[:-1], stopped=False), g)
    two = [*seq["open"], *pt, *seq["point_sep"], *pt, *seq["close"], STOP[0]]
    with pytest.raises(PolyGrammarError):
        check_poly_output(res(two), g)  # 只有 2 个点：close 不合语法
    with pytest.raises(PolyGrammarError):
        check_poly_output(res([*seq["open"], d[1], d[2], d[3], d[4], STOP[0]]), g)


def test_poly_processor_on_tiny_qwen():
    from PIL import Image

    from src.adapters.factory import make_adapter
    from src.adapters.image_policy import PixelBudget

    ad = make_adapter(QWEN, device="cpu", dtype=torch.float32, image_policy=PixelBudget(64), tiny=True, init_std=0.05)
    fmt = poly_format_ids(ad.tokenizer, PIECES)
    rng = np.random.default_rng(1)
    img = Image.fromarray(rng.integers(0, 256, (64, 80, 3), dtype=np.uint8))
    inputs = [ad.build_inputs([img], "Outline the cat with a polygon.", "image_first"),
              ad.build_inputs([], "A longer instruction so that the left padding differs between rows.", "instruction_first")]
    for mn in (3, 8):
        g = PolyGrammar(fmt, min_points=mn, max_points=12, stop_ids=ad.stop_token_ids())
        proc = PolyStructureProcessor(g)
        out = ad.generate_batch(inputs, enable_thinking=False, max_new_tokens=g.max_tokens(), logits_processor=[proc],
                                do_sample=False)
        for r in out:
            n = check_poly_output(r, g)
            assert mn <= n <= 12 and r.text.startswith("[[") and r.text.endswith("]]")
        with pytest.raises(Exception):
            proc(torch.zeros(2, 3, dtype=torch.long), torch.zeros(2, 8))  # 一个实例只用于一次 generate


def test_near_rectangle_rule():
    th = {"n_points": 4, "min_area_ratio": 0.95}
    rect = [[[0, 0], [10, 0], [10, 5], [0, 5]]]
    assert near_rectangle(rect, **th) == {"near_rect": True, "n_polygons": 1, "n_points": 4, "area_ratio": 1.0}
    bow = [[[0, 0], [10, 5], [10, 0], [0, 5]]]  # 自交的"领结"：鞋带面积 0
    assert near_rectangle(bow, **th)["near_rect"] is False
    skew = [[[0, 0], [10, 1], [10, 5], [0, 4]]]  # 平行四边形：面积 40 / 外接框 50 = 0.8
    assert near_rectangle(skew, **th)["area_ratio"] == pytest.approx(0.8) and not near_rectangle(skew, **th)["near_rect"]
    almost = [[[0, 0], [10, 0.5], [10, 5], [0, 5]]]  # 47.5 / 50 = 0.95：≥ 取
    assert near_rectangle(almost, **th)["near_rect"] is True
    assert not near_rectangle([[[0, 0], [10, 0], [10, 5], [5, 7], [0, 5]]], **th)["near_rect"]  # 5 个点
    assert not near_rectangle([[[0, 0], [10, 0], [5, 5]]], **th)["near_rect"]
    assert not near_rectangle([[[0, 0], [10, 0], [10, 0], [0, 0]]], **th)["near_rect"]  # 外接框面积为 0
    assert not near_rectangle(rect + rect, **th)["near_rect"]  # 两个多边形
    assert near_rectangle(None, **th)["near_rect"] is False
    assert polygon_area_ratio([[0, 0], [4, 0], [0, 4]]) == pytest.approx(0.5)
    s = near_rect_summary([{"near_rect": True}, {"near_rect": False}, {"near_rect": True}], [True, True, False])
    assert s == {"n": 3, "n_near_rect": 2, "share_of_all": pytest.approx(2 / 3), "n_valid": 2, "n_near_rect_valid": 1,
                 "share_of_valid": 0.5}


# ---------------------------------------------------------------------------
# 解码时屏蔽网格 token
# ---------------------------------------------------------------------------


def test_grid_token_mask():
    m = GridTokenMask([5, 2, 5])
    s = torch.arange(16.0).reshape(2, 8)
    out = m(torch.zeros(2, 3, dtype=torch.long), s)
    assert torch.isinf(out[:, [2, 5]]).all() and torch.equal(out[:, [0, 1, 3, 4, 6, 7]], s[:, [0, 1, 3, 4, 6, 7]])
    assert not torch.isinf(s).any()  # 不改输入
    with pytest.raises(ValueError):
        GridTokenMask([])
    with pytest.raises(RuntimeError):
        GridTokenMask([9])(torch.zeros(1, 1, dtype=torch.long), torch.zeros(1, 8))
    rec = mask_record(SimpleNamespace(grid_token_ids={"a": 7, "b": 3}))
    assert rec["enabled"] is True and rec["n_ids"] == 2
    assert rec["ids_sha256"] == hashlib.sha256(b"3,7").hexdigest()
    assert mask_record(SimpleNamespace(grid_token_ids={}))["n_ids"] == 0


def test_runner_mask_applies_and_generate_rows_passthrough():
    from src.e1 import runner as R

    assert R.mask_applies("text", 3) and not R.mask_applies("letter", 3) and not R.mask_applies("text", 0)
    calls = []

    class Ad:
        def generate_batch(self, inputs, **kw):
            calls.append(kw)
            return [None] * len(inputs)

    R.generate_rows(Ad(), [1, 2], batch_size=2, enable_thinking=False, max_new_tokens=4, decoding={"do_sample": False})
    assert "logits_processor" not in calls[-1]  # 不给就不传：调用与原来逐个相同
    R.generate_rows(Ad(), [1, 2], batch_size=2, enable_thinking=False, max_new_tokens=4, decoding={"do_sample": False},
                    logits_processor=[GridTokenMask([1])])
    assert len(calls[-1]["logits_processor"]) == 1
    with pytest.raises(ValueError, match="预算强制收尾"):
        R.generate_rows(Ad(), [1], batch_size=1, enable_thinking=True, max_new_tokens=5, decoding={"do_sample": False},
                        think_budget=(3, 2), logits_processor=[GridTokenMask([1])])


# ---------------------------------------------------------------------------
# "模型自己的框"
# ---------------------------------------------------------------------------


def test_own_box_record_raster(summ):
    gt = np.zeros((48, 64), dtype=bool)
    gt[8:24, 16:48] = True  # 32 × 16 的矩形
    rec = {"id": "r/1", "failure": None, "text": "x", "parsed": {"box": [16.0, 8.0, 48.0, 24.0]}}
    poly = summ.own_box_record("refexp_poly", rec, gt)
    assert poly["score"]["iou"] == pytest.approx(1.0, abs=0.08)  # pycocotools 的多边形光栅化在边上差一圈
    assert poly["parsed"]["polygons"] == [[[16.0, 8.0], [48.0, 8.0], [48.0, 24.0], [16.0, 24.0]]]
    grid = summ.own_box_record("grid16", rec, gt)
    assert len(grid["parsed"]["grid"]) == 16 and all(len(r) == 16 for r in grid["parsed"]["grid"])
    from src.executors.raster import polygons_to_mask
    from src.grid.codec import grid_to_mask, mask_to_grid
    from src.metrics.seg import mask_inter_union

    want = grid_to_mask(mask_to_grid(polygons_to_mask(summ.box_polygon(rec["parsed"]["box"]), 48, 64), 16), 48, 64)
    st = mask_inter_union(want, gt)
    assert (grid["score"]["inter"], grid["score"]["union"]) == (st.inter, st.union)
    fail = summ.own_box_record("refexp_poly", {"id": "r/2", "failure": "no_box", "text": "no", "parsed": None}, gt)
    assert fail["score"]["inter"] == 0 and fail["score"]["union"] == int(gt.sum()) and fail["parsed"] is None


def test_match_boxes_checks_same_sentence(summ, prompts, tmp_path):
    from src.e1.scoring import gt_mask

    paths = DataPaths.posthoc_fixtures(nyu_cache=tmp_path / "nyu")
    samples = select("refexp_poly", "think", paths).samples
    gts = {s.id: gt_mask(s) for s in samples}
    recs = []
    for s in samples:
        h, w = gts[s.id].shape
        recs.append({"id": s.id, "instruction": prompts.render("refexp_box", "qwen3.5", s).text, "image_size": [w, h],
                     "failure": None, "parsed": {"box": [0, 0, w / 2, h / 2]}})
    ok, bad = summ.match_boxes(samples, recs, prompts, "qwen3.5", gts)
    assert list(ok) == [s.id for s in samples] and all(not v for v in bad.values())
    recs[0] = {**recs[0], "instruction": recs[0]["instruction"].replace("Locate", "Find")}
    recs[1] = {**recs[1], "image_size": [1, 1]}
    ok, bad = summ.match_boxes(samples, recs[:-1], prompts, "qwen3.5", gts)
    assert bad["instruction_differs"] == [samples[0].id] and bad["image_size_differs"] == [samples[1].id]
    assert bad["missing_id"] == [samples[-1].id] and len(ok) == len(samples) - 3
    # 同一句话用另一个家族的模板渲染就对不上（模板随家族）
    ok2, bad2 = summ.match_boxes(samples[:1], recs[1:2], prompts, "gemma4", gts)
    assert not ok2


def test_ownbox_cell_pairs_against_own_box(summ, prompts, tmp_path):
    from src.e1.scoring import gt_mask

    paths = DataPaths.posthoc_fixtures(nyu_cache=tmp_path / "nyu")
    samples = select("grid16", "think", paths).samples
    gts = {s.id: gt_mask(s) for s in samples}
    boxes, readout = [], []
    for k, s in enumerate(samples):
        h, w = gts[s.id].shape
        boxes.append({"id": s.id, "instruction": prompts.render("refexp_box", "qwen3.5", s).text, "image_size": [w, h],
                      "failure": None if k else "no_box", "text": "x", "parsed": {"box": [0, 0, w, h]} if k else None})
        readout.append({"id": s.id, "failure": None, "text": "g", "parsed": {"grid": ["1" * 16] * 16},
                        "score": {"inter": int(gts[s.id].sum()), "union": h * w, "iou": float(gts[s.id].sum()) / (h * w)}})
    nr = {"n_points": 4, "min_area_ratio": 0.95}
    c = summ.ownbox_cell("grid16", samples, gts, prompts, "qwen3.5", boxes, boxes[:2],
                         {"main": readout, "think": readout[::-1], "restricted": None, "structured": readout[:1]}, nr)
    rows = {r["readout"]: r for r in c["rows"]}
    assert c["n_matched"] == len(samples) and set(rows) == {"own_box", "main", "think"}
    assert c["missing"]["restricted"] and "缺" in c["missing"]["structured"]
    own = rows["own_box"]
    assert own["valid_rate"]["value"] == pytest.approx((len(samples) - 1) / len(samples))
    # 全图的框填进网格 = 全 1 网格：除了解析失败的那条，与全 1 的读数逐条相同
    d = rows["main"]["paired_diff"]
    assert rows["main"]["minus"] == "own_box" and d["gIoU"]["value"] == pytest.approx(
        float(np.mean([r["score"]["iou"] for r in readout[:1]])) / len(samples))
    assert rows["think"]["n"] == len(samples)  # 读数的顺序不同也按 id 对齐
    tb = c["think_box"]
    assert tb["n_intersection"] == 2 and tb["row"]["readout"] == "own_box_think" and tb["think_on_intersection"]["n"] == 2


# ---------------------------------------------------------------------------
# 汇总脚本：模型目录与 D72-2 的标题表
# ---------------------------------------------------------------------------


def test_models_in_skips_non_model_dirs(summ, tmp_path):
    (tmp_path / "probe_scripts").mkdir()
    (tmp_path / "probe_scripts" / "p30.py").write_text("x")
    (tmp_path / "logs").mkdir()
    (tmp_path / QWEN / "rel_depth_nyu__debiased").mkdir(parents=True)
    (tmp_path / "my-ckpt" / "grid16__think").mkdir(parents=True)  # 没登记的名字，但有任务叶目录：算
    (tmp_path / "notes").mkdir()
    assert summ.models_in(tmp_path) == [QWEN, "my-ckpt"]


def _w(path: Path, recs):
    path.mkdir(parents=True, exist_ok=True)
    (path / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))


def test_validity_with_letter_swap_and_headline(summ, tmp_path):
    deb, free, ls = tmp_path / "deb", tmp_path / "free", tmp_path / "ls"
    ids = [f"s/{i}" for i in range(10)]

    def letter(i, correct, k=2):
        return {"id": f"s/{i}", "k": k, "score": {"correct": correct, "original_correct": False},
                "parsed": {"letter": "A", "original_letter": "A"}}

    for t in summ.LETTER_TASKS:
        _w(deb / QWEN / f"{t}__debiased", [letter(i, i < 5) for i in range(10)])  # 0.5
        _w(free / QWEN / f"{t}__free", [{"id": i, "failure": None, "text": "A", "score": {"correct": n < 9}}
                                         for n, i in enumerate(ids)])  # 0.9
    _w(ls / QWEN / "rel_depth_nyu__letter_swap", [letter(i, i < 8) for i in range(10)])  # 0.8
    (deb / "probe_scripts").mkdir()
    out = tmp_path / "v.json"
    assert summ.main(["validity", "--debiased", str(deb), "--free", str(free), "--letter-swap", str(ls),
                      "--out-json", str(out), "--out-md", str(tmp_path / "v.md")]) == 0
    doc = json.loads(out.read_text())
    assert {c["model"] for c in doc["cells"]} == {QWEN}  # probe_scripts 不再当成模型
    cells = {c["task"]: c for c in doc["cells"]}
    assert cells["rel_depth_nyu"]["readout"] == "letter_swap" and cells["rel_depth_nyu"]["debiased_on_free"] == 0.8
    assert not cells["rel_depth_nyu"]["flag_not_reflecting_ability"]
    assert cells["rel_depth_nyu"]["superseded_cyclic_debiased_eval"]["value"] == 0.5
    assert cells["rel_depth_blink"]["flag_not_reflecting_ability"]  # 循环移位 0.5 = 随机、自由作答 0.9：照旧标出
    row = doc["headline_d72"]["rows"][0]["cells"]
    assert row["rel_depth_nyu"]["readout"] == "letter_swap" and row["rel_depth_nyu"]["value"] == 0.8
    assert row["rel_depth_blink"]["readout"] == "free" and row["rel_depth_blink"]["value"] == pytest.approx(0.9)
    assert row["rel_depth_blink"]["valid_rate"] == 1.0 and row["rel_depth_blink"]["covers_eval"] is True
    assert row["spatial_cvbench"]["readout"] == "letter_debiased" and row["spatial_cvbench"]["value"] == 0.5
    md = (tmp_path / "v.md").read_text()
    assert "D72-2 的标题表" in md and "†" in md
    # 不给 --letter-swap：NYU 照旧用循环移位，标题表里 NYU 记为缺
    assert summ.main(["validity", "--debiased", str(deb), "--free", str(free), "--out-json", str(out)]) == 0
    doc = json.loads(out.read_text())
    assert {c["task"]: c for c in doc["cells"]}["rel_depth_nyu"]["readout"] == "letter_debiased"
    assert doc["headline_d72"]["rows"][0]["cells"]["rel_depth_nyu"]["missing"] == "letter_swap"
