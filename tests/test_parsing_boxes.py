"""三家族原生框格式的解析（DEV-0006）：适配器的 parse_boxes / parse_one_box，以及生成 → 解析 → 记账的端到端彩排。

真实输出逐字取自 docs/model_notes.md（Qwen §5.5 ≈ 第 321–326 行，Gemma §6.5 ≈ 第 758–767 行，InternVL §7.5 ≈ 第 1057–1066 行），
并与 RUN-0001/0002 的原始证据文件逐字比对。期望值按换算公式手算，写在注释里：
x 在前时 x = a/1000·W、y = b/1000·H；Gemma 的 box_2d 是 [y1, x1, y2, x2]，先换成 x 在前再换算。
COCO GT（xywh，取自 experiments/P0/0001_env_probe/raw/p5/coco_gt_2images.json）：左边的猫 [17.5, 54.38, 301.25, 415.0]，
左边的电视 [7.03, 167.76, 149.32, 94.87]。
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.adapters import gemma4, internvl35, qwen35, registry
from src.adapters.base import GenerationResult, ModelAdapter
from src.constants import BOX_ACC_IOU_THRESHOLD
from src.executors.geometry import box_iou
from src.parsing import BoxParse, OneBoxParse, ParseFailure, ParseTally, one_box, parse_count, parse_generation

PF = ParseFailure
REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_QWEN = REPO_ROOT / "experiments/P0/0001_env_probe/raw/p5"
RAW_FAMILY = REPO_ROOT / "experiments/P0/0002_family_probe/raw"

CAT_WH, TV_WH = (640, 480), (640, 426)  # 000000039769.jpg、000000000139.jpg 的原图宽高

# ---------------------------------------------------------------------------
# 真实输出（逐字）与手算的期望值
# ---------------------------------------------------------------------------

# (模型, 图, 宽高, 原始输出（含特殊 token）, 期望的绝对像素 xyxy, 期望的 label)
QWEN_CASES = [
    # bbox_2d [14, 114, 492, 988]，640×480：x1 = 14·0.64 = 8.96，y1 = 114·0.48 = 54.72，x2 = 492·0.64 = 314.88，y2 = 988·0.48 = 474.24
    ("qwen3.5-0.8b", "000000039769.jpg", CAT_WH,
     '```json\n[\n\t{"bbox_2d": [14, 114, 492, 988], "label": "cat on the left"}\n]\n```<|im_end|>\n<|endoftext|>',
     (8.96, 54.72, 314.88, 474.24), "cat on the left"),
    # [10, 391, 242, 610]，640×426：x1 = 10·0.64 = 6.4，y1 = 391·0.426 = 166.566，x2 = 242·0.64 = 154.88，y2 = 610·0.426 = 259.86
    ("qwen3.5-0.8b", "000000000139.jpg", TV_WH,
     '```json\n[\n\t{"bbox_2d": [10, 391, 242, 610], "label": "television"}\n]\n```<|im_end|>\n<|endoftext|>',
     (6.4, 166.566, 154.88, 259.86), "television"),
    # [11, 119, 494, 989]，640×480：x1 = 11·0.64 = 7.04，y1 = 119·0.48 = 57.12，x2 = 494·0.64 = 316.16，y2 = 989·0.48 = 474.72
    ("qwen3.5-9b", "000000039769.jpg", CAT_WH,
     '```json\n[\n\t{"bbox_2d": [11, 119, 494, 989], "label": "cat on the left"}\n]\n```<|im_end|>\n<|endoftext|>',
     (7.04, 57.12, 316.16, 474.72), "cat on the left"),
    # [10, 388, 241, 624]，640×426：x1 = 6.4，y1 = 388·0.426 = 165.288，x2 = 241·0.64 = 154.24，y2 = 624·0.426 = 265.824
    ("qwen3.5-9b", "000000000139.jpg", TV_WH,
     '```json\n[\n\t{"bbox_2d": [10, 388, 241, 624], "label": "television"}\n]\n```<|im_end|>\n<|endoftext|>',
     (6.4, 165.288, 154.24, 265.824), "television"),
]
GEMMA_CASES = [
    # box_2d [107, 13, 987, 496] = [y1, x1, y2, x2]，640×480：x1 = 13·0.64 = 8.32，y1 = 107·0.48 = 51.36，
    # x2 = 496·0.64 = 317.44，y2 = 987·0.48 = 473.76
    ("gemma-4-e2b-it", "000000039769.jpg", CAT_WH,
     '```json\n[\n  {"box_2d": [107, 13, 987, 496], "label": "the cat on the left"}\n]\n```<turn|>',
     (8.32, 51.36, 317.44, 473.76), "the cat on the left"),
    # [394, 13, 621, 244]，640×426：x1 = 13·0.64 = 8.32，y1 = 394·0.426 = 167.844，x2 = 244·0.64 = 156.16，y2 = 621·0.426 = 264.546
    ("gemma-4-e2b-it", "000000000139.jpg", TV_WH,
     '```json\n[\n  {"box_2d": [394, 13, 621, 244], "label": "the television"}\n]\n```<turn|>',
     (8.32, 167.844, 156.16, 264.546), "the television"),
]
INTERNVL_CASES = [
    # [[24, 114, 492, 984]]，640×480：x1 = 24·0.64 = 15.36，y1 = 114·0.48 = 54.72，x2 = 492·0.64 = 314.88，y2 = 984·0.48 = 472.32
    ("internvl3.5-2b-hf", "000000039769.jpg", CAT_WH,
     "<ref>the cat on the left</ref><box>[[24, 114, 492, 984]]</box><|im_end|>",
     (15.36, 54.72, 314.88, 472.32), "the cat on the left"),
    # [[16, 395, 224, 612]]，640×426：x1 = 16·0.64 = 10.24，y1 = 395·0.426 = 168.27，x2 = 224·0.64 = 143.36，y2 = 612·0.426 = 260.712
    ("internvl3.5-2b-hf", "000000000139.jpg", TV_WH,
     "<ref>the television</ref><box>[[16, 395, 224, 612]]</box><|im_end|>",
     (10.24, 168.27, 143.36, 260.712), "the television"),
]
FAMILY_CASES = [(qwen35, c) for c in QWEN_CASES] + [(gemma4, c) for c in GEMMA_CASES] + [
    (internvl35, c) for c in INTERNVL_CASES]


def _coco_gt_xyxy() -> dict[str, list[list[float]]]:
    raw = json.loads((RAW_QWEN / "coco_gt_2images.json").read_text())
    return {v["file"]: [[x, y, x + w, y + h] for name, (x, y, w, h) in v["objects"] if name in ("cat", "tv")]
            for v in raw.values()}


def _evidence_output(model: str, image: str) -> str:
    if model.startswith("qwen"):
        path = RAW_QWEN / model / "box_zeroshot.json"
    else:
        path = RAW_FAMILY / {"gemma-4-e2b-it": "gemma4-e2b-it", "internvl3.5-2b-hf": "internvl3.5-2b-hf"}[model] / (
            "box_zeroshot.json")
    (case,) = [c for c in json.loads(path.read_text())["cases"] if c["image"] == image]
    return case["raw_output_with_special"]


def test_literals_match_the_run_evidence():
    """测试里逐字抄写的输出与 RUN-0001/0002 落盘的原始输出逐字相同，也出现在 model_notes 里。"""
    notes = (REPO_ROOT / "docs/model_notes.md").read_text()
    for _, (model, image, _, raw, _, _) in FAMILY_CASES:
        assert _evidence_output(model, image) == raw
        assert repr(raw) in notes


@pytest.mark.parametrize("fam, case", FAMILY_CASES, ids=[f"{c[0]}-{c[1]}" for _, c in FAMILY_CASES])
def test_real_outputs_parse_to_hand_computed_boxes(fam, case):
    model, image, (w, h), raw, expected, label = case
    got = fam.parse_boxes(raw, width=w, height=h)
    assert got.failure is None and got.labels == (label,)
    np.testing.assert_allclose(got.boxes[0], expected, rtol=0, atol=1e-9)
    one = fam.parse_one_box(raw, width=w, height=h)
    assert one.failure is None and one.box == got.boxes[0]
    # 与 COCO GT 吻合（Acc@0.5 意义下命中），把两根轴读反则不命中：轴序写反会被抓住
    gts = _coco_gt_xyxy()[image]
    assert max(box_iou(one.box, g) for g in gts) >= BOX_ACC_IOU_THRESHOLD
    x0, y0, x1, y1 = one.box
    swapped = (y0 / h * w, x0 / w * h, y1 / h * w, x1 / w * h)  # 原生的两根轴互换后的读法
    assert max(box_iou(swapped, g) for g in gts) < BOX_ACC_IOU_THRESHOLD


def test_family_modules_are_distinct_about_axis_order():
    """同一个四元组：Qwen（x 在前）与 Gemma（y 在前）必须给出不同的框，且各自等于手算值。W ≠ H，四个数互不相同。"""
    text = "[100, 200, 300, 600]"
    q = qwen35.parse_boxes(text, width=1000, height=500)
    g = gemma4.parse_boxes(text, width=1000, height=500)
    # Qwen：x1 = 100，y1 = 200·0.5 = 100，x2 = 300，y2 = 600·0.5 = 300
    np.testing.assert_allclose(q.boxes[0], (100, 100, 300, 300), atol=1e-9)
    # Gemma：[y1, x1, y2, x2] → x1 = 200，y1 = 100·0.5 = 50，x2 = 600，y2 = 300·0.5 = 150
    np.testing.assert_allclose(g.boxes[0], (200, 50, 600, 150), atol=1e-9)
    # 同一条带键的输出，用另一家的键名 → malformed（不会被静默按错的轴序读）
    assert qwen35.parse_boxes('[{"box_2d": [1, 2, 3, 4]}]', width=10, height=10).failure is PF.MALFORMED
    assert gemma4.parse_boxes('[{"bbox_2d": [1, 2, 3, 4]}]', width=10, height=10).failure is PF.MALFORMED


# ---------------------------------------------------------------------------
# JSON 框的通用规则（Qwen 与 Gemma 共用；参数化到两个家族上）
# ---------------------------------------------------------------------------

JSON_FAMILIES = [(qwen35, "bbox_2d"), (gemma4, "box_2d")]


@pytest.fixture(params=JSON_FAMILIES, ids=["qwen", "gemma"])
def jfam(request):
    return request.param


def _xyxy(fam, quad, w, h):
    """期望值的独立写法：x 在前的家族直接缩放；Gemma 先把 [y1, x1, y2, x2] 换成 [x1, y1, x2, y2]。"""
    q = list(quad) if fam is qwen35 else [quad[1], quad[0], quad[3], quad[2]]
    return (q[0] * w / 1000, q[1] * h / 1000, q[2] * w / 1000, q[3] * h / 1000)


def test_json_boxes_shapes(jfam):
    fam, key = jfam
    w, h = 300, 200
    a, b = [100, 200, 300, 400], [500, 600, 700, 800]
    cases = [
        (json.dumps([{key: a, "label": "cat"}]), [a], ["cat"]),  # 对象的列表
        (json.dumps({key: a, "label": "cat"}), [a], ["cat"]),  # 单个对象
        (json.dumps([{key: a}, {key: b, "label": None}]), [a, b], [None, None]),  # label 可缺、可为 null
        (json.dumps([{key: a, "label": "x", "score": 0.9, "point_2d": [1, 2]}]), [a], ["x"]),  # 其他键忽略
        (json.dumps(a), [a], [None]),  # 裸的四元组
        (json.dumps([a, b]), [a, b], [None, None]),  # 四元组的列表
        ("Sure! The box is " + json.dumps(a) + ".", [a], [None]),  # 裸 JSON 前后的说明文字
        ("Result:\n```json\n" + json.dumps([{key: a, "label": "cat"}, {key: b, "label": "dog"}]) + "\n```\nDone.",
         [a, b], ["cat", "dog"]),  # 代码块、多个框、前后说明文字
        ("```\n" + json.dumps([{key: a}]) + "\n```", [a], [None]),  # 没有信息串的代码块
        (json.dumps([{key: [10.5, 20.25, 30, 40.0]}]), [[10.5, 20.25, 30, 40.0]], [None]),  # 小数照收
        (json.dumps([{key: [0, 0, 1000, 1000]}]), [[0, 0, 1000, 1000]], [None]),  # 量程两端
        (json.dumps([{key: [5, 5, 5, 9]}]), [[5, 5, 5, 9]], [None]),  # 退化框照样返回
    ]
    for text, quads, labels in cases:
        got = fam.parse_boxes(text, width=w, height=h)
        assert got.failure is None, text
        assert got.labels == tuple(labels)
        assert len(got.boxes) == len(quads)
        for box, quad in zip(got.boxes, quads):
            assert all(type(v) is float for v in box)
            np.testing.assert_allclose(box, _xyxy(fam, quad, w, h), rtol=0, atol=1e-9)


def test_json_boxes_empty_list_is_an_explicit_no_box(jfam):
    fam, _ = jfam
    assert fam.parse_boxes("```json\n[]\n```", width=10, height=10) == ((), (), None)
    assert fam.parse_one_box("[]", width=10, height=10) == (None, PF.NO_BOX)


def test_json_boxes_failures(jfam):
    fam, key = jfam
    good = {key: [1, 2, 3, 4]}
    cases = [
        ("There is no cat.", PF.NO_BOX),
        ("", PF.NO_BOX),
        (json.dumps([good]) + "\n" + json.dumps([good]), PF.AMBIGUOUS),  # 两个顶层 JSON
        ("```json\n[]\n```\n```json\n" + json.dumps([good]) + "\n```", PF.AMBIGUOUS),  # 两个代码块
        ('[{"%s": [1, 2, 3, 4], "%s": [1, 2, 3, 5]}]' % (key, key), PF.AMBIGUOUS),  # 重复键
        ("```json\n" + json.dumps([good])[:-1], PF.MALFORMED),  # 代码块没闭合（截断）
        ("[{'%s': [1, 2, 3, 4]}]" % key, PF.MALFORMED),  # 单引号
        (json.dumps([{"box": [1, 2, 3, 4]}]), PF.MALFORMED),  # 缺坐标键
        (json.dumps([{key: [1, 2, 3]}]), PF.MALFORMED),
        (json.dumps([{key: [1, 2, 3, 4, 5]}]), PF.MALFORMED),
        (json.dumps([{key: [[1, 2, 3, 4]]}]), PF.MALFORMED),
        (json.dumps([{key: ["1", 2, 3, 4]}]), PF.MALFORMED),
        (json.dumps([{key: [True, 2, 3, 4]}]), PF.MALFORMED),
        (json.dumps([{key: [None, 2, 3, 4]}]), PF.MALFORMED),
        ('[{"%s": [1e999, 2, 3, 4]}]' % key, PF.MALFORMED),  # 非有限
        (json.dumps([{key: [1, 2, 3, 4], "label": 7}]), PF.MALFORMED),  # label 不是字符串
        (json.dumps([good, [1, 2, 3, 4]]), PF.MALFORMED),  # 对象与四元组混在一起
        (json.dumps([1, 2, 3]), PF.MALFORMED),
        (json.dumps("cat"), PF.NO_BOX),  # 不是 [...] 或 {...}：找不到 JSON
        ("```json\n\"cat\"\n```", PF.MALFORMED),  # 代码块里是字符串：形状不对
        (json.dumps([good, {key: [1, 2, 3, 4000]}]), PF.OUT_OF_RANGE),  # 一个框坏了整条判失败
        (json.dumps([{key: [1, 2, 3, 1000.5]}]), PF.OUT_OF_RANGE),
        (json.dumps([{key: [-1, 2, 3, 4]}]), PF.OUT_OF_RANGE),  # JSON 负数合法，按量程判
        (json.dumps([{key: [5, 2, 3, 4]}]), PF.INVERTED),  # 第 3 个数 < 第 1 个
        (json.dumps([{key: [1, 5, 3, 4]}]), PF.INVERTED),  # 第 4 个数 < 第 2 个
        (json.dumps([good, {key: [500, 0, 100, 5]}]), PF.INVERTED),
        # 检查顺序：malformed 先于 out_of_range 先于 inverted
        (json.dumps([{key: [5, 2, 3, 4]}, {key: [1, 2, 3, 4000]}, {key: [1, 2, 3]}]), PF.MALFORMED),
        (json.dumps([{key: [5, 2, 3, 4]}, {key: [1, 2, 3, 4000]}]), PF.OUT_OF_RANGE),
    ]
    for text, failure in cases:
        assert fam.parse_boxes(text, width=640, height=480) == (None, None, failure), text
        assert fam.parse_one_box(text, width=640, height=480) == (None, failure), text


def test_json_boxes_inverted_is_judged_on_same_axis_pairs():
    """颠倒只比较同一根轴的起点与终点：原生四元组的第 1 与第 3 个数、第 2 与第 4 个数（两家族都是这两对）。
    跨轴比较不算：Gemma 的 [y1, x1, y2, x2] = [100, 900, 200, 950] 里 x1 = 900 > y2 = 200，但它不是颠倒的框。"""
    cross_axis = "[100, 900, 200, 950]"
    for fam in (gemma4, qwen35):
        assert fam.parse_boxes(cross_axis, width=10, height=10).failure is None
    for text in ("[300, 10, 200, 20]",  # 第 3 个数 < 第 1 个（Gemma：y2 < y1；Qwen：x2 < x1）
                 "[10, 300, 20, 200]"):  # 第 4 个数 < 第 2 个（Gemma：x2 < x1；Qwen：y2 < y1）
        for fam in (gemma4, qwen35):
            assert fam.parse_boxes(text, width=10, height=10).failure is PF.INVERTED


def test_json_boxes_arguments(jfam):
    fam, _ = jfam
    with pytest.raises(TypeError):
        fam.parse_boxes(None, width=10, height=10)
    with pytest.raises(ValueError):
        fam.parse_boxes("[1, 2, 3, 4]", width=0, height=10)
    with pytest.raises(ValueError):
        fam.parse_boxes("[1, 2, 3, 4]", width=10, height=-1)


def test_json_multiple_boxes_for_one_box():
    text = '[{"bbox_2d": [1, 2, 3, 4], "label": "a"}, {"bbox_2d": [1, 2, 3, 4], "label": "a"}]'
    assert qwen35.parse_boxes(text, width=10, height=10).failure is None
    assert qwen35.parse_one_box(text, width=10, height=10) == (None, PF.MULTIPLE_BOXES)  # 相同的框也不合并


# ---------------------------------------------------------------------------
# InternVL：<ref> 作为 label（其余规则由 tests/test_adapter_internvl35.py 覆盖，D19）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text, labels", [
    ("<ref>cat</ref><box>[[1, 2, 3, 4]]</box>", ("cat",)),
    ("<ref>cat</ref> \n<box>[[1, 2, 3, 4]]</box>", ("cat",)),  # 中间只有空白
    ("<ref> the cat </ref><box>[[1, 2, 3, 4]]</box>", (" the cat ",)),  # 原样，不去空白
    ("<box>[[1, 2, 3, 4]]</box>", (None,)),
    ("<ref>cat</ref> is at <box>[[1, 2, 3, 4]]</box>", (None,)),  # 不紧挨
    ("<ref>cat</ref><box>[[1, 2, 3, 4], [5, 6, 7, 8]]</box>", ("cat", "cat")),  # 同一个标签里的框共用
    ("<ref>a</ref><box>[[1,2,3,4]]</box> and <ref>b</ref><box>[[5,6,7,8]]</box>", ("a", "b")),
    ("<ref>a</ref><box>[[1,2,3,4]]</box><box>[[5,6,7,8]]</box>", ("a", None)),
    ("<ref>a<ref>b</ref><box>[[1,2,3,4]]</box>", ("b",)),  # 取最近的 <ref>
    ("<ref>a</ref>b</ref><box>[[1,2,3,4]]</box>", (None,)),  # 标签乱了：不猜
    ("</ref><box>[[1,2,3,4]]</box>", (None,)),
])
def test_internvl_labels_from_ref(text, labels):
    got = internvl35.parse_boxes(text, width=10, height=10)
    assert got.failure is None and got.labels == labels and len(got.boxes) == len(labels)


# ---------------------------------------------------------------------------
# 基类接口与旧名字
# ---------------------------------------------------------------------------


def _unloaded_adapters():
    """不加载模型也能解析：三个家族的适配器只构造、不 load。"""
    return [
        qwen35.Qwen35Adapter(device="cpu", image_size=(64, 96)),
        gemma4.Gemma4Adapter(device="cpu", max_soft_tokens=70),
        internvl35.InternVL35Adapter(device="cpu", crop_to_patches=False, max_patches=1),
    ]


def test_base_declares_parse_boxes_abstract_and_parse_one_box_concrete():
    assert "parse_boxes" in ModelAdapter.__abstractmethods__
    assert "parse_one_box" not in ModelAdapter.__abstractmethods__
    for cls in (qwen35.Qwen35Adapter, gemma4.Gemma4Adapter, internvl35.InternVL35Adapter):
        assert cls.parse_boxes is not ModelAdapter.parse_boxes
        assert cls.parse_one_box is ModelAdapter.parse_one_box  # 规则只在一处（src.parsing.one_box）


def test_parse_boxes_width_height_are_required_keywords():
    """base._REQUIRED_KWONLY：子类的 parse_boxes 若给 width/height 默认值或把它们做成位置参数，定义类时就报 TypeError。"""
    with pytest.raises(TypeError, match="width"):

        class _DefaultWidth(ModelAdapter):
            def parse_boxes(self, text, *, width=640, height): ...

    with pytest.raises(TypeError, match="height"):

        class _DefaultHeight(ModelAdapter):
            def parse_boxes(self, text, *, width, height=480): ...

    with pytest.raises(TypeError, match="width"):

        class _Positional(ModelAdapter):
            def parse_boxes(self, text, width, height): ...

    for cls in (qwen35.Qwen35Adapter, gemma4.Gemma4Adapter, internvl35.InternVL35Adapter):
        params = inspect.signature(cls.parse_boxes).parameters
        for p in ("width", "height"):
            assert params[p].kind is inspect.Parameter.KEYWORD_ONLY and params[p].default is inspect.Parameter.empty


def test_adapter_methods_match_module_functions_without_loading():
    samples = {
        qwen35: ['[{"bbox_2d": [1, 2, 3, 4]}, {"bbox_2d": [5, 6, 7, 8]}]', '[{"bbox_2d": [9, 8, 7, 6]}]', "none", "[]"],
        gemma4: ['[{"box_2d": [1, 2, 3, 4]}]', '{"box_2d": [1, 2, 3, 4000]}', "[]"],
        internvl35: ["<ref>a</ref><box>[[1, 2, 3, 4]]</box>", "<box>[[1, 2, 3, 4], [1, 2, 3, 4]]</box>", "<box>"],
    }
    for ad, fam in zip(_unloaded_adapters(), (qwen35, gemma4, internvl35)):
        for text in samples[fam]:
            assert ad.parse_boxes(text, width=640, height=426) == fam.parse_boxes(text, width=640, height=426)
            assert ad.parse_one_box(text, width=640, height=426) == fam.parse_one_box(text, width=640, height=426)
            assert ad.parse_one_box(text, width=640, height=426) == one_box(fam.parse_boxes(text, width=640, height=426))


def test_old_internvl_names_stay_importable():
    from src.adapters.internvl35 import BoxParse as OldBoxParse
    from src.adapters.internvl35 import BoxParseFailure, OneBoxParse as OldOneBoxParse

    assert BoxParseFailure is ParseFailure and OldBoxParse is BoxParse and OldOneBoxParse is OneBoxParse
    assert [m.value for m in BoxParseFailure][:6] == [
        "no_box", "unbalanced_tags", "malformed", "out_of_range", "inverted", "multiple_boxes"]


# ---------------------------------------------------------------------------
# 端到端彩排：真实的 token 路径（分词 → 适配器的 split_answer / generate → GenerationResult → parse_generation → 记账）
# ---------------------------------------------------------------------------

REHEARSAL_IMAGE = Image.fromarray(np.random.default_rng(20260930).integers(0, 256, (72, 104, 3), dtype=np.uint8))


@pytest.fixture(scope="module")
def tiny_adapters():
    out = {}
    try:
        out[qwen35] = qwen35.make_tiny_adapter(qwen35.load_pinned_components(), image_size=(64, 96))
        out[gemma4] = gemma4.make_tiny_adapter(gemma4.load_pinned_components())
        out[internvl35] = internvl35.make_tiny_adapter(internvl35.load_pinned_components(), crop_to_patches=False,
                                                       max_patches=1)
    except Exception as e:  # noqa: BLE001 — 取不到小文件时给出原因后失败，不跳过
        keys = [registry.get_model(m.TINY_SOURCE_KEY).repo for m in (qwen35, gemma4, internvl35)]
        pytest.fail(f"取不到 {keys} 的 config/tokenizer/处理器/模板小文件：{e!r}。测试需要联网下载一次。", pytrace=False)
    return out


def _result_from_ids(ad, ids, *, enable_thinking):
    """与各家族 generate() 的收尾相同：split_answer 拆出 text / raw_text / thinking_unclosed。"""
    text, raw, unclosed = ad.split_answer(ids, enable_thinking=enable_thinking)
    return GenerationResult(text, tuple(ids), len(ids), 10, 0, 0.0, raw_text=raw, thinking_unclosed=unclosed)


def _ids(ad, s):
    return ad.tokenizer(s, add_special_tokens=False)["input_ids"]


THINK_MARKERS = {  # 各家族的思考段开头（Qwen 的开头由提示预填，输出里只有结束标记）
    qwen35: ("", qwen35.THINK_END),
    gemma4: (gemma4.THOUGHT_OPEN, gemma4.CHANNEL_CLOSE),
    internvl35: (internvl35.THINK_START, internvl35.THINK_END),
}
STOPS = {qwen35: qwen35.IM_END, gemma4: gemma4.TURN_CLOSE, internvl35: internvl35.IM_END}


@pytest.mark.parametrize("fam", [qwen35, gemma4, internvl35], ids=["qwen", "gemma", "internvl"])
def test_rehearsal_generation_to_tally(tiny_adapters, fam):
    ad = tiny_adapters[fam]
    tally = ParseTally()
    cases = [c for f, c in FAMILY_CASES if f is fam]
    open_mark, close_mark = THINK_MARKERS[fam]
    for model, image, (w, h), raw, expected, label in cases:
        # 1. 真实输出逐 token 走一遍（含 RUN 里原样出现的特殊 token），关思考
        ids = _ids(ad, raw)
        res = _result_from_ids(ad, ids, enable_thinking=False)
        assert res.raw_text == raw and not res.thinking_unclosed
        p = parse_generation(res, ad.parse_one_box, width=w, height=h)
        assert p.failure is None
        np.testing.assert_allclose(p.box, expected, rtol=0, atol=1e-9)
        tally.add(p)
        # 2. 开思考：思考段闭合后作答 → 同一个框
        answer = raw[: raw.rindex(STOPS[fam])] if STOPS[fam] in raw else raw
        ids = _ids(ad, open_mark + "The cat is on the left, so the box is " + answer.replace("```", "") + ".")
        ids += [ad.token_id(close_mark)] + _ids(ad, "\n\n" + answer) + [ad.token_id(STOPS[fam])]
        res = _result_from_ids(ad, ids, enable_thinking=True)
        assert not res.thinking_unclosed
        p = parse_generation(res, ad.parse_one_box, width=w, height=h)
        np.testing.assert_allclose(p.box, expected, rtol=0, atol=1e-9)
        tally.add(p)
        # 3. 思考段没闭合（预算用完）→ thinking_unclosed，结果类型与成功时相同
        res = _result_from_ids(ad, _ids(ad, open_mark + "Let me look at the image carefully"), enable_thinking=True)
        assert res.thinking_unclosed and res.text == ""
        p = parse_generation(res, ad.parse_one_box, width=w, height=h)
        assert type(p) is OneBoxParse and p == (None, PF.THINKING_UNCLOSED)
        tally.add(p)
    # 4. 真的调一次 generate：只允许停止 token → 空回答（Qwen 开思考时思考段未闭合）
    inp = ad.build_inputs([REHEARSAL_IMAGE], "How many cats are there?", "image_first")
    stop = ad.token_id(STOPS[fam])
    for thinking in (False, True):
        res = ad.generate(inp, enable_thinking=thinking, max_new_tokens=2, allowed_token_ids=[stop])
        assert res.token_ids == (stop,)
        p = parse_generation(res, parse_count)
        expect = PF.THINKING_UNCLOSED if (fam is qwen35 and thinking) else PF.EMPTY
        assert type(p) is type(parse_count("")) and p == (None, expect)
        tally.add(p)
    n_cases = len(cases)
    d = tally.to_dict()
    assert json.loads(json.dumps(d)) == d
    assert d["n"] == 3 * n_cases + 2 and d["n_ok"] == 2 * n_cases
    qwen_thinking = 1 if fam is qwen35 else 0  # 第 4 步开思考那一次：Qwen 算未闭合，另两家算空回答
    assert d["failures"] == {"empty": 2 - qwen_thinking, "thinking_unclosed": n_cases + qwen_thinking}
    assert list(d["failures"]) == ["empty", "thinking_unclosed"]
