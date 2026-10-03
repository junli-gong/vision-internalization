"""DEV-0018：E5 的评测集序号规则、标题口径与分列维度（src/e5/protocol.py），提示配置（configs/e5_prompts.json；DEV-0019 从 src/e5/ 搬来）。"""

import copy
import json
from pathlib import Path

import pytest

from src.e5 import prompts as P
from src.e5.protocol import (
    A_HEADLINE_BINS,
    DEFAULT_N_PER_BIN_DEV,
    DEFAULT_N_PER_BIN_EVAL,
    E5_GRID_SIZE,
    E5_TASKS,
    eval_items,
    groups_of,
    in_headline,
    num_bins,
    parse_sample_id,
    sample_id,
)
from src.grid.codec import RELIEF_LEVEL_SYMBOLS, RELIEF_NUM_LEVELS
from src.synth import generate
from src.synth.relief import HALF_LEVELS


def test_tasks_and_defaults():
    assert E5_TASKS == ("rotation3d", "viewpoint", "occlusion", "physics", "imagery")
    assert E5_GRID_SIZE == 32 and DEFAULT_N_PER_BIN_EVAL == 100 and DEFAULT_N_PER_BIN_DEV == 2
    assert A_HEADLINE_BINS == (1, 2, 3)


@pytest.mark.parametrize("task", E5_TASKS)
def test_eval_index_rule(task):
    b = num_bins(task)
    items = eval_items(task, "eval", 5)
    assert [it.index for it in items] == list(range(5 * b))
    assert all(it.bin == it.index % b and it.split == "test" for it in items)
    for k in range(5):  # 第 b 档的第 k 条是序号 k·B + b
        for bb in range(b):
            assert items[k * b + bb].bin == bb
    assert [it.id for it in eval_items(task, "eval", 3)] == [it.id for it in items[: 3 * b]]  # 前缀性质
    dev = eval_items(task, "dev", 1)
    assert all(it.split == "val" for it in dev) and len(dev) == b
    assert parse_sample_id(items[3].id) == (task, "test", 3) and items[3].id == sample_id(task, "test", 3)
    s = generate(task, "test", items[3].index, bin=items[3].bin)  # 显式 bin 与默认轮转一致
    assert s.digest() == generate(task, "test", items[3].index).digest()
    with pytest.raises(ValueError):
        eval_items(task, "eval", 0)
    with pytest.raises(ValueError):
        eval_items(task, "train", 1)


def test_groups_and_headline():
    for i in range(8):
        a = generate("rotation3d", "test", i)
        g = groups_of("rotation3d", a.meta)
        assert g == {"family": a.meta["scene"]["objects"][0]["family"]}
        assert in_headline("rotation3d", a.meta["difficulty"]["bin"], g) == (a.meta["rotation"]["angle_deg"] >= 30)
        b = generate("viewpoint", "test", i)
        gb = groups_of("viewpoint", b.meta)
        assert gb["move"] == b.meta["move"]["type"] and in_headline("viewpoint", 0, gb) == (gb["move"] == "orbit")
    assert set(groups_of("occlusion", generate("c", "test", 0).meta)) == {"family"}
    assert set(groups_of("physics", generate("d", "test", 0).meta)) == {"action"}
    assert set(groups_of("imagery", generate("e", "test", 0).meta)) == {"transform"}


# ---------------------------------------------------------------------------
# 提示
# ---------------------------------------------------------------------------


def test_prompt_config_loads_and_formats():
    c = P.load_prompt_config()
    assert len(c.sha256) == 64 and c.order == "image_first" and c.batch_size() == 64
    assert c.enable_thinking("zeroshot") is False and c.enable_thinking("cot") is True
    # D59-5：思维链是预算强制收尾，思考 4096（E1 的思考预算）+ 作答 1024；零样本照旧 generate_batch
    assert c.generation("zeroshot") == "plain" and c.max_new_tokens("zeroshot") == 2048
    assert c.generation("cot") == "budgeted" and c.budgets("cot") == (4096, 1024)
    with pytest.raises(KeyError):
        c.max_new_tokens("cot")
    with pytest.raises(KeyError):
        c.budgets("zeroshot")
    # a、b 用 33 符号的起伏网格（b 自 DEV-0028）；思维链的 a、b 也是（D59-4）
    assert {c.kind_of("zeroshot", t) for t in ("a", "b")} == {"relief_grid"}
    assert {c.kind_of("zeroshot", t) for t in ("c", "d", "e")} == {"binary_grid"}
    assert {c.kind_of("cot", t) for t in ("a", "b")} == {"relief_grid"}
    assert {c.kind_of("cot", t) for t in ("c", "d", "e")} == {"polygon"}
    assert c.format_of("zeroshot", "e") == "binary_canvas" and c.format_of("cot", "e") == "polygon_canvas"


def test_binary_template_is_the_e1_grid16_wording_scaled_to_32():
    """E1 grid16 的模板（configs/e1_prompts.json，DEV-0014）把 16 换成 {n}，目标换成 {target} as described above。"""
    e1 = ("Segment {expression} in the image as a 16x16 binary grid. Split the image into 16 rows (top to bottom) and 16 "
          "columns (left to right) of equal size. Output exactly 16 lines, each with 16 digits and no spaces: 1 if the cell "
          "is mostly covered by the object, otherwise 0. Output only the grid.")
    tail = e1.split("binary grid. ", 1)[1].replace("16", "{n}").replace("the object", "the {target} as described above")
    c = P.load_prompt_config()
    assert c.template("zeroshot", "c").endswith(tail) and c.template("zeroshot", "d").endswith(tail)  # b 自 DEV-0028 用起伏网格
    assert c.template("zeroshot", "e").endswith(tail.replace("image", "canvas"))


def test_relief_template_matches_the_codec_and_relief_definition():
    t = P.load_prompt_config().template("zeroshot", "rotation3d")
    assert " ".join(RELIEF_LEVEL_SYMBOLS) in t and f"using {RELIEF_NUM_LEVELS} steps" in t
    assert "Write . for a cell" in t
    assert "Each step is 1/16 of" in t and 2 / RELIEF_NUM_LEVELS == 1 / 16  # 档宽 2r / 32
    assert RELIEF_LEVEL_SYMBOLS[HALF_LEVELS] == "G" and "G covers from the average depth to 1/16 radius farther" in t
    assert RELIEF_LEVEL_SYMBOLS[HALF_LEVELS - 1] == "F" and "F covers from 1/16 radius nearer up to the average depth" in t
    assert "written 0 or V" in t and RELIEF_LEVEL_SYMBOLS[0] == "0" and RELIEF_LEVEL_SYMBOLS[-1] == "V"


# 零样本 a 的模板（DEV-0018 定稿，DEV-0028 未改）：开头一句 + 符号表说明 + "Output only the grid."
A_HEAD = ("{instruction} Answer with a {n}x{n} grid over the image that shows both where the {target} is after the rotation "
          "and how far each part of it is from the camera. ")


def _symbol_table() -> str:
    t = P.load_prompt_config().template("zeroshot", "rotation3d")
    assert t.startswith(A_HEAD) and t.endswith(" Output only the grid.")
    return t[len(A_HEAD): -len(" Output only the grid.")]


def test_all_relief_templates_share_the_zero_shot_symbol_table():
    """D59-4、DEV-0028：b 的零样本、a 与 b 的思维链都输出同一种 33 符号网格，符号表说明逐字取自零样本 a 的模板。"""
    c = P.load_prompt_config()
    table = _symbol_table()
    assert table.startswith("Split the image into {n} rows") and table.endswith("is written 0 or V.")
    b0 = c.template("zeroshot", "viewpoint")
    assert b0 == A_HEAD.replace("after the rotation", "in the new view").replace(
        "from the camera.", "from the camera at its new position.") + table + " Output only the grid."
    ca, cb = c.template("cot", "rotation3d"), c.template("cot", "viewpoint")
    for t, where, cam in ((ca, "after the rotation", "the camera"), (cb, "in the new view", "the camera at its new position")):
        assert t.startswith("{instruction} First reason step by step in words about where the {target} ")
        assert f"Then answer with a {{n}}x{{n}} grid over the image that shows both where the {{target}} is {where} and how far " \
               f"each part of it is from {cam}. " + table + " Give the final answer as exactly one such grid." in t
        assert t.endswith(table + " Give the final answer as exactly one such grid.")
    # c、d、e 的思维链仍是多边形，措辞不变
    assert c.template("cot", "occlusion").startswith("{instruction} First reason step by step in words about where the {target} "
                                                     "ends up. Then outline the {target} as described above with a polygon.")


def test_render():
    c = P.load_prompt_config()
    s = generate("imagery", "val", 0)
    txt = c.render("zeroshot", "imagery", s.instruction + "  ", s.meta["target_description"])
    assert txt.startswith(s.instruction) and "32x32 binary grid over the canvas" in txt and txt == txt.strip()
    assert "{" not in txt
    cot = c.render("cot", "occlusion", "Complete it.", "red square")
    assert "relative to the image size and range from 0 to 1000" in cot and "red square" in cot
    cot_a = c.render("cot", "rotation3d", "Rotate it.", "red cube")
    assert "32x32 grid over the image" in cot_a and "red cube" in cot_a and "{" not in cot_a


# DEV-0018 定稿 d84b6efc…；DEV-0019 搬家不变；D43 批大小 16 → 64（管理员常量级修改）；
# DEV-0028（D59-4、D59-5）：schema 2，b 与思维链的 a、b 用起伏网格，思维链预算强制收尾
E5_PROMPTS_SHA256 = "7833f9d657dd28bd8283c39af857858e912540a14e1aba3ff44ea49f2c9a1ea3"


def test_prompt_config_lives_in_configs_unchanged(tmp_path):
    """DEV-0019：提示配置搬到 configs/e5_prompts.json，逐字节不变（sha256 同 DEV-0018）；src/e5/ 里不再有副本；
    运行目录的 meta.json 记的是新位置的 sha256。"""
    from src.e5 import runner as RN

    repo = Path(__file__).resolve().parents[1]
    assert P.PROMPTS_PATH == repo / "configs" / "e5_prompts.json"
    assert P.file_sha256(P.PROMPTS_PATH) == E5_PROMPTS_SHA256 and P.load_prompt_config().sha256 == E5_PROMPTS_SHA256
    assert not (repo / "src" / "e5" / "e5_prompts.json").exists()
    spec = RN.RunSpec(system="baseline", model_key="visible", task="occlusion", subset="dev", n_per_bin=1, n_resamples=0, limit=1)
    RN.run_baseline_task(spec, tmp_path)
    meta = json.loads((RN.leaf_dir(tmp_path, spec) / "meta.json").read_text())
    assert meta["data_sha256"] == {str(P.PROMPTS_PATH): E5_PROMPTS_SHA256}


def test_validate_rejects_bad_configs():
    raw = json.loads(P.PROMPTS_PATH.read_text(encoding="utf-8"))
    P.validate(raw)
    bad = copy.deepcopy(raw)
    bad["systems"]["zeroshot"]["format_of_task"]["imagery"] = "binary_image"
    with pytest.raises(P.PromptConfigError):
        P.validate(bad)
    bad = copy.deepcopy(raw)
    bad["systems"]["cot"]["templates"]["polygon_image"] += " {oops}"
    with pytest.raises(P.PromptConfigError):
        P.validate(bad)
    bad = copy.deepcopy(raw)
    bad["decoding"]["do_sample"] = True
    with pytest.raises(P.PromptConfigError):
        P.validate(bad)
    # schema 2（DEV-0028）：生成方式与预算
    for mutate in (lambda d: d["systems"]["cot"].update(generation="free"),
                   lambda d: d["systems"]["cot"].pop("answer_budget"),
                   lambda d: d["systems"]["cot"].update(enable_thinking=False),
                   lambda d: d["systems"]["cot"].update(max_new_tokens=4096),
                   lambda d: d["systems"]["zeroshot"].update(thinking_budget=10),
                   lambda d: d["systems"]["zeroshot"].pop("max_new_tokens"),
                   lambda d: d["systems"]["zeroshot"]["format_of_task"].update(occlusion="relief_image"),  # 起伏只给 a、b
                   lambda d: d.update(schema_version=1)):
        bad = copy.deepcopy(raw)
        mutate(bad)
        with pytest.raises(P.PromptConfigError):
            P.validate(bad)
