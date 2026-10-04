"""DEV-0033：E1 事后读数（D66；E1 预注册 §10）的口径、拆分、分批、合成与指标（进程内；命令行与续跑见 test_e1_posthoc_cli.py）。

- 配置：configs/e1_posthoc.json 的 sha256 钉住；它依赖的 configs/e1_prompts.json 仍是 f3353de0…（主评测的配置一个字节都不动）；
  任务、写法、网格长度（271 = 16 × 16 + 15）、换行周期（17）、Qwen 四个尺寸与 E1 的约定一致；改坏任一处都报错。
- 选项拆分（§10.1）：真实格式的 BLINK、CV-Bench 提示（RUN-0027 的 eval 与 0020 的 dev 里全部 2,762 条）原顺序拼回逐字节相同；
  拆不出（fixture 里占位的 "p"、选项行多一行、选项文字对不上、没有题干）就报错；三家族在 fixture 上的 k 种顺序：s = 0 与主评测的
  渲染逐字节相同，其余顺序把选项内容按循环移位换位（字母与题干不变），EmbSpatial 用 option_format 重拼。
- 分批（order_major）：s = 0 那一段与主评测的 batch_plan 逐批相同；同一批的顺序与字母集合相同；混合 k 时 s 只取 k > s 的样本。
- 合成：k 种顺序的字母概率映射回选项内容后取平均；纯位置偏好被抵消；平票取原顺序靠前的选项；per_order、paired_diff、random_level。
- 格式强制：0/1/换行三个 token 的构造与 scripts/check_training_interfaces.grid_allowed 相同（id 一致，那边另加停止 token）；结构强制的
  logits 处理器在小模型上逐位置生效；入口断言抓得住长度、允许集合与换行位置的错误。
- 事后 fixture（tests/fixtures/e1_posthoc）：与 e1 的 fixture 只差说明里写的几处。
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

from src.adapters import registry
from src.adapters.factory import make_adapter
from src.constants import GRID_SIZE_E1
from src.data.records import Sample
from src.e1 import debiased as DB
from src.e1 import forced as FC
from src.e1 import posthoc as PH
from src.e1.prompts import load_prompt_config
from src.e1.runner import batch_keys, batch_plan
from src.e1.tasks import CHOICE_TASKS, FIXTURE_DIR, FIXTURE_POSTHOC_DIR, DataPaths, select

REPO = Path(__file__).resolve().parents[1]
E1_PROMPTS_SHA256 = "f3353de0d53de22b8f2226e07afef94b643d02ec106868eaf6bf2766a45bf452"  # D60：主评测、free、extra、think 的冻结配置
POSTHOC_SHA256 = "e02971cd232a57891e193f581eb2e09de5af164440f60409e8ba11383471b8cc"  # configs/e1_posthoc.json（DEV-0033）
FAMILY_KEYS = ("qwen3.5-0.8b", "gemma-4-e2b-it", "internvl3.5-2b-hf")


@pytest.fixture(scope="module")
def prompts():
    return load_prompt_config()


@pytest.fixture(scope="module")
def posthoc(prompts):
    return PH.load_posthoc_config(prompts)


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------


def test_config_hashes_and_pins(prompts, posthoc):
    assert prompts.sha256 == E1_PROMPTS_SHA256  # 主评测的配置不动
    assert posthoc.sha256 == POSTHOC_SHA256 == hashlib.sha256((REPO / "configs/e1_posthoc.json").read_bytes()).hexdigest()
    assert posthoc.data["prompt_config"]["sha256"] == prompts.sha256
    assert PH.GRID_FIXED_TOKENS == GRID_SIZE_E1 * GRID_SIZE_E1 + GRID_SIZE_E1 - 1 == 271
    assert PH.GRID_NEWLINE_PERIOD == GRID_SIZE_E1 + 1 == 17
    for v in ("restricted", "structured"):
        assert posthoc.forced["variants"][v]["n_tokens"] == 271
    assert posthoc.forced["variants"]["structured"]["newline_period"] == 17
    assert tuple(posthoc.debiased["tasks"]) == tuple(prompts.data["letter_readout"]["tasks"]) == CHOICE_TASKS
    qwen = [k for k, e in registry.MODELS.items() if e.family == registry.FAMILY_QWEN35]
    assert sorted(posthoc.forced["models"]) == sorted(qwen) and len(qwen) == 4
    assert posthoc.forced["families"] == [registry.FAMILY_QWEN35]
    assert posthoc.base_subset("debiased") == "eval" and posthoc.base_subset("debiased_dev") == "dev"
    assert posthoc.base_subset("forced") == "think"
    assert posthoc.tasks_for("forced") == ("refexp_poly", "grid16")
    assert posthoc.forced_variants("grid16") == ("restricted", "structured")
    assert posthoc.forced_variants("refexp_poly") == ("prefill",)
    assert posthoc.validity_thresholds() == {"near_random_margin": 0.05, "free_valid_rate_min": 0.9,
                                             "free_minus_debiased_min": 0.2}
    assert posthoc.forced["variants"]["prefill"]["answer_prefix"] == "[["


@pytest.mark.parametrize("mutate", [
    lambda d: d["prompt_config"].__setitem__("sha256", "0" * 64),
    lambda d: d["debiased_letter"].__setitem__("tasks", ["rel_depth_nyu"]),
    lambda d: d["debiased_letter"]["option_split"].__setitem__("line_format", "{letter}. {text}"),
    lambda d: d["format_forced"]["variants"]["restricted"].__setitem__("n_tokens", 272),
    lambda d: d["format_forced"]["variants"]["structured"].__setitem__("newline_period", 16),
    lambda d: d["format_forced"]["variants"]["prefill"].__setitem__("answer_prefix", "["),
    lambda d: d["format_forced"].__setitem__("enable_thinking", True),
    lambda d: d["format_forced"]["models"].append("gemma-4-e2b-it"),
    lambda d: d["debiased_letter"]["batch_layout"].__setitem__("layout", "sample_major"),
])
def test_config_validation_catches(prompts, posthoc, mutate):
    d = copy.deepcopy(dict(posthoc.data))
    mutate(d)
    with pytest.raises(PH.PosthocConfigError):
        PH.validate(d, prompts)


# ---------------------------------------------------------------------------
# 选项拆分
# ---------------------------------------------------------------------------

BLINK = ("Two points are circled on the image, labeled by A and B beside each circle. Which point is closer to the camera?"
         "\nSelect from the following choices.\n(A) A is closer\n(B) B is closer")


def _split(text, options):
    return DB.split_option_lines(text, [chr(65 + i) for i in range(len(options))], options, line_format="({letter}) {text}",
                                 joiner="\n")


def test_split_real_formats_and_compose():
    b = _split(BLINK, ["A is closer", "B is closer"])
    assert b.compose((0, 1)) == BLINK
    assert b.compose((1, 0)).endswith("choices.\n(A) B is closer\n(B) A is closer")
    cv = "How many organs are in the image? Select from the following choices.\n(A) 3\n(B) 2\n(C) 1\n(D) 0"
    b = _split(cv, ["3", "2", "1", "0"])
    assert b.head == "How many organs are in the image? Select from the following choices."
    assert b.compose(DB.cyclic_perm(4, 1)) == b.head + "\n(A) 2\n(B) 1\n(C) 0\n(D) 3"
    assert b.compose(DB.cyclic_perm(4, 3)) == b.head + "\n(A) 0\n(B) 3\n(C) 2\n(D) 1"


@pytest.mark.parametrize("text, options, match", [
    ("p", ["1", "2"], "放不下"),
    ("Q?\n(C) 9\n(A) 1\n(B) 2", ["1", "2"], "像选项行"),  # 选项行比 choices 多一行
    ("Q?\n(A) 1\n(B) 2\n(C) 3", ["2", "3"], "应为选项行"),
    ("Q?\n(A) 1\n(B) 2", ["1", "3"], "应为选项行"),
    ("\n(A) 1\n(B) 2", ["1", "2"], "没有题干"),
    ("Q?\n(A) 1\n(C) 2", ["1", "2"], "应为选项行"),
])
def test_split_errors(text, options, match):
    with pytest.raises(DB.OptionSplitError, match=match):
        _split(text, options)


def _real_mcq_records():
    """RUN-0027 的 eval 与 0020/0024 验收的 dev 里 BLINK、CV-Bench 的全部指令（每个模型相同，取一个模型；仓库里有时才测）。"""
    out = {}
    for pat in ("experiments/E1/0027_e1_main/qwen3.5-4b/{t}/predictions.jsonl",
                "experiments/E1/0020_acceptance_v3/*/acceptance/A1/*/{t}__dev/predictions.jsonl"):
        for t in ("rel_depth_blink", "spatial_cvbench"):
            for p in sorted(REPO.glob(pat.format(t=t))):
                for line in p.read_text().splitlines():
                    r = json.loads(line)
                    out[(t, r["id"])] = r
    return out


def test_split_every_real_prompt_roundtrips():
    recs = _real_mcq_records()
    if not recs:
        pytest.skip("仓库里没有 RUN-0027 的 predictions（只在完整检出时核对）")
    n = {"rel_depth_blink": 0, "spatial_cvbench": 0}
    for (t, _), r in recs.items():
        b = _split(r["instruction"], r["extras"]["options"])
        k = len(r["extras"]["options"])
        assert b.compose(tuple(range(k))) == r["instruction"]  # split 内部也断言了
        assert len({b.compose(DB.cyclic_perm(k, s)) for s in range(k)}) == k
        n[t] += 1
    assert n["rel_depth_blink"] >= 112 and n["spatial_cvbench"] >= 2588


@pytest.fixture(scope="module")
def posthoc_paths(tmp_path_factory):
    return DataPaths.posthoc_fixtures(nyu_cache=tmp_path_factory.mktemp("nyu"))


@pytest.mark.parametrize("family", registry.FAMILIES)
@pytest.mark.parametrize("task", CHOICE_TASKS)
def test_ordered_prompts_on_fixture(prompts, posthoc, posthoc_paths, family, task):
    samples = select(task, "eval", posthoc_paths).samples + select(task, "dev", posthoc_paths).samples
    assert samples
    for s in samples:
        rendered, orders = DB.ordered_prompts(prompts, posthoc, task, family, s)
        assert rendered.text == prompts.render(task, family, s).text  # 主评测的渲染
        k = len(rendered.extras["options"])
        assert [o.order for o in orders] == list(range(k)) and orders[0].text == rendered.text
        opts = rendered.extras["options"]
        for o in orders:
            assert o.perm == DB.cyclic_perm(k, o.order)
            if task == "spatial_embspatial":
                fmt = prompts.data["tasks"][task]["option_format"]
                want = "\n".join(fmt.format(letter=chr(65 + i), text=opts[j]) for i, j in enumerate(o.perm))
            else:
                want = "\n".join(f"({chr(65 + i)}) {opts[j]}" for i, j in enumerate(o.perm))
            assert want in o.text
            head = rendered.text.split("\n(A) ")[0] if task != "spatial_embspatial" else rendered.text.split("\nA. ")[0]
            assert o.text.startswith(head)


def test_ordered_prompts_keeps_prompt_field_whitespace(prompts, posthoc, posthoc_paths):
    """prompt 字段首尾若有空白：单独留出、原样放回（原顺序与 prompt 字段逐字节相同；渲染时本来就去掉）。"""
    import dataclasses

    s = select("spatial_cvbench", "eval", posthoc_paths).samples[0]
    raw = "\n " + s.fields["prompt"] + " \n"
    s2 = dataclasses.replace(s, fields={**s.fields, "prompt": raw})
    rendered, orders = DB.ordered_prompts(prompts, posthoc, "spatial_cvbench", registry.FAMILY_QWEN35, s2)
    assert rendered.text == s.fields["prompt"] == orders[0].text
    assert orders[1].text != orders[0].text and not orders[1].text.endswith(" ")


def test_rule_tolerance_documented():
    spec = importlib.util.spec_from_file_location("_summ_eps", REPO / "scripts" / "summarize_e1_posthoc.py")
    mod = importlib.util.module_from_spec(spec)
    saved = list(sys.path)
    spec.loader.exec_module(mod)
    sys.path[:] = saved
    rule = json.loads((REPO / "configs/e1_posthoc.json").read_text())["debiased_letter"]["validity_check"]["rule"]
    assert mod.RULE_EPS == 1e-9 and "1e-9" in rule


@pytest.mark.parametrize("task", ["rel_depth_blink", "spatial_cvbench"])
def test_ordered_prompts_refuses_placeholder_prompt(prompts, posthoc, task, tmp_path):
    """e1 的 fixture 里 BLINK、CV-Bench 的 prompt 是占位的 "p"：拆不出就报错，不跳过。"""
    paths = DataPaths.fixtures(nyu_cache=tmp_path)
    s = select(task, "eval", paths).samples[0]
    with pytest.raises(DB.OptionSplitError):
        DB.ordered_prompts(prompts, posthoc, task, registry.FAMILY_QWEN35, s)


def test_cyclic_perm_rules():
    assert DB.cyclic_perm(3, 0) == (0, 1, 2) and DB.cyclic_perm(3, 2) == (2, 0, 1)
    for bad in ((1, 0), (3, 3), (3, -1)):
        with pytest.raises(ValueError):
            DB.cyclic_perm(*bad)


# ---------------------------------------------------------------------------
# 分批（order_major）
# ---------------------------------------------------------------------------


def test_reading_rows_order_major_and_plan():
    lts = [tuple("AB"), tuple("AB"), tuple("ABCD"), tuple("ABCD"), tuple("AB"), tuple("ABC")]
    ks = [len(x) for x in lts]
    rows, keys = DB.reading_rows(ks, lts)
    assert len(rows) == sum(ks)
    assert [s for _, s in rows] == sorted(s for _, s in rows)  # 顺序优先
    assert rows[:6] == [(j, 0) for j in range(6)]
    assert [j for j, s in rows if s == 2] == [2, 3, 5] and [j for j, s in rows if s == 3] == [2, 3]
    plan = batch_plan(keys, 2)
    for a, b in plan:
        assert len({keys[i] for i in range(a, b)}) == 1  # 同一批的 (顺序, 字母集合) 相同
    main = batch_plan(batch_keys("letter", [SimpleNamespace(extras={"letters": list(x)}) for x in lts]), 2)
    assert [p for p in plan if p[1] <= 6] == main  # s = 0 那一段与主评测的批划分逐批相同


# ---------------------------------------------------------------------------
# 合成与指标
# ---------------------------------------------------------------------------


def _sample(task, sid, gt):
    gtd = {"closer": gt} if task == "rel_depth_nyu" else {"answer": gt}
    return Sample(id=sid, task=task, images=("x",), image_keys=("k",), fields={}, gt=gtd, source="s", split="test")


def _readings(sid, probs_by_order, k):
    out = []
    for s, p in enumerate(probs_by_order):
        perm = DB.cyclic_perm(k, s)
        i = int(np.argmax(p))
        out.append({"id": sid, "order": s, "k": k, "perm": list(perm), "letter_probs": list(p), "choice_letter": chr(65 + i),
                    "choice_option": perm[i], "n_prompt_tokens": 10, "n_visual_tokens": 4, "n_image_pixels": 64})
    return out


def _rendered(k):
    return SimpleNamespace(text="t", extras={"letters": [chr(65 + i) for i in range(k)], "options": [str(i) for i in range(k)]})


def test_combine_cancels_pure_position_bias_and_breaks_ties_to_first():
    # 总选 A（0.9）：两种顺序映射回内容后各 0.5，平票取原顺序靠前的（A）
    s = _sample("rel_depth_nyu", "x/1", "B")
    rec = DB.combine_sample("rel_depth_nyu", s, _rendered(2), (8, 8), _readings("x/1", [[0.9, 0.1], [0.9, 0.1]], 2))
    assert rec["mean_probs"] == [0.5, 0.5]
    assert rec["parsed"]["letter"] == "A" and rec["parsed"]["original_letter"] == "A"
    assert rec["score"] == {"gt": "B", "correct": False, "original_correct": False, "p_gold": 0.5, "original_p_gold": 0.1}
    assert rec["n_generated_tokens"] == 2 and rec["n_prompt_tokens"] == 20 and rec["k"] == 2


def test_combine_recovers_content_preference_under_bias():
    # 内容 2 的偏好 + 位置偏好（A）：原顺序选 A（内容 0，错），平均之后选内容 2（对）
    k = 3
    s = _sample("spatial_cvbench", "x/2", "C")
    probs = [[0.5, 0.1, 0.4], [0.45, 0.4, 0.15], [0.6, 0.2, 0.2]]  # s=0:[c0,c1,c2]；s=1:[c1,c2,c0]；s=2:[c2,c0,c1]
    rec = DB.combine_sample("spatial_cvbench", s, _rendered(k), (8, 8), _readings("x/2", probs, k))
    want = [(0.5 + 0.15 + 0.2) / 3, (0.1 + 0.45 + 0.2) / 3, (0.4 + 0.4 + 0.6) / 3]
    assert rec["mean_probs"] == pytest.approx(want, abs=1e-15)
    assert rec["parsed"]["letter"] == "C" and rec["score"]["correct"] and not rec["score"]["original_correct"]
    assert [o["choice_option"] for o in rec["orders"]] == [0, 1, 2]


def test_combine_rejects_inconsistent_readings():
    s = _sample("rel_depth_nyu", "x/1", "A")
    rs = _readings("x/1", [[0.6, 0.4], [0.3, 0.7]], 2)
    rs[1]["perm"] = [0, 1]
    with pytest.raises(ValueError, match="循环移位"):
        DB.combine_sample("rel_depth_nyu", s, _rendered(2), (8, 8), rs)
    rs = _readings("x/1", [[0.6, 0.4], [0.3, 0.7]], 2)
    rs[1]["n_visual_tokens"] = 5
    with pytest.raises(RuntimeError, match="视觉 token"):
        DB.combine_sample("rel_depth_nyu", s, _rendered(2), (8, 8), rs)


def test_debiased_metrics_sections():
    task = "rel_depth_nyu"
    samples, recs = [], []
    table = [("B", [[0.9, 0.1], [0.8, 0.2]]),  # 原顺序选 A（错）；平均：c0 = (0.9 + 0.2)/2，c1 = (0.1 + 0.8)/2 → A（错）
             ("B", [[0.6, 0.4], [0.9, 0.1]]),  # c0 = 0.35，c1 = 0.65 → B（对），原顺序 A（错）
             ("A", [[0.7, 0.3], [0.6, 0.4]])]  # c0 = 0.55 → A（对），原顺序 A（对）
    for n, (gt, probs) in enumerate(table):
        s = _sample(task, f"x/{n}", gt)
        samples.append(s)
        recs.append(DB.combine_sample(task, s, _rendered(2), (8, 8), _readings(s.id, probs, 2)))
    m = DB.debiased_metrics(task, samples, recs, has_gt=True)
    assert m["headline"][0]["value"] == pytest.approx(2 / 3)
    d = m["debiased"]
    assert d["original_letter"]["headline"][0]["value"] == pytest.approx(1 / 3)
    assert d["paired_diff"]["value"] == pytest.approx(1 / 3) and d["paired_diff"]["n_wrong_to_right"] == 1
    assert d["paired_diff"]["n_answer_changed"] == 1 and d["paired_diff"]["n_right_to_wrong"] == 0
    assert d["random_level"] == 0.5 and d["k_counts"] == {"2": 3}
    po = d["per_order"]
    assert [e["order"] for e in po] == [0, 1]
    assert po[0]["most_common_letter"] == "A" and po[0]["most_common_letter_share"] == 1.0
    # s = 1 三条都选字母 A，A 在这种顺序下放的是内容 1（原字母 B）：gt 为 B 的前两条对
    assert po[1]["accuracy"]["value"] == pytest.approx(2 / 3) and po[1]["letter_counts"] == {"A": 3}
    assert m["readout"] == {"letter_debiased": 3} and m["parse_routes"] == {"letter_debiased": 3}


# ---------------------------------------------------------------------------
# 格式强制：允许集合、处理器、入口断言
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", params=FAMILY_KEYS)
def tiny(request):
    from src.adapters.image_policy import PixelBudget, SoftTokenBudget, TilePolicy

    pol = {"qwen3.5-0.8b": PixelBudget(64), "gemma-4-e2b-it": SoftTokenBudget(70),
           "internvl3.5-2b-hf": TilePolicy(crop_to_patches=True, max_patches=4)}[request.param]
    try:
        return make_adapter(request.param, device="cpu", dtype=torch.float32, image_policy=pol, tiny=True, init_std=0.05)
    except OSError as e:
        pytest.fail(f"取不到 {request.param} 的小文件：{e!r}", pytrace=False)


def _cti():
    spec = importlib.util.spec_from_file_location("_cti_for_posthoc", REPO / "scripts" / "check_training_interfaces.py")
    mod = importlib.util.module_from_spec(spec)
    saved = list(sys.path)
    spec.loader.exec_module(mod)
    sys.path[:] = saved
    return mod


def test_grid_format_ids_same_construction_as_training_interfaces(tiny):
    fmt = FC.grid_format_ids(tiny.tokenizer)
    ref = _cti().grid_allowed(tiny)
    assert fmt["ids"] == sorted(set(ref["ids"]) - set(ref["stop_ids"]))
    assert [r["id"] for r in fmt["tokens"]] == [r["id"] for r in ref["tokens"] if r["role"] != "stop"]
    assert not set(fmt["ids"]) & set(tiny.stop_token_ids())


def test_structure_processor_on_tiny_model(tiny):
    fmt = FC.grid_format_ids(tiny.tokenizer)
    rng = np.random.default_rng(3)
    from PIL import Image

    img = Image.fromarray(rng.integers(0, 256, (64, 80, 3), dtype=np.uint8))
    inputs = [tiny.build_inputs([img], "Segment the cat as a 16x16 binary grid.", "image_first"),
              tiny.build_inputs([], "Grid please, a long instruction so that padding differs.", "instruction_first")]
    proc = FC.GridStructureProcessor(fmt["digit_ids"], fmt["newline_id"])
    res = tiny.generate_batch(inputs, enable_thinking=False, max_new_tokens=PH.GRID_FIXED_TOKENS,
                              allowed_token_ids=fmt["ids"], logits_processor=[proc], do_sample=False)
    assert proc.n_calls == PH.GRID_FIXED_TOKENS
    for r in res:
        FC.check_fixed_output(r, variant="structured", fmt=fmt)
        lines = r.text.split("\n")
        assert len(lines) == GRID_SIZE_E1 and all(len(x) == GRID_SIZE_E1 and set(x) <= {"0", "1"} for x in lines)
    free = tiny.generate_batch(inputs, enable_thinking=False, max_new_tokens=PH.GRID_FIXED_TOKENS,
                               allowed_token_ids=fmt["ids"], do_sample=False)
    for r in free:
        FC.check_fixed_output(r, variant="restricted", fmt=fmt)


def test_check_fixed_output_catches_errors():
    fmt = {"ids": [5, 6, 7], "newline_id": 7}
    good = [5 if not FC.newline_position(t) else 7 for t in range(271)]

    def res(ids, stopped=False):
        return SimpleNamespace(token_ids=tuple(ids), stopped=stopped)

    FC.check_fixed_output(res(good), variant="structured", fmt=fmt)
    with pytest.raises(FC.ForcedOutputError, match="恰好生成"):
        FC.check_fixed_output(res(good[:-1]), variant="restricted", fmt=fmt)
    with pytest.raises(FC.ForcedOutputError, match="允许集合"):
        FC.check_fixed_output(res(good[:-1] + [9]), variant="restricted", fmt=fmt)
    with pytest.raises(FC.ForcedOutputError, match="换行"):
        FC.check_fixed_output(res([5] * 271), variant="structured", fmt=fmt)
    FC.check_fixed_output(res([5] * 271), variant="restricted", fmt=fmt)  # 主读法不管换行在哪里
    with pytest.raises(FC.ForcedOutputError, match="不停止"):
        FC.check_fixed_output(res(good, stopped=True), variant="restricted", fmt=fmt)


def test_structure_processor_single_use():
    p = FC.GridStructureProcessor([1, 2], 3)
    s = torch.zeros(2, 8)
    out = p(torch.zeros(2, 5, dtype=torch.long), s)
    assert torch.isinf(out[:, 3]).all() and not torch.isinf(out[:, 1]).any()
    with pytest.raises(FC.ForcedOutputError):
        p(torch.zeros(2, 5, dtype=torch.long), s)  # 同一个宽度再调一次：不是下一步


def test_parse_view_drops_stopped_only_for_fixed_length():
    from src.adapters.base import GenerationResult

    r = GenerationResult(text="0", token_ids=(1,), n_generated_tokens=1, n_prompt_tokens=2, n_visual_tokens=0,
                         wall_time_s=0.0, raw_text="0", stopped=False)
    assert FC.parse_view(r, fixed_length=True).stopped is None
    assert FC.parse_view(r, fixed_length=False) is r


# ---------------------------------------------------------------------------
# 事后 fixture
# ---------------------------------------------------------------------------


def _jsonl(p):
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def test_posthoc_fixture_matches_e1_fixture_except_documented_parts():
    e1, ph = FIXTURE_DIR / "root", FIXTURE_POSTHOC_DIR / "root"
    for name in ("nyuv2", "embspatial"):
        for f in sorted((e1 / name).rglob("*")):
            if f.is_file():
                assert (ph / name / f.relative_to(e1 / name)).read_bytes() == f.read_bytes(), f
    for name, fn in (("blink_reldepth", "val.jsonl"), ("cvbench", "test.jsonl")):
        a, b = _jsonl(e1 / name / fn), _jsonl(ph / name / fn)
        assert [(r["idx"], r["image"], r["image_sha256"]) for r in a] == [(r["idx"], r["image"], r["image_sha256"]) for r in b]
        for r in b:  # 提示是真实格式：最后 k 行是选项行
            DB.split_option_lines(r["prompt"], [chr(65 + i) for i in range(len(r["choices"]))], r["choices"],
                                  line_format="({letter}) {text}", joiner="\n")
            assert r["answer"] == f"({r['answer_letter']})" and r["answer_letter"] in [chr(65 + i) for i in range(len(r["choices"]))]
        for img in sorted((e1 / name / "images").rglob("*.jpg")):
            assert (ph / name / img.relative_to(e1 / name)).read_bytes() == img.read_bytes()
    assert {len(r["choices"]) for r in _jsonl(ph / "cvbench" / "test.jsonl")} == {2, 3, 4}
    man = json.loads((ph / "refcoco" / "MANIFEST.json").read_text())
    man0 = json.loads((e1 / "refcoco" / "MANIFEST.json").read_text())
    for rel, rec in man["files"].items():
        assert hashlib.sha256((ph / "refcoco" / rel).read_bytes()).hexdigest() == rec["sha256"]
        if rec["kind"] == "image":
            assert (rec["width"], rec["height"]) == (4 * man0["files"][rel]["width"], 4 * man0["files"][rel]["height"])
            assert rec["phash"] == man0["files"][rel]["phash"]
    for fn in ("refcoco_unc.json", "refcoco+_unc.json", "refcocog_umd.json"):
        a, b = json.loads((e1 / "refcoco" / fn).read_text()), json.loads((ph / "refcoco" / fn).read_text())
        assert a["refs"] == b["refs"] and a["categories"] == b["categories"]
        for k, ann in a["anns"].items():
            assert b["anns"][k]["bbox"] == [4 * v for v in ann["bbox"]]
            assert b["anns"][k]["segmentation"] == [[4 * v for v in poly] for poly in ann["segmentation"]]


def test_posthoc_fixture_same_samples_as_e1_fixture(tmp_path):
    a, b = DataPaths.fixtures(nyu_cache=tmp_path / "a"), DataPaths.posthoc_fixtures(nyu_cache=tmp_path / "b")
    for task in CHOICE_TASKS + ("refexp_poly", "grid16"):
        for sub in ("eval", "dev", "think"):
            assert [s.id for s in select(task, sub, a).samples] == [s.id for s in select(task, sub, b).samples], (task, sub)
