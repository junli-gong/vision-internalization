"""DEV-0064：大题量的语言读数（scripts/lang_battery.py）。

- 抽样：MMLU-Pro test 的总体（tests/fixtures/health/population_index.json，就是建健康检查题目时的那份原始文件的 (id, 学科)）
  去掉健康检查的 100 题，按 hash_rank(salt = lang_battery_v1) 分层取 2,000 题：可复现（与独立按 sha256 公式重算的结果相同、
  与输入顺序无关、id 清单的 sha256 钉住），与健康检查的 200 题不重叠；IFEval 全部 541 条，其中 50 条是健康检查的（逐题标出）。
- 题目字段与健康检查题目文件相同；原始文件的布局与 scripts/build_health_items.py 的 fetch 相同；真模型运行核对 sha256。
- 打分与健康检查同一套函数：在健康检查的题上，逐题记录与 src/health/run.run_one 的记录逐字段相同（墙钟除外），汇总与
  scoring.summarize 相同。
- 端到端（子进程，--tiny）：基座与带 --checkpoint（合并核对 → load_trained(merge=True)）各一次，输出齐全；--compare 的命令行。
- --compare 的配对：差值、丢题 / 得题、符号检验，配对 bootstrap CI 与独立实现的配对重采样相同；题目或数据不同时拒绝。
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from src.constants import BOOTSTRAP_N_RESAMPLES, BOOTSTRAP_SEED
from src.health import items as I
from src.health import protocol as P
from src.health.run import run_one
from src.health.scoring import paired_ci_pp, summarize
from tests import _lang_battery_fixtures as FX

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "lang_battery.py"
SYNTHETIC_NLTK_DATA = REPO / "tests/fixtures/health/nltk_data"
POP_INDEX = REPO / "tests/fixtures/health/population_index.json"
QWEN = "qwen3.5-0.8b"
LB = FX.load_script()
LETTER = P.MMLU_PRO_LETTER
# 2,000 题（question_id 升序）的指纹：抽样规则（盐、purpose、排除、分层）写死在这里，改了任何一项都会报出来
EXPECTED_MMLU_PRO_IDS_SHA256 = "3ddefcec39ad84c9c9dc955e1f41475031e70ab889dae8506b0c9b7b2bd7f6ea"
EXPECTED_IFEVAL_IDS_SHA256 = "731535915e370e828a0d5c3c256abbf3a1f9dedb4591f58a858360da223a437e"


@pytest.fixture(scope="module")
def health():
    obj, _ = I.load_items()
    return obj


@pytest.fixture(scope="module")
def population():
    pop, _, keys = I.expand_population_index(json.loads(POP_INDEX.read_text(encoding="utf-8")))
    return pop, keys


def _key(purpose: str, qid: int) -> str:
    return hashlib.sha256(f"lang_battery_v1|{purpose}|{qid}".encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# 抽样
# ---------------------------------------------------------------------------


def test_mmlu_pro_selection_reproducible_and_disjoint_from_health(health, population):
    pop, _ = population
    assert len(pop) == P.SOURCES["mmlu_pro"].n_rows
    hids = [x["id"] for x in health["items"]["mmlu_pro"]]
    chosen = LB.select_mmlu_pro(pop, 2000, exclude=hids)
    assert len(chosen) == 2000 and Counter(chosen.values()) == {"stratified": 1988, "fill": 12}
    cats = dict(pop)
    assert set(Counter(cats[q] for q, s in chosen.items() if s == "stratified").values()) == {142}
    assert len(set(Counter(cats[q] for q in chosen))) == 14
    # 与健康检查的 200 题不重叠（MMLU-Pro 的 100 题被排除在总体之外；GSM8K 不在本 battery 里）
    assert not set(chosen) & set(hids)
    # 独立重算：sha256("lang_battery_v1|<purpose>|<question_id>") 升序，各学科前 142，其余前 12
    pool = [(q, c) for q, c in pop if q not in set(hids)]
    want = {}
    for cat in sorted({c for _, c in pool}):
        for q in sorted((q for q, c in pool if c == cat), key=lambda q: _key("mmlu_pro_stratified", q))[:142]:
            want[q] = "stratified"
    for q in sorted((q for q, _ in pool if q not in want), key=lambda q: _key("mmlu_pro_fill", q))[:12]:
        want[q] = "fill"
    assert chosen == want
    # 与输入顺序无关，两次相同；id 清单钉住
    assert LB.select_mmlu_pro(list(reversed(pop)), 2000, exclude=reversed(hids)) == chosen
    assert LB.ids_sha256(sorted(chosen)) == EXPECTED_MMLU_PRO_IDS_SHA256
    # 不排除的话会选中健康检查的 17 题：排除确实起作用
    assert len(set(LB.select_mmlu_pro(pop, 2000)) & set(hids)) == 17


def test_ifeval_is_all_541_with_50_from_health(health, population):
    _, keys = population
    raw = {"mmlu_pro": {"rows": _dummy_mmlu_rows(population[0]), "path": "m", "sha256": "x", "matches_pinned": True},
           "ifeval": {"rows": _dummy_ifeval_rows(keys), "path": "i", "sha256": "y", "matches_pinned": True}}
    items, metas, data = LB.build_battery(raw, n_mmlu_pro=2000, health_items=health, pinned=True)
    assert [it["id"] for it in items["ifeval"]] == sorted(keys) and len(keys) == 541
    hk = {x["id"] for x in health["items"]["ifeval"]}
    assert [m["in_health_check"] for m in metas["ifeval"]] == [k in hk for k in sorted(keys)]
    assert data["ifeval"]["health_check"]["n_overlap"] == 50 == len(data["ifeval"]["health_check"]["overlap_ids"])
    assert data["ifeval"]["ids_sha256"] == EXPECTED_IFEVAL_IDS_SHA256
    # MMLU-Pro：真模型路径（pinned）的排除与记录
    mp = data["mmlu_pro"]
    assert mp["health_check"] == {"n_items": 100, "n_in_population": 100, "n_excluded_from_population": 100, "n_overlap": 0}
    assert mp["ids"] == [it["id"] for it in items["mmlu_pro"]] == sorted(mp["ids"])
    assert mp["ids_sha256"] == EXPECTED_MMLU_PRO_IDS_SHA256 and mp["n_items"] == 2000
    assert [m["category"] for m in metas["mmlu_pro"]] == [it["category"] for it in items["mmlu_pro"]]
    assert sum(v.get("fill", 0) for v in mp["per_category"].values()) == 12
    # 总体里缺了健康检查的题（不是同一份数据）：真模型拒绝，--tiny 只记录
    rows = [r for r in raw["mmlu_pro"]["rows"] if r["question_id"] != health["items"]["mmlu_pro"][0]["id"]]
    short = {**raw, "mmlu_pro": {**raw["mmlu_pro"], "rows": rows}}
    with pytest.raises(LB.DataError, match="不是同一份"):
        LB.build_battery(short, n_mmlu_pro=2000, health_items=health, pinned=True)
    assert LB.build_battery(short, n_mmlu_pro=2000, health_items=health, pinned=False)[2]["mmlu_pro"]["health_check"][
        "n_in_population"] == 99


def _dummy_mmlu_rows(pop):
    return [{"question_id": q, "category": c, "src": "x", "question": f"Q{q}", "options": ["a", "b", "c", "d"],
             "answer": "A", "answer_index": 0, "cot_content": ""} for q, c in pop]


def _dummy_ifeval_rows(keys):
    return [{"key": k, "prompt": f"p{k}", "instruction_id_list": ["punctuation:no_comma"], "kwargs": [{}]} for k in keys]


def test_selection_errors():
    cats = [f"c{i}" for i in range(14)]
    pop = [(10 * i + j, c) for i, c in enumerate(cats) for j in range(2)]
    for bad in (0, -1, True, 2.0):
        with pytest.raises(LB.DataError, match="正整数"):
            LB.select_mmlu_pro(pop, bad)
    with pytest.raises(LB.DataError, match="不够每学科"):
        LB.select_mmlu_pro(pop, 42)
    with pytest.raises(LB.DataError, match="14 个学科"):
        LB.select_mmlu_pro(pop[:-2], 14)
    one = [(i, c) for i, c in enumerate(cats)]
    with pytest.raises(LB.DataError, match="补足"):
        LB.select_mmlu_pro(one, 15)
    # 排除之后某学科不够
    with pytest.raises(LB.DataError, match="不够每学科"):
        LB.select_mmlu_pro(pop, 28, exclude=[0])
    assert set(LB.select_mmlu_pro(pop, 28)) == {q for q, _ in pop}


# ---------------------------------------------------------------------------
# 题目字段、原始文件
# ---------------------------------------------------------------------------


def test_item_fields_match_health_items(health):
    for it in health["items"]["mmlu_pro"]:
        row = {"question_id": it["id"], "question": it["question"], "options": list(it["options"]), "answer": it["answer"],
               "answer_index": it["answer_index"], "cot_content": "", "category": it["category"], "src": it["src"]}
        assert LB.mmlu_pro_item(row, it["stage"]) == it
        assert P.mmlu_pro_prompt(LB.mmlu_pro_item(row, "fill")) == P.mmlu_pro_prompt(it)
    for it in health["items"]["ifeval"]:
        row = {"key": it["id"], "prompt": it["prompt"], "instruction_id_list": list(it["instruction_id_list"]),
               "kwargs": copy.deepcopy(it["kwargs"])}
        assert LB.ifeval_item(row) == it


def test_load_raw_pins_files(tmp_path):
    raw_dir = FX.write_tiny_raw_dir(tmp_path / "raw")
    raw = LB.load_raw(raw_dir, pinned=False)
    assert len(raw["mmlu_pro"]["rows"]) == 70 and len(raw["ifeval"]["rows"]) == 52
    assert raw["mmlu_pro"]["matches_pinned"] is False and raw["mmlu_pro"]["sha256"] == hashlib.sha256(
        LB.raw_path(raw_dir, "mmlu_pro").read_bytes()).hexdigest()
    with pytest.raises(LB.DataError, match="sha256"):
        LB.load_raw(raw_dir, pinned=True)
    with pytest.raises(LB.DataError, match="--fetch"):
        LB.load_raw(tmp_path / "missing", pinned=False)
    assert LB.raw_path("R", "mmlu_pro") == Path("R/TIGER-Lab__MMLU-Pro/data/test-00000-of-00001.parquet")
    assert LB.raw_path("R", "ifeval") == Path("R/google__IFEval/ifeval_input_data.jsonl")


def test_raw_layout_is_build_health_items_fetch(tmp_path, monkeypatch):
    """--raw-dir 的布局就是 build_health_items.fetch 的：文件放在 raw_path 上时 fetch 直接核对它的 sha256（不下载）。"""
    import huggingface_hub

    def no_download(*a, **k):
        raise AssertionError(f"fetch 没在 raw_path 上找到文件，想去下载：{a} {k}")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", no_download)
    build = LB._load_script(LB.BUILD_ITEMS_SCRIPT, "_build_health_items_under_test")
    for name in LB.BENCHMARKS:
        p = LB.raw_path(tmp_path, name)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"not the real file")
        with pytest.raises(SystemExit, match="sha256"):
            build.fetch(name, tmp_path)


# ---------------------------------------------------------------------------
# 与健康检查同一套函数：逐题相同
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_adapter():
    hc = LB.health_check_script()
    ad, _, _ = hc.make_adapter(hc.parse_args(["--tiny", "--model", QWEN, "--out", "unused.json"]))
    return ad


def _no_wall(rec):
    return {k: v for k, v in rec.items() if k != "wall_time_s"}


def test_records_identical_to_health_check(tiny_adapter, health, monkeypatch):
    import nltk

    monkeypatch.setattr(nltk.data, "path", [str(SYNTHETIC_NLTK_DATA), *nltk.data.path])  # 自造的空 punkt_tab（同 test_health_run）
    ad = tiny_adapter
    _, spellings = LB.warmup(ad)
    mmlu, ife = health["items"]["mmlu_pro"][:4], health["items"]["ifeval"][:2]
    ours = {LETTER: [], "ifeval": []}
    theirs = {LETTER: [], "ifeval": []}
    for it in mmlu:
        rec, comp = LB.run_letter(ad, it, spellings)
        ref = run_one(ad, "mmlu_pro", it, spellings=spellings)[0][LETTER]
        assert list(rec) == list(ref) and _no_wall(rec) == _no_wall(ref)
        assert (comp.n_samples, comp.n_calls, comp.generated_tokens) == (1, 1, P.LETTER_MAX_NEW_TOKENS)
        ours[LETTER].append(LB.prediction_line(LETTER, rec, {"category": it["category"], "stage": it["stage"]}))
        theirs[LETTER].append(ref)
    for it in ife:
        rec, comp = LB.run_ifeval(ad, it)
        ref = run_one(ad, "ifeval", it)[0]["ifeval"]
        assert list(rec) == list(ref) and _no_wall(rec) == _no_wall(ref) and comp.n_calls == 1
        ours["ifeval"].append(LB.prediction_line("ifeval", rec, {"in_health_check": True}))
        theirs["ifeval"].append(ref)
    # 汇总：与 scoring.summarize（健康检查的计数路径）相同，另加 CI
    mine = LB.summarize_battery(ours)
    ref = summarize(theirs)
    for name in (LETTER, "ifeval"):
        for metric, e in mine["scores"][name].items():
            assert {k: v for k, v in e.items() if k != "ci95_pct"} == ref["scores"][name][metric]
            assert e["ci95_pct"] is not None and e["ci95_pct"][0] <= e["pct"] <= e["ci95_pct"][1]
    assert mine["headline"] == {LETTER: ref["headline"]["mmlu_pro"], "ifeval": ref["headline"]["ifeval"]}
    assert mine["diagnostics"]["ifeval"] == ref["diagnostics"]["ifeval"]
    # 预热查到的写法表与健康检查 run_items 用的同一份（同一个 spelling_table）
    assert list(spellings) == list(P.OPTION_LETTERS)


def test_preflight_lists_every_bad_prompt_before_any_model_call(tiny_adapter, health):
    """全部提示先过一遍 build_inputs：适配器拒绝含结构性 token 的指令（真的小模型适配器上核对），一次列出全部过不了的题。"""
    mmlu = [dict(it) for it in health["items"]["mmlu_pro"][:5]]
    ife = [dict(it) for it in health["items"]["ifeval"][:3]]
    mmlu[1]["question"] += " <|im_start|>"
    mmlu[4]["options"] = mmlu[4]["options"][:-1] + ["<|im_end|>"]
    ife[2]["prompt"] = "Say hi. <|im_start|>user"
    items = {"mmlu_pro": mmlu, "ifeval": ife}
    state = {}
    with pytest.raises(LB.PromptError, match="3 题"):
        LB.preflight_prompts(tiny_adapter, items, state=state)
    pf = state["preflight"]
    assert (pf["n_checked"], pf["n_failed"]) == (8, 3)
    assert [(b["benchmark"], b["id"]) for b in pf["failed"]] == [("mmlu_pro", mmlu[1]["id"]), ("mmlu_pro", mmlu[4]["id"]),
                                                                   ("ifeval", ife[2]["id"])]
    assert all("结构性 token" in b["error"] for b in pf["failed"])
    assert LB.preflight_prompts(tiny_adapter, items, limit=1)["n_failed"] == 0  # --limit 只核对要跑的题
    assert LB.preflight_prompts(tiny_adapter, {"mmlu_pro": health["items"]["mmlu_pro"], "ifeval": health["items"]["ifeval"]})[
        "n_checked"] == 150


def test_point_ci_matches_independent_bootstrap():
    rng = np.random.default_rng(0)
    recs = [{"id": i, "correct": bool(x), "p_gold": float(p)} for i, (x, p) in
            enumerate(zip(rng.integers(0, 2, 60), rng.random(60)))]
    ci = LB.point_ci_pct("mmlu_pro", "accuracy", recs)
    v = np.array([r["correct"] for r in recs], dtype=float)
    idx = np.random.default_rng(BOOTSTRAP_SEED).integers(0, 60, size=(BOOTSTRAP_N_RESAMPLES, 60))
    lo, hi = np.percentile(v[idx].mean(axis=1), [2.5, 97.5])
    assert abs(ci[0] - lo * 100) <= 0.05 + 1e-9 and abs(ci[1] - hi * 100) <= 0.05 + 1e-9
    p = np.array([r["p_gold"] for r in recs])
    lo, hi = np.percentile(p[idx].mean(axis=1), [2.5, 97.5])
    ci = LB.point_ci_pct("mmlu_pro", "mean_p_gold", recs)
    assert abs(ci[0] - lo * 100) <= 0.05 + 1e-9 and abs(ci[1] - hi * 100) <= 0.05 + 1e-9
    assert LB.point_ci_pct("mmlu_pro", "accuracy", []) is None


# ---------------------------------------------------------------------------
# 端到端（子进程，与服务器同一流程）
# ---------------------------------------------------------------------------


def _run(args, tmp: Path, *, ok: bool = True, timeout: int = 1800):
    env = {**os.environ, "NLTK_DATA": str(SYNTHETIC_NLTK_DATA), "HF_HUB_OFFLINE": "1", "TMPDIR": str(tmp)}
    env.pop("VISINT_DATA_ROOT", None)
    p = subprocess.run([sys.executable, str(SCRIPT), *args], cwd=REPO, env=env, capture_output=True, text=True, timeout=timeout)
    if ok:
        assert p.returncode == 0, p.stderr[-3000:]
    return p


def _make_checkpoint(dest: Path) -> Path:
    """小模型 + LoRA（peft 默认初始化：B = 0，合并前后相同）→ save_trained；构造方式同 health_check 的 make_adapter。"""
    hc = LB.health_check_script()
    ad, _, _ = hc.make_adapter(hc.parse_args(["--tiny", "--model", QWEN, "--out", "unused.json"]))
    ad.inject_lora(r=4, lora_alpha=8.0)
    return ad.save_trained(dest)


@pytest.fixture(scope="module")
def e2e(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("lang_battery")
    raw = FX.write_tiny_raw_dir(tmp / "raw")
    base, ck_out = tmp / "base", tmp / "ckpt"
    procs = {"base": _run(["--tiny", "--model", QWEN, "--raw-dir", str(raw), "--limit", "3", "--out", str(base)], tmp)}
    ck = _make_checkpoint(tmp / "ck")
    procs["ckpt"] = _run(["--tiny", "--model", QWEN, "--raw-dir", str(raw), "--limit", "3", "--checkpoint", str(ck),
                          "--label", "tiny_lora", "--out", str(ck_out)], tmp)
    return {"tmp": tmp, "raw": raw, "base": base, "ckpt": ck_out, "ck": ck, "procs": procs}


def _read_preds(d: Path):
    return [json.loads(x) for x in (d / "predictions.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]


def test_end_to_end_tiny(e2e, health):
    d = e2e["base"]
    for name in LB.OUTPUT_FILES:
        assert (d / name).is_file(), name
    rep = json.loads((d / "lang_battery.json").read_text(encoding="utf-8"))
    assert rep["ok"] is True and (rep["mode"], rep["limit"], rep["battery_version"]) == ("tiny", 3, LB.BATTERY_VERSION)
    assert rep["model"]["key"] == QWEN and "checkpoint" not in rep
    gen = rep["generation"]
    assert (gen["enable_thinking"], gen["order"], gen["n_images"], gen["greedy_kwargs"]) == (
        False, "instruction_first", 0, dict(P.GREEDY_KWARGS))
    assert gen["readings"] == {LETTER: {"benchmark": "mmlu_pro", "kind": "option_letter", "max_new_tokens": 1},
                               "ifeval": {"benchmark": "ifeval", "kind": "generate", "max_new_tokens": 1024}}
    mp = rep["data"]["mmlu_pro"]
    assert (mp["n_population"], mp["n_items"], mp["sampling"]["n"], mp["sampling"]["salt"]) == (70, 28, 28, "lang_battery_v1")
    assert mp["matches_pinned"] is False and mp["health_check"]["n_overlap"] == 0
    assert rep["data"]["ifeval"]["n_items"] == 52 and rep["data"]["ifeval"]["health_check"]["n_overlap"] == 50
    assert rep["data"]["health_check_items_file"]["sha256"] == P.ITEMS_SHA256
    preds = _read_preds(d)
    assert [p["reading"] for p in preds] == [LETTER] * 3 + ["ifeval"] * 3
    assert [p["id"] for p in preds[:3]] == mp["ids"][:3]
    ife_ids = sorted(r["key"] for r in FX.health_ifeval_rows() + list(FX.EXTRA_IFEVAL))
    assert [p["id"] for p in preds[3:]] == ife_ids[:3]
    hk = {x["id"] for x in health["items"]["ifeval"]}
    assert all(p["meta"] == {"in_health_check": p["id"] in hk} for p in preds[3:])
    assert all(set(p["meta"]) == {"category", "stage"} for p in preds[:3])
    assert rep["predictions"]["sha256"] == hashlib.sha256((d / "predictions.jsonl").read_bytes()).hexdigest()
    # 分数从逐题记录重算一致；每个读数都有 CI
    again = LB.summarize_battery({LETTER: preds[:3], "ifeval": preds[3:]})
    assert again["scores"] == rep["scores"] and again["headline"] == rep["headline"]
    assert all(e["ci95_pct"] is not None for s in rep["scores"].values() for e in s.values())
    assert rep["scores"]["ifeval"]["instruction_level_strict"]["n"] == sum(len(p["instruction_id_list"]) for p in preds[3:])
    # 记账：MMLU-Pro 每题一次 option_logits，IFEval 每题一次生成；compute.json 是总量；预热不计
    comp = json.loads((d / "compute.json").read_text())
    assert comp == rep["compute"]["total"] and (comp["n_samples"], comp["n_calls"], comp["visual_tokens"]) == (6, 6, 0)
    assert rep["compute"]["mmlu_pro"]["generated_tokens"] == 3 and rep["warmup"]["counted_in_compute"] is False
    assert (rep["preflight"]["n_checked"], rep["preflight"]["n_failed"]) == (6, 0)
    assert rep["scoring"]["ifeval"]["scored"] is True and rep["scoring"]["ifeval"]["punkt_tab"]["matches_pinned"] is False
    assert rep["run"]["config"]["mmlu_pro_n"] == LB.TINY_MMLU_PRO_N and rep["run"]["meta"]["data_sha256"]
    readme = (d / "README.md").read_text(encoding="utf-8")
    assert "lang_battery_v1" in readme and "不再分发" in readme and "mmlu_pro_letter | accuracy" in readme
    # 同一个 --out 不覆盖
    p = _run(["--tiny", "--model", QWEN, "--raw-dir", str(e2e["raw"]), "--limit", "1", "--out", str(d)], e2e["tmp"], ok=False)
    assert p.returncode == 2 and "拒绝覆盖" in p.stderr


def test_end_to_end_checkpoint(e2e):
    d, ck = e2e["ckpt"], e2e["ck"]
    rep = json.loads((d / "lang_battery.json").read_text(encoding="utf-8"))
    man_sha = hashlib.sha256((ck / "manifest.json").read_bytes()).hexdigest()
    assert rep["ok"] is True and rep["label"] == "tiny_lora"
    assert rep["checkpoint"]["manifest_sha256"] == man_sha and rep["checkpoint"]["merge_check"]["pass"] is True
    assert rep["run"]["config"]["checkpoint"]["manifest_sha256"] == man_sha
    assert json.loads((d / "lang_battery_merge_check.json").read_text())["pass"] is True
    # B = 0 的 LoRA：合并后与 base 相同，逐题输出相同
    base = _read_preds(e2e["base"])
    assert [_no_wall(p) for p in _read_preds(d)] == [_no_wall(p) for p in base]
    # 构造方式与检查点不符：拒绝（与 health_check --checkpoint 同一处核对）
    p = _run(["--tiny", "--model", "gemma-4-e2b-it", "--raw-dir", str(e2e["raw"]), "--limit", "1", "--checkpoint", str(ck),
              "--out", str(e2e["tmp"] / "bad")], e2e["tmp"], ok=False)
    assert p.returncode == 2 and "构造方式" in json.loads((e2e["tmp"] / "bad" / "lang_battery.json").read_text())["error"]


def test_failure_exit_codes_keep_evidence(e2e, tmp_path, monkeypatch, capsys):
    """合并核对不通过 → 退出码 1；中途出错 → 退出码 2；两种都先写出 lang_battery.json（已跑的题留在 predictions.jsonl）。"""
    import types

    import nltk

    from src.e3.merge_check import MergeCheckFailed

    monkeypatch.setattr(nltk.data, "path", [str(SYNTHETIC_NLTK_DATA), *nltk.data.path])
    real = LB.health_check_script()

    def refuse(_args):
        raise MergeCheckFailed("合并前后不一致（测试注入）", {"pass": False})

    monkeypatch.setattr(LB, "health_check_script", lambda: types.SimpleNamespace(
        **{k: getattr(real, k) for k in ("check_ifeval_ready", "make_adapter", "scoring_block", "IFEVAL_PACKAGES")},
        make_checkpoint_adapter=refuse))
    out = tmp_path / "refused"
    rc = LB.main(["--tiny", "--model", QWEN, "--raw-dir", str(e2e["raw"]), "--checkpoint", str(e2e["ck"]), "--out", str(out)])
    rep = json.loads((out / "lang_battery.json").read_text(encoding="utf-8"))
    assert rc == 1 and rep["ok"] is False and "测试注入" in rep["error"] and "拒绝评测" in capsys.readouterr().err
    monkeypatch.setattr(LB, "health_check_script", lambda: real)
    calls = []

    def flaky(adapter, item):
        calls.append(item["id"])
        if len(calls) == 2:
            raise RuntimeError("boom")
        records, compute = run_one(adapter, "ifeval", item)
        return records["ifeval"], compute

    monkeypatch.setattr(LB, "run_ifeval", flaky)
    out = tmp_path / "broken"
    rc = LB.main(["--tiny", "--model", QWEN, "--raw-dir", str(e2e["raw"]), "--limit", "3", "--out", str(out)])
    rep = json.loads((out / "lang_battery.json").read_text(encoding="utf-8"))
    assert rc == 2 and rep["ok"] is False and "boom" in rep["error"] and rep["n_done"] == {LETTER: 3, "ifeval": 1}
    assert [p["reading"] for p in _read_preds(out)] == [LETTER] * 3 + ["ifeval"]
    assert rep["compute"]["mmlu_pro"]["n_samples"] == 3 and "total" not in rep["compute"]


def test_compare_cli(e2e):
    out = e2e["tmp"] / "cmp.json"
    p = _run(["--compare", str(e2e["base"]), str(e2e["ckpt"] / "lang_battery.json"), "--out", str(out)], e2e["tmp"])
    res = json.loads(out.read_text(encoding="utf-8"))
    assert res["base"]["label"] == QWEN and res["other"]["label"] == "tiny_lora" and res["partial_run"] is True
    assert [(r["reading"], r["metric"]) for r in res["rows"]] == [(LETTER, "accuracy"), (LETTER, "mean_p_gold")] + [
        ("ifeval", m) for m in ("prompt_level_strict", "instruction_level_strict", "prompt_level_loose", "instruction_level_loose")]
    assert all(r["delta_pp"] == 0.0 and r["ci95_pp"] == [0.0, 0.0] for r in res["rows"])
    assert res["mmlu_pro_by_category"] and "CI95=[+0.0, +0.0]" in p.stdout


# ---------------------------------------------------------------------------
# --compare 的配对
# ---------------------------------------------------------------------------


def _write_run(d: Path, letter: list[dict], ifeval: list[dict], *, ids_sha: str = "sel", label: str = "x") -> Path:
    d.mkdir(parents=True)
    lines = [{"reading": LETTER, **r} for r in letter] + [{"reading": "ifeval", **r} for r in ifeval]
    (d / "predictions.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines), encoding="utf-8")
    doc = {"ok": True, "battery_version": LB.BATTERY_VERSION, "label": label, "limit": None, "model": {"key": "m"},
           "headline": {}, "data": {"mmlu_pro": {"sha256": "a", "ids_sha256": ids_sha, "sampling": {"n": 80}},
                                     "ifeval": {"sha256": "b", "ids_sha256": "c"}},
           "predictions": {"sha256": hashlib.sha256((d / "predictions.jsonl").read_bytes()).hexdigest()}}
    (d / "lang_battery.json").write_text(json.dumps(doc), encoding="utf-8")
    return d


def _synthetic(seed: int, n_letter=80, n_ife=40):
    rng = np.random.default_rng(seed)
    letter = [{"id": 1000 + i, "correct": bool(rng.random() < 0.6), "p_gold": float(rng.random()),
               "meta": {"category": f"c{i % 3}", "stage": "stratified"}} for i in range(n_letter)]
    ife = []
    for i in range(n_ife):
        k = int(rng.integers(1, 4))
        fs, fl = [bool(rng.random() < 0.7) for _ in range(k)], [bool(rng.random() < 0.8) for _ in range(k)]
        ife.append({"id": 5000 + i, "instruction_id_list": ["x"] * k, "strict": {"follow_all": all(fs), "follow_list": fs},
                    "loose": {"follow_all": all(fl), "follow_list": fl}, "meta": {"in_health_check": False}})
    return letter, ife


def test_compare_pairing(tmp_path):
    lb, ib = _synthetic(1)
    lo, io = _synthetic(2)
    for a, b in zip(lo, lb):  # 同一批题
        a["id"], a["meta"] = b["id"], b["meta"]
    for a, b in zip(io, ib):
        a["id"], a["instruction_id_list"] = b["id"], b["instruction_id_list"]
        for side in ("strict", "loose"):
            fl = (a[side]["follow_list"] + [True] * 3)[:len(b["instruction_id_list"])]
            a[side] = {"follow_all": all(fl), "follow_list": fl}
    base = _write_run(tmp_path / "b", lb, ib, label="base")
    other = _write_run(tmp_path / "o", lo, io, label="other")
    res = LB.compare_runs(base, other)
    rows = {(r["reading"], r["metric"]): r for r in res["rows"]}
    acc = rows[(LETTER, "accuracy")]
    vb = np.array([r["correct"] for r in lb], dtype=float)
    vo = np.array([r["correct"] for r in lo], dtype=float)
    assert acc["n"] == 80 and acc["delta_pp"] == round(float(Fraction(int(vo.sum() - vb.sum()) * 100, 80)), 1)
    assert acc["lost"] == int(((vb == 1) & (vo == 0)).sum()) and acc["gained"] == int(((vb == 0) & (vo == 1)).sum())
    assert acc["sign_test"]["n_better"] == acc["gained"] and acc["sign_test"]["n_worse"] == acc["lost"]
    # 配对 bootstrap：同一组下标同时作用于两边（独立实现），OTHER − BASE
    idx = np.random.default_rng(BOOTSTRAP_SEED).integers(0, 80, size=(BOOTSTRAP_N_RESAMPLES, 80))
    lo_, hi_ = np.percentile(vo[idx].mean(axis=1) - vb[idx].mean(axis=1), [2.5, 97.5])
    assert abs(acc["ci95_pp"][0] - lo_ * 100) <= 0.05 + 1e-9 and abs(acc["ci95_pp"][1] - hi_ * 100) <= 0.05 + 1e-9
    # mean_p_gold：均值之差与配对 CI；没有丢题、符号检验
    mp = rows[(LETTER, "mean_p_gold")]
    d = np.array([r["p_gold"] for r in lo]) - np.array([r["p_gold"] for r in lb])
    assert abs(mp["delta_pp"] - d.mean() * 100) <= 0.05 + 1e-9 and mp["lost"] is None and mp["sign_test"] is None
    lo_, hi_ = np.percentile(d[idx].mean(axis=1), [2.5, 97.5])
    assert abs(mp["ci95_pp"][0] - lo_ * 100) <= 0.05 + 1e-9 and abs(mp["ci95_pp"][1] - hi_ * 100) <= 0.05 + 1e-9
    # IFEval instruction-level：按题配对、Σ遵循 / Σ指令（与健康检查 --compare 的同一个函数）
    il = rows[("ifeval", "instruction_level_strict")]
    cb = [(sum(r["strict"]["follow_list"]), len(r["strict"]["follow_list"])) for r in ib]
    co = [(sum(r["strict"]["follow_list"]), len(r["strict"]["follow_list"])) for r in io]
    assert il["n"] == sum(n for _, n in cb) and il["ci95_pp"] == list(paired_ci_pp("instruction_level_strict", cb, co))
    assert il["delta_pp"] == round(float(Fraction(100 * (sum(c for c, _ in co) - sum(c for c, _ in cb)), il["n"])), 1)
    # 分学科
    cats = {r["category"]: r for r in res["mmlu_pro_by_category"]}
    assert sorted(cats) == ["c0", "c1", "c2"] and sum(r["n"] for r in cats.values()) == 80
    # 反过来比较：差值变号
    rev = {(r["reading"], r["metric"]): r for r in LB.compare_runs(other, base)["rows"]}
    assert rev[(LETTER, "accuracy")]["delta_pp"] == -acc["delta_pp"]
    assert (rev[(LETTER, "accuracy")]["lost"], rev[(LETTER, "accuracy")]["gained"]) == (acc["gained"], acc["lost"])


def test_compare_refuses_unpaired(tmp_path):
    lb, ib = _synthetic(1)
    base = _write_run(tmp_path / "b", lb, ib)
    with pytest.raises(LB.CompareError, match="题目或数据不同"):
        LB.compare_runs(base, _write_run(tmp_path / "s", lb, ib, ids_sha="another"))
    with pytest.raises(LB.CompareError, match="题目不同"):
        LB.compare_runs(base, _write_run(tmp_path / "l", lb[:-1], ib))
    tampered = _write_run(tmp_path / "t", lb, ib)
    (tampered / "predictions.jsonl").write_text((tampered / "predictions.jsonl").read_text().replace("true", "false", 1))
    with pytest.raises(LB.CompareError, match="sha256"):
        LB.compare_runs(base, tampered)
    failed = _write_run(tmp_path / "f", lb, ib)
    doc = json.loads((failed / "lang_battery.json").read_text())
    (failed / "lang_battery.json").write_text(json.dumps({**doc, "ok": False}))
    with pytest.raises(LB.CompareError, match="跑完"):
        LB.compare_runs(base, failed)
    p = subprocess.run([sys.executable, str(SCRIPT), "--compare", str(base), str(failed)], cwd=REPO, capture_output=True, text=True)
    assert p.returncode == 2 and "比较失败" in p.stderr


# ---------------------------------------------------------------------------
# 命令行规则
# ---------------------------------------------------------------------------


def test_parse_args_rules(tmp_path):
    a = LB.parse_args(["--tiny", "--raw-dir", "r", "--out", "o"])
    assert (a.device, a.dtype, a.mmlu_pro_n, a.model) == ("cpu", "fp32", LB.TINY_MMLU_PRO_N, None)
    a = LB.parse_args(["--model", "qwen3.5-9b", "--raw-dir", "r", "--out", "o"])
    assert (a.device, a.dtype, a.mmlu_pro_n) == ("cuda", "bf16", LB.DEFAULT_MMLU_PRO_N) and LB.DEFAULT_MMLU_PRO_N == 2000
    assert LB.parse_args(["--compare", "a", "b", "--out", "c.json"]).compare == ["a", "b"]
    assert LB.parse_args(["--fetch", "--raw-dir", "r"]).fetch is True
    ck = tmp_path / "ck"
    ck.mkdir()
    for bad in (["--tiny", "--out", "o"], ["--tiny", "--raw-dir", "r"], ["--raw-dir", "r", "--out", "o"],
                ["--tiny", "--raw-dir", "r", "--out", "o", "--limit", "0"],
                ["--tiny", "--raw-dir", "r", "--out", "o", "--mmlu-pro-n", "0"],
                ["--tiny", "--raw-dir", "r", "--out", "o", "--merge-check-n", "3"],
                ["--tiny", "--raw-dir", "r", "--out", "o", "--checkpoint", str(ck)],
                ["--tiny", "--model", QWEN, "--raw-dir", "r", "--out", "o", "--checkpoint", str(tmp_path / "nope")],
                ["--compare", "a", "b", "--tiny"], ["--compare", "a", "b", "--raw-dir", "r"], ["--fetch"],
                ["--fetch", "--raw-dir", "r", "--out", "o"], ["--fetch", "--compare", "a", "b"]):
        with pytest.raises(SystemExit):
            LB.parse_args(bad)


def test_health_check_script_is_reused_not_copied():
    """模型加载、合并核对、IFEval 依赖核对与评分记录都从 scripts/health_check.py 来（按路径载入，不复制）。"""
    hc = LB.health_check_script()
    src = SCRIPT.read_text(encoding="utf-8")
    for fn in ("make_adapter", "make_checkpoint_adapter", "check_ifeval_ready", "scoring_block"):
        assert callable(getattr(hc, fn)) and f"hc.{fn}(" in src
    assert "def make_checkpoint_adapter" not in src and "def vision_policy" not in src
    args = LB.health_args(LB.parse_args(["--tiny", "--raw-dir", "r", "--out", "o"]), Path("o"))
    assert hc.merge_check_path(args.out) == Path("o/lang_battery_merge_check.json")
