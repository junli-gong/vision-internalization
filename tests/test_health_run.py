"""健康检查的运行流程（src/health/run.py 与 scripts/health_check.py，DEV-0008）。

- 假适配器：核对每次调用都按协议（零张图、instruction_first、关思考、greedy 参数、各项的 max_new_tokens），分数与记账
  从逐题记录数出来，预热不计入，中途出错时已有结果留在 state 里。
- 三家族的真实 tokenizer / 模板（小型随机模型）：200 题的提示都能零张图 build_inputs，指令区间解码回来就是提示。
- 端到端：三家族各跑 `--tiny --limit 3`，退出码 0，输出 JSON 的结构齐全；`--compare` 命令行；真模型模式在加载模型之前
  就核对 punkt_tab。子进程的 NLTK_DATA 指向 tests/fixtures/health/nltk_data：一份自造的空 punkt_tab（不是 nltk 的数据，
  哈希与协议钉住的不同），--tiny 只记录哈希，所以彩排不依赖下载；真模型模式会拒绝它。
- 先生成、后评分（审查第 1 轮）：--skip-ifeval-scoring 在拿不到 IFEval 依赖与 punkt_tab 时照样跑完；--rescore 与
  跑完即评的结果逐字节相同（子进程的 --tiny 输出，与假适配器上的全部 200 题）。
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from src.adapters import registry
from src.adapters.base import GenerationResult, ModelInputs, Span
from src.health import protocol as P
from src.health.items import load_items
from src.health.run import RescoreError, TextOnlyViolation, rescore_records, run_items
from src.health.scoring import IFEVAL_SKIP_REASON, SCORED_SECTIONS, compare, ifeval_follow, summarize

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts/health_check.py"
SYNTHETIC_NLTK_DATA = REPO_ROOT / "tests/fixtures/health/nltk_data"
FAMILY_KEYS = ("qwen3.5-0.8b", "gemma-4-e2b-it", "internvl3.5-2b-hf")


@pytest.fixture(scope="module")
def items():
    obj, _ = load_items()
    return obj


# ---------------------------------------------------------------------------
# 假适配器
# ---------------------------------------------------------------------------


class FakeAdapter:
    """只实现 build_inputs / generate：回答由 responder(prompt) 给出，调用逐条记下。"""

    def __init__(self, responder, *, visual_tokens=0, fail_on_call=None):
        self.responder, self.visual_tokens, self.fail_on_call = responder, visual_tokens, fail_on_call
        self.calls = []

    def build_inputs(self, images, instruction, order):
        self.calls.append(("build_inputs", list(images), instruction, order))
        return ModelInputs(tensors={"prompt": instruction}, order=order, images=(), instruction_span=Span(0, 1))

    def generate(self, inputs, *, enable_thinking, max_new_tokens, **kw):
        prompt = inputs.tensors["prompt"]
        self.calls.append(("generate", prompt, enable_thinking, max_new_tokens, kw))
        if self.fail_on_call is not None and sum(c[0] == "generate" for c in self.calls) == self.fail_on_call:
            raise RuntimeError("boom")
        text = self.responder(prompt)
        n = min(max_new_tokens, 1 + len(text) // 4)
        return GenerationResult(text=text, token_ids=tuple(range(n)), n_generated_tokens=n,
                                n_prompt_tokens=5 + len(prompt) // 4, n_visual_tokens=self.visual_tokens,
                                wall_time_s=0.001, raw_text=text + "<stop>")


def canned_responder(items):
    """MMLU-Pro：偶数位置答对；GSM8K：每 3 题一轮（带 ####、不带 ####、答错）；IFEval：空回答（官方规则下全部未遵循）。"""
    answers = {P.WARMUP_PROMPT: "OK"}
    for k, it in enumerate(items["items"]["mmlu_pro"]):
        answers[P.mmlu_pro_prompt(it)] = f"The answer is ({it['answer']})." if k % 2 == 0 else "I am not sure."
    for k, it in enumerate(items["items"]["gsm8k"]):
        answers[P.gsm8k_prompt(it)] = [f"Work.\n#### {it['answer']:,}", f"So it is {it['answer']}.",
                                       f"#### {it['answer'] + 1}"][k % 3]
    for it in items["items"]["ifeval"]:
        answers[P.ifeval_prompt(it)] = ""
    return answers.__getitem__


def test_full_run_follows_protocol(items):
    ad = FakeAdapter(canned_responder(items))
    state = run_items(ad, items)
    builds = [c for c in ad.calls if c[0] == "build_inputs"]
    gens = [c for c in ad.calls if c[0] == "generate"]
    assert len(builds) == len(gens) == 201  # 预热 1 次 + 200 题
    assert gens[0][1:4] == (P.WARMUP_PROMPT, False, P.WARMUP_MAX_NEW_TOKENS)
    expected = [(b, P.PROMPT_BUILDERS[b](it)) for b in P.BENCHMARKS for it in items["items"][b]]
    for (bench, prompt), build, g in zip(expected, builds[1:], gens[1:]):
        assert build == ("build_inputs", [], prompt, "instruction_first")
        assert g[1:4] == (prompt, False, P.MAX_NEW_TOKENS[bench])
    for g in gens:
        assert g[4] == dict(P.GREEDY_KWARGS) and g[4]["do_sample"] is False  # 每次都显式给 greedy
    # 逐题记录按题目文件的顺序
    for bench in P.BENCHMARKS:
        assert [r["id"] for r in state["records"][bench]] == [it["id"] for it in items["items"][bench]]
    s = summarize(state["records"])
    assert s["headline"] == {"mmlu_pro": 50.0, "gsm8k": 68.0, "ifeval": 0.0}
    assert s["scores"]["gsm8k"]["strict"] == {"n": 50, "n_correct": 17, "pct": 34.0}
    assert s["parse_tally"]["mmlu_pro"]["failures"] == {"no_choice": 50}
    assert s["parse_tally"]["gsm8k_strict"]["failures"] == {"no_marker": 17}
    assert s["scores"]["ifeval"]["instruction_level_loose"]["n"] == 77
    # 记账：每题一个样本、一次调用、一轮；预热不计入
    comp = state["compute"]
    assert [comp[b]["n_samples"] for b in P.BENCHMARKS] == [100, 50, 50]
    assert comp["total"]["n_calls"] == comp["total"]["n_rounds"] == comp["total"]["n_samples"] == 200
    assert comp["total"]["visual_tokens"] == 0
    assert comp["total"]["generated_tokens"] == sum(r["n_generated_tokens"] for b in P.BENCHMARKS
                                                     for r in state["records"][b])
    assert state["warmup"]["counted_in_compute"] is False


def test_limited_run_records(items):
    responses = {P.WARMUP_PROMPT: "OK"}
    ifeval_texts = ["First paragraph.\n\nSecond one here", "<<A Title>>\nbody", "*one* and *two*, " + "word " * 30]
    for it, text in zip(items["items"]["ifeval"][:3], ifeval_texts):
        responses[P.ifeval_prompt(it)] = text
    ad = FakeAdapter(lambda p: responses.get(p, "B"))
    msgs = []
    state = run_items(ad, items, limit=3, progress=msgs.append)
    assert {b: len(v) for b, v in state["records"].items()} == {"mmlu_pro": 3, "gsm8k": 3, "ifeval": 3}
    for it, rec in zip(items["items"]["ifeval"][:3], state["records"]["ifeval"]):
        assert {k: rec[k] for k in ("strict", "loose")} == ifeval_follow(it, rec["text"])
        assert rec["instruction_id_list"] == it["instruction_id_list"]
    rec = state["records"]["mmlu_pro"][0]
    assert rec["raw_text"] == "B<stop>" and rec["text"] == "B" and rec["parsed"] == "B"
    assert rec["reached_max_new_tokens"] is False and len(rec["prompt_sha256"]) == 64
    assert msgs == ["mmlu_pro 3/3  生成 token 累计 3", "gsm8k 3/3  生成 token 累计 3",
                    f"ifeval 3/3  生成 token 累计 {sum(r['n_generated_tokens'] for r in state['records']['ifeval'])}"]


def test_failure_keeps_partial_results(items):
    ad = FakeAdapter(lambda p: "A", fail_on_call=5)  # 预热是第 1 次，第 4 题出错
    state = {}
    with pytest.raises(RuntimeError, match="boom"):
        run_items(ad, items, state=state)
    assert len(state["records"]["mmlu_pro"]) == 3 and state["compute"]["mmlu_pro"]["n_samples"] == 3
    assert "warmup" in state and "total" not in state["compute"]


def test_visual_tokens_are_rejected(items):
    with pytest.raises(TextOnlyViolation):
        run_items(FakeAdapter(lambda p: "A", visual_tokens=1), items, limit=1)


@pytest.mark.parametrize("bad", [0, -1, True, 2.0])
def test_limit_must_be_positive_int(items, bad):
    with pytest.raises(ValueError):
        run_items(FakeAdapter(lambda p: "A"), items, limit=bad)


# ---------------------------------------------------------------------------
# 三家族：零张图输入与全部 200 个提示（真实 tokenizer 与模板，小型随机模型）
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location("_health_check_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("key", FAMILY_KEYS)
def test_every_prompt_builds_with_zero_images(script, items, key):
    args = script.parse_args(["--tiny", "--model", key, "--out", "unused.json"])
    ad, entry, _ = script.make_adapter(args)
    assert entry.family == registry.get_model(key).family and ad.model_key == f"tiny({key})"
    for bench in P.BENCHMARKS:
        for it in items["items"][bench]:
            prompt = P.PROMPT_BUILDERS[bench](it)
            inp = ad.build_inputs([], prompt, P.ORDER)
            assert inp.images == () and inp.query_span is None
            ids = inp.tensors["input_ids"][0].tolist()
            assert ad.tokenizer.decode(ids[inp.instruction_span.start : inp.instruction_span.end]) == prompt, (bench, it["id"])
            assert "pixel_values" not in inp.tensors


def test_parse_args_rules(script):
    a = script.parse_args(["--tiny", "--out", "x.json"])
    assert (a.device, a.dtype, a.model) == ("cpu", "fp32", None)
    a = script.parse_args(["--model", "qwen3.5-2b", "--out", "x.json"])
    assert (a.device, a.dtype) == ("cuda", "bf16")
    for bad in (["--tiny"], ["--out", "x.json"], ["--tiny", "--out", "x.json", "--limit", "0"],
                ["--tiny", "--out", "x.json", "--dtype", "bfloat16"]):
        with pytest.raises(SystemExit):
            script.parse_args(bad)


# ---------------------------------------------------------------------------
# 端到端（子进程，与服务器同一流程）
# ---------------------------------------------------------------------------


def _run_script(args, *, nltk_data=SYNTHETIC_NLTK_DATA, timeout=900):
    env = {**os.environ, "NLTK_DATA": str(nltk_data)}
    return subprocess.run([sys.executable, str(SCRIPT), *args], cwd=REPO_ROOT, env=env, capture_output=True,
                          text=True, timeout=timeout)


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory):
    cache = {}
    root = tmp_path_factory.mktemp("health_tiny")

    def get(key):
        if key not in cache:
            out = root / f"{key}.json"
            cache[key] = (_run_script(["--tiny", "--model", key, "--limit", "3", "--out", str(out)]), out)
        return cache[key]

    return get


@pytest.mark.parametrize("key", FAMILY_KEYS)
def test_tiny_end_to_end(tiny_run, items, key):
    proc, out = tiny_run(key)
    assert out.exists(), proc.stderr[-3000:]
    rep = json.loads(out.read_text(encoding="utf-8"))
    assert proc.returncode == 0 and rep["ok"] is True, (rep.get("error"), rep.get("traceback"))
    assert (rep["mode"], rep["limit"], rep["protocol_version"]) == ("tiny", 3, P.PROTOCOL_VERSION)
    entry = registry.get_model(key)
    assert (rep["model"]["key"], rep["model"]["family"], rep["model"]["revision"]) == (entry.key, entry.family, entry.revision)
    assert rep["model"]["device"] == "cpu" and rep["model"]["dtype"] == "torch.float32"
    assert rep["generation"] == {"order": "instruction_first", "n_images": 0, "enable_thinking": False,
                                 "greedy_kwargs": dict(P.GREEDY_KWARGS), "max_new_tokens": dict(P.MAX_NEW_TOKENS)}
    assert rep["items_file"] == {"path": P.ITEMS_PATH, "sha256": P.ITEMS_SHA256}
    for bench in P.BENCHMARKS:
        recs = rep["records"][bench]
        assert [r["id"] for r in recs] == [it["id"] for it in items["items"][bench][:3]]
        for r in recs:
            assert isinstance(r["raw_text"], str) and isinstance(r["text"], str)
            assert 1 <= r["n_generated_tokens"] <= P.MAX_NEW_TOKENS[bench]
    assert set(rep["headline"]) == set(P.BENCHMARKS)
    assert {k: v["n"] for k, v in rep["parse_tally"].items()} == {"mmlu_pro": 3, "gsm8k_strict": 3, "gsm8k_flexible": 3}
    assert rep["scores"]["ifeval"]["instruction_level_strict"]["n"] == sum(
        len(it["instruction_id_list"]) for it in items["items"]["ifeval"][:3])
    total = rep["compute"]["total"]
    assert (total["n_samples"], total["n_calls"], total["visual_tokens"]) == (9, 9, 0)
    assert rep["warmup"]["counted_in_compute"] is False
    scoring = rep["scoring"]
    assert (scoring["mode"], scoring["benchmarks_scored"], scoring["source"]) == ("inline", list(P.BENCHMARKS), None)
    assert scoring["code"]["commit"] == rep["run"]["meta"]["git"]["commit"] and scoring["scored_at_utc"]
    punkt = scoring["ifeval"]["punkt_tab"]
    assert str(SYNTHETIC_NLTK_DATA) in punkt["path"] and punkt["matches_pinned"] is False
    assert (scoring["ifeval"]["scored"], scoring["ifeval"]["skip_reason"], scoring["ifeval"]["langdetect_seed"]) == (
        True, None, 0)
    assert rep["unscored"] == {}
    run = rep["run"]
    assert run["config"]["seed"] == run["meta"]["seed"] == P.SEED
    assert list(run["meta"]["data_sha256"].values()) == [P.ITEMS_SHA256]
    assert run["meta"]["git"]["commit"] is None or len(run["meta"]["git"]["commit"]) == 40
    assert set(run["packages_ifeval"]) == {"nltk", "langdetect", "immutabledict"}
    # 汇总与逐题记录一致（门报告的数从记录重算得到同一结果）
    assert summarize(rep["records"])["headline"] == rep["headline"]


def test_compare_cli(tiny_run, tmp_path):
    _, out = tiny_run("internvl3.5-2b-hf")
    cmp_out = tmp_path / "cmp.json"
    proc = _run_script(["--compare", str(out), str(out), "--out", str(cmp_out)], timeout=300)
    assert proc.returncode == 0, proc.stderr[-2000:]
    res = json.loads(cmp_out.read_text(encoding="utf-8"))
    assert res["any_flag"] is False and res["partial_run"] is True and res["unscored"] == []
    assert all(r["delta_pp"] == 0.0 and r["ci95_pp"] == [0.0, 0.0] for r in res["rows"])
    assert all(r["sign_test"] == {"n_better": 0, "n_worse": 0, "p_value": 1.0} for r in res["rows"])
    assert "any_flag=False" in proc.stdout and "CI95=[+0.0, +0.0]" in proc.stdout


# ---------------------------------------------------------------------------
# 先生成、后评分（审查第 1 轮）：--skip-ifeval-scoring 与 --rescore
# ---------------------------------------------------------------------------

SKIP_KEY = "internvl3.5-2b-hf"  # 三家族里最快的；生成是确定的（同一小模型、greedy、CPU）


def _blocked_ifeval_env(tmp_path: Path) -> dict:
    """子进程环境：nltk、langdetect、immutabledict 一 import 就报错；NLTK_DATA 指向空目录（没有 punkt_tab）。"""
    blocked = tmp_path / "blocked"
    for name in ("nltk", "langdetect", "immutabledict"):
        (blocked / name).mkdir(parents=True)
        (blocked / name / "__init__.py").write_text(f"raise ImportError('{name} is blocked in this test')\n")
    (tmp_path / "empty_nltk_data").mkdir()
    pythonpath = os.pathsep.join(p for p in (str(blocked), os.environ.get("PYTHONPATH", "")) if p)
    return {**os.environ, "PYTHONPATH": pythonpath, "NLTK_DATA": str(tmp_path / "empty_nltk_data")}


@pytest.fixture(scope="module")
def skip_run(tmp_path_factory):
    """--skip-ifeval-scoring：IFEval 的依赖与 punkt_tab 都拿不到时照样跑完。"""
    root = tmp_path_factory.mktemp("health_skip")
    out = root / "skip.json"
    proc = subprocess.run([sys.executable, str(SCRIPT), "--tiny", "--model", SKIP_KEY, "--limit", "3",
                           "--skip-ifeval-scoring", "--out", str(out)], cwd=REPO_ROOT, env=_blocked_ifeval_env(root),
                          capture_output=True, text=True, timeout=900)
    return proc, out


def _dump(obj) -> str:
    return json.dumps(obj, indent=1, ensure_ascii=False)


def _scored_view(rep: dict, *, drop_timing: bool) -> str:
    """跑完即评与重评应当逐字节相同的部分；比较两次不同的运行时去掉逐题的墙钟（唯一与评分无关、每次都变的逐题字段）。"""
    view = {s: rep[s] for s in SCORED_SECTIONS}
    if drop_timing:
        view["records"] = {b: [{k: v for k, v in r.items() if k != "wall_time_s"} for r in recs]
                           for b, recs in rep["records"].items()}
    return _dump(view)


def test_skip_ifeval_scoring_needs_no_ifeval_dependencies(skip_run, items):
    proc, out = skip_run
    assert out.exists(), proc.stderr[-3000:]
    rep = json.loads(out.read_text(encoding="utf-8"))
    assert proc.returncode == 0 and rep["ok"] is True, (rep.get("error"), rep.get("traceback"))
    assert "is blocked in this test" not in proc.stderr
    # IFEval：生成照常、原始输出照常保存；分数为 null 并写明原因
    ife = rep["records"]["ifeval"]
    assert [r["id"] for r in ife] == [it["id"] for it in items["items"]["ifeval"][:3]]
    assert all(r["raw_text"] and r["strict"] is None and r["loose"] is None for r in ife)
    assert rep["headline"]["ifeval"] is None and rep["unscored"] == {"ifeval": IFEVAL_SKIP_REASON}
    assert all(v["n_correct"] is None and v["pct"] is None for v in rep["scores"]["ifeval"].values())
    assert rep["scores"]["ifeval"]["instruction_level_strict"]["n"] == sum(
        len(it["instruction_id_list"]) for it in items["items"]["ifeval"][:3])
    # 另两项照常评分
    assert rep["headline"]["mmlu_pro"] is not None and rep["headline"]["gsm8k"] is not None
    assert rep["parse_tally"]["mmlu_pro"]["n"] == 3
    ifeval = rep["scoring"]["ifeval"]
    assert rep["scoring"]["benchmarks_scored"] == ["mmlu_pro", "gsm8k"]
    assert (ifeval["scored"], ifeval["skip_reason"], ifeval["punkt_tab"], ifeval["langdetect_seed"]) == (
        False, IFEVAL_SKIP_REASON, None, None)
    assert rep["run"]["config"]["skip_ifeval_scoring"] is True
    assert "未评分" in proc.stdout


def test_rescore_is_byte_identical_to_inline_scoring(tiny_run, skip_run, tmp_path):
    """同一套规则：一次跑完即评的输出，与"只存输出、事后 --rescore"的结果，由逐题记录算出的部分逐字节相同。"""
    _, inline_path = tiny_run(SKIP_KEY)
    inline_raw = inline_path.read_bytes()
    inline = json.loads(inline_raw)
    # 1) 对一次跑完的输出重评三项：连逐题墙钟在内逐字节相同
    re_all = tmp_path / "rescored_inline.json"
    proc = _run_script(["--rescore", str(inline_path), "--out", str(re_all)], timeout=300)
    assert proc.returncode == 0, proc.stderr[-3000:]
    rep = json.loads(re_all.read_text(encoding="utf-8"))
    assert _scored_view(rep, drop_timing=False) == _scored_view(inline, drop_timing=False)
    for key in ("model", "run", "generation", "compute", "warmup", "items_file", "argv", "started_utc"):
        assert rep[key] == inline[key], key  # 生成侧的记录原样保留
    scoring = rep["scoring"]
    assert (scoring["mode"], scoring["benchmarks_scored"]) == ("rescore", list(P.BENCHMARKS))
    assert scoring["source"] == {"path": str(inline_path), "sha256": hashlib.sha256(inline_raw).hexdigest()}
    assert scoring["scored_at_utc"] and "commit" in scoring["code"]
    assert scoring["ifeval"]["punkt_tab"]["files_sha256"] == inline["scoring"]["ifeval"]["punkt_tab"]["files_sha256"]
    assert rep["scoring_history"] == [inline["scoring"]]
    assert inline_path.read_bytes() == inline_raw  # 输入原样不动

    # 2) 先生成（不评 IFEval）、后评分：与一次跑完的评分逐字节相同（两次运行只差逐题墙钟）
    _, skip_path = skip_run
    skipped = json.loads(skip_path.read_text(encoding="utf-8"))
    for bench in P.BENCHMARKS:  # 前提：两次运行的生成相同
        assert [r["raw_text"] for r in skipped["records"][bench]] == [r["raw_text"] for r in inline["records"][bench]]
    for extra in ([], ["--only-ifeval"]):
        out = tmp_path / f"rescored_skip{'_ifeval' if extra else ''}.json"
        proc = _run_script(["--rescore", str(skip_path), "--out", str(out), *extra], timeout=300)
        assert proc.returncode == 0, proc.stderr[-3000:]
        rep = json.loads(out.read_text(encoding="utf-8"))
        assert _scored_view(rep, drop_timing=True) == _scored_view(inline, drop_timing=True), extra
        assert rep["unscored"] == {} and rep["headline"]["ifeval"] == inline["headline"]["ifeval"]
        assert rep["scoring"]["benchmarks_scored"] == (["ifeval"] if extra else list(P.BENCHMARKS))
        assert rep["scoring_history"] == [skipped["scoring"]] and rep["scoring_history"][0]["ifeval"]["scored"] is False
    # 3) 未评 IFEval 的一边参与比较：该项不比，另两项照常
    res = compare(inline, skipped)
    assert res["unscored"] == ["ifeval"] and not any(r["scored"] for r in res["rows"] if r["benchmark"] == "ifeval")
    assert all(r["scored"] and r["delta_pp"] == 0.0 for r in res["rows"] if r["benchmark"] != "ifeval")


def test_rescore_real_output_requires_pinned_punkt(tiny_run, tmp_path):
    _, inline_path = tiny_run(SKIP_KEY)
    rep = json.loads(inline_path.read_text(encoding="utf-8"))
    rep["mode"] = "real"  # 真模型的输出：重评 IFEval 时要求钉住的 punkt_tab（这里只有自造的空 punkt_tab）
    fake_real = tmp_path / "real.json"
    fake_real.write_text(json.dumps(rep), encoding="utf-8")
    out = tmp_path / "out.json"
    proc = _run_script(["--rescore", str(fake_real), "--out", str(out)], timeout=300)
    assert proc.returncode == 2
    err = json.loads(out.read_text(encoding="utf-8"))
    assert err["ok"] is False and "PunktDataError" in err["error"]


def test_rescore_rejects_mismatched_records(items):
    ad = FakeAdapter(canned_responder(items))
    records = run_items(ad, items, limit=2)["records"]
    assert rescore_records(records, items) == records
    bad = copy.deepcopy(records)
    bad["gsm8k"][0]["prompt_sha256"] = "0" * 64
    with pytest.raises(RescoreError, match="prompt_sha256"):
        rescore_records(bad, items)
    bad = copy.deepcopy(records)
    bad["mmlu_pro"].reverse()
    with pytest.raises(RescoreError, match="前 2 题"):
        rescore_records(bad, items)
    bad = copy.deepcopy(records)
    del bad["ifeval"][1]["text"]
    with pytest.raises(RescoreError, match="text"):
        rescore_records(bad, items, benchmarks=("ifeval",))
    rescore_records(bad, items, benchmarks=("mmlu_pro",))  # 不重评的项原样保留，不核对生成字段
    with pytest.raises(RescoreError, match="未知"):
        rescore_records(records, items, benchmarks=("mmlu",))


def test_rescore_in_process_equals_inline(items):
    """全部 200 题（假适配器）：先不评 IFEval、再重评，与跑完即评的逐题记录与汇总逐字节相同。
    需要 punkt_tab 的题给空回答（官方规则下空回答直接判未遵循，不调用句子切分），其余题给会触发各种 checker 的回答。"""
    punkt_users = {"length_constraints:number_sentences", "change_case:capital_word_frequency"}
    texts = {}
    for k, it in enumerate(items["items"]["ifeval"]):
        texts[P.ifeval_prompt(it)] = "" if punkt_users & set(it["instruction_id_list"]) else [
            "<<A Title>>\n*first* and *second* part, with a comma.\nP.S. done",
            '"THIS IS ALL CAPS AND QUOTED"',
            "lowercase only here\n\n***\n\nSECTION 1\nmore text"][k % 3]
    base = canned_responder(items)

    def responder(prompt):
        return texts[prompt] if prompt in texts else base(prompt)

    inline = run_items(FakeAdapter(responder), items)["records"]
    skipped = run_items(FakeAdapter(responder), items, score_ifeval=False)["records"]
    assert all(r["strict"] is None for r in skipped["ifeval"])
    assert summarize(skipped)["unscored"] == {"ifeval": IFEVAL_SKIP_REASON}
    no_timing = lambda recs: {b: [{k: v for k, v in r.items() if k != "wall_time_s"} for r in rs] for b, rs in recs.items()}
    for benchmarks in (P.BENCHMARKS, ("ifeval",)):
        rescored = rescore_records(skipped, items, benchmarks=benchmarks)
        assert _dump(no_timing(rescored)) == _dump(no_timing(inline))
        assert _dump(summarize(rescored)) == _dump(summarize(inline))
    assert any(r["strict"]["follow_all"] for r in inline["ifeval"])  # 确实有题被判遵循，不是全 False 的平凡情形


def test_cli_mode_rules(script, tmp_path):
    ok = tmp_path / "in.json"
    for bad in (["--rescore", str(ok)],  # 缺 --out
                ["--rescore", str(ok), "--out", str(ok)],  # 覆盖输入
                ["--rescore", str(ok), "--out", "x.json", "--tiny"],
                ["--rescore", str(ok), "--out", "x.json", "--skip-ifeval-scoring"],
                ["--tiny", "--out", "x.json", "--only-ifeval"],
                ["--compare", "a.json", "b.json", "--rescore", str(ok), "--out", "x.json"],
                ["--compare", "a.json", "b.json", "--limit", "3"]):
        with pytest.raises(SystemExit):
            script.parse_args(bad)
    a = script.parse_args(["--rescore", str(ok), "--out", "x.json", "--only-ifeval"])
    assert a.only_ifeval and a.rescore == str(ok)
    a = script.parse_args(["--tiny", "--out", "x.json", "--skip-ifeval-scoring"])
    assert a.skip_ifeval_scoring and not a.only_ifeval


def test_rescore_refuses_failed_run(tmp_path):
    failed = tmp_path / "failed.json"
    failed.write_text(json.dumps({"ok": False, "error": "RuntimeError('boom')"}), encoding="utf-8")
    out = tmp_path / "out.json"
    proc = _run_script(["--rescore", str(failed), "--out", str(out)], timeout=300)
    assert proc.returncode == 2
    assert json.loads(out.read_text(encoding="utf-8"))["ok"] is False


def test_real_mode_checks_punkt_before_loading_model(tmp_path):
    out = tmp_path / "real.json"
    proc = _run_script(["--model", "qwen3.5-0.8b", "--device", "cpu", "--dtype", "fp32", "--out", str(out)], timeout=300)
    assert proc.returncode == 2
    rep = json.loads(out.read_text(encoding="utf-8"))
    assert rep["ok"] is False and "PunktDataError" in rep["error"]
    assert "model" not in rep and "records" not in rep  # 在加载模型之前就停了
