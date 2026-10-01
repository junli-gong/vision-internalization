"""健康检查的固定题目（DEV-0008）：题目文件的数量、分层、字段、哈希，离线重抽，提示模板。

复现方式（择一，见完成总结）：离线重抽——tests/fixtures/health/population_index.json 是 build_health_items.py 从固定
revision 的原始文件导出的总体索引（MMLU-Pro 的 id 与类别、GSM8K 行数、IFEval 的 key），这里用它按同一抽样规则重新挑一次，
与入库题目文件的 id（及 MMLU-Pro 的类别与阶段）逐个比较；题目内容由 protocol.ITEMS_SHA256 钉住。联网的逐字节复现是
`python scripts/build_health_items.py --raw-dir <目录> --check`。
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from src.health import items as I
from src.health import protocol as P

REPO_ROOT = Path(__file__).resolve().parents[1]
FIX = REPO_ROOT / "tests/fixtures/health"


@pytest.fixture(scope="module")
def items_obj():
    obj, digest = I.load_items()
    assert digest == P.ITEMS_SHA256
    return obj


# ---------------------------------------------------------------------------
# 题目文件
# ---------------------------------------------------------------------------


def test_items_file_hash_is_pinned_and_canonical(items_obj):
    raw = (REPO_ROOT / P.ITEMS_PATH).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == P.ITEMS_SHA256
    # 入库文件就是规范序列化的结果：同一内容重新写出逐字相同
    assert I.dumps_items(items_obj) == raw.decode("utf-8")


def test_counts_and_stratification(items_obj):
    it = items_obj["items"]
    assert list(it) == ["mmlu_pro", "gsm8k", "ifeval"] == list(P.BENCHMARKS)
    assert {k: len(v) for k, v in it.items()} == {"mmlu_pro": 100, "gsm8k": 50, "ifeval": 50} == dict(P.COUNTS)
    assert sum(len(v) for v in it.values()) == 200
    strat = Counter(x["category"] for x in it["mmlu_pro"] if x["stage"] == "stratified")
    assert len(strat) == 14 and set(strat.values()) == {7}
    fill = [x for x in it["mmlu_pro"] if x["stage"] == "fill"]
    assert len(fill) == 2
    assert set(Counter(x["stage"] for x in it["mmlu_pro"])) == {"stratified", "fill"}
    for name, xs in it.items():
        ids = [x["id"] for x in xs]
        assert ids == sorted(set(ids)), name  # 升序、不重复


def test_fields_complete(items_obj):
    it = items_obj["items"]
    for x in it["mmlu_pro"]:
        assert set(x) == {"id", "category", "src", "stage", "question", "options", "answer", "answer_index"}
        assert isinstance(x["question"], str) and x["question"].strip()
        assert 1 <= len(x["options"]) <= 10 and all(isinstance(o, str) and o.strip() for o in x["options"])
        assert x["answer"] == "ABCDEFGHIJ"[x["answer_index"]] and x["answer_index"] < len(x["options"])
    for x in it["gsm8k"]:
        assert set(x) == {"id", "question", "answer"}
        assert type(x["answer"]) is int and x["question"].strip()
        assert 0 <= x["id"] < P.SOURCES["gsm8k"].n_rows
    for x in it["ifeval"]:
        assert set(x) == {"id", "prompt", "instruction_id_list", "kwargs"}
        assert len(x["instruction_id_list"]) == len(x["kwargs"]) >= 1
        assert all(isinstance(kw, dict) for kw in x["kwargs"])
        assert all(v is not None for kw in x["kwargs"] for v in kw.values())  # HF 文件里是稀疏写法，没有 null 占位


def test_sources_recorded(items_obj):
    src = items_obj["sources"]
    assert list(src) == list(P.BENCHMARKS)
    for name, s in P.SOURCES.items():
        assert src[name] == {"repo": s.repo, "revision": s.revision, "config": s.config, "split": s.split,
                             "file": s.file, "file_sha256": s.file_sha256, "license": s.license, "n_rows": s.n_rows}
        assert len(s.revision) == 40 and int(s.revision, 16) >= 0
        assert len(s.file_sha256) == 64
    assert {n: s["license"] for n, s in src.items()} == {"mmlu_pro": "MIT", "gsm8k": "MIT", "ifeval": "Apache-2.0"}
    assert (src["mmlu_pro"]["split"], src["gsm8k"]["config"], src["gsm8k"]["split"], src["ifeval"]["split"]) == (
        "test", "main", "test", "train")
    assert items_obj["seed"] == P.SEED == 20260930


# ---------------------------------------------------------------------------
# 离线重抽
# ---------------------------------------------------------------------------


def test_offline_resample_reproduces_items_file(items_obj):
    index = json.loads((FIX / "population_index.json").read_text(encoding="utf-8"))
    pop, n_gsm, keys = I.expand_population_index(index)
    assert len(pop) == P.SOURCES["mmlu_pro"].n_rows and n_gsm == P.SOURCES["gsm8k"].n_rows
    assert len(keys) == P.SOURCES["ifeval"].n_rows == len(set(keys))
    it = items_obj["items"]
    chosen = I.select_mmlu_pro(pop)
    assert chosen == {x["id"]: x["stage"] for x in it["mmlu_pro"]}
    category = dict(pop)
    assert all(category[x["id"]] == x["category"] for x in it["mmlu_pro"])
    assert sorted(I.select_uniform("gsm8k", range(n_gsm))) == [x["id"] for x in it["gsm8k"]]
    assert sorted(I.select_uniform("ifeval", keys)) == [x["id"] for x in it["ifeval"]]
    # 换一个种子就挑出别的题（排序键确实依赖种子）
    assert sorted(I.select_uniform("gsm8k", range(n_gsm), seed=P.SEED + 1)) != [x["id"] for x in it["gsm8k"]]


def test_rank_key_formula_pinned():
    # 抽样规则写死在这条公式上；改公式（或阶段名、种子）会改变题目，测试立刻报出
    assert I.rank_key("gsm8k", "uniform", 7) == hashlib.sha256(b"20260930|gsm8k|uniform|7").hexdigest()
    assert (P.STAGE_MMLU_PRO_STRATIFIED, P.STAGE_MMLU_PRO_FILL, P.STAGE_UNIFORM) == ("stratified", "fill", "uniform")
    with pytest.raises(TypeError):
        I.rank_key("gsm8k", "uniform", True)


def test_stratified_sampler_on_synthetic_population():
    # 类别大小各不相同（含恰好 7 题的类别）；补足的 2 题只能来自各类别剩下的题
    sizes = {f"c{k}": 7 + 3 * k for k in range(14)}
    pop, q = [], 1000
    for cat, n in sizes.items():
        for _ in range(n):
            pop.append((q, cat))
            q += 1
    chosen = I.select_mmlu_pro(pop, seed=5)
    per_cat = Counter(dict(pop)[q] for q, s in chosen.items() if s == "stratified")
    assert per_cat == Counter({c: 7 for c in sizes})
    fill = [q for q, s in chosen.items() if s == "fill"]
    assert len(fill) == 2 and dict(pop)[fill[0]] != "c0" and dict(pop)[fill[1]] != "c0"  # c0 只有 7 题，全被分层取走
    assert I.select_mmlu_pro(list(reversed(pop)), seed=5) == chosen  # 与总体的行序无关
    with pytest.raises(ValueError, match="14"):
        I.select_mmlu_pro(pop[:-sizes["c13"]], seed=5)


def test_hash_sample_errors():
    with pytest.raises(ValueError, match="重复"):
        I.hash_sample([1, 1, 2], 1, benchmark="x", stage="y")
    with pytest.raises(ValueError, match="只有"):
        I.hash_sample([1, 2], 3, benchmark="x", stage="y")
    assert I.hash_sample(range(10), 10, benchmark="x", stage="y") != list(range(10))  # 是置换，不是原序


def test_gsm8k_final_answer():
    assert I.gsm8k_final_answer("She has 3 + 4 = <<3+4=7>>7 apples.\n#### 7") == 7
    assert I.gsm8k_final_answer("...\n#### 1,450,000") == 1450000
    assert I.gsm8k_final_answer("...\n#### -3") == -3
    for bad in ("#### 7", "...\n#### 7.5", "...\n#### seven", "...\n#### 7\n"):
        with pytest.raises(ValueError):
            I.gsm8k_final_answer(bad)


def test_validate_items_rejects_tampering(items_obj):
    I.validate_items(items_obj)

    def broken(mutate):
        obj = copy.deepcopy(items_obj)
        mutate(obj["items"])
        return obj

    cases = [
        lambda it: it["mmlu_pro"].pop(),  # 少一题
        lambda it: it["gsm8k"].reverse(),  # 顺序乱了
        lambda it: it["mmlu_pro"][0].update(stage="fill"),  # 分层被破坏
        lambda it: it["mmlu_pro"][0].update(answer_index=(it["mmlu_pro"][0]["answer_index"] + 1) % 3),
        lambda it: it["gsm8k"][0].update(answer=str(it["gsm8k"][0]["answer"])),
        lambda it: it["ifeval"][0]["kwargs"].append({}),
    ]
    for mutate in cases:
        with pytest.raises(ValueError):
            I.validate_items(broken(mutate))


def test_load_items_checks_sha256(tmp_path, items_obj):
    obj = copy.deepcopy(items_obj)
    obj["items"]["gsm8k"][0]["answer"] += 1  # 结构仍合法，只有内容变了
    p = tmp_path / "items.json"
    p.write_text(I.dumps_items(obj), encoding="utf-8")
    with pytest.raises(ValueError, match="sha256"):
        I.load_items(p)
    loaded, digest = I.load_items(p, expected_sha256=None)
    assert loaded == obj and digest != P.ITEMS_SHA256


# ---------------------------------------------------------------------------
# 提示模板
# ---------------------------------------------------------------------------


def test_mmlu_pro_prompt_format():
    item = {"question": "  What is 2+2?\n", "options": [" 3", "4 ", "5", "22"]}
    assert P.mmlu_pro_prompt(item) == (
        "What is 2+2?\nA. 3\nB. 4\nC. 5\nD. 22\nAnswer with the letter of the correct option only.")
    assert P.option_letters(4) == "ABCD" and P.option_letters(10) == "ABCDEFGHIJ"
    for bad in (0, 11, True, 3.0):
        with pytest.raises(ValueError):
            P.option_letters(bad)


def test_gsm8k_and_ifeval_prompt_format():
    assert P.gsm8k_prompt({"question": "Tom has 3 apples. How many?"}) == (
        "Tom has 3 apples. How many?\n"
        "Solve the problem step by step. On the last line, write the final answer as '#### <number>'.")
    assert P.ifeval_prompt({"prompt": "Write a poem.\n\n"}) == "Write a poem."


def test_all_prompts_have_no_edge_whitespace(items_obj):
    # 适配器（Qwen/Gemma 的模板 trim 指令）要求指令没有首尾空白；只有 IFEval key 1999 的原始提示末尾有空白
    for name in P.BENCHMARKS:
        for x in items_obj["items"][name]:
            prompt = P.PROMPT_BUILDERS[name](x)
            assert prompt == prompt.strip(), (name, x["id"])
    changed = [x["id"] for x in items_obj["items"]["ifeval"] if P.ifeval_prompt(x) != x["prompt"]]
    assert changed == [1999]
    lead = [x["id"] for x in items_obj["items"]["mmlu_pro"] if x["question"] != x["question"].lstrip()]
    assert lead == [10873, 11161, 11271]  # 这三题的题干开头有空白，模板里去掉


def test_protocol_constants_pinned():
    # 协议 v2（D28-7）：MMLU-Pro 的标题读数 256 token，32 token 留作格式诊断（同一次生成的前 32 个 token）
    assert P.PROTOCOL_VERSION == 2 and P.SUPPORTED_PROTOCOL_VERSIONS == (1, 2)
    assert P.ITEMS_PROTOCOL_VERSION == 1  # 题目文件建于 v1，v2 题目不变
    assert dict(P.MAX_NEW_TOKENS) == {"mmlu_pro": 256, "gsm8k": 512, "ifeval": 1024}
    assert P.FMT32_MAX_NEW_TOKENS == 32 and P.TRUNCATION_NOTICE_PP == 10
    assert [(r.name, r.benchmark, r.max_new_tokens, r.role, r.prefix_of) for r in P.READINGS] == [
        ("mmlu_pro", "mmlu_pro", 256, "headline", None),
        ("mmlu_pro_fmt32", "mmlu_pro", 32, "diagnostic", "mmlu_pro"),
        ("gsm8k", "gsm8k", 512, "headline", None),
        ("ifeval", "ifeval", 1024, "headline", None),
    ]
    # v1 的输出（D26）：MMLU-Pro 32、GSM8K 512、IFEval 1024；v1 的 MMLU-Pro 对上 v2 的 fmt32
    assert {k: (r.name, r.max_new_tokens) for k, r in P.record_readings(1).items()} == {
        "mmlu_pro": ("mmlu_pro_fmt32", 32), "gsm8k": ("gsm8k", 512), "ifeval": ("ifeval", 1024)}
    assert list(P.record_readings(2)) == [r.name for r in P.READINGS]
    assert [r.name for r in P.readings_of("mmlu_pro")] == ["mmlu_pro", "mmlu_pro_fmt32"]
    for bad in (0, 3, None, True, "2"):
        with pytest.raises(ValueError):
            P.check_protocol_version(bad)
    assert P.ORDER == "instruction_first" and P.ENABLE_THINKING is False
    assert P.GREEDY_KWARGS["do_sample"] is False and P.GREEDY_KWARGS["num_beams"] == 1
    assert dict(P.HEADLINE) == {"mmlu_pro": "accuracy", "gsm8k": "flexible", "ifeval": "prompt_level_strict"}
    assert P.RED_FLAG_DROP_PP == 5
    assert P.MMLU_PRO_INSTRUCTION == "Answer with the letter of the correct option only."
    assert P.GSM8K_INSTRUCTION == (
        "Solve the problem step by step. On the last line, write the final answer as '#### <number>'.")
