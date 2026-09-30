"""健康检查的固定题目：抽样、从原始数据构造题目、题目文件的读写与校验（简报 §8.11；DEV-0008）。

抽样规则（种子 protocol.SEED，在看到任何分数之前固定）：
- 排序键：sha256(f"{seed}|{benchmark}|{stage}|{id}") 的十六进制串。按键升序排列就是一个以种子为盐的随机置换，
  取前 k 个即从总体里均匀、不放回地抽 k 个。不依赖任何随机数库的实现，换语言、换版本都能复现。
- MMLU-Pro（test，按 question_id）：14 个类别各取排序键（stage="stratified"）最小的 7 题，共 98 题；再在其余题目里
  （不分类别）取排序键（stage="fill"）最小的 2 题，凑满 100。
- GSM8K（main/test，id = 行号，从 0 起）：排序键（stage="uniform"）最小的 50 题。
- IFEval（train，id = key）：排序键（stage="uniform"）最小的 50 题。
- 题目文件里每项的题目按 id 升序排列（--limit N 取的是每项的前 N 题）。
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.health import protocol as P

# GSM8K 参考答案的最后一行："#### 72"、"#### 1,450,000"、"#### -3"
_GSM8K_FINAL = re.compile(r"\n#### (-?(?:\d{1,3}(?:,\d{3})+|\d+))\Z")


# ---------------------------------------------------------------------------
# 抽样
# ---------------------------------------------------------------------------


def rank_key(benchmark: str, stage: str, item_id: int, *, seed: int = P.SEED) -> str:
    """抽样排序键（见模块说明）。"""
    if isinstance(item_id, bool) or not isinstance(item_id, int):
        raise TypeError(f"id 必须是 int，收到 {item_id!r}")
    return hashlib.sha256(f"{seed}|{benchmark}|{stage}|{item_id}".encode("utf-8")).hexdigest()


def hash_sample(ids: Iterable[int], k: int, *, benchmark: str, stage: str, seed: int = P.SEED) -> list[int]:
    """按排序键取最小的 k 个 id（返回顺序即排序键顺序）。id 重复或不足 k 个时报错。"""
    ids = list(ids)
    if len(set(ids)) != len(ids):
        raise ValueError(f"{benchmark}/{stage}：id 有重复")
    if not 0 <= k <= len(ids):
        raise ValueError(f"{benchmark}/{stage}：要抽 {k} 个，总体只有 {len(ids)} 个")
    return sorted(ids, key=lambda i: (rank_key(benchmark, stage, i, seed=seed), i))[:k]


def select_mmlu_pro(population: Sequence[tuple[int, str]], *, seed: int = P.SEED) -> dict[int, str]:
    """population：(question_id, category) 的序列。返回 {question_id: stage}，stage 为 stratified 或 fill。"""
    by_cat: dict[str, list[int]] = defaultdict(list)
    for qid, cat in population:
        by_cat[cat].append(qid)
    if len(by_cat) != P.MMLU_PRO_N_CATEGORIES:
        raise ValueError(f"MMLU-Pro 应有 {P.MMLU_PRO_N_CATEGORIES} 个类别，实有 {len(by_cat)} 个：{sorted(by_cat)}")
    chosen: dict[int, str] = {}
    for cat in sorted(by_cat):
        for qid in hash_sample(by_cat[cat], P.MMLU_PRO_PER_CATEGORY, benchmark="mmlu_pro",
                               stage=P.STAGE_MMLU_PRO_STRATIFIED, seed=seed):
            chosen[qid] = P.STAGE_MMLU_PRO_STRATIFIED
    rest = [qid for qid, _ in population if qid not in chosen]
    for qid in hash_sample(rest, P.MMLU_PRO_FILL, benchmark="mmlu_pro", stage=P.STAGE_MMLU_PRO_FILL, seed=seed):
        chosen[qid] = P.STAGE_MMLU_PRO_FILL
    return chosen


def select_uniform(benchmark: str, ids: Iterable[int], *, seed: int = P.SEED) -> list[int]:
    """GSM8K / IFEval：均匀抽 COUNTS[benchmark] 个。"""
    return hash_sample(ids, P.COUNTS[benchmark], benchmark=benchmark, stage=P.STAGE_UNIFORM, seed=seed)


# ---------------------------------------------------------------------------
# 从原始数据构造题目
# ---------------------------------------------------------------------------


def gsm8k_final_answer(answer_text: str) -> int:
    """GSM8K 参考解答最后一行 "#### N" 里的整数（去掉千分位逗号）。"""
    m = _GSM8K_FINAL.search(answer_text)
    if m is None:
        raise ValueError(f"GSM8K 参考答案没有以 '\\n#### <整数>' 结尾：{answer_text[-80:]!r}")
    return int(m.group(1).replace(",", ""))


def _check_mmlu_pro_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    ids = [r["question_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("MMLU-Pro 的 question_id 有重复")
    for r in rows:
        opts = r["options"]
        if not isinstance(opts, list) or not all(isinstance(o, str) for o in opts):
            raise ValueError(f"MMLU-Pro {r['question_id']}：options 不是字符串列表")
        letters = P.option_letters(len(opts))
        if not 0 <= r["answer_index"] < len(opts) or r["answer"] != letters[r["answer_index"]]:
            raise ValueError(f"MMLU-Pro {r['question_id']}：answer={r['answer']!r} 与 answer_index={r['answer_index']} 不一致")


def build_items(mmlu_pro_rows: Sequence[Mapping[str, Any]], gsm8k_rows: Sequence[Mapping[str, Any]],
                ifeval_rows: Sequence[Mapping[str, Any]], *, seed: int = P.SEED) -> dict[str, Any]:
    """三个来源的全部原始行（按文件行序）→ 题目文件的内容（dict）。只保留评分需要的字段。"""
    for name, rows in (("mmlu_pro", mmlu_pro_rows), ("gsm8k", gsm8k_rows), ("ifeval", ifeval_rows)):
        if len(rows) != P.SOURCES[name].n_rows:
            raise ValueError(f"{name}：{len(rows)} 行，固定 revision 应有 {P.SOURCES[name].n_rows} 行")
    _check_mmlu_pro_rows(mmlu_pro_rows)
    chosen = select_mmlu_pro([(r["question_id"], r["category"]) for r in mmlu_pro_rows], seed=seed)
    mmlu = [
        {
            "id": r["question_id"],
            "category": r["category"],
            "src": r["src"],
            "stage": chosen[r["question_id"]],
            "question": r["question"],
            "options": list(r["options"]),
            "answer": r["answer"],
            "answer_index": r["answer_index"],
        }
        for r in sorted(mmlu_pro_rows, key=lambda r: r["question_id"])
        if r["question_id"] in chosen
    ]
    gsm_ids = sorted(select_uniform("gsm8k", range(len(gsm8k_rows)), seed=seed))
    gsm = [{"id": i, "question": gsm8k_rows[i]["question"], "answer": gsm8k_final_answer(gsm8k_rows[i]["answer"])}
           for i in gsm_ids]
    by_key = {r["key"]: r for r in ifeval_rows}
    if len(by_key) != len(ifeval_rows):
        raise ValueError("IFEval 的 key 有重复")
    ife_ids = sorted(select_uniform("ifeval", by_key, seed=seed))
    ife = [
        {
            "id": k,
            "prompt": by_key[k]["prompt"],
            "instruction_id_list": list(by_key[k]["instruction_id_list"]),
            "kwargs": [dict(kw) for kw in by_key[k]["kwargs"]],
        }
        for k in ife_ids
    ]
    return {
        "schema_version": P.ITEMS_SCHEMA_VERSION,
        "protocol_version": P.PROTOCOL_VERSION,
        "generator": "scripts/build_health_items.py (DEV-0008)",
        "seed": seed,
        "sampling": {
            "rank_key": "sha256(f'{seed}|{benchmark}|{stage}|{id}') hex, ascending; take the first k",
            "mmlu_pro": f"{P.MMLU_PRO_PER_CATEGORY} per category ({P.MMLU_PRO_N_CATEGORIES} categories, "
                        f"stage '{P.STAGE_MMLU_PRO_STRATIFIED}') + {P.MMLU_PRO_FILL} from the rest "
                        f"(stage '{P.STAGE_MMLU_PRO_FILL}'); id = question_id",
            "gsm8k": f"{P.COUNTS['gsm8k']} uniform (stage '{P.STAGE_UNIFORM}'); id = 0-based row index in {P.SOURCES['gsm8k'].split}",
            "ifeval": f"{P.COUNTS['ifeval']} uniform (stage '{P.STAGE_UNIFORM}'); id = key",
            "order": "items sorted by id within each benchmark",
        },
        "sources": {
            name: {
                "repo": s.repo,
                "revision": s.revision,
                "config": s.config,
                "split": s.split,
                "file": s.file,
                "file_sha256": s.file_sha256,
                "license": s.license,
                "n_rows": s.n_rows,
            }
            for name, s in P.SOURCES.items()
        },
        "counts": dict(P.COUNTS),
        "items": {"mmlu_pro": mmlu, "gsm8k": gsm, "ifeval": ife},
    }


def population_index(mmlu_pro_rows: Sequence[Mapping[str, Any]], gsm8k_rows: Sequence[Any],
                     ifeval_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """重新抽样所需的最小信息（测试 fixture，离线复现抽样）：MMLU-Pro 的 (id, 类别)、GSM8K 的行数、IFEval 的 key。

    MMLU-Pro 的 id 以升序连续段 [first, last] 存，类别按行序存成 (类别, 行数) 的游程；两者都对原始行序成立时才写出。
    """
    ids = [r["question_id"] for r in mmlu_pro_rows]
    if ids != sorted(ids):
        raise ValueError("MMLU-Pro 的 question_id 不是按行序升序，不能用连续段表示")
    ranges: list[list[int]] = []
    for qid in ids:
        if ranges and qid == ranges[-1][1] + 1:
            ranges[-1][1] = qid
        else:
            ranges.append([qid, qid])
    runs: list[list[Any]] = []
    for r in mmlu_pro_rows:
        if runs and runs[-1][0] == r["category"]:
            runs[-1][1] += 1
        else:
            runs.append([r["category"], 1])
    return {
        "note": "population index for offline re-sampling (tests/test_health_items.py); generated by scripts/build_health_items.py",
        "mmlu_pro": {"id_ranges_inclusive": ranges, "category_runs": runs},
        "gsm8k": {"n_rows": len(gsm8k_rows)},
        "ifeval": {"keys": [r["key"] for r in ifeval_rows]},
    }


def expand_population_index(index: Mapping[str, Any]) -> tuple[list[tuple[int, str]], int, list[int]]:
    """population_index 的逆：返回 (MMLU-Pro 的 (id, 类别) 列表, GSM8K 行数, IFEval key 列表)。"""
    ids = [q for lo, hi in index["mmlu_pro"]["id_ranges_inclusive"] for q in range(lo, hi + 1)]
    cats = [c for c, n in index["mmlu_pro"]["category_runs"] for _ in range(n)]
    if len(ids) != len(cats):
        raise ValueError(f"MMLU-Pro 索引里 id {len(ids)} 个、类别 {len(cats)} 个，不一致")
    return list(zip(ids, cats)), int(index["gsm8k"]["n_rows"]), list(index["ifeval"]["keys"])


# ---------------------------------------------------------------------------
# 题目文件
# ---------------------------------------------------------------------------


def dumps_items(obj: Mapping[str, Any]) -> str:
    """题目文件的规范文本：键序即构造顺序、缩进 1、非 ASCII 原样、末尾换行。同一内容总是得到逐字相同的文本。"""
    return json.dumps(obj, indent=1, ensure_ascii=False) + "\n"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_items(obj: Mapping[str, Any]) -> None:
    """结构校验：数量、分层、字段齐全、类型。不合格抛 ValueError。"""
    if obj.get("schema_version") != P.ITEMS_SCHEMA_VERSION or obj.get("protocol_version") != P.PROTOCOL_VERSION:
        raise ValueError(f"题目文件版本 {obj.get('schema_version')}/{obj.get('protocol_version')} 与代码不一致")
    items = obj["items"]
    if list(items) != list(P.BENCHMARKS):
        raise ValueError(f"题目文件的项 {list(items)} 应为 {list(P.BENCHMARKS)}")
    for name in P.BENCHMARKS:
        ids = [it["id"] for it in items[name]]
        if len(ids) != P.COUNTS[name] or ids != sorted(set(ids)):
            raise ValueError(f"{name}：应有 {P.COUNTS[name]} 题、id 升序且不重复")
    stages = defaultdict(lambda: defaultdict(int))
    for it in items["mmlu_pro"]:
        if set(it) != {"id", "category", "src", "stage", "question", "options", "answer", "answer_index"}:
            raise ValueError(f"MMLU-Pro {it.get('id')}：字段 {sorted(it)} 不符")
        letters = P.option_letters(len(it["options"]))
        if it["answer"] != letters[it["answer_index"]]:
            raise ValueError(f"MMLU-Pro {it['id']}：答案与下标不一致")
        stages[it["stage"]][it["category"]] += 1
    summary = {stage: dict(per_cat) for stage, per_cat in stages.items()}
    strat = summary.get(P.STAGE_MMLU_PRO_STRATIFIED, {})
    if len(strat) != P.MMLU_PRO_N_CATEGORIES or set(strat.values()) != {P.MMLU_PRO_PER_CATEGORY}:
        raise ValueError(f"MMLU-Pro 分层不符：{summary}")
    if set(summary) != {P.STAGE_MMLU_PRO_STRATIFIED, P.STAGE_MMLU_PRO_FILL} or sum(
            summary[P.STAGE_MMLU_PRO_FILL].values()) != P.MMLU_PRO_FILL:
        raise ValueError(f"MMLU-Pro 补足阶段不符：{summary}")
    for it in items["gsm8k"]:
        if set(it) != {"id", "question", "answer"} or isinstance(it["answer"], bool) or not isinstance(it["answer"], int):
            raise ValueError(f"GSM8K {it.get('id')}：字段或答案类型不符")
    for it in items["ifeval"]:
        if set(it) != {"id", "prompt", "instruction_id_list", "kwargs"}:
            raise ValueError(f"IFEval {it.get('id')}：字段 {sorted(it)} 不符")
        if len(it["kwargs"]) != len(it["instruction_id_list"]) or not it["instruction_id_list"]:
            raise ValueError(f"IFEval {it['id']}：kwargs 与 instruction_id_list 不等长或为空")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_items(path: str | Path | None = None, *, expected_sha256: str | None = P.ITEMS_SHA256) -> tuple[dict[str, Any], str]:
    """读题目文件，核对 sha256（expected_sha256=None 时不核对）并做结构校验。返回 (内容, 实际 sha256)。"""
    p = Path(path) if path is not None else repo_root() / P.ITEMS_PATH
    raw = p.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"{p} 的 sha256={digest}，与协议固定的 {expected_sha256} 不一致：题目被改动过，或代码与题目文件不同步")
    obj = json.loads(raw.decode("utf-8"))
    validate_items(obj)
    return obj, digest
