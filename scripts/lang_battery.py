#!/usr/bin/env python
"""大题量的语言读数（DEV-0064；D125-2 (c)、D125-3）：MMLU-Pro 2,000 题的字母作答与 IFEval 全部 541 条，与基座逐题配对。

健康检查（简报 §8.11，scripts/health_check.py，固定 200 题）不动；这里只是把它的两个读数换成更大的题量，口径与它相同：
- **MMLU-Pro 字母作答**（健康检查 v3 的标题读数 mmlu_pro_letter）：提示 protocol.mmlu_pro_prompt，一次 adapter.option_logits
  （src/health/run.letter_text_only，关思考，选项为本题的实际字母），评分 scoring.score_mmlu_pro_letter；口径 accuracy 与
  mean_p_gold。逐题记录与健康检查的 mmlu_pro_letter 记录逐字段相同（同一组函数：letter_text_only → check_spellings →
  letter_record → score_mmlu_pro_letter）。不跑健康检查的自由作答诊断（free256 / fmt32）。
- **IFEval**：提示 protocol.ifeval_prompt，一次 greedy 生成（max_new_tokens 1024，关思考），官方 checker 的 strict 与 loose
  （src/health/ifeval；langdetect 种子 0）；四个读数 prompt_level_strict、instruction_level_strict、prompt_level_loose、
  instruction_level_loose。每题就是健康检查的 run.run_one("ifeval", …)，原样调用。
- 预热（generate 一次、option_logits 一次，不计入记账）、A–J 写法的核对、零张图、instruction_first、greedy 参数、记账口径
  都与健康检查相同（src/health/run.py 的同一批函数与 protocol.py 的同一批常量）。
- 计数：scoring.summarize（健康检查的同一条汇总路径）。
- 跑任何一题之前，要跑的全部提示先过一遍适配器的 build_inputs（只构造输入，不调用模型；几秒）：适配器拒绝含结构性 token
  或内部哨兵的指令，健康检查只核对过自己的 150 题。有过不了的，一次列出全部 id（lang_battery.json 的 preflight），退出码 2。

题目（在看到任何分数之前固定；battery 版本 BATTERY_VERSION）：
- 来源与健康检查相同（protocol.SOURCES，固定 revision 与 sha256）：
  - MMLU-Pro：TIGER-Lab/MMLU-Pro@b189ec76…，default/test，12,032 题，HF card license MIT；
  - IFEval：google/IFEval@966cd895…，train，541 条，Apache-2.0。
  数据只限非商业研究，不再分发：原始文件放 --raw-dir（仓库外），输出只存题目 id、提示的 sha256 与模型输出，不存题面。
- MMLU-Pro：总体是 test 全部题目去掉健康检查的 100 题（configs/health_check_items.json，sha256 由 protocol.ITEMS_SHA256 钉住；
  真模型运行要求 100 题都在总体里，排除后重叠为 0，输出里记下）。抽样同健康检查的结构、换一个盐：排序键
  sha256(f"{SALT}|{purpose}|{question_id}")（src/data/subsets.hash_rank，SALT = "lang_battery_v1"），
  14 个学科各取排序键（purpose = "mmlu_pro_stratified"）最小的 N // 14 题，再在其余题目里不分学科取排序键
  （purpose = "mmlu_pro_fill"）最小的 N mod 14 题。N = 2,000 时各学科 142 题、补足 12 题。题目按 question_id 升序。
- IFEval：全部 541 条（按 key 升序），不抽样；其中 50 条也是健康检查的题，逐题记 in_health_check。
- 原始文件在 --raw-dir 下的布局同 scripts/build_health_items.py：<raw-dir>/<仓库名里的 / 换成 __>/<文件>。真模型运行核对
  文件 sha256 与行数（与 protocol.SOURCES 相同）；--tiny 只记录。

统计：每个读数给点估计与 bootstrap 95% CI（src.stats.bootstrap_ci，percentile 法，重采样次数、置信水平、种子取
src.constants）；按题重采样，instruction-level 的统计量是 Σ遵循的指令数 / Σ指令数（src.stats.ratio_of_sums），mean_p_gold
是 p_gold 的均值。MMLU-Pro 另给分学科的准确率与 CI。
--compare BASE OTHER：逐题配对（两边的题目选择、数据文件与逐题 id 必须相同），每个读数给 OTHER − BASE 的差值（百分点）、
配对 bootstrap 95% CI（scoring.paired_ci_pp，即健康检查 --compare 的同一个函数；mean_p_gold 用 src.stats.paired_bootstrap_ci）、
丢题 / 得题数与精确符号检验 p 值（scoring.sign_test_p），另给 MMLU-Pro 分学科的差值与 CI。只描述，不设标红线
（标红是健康检查 §8.11 的规则）。

输出（--out 是一个目录；其中的 lang_battery.json、predictions.jsonl、compute.json、README.md、config.yaml、meta.json 已存在时拒绝）：
- lang_battery.json：运行记录（runlog 的 config 与 meta）、模型与检查点、生成参数、数据来源与抽样（全部题目 id 与其 sha256）、
  各读数的分数与 CI、分学科、诊断、记账、评分记录（punkt_tab 哈希等）、predictions.jsonl 的 sha256；
- predictions.jsonl：一行一题：{"reading", <与健康检查相同的逐题记录>, "meta": {学科与抽样阶段，或 in_health_check}}，边跑边写；
  按 RUN-0069 的逐题记录大小（字母作答约 0.7 KB、IFEval 约 3.2 KB）估计每次约 3 MB，在推送的 5 MB 大文件线以下；
- compute.json：src/accounting 的总量（按项的分量在 lang_battery.json 的 compute 里）；
- README.md：这次运行的摘要；
- config.yaml、meta.json：src/runlog.write_run_metadata；
- --checkpoint 时另有 lang_battery_merge_check.json（评测前的合并核对）。

模型加载与 --checkpoint：调用 scripts/health_check.py 的 make_adapter / make_checkpoint_adapter（按文件路径载入那个脚本，
不复制）：--checkpoint 先做评测前的合并核对（t_text 口径；--merge-check-n、--data-root 同健康检查），不通过退出码 1，
再 factory.load_trained(merge=True)。--model 必须与检查点 manifest 的构造方式相同。思考开关与健康检查相同（关）。

数据下载（一次；不进仓库；HPC 的计算节点设了 HF_HUB_OFFLINE=1，下载在 hpcmgt 上做，经 hf-mirror）：
    VI=/home/mail/chenyuma/visint; WT=<任务的工作树>; PY=$VI/env/bin/python; RAW=$VI/data/lang_battery_raw
    cd $WT && env -u HF_HUB_OFFLINE HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 HF_HOME=$VI/hf \\
        $PY scripts/lang_battery.py --fetch --raw-dir $RAW
  --fetch 调 scripts/build_health_items.py 的 fetch（健康检查建题时的同一个函数）：按固定 revision 下载 MMLU-Pro 与 IFEval 的
  原始文件、逐个核对 sha256，打印路径与哈希。两个文件共几 MB（未在开发机实测）。

服务器命令（hpcgpu06，Slurm 157296 上 srun，同一节点配对；环境变量同 RUN-0069 的 w2c.sh：HF_HOME=$VI/hf、
HF_HUB_CACHE=$VI/hf/hub、HF_HUB_OFFLINE=1、TMPDIR=$VI/tmp、PIP_NO_CACHE_DIR=1、NLTK_DATA=$VI/b200/nltk_data、PYTHONHASHSEED=0）：
    OUT=$VI/repo/experiments/W2/NNNN_lang_battery_hpcgpu06
    $PY scripts/lang_battery.py --model qwen3.5-9b --raw-dir $RAW --label base --out $OUT/base
    $PY scripts/lang_battery.py --model qwen3.5-9b --raw-dir $RAW --label w2c --data-root $VI/b200/data \\
        --checkpoint $VI/repo/experiments/W2/0069_w2c_hpcgpu06/0069_train/w2c/qwen3.5-9b/seed0/ckpt/final/model --out $OUT/w2c
    $PY scripts/lang_battery.py --model qwen3.5-9b --raw-dir $RAW --label e3b_c3 --data-root $VI/b200/data \\
        --checkpoint $VI/ckpt/hpcgpu06/0056/e3b_c3/qwen3.5-9b/seed0/ckpt/final/model --out $OUT/e3b_c3
    $PY scripts/lang_battery.py --compare $OUT/base $OUT/w2c --out $OUT/compare_w2c.json
    $PY scripts/lang_battery.py --compare $OUT/base $OUT/e3b_c3 --out $OUT/compare_e3b_c3.json
  W2-c 加回放（D125-2 (b)）训完以后用同一条命令，换 --checkpoint、--label 与 --out，再与 $OUT/base 配对。
  E3b+C3 的检查点是 RUN-0056 的 e3b_c3 种子 0（reports/0056-hpc-06-9b.md："e3b_c3_s0 完成：final → …"）。

用时估计（按 RUN-0069 在 hpcgpu06 上的 9B 健康检查实测折算；A100 bf16，逐题单条调用，与健康检查相同）：
- MMLU-Pro 字母作答：每题约 0.063 s（RUN-0069 基座 100 题 6.3 s）→ 2,000 题约 2–3 分钟；
- IFEval：基座每题约 10.1 s（50 题 507 s），W2-c 约 8.6 s（428 s）→ 541 题基座约 1.5 小时、W2-c 约 1.3 小时；E3b+C3 未实测，
  按基座估；这 50 题的回答长度不一定代表全部 541 条；
- 加载：基座约 0.5 分钟；带检查点另加合并核对，约 4 分钟（RUN-0069 health_w2c 墙钟 1,065 s 减去模型调用 830 s）；
- 合计每个模型约 1.4–1.6 小时，三个约 4.5 小时（建议上限 7 小时）；--compare 几秒。

开发机彩排（小型随机模型，CPU；数值无意义，只验证全流程）：
    python -m tests._lang_battery_fixtures <raw 目录>        # MMLU-Pro validation 的 70 题当总体，IFEval 52 条
    NLTK_DATA=tests/fixtures/health/nltk_data python scripts/lang_battery.py --tiny --model qwen3.5-0.8b \\
        --raw-dir <raw 目录> --limit 3 --out <目录>
  --tiny 时 --mmlu-pro-n 默认 TINY_MMLU_PRO_N（每学科 2 题）；--limit N：每项只跑前 N 题（调试用，结果不拿去比较）。

退出码：0 = 完成；1 = --checkpoint 的合并核对没有通过，拒绝评测；2 = 出错（已有结果与报错先写进 lang_battery.json）或用法错误。
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import sys
import time
import traceback
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from src import runlog  # noqa: E402
from src.accounting import Compute, merge, write_compute_json  # noqa: E402
from src.constants import BOOTSTRAP_CI_LEVEL, BOOTSTRAP_N_RESAMPLES, BOOTSTRAP_SEED  # noqa: E402
from src.data.subsets import hash_rank  # noqa: E402
from src.health import protocol as P  # noqa: E402
from src.health.items import _check_mmlu_pro_rows, load_items  # noqa: E402
from src.health.parquet_lite import read_rows  # noqa: E402
from src.health.run import (  # noqa: E402
    check_spellings,
    generate_text_only,
    letter_record,
    letter_text_only,
    run_one,
    spelling_table,
)
from src.health.scoring import (  # noqa: E402
    MEAN_P_GOLD,
    METRICS,
    item_counts,
    mean_fraction,
    paired_ci_pp,
    pct1,
    round1,
    score_mmlu_pro_letter,
    sign_test_p,
    summarize,
    unit_outcomes,
)
from src.stats import bootstrap_ci, paired_bootstrap_ci, ratio_of_sums  # noqa: E402

BATTERY_VERSION = 1
SALT = "lang_battery_v1"
STAGES = (P.STAGE_MMLU_PRO_STRATIFIED, P.STAGE_MMLU_PRO_FILL)
PURPOSES: Mapping[str, str] = {stage: f"mmlu_pro_{stage}" for stage in STAGES}  # 进入排序键，改名就改变抽样结果
DEFAULT_MMLU_PRO_N = 2000
TINY_MMLU_PRO_N = 2 * P.MMLU_PRO_N_CATEGORIES  # --tiny 的总体是 MMLU-Pro validation（每学科 5 题）

LETTER = P.MMLU_PRO_LETTER
IFEVAL = "ifeval"
# 读数 → 项；顺序即运行与输出的顺序
READINGS: Mapping[str, str] = {LETTER: "mmlu_pro", IFEVAL: "ifeval"}
BENCHMARKS = tuple(READINGS.values())
LETTER_METRICS = ("accuracy", MEAN_P_GOLD)
READING_METRICS: Mapping[str, tuple[str, ...]] = {LETTER: LETTER_METRICS, IFEVAL: METRICS["ifeval"]}
HEADLINE_METRIC: Mapping[str, str] = {LETTER: P.HEADLINE["mmlu_pro"], IFEVAL: P.HEADLINE["ifeval"]}

OUT_JSON = "lang_battery.json"
PREDICTIONS = "predictions.jsonl"
COMPUTE_JSON = "compute.json"
README = "README.md"
OUTPUT_FILES = (OUT_JSON, PREDICTIONS, COMPUTE_JSON, README, "config.yaml", "meta.json")
STATE_KEYS = ("preflight", "warmup", "option_spellings", "compute")  # run_battery 的 state 里原样写进 lang_battery.json 的键

HEALTH_CHECK_SCRIPT = REPO_ROOT / "scripts" / "health_check.py"
BUILD_ITEMS_SCRIPT = REPO_ROOT / "scripts" / "build_health_items.py"
LICENSE_NOTE = ("data for non-commercial research only, not redistributed: raw files stay outside the repository (--raw-dir); "
                "outputs keep item ids, prompt sha256 and model outputs, never the question text")


class DataError(ValueError):
    """原始数据缺失、与固定的 sha256 / 行数不符，或抽样做不出来。"""


class CompareError(ValueError):
    """两次运行不能配对（题目、数据或 battery 版本不同，或不是一次跑完的运行）。"""


class PromptError(ValueError):
    """有的题的提示过不了适配器的 build_inputs（如含结构性 token）；跑任何一题之前一次核对完。"""


def _load_script(path: Path, name: str) -> Any:
    """按文件路径载入一个脚本（同 src/e3/merge_check.load_cti 的做法）。"""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def health_check_script() -> Any:
    """scripts/health_check.py：模型加载（make_adapter / make_checkpoint_adapter）、IFEval 依赖核对、评分记录（复用，不复制）。"""
    return _load_script(HEALTH_CHECK_SCRIPT, "_visint_health_check_script")


# ---------------------------------------------------------------------------
# 原始数据与题目
# ---------------------------------------------------------------------------


def raw_path(raw_dir: str | Path, name: str) -> Path:
    """原始文件的位置：<raw-dir>/<仓库名里的 / 换成 __>/<文件>（scripts/build_health_items.py 的 fetch 的布局；tests 核对相同）。"""
    s = P.SOURCES[name]
    return Path(raw_dir) / s.repo.replace("/", "__") / s.file


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_raw(raw_dir: str | Path, *, pinned: bool) -> dict[str, dict[str, Any]]:
    """读 MMLU-Pro test 与 IFEval 的原始文件。pinned（真模型）时文件 sha256 与行数必须与 protocol.SOURCES 相同。"""
    out: dict[str, dict[str, Any]] = {}
    for name in BENCHMARKS:
        src = P.SOURCES[name]
        path = raw_path(raw_dir, name)
        if not path.is_file():
            raise DataError(f"没有 {path}（{src.repo}@{src.revision}:{src.file}）。下载：scripts/lang_battery.py --fetch --raw-dir "
                            f"{raw_dir}（见模块说明；HPC 上经 hf-mirror，在 hpcmgt 上做）")
        digest = runlog.sha256_file(path)
        if pinned and digest != src.file_sha256:
            raise DataError(f"{path} 的 sha256={digest}，与固定的 {src.file_sha256}（{src.repo}@{src.revision}）不同")
        rows = read_rows(path) if path.suffix == ".parquet" else read_jsonl(path)
        if pinned and len(rows) != src.n_rows:
            raise DataError(f"{path}：{len(rows)} 行，固定 revision 应有 {src.n_rows} 行")
        out[name] = {"path": str(path), "sha256": digest, "matches_pinned": digest == src.file_sha256, "rows": rows}
    _check_mmlu_pro_rows(out["mmlu_pro"]["rows"])
    keys = [r["key"] for r in out["ifeval"]["rows"]]
    if len(set(keys)) != len(keys):
        raise DataError("IFEval 的 key 有重复")
    return out


def mmlu_pro_item(row: Mapping[str, Any], stage: str) -> dict[str, Any]:
    """MMLU-Pro 的一行 → 题目（字段同健康检查题目文件的 MMLU-Pro 题，src/health/items.build_items；stage 是本 battery 的抽样阶段）。"""
    return {"id": row["question_id"], "category": row["category"], "src": row["src"], "stage": stage,
            "question": row["question"], "options": list(row["options"]), "answer": row["answer"],
            "answer_index": row["answer_index"]}


def ifeval_item(row: Mapping[str, Any]) -> dict[str, Any]:
    """IFEval 的一行 → 题目（字段同健康检查题目文件的 IFEval 题）。"""
    return {"id": row["key"], "prompt": row["prompt"], "instruction_id_list": list(row["instruction_id_list"]),
            "kwargs": [dict(kw) for kw in row["kwargs"]]}


def select_mmlu_pro(population: Sequence[tuple[int, str]], n: int, *, exclude: Iterable[int] = ()) -> dict[int, str]:
    """population：(question_id, 学科)。去掉 exclude 之后，每学科取 n // 学科数 题、再不分学科补 n mod 学科数 题。
    返回 {question_id: 抽样阶段}（规则见模块说明）。学科必须是 protocol.MMLU_PRO_N_CATEGORIES 个，题不够时报错。"""
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise DataError(f"n 必须是正整数，收到 {n!r}")
    excluded = set(exclude)
    by_cat: dict[str, list[int]] = defaultdict(list)
    for qid, cat in population:
        if qid not in excluded:
            by_cat[cat].append(qid)
    if len(by_cat) != P.MMLU_PRO_N_CATEGORIES:
        raise DataError(f"MMLU-Pro 应有 {P.MMLU_PRO_N_CATEGORIES} 个学科，实有 {len(by_cat)} 个：{sorted(by_cat)}")
    per_cat, fill = divmod(n, len(by_cat))
    chosen: dict[int, str] = {}
    for cat in sorted(by_cat):
        if len(by_cat[cat]) < per_cat:
            raise DataError(f"MMLU-Pro 学科 {cat!r} 排除之后只有 {len(by_cat[cat])} 题，不够每学科 {per_cat} 题")
        for q in hash_rank(by_cat[cat], PURPOSES[P.STAGE_MMLU_PRO_STRATIFIED], SALT)[:per_cat]:
            chosen[int(q)] = P.STAGE_MMLU_PRO_STRATIFIED
    rest = [q for c in sorted(by_cat) for q in by_cat[c] if q not in chosen]
    if len(rest) < fill:
        raise DataError(f"MMLU-Pro 补足阶段要 {fill} 题，其余只有 {len(rest)} 题")
    for q in hash_rank(rest, PURPOSES[P.STAGE_MMLU_PRO_FILL], SALT)[:fill]:
        chosen[int(q)] = P.STAGE_MMLU_PRO_FILL
    return chosen


def ids_sha256(ids: Iterable[Any]) -> str:
    """一组题目 id 的指纹：按给出的顺序转成字符串、换行连接后的 sha256。"""
    return hashlib.sha256("\n".join(str(i) for i in ids).encode("utf-8")).hexdigest()


def _source_block(name: str) -> dict[str, Any]:
    s = P.SOURCES[name]
    return {"repo": s.repo, "revision": s.revision, "config": s.config, "split": s.split, "file": s.file,
            "file_sha256": s.file_sha256, "license": s.license, "n_rows": s.n_rows}


def build_battery(raw: Mapping[str, Mapping[str, Any]], *, n_mmlu_pro: int, health_items: Mapping[str, Any],
                  pinned: bool) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """原始行 + 健康检查的题目 → ({项: 题目列表（按 id 升序）}, {项: 与题目一一对应的元数据}, 输出 JSON 的 data 一节)。

    元数据：MMLU-Pro 是学科与抽样阶段，IFEval 是 in_health_check（这条是否也是健康检查的题）。
    pinned（真模型）时健康检查的 100 题 MMLU-Pro 必须都在总体里（同一份来源）。排除之后选出的题与它们的重叠必须为 0。"""
    rows = raw["mmlu_pro"]["rows"]
    health_mmlu = [it["id"] for it in health_items["items"]["mmlu_pro"]]
    health_ifeval = {it["id"] for it in health_items["items"]["ifeval"]}
    pop_ids = {r["question_id"] for r in rows}
    in_pop = sorted(set(health_mmlu) & pop_ids)
    if pinned and len(in_pop) != len(health_mmlu):
        raise DataError(f"健康检查的 MMLU-Pro {len(health_mmlu)} 题只有 {len(in_pop)} 题在总体里：数据不是同一份")
    chosen = select_mmlu_pro([(r["question_id"], r["category"]) for r in rows], n_mmlu_pro, exclude=health_mmlu)
    overlap = sorted(set(chosen) & set(health_mmlu))
    if overlap:  # 按构造不会发生
        raise DataError(f"选出的 MMLU-Pro 题与健康检查重叠：{overlap}")
    mmlu = [mmlu_pro_item(r, chosen[r["question_id"]]) for r in sorted(rows, key=lambda r: r["question_id"])
            if r["question_id"] in chosen]
    ife = [ifeval_item(r) for r in sorted(raw["ifeval"]["rows"], key=lambda r: r["key"])]
    per_cat: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for it in mmlu:
        per_cat[it["category"]][it["stage"]] += 1
    per_cat_n, fill_n = divmod(n_mmlu_pro, P.MMLU_PRO_N_CATEGORIES)
    data = {
        "license_note": LICENSE_NOTE,
        "mmlu_pro": {
            "source": _source_block("mmlu_pro"),
            "path": raw["mmlu_pro"]["path"], "sha256": raw["mmlu_pro"]["sha256"],
            "matches_pinned": raw["mmlu_pro"]["matches_pinned"], "n_population": len(rows),
            "sampling": {
                "rank_key": f"sha256(f'{{salt}}|{{purpose}}|{{question_id}}') hex, ascending (src.data.subsets.hash_rank); "
                            f"salt '{SALT}'",
                "salt": SALT, "purposes": dict(PURPOSES), "n": n_mmlu_pro,
                "rule": f"population = test minus the health check's {len(health_mmlu)} MMLU-Pro items; "
                        f"{per_cat_n} per category ({P.MMLU_PRO_N_CATEGORIES} categories, purpose "
                        f"'{PURPOSES[P.STAGE_MMLU_PRO_STRATIFIED]}') + {fill_n} from the rest (purpose "
                        f"'{PURPOSES[P.STAGE_MMLU_PRO_FILL]}'); items sorted by question_id",
            },
            "health_check": {"n_items": len(health_mmlu), "n_in_population": len(in_pop),
                             "n_excluded_from_population": len(in_pop), "n_overlap": len(overlap)},
            "n_items": len(mmlu),
            "per_category": {c: dict(v) for c, v in sorted(per_cat.items())},
            "ids": [it["id"] for it in mmlu],
            "ids_sha256": ids_sha256(it["id"] for it in mmlu),
        },
        "ifeval": {
            "source": _source_block("ifeval"),
            "path": raw["ifeval"]["path"], "sha256": raw["ifeval"]["sha256"],
            "matches_pinned": raw["ifeval"]["matches_pinned"], "n_population": len(ife),
            "sampling": "all prompts, sorted by key",
            "health_check": {"n_items": len(health_ifeval), "n_overlap": sum(it["id"] in health_ifeval for it in ife),
                             "overlap_ids": sorted(it["id"] for it in ife if it["id"] in health_ifeval)},
            "n_items": len(ife),
            "ids_sha256": ids_sha256(it["id"] for it in ife),
        },
    }
    metas = {"mmlu_pro": [{"category": it["category"], "stage": it["stage"]} for it in mmlu],
             "ifeval": [{"in_health_check": it["id"] in health_ifeval} for it in ife]}
    return {"mmlu_pro": mmlu, "ifeval": ife}, metas, data


# ---------------------------------------------------------------------------
# 运行（每题与健康检查同一组函数）
# ---------------------------------------------------------------------------


def preflight_prompts(adapter: Any, items: Mapping[str, list[Mapping[str, Any]]], *, limit: int | None = None,
                      state: dict[str, Any] | None = None) -> dict[str, Any]:
    """跑任何一题之前，用适配器的 build_inputs 把要跑的全部提示构造一遍（只构造输入，不调用模型）。健康检查只核对过
    自己的 150 题；这里有过不了的（适配器拒绝含结构性 token 或内部哨兵的指令），一次列出全部 id 并抛 PromptError，
    不在跑了一个多小时之后才失败。结果（含过不了的题）记进 state["preflight"]。"""
    t0 = time.perf_counter()
    bad, n = [], 0
    for bench in BENCHMARKS:
        for it in (items[bench][:limit] if limit is not None else items[bench]):
            n += 1
            try:
                adapter.build_inputs([], P.PROMPT_BUILDERS[bench](it), P.ORDER)
            except (ValueError, TypeError) as e:
                bad.append({"benchmark": bench, "id": it["id"], "error": str(e)})
    out = {"n_checked": n, "n_failed": len(bad), "failed": bad, "wall_time_s": round(time.perf_counter() - t0, 2)}
    if state is not None:
        state["preflight"] = out
    if bad:
        raise PromptError(f"{len(bad)} 题的提示过不了 build_inputs（前 10 题：{bad[:10]}）；全部见 preflight.failed")
    return out


def warmup(adapter: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """计时前的预热（同 src/health/run.run_items：generate 一次、option_logits 一次，不计入记账）；返回 (记录, A–J 的写法表)。"""
    warm, warm_wall = generate_text_only(adapter, P.WARMUP_PROMPT, max_new_tokens=P.WARMUP_MAX_NEW_TOKENS)
    letters, letters_wall = letter_text_only(adapter, P.WARMUP_PROMPT, P.OPTION_LETTERS)
    info = {"prompt": P.WARMUP_PROMPT, "max_new_tokens": P.WARMUP_MAX_NEW_TOKENS,
            "n_generated_tokens": warm.n_generated_tokens, "wall_time_s": round(warm_wall, 4),
            "option_logits": {"options": P.OPTION_LETTERS, "wall_time_s": round(letters_wall, 4)},
            "counted_in_compute": False}
    return info, spelling_table(letters)


def run_letter(adapter: Any, item: Mapping[str, Any], spellings: Mapping[str, Any] | None) -> tuple[dict[str, Any], Compute]:
    """MMLU-Pro 的受限字母读数一题：健康检查 run.run_one 里 mmlu_pro_letter 那一支的同一组函数（不跑自由作答）。"""
    prompt = P.mmlu_pro_prompt(item)
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    scores, wall = letter_text_only(adapter, prompt, P.option_letters(len(item["options"])))
    check_spellings(spellings, scores, item["id"])
    generation = letter_record(item, prompt_hash, scores, wall=wall)
    record = {**generation, **score_mmlu_pro_letter(item, generation)}
    return record, Compute.from_generation(scores, new_round=True).as_sample(wall_time_s=wall)


def run_ifeval(adapter: Any, item: Mapping[str, Any]) -> tuple[dict[str, Any], Compute]:
    """IFEval 一题：就是健康检查的 run.run_one("ifeval", …)。"""
    records, compute = run_one(adapter, "ifeval", item, score_ifeval=True)
    return records["ifeval"], compute


def prediction_line(reading: str, record: Mapping[str, Any], meta: Mapping[str, Any]) -> dict[str, Any]:
    """predictions.jsonl 的一行：读数名、与健康检查相同的逐题记录、题目的元数据。"""
    return {"reading": reading, **record, "meta": dict(meta)}


def run_battery(adapter: Any, items: Mapping[str, list[Mapping[str, Any]]], metas: Mapping[str, list[Mapping[str, Any]]], *,
                predictions: Path, limit: int | None = None, state: dict[str, Any] | None = None,
                progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    """先核对全部提示（preflight_prompts），预热，然后按 READINGS 的顺序逐题跑，每题一行写进 predictions（边跑边写；
    中途出错时 state 与文件里都有已跑的题）。

    state 的键：preflight、warmup、option_spellings、records（读数 → predictions 的行）、compute（项 → compute.json 字典，
    外加 total）、compute_total（总量的 Compute，写 compute.json 用）。"""
    state = {} if state is None else state
    preflight_prompts(adapter, items, limit=limit, state=state)
    state["warmup"], state["option_spellings"] = warmup(adapter)
    state["records"] = {name: [] for name in READINGS}
    state["compute"] = {}
    totals: dict[str, Compute] = {}
    with predictions.open("w", encoding="utf-8") as f:
        for reading, bench in READINGS.items():
            todo = list(zip(items[bench], metas[bench]))
            todo = todo[:limit] if limit is not None else todo
            totals[bench] = Compute()
            t0 = time.perf_counter()
            every = 100 if bench == "mmlu_pro" else 10
            for k, (item, meta) in enumerate(todo, 1):
                if bench == "mmlu_pro":
                    record, compute = run_letter(adapter, item, state["option_spellings"])
                else:
                    record, compute = run_ifeval(adapter, item)
                line = prediction_line(reading, record, meta)
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
                f.flush()
                state["records"][reading].append(line)
                totals[bench] = totals[bench] + compute
                state["compute"][bench] = totals[bench].to_json_dict()
                if progress is not None and (k % every == 0 or k == len(todo)):
                    el = time.perf_counter() - t0
                    progress(f"{reading} {k}/{len(todo)}  {el:.0f} s，预计还要 {el / k * (len(todo) - k):.0f} s")
    state["compute_total"] = merge(totals.values())
    state["compute"]["total"] = state["compute_total"].to_json_dict()
    return state


# ---------------------------------------------------------------------------
# 汇总与 CI
# ---------------------------------------------------------------------------


def _ci_pct(ci: Any) -> list[float] | None:
    return None if ci is None else [round1(Fraction(ci.low) * 100), round1(Fraction(ci.high) * 100)]


def point_ci_pct(bench: str, metric: str, records: list[Mapping[str, Any]]) -> list[float] | None:
    """一个读数的 bootstrap 95% CI（百分数，一位小数）。按题重采样：二值口径是均值；instruction-level 是 Σ遵循 / Σ指令
    （ratio_of_sums，同一题的几条指令一起抽）；mean_p_gold 是 p_gold 的均值。没有题时为 None。"""
    if not records:
        return None
    if metric == MEAN_P_GOLD:
        return _ci_pct(bootstrap_ci(np.array([float(r["p_gold"]) for r in records], dtype=np.float64)))
    counts = item_counts(bench, metric, records)
    if metric.startswith("instruction_level_"):
        return _ci_pct(bootstrap_ci(np.array(counts, dtype=np.float64), statistic=ratio_of_sums))
    return _ci_pct(bootstrap_ci(np.array([c for c, _ in counts], dtype=np.float64)))


def summarize_battery(records: Mapping[str, list[Mapping[str, Any]]]) -> dict[str, Any]:
    """逐题记录（predictions 的行）→ 分数（scoring.summarize 的同一条计数路径）加 CI、标题、分学科、诊断。"""
    s = summarize({name: list(records[name]) for name in READINGS})
    if s["unscored"]:
        raise ValueError(f"有未评分的读数：{s['unscored']}")
    scores: dict[str, Any] = {}
    for name, bench in READINGS.items():
        scores[name] = {metric: {**s["scores"][name][metric], "ci95_pct": point_ci_pct(bench, metric, list(records[name]))}
                        for metric in READING_METRICS[name]}
    by_cat: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for r in records[LETTER]:
        by_cat[r["meta"]["category"]].append(r)
    by_category = {}
    for cat in sorted(by_cat):
        units = unit_outcomes("mmlu_pro", "accuracy", by_cat[cat])
        by_category[cat] = {"n": len(units), "n_correct": sum(units.values()), "pct": pct1(sum(units.values()), len(units)),
                            "ci95_pct": point_ci_pct("mmlu_pro", "accuracy", by_cat[cat])}
    return {
        "scores": scores,
        "headline": {name: scores[name][HEADLINE_METRIC[name]]["pct"] for name in READINGS},
        "mmlu_pro_by_category": by_category,
        "diagnostics": {name: s["diagnostics"][name] for name in READINGS},
        "statistics": {"ci95_pct": "bootstrap over questions, percentile interval (src.stats.bootstrap_ci); instruction-level "
                                   "metrics resample questions with src.stats.ratio_of_sums; mean_p_gold is the mean of p_gold",
                       "bootstrap": {"n_resamples": BOOTSTRAP_N_RESAMPLES, "level": BOOTSTRAP_CI_LEVEL, "seed": BOOTSTRAP_SEED}},
    }


# ---------------------------------------------------------------------------
# 比较：逐题配对
# ---------------------------------------------------------------------------


def load_run(path: str | Path) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], Path]:
    """一次运行（--out 目录或其中的 lang_battery.json）→ (lang_battery.json, {读数: predictions 的行}, json 的路径)。"""
    p = Path(path)
    if p.is_dir():
        p = p / OUT_JSON
    doc = json.loads(p.read_text(encoding="utf-8"))
    if doc.get("ok") is not True:
        raise CompareError(f"{p} 不是一次跑完的运行（ok={doc.get('ok')!r}）")
    if doc.get("battery_version") != BATTERY_VERSION:
        raise CompareError(f"{p} 的 battery_version={doc.get('battery_version')!r}，本脚本是 {BATTERY_VERSION}")
    pred = p.parent / PREDICTIONS
    raw = pred.read_bytes()
    if hashlib.sha256(raw).hexdigest() != doc["predictions"]["sha256"]:
        raise CompareError(f"{pred} 的 sha256 与 {p} 记下的不同（文件被改过，或不是同一次运行）")
    records: dict[str, list[dict[str, Any]]] = {name: [] for name in READINGS}
    for line in raw.decode("utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            records[r["reading"]].append(r)
    return doc, records, p


def _identity(doc: Mapping[str, Any]) -> dict[str, Any]:
    """配对要求两边相同的部分：battery 版本、两个数据文件的 sha256、题目选择。"""
    d = doc["data"]
    return {"battery_version": doc["battery_version"], "mmlu_pro_sha256": d["mmlu_pro"]["sha256"],
            "mmlu_pro_ids_sha256": d["mmlu_pro"]["ids_sha256"], "mmlu_pro_n": d["mmlu_pro"]["sampling"]["n"],
            "ifeval_sha256": d["ifeval"]["sha256"], "ifeval_ids_sha256": d["ifeval"]["ids_sha256"]}


def paired_row(bench: str, metric: str, rb: list[Mapping[str, Any]], ro: list[Mapping[str, Any]]) -> dict[str, Any]:
    """一个口径的配对比较：两边的分数、差值（OTHER − BASE，百分点，按精确分数算后一位小数）、配对 bootstrap 95% CI、
    丢题（BASE 对、OTHER 错）/ 得题数与符号检验（mean_p_gold 是连续读数：只有差值与 CI）。"""
    if [r["id"] for r in rb] != [r["id"] for r in ro]:
        raise CompareError(f"{bench}/{metric}：两边的题目不同")
    if metric == MEAN_P_GOLD:
        vb, vo = [float(r["p_gold"]) for r in rb], [float(r["p_gold"]) for r in ro]
        mb, mo = mean_fraction(vb), mean_fraction(vo)
        ci = paired_bootstrap_ci(np.array(vo, dtype=np.float64), np.array(vb, dtype=np.float64))
        return {"n": len(vb), "base_pct": None if mb is None else round1(mb * 100),
                "other_pct": None if mo is None else round1(mo * 100),
                "delta_pp": None if mb is None else round1((mo - mb) * 100), "ci95_pp": _ci_pct(ci),
                "lost": None, "gained": None, "sign_test": None}
    ub, uo = unit_outcomes(bench, metric, rb), unit_outcomes(bench, metric, ro)
    if set(ub) != set(uo):
        raise CompareError(f"{bench}/{metric}：评分单位不同")
    n = len(ub)
    cb, co = sum(ub.values()), sum(uo.values())
    per_b, per_o = item_counts(bench, metric, rb), item_counts(bench, metric, ro)
    better = sum(1 for (x, _), (y, _) in zip(per_b, per_o) if y > x)
    worse = sum(1 for (x, _), (y, _) in zip(per_b, per_o) if y < x)
    ci = paired_ci_pp(metric, per_b, per_o)
    return {"n": n, "base_pct": pct1(cb, n), "other_pct": pct1(co, n),
            "delta_pp": None if n == 0 else round1(Fraction(100 * (co - cb), n)),
            "ci95_pp": None if ci is None else list(ci),
            "lost": sum(1 for k in ub if ub[k] and not uo[k]), "gained": sum(1 for k in ub if uo[k] and not ub[k]),
            "sign_test": {"n_better": better, "n_worse": worse, "p_value": sign_test_p(better, worse)}}


def _side(doc: Mapping[str, Any], path: Path) -> dict[str, Any]:
    ck = doc.get("checkpoint")
    return {"label": doc.get("label"), "path": str(path), "model": doc.get("model", {}).get("key"),
            "checkpoint": None if ck is None else {"dir": ck.get("dir"), "manifest_sha256": ck.get("manifest_sha256")},
            "limit": doc.get("limit"), "headline": doc.get("headline")}


def compare_runs(base: str | Path, other: str | Path) -> dict[str, Any]:
    """两次运行逐题配对（规则见模块说明）。两边的 battery 版本、数据文件、题目选择与逐题 id 必须相同，否则 CompareError。"""
    db, rb, pb = load_run(base)
    do, ro, po = load_run(other)
    ib, io_ = _identity(db), _identity(do)
    if ib != io_:
        raise CompareError(f"两次运行的题目或数据不同：{ {k: (ib[k], io_[k]) for k in ib if ib[k] != io_[k]} }")
    for name in READINGS:
        if [r["id"] for r in rb[name]] != [r["id"] for r in ro[name]]:
            raise CompareError(f"{name}：两次运行的题目不同（--limit 不同？）")
    rows = []
    for name, bench in READINGS.items():
        for metric in READING_METRICS[name]:
            rows.append({"reading": name, "metric": metric, "headline": metric == HEADLINE_METRIC[name],
                         **paired_row(bench, metric, rb[name], ro[name])})
    cats_b = {r["id"]: r["meta"]["category"] for r in rb[LETTER]}
    if cats_b != {r["id"]: r["meta"]["category"] for r in ro[LETTER]}:
        raise CompareError("MMLU-Pro 两边的学科标注不同")
    by_category = []
    for cat in sorted(set(cats_b.values())):
        sb = [r for r in rb[LETTER] if r["meta"]["category"] == cat]
        so = [r for r in ro[LETTER] if r["meta"]["category"] == cat]
        by_category.append({"category": cat, "metric": "accuracy", **paired_row("mmlu_pro", "accuracy", sb, so)})
    return {
        "script": "scripts/lang_battery.py --compare",
        "battery_version": BATTERY_VERSION,
        "base": _side(db, pb),
        "other": _side(do, po),
        "identity": ib,
        "partial_run": db.get("limit") is not None or do.get("limit") is not None,
        "direction": "other - base, percentage points",
        "statistics": {
            "ci95_pp": "paired bootstrap over questions, other - base, percentile interval (src.health.scoring.paired_ci_pp, "
                       "the health check's --compare function; instruction-level metrics resample questions with "
                       "src.stats.ratio_of_sums; mean_p_gold uses src.stats.paired_bootstrap_ci of the mean)",
            "bootstrap": {"n_resamples": BOOTSTRAP_N_RESAMPLES, "level": BOOTSTRAP_CI_LEVEL, "seed": BOOTSTRAP_SEED},
            "sign_test": "exact two-sided binomial test (p = 1/2) on questions whose count of correct units differs "
                         "(src.health.scoring.sign_test_p)",
            "note": "descriptive only; no red-flag threshold (the red flag is the health check's rule, brief §8.11)",
        },
        "rows": rows,
        "mmlu_pro_by_category": by_category,
    }


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------


def _fmt_ci(ci: Sequence[float] | None, signed: bool = False) -> str:
    if ci is None:
        return "n/a"
    return f"[{ci[0]:+.1f}, {ci[1]:+.1f}]" if signed else f"[{ci[0]:.1f}, {ci[1]:.1f}]"


def readme_text(report: Mapping[str, Any]) -> str:
    """一次运行的 README.md（全部数字取自 lang_battery.json）。"""
    d, m, ck = report["data"], report["model"], report.get("checkpoint")
    lines = [f"# 语言读数（大题量）：{report['label']}", "",
             f"- 脚本：`scripts/lang_battery.py`（DEV-0064），battery v{report['battery_version']}；commit "
             f"`{report['scoring']['code'].get('commit')}`；模式 {report['mode']}；--limit {report['limit']}",
             f"- 模型：`{m['key']}`（{m['repo']}@{m['revision']}，{m['device']}，{m['dtype']}）"]
    if ck is not None:
        lines.append(f"- 检查点：`{ck['dir']}`，manifest sha256 `{ck['manifest_sha256']}`；评测前的合并核对 "
                     f"pass={ck['merge_check'].get('pass')}；之后 load_trained(merge=True)")
    mp, ie = d["mmlu_pro"], d["ifeval"]
    lines += [
        f"- MMLU-Pro：{mp['source']['repo']}@{mp['source']['revision']} {mp['source']['split']}（{mp['source']['license']}），"
        f"文件 sha256 `{mp['sha256']}`（与固定值相同：{mp['matches_pinned']}）。总体 {mp['n_population']} 题去掉健康检查的 "
        f"{mp['health_check']['n_excluded_from_population']} 题，按 `{SALT}` 分层抽 {mp['n_items']} 题；与健康检查重叠 "
        f"{mp['health_check']['n_overlap']} 题。",
        f"- IFEval：{ie['source']['repo']}@{ie['source']['revision']}（{ie['source']['license']}），全部 {ie['n_items']} 条；"
        f"其中 {ie['health_check']['n_overlap']} 条也是健康检查的题（逐题 meta.in_health_check）。",
        f"- 数据只限非商业研究，不再分发；本目录只存题目 id、提示的 sha256 与模型输出，不存题面。",
        "- 口径与健康检查 v3 相同：MMLU-Pro 受限字母读数（option_logits）、IFEval 官方 checker；关思考、greedy、零张图。",
        "", "| 读数 | 口径 | n | 点估计（%） | bootstrap 95% CI |", "|---|---|---:|---:|---|"]
    for name in READINGS:
        for metric, e in report["scores"][name].items():
            lines.append(f"| {name} | {metric} | {e['n']} | {e['pct']} | {_fmt_ci(e['ci95_pct'])} |")
    lines += ["", "MMLU-Pro 分学科（准确率）：", "", "| 学科 | n | % | 95% CI |", "|---|---:|---:|---|"]
    for cat, e in report["mmlu_pro_by_category"].items():
        lines.append(f"| {cat} | {e['n']} | {e['pct']} | {_fmt_ci(e['ci95_pct'])} |")
    t = report["compute"]["total"]
    lines += ["", f"- 计算记账（预热不计）：{t['n_samples']} 题，{t['n_calls']} 次调用，生成 {t['generated_tokens']} token，"
                  f"提示 {t['prompt_tokens']} token，调用耗时 {t['call_time_s']:.1f} s；整次墙钟 {report['wall_time_s']} s",
              f"- IFEval 诊断：思考段未闭合 {report['diagnostics'][IFEVAL]['n_thinking_unclosed']}，空回答 "
              f"{report['diagnostics'][IFEVAL]['n_empty_text']}，写满 {P.MAX_NEW_TOKENS['ifeval']} token "
              f"{report['diagnostics'][IFEVAL]['n_reached_max_new_tokens']}",
              f"- 命令：`python scripts/lang_battery.py {' '.join(report['argv'])}`", ""]
    return "\n".join(lines)


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _write_json(path: Path, obj: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def _print_summary(report: Mapping[str, Any], out_dir: Path) -> None:
    for name in READINGS:
        for metric, e in report["scores"][name].items():
            print(f"{name:16s} {metric:25s} {e['pct']}  CI95={_fmt_ci(e['ci95_pct'])}  (n={e['n']})")
    t = report["compute"]["total"]
    print(f"compute: samples={t['n_samples']} calls={t['n_calls']} generated={t['generated_tokens']} "
          f"prompt={t['prompt_tokens']} call_time={t['call_time_s']:.1f}s  wall={report['wall_time_s']} s  -> {out_dir}")


def _print_compare(res: Mapping[str, Any]) -> None:
    print(f"base={res['base']['label']}  other={res['other']['label']}  ({res['direction']}; partial_run={res['partial_run']})")
    for r in [*res["rows"], *res["mmlu_pro_by_category"]]:
        name = r.get("category") or r["reading"]
        st = r["sign_test"]
        sign = "" if st is None else f"  sign test {st['n_better']}+/{st['n_worse']}- p={st['p_value']:.3g}"
        paired = "" if r["lost"] is None else f"lost {r['lost']}, gained {r['gained']}, "
        delta = "n/a" if r["delta_pp"] is None else f"{r['delta_pp']:+.1f}"
        print(f"{name:16s} {r['metric']:25s} base={r['base_pct']} other={r['other_pct']} delta={delta} pp "
              f"CI95={_fmt_ci(r['ci95_pp'], signed=True)}{sign}  ({paired}n={r['n']})")


# ---------------------------------------------------------------------------
# 命令行
# ---------------------------------------------------------------------------


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", help="注册键，如 qwen3.5-9b（src/adapters/registry.py）；--tiny 时选家族与小模型 config 的来源")
    p.add_argument("--device", help="默认：真模型 cuda，--tiny 时 cpu")
    p.add_argument("--dtype", choices=("bf16", "fp32"), help="默认：真模型 bf16，--tiny 时 fp32")
    p.add_argument("--out", help="运行：输出目录（必填）；--compare：比较结果的 JSON（可选）")
    p.add_argument("--tiny", action="store_true", help="本地构造的小型随机模型，CPU 彩排（数据不核对固定的 sha256）")
    p.add_argument("--raw-dir", help="原始数据目录（仓库外；布局同 scripts/build_health_items.py --raw-dir）")
    p.add_argument("--mmlu-pro-n", type=int,
                   help=f"MMLU-Pro 抽多少题（默认 {DEFAULT_MMLU_PRO_N}；--tiny 时 {TINY_MMLU_PRO_N}）；改了就是另一组题，不能与默认的配对")
    p.add_argument("--limit", type=int, help="每项只跑前 N 题（调试用；结果不拿去比较）")
    p.add_argument("--label", help="这次运行的名字（写进输出与 README；默认取 --model，带 --checkpoint 时加检查点目录）")
    p.add_argument("--checkpoint", help="训练后的检查点目录：合并核对 → load_trained(merge=True)，同 scripts/health_check.py --checkpoint")
    p.add_argument("--merge-check-n", type=int, help="--checkpoint 的合并核对的样本数（同 health_check.py）")
    p.add_argument("--data-root", help="--checkpoint 的合并核对读 E1 grid16 开发子集的落地根（同 health_check.py）")
    p.add_argument("--compare", nargs=2, metavar=("BASE", "OTHER"), help="两次运行（目录或其中的 lang_battery.json）逐题配对，不跑模型")
    p.add_argument("--fetch", action="store_true", help="只下载并核对原始数据到 --raw-dir（scripts/build_health_items.py 的 fetch），不跑模型")
    args = p.parse_args(argv)
    run_only = {"--model": args.model, "--device": args.device, "--dtype": args.dtype, "--tiny": args.tiny,
                "--mmlu-pro-n": args.mmlu_pro_n, "--limit": args.limit, "--label": args.label,
                "--checkpoint": args.checkpoint, "--merge-check-n": args.merge_check_n, "--data-root": args.data_root}
    if args.compare and args.fetch:
        p.error("--compare 与 --fetch 不能同时用")
    if args.compare or args.fetch:
        extra = {"--raw-dir": args.raw_dir} if args.compare else {"--out": args.out}
        used = [k for k, v in {**run_only, **extra}.items() if v not in (None, False)]
        if used:
            p.error(f"{'--compare' if args.compare else '--fetch'} 不跑模型，不能与 {used} 一起用")
        if args.fetch and not args.raw_dir:
            p.error("--fetch 需要 --raw-dir")
        return args
    if not args.out or not args.raw_dir:
        p.error("运行需要 --out 与 --raw-dir")
    for name in ("limit", "mmlu_pro_n", "merge_check_n"):
        v = getattr(args, name)
        if v is not None and v < 1:
            p.error(f"--{name.replace('_', '-')} 必须是正整数")
    if args.checkpoint is not None:
        if not args.model:
            p.error("--checkpoint 要给 --model（必须与检查点 manifest 的构造方式相同）")
        if not Path(args.checkpoint).is_dir():
            p.error(f"--checkpoint {args.checkpoint!r} 不是目录")
    elif args.merge_check_n is not None or args.data_root is not None:
        p.error("--merge-check-n、--data-root 只配 --checkpoint 用")
    if args.tiny:
        args.device = args.device or "cpu"
        args.dtype = args.dtype or "fp32"
        args.mmlu_pro_n = args.mmlu_pro_n or TINY_MMLU_PRO_N
    else:
        if not args.model:
            p.error("真模型运行需要 --model")
        args.device = args.device or "cuda"
        args.dtype = args.dtype or "bf16"
        args.mmlu_pro_n = args.mmlu_pro_n or DEFAULT_MMLU_PRO_N
    return args


def health_args(args, out_dir: Path) -> argparse.Namespace:
    """scripts/health_check.py 的 make_adapter / make_checkpoint_adapter 读的参数（合并核对写在 <out>/lang_battery_merge_check.json）。"""
    return argparse.Namespace(model=args.model, device=args.device, dtype=args.dtype, tiny=args.tiny, checkpoint=args.checkpoint,
                              merge_check_n=args.merge_check_n, data_root=args.data_root, out=str(out_dir / OUT_JSON))


def main_fetch(args) -> int:
    build = _load_script(BUILD_ITEMS_SCRIPT, "_visint_build_health_items_script")
    raw_dir = Path(args.raw_dir)
    for name in BENCHMARKS:
        path = build.fetch(name, raw_dir)  # 下载（已存在且相符就不下）并核对 sha256；不符时那个函数直接退出
        assert Path(path).resolve() == raw_path(raw_dir, name).resolve(), (path, raw_path(raw_dir, name))
        print(f"{name}: {path}  sha256={runlog.sha256_file(path)} (pinned {P.SOURCES[name].file_sha256})")
    return 0


def main_compare(args) -> int:
    try:
        res = compare_runs(*args.compare)
    except (CompareError, OSError, KeyError, ValueError) as e:
        print(f"比较失败：{e}", file=sys.stderr)
        return 2
    _print_compare(res)
    if args.out:
        _write_json(Path(args.out), res)
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.fetch:
        return main_fetch(args)
    if args.compare:
        return main_compare(args)
    out_dir = Path(args.out)
    existing = [n for n in OUTPUT_FILES if (out_dir / n).exists()]
    if existing:
        print(f"{out_dir} 里已有 {existing}，拒绝覆盖（换一个 --out）", file=sys.stderr)
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)
    label = args.label or (args.model or "tiny") + ("" if args.checkpoint is None else f" @ {args.checkpoint}")
    report: dict[str, Any] = {
        "script": "scripts/lang_battery.py",
        "battery_version": BATTERY_VERSION,
        "argv": sys.argv[1:] if argv is None else list(argv),
        "label": label,
        "mode": "tiny" if args.tiny else "real",
        "limit": args.limit,
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "ok": False,
    }
    state: dict[str, Any] = {}
    t_all = time.perf_counter()
    pinned = not args.tiny
    try:
        hc = health_check_script()
        health_obj, health_sha = load_items()
        raw = load_raw(args.raw_dir, pinned=pinned)
        items, metas, report["data"] = build_battery(raw, n_mmlu_pro=args.mmlu_pro_n, health_items=health_obj, pinned=pinned)
        report["data"]["health_check_items_file"] = {"path": P.ITEMS_PATH, "sha256": health_sha}
        # IFEval 的依赖与 punkt_tab 在加载模型之前核对（真模型要求钉住的哈希，同健康检查）
        punkt = hc.check_ifeval_ready(pinned=pinned)
        config = {"script": "scripts/lang_battery.py", "battery_version": BATTERY_VERSION, "label": label, "model": args.model,
                  "tiny": args.tiny, "device": args.device, "dtype": args.dtype, "limit": args.limit,
                  "mmlu_pro_n": args.mmlu_pro_n, "salt": SALT, "raw_dir": str(args.raw_dir),
                  "health_check_protocol_version": P.PROTOCOL_VERSION, "seed": BOOTSTRAP_SEED}
        hargs = health_args(args, out_dir)
        if args.checkpoint is not None:
            ad, entry, vision, report["checkpoint"] = hc.make_checkpoint_adapter(hargs)
            config["checkpoint"] = {k: report["checkpoint"][k] for k in ("manifest_sha256", "build_spec", "lora")}
        else:
            ad, entry, vision = hc.make_adapter(hargs)
        # seed：这里唯一用到随机数的是 bootstrap（生成是 greedy，langdetect 的种子固定）
        meta = runlog.write_run_metadata(out_dir, config=config, seed=BOOTSTRAP_SEED, repo_root=REPO_ROOT,
                                         data_files=[raw["mmlu_pro"]["path"], raw["ifeval"]["path"], REPO_ROOT / P.ITEMS_PATH])
        report["run"] = {"config": config, "meta": meta, "packages_ifeval": runlog.package_versions(hc.IFEVAL_PACKAGES)}
        report["model"] = {
            "key": entry.key, "family": entry.family, "repo": entry.repo, "revision": entry.revision,
            "adapter_model_key": ad.model_key, "device": str(ad.device), "dtype": str(ad.dtype),
            "num_layers": ad.num_layers(), "hidden_size": ad.hidden_size(), "vision_settings_unused": vision,
            "model_generation_config": ad.model.generation_config.to_diff_dict(),
        }
        report["generation"] = {
            "order": P.ORDER, "n_images": 0, "enable_thinking": P.ENABLE_THINKING, "greedy_kwargs": dict(P.GREEDY_KWARGS),
            "health_check_protocol_version": P.PROTOCOL_VERSION,
            "readings": {name: {"benchmark": P.READINGS_BY_NAME[name].benchmark, "kind": P.READINGS_BY_NAME[name].kind,
                                "max_new_tokens": P.READINGS_BY_NAME[name].max_new_tokens} for name in READINGS},
            "per_item": {LETTER: "src.health.run: letter_text_only -> check_spellings -> letter_record -> "
                                 "scoring.score_mmlu_pro_letter (the health check's mmlu_pro_letter branch of run_one)",
                         IFEVAL: "src.health.run.run_one('ifeval', ...) as in the health check"},
        }
        _log(f"模型 {entry.key}（{report['mode']}，{ad.device}，{ad.dtype}）已就绪；MMLU-Pro {len(items['mmlu_pro'])} 题、"
             f"IFEval {len(items['ifeval'])} 条" + (f"，每项只跑前 {args.limit} 题" if args.limit else ""))
        run_battery(ad, items, metas, predictions=out_dir / PREDICTIONS, limit=args.limit, state=state, progress=_log)
        report.update({k: state[k] for k in STATE_KEYS})
        report.update(summarize_battery(state["records"]))
        report["scoring"] = hc.scoring_block(mode="inline", benchmarks=list(BENCHMARKS), punkt=punkt)
        report["predictions"] = {"path": PREDICTIONS, "n_lines": sum(len(v) for v in state["records"].values()),
                                 "sha256": runlog.sha256_file(out_dir / PREDICTIONS)}
        report["ok"] = True
    except Exception as e:  # noqa: BLE001 — 证据先落盘，再以非零码退出
        from src.e3.merge_check import MergeCheckFailed

        report.update({k: state[k] for k in STATE_KEYS if k in state})
        report["n_done"] = {name: len(v) for name, v in state.get("records", {}).items()}
        report.update(error=repr(e), traceback=traceback.format_exc()[-4000:],
                      wall_time_s=round(time.perf_counter() - t_all, 2))
        _write_json(out_dir / OUT_JSON, report)
        if isinstance(e, MergeCheckFailed):
            print(f"拒绝评测：{e}（已写出 {out_dir / OUT_JSON}）", file=sys.stderr)
            return 1
        print(f"失败：{e!r}（已写出 {out_dir / OUT_JSON}）", file=sys.stderr)
        return 2
    report["wall_time_s"] = round(time.perf_counter() - t_all, 2)
    write_compute_json(out_dir / COMPUTE_JSON, state["compute_total"])
    _write_json(out_dir / OUT_JSON, report)
    (out_dir / README).write_text(readme_text(report), encoding="utf-8")
    _print_summary(report, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
