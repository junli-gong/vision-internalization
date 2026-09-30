"""健康检查的协议常量与提示模板（简报 §8.11；DEV-0008）。约定只在这里定义一处，其余模块与脚本 import。

协议全文见 src/health/__init__.py 的模块说明；这里的数字由 tests/test_health_items.py 与 tests/test_health_scoring.py 钉住。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

PROTOCOL_VERSION = 1

# ---------------------------------------------------------------------------
# 题目来源（HF 数据集，固定 revision；原始文件的 sha256 在开发机上实测）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Source:
    """一个题目来源：HF 数据集仓库、固定的 commit、配置与划分、原始文件、许可（HF API 的 card license）。"""

    repo: str
    revision: str
    config: str
    split: str
    file: str
    file_sha256: str
    license: str
    n_rows: int


BENCHMARKS: tuple[str, ...] = ("mmlu_pro", "gsm8k", "ifeval")

SOURCES: Mapping[str, Source] = {
    "mmlu_pro": Source(
        repo="TIGER-Lab/MMLU-Pro",
        revision="b189ec765aa7ed75c8acfea42df31fdae71f97be",
        config="default",
        split="test",
        file="data/test-00000-of-00001.parquet",
        file_sha256="0e24a191921c2f453518a537a8b2117bd137e7714d4ef1565e9ba06c1ecb9ad8",
        license="MIT",
        n_rows=12032,
    ),
    "gsm8k": Source(
        repo="openai/gsm8k",
        revision="740312add88f781978c0658806c59bc2815b9866",
        config="main",
        split="test",
        file="main/test-00000-of-00001.parquet",
        file_sha256="ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59",
        license="MIT",
        n_rows=1319,
    ),
    "ifeval": Source(
        repo="google/IFEval",
        revision="966cd89545d6b6acfd7638bc708b98261ca58e84",
        config="default",
        split="train",
        file="ifeval_input_data.jsonl",
        file_sha256="6a85310ca8ce15eff755aa08a3a4ff931c7e273e7515ebb3c492ea85fd8288f2",
        license="Apache-2.0",
        n_rows=541,
    ),
}

# ---------------------------------------------------------------------------
# 抽样（在看到任何分数之前固定）
# ---------------------------------------------------------------------------

SEED = 20260930
MMLU_PRO_N_CATEGORIES = 14
MMLU_PRO_PER_CATEGORY = 7
MMLU_PRO_FILL = 2
COUNTS: Mapping[str, int] = {
    "mmlu_pro": MMLU_PRO_N_CATEGORIES * MMLU_PRO_PER_CATEGORY + MMLU_PRO_FILL,
    "gsm8k": 50,
    "ifeval": 50,
}
# 抽样阶段名：进入排序键，改名就改变抽样结果
STAGE_MMLU_PRO_STRATIFIED = "stratified"
STAGE_MMLU_PRO_FILL = "fill"
STAGE_UNIFORM = "uniform"

ITEMS_PATH = "configs/health_check_items.json"  # 相对仓库根
ITEMS_SCHEMA_VERSION = 1
# 入库的题目文件的 sha256：运行脚本加载时核对，防止题目被静默改动（重建题目文件后必须同步更新，见 build_health_items.py）
ITEMS_SHA256 = "0239675f9c226f17963419e2e12a282dd4ced8ed42452b53a939145372a525df"

# ---------------------------------------------------------------------------
# 输入与生成
# ---------------------------------------------------------------------------

ORDER = "instruction_first"  # build_inputs([], prompt, ORDER)：纯文本，零张图
ENABLE_THINKING = False
# greedy：每次 generate 都显式传入。transformers 5.x 里未显式给出的生成参数取仓库的 generation_config
# （Gemma 默认采样），所以除 do_sample 外，把会改变 greedy 结果的常见参数也显式钉成中性值。
GREEDY_KWARGS: Mapping[str, Any] = {
    "do_sample": False,
    "num_beams": 1,
    "repetition_penalty": 1.0,
    "no_repeat_ngram_size": 0,
}
MAX_NEW_TOKENS: Mapping[str, int] = {"mmlu_pro": 32, "gsm8k": 512, "ifeval": 1024}
WARMUP_PROMPT = "Reply with the word OK."  # 计时前的预热调用（不计入记账，D21）
WARMUP_MAX_NEW_TOKENS = 2

# ---------------------------------------------------------------------------
# 提示模板
# ---------------------------------------------------------------------------

OPTION_LETTERS = "ABCDEFGHIJ"  # MMLU-Pro 最多 10 个选项
MMLU_PRO_INSTRUCTION = "Answer with the letter of the correct option only."
GSM8K_INSTRUCTION = "Solve the problem step by step. On the last line, write the final answer as '#### <number>'."


def option_letters(n_options: int) -> str:
    """本题的选项字母：前 n 个（n ≤ 10）。"""
    if isinstance(n_options, bool) or not isinstance(n_options, int) or not 1 <= n_options <= len(OPTION_LETTERS):
        raise ValueError(f"选项个数必须在 1..{len(OPTION_LETTERS)}，收到 {n_options!r}")
    return OPTION_LETTERS[:n_options]


def mmlu_pro_prompt(item: Mapping[str, Any]) -> str:
    """题干一行（去掉首尾空白），每个选项一行 "A. …"（选项去掉首尾空白），最后一行固定指令，行间单个换行。"""
    options = item["options"]
    lines = [item["question"].strip()]
    lines += [f"{letter}. {opt.strip()}" for letter, opt in zip(option_letters(len(options)), options)]
    lines.append(MMLU_PRO_INSTRUCTION)
    return "\n".join(lines)


def gsm8k_prompt(item: Mapping[str, Any]) -> str:
    """题干（去掉首尾空白），换行，固定指令。"""
    return item["question"].strip() + "\n" + GSM8K_INSTRUCTION


def ifeval_prompt(item: Mapping[str, Any]) -> str:
    """IFEval 的提示原样使用，只去掉首尾空白。

    入库的 50 题里只有 key 1999 受影响（末尾两个换行）：Qwen3.5 与 Gemma 4 的聊天模板本来就 trim 文本，适配器为精确定位
    指令区间，拒绝带首尾空白的指令（build_inputs 报 ValueError）；InternVL3.5 的模板不 trim。统一去掉后三个家族看到的
    提示相同。评分仍用题目文件里的原始 prompt（官方 checker 的 InputExample.prompt）。
    """
    return item["prompt"].strip()


PROMPT_BUILDERS = {"mmlu_pro": mmlu_pro_prompt, "gsm8k": gsm8k_prompt, "ifeval": ifeval_prompt}

# ---------------------------------------------------------------------------
# 评分与比较
# ---------------------------------------------------------------------------

# 每项的标题口径（门报告里比较的就是这三个数）
HEADLINE: Mapping[str, str] = {
    "mmlu_pro": "accuracy",
    "gsm8k": "flexible",
    "ifeval": "prompt_level_strict",
}
RED_FLAG_DROP_PP = 5  # 合并后比基座低超过 5 个百分点（严格大于）即标红
LANGDETECT_SEED = 0  # IFEval 的语言判定（langdetect）默认每次随机；固定种子使评分可复现

# IFEval 的句子切分用 nltk 的 punkt_tab（english）。nltk_data 不分版本，这里钉住开发机 2026-09-30 下载的文件哈希；
# 真模型运行时要求逐个相同（--tiny 彩排只记录）。
PUNKT_TAB_ENGLISH_SHA256: Mapping[str, str] = {
    "abbrev_types.txt": "92a3e070f43d9b4c5534758ca40ad7343b04e7e29bfe0c2eb658a39445a4f779",
    "collocations.tab": "8e2da1225e4dd2cc9dba261ee231ccb134859e21b46006e7f472c5ee269af0cf",
    "ortho_context.tab": "4bbcca25ed3d3f06c02402abf8419b9f033b8adc06e7b482eca4e45f81a5dc4c",
    "sent_starters.txt": "f3f8535483e1dba487241b764945168123bca3209a9645e59acd1225dc76edac",
}
