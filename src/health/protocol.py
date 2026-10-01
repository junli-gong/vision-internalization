"""健康检查的协议常量与提示模板（简报 §8.11；DEV-0008；协议 v2：DEV-0013，D28-7；协议 v3：DEV-0016，D36-4）。
约定只在这里定义一处，其余模块与脚本 import。

协议全文见 src/health/__init__.py 的模块说明；这里的数字由 tests/test_health_items.py 与 tests/test_health_scoring.py 钉住。

版本：
- v1（D26）：MMLU-Pro 最多生成 32 个 token，那一个读数就是标题。RUN-0007、RUN-0009 的输出是 v1。
- v2（D28-7，取代 D26 的 32 token）：MMLU-Pro 最多生成 256 个 token，标题读数（v2 输出里的键 "mmlu_pro"）取全部生成；
  诊断读数 mmlu_pro_fmt32 取同一次生成的前 32 个 token（格式诊断，不标红）。RUN-0011 的输出是 v2。
- v3（D36-4，取代 D28-7 的标题读数；**冻结，G3 之前不再动**）：MMLU-Pro 的标题读数改为受限字母读数 mmlu_pro_letter：
  适配器的 option_logits 只看第一个生成位置，限制在本题的选项字母上，每个字母取 "X" 与 " X" 两种写法里较大的 logit；
  准确率是标题（标红只看它），另记正确选项在选项集合上的平均概率（诊断，给差值与 CI）。v2 的 256 token 自由作答
  （v3 的名字 mmlu_pro_free256）与同一次生成的 32 token 前缀（mmlu_pro_fmt32）保留为诊断，不标红。题目文件与其余协议不变。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

PROTOCOL_VERSION = 3  # D36-4：冻结为健康检查 v3，G3 之前不再动
# 能读、能 --rescore、能 --compare 的输出版本（v1、v2 的输出按各自的布局读，见 RECORD_KEYS_BY_VERSION）
SUPPORTED_PROTOCOL_VERSIONS: tuple[int, ...] = (1, 2, 3)

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
# 题目文件里记的协议版本：题目建于 v1，v2 的题目不变（D28-7），所以题目文件仍是 1，与 PROTOCOL_VERSION 分开
ITEMS_PROTOCOL_VERSION = 1
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
# 每项一次自由生成的上限（v2 起）。MMLU-Pro 256（D28-7；v1 是 32，见 RECORD_KEYS_BY_VERSION；v3 里它是诊断读数 mmlu_pro_free256）
MAX_NEW_TOKENS: Mapping[str, int] = {"mmlu_pro": 256, "gsm8k": 512, "ifeval": 1024}
# 受限字母读数（v3，D36-4）：只看第一个生成位置，相当于只生成 1 个 token
LETTER_MAX_NEW_TOKENS = 1
# MMLU-Pro 的格式诊断：同一次生成的前 32 个 token（D28-7）
FMT32_MAX_NEW_TOKENS = 32
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
# 截断率提示（D28-7）：某读数写满 max_new_tokens 的题所占比例，基座与合并后相差超过 10 个百分点（严格大于）即列出，与标红分开
TRUNCATION_NOTICE_PP = 10

# ---------------------------------------------------------------------------
# 读数（D28-7）：逐题记录、分数、诊断与比较的单位
# ---------------------------------------------------------------------------

ROLE_HEADLINE = "headline"  # 标题读数：标题口径参与标红
ROLE_DIAGNOSTIC = "diagnostic"  # 诊断读数：给出差值、CI 与符号检验，不标红

KIND_GENERATE = "generate"  # 自由生成（greedy，最多 max_new_tokens 个 token）；prefix_of 不为 None 时是它的前缀读数
KIND_OPTION_LETTER = "option_letter"  # 受限字母读数（v3，D36-4）：adapter.option_logits，选项为本题的实际字母


@dataclass(frozen=True)
class Reading:
    """一个读数：题目、提示与评分规则取自 benchmark，生成上限 max_new_tokens。

    kind = KIND_GENERATE：自由生成。prefix_of 不为 None 时是前缀读数：不另外生成，取 prefix_of 那个读数的同一次 greedy 生成的
    前 max_new_tokens 个 token，按"只生成 max_new_tokens 个"重建生成部分（run.prefix_generation）。greedy 下两者的 token 逐个相同。
    kind = KIND_OPTION_LETTER：受限字母读数，一次 option_logits（max_new_tokens 记 1），不解析文本。
    """

    name: str
    benchmark: str
    max_new_tokens: int
    role: str
    prefix_of: str | None = None
    kind: str = KIND_GENERATE

    def __post_init__(self) -> None:
        if self.kind not in (KIND_GENERATE, KIND_OPTION_LETTER):
            raise ValueError(f"未知的读数类型 {self.kind!r}")
        if self.kind == KIND_OPTION_LETTER and (self.prefix_of is not None or self.max_new_tokens != LETTER_MAX_NEW_TOKENS
                                                or self.benchmark != "mmlu_pro"):
            raise ValueError(f"受限字母读数 {self.name} 只用于 MMLU-Pro，max_new_tokens 为 {LETTER_MAX_NEW_TOKENS}，不能是前缀读数")


MMLU_PRO_LETTER = "mmlu_pro_letter"
MMLU_PRO_FREE256 = "mmlu_pro_free256"
MMLU_PRO_FMT32 = "mmlu_pro_fmt32"

# v3 的读数，按输出里的顺序。比较的行名就是这里的 name；口径（读数名所指的项、类型与 max_new_tokens）相同才可比
READINGS: tuple[Reading, ...] = (
    Reading(MMLU_PRO_LETTER, "mmlu_pro", LETTER_MAX_NEW_TOKENS, ROLE_HEADLINE, kind=KIND_OPTION_LETTER),
    Reading(MMLU_PRO_FREE256, "mmlu_pro", MAX_NEW_TOKENS["mmlu_pro"], ROLE_DIAGNOSTIC),
    Reading(MMLU_PRO_FMT32, "mmlu_pro", FMT32_MAX_NEW_TOKENS, ROLE_DIAGNOSTIC, prefix_of=MMLU_PRO_FREE256),
    Reading("gsm8k", "gsm8k", MAX_NEW_TOKENS["gsm8k"], ROLE_HEADLINE),
    Reading("ifeval", "ifeval", MAX_NEW_TOKENS["ifeval"], ROLE_HEADLINE),
)
READINGS_BY_NAME: Mapping[str, Reading] = {r.name: r for r in READINGS}

# 各版本输出里 records / scores / diagnostics 的键 → 它对应的 v3 读数。
# v1 的 "mmlu_pro" 是 32 个 token 的那一个读数，口径与 mmlu_pro_fmt32 相同；v1 没有 256 的读数。
# v2 的 "mmlu_pro" 是 256 token 的自由作答，口径与 v3 的 mmlu_pro_free256 相同。v1、v2 都没有受限字母读数。
RECORD_KEYS_BY_VERSION: Mapping[int, Mapping[str, str]] = {
    1: {"mmlu_pro": MMLU_PRO_FMT32, "gsm8k": "gsm8k", "ifeval": "ifeval"},
    2: {"mmlu_pro": MMLU_PRO_FREE256, MMLU_PRO_FMT32: MMLU_PRO_FMT32, "gsm8k": "gsm8k", "ifeval": "ifeval"},
    3: {r.name: r.name for r in READINGS},
}
# 各版本自己的汇总里，headline 每项取哪个记录键（v1、v2 照当时的规则：键名就是项名；重评 v1、v2 的输出时汇总逐字节不变）
HEADLINE_KEYS_BY_VERSION: Mapping[int, Mapping[str, str]] = {
    1: {"mmlu_pro": "mmlu_pro", "gsm8k": "gsm8k", "ifeval": "ifeval"},
    2: {"mmlu_pro": "mmlu_pro", "gsm8k": "gsm8k", "ifeval": "ifeval"},
    3: {"mmlu_pro": MMLU_PRO_LETTER, "gsm8k": "gsm8k", "ifeval": "ifeval"},
}


def check_protocol_version(version: Any) -> int:
    """输出文件的协议版本必须是 SUPPORTED_PROTOCOL_VERSIONS 之一，否则 ValueError。"""
    if isinstance(version, bool) or version not in SUPPORTED_PROTOCOL_VERSIONS:
        raise ValueError(f"协议版本 {version!r} 不受支持（支持 {list(SUPPORTED_PROTOCOL_VERSIONS)}）")
    return version


def record_readings(version: int) -> dict[str, Reading]:
    """某版本输出的 records 键 → 读数（按输出里的顺序）。"""
    return {key: READINGS_BY_NAME[name] for key, name in RECORD_KEYS_BY_VERSION[check_protocol_version(version)].items()}


def readings_of(benchmark: str) -> tuple[Reading, ...]:
    """v3 里 benchmark 的全部读数，按 READINGS 的顺序（前缀读数排在它截取的读数之后）。"""
    out = tuple(r for r in READINGS if r.benchmark == benchmark)
    if not out:
        raise KeyError(f"没有 {benchmark} 的读数")
    return out


def _check_readings() -> None:
    names = [r.name for r in READINGS]
    if len(set(names)) != len(names):
        raise RuntimeError(f"读数名重复：{names}")
    for k, r in enumerate(READINGS):
        if r.prefix_of is not None:
            src = [s for s in READINGS[:k] if s.name == r.prefix_of]
            if not (src and src[0].benchmark == r.benchmark and src[0].kind == KIND_GENERATE and src[0].prefix_of is None
                    and r.max_new_tokens < src[0].max_new_tokens):
                raise RuntimeError(f"前缀读数 {r.name} 必须排在它截取的同一项自由生成读数之后，且更短")
    for bench in BENCHMARKS:
        heads = [r for r in READINGS if r.benchmark == bench and r.role == ROLE_HEADLINE]
        if [r.name for r in heads] != [HEADLINE_KEYS_BY_VERSION[PROTOCOL_VERSION][bench]]:
            raise RuntimeError(f"{bench} 应恰有一个标题读数 {HEADLINE_KEYS_BY_VERSION[PROTOCOL_VERSION][bench]}")
        if sum(r.kind == KIND_GENERATE and r.prefix_of is None for r in READINGS if r.benchmark == bench) != 1:
            raise RuntimeError(f"{bench} 应恰有一次自由生成")
    for v in SUPPORTED_PROTOCOL_VERSIONS:
        if set(HEADLINE_KEYS_BY_VERSION[v].values()) - set(RECORD_KEYS_BY_VERSION[v]):
            raise RuntimeError(f"v{v} 的标题键不在它的记录键里")


_check_readings()

LANGDETECT_SEED = 0  # IFEval 的语言判定（langdetect）默认每次随机；固定种子使评分可复现

# IFEval 的句子切分用 nltk 的 punkt_tab（english）。nltk_data 不分版本，这里钉住开发机 2026-09-30 下载的文件哈希；
# 真模型运行时要求逐个相同（--tiny 彩排只记录）。
PUNKT_TAB_ENGLISH_SHA256: Mapping[str, str] = {
    "abbrev_types.txt": "92a3e070f43d9b4c5534758ca40ad7343b04e7e29bfe0c2eb658a39445a4f779",
    "collocations.tab": "8e2da1225e4dd2cc9dba261ee231ccb134859e21b46006e7f472c5ee269af0cf",
    "ortho_context.tab": "4bbcca25ed3d3f06c02402abf8419b9f033b8adc06e7b482eca4e45f81a5dc4c",
    "sent_starters.txt": "f3f8535483e1dba487241b764945168123bca3209a9645e59acd1225dc76edac",
}
