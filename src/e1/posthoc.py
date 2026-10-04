"""E1 事后读数的配置（configs/e1_posthoc.json；D66、E1 预注册 §10；DEV-0033）：读取、结构校验、sha256，以及与冻结的 E1 提示配置的核对。

两项事后读数（实现：src/e1/debiased.py、src/e1/forced.py；入口：scripts/run_e1.py --subset debiased | debiased_dev | forced）：
- debiased_letter（§10.1，N22 → (b)）：去位置偏好的字母读数——k 种循环移位各读一次，合法字母上 softmax、映射回选项内容、取平均；
- format_forced（§10.2）：格式强制诊断——多边形预填 "[["，网格受限解码（固定 271 个 token；主读法与结构强制两种）。

为什么另起一份配置：E1 主评测的配置 configs/e1_prompts.json（sha256 f3353de0…）一个字节都不动——往里加条目会改变它的 sha256，主评测的
config.yaml 就不再逐字节相同。本配置记下它依赖的那份提示配置的 sha256，load_posthoc_config 在运行时核对，不等就拒绝。

约定只在一处定义：字母写法取 e1_prompts.json 的 letter_readout.spellings（这里只核对它与 option_logits 的写法模板一致）；批配置取它的
batch_protocol；网格边长取 src.constants.GRID_SIZE_E1（n_tokens、newline_period 由它推出，validate 核对配置里的数字）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from src.adapters import registry
from src.adapters.base import OPTION_SPELLING_TEMPLATES
from src.constants import GRID_SIZE_E1
from src.e1.prompts import PromptConfig, file_sha256

REPO_ROOT = Path(__file__).resolve().parents[2]
POSTHOC_PATH = REPO_ROOT / "configs" / "e1_posthoc.json"
POSTHOC_SCHEMA_VERSION = 1

DEBIASED_READOUT = "letter_debiased"
DEBIASED_SUBSETS: tuple[str, ...] = ("debiased", "debiased_dev")
FORCED_SUBSET = "forced"
POSTHOC_SUBSETS: tuple[str, ...] = DEBIASED_SUBSETS + (FORCED_SUBSET,)
SPLIT_HOWS: tuple[str, ...] = ("prompt_field", "rendered_template", "option_format")
FORCED_VARIANTS: tuple[str, ...] = ("prefill", "restricted", "structured")
GRID_FIXED_TOKENS = GRID_SIZE_E1 * GRID_SIZE_E1 + (GRID_SIZE_E1 - 1)  # 16 × 16 个数字 + 15 个换行 = 271
GRID_NEWLINE_PERIOD = GRID_SIZE_E1 + 1  # 每行 16 个数字之后是换行：生成位置 t mod 17 = 16 处
GRID_FORMAT_TEXTS: tuple[str, ...] = ("0", "1", "\n")


class PosthocConfigError(ValueError):
    """事后读数的配置不合格：列出全部问题。"""


def _num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def validate(cfg: Mapping[str, Any], prompts: PromptConfig) -> None:
    """结构校验与对 E1 提示配置的核对（任务、写法、批配置、网格长度）。问题一次报全（PosthocConfigError）。"""
    bad: list[str] = []
    if cfg.get("schema_version") != POSTHOC_SCHEMA_VERSION:
        bad.append(f"schema_version 应为 {POSTHOC_SCHEMA_VERSION}，收到 {cfg.get('schema_version')!r}")
    pc = cfg.get("prompt_config", {})
    if pc.get("sha256") != prompts.sha256:
        bad.append(f"prompt_config.sha256 = {pc.get('sha256')!r}，而本次的提示配置 {prompts.path} 是 {prompts.sha256}："
                   "事后读数只在冻结的那份提示配置上做")
    d = cfg.get("debiased_letter", {})
    lr = prompts.data["letter_readout"]
    if d.get("readout") != DEBIASED_READOUT:
        bad.append(f"debiased_letter.readout 必须是 {DEBIASED_READOUT}")
    if list(d.get("tasks", [])) != list(lr["tasks"]):
        bad.append(f"debiased_letter.tasks 必须与 letter_readout.tasks {lr['tasks']} 相同（同一组选择题与点对任务）")
    if d.get("subsets") != {"debiased": "eval", "debiased_dev": "dev"}:
        bad.append("debiased_letter.subsets 必须是 {debiased: eval, debiased_dev: dev}")
    if d.get("orders") != "cyclic_shifts":
        bad.append("debiased_letter.orders 必须是 cyclic_shifts")
    # 字母的两种写法：letter_readout.spellings 与 option_logits 的写法模板必须是同一组（逐个对应）
    if [s.replace("{letter}", "{}") for s in lr["spellings"]] != list(OPTION_SPELLING_TEMPLATES):
        bad.append(f"letter_readout.spellings {lr['spellings']} 与 option_logits 的写法模板 {OPTION_SPELLING_TEMPLATES} 不同")
    sp = d.get("option_split", {})
    if sp.get("line_format") != "({letter}) {text}" or sp.get("line_joiner") != "\n":
        bad.append("option_split 的 line_format 必须是 \"({letter}) {text}\"、line_joiner 必须是换行")
    how = sp.get("how", {})
    if set(how) != set(lr["tasks"]) or any(v not in SPLIT_HOWS for v in how.values()):
        bad.append(f"option_split.how 必须对每个任务给出 {SPLIT_HOWS} 之一")
    if how.get("spatial_embspatial") != "option_format" or how.get("rel_depth_nyu") != "rendered_template":
        bad.append("EmbSpatial 用 option_format 重拼，NYU 点对拆渲染好的模板")
    if not str(sp.get("rule", "")).strip():
        bad.append("option_split.rule 不能为空")
    if d.get("batch_layout", {}).get("layout") != "order_major":
        bad.append("debiased_letter.batch_layout.layout 必须是 order_major")
    if d.get("original_letter_control") != "same_run_order0":
        bad.append("debiased_letter.original_letter_control 必须是 same_run_order0")
    vc = d.get("validity_check", {})
    if (vc.get("free_subset") != "free" or not _num(vc.get("near_random_margin")) or not _num(vc.get("free_valid_rate_min"))
            or not _num(vc.get("free_minus_debiased_min"))):
        bad.append("validity_check 要给 free_subset = free 与三个阈值")
    f = cfg.get("format_forced", {})
    if f.get("subset") != FORCED_SUBSET or f.get("sample_subset") != "think" or f.get("enable_thinking") is not False:
        bad.append("format_forced：subset = forced、sample_subset = think、enable_thinking = false")
    if not set(f.get("families", [])) <= set(registry.FAMILIES) or not f.get("families"):
        bad.append("format_forced.families 必须是登记过的家族")
    for m in f.get("models", []):
        if m not in registry.MODELS or registry.get_model(m).family not in f.get("families", []):
            bad.append(f"format_forced.models 里的 {m!r} 没有登记或不属于 families")
    vs = f.get("variants", {})
    if set(vs) != set(FORCED_VARIANTS):
        bad.append(f"format_forced.variants 必须恰好是 {FORCED_VARIANTS}")
    else:
        if vs["prefill"].get("task") != "refexp_poly" or vs["prefill"].get("answer_prefix") != "[[" \
                or vs["prefill"].get("max_new_tokens") != "entry" or vs["prefill"].get("fixed_length") is not False:
            bad.append("prefill：task = refexp_poly、answer_prefix = \"[[\"、max_new_tokens = entry、fixed_length = false")
        for name in ("restricted", "structured"):
            v = vs[name]
            if (v.get("task") != "grid16" or tuple(v.get("allowed", ())) != GRID_FORMAT_TEXTS
                    or v.get("stop_tokens_allowed") is not False or v.get("n_tokens") != GRID_FIXED_TOKENS
                    or v.get("fixed_length") is not True):
                bad.append(f"{name}：task = grid16、allowed = {list(GRID_FORMAT_TEXTS)}、stop_tokens_allowed = false、"
                           f"n_tokens = {GRID_FIXED_TOKENS}（16 × 16 + 15）、fixed_length = true")
        if vs["structured"].get("newline_period") != GRID_NEWLINE_PERIOD:
            bad.append(f"structured.newline_period 必须是 {GRID_NEWLINE_PERIOD}")
        for name, v in vs.items():
            if not str(v.get("rule", "")).strip():
                bad.append(f"variants.{name}.rule 不能为空")
    if bad:
        raise PosthocConfigError("事后读数的配置不合格：\n- " + "\n- ".join(bad))


@dataclass(frozen=True)
class PosthocConfig:
    data: Mapping[str, Any]
    path: str
    sha256: str

    @property
    def debiased(self) -> Mapping[str, Any]:
        return self.data["debiased_letter"]

    @property
    def forced(self) -> Mapping[str, Any]:
        return self.data["format_forced"]

    def base_subset(self, subset: str) -> str:
        """--subset（debiased、debiased_dev、forced）→ 取样本用的 E1 子集（eval、dev、think）。"""
        if subset in DEBIASED_SUBSETS:
            return self.debiased["subsets"][subset]
        if subset == FORCED_SUBSET:
            return self.forced["sample_subset"]
        raise ValueError(f"{subset!r} 不是事后读数的子集 {POSTHOC_SUBSETS}")

    def tasks_for(self, subset: str) -> tuple[str, ...]:
        """这个事后子集支持的任务（debiased：四个选择题与点对任务；forced：refexp_poly、grid16）。"""
        if subset in DEBIASED_SUBSETS:
            return tuple(self.debiased["tasks"])
        if subset == FORCED_SUBSET:
            return tuple(dict.fromkeys(v["task"] for v in self.forced["variants"].values()))
        raise ValueError(f"{subset!r} 不是事后读数的子集 {POSTHOC_SUBSETS}")

    def forced_variants(self, task: str) -> tuple[str, ...]:
        """一个任务的格式强制读法（按 FORCED_VARIANTS 的顺序）。"""
        return tuple(k for k in FORCED_VARIANTS if self.forced["variants"][k]["task"] == task)

    def split_how(self, task: str) -> str:
        return self.debiased["option_split"]["how"][task]

    @property
    def line_format(self) -> str:
        return self.debiased["option_split"]["line_format"]

    @property
    def line_joiner(self) -> str:
        return self.debiased["option_split"]["line_joiner"]

    def validity_thresholds(self) -> dict[str, float]:
        vc = self.debiased["validity_check"]
        return {k: float(vc[k]) for k in ("near_random_margin", "free_valid_rate_min", "free_minus_debiased_min")}


def load_posthoc_config(prompts: PromptConfig, path: str | Path = POSTHOC_PATH) -> PosthocConfig:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    validate(data, prompts)
    return PosthocConfig(data=data, path=str(p), sha256=file_sha256(p))


__all__ = [
    "DEBIASED_READOUT",
    "DEBIASED_SUBSETS",
    "FORCED_SUBSET",
    "FORCED_VARIANTS",
    "GRID_FIXED_TOKENS",
    "GRID_FORMAT_TEXTS",
    "GRID_NEWLINE_PERIOD",
    "POSTHOC_PATH",
    "POSTHOC_SUBSETS",
    "PosthocConfig",
    "PosthocConfigError",
    "load_posthoc_config",
    "validate",
]
