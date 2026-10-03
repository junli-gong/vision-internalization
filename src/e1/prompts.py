"""E1 提示配置（configs/e1_prompts.json）：读取、结构校验、sha256、逐条渲染（DEV-0014 任务书 §1）。

配置的结构：任务 × 家族 → 提示模板原文、是否原生、出处链接与引用原文、max_new_tokens、解析函数名。约定只在配置里写一处，
这里只读它；模板之外的规则（CountBench 问句的对象怎么取、多选题选项怎么排）写在本模块，配置的 count_question_rule 是它的
说明文字，tests 钉住两者一致。渲染后去掉首尾空白（Qwen 的聊天模板会 trim 整段内容）。

CountBench 的问句有两个来源（DEV-0015 §4；run_e1.py --count-questions）：countbenchqa（默认，D36-2：big_vision 的人工问句，
配置的 countbenchqa_questions）与 heuristic（可选，count_question_rule 从 caption 取对象）。用哪个由样本决定：
src/e1/tasks.select 在 countbenchqa 模式下把问句放进 fields[COUNTBENCHQA_FIELD]，render 见到它就用整句问句
（规则见 countbenchqa_instruction）。

协议 v2（DEV-0017，D36）在配置里另有五节，结构由 validate 核对，规则文字写在配置里：
- letter_readout：选择题与点对任务在 eval、dev 上的受限字母读数（readout、max_new_tokens、letter_spellings；实现见
  src/e1/letters.py）；自由作答的诊断用 free 子集（think:<任务> 的 id，关思考）。
- report_columns、refusal_by_person：三列（严格读数、有效率、条件读数）与按人物 / 非人物分列的拒答诊断（src/e1/scoring）。
- batch_protocol：冻结的批配置（批大小、左填充、样本顺序、按字母集合分批）；acceptance：批处理验收 v3（src/e1/acceptance）。

DEV-0027（D59）：
- 思考子集（think）改用预算强制收尾：generation.think_budget 给出 thinking_budget（4096）与 answer_budget（1024），生成走
  adapter.generate_batch_budgeted（D59-1）；删掉 think8192 子集与 think_max_new_tokens（不再做 8192 的敏感性检查）。
- 验收 v3 的 δ 按组设置（每组的 delta，取值见配置的 acceptance.delta_rule；D59-3）。
- eval、dev、free、extra 的提示渲染、生成参数与解析不变（与 sha256 5de1aec7… 的配置逐字节相同，tests/test_e1_d59.py 核对）。
"""

from __future__ import annotations

import hashlib
import json
import re
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from src.adapters import registry
from src.data.records import Sample, letter
from src.parsing.answers import NUMBER_WORDS

REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPTS_PATH = REPO_ROOT / "configs" / "e1_prompts.json"
PROMPTS_SCHEMA_VERSION = 2  # DEV-0017：加了 letter_readout、report_columns、refusal_by_person、batch_protocol、acceptance

# 9 个任务（键名与 src/data/eval_subsets.THINK_TASKS 相同）
TASKS: tuple[str, ...] = (
    "refexp_box",
    "refexp_poly",
    "grid16",
    "count",
    "ovd",
    "rel_depth_nyu",
    "rel_depth_blink",
    "spatial_cvbench",
    "spatial_embspatial",
)
PARSERS: tuple[str, ...] = ("one_box", "boxes", "polygon", "grid16", "count", "choice", "letter_token")
ENTRY_PARSERS: tuple[str, ...] = PARSERS[:-1]  # 任务条目里的 parse（自由作答）；letter_token 只由 letter_readout 选用
PROVENANCES: tuple[str, ...] = (
    "official_doc",  # 本家族官方文档的原文
    "predecessor_cookbook",  # 前代模型的官方 cookbook（Qwen3-VL → Qwen3.5）
    "adapted_official_doc",  # 官方文档的句子改写（非原生）
    "benchmark",  # benchmark 自带的提示
    "common_practice",  # 常用评测工具的写法（非原生）
    "unified",  # 统一提示（非原生）
    "unified_common_practice",  # 统一提示，模板取自常见评测写法（非原生）
    "unified_benchmark_wording",  # 统一提示，措辞取自别的 benchmark（非原生）
)
NATIVE_PROVENANCES = frozenset({"official_doc", "predecessor_cookbook", "benchmark"})
SUBSET_KINDS: tuple[str, ...] = ("eval", "dev", "think", "extra", "free")  # D59-1：删掉 think8192
THINK_SUBSETS: tuple[str, ...] = ("think",)
THINK_METHOD = "generate_batch_budgeted"  # 思考子集的生成方式（D59-1：预算强制收尾）
FREE_SUBSET = "free"  # 选择题与点对任务的自由作答诊断（D36-6）：think:<任务> 的 id，关思考，上限取任务条目的 max_new_tokens
READOUTS: tuple[str, ...] = ("text", "letter")  # text = 自由生成后解析；letter = 受限字母读数（letter_readout）
PADDING_SIDES: tuple[str, ...] = ("left",)  # src/adapters/batching 只实现了左填充
BATCH_ORDERS: tuple[str, ...] = ("selection",)  # 样本顺序 = src/e1/tasks.select 的固定顺序
LETTER_BATCHES: tuple[str, ...] = ("same_letters",)  # 受限字母读数：字母集合一变就另起一批
ACCEPTANCE_VERSION = 3
ACCEPTANCE_METRICS: tuple[str, ...] = ("accuracy_pooled", "cIoU", "gIoU", "F1@0.5")
ACCEPTANCE_METRIC_CLASSES: tuple[str, ...] = ("accuracy", "iou")
REQUIRED_FAMILY_FIELDS: tuple[str, ...] = (
    "template",
    "native",
    "provenance",
    "source_url",
    "source_quote",
    "note",
    "max_new_tokens",
    "parse",
)
ORDERS_ALLOWED = ("image_first", "instruction_first")


class PromptConfigError(ValueError):
    """提示配置不合格：列出全部问题。"""


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _placeholders(template: str) -> set[str]:
    return {name for _, name, _, _ in string.Formatter().parse(template) if name is not None}


def validate(cfg: Mapping[str, Any]) -> None:
    """结构校验：任务、家族、必填字段、占位符、解析函数名、生成参数。问题一次报全（PromptConfigError）。"""
    bad: list[str] = []
    if cfg.get("schema_version") != PROMPTS_SCHEMA_VERSION:
        bad.append(f"schema_version 应为 {PROMPTS_SCHEMA_VERSION}，收到 {cfg.get('schema_version')!r}")
    order = cfg.get("image_order", {})
    if order.get("value") not in ORDERS_ALLOWED:
        bad.append(f"image_order.value 应为 {ORDERS_ALLOWED} 之一")
    if set(order.get("sources", {})) != set(registry.FAMILIES):
        bad.append("image_order.sources 必须逐家族给出出处")
    gen = cfg.get("generation", {})
    dec = gen.get("decoding", {})
    if dec.get("do_sample") is not False or dec.get("num_beams") != 1:
        bad.append("generation.decoding 必须 do_sample=false、num_beams=1（greedy）")
    et = gen.get("enable_thinking", {})
    if set(et) != set(SUBSET_KINDS) or any(not isinstance(v, bool) for v in et.values()):
        bad.append(f"generation.enable_thinking 必须对 {SUBSET_KINDS} 各给一个 bool")
    elif any(et[s] for s in SUBSET_KINDS if s not in THINK_SUBSETS) or not all(et[s] for s in THINK_SUBSETS):
        bad.append("只有思考子集开思考（简报 §6.3）")
    if "think_max_new_tokens" in gen:
        bad.append("generation.think_max_new_tokens 已由 D59-1 取消（思考子集用 think_budget 的预算强制收尾）")
    tb = gen.get("think_budget", {})
    if (not _posint(tb.get("thinking_budget")) or not _posint(tb.get("answer_budget")) or tb.get("method") != THINK_METHOD
            or not str(tb.get("basis", "")).strip()):
        bad.append(f"generation.think_budget 必须给正整数的 thinking_budget、answer_budget，method = {THINK_METHOD}，以及 basis")
    if not isinstance(gen.get("free_note"), str):
        bad.append("generation.free_note 必须说明 free 子集")
    bad += _validate_protocol(cfg)
    if not isinstance(cfg.get("max_new_tokens_adjustments"), list):
        bad.append("max_new_tokens_adjustments 必须是列表（调整记录）")
    if set(cfg.get("parse_functions", {})) != set(PARSERS):
        bad.append(f"parse_functions 必须恰好说明 {PARSERS}")
    if not str(cfg.get("countbenchqa_questions", {}).get("provenance_note", "")).strip():
        bad.append("countbenchqa_questions.provenance_note 必须注明问句的出处（D36-2）")
    if "rule" not in cfg.get("count_question_rule", {}) or cfg["count_question_rule"].get("version") != COUNT_RULE_VERSION:
        bad.append(f"count_question_rule 必须有 rule，version = {COUNT_RULE_VERSION}（src/e1/prompts.py）")
    qpart = cfg.get("countbenchqa_questions", {}).get("question_part")
    if not isinstance(qpart, str) or qpart.count("{object}") != 1 or _placeholders(qpart) != {"object"}:
        bad.append("countbenchqa_questions.question_part 必须是只含一个 {object} 的问句")
    else:
        for fam, e in cfg.get("tasks", {}).get("count", {}).get("families", {}).items():
            if not str(e.get("template", "")).startswith(qpart):
                bad.append(f"count/{fam}: 模板必须以 countbenchqa_questions.question_part 开头（人工问句替换的就是这一段）")
    tasks = cfg.get("tasks", {})
    if set(tasks) != set(TASKS):
        bad.append(f"tasks 必须恰好是 {sorted(TASKS)}，收到 {sorted(tasks)}")
    for t, spec in tasks.items():
        allowed = set(spec.get("placeholders", {}))
        fams = spec.get("families", {})
        if set(fams) != set(registry.FAMILIES):
            bad.append(f"{t}: families 必须恰好是 {registry.FAMILIES}")
        for fam, e in fams.items():
            where = f"{t}/{fam}"
            missing = [k for k in REQUIRED_FAMILY_FIELDS if k not in e]
            if missing:
                bad.append(f"{where}: 缺字段 {missing}")
                continue
            for k in ("template", "provenance", "source_url", "source_quote", "note"):
                if not isinstance(e[k], str) or not e[k].strip():
                    bad.append(f"{where}: {k} 必须是非空字符串")
            if not isinstance(e["native"], bool):
                bad.append(f"{where}: native 必须是 bool")
            if e["provenance"] not in PROVENANCES:
                bad.append(f"{where}: provenance {e['provenance']!r} 不在 {PROVENANCES}")
            elif e["native"] != (e["provenance"] in NATIVE_PROVENANCES):
                bad.append(f"{where}: native={e['native']} 与 provenance={e['provenance']} 不符")
            n = e["max_new_tokens"]
            if isinstance(n, bool) or not isinstance(n, int) or n < 1:
                bad.append(f"{where}: max_new_tokens 必须是正整数")
            if e["parse"] not in ENTRY_PARSERS:
                bad.append(f"{where}: parse {e['parse']!r} 不在 {ENTRY_PARSERS}")
            if isinstance(e["template"], str):
                try:
                    used = _placeholders(e["template"])
                except ValueError as exc:
                    bad.append(f"{where}: 模板的占位符写法不对：{exc}")
                    continue
                if used - allowed:
                    bad.append(f"{where}: 模板用了未声明的占位符 {sorted(used - allowed)}")
                if e["template"] != e["template"].strip():
                    bad.append(f"{where}: 模板首尾有空白")
            if t == "ovd" and not isinstance(e.get("category_joiner"), str):
                bad.append(f"{where}: 检测要给 category_joiner")
        if t == "spatial_embspatial":
            for k in ("option_format", "option_joiner"):
                if not isinstance(spec.get(k), str):
                    bad.append(f"{t}: 缺 {k}")
        if t == "rel_depth_nyu" and not (isinstance(spec.get("choices"), list) and len(spec["choices"]) == 2):
            bad.append(f"{t}: 要给两个选项原文 choices（选项原文匹配用）")
    if bad:
        raise PromptConfigError("提示配置不合格：\n- " + "\n- ".join(bad))


def _posint(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and v >= 1


def _validate_protocol(cfg: Mapping[str, Any]) -> list[str]:
    """协议 v2 的五节（DEV-0017）：letter_readout、report_columns、refusal_by_person、batch_protocol、acceptance。"""
    bad: list[str] = []
    tasks = cfg.get("tasks", {})
    lr = cfg.get("letter_readout", {})
    lt = lr.get("tasks", [])
    if not lt or not set(lt) <= set(TASKS) or len(set(lt)) != len(lt):
        bad.append("letter_readout.tasks 必须是不重复的任务名")
    for t in lt:
        for fam, e in tasks.get(t, {}).get("families", {}).items():
            if e.get("parse") != "choice":
                bad.append(f"letter_readout：{t}/{fam} 的自由作答解析必须是 choice")
    subs = lr.get("subsets", [])
    if not subs or not set(subs) <= set(SUBSET_KINDS) - set(THINK_SUBSETS) - {FREE_SUBSET}:
        bad.append("letter_readout.subsets 只能取关思考、非 free 的子集")
    if lr.get("max_new_tokens") != 1:
        bad.append("letter_readout.max_new_tokens 必须是 1")
    sp = lr.get("spellings", [])
    if (not isinstance(sp, list) or not sp or len(set(sp)) != len(sp)
            or any(not isinstance(x, str) or x.count("{letter}") != 1 or _placeholders(x) != {"letter"} for x in sp)):
        bad.append("letter_readout.spellings 必须是不重复、各含一个 {letter} 的写法")
    if lr.get("parse") != "letter_token":
        bad.append("letter_readout.parse 必须是 letter_token")
    for k in ("rule", "free_form_diagnostic", "basis"):
        if not str(lr.get(k, "")).strip():
            bad.append(f"letter_readout.{k} 不能为空")
    rc = cfg.get("report_columns", {})
    if any(not str(rc.get(k, "")).strip() for k in ("strict", "valid_rate", "conditional")):
        bad.append("report_columns 必须定义 strict、valid_rate、conditional")
    rp = cfg.get("refusal_by_person", {})
    if not rp.get("tasks") or not set(rp["tasks"]) <= {"refexp_box", "refexp_poly", "grid16"} or not rp.get("person_category"):
        bad.append("refusal_by_person 要给 RefCOCO 系的任务与 person_category")
    bp = cfg.get("batch_protocol", {})
    if not _posint(bp.get("batch_size")):
        bad.append("batch_protocol.batch_size 必须是正整数")
    if (bp.get("padding_side") not in PADDING_SIDES or bp.get("order") not in BATCH_ORDERS
            or bp.get("letter_batches") not in LETTER_BATCHES):
        bad.append(f"batch_protocol：padding_side ∈ {PADDING_SIDES}，order ∈ {BATCH_ORDERS}，letter_batches ∈ {LETTER_BATCHES}")
    ac = cfg.get("acceptance", {})
    if ac.get("version") != ACCEPTANCE_VERSION:
        bad.append(f"acceptance.version 必须是 {ACCEPTANCE_VERSION}")
    bb = ac.get("composition_b", {}).get("batch_size")
    if not _posint(bb) or bb == bp.get("batch_size"):
        bad.append("acceptance.composition_b.batch_size 必须是正整数，且与 batch_protocol.batch_size 不同")
    if "delta" in ac:
        bad.append("acceptance.delta（按 metric_class 取）已由 D59-3 改为按组设置：每组给 delta")
    if not str(ac.get("delta_rule", "")).strip():
        bad.append("acceptance.delta_rule 必须说明每组 δ 的取法与依据")
    groups = ac.get("groups", [])
    covered = [t for g in groups for t in g.get("tasks", [])]
    if sorted(covered) != sorted(TASKS):
        bad.append("acceptance.groups 必须恰好覆盖每个任务一次")
    if len({g.get("name") for g in groups}) != len(groups):
        bad.append("acceptance.groups 的组名不能重复")
    for g in groups:
        if g.get("metric") not in ACCEPTANCE_METRICS or g.get("metric_class") not in ACCEPTANCE_METRIC_CLASSES:
            bad.append(f"acceptance 组 {g.get('name')!r}：metric ∈ {ACCEPTANCE_METRICS}，metric_class ∈ {ACCEPTANCE_METRIC_CLASSES}")
        d = g.get("delta")
        if isinstance(d, bool) or not isinstance(d, (int, float)) or not 0 < d < 1:
            bad.append(f"acceptance 组 {g.get('name')!r}：delta 必须是 (0, 1) 之间的数（D59-3）")
        if not (g.get("take") == "all" or _posint(g.get("take"))):
            bad.append(f"acceptance 组 {g.get('name')!r}：take 必须是 all 或正整数")
        if g.get("n_min") is not None and not _posint(g.get("n_min")):
            bad.append(f"acceptance 组 {g.get('name')!r}：n_min 必须是正整数或 null")
        multi = {"refexp_poly", "grid16", "ovd"} & set(g.get("tasks", []))
        if multi and (g.get("metric") == "accuracy_pooled" or len(g.get("tasks", [])) != 1):
            bad.append(f"acceptance 组 {g.get('name')!r}：多边形、网格、检测各自单独成组，不进 0/1 的池")
    if not str(ac.get("take_salt_purpose", "")).strip():
        bad.append("acceptance.take_salt_purpose 不能为空")
    return bad


@dataclass(frozen=True)
class Rendered:
    """一条样本渲染好的指令，以及解析时要用的附带信息（多选题的字母与选项原文、计数的对象与规则分支）。"""

    text: str
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PromptConfig:
    data: Mapping[str, Any]
    path: str
    sha256: str

    @property
    def order(self) -> str:
        return self.data["image_order"]["value"]

    def entry(self, task: str, family: str) -> Mapping[str, Any]:
        try:
            return self.data["tasks"][task]["families"][family]
        except KeyError:
            raise KeyError(f"提示配置里没有 {task}/{family}") from None

    def decoding(self) -> dict[str, Any]:
        return dict(self.data["generation"]["decoding"])

    def enable_thinking(self, subset: str) -> bool:
        return bool(self.data["generation"]["enable_thinking"][subset_kind(subset)])

    def readout(self, task: str, subset: str) -> str:
        """这个任务在这个子集上怎么读答案：letter（受限字母读数：letter_readout 的任务 × 子集）或 text（自由生成后解析）。"""
        if task not in TASKS:
            raise KeyError(f"未知任务 {task!r}")
        lr = self.data["letter_readout"]
        return "letter" if task in lr["tasks"] and subset_kind(subset) in lr["subsets"] else "text"

    def max_new_tokens(self, task: str, family: str, subset: str) -> int:
        """模型生成的 token 数上限：受限字母读数取 letter_readout.max_new_tokens（1）；思考子集是预算强制收尾（D59-1），上限 =
        thinking_budget + answer_budget（补的 token 不计入）；其余（含 free 子集）取该任务该家族条目的 max_new_tokens。"""
        kind = subset_kind(subset)
        if self.readout(task, kind) == "letter":
            return int(self.data["letter_readout"]["max_new_tokens"])
        if kind in THINK_SUBSETS:
            return sum(self.think_budget())
        return int(self.entry(task, family)["max_new_tokens"])

    def think_budget(self) -> tuple[int, int]:
        """思考子集的预算强制收尾（D59-1）：(thinking_budget, answer_budget)。"""
        tb = self.data["generation"]["think_budget"]
        return int(tb["thinking_budget"]), int(tb["answer_budget"])

    def uses_think_budget(self, subset: str) -> bool:
        """这个子集是否走 generate_batch_budgeted（只有思考子集）。"""
        return subset_kind(subset) in THINK_SUBSETS

    def parser(self, task: str, family: str, subset: str | None = None) -> str:
        """解析函数名。给了 subset 且是受限字母读数时为 letter_token，否则取条目的 parse。"""
        if subset is not None and self.readout(task, subset) == "letter":
            return self.data["letter_readout"]["parse"]
        return self.entry(task, family)["parse"]

    def letter_spellings(self) -> tuple[str, ...]:
        return tuple(self.data["letter_readout"]["spellings"])

    def batch_protocol(self) -> dict[str, Any]:
        """冻结的批配置（batch_size、padding_side、order、letter_batches 与规则文字）。"""
        return dict(self.data["batch_protocol"])

    def acceptance(self) -> dict[str, Any]:
        """批处理验收 v3 的协议（composition_b、delta、groups、take 的规则；深拷贝）。"""
        return json.loads(json.dumps(self.data["acceptance"]))

    @property
    def countbenchqa_note(self) -> str:
        return self.data["countbenchqa_questions"]["provenance_note"]

    def render(self, task: str, family: str, sample: Sample) -> Rendered:
        e = self.entry(task, family)
        spec = self.data["tasks"][task]
        values: dict[str, Any] = {}
        extras: dict[str, Any] = {}
        f = sample.fields
        if task in ("refexp_box", "refexp_poly", "grid16"):
            values["expression"] = f["expression"]
        elif task == "ovd":
            values["categories"] = e["category_joiner"].join(f["categories"])
        elif task == "count":
            if COUNTBENCHQA_FIELD in f:  # 人工问句（--count-questions countbenchqa）：整句替换模板的问句部分
                text, obj, rule = countbenchqa_instruction(e["template"], self.data["countbenchqa_questions"]["question_part"],
                                                           f[COUNTBENCHQA_FIELD])
                return Rendered(text, {"object": obj, "object_rule": rule, "question": f[COUNTBENCHQA_FIELD]})
            obj, rule = count_object(f["text"], int(sample.gt["count"]))
            values["object"] = obj
            extras.update(object=obj, object_rule=rule)
        elif task in ("rel_depth_blink", "spatial_cvbench"):
            values["prompt"] = f["prompt"]
            extras.update(letters=[letter(i) for i in range(len(f["choices"]))], options=list(f["choices"]))
        elif task == "spatial_embspatial":
            opts = list(f["choices"])
            values["question"] = f["question"]
            values["options"] = spec["option_joiner"].join(
                spec["option_format"].format(letter=letter(i), text=t) for i, t in enumerate(opts))
            extras.update(letters=[letter(i) for i in range(len(opts))], options=opts)
        elif task == "rel_depth_nyu":
            extras.update(letters=[letter(i) for i in range(len(spec["choices"]))], options=list(spec["choices"]))
        else:
            raise KeyError(f"未知任务 {task!r}")
        return Rendered(e["template"].format(**values).strip(), extras)


def subset_kind(subset: str) -> str:
    """命令行的 --subset（eval、dev、think、extra、free）→ 配置里的子集类别。"""
    if subset not in SUBSET_KINDS:
        raise ValueError(f"subset 必须是 {SUBSET_KINDS} 之一，收到 {subset!r}")
    return subset


def load_prompt_config(path: str | Path = PROMPTS_PATH) -> PromptConfig:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    validate(data)
    return PromptConfig(data=data, path=str(p), sha256=file_sha256(p))


# ---------------------------------------------------------------------------
# CountBench 问句的对象（configs/e1_prompts.json 的 count_question_rule；版本号改了要同步改配置）
# ---------------------------------------------------------------------------

COUNT_RULE_VERSION = 1
COUNT_MAX_OBJECT_WORDS = 4
COUNT_FALLBACK_OBJECT = "objects"
COUNT_STOP_WORDS = frozenset(
    """in on at with and or of for from by to the a an that which who whose are is was were be been being as into onto
    over under near beside behind between among around about against along across through during after before while
    each per this these those its their his her our your my all both every isolated standing sitting lying hanging made
    set sets including""".split()
)
_DETERMINERS = frozenset("the a an these those my our your his her their its".split())
_BACK_SKIP = frozenset("of set sets pack group lot series collection block a an the".split())
_TOKEN = re.compile(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*|\S")
_ALPHA_WORD = re.compile(r"[A-Za-z]+(?:['’-][A-Za-z]+)*")
COUNT_RULE_BRANCHES = ("after_number", "before_number", "fallback_no_phrase", "fallback_no_mention")


def _is_count(tok: str, count: int) -> bool:
    return tok == str(count) or NUMBER_WORDS.get(tok.lower()) == count


def _object_word(tok: str) -> bool:
    return bool(_ALPHA_WORD.fullmatch(tok)) and tok.lower() not in COUNT_STOP_WORDS


def count_object(caption: str, count: int) -> tuple[str, str]:
    """CountBench 的 caption 与 GT 计数 → (问句里的计数对象, 规则分支)。规则见 configs/e1_prompts.json 的 count_question_rule。

    分支：after_number（从计数提及往后取到）、before_number（往后为空，往前取到）、fallback_no_phrase（两边都为空）、
    fallback_no_mention（caption 里没有与 GT 相等的数）；后两种取 COUNT_FALLBACK_OBJECT。
    """
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError(f"count 必须是非负整数，收到 {count!r}")
    toks = _TOKEN.findall(caption)
    k = next((i for i, t in enumerate(toks) if _is_count(t, count)), None)
    if k is None:
        return COUNT_FALLBACK_OBJECT, "fallback_no_mention"
    i = k + 1
    while i < len(toks) and (_is_count(toks[i], count) or toks[i] in "()"):
        i += 1
    if i < len(toks) and toks[i].lower() == "of":
        i += 1
        while i < len(toks) and toks[i].lower() in _DETERMINERS:
            i += 1
    words: list[str] = []
    while i < len(toks) and len(words) < COUNT_MAX_OBJECT_WORDS and _object_word(toks[i]):
        words.append(toks[i])
        i += 1
    if words:
        return " ".join(words), "after_number"
    j = k - 1
    while j >= 0 and toks[j].lower() in _BACK_SKIP:
        j -= 1
    back: list[str] = []
    while j >= 0 and len(back) < COUNT_MAX_OBJECT_WORDS and _object_word(toks[j]):
        back.append(toks[j])
        j -= 1
    if back:
        return " ".join(reversed(back)), "before_number"
    return COUNT_FALLBACK_OBJECT, "fallback_no_phrase"


# ---------------------------------------------------------------------------
# CountBench 的人工问句（configs/e1_prompts.json 的 countbenchqa_questions；DEV-0015 §4）
# ---------------------------------------------------------------------------

COUNT_QUESTION_SOURCES: tuple[str, ...] = ("heuristic", "countbenchqa")
DEFAULT_COUNT_QUESTIONS = "countbenchqa"  # D36-2：默认用人工问句；启发式保留为可选（--count-questions heuristic）
COUNTBENCHQA_FIELD = "countbenchqa_question"  # src/e1/tasks.select 在 countbenchqa 模式下放进 Sample.fields 的键
COUNTBENCHQA_RULES: tuple[str, ...] = ("countbenchqa", "countbenchqa_free_form")


def countbenchqa_instruction(template: str, question_part: str, question: str) -> tuple[str, str | None, str]:
    """人工问句 → (指令, 对象, 规则分支)。用整句问句（不抽对象）：指令 = 问句 + 模板里 question_part 之后的部分，去掉首尾空白；
    模板本身不变。问句恰是 question_part 的句式（如 "How many cars are there in the image?"）时，结果与把 {object} 换成那个对象
    逐字相同，分支记 countbenchqa、对象记那个短语；否则按原句使用，对象记 None，分支记 countbenchqa_free_form。"""
    if not template.startswith(question_part):
        raise PromptConfigError(f"计数模板 {template!r} 不以 question_part {question_part!r} 开头")
    q = question.strip()
    if not q:
        raise ValueError("人工问句是空的")
    pre, post = question_part.split("{object}")
    m = re.fullmatch(re.escape(pre) + r"(?P<object>.+?)" + re.escape(post), q)
    text = (q + template[len(question_part):]).strip()
    if m:
        return text, m.group("object"), "countbenchqa"
    return text, None, "countbenchqa_free_form"


__all__ = [
    "ACCEPTANCE_METRICS",
    "ACCEPTANCE_METRIC_CLASSES",
    "ACCEPTANCE_VERSION",
    "COUNTBENCHQA_FIELD",
    "COUNTBENCHQA_RULES",
    "COUNT_FALLBACK_OBJECT",
    "COUNT_MAX_OBJECT_WORDS",
    "COUNT_QUESTION_SOURCES",
    "COUNT_RULE_BRANCHES",
    "COUNT_RULE_VERSION",
    "COUNT_STOP_WORDS",
    "DEFAULT_COUNT_QUESTIONS",
    "ENTRY_PARSERS",
    "FREE_SUBSET",
    "PARSERS",
    "PROMPTS_PATH",
    "PromptConfig",
    "PromptConfigError",
    "READOUTS",
    "Rendered",
    "SUBSET_KINDS",
    "TASKS",
    "THINK_METHOD",
    "THINK_SUBSETS",
    "count_object",
    "countbenchqa_instruction",
    "file_sha256",
    "load_prompt_config",
    "subset_kind",
    "validate",
]
