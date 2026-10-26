"""W2-c 加回放（DEV-0065；D125-2 (b)）的自蒸馏回放数据：基座模型自己对一组通用指令与计数提示的回答（贪心、关思考）。
不引入任何外部模型或新的监督：提示来自一个公开的指令数据集与 W2-c 的训练图，回答由 base 自己生成；训练时目标就是这些回答的 token 原样
（src/w2/w2c_train.py 的回放流）。

**两类提示**（plan，只用 CPU；结果写进 <out>/replay/manifest.json，不含机器相关的路径，逐字节可复现）：
1. 通用指令（general）：databricks-dolly-15k（DOLLY：固定 revision 与文件 sha256；CC BY-SA 3.0，Databricks 员工手写的指令，
   不是模型生成的）。
   - 提示 = instruction.strip()，context 不为空时再接 "\\n\\n" + context.strip()（general_prompt）；id = dolly/<行号>（0 起，文件行序）。
   - 依次剔除（每步记条数与 id 的 sha256）：提示超过 MAX_PROMPT_CHARS 个字符（too_long）；含对话模板的结构性标记（STRUCTURAL_MARKERS，
     structural_marker）；与评测题重叠（eval_overlap，见下，逐条记命中的参照题）；规范化后与行号更小的某一行相同（duplicate）。
   - 剩下的按 hash_rank(salt = SALT, purpose = PURPOSE_GENERAL) 升序取前 N 条（默认 DEFAULT_N["general"] = 2,000）。
   - **与评测不重叠**（OverlapIndex）：参照集 = 健康检查的固定 200 题（configs/health_check_items.json，sha256 = src/health/protocol.
     ITEMS_SHA256）+ MMLU-Pro test 全部 12,032 题（题干与每个选项）+ GSM8K test 全部 1,319 题 + IFEval 全部 541 条（三者的固定 revision
     与 sha256 同 src/health/protocol.SOURCES）。lang_battery（DEV-0064）的题取自 MMLU-Pro test 与 IFEval 全部，都在参照集里。
     规则：文字规范化（NFKC、casefold、取 \\w+ 的词，norm_words）后，(a) 提示与某个题干 / 提示全文相同，或 (b) 两者有一个相同的连续
     NGRAM（13）个词，就算重叠（GPT-3 的污染分析口径）。
2. 计数（count）：W2-c 训练数据（DEV-0060 的 w2c_data，--w2c-data）的句子所在的 train 图。
   - 这些图已按 COCO 图像 id 排除 W2 的全部评测图、按 E3 训练池的剔除清单去重（非 --tiny 时核对它的 manifest.json：
     src/w2/w2c_data.check_manifest 通过、选中的 id 恰是数据的行；E3 训练池的剔除清单比较过 countbench/eval，load_train_pool 的 check_dedupe）。
   - 每张图里"有标注的 COCO 类别" = 落地的 RefCOCO 系 train 标注（E3 训练池，src/w2/w2c_data.load_train_pool）里这张图上全部被指代物体
     的类别（三个数据集合并）。落地的标注只含被指代的物体，不是这张图的全部 COCO 标注；回答由 base 生成、不用真值计数，所以只需要
     "图里确有这一类"。
   - **与 CountBench 的图不重叠**：E1 落地的 countbench 清单（<E1 数据根>/countbench/MANIFEST.json，src/data/train_dedupe.export_manifest）
     与这些图按图像键与感知哈希比较（src/data/dedupe.cross_overlap，汉明距离 ≤ PHASH_MAX_HAMMING），重合的图整张剔除（预期 0）。
   - 总体 = (图, 类别) 对，id = <图像键>|<类别>；按 hash_rank(salt = SALT, purpose = PURPOSE_COUNT) 取前 N 条（默认 2,000）。
   - 提示 = E1 计数提示的原文（configs/e1_prompts.json 的 count 本家族的 template，同 src/e1/prompts.PromptConfig.render 的写法）把
     {object} 换成类别的复数（COCO_PLURALS；表外的类别照抄类别名，object_rule 记 fallback_name，metrics 计数）。

**回答**（GPU；--stage general、count）：
- 通用：健康检查的生成口径（src/health/run.generate_text_only：零张图、src/health/protocol.ORDER、关思考、protocol.GREEDY_KWARGS），上限
  MAX_NEW_TOKENS["general"] = 512；
- 计数：E1 计数的口径（原图 = src/w2/w2c_train.load_pil，同 src/e1/runner.load_images；configs/e1_prompts.json 的 image_order、eval 子集的
  思考开关（关）与 decoding），上限 MAX_NEW_TOKENS["count"] = 16；
- 批：manifest 的顺序里连续取 batch_size 条（默认 E1 冻结的批大小 64），src/e1/runner.generate_rows（左填充的 generate_batch；
  健康检查本身是逐条 generate，回放的通用回答为了吞吐按批生成，回答仍是 base 自己贪心的输出）；
- 每条一行（ROW_KEYS）：提示、图（计数）、回答（split_answer 的 text）、raw_text、生成的 token（含停止 token）、生成长度、是否停下、
  是否截断（truncated = 没有生成停止 token、写满上限）。**截断的照实记，训练时不用**。
- 续跑：已写完的整批核对 id 后保留，没写完的那一批整批重算（批组成与一次跑完相同）；除耗时（logs/、meta.json、compute.json）外与一次
  跑完逐字节相同。

**输出**（<out>/，<out> 就是训练时的 --replay-data）：config.yaml（运行身份：模型与 revision、图像口径、两类的生成口径、manifest 的
sha256）、meta.json、replay/manifest.json、replay/general.jsonl、replay/count.jsonl、metrics.json（只由各行算）、README.md、
compute.json（src/accounting.py；墙钟来自 logs/timing.jsonl，同一 id 取最后一条）、logs/。

**训练时读**（load_replay_data）：config.yaml 的模型（注册键、revision、tiny）必须与训练的模型相同；两类的行恰是 manifest 的条目（跑完）；
每行的格式、思考开关与顺序对得上；池 = 指定种类里没有截断的行。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml

from src.accounting import Compute, merge, write_compute_json
from src.constants import BOOTSTRAP_SEED
from src.data.constants import PHASH_MAX_HAMMING
from src.data.dedupe import cross_overlap
from src.data.subsets import dump_config, hash_rank, population_digest, sort_ids
from src.e1.runner import CONFIG, METRICS, README, _append, _identity, _jsonl_line, generate_rows, ids_digest, read_jsonl_tolerant
from src.e1.scoring import distribution
from src.health import protocol as HP
from src.health.parquet_lite import read_rows as read_parquet_rows
from src.runlog import write_run_metadata

REPO_ROOT = Path(__file__).resolve().parents[2]
FRAMEWORK = "src/w2/w2c_replay（DEV-0065；D125-2 (b)）"
SCHEMA = 1
SALT = "w2c_replay_v1"
PURPOSE_GENERAL = "w2c_replay_general"
PURPOSE_COUNT = "w2c_replay_count"
GENERAL = "general"
COUNT = "count"
KINDS: tuple[str, ...] = (GENERAL, COUNT)
PURPOSES = {GENERAL: PURPOSE_GENERAL, COUNT: PURPOSE_COUNT}
DEFAULT_N = {GENERAL: 2_000, COUNT: 2_000}
TINY_N = {GENERAL: 6, COUNT: 6}
MAX_NEW_TOKENS = {GENERAL: 512, COUNT: 16}  # 任务书 DEV-0065：作答上限够长
TINY_MAX_NEW_TOKENS = {GENERAL: 8, COUNT: 4}  # --tiny：小型随机模型不会停，只要跑通
MAX_PROMPT_CHARS = 4_000  # dolly 的 15,011 条里 157 条超过（1.0%；开发机按固定 revision 实测）
NGRAM = 13
DATA_DIR = "replay"
MANIFEST = "manifest.json"
FILES = {GENERAL: "general.jsonl", COUNT: "count.jsonl"}
TIMING = "logs/timing.jsonl"
COMPUTE = "compute.json"
KIND_DATA = "w2c_replay_data"
KIND_SELECTION = "w2c_replay_selection"
E1_COUNT_TASK = "count"
E1_SUBSET = "eval"  # E1 的主口径（思考开关、decoding 按 eval 子集取）
COUNTBENCH_MANIFEST = "countbench/MANIFEST.json"  # 相对 E1 数据根
COUNTBENCH_SET = "countbench/eval"  # src/data/train_dedupe.E1_SET_NAMES["countbench"]
TINY_RAW_DIR = REPO_ROOT / "tests" / "fixtures" / "w2c_replay" / "raw"  # --tiny 的原始文件（测试造的，不是真数据）
TINY_EVAL_ROOT = REPO_ROOT / "tests" / "fixtures" / "e1" / "root"  # --tiny 的 E1 数据根（countbench 的 fixture）

DOLLY = HP.Source(
    repo="databricks/databricks-dolly-15k",
    revision="bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a",  # HF API 2026-10-25：main 的 commit（文件最后改于 d72c16e4，2023-05-04）
    config="default",
    split="train",
    file="databricks-dolly-15k.jsonl",
    file_sha256="2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec",  # HF 的 LFS oid；开发机下载后实测相同
    license="CC-BY-SA-3.0",
    n_rows=15_011,
)
DOLLY_NAME = "databricks-dolly-15k"
REFERENCE_SETS: tuple[str, ...] = ("health_check", "mmlu_pro", "gsm8k", "ifeval")

# 对话模板的结构性标记（三个家族的特殊 token 写法：<|…|>、<think>、<image>、Gemma 的 <start_of_turn>……）；含有的提示不用
STRUCTURAL_MARKERS = re.compile(r"<\|[^<>|\n]{1,40}\|>|</?(?:think|image|img|tool_call|tool_response|start_of_turn|end_of_turn)>"
                                r"|<IMG_CONTEXT>")
_WORD = re.compile(r"\w+")

# COCO 80 类（instances_train2014.json 的 categories 名）→ 计数提示里的复数
COCO_PLURALS: Mapping[str, str] = {
    "person": "people", "bicycle": "bicycles", "car": "cars", "motorcycle": "motorcycles", "airplane": "airplanes", "bus": "buses",
    "train": "trains", "truck": "trucks", "boat": "boats", "traffic light": "traffic lights", "fire hydrant": "fire hydrants",
    "stop sign": "stop signs", "parking meter": "parking meters", "bench": "benches", "bird": "birds", "cat": "cats", "dog": "dogs",
    "horse": "horses", "sheep": "sheep", "cow": "cows", "elephant": "elephants", "bear": "bears", "zebra": "zebras",
    "giraffe": "giraffes", "backpack": "backpacks", "umbrella": "umbrellas", "handbag": "handbags", "tie": "ties",
    "suitcase": "suitcases", "frisbee": "frisbees", "skis": "skis", "snowboard": "snowboards", "sports ball": "sports balls",
    "kite": "kites", "baseball bat": "baseball bats", "baseball glove": "baseball gloves", "skateboard": "skateboards",
    "surfboard": "surfboards", "tennis racket": "tennis rackets", "bottle": "bottles", "wine glass": "wine glasses", "cup": "cups",
    "fork": "forks", "knife": "knives", "spoon": "spoons", "bowl": "bowls", "banana": "bananas", "apple": "apples",
    "sandwich": "sandwiches", "orange": "oranges", "broccoli": "broccoli", "carrot": "carrots", "hot dog": "hot dogs",
    "pizza": "pizzas", "donut": "donuts", "cake": "cakes", "chair": "chairs", "couch": "couches", "potted plant": "potted plants",
    "bed": "beds", "dining table": "dining tables", "toilet": "toilets", "tv": "TVs", "laptop": "laptops", "mouse": "computer mice",
    "remote": "remotes", "keyboard": "keyboards", "cell phone": "cell phones", "microwave": "microwaves", "oven": "ovens",
    "toaster": "toasters", "sink": "sinks", "refrigerator": "refrigerators", "book": "books", "clock": "clocks", "vase": "vases",
    "scissors": "scissors", "teddy bear": "teddy bears", "hair drier": "hair dryers", "toothbrush": "toothbrushes",
}
OBJECT_RULES = ("coco_plural", "fallback_name")

ROW_KEYS: tuple[str, ...] = ("id", "kind", "source", "prompt", "images", "image_size", "order", "enable_thinking", "max_new_tokens",
                             "answer", "raw_text", "token_ids", "n_generated_tokens", "n_prompt_tokens", "n_visual_tokens",
                             "n_image_pixels", "stopped", "truncated")


class ReplayDataError(ValueError):
    """回放数据不能生成，或与已有的输出、训练的模型对不上。"""


# ---------------------------------------------------------------------------
# 原始文件（<raw>/<仓库名里的 / 换成 __>/<文件>，同 scripts/build_health_items.py）
# ---------------------------------------------------------------------------


def raw_path(raw_dir: str | Path, src: HP.Source) -> Path:
    return Path(raw_dir) / src.repo.replace("/", "__") / src.file


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fetch(src: HP.Source, raw_dir: str | Path, *, download: bool) -> Path:
    """一个来源的原始文件：不在就下载（download = True；huggingface_hub，固定 revision；服务器上设 HF_ENDPOINT 用镜像），核对 sha256。"""
    p = raw_path(raw_dir, src)
    if not p.exists():
        if not download:
            raise ReplayDataError(f"{p} 不在：先在能联网的节点上跑 scripts/build_w2c_replay.py --stage fetch --raw-dir {raw_dir}")
        from huggingface_hub import hf_hub_download

        hf_hub_download(src.repo, src.file, repo_type="dataset", revision=src.revision,
                        local_dir=str(Path(raw_dir) / src.repo.replace("/", "__")))
    digest = sha256_file(p)
    if digest != src.file_sha256:
        raise ReplayDataError(f"{src.repo}@{src.revision}:{src.file} 的 sha256 {digest} 与固定的 {src.file_sha256} 不同")
    return p


def all_sources() -> dict[str, HP.Source]:
    return {"dolly": DOLLY, **{k: HP.SOURCES[k] for k in HP.BENCHMARKS}}


def source_record(src: HP.Source, path: Path | None) -> dict[str, Any]:
    rec = {"repo": src.repo, "revision": src.revision, "file": src.file, "pinned_sha256": src.file_sha256, "license": src.license,
           "n_rows_pinned": src.n_rows}
    if path is None:
        return {**rec, "missing": True}
    return {**rec, "sha256": sha256_file(path), "missing": False}


def resolve_raw(raw_dir: str | Path, *, tiny: bool) -> dict[str, Path | None]:
    """dolly 与三个评测来源的原始文件。非 tiny：都必须在且 sha256 = 固定值；tiny：不核 sha256，评测来源可以缺（记 missing）。"""
    out: dict[str, Path | None] = {}
    for name, src in all_sources().items():
        if tiny:
            p = raw_path(raw_dir, src)
            if not p.exists() and name == "dolly":
                raise ReplayDataError(f"{p} 不在（--tiny 的 dolly fixture）")
            out[name] = p if p.exists() else None
        else:
            out[name] = fetch(src, raw_dir, download=False)
    return out


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


# ---------------------------------------------------------------------------
# 与评测不重叠
# ---------------------------------------------------------------------------


def norm_words(text: str) -> list[str]:
    """规范化后的词序列：NFKC、casefold、取 \\w+（含数字与各种文字；标点与空白都当分隔）。"""
    return _WORD.findall(unicodedata.normalize("NFKC", str(text)).casefold())


def ngrams(words: Sequence[str], n: int = NGRAM) -> list[str]:
    return [" ".join(words[i: i + n]) for i in range(len(words) - n + 1)]


@dataclass(frozen=True)
class RefItem:
    """一道参照题：set 名、题号、主文字（题干或提示：全文相同也算重叠）、其余文字（选项：只比 n-gram）。"""

    set: str
    id: str
    main: str
    extra: tuple[str, ...] = ()


def health_refs(items_path: str | Path, *, check_sha: bool = True) -> list[RefItem]:
    """健康检查的固定 200 题（configs/health_check_items.json；sha256 必须是 protocol.ITEMS_SHA256，同 scripts/health_check.py 的核对）。"""
    p = Path(items_path)
    raw = p.read_bytes()
    if check_sha and hashlib.sha256(raw).hexdigest() != HP.ITEMS_SHA256:
        raise ReplayDataError(f"{p} 的 sha256 不是 src/health/protocol.ITEMS_SHA256（健康检查的题被改过）")
    items = json.loads(raw)["items"]
    out = [RefItem("health_check", f"mmlu_pro/{it['id']}", it["question"], tuple(it["options"])) for it in items["mmlu_pro"]]
    out += [RefItem("health_check", f"gsm8k/{it['id']}", it["question"]) for it in items["gsm8k"]]
    out += [RefItem("health_check", f"ifeval/{it['id']}", it["prompt"]) for it in items["ifeval"]]
    return out


def source_refs(name: str, path: str | Path) -> list[RefItem]:
    """评测来源的全部题（MMLU-Pro test 的题干与选项、GSM8K test 的题、IFEval 的提示）。"""
    if name == "mmlu_pro":
        return [RefItem(name, str(r["question_id"]), r["question"], tuple(r["options"])) for r in read_parquet_rows(path)]
    if name == "gsm8k":
        return [RefItem(name, str(i), r["question"]) for i, r in enumerate(read_parquet_rows(path))]
    if name == "ifeval":
        return [RefItem(name, str(r["key"]), r["prompt"]) for r in read_jsonl(path)]
    raise KeyError(name)


def reference_sets(raw: Mapping[str, Path | None], *, items_path: str | Path, tiny: bool) -> tuple[dict[str, list[RefItem]], dict[str, Any]]:
    """参照集（REFERENCE_SETS 的顺序）与它们的记录（来源、条数、sha256）。非 tiny 时三个来源的条数必须等于固定值。"""
    refs: dict[str, list[RefItem]] = {"health_check": health_refs(items_path)}
    info: dict[str, Any] = {"health_check": {"file": "configs/health_check_items.json", "sha256": HP.ITEMS_SHA256,
                                             "n_items": len(refs["health_check"])}}
    for name in REFERENCE_SETS[1:]:
        src = HP.SOURCES[name]
        p = raw.get(name)
        if p is None:
            if not tiny:
                raise ReplayDataError(f"{name} 的原始文件缺失")
            refs[name] = []
            info[name] = source_record(src, None)
            continue
        refs[name] = source_refs(name, p)
        if not tiny and len(refs[name]) != src.n_rows:
            raise ReplayDataError(f"{name}：{len(refs[name])} 题，固定 revision 应有 {src.n_rows} 题")
        info[name] = {**source_record(src, p), "n_items": len(refs[name])}
    return refs, info


class OverlapIndex:
    """参照集的规范化全文与全部 NGRAM 词 n-gram（各取第一次出现的题，按 REFERENCE_SETS 与题序）。"""

    def __init__(self, refs: Mapping[str, Sequence[RefItem]], *, n: int = NGRAM) -> None:
        self.n = int(n)
        self.exact: dict[str, RefItem] = {}
        self.grams: dict[str, RefItem] = {}
        for name in REFERENCE_SETS:
            for it in refs.get(name, ()):
                w = norm_words(it.main)
                if w:
                    self.exact.setdefault(" ".join(w), it)
                for text in (it.main, *it.extra):
                    for g in ngrams(norm_words(text), self.n):
                        self.grams.setdefault(g, it)

    def find(self, text: str) -> dict[str, Any] | None:
        """第一个命中（先比全文，再按位置比 n-gram）；没有返回 None。"""
        w = norm_words(text)
        hit = self.exact.get(" ".join(w)) if w else None
        if hit is not None:
            return {"rule": "exact", "set": hit.set, "item": hit.id}
        for g in ngrams(w, self.n):
            hit = self.grams.get(g)
            if hit is not None:
                return {"rule": f"ngram{self.n}", "set": hit.set, "item": hit.id, "ngram": g}
        return None


# ---------------------------------------------------------------------------
# 通用指令（dolly）
# ---------------------------------------------------------------------------


def general_prompt(instruction: str, context: str) -> str:
    ins, ctx = str(instruction).strip(), str(context or "").strip()
    return f"{ins}\n\n{ctx}" if ctx else ins


def _ids_step(ids: Sequence[str]) -> dict[str, Any]:
    return {"n": len(ids), "sha256": population_digest(ids)}


def select_general(rows: Sequence[Mapping[str, Any]], index: OverlapIndex, *, n: int, allow_fewer: bool = False,
                   max_chars: int = MAX_PROMPT_CHARS) -> dict[str, Any]:
    """dolly 的行（文件行序）→ 选中的条目（hash_rank 顺序）与各步的记录（规则见模块说明）。"""
    removed: dict[str, list[str]] = {"too_long": [], "structural_marker": [], "eval_overlap": [], "duplicate": []}
    overlaps: list[dict[str, Any]] = []
    seen: dict[str, str] = {}
    kept: list[str] = []
    by_id: dict[str, dict[str, Any]] = {}
    for i, r in enumerate(rows):
        rid = f"dolly/{i}"
        prompt = general_prompt(r["instruction"], r.get("context", ""))
        if not prompt:
            raise ReplayDataError(f"{rid}：instruction 为空")
        if len(prompt) > max_chars:
            removed["too_long"].append(rid)
            continue
        if STRUCTURAL_MARKERS.search(prompt):
            removed["structural_marker"].append(rid)
            continue
        hit = index.find(prompt)
        if hit is not None:
            removed["eval_overlap"].append(rid)
            overlaps.append({"id": rid, **hit})
            continue
        key = " ".join(norm_words(prompt))
        if key in seen:
            removed["duplicate"].append(rid)
            continue
        seen[key] = rid
        kept.append(rid)
        by_id[rid] = {"id": rid, "source": {"dataset": DOLLY_NAME, "row": i, "category": str(r.get("category"))}, "prompt": prompt}
    if len(kept) < n and not allow_fewer:
        raise ReplayDataError(f"dolly 剔除之后只剩 {len(kept)} 条，不够 n = {n}")
    chosen = hash_rank(kept, PURPOSE_GENERAL, SALT)[:n]
    items = [by_id[i] for i in chosen]
    cats = Counter(it["source"]["category"] for it in items)
    return {
        "items": items,
        "steps": {
            "population": {"n_rows": len(rows), "rule": "dolly 的全部行；id = dolly/<行号>（0 起）"},
            "removed": {k: _ids_step(v) for k, v in removed.items()},
            "eval_overlap": {"rule": f"norm_words 后全文相同（exact），或有一个相同的连续 {index.n} 词（ngram{index.n}）",
                             "hits": overlaps, "by_set": dict(sorted(Counter(h["set"] for h in overlaps).items()))},
            "kept": _ids_step(kept),
            "selected": {"salt": SALT, "purpose": PURPOSE_GENERAL, "n_requested": n, "n": len(chosen), "sha256": population_digest(chosen),
                         "ordered_sha256": ids_digest(chosen), "by_category": dict(sorted(cats.items())),
                         "n_with_context": sum(1 for it in items if "\n\n" in it["prompt"])},
        },
    }


# ---------------------------------------------------------------------------
# 计数（W2-c 的训练图 × 图里有标注的 COCO 类别）
# ---------------------------------------------------------------------------


def count_object(category: str) -> tuple[str, str]:
    """类别 → (提示里的对象, 规则)：COCO_PLURALS 的复数，表外的照抄类别名。"""
    if category in COCO_PLURALS:
        return COCO_PLURALS[category], OBJECT_RULES[0]
    return str(category), OBJECT_RULES[1]


def count_prompt(template: str, obj: str) -> str:
    """同 src/e1/prompts.PromptConfig.render 的计数分支：template.format(object = …).strip()。"""
    return template.format(object=obj).strip()


def read_w2c_rows(w2c_data: str | Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """DEV-0060 的数据目录：*.jsonl 按文件名升序拼接（同 src/w2/w2c_train.load_w2c_data）。返回 (行, 分片的记录)。"""
    d = Path(w2c_data)
    shards = sorted(d.glob("*.jsonl"))
    if not shards:
        raise ReplayDataError(f"{d} 里没有 *.jsonl 分片")
    rows, info = [], []
    for f in shards:
        raw = f.read_bytes()
        lines = [x for x in raw.decode("utf-8").splitlines() if x.strip()]
        info.append({"file": f.name, "n": len(lines), "sha256": hashlib.sha256(raw).hexdigest()})
        rows += [json.loads(x) for x in lines]
    return rows, info


def check_w2c_manifest(w2c_data: str | Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """非 tiny：DEV-0060 的 manifest.json 自洽（check_manifest：不在 W2 评测图上……），选中的 id 恰是数据的行，image_of 与行的图对得上。"""
    from src.w2.w2c_data import MANIFEST as W2C_MANIFEST, check_manifest

    p = Path(w2c_data) / W2C_MANIFEST
    if not p.is_file():
        raise ReplayDataError(f"{p} 不在（DEV-0060 的去重与抽样记录）")
    doc = json.loads(p.read_text(encoding="utf-8"))
    bad = check_manifest(doc)
    if bad:
        raise ReplayDataError(f"{p} 的核对不过：{bad}")
    ids = [str(r["id"]) for r in rows]
    if sorted(ids) != sorted(str(i) for i in doc["selected"]["ids"]):
        raise ReplayDataError(f"{p} 选中的 id 与 {w2c_data} 的行不同")
    return {"manifest_sha256": sha256_file(p), "checks": dict(doc["checks"]),
            "w2_eval_images_sha256": doc["steps"]["w2_eval_images"]["images_sha256"], "image_of": doc["selected"]["image_of"]}


def countbench_hashes(eval_root: str | Path) -> tuple[dict[str, str], dict[str, Any]]:
    """E1 落地的 countbench 清单 → {图像键: 感知哈希}（src/data/train_dedupe.export_manifest 的 countbench/eval 集合）。"""
    from src.data.train_dedupe import export_manifest

    p = Path(eval_root) / COUNTBENCH_MANIFEST
    if not p.is_file():
        raise ReplayDataError(f"{p} 不在（E1 落地的 CountBench 清单）")
    sets, _ = export_manifest(p)
    if COUNTBENCH_SET not in sets:
        raise ReplayDataError(f"{p} 里没有 {COUNTBENCH_SET} 集合")
    items = sets[COUNTBENCH_SET]["items"]
    m = {str(k): h for k, h, _ in items}
    return m, {"manifest": COUNTBENCH_MANIFEST, "manifest_sha256": sha256_file(p), "n_images": len(m),
               "n_with_phash": sum(1 for h in m.values() if h)}


def _rel(path: str | Path, root: str | Path) -> str:
    try:
        return Path(path).relative_to(Path(root)).as_posix()
    except ValueError:
        return Path(os.path.relpath(Path(path).resolve(), Path(root).resolve())).as_posix()


def select_count(w2c_rows: Sequence[Mapping[str, Any]], pool: Sequence[Any], countbench: Mapping[str, str], *, train_root: str | Path,
                 template: str, n: int, allow_fewer: bool = False, image_of: Mapping[str, str] | None = None) -> dict[str, Any]:
    """W2-c 的行 + E3 训练池（Sample）+ CountBench 的 {键: 感知哈希} → 选中的条目（hash_rank 顺序）与各步的记录（规则见模块说明）。
    image_of（非 tiny：DEV-0060 manifest 的 selected.image_of）给了就逐行核对图像键。"""
    by_id = {s.id: s for s in pool}
    images: dict[str, dict[str, Any]] = {}
    for r in w2c_rows:
        s = by_id.get(str(r["id"]))
        if s is None:
            raise ReplayDataError(f"W2-c 的行 {r['id']} 不在 E3 的训练池里")
        if len(s.image_keys) != 1:
            raise ReplayDataError(f"{s.id}：应恰有一个图像键，收到 {s.image_keys}")
        key = str(s.image_keys[0])
        if image_of is not None and str(image_of.get(str(r["id"]))) != key:
            raise ReplayDataError(f"{r['id']}：DEV-0060 manifest 的图 {image_of.get(str(r['id']))} 与训练池的 {key} 不同")
        rel = _rel(s.images[0], train_root)
        if rel != r["image"] or list(s.fields["image_size"]) != list(r["image_size"]):
            raise ReplayDataError(f"{r['id']}：W2-c 行的图 {r['image']} {r['image_size']} 与训练池的 {rel} {s.fields['image_size']} 不同")
        images.setdefault(key, {"image": rel, "image_size": [int(v) for v in r["image_size"]], "phash": s.phash[0] if s.phash else None})
    cats: dict[str, set[str]] = {k: set() for k in images}
    for s in pool:
        k = str(s.image_keys[0])
        if k in cats:
            cats[k].add(str(s.fields["category"]))
    ov = cross_overlap({k: v["phash"] for k, v in images.items()}, dict(countbench), max_hamming=PHASH_MAX_HAMMING)
    hit = set(ov["same_key"]) | {h["a"] for h in ov["phash_near"]}
    keys = [k for k in sort_ids(images) if k not in hit]
    pairs = {f"{k}|{c}": (k, c) for k in keys for c in sorted(cats[k])}
    if len(pairs) < n and not allow_fewer:
        raise ReplayDataError(f"计数的 (图, 类别) 对只有 {len(pairs)} 个，不够 n = {n}")
    chosen = hash_rank(list(pairs), PURPOSE_COUNT, SALT)[:n]
    items = []
    for pid in chosen:
        k, c = pairs[pid]
        obj, rule = count_object(c)
        items.append({"id": pid, "source": {"image_key": k, "category": c, "object": obj, "object_rule": rule},
                      "image": images[k]["image"], "image_size": images[k]["image_size"], "prompt": count_prompt(template, obj)})
    by_cat = Counter(it["source"]["category"] for it in items)
    return {
        "items": items,
        "steps": {
            "w2c_images": {"n_rows": len(w2c_rows), "n_images": len(images), "images_sha256": population_digest(images),
                           "rule": "W2-c 训练数据的句子所在的图（图像键取 E3 训练池的 coco:<id>）"},
            "categories": {"n_pairs_before_countbench": sum(len(v) for v in cats.values()),
                           "n_images_without_category": sum(1 for v in cats.values() if not v),
                           "rule": "E3 训练池（三个数据集合并）里这张图上全部被指代物体的 COCO 类别"},
            "countbench": {"rule": f"图像键相同，或感知哈希汉明距离 ≤ {PHASH_MAX_HAMMING}（src/data/dedupe.cross_overlap）",
                           "n_same_key": ov["n_same_key"], "n_phash_near": ov["n_phash_near"], "near": ov["phash_near"][:20],
                           "n_images_removed": len(hit), "images_removed": sort_ids(hit),
                           "n_w2c_images_without_phash": sum(1 for v in images.values() if not v["phash"])},
            "population": {"n_images": len(keys), "n_pairs": len(pairs), "sha256": population_digest(pairs)},
            "selected": {"salt": SALT, "purpose": PURPOSE_COUNT, "n_requested": n, "n": len(chosen), "sha256": population_digest(chosen),
                         "ordered_sha256": ids_digest(chosen), "n_images": len({it["source"]["image_key"] for it in items}),
                         "by_category": dict(sorted(by_cat.items(), key=lambda kv: (-kv[1], kv[0]))),
                         "n_object_fallback": sum(1 for it in items if it["source"]["object_rule"] != OBJECT_RULES[0])},
        },
    }


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------


def build_manifest(general: Mapping[str, Any], count: Mapping[str, Any], *, tiny: bool, family: str, sources: Mapping[str, Any],
                   references: Mapping[str, Any], w2c: Mapping[str, Any], pool_steps: Mapping[str, Any], countbench: Mapping[str, Any],
                   count_template: Mapping[str, Any]) -> dict[str, Any]:
    doc = {
        "schema": SCHEMA, "kind": KIND_SELECTION, "framework": FRAMEWORK, "tiny": bool(tiny), "family": family, "salt": SALT,
        GENERAL: {"source": dict(sources["dolly"]), "prompt_rule": "instruction.strip()；context 不为空时 + \"\\n\\n\" + context.strip()",
                  "max_prompt_chars": MAX_PROMPT_CHARS, "structural_markers": STRUCTURAL_MARKERS.pattern,
                  "reference_sets": dict(references), **general["steps"]},
        COUNT: {"w2c_data": dict(w2c), "train_pool": {k: pool_steps[k] for k in ("population", "e3_exclusions")},
                "countbench_source": dict(countbench), "prompt": dict(count_template),
                "object_rule": "COCO_PLURALS（src/w2/w2c_replay.py）的复数；表外的类别照抄类别名", **count["steps"]},
        "items": {GENERAL: list(general["items"]), COUNT: list(count["items"])},
    }
    bad = check_manifest(doc)
    if bad:
        raise ReplayDataError("manifest 自洽核对不过：" + "；".join(bad))
    return doc


def check_manifest(doc: Mapping[str, Any]) -> list[str]:
    """manifest 的自洽（不要数据）：两类的条目 id 不重复、条数与 sha256 对得上、是 hash_rank 顺序；计数的图不在 CountBench 剔除的图里。"""
    bad: list[str] = []
    for kind in KINDS:
        items = doc["items"][kind]
        ids = [str(it["id"]) for it in items]
        st = doc[kind]["selected"]
        if len(set(ids)) != len(ids):
            bad.append(f"{kind}：id 有重复")
            continue
        if len(ids) != st["n"] or population_digest(ids) != st["sha256"] or ids_digest(ids) != st["ordered_sha256"]:
            bad.append(f"{kind}：条目与记的条数或 sha256 不符")
        if ids != hash_rank(ids, st["purpose"], st["salt"]) or st["salt"] != SALT or st["purpose"] != PURPOSES[kind]:
            bad.append(f"{kind}：条目不是 hash_rank({SALT}, {PURPOSES[kind]}) 的顺序")
    removed = set(doc[COUNT]["countbench"]["images_removed"])
    if any(it["source"]["image_key"] in removed for it in doc["items"][COUNT]):
        bad.append("count：有条目在与 CountBench 重合的图上")
    return bad


def manifest_text(doc: Mapping[str, Any]) -> str:
    return dump_config(doc)


def manifest_path(out: str | Path) -> Path:
    return Path(out) / DATA_DIR / MANIFEST


def ensure_manifest(out: str | Path, doc: Mapping[str, Any]) -> str:
    """<out>/replay/manifest.json：没有就写；已有必须逐字节相同。返回它的 sha256。"""
    text = manifest_text(doc)
    p = manifest_path(out)
    if p.exists():
        if p.read_text(encoding="utf-8") != text:
            raise ReplayDataError(f"{p} 与这次重算的不同（原始文件、训练池或规则变了），拒绝混用；换一个 --out")
    else:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_manifest(out: str | Path) -> tuple[dict[str, Any], str]:
    p = manifest_path(out)
    if not p.is_file():
        raise ReplayDataError(f"{p} 不在：先跑 --stage plan")
    raw = p.read_bytes()
    doc = json.loads(raw)
    bad = check_manifest(doc)
    if bad:
        raise ReplayDataError(f"{p} 的自洽核对不过：{bad}")
    return doc, hashlib.sha256(raw).hexdigest()


# ---------------------------------------------------------------------------
# 生成
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GenSpec:
    """一类提示的生成口径（进运行身份）。"""

    kind: str
    order: str
    enable_thinking: bool
    max_new_tokens: int
    decoding: Mapping[str, Any]
    batch_size: int
    protocol: str

    def as_dict(self) -> dict[str, Any]:
        method = "generate" if self.batch_size == 1 else "generate_batch"
        return {"order": self.order, "enable_thinking": self.enable_thinking, "max_new_tokens": self.max_new_tokens,
                "decoding": dict(self.decoding), "batch_size": self.batch_size, "method": method, "protocol": self.protocol}


def gen_specs(prompts: Any, *, batch_size: int, tiny: bool) -> dict[str, GenSpec]:
    """通用 = 健康检查的口径；计数 = E1 计数的口径（模块说明）。prompts：src/e1/prompts.PromptConfig。"""
    cap = TINY_MAX_NEW_TOKENS if tiny else MAX_NEW_TOKENS
    return {
        GENERAL: GenSpec(GENERAL, HP.ORDER, bool(HP.ENABLE_THINKING), int(cap[GENERAL]), dict(HP.GREEDY_KWARGS), int(batch_size),
                         "健康检查的生成口径（src/health/run.generate_text_only：零张图、protocol.ORDER、关思考、protocol.GREEDY_KWARGS）；"
                         "批量生成（左填充，同 E1 的批配置；健康检查是逐条 generate）"),
        COUNT: GenSpec(COUNT, prompts.order, bool(prompts.enable_thinking(E1_SUBSET)), int(cap[COUNT]), prompts.decoding(),
                       int(batch_size), "E1 计数的口径（configs/e1_prompts.json 的 image_order、eval 子集的思考开关与 decoding；原图）"),
    }


def data_file(out: str | Path, kind: str) -> Path:
    return Path(out) / DATA_DIR / FILES[kind]


def item_images(item: Mapping[str, Any], kind: str) -> list[str]:
    return [item["image"]] if kind == COUNT else []


def make_row(item: Mapping[str, Any], kind: str, spec: GenSpec, res: Any) -> dict[str, Any]:
    """一条的记录（键序 = ROW_KEYS）。res：src/adapters/base.GenerationResult。"""
    ids = [int(t) for t in res.token_ids]
    row = {"id": item["id"], "kind": kind, "source": dict(item["source"]), "prompt": item["prompt"], "images": item_images(item, kind),
           "image_size": list(item["image_size"]) if kind == COUNT else None, "order": spec.order, "enable_thinking": spec.enable_thinking,
           "max_new_tokens": spec.max_new_tokens, "answer": res.text, "raw_text": res.raw_text, "token_ids": ids,
           "n_generated_tokens": int(res.n_generated_tokens), "n_prompt_tokens": int(res.n_prompt_tokens),
           "n_visual_tokens": int(res.n_visual_tokens), "n_image_pixels": int(res.n_image_pixels or 0), "stopped": bool(res.stopped),
           "truncated": not bool(res.stopped)}
    assert tuple(row) == ROW_KEYS
    if row["n_generated_tokens"] != len(ids):
        raise ReplayDataError(f"{item['id']}：n_generated_tokens {row['n_generated_tokens']} 与 token 数 {len(ids)} 不同")
    return row


def resume_rows(path: Path, items: Sequence[Mapping[str, Any]], batch_size: int) -> list[dict[str, Any]]:
    """已写的行：末尾不完整的行丢掉；id 必须是 items 的前缀；不在批的边界上时截回那一批的开头（整批重算）。文件随之改写。"""
    recs = read_jsonl_tolerant(path, repair=True)
    want = [str(it["id"]) for it in items]
    got = [str(r["id"]) for r in recs]
    if got != want[: len(got)]:
        raise ReplayDataError(f"{path}：已写的 id 不是 manifest 条目的前缀，拒绝续跑")
    keep = len(recs) if len(recs) == len(items) else (len(recs) // batch_size) * batch_size
    if keep != len(recs):
        tmp = path.with_name(path.name + ".part")
        tmp.write_text("".join(_jsonl_line(r) for r in recs[:keep]), encoding="utf-8")
        tmp.replace(path)
    return recs[:keep]


def run_kind(adapter: Any, kind: str, items: Sequence[Mapping[str, Any]], out: str | Path, *, spec: GenSpec, data_root: str | Path,
             run_k: int, log: Callable[[str], None] | None = None) -> dict[str, Any]:
    """一类提示的生成（规则见模块说明）；返回本次的统计。"""
    from src.w2.w2c_train import load_pil  # 训练时的回放样本也用它读图（src/w2/w2c_train.ReplayBuilder）

    log = log or (lambda m: print(m, file=sys.stderr, flush=True))
    out = Path(out)
    path = data_file(out, kind)
    path.parent.mkdir(parents=True, exist_ok=True)
    done = resume_rows(path, items, spec.batch_size)
    todo = list(items[len(done):])
    stat = {"kind": kind, "n_done_before": len(done), "n_this_run": len(todo), "warmup_wall_s": None}
    if not todo:
        return stat
    root = Path(data_root)

    def load(item: Mapping[str, Any]) -> list[Any]:
        imgs = [load_pil(root / p) for p in item_images(item, kind)]
        for im in imgs:
            if list(im.size) != list(item["image_size"]):
                raise ReplayDataError(f"{item['id']}：图 {im.size} 与 image_size {item['image_size']} 不同")
        return imgs

    first = todo[0]
    stat["warmup_wall_s"] = adapter.warmup(load(first), instruction=first["prompt"], order=spec.order,
                                           batch_size=min(spec.batch_size, len(todo)))
    t_all = time.perf_counter()
    n_batches = 0
    for a in range(0, len(todo), spec.batch_size):
        batch = todo[a: a + spec.batch_size]
        t0 = time.perf_counter()
        inputs = [adapter.build_inputs(load(it), it["prompt"], spec.order) for it in batch]
        res = generate_rows(adapter, inputs, batch_size=spec.batch_size, enable_thinking=spec.enable_thinking,
                            max_new_tokens=spec.max_new_tokens, decoding=spec.decoding)
        if len(res) != len(batch):
            raise RuntimeError("generate_rows 返回的行数与输入不同")
        wall = time.perf_counter() - t0
        rows = [make_row(it, kind, spec, r) for it, r in zip(batch, res)]
        _append(out / TIMING, [_jsonl_line({"id": it["id"], "kind": kind, "run": run_k, "call_time_s": float(r.wall_time_s),
                                            "wall_time_s": wall / len(batch)}) for it, r in zip(batch, res)])
        _append(path, [_jsonl_line(r) for r in rows])
        n_batches += 1
        n_now = len(done) + a + len(batch)
        log(f"[回放 {kind}] {n_now}/{len(items)}（截断 {sum(r['truncated'] for r in rows)}/{len(rows)}；{wall:.1f} s/批）")
    stat.update(n_batches=n_batches, wall_s=round(time.perf_counter() - t_all, 3))
    return stat


# ---------------------------------------------------------------------------
# 运行身份、统计与 README
# ---------------------------------------------------------------------------


def run_identity(*, model: Mapping[str, Any], image_policy: str, dtype: str, manifest_sha256: str, specs: Mapping[str, GenSpec],
                 prompts: Any, tiny: bool) -> dict[str, Any]:
    """<out>/config.yaml：回放数据的运行身份（不含时间与机器相关的路径）。"""
    return {
        "experiment": "W2", "kind": KIND_DATA, "framework": FRAMEWORK, "model": dict(model), "tiny": bool(tiny), "image_policy": image_policy,
        "dtype": dtype, "manifest": f"{DATA_DIR}/{MANIFEST}", "manifest_sha256": manifest_sha256,
        "generation": {k: specs[k].as_dict() for k in KINDS},
        "prompts": {"e1_prompts": {"path": "configs/e1_prompts.json", "sha256": prompts.sha256},
                    "health_protocol": {"version": HP.PROTOCOL_VERSION, "order": HP.ORDER, "greedy": dict(HP.GREEDY_KWARGS)}},
        "files": {k: f"{DATA_DIR}/{FILES[k]}" for k in KINDS},
        "row_keys": list(ROW_KEYS),
        "truncation": "truncated = 没有生成停止 token（写满 max_new_tokens）；照实记，训练时不用",
    }


def prepare_root(out: Path, ident: Mapping[str, Any], files: Sequence[str | Path]) -> None:
    """<out>/config.yaml：没有就写（连同 meta.json）；已有必须身份相同，否则拒绝混写。"""
    p = out / CONFIG
    if p.exists():
        old = yaml.safe_load(p.read_text(encoding="utf-8"))
        a, b = _identity(old), _identity(ident)
        if a != b:
            diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
            raise ReplayDataError(f"{out}：已有的运行身份与这次的不同（{diff}），拒绝混写；换一个 --out")
    else:
        write_run_metadata(out, config=dict(ident), seed=BOOTSTRAP_SEED, data_files=list(files), repo_root=REPO_ROOT)
    (out / "logs").mkdir(parents=True, exist_ok=True)


def kind_metrics(rows: Sequence[Mapping[str, Any]], n_expected: int, kind: str) -> dict[str, Any]:
    n = len(rows)
    trunc = sum(1 for r in rows if r["truncated"])
    m: dict[str, Any] = {"n_expected": n_expected, "n": n, "complete": n == n_expected, "n_truncated": trunc,
                         "truncated_rate": (trunc / n) if n else None, "n_usable": n - trunc,
                         "n_generated_tokens": distribution([r["n_generated_tokens"] for r in rows]),
                         "n_generated_tokens_usable": distribution([r["n_generated_tokens"] for r in rows if not r["truncated"]]),
                         "n_prompt_tokens": distribution([r["n_prompt_tokens"] for r in rows])}
    if kind == COUNT:
        from src.parsing import parse_count

        parsed = [parse_count(r["answer"]) for r in rows if not r["truncated"]]
        vals = [p.value for p in parsed if p.failure is None]
        m["answers"] = {"n_parsed": len(vals), "n_unparsed": len(parsed) - len(vals), "values": distribution(vals),
                        "note": "诊断：解析成非负整数的比例与取值分布（src.parsing.parse_count）；不与任何真值比较，不进训练"}
    return m


def finalize(out: str | Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    """metrics.json（只由各行算）、compute.json（墙钟来自 logs/timing.jsonl）、README.md。"""
    out = Path(out)
    body: dict[str, Any] = {}
    comps: list[Compute] = []
    timing: dict[str, Mapping[str, Any]] = {}
    for r in read_jsonl_tolerant(out / TIMING, repair=False):
        timing[str(r["id"])] = r
    for kind in KINDS:
        rows = read_jsonl_tolerant(data_file(out, kind), repair=False)
        body[kind] = kind_metrics(rows, len(manifest["items"][kind]), kind)
        for r in rows:
            t = timing.get(str(r["id"]), {})
            comps.append(Compute.call(generated_tokens=r["n_generated_tokens"], prompt_tokens=r["n_prompt_tokens"],
                                      visual_tokens=r["n_visual_tokens"], image_pixels=r["n_image_pixels"],
                                      call_time_s=float(t.get("call_time_s", 0.0)), new_round=True)
                         .as_sample(wall_time_s=float(t.get("wall_time_s", 0.0))))
    m = {"experiment": "W2", "kind": KIND_DATA, "manifest_sha256": hashlib.sha256(manifest_path(out).read_bytes()).hexdigest(),
         **body, "complete": all(body[k]["complete"] for k in KINDS)}
    (out / METRICS).write_text(json.dumps(m, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    write_compute_json(out / COMPUTE, merge(comps))
    (out / README).write_text(readme_text(m, manifest), encoding="utf-8")
    return m


def _fmt(v: Any) -> str:
    return "null" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))


def readme_text(m: Mapping[str, Any], man: Mapping[str, Any]) -> str:
    g, c = man[GENERAL], man[COUNT]
    lines = [f"# W2-c 加回放的回放数据（DEV-0065）· complete = {m['complete']}", ""]
    for kind in KINDS:
        k = m[kind]
        lines.append(f"- {kind}：{k['n']}/{k['n_expected']} 条，截断 {k['n_truncated']}（{_fmt(k['truncated_rate'])}），可用 {k['n_usable']}；"
                     f"生成长度 p50 {_fmt((k['n_generated_tokens'] or {}).get('p50'))}")
    lines += [
        f"- 通用：{g['source']['repo']}@{g['source']['revision'][:12]}（{g['source']['license']}）；剔除 "
        f"{ {k: v['n'] for k, v in g['removed'].items()} }；与评测重叠的按参照集 {g['eval_overlap']['by_set']}；类别 {g['selected']['by_category']}",
        f"- 计数：W2-c 的 {c['w2c_images']['n_images']} 张图，(图, 类别) 对 {c['population']['n_pairs']}；与 CountBench 重合的图 "
        f"{c['countbench']['n_images_removed']}；选中 {c['selected']['n']} 条、{c['selected']['n_images']} 张图",
        "", "## 文件", "",
        "- replay/manifest.json：选择与去重的记录；replay/general.jsonl、replay/count.jsonl：每条一行（提示、图、回答、生成的 token、"
        "是否截断）；config.yaml：运行身份；metrics.json：由各行算；compute.json、logs/：记账与计时。训练：scripts/train_w2c.py "
        "--config configs/w2c_replay_train.json --replay-data <本目录>。", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 训练时读（src/w2/w2c_train.py 的回放流）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReplayData:
    """回放数据：rows = general.jsonl 再 count.jsonl 的行（文件行序）；pools[流] = 这条流的种类里没有截断的行（rows 的下标）。"""

    run_dir: Path
    rows: tuple[Mapping[str, Any], ...]
    pools: Mapping[str, tuple[int, ...]]
    kinds: Mapping[str, tuple[str, ...]]
    files: tuple[dict[str, Any], ...]
    manifest_sha256: str
    config: Mapping[str, Any]
    stats: Mapping[str, Any]

    def pool(self, stream: str) -> tuple[int, ...]:
        return self.pools[stream]

    def file_paths(self) -> list[Path]:
        return [self.run_dir / CONFIG, manifest_path(self.run_dir), *[data_file(self.run_dir, k) for k in KINDS]]

    def identity(self) -> dict[str, Any]:
        """训练运行身份里的回放数据一节：内容的指纹与统计；不含路径。"""
        from src.e3.train import identity_sha256

        c = self.config
        return {"format": "src/w2/w2c_replay.py（DEV-0065；scripts/build_w2c_replay.py 的输出）", "files": [dict(f) for f in self.files],
                "manifest_sha256": self.manifest_sha256, "data_identity_sha256": identity_sha256(c), "model": dict(c["model"]),
                "image_policy": c["image_policy"], "generation": dict(c["generation"]),
                "target": "base 贪心生成的 token 原样（含停止 token；截断的行不进池）",
                "pools": {s: {"kinds": list(self.kinds[s]), "n": len(p), "ids_sha256": ids_digest([str(self.rows[i]["id"]) for i in p])}
                          for s, p in self.pools.items()},
                **dict(self.stats)}


def row_problems(r: Mapping[str, Any], kind: str, gen: Mapping[str, Any]) -> list[str]:
    if tuple(r) != ROW_KEYS:
        return [f"键 {list(r)} ≠ {list(ROW_KEYS)}"]
    bad = []
    if r["kind"] != kind:
        bad.append(f"kind {r['kind']!r} ≠ {kind!r}")
    if r["enable_thinking"] is not False or gen["enable_thinking"] is not False:
        bad.append("回放的回答必须关思考生成")
    if r["order"] != gen["order"] or r["max_new_tokens"] != gen["max_new_tokens"]:
        bad.append(f"order / max_new_tokens（{r['order']}，{r['max_new_tokens']}）与运行身份的不同")
    ids = r["token_ids"]
    if not isinstance(ids, list) or not ids or any(isinstance(t, bool) or not isinstance(t, int) or t < 0 for t in ids):
        bad.append("token_ids 必须是非空的非负整数列表")
    elif r["n_generated_tokens"] != len(ids):
        bad.append("n_generated_tokens 与 token_ids 的长度不同")
    if r["truncated"] is not (not r["stopped"]):
        bad.append("truncated 必须等于 not stopped")
    if kind == COUNT:
        if not (isinstance(r["images"], list) and len(r["images"]) == 1 and isinstance(r["image_size"], list)):
            bad.append("计数的行要恰有一张图与 image_size")
    elif r["images"] != [] or r["image_size"] is not None:
        bad.append("通用指令的行没有图")
    if not isinstance(r["prompt"], str) or not r["prompt"].strip():
        bad.append("prompt 为空")
    return bad


def load_replay_data(run_dir: str | Path, *, streams: Mapping[str, Sequence[str]], model_key: str, revision: str, tiny: bool,
                     data_root: str | Path) -> ReplayData:
    """scripts/build_w2c_replay.py 的输出（--out）→ ReplayData。核对见模块说明；streams = {流名: 它的种类}。不合格抛 ReplayDataError。"""
    d = Path(run_dir)
    if not (d / CONFIG).is_file():
        raise ReplayDataError(f"{d} 里没有 {CONFIG}（--replay-data 要给 scripts/build_w2c_replay.py 的 --out）")
    cfg = yaml.safe_load((d / CONFIG).read_text(encoding="utf-8")) or {}
    if cfg.get("kind") != KIND_DATA:
        raise ReplayDataError(f"{d / CONFIG} 不是回放数据的运行（kind = {cfg.get('kind')!r}）")
    m = cfg.get("model") or {}
    if (m.get("key"), m.get("revision"), bool(m.get("tiny"))) != (model_key, revision, bool(tiny)):
        raise ReplayDataError(f"回放数据由 {m.get('key')}@{str(m.get('revision'))[:12]}（tiny {m.get('tiny')}）生成，训练的是 "
                              f"{model_key}@{revision[:12]}（tiny {tiny}）：自蒸馏的回答必须来自同一个 base")
    man, man_sha = load_manifest(d)
    if man_sha != cfg.get("manifest_sha256"):
        raise ReplayDataError(f"{manifest_path(d)} 的 sha256 与 {d / CONFIG} 记的不同")
    rows: list[Mapping[str, Any]] = []
    files: list[dict[str, Any]] = []
    by_kind: dict[str, list[int]] = {k: [] for k in KINDS}
    problems: list[str] = []
    root = Path(data_root)
    for kind in KINDS:
        p = data_file(d, kind)
        if not p.is_file():
            raise ReplayDataError(f"{p} 不在：回放数据还没生成完")
        raw = p.read_bytes()
        recs = [json.loads(x) for x in raw.decode("utf-8").splitlines() if x.strip()]
        files.append({"file": f"{DATA_DIR}/{FILES[kind]}", "n": len(recs), "sha256": hashlib.sha256(raw).hexdigest()})
        want = [str(it["id"]) for it in man["items"][kind]]
        if [str(r.get("id")) for r in recs] != want:
            raise ReplayDataError(f"{p}：行的 id 不是 manifest 的全部条目（{len(recs)}/{len(want)}）；回放数据没有跑完或不是这份 manifest")
        for r in recs:
            bad = row_problems(r, kind, cfg["generation"][kind])
            if not bad and kind == COUNT and not r["truncated"] and not (root / r["images"][0]).is_file():
                bad.append(f"图不在：{root / r['images'][0]}")
            if bad:
                problems.append(f"{r.get('id')}：{'；'.join(bad)}")
                continue
            by_kind[kind].append(len(rows))
            rows.append(r)
    if problems:
        raise ReplayDataError(f"回放数据有 {len(problems)} 行不合格：\n- " + "\n- ".join(problems[:10]))
    kinds: dict[str, tuple[str, ...]] = {}
    pools: dict[str, tuple[int, ...]] = {}
    used: set[str] = set()
    for s, ks in streams.items():
        ks = tuple(ks)
        if not ks or set(ks) - set(KINDS) or len(set(ks)) != len(ks) or used & set(ks):
            raise ReplayDataError(f"回放流 {s} 的种类 {list(ks)} 不合格（{list(KINDS)} 的非空子集，各流不重叠）")
        used |= set(ks)
        kinds[s] = ks
        pools[s] = tuple(i for k in KINDS if k in ks for i in by_kind[k] if not rows[i]["truncated"])
        if not pools[s]:
            raise ReplayDataError(f"回放流 {s} 的池是空的（{list(ks)} 的行全被截断？）")
    stats = {"by_kind": {k: {"n_rows": len(by_kind[k]), "n_truncated": sum(1 for i in by_kind[k] if rows[i]["truncated"]),
                             "n_usable": sum(1 for i in by_kind[k] if not rows[i]["truncated"]),
                             "mean_target_tokens": (sum(len(rows[i]["token_ids"]) for i in by_kind[k] if not rows[i]["truncated"]) /
                                                    max(1, sum(1 for i in by_kind[k] if not rows[i]["truncated"])))}
                         for k in KINDS}}
    return ReplayData(run_dir=d, rows=tuple(rows), pools=pools, kinds=kinds, files=tuple(files), manifest_sha256=man_sha, config=cfg,
                      stats=stats)


__all__ = [
    "COCO_PLURALS",
    "COUNT",
    "DATA_DIR",
    "DEFAULT_N",
    "DOLLY",
    "FILES",
    "GENERAL",
    "GenSpec",
    "KINDS",
    "MAX_NEW_TOKENS",
    "MAX_PROMPT_CHARS",
    "NGRAM",
    "OverlapIndex",
    "ROW_KEYS",
    "RefItem",
    "ReplayData",
    "ReplayDataError",
    "SALT",
    "TINY_N",
    "build_manifest",
    "check_manifest",
    "check_w2c_manifest",
    "count_object",
    "count_prompt",
    "countbench_hashes",
    "ensure_manifest",
    "fetch",
    "finalize",
    "gen_specs",
    "general_prompt",
    "health_refs",
    "load_manifest",
    "load_replay_data",
    "make_row",
    "norm_words",
    "prepare_root",
    "read_w2c_rows",
    "reference_sets",
    "resolve_raw",
    "run_identity",
    "run_kind",
    "select_count",
    "select_general",
    "source_refs",
]
