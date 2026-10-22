"""W2 的配置（configs/w2.json；DEV-0059；docs/w2_plan.md §3）：读取、结构校验、sha256；按配置给出挑选一步的协议与提示。

约定只写在配置里一处（提示原文、预算、解析规则、候选清单、渲染样式、随机挑选的 salt、批参数、划分规则），这里只读它；
tests/test_w2_config.py 钉住配置里的数字与这里的读法一致。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from src.w1.loop import TASKS, template_fields

REPO_ROOT = Path(__file__).resolve().parents[2]
W2_CONFIG_PATH = REPO_ROOT / "configs" / "w2.json"
W2_SCHEMA_VERSION = 1
MAX_THINKING_BUDGET = 2048  # D113：每轮思考至多 2,048 个 token
# 设置（docs/w2_plan.md §3）。按输出类型路由：要 mask 的任务（refseg）走 提出 → 候选 → 挑选；要框或数字的任务（refexp_box、count）
# 的 w2a 按 E1 单轮作答（= baseline_e1 的结果）
SETTINGS: tuple[str, ...] = ("w2a", "random_pick", "oracle_pick", "w1f", "ownbox", "baseline_e1")
SETTING_TASKS: Mapping[str, tuple[str, ...]] = {"w2a": TASKS, "random_pick": ("refseg",), "oracle_pick": ("refseg",),
                                                "w1f": ("refseg",), "ownbox": ("refseg",), "baseline_e1": TASKS}
CANDIDATE_SETTINGS: tuple[str, ...] = ("w2a", "random_pick", "oracle_pick", "w1f")  # 共用同一份候选（refseg）
OUTPUT_KIND: Mapping[str, str] = {"refseg": "mask", "refexp_box": "box", "count": "count"}
MASK_TASKS: tuple[str, ...] = tuple(t for t in TASKS if OUTPUT_KIND[t] == "mask")
PROPOSE_BOX = "box"
PROPOSE_BOX_POINTS = "box_points"
PROPOSE_MODES: tuple[str, ...] = (PROPOSE_BOX, PROPOSE_BOX_POINTS)
LAYOUTS: tuple[str, ...] = ("grid", "separate")
FIRST_GENERATOR = {"type": "grabcut_default"}  # docs/w2_plan.md §1：1 号候选 = W1-F
SELECT_PLACEHOLDERS: tuple[str, ...] = ("k", "expression")
BOX_POINTS_PLACEHOLDERS: tuple[str, ...] = ("expression", "max_points")


class W2ConfigError(ValueError):
    """configs/w2.json 不合格：列出全部问题。"""


def _posint(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and v >= 1


def _check_template(bad: list[str], where: str, text: Any, declared: Any, allowed: tuple[str, ...]) -> None:
    if not isinstance(text, str) or not text.strip():
        bad.append(f"{where} 必须是非空字符串")
        return
    try:
        used = template_fields(text)
    except ValueError as e:
        bad.append(f"{where} 的占位符写法不对：{e}")
        return
    if not isinstance(declared, list) or used != set(declared):
        bad.append(f"{where} 用的占位符 {sorted(used)} 与 placeholders 声明的 {declared} 不同")
    if used - set(allowed):
        bad.append(f"{where} 的占位符只能是 {allowed}，收到 {sorted(used - set(allowed))}")


def validate(cfg: Mapping[str, Any]) -> None:
    bad: list[str] = []
    if cfg.get("schema_version") != W2_SCHEMA_VERSION:
        bad.append(f"schema_version 应为 {W2_SCHEMA_VERSION}")
    if not isinstance(cfg.get("frozen"), bool):
        bad.append("frozen 必须是 bool")
    pr = cfg.get("propose") or {}
    if pr.get("mode") not in PROPOSE_MODES:
        bad.append(f"propose.mode 必须是 {PROPOSE_MODES} 之一")
    bp = pr.get("box_points") or {}
    _check_template(bad, "propose.box_points.prompt", bp.get("prompt"), bp.get("placeholders"), BOX_POINTS_PLACEHOLDERS)
    for k in ("max_points", "max_new_tokens"):
        if not _posint(bp.get(k)):
            bad.append(f"propose.box_points.{k} 必须是正整数")
    fams = bp.get("families")
    if not isinstance(fams, list) or not fams or any(not isinstance(f, str) for f in fams):
        bad.append("propose.box_points.families 必须是非空的家族名列表")
    gens = (cfg.get("candidates") or {}).get("generators")
    if not isinstance(gens, list) or len(gens) < 2 or any(not isinstance(g, dict) or not isinstance(g.get("type"), str) for g in gens):
        bad.append("candidates.generators 必须是至少两项的列表，每项是带 type 的对象")
    elif gens[0] != FIRST_GENERATOR:
        bad.append(f"candidates.generators 的第一项必须是 {FIRST_GENERATOR}（1 号候选 = W1-F，docs/w2_plan.md §1）")
    else:
        try:
            json.dumps(gens, sort_keys=True)
        except (TypeError, ValueError) as e:
            bad.append(f"candidates.generators 不能 JSON 序列化：{e}")
    rd = cfg.get("render") or {}
    if rd.get("layout") not in LAYOUTS:
        bad.append(f"render.layout 必须是 {LAYOUTS} 之一")
    if not isinstance(rd.get("style"), dict):
        bad.append("render.style 必须是对象")
    sl = cfg.get("select") or {}
    if sl.get("system") is not None and (not isinstance(sl.get("system"), str) or not sl["system"].strip()):
        bad.append("select.system 必须是 null 或非空字符串")
    _check_template(bad, "select.prompt", sl.get("prompt"), sl.get("placeholders"), SELECT_PLACEHOLDERS)
    if not isinstance(sl.get("thinking"), bool):
        bad.append("select.thinking 必须是 bool")
    if not _posint(sl.get("thinking_budget")) or sl.get("thinking_budget", 0) > MAX_THINKING_BUDGET:
        bad.append(f"select.thinking_budget 必须是 1..{MAX_THINKING_BUDGET} 的整数（D113：每轮思考至多 2,048 个 token）")
    if not _posint(sl.get("answer_budget")):
        bad.append("select.answer_budget 必须是正整数")
    ps = sl.get("parse") or {}
    nw = ps.get("none_words")
    if not isinstance(nw, list) or not nw or any(not isinstance(w, str) or not w.strip() or w != w.strip().lower() for w in nw):
        bad.append("select.parse.none_words 必须是非空的小写词列表（不带首尾空白）")
    sc = ps.get("strip_chars")
    if not isinstance(sc, str) or any(ch.isalnum() for ch in sc):
        bad.append("select.parse.strip_chars 必须是字符串，且不含字母与数字")
    rp = cfg.get("random_pick") or {}
    if not isinstance(rp.get("salt"), str) or not rp.get("salt"):
        bad.append("random_pick.salt 必须是非空字符串")
    bt = cfg.get("batching") or {}
    for k in ("batch_size", "chunk_size", "max_batch_tokens"):
        if not _posint(bt.get(k)):
            bad.append(f"batching.{k} 必须是正整数")
    sp = cfg.get("splits") or {}
    dt = sp.get("devtune800") or {}
    if dt.get("w1_splits") != ["devtune", "devcheck"]:
        bad.append("splits.devtune800.w1_splits 必须是 [devtune, devcheck]（W1 的 800 句原样）")
    d2 = sp.get("devcheck2") or {}
    want = {"population": "refcoco_unc_val", "dataset": "refcoco", "split_by": "unc", "split": "val",
            "exclude_images_of": "devtune800", "salt": "w2_dev_v1", "purpose": "w2_devcheck", "n": 500}
    got = {k: d2.get(k) for k in want}
    if got != want:
        bad.append(f"splits.devcheck2 必须是 {want}（D113；任务书 DEV-0059），收到 {got}")
    if not isinstance(sp.get("test"), list) or not sp["test"] or len(set(sp["test"])) != len(sp["test"]):
        bad.append("splits.test 必须是不重复的 E1 poly 子集名")
    if sp.get("count") != ["devtune_countbench", "test_countbench"]:
        bad.append("splits.count 必须是 [devtune_countbench, test_countbench]（同 W1）")
    if not _posint((sp.get("tiny") or {}).get("devcheck2_n")):
        bad.append("splits.tiny.devcheck2_n 必须是正整数")
    bl = cfg.get("baselines") or {}
    if (bl.get("e1_task") or {}) != {"refseg": "refexp_poly", "refexp_box": "refexp_box", "count": "count"}:
        bad.append("baselines.e1_task 必须是 refseg → refexp_poly、refexp_box → refexp_box、count → count（E1 零样本，同 W1）")
    tiny = cfg.get("tiny") or {}
    if not _posint(tiny.get("batch_size")) or not _posint(tiny.get("chunk_size")):
        bad.append("tiny.batch_size、tiny.chunk_size 必须是正整数")
    ts = tiny.get("select") or {}
    if set(ts) - {"thinking_budget", "answer_budget"} or any(not _posint(v) for v in ts.values()):
        bad.append("tiny.select 只能缩小 thinking_budget、answer_budget（正整数）")
    if bad:
        raise W2ConfigError("configs/w2.json 不合格：\n- " + "\n- ".join(bad))


@dataclass(frozen=True)
class SelectProtocol:
    """挑选一步的协议（configs/w2.json 的 select）。"""

    thinking: bool
    thinking_budget: int
    answer_budget: int
    none_words: tuple[str, ...]
    strip_chars: str

    def __post_init__(self) -> None:
        for name in ("thinking_budget", "answer_budget"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise ValueError(f"SelectProtocol.{name} 必须是正整数，收到 {v!r}")
        if self.thinking_budget > MAX_THINKING_BUDGET:
            raise ValueError(f"thinking_budget {self.thinking_budget} 超过 D113 的上限 {MAX_THINKING_BUDGET}")

    def as_dict(self) -> dict[str, Any]:
        return {"thinking": self.thinking, "thinking_budget": self.thinking_budget, "answer_budget": self.answer_budget,
                "none_words": list(self.none_words), "strip_chars": self.strip_chars}


@dataclass(frozen=True)
class W2Config:
    data: Mapping[str, Any]
    path: str
    sha256: str

    @property
    def frozen(self) -> bool:
        return bool(self.data["frozen"])

    @property
    def propose_mode(self) -> str:
        return self.data["propose"]["mode"]

    @property
    def box_points(self) -> Mapping[str, Any]:
        return self.data["propose"]["box_points"]

    def box_points_prompt(self, expression: str) -> str:
        bp = self.box_points
        return bp["prompt"].format(expression=expression, max_points=int(bp["max_points"])).strip()

    @property
    def generators(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(g) for g in self.data["candidates"]["generators"])

    @property
    def k(self) -> int:
        return len(self.data["candidates"]["generators"])

    def candidates_digest(self) -> str:
        """候选清单的 sha256（规范 JSON，键排序）：候选缓存的身份。"""
        return hashlib.sha256(json.dumps(list(self.generators), sort_keys=True, ensure_ascii=False,
                                         separators=(",", ":")).encode("utf-8")).hexdigest()

    @property
    def layout(self) -> str:
        return self.data["render"]["layout"]

    @property
    def style(self) -> dict[str, Any]:
        return json.loads(json.dumps(self.data["render"]["style"]))

    def select_protocol(self, *, tiny: bool = False) -> SelectProtocol:
        s = self.data["select"]
        p = SelectProtocol(thinking=bool(s["thinking"]), thinking_budget=int(s["thinking_budget"]),
                           answer_budget=int(s["answer_budget"]), none_words=tuple(s["parse"]["none_words"]),
                           strip_chars=s["parse"]["strip_chars"])
        if tiny:
            p = replace(p, **{k: int(v) for k, v in self.data["tiny"]["select"].items()})
        return p

    @property
    def select_system(self) -> str | None:
        return self.data["select"]["system"]

    def select_prompt(self, k: int, expression: str) -> str:
        return self.data["select"]["prompt"].format(k=int(k), expression=expression).strip()

    @property
    def random_salt(self) -> str:
        return self.data["random_pick"]["salt"]

    def batching(self, *, tiny: bool = False) -> dict[str, int]:
        b = self.data["batching"]
        out = {"batch_size": int(b["batch_size"]), "chunk_size": int(b["chunk_size"]), "max_batch_tokens": int(b["max_batch_tokens"])}
        if tiny:
            out.update(batch_size=int(self.data["tiny"]["batch_size"]), chunk_size=int(self.data["tiny"]["chunk_size"]))
        return out

    @property
    def splits(self) -> Mapping[str, Any]:
        return self.data["splits"]

    def e1_task(self, task: str) -> str:
        return self.data["baselines"]["e1_task"][task]

    def identity(self) -> dict[str, Any]:
        """运行身份里的配置一节：文件、sha256、是否冻结。"""
        p = Path(self.path).resolve()
        rel = str(p.relative_to(REPO_ROOT)) if p.is_relative_to(REPO_ROOT) else str(self.path)
        return {"config": rel, "config_sha256": self.sha256, "frozen": self.frozen}


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_w2_config(path: str | Path = W2_CONFIG_PATH) -> W2Config:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    validate(data)
    return W2Config(data=data, path=str(p), sha256=file_sha256(p))


__all__ = [
    "BOX_POINTS_PLACEHOLDERS",
    "CANDIDATE_SETTINGS",
    "FIRST_GENERATOR",
    "LAYOUTS",
    "MASK_TASKS",
    "MAX_THINKING_BUDGET",
    "OUTPUT_KIND",
    "PROPOSE_BOX",
    "PROPOSE_BOX_POINTS",
    "PROPOSE_MODES",
    "SELECT_PLACEHOLDERS",
    "SETTINGS",
    "SETTING_TASKS",
    "SelectProtocol",
    "W2Config",
    "W2ConfigError",
    "W2_CONFIG_PATH",
    "file_sha256",
    "load_w2_config",
    "validate",
]
