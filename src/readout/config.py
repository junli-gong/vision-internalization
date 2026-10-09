"""E3a 的配置（configs/e3a_train.json）：读取、结构核对、sha256、--tiny 的规模覆盖、生效参数的展开（resolve_run）与读出结构（readout_spec）。

配置是运行身份的一部分：它的 sha256 与本次实际生效的各项一起写进 config.yaml，续跑时必须完全相同。数字只写在配置里；硬约束的上限
（DECODER_MAX_PARAMS、HEAD_MAX_PARAMS）在 src/readout/model.py；与 E2、E3b 共用的取值（损失参数、LoRA、去重名单）由 tests 钉住。

**E3a+C3**（DEV-0044；运行变体 e3a_c3，配置的 mixed 节）：读出、LoRA、查询、优化器都同 full（readout_variant），另有两条样本流：
C3 的文本流（mixed.text_stream.config 指向的 E3b 配置，src/e3 原样读取）与 E3a full 的稠密流。mixed_spec 一次核对条数、步数的来历、
1 : 1 的权重与两份配置的相容（同一个训练池、同一组种子、同一个思考开关、C3 的去重名单是 E3a 的子集）。
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.adapters import registry
from src.adapters.training import join_class_names
from src.e3 import config as E3C
from src.readout.model import LINEAR_HEAD_TASKS, TASKS, VARIANTS, ReadoutSpec

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "e3a_train.json"
SCHEMA = 1
OPTIM_GROUPS: tuple[str, ...] = ("lora", "query", "mix", "readout")
_TINY_KEYS = ("steps", "micro_batch_size", "grad_accum_steps", "ckpt_every")
LOSS_KINDS = {"refseg": "bce_dice", "depth": "silog", "semseg": "cross_entropy", "normals": "cosine"}

# E3a+C3（DEV-0044）：运行变体名（输出目录、config.yaml 的 variant）；它的读出结构、LoRA、查询与优化器同 MIXED_BASE（readout.json 的
# 结构变体也是它，所以 eval_e3a --variant full 直接能读）。两条样本流：text = C3 的文本流，dense = E3a full 的稠密流
E3A_C3 = "e3a_c3"
MIXED_BASE = "full"
MIXED_STREAMS: tuple[str, ...] = ("text", "dense")
TEXT_SUPERVISION = "text"  # 文本流在 src/e3 里的监督名（C3）
RUN_VARIANTS: tuple[str, ...] = VARIANTS + (E3A_C3,)


class ConfigError(ValueError):
    pass


def readout_variant(variant: str) -> str:
    """运行变体 → 读出结构的变体（model.VARIANTS 之一）：e3a_c3 → full，其余原样。"""
    if variant == E3A_C3:
        return MIXED_BASE
    if variant not in VARIANTS:
        raise ConfigError(f"variant 必须是 {RUN_VARIANTS} 之一，收到 {variant!r}")
    return variant


def groups_for(variant: str) -> tuple[str, ...]:
    """各变体的参数组（没有的组不建）：full 四组；c1 没有 lora；c2 只有 readout；e3a_c3 同 full。"""
    variant = readout_variant(variant)
    if variant == "full":
        return OPTIM_GROUPS
    if variant == "c1":
        return ("query", "mix", "readout")
    return ("readout",)


@dataclass(frozen=True)
class E3aConfig:
    path: str
    sha256: str
    doc: Mapping[str, Any]

    def task(self, name: str) -> dict[str, Any]:
        if name not in self.doc["tasks"]:
            raise ConfigError(f"未知任务 {name!r}，配置里有 {sorted(self.doc['tasks'])}")
        return dict(self.doc["tasks"][name])

    @property
    def mix_weights(self) -> dict[str, int]:
        return {k: int(v) for k, v in self.doc["mix"]["weights"].items()}

    @property
    def tasks(self) -> tuple[str, ...]:
        """参与混合的任务（权重 > 0），按 TASKS 的顺序。"""
        w = self.mix_weights
        return tuple(t for t in TASKS if w.get(t, 0) > 0)

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(int(s) for s in self.doc["seeds"])

    @property
    def class_names(self) -> list[str]:
        return [str(n) for n in self.doc["tasks"]["semseg"]["class_names"]]

    @property
    def enable_thinking(self) -> bool:
        return bool(self.doc["enable_thinking"])

    @property
    def query_n(self) -> int:
        return int(self.doc["query"]["n"])

    @property
    def query_grid(self) -> int:
        return int(self.doc["query"]["grid"])

    def position_mode(self, family: str) -> str:
        modes = self.doc["query"]["position_mode"]
        if family not in modes:
            raise ConfigError(f"query.position_mode 没有家族 {family!r}")
        return str(modes[family])

    def instruction(self, task: str, expression: str | None = None) -> str:
        """这个任务送进 build_inputs 的指令：指代分割把指代句（去首尾空白）填进模板；语义分割把类别名按 CLASS_NAME_SEPARATOR 拼进
        模板（class_name_spans 的拼法）；深度与法向是固定的一句。"""
        tmpl = str(self.doc["tasks"][task]["instruction_template"])
        if task == "refseg":
            if expression is None or not str(expression).strip():
                raise ConfigError("refseg 要给非空的指代句")
            return tmpl.format(expression=str(expression).strip())
        if expression is not None:
            raise ConfigError(f"{task} 的指令不接受指代句")
        if task == "semseg":
            return tmpl.format(classes=join_class_names(self.class_names))
        return tmpl

    def order(self, task: str) -> str:
        return str(self.doc["tasks"][task]["order"])


def load_config(path: str | Path = CONFIG_PATH) -> E3aConfig:
    p = Path(path)
    raw = p.read_bytes()
    doc = json.loads(raw)
    if doc.get("schema") != SCHEMA:
        raise ConfigError(f"{p}：schema {doc.get('schema')!r} 不是 {SCHEMA}")
    if set(doc["tasks"]) != set(TASKS):
        raise ConfigError(f"tasks 必须恰好是 {TASKS}，收到 {sorted(doc['tasks'])}")
    for t in TASKS:
        tc = doc["tasks"][t]
        if tc["loss"]["kind"] != LOSS_KINDS[t]:
            raise ConfigError(f"tasks.{t}.loss.kind 必须是 {LOSS_KINDS[t]}（同 E2），收到 {tc['loss']['kind']!r}")
        if tc["order"] != "instruction_first":
            raise ConfigError(f"tasks.{t}.order 必须是 instruction_first（简报 §8.4：指令在前）")
    names = doc["tasks"]["semseg"]["class_names"]
    join_class_names(names)  # 非空、不含分隔符、不重复
    from src.constants import ADE_NUM_CLASSES

    if len(names) != ADE_NUM_CLASSES:
        raise ConfigError(f"class_names 有 {len(names)} 个，ADE_NUM_CLASSES = {ADE_NUM_CLASSES}")
    w = doc["mix"]["weights"]
    if set(w) - set(TASKS) or any(isinstance(v, bool) or int(v) != v or int(v) < 0 for v in w.values()) \
            or sum(int(v) for v in w.values()) < 1:
        raise ConfigError(f"mix.weights 必须是 {TASKS} 的非负整数权重、至少一个为正，收到 {w}")
    tw = doc["loss"]["task_weights"]
    if set(tw) != set(TASKS) or any(not float(v) >= 0 for v in tw.values()):
        raise ConfigError(f"loss.task_weights 必须给全 {TASKS}、非负，收到 {tw}")
    if doc["loss"]["reduction"] != "sample_mean":
        raise ConfigError("loss.reduction 只实现了 sample_mean")
    q = doc["query"]
    if int(q["grid"]) ** 2 != int(q["n"]):
        raise ConfigError(f"query.n {q['n']} 必须是 query.grid {q['grid']} 的平方（解码器按 grid × grid 排列）")
    for fam in registry.FAMILIES:
        if q["position_mode"].get(fam) not in ("grid2d", "seq1d"):
            raise ConfigError(f"query.position_mode.{fam} 必须是 grid2d 或 seq1d")
    if set(doc["heads"]["channels"]) != set(LINEAR_HEAD_TASKS):
        raise ConfigError(f"heads.channels 必须恰好给 {LINEAR_HEAD_TASKS}")
    if set(doc["optim"]["groups"]) != set(OPTIM_GROUPS):
        raise ConfigError(f"optim.groups 必须恰好是 {OPTIM_GROUPS}")
    if doc["optim"]["optimizer"] != "adamw" or doc["optim"]["schedule"] != "warmup_cosine":
        raise ConfigError("只实现了 adamw + warmup_cosine")
    for g in ("lora", "query", "mix"):
        if float(doc["optim"]["groups"][g]["weight_decay"]) != 0.0:
            raise ConfigError(f"optim.groups.{g} 必须 weight_decay = 0（LoRA 同 E3b；查询嵌入同 D45-3；混合是标量）")
    if len(set(int(s) for s in doc["seeds"])) != len(doc["seeds"]):
        raise ConfigError("seeds 有重复")
    req = doc["dedupe"]["required_eval_sets"]
    if set(req) != set(TASKS):
        raise ConfigError(f"dedupe.required_eval_sets 必须给全 {TASKS}")
    up = doc["eval"]["upsample"]
    if up["primary"] not in up["methods"] or up["methods"][0] != up["primary"] or set(up["methods"]) - {"guided", "bilinear"}:
        raise ConfigError(f"eval.upsample：primary 必须是 methods 的第一个，methods 只能取 guided / bilinear，收到 {up}")
    if set(doc["eval"]["leak_checks"]["no_instruction"]["tasks"]) - set(TASKS):
        raise ConfigError("eval.leak_checks.no_instruction.tasks 有未知任务")
    for t in TASKS:
        if t not in doc["eval"]["sets"]:
            raise ConfigError(f"eval.sets 缺 {t}")
    readout_spec(doc, variant="full", hidden_size=64, n_layers=4, tiny=False)  # 结构能展开
    readout_spec(doc, variant="full", hidden_size=64, n_layers=4, tiny=True)
    if "mixed" in doc:  # E3a+C3（DEV-0044）：两条流的条数、权重、步数的来历与两份配置的相容一次核对
        for tiny in (False, True):
            mixed_spec(doc, tiny=tiny)
    rel = str(p.relative_to(REPO_ROOT)) if p.resolve().is_relative_to(REPO_ROOT) else str(p)
    return E3aConfig(path=rel, sha256=hashlib.sha256(raw).hexdigest(), doc=doc)


# ---------------------------------------------------------------------------
# E3a+C3 的两条样本流（配置的 mixed 节）
# ---------------------------------------------------------------------------


def text_stream_config(doc: Mapping[str, Any]) -> E3C.E3Config:
    """mixed.text_stream.config 指向的 E3b 配置（C3 的样本流、样本构造与损失都按它；src/e3/config.load_config 原样读取）。"""
    ts = doc["mixed"]["text_stream"]
    try:
        return E3C.load_config(REPO_ROOT / str(ts["config"]))
    except (OSError, ValueError, KeyError) as e:
        raise ConfigError(f"mixed.text_stream.config {ts.get('config')!r} 读不了：{e!r}") from None


def micro_batch_groups(tasks: Sequence[str], micro: int) -> list[list[int]]:
    """E3a 的微批规则（Trainer.micro_batches 用的就是它）：一步的样本按任务名分成语义分割与其余两部分（语义分割在前；类别名区间
    只在语义分割的行上取），各自按原顺序每 micro 条切一批。返回每个微批里样本的下标。"""
    m = int(micro)
    out: list[list[int]] = []
    for part in ([i for i, t in enumerate(tasks) if t == "semseg"], [i for i, t in enumerate(tasks) if t != "semseg"]):
        out += [part[s: s + m] for s in range(0, len(part), m)]
    return out


def _check_mixed_compat(doc: Mapping[str, Any], tc: E3C.E3Config) -> None:
    """E3a 与 C3 两份配置的相容（文本流的样本要与 C3 逐个相同、又用 E3a 已加载的训练池）。"""
    ts = doc["mixed"]["text_stream"]
    if ts.get("supervision") != TEXT_SUPERVISION or ts.get("variant") != E3C.VARIANT_OF[TEXT_SUPERVISION]:
        raise ConfigError(f"mixed.text_stream 必须是 supervision {TEXT_SUPERVISION!r}、variant "
                          f"{E3C.VARIANT_OF[TEXT_SUPERVISION]!r}（C3），收到 {dict(ts)}")
    w = doc["mix"]["weights"]
    missing = [t for t in tc.tasks if int(w.get(t, 0)) < 1]
    if missing:
        raise ConfigError(f"C3 的任务 {missing} 不在 E3a 的 mix.weights 里（文本流用 E3a 加载的训练池）")
    if list(tc.task("refseg")["datasets"]) != list(doc["tasks"]["refseg"]["datasets"]):
        raise ConfigError(f"两份配置的 tasks.refseg.datasets 不同（C3 {tc.task('refseg')['datasets']}，E3a "
                          f"{doc['tasks']['refseg']['datasets']}）：文本流与稠密流必须是同一个训练池")
    for t in tc.tasks:
        extra = sorted(set(tc.doc["dedupe"]["required_eval_sets"].get(t, [])) - set(doc["dedupe"]["required_eval_sets"][t]))
        if extra:
            raise ConfigError(f"C3 的 {t} 去重名单多出 {extra}：E3a 的开跑核对必须覆盖 C3 的")
    if not set(int(s) for s in doc["seeds"]) <= set(tc.seeds):
        raise ConfigError(f"E3a 的 seeds {doc['seeds']} 不全在 C3 的 seeds {list(tc.seeds)} 里（两条流用同一个种子）")
    if bool(tc.enable_thinking) != bool(doc["enable_thinking"]):
        raise ConfigError("两份配置的 enable_thinking 不同")


def mixed_spec(doc: Mapping[str, Any], *, tiny: bool) -> dict[str, Any]:
    """配置的 mixed 节（E3a+C3）→ 本次生效的两条流（规则见配置的 mixed.basis）。核对：
    - per_step 恰好是 text、dense 两项，正整数；两项之和 = 每步条数（batch.micro_batch_size × batch.grad_accum_steps，tiny 取 tiny 节），
      且各自是微批大小的整数倍；
    - steps × per_step.dense = E3a 的 steps × 每步条数，steps × per_step.text = C3 的 steps × 每步条数（C3 = text_stream.config 按
      src/e3/config.resolve_run(supervision = text) 展开；全程的稠密样本恰好是 E3a full 的、文本样本恰好是 C3 的，一条不多一条不少）；
    - loss_weights 恰好是 text 1、dense 1（D83-2：1 : 1 事先固定，不调）；order 是两条流的一个排列；
    - 两份配置相容（_check_mixed_compat）。"""
    m = doc["mixed"]
    sec = m["tiny"] if tiny else m
    t = doc["tiny"]
    micro = int(t["micro_batch_size"] if tiny else doc["batch"]["micro_batch_size"])
    batch = micro * int(t["grad_accum_steps"] if tiny else doc["batch"]["grad_accum_steps"])
    ref_steps = int(t["steps"] if tiny else doc["steps"])
    where = "mixed.tiny" if tiny else "mixed"
    per = sec["per_step"]
    if set(per) != set(MIXED_STREAMS) or any(isinstance(v, bool) or int(v) != v or int(v) < 1 for v in per.values()):
        raise ConfigError(f"{where}.per_step 必须恰好是 {list(MIXED_STREAMS)} 的正整数条数，收到 {per}")
    per = {s: int(per[s]) for s in MIXED_STREAMS}
    if sum(per.values()) != batch or any(v % micro for v in per.values()):
        raise ConfigError(f"{where}.per_step {per}：两项之和必须是每步条数 {batch}，且各自是微批大小 {micro} 的整数倍")
    tc = text_stream_config(doc)
    _check_mixed_compat(doc, tc)
    c3 = E3C.resolve_run(tc, supervision=TEXT_SUPERVISION, tiny=tiny)
    steps = sec["steps"]
    want = {"text": int(c3["steps"]) * int(c3["batch_size"]), "dense": ref_steps * batch}
    if isinstance(steps, bool) or int(steps) != steps or any(int(steps) * per[s] != want[s] for s in MIXED_STREAMS):
        raise ConfigError(f"{where}.steps = {steps}：steps × per_step.<流> 必须等于 C3、E3a full 各自的样本数 {want}"
                          "（逐样本相同，D83-2）")
    w = m["loss_weights"]
    if set(w) != set(MIXED_STREAMS) or any(isinstance(v, bool) or float(v) != 1.0 for v in w.values()):
        raise ConfigError(f"mixed.loss_weights 必须恰好是 text 1、dense 1（D83-2：两种损失 1 : 1，事先固定，不调），收到 {w}")
    order = [str(x) for x in m["order"]]
    if sorted(order) != sorted(MIXED_STREAMS):
        raise ConfigError(f"mixed.order 必须是 {list(MIXED_STREAMS)} 的一个排列，收到 {order}")
    steps = int(steps)
    variants = {"text": E3C.VARIANT_OF[TEXT_SUPERVISION], "dense": MIXED_BASE}
    labels = {"text": f"C3（变体 {variants['text']}）", "dense": f"E3a {variants['dense']} "}
    return {"version": str(m["version"]), "per_step": per, "order": order,
            "loss_weights": {s: float(w[s]) for s in MIXED_STREAMS}, "steps": steps,
            "micro_batches": {"text": per["text"] // micro, "dense": _dense_composition(doc, per["dense"], micro)},
            "reference": {
                "text": {"variant": variants["text"], "config": {"path": tc.path, "sha256": tc.sha256, "version": tc.doc["version"]},
                         "steps": int(c3["steps"]), "batch_size": int(c3["batch_size"]),
                         "micro_batch_size": int(c3["micro_batch_size"]), "samples": want["text"]},
                "dense": {"variant": variants["dense"], "steps": ref_steps, "batch_size": batch, "micro_batch_size": micro,
                          "samples": want["dense"]}},
            "schedule_length_ratio": {"text": steps / int(c3["steps"]), "dense": steps / ref_steps},
            "streams": {s: f"第 t 步取 {labels[s]}同一种子的样本流第 {per[s]}t … {per[s]}t + {per[s] - 1} 个槽位"
                        for s in MIXED_STREAMS}}


def _dense_composition(doc: Mapping[str, Any], per_dense: int, micro: int) -> dict[str, Any]:
    """稠密流每步的微批组成（E3a 的规则，micro_batch_groups）。混合循环的长度整除每步条数时每一步都相同，记下各微批的任务；
    否则随步变化，只记规则。"""
    from src.readout.data import mix_cycle  # 延迟 import：data 依赖本模块

    cycle = mix_cycle({k: int(v) for k, v in doc["mix"]["weights"].items()})
    rule = "E3a 的规则：语义分割的样本单独成批，其余按原顺序每 micro_batch_size 条一批（语义分割在前）"
    if per_dense % len(cycle):
        return {"rule": rule, "per_step": None}
    tasks = [cycle[i % len(cycle)] for i in range(per_dense)]
    return {"rule": rule, "per_step": [[tasks[i] for i in g] for g in micro_batch_groups(tasks, micro)]}


def decoder_block(doc: Mapping[str, Any], *, tiny: bool) -> dict[str, Any]:
    """生效的解码器结构：正式配置的 decoder 节，--tiny 时按 tiny.decoder 覆盖通道与残差块数（嵌入维度与输出分辨率不变）。"""
    d = {k: copy.deepcopy(v) for k, v in doc["decoder"].items() if k != "basis"}
    if tiny:
        for k, v in doc["tiny"]["decoder"].items():
            if k not in ("widths", "res_blocks"):
                raise ConfigError(f"tiny.decoder 只能覆盖 widths、res_blocks，收到 {k!r}")
            d[k] = copy.deepcopy(v)
    return d


def readout_spec(doc: Mapping[str, Any], *, variant: str, hidden_size: int, n_layers: int, tiny: bool) -> ReadoutSpec:
    """读出结构。n_layers = 适配器的 num_layers()：full / c1 的来源是层号 0..n_layers（共 n_layers + 1 个），c2 只有 1 个。"""
    if variant not in VARIANTS:
        raise ConfigError(f"variant 必须是 {VARIANTS} 之一，收到 {variant!r}")
    d = decoder_block(doc, tiny=tiny)
    return ReadoutSpec(variant=variant, hidden_size=int(hidden_size), n_sources=1 if variant == "c2" else int(n_layers) + 1,
                       grid=int(doc["query"]["grid"]), widths=tuple(d["widths"]), res_blocks=tuple(d["res_blocks"]),
                       embed_dim=int(d["embed_dim"]), out_size=int(d["out_size"]),
                       head_channels={k: int(v) for k, v in doc["heads"]["channels"].items()},
                       norm_eps=float(doc["layer_mix"]["norm_eps"]))


def resolve_run(cfg: E3aConfig, *, variant: str, tiny: bool) -> dict[str, Any]:
    """本次运行实际生效的训练参数（写进 config.yaml）：--tiny 时按 cfg.tiny 覆盖规模参数。
    变体 e3a_c3（DEV-0044）：其余同 full，steps = mixed.steps（--tiny 取 mixed.tiny.steps），warmup_steps 按它算；另记 mixed 一节
    （mixed_spec）。full、c1、c2 的结果与原来逐项相同（不加键）。"""
    d = cfg.doc
    rv = readout_variant(variant)
    groups = groups_for(variant)
    out = {
        "variant": variant,
        "steps": int(d["steps"]),
        "micro_batch_size": int(d["batch"]["micro_batch_size"]),
        "grad_accum_steps": int(d["batch"]["grad_accum_steps"]),
        "ckpt_every": int(d["ckpt_every"]),
        "lora": {k: d["lora"][k] for k in ("r", "alpha", "dropout")} if rv == "full" else None,
        "optim": {"optimizer": d["optim"]["optimizer"], "betas": list(d["optim"]["betas"]), "eps": float(d["optim"]["eps"]),
                  "groups": {g: dict(d["optim"]["groups"][g]) for g in groups},
                  "max_grad_norm": d["optim"]["max_grad_norm"], "schedule": d["optim"]["schedule"],
                  "warmup_ratio": float(d["optim"]["warmup_ratio"]), "min_lr_ratio": float(d["optim"]["min_lr_ratio"])},
        "loss": {"task_weights": {t: float(v) for t, v in d["loss"]["task_weights"].items()},
                 "reduction": d["loss"]["reduction"]},
        "gradient_checkpointing": bool(d["model"]["gradient_checkpointing"]) and rv != "c2",
        "enable_thinking": cfg.enable_thinking,
        "mix": cfg.mix_weights,
        "data_workers": int(d["data"]["workers"]),
        "query": None if rv == "c2" else {"n": cfg.query_n, "grid": cfg.query_grid,
                                          "init_noise_ratio": float(d["query"]["init_noise_ratio"])},
        "decoder": decoder_block(d, tiny=tiny),
    }
    if tiny:
        t = d["tiny"]
        for k in _TINY_KEYS:
            out[k] = int(t[k])
    if variant == E3A_C3:
        if "mixed" not in d:
            raise ConfigError("配置里没有 mixed 节")
        ms = mixed_spec(d, tiny=tiny)
        out["steps"] = ms["steps"]
        out["mixed"] = ms
    for k in ("steps", "micro_batch_size", "grad_accum_steps", "ckpt_every"):
        if out[k] < 1:
            raise ConfigError(f"{k} 必须 ≥ 1，收到 {out[k]}")
    out["batch_size"] = out["micro_batch_size"] * out["grad_accum_steps"]
    out["warmup_steps"] = int(d["tiny"]["warmup_steps"]) if tiny else int(math.ceil(float(d["optim"]["warmup_ratio"]) * out["steps"]))
    if not 0 <= out["warmup_steps"] <= out["steps"]:
        raise ConfigError(f"warmup_steps {out['warmup_steps']} 不在 [0, steps]")
    return out


__all__ = ["CONFIG_PATH", "ConfigError", "E3A_C3", "E3aConfig", "MIXED_BASE", "MIXED_STREAMS", "OPTIM_GROUPS", "RUN_VARIANTS",
           "TEXT_SUPERVISION", "decoder_block", "groups_for", "load_config", "micro_batch_groups", "mixed_spec", "readout_spec",
           "readout_variant", "resolve_run", "text_stream_config"]
