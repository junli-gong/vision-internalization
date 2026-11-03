"""W2-c 的训练（DEV-0061；D113 第二节第 4 条；接口 docs/w2_plan.md §5）：只有文本的 LoRA 监督——"提出的框"与"挑选的编号"。

**为什么另写一个模块、不在 E3 的训练框架里加一种监督**（侵入小者优先）：src/e3/train.run_training 的数据一侧（E3 的训练池与去重核对、
mix 循环的样本日程、ExampleBuilder、运行身份、README、accounting 的 by_supervision）处处按 E3 的任务（refseg、depth）与监督名写死，
加一种监督要改其中十来处，E3 的运行身份与续跑都要跟着证明不变。这里只 import 它的机制、不改它的一行：
- 训练器：src/e3/train.Trainer 的子类 W2CTrainer。搭建（torch.manual_seed → inject_lora(r, alpha, dropout) → 梯度检查点 → AdamW +
  warmup / 余弦）、运行期审计（只有 LoRA 求梯度、LoRA 是 fp32）、续跑（load_trained(merge=False) → 优化器、调度器、随机数状态）都是
  父类的原样；只覆盖 optimize（两条文本样本流各自归一，见下）。
- 检查点：adapter.save_trained（LoRA + tokens.json + manifest，没有新增 token 行），与 C3 同格式：scripts/eval_e3b.py、
  scripts/run_e1.py --checkpoint、scripts/health_check.py --checkpoint、scripts/run_w2.py --setting w2c 都能直接读。
- 输出与记账：叶目录 <out>/w2c/<模型>/seed<k>/ 的文件与 E3 相同（config.yaml、meta.json、train_log.jsonl、train_summary.json、
  accounting.json、compute.json、README.md、logs/run_<k>.json、logs/timing_run_<k>.jsonl、ckpt/latest、ckpt/final/model），合计用
  src/e3/train.accounting_doc；续跑、分段运行（stop_after_steps）、检查点的写法与 run_training 逐条对应。

**数据**（load_w2c_data；DEV-0060 产出，格式 docs/w2_plan.md §5）：<w2c_data>/*.jsonl 按文件名升序拼接，每行一句。核对（不过就拒绝训练）：
- 只读训练集：split 必须是 train、source 必须在配置 data.sources 里（RefCOCO、RefCOCO+、RefCOCOg）；id 不重复；
- 与 W2 配置对得上：w2_config_sha256 = 本次用的 configs/w2.json 的 sha256，spec_sha256 = 它的候选清单 sha256（W2Config.candidates_digest），
  候选恰 K 个、编号 1..K、RLE 的尺寸等于原图；
- 结构：proposal.failure 不为空 ⟺ candidates 为 null（此时 label 也为 null）；label 是 1..K 或 null；label 不为 null 时必须是候选 iou
  最高者里编号最小的（§5 的标签规则，按记下的 iou 逐个比较）。
提出的池 = 全部行；挑选的池 = candidates 与 label 都不为 null 的行。

**样本日程**（无状态，同 E3 的 Schedule：一切由 (种子, k) 算出，续跑时按步号重算）：第 t 步 = 提出流第 P·t … P·t + P − 1 个槽位 +
挑选流第 S·t … S·t + S − 1 个槽位（P、S = 配置 streams.per_step；流的先后 = streams.order）。流 s 的第 j 个槽位取第 j // N_s 个 epoch
的置换 np.random.default_rng([种子, 流号, epoch]).permutation(N_s) 的第 j mod N_s 个（与 E2 的 batch_indices、E3 的 pool_position 同一规则；
流号 propose = 0、select = 1）。

**样本**（W2CBuilder）：
- 提出：原图（src/e3/data.load_rgb）+ E1 refexp_box 本家族的提示原文（src/e3/c3.load_box_template、render_e1_template(模板, 指代句.strip())）
  → 目标 src/e3/c3.box_target(adapter.format_boxes, gt_box, W, H, 指代句.strip()) + 回合结束 token：与 C3 的框监督逐字相同。
- 挑选：原图（PIL，Image.open(...).convert("RGB")，同 src/e1/runner.load_images）→ 候选 = src/w2/loop.selection_candidates（RLE 解码；
  1 号 ok = False 时画空叠加，W2 的裁定 A）→ render_selection(原图, 候选, layout, style)（冻结的 W2 配置）；提示 = W2Config.select_prompt(K,
  指代句)；目标 = str(label) + 回合结束 token。这三步与 W2-a 评测时 src/w2/runner.run_candidate_setting 的调用逐个相同，所以挑选图逐位相同
  （tests/test_w2c_train.py 钉住）。
- 顺序（build_inputs 的 order）：配置 propose.order、select.order；select.order 必须等于 E1 的 image_order（评测时挑选用的就是它）。

**损失**（W2CTrainer.optimize）：一步之内按流切微批（每个微批只含一种样本，流的先后 = streams.order），文本损失 chunked_lm_loss（全词表，
分块）；每种样本各自除以这一步里它自己的监督 token 总数，再乘 loss_weights（1 : 1）相加：loss = L_propose + L_select。train_log 的 loss 是
这个和，loss_by_stream 是两项各自的逐 token 平均；select_number_acc 是挑选样本第一个目标位置上、只在编号 1..K 的 token 之间比较时
argmax 等于标签的比例（教师强制的诊断，不进损失；编号不都是单 token 时为 null）。

**回放（DEV-0065，D125-2 (b)；可选，配置 configs/w2c_replay_train.json 的 replay 一节）**：配置没有 replay 一节时，以下都不发生，
run_cfg、运行身份、train_log、README、检查点与以前逐字节相同（tests/test_w2c_replay.py 与 7f53f240 的原文件对照钉住）。有这一节时：
- 数据：src/w2/w2c_replay.load_replay_data（scripts/build_w2c_replay.py 的输出；回答由同一个 base 贪心、关思考生成，截断的行不进池）；
- 日程：每条回放流 r 每步 per_step 条，排在提出、挑选之后（ReplaySchedule；规则同 W2CSchedule，流号 = 2、3……）；
- 样本（ReplayBuilder）：提示、图、顺序取行原样，目标 = base 生成的 token 原样（含停止 token）；
- 损失：每条回放流也各自除以这一步里自己的监督 token 数，再乘 loss_weight，与提出、挑选两项相加；回放流的微批同样只含这一条流的样本，
  per_step 不是微批的整数倍时最后一个微批不满。configs/w2c_replay_train.json（管理员裁定，DEV-0065 待裁第 1 条）：两条回放流
  replay_general（通用 3 条）、replay_count（计数 2 条），每步 8 + 8 + 3 + 2，权重 1 : 1 : 0.5 : 0.5——回放在损失里的总权重仍是 1/3，
  计数与通用各占一半（合成一条流按 token 平均时，约 2 个 token 的计数回答只占回放损失约 1%）。
- 记录：train_log 的 loss_by_stream、by_stream……多出回放流；运行身份多 replay 一节（回放数据的指纹、池、流）；检查点格式不变。

**X-2 的两条流（DEV-0069，探索 X-2；可选，配置 configs/x2_train.json 的 x2_data、spline、correct 三节）**：配置没有 spline、correct 两节时，
以下都不发生，run_cfg、运行身份、train_log、README、检查点与以前逐字节相同（tests/test_x2_train.py 与 7f53f240 的原文件对照钉住）。有时：
- 数据：src/explore/x2_data.load_x2_data（scripts/build_x2_data.py 的输出；目标只来自真值；参数必须等于配置的、来源必须是同一份 w2c_data）；
- 日程：spline、correct 每步各 per_step 条（X2Schedule；规则同 W2CSchedule，流号 spline = 4、correct = 5），排在提出、挑选之后、回放之前；
- 样本（X2Builder）：描样条 = 原图（load_pil）+ x2_prompts.spline_prompt(指代句, 提出的框的网格整数, n) → 目标文本（n 个控制点的 JSON）；
  指修正框 = x2_render.render_overlay(原图, 起点 mask, 提出的框, 冻结的 W2 render.style)（训练时现画）+ x2_prompts.correct_prompt(…, m) →
  目标文本（至多 m 个带符号的框，或 []）；目标都加回合结束 token，全部监督；关思考；order = E1 的 image_order；
- 损失：两条流也各自除以这一步里自己的监督 token 数，再乘 loss_weight（1、1），与其余各项相加；微批同样只含一种样本，per_step 不是微批的
  整数倍时最后一个微批不满（同回放流）；
- 记录：loss_by_stream、by_stream……多出 spline、correct；运行身份多 x2 一节（数据的指纹、参数、提示原文、流）；检查点格式不变（variant 仍是
  w2c，scripts/run_w2.py --setting w2c 与 scripts/run_x2.py 都能加载）。

**只有样条一条流（DEV-0070，探索 X-3；配置 configs/x3_train.json = X-2 的配置去掉 correct 一节，样条每步 10 条）**：配置有 spline、没有 correct
一节时只开样条流（流号仍是 4），其余同上；数据仍是 X-2 的同一份（scripts/build_x2_data.py 的输出），只读、只核对其中的样条行与样条文件，
数据参数只比 spline 一节（src/explore/x2_data.load_x2_data 的 kinds）。运行身份的 x2 一节：参数、池只有 spline，提示只有描样条的，没有
叠加图（render）一项；scripts/run_x2.py 据此跳过修正类设置。两节都在时与 DEV-0069 逐字节相同（tests/test_x2_spline_only.py 与 51dffb25 的原文件
对照钉住）；只有 correct 一节仍然报错。
"""

from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
import math
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import yaml
from PIL import Image

from src.accounting import Compute, merge, write_compute_json
from src.adapters.base import ORDERS
from src.e1.prompts import PROMPTS_PATH, load_prompt_config
from src.e3 import c3 as C3
from src.e3.data import load_rgb
from src.e3.train import (
    ACCOUNTING,
    COMPUTE,
    CONFIG,
    FINAL,
    LATEST,
    MODEL_SUB,
    README,
    RESUME_STATE,
    SUMMARY,
    TRAIN_LOG,
    StopRequested,
    Trainer,
    _append,
    _atomic_write_text,
    _dir_bytes,
    _identity,
    _jsonl,
    _read_log,
    _sync,
    accounting_doc,
    identity_sha256,
    leaf_dir,
)
from src.runlog import write_run_metadata
from src.executors.raster import rle_to_mask
from src.explore import x2_data as XD
from src.explore import x2_prompts as XP
from src.explore import x2_render as XR
from src.w2 import w2c_replay as RP
from src.w2.config import W2Config, load_w2_config
from src.w2.loop import selection_candidates

REPO_ROOT = Path(__file__).resolve().parents[2]
W2C_CONFIG_PATH = REPO_ROOT / "configs" / "w2c_train.json"
W2C_REPLAY_CONFIG_PATH = REPO_ROOT / "configs" / "w2c_replay_train.json"  # DEV-0065：W2-c 加回放（除 replay 一节外与上面逐项相同）
X2_CONFIG_PATH = REPO_ROOT / "configs" / "x2_train.json"  # DEV-0069：探索 X-2（W2-c 加回放，再加 spline、correct 两条流）
X3_CONFIG_PATH = REPO_ROOT / "configs" / "x3_train.json"  # DEV-0070：探索 X-3（X-2 去掉 correct 流，样条每步 10 条）
REPLAY_PREFIX = "replay"  # 回放流的名字以它开头
SCHEMA = 1
SUPERVISION = "w2c"
VARIANT = "w2c"  # 叶目录 <out>/w2c/<模型>/seed<k>/；run_w2.py --setting w2c 核对检查点来自这个变体
STREAMS: tuple[str, ...] = ("propose", "select")
STREAM_INDEX = {s: i for i, s in enumerate(STREAMS)}
X2_STREAMS: tuple[str, ...] = ("spline", "correct")  # DEV-0069：X-2 的两条流（配置的 spline、correct 两节）；DEV-0070：可以只有 spline
assert X2_STREAMS == XD.KINDS
X2_STREAM_INDEX = {"spline": 4, "correct": 5}  # pool_position 的流号：回放流至多 len(RP.KINDS) 条，占 2、3
assert len(STREAMS) + len(RP.KINDS) <= min(X2_STREAM_INDEX.values())
X2_DOC_KEYS = ("rule", "basis", "note")  # spline、correct 两节里只作说明的键
X2_DATA_KEYS = {"spline": ("n", "min_box_iou"),
                "correct": ("m", "min_box_iou", "start_salt", "round2_ratio", "round2_salt", "round2_apply")}
X2_TRAIN_KEYS = ("per_step", "tiny_per_step", "loss_weight", "order")
DATA_MANIFEST = "manifest.json"  # DEV-0060 的去重与抽样记录（与数据分片同一目录）
_TINY_KEYS = ("steps", "micro_batch_size", "grad_accum_steps", "ckpt_every")
_HEX = set("0123456789abcdef")


class W2CConfigError(ValueError):
    """configs/w2c_train.json 不合格。"""


class W2CDataError(ValueError):
    """W2-c 的训练数据不合格（不是训练集、与 W2 配置对不上、结构不对）：拒绝训练。"""


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class W2CConfig:
    path: str
    sha256: str
    doc: Mapping[str, Any]

    @property
    def seed(self) -> int:
        return int(self.doc["seed"])


def _posint(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and v >= 1


def _e1_order() -> str:
    return load_prompt_config(PROMPTS_PATH).order


def streams_spec(doc: Mapping[str, Any], *, tiny: bool) -> dict[str, Any]:
    """配置的 streams 节 → 本次生效的两条流。核对：per_step 恰好是 propose、select 两项正整数，和 = 每步条数（微批 × 累积，tiny 取 tiny 节），
    各自是微批的整数倍（每个微批只含一种样本）；loss_weights 恰好是 propose 1、select 1（1 : 1 事先固定）；order 是两条流的一个排列。"""
    s = doc["streams"]
    t = doc["tiny"]
    micro = int(t["micro_batch_size"] if tiny else doc["batch"]["micro_batch_size"])
    batch = micro * int(t["grad_accum_steps"] if tiny else doc["batch"]["grad_accum_steps"])
    per = t["per_step"] if tiny else s["per_step"]
    where = "tiny.per_step" if tiny else "streams.per_step"
    if set(per) != set(STREAMS) or any(not _posint(v) for v in per.values()):
        raise W2CConfigError(f"{where} 必须恰好是 {list(STREAMS)} 的正整数条数，收到 {per}")
    per = {k: int(per[k]) for k in STREAMS}
    if sum(per.values()) != batch or any(v % micro for v in per.values()):
        raise W2CConfigError(f"{where} {per}：两项之和必须是每步条数 {batch}，且各自是微批大小 {micro} 的整数倍")
    w = s["loss_weights"]
    if set(w) != set(STREAMS) or any(isinstance(v, bool) or float(v) != 1.0 for v in w.values()):
        raise W2CConfigError(f"streams.loss_weights 必须恰好是 propose 1、select 1（两种损失 1 : 1，事先固定），收到 {w}")
    order = [str(x) for x in s["order"]]
    if sorted(order) != sorted(STREAMS):
        raise W2CConfigError(f"streams.order 必须是 {list(STREAMS)} 的一个排列，收到 {order}")
    return {"per_step": per, "order": order, "loss_weights": {k: float(w[k]) for k in STREAMS},
            "micro_batches": {k: per[k] // micro for k in STREAMS}}


def replay_spec(doc: Mapping[str, Any], *, tiny: bool) -> dict[str, Any] | None:
    """配置的 replay 节（DEV-0065）→ 本次生效的回放流；没有这一节返回 None（W2-c 原样，运行身份里也没有 replay 键）。
    每条回放流：name 以 replay 开头、不与提出、挑选重名；kinds 是 src/w2/w2c_replay.KINDS 的非空子集，各流不重叠；per_step（tiny 取
    tiny_per_step）正整数，不必是微批的整数倍（最后一个微批可以不满）；loss_weight 正数。流的先后 = 列表的顺序，排在提出、挑选之后；
    流号（pool_position 的随机数种子里的那一项）= 2、3……"""
    r = doc.get("replay")
    if r is None:
        return None
    if not isinstance(r, Mapping):
        raise W2CConfigError("replay 必须是一节（对象）")
    t = doc["tiny"]
    micro = int(t["micro_batch_size"] if tiny else doc["batch"]["micro_batch_size"])
    data = r.get("data")
    if not isinstance(data, Mapping) or not (data.get("path") is None or (isinstance(data.get("path"), str) and data["path"])):
        raise W2CConfigError("replay.data 必须是一节，path 是 null（用 --replay-data）或相对仓库根的非空路径")
    streams = r.get("streams")
    if not isinstance(streams, list) or not streams:
        raise W2CConfigError("replay.streams 必须是非空列表")
    out: dict[str, dict[str, Any]] = {}
    used: set[str] = set()
    for i, s in enumerate(streams):
        name = s.get("name") if isinstance(s, Mapping) else None
        if not isinstance(name, str) or not name.startswith(REPLAY_PREFIX) or name in STREAMS or name in out:
            raise W2CConfigError(f"replay.streams[{i}].name 必须以 {REPLAY_PREFIX} 开头且不重名，收到 {name!r}")
        kinds = s.get("kinds")
        if (not isinstance(kinds, list) or not kinds or len(set(kinds)) != len(kinds) or set(kinds) - set(RP.KINDS)
                or used & set(kinds)):
            raise W2CConfigError(f"replay.streams[{i}].kinds 必须是 {list(RP.KINDS)} 的非空子集，各流不重叠，收到 {kinds!r}")
        used |= set(kinds)
        per = s.get("tiny_per_step" if tiny else "per_step")
        if not _posint(per):
            raise W2CConfigError(f"replay.streams[{i}].{'tiny_per_step' if tiny else 'per_step'} 必须是正整数，收到 {per!r}")
        w = s.get("loss_weight")
        if isinstance(w, bool) or not isinstance(w, (int, float)) or not math.isfinite(float(w)) or float(w) <= 0:
            raise W2CConfigError(f"replay.streams[{i}].loss_weight 必须是正数，收到 {w!r}")
        out[name] = {"kinds": [str(k) for k in kinds], "per_step": int(per), "loss_weight": float(w),
                     "micro_batches": int(math.ceil(int(per) / micro)), "index": len(STREAMS) + i}
    return {"streams": out, "order": list(out), "per_step_total": sum(v["per_step"] for v in out.values())}


def x2_spec(doc: Mapping[str, Any], *, tiny: bool) -> dict[str, Any] | None:
    """配置的 spline、correct 两节（DEV-0069）→ 本次生效的 X-2 流；两节都没有返回 None（W2-c 原样，运行身份里也没有 x2 键）。
    两节都有，或只有 spline 一节（DEV-0070：只开样条流，数据参数只核对 spline 一节）；只有 correct 一节报错。要有 x2_data 一节。
    每条流：per_step（tiny 取 tiny_per_step）正整数，不必是微批的整数倍；loss_weight 正数；order 等于 E1 的 image_order；数据参数按
    src/explore/x2_data.check_params（kinds = 有的那几节）核对。流的先后 = (spline, correct) 里有的，排在提出、挑选之后、回放之前；
    流号 = X2_STREAM_INDEX。"""
    have = [s for s in X2_STREAMS if s in doc]
    if not have:
        return None
    if "spline" not in have:
        raise W2CConfigError(f"correct 一节不能单独有（X-2 的流要有 spline 一节；DEV-0070 起可以只有 spline），只有 {have}")
    xd = doc.get("x2_data")
    if not isinstance(xd, Mapping) or not (xd.get("path") is None or (isinstance(xd.get("path"), str) and xd["path"])):
        raise W2CConfigError("x2_data 必须是一节，path 是 null（用 --x2-data）或相对仓库根的非空路径")
    t = doc["tiny"]
    micro = int(t["micro_batch_size"] if tiny else doc["batch"]["micro_batch_size"])
    out: dict[str, dict[str, Any]] = {}
    for name in have:
        sec = doc[name]
        if not isinstance(sec, Mapping):
            raise W2CConfigError(f"{name} 必须是一节（对象）")
        extra = set(sec) - set(X2_DATA_KEYS[name]) - set(X2_TRAIN_KEYS) - set(X2_DOC_KEYS)
        if extra:
            raise W2CConfigError(f"{name} 有未知的键 {sorted(extra)}")
        per = sec.get("tiny_per_step" if tiny else "per_step")
        if not _posint(per):
            raise W2CConfigError(f"{name}.{'tiny_per_step' if tiny else 'per_step'} 必须是正整数，收到 {per!r}")
        w = sec.get("loss_weight")
        if isinstance(w, bool) or not isinstance(w, (int, float)) or not math.isfinite(float(w)) or float(w) <= 0:
            raise W2CConfigError(f"{name}.loss_weight 必须是正数，收到 {w!r}")
        if sec.get("order") not in ORDERS or sec["order"] != _e1_order():
            raise W2CConfigError(f"{name}.order 必须等于 configs/e1_prompts.json 的 image_order {_e1_order()!r}，收到 {sec.get('order')!r}")
        out[name] = {"per_step": int(per), "loss_weight": float(w), "order": str(sec["order"]),
                     "micro_batches": int(math.ceil(int(per) / micro)), "index": X2_STREAM_INDEX[name]}
    try:
        params = XD.check_params({k: {kk: doc[k][kk] for kk in X2_DATA_KEYS[k] if kk in doc[k]} for k in have}, kinds=have)
    except XD.X2DataError as e:
        raise W2CConfigError(str(e)) from None
    return {"streams": out, "order": list(have), "per_step_total": sum(v["per_step"] for v in out.values()), "data_params": params}


def stream_names(rc: Mapping[str, Any]) -> tuple[str, ...]:
    """本次的全部样本流：提出、挑选，再加 X-2 的流（有 spline、correct 两节时两条，只有 spline 时一条）、回放流（有 replay 一节时）。"""
    x, r = rc.get("x2"), rc.get("replay")
    return STREAMS + (() if x is None else tuple(x["order"])) + (() if r is None else tuple(r["order"]))


def stream_order(rc: Mapping[str, Any]) -> list[str]:
    """一步之内微批的先后：streams.order，再接 X-2 的流、回放流的顺序。"""
    x, r = rc.get("x2"), rc.get("replay")
    return list(rc["streams"]["order"]) + ([] if x is None else list(x["order"])) + ([] if r is None else list(r["order"]))


def stream_weights(rc: Mapping[str, Any]) -> dict[str, float]:
    w = {s: float(rc["streams"]["loss_weights"][s]) for s in STREAMS}
    x = rc.get("x2")
    if x is not None:
        w.update({s: float(v["loss_weight"]) for s, v in x["streams"].items()})
    r = rc.get("replay")
    if r is not None:
        w.update({s: float(v["loss_weight"]) for s, v in r["streams"].items()})
    return w


def validate(doc: Mapping[str, Any]) -> None:
    bad: list[str] = []
    if doc.get("schema") != SCHEMA:
        bad.append(f"schema 应为 {SCHEMA}")
    if doc.get("enable_thinking") is not False:
        bad.append("enable_thinking 必须是 false（D113：W2-c 的提出与挑选都关思考）")
    wc = doc.get("w2_config") or {}
    sha = wc.get("sha256")
    if not isinstance(wc.get("path"), str) or not wc["path"]:
        bad.append("w2_config.path 必须是非空字符串")
    if sha is not None and (not isinstance(sha, str) or len(sha) != 64 or set(sha) - _HEX):
        bad.append("w2_config.sha256 必须是 null（占位）或 64 位十六进制小写")
    d = doc.get("data") or {}
    if d.get("split") != "train":
        bad.append("data.split 必须是 train（只读训练集）")
    if not isinstance(d.get("sources"), list) or not d["sources"] or any(not isinstance(x, str) for x in d["sources"]):
        bad.append("data.sources 必须是非空的数据集名列表")
    if not isinstance(d.get("workers"), int) or isinstance(d.get("workers"), bool) or d["workers"] < 0:
        bad.append("data.workers 必须是非负整数")
    p = doc.get("propose") or {}
    if p.get("order") not in ORDERS:
        bad.append(f"propose.order 必须是 {ORDERS} 之一")
    ref = p.get("box_template_from") or {}
    if not isinstance(ref.get("config"), str) or not isinstance(ref.get("task"), str):
        bad.append("propose.box_template_from 必须给 config 与 task")
    sl = doc.get("select") or {}
    if sl.get("target") != "decimal":
        bad.append("select.target 必须是 decimal（编号的十进制文本）")
    if sl.get("order") not in ORDERS:
        bad.append(f"select.order 必须是 {ORDERS} 之一")
    elif sl["order"] != _e1_order():
        bad.append(f"select.order {sl['order']!r} 必须等于 configs/e1_prompts.json 的 image_order {_e1_order()!r}"
                   "（W2 评测时挑选一步 build_inputs 用的就是它）")
    lo = doc.get("lora") or {}
    if not _posint(lo.get("r")) or not isinstance(lo.get("alpha"), (int, float)) or not isinstance(lo.get("dropout"), (int, float)):
        bad.append("lora 必须给正整数 r、数值 alpha 与 dropout")
    o = doc.get("optim") or {}
    if o.get("optimizer") != "adamw" or o.get("schedule") != "warmup_cosine" or "token_rows" in o:
        bad.append("optim 必须是 adamw + warmup_cosine，且没有 token_rows（W2-c 不加新 token）")
    if (doc.get("loss") or {}).get("reduction") != "per_stream_token_mean" or not _posint((doc.get("loss") or {}).get("chunk_size")):
        bad.append("loss.reduction 必须是 per_stream_token_mean，loss.chunk_size 必须是正整数")
    for k in ("steps", "ckpt_every"):
        if not _posint(doc.get(k)):
            bad.append(f"{k} 必须是正整数")
    if not isinstance(doc.get("seed"), int) or isinstance(doc.get("seed"), bool) or doc["seed"] < 0:
        bad.append("seed 必须是非负整数")
    if bad:
        raise W2CConfigError("configs/w2c_train.json 不合格：\n- " + "\n- ".join(bad))
    for tiny in (False, True):
        streams_spec(doc, tiny=tiny)
        replay_spec(doc, tiny=tiny)
        x2_spec(doc, tiny=tiny)


def load_w2c_config(path: str | Path = W2C_CONFIG_PATH) -> W2CConfig:
    p = Path(path)
    raw = p.read_bytes()
    doc = json.loads(raw)
    validate(doc)
    rel = str(p.resolve().relative_to(REPO_ROOT)) if p.resolve().is_relative_to(REPO_ROOT) else str(p)
    return W2CConfig(path=rel, sha256=hashlib.sha256(raw).hexdigest(), doc=doc)


def resolve_w2c_run(cfg: W2CConfig, *, tiny: bool, steps: int | None = None) -> dict[str, Any]:
    """本次实际生效的训练参数（写进 config.yaml 的 training）：--tiny 时按 tiny 节覆盖规模参数；steps 给了就覆盖配置的 steps（记 steps_override）。
    键名与 E3 的 resolve_run 相同的那些，src/e3/train.Trainer 照读（lora、optim、warmup_steps、steps、micro_batch_size、gradient_checkpointing、
    enable_thinking）。"""
    d = cfg.doc
    out: dict[str, Any] = {
        "supervision": SUPERVISION,
        "variant": VARIANT,
        "steps": int(d["steps"]),
        "micro_batch_size": int(d["batch"]["micro_batch_size"]),
        "grad_accum_steps": int(d["batch"]["grad_accum_steps"]),
        "ckpt_every": int(d["ckpt_every"]),
        "lora": {k: d["lora"][k] for k in ("r", "alpha", "dropout")},
        "optim": {k: copy.deepcopy(v) for k, v in d["optim"].items() if k != "basis"},
        "loss_reduction": d["loss"]["reduction"],
        "chunk_size": int(d["loss"]["chunk_size"]),
        "gradient_checkpointing": bool(d["model"]["gradient_checkpointing"]),
        "enable_thinking": bool(d["enable_thinking"]),
        "data_workers": int(d["data"]["workers"]),
        "streams": streams_spec(d, tiny=tiny),
    }
    rs = replay_spec(d, tiny=tiny)
    if rs is not None:  # DEV-0065：只有配置有 replay 一节时才有这个键（默认配置的 run_cfg 与以前逐项相同）
        out["replay"] = rs
    xs = x2_spec(d, tiny=tiny)
    if xs is not None:  # DEV-0069：只有配置有 spline、correct 两节（DEV-0070：或只有 spline）时才有这个键
        out["x2"] = xs
    if tiny:
        for k in _TINY_KEYS:
            out[k] = int(d["tiny"][k])
    if steps is not None:
        if not _posint(steps):
            raise W2CConfigError(f"steps 必须是正整数，收到 {steps!r}")
        out["steps_override"] = {"config": out["steps"], "used": int(steps)}
        out["steps"] = int(steps)
    out["batch_size"] = out["micro_batch_size"] * out["grad_accum_steps"]
    out["warmup_steps"] = int(d["tiny"]["warmup_steps"]) if tiny else int(math.ceil(float(d["optim"]["warmup_ratio"]) * out["steps"]))
    if not 0 <= out["warmup_steps"] <= out["steps"]:
        raise W2CConfigError(f"warmup_steps {out['warmup_steps']} 不在 [0, steps]")
    if sum(out["streams"]["per_step"].values()) != out["batch_size"]:
        raise W2CConfigError("streams.per_step 的和与每步条数不同")
    return out


def resolve_w2_config(cfg: W2CConfig, *, tiny: bool, override: str | Path | None = None) -> tuple[W2Config, dict[str, Any]]:
    """W2-a 冻结的 configs/w2.json（规则见配置的 w2_config.rule）→ (W2Config, 运行身份里的 w2_config 一节)。"""
    if override is not None and not tiny:
        raise W2CConfigError("--w2-config 只配 --tiny（正式运行用 configs/w2c_train.json 写死的那一份）")
    pinned = cfg.doc["w2_config"]["sha256"]
    path = Path(override) if override is not None else REPO_ROOT / cfg.doc["w2_config"]["path"]
    w2 = load_w2_config(path)
    if not tiny:
        if pinned is None:
            raise W2CConfigError("configs/w2c_train.json 的 w2_config.sha256 还是占位（null）：W2-a 冻结之后由管理员填，之前不能正式训练")
        if w2.sha256 != pinned:
            raise W2CConfigError(f"{path} 的 sha256 {w2.sha256[:16]}… 不是 configs/w2c_train.json 写死的 {pinned[:16]}…")
        if not w2.frozen:
            raise W2CConfigError(f"{path} 没有冻结（frozen = false）：W2-c 用 W2-a 冻结的那一份")
    if w2.select_system is not None:
        raise W2CConfigError(f"{path} 的 select.system 不是 null：W2-c 关思考的挑选走 build_inputs，没有 system 轮")
    info = {**w2.identity(), "pinned_sha256": pinned, "override": override is not None, "k": w2.k,
            "candidates_sha256": w2.candidates_digest(), "layout": w2.layout, "style": w2.style,
            "select_prompt": w2.data["select"]["prompt"], "select_system": w2.select_system}
    return w2, info


# ---------------------------------------------------------------------------
# 数据（docs/w2_plan.md §5）
# ---------------------------------------------------------------------------


ROW_KEYS = ("id", "source", "split", "image", "image_size", "expression", "gt_box", "proposal", "candidates", "label", "spec_sha256",
            "w2_config_sha256")


@dataclass(frozen=True)
class W2CData:
    """DEV-0060 的数据：rows 按分片文件名、行号的顺序；propose / select 是两个池（rows 的下标）。"""

    data_dir: Path
    rows: tuple[Mapping[str, Any], ...]
    propose: tuple[int, ...]
    select: tuple[int, ...]
    shards: tuple[dict[str, Any], ...]
    manifest_sha256: str | None
    stats: Mapping[str, Any]

    def pool(self, stream: str) -> tuple[int, ...]:
        return self.propose if stream == "propose" else self.select

    def files(self) -> list[Path]:
        out = [self.data_dir / s["file"] for s in self.shards]
        if self.manifest_sha256 is not None:
            out.append(self.data_dir / DATA_MANIFEST)
        return out

    def identity(self) -> dict[str, Any]:
        """运行身份里的数据一节：内容的指纹（分片的 sha256 与条数、manifest 的 sha256、两个池的 id 指纹）与统计；不含路径。"""
        return {"format": "docs/w2_plan.md §5（DEV-0060）", "shards": [dict(s) for s in self.shards],
                "manifest_sha256": self.manifest_sha256, **dict(self.stats)}


def _ids_digest(ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def _num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))


def check_label(cands: Sequence[Mapping[str, Any]], label: int) -> bool:
    """label 是否是 iou 最高者里编号最小的（§5 的标签规则；按记下的 iou 精确比较）。"""
    ious = [float(c["iou"]) for c in cands]
    best = max(ious)
    return ious[label - 1] == best and all(v < best for v in ious[: label - 1])


def row_problems(r: Mapping[str, Any], *, k: int, sources: Sequence[str], w2_sha: str, spec_sha: str) -> list[str]:
    """一行的问题（空列表 = 合格）。规则见模块说明。"""
    bad = [f"缺少 {key}" for key in ROW_KEYS if key not in r]
    if bad:
        return bad
    if r["split"] != "train":
        bad.append(f"split = {r['split']!r}（只读训练集）")
    if r["source"] not in sources:
        bad.append(f"source = {r['source']!r} 不在 {list(sources)} 里")
    if r["w2_config_sha256"] != w2_sha:
        bad.append(f"w2_config_sha256 {str(r['w2_config_sha256'])[:16]}… ≠ 本次的 W2 配置 {w2_sha[:16]}…")
    if r["spec_sha256"] != spec_sha:
        bad.append(f"spec_sha256 {str(r['spec_sha256'])[:16]}… ≠ 本次的候选清单 {spec_sha[:16]}…")
    img = r["image"]
    if not isinstance(img, str) or not img or Path(img).is_absolute() or ".." in Path(img).parts:
        bad.append(f"image 必须是相对数据根的路径，收到 {img!r}")
    sz = r["image_size"]
    if not (isinstance(sz, list) and len(sz) == 2 and all(_posint(v) for v in sz)):
        bad.append(f"image_size 必须是 [W, H] 的正整数，收到 {sz!r}")
        return bad
    if not isinstance(r["expression"], str) or not r["expression"].strip():
        bad.append("expression 为空")
    b = r["gt_box"]
    if not (isinstance(b, list) and len(b) == 4 and all(_num(v) for v in b)) or b[2] < b[0] or b[3] < b[1]:
        bad.append(f"gt_box 必须是 [x0, y0, x1, y1]（x1 ≥ x0、y1 ≥ y0），收到 {b!r}")
    p = r["proposal"]
    if not isinstance(p, Mapping) or "box" not in p or "failure" not in p or (p["box"] is None) == (p["failure"] is None):
        bad.append("proposal 必须有 box 与 failure，且恰有一个为 null")
        return bad
    cands, label = r["candidates"], r["label"]
    if (cands is None) != (p["failure"] is not None):
        bad.append("proposal.failure 不为空 ⟺ candidates 为 null（§5）")
    if cands is None:
        if label is not None:
            bad.append("candidates 为 null 时 label 也必须为 null")
        return bad
    if not isinstance(cands, list) or len(cands) != k:
        bad.append(f"candidates 必须恰有 K = {k} 个，收到 {len(cands) if isinstance(cands, list) else type(cands).__name__}")
        return bad
    w, h = sz
    for i, c in enumerate(cands, start=1):
        if not isinstance(c, Mapping) or c.get("index") != i or not isinstance(c.get("name"), str) or not isinstance(c.get("ok"), bool):
            bad.append(f"候选 {i}：index 必须是 {i}、name 是字符串、ok 是 bool")
            continue
        rle = c.get("rle")
        if not isinstance(rle, Mapping) or list(rle.get("size") or []) != [h, w] or "counts" not in rle:
            bad.append(f"候选 {i}：rle 的 size 必须是 [H, W] = {[h, w]}")
        if not _num(c.get("iou")) or not 0.0 <= float(c["iou"]) <= 1.0:
            bad.append(f"候选 {i}：iou 必须是 [0, 1] 的数")
    if bad:
        return bad
    if label is not None:
        if not _posint(label) or label > k:
            bad.append(f"label 必须是 1..{k} 或 null，收到 {label!r}")
        elif not check_label(cands, label):
            bad.append(f"label {label} 不是 iou 最高者里编号最小的（iou {[c['iou'] for c in cands]}）")
    return bad


def load_w2c_data(data_dir: str | Path, *, w2: W2Config, sources: Sequence[str], data_root: str | Path,
                  require_manifest: bool) -> W2CData:
    """读 DEV-0060 的数据并逐行核对（规则见模块说明）；不合格就抛 W2CDataError（列出前几条）。"""
    d = Path(data_dir)
    shards = sorted(d.glob("*.jsonl"))
    if not shards:
        raise W2CDataError(f"{d} 里没有 *.jsonl 分片")
    man = d / DATA_MANIFEST
    if require_manifest and not man.is_file():
        raise W2CDataError(f"{d} 里没有 {DATA_MANIFEST}（DEV-0060 的去重与抽样记录，硬约束 7）")
    k, w2_sha, spec_sha = w2.k, w2.sha256, w2.candidates_digest()
    rows: list[Mapping[str, Any]] = []
    shard_info = []
    problems: list[str] = []
    seen: set[str] = set()
    root = Path(data_root)
    for f in shards:
        raw = f.read_bytes()
        lines = [ln for ln in raw.decode("utf-8").splitlines() if ln.strip()]
        shard_info.append({"file": f.name, "n": len(lines), "sha256": hashlib.sha256(raw).hexdigest()})
        for n, ln in enumerate(lines, start=1):
            try:
                r = json.loads(ln)
            except ValueError as e:
                problems.append(f"{f.name}:{n}：不是 JSON（{e}）")
                continue
            if not isinstance(r, dict):
                problems.append(f"{f.name}:{n}：一行必须是一个 JSON 对象")
                continue
            bad = row_problems(r, k=k, sources=sources, w2_sha=w2_sha, spec_sha=spec_sha)
            if not bad:
                if r["id"] in seen:
                    bad.append("id 重复")
                elif not (root / r["image"]).is_file():
                    bad.append(f"图像不在：{root / r['image']}")
            if bad:
                problems.append(f"{f.name}:{n}（{r.get('id')}）：{'；'.join(bad)}")
                continue
            seen.add(r["id"])
            rows.append(r)
    if problems:
        raise W2CDataError(f"W2-c 的数据有 {len(problems)} 行不合格，拒绝训练：\n- " + "\n- ".join(problems[:10]))
    prop = tuple(range(len(rows)))
    sel = tuple(i for i, r in enumerate(rows) if r["candidates"] is not None and r["label"] is not None)
    labels: dict[str, int] = {}
    for i in sel:
        labels[str(rows[i]["label"])] = labels.get(str(rows[i]["label"]), 0) + 1
    by_source: dict[str, int] = {}
    for r in rows:
        by_source[r["source"]] = by_source.get(r["source"], 0) + 1
    stats = {"n_rows": len(rows), "n_propose": len(prop), "n_select": len(sel),
             "n_proposal_failed": sum(1 for r in rows if r["candidates"] is None),
             "n_label_null_with_candidates": sum(1 for r in rows if r["candidates"] is not None and r["label"] is None),
             "label_counts": {key: labels[key] for key in sorted(labels, key=int)}, "by_source": dict(sorted(by_source.items())),
             "k": k, "propose_ids_sha256": _ids_digest([rows[i]["id"] for i in prop]),
             "select_ids_sha256": _ids_digest([rows[i]["id"] for i in sel])}
    if not sel:
        raise W2CDataError(f"{d}：挑选的池是空的（没有 candidates 与 label 都不为 null 的行）")
    return W2CData(data_dir=d, rows=tuple(rows), propose=prop, select=sel, shards=tuple(shard_info),
                   manifest_sha256=hashlib.sha256(man.read_bytes()).hexdigest() if man.is_file() else None, stats=stats)


# ---------------------------------------------------------------------------
# 样本日程（无状态）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class W2CPlan:
    """日程里的一格：流 stream 的第 j 个槽位（第 step 步、本步全局下标 k）→ 池里的第 index 个（rows 的下标 row）、第 epoch 个 epoch。"""

    stream: str
    k: int
    step: int
    j: int
    index: int
    row: int
    epoch: int
    sample_id: str


def pool_position(j: int, n: int, *, seed: int, stream: str, index: int | None = None) -> tuple[int, int]:
    """流内的第 j 个槽位 → (池下标, epoch)：第 j // n 个 epoch 的置换的第 j mod n 个（规则见模块说明）。index：流号（默认
    STREAM_INDEX[stream]；回放流取 replay_spec 的 index）。"""
    if n < 1:
        raise ValueError(f"{stream}：池为空")
    ep = j // n
    perm = np.random.default_rng([int(seed), STREAM_INDEX[stream] if index is None else int(index), ep]).permutation(n)
    return int(perm[j % n]), ep


class W2CSchedule:
    """extra_per_step：同一步里排在提出、挑选之后的回放条数（DEV-0065；只影响 W2CPlan.k 的全局编号，默认 0 = W2-c 原样）。"""

    def __init__(self, data: W2CData, *, seed: int, per_step: Mapping[str, int], order: Sequence[str], extra_per_step: int = 0) -> None:
        if sorted(order) != sorted(STREAMS) or set(per_step) != set(STREAMS):
            raise ValueError(f"W2-c 的样本流必须恰好是 {STREAMS}")
        self.data = data
        self.seed = int(seed)
        self.per_step = {s: int(per_step[s]) for s in STREAMS}
        self.order = tuple(order)
        self.batch_size = sum(self.per_step.values())
        self.step_size = self.batch_size + int(extra_per_step)

    def plan(self, stream: str, j: int) -> W2CPlan:
        pool = self.data.pool(stream)
        idx, ep = pool_position(j, len(pool), seed=self.seed, stream=stream)
        row = pool[idx]
        step = j // self.per_step[stream]
        off = sum(self.per_step[s] for s in self.order[: self.order.index(stream)])
        return W2CPlan(stream=stream, k=step * self.step_size + off + j % self.per_step[stream], step=step, j=j, index=idx, row=row,
                       epoch=ep, sample_id=str(self.data.rows[row]["id"]))

    def step_plans(self, step: int) -> list[W2CPlan]:
        return [self.plan(s, step * self.per_step[s] + i) for s in self.order for i in range(self.per_step[s])]


class ReplaySchedule:
    """回放流的日程（DEV-0065）：规则同 W2CSchedule（无状态；流 s 的第 j 个槽位 = pool_position(j, |池|, 种子, 流号)），每步排在提出、
    挑选之后；W2CPlan.k 从 base_batch（提出 + 挑选的条数）往后接。"""

    def __init__(self, data: RP.ReplayData, *, seed: int, spec: Mapping[str, Any], base_batch: int) -> None:
        self.data = data
        self.seed = int(seed)
        self.streams = {s: dict(v) for s, v in spec["streams"].items()}
        self.order = tuple(spec["order"])
        self.base_batch = int(base_batch)
        self.batch_size = self.base_batch + sum(v["per_step"] for v in self.streams.values())
        if set(self.order) != set(data.pools):
            raise ValueError(f"回放流 {list(self.order)} 与回放数据的池 {list(data.pools)} 不同")

    def plan(self, stream: str, j: int) -> W2CPlan:
        v = self.streams[stream]
        pool = self.data.pool(stream)
        idx, ep = pool_position(j, len(pool), seed=self.seed, stream=stream, index=v["index"])
        row = pool[idx]
        per = int(v["per_step"])
        step = j // per
        off = self.base_batch + sum(self.streams[s]["per_step"] for s in self.order[: self.order.index(stream)])
        return W2CPlan(stream=stream, k=step * self.batch_size + off + j % per, step=step, j=j, index=idx, row=row, epoch=ep,
                       sample_id=str(self.data.rows[row]["id"]))

    def step_plans(self, step: int) -> list[W2CPlan]:
        return [self.plan(s, step * self.streams[s]["per_step"] + i) for s in self.order for i in range(self.streams[s]["per_step"])]


class X2Schedule:
    """X-2 两条流的日程（DEV-0069）：规则同 W2CSchedule（无状态；流 s 的第 j 个槽位 = pool_position(j, |池|, 种子, 流号)），每步排在
    提出、挑选之后、回放之前；W2CPlan.k 从 base_batch（提出 + 挑选的条数）往后接，step_size 是一步的总条数（含回放）。"""

    def __init__(self, data: XD.X2Data, *, seed: int, spec: Mapping[str, Any], base_batch: int, step_size: int) -> None:
        self.data = data
        self.seed = int(seed)
        self.streams = {s: dict(v) for s, v in spec["streams"].items()}
        self.order = tuple(spec["order"])
        self.base_batch = int(base_batch)
        self.step_size = int(step_size)
        self.batch_size = self.base_batch + sum(v["per_step"] for v in self.streams.values())
        if set(self.order) - set(data.pools):
            raise ValueError(f"X-2 的流 {list(self.order)} 在数据里没有池（有 {list(data.pools)}）")

    def plan(self, stream: str, j: int) -> W2CPlan:
        v = self.streams[stream]
        pool = self.data.pool(stream)
        idx, ep = pool_position(j, len(pool), seed=self.seed, stream=stream, index=v["index"])
        row = pool[idx]
        per = int(v["per_step"])
        step = j // per
        off = self.base_batch + sum(self.streams[s]["per_step"] for s in self.order[: self.order.index(stream)])
        return W2CPlan(stream=stream, k=step * self.step_size + off + j % per, step=step, j=j, index=idx, row=row, epoch=ep,
                       sample_id=str(self.data.rows[row]["id"]))

    def step_plans(self, step: int) -> list[W2CPlan]:
        return [self.plan(s, step * self.streams[s]["per_step"] + i) for s in self.order for i in range(self.streams[s]["per_step"])]


# ---------------------------------------------------------------------------
# 样本
# ---------------------------------------------------------------------------


@dataclass
class W2CExample:
    plan: W2CPlan
    images: list[Any]
    instruction: str
    order: str
    target_ids: list[int]
    weights: list[int]
    target_text: str
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def n_supervised(self) -> int:
        return int(sum(self.weights))


def load_pil(path: str | Path) -> Image.Image:
    """同 src/e1/runner.load_images：打开后转 RGB（W2-a 评测时挑选图的原图就是这样读的）。"""
    with Image.open(path) as im:
        return im.convert("RGB")


def selection_row_view(row: Mapping[str, Any]) -> dict[str, Any]:
    """§5 的一行 → candidates.jsonl 的行的形状（src/w2/loop.selection_candidates 读的那几个键），让挑选图走评测时的同一个函数。"""
    return {"id": row["id"], "image_size": list(row["image_size"]),
            "candidates": [{"index": c["index"], "name": c["name"], "ok": c["ok"], "info": {}, "mask": {"rle": c["rle"]}}
                           for c in row["candidates"]]}


class W2CBuilder:
    """W2CPlan → W2CExample（规则见模块说明）。impl：src/w2/runner.Impl（Candidate 类与 render_selection，同 W2-a 评测）。"""

    def __init__(self, cfg: W2CConfig, data: W2CData, *, w2: W2Config, impl: Any, data_root: str | Path, stop_id: int,
                 encode: Callable[[str], list[int]], family: str, format_boxes: Callable[..., str]) -> None:
        if impl.render is None:
            raise ValueError("W2-c 的挑选样本要渲染（impl.render）")
        self.cfg = cfg
        self.data = data
        self.w2 = w2
        self.impl = impl
        self.root = Path(data_root)
        self.stop_id = int(stop_id)
        self.encode = encode
        self.format_boxes = format_boxes
        self.box_template, self.box_source = C3.load_box_template({"c3": {"box_template_from": cfg.doc["propose"]["box_template_from"]}},
                                                                  family, REPO_ROOT)
        self.orders = {"propose": str(cfg.doc["propose"]["order"]), "select": str(cfg.doc["select"]["order"])}

    def build(self, plan: W2CPlan) -> W2CExample:
        row = self.data.rows[plan.row]
        if row["id"] != plan.sample_id:
            raise RuntimeError(f"日程与数据对不上：{plan.sample_id} ≠ {row['id']}")
        return self._propose(plan, row) if plan.stream == "propose" else self._select(plan, row)

    def _target(self, text: str) -> list[int]:
        return list(self.encode(text)) + [self.stop_id]

    def propose_inputs(self, row: Mapping[str, Any]) -> tuple[np.ndarray, str, str]:
        """(原图, 提示, 目标文本)：与 C3 的框监督逐字相同（src/e3/data.ExampleBuilder._text 的 box 分支）。"""
        image = load_rgb(self.root / row["image"])
        h, w = image.shape[:2]
        if [w, h] != list(row["image_size"]):
            raise ValueError(f"{row['id']}：图像 {w}×{h} 与 image_size {row['image_size']} 不符")
        expr = str(row["expression"]).strip()
        return image, C3.render_e1_template(self.box_template, expr), C3.box_target(self.format_boxes, row["gt_box"], w, h, expr)

    def select_inputs(self, row: Mapping[str, Any]) -> tuple[list[Image.Image], str]:
        """(挑选图, 提示)：与 W2-a 评测时 run_candidate_setting 的调用逐个相同（selection_candidates → render → select_prompt）。"""
        image = load_pil(self.root / row["image"])
        if list(image.size) != list(row["image_size"]):
            raise ValueError(f"{row['id']}：图像 {image.size} 与 image_size {row['image_size']} 不符")
        cands = selection_candidates(selection_row_view(row), self.impl.Candidate)
        imgs = self.impl.render(image, cands, layout=self.w2.layout, style=self.w2.style)
        if not isinstance(imgs, list) or not imgs or any(not hasattr(im, "size") for im in imgs):
            raise RuntimeError(f"{row['id']}：render_selection 应返回非空的 PIL 图列表，收到 {type(imgs).__name__}")
        return imgs, self.w2.select_prompt(len(cands), row["expression"])

    def _propose(self, plan: W2CPlan, row: Mapping[str, Any]) -> W2CExample:
        image, instruction, text = self.propose_inputs(row)
        ids = self._target(text)
        return W2CExample(plan, [image], instruction, self.orders["propose"], ids, [1] * len(ids), text,
                          info={"proposal_failed": row["proposal"]["failure"] is not None})

    def _select(self, plan: W2CPlan, row: Mapping[str, Any]) -> W2CExample:
        imgs, prompt = self.select_inputs(row)
        text = str(int(row["label"]))
        ids = self._target(text)
        return W2CExample(plan, list(imgs), prompt, self.orders["select"], ids, [1] * len(ids), text,
                          info={"label": int(row["label"]), "k": len(row["candidates"])})


class ReplayBuilder:
    """回放样本（DEV-0065）：提示、图（计数：原图 load_pil，与生成回答时同一个函数）与顺序取回放数据的行原样，目标 = base 生成的 token
    原样（含停止 token），全部监督。stop_ids：本家族的停止 token（目标的最后一个必须是其中之一，开训前核对）。"""

    def __init__(self, data: RP.ReplayData, *, data_root: str | Path, stop_ids: Sequence[int]) -> None:
        self.data = data
        self.root = Path(data_root)
        self.stop_ids = {int(i) for i in stop_ids}
        bad = [str(data.rows[i]["id"]) for p in data.pools.values() for i in p if int(data.rows[i]["token_ids"][-1]) not in self.stop_ids]
        if bad:
            raise RP.ReplayDataError(f"{len(bad)} 条回放样本的目标不以本家族的停止 token 结尾（例如 {bad[:3]}）")

    def build(self, plan: W2CPlan) -> W2CExample:
        row = self.data.rows[plan.row]
        if str(row["id"]) != plan.sample_id:
            raise RuntimeError(f"日程与回放数据对不上：{plan.sample_id} ≠ {row['id']}")
        images = [load_pil(self.root / p) for p in row["images"]]
        for im in images:
            if list(im.size) != list(row["image_size"]):
                raise ValueError(f"{row['id']}：图像 {im.size} 与 image_size {row['image_size']} 不符")
        ids = [int(t) for t in row["token_ids"]]
        return W2CExample(plan, images, row["prompt"], row["order"], ids, [1] * len(ids), row["answer"],
                          info={"kind": row["kind"], "replay_id": row["id"]})


class X2Builder:
    """X-2 的样本（DEV-0069；规则见模块说明）。原图用 load_pil（同挑选；推理时 src/explore/x2_runner 用 E1 的 load_images，同一种读法）；
    修正的叠加图用冻结的 W2 配置的 render.style 现画；提示与目标文本都来自 src/explore/x2_prompts。"""

    def __init__(self, data: XD.X2Data, *, w2: W2Config, data_root: str | Path, stop_id: int, encode: Callable[[str], list[int]],
                 spec: Mapping[str, Any]) -> None:
        self.data = data
        self.w2 = w2
        self.root = Path(data_root)
        self.stop_id = int(stop_id)
        self.encode = encode
        self.orders = {s: str(v["order"]) for s, v in spec["streams"].items()}
        self.params = spec["data_params"]

    def inputs(self, row: Mapping[str, Any]) -> tuple[list[Image.Image], str]:
        """(图, 提示)：描样条是原图，指修正框是叠加图（起点 mask + 提出的框）。推理时的运行器调用同样的两个函数。"""
        image = load_pil(self.root / row["image"])
        if list(image.size) != list(row["image_size"]):
            raise ValueError(f"{row['id']}：图像 {image.size} 与 image_size {row['image_size']} 不符")
        if row["kind"] == "spline":
            return [image], XP.spline_prompt(row["expression"], row["box_q"], self.params["spline"]["n"])
        start = rle_to_mask(row["start_rle"]).astype(bool)
        return ([XR.render_overlay(image, start, row["box"], style=self.w2.style)],
                XP.correct_prompt(row["expression"], row["box_q"], self.params["correct"]["m"]))

    def build(self, plan: W2CPlan) -> W2CExample:
        row = self.data.rows[plan.row]
        if str(row["id"]) != plan.sample_id or row["kind"] != plan.stream:
            raise RuntimeError(f"日程与 X-2 数据对不上：{plan.sample_id}（{plan.stream}）≠ {row['id']}（{row['kind']}）")
        images, prompt = self.inputs(row)
        text = str(row["target_text"])
        ids = list(self.encode(text)) + [self.stop_id]
        info = {"kind": row["kind"]} if row["kind"] == "spline" else {"kind": "correct", "round": int(row["round"]),
                                                                      "n_boxes": len(row["target"])}
        return W2CExample(plan, images, prompt, self.orders[plan.stream], ids, [1] * len(ids), text, info=info)


# ---------------------------------------------------------------------------
# 训练器：两条文本样本流各自归一
# ---------------------------------------------------------------------------


def number_token_ids(encode: Callable[[str], list[int]], k: int) -> list[int] | None:
    """编号 1..k 各自的 token id；有一个不是单 token 就返回 None（select_number_acc 记 null）。"""
    out = []
    for i in range(1, int(k) + 1):
        ids = list(encode(str(i)))
        if len(ids) != 1:
            return None
        out.append(int(ids[0]))
    return out if len(set(out)) == len(out) else None


class W2CTrainer(Trainer):
    """src/e3/train.Trainer 的子类：搭建、审计、续跑状态全用父类（supervision = w2c：没有网格 token、没有 token 行、注入 LoRA）；
    只覆盖 optimize（模块说明的损失）。number_ids：编号 1..K 的 token id（只用于 select_number_acc）。"""

    def __init__(self, adapter: Any, cfg: W2CConfig, run_cfg: Mapping[str, Any], *, seed: int,
                 number_ids: Sequence[int] | None = None) -> None:
        super().__init__(adapter, cfg, run_cfg, supervision=SUPERVISION, seed=seed)
        if self.with_grid_tokens or self.grid_like or self.mixed or self.frozen_llm:
            raise RuntimeError("W2-c 的训练器不该有网格 token / 冻结 LLM")
        self.number_ids = None if number_ids is None else [int(i) for i in number_ids]

    def _select_hits(self, out: Any, group: Sequence[W2CExample]) -> tuple[int, int]:
        """挑选样本第一个目标位置上，只在编号 token 之间比较时 argmax 是否等于标签（教师强制的诊断）。"""
        if self.number_ids is None:
            return 0, 0
        trow = out.target_rows
        firsts = []
        for b in range(len(group)):
            pos = (trow == b).nonzero().flatten()
            firsts.append(int(pos[0]))
        h = out.target_hidden.detach().index_select(0, torch.tensor(firsts, dtype=torch.long, device=out.target_hidden.device))
        lg = self.ad.restricted_logits(h, self.number_ids)
        pred = (lg.argmax(-1) + 1).tolist()
        return sum(int(p == e.info["label"]) for p, e in zip(pred, group)), len(group)

    def optimize(self, examples: Sequence[W2CExample], *, device: Any = None) -> tuple[dict[str, Any], dict[str, Any], list[Compute]]:
        if not examples:
            raise ValueError("这一步没有样本")
        dev = device if device is not None else self.ad.device
        names = stream_names(self.rc)  # 默认配置：恰是 STREAMS（DEV-0065 之前的两条流）
        order = stream_order(self.rc)
        micro = int(self.rc["micro_batch_size"])
        if any(e.plan.stream not in names for e in examples):
            raise ValueError(f"样本流 {sorted({e.plan.stream for e in examples} - set(names))} 不在本次的流 {list(names)} 里")
        by = {s: [e for e in examples if e.plan.stream == s] for s in names}
        n_by = {s: sum(e.n_supervised for e in by[s]) for s in names}
        if any(by[s] and n_by[s] <= 0 for s in names):
            raise ValueError("某种样本没有监督 token")
        weights = stream_weights(self.rc)
        chunks = [(s, by[s][i: i + micro]) for s in order for i in range(0, len(by[s]), micro)]
        chunk_size = int(self.rc["chunk_size"])
        self.opt.zero_grad(set_to_none=True)
        sums = {s: 0.0 for s in names}
        cnt_by = {s: {"n_samples": len(by[s]), "n_supervised_tokens": int(n_by[s]), "n_target_tokens": 0, "n_prompt_tokens": 0,
                      "n_visual_tokens": 0, "n_image_pixels": 0} for s in names}
        hits = [0, 0]
        times = {"build_inputs_s": 0.0, "fwd_s": 0.0, "bwd_s": 0.0, "opt_s": 0.0}
        computes: list[Compute] = []
        for key, group in chunks:
            t0 = time.perf_counter()
            inputs = [self.build_inputs(e) for e in group]
            _sync(dev)
            t1 = time.perf_counter()
            out = self.ad.training_forward(inputs, [e.target_ids for e in group], enable_thinking=bool(self.rc["enable_thinking"]))
            ls = self.ad.chunked_lm_loss(out.target_hidden, out.target_ids, chunk_size=chunk_size, reduction="sum")
            if key == "select":
                with torch.no_grad():
                    a, n = self._select_hits(out, group)
                hits[0] += a
                hits[1] += n
            _sync(dev)
            t2 = time.perf_counter()
            (ls / float(n_by[key]) * weights[key]).backward()
            _sync(dev)
            t3 = time.perf_counter()
            times["build_inputs_s"] += t1 - t0
            times["fwd_s"] += t2 - t1
            times["bwd_s"] += t3 - t2
            sums[key] += float(ls.detach())
            share = (t3 - t1) / len(group)
            c = cnt_by[key]
            for b, (e, inp) in enumerate(zip(group, inputs)):
                nv = sum(len(i.span) for i in inp.images)
                npx = sum(int(i.n_pixels or 0) for i in inp.images)
                c["n_target_tokens"] += int(out.target_lengths[b])
                c["n_prompt_tokens"] += int(out.prompt_lengths[b])
                c["n_visual_tokens"] += int(nv)
                c["n_image_pixels"] += int(npx)
                computes.append(Compute.call(generated_tokens=int(out.target_lengths[b]), prompt_tokens=int(out.prompt_lengths[b]),
                                             visual_tokens=int(nv), image_pixels=int(npx), call_time_s=share, new_round=True))
            del out, ls
        t4 = time.perf_counter()
        params = [p for g in self.opt.param_groups for p in g["params"]]
        mx = self.rc["optim"]["max_grad_norm"]
        gn = torch.nn.utils.clip_grad_norm_(params, float("inf") if mx is None else float(mx))
        lr = self.lr_now()
        self.opt.step()
        self.sched.step()
        self.opt.zero_grad(set_to_none=True)
        _sync(dev)
        times["opt_s"] = time.perf_counter() - t4
        loss_by = {s: (sums[s] / n_by[s] if n_by[s] else None) for s in names}
        loss = sum(weights[s] * loss_by[s] for s in order if loss_by[s] is not None)
        tot = {k: int(sum(cnt_by[s][k] for s in names)) for k in ("n_target_tokens", "n_prompt_tokens", "n_visual_tokens",
                                                                    "n_image_pixels")}
        rec = {"lr": lr, "loss": loss, "loss_by_stream": loss_by, "n_supervised_by_stream": {s: int(n_by[s]) for s in names},
               "n_samples": len(examples), "n_by_stream": {s: len(by[s]) for s in names},
               "n_supervised_tokens": int(sum(n_by.values())), **tot, "grad_norm": float(gn),
               "select_number_acc": (hits[0] / hits[1]) if hits[1] else None, "by_stream": cnt_by}
        return rec, times, computes


# ---------------------------------------------------------------------------
# 运行身份与输出
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class W2CRunSpec:
    model_key: str
    family: str
    repo: str
    revision: str
    adapter_model_key: str
    tiny: bool
    image_policy: str
    model_dtype: str
    seed: int

    @property
    def variant(self) -> str:
        return VARIANT


def run_identity(spec: W2CRunSpec, cfg: W2CConfig, rc: Mapping[str, Any], trainer: W2CTrainer, builder: W2CBuilder, data: W2CData,
                 w2_info: Mapping[str, Any], replay: RP.ReplayData | None = None, x2: XD.X2Data | None = None) -> dict[str, Any]:
    """运行身份（config.yaml）。replay（DEV-0065）给了才加 replay 一节、x2（DEV-0069）给了才加 x2 一节；都不给时与以前逐项相同。"""
    ident = {
        "experiment": "W2-c",
        "framework": "src/w2/w2c_train.py（DEV-0061；docs/w2_plan.md §5）；训练器 src/e3/train.Trainer",
        "model": {"key": spec.model_key, "family": spec.family, "repo": spec.repo, "revision": spec.revision,
                  "adapter_model_key": spec.adapter_model_key, "tiny": spec.tiny, "dtype": spec.model_dtype},
        "image_policy": spec.image_policy,
        "supervision": SUPERVISION,
        "variant": VARIANT,
        "seed": spec.seed,
        "training": dict(rc),
        "stop_id": trainer.stop_id,
        "config": {"path": cfg.path, "sha256": cfg.sha256, "version": cfg.doc["version"]},
        "w2_config": dict(w2_info),
        "impl": {"candidates_module": builder.impl.candidates_module, "render_module": builder.impl.render_module},
        "propose": {"box_template": builder.box_template, "box_template_source": builder.box_source, "order": builder.orders["propose"],
                    "box_format": "adapter.format_boxes（本家族原生格式，label = 指代句；与 C3 相同，src/e3/c3.box_target）",
                    "rows": "全部行"},
        "select": {"prompt": w2_info["select_prompt"], "order": builder.orders["select"], "target": "str(label)（十进制）+ 回合结束 token",
                   "render": "src/w2/loop.selection_candidates（1 号 ok = False 画空）→ render_selection(原图, 候选, layout, style)",
                   "rows": "candidates 与 label 都不为 null 的行", "number_token_ids": trainer.number_ids},
        "data": data.identity(),
    }
    if replay is not None:
        ident["replay"] = {**replay.identity(), "streams": {k: dict(v) for k, v in rc["replay"]["streams"].items()},
                           "sample": "提示、图（计数：原图 load_pil）、order 取回放数据的行原样；目标 = base 生成的 token 原样（含停止 token）",
                           "loss": "每条回放流各自除以这一步里自己的监督 token 数，再乘 loss_weight，与提出、挑选两项相加"}
    if x2 is not None and "correct" in rc["x2"]["streams"]:
        ident["x2"] = {**x2.identity(), "streams": {k: dict(v) for k, v in rc["x2"]["streams"].items()},
                       "prompts": {"version": XP.PROMPT_VERSION, "spline": XP.SPLINE_TEMPLATE, "correct": XP.CORRECT_TEMPLATE},
                       "render": {"version": XR.RENDER_VERSION, "module": "src/explore/x2_render.render_overlay", "box_rgb": list(XR.BOX_RGB),
                                  "box_px": XR.BOX_PX, "style": "冻结的 W2 配置的 render.style（w2_config 一节）"},
                       "sample": "描样条：原图 + spline_prompt → 控制点 JSON；指修正框：叠加图（起点 mask + 提出的框，训练时现画）+ correct_prompt → "
                                 "修正框 JSON 或 []；目标加回合结束 token，全部监督",
                       "loss": "两条流各自除以这一步里自己的监督 token 数，再乘 loss_weight，与其余各项相加"}
    elif x2 is not None:  # DEV-0070：只有样条一条流（没有修正：提示只有描样条的，没有叠加图）
        ident["x2"] = {**x2.identity(), "streams": {k: dict(v) for k, v in rc["x2"]["streams"].items()},
                       "prompts": {"version": XP.PROMPT_VERSION, "spline": XP.SPLINE_TEMPLATE},
                       "sample": "描样条：原图 + spline_prompt → 控制点 JSON；目标加回合结束 token，全部监督。没有修正流（DEV-0070：配置没有 correct 一节，"
                                 "X-2 数据里的修正行不读）",
                       "loss": "样条流除以这一步里自己的监督 token 数，再乘 loss_weight，与其余各项相加"}
    return ident


def readme_text(spec: W2CRunSpec, ident: Mapping[str, Any], summary: Mapping[str, Any], log: Sequence[Mapping[str, Any]]) -> str:
    tr = ident["training"]
    st = tr["streams"]
    first, last = (log[0], log[-1]) if log else ({}, {})
    d = ident["data"]
    tot = {s: {k: int(sum(r["by_stream"][s][k] for r in log)) for k in ("n_samples", "n_supervised_tokens")}
           for s in stream_names(tr)} if log else {}
    w2 = ident["w2_config"]
    rp = ident.get("replay")
    replay_lines = [] if rp is None else [
        f"- 回放（DEV-0065，自蒸馏）：每步另加 {tr['replay']['per_step_total']} 条，"
        + "；".join(f"{s} {v['per_step']} 条（{'+'.join(v['kinds'])}，损失权重 {v['loss_weight']:g}，按本流的监督 token 数平均）"
                    for s, v in tr["replay"]["streams"].items())
        + f"（回放的权重合计 {sum(v['loss_weight'] for v in tr['replay']['streams'].values()):g}，提出、挑选各 1）"
        + f"；数据 manifest sha256 {rp['manifest_sha256'][:16]}…，由 {rp['model']['key']}@{str(rp['model']['revision'])[:12]} 贪心、关思考生成；"
        f"池 { {s: v['n'] for s, v in rp['pools'].items()} }（截断的不用）"]
    x2 = ident.get("x2")
    x2_lines = [] if x2 is None else [
        f"- X-2（DEV-0069，目标来自真值）：每步另加 {tr['x2']['per_step_total']} 条，"
        + "；".join(f"{s} {v['per_step']} 条（损失权重 {v['loss_weight']:g}，按本流的监督 token 数平均）" for s, v in tr["x2"]["streams"].items())
        + f"；参数 {x2['params']}；数据 manifest sha256 {x2['manifest_sha256'][:16]}…，池 { {s: v['n'] for s, v in x2['pools'].items()} }"
        + ("" if "correct" in tr["x2"]["streams"] else "；没有修正流（DEV-0070，X-3：只读 X-2 数据里的样条行）")]
    return "\n".join([
        f"# W2-c 训练 · {spec.model_key} · seed {spec.seed}", "",
        f"- 模型：{spec.repo}@{spec.revision}（{spec.family}；适配器 {spec.adapter_model_key}；{spec.model_dtype}）；图像口径 {spec.image_policy}",
        f"- 监督：只有文本——提出（E1 refexp_box 提示 + 原图 → GT 框，与 C3 的框监督逐字相同，order {ident['propose']['order']}）与挑选"
        f"（挑选图 + 挑选提示 → 编号，order {ident['select']['order']}）；关思考；不加任何新 token",
        f"- 训练：{tr['steps']} 步 × {tr['batch_size']} 条（提出 {st['per_step']['propose']} + 挑选 {st['per_step']['select']}；微批 "
        f"{tr['micro_batch_size']} × 累积 {tr['grad_accum_steps']}，每个微批只含一种样本，先后 {st['order']}）；损失 = "
        f"{st['loss_weights']['propose']:g} × 提出的逐 token 平均 + {st['loss_weights']['select']:g} × 挑选的逐 token 平均",
        f"- LoRA r {tr['lora']['r']} / alpha {tr['lora']['alpha']} / dropout {tr['lora']['dropout']}，AdamW lr {tr['optim']['lora']['lr']}，"
        f"warmup {tr['warmup_steps']} 步 + 余弦（同 C3）",
        *x2_lines,
        *replay_lines,
        f"- W2 配置：{w2['config']} sha256 {w2['config_sha256'][:16]}…（frozen = {w2['frozen']}；K = {w2['k']}，layout = {w2['layout']}）",
        f"- 数据：{len(d['shards'])} 个分片，{d['n_rows']} 句（提出池 {d['n_propose']}，挑选池 {d['n_select']}；提出失败 {d['n_proposal_failed']}）；"
        f"标签分布 {d['label_counts']}；manifest sha256 {d['manifest_sha256']}",
        f"- 第一步损失 {first.get('loss')}，最后一步损失 {last.get('loss')}（{len(log)} 步）；本次合计 {tot}",
        f"- 检查点：{summary.get('final_checkpoint')}（{summary.get('final_checkpoint_bytes')} 字节；只有 LoRA、tokens.json、manifest，与 C3 同格式）",
        "", "## 文件", "",
        "- train_log.jsonl：每步的确定性记录（loss、loss_by_stream、by_stream、select_number_acc……）；logs/：每次调用的记录与逐步计时；"
        "accounting.json、compute.json：记账；config.yaml：运行身份；meta.json：git、主机、包版本、数据分片的 sha256。", ""])


def run_w2c_training(adapter: Any, cfg: W2CConfig, spec: W2CRunSpec, run_cfg: Mapping[str, Any], data: W2CData, *, w2: W2Config,
                     w2_info: Mapping[str, Any], impl: Any, data_root: str | Path, out_dir: str | Path, resume: bool = False,
                     argv: Sequence[str] = (), ckpt_root: str | Path | None = None, stop_after_steps: int | None = None,
                     keep_optimizer_state: bool = False, log: Callable[[str], None] | None = None,
                     _fail_after_step: int | None = None, replay: RP.ReplayData | None = None,
                     x2: XD.X2Data | None = None) -> dict[str, Any] | None:
    """跑一次 W2-c 训练；返回 train_summary（stop_after_steps 到了时返回 None）。adapter 必须是刚挂好的 base。
    布局、续跑、分段运行、检查点、记账与 src/e3/train.run_training 逐条对应（模块说明）。
    replay（DEV-0065）：回放数据（src/w2/w2c_replay.load_replay_data），当且仅当 run_cfg 有 replay 一节时给；每步在提出、挑选之后
    另加回放流的样本（ReplaySchedule、ReplayBuilder），损失多一项（W2CTrainer.optimize）。不给时与以前逐字节相同。
    x2（DEV-0069）：X-2 的数据（src/explore/x2_data.load_x2_data），当且仅当 run_cfg 有 x2 一节时给；每步在提出、挑选之后、回放之前另加
    spline、correct 两条流（DEV-0070：或只有 spline 一条）的样本（X2Schedule、X2Builder）；数据读进来的类必须恰是 run_cfg 的流、参数
    等于 run_cfg 的 data_params。不给时与以前逐字节相同。"""
    log = log or (lambda m: print(m, file=sys.stderr, flush=True))
    rc = dict(run_cfg)
    if rc["supervision"] != SUPERVISION or rc["variant"] != VARIANT:
        raise ValueError(f"run_cfg（{rc['supervision']}/{rc['variant']}）不是 W2-c 的")
    if (replay is None) != (rc.get("replay") is None):
        raise ValueError("回放数据与 run_cfg 的 replay 一节必须同时有或同时没有（--replay-data 只配有 replay 一节的配置）")
    if (x2 is None) != (rc.get("x2") is None):
        raise ValueError("X-2 的数据与 run_cfg 的 x2 一节必须同时有或同时没有（--x2-data 只配有 spline 一节、可再有 correct 一节的配置）")
    if x2 is not None and x2.params != dict(rc["x2"]["data_params"]):  # x2.params：读进来的类的参数（两类都读时 = manifest 的 params）
        raise ValueError(f"X-2 数据的参数 {x2.params} 与 run_cfg 的 {rc['x2']['data_params']} 不同")
    leaf = leaf_dir(out_dir, VARIANT, spec.model_key, spec.seed)
    ckpt = (Path(ckpt_root) / VARIANT / spec.model_key / f"seed{spec.seed}") if ckpt_root else leaf / "ckpt"
    latest = ckpt / LATEST

    exists = leaf.exists() and any(leaf.iterdir())
    if exists and not resume:
        raise FileExistsError(f"{leaf} 已存在且非空；续跑请加 --resume")
    if exists and (leaf / SUMMARY).exists():
        s = json.loads((leaf / SUMMARY).read_text(encoding="utf-8"))
        if s.get("complete"):
            log(f"[{VARIANT} {spec.model_key} seed{spec.seed}] 已完成，跳过（{leaf / SUMMARY}）")
            return s

    encode = (lambda s: list(adapter.tokenizer(s, add_special_tokens=False)["input_ids"]))
    trainer = W2CTrainer(adapter, cfg, rc, seed=spec.seed, number_ids=number_token_ids(encode, w2.k))
    st = None
    if exists and (latest / RESUME_STATE).exists():
        trainer.setup_resume(latest / MODEL_SUB)
        st = torch.load(latest / RESUME_STATE, map_location="cpu", weights_only=False)
    else:
        trainer.setup_fresh()
    builder = W2CBuilder(cfg, data, w2=w2, impl=impl, data_root=data_root, stop_id=trainer.stop_id, encode=encode, family=spec.family,
                         format_boxes=adapter.format_boxes)
    rbuilder = rsched = None
    if replay is not None:
        if not spec.tiny and replay.config.get("image_policy") != spec.image_policy:
            raise ValueError(f"回放数据生成时的图像口径 {replay.config.get('image_policy')} 与训练的 {spec.image_policy} 不同")
        rbuilder = ReplayBuilder(replay, data_root=data_root, stop_ids=adapter.stop_token_ids())
    xbuilder = xsched = None
    if x2 is not None:
        xbuilder = X2Builder(x2, w2=w2, data_root=data_root, stop_id=trainer.stop_id, encode=encode, spec=rc["x2"])
    x2_total = 0 if x2 is None else int(rc["x2"]["per_step_total"])
    schedule = W2CSchedule(data, seed=spec.seed, per_step=rc["streams"]["per_step"], order=rc["streams"]["order"],
                           extra_per_step=(0 if replay is None else int(rc["replay"]["per_step_total"])) + x2_total)
    if schedule.batch_size != int(rc["batch_size"]):
        raise ValueError(f"W2-c 每步 {schedule.batch_size} 条，与 run_cfg 的 batch_size {rc['batch_size']} 不同")
    if x2 is not None:
        xsched = X2Schedule(x2, seed=spec.seed, spec=rc["x2"], base_batch=schedule.batch_size, step_size=schedule.step_size)
    if replay is not None:
        rsched = ReplaySchedule(replay, seed=spec.seed, spec=rc["replay"], base_batch=schedule.batch_size + x2_total)
    names = stream_names(rc)
    ident = run_identity(spec, cfg, rc, trainer, builder, data, w2_info, replay=replay, **({} if x2 is None else {"x2": x2}))
    ident_sha = identity_sha256(ident)
    if exists:
        if not (leaf / CONFIG).exists():
            raise FileExistsError(f"{leaf} 里没有 {CONFIG}，不能续跑")
        old = yaml.safe_load((leaf / CONFIG).read_text(encoding="utf-8"))
        if _identity(old) != _identity(ident):
            diff = sorted(k for k in set(_identity(old)) | set(_identity(ident)) if _identity(old).get(k) != _identity(ident).get(k))
            raise ValueError(f"{leaf}：续跑的运行身份与已有的不同（{diff}），拒绝混写")
    else:
        leaf.mkdir(parents=True, exist_ok=True)
        files = [REPO_ROOT / cfg.path if not Path(cfg.path).is_absolute() else Path(cfg.path), Path(w2.path), *data.files(),
                 *([] if replay is None else replay.file_paths()), *([] if x2 is None else x2.file_paths())]
        write_run_metadata(leaf, config=ident, seed=spec.seed, data_files=files, repo_root=REPO_ROOT)
    start = 0
    if st is not None:
        if st["identity_sha256"] != ident_sha:
            raise ValueError(f"{latest / RESUME_STATE} 的运行身份与本次不同，拒绝续跑")
        trainer.load_state_dict(st)
        start = int(st["step"])
        log(f"[{VARIANT} {spec.model_key} seed{spec.seed}] 从第 {start} 步续跑")
    tl = [r for r in _read_log(leaf / TRAIN_LOG) if r["step"] < start]
    _atomic_write_text(leaf / TRAIN_LOG, "".join(_jsonl(r) for r in tl))

    (leaf / "logs").mkdir(exist_ok=True)
    dev = torch.device(adapter.device)
    run_k = len(list((leaf / "logs").glob("run_*.json"))) + 1
    run_path = leaf / "logs" / f"run_{run_k:03d}.json"
    timing_path = leaf / "logs" / f"timing_run_{run_k:03d}.jsonl"
    run_log: dict[str, Any] = {"argv": list(argv), "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                               "device": str(dev), "gpu": torch.cuda.get_device_name(dev) if dev.type == "cuda" else None,
                               "paths": {"w2c_data": str(data.data_dir), "data_root": str(data_root), "w2_config": str(w2.path)},
                               "ckpt_dir": str(ckpt), "start_step": start, "finished_utc": None, "audit": trainer.audit}
    if replay is not None:
        run_log["paths"]["replay_data"] = str(replay.run_dir)
    if x2 is not None:
        run_log["paths"]["x2_data"] = str(x2.run_dir)
    run_path.write_text(json.dumps(run_log, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    t_all = time.perf_counter()
    done_box = [0]

    def finish_log(status: str) -> None:
        _sync(dev)
        run_log.update(status=status, finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                       wall_s=round(time.perf_counter() - t_all, 3), steps_this_run=done_box[0])
        run_path.write_text(json.dumps(run_log, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        acc, comp = accounting_doc(leaf)
        tlog = _read_log(leaf / TRAIN_LOG)
        if tlog:
            acc["by_stream"] = {s: {k: int(sum(r["by_stream"][s][k] for r in tlog)) for k in tlog[0]["by_stream"][s]} for s in names}
        (leaf / ACCOUNTING).write_text(json.dumps(acc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        write_compute_json(leaf / COMPUTE, comp)

    def save_latest(step_done: int) -> None:
        tmp = ckpt / (LATEST + ".tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        adapter.save_trained(tmp / MODEL_SUB)
        torch.save({"identity_sha256": ident_sha, "step": step_done, **trainer.state_dict()}, tmp / RESUME_STATE)
        old = ckpt / (LATEST + ".old")
        if old.exists():
            shutil.rmtree(old)
        if latest.exists():
            latest.replace(old)
        tmp.replace(latest)
        if old.exists():
            shutil.rmtree(old)

    steps = int(rc["steps"])
    every = int(rc["ckpt_every"])
    workers = int(rc["data_workers"])
    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="w2c-data") if workers > 0 else None

    def build(plan: W2CPlan) -> W2CExample:
        if xbuilder is not None and plan.stream in rc["x2"]["streams"]:
            return xbuilder.build(plan)
        return rbuilder.build(plan) if rbuilder is not None and plan.stream in rc["replay"]["streams"] else builder.build(plan)

    def submit(step: int) -> list[Any]:
        plans = (schedule.step_plans(step) + ([] if xsched is None else xsched.step_plans(step))
                 + ([] if rsched is None else rsched.step_plans(step)))
        return [pool.submit(build, p) for p in plans] if pool is not None else plans

    def collect(handle: list[Any]) -> list[W2CExample]:
        return [h.result() for h in handle] if pool is not None else [build(p) for p in handle]

    try:
        adapter.train_mode()
        done = 0
        pending = submit(start) if start < steps else None
        for step in range(start, steps):
            if dev.type == "cuda":
                torch.cuda.reset_peak_memory_stats(dev)
            t0 = time.perf_counter()
            examples = collect(pending)
            pending = submit(step + 1) if step + 1 < steps else None
            t_data = time.perf_counter() - t0
            rec, times, comps = trainer.optimize(examples, device=dev)
            step_s = time.perf_counter() - t0
            n = len(examples)
            comps = [c.as_sample(wall_time_s=step_s / n) for c in comps]
            if step == start:
                trainer.audit["first_step_examples"] = [
                    {"id": e.plan.sample_id, "stream": e.plan.stream, "n_images": len(e.images), "n_target": len(e.target_ids),
                     "target_text": e.target_text} for s in names for e in [x for x in examples if x.plan.stream == s][:4]]
            _append(leaf / TRAIN_LOG, [_jsonl({"step": step, **rec})])
            _append(timing_path, [_jsonl({"step": step, "data_s": t_data, **times, "step_s": step_s,
                                          "peak_mem_bytes": int(torch.cuda.max_memory_allocated(dev)) if dev.type == "cuda" else None,
                                          "compute": {k: v for k, v in merge(comps).to_json_dict().items()
                                                      if k not in ("schema_version", "per_sample_mean")}})])
            done += 1
            done_box[0] = done
            s1 = step + 1
            if s1 % 10 == 0 or s1 == steps:
                extra = "".join(f"，{s} {rec['loss_by_stream'][s]}" for s in names[len(STREAMS):])
                log(f"[{VARIANT} {spec.model_key} seed{spec.seed}] 第 {s1}/{steps} 步 loss {rec['loss']:.4f}（提出 "
                    f"{rec['loss_by_stream']['propose']}，挑选 {rec['loss_by_stream']['select']}{extra}；{step_s:.2f} s/步）")
            stop = stop_after_steps is not None and done >= stop_after_steps and s1 < steps
            if (s1 % every == 0 and s1 < steps) or stop:
                save_latest(s1)
            if _fail_after_step is not None and s1 == _fail_after_step:
                raise RuntimeError(f"测试注入的中断（第 {s1} 步之后）")
            if stop:
                raise StopRequested(f"本次跑了 {done} 步（到第 {s1}/{steps} 步），检查点已写")
        adapter.eval_mode()
        final = ckpt / FINAL / MODEL_SUB
        if final.exists():
            shutil.rmtree(final)
        final.parent.mkdir(parents=True, exist_ok=True)
        adapter.save_trained(final)
        if not keep_optimizer_state and latest.exists():
            shutil.rmtree(latest)
        man = json.loads((final / "manifest.json").read_text(encoding="utf-8"))
        tlog = _read_log(leaf / TRAIN_LOG)
        summary = {"complete": True, "steps": steps, "variant": VARIANT, "model": spec.model_key, "seed": spec.seed,
                   "identity_sha256": ident_sha, "final_loss": tlog[-1]["loss"] if tlog else None,
                   "final_loss_by_stream": tlog[-1]["loss_by_stream"] if tlog else None,
                   "final_checkpoint": str(final.relative_to(leaf)) if final.is_relative_to(leaf) else str(final),
                   "final_checkpoint_bytes": _dir_bytes(final),
                   "final_checkpoint_files": {k: v["bytes"] for k, v in man["files"].items()},
                   "lora": man.get("lora"), "trainable_token_rows": man.get("trainable_token_rows"),
                   "w2_config_sha256": w2.sha256,
                   "by_stream": {s: {k: int(sum(r["by_stream"][s][k] for r in tlog)) for k in tlog[0]["by_stream"][s]}
                                 for s in names} if tlog else None}
        if replay is not None:
            summary["replay"] = {"manifest_sha256": replay.manifest_sha256, "pools": {s: len(p) for s, p in replay.pools.items()}}
        if x2 is not None:
            summary["x2"] = {"manifest_sha256": x2.manifest_sha256, "pools": {s: len(p) for s, p in x2.pools.items()},
                             "params": x2.params}
        (leaf / README).write_text(readme_text(spec, ident, summary, tlog), encoding="utf-8")
        _atomic_write_text(leaf / SUMMARY, json.dumps(summary, indent=1, ensure_ascii=False) + "\n")
        finish_log("complete")
        return summary
    except StopRequested as e:
        log(f"[停止] {e}")
        adapter.eval_mode()
        finish_log("stopped")
        return None
    except BaseException:
        finish_log("error")
        raise
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)


__all__ = [
    "DATA_MANIFEST",
    "REPLAY_PREFIX",
    "ReplayBuilder",
    "ReplaySchedule",
    "STREAMS",
    "SUPERVISION",
    "VARIANT",
    "W2CBuilder",
    "W2CConfig",
    "W2CConfigError",
    "W2CData",
    "W2CDataError",
    "W2CExample",
    "W2CPlan",
    "W2CRunSpec",
    "W2CSchedule",
    "W2CTrainer",
    "W2C_CONFIG_PATH",
    "W2C_REPLAY_CONFIG_PATH",
    "X2Builder",
    "X2Schedule",
    "X2_CONFIG_PATH",
    "X3_CONFIG_PATH",
    "X2_STREAMS",
    "X2_STREAM_INDEX",
    "check_label",
    "load_pil",
    "load_w2c_config",
    "load_w2c_data",
    "number_token_ids",
    "pool_position",
    "readme_text",
    "replay_spec",
    "resolve_w2_config",
    "resolve_w2c_run",
    "row_problems",
    "run_identity",
    "run_w2c_training",
    "selection_row_view",
    "stream_names",
    "stream_order",
    "stream_weights",
    "streams_spec",
    "x2_spec",
]
