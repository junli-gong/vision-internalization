"""X-2 的推理运行器（DEV-0069；探索 X-2；预注册 X）：一个检查点（scripts/train_w2c.py --config configs/x2_train.json 的输出），几条管线，
每条一个设置名；输出 W2 格式（<out>/<模型>/refseg/<设置>/<划分>/ 的 predictions.jsonl、metrics.json、compute.json、config.yaml、
meta.json），scripts/gate_explore.py 能直接读。参数在 configs/x2_run.json（第一个检查点的读数出来之前不调）。

**一个划分的流程**（run_split；模型只加载一次，提出、候选与 W2-c 同名、同规则，所以与 run_w2.py --setting w2c 共用同一个 --out 时直接复用）：
1. 提出：训练过的模型按 E1 refexp_box 协议出框（src/w2/w2c.ensure_proposal → <out>/<模型>/refexp_box/propose_w2c/<划分>）；
2. 候选：冻结的 W2 配置的 K = 4 个（src/w2/runner.ensure_candidates，cand_setting = candidates_w2c）；
3. 轨迹（<out>/<模型>/refseg/x2_trace/<划分>/trace.jsonl；每 chunk_size 句一组，没写完的那一组整组重跑）。每组依次：
   - 挑选（K = 4）：W2-c 的挑选原样（render_selection、select_prompt(4)、关思考，src/w2/loop.run_selection）；none 或解析失败交 1 号；
   - 描样条：原图 + x2_prompts.spline_prompt(指代句, 提出的框的网格整数, n) → 严格解析（截断记失败）→ 光栅化（x2_prompts.spline_mask，
     采样点夹在提出的框内）与 trimap（guidance.spline_grabcut(mode = trimap)）；
   - 椭圆（消融，不调模型）：提出的框的内切椭圆拟合成 n 个控制点的样条（网格整数），光栅化与 trimap 同上；
   - 挑选（K = 5）：4 个候选 + 第 5 个 = 样条的 trimap（样条解析失败时第 5 个画空）；提示与渲染沿用 W2（select_prompt(5)）；
   - 修正：两条轨迹，起点分别是 K = 4、K = 5 挑出的 mask。第 r 轮（r = 1..rounds）：叠加图（x2_render.render_overlay：当前 mask + 提出的框）
     + x2_prompts.correct_prompt → 严格解析 → guidance.correction_round（grabcut 规则、软锁定、effect gate、约束框 = 提出的框；承诺场随轨迹
     累积）。模型输出 [] 或解析失败就停（mask 不变）；
   - 随机框（消融，不调模型）：模型轨迹第 r 轮给了几个框，随机轨迹第 r 轮就用几个同大小、同符号、位置在提出的框里均匀抽的框，施加规则相同，
     停在同一轮。
4. 设置（derive；只读轨迹，不调模型，每次整份重写）：
   - x2_select：K = 4 的挑选（W2-c 的管线换成新检查点）；x2_select_cor{R}：它的修正轨迹的前 R 轮；
   - x2_spline_raster、x2_spline_trimap：样条直接光栅化、trimap；解析失败交 1 号候选（fallback = parse_failure）；
   - x2_select5：K = 5 的挑选（**挑选没在 K = 5 上训练过**）；x2_select5_cor{R}；
   - 诊断：x2_oracle4、x2_oracle5（用真值挑 4 个、5 个里 IoU 最高的；并列取编号小的）；
   - 消融：x2_select_cor{R}_randbox、x2_select5_cor{R}_randbox（随机框）；x2_ellipse_raster、x2_ellipse_trimap（椭圆代替样条）。
   提出失败（没有框）的句子在所有设置里都没有 mask（failure = 提出的失败原因，IoU 按空 mask 计，同 W2）。
5. 记账（硬约束 11，同 W2）：每句每个设置的模型调用次数（提出 1 + 用到的各阶段）、各阶段的 token（提示、生成、插入、视觉）、墙钟
   （模型调用 + 工具）；工具调用次数。随机框的消融记成与它所依的模型轨迹相同的调用（它用了模型的框数与大小）；椭圆只记提出。
   summary.json（<out>/<模型>/refseg/x2_summary/<划分>.json）列出各设置的 cIoU、gIoU、每句平均调用与 token，以及多轮设置与单轮设置
   的计算量配对（每句平均调用次数不超过它的单轮设置里最多的那个）。

**划分与守卫**：--splits-file configs/explore_splits.json 的 explore_val 三层，或 W2 的 devtune800、devcheck2；devcheck3 只经
--unseal-devcheck3 G-EX（src/data/explore_splits.require_unsealed）；test 一律拒绝（scripts/run_x2.py）。
检查点：<叶目录>/ckpt/final/model，叶目录的 config.yaml 必须有 x2 一节（n、m 取它的 params）、variant = w2c、训练用的 W2 配置与本次相同
（src/w2/w2c.load_checkpoint_adapter 的核对与合并核对）。

工具只用 src/explore/guidance（不含学习）；本模块不 import torch 等（tests/test_explore_guidance.py 扫描 src/explore 的源码）。工具一步可以
用 spawn 的进程池（tool_workers；子进程单线程），结果与单进程逐位相同。
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import yaml

from src.accounting import Compute, merge, write_compute_json
from src.data.records import Sample
from src.e1.posthoc_runner import _finish_run_log, _prepare_leaf, _resume_point, _start_run_log
from src.e1.prompts import PromptConfig
from src.e1.runner import CONFIG, METRICS, PREDICTIONS, README, TIMING, _append, _jsonl_line, batch_plan, generate_rows, load_images
from src.e1.runner import COMPUTE, read_jsonl_tolerant
from src.e1.scoring import distribution, gt_mask
from src.executors.raster import mask_to_rle, rle_to_mask
from src.explore import guidance as G
from src.explore import oracle_sim as OS
from src.explore import x2_prompts as XP
from src.explore import x2_render as XR
from src.explore.x2_tools import ToolPool, correction_tool_job, ellipse_ctrl_q, random_items, spline_tool_job
from src.metrics.seg import mask_inter_union, sample_iou
from src.parsing import parse_generation
from src.w1 import baselines as B
from src.w1.runner import model_record
from src.w1.scoring import headline
from src.w2 import runner as R
from src.w2.config import W2C_SETTING, W2Config
from src.w2.loop import SelectItem, candidates_digest_rows, effective_masks, oracle_index, run_selection

REPO_ROOT = Path(__file__).resolve().parents[2]
X2_RUN_CONFIG_PATH = REPO_ROOT / "configs" / "x2_run.json"
FRAMEWORK = "src/explore/x2_runner.py（DEV-0069；探索 X-2）"
TRACE_SETTING = "x2_trace"
SUMMARY_DIR = "x2_summary"
TRACE = "trace.jsonl"
TASK = "refseg"
BASES: tuple[str, ...] = ("select", "select5")
FIFTH_NAME = "spline_trimap"
STOP_EMPTY = "empty"
STOP_PARSE = "parse_failure"


class X2RunError(ValueError):
    """X-2 运行器的输入不合格（配置、检查点、划分）。"""


# ---------------------------------------------------------------------------
# 配置与设置
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class X2RunConfig:
    path: str
    sha256: str
    doc: Mapping[str, Any]

    @property
    def rounds(self) -> int:
        return int(self.doc["rounds"])

    def batching(self, *, tiny: bool) -> dict[str, int]:
        b = dict(self.doc["batching"])
        out = {"batch_size": int(b["batch_size"]), "chunk_size": int(b["chunk_size"]), "max_batch_tokens": int(b["max_batch_tokens"])}
        if tiny:
            out.update({k: int(v) for k, v in self.doc["tiny"].items()})
        return out

    def identity(self) -> dict[str, Any]:
        return {"path": self.path, "sha256": self.sha256, "version": self.doc["version"]}


def load_run_config(path: str | Path = X2_RUN_CONFIG_PATH) -> X2RunConfig:
    p = Path(path)
    raw = p.read_bytes()
    doc = json.loads(raw)
    bad = []
    if doc.get("schema") != 1:
        bad.append("schema 应为 1")
    if not isinstance(doc.get("rounds"), int) or isinstance(doc.get("rounds"), bool) or not 1 <= doc["rounds"] <= 8:
        bad.append("rounds 必须是 1..8 的整数")
    c = doc.get("correction") or {}
    if c.get("rule") not in G.CORRECTION_RULES or not isinstance(c.get("soft_lock"), bool) or c.get("gate") is not True:
        bad.append(f"correction 必须给 rule（{G.CORRECTION_RULES}）、soft_lock（bool）、gate = true")
    if c.get("early_stop") != "model_empty_or_parse_failure":
        bad.append("correction.early_stop 必须是 model_empty_or_parse_failure")
    if (doc.get("spline") or {}).get("trimap", {}).get("mode") not in G.SPLINE_GC_MODES or doc["spline"].get("failure_fallback") != "candidate_1":
        bad.append("spline.trimap.mode 必须是 guidance 的模式之一，failure_fallback 必须是 candidate_1")
    if (doc.get("select5") or {}).get("fifth") != FIFTH_NAME or doc["select5"].get("spline_failure") != "empty":
        bad.append(f"select5.fifth 必须是 {FIFTH_NAME}，spline_failure 必须是 empty")
    if not isinstance(((doc.get("ablations") or {}).get("randbox") or {}).get("salt"), str):
        bad.append("ablations.randbox.salt 必须是字符串")
    b = doc.get("batching") or {}
    if any(not isinstance(b.get(k), int) or isinstance(b.get(k), bool) or b[k] < 1 for k in ("batch_size", "chunk_size", "max_batch_tokens")):
        bad.append("batching 的 batch_size、chunk_size、max_batch_tokens 必须是正整数")
    if bad:
        raise X2RunError(f"{p} 不合格：\n- " + "\n- ".join(bad))
    rel = str(p.resolve().relative_to(REPO_ROOT)) if p.resolve().is_relative_to(REPO_ROOT) else str(p)
    return X2RunConfig(path=rel, sha256=hashlib.sha256(raw).hexdigest(), doc=doc)


def setting_defs(rounds: int) -> dict[str, dict[str, Any]]:
    """设置名 → 定义（kind、基础轨迹、轮数、是否诊断 / 消融、用到的模型阶段）。顺序即输出顺序。"""
    d: dict[str, dict[str, Any]] = {}

    def add(name, kind, stages, **kw):
        d[name] = {"kind": kind, "stages": list(stages), "multi_round": False, "diagnostic": False, "ablation": None, **kw}

    add("x2_select", "select", ["propose", "select4"])
    for r in range(1, rounds + 1):
        add(f"x2_select_cor{r}", "cor", ["propose", "select4", "cor_select"], base="select", rounds=r, multi_round=True)
    add("x2_spline_raster", "spline", ["propose", "spline"], mode="raster")
    add("x2_spline_trimap", "spline", ["propose", "spline"], mode="trimap")
    add("x2_select5", "select5", ["propose", "spline", "select5"], note="挑选没在 K = 5 上训练过")
    for r in range(1, rounds + 1):
        add(f"x2_select5_cor{r}", "cor", ["propose", "spline", "select5", "cor_select5"], base="select5", rounds=r, multi_round=True)
    add("x2_oracle4", "oracle", ["propose"], k=4, diagnostic=True)
    add("x2_oracle5", "oracle", ["propose", "spline"], k=5, diagnostic=True)
    for base, st in (("select", ["propose", "select4", "cor_select"]), ("select5", ["propose", "spline", "select5", "cor_select5"])):
        for r in range(1, rounds + 1):
            add(f"x2_{base}_cor{r}_randbox", "randbox", st, base=base, rounds=r, multi_round=True, ablation="randbox")
    add("x2_ellipse_raster", "ellipse", ["propose"], mode="raster", ablation="ellipse")
    add("x2_ellipse_trimap", "ellipse", ["propose"], mode="trimap", ablation="ellipse")
    return d


def checkpoint_x2_params(checkpoint: str | Path) -> dict[str, Any]:
    """检查点的训练叶目录（<叶>/ckpt/final/model）的 config.yaml 的 x2 一节 → {"n", "m", "params", "manifest_sha256"}。"""
    from src.e3.train import checkpoint_leaf

    leaf = checkpoint_leaf(checkpoint)
    if leaf is None or not (leaf / CONFIG).is_file():
        raise X2RunError(f"检查点 {checkpoint} 不在 <训练叶目录>/ckpt/final/model 下，或叶目录没有 {CONFIG}")
    doc = yaml.safe_load((leaf / CONFIG).read_text(encoding="utf-8")) or {}
    x = doc.get("x2")
    if not x or not isinstance(x.get("params"), Mapping):
        raise X2RunError(f"{leaf / CONFIG} 没有 x2 一节：这个检查点不是 X-2 的训练结果（configs/x2_train.json）")
    if (x.get("prompts") or {}).get("version") != XP.PROMPT_VERSION or (x.get("render") or {}).get("version") != XR.RENDER_VERSION:
        raise X2RunError(f"检查点训练时的提示 / 叠加图版本 {x.get('prompts', {}).get('version')} / {x.get('render', {}).get('version')} "
                         f"与本次的 {XP.PROMPT_VERSION} / {XR.RENDER_VERSION} 不同")
    return {"n": int(x["params"]["spline"]["n"]), "m": int(x["params"]["correct"]["m"]), "params": dict(x["params"]),
            "manifest_sha256": x.get("manifest_sha256")}


# ---------------------------------------------------------------------------
# 模型调用
# ---------------------------------------------------------------------------


def _call_record(res: Any, failure: Any) -> dict[str, Any]:
    px = getattr(res, "n_image_pixels", None)
    return {"text": res.text, "stopped": getattr(res, "stopped", None), "n_prompt_tokens": int(res.n_prompt_tokens),
            "n_generated_tokens": int(res.n_generated_tokens), "n_inserted_tokens": int(getattr(res, "n_inserted_tokens", 0) or 0),
            "n_visual_tokens": int(res.n_visual_tokens), "n_image_pixels": int(px or 0), "failure": XP.failure_name(failure)}


def _rejected_record(err: Exception) -> dict[str, Any]:
    return {"text": None, "stopped": None, "n_prompt_tokens": 0, "n_generated_tokens": 0, "n_inserted_tokens": 0, "n_visual_tokens": 0,
            "n_image_pixels": 0, "failure": "input_rejected", "input_error": f"{type(err).__name__}: {err}"}


def generate_parse(adapter: Any, items: Sequence[tuple[list, str]], parse_fn: Callable[..., Any], *, batch_size: int, max_new_tokens: int,
                   prompts: PromptConfig, **kw: Any) -> list[tuple[dict[str, Any], Any, float]]:
    """一组单轮生成（关思考、贪心、E1 的解码参数、prompts.order；build_inputs 拒收的记 input_rejected，不进批），按顺序每 batch_size 条
    一批；回答经 parse_generation（截断、空回答、思考未闭合判失败）与 parse_fn 解析。返回与 items 同序的 (调用记录, 解析值, 墙钟)。"""
    out: list[Any] = [None] * len(items)
    ready, inputs = [], []
    for i, (images, prompt) in enumerate(items):
        try:
            inputs.append(adapter.build_inputs(list(images), prompt, prompts.order))
            ready.append(i)
        except ValueError as e:
            out[i] = (_rejected_record(e), None, 0.0)
    for a in range(0, len(ready), batch_size):
        part = list(range(a, min(a + batch_size, len(ready))))
        res = generate_rows(adapter, [inputs[k] for k in part], batch_size=batch_size, enable_thinking=False, max_new_tokens=max_new_tokens,
                            decoding=prompts.decoding())
        if len(res) != len(part):
            raise RuntimeError(f"generate_batch 返回 {len(res)} 行，输入 {len(part)} 行")
        for k, r in zip(part, res):
            p = parse_generation(r, parse_fn, **kw)
            out[ready[k]] = (_call_record(r, p.failure), p.value, float(getattr(r, "wall_time_s", 0.0)))
    return out


# ---------------------------------------------------------------------------
# 轨迹（一组样本）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChunkContext:
    adapter: Any
    w2: W2Config
    rc: X2RunConfig
    impl: Any
    prompts: PromptConfig
    n: int
    m: int
    batch_size: int
    max_batch_tokens: int | None
    tiny: bool
    pool: ToolPool


def _select(ctx: ChunkContext, chunk: Sequence[Sample], images: Mapping[str, Any], masks: Mapping[str, list[np.ndarray]],
            names: Mapping[str, list[str]], oks: Mapping[str, list[bool]]) -> dict[str, Any]:
    """挑选（K = len(masks[id])）：W2-c 的挑选原样（关思考）。返回 {id: SelectOutcome}。"""
    protocol = ctx.w2.select_protocol_for(W2C_SETTING, tiny=ctx.tiny)
    items = []
    for s in chunk:
        if s.id not in masks:
            continue
        cands = [ctx.impl.Candidate(name=nm, mask=mk, ok=ok, info={}) for nm, mk, ok in zip(names[s.id], masks[s.id], oks[s.id])]
        imgs = ctx.impl.render(images[s.id], cands, layout=ctx.w2.layout, style=ctx.w2.style)
        items.append(SelectItem(s.id, tuple(imgs), ctx.w2.select_prompt(len(cands), s.fields["expression"]), len(cands)))
    outs = run_selection(ctx.adapter, items, protocol, system=ctx.w2.select_system, batch_size=ctx.batch_size,
                         max_batch_tokens=ctx.max_batch_tokens, decoding=ctx.prompts.decoding(), order=ctx.prompts.order)
    return {o.id: o for o in outs}


def _pick(outcome: Any) -> tuple[int, str | None]:
    """(交的编号, fallback)：none 或解析失败交 1 号（同 W2）。"""
    if outcome.pick is not None:
        return int(outcome.pick), None
    return 1, outcome.fallback


def process_chunk(ctx: ChunkContext, chunk: Sequence[Sample], cand_rows: Mapping[str, Mapping[str, Any]]) -> tuple[list[dict], list[dict]]:
    """一组样本的轨迹（模块说明第 3 条）。返回 (trace 的行, timing 的行)；行里不含墙钟（墙钟进 timing）。"""
    rc = ctx.rc.doc
    images = {s.id: load_images(s)[0] for s in chunk}
    paths = {s.id: str(s.images[0]) for s in chunk}
    tr: dict[str, dict[str, Any]] = {}
    tm: dict[str, dict[str, Any]] = {}
    cand_masks, cand_names, cand_ok = {}, {}, {}
    for s in chunk:
        row = cand_rows[s.id]
        prop = row["proposal"]
        w, h = row["image_size"]
        box = prop["box"]
        tr[s.id] = {"id": s.id, "box": box, "box_q": None if box is None else XP.box_q(box, w, h), "failure": prop["failure"]}
        tm[s.id] = {"id": s.id, "model_s": {}, "tool_s": {}}
        if row["candidates"] is not None:
            cand_masks[s.id] = effective_masks(row)
            cand_names[s.id] = [c["name"] for c in row["candidates"]]
            cand_ok[s.id] = [bool(c["ok"]) for c in row["candidates"]]
    live = [s for s in chunk if s.id in cand_masks]
    # 1. 挑选（K = 4）
    t0 = time.perf_counter()
    o4 = _select(ctx, live, images, cand_masks, cand_names, cand_ok)
    for s in live:
        tr[s.id]["select4"] = dict(o4[s.id].record)
        tm[s.id]["model_s"]["select4"] = float(o4[s.id].wall_time_s)
    # 2. 描样条（模型）与 3. 样条、椭圆的工具
    sp_items = [([images[s.id]], XP.spline_prompt(s.fields["expression"], tr[s.id]["box_q"], ctx.n)) for s in live]
    sp = generate_parse(ctx.adapter, sp_items, XP.parse_spline, batch_size=ctx.batch_size, max_new_tokens=XP.max_new_tokens_spline(ctx.n),
                        prompts=ctx.prompts, n=ctx.n)
    mode = rc["spline"]["trimap"]["mode"]
    jobs = []
    for s, (call, val, wall) in zip(live, sp):
        w, h = cand_rows[s.id]["image_size"]
        tr[s.id]["spline"] = {"call": call, "ctrl_q": val}
        tm[s.id]["model_s"]["spline"] = wall
        jobs.append((paths[s.id], tr[s.id]["box"], val, ellipse_ctrl_q(tr[s.id]["box"], w, h, ctx.n), mode))
    for s, job, res in zip(live, jobs, ctx.pool.map(spline_tool_job, jobs)):
        tm[s.id]["tool_s"]["spline_tools"] = res.pop("wall_s")
        if res["spline"] is not None:
            tr[s.id]["spline"].update(res["spline"])
        tr[s.id]["ellipse"] = {"ctrl_q": job[3], **res["ellipse"]}
    # 4. 挑选（K = 5）
    m5, n5, k5 = {}, {}, {}
    for s in live:
        spl = tr[s.id]["spline"]
        fifth = rle_to_mask(spl["trimap_rle"]).astype(bool) if spl.get("trimap_rle") else np.zeros_like(cand_masks[s.id][0])
        m5[s.id] = cand_masks[s.id] + [fifth]
        n5[s.id] = cand_names[s.id] + [FIFTH_NAME]
        k5[s.id] = cand_ok[s.id] + [bool(spl.get("trimap_rle"))]
    o5 = _select(ctx, live, images, m5, n5, k5)
    for s in live:
        tr[s.id]["select5"] = {**dict(o5[s.id].record), "fifth": FIFTH_NAME if tr[s.id]["spline"].get("trimap_rle") else "empty"}
        tm[s.id]["model_s"]["select5"] = float(o5[s.id].wall_time_s)
    # 5. 修正（两条轨迹）与随机框
    cor = rc["correction"]
    salt = rc["ablations"]["randbox"]["salt"]
    for base, outs, cands in (("select", o4, cand_masks), ("select5", o5, m5)):
        state = {}
        for s in live:
            idx, _ = _pick(outs[s.id])
            start = cands[s.id][idx - 1]
            state[s.id] = {"cur": start, "lock": None, "rcur": start, "rlock": None, "go": True}
            tr[s.id].setdefault("cor", {})[base] = []
            tr[s.id].setdefault("randbox", {})[base] = []
            tm[s.id]["model_s"][f"cor_{base}"] = []
            tm[s.id]["tool_s"][f"cor_{base}"] = []
            tm[s.id]["tool_s"][f"randbox_{base}"] = []
        for r in range(1, ctx.rc.rounds + 1):
            act = [s for s in live if state[s.id]["go"]]
            if not act:
                break
            items = []
            for s in act:
                ov = XR.render_overlay(images[s.id], state[s.id]["cur"], tr[s.id]["box"], style=ctx.w2.style)
                items.append(([ov], XP.correct_prompt(s.fields["expression"], tr[s.id]["box_q"], ctx.m)))
            res = generate_parse(ctx.adapter, items, XP.parse_corrections, batch_size=ctx.batch_size,
                                 max_new_tokens=XP.max_new_tokens_correct(ctx.m), prompts=ctx.prompts, m=ctx.m)
            apply, rand = [], []
            for s, (call, val, wall) in zip(act, res):
                tm[s.id]["model_s"][f"cor_{base}"].append(wall)
                rec = {"round": r, "call": call, "items": val, "stop": None, "info": None, "mask_rle": None}
                if val is None:
                    rec["stop"] = STOP_PARSE
                elif not val:
                    rec["stop"] = STOP_EMPTY
                if rec["stop"] is not None:
                    state[s.id]["go"] = False
                else:
                    apply.append((s, rec))
                    ri = random_items(val, tr[s.id]["box_q"], OS.rng_for(salt, s.id, base, r))
                    rand.append((s, ri))
                tr[s.id]["cor"][base].append(rec)
            jobs = [(paths[s.id], state[s.id]["cur"], state[s.id]["lock"],
                     XP.proposals_abs(rec["items"], *cand_rows[s.id]["image_size"]), tr[s.id]["box"], cor["rule"], cor["soft_lock"])
                    for s, rec in apply]
            jobs += [(paths[s.id], state[s.id]["rcur"], state[s.id]["rlock"], XP.proposals_abs(ri, *cand_rows[s.id]["image_size"]),
                      tr[s.id]["box"], cor["rule"], cor["soft_lock"]) for s, ri in rand]
            results = ctx.pool.map(correction_tool_job, jobs)
            for (s, rec), (new, lk, info, wall) in zip(apply, results[: len(apply)]):
                rec.update(info={k: int(v) for k, v in info.items()}, mask_rle=mask_to_rle(new))
                state[s.id].update(cur=new, lock=lk if cor["soft_lock"] else None)
                tm[s.id]["tool_s"][f"cor_{base}"].append(wall)
            for (s, ri), (new, lk, info, wall) in zip(rand, results[len(apply):]):
                tr[s.id]["randbox"][base].append({"round": r, "items": ri, "info": {k: int(v) for k, v in info.items()},
                                                  "mask_rle": mask_to_rle(new)})
                state[s.id].update(rcur=new, rlock=lk if cor["soft_lock"] else None)
                tm[s.id]["tool_s"][f"randbox_{base}"].append(wall)
    return [tr[s.id] for s in chunk], [tm[s.id] for s in chunk]


# ---------------------------------------------------------------------------
# 一个划分
# ---------------------------------------------------------------------------


def trace_identity(spec: R.W2RunSpec, cfg: W2Config, rc: X2RunConfig, splits_doc: Mapping[str, Any], samples: Sequence[Sample], out: str | Path,
                   *, prop_dir: Path, cand_leaf: Path, cand_rows: Sequence[Mapping[str, Any]], impl: Any, checkpoint: Mapping[str, Any],
                   x2: Mapping[str, Any], batching: Mapping[str, Any]) -> dict[str, Any]:
    tspec = dataclasses.replace(spec, setting=TRACE_SETTING)
    return {"experiment": "X-2", "framework": FRAMEWORK, "w2": R.w2_record(tspec, splits_doc, samples), "model": model_record(spec),
            "image_policy": dict(spec.image_policy), "w2_config": cfg.identity(),
            "propose": {"mode": "box", "run": R._run_ref(out, prop_dir)},
            "candidates": {"leaf": B._under_out(out, cand_leaf), "module": impl.candidates_module, "generators": list(cfg.generators),
                           "k": cfg.k, "content_sha256": candidates_digest_rows(cand_rows)},
            "render": {"select_module": impl.render_module, "layout": cfg.layout, "style": cfg.style, "overlay": XR.RENDER_VERSION},
            "checkpoint": json.loads(json.dumps(dict(checkpoint), ensure_ascii=False)),
            "x2": {"n": int(x2["n"]), "m": int(x2["m"]), "train_params": dict(x2["params"]), "prompts": XP.PROMPT_VERSION,
                   "run_config": rc.identity(), "rounds": rc.rounds},
            "batching": dict(batching)}


def run_trace(ctx: ChunkContext, spec: R.W2RunSpec, samples: Sequence[Sample], cand_rows: Mapping[str, Mapping[str, Any]], out: str | Path,
              *, ident: Mapping[str, Any], files: Sequence[str | Path], chunk_size: int, resume: bool, argv: Sequence[str] = (),
              log: Callable[[str], None] | None = None) -> tuple[Path, list[dict], dict[str, dict]]:
    """轨迹（模块说明第 3 条），按 chunk_size 一组，续跑同 W2。返回 (目录, trace 的行, {id: timing 的行})。"""
    log = log or (lambda m: print(m, file=sys.stderr, flush=True))
    leaf = R.w2_leaf(out, spec.model_key, TASK, TRACE_SETTING, spec.split)
    _prepare_leaf(leaf, ident, files, resume=resume)
    ids = [s.id for s in samples]
    plan = batch_plan([None] * len(ids), chunk_size)
    what = f"X-2 {spec.model_key} {TRACE_SETTING}/{spec.split}"
    keep, trimmed = _resume_point(leaf / TRACE, ids, lambda r: r["id"], plan, log, what)
    todo = [(a, b) for a, b in plan if a >= keep]
    run_k, run_path, run_log = _start_run_log(leaf, argv, n_done_before=keep, n_remaining=len(ids) - keep, chunk_size=chunk_size,
                                              tool_workers=ctx.pool.workers, trimmed_to_batch_boundary=trimmed)
    t_all = time.perf_counter()
    for n_chunk, (a, b) in enumerate(todo):
        recs, tms = process_chunk(ctx, samples[a:b], cand_rows)
        _append(leaf / TIMING, [_jsonl_line({**t, "run": run_k, "chunk_index": n_chunk}) for t in tms])
        _append(leaf / TRACE, [_jsonl_line(r) for r in recs])
        log(f"[{what}] {b}/{len(ids)}")
    _finish_run_log(run_path, run_log, n_batches=len(todo), t_all=t_all)
    recs = read_jsonl_tolerant(leaf / TRACE, repair=False)
    if [r["id"] for r in recs] != ids:
        raise RuntimeError(f"{leaf / TRACE} 的 id 与样本不同")
    timing = {t["id"]: t for t in read_jsonl_tolerant(leaf / TIMING, repair=False)}
    return leaf, recs, timing


# ---------------------------------------------------------------------------
# 设置（只读轨迹）
# ---------------------------------------------------------------------------


def _mask(rle: Mapping[str, Any] | None) -> np.ndarray | None:
    return None if rle is None else rle_to_mask(rle).astype(bool)


def _cor_final(rounds: Sequence[Mapping[str, Any]], R_: int, start: np.ndarray) -> tuple[np.ndarray, int, int]:
    """修正轨迹的前 R 轮 → (mask, 用了几次模型调用, 施加了几轮)。"""
    calls = rounds[:R_]
    applied = [r for r in calls if r.get("mask_rle") is not None]
    return (_mask(applied[-1]["mask_rle"]) if applied else start), len(calls), len(applied)


def final_of(setting: str, d: Mapping[str, Any], t: Mapping[str, Any], row: Mapping[str, Any], gt: np.ndarray) -> dict[str, Any]:
    """一句在一个设置下交的 mask 与说明：{"mask"（None = 没有 mask）, "fallback", "failure", "calls"（[(阶段, 第几次)]）, "info"}。"""
    if t["failure"] is not None or row["candidates"] is None:
        return {"mask": None, "fallback": None, "failure": t["failure"] or "no_candidates", "calls": [("propose", 0)], "info": {}}
    cands = effective_masks(row)
    kind = d["kind"]
    calls = [("propose", 0)]
    info: dict[str, Any] = {}
    fb = None
    spl = t["spline"]
    if kind in ("spline", "ellipse"):
        src = spl if kind == "spline" else t["ellipse"]
        if kind == "spline":
            calls.append(("spline", 0))
        if src.get("raster_rle") is None:
            mask, fb = cands[0], "parse_failure"
            info["spline_failure"] = spl["call"]["failure"]
        else:
            mask = _mask(src["raster_rle"] if d["mode"] == "raster" else src["trimap_rle"])
            if d["mode"] == "trimap":
                info["trimap_ok"] = src["trimap_ok"]
        return {"mask": mask, "fallback": fb, "failure": None, "calls": calls, "info": info}
    if kind == "oracle":
        ms = cands + ([_mask(spl["trimap_rle"]) if spl.get("trimap_rle") else np.zeros_like(cands[0])] if d["k"] == 5 else [])
        if d["k"] == 5:
            calls.append(("spline", 0))
        st = [mask_inter_union(m, gt) for m in ms]
        best = oracle_index([(int(s.inter), int(s.union)) for s in st])
        return {"mask": ms[best - 1], "fallback": None, "failure": None, "calls": calls, "info": {"best_index": best}}
    base = d.get("base", "select" if kind == "select" else "select5")
    if base == "select":
        rec, ms = t["select4"], cands
        calls.append(("select4", 0))
    else:
        rec = t["select5"]
        ms = cands + [_mask(spl["trimap_rle"]) if spl.get("trimap_rle") else np.zeros_like(cands[0])]
        calls += [("spline", 0), ("select5", 0)]
        info["fifth"] = rec["fifth"]
    ans = rec["answer"]
    idx = ans if isinstance(ans, int) else 1
    fb = None if isinstance(ans, int) else ("none" if ans == "none" else "parse_failure")
    info.update(pick=idx, k=len(ms))
    start = ms[idx - 1]
    if kind in ("select", "select5"):
        return {"mask": start, "fallback": fb, "failure": None, "calls": calls, "info": info}
    rounds = t["cor"][base]
    if kind == "cor":
        mask, n_calls, n_applied = _cor_final(rounds, d["rounds"], start)
        stops = [r["stop"] for r in rounds[: d["rounds"]] if r["stop"] is not None]
    else:  # randbox：模型轨迹前 R 轮里施加了几轮，随机轨迹就取几轮
        _, n_calls, n_applied = _cor_final(rounds, d["rounds"], start)
        rb = t["randbox"][base][:n_applied]
        mask = _mask(rb[-1]["mask_rle"]) if rb else start
        stops = [r["stop"] for r in rounds[: d["rounds"]] if r["stop"] is not None]
    calls += [(f"cor_{base}", k) for k in range(n_calls)]
    info.update(rounds_called=n_calls, rounds_applied=n_applied, stop=stops[0] if stops else None,
                round_failures=[r["call"]["failure"] for r in rounds[: d["rounds"]] if r["call"]["failure"]])
    return {"mask": mask, "fallback": fb, "failure": None, "calls": calls, "info": info}


def _call_of(stage: str, k: int, t: Mapping[str, Any], prop: Mapping[str, Any]) -> Mapping[str, Any]:
    if stage == "propose":
        return prop
    if stage == "select4":
        return t["select4"]
    if stage == "select5":
        return t["select5"]
    if stage == "spline":
        return t["spline"]["call"]
    base = stage[len("cor_"):]
    return t["cor"][base][k]["call"]


def _wall_of(stage: str, k: int, tm: Mapping[str, Any], prop_wall: float) -> float:
    if stage == "propose":
        return prop_wall
    v = tm["model_s"].get(stage, 0.0)
    return float(v[k]) if isinstance(v, list) else float(v)


def _tool_wall(setting_def: Mapping[str, Any], f: Mapping[str, Any], tm: Mapping[str, Any], cand_wall: float) -> float:
    w = cand_wall
    ts = tm.get("tool_s", {})
    if any(st in setting_def["stages"] for st in ("spline",)) or setting_def["kind"] == "ellipse":
        w += float(ts.get("spline_tools", 0.0))
    if setting_def["kind"] in ("cor", "randbox"):
        key = f"{'cor' if setting_def['kind'] == 'cor' else 'randbox'}_{setting_def['base']}"
        w += float(sum(ts.get(key, [])[: f["info"].get("rounds_applied", 0)]))
    return w


def setting_record(name: str, d: Mapping[str, Any], sample: Sample, t: Mapping[str, Any], tm: Mapping[str, Any], row: Mapping[str, Any],
                   prop: Mapping[str, Any], prop_wall: float, cand_wall: float, split: str) -> tuple[dict[str, Any], Compute]:
    """predictions.jsonl 的一行（W2 格式：id、score 的 inter / union / iou……；不含墙钟）与这句的计算量。"""
    gt = gt_mask(sample)
    f = final_of(name, d, t, row, gt)
    st = mask_inter_union(f["mask"] if f["failure"] is None else None, gt)
    calls = [_call_of(s, k, t, prop) for s, k in f["calls"]]

    def tok(c: Mapping[str, Any], key: str) -> int:  # 预检拒收（input_rejected）的调用没有 token 数：记 0
        return int(c.get(key) or 0)

    inserted = int(sum(tok(c, "n_inserted_tokens") for c in calls))
    tokens = {"prompt": int(sum(tok(c, "n_prompt_tokens") for c in calls)), "generated": int(sum(tok(c, "n_generated_tokens") for c in calls)),
              "inserted": inserted, "visual": int(sum(tok(c, "n_visual_tokens") for c in calls)),
              "image_pixels": int(sum(tok(c, "n_image_pixels") for c in calls))}
    by_stage: dict[str, dict[str, int]] = {}
    for (s, _), c in zip(f["calls"], calls):
        b = by_stage.setdefault(s, {"calls": 0, "prompt": 0, "generated": 0, "visual": 0})
        b["calls"] += 1
        b["prompt"] += tok(c, "n_prompt_tokens") + tok(c, "n_inserted_tokens")
        b["generated"] += tok(c, "n_generated_tokens")
        b["visual"] += tok(c, "n_visual_tokens")
    n_tools = 0 if f["failure"] is not None else 1 + (1 if ("spline" in d["stages"] or d["kind"] == "ellipse") else 0) + \
        int(f["info"].get("rounds_applied", 0))
    rec = {"id": sample.id, "w2_split": split, "source": sample.source, "split": sample.split, "setting": name, "task": TASK,
           "instruction": sample.fields["expression"], "image_size": list(row["image_size"]), "proposal": dict(row["proposal"]),
           "failure": f["failure"], "fallback": f["fallback"], "info": f["info"],
           "answer": None if f["mask"] is None or f["failure"] is not None else
           {"kind": "mask", "mask": {"rle": mask_to_rle(f["mask"]), "area": int(np.count_nonzero(f["mask"]))}},
           "tokens": tokens, "tokens_by_stage": by_stage, "n_model_calls": len(calls), "n_tool_calls": n_tools,
           "score": {"inter": st.inter, "union": st.union, "iou": sample_iou(*st)}}
    call_s = float(sum(_wall_of(s, k, tm, prop_wall) for s, k in f["calls"]))
    comp = Compute(n_calls=len(calls), n_rounds=len(calls), generated_tokens=tokens["generated"], prompt_tokens=tokens["prompt"] + inserted,
                   visual_tokens=tokens["visual"], image_pixels=tokens["image_pixels"], call_time_s=call_s
                   ).as_sample(wall_time_s=call_s + _tool_wall(d, f, tm, cand_wall))
    return rec, comp


def aggregate(name: str, d: Mapping[str, Any], recs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    from collections import Counter

    def cnt(vals):
        c = Counter("ok" if v is None else str(v) for v in vals)
        return {k: c[k] for k in sorted(c)}

    m: dict[str, Any] = {"n": len(recs), "headline": headline(TASK, recs) if recs else [], "failures": cnt(r["failure"] for r in recs),
                         "fallback": cnt(r["fallback"] for r in recs), "n_model_calls": distribution(r["n_model_calls"] for r in recs),
                         "n_tool_calls": distribution(r["n_tool_calls"] for r in recs)}
    for key in ("prompt", "generated", "inserted", "visual"):
        m.setdefault("tokens", {})[key] = distribution(r["tokens"][key] for r in recs)
    if d["kind"] in ("cor", "randbox"):
        ok = [r for r in recs if r["failure"] is None]
        m["rounds_called"] = distribution(r["info"]["rounds_called"] for r in ok)
        m["rounds_applied"] = distribution(r["info"]["rounds_applied"] for r in ok)
        m["stop"] = cnt(r["info"]["stop"] for r in ok)
    if d["kind"] in ("spline", "ellipse"):
        m["spline_failure"] = cnt(r["info"].get("spline_failure") for r in recs if r["failure"] is None)
    return m


def setting_identity(trace_ident: Mapping[str, Any], name: str, d: Mapping[str, Any], spec: R.W2RunSpec, splits_doc: Mapping[str, Any],
                     samples: Sequence[Sample], trace_leaf: Path, out: str | Path) -> dict[str, Any]:
    sspec = dataclasses.replace(spec, setting=name)
    doc = {k: v for k, v in trace_ident.items() if k not in ("w2", "batching")}
    doc["w2"] = R.w2_record(sspec, splits_doc, samples)
    doc["x2"] = {**dict(trace_ident["x2"]), "setting": name, "definition": dict(d),
                 "trace": {"leaf": B._under_out(out, trace_leaf), "config_sha256": R._sha(trace_leaf / CONFIG)}}
    return doc


def derive_settings(spec: R.W2RunSpec, rc: X2RunConfig, samples: Sequence[Sample], out: str | Path, *, splits_doc: Mapping[str, Any],
                    trace_leaf: Path, trace_ident: Mapping[str, Any], trace: Sequence[Mapping[str, Any]], timing: Mapping[str, Any],
                    cand_rows: Mapping[str, Mapping[str, Any]], cand_leaf: Path, prop_recs: Mapping[str, Mapping[str, Any]], prop_dir: Path,
                    files: Sequence[str | Path], settings: Sequence[str] | None = None) -> dict[str, Any]:
    """各设置的输出目录（只读轨迹，整份重写）与 summary.json。返回 summary。"""
    defs = setting_defs(rc.rounds)
    names = list(defs) if settings is None else [s for s in defs if s in settings]
    by_t = {t["id"]: t for t in trace}
    prop_t = R._timing_by_id(prop_dir)
    cand_t = R._timing_by_id(cand_leaf)
    summ: dict[str, Any] = {"experiment": "X-2", "split": spec.split, "model": spec.model_key, "checkpoint": trace_ident["checkpoint"]["manifest_sha256"],
                            "n": len(samples), "settings": {}}
    for name in names:
        d = defs[name]
        leaf = R.w2_leaf(out, spec.model_key, TASK, name, spec.split)
        ident = setting_identity(trace_ident, name, d, spec, splits_doc, samples, trace_leaf, out)
        B._prepare_derived(leaf, ident, files, resume=True)
        recs, comps = [], []
        for s in samples:
            rec, comp = setting_record(name, d, s, by_t[s.id], timing.get(s.id, {"model_s": {}, "tool_s": {}}), cand_rows[s.id], prop_recs[s.id],
                                       float(prop_t.get(s.id, {}).get("wall_time_s", 0.0)), float(cand_t.get(s.id, {}).get("wall_time_s", 0.0)),
                                       spec.split)
            recs.append(rec)
            comps.append(comp)
        (leaf / PREDICTIONS).write_text("".join(_jsonl_line(r) for r in recs), encoding="utf-8")
        body = aggregate(name, d, recs)
        metrics = {"experiment": "X-2", "w2": ident["w2"], "model": spec.model_key, "revision": spec.revision, "setting": name,
                   "definition": d, **body, "complete": len(recs) == len(samples), "n_expected": len(samples)}
        (leaf / METRICS).write_text(json.dumps(metrics, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        total = merge(comps)
        write_compute_json(leaf / COMPUTE, total)
        head = {h["name"]: h for h in body["headline"]}
        (leaf / README).write_text("\n".join([
            f"# X-2 · {spec.model_key} · refseg · {name} · {spec.split}", "",
            f"- 定义：{json.dumps(d, ensure_ascii=False)}",
            f"- 检查点：{trace_ident['checkpoint'].get('path')}（manifest {str(trace_ident['checkpoint'].get('manifest_sha256'))[:16]}…）",
            f"- 轨迹：{B._under_out(out, trace_leaf)}；运行配置 {rc.path}（{rc.sha256[:16]}…）",
            *[f"- {k}：{h.get('value')}（95% CI {h.get('ci')}）" for k, h in head.items()],
            f"- 失败 {body['failures']}；fallback {body['fallback']}；每句模型调用 {(body['n_model_calls'] or {}).get('mean')}", ""]), encoding="utf-8")
        summ["settings"][name] = {"cIoU": head.get("cIoU", {}).get("value"), "cIoU_ci": head.get("cIoU", {}).get("ci"),
                                  "gIoU": head.get("gIoU", {}).get("value"), "multi_round": d["multi_round"], "diagnostic": d["diagnostic"],
                                  "ablation": d["ablation"], "per_sample": total.per_sample_mean(), "fallback": body["fallback"]}
    singles = {k: v for k, v in summ["settings"].items() if not v["multi_round"] and not v["diagnostic"] and v["ablation"] is None}
    pairing = {}
    for k, v in summ["settings"].items():
        if not v["multi_round"] or v["per_sample"] is None:
            continue
        calls = v["per_sample"]["n_calls"]
        le = [(s, sv["per_sample"]["n_calls"]) for s, sv in singles.items() if sv["per_sample"] and sv["per_sample"]["n_calls"] <= calls]
        pairing[k] = {"calls_per_sample": calls, "single_round_at_most_this_compute": sorted(le, key=lambda x: -x[1])}
    summ["compute_pairing"] = pairing
    summ["rule"] = ("多轮设置与单轮设置的计算量配对：列出每句平均模型调用不超过它的单轮设置（不含诊断与消融），按调用数从多到少；"
                    "单轮结果是主要数字（硬约束 11）")
    sp = R.w2_leaf(out, spec.model_key, TASK, SUMMARY_DIR, "")
    sp.mkdir(parents=True, exist_ok=True)
    (sp / f"{spec.split}.json").write_text(json.dumps(summ, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return summ


# ---------------------------------------------------------------------------
# 一个划分的全流程
# ---------------------------------------------------------------------------


def run_split(adapter: Any, entry: Any, spec: R.W2RunSpec, w2: W2Config, rc: X2RunConfig, samples: Sequence[Sample], paths: Any,
              out: str | Path, *, checkpoint: Mapping[str, Any], x2: Mapping[str, Any], splits_doc: Mapping[str, Any], prompts: PromptConfig,
              impl: Any, e1_batch_size: int, resume: bool, tool_workers: int = 1, candidate_workers: int = 1, argv: Sequence[str] = (),
              run_extra: Mapping[str, Any] | None = None, settings: Sequence[str] | None = None,
              log: Callable[[str], None] | None = None) -> dict[str, Any]:
    """模块说明的 1–5：提出（W2-c 的同一个函数与目录）→ 候选（candidates_w2c）→ 轨迹 → 各设置与 summary。返回 summary。"""
    from src.w2 import w2c as W2C
    from src.w2.config import PROPOSE_BOX, W2C_CANDIDATES_SETTING

    log = log or (lambda m: print(m, file=sys.stderr, flush=True))
    if spec.task != TASK:
        raise X2RunError(f"X-2 只做 {TASK}，收到 {spec.task}")
    if w2.select_system is not None:
        raise X2RunError("X-2 的挑选关思考（同 W2-c）：configs/w2.json 的 select.system 必须是 null")
    bat = rc.batching(tiny=spec.tiny)
    common = dict(splits_doc=splits_doc, prompts=prompts, argv=argv)
    prop_dir, prop_list = W2C.ensure_proposal(adapter, entry, spec, w2, samples, paths, out, checkpoint=checkpoint, batch_size=e1_batch_size,
                                              run_extra=run_extra, log=log, **common)
    cand_leaf, cand_rows = R.ensure_candidates(spec, w2, samples, out, mode=PROPOSE_BOX, prop_dir=prop_dir, prop_recs=prop_list, impl=impl,
                                               chunk_size=bat["chunk_size"], workers=candidate_workers, cand_setting=W2C_CANDIDATES_SETTING,
                                               log=log, **common)
    rows = [cand_rows[s.id] for s in samples]
    ident = trace_identity(spec, w2, rc, splits_doc, samples, out, prop_dir=prop_dir, cand_leaf=cand_leaf, cand_rows=rows, impl=impl,
                           checkpoint=checkpoint, x2=x2, batching=bat)
    files = [w2.path, REPO_ROOT / rc.path if not Path(rc.path).is_absolute() else Path(rc.path), *R._splits_files(splits_doc),
             *paths.config_files()]
    with ToolPool(tool_workers) as pool:
        ctx = ChunkContext(adapter=adapter, w2=w2, rc=rc, impl=impl, prompts=prompts, n=int(x2["n"]), m=int(x2["m"]),
                           batch_size=bat["batch_size"], max_batch_tokens=bat["max_batch_tokens"], tiny=spec.tiny, pool=pool)
        trace_leaf, trace, timing = run_trace(ctx, spec, samples, cand_rows, out, ident=ident, files=files, chunk_size=bat["chunk_size"],
                                              resume=resume, argv=argv, log=log)
    props = R._proposals_by_id(PROPOSE_BOX, samples, prop_list, prompts, spec, w2)
    return derive_settings(spec, rc, samples, out, splits_doc=splits_doc, trace_leaf=trace_leaf, trace_ident=ident, trace=trace, timing=timing,
                           cand_rows=cand_rows, cand_leaf=cand_leaf, prop_recs=props, prop_dir=prop_dir, files=files, settings=settings)


__all__ = [
    "BASES", "ChunkContext", "FIFTH_NAME", "TRACE_SETTING", "X2RunConfig", "X2RunError", "X2_RUN_CONFIG_PATH", "aggregate",
    "checkpoint_x2_params", "derive_settings", "final_of", "generate_parse", "load_run_config", "process_chunk", "run_trace",
    "run_split", "setting_defs", "setting_record", "trace_identity",
]
