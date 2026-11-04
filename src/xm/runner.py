"""X-M 的推理运行器（DEV-0080；指令第三、四节）：一个模型（X-M 的检查点，或不加 LoRA 的基座）× 一个划分，各设置共用同一份模型输出。

**一个划分的流程**（run_split）：
1. 轨迹（<out>/<模型>/xm/<轨迹>/<划分>/trace.jsonl，轨迹 = xm_trace（X-M 的检查点）或 base_trace（基座）；每 chunk_size 个对一组，
   没写完的那一组整组重跑，续跑同 W2 / X-2）。每组依次：
   - 调用 1（列出实例）：原图 + src/xm/prompts.list_prompt(类别)，关思考、贪心、E1 的解码参数与图像口径（检查点的口径必须就是 E1 冻结的，
     基座用 default_image_policy），E1 冻结的批大小。X-M 用 prompts.parse_boxes（严格 JSON）；基座按 E1 协议用适配器的
     parse_boxes（E1 ovd 的 boxes 解析：认代码块与本家族的原生格式，是对基座最宽的读法），之后同样只取前 MAX_INSTANCES 个；
   - NMS（src/xm/post.nms，输出顺序贪心，IoU > NMS_IOU 去掉后者）；
   - 调用 2（描轮廓，只 X-M、只对至少有一个框的对）：原图 + prompts.trace_prompt(类别, NMS 之后的框)，组内按框数升序分批（批大小
     TRACE_BATCH_SIZE），每批的生成上限取批内各对 max_new_tokens_trace(k) 的最大值；prompts.parse_splines 逐条解析；
   - 工具（src/xm/post.pair_tool_job，进程池可选）：每个框的样条光栅化与 trimap、椭圆的光栅化与 trimap、框 → W1-F 的缺省 GrabCut；
     基座只有框 → GrabCut。
2. 设置（derive_settings；只读轨迹，每次整份重写）：<out>/<模型>/xm/<设置>/<划分>/ 的 predictions.jsonl、metrics.json、compute.json、
   config.yaml、meta.json、README.md；汇总 <out>/<模型>/xm/xm_summary/<划分>__<轨迹>.json。
   - xm_spline_trimap（主设置）、xm_box_grabcut（判定第 1 条的对照）、xm_ellipse_trimap（椭圆消融）、xm_spline_raster（诊断）：
     检测的读数都来自 X-M 调用 1 的框（NMS 之后），只算一次，四个设置的 det 一节相同；实例分割各用各的 mask；
   - base_det（判定第 2 条的对照）：只有检测；base_box_grabcut：基座的框 → 缺省 GrabCut 的实例分割，检测一节同 base_det。
   - 样条失败的实例按 src/xm/post.SPLINE_FAILURE_FALLBACK 退回并计数；调用 1 失败的对没有任何预测（真值全记 FN）。
3. 指标见 src/xm/metrics（主读数 f1_rexomni，XM-3a；按图像重抽的 bootstrap）。
4. 记账（硬约束 11）：每个对每个设置的模型调用次数（调用 1；样条类设置另加调用 2）、token（提示、生成、视觉）、墙钟（模型调用 + 工具）。

只用不含学习的工具（src/explore/guidance 经 src/xm/post）；本模块不 import 学习型的视觉模块（tests/test_xm_runner.py 核对）。
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from PIL import Image
from pycocotools import mask as mask_utils

from src.accounting import Compute, merge, write_compute_json
from src.e1.posthoc_runner import _finish_run_log, _prepare_leaf, _resume_point, _start_run_log
from src.e1.prompts import PromptConfig
from src.e1.runner import COMPUTE, CONFIG, METRICS, PREDICTIONS, README, TIMING, _append, _jsonl_line, batch_plan, generate_rows
from src.e1.runner import read_jsonl_tolerant
from src.e1.scoring import distribution
from src.explore import x2_prompts as XP
from src.explore.x2_runner import _call_record, _rejected_record
from src.parsing import parse_generation
from src.w1 import baselines as B
from src.xm import data as XD
from src.xm import metrics as XMM
from src.xm import post as XPOST
from src.xm import prompts as XMP

REPO_ROOT = Path(__file__).resolve().parents[2]
FRAMEWORK = "src/xm/runner.py（DEV-0080；X-M）"
TASK = "xm"
TRACE = "trace.jsonl"
TRACES = {"xm": "xm_trace", "base": "base_trace"}
SUMMARY_DIR = "xm_summary"
TRACE_BATCH_SIZE = 16  # 调用 2（同 configs/x2_run.json 的 batching.batch_size：描样条）
CHUNK_SIZE = 64  # 续跑的粒度（同 X-2）
TINY_BATCHING = {"list_batch_size": 2, "trace_batch_size": 2, "chunk_size": 2}
TRACE_ORDER = "调用 2 在每组（chunk）内按框数升序（同框数保持原顺序）分批"
XM_SPLITS_PATH = REPO_ROOT / "configs" / "xm_splits.json"  # DEV-0079 写；存在时它的 sha256 记进 meta.json
BASE_PARSER = "adapter.parse_boxes（E1 ovd 的 boxes 解析）"
XM_PARSER = "src/xm/prompts.parse_boxes（严格 JSON）"

SETTINGS: dict[str, dict[str, Any]] = {
    "xm_spline_trimap": {"system": "xm", "masks": "spline_trimap", "stages": ["list", "trace"], "role": "主设置：X-M 的框 + X-M 的样条 → 三元图贴边"},
    "xm_box_grabcut": {"system": "xm", "masks": "box_grabcut", "stages": ["list"], "role": "对照（判定第 1 条）：X-M 的框 → W1-F 的缺省 GrabCut，不描样条"},
    "xm_ellipse_trimap": {"system": "xm", "masks": "ellipse_trimap", "stages": ["list"], "role": "消融：X-M 的框的内切椭圆 → 三元图贴边（同 X-2）"},
    "xm_spline_raster": {"system": "xm", "masks": "spline_raster", "stages": ["list", "trace"], "role": "诊断：样条直接光栅化，不贴边"},
    "base_det": {"system": "base", "masks": None, "stages": ["list"], "role": "对照（判定第 2 条）：基座零样本，只出框"},
    "base_box_grabcut": {"system": "base", "masks": "box_grabcut", "stages": ["list"], "role": "起点：基座零样本的框 → 缺省 GrabCut"},
}


class XMRunError(ValueError):
    """X-M 运行器的输入不合格。"""


def settings_of(system: str) -> list[str]:
    return [k for k, v in SETTINGS.items() if v["system"] == system]


def system_of(settings: Sequence[str]) -> str:
    bad = [s for s in settings if s not in SETTINGS]
    if bad:
        raise XMRunError(f"未知的设置 {bad}；可选 {list(SETTINGS)}")
    sys_ = sorted({SETTINGS[s]["system"] for s in settings})
    if len(sys_) != 1:
        raise XMRunError(f"一次只跑一个系统的设置（xm_* 要 --checkpoint，base_* 不配检查点）：收到 {list(settings)}")
    return sys_[0]


XM_CHECKPOINT_SECTION = "xm"  # 训练叶目录 config.yaml 里 X-M 的一节（DEV-0081 写；键 prompts.version、可选 n）


def checkpoint_xm_params(checkpoint: str | Path) -> dict[str, Any]:
    """检查点（<训练叶目录>/ckpt/final/model）的叶目录 config.yaml 的 xm 一节 → {"section", "prompts_version", "n"}。没有这一节、提示
    版本不是本次的 PROMPT_VERSION、或 n 不是 SPLINE_N 时报错（这个检查点不是用本次的提示训的 X-M）。"""
    import yaml

    from src.e3.train import checkpoint_leaf

    leaf = checkpoint_leaf(checkpoint)
    if leaf is None or not (leaf / CONFIG).is_file():
        raise XMRunError(f"检查点 {checkpoint} 不在 <训练叶目录>/ckpt/final/model 下，或叶目录没有 {CONFIG}")
    doc = yaml.safe_load((leaf / CONFIG).read_text(encoding="utf-8")) or {}
    x = doc.get(XM_CHECKPOINT_SECTION)
    if not isinstance(x, Mapping):
        raise XMRunError(f"{leaf / CONFIG} 没有 {XM_CHECKPOINT_SECTION} 一节：这个检查点不是 X-M 的训练结果（DEV-0081）")
    ver = (x.get("prompts") or {}).get("version")
    if ver != XMP.PROMPT_VERSION:
        raise XMRunError(f"检查点训练时的 X-M 提示版本 {ver!r} 与本次的 {XMP.PROMPT_VERSION} 不同")
    n = int(x.get("n", XMP.SPLINE_N))
    if n != XMP.SPLINE_N:
        raise XMRunError(f"检查点训练的样条控制点数 {n} 不是 {XMP.SPLINE_N}")
    return {"section": json.loads(json.dumps(dict(x), ensure_ascii=False, default=str)), "prompts_version": ver, "n": n}


def refuse_mixing(out: str | Path, model_key: str, split: str, manifest_sha256: str) -> None:
    """加载与合并核对之前先看同一个 --out 里已有的 xm_trace：来自另一个检查点就拒绝（换一个 --out）。"""
    import yaml

    cfg = trace_leaf(out, model_key, "xm", split) / CONFIG
    if cfg.exists():
        have = ((yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}).get("checkpoint") or {}).get("manifest_sha256")
        if have != manifest_sha256:
            raise XMRunError(f"{cfg.parent}：已有的结果来自另一个检查点（manifest sha256 {str(have)[:16]}…，本次 {manifest_sha256[:16]}…），"
                             "拒绝混用；换一个 --out")


# ---------------------------------------------------------------------------
# 对（XMPair）的读写与键
# ---------------------------------------------------------------------------


def image_key(pair: XD.XMPair) -> str:
    """按图像重抽与"同一张图"的键：<source>:<相对路径>。"""
    return f"{pair.source}:{pair.image}"


NOT_EXHAUSTIVE_KEY = "not_exhaustive"  # DEV-0079：LVIS 的对在 XMPair.meta 里记本对的类别是否非穷尽（另有图像级的 not_exhaustive_category_ids）
IGNORE_KEY = "ignore_rle"  # DEV-0079：ReasonSeg 的忽略区域（计分时逐像素忽略），记在 XMPair.meta；没有时为 None


def not_exhaustive(pair: XD.XMPair) -> bool:
    """LVIS 的非穷尽类别（LVIS 官方惯例：没配上的预测不记 FP）：meta.not_exhaustive，或类别在 meta.not_exhaustive_category_ids 里。"""
    meta = getattr(pair, "meta", None) or {}
    if meta.get(NOT_EXHAUSTIVE_KEY) is not None:
        return bool(meta[NOT_EXHAUSTIVE_KEY])
    return pair.category_id in {int(c) for c in meta.get("not_exhaustive_category_ids") or ()}


def pair_to_json(p: XD.XMPair) -> dict[str, Any]:
    return XD.pair_to_dict(p)


def pairs_from_json(path: str | Path) -> list[XD.XMPair]:
    """XD.pair_to_dict 的列表（JSON 文件）→ XMPair（--tiny 的彩排与测试；真运行走 src/xm/data.load_pairs）。"""
    return [XD.pair_from_dict(d) for d in json.loads(Path(path).read_text(encoding="utf-8"))]


def pairs_digest(pairs: Sequence[XD.XMPair]) -> str:
    return hashlib.sha256("\n".join(p.id for p in pairs).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# 模型调用
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Context:
    adapter: Any
    system: str
    prompts: PromptConfig  # E1 的提示配置：解码参数与图像顺序
    data_root: Path
    list_batch_size: int
    trace_batch_size: int
    pool: XPOST.ToolPool
    n: int = XMP.SPLINE_N


def generate(adapter: Any, items: Sequence[tuple[list, str]], parsers: Sequence[Callable[[Any], Any]], budgets: Sequence[int], *,
             batch_size: int, prompts: PromptConfig) -> list[tuple[dict[str, Any], Any, float, int]]:
    """一组单轮生成（关思考、贪心、E1 的解码参数与 prompts.order；build_inputs 拒收的记 input_rejected，不进批）：按顺序每 batch_size 条一批，
    每批的生成上限取批内 budgets 的最大值；第 i 条的回答经 parse_generation(结果, parsers[i])。返回 (调用记录, 解析结果, 墙钟, 生成上限)。"""
    out: list[Any] = [None] * len(items)
    ready, inputs = [], []
    for i, (images, prompt) in enumerate(items):
        try:
            inputs.append(adapter.build_inputs(list(images), prompt, prompts.order))
            ready.append(i)
        except ValueError as e:
            out[i] = (_rejected_record(e), None, 0.0, 0)
    for a in range(0, len(ready), batch_size):
        part = list(range(a, min(a + batch_size, len(ready))))
        mnt = max(int(budgets[ready[k]]) for k in part)
        res = generate_rows(adapter, [inputs[k] for k in part], batch_size=batch_size, enable_thinking=False, max_new_tokens=mnt,
                            decoding=prompts.decoding())
        if len(res) != len(part):
            raise RuntimeError(f"generate_batch 返回 {len(res)} 行，输入 {len(part)} 行")
        for k, r in zip(part, res):
            p = parse_generation(r, parsers[ready[k]])
            out[ready[k]] = (_call_record(r, p.failure), p, float(getattr(r, "wall_time_s", 0.0)), mnt)
    return out


def _load_image(path: Path, pair: XD.XMPair) -> Image.Image:
    with Image.open(path) as im:
        img = im.convert("RGB")
    if img.size != (pair.width, pair.height):
        raise XMRunError(f"{pair.id}：图像 {path} 的尺寸 {img.size} 与对的 (width, height) = {(pair.width, pair.height)} 不符")
    return img


def _list_value(system: str, parsed: Any, w: int, h: int) -> tuple[list[list[float]] | None, dict[str, Any] | None]:
    """调用 1 的解析结果 → (原图像素的框（NMS 之前，至多 MAX_INSTANCES 个）, 计数)；失败为 (None, None)。"""
    if parsed is None or parsed.failure is not None:
        return None, None
    if system == "xm":
        v = parsed.value
        return [XP.box_abs(q, w, h) for q in v["boxes_q"]], {k: v[k] for k in ("n_raw", "n_capped", "n_clipped", "n_inverted")}
    boxes = [list(map(float, b)) for b in parsed.boxes]  # src.parsing 的 BoxParse（E1 的 boxes 解析）
    kept = boxes[: XMP.MAX_INSTANCES]
    return kept, {"n_raw": len(boxes), "n_capped": len(boxes) - len(kept), "n_clipped": 0, "n_inverted": 0}


def process_chunk(ctx: Context, pairs: Sequence[XD.XMPair]) -> tuple[list[dict], list[dict]]:
    """一组对的轨迹（模块说明第 1 条）。返回 (trace 的行, timing 的行)；trace 不含墙钟。"""
    paths = {p.id: ctx.data_root / p.image for p in pairs}
    images = {p.id: _load_image(paths[p.id], p) for p in pairs}
    tr: dict[str, dict[str, Any]] = {}
    tm: dict[str, dict[str, Any]] = {p.id: {"id": p.id, "model_s": {}, "tool_s": {}} for p in pairs}
    # 调用 1
    if ctx.system == "xm":
        parsers = [XMP.parse_boxes] * len(pairs)
    else:
        parsers = [lambda t, _p=p: ctx.adapter.parse_boxes(t, width=float(_p.width), height=float(_p.height)) for p in pairs]
    res = generate(ctx.adapter, [([images[p.id]], XMP.list_prompt(p.category)) for p in pairs], parsers,
                   [XMP.max_new_tokens_list()] * len(pairs), batch_size=ctx.list_batch_size, prompts=ctx.prompts)
    for p, (call, parsed, wall, mnt) in zip(pairs, res):
        boxes, info = _list_value(ctx.system, parsed, p.width, p.height)
        kept = XPOST.nms(boxes) if boxes else []
        kb = [boxes[i] for i in kept] if boxes else []
        tr[p.id] = {"id": p.id, "image_key": image_key(p), "image_size": [p.width, p.height], "category_id": p.category_id,
                    "category": p.category,
                    "list": {"call": {**call, "max_new_tokens": mnt}, "parser": XM_PARSER if ctx.system == "xm" else BASE_PARSER,
                             "info": info, "boxes_raw": boxes},
                    "nms": {"n_in": len(boxes or []), "kept": kept}, "boxes": kb, "boxes_q": [XP.box_q(b, p.width, p.height) for b in kb],
                    "trace": None, "instances": []}
        tm[p.id]["model_s"]["list"] = wall
    # 调用 2（只 X-M）
    live = [p for p in pairs if tr[p.id]["boxes"]]
    splines: dict[str, list | None] = {p.id: None for p in pairs}
    if ctx.system == "xm" and live:
        # 组内按框数升序排（同框数保持原顺序）再按批生成：输出长度约与框数成正比（一条样条约 157 个 token），同一批的长度相近，
        # 少等最长的那条（TRACE_ORDER，进运行身份）
        tl = sorted(live, key=lambda p: (len(tr[p.id]["boxes"]), live.index(p)))
        ks = [len(tr[p.id]["boxes"]) for p in tl]
        items = [([images[p.id]], XMP.trace_prompt(p.category, tr[p.id]["boxes_q"], ctx.n)) for p in tl]
        prs = [lambda t, _k=k: XMP.parse_splines(t, k=_k, n=ctx.n) for k in ks]
        res2 = generate(ctx.adapter, items, prs, [XMP.max_new_tokens_trace(k) for k in ks], batch_size=ctx.trace_batch_size, prompts=ctx.prompts)
        for p, k, (call, parsed, wall, mnt) in zip(tl, ks, res2):
            if parsed is not None and parsed.failure is None:
                v = parsed.value
                rec = {"splines": v["splines"], "failures": v["failures"], "n_clipped": v["n_clipped"]}
            else:
                why = call["failure"] or "input_rejected"
                rec = {"splines": [None] * k, "failures": [why] * k, "n_clipped": 0}
            tr[p.id]["trace"] = {"call": {**call, "max_new_tokens": mnt}, **rec}
            splines[p.id] = rec["splines"]
            tm[p.id]["model_s"]["trace"] = wall
    # 工具
    jobs = [(str(paths[p.id]), ctx.system, tr[p.id]["boxes"], splines[p.id], ctx.n) for p in live]
    for p, r in zip(live, ctx.pool.map(XPOST.pair_tool_job, jobs)):
        tm[p.id]["tool_s"]["instances"] = r["wall_s"]
        tr[p.id]["instances"] = r["instances"]
    return [tr[p.id] for p in pairs], [tm[p.id] for p in pairs]


# ---------------------------------------------------------------------------
# 轨迹
# ---------------------------------------------------------------------------


def batching(*, tiny: bool, list_batch_size: int) -> dict[str, Any]:
    out: dict[str, Any] = dict(TINY_BATCHING) if tiny else {"list_batch_size": int(list_batch_size), "trace_batch_size": TRACE_BATCH_SIZE,
                                                             "chunk_size": CHUNK_SIZE}
    return {**out, "trace_order": TRACE_ORDER}


def run_record(split: str, system: str, pairs: Sequence[XD.XMPair], *, limit: int | None, unseal: Mapping[str, Any] | None = None,
               xm_gate: Mapping[str, Any] | None = None, lvis_cap: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """运行身份里的 X-M 一节。lvis_cap：lvis_sub 载入时 XM-2b 类别上限的计数（src/xm/data.cap_lvis_pairs；DEV-0082），只在给了时加这个键
    （其余划分的身份不变）。"""
    src = Counter(p.source for p in pairs)
    return {"task": TASK, "split": split, "system": system, "n_pairs": len(pairs), "pair_ids_sha256": pairs_digest(pairs),
            "n_images": len({image_key(p) for p in pairs}), "n_instances": sum(len(p.instances) for p in pairs),
            "sources": {k: src[k] for k in sorted(src)}, "limit": limit,
            **({"unseal": dict(unseal)} if unseal is not None else {}), **({"xm_gate": dict(xm_gate)} if xm_gate is not None else {}),
            **({"lvis_cap": dict(lvis_cap)} if lvis_cap is not None else {})}


def trace_identity(model: Mapping[str, Any], image_policy: Mapping[str, Any], checkpoint: Mapping[str, Any] | None, xm: Mapping[str, Any],
                   prompts: PromptConfig, bat: Mapping[str, int]) -> dict[str, Any]:
    ident = {"experiment": "X-M", "framework": FRAMEWORK, "xm": dict(xm), "model": dict(model), "image_policy": dict(image_policy),
             "checkpoint": None if checkpoint is None else json.loads(json.dumps(dict(checkpoint), ensure_ascii=False)),
             "prompts": XMP.identity(), "post": XPOST.identity(),
             "generation": {"enable_thinking": False, "decoding": prompts.decoding(), "order": prompts.order,
                            "e1_prompts_sha256": prompts.sha256,
                            "list_parser": XM_PARSER if xm["system"] == "xm" else BASE_PARSER},
             "batching": dict(bat)}
    return ident


def trace_leaf(out: str | Path, model_key: str, system: str, split: str) -> Path:
    return Path(out) / model_key / TASK / TRACES[system] / split


def run_trace(ctx: Context, pairs: Sequence[XD.XMPair], out: str | Path, model_key: str, split: str, *, ident: Mapping[str, Any],
              files: Sequence[str | Path], chunk_size: int, resume: bool, argv: Sequence[str] = (),
              log: Callable[[str], None] | None = None) -> tuple[Path, list[dict], dict[str, dict]]:
    log = log or (lambda m: print(m, file=sys.stderr, flush=True))
    leaf = trace_leaf(out, model_key, ctx.system, split)
    _prepare_leaf(leaf, ident, files, resume=resume)
    ids = [p.id for p in pairs]
    plan = batch_plan([None] * len(ids), chunk_size)
    what = f"X-M {model_key} {TRACES[ctx.system]}/{split}"
    keep, trimmed = _resume_point(leaf / TRACE, ids, lambda r: r["id"], plan, log, what)
    todo = [(a, b) for a, b in plan if a >= keep]
    run_k, run_path, run_log = _start_run_log(leaf, argv, n_done_before=keep, n_remaining=len(ids) - keep, chunk_size=chunk_size,
                                              tool_workers=ctx.pool.workers, trimmed_to_batch_boundary=trimmed)
    t_all = time.perf_counter()
    for n_chunk, (a, b) in enumerate(todo):
        recs, tms = process_chunk(ctx, pairs[a:b])
        _append(leaf / TIMING, [_jsonl_line({**t, "run": run_k, "chunk_index": n_chunk}) for t in tms])
        _append(leaf / TRACE, [_jsonl_line(r) for r in recs])
        log(f"[{what}] {b}/{len(ids)}")
    _finish_run_log(run_path, run_log, n_batches=len(todo), t_all=t_all)
    recs = read_jsonl_tolerant(leaf / TRACE, repair=False)
    if [r["id"] for r in recs] != ids:
        raise RuntimeError(f"{leaf / TRACE} 的 id 与对不同")
    timing = {t["id"]: t for t in read_jsonl_tolerant(leaf / TIMING, repair=False)}
    return leaf, recs, timing


# ---------------------------------------------------------------------------
# 设置（只读轨迹）
# ---------------------------------------------------------------------------


def det_eval(pair: XD.XMPair, t: Mapping[str, Any]) -> dict[str, Any]:
    ious, crowd = XMM.box_ious(t["boxes"], [i.box_xyxy for i in pair.instances], pair.crowd_rles)
    return XMM.eval_pair(ious, crowd, len(pair.instances), not_exhaustive=not_exhaustive(pair))


def seg_eval(pair: XD.XMPair, rles: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ious, crowd = XMM.mask_ious(rles, [i.rle for i in pair.instances], pair.crowd_rles)
    return XMM.eval_pair(ious, crowd, len(pair.instances), not_exhaustive=not_exhaustive(pair))


def _stage_calls(d: Mapping[str, Any], t: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    calls = [("list", t["list"]["call"])]
    if "trace" in d["stages"] and t["trace"] is not None:
        calls.append(("trace", t["trace"]["call"]))
    return calls


def _tool_ok(tool: Mapping[str, Any], kind: str, fb: str | None) -> bool:
    """交的 mask 是不是工具正常产出的（GrabCut 跑不起来时 guidance 用光栅化或框本身，ok = False）。光栅化总是正常。"""
    if kind == "box_grabcut" or fb == "box_grabcut":
        return bool(tool["box_gc"]["ok"])
    if kind == "spline_raster":
        return True
    src = tool["ellipse"] if (kind == "ellipse_trimap" or fb == "ellipse") else tool["spline"]
    return bool(src["trimap_ok"])


def setting_record(name: str, d: Mapping[str, Any], pair: XD.XMPair, t: Mapping[str, Any], tm: Mapping[str, Any],
                   det: Mapping[str, Any]) -> tuple[dict[str, Any], Compute]:
    """predictions.jsonl 的一行（不含墙钟）与这个对的计算量。det：这个对的检测计数（同一系统的各设置共用）。"""
    inst = []
    rles = []
    for k, (box, q) in enumerate(zip(t["boxes"], t["boxes_q"])):
        tool = t["instances"][k]
        e: dict[str, Any] = {"box": box, "box_q": q}
        if d["system"] == "xm":
            e["spline_q"] = None if t["trace"] is None else t["trace"]["splines"][k]
            e["spline_failure"] = None if t["trace"] is None else t["trace"]["failures"][k]
        if d["masks"] is not None:
            rle, fb = XPOST.instance_mask(tool, d["masks"])
            e["mask"] = {"rle": rle, "area": int(mask_utils.area(XMM.coco_rle(rle)))}
            e["fallback"] = fb
            e["tool_ok"] = _tool_ok(tool, d["masks"], fb)
            rles.append(rle)
        inst.append(e)
    calls = _stage_calls(d, t)

    def tok(c: Mapping[str, Any], key: str) -> int:
        return int(c.get(key) or 0)

    tokens = {k: int(sum(tok(c, f"n_{k}_tokens") for _, c in calls)) for k in ("prompt", "generated", "inserted", "visual")}
    tokens["image_pixels"] = int(sum(tok(c, "n_image_pixels") for _, c in calls))
    by_stage = {s: {"calls": 1, "prompt": tok(c, "n_prompt_tokens") + tok(c, "n_inserted_tokens"), "generated": tok(c, "n_generated_tokens"),
                    "visual": tok(c, "n_visual_tokens")} for s, c in calls}
    rec = {"id": pair.id, "image_key": image_key(pair), "source": pair.source, "split": pair.split, "setting": name, "task": TASK,
           "category_id": pair.category_id, "category": pair.category, "image_size": [pair.width, pair.height],
           "n_gt": len(pair.instances), "n_crowd": len(pair.crowd_rles), "not_exhaustive": not_exhaustive(pair),
           "raw": {s: c.get("text") for s, c in calls},
           "failure": t["list"]["call"]["failure"], "list_info": t["list"]["info"], "nms": t["nms"],
           "trace_failure": None if "trace" not in d["stages"] or t["trace"] is None else t["trace"]["call"]["failure"],
           "instances": inst, "det": dict(det), "seg": None if d["masks"] is None else seg_eval(pair, rles),
           "tokens": tokens, "tokens_by_stage": by_stage, "n_model_calls": len(calls),
           "n_tool_calls": len(inst) if d["masks"] is not None else 0}
    call_s = float(sum(float(tm.get("model_s", {}).get(s, 0.0)) for s, _ in calls))
    tool_s = float(tm.get("tool_s", {}).get("instances", 0.0)) if d["masks"] is not None else 0.0
    comp = Compute(n_calls=len(calls), n_rounds=len(calls), generated_tokens=tokens["generated"], prompt_tokens=tokens["prompt"] + tokens["inserted"],
                   visual_tokens=tokens["visual"], image_pixels=tokens["image_pixels"], call_time_s=call_s).as_sample(wall_time_s=call_s + tool_s)
    return rec, comp


def _cnt(vals) -> dict[str, int]:
    c = Counter("ok" if v is None else str(v) for v in vals)
    return {k: c[k] for k in sorted(c)}


def failure_summary(d: Mapping[str, Any], recs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    info = [r["list_info"] for r in recs if r["list_info"] is not None]
    out: dict[str, Any] = {"list_failure": _cnt(r["failure"] for r in recs),
                           "list_counts": {k: int(sum(i[k] for i in info)) for k in ("n_raw", "n_capped", "n_clipped", "n_inverted")},
                           "nms_suppressed": int(sum(r["nms"]["n_in"] - len(r["nms"]["kept"]) for r in recs)),
                           "n_pred_boxes": distribution(len(r["instances"]) for r in recs)}
    if "trace" in d["stages"]:
        out["trace_failure"] = _cnt(r["trace_failure"] for r in recs if r["instances"])
        out["spline_failure"] = _cnt(e["spline_failure"] for r in recs for e in r["instances"])
    if d["masks"] is not None:
        out["fallback"] = _cnt(e["fallback"] for r in recs for e in r["instances"])
        out["tool_not_ok"] = int(sum(1 for r in recs for e in r["instances"] if not e["tool_ok"]))
    return out


def setting_identity(trace_ident: Mapping[str, Any], name: str, d: Mapping[str, Any], trace_leaf_: Path, out: str | Path) -> dict[str, Any]:
    doc = {k: v for k, v in trace_ident.items() if k != "batching"}
    doc["xm"] = {**dict(trace_ident["xm"]), "setting": name, "definition": dict(d)}
    doc["metrics"] = XMM.identity()
    doc["trace"] = {"leaf": B._under_out(out, trace_leaf_), "config_sha256": hashlib.sha256((trace_leaf_ / CONFIG).read_bytes()).hexdigest()}
    return doc


def _fmt(v: Any) -> str:
    return "null" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))


def _readme(name: str, d: Mapping[str, Any], model_key: str, split: str, m: Mapping[str, Any]) -> str:
    lines = [f"# X-M · {model_key} · {name} · {split}", "", f"- 定义：{d['role']}（mask：{d['masks']}；调用：{d['stages']}）", ""]
    for kind in XMM.KINDS:
        sec = m.get(kind)
        if not sec:
            continue
        lines.append(f"## {kind}（{sec['iou']} IoU；{sec['n_pairs']} 个对、{sec['n_images']} 张图、{sec['n_categories']} 类）")
        for st in (*XMM.STATISTICS, *XMM.EXACT_STATISTICS):
            e = sec[st]
            ci = e.get("ci")
            lines.append(f"- {st}：{_fmt(e['value'])}" + (f"（95% CI [{_fmt(ci['low'])}, {_fmt(ci['high'])}]）" if ci else "")
                         + (f"；P {_fmt(e.get('precision'))}、R {_fmt(e.get('recall'))}" if "precision" in e else ""))
        lines.append("")
    lines += [f"- 失败与退回：{json.dumps(m['failures'], ensure_ascii=False)}", ""]
    return "\n".join(lines)


def derive_settings(pairs: Sequence[XD.XMPair], out: str | Path, model: Mapping[str, Any], split: str, system: str, *, trace_leaf_: Path,
                    trace_ident: Mapping[str, Any], trace: Sequence[Mapping[str, Any]], timing: Mapping[str, Any], files: Sequence[str | Path],
                    settings: Sequence[str] | None = None, n_resamples: int | None = XMM.BOOTSTRAP_N_RESAMPLES) -> dict[str, Any]:
    """各设置的输出目录（只读轨迹，整份重写）与汇总。返回汇总。检测的计数与读数在同一系统的各设置之间只算一次。"""
    names = [s for s in settings_of(system) if settings is None or s in settings]
    by_t = {t["id"]: t for t in trace}
    dets = {p.id: det_eval(p, by_t[p.id]) for p in pairs}
    model_key = model["key"]
    summ: dict[str, Any] = {"experiment": "X-M", "split": split, "system": system, "model": model_key,
                            "checkpoint": (trace_ident.get("checkpoint") or {}).get("manifest_sha256"), "n_pairs": len(pairs), "settings": {}}
    det_readout = None
    for name in names:
        d = SETTINGS[name]
        leaf = Path(out) / model_key / TASK / name / split
        ident = setting_identity(trace_ident, name, d, trace_leaf_, out)
        B._prepare_derived(leaf, ident, files, resume=True)
        recs, comps = [], []
        for p in pairs:
            rec, comp = setting_record(name, d, p, by_t[p.id], timing.get(p.id, {}), dets[p.id])
            recs.append(rec)
            comps.append(comp)
        (leaf / PREDICTIONS).write_text("".join(_jsonl_line(r) for r in recs), encoding="utf-8")
        if det_readout is None:
            det_readout = XMM.readout(recs, "det", n_resamples=n_resamples)
        metrics = {"experiment": "X-M", "xm": ident["xm"], "model": model_key, "revision": model.get("revision"), "setting": name,
                   "definition": dict(d), "det": det_readout,
                   "det_note": "检测的读数来自本系统调用 1 的框（NMS 之后），同一系统的各设置相同、只算一次",
                   "seg": None if d["masks"] is None else XMM.readout(recs, "seg", n_resamples=n_resamples),
                   "failures": failure_summary(d, recs), "n_model_calls": distribution(r["n_model_calls"] for r in recs),
                   "complete": len(recs) == len(pairs), "n": len(recs), "n_expected": len(pairs)}
        (leaf / METRICS).write_text(json.dumps(metrics, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        total = merge(comps)
        write_compute_json(leaf / COMPUTE, total)
        (leaf / README).write_text(_readme(name, d, model_key, split, metrics), encoding="utf-8")
        summ["settings"][name] = {k: (None if metrics[k] is None else {st: metrics[k][st]["value"]
                                                                        for st in (*XMM.STATISTICS, *XMM.EXACT_STATISTICS)})
                                  for k in XMM.KINDS} | {"per_pair": total.per_sample_mean(), "role": d["role"]}
    sp = Path(out) / model_key / TASK / SUMMARY_DIR
    sp.mkdir(parents=True, exist_ok=True)
    (sp / f"{split}__{TRACES[system]}.json").write_text(json.dumps(summ, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return summ


# ---------------------------------------------------------------------------
# ReasonSeg（指代分割走 X-2 的运行器，这里只把对换成 W2 / E1 的样本）
# ---------------------------------------------------------------------------


def reasonseg_samples(pairs: Sequence[XD.XMPair], data_root: str | Path) -> list[Any]:
    """ReasonSeg 的对（DEV-0079：category 放表达式原文，instances 至多 1 个；真值为空的条目没有实例）→ src.data.records.Sample
    （refexp；fields.expression、gt.box / gt.mask，同 RefCOCO 的加载器），交给 src/explore/x2_runner.run_split。"""
    from src.data.records import Sample
    from src.executors.raster import mask_to_rle

    out = []
    for p in pairs:
        if len(p.instances) > 1:
            raise XMRunError(f"{p.id}：ReasonSeg 的对至多 1 个实例（表达式对应的 mask），收到 {len(p.instances)} 个")
        if p.instances:
            box, rle = [float(v) for v in p.instances[0].box_xyxy], dict(p.instances[0].rle)
        else:  # 真值为空（LISA 的 val 有 4 条）
            import numpy as np

            box, rle = [0.0, 0.0, 0.0, 0.0], mask_to_rle(np.zeros((p.height, p.width), bool))
        out.append(Sample(id=p.id, task="refexp", images=(str(Path(data_root) / p.image),), image_keys=(image_key(p),),
                          fields={"expression": p.category, "image_size": [p.width, p.height]},
                          gt={"box": box, "mask": {"rle": rle, "height": p.height, "width": p.width}}, source=p.source, split=p.split))
    return out


def reasonseg_readout(recs: Sequence[Mapping[str, Any]], pairs: Sequence[XD.XMPair]) -> dict[str, Any]:
    """ReasonSeg 的 gIoU、cIoU，忽略区域（XMPair.meta 的 ignore_rle）逐像素不计（LISA 的 ignore_index = 255）：X-2 的运行器写的
    W2 格式 predictions.jsonl（answer.mask.rle；没有 mask 的按空 mask）→ src/xm/metrics.refseg_readout。另报不扣忽略区域的读数（= W2 的
    headline，核对用）。"""
    import numpy as np

    from src.executors.raster import rle_to_mask

    by = XD.pair_by_id(pairs)
    rows, plain = [], []
    for r in recs:
        p = by[r["id"]]
        gt = rle_to_mask(XMM.coco_rle_json(p.instances[0].rle)) if p.instances else np.zeros((p.height, p.width), bool)
        ans = r.get("answer")
        pred = rle_to_mask(ans["mask"]["rle"]) if ans else None
        ig = (getattr(p, "meta", None) or {}).get(IGNORE_KEY)
        keep = ~rle_to_mask(XMM.coco_rle_json(ig)) if ig else np.ones_like(gt)
        g = gt & keep
        if pred is None:
            rows.append({"score": {"inter": 0, "union": int(g.sum())}, "answer": None})
        else:
            q = pred & keep
            rows.append({"score": {"inter": int((q & g).sum()), "union": int((q | g).sum())}, "answer": ans})
        plain.append(r)
    return {"rule": "忽略区域（meta.ignore_rle）里的像素不进交与并（LISA 的 ignore_index）；真值为空且预测为空记 1（同 W2 / LISA）",
            "with_ignore": XMM.refseg_readout(rows), "without_ignore": XMM.refseg_readout(plain),
            "n_with_ignore_region": int(sum(1 for p in pairs if (getattr(p, "meta", None) or {}).get(IGNORE_KEY)))}


# ---------------------------------------------------------------------------
# 一个划分的全流程
# ---------------------------------------------------------------------------


def run_split(adapter: Any, pairs: Sequence[XD.XMPair], out: str | Path, *, system: str, split: str, model: Mapping[str, Any],
              image_policy: Mapping[str, Any], checkpoint: Mapping[str, Any] | None, prompts: PromptConfig, data_root: str | Path,
              tiny: bool, list_batch_size: int, resume: bool, tool_workers: int = 1, settings: Sequence[str] | None = None,
              limit: int | None = None, unseal: Mapping[str, Any] | None = None, xm_gate: Mapping[str, Any] | None = None,
              files: Sequence[str | Path] = (), argv: Sequence[str] = (), log: Callable[[str], None] | None = None,
              lvis_cap: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """模块说明的 1–4。返回汇总。lvis_cap 见 run_record。"""
    if system not in TRACES:
        raise XMRunError(f"system 必须是 {list(TRACES)} 之一，收到 {system!r}")
    if (system == "xm") != (checkpoint is not None):
        raise XMRunError("xm 系统要检查点，base 系统不配检查点")
    if settings is not None and system_of(settings) != system:
        raise XMRunError(f"设置 {list(settings)} 不属于系统 {system}")
    bat = batching(tiny=tiny, list_batch_size=list_batch_size)
    xm = run_record(split, system, pairs, limit=limit, unseal=unseal, xm_gate=xm_gate, lvis_cap=lvis_cap)
    ident = trace_identity(model, image_policy, checkpoint, xm, prompts, bat)
    files = [*files, *([XM_SPLITS_PATH] if XM_SPLITS_PATH.is_file() else [])]
    with XPOST.ToolPool(tool_workers) as pool:
        ctx = Context(adapter=adapter, system=system, prompts=prompts, data_root=Path(data_root), list_batch_size=bat["list_batch_size"],
                      trace_batch_size=bat["trace_batch_size"], pool=pool)
        tleaf, trace, timing = run_trace(ctx, pairs, out, model["key"], split, ident=ident, files=files, chunk_size=bat["chunk_size"],
                                         resume=resume, argv=argv, log=log)
    return derive_settings(pairs, out, model, split, system, trace_leaf_=tleaf, trace_ident=ident, trace=trace, timing=timing, files=files,
                           settings=settings)


__all__ = [
    "IGNORE_KEY", "NOT_EXHAUSTIVE_KEY", "XM_CHECKPOINT_SECTION", "checkpoint_xm_params", "refuse_mixing", "reasonseg_readout",
    "reasonseg_samples", "BASE_PARSER", "CHUNK_SIZE", "Context", "FRAMEWORK", "SETTINGS", "SUMMARY_DIR", "TASK", "TRACES", "TRACE_BATCH_SIZE", "XMRunError",
    "XM_PARSER", "batching", "derive_settings", "det_eval", "failure_summary", "generate", "image_key", "not_exhaustive", "pair_to_json",
    "pairs_digest", "pairs_from_json", "process_chunk", "run_record", "run_split", "run_trace", "seg_eval", "setting_identity",
    "setting_record", "settings_of", "system_of", "trace_identity", "trace_leaf",
]
