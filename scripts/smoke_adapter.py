#!/usr/bin/env python
"""Qwen3.5 适配器冒烟（DEV-0003；G0 看它的结果）。

输出一个 JSON。任何一项失败：先把 JSON 完整写出（证据先落盘），再以非零码退出。

服务器（真模型，权重按注册表的固定 revision 加载）：
    python scripts/smoke_adapter.py --model qwen3.5-0.8b --device cuda --dtype bf16 --image <图路径> --out <json>
开发机彩排（本地构造的小型随机 Qwen3.5，CPU，只下载 tokenizer / 处理器 / config 小文件）：
    python scripts/smoke_adapter.py --tiny --out /tmp/smoke_tiny.json

检查项（1–7 的编号同任务书；0、8 是附加项）：
  0 环境与模型事实（只记录）
  1 两种顺序 × 思考开关各 build + generate 一次（max_new_tokens 取小值）；prompt + assistant 头与
    apply_chat_template 逐个相同；load 路径下另与官方 AutoProcessor 的输出逐项比对
  2 经适配器 API 复现 RUN-0001 §5.7 的仪器：图在前时两条等长指令下各层图像 token 逐位相同；
    指令在前时第 0 层相同、之后各层不同（逐层 max|Δ| 与 model_notes §5.7 的表同口径）
  3 add_grid_tokens(["<g0>", "<g1>"]) 后用 allowed_token_ids 限定生成，产出的 id 全在允许集合内
  4 inject_query_tokens(16, grid2d / seq1d) 都能前向、形状正确；查询块之前各位置两种模式逐位相同；位置合规
  5 不带查询 token：传自构 position_ids 与让模型自己算，隐藏状态逐位相同（另比适配器整条路径）
  6 vision_features 与第 0 层图像位置的隐藏状态逐位相同
  7 思考开启时 text 不含思考段、raw_text 含思考段（未闭合时 text 为空并标记）
  8 设备审计：参数与 buffer 全在目标设备；一次前向加一次生成里，每个子模块的张量输入都在目标设备
执行顺序：0 1 2 5 6 7 3 4 8（会加 token / 注入查询的 3、4 放在后面；8 最后）。

其他家族按 --model 的注册键分派；开发机彩排加 --tiny（--tiny 时 --model 选家族与小模型 config 的来源，不给 --model
即 Qwen3.5）。按家族分派只在 family_suite() 一处；每个家族的 setup 返回 (adapter, ctx)，ctx["checks"] 是它的检查表：
- Gemma 4（DEV-0004）：--model gemma-4-e2b-it。构造与 0、1、3–7 项检查在 src/adapters/gemma4.py（smoke_setup），
  2、8 两项沿用本脚本的通用实现；编号与含义同上，Gemma 的差异写在那里。
- InternVL3.5（DEV-0005）：--model internvl3.5-2b-hf。构造与 0、1、4–7 项检查，以及附加的 9（多 tile），在
  src/adapters/internvl35.py（smoke_suite），2、3、8 三项沿用本脚本的通用实现；差异写在那里。
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import math
import platform
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402

from src.adapters import qwen35, registry  # noqa: E402
from src.adapters.base import ORDERS  # noqa: E402
from src.adapters.qwen35 import IMAGE_PAD, IM_END, END_OF_TEXT, THINK_END  # noqa: E402

INSTR_A, INSTR_B = "Describe the cat.", "Describe the sofa."  # RUN-0001 §5.7 用的同一对指令（各 4 个 token）
INSTR_THINK = "What is the main color in this image? Answer in one word."
GRID_NAMES = ["<g0>", "<g1>"]
N_QUERY = 16
DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}
CHECK_NAMES = [
    "0_facts", "1_build_generate", "2_task_modulation", "3_grid_tokens", "4_query_tokens",
    "5_position_ids", "6_vision_features", "7_thinking", "8_device_audit",
]
RUN_ORDER = [
    "0_facts", "1_build_generate", "2_task_modulation", "5_position_ids", "6_vision_features",
    "7_thinking", "3_grid_tokens", "4_query_tokens", "8_device_audit",
]


# ---------------------------------------------------------------------------
# 准备
# ---------------------------------------------------------------------------


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", help="注册键，如 qwen3.5-0.8b（见 src/adapters/registry.py）")
    p.add_argument("--device", help="默认：真模型 cuda，--tiny 时 cpu")
    p.add_argument("--dtype", choices=sorted(DTYPES), help="默认：真模型 bf16，--tiny 时 fp32")
    p.add_argument("--image", help="图路径；--tiny 时可省略（用合成图，H≠W）")
    p.add_argument("--out", required=True, help="输出 JSON 路径")
    p.add_argument("--image-size", nargs=2, type=int, default=[480, 640], metavar=("H", "W"),
                   help="固定分辨率，默认 480 640（与 RUN-0001 §5.7 的 300 个视觉 token 同设置）")
    p.add_argument("--tiny", action="store_true", help="用本地构造的小型随机模型在 CPU 上彩排（家族取 --model，不给即 Qwen3.5）")
    p.add_argument("--max-new-tokens", type=int, default=8, help="检查 1、3、7（关思考对照）的生成长度")
    p.add_argument("--think-max-new-tokens", type=int, help="检查 7 的思考生成长度；默认真模型 512、--tiny 16")
    args = p.parse_args(argv)
    if args.tiny:  # --model 可省略（即 Qwen3.5）；给了就按它的家族彩排（DEV-0004）
        args.device = args.device or "cpu"
        args.dtype = args.dtype or "fp32"
        args.think_max_new_tokens = args.think_max_new_tokens or 16
    else:
        if not args.model or not args.image:
            p.error("真模型冒烟需要 --model 与 --image")
        args.device = args.device or "cuda"
        args.dtype = args.dtype or "bf16"
        args.think_max_new_tokens = args.think_max_new_tokens or 512
    return args


def synthetic_image() -> Image.Image:
    """H≠W 的合成图：两个方向的渐变、一个矩形、噪声。"""
    rng = np.random.default_rng(0)
    h, w = 360, 540
    img = np.zeros((h, w, 3), np.int64)
    img[..., 0] = np.linspace(0, 255, w)[None, :]
    img[..., 2] = np.linspace(255, 0, h)[:, None]
    img[80:200, 100:300, 1] = 220
    img = img + rng.integers(0, 40, img.shape)
    return Image.fromarray(img.clip(0, 255).astype(np.uint8))


def base_context(args):
    """各家族共用的准备：固定种子、测试图（--image 或合成图）与第二张图。返回 ctx，各家族的 setup 在上面补充。"""
    torch.manual_seed(0)
    if args.image:
        path = Path(args.image)
        img = Image.open(path).convert("RGB")
        image_info = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                      "size_wh": list(img.size)}
    else:
        img = synthetic_image()
        image_info = {"path": None, "synthetic": True, "size_wh": list(img.size)}
    img2 = ImageOps.mirror(img)  # 多图检查用的第二张：左右翻转，内容不同
    return {"img": img, "img2": img2, "image_info": image_info, "args": args}


def setup(args):
    """Qwen3.5 的构造（不给 --model 时的默认家族）。其他家族的构造见 family_suite。"""
    ctx = base_context(args)
    dtype = DTYPES[args.dtype]
    hw = tuple(args.image_size)
    if args.tiny:
        comps = qwen35.load_pinned_components()
        model = qwen35.build_tiny_model(comps[0], tie_word_embeddings=True, dtype=dtype)
        ad = qwen35.make_tiny_adapter(comps, image_size=hw, device=args.device, model=model)
    else:
        ad = qwen35.Qwen35Adapter(device=args.device, image_size=hw)
        ad.load(args.model, dtype)
    return ad, {**ctx, "checks": CHECKS}


def _layers(ad):
    return list(range(ad.num_layers() + 1))


def _collapse_image_runs(ids, image_id):
    out = []
    for t in ids:
        if t == image_id and out and out[-1] == image_id:
            continue
        out.append(t)
    return out


def _content(n_images, instruction, order, images=None):
    imgs = [{"type": "image", "image": im} for im in images] if images else [{"type": "image"}] * n_images
    text = {"type": "text", "text": instruction}
    return [text, *imgs] if order == "instruction_first" else [*imgs, text]


def _base_mean(weight, n_base):
    """冒烟里独立重算的基础行均值（float64 分块求和），不复用适配器的实现。"""
    acc = torch.zeros(weight.shape[1], dtype=torch.float64, device=weight.device)
    for s in range(0, n_base, 16384):
        acc += weight[s : min(s + 16384, n_base)].double().sum(0)
    return (acc / n_base).to(weight.dtype)


# ---------------------------------------------------------------------------
# 检查项
# ---------------------------------------------------------------------------


def check_0_facts(ad, ctx):
    import transformers
    from transformers.models.qwen3_5 import modeling_qwen3_5 as mq

    def fn_name(f):
        # 穿过 transformers 的分派包装：包装的 __module__/__qualname__ 抄自 torch 原函数，只看名字分不出 fla 与 torch
        f = qwen35.kernel_implementation(f)
        return f"{getattr(f, '__module__', '?')}.{getattr(f, '__qualname__', type(f).__name__)}"

    entry = registry.MODELS.get(ad.model_key) if ad.model_key in registry.MODELS else None
    tok = ad.tokenizer
    return {
        "ok": True,
        "mode": "tiny" if ctx["args"].tiny else "real",
        "model_key": ad.model_key,
        "repo": entry.repo if entry else None,
        "revision": entry.revision if entry else None,
        "device": str(ad.device),
        "dtype": str(ad.dtype),
        "image_size_hw": list(ad.image_size),
        "llm_grid_hw": list(ad.llm_grid_hw),
        "num_layers": ad.num_layers(),
        "hidden_size": ad.hidden_size(),
        "tied_word_embeddings": ad.tied_word_embeddings,
        "len_tokenizer": len(tok),
        "n_base_vocab": ad.n_base_vocab,
        "embed_rows": ad.model.get_input_embeddings().weight.shape[0],
        "lm_head_rows": ad.model.get_output_embeddings().weight.shape[0],
        "attn_implementation": ad.model.config._attn_implementation,
        "image_processor_class": type(ad.image_processor).__name__,
        "hf_processor_class": type(ad.hf_processor).__name__ if ad.hf_processor is not None else None,
        "assistant_header_ids": {"thinking_off": list(ad.assistant_header_ids(False)),
                                 "thinking_on": list(ad.assistant_header_ids(True))},
        "stop_token_ids": {IM_END: ad.token_id(IM_END), END_OF_TEXT: ad.token_id(END_OF_TEXT)},
        "image": ctx["image_info"],
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "transformers": transformers.__version__},
        # transformers 在 import 时的模块级绑定实际调用的函数（服务器装了 fla 时，GDN 两项是 fla 的 Triton 内核）
        "kernels": {n: fn_name(getattr(mq, n)) for n in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule",
                                                          "causal_conv1d_fn", "causal_conv1d_update") if hasattr(mq, n)},
        # 适配器的前向与生成实际走的 GDN 实现：CPU 上是切换之后的 torch 原函数，GPU 上同上面的绑定（DEV-0003-a1）。
        # 只有带 GDN 的家族（Qwen3.5）的适配器有这个方法，其他家族记 None
        "gdn_in_adapter_forward": ad.gdn_implementation() if hasattr(ad, "gdn_implementation") else None,
    }


def _compare_official(ad, images, instruction, order, thinking, inp):
    """load 路径：适配器的输入张量（加 assistant 头后）与官方 AutoProcessor 的输出逐项相同。"""
    proc = ad.hf_processor
    h, w = ad.image_size
    resized = [ad.resize_image(im) for im in images]
    text = proc.apply_chat_template([{"role": "user", "content": _content(len(images), instruction, order)}],
                                    tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
    enc = proc(text=[text], images=resized, min_pixels=h * w, max_pixels=h * w, return_tensors="pt")
    t = inp.tensors
    header = torch.tensor([ad.assistant_header_ids(thinking)], dtype=torch.long)
    ids = torch.cat([t["input_ids"].cpu(), header], dim=1)
    mm = torch.cat([t["mm_token_type_ids"].cpu(), torch.zeros_like(header)], dim=1)
    res = {
        "input_ids": torch.equal(enc["input_ids"].cpu(), ids),
        "attention_mask": torch.equal(enc["attention_mask"].cpu(), torch.ones_like(ids)),
        "mm_token_type_ids": torch.equal(enc["mm_token_type_ids"].cpu(), mm),
        "image_grid_thw": torch.equal(enc["image_grid_thw"].cpu(), t["image_grid_thw"].cpu()),
        "pixel_values": torch.equal(enc["pixel_values"].to(ad.dtype).cpu(), t["pixel_values"].cpu()),
    }
    return {"ok": all(res.values()), **res}


def check_1_build_generate(ad, ctx):
    args, img, tok = ctx["args"], ctx["img"], ad.tokenizer
    cases = []
    for order in ORDERS:
        inp = ad.build_inputs([img], INSTR_A, order)
        for thinking in (False, True):
            res = ad.generate(inp, enable_thinking=thinking, max_new_tokens=args.max_new_tokens)
            full = inp.tensors["input_ids"][0].tolist() + list(ad.assistant_header_ids(thinking))
            ref = tok.apply_chat_template([{"role": "user", "content": _content(1, INSTR_A, order)}],
                                          tokenize=True, add_generation_prompt=True, enable_thinking=thinking)["input_ids"]
            template_equal = _collapse_image_runs(full, ad.token_id(IMAGE_PAD)) == list(ref)
            official = _compare_official(ad, [img], INSTR_A, order, thinking, inp) if ad.hf_processor is not None else None
            raw_ok = res.raw_text == tok.decode(list(res.token_ids), skip_special_tokens=False)
            ok = (template_equal and raw_ok and 1 <= res.n_generated_tokens <= args.max_new_tokens
                  and (official is None or official["ok"]))
            cases.append({
                "order": order, "enable_thinking": thinking, "ok": ok,
                "template_equal": template_equal, "official_processor": official, "raw_text_is_full_decode": raw_ok,
                "instruction_span": [inp.instruction_span.start, inp.instruction_span.end],
                "image_spans": [[i.span.start, i.span.end] for i in inp.images],
                "n_prompt_tokens": res.n_prompt_tokens, "n_generated_tokens": res.n_generated_tokens,
                "n_visual_tokens": res.n_visual_tokens, "wall_time_s": round(res.wall_time_s, 4),
                "thinking_unclosed": res.thinking_unclosed, "text": res.text[:200], "raw_text": res.raw_text[:300],
            })
    return {"ok": all(c["ok"] for c in cases), "official_processor_compared": ad.hf_processor is not None,
            "cases": cases}


def check_2_task_modulation(ad, ctx):
    img = ctx["img"]
    layers = _layers(ad)
    out = {"instructions": [INSTR_A, INSTR_B]}
    for order in ("image_first", "instruction_first"):
        a, b = ad.build_inputs([img], INSTR_A, order), ad.build_inputs([img], INSTR_B, order)
        lens = [len(a.instruction_span), len(b.instruction_span)]
        same_spans = [i.span for i in a.images] == [i.span for i in b.images]
        with torch.no_grad():
            ha, hb = ad.hidden_states(a, layers), ad.hidden_states(b, layers)
        rows = []
        for l in layers:
            x, y = ha.image[l], hb.image[l]
            d = (x.float() - y.float()).abs()
            rows.append({"layer": l, "equal": bool(torch.equal(x, y)), "max_abs_delta": d.max().item(),
                         "n_tokens_changed": int((d.amax(-1) > 0).sum())})
        eq = [r["equal"] for r in rows]
        expect = all(eq) if order == "image_first" else (eq[0] and not any(eq[1:]))
        out[order] = {"ok": bool(expect and lens[0] == lens[1] and same_spans),
                      "instr_token_lens": lens, "n_image_tokens": int(ha.image[0].shape[0]),
                      "seq_len": int(a.tensors["input_ids"].shape[1]),
                      "equal_layers": f"{sum(eq)}/{len(eq)}", "layers": rows}
    out["ok"] = out["image_first"]["ok"] and out["instruction_first"]["ok"]
    return out


def check_3_grid_tokens(ad, ctx):
    args, img, tok = ctx["args"], ctx["img"], ad.tokenizer
    rows_before = ad.model.get_input_embeddings().weight.shape[0]
    ids = ad.add_grid_tokens(GRID_NAMES)
    allowed = list(ids.values())
    single = {n: tok(n, add_special_tokens=False)["input_ids"] == [i] for n, i in ids.items()}
    emb = ad.model.get_input_embeddings().weight
    head = ad.model.get_output_embeddings().weight
    with torch.no_grad():
        emb_ok = torch.equal(emb[allowed], _base_mean(emb, ad.n_base_vocab).expand(len(allowed), -1))
        head_ok = True if ad.tied_word_embeddings else torch.equal(
            head[allowed], _base_mean(head, ad.n_base_vocab).expand(len(allowed), -1))
    inp = ad.build_inputs([img], INSTR_A, "instruction_first")
    res = ad.generate(inp, enable_thinking=False, max_new_tokens=args.max_new_tokens, allowed_token_ids=allowed)
    in_set = set(res.token_ids) <= set(allowed)
    rows_after = emb.shape[0]
    ok = (in_set and res.n_generated_tokens == args.max_new_tokens and all(single.values())
          and emb_ok and head_ok and rows_after == rows_before)
    return {"ok": ok, "ids": ids, "single_token": single, "rows_before": rows_before, "rows_after": rows_after,
            "mean_init_embed": emb_ok, "mean_init_lm_head": head_ok, "all_generated_in_allowed": in_set,
            "generated_ids": list(res.token_ids), "text": res.text[:200]}


def _query_rule(pos, span, mode):
    p3 = pos[1:, 0].cpu()
    qs, qe = span.start, span.end
    n = qe - qs
    pre = p3[:, qs - 1].tolist()
    t0 = pre[0] + 1
    if mode == "grid2d":
        q = math.isqrt(n)
        want = [[t0] * n, [t0 + r for r in range(q) for _ in range(q)], [t0 + c for _ in range(q) for c in range(q)]]
        after = t0 + q
    else:
        want = [list(range(t0, t0 + n))] * 3
        after = t0 + n
    ok = (len(set(pre)) == 1 and p3[:, qs:qe].tolist() == want and p3[:, qe].tolist() == [after] * 3
          and torch.equal(pos[0, 0].cpu(), torch.arange(pos.shape[-1])))
    return bool(ok), {"p_before_block": pre, "t0": t0, "after_block": after,
                      "first_THW": p3[:, qs : qs + min(n, 6)].T.tolist()}


def check_4_query_tokens(ad, ctx):
    img = ctx["img"]
    layers = _layers(ad)
    n_layers, hidden = ad.num_layers(), ad.hidden_size()
    g = torch.Generator().manual_seed(0)
    qe = (torch.randn(N_QUERY, hidden, generator=g) * 0.02).to(device=ad.device, dtype=ad.dtype)
    seq, modes, span = {}, {}, None
    for mode in ("grid2d", "seq1d"):
        ad.inject_query_tokens(N_QUERY, mode)
        inp = ad.build_inputs([img], INSTR_A, "instruction_first")
        span = inp.query_span
        rule_ok, rule = _query_rule(ad.position_ids(inp), span, mode)
        with torch.no_grad():
            seq[mode] = ad.sequence_hidden_states(inp, layers, query_embeds=qe)
            hs = ad.hidden_states(inp, [0, n_layers], query_embeds=qe)
        shape_ok = (tuple(hs.query[n_layers].shape) == (N_QUERY, hidden)
                    and tuple(hs.image[n_layers].shape) == (sum(len(i.span) for i in inp.images), hidden))
        modes[mode] = {"ok": bool(rule_ok and shape_ok and torch.equal(hs.query[0], qe)),
                       "query_span": [span.start, span.end], "position_rule": rule_ok, **rule,
                       "shape_ok": shape_ok, "layer0_query_equals_query_embeds": bool(torch.equal(hs.query[0], qe))}
    ad.clear_query_tokens()
    qs = span.start
    prefix = [bool(torch.equal(seq["grid2d"][l][:qs], seq["seq1d"][l][:qs])) for l in layers]
    q_differ = [not torch.equal(seq["grid2d"][l][qs : span.end], seq["seq1d"][l][qs : span.end]) for l in layers]
    ok = all(m["ok"] for m in modes.values()) and all(prefix)
    return {"ok": ok, "n_query": N_QUERY, "query_token_id": ad.query_token_id, "modes": modes,
            "prefix_bitwise_equal_by_layer": prefix, "query_states_differ_by_layer": q_differ}


def check_5_position_ids(ad, ctx):
    import contextlib

    imgs = [ctx["img"], ctx["img2"]]
    layers = _layers(ad)
    out = {}
    for order in ORDERS:
        inp = ad.build_inputs(imgs, INSTR_A, order)
        t = inp.tensors
        pos = ad.position_ids(inp)  # 内部已与 get_rope_index 比对
        ref, _ = ad.model.model.get_rope_index(t["input_ids"], mm_token_type_ids=t["mm_token_type_ids"],
                                               image_grid_thw=t["image_grid_thw"], attention_mask=t["attention_mask"])
        rope_eq = bool(torch.equal(pos[1:].cpu(), ref.cpu()))
        row0 = bool(torch.equal(pos[0, 0].cpu(), torch.arange(pos.shape[-1])))
        kw = dict(output_hidden_states=True, logits_to_keep=1)
        with torch.no_grad():
            ours = ad.sequence_hidden_states(inp, layers)
            # 直接调 ad.model 的 HF 原生参照路径，与适配器走同一个 GDN 实现（Qwen3.5 在 CPU 上是 torch，DEV-0003-a1）；
            # 适配器没有 gdn_scope 的家族（Gemma 4、InternVL3.5）不切换
            with getattr(ad, "gdn_scope", contextlib.nullcontext)():
                auto = ad.model(**t, use_cache=False, **kw).hidden_states
                given = ad.model(**t, position_ids=pos, use_cache=False, **kw).hidden_states
                default = ad.model(**t, **kw).hidden_states  # HF 默认（use_cache 取 config），只记录
        given_eq = [bool(torch.equal(auto[l][0], given[l][0])) for l in layers]
        ours_eq = [bool(torch.equal(auto[l][0], ours[l])) for l in layers]
        default_eq = [bool(torch.equal(default[l][0], ours[l])) for l in layers]
        out[order] = {"ok": all(given_eq) and all(ours_eq) and rope_eq and row0,
                      "positions_equal_get_rope_index": rope_eq, "row0_is_arange": row0,
                      "own_position_ids_vs_model_computed": given_eq,
                      "adapter_path_vs_model_path": ours_eq,
                      "info_adapter_path_vs_hf_default_use_cache": default_eq,
                      "seq_len": int(pos.shape[-1])}
    out["ok"] = all(out[o]["ok"] for o in ORDERS)
    return out


def check_6_vision_features(ad, ctx):
    imgs = [ctx["img"], ctx["img2"]]
    gh, gw = ad.llm_grid_hw
    out = {}
    for order in ORDERS:
        inp = ad.build_inputs(imgs, INSTR_A, order)
        with torch.no_grad():
            vf = ad.vision_features(inp)
            h0 = ad.hidden_states(inp, [0]).image[0]
        out[order] = {"ok": bool(tuple(vf.shape) == (2 * gh * gw, ad.hidden_size()) and torch.equal(vf, h0)),
                      "shape": list(vf.shape), "equal": bool(torch.equal(vf, h0))}
    out["ok"] = all(out[o]["ok"] for o in ORDERS)
    return out


def check_7_thinking(ad, ctx):
    args, img, tok = ctx["args"], ctx["img"], ad.tokenizer
    inp = ad.build_inputs([img], INSTR_THINK, "image_first")
    res = ad.generate(inp, enable_thinking=True, max_new_tokens=args.think_max_new_tokens)
    ids = list(res.token_ids)
    end = ad.token_id(THINK_END)
    stops = {ad.token_id(IM_END), ad.token_id(END_OF_TEXT)}
    raw_ok = res.raw_text == tok.decode(ids, skip_special_tokens=False)
    closed = end in ids
    if closed:
        k = len(ids) - 1 - ids[::-1].index(end)
        ans = ids[k + 1 :]
        if ans and ans[-1] in stops:
            ans = ans[:-1]
        expect = tok.decode(ans, skip_special_tokens=False).strip("\n")
        ok = (raw_ok and not res.thinking_unclosed and res.text == expect and THINK_END not in res.text
              and THINK_END in res.raw_text)
    else:
        ok = raw_ok and res.thinking_unclosed and res.text == ""
    off = ad.generate(inp, enable_thinking=False, max_new_tokens=args.max_new_tokens)
    off_ok = (not off.thinking_unclosed) and THINK_END not in off.text
    return {"ok": bool(ok and off_ok), "closed": closed, "closed_case_exercised": closed,
            "n_generated_tokens": res.n_generated_tokens, "max_new_tokens": args.think_max_new_tokens,
            "thinking_unclosed": res.thinking_unclosed, "text": res.text[:300], "raw_text": res.raw_text[:800],
            "thinking_off_control": {"ok": off_ok, "text": off.text[:200], "thinking_unclosed": off.thinking_unclosed}}


def _iter_tensors(args, kwargs):
    items = [(f"arg{i}", a) for i, a in enumerate(args)] + list(kwargs.items())
    for name, v in items:
        if torch.is_tensor(v):
            yield name, v
        elif isinstance(v, (tuple, list)):
            for j, x in enumerate(v):
                if torch.is_tensor(x):
                    yield f"{name}[{j}]", x


def check_8_device_audit(ad, ctx):
    dev = ad.device
    static_bad = [(n, str(t.device)) for n, t in list(ad.model.named_parameters()) + list(ad.model.named_buffers())
                  if t.device != dev][:5]
    census = collections.Counter(str(p.dtype) for p in ad.model.parameters())
    first, calls = {}, [0]

    def make_hook(mod_name):
        def hook(mod, args, kwargs):
            calls[0] += 1
            if first:
                return
            for label, t in _iter_tensors(args, kwargs):
                if t.device != dev:
                    first.update(module=mod_name, input=label, device=str(t.device), expected=str(dev))
                    return
        return hook

    handles = [m.register_forward_pre_hook(make_hook(n or "<root>"), with_kwargs=True)
               for n, m in ad.model.named_modules()]
    try:
        inp = ad.build_inputs([ctx["img"]], INSTR_A, "instruction_first")
        with torch.no_grad():
            ad.hidden_states(inp, [0, ad.num_layers()])
        ad.generate(inp, enable_thinking=False, max_new_tokens=2)
    finally:
        for h in handles:
            h.remove()
    return {"ok": not static_bad and not first and calls[0] > 0, "device": str(dev),
            "static_offenders": static_bad, "first_runtime_offender": first or None,
            "n_modules_hooked": len(handles), "n_hook_calls": calls[0], "param_dtype_census": dict(census)}


CHECKS = {
    "0_facts": check_0_facts,
    "1_build_generate": check_1_build_generate,
    "2_task_modulation": check_2_task_modulation,
    "3_grid_tokens": check_3_grid_tokens,
    "4_query_tokens": check_4_query_tokens,
    "5_position_ids": check_5_position_ids,
    "6_vision_features": check_6_vision_features,
    "7_thinking": check_7_thinking,
    "8_device_audit": check_8_device_audit,
}


def family_suite(args):
    """按家族分派的唯一入口：返回 (setup, check_names, run_order)。没给 --model（只有 --tiny）时是 Qwen3.5。

    setup(args) 返回 (adapter, ctx)，ctx["checks"] 是该家族的检查表（名字 → 函数）。其他家族的构造与检查逻辑放在
    各自的适配器模块里，这里只加入口；它们拿到本模块，复用 base_context、通用检查与常量。
    """
    family = registry.get_model(args.model).family if args.model else registry.FAMILY_QWEN35
    smoke = sys.modules[__name__]
    if family == registry.FAMILY_GEMMA4:  # DEV-0004：构造与检查表在 gemma4.smoke_setup，检查项名字与顺序同 Qwen
        from src.adapters import gemma4

        def setup_gemma4(a):
            return gemma4.smoke_setup(a, DTYPES[a.dtype], base_context(a), smoke)

        return setup_gemma4, CHECK_NAMES, RUN_ORDER
    if family == registry.FAMILY_INTERNVL35:  # DEV-0005：构造、检查表与附加的 9_multi_tile 在 internvl35.smoke_suite
        from src.adapters import internvl35

        return internvl35.smoke_suite(smoke)
    if family != registry.FAMILY_QWEN35:
        raise ValueError(f"冒烟脚本还不支持 {family} 家族（{args.model}）")
    return setup, CHECK_NAMES, RUN_ORDER


def main(argv=None) -> int:
    args = parse_args(argv)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report = {"script": "scripts/smoke_adapter.py", "argv": sys.argv[1:] if argv is None else list(argv),
              "mode": "tiny" if args.tiny else "real",
              "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    t_all = time.perf_counter()
    try:
        setup_fn, check_names, run_order = family_suite(args)
        ad, ctx = setup_fn(args)
    except Exception as e:  # noqa: BLE001 — 失败也先落盘
        report.update(setup_error=repr(e), traceback=traceback.format_exc()[-4000:], all_ok=False)
        out_path.write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str))
        print(f"setup 失败：{e!r}", file=sys.stderr)
        return 2
    results = {}
    for name in run_order:
        t0 = time.perf_counter()
        try:
            r = ctx["checks"][name](ad, ctx)
        except Exception as e:  # noqa: BLE001 — 单项失败不影响其他项，全部落盘后再判
            r = {"ok": False, "error": repr(e), "traceback": traceback.format_exc()[-4000:]}
        r["wall_time_s"] = round(time.perf_counter() - t0, 3)
        results[name] = r
    report["checks"] = {n: results[n] for n in check_names}
    report["ok"] = {n: bool(results[n].get("ok")) for n in check_names}
    report["all_ok"] = all(report["ok"].values())
    report["wall_time_s"] = round(time.perf_counter() - t_all, 2)
    out_path.write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str))
    for n in check_names:
        print(f"{n:22s} {'PASS' if report['ok'][n] else 'FAIL'}  ({results[n]['wall_time_s']} s)")
    print(f"all_ok={report['all_ok']}  wall={report['wall_time_s']} s  -> {out_path}")
    return 0 if report["all_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
