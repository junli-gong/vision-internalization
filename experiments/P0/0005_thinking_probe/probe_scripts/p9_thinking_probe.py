#!/usr/bin/env python
"""RUN-0005 思考行为事实探针（Phase 0；不评分、不调提示词）。一个模型跑一次本脚本：12 条提示 × 3 种生成设置。

适配器的构造与冒烟完全相同：复用 scripts/smoke_adapter.py 的 parse_args 与 family_suite(args)，调用返回的 setup(args)，
只取适配器（不跑冒烟检查，不调用 add_grid_tokens / inject_query_tokens，模型保持原样）。

usage (在 $REPO 下):
  python experiments/P0/0005_thinking_probe/probe_scripts/p9_thinking_probe.py --model KEY --image-a A.jpg --image-b B.jpg \
      --out-dir experiments/P0/0005_thinking_probe [--budget-s 14400]
records.jsonl 逐条追加；gen_params.json 按模型键合并；probe_start_epoch.txt 记第一条生成的时刻（4 小时预算从这里算）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import torch  # noqa: E402
from PIL import Image  # noqa: E402

import smoke_adapter as smoke  # noqa: E402
from src.adapters import registry  # noqa: E402

PROMPTS = [  # 任务书 §3.4，原样照抄
    ("A1", "A", "What is the main color in this image? Answer in one word."),
    ("A2", "A", "How many cats are in this image? Answer with a single number."),
    ("A3", "A", "How many remote controls are in this image? Answer with a single number."),
    ("A4", "A", "Locate the cat on the left in this image and output its bounding box."),
    ("A5", "A", "Which cat is closer to the camera, the left one or the right one? Answer 'left' or 'right'."),
    ("A6", "A", "If this image were rotated 90 degrees clockwise, which edge of the image would the remote controls be closest to? Answer top, bottom, left, or right."),
    ("B1", "B", "Describe this image in detail."),
    ("B2", "B", "How many chairs are in this image? Answer with a single number."),
    ("B3", "B", "Which is closer to the camera, the television or the dining table? Answer with one of the two."),
    ("B4", "B", "Output the bounding box of the television in this image."),
    ("B5", "B", "Where is the clock relative to the television? (A) left (B) right (C) above (D) below. Answer with the option letter."),
    ("B6", "B", "Imagine walking from the camera position straight to the television. Which objects would you pass on your left? Think it through, then answer."),
]
CONDITIONS = [("nothink_greedy", False, 512), ("think_greedy", True, 4096), ("think_sampled", True, 4096)]
# 思考标记（字符串；id 运行时从 tokenizer 查）与停止 token（适配器模块常量）
MARKERS = {
    registry.FAMILY_QWEN35: {"open": "<think>", "close": "</think>", "open_in_prompt_when_thinking": True},
    registry.FAMILY_GEMMA4: {"open": "<|channel>", "close": "<channel|>", "open_in_prompt_when_thinking": False, "extra": ["<|think|>"]},
    registry.FAMILY_INTERNVL35: {"open": "<think>", "close": "</think>", "open_in_prompt_when_thinking": False},
}


def sampled_kwargs(entry, family):
    """think_sampled 的参数与出处（任务书 §3.5）。返回 (kwargs 或 None, 说明)。"""
    if family == registry.FAMILY_INTERNVL35:
        return {"do_sample": True, "temperature": 0.6}, ("model card §Thinking Mode 原文: \"When enabling Thinking mode, we recommend setting "
                                                        "`do_sample=True` and `temperature=0.6`\"; generation_config.json 无采样参数")
    from huggingface_hub import HfApi, hf_hub_download
    files = {s.rfilename for s in HfApi().model_info(entry.repo, revision=entry.revision).siblings}
    if "generation_config.json" not in files:
        return None, f"{entry.repo}@{entry.revision} 没有 generation_config.json → 按任务书跳过 think_sampled"
    gc = json.load(open(hf_hub_download(entry.repo, "generation_config.json", revision=entry.revision)))
    if gc.get("do_sample") is not True:
        return None, f"generation_config.json 的 do_sample={gc.get('do_sample')!r}（不是 True）→ 按任务书跳过 think_sampled"
    kw = {k: gc[k] for k in ("do_sample", "temperature", "top_p", "top_k") if k in gc}
    return kw, f"{entry.repo}@{entry.revision} generation_config.json 原样: {json.dumps(gc)}"


def analyse(ids, family, thinking, mk_ids, stop_ids):
    """按任务书 §3.6 的口径从生成的 token id 计算思考段字段。"""
    m = MARKERS[family]
    o, c = mk_ids["open"], mk_ids["close"]
    n = len(ids)
    trailing_stop = n > 0 and ids[-1] in stop_ids
    body_end = n - 1 if trailing_stop else n  # 末尾停止 token 不计入思考/回答
    open_pos = [i for i, t in enumerate(ids) if t == o]
    close_pos = [i for i, t in enumerate(ids) if t == c]
    rec = {"n_open_markers_in_output": len(open_pos), "n_close_markers_in_output": len(close_pos)}
    if thinking and m["open_in_prompt_when_thinking"]:
        think_open, src, start = True, "prompt", 0
    elif open_pos:
        think_open, src, start = True, "output", open_pos[0] + 1
    else:
        think_open, src, start = False, None, None
    rec["think_open"], rec["think_open_source"] = think_open, src
    if think_open:
        closes_after = [p for p in close_pos if p >= start]
        think_close = bool(closes_after)
        end = closes_after[0] if think_close else body_end
        rec["think_close"] = think_close
        rec["n_think_tokens"] = max(0, end - start)
        if think_close:
            last_close = max(close_pos)
            rec["n_answer_tokens"] = sum(1 for t in ids[last_close + 1:] if t not in stop_ids)
        else:
            rec["n_answer_tokens"] = 0
    else:
        rec["think_close"] = bool(close_pos)
        rec["n_think_tokens"] = 0
        rec["n_answer_tokens"] = sum(1 for t in ids if t not in stop_ids)
    extra = {s: sum(1 for t in ids if t == i) for s, i in mk_ids.get("extra", {}).items()}
    rec["any_think_marker_in_output"] = bool(open_pos or close_pos or any(extra.values()))
    rec["extra_marker_counts"] = extra
    return rec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--image-a", required=True)
    p.add_argument("--image-b", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--budget-s", type=float, default=4 * 3600)
    a = p.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    entry = registry.get_model(a.model); family = entry.family
    sargs = smoke.parse_args(["--model", a.model, "--device", "cuda", "--dtype", "bf16", "--image", a.image_a,
                              "--out", str(Path(os.environ.get("TMPDIR", "/tmp")) / f"p9_unused_{a.model}.json")])
    setup_fn, _, _ = smoke.family_suite(sargs)
    t_load = time.perf_counter()
    ad, _ctx = setup_fn(sargs)
    load_s = time.perf_counter() - t_load
    imgs = {"A": Image.open(a.image_a).convert("RGB"), "B": Image.open(a.image_b).convert("RGB")}
    img_names = {"A": Path(a.image_a).name, "B": Path(a.image_b).name}
    m = MARKERS[family]

    def tid(tok_str):
        """标记/停止 token 的 id 一律从 tokenizer 查（任务书 §3.6），并确认它是单个 token。"""
        i = ad.tokenizer.convert_tokens_to_ids(tok_str)
        enc = ad.tokenizer(tok_str, add_special_tokens=False)["input_ids"]
        if i is None or i == ad.tokenizer.unk_token_id or enc != [i]:
            raise ValueError(f"{tok_str!r} 不是单个 token：convert_tokens_to_ids={i}, encode={enc}")
        return i

    mk_ids = {"open": tid(m["open"]), "close": tid(m["close"]), "extra": {s: tid(s) for s in m.get("extra", [])}}
    stop_names = {registry.FAMILY_QWEN35: "qwen35", registry.FAMILY_GEMMA4: "gemma4", registry.FAMILY_INTERNVL35: "internvl35"}[family]
    mod = __import__(f"src.adapters.{stop_names}", fromlist=["STOP_TOKENS"])
    stop_ids = {tid(s): s for s in mod.STOP_TOKENS}
    skw, skw_src = sampled_kwargs(entry, family)
    gp_path = out / "gen_params.json"
    gp = json.loads(gp_path.read_text()) if gp_path.exists() else {}
    gp[a.model] = {"repo": entry.repo, "revision": entry.revision, "family": family,
                   "adapter_settings": {k: getattr(ad, k, None) for k in ("image_size", "max_soft_tokens", "tile_policy") if getattr(ad, k, None) is not None},
                   "conditions": {"nothink_greedy": {"enable_thinking": False, "max_new_tokens": 512, "gen_kwargs": {}, "decoding": "greedy (adapter default do_sample=False)"},
                                  "think_greedy": {"enable_thinking": True, "max_new_tokens": 4096, "gen_kwargs": {}, "decoding": "greedy (adapter default do_sample=False)"},
                                  "think_sampled": ({"enable_thinking": True, "max_new_tokens": 4096, "gen_kwargs": skw, "source": skw_src} if skw is not None
                                                    else {"skipped": True, "reason": skw_src})},
                   "think_marker_ids": {m["open"]: mk_ids["open"], m["close"]: mk_ids["close"], **mk_ids["extra"]},
                   "think_open_in_prompt_when_thinking": m["open_in_prompt_when_thinking"],
                   "stop_token_ids": {v: k for k, v in stop_ids.items()},
                   "seed_rule": "torch.manual_seed(s); torch.cuda.manual_seed_all(s); s = 1000 + prompt index (A1=0 … B6=11), set before every generation",
                   "load_s": round(load_s, 2)}
    gp_path.write_text(json.dumps(gp, indent=1, ensure_ascii=False, default=str))
    start_file = out / "probe_start_epoch.txt"
    rec_path = out / "records.jsonl"
    n_done = 0
    for pi, (pid, imk, prompt) in enumerate(PROMPTS):
        for cond, thinking, mnt in CONDITIONS:
            if cond == "think_sampled" and skw is None:
                continue
            if start_file.exists() and time.time() - float(start_file.read_text()) > a.budget_s:
                print(f"BUDGET EXCEEDED before {a.model} {pid} {cond}", flush=True)
                (out / "budget_stop.txt").write_text(json.dumps({"model": a.model, "prompt_id": pid, "condition": cond, "utc": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}))
                return 4
            if not start_file.exists():
                start_file.write_text(str(time.time()))
            s = 1000 + pi
            torch.manual_seed(s); torch.cuda.manual_seed_all(s)
            kw = dict(skw) if cond == "think_sampled" else {}
            inputs = ad.build_inputs([imgs[imk]], prompt, "image_first")
            try:
                res = ad.generate(inputs, enable_thinking=thinking, max_new_tokens=mnt, **kw)
            except Exception as e:  # noqa: BLE001
                with open(out / "errors.jsonl", "a") as f:
                    f.write(json.dumps({"model": a.model, "prompt_id": pid, "condition": cond, "error": repr(e), "traceback": traceback.format_exc()[-3000:]}) + "\n")
                print(f"ERROR {a.model} {pid} {cond}: {e!r}", flush=True)
                continue
            ids = list(res.token_ids)
            r = {"model": a.model, "prompt_id": pid, "image": img_names[imk], "condition": cond, "enable_thinking": thinking,
                 "gen_kwargs": kw, "seed": s, "max_new_tokens": mnt, "n_prompt_tokens": res.n_prompt_tokens,
                 "n_visual_tokens": res.n_visual_tokens, "n_generated_tokens": res.n_generated_tokens,
                 "ended_with_stop_token": bool(ids) and ids[-1] in stop_ids,
                 "hit_max_new_tokens": res.n_generated_tokens == mnt and not (bool(ids) and ids[-1] in stop_ids)}
            r.update(analyse(ids, family, thinking, mk_ids, stop_ids))
            r.update({"thinking_unclosed": res.thinking_unclosed, "text": res.text, "raw_text": res.raw_text,
                      "wall_time_s": round(res.wall_time_s, 3),
                      "tokens_per_s": round(res.n_generated_tokens / res.wall_time_s, 2) if res.wall_time_s > 0 else None})
            with open(rec_path, "a") as f:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n_done += 1
            print(f"{a.model} {pid} {cond}: gen={res.n_generated_tokens} stop={r['ended_with_stop_token']} open={r['think_open']} close={r['think_close']} think={r['n_think_tokens']} ans={r['n_answer_tokens']} {r['wall_time_s']}s", flush=True)
    print(f"DONE {a.model} n={n_done}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
