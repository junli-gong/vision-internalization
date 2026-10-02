#!/usr/bin/env python
"""RUN-0025 §3：qwen3.5-2b 的 refexp_box 开发子集，取 RUN-0024 A1 的同一种批划分（冻结批大小 64）里的第 4 批（下标 3），
在同一个进程里用 adapter.generate_batch 重复生成 N 遍，记录每遍每条的 raw_text 与生成 token id，以及与第 1 遍的差别。
只调用现有接口：scripts/run_e1.py 的 data_paths / build_adapter / selection、src.e1.acceptance.take_samples、
src.e1.prompts.load_prompt_config、src.e1.runner 的 batch_keys / batch_plan / load_images / generate_rows、adapter.warmup。
与 run_e1 的一处不同：generate_rows 的 decoding 里多传一个只读的 logits 处理器（TopkRecorder，原样返回 scores），记下每一步
每行 fp32 logits 的前两名（token id 与值），用于第一个分歧位置的读数。它在 HF 的 logits_processor 列表里（generate_batch 接受这个
gen_kwargs，见 batching.run_generate），在前向之后执行，不改 logits。
--deterministic：模型加载之前 torch.use_deterministic_algorithms(True, warn_only=True)，并记录全程的 warnings（去重）。
环境变量（CUBLAS_WORKSPACE_CONFIG、FLA_CACHE_MODE 等）由启动脚本设置，原样记进输出。
usage (在仓库根下): python rerun_batch.py --setting S1 --passes 20 --out <json> [--deterministic] [--model qwen3.5-2b] [--batch-index 3]
"""
import argparse, hashlib, json, os, sys, time, warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
ap = argparse.ArgumentParser()
ap.add_argument("--setting", required=True); ap.add_argument("--passes", type=int, required=True); ap.add_argument("--out", required=True)
ap.add_argument("--deterministic", action="store_true"); ap.add_argument("--model", default="qwen3.5-2b"); ap.add_argument("--batch-index", type=int, default=3)
args = ap.parse_args()
warnings.simplefilter("always")
caught = []
_orig_showwarning = warnings.showwarning
def _show(message, category, filename, lineno, file=None, line=None):
    caught.append({"category": category.__name__, "message": str(message)[:500], "file": filename, "line": lineno})
    _orig_showwarning(message, category, filename, lineno, file, line)
warnings.showwarning = _show

import torch  # noqa: E402
if args.deterministic:
    torch.use_deterministic_algorithms(True, warn_only=True)
import run_e1 as R  # noqa: E402
from transformers import LogitsProcessor  # noqa: E402
from src.e1 import acceptance as A  # noqa: E402
from src.e1.prompts import PROMPTS_PATH, DEFAULT_COUNT_QUESTIONS, load_prompt_config  # noqa: E402
from src.e1.runner import batch_keys, batch_plan, generate_rows, load_images  # noqa: E402


class TopkRecorder(LogitsProcessor):
    """只读：记下每一步每行 scores（fp32）的前两名，原样返回 scores。"""
    def __init__(self):
        self.steps = []
    def __call__(self, input_ids, scores):
        t = torch.topk(scores, 2, dim=-1)
        self.steps.append((t.values.detach().float().cpu(), t.indices.detach().cpu()))
        return scores


a = argparse.Namespace(model=args.model, policy="default", tiny=False, device="cuda", dtype="bf16",
                       data_root=os.environ["VISINT_DATA_ROOT"], nyu_cache=None, dev_subsets=None, eval_subsets=None, nyu_pairs=None)
prompts = load_prompt_config(PROMPTS_PATH)
paths = R.data_paths(a)
t0 = time.perf_counter()
entry, policy, ad = R.build_adapter(a)
load_s = time.perf_counter() - t0
task = "refexp_box"
sel = R.selection(task, "dev", paths, None, DEFAULT_COUNT_QUESTIONS)
samples, take = A.take_samples(task, sel.samples, prompts.acceptance())
bs = prompts.batch_protocol()["batch_size"]
rendered_all = [prompts.render(task, entry.family, s) for s in samples]
plan = batch_plan(batch_keys(prompts.readout(task, "dev"), rendered_all), bs)
lo, hi = plan[args.batch_index]
batch, rendered = samples[lo:hi], rendered_all[lo:hi]
imgs = [load_images(s) for s in batch]
inputs = [ad.build_inputs(im, r.text, prompts.order) for im, r in zip(imgs, rendered)]
max_new = prompts.max_new_tokens(task, entry.family, "dev"); decoding = prompts.decoding(); thinking = prompts.enable_thinking("dev")
warm = ad.warmup(load_images(samples[0]), batch_size=min(bs, len(samples)))
passes = []
for k in range(args.passes):
    rec = TopkRecorder()
    res = generate_rows(ad, inputs, batch_size=bs, enable_thinking=thinking, max_new_tokens=max_new, decoding={**decoding, "logits_processor": [rec]})
    rows = [{"id": s.id, "raw_text": r.raw_text, "token_ids": list(r.token_ids)} for s, r in zip(batch, res)]
    passes.append({"pass": k + 1, "wall_s": res[0].batch_wall_time_s, "rows": rows, "rec": rec,
                   "sha256": hashlib.sha256(json.dumps([x["token_ids"] for x in rows]).encode()).hexdigest()})


def first_div(x, y):
    n = min(len(x), len(y))
    for i in range(n):
        if x[i] != y[i]:
            return i
    return None if len(x) == len(y) else n


ref = passes[0]
summary, first = [], None
for p in passes:
    diffs = []
    for ri, (r0, r1) in enumerate(zip(ref["rows"], p["rows"])):
        if r0["token_ids"] != r1["token_ids"] or r0["raw_text"] != r1["raw_text"]:
            pos = first_div(r0["token_ids"], r1["token_ids"])
            diffs.append({"row": ri, "batch_pos": lo + ri, "id": r0["id"], "first_divergence_token": pos})
            if first is None and pos is not None:
                def top2(rec, step, row):
                    v, ix = rec.steps[step]
                    tok = getattr(ad, "tokenizer", None) or getattr(ad, "_tok", None)  # 只读，用来解码 token 文本
                    return {"ids": [int(t) for t in ix[row]], "tokens": [tok.decode([int(t)]) if tok is not None else None for t in ix[row]],
                            "logits": [float(t) for t in v[row]], "gap": float(v[row][0] - v[row][1])}
                first = {"pass": p["pass"], "row": ri, "id": r0["id"], "step": pos,
                         "pass1": top2(ref["rec"], pos, ri), f"pass{p['pass']}": top2(p["rec"], pos, ri),
                         "pass1_raw_text": r0["raw_text"], f"pass{p['pass']}_raw_text": r1["raw_text"]}
    summary.append({"pass": p["pass"], "n_rows_diff_vs_pass1": len(diffs), "diffs": diffs, "wall_s": p["wall_s"], "sha256": p["sha256"]})
out = {"setting": args.setting, "deterministic_algorithms": args.deterministic,
       "env": {k: os.environ.get(k) for k in ("CUBLAS_WORKSPACE_CONFIG", "FLA_CACHE_MODE", "PYTORCH_CUDA_ALLOC_CONF", "TMPDIR")},
       "versions": {"torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                    "cudnn_deterministic": torch.backends.cudnn.deterministic, "cudnn_benchmark": torch.backends.cudnn.benchmark,
                    "are_deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled()},
       "pid": os.getpid(), "model": entry.key, "revision": entry.revision, "task": task, "take": take, "batch_size": bs,
       "batch_plan_len": len(plan), "batch_index": args.batch_index, "batch_range": [lo, hi],
       "batch_ids_sha256": hashlib.sha256(json.dumps([s.id for s in batch]).encode()).hexdigest(),
       "contains_7909": any(s.id == "refcoco+_unc/val/7909" for s in batch), "max_new_tokens": max_new, "enable_thinking": thinking,
       "load_s": round(load_s, 2), "warmup_wall_s": warm, "passes": summary, "first_divergence": first,
       "warnings_unique": sorted({(w["category"], w["message"], w["file"].split("site-packages/")[-1], w["line"]) for w in caught}),
       "rows_by_pass": [[{"id": r["id"], "raw_text": r["raw_text"], "token_ids": r["token_ids"]} for r in p["rows"]] for p in passes]}
Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False, default=str) + "\n")
print(f"{args.setting} pid {os.getpid()}: passes {len(passes)}, rows differing vs pass 1 per pass: {[s['n_rows_diff_vs_pass1'] for s in summary]}; "
      f"first_divergence {None if first is None else (first['pass'], first['id'], first['step'])}; -> {args.out}", flush=True)
