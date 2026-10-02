#!/usr/bin/env python
"""RUN-0017 §3.1：GPU bf16 上 option_logits 的字母读数与受限生成的逐题核对（只读不改；只调用仓库里现有的函数）。

- 适配器：scripts/health_check.py 的 make_adapter（同一个 factory.make_adapter、同样的参数：--device cuda --dtype bf16）。
- 题目：configs/health_check_items.json（load_items，核对 sha256）的 MMLU-Pro 前 N 题；提示 P.PROMPT_BUILDERS["mmlu_pro"]，
  选项字母 P.option_letters(本题选项数)。
- 每题：
  1. src.health.run.letter_text_only(adapter, prompt, letters)：零张图、P.ORDER、关思考，一次 option_logits；
  2. adapter.generate(build_inputs([], prompt, P.ORDER), enable_thinking=P.ENABLE_THINKING, max_new_tokens=1,
     allowed_token_ids=本题全部字母的全部单 token 写法（取自第 1 步返回的 spellings）, **P.GREEDY_KWARGS)；
  3. 记三项：生成的 token 与 option_logits 选中的 token（token_ids[选中字母]）是否相同；两者对应的字母是否相同；
     第 1 步选中的字母与 health_<键>.json 里同一题 mmlu_pro_letter 的 choice 是否相同。
不计入记账，不改分数。
usage (在 $REPO 下): python audit_letter.py <模型键> <health_<键>.json> <out.json> [N=100]
"""
import argparse, json, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import health_check as HC  # noqa: E402  （只用它的 make_adapter）
from src.health import protocol as P  # noqa: E402
from src.health.items import load_items  # noqa: E402
from src.health.run import letter_text_only  # noqa: E402

key, health_path, out_path = sys.argv[1:4]
n = int(sys.argv[4]) if len(sys.argv) > 4 else 100
items_obj, items_sha = load_items()
items = items_obj["items"]["mmlu_pro"][:n]
health = json.load(open(health_path))
hc = {r["id"]: r for r in health["records"]["mmlu_pro_letter"]}

t0 = time.perf_counter()
ad, entry, vision = HC.make_adapter(argparse.Namespace(model=key, device="cuda", dtype="bf16", tiny=False))
load_s = time.perf_counter() - t0
rows = []
t1 = time.perf_counter()
for item in items:
    prompt = P.PROMPT_BUILDERS["mmlu_pro"](item)
    letters = P.option_letters(len(item["options"]))
    scores, _ = letter_text_only(ad, prompt, letters)
    allowed = sorted({sp.token_id for sps in scores.spellings for sp in sps})
    tok_to_letter = {sp.token_id: o for o, sps in zip(scores.options, scores.spellings) for sp in sps}
    chosen_tok = scores.token_ids[list(scores.options).index(scores.choice)]
    inputs = ad.build_inputs([], prompt, P.ORDER)
    res = ad.generate(inputs, enable_thinking=P.ENABLE_THINKING, allowed_token_ids=allowed, max_new_tokens=1, **dict(P.GREEDY_KWARGS))
    gen_tok = res.token_ids[0] if res.token_ids else None
    h = hc.get(item["id"])
    rows.append({"id": item["id"], "letters": letters, "n_allowed_token_ids": len(allowed),
                 "option_logits_choice": scores.choice, "option_logits_token_id": chosen_tok,
                 "generate_token_id": gen_tok, "generate_n_tokens": len(res.token_ids), "generate_raw_text": res.raw_text,
                 "generate_letter": tok_to_letter.get(gen_tok),
                 "health_choice": None if h is None else h.get("choice"),
                 "same_token": gen_tok == chosen_tok, "same_letter": tok_to_letter.get(gen_tok) == scores.choice,
                 "same_as_health": h is not None and h.get("choice") == scores.choice})
out = {"model": key, "revision": entry.revision, "items_sha256": items_sha, "n": len(rows), "load_s": round(load_s, 2),
       "audit_wall_s": round(time.perf_counter() - t1, 2),
       "n_same_token": sum(r["same_token"] for r in rows), "n_same_letter": sum(r["same_letter"] for r in rows),
       "n_same_as_health": sum(r["same_as_health"] for r in rows),
       "health_json": health_path, "health_protocol_version": health.get("protocol_version"),
       "method": {"option_logits": "src.health.run.letter_text_only", "generate": "adapter.generate(max_new_tokens=1, "
                  "enable_thinking=P.ENABLE_THINKING, allowed_token_ids=all single-token spellings of the item's letters, **P.GREEDY_KWARGS)",
                  "adapter": "scripts/health_check.make_adapter(model, device=cuda, dtype=bf16, tiny=False)"},
       "rows": rows}
Path(out_path).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"{key}: same_token {out['n_same_token']}/{out['n']}  same_letter {out['n_same_letter']}/{out['n']}  "
      f"same_as_health {out['n_same_as_health']}/{out['n']}  load {out['load_s']}s audit {out['audit_wall_s']}s -> {out_path}")
