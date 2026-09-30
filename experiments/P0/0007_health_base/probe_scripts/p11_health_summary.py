#!/usr/bin/env python
"""RUN-0007：从 health_<键>.json 生成分数表（所有数字只从 JSON 读）。usage: python p11_health_summary.py <dir> > health_summary.md"""
import json, os, sys
D = sys.argv[1]
ORDER = ['qwen3.5-0.8b', 'qwen3.5-2b', 'gemma-4-e2b-it', 'internvl3.5-2b-hf', 'qwen3.5-4b', 'gemma-4-e4b-it', 'internvl3.5-4b-hf', 'qwen3.5-9b']
Rs = {m: json.load(open(os.path.join(D, f'health_{m}.json'))) for m in ORDER if os.path.exists(os.path.join(D, f'health_{m}.json'))}
pct = lambda s: '—' if s is None or s.get('pct') is None else f"{s['pct']:.1f}（{s['n_correct']}/{s['n']}）"
out = ['## 分数表（D26 协议；IFEval 按 --skip-ifeval-scoring 只生成、未评分）', '',
       '| 模型 | ok | MMLU-Pro 准确率 | GSM8K flexible（标题口径） | GSM8K strict | IFEval |', '|---|---|---|---|---|---|']
for m, R in Rs.items():
    sc = R.get('scores') or {}
    out.append(f"| {m} | {R.get('ok')} | {pct(sc.get('mmlu_pro', {}).get('accuracy'))} | {pct(sc.get('gsm8k', {}).get('flexible'))} | {pct(sc.get('gsm8k', {}).get('strict'))} | 未评分 |")
out += ['', '## 解析失败、写满 max_new_tokens、计算量', '',
        '| 模型 | MMLU-Pro 解析失败（原因） | GSM8K strict 解析失败 | GSM8K flexible 解析失败 | 写满 max_new_tokens：MMLU-Pro(32) / GSM8K(512) / IFEval(1024) | 思考未闭合 / 空回答（三项合计） | 生成 token：MMLU / GSM8K / IFEval / 合计 | 墙钟 s（运行合计 / 三项计时合计） |',
        '|---|---|---|---|---|---|---|---|']
for m, R in Rs.items():
    pt, dg, cp = R.get('parse_tally') or {}, R.get('diagnostics') or {}, R.get('compute') or {}
    f = lambda k: f"{pt[k]['n_failed']}/{pt[k]['n']} {json.dumps(pt[k]['failures'], ensure_ascii=False)}" if k in pt else '—'
    mx = ' / '.join(str(dg.get(b, {}).get('n_reached_max_new_tokens', '—')) for b in ('mmlu_pro', 'gsm8k', 'ifeval'))
    tu = sum(dg.get(b, {}).get('n_thinking_unclosed', 0) for b in ('mmlu_pro', 'gsm8k', 'ifeval'))
    em = sum(dg.get(b, {}).get('n_empty_text', 0) for b in ('mmlu_pro', 'gsm8k', 'ifeval'))
    gt = ' / '.join(str(cp.get(b, {}).get('generated_tokens', '—')) for b in ('mmlu_pro', 'gsm8k', 'ifeval', 'total'))
    out.append(f"| {m} | {f('mmlu_pro')} | {f('gsm8k_strict')} | {f('gsm8k_flexible')} | {mx} | {tu} / {em} | {gt} | {R.get('wall_time_s')} / {round(cp.get('total', {}).get('wall_time_s', 0), 1)} |")
errs = {m: R for m, R in Rs.items() if not R.get('ok')}
if errs:
    out += ['', '## 出错的模型（原样）', '']
    for m, R in errs.items():
        out.append(f"- {m}: `{json.dumps({k: R[k] for k in R if 'error' in k or 'traceback' in k}, ensure_ascii=False)[:3000]}`")
print('\n'.join(out))
