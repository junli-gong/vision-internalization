#!/usr/bin/env python
"""RUN-0005：从 records.jsonl 生成 summary.md（表 1–3，任务书 §3.7）。所有数字只从 records.jsonl 读。
usage: python p9_summary.py <records.jsonl> <gen_params.json> > summary.md
"""
import json, sys, statistics
recs = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
gp = json.load(open(sys.argv[2]))
ORDER = ['internvl3.5-2b-hf', 'qwen3.5-0.8b', 'gemma-4-e2b-it', 'qwen3.5-2b', 'internvl3.5-4b-hf', 'gemma-4-e4b-it', 'qwen3.5-4b', 'qwen3.5-9b']
CONDS = ['nothink_greedy', 'think_greedy', 'think_sampled']
PIDS = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6']
models = [m for m in ORDER if any(r['model'] == m for r in recs)]
med = lambda xs: (statistics.median(xs) if xs else None)
fmt = lambda x: '—' if x is None else (f'{x:g}' if isinstance(x, (int, float)) else str(x))
out = ['# RUN-0005 summary（由 p9_summary.py 从 records.jsonl 生成）', '',
       f'records: {len(recs)} 条；模型：{", ".join(models)}', '',
       '## 表 1：模型 × 设置（每格应为 n=12）', '',
       '| 模型 | 设置 | n | 开思考 | 闭合 | 写满预算 | 以停止 token 结束 | 思考 token 中位数（仅开思考） | 思考 token 最大（仅开思考） | 回答 token 中位数 | tokens/s 均值 | 墙钟合计 s |',
       '|---|---|---|---|---|---|---|---|---|---|---|---|']
for m in models:
    for c in CONDS:
        rs = [r for r in recs if r['model'] == m and r['condition'] == c]
        if not rs:
            reason = gp.get(m, {}).get('conditions', {}).get(c, {})
            out.append(f"| {m} | {c} | 0 | 跳过：{reason.get('reason', '无记录')} ||||||||||")
            continue
        op = [r for r in rs if r['think_open']]
        tk = [r['n_think_tokens'] for r in op]
        out.append(f"| {m} | {c} | {len(rs)} | {len(op)} | {sum(1 for r in op if r['think_close'])} | {sum(r['hit_max_new_tokens'] for r in rs)} | "
                   f"{sum(r['ended_with_stop_token'] for r in rs)} | {fmt(med(tk))} | {fmt(max(tk) if tk else None)} | {fmt(med([r['n_answer_tokens'] for r in rs]))} | "
                   f"{fmt(round(statistics.mean(r['tokens_per_s'] for r in rs if r['tokens_per_s'] is not None), 2))} | {round(sum(r['wall_time_s'] for r in rs), 1)} |")
out += ['', '## 表 2：think_greedy 下没开思考的提示 id', '', '| 模型 | 没开思考的提示 | 个数 |', '|---|---|---|']
for m in models:
    rs = {r['prompt_id']: r for r in recs if r['model'] == m and r['condition'] == 'think_greedy'}
    no = [p for p in PIDS if p in rs and not rs[p]['think_open']]
    miss = [p for p in PIDS if p not in rs]
    out.append(f"| {m} | {', '.join(no) if no else '（无）'}{'；缺记录：' + ', '.join(miss) if miss else ''} | {len(no)} |")
out += ['', '## 表 3：nothink_greedy 下输出里是否出现任何思考标记', '', '| 模型 | 出现思考标记的提示（open/close/其他标记计数） | 个数 / n |', '|---|---|---|']
for m in models:
    rs = [r for r in recs if r['model'] == m and r['condition'] == 'nothink_greedy']
    hit = [r for r in rs if r['any_think_marker_in_output']]
    desc = ', '.join(f"{r['prompt_id']}(open={r['n_open_markers_in_output']},close={r['n_close_markers_in_output']},extra={r['extra_marker_counts']})" for r in hit)
    out.append(f"| {m} | {desc if desc else '（无）'} | {len(hit)} / {len(rs)} |")
out += ['', '## 附：每条记录一行（模型、提示、设置、生成 token、停止、开/闭、思考/回答 token、墙钟）', '',
        '| 模型 | 提示 | 设置 | 生成 | 停止 | 满预算 | 开 | 闭 | 思考 | 回答 | unclosed | 墙钟 s |', '|---|---|---|---|---|---|---|---|---|---|---|---|']
for m in models:
    for p in PIDS:
        for c in CONDS:
            for r in recs:
                if r['model'] == m and r['prompt_id'] == p and r['condition'] == c:
                    out.append(f"| {m} | {p} | {c} | {r['n_generated_tokens']} | {r['ended_with_stop_token']} | {r['hit_max_new_tokens']} | {r['think_open']} | {r['think_close']} | {r['n_think_tokens']} | {r['n_answer_tokens']} | {r['thinking_unclosed']} | {r['wall_time_s']} |")
print('\n'.join(out))
