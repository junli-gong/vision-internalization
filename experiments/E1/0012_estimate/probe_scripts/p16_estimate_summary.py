#!/usr/bin/env python
"""RUN-0012 §4：从 run_e1 --estimate 的 estimate.json 生成 report 第 3 节的估算表（所有数字只从 JSON 读，不重算外推）。
usage (在 $REPO 下): python p16_estimate_summary.py <estimate_root> <out.json> > estimate_summary.md
"""
import json, os, sys
ROOT, OUT = sys.argv[1:3]
ORDER = ['qwen3.5-0.8b', 'qwen3.5-2b', 'gemma-4-e2b-it', 'internvl3.5-2b-hf', 'qwen3.5-4b', 'gemma-4-e4b-it', 'internvl3.5-4b-hf', 'qwen3.5-9b']
f = lambda v, d=3: '—' if v is None else (f'{v:.{d}f}' if isinstance(v, float) else str(v))
docs = {}
for m in ORDER:
    for sub in ('estimate', 'estimate__qwen_1024', 'estimate__internvl_native'):
        p = os.path.join(ROOT, m, sub, 'estimate.json')
        if os.path.exists(p):
            docs[(m, sub)] = json.load(open(p))
# 子目录名以实际存在的为准
for m in ORDER:
    d = os.path.join(ROOT, m)
    if os.path.isdir(d):
        for sub in os.listdir(d):
            p = os.path.join(d, sub, 'estimate.json')
            if sub.startswith('estimate') and os.path.exists(p) and (m, sub) not in docs:
                docs[(m, sub)] = json.load(open(p))
json.dump({f'{m}/{s}': {'gpu_hours_total': D['gpu_hours_total'], 'header': D['header'],
                        'tasks': [{k: t.get(k) for k in ('task', 'n_dev_used', 'max_new_tokens', 'best_batch_size', 'samples_per_s', 'n', 'gpu_hours')} for t in D['tasks']]}
           for (m, s), D in docs.items()}, open(OUT, 'w'), indent=1, ensure_ascii=False)
default = [(m, D) for (m, s), D in docs.items() if s == 'estimate']
extra = [(m, s, D) for (m, s), D in docs.items() if s != 'estimate']
out = ['#### 估算汇总（主口径；推荐批大小 = 可行的批大小里样本/秒最高者；GPU 小时 = n / 样本每秒，由 run_e1 外推）', '',
       '| 模型 | 任务 | 推荐批大小 | 样本/秒 | n(eval) | GPU h(eval) | 思考实测批大小 | 思考样本/秒 | n(think) | GPU h(think) | n(8192) | GPU h(8192, 上界) |', '|---|---|---|---|---|---|---|---|---|---|---|---|']
for m, D in default:
    for t in D['tasks']:
        g, n, th = t['gpu_hours'], t['n'], t.get('think_measurement') or {}
        out.append(f"| {m} | {t['task']} | {f(t['best_batch_size'])} | {f(t['samples_per_s'], 2)} | {n.get('eval')} | {f(g.get('eval'))} | {f(th.get('batch_size'))} | "
                   f"{f(th.get('samples_per_s'), 2)} | {n.get('think')} | {f(g.get('think'))} | {n.get('think8192')} | {f(g.get('think8192'))} |")
    T = D['gpu_hours_total']
    out.append(f"| {m} | **合计** | | | | **{f(T['eval'])}** | | | | **{f(T['think'])}** | | **{f(T['think8192'])}** |")
out += ['', '#### 每个模型的合计（GPU 小时，主口径）', '', '| 模型 | eval | think | think8192（上界） | eval+think+think8192 |', '|---|---|---|---|---|']
S = {'eval': 0.0, 'think': 0.0, 'think8192': 0.0}
for m, D in default:
    T = D['gpu_hours_total']; tot = sum(T[k] or 0.0 for k in S)
    for k in S:
        S[k] += T[k] or 0.0
    out.append(f"| {m} | {f(T['eval'])} | {f(T['think'])} | {f(T['think8192'])} | {f(tot)} |")
out.append(f"| 已测 {len(default)} 个模型之和 | {f(S['eval'])} | {f(S['think'])} | {f(S['think8192'])} | {f(sum(S.values()))} |")
missing = [m for m in ORDER if m not in dict(default)]
out.append(f"\n未测的模型：{', '.join(missing) if missing else '无'}")
out += ['', '#### 附加口径（只跑 refexp_box、ovd；只外推 extra 子集）', '', '| 模型 | 口径 | 任务 | 推荐批大小 | 样本/秒 | n(extra) | GPU h(extra) | 主口径同任务样本/秒 |', '|---|---|---|---|---|---|---|---|']
for m, s, D in extra:
    base = {t['task']: t for t in docs.get((m, 'estimate'), {'tasks': []})['tasks']}
    for t in D['tasks']:
        out.append(f"| {m} | {D['header']['policy']} | {t['task']} | {f(t['best_batch_size'])} | {f(t['samples_per_s'], 2)} | {t['n'].get('extra')} | {f(t['gpu_hours'].get('extra'))} | {f(base.get(t['task'], {}).get('samples_per_s'), 2)} |")
    out.append(f"| {m} | {D['header']['policy']} | **合计** | | | | **{f(D['gpu_hours_total']['extra'])}** | |")
out += ['', '#### 逐批大小明细（关思考；峰值显存 = torch.cuda.max_memory_allocated）', '', '| 模型 | 口径 | 任务 | 批大小 | ok | 样本/秒 | token/秒 | 平均生成 token | 截断率 | 峰值显存 GB |', '|---|---|---|---|---|---|---|---|---|---|']
for (m, s), D in docs.items():
    for t in D['tasks']:
        for r in t['by_batch_size']:
            pk = r.get('peak_mem_bytes')
            out.append(f"| {m} | {D['header']['policy']} | {t['task']} | {r['batch_size']} | {r['ok']}{' OOM' if r.get('oom') else ''} | {f(r.get('samples_per_s'), 2)} | {f(r.get('tokens_per_s'), 1)} | "
                       f"{f(r.get('mean_generated_tokens'), 1)} | {f(r.get('truncation_rate'), 3)} | {f(pk / 1e9 if pk else None, 2)} |")
out += ['', '#### 开思考的实测（预算 4096；开发样本 ≤ 8 条）', '', '| 模型 | 任务 | 批大小 | 样本数 | ok | 样本/秒 | 平均生成 token | 截断率 | 峰值显存 GB |', '|---|---|---|---|---|---|---|---|---|']
for m, D in default:
    for t in D['tasks']:
        r = t.get('think_measurement')
        if r:
            pk = r.get('peak_mem_bytes')
            out.append(f"| {m} | {t['task']} | {r['batch_size']} | {r.get('n_rows')} | {r['ok']} | {f(r.get('samples_per_s'), 2)} | {f(r.get('mean_generated_tokens'), 1)} | {f(r.get('truncation_rate'), 3)} | {f(pk / 1e9 if pk else None, 2)} |")
print('\n'.join(out))
