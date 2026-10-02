#!/usr/bin/env python
"""RUN-0016 §5：从 run_e1 --estimate 的 estimate.json 生成 report 第 3 节的表（在 RUN-0013 的 p16_estimate_summary.py 基础上改）。
数字只从文件读，不重算外推：
- 本次：experiments/E1/0016_estimate/<键>/estimate*/estimate.json；
- 思考吞吐对照：RUN-0013 的 experiments/E1/0013_estimate/<键>/estimate/estimate.json（think_measurement，批大小 8、8 条）；
- nvidia-smi 峰值：按驱动日志里每个模型的起止时间，把 *_apps.csv（每 5 s 一次，按进程）的样本归到模型，取 used_memory 的最大值。
usage (在 $REPO 下): python p20_estimate_summary.py <out.json> > estimate_summary.md
"""
import csv, json, os, re, sys
OUT = sys.argv[1]
R16, R13 = 'experiments/E1/0016_estimate', 'experiments/E1/0013_estimate'
ORDER = ['qwen3.5-4b', 'gemma-4-e4b-it', 'internvl3.5-4b-hf', 'qwen3.5-9b', 'qwen3.5-0.8b', 'qwen3.5-2b', 'gemma-4-e2b-it', 'internvl3.5-2b-hf']
SMALL = ['qwen3.5-0.8b', 'qwen3.5-2b', 'gemma-4-e2b-it', 'internvl3.5-2b-hf']
f = lambda v, d=3: '—' if v is None else (f'{v:.{d}f}' if isinstance(v, float) else str(v))


def load(root):
    docs = {}
    for m in ORDER:
        d = os.path.join(root, m)
        if os.path.isdir(d):
            for sub in sorted(os.listdir(d)):
                p = os.path.join(d, sub, 'estimate.json')
                if sub.startswith('estimate') and os.path.exists(p):
                    docs[(m, sub)] = json.load(open(p))
    return docs


D16, D13 = load(R16), load(R13)
default = [(m, D) for (m, s), D in D16.items() if s == 'estimate']
extra = [(m, s, D) for (m, s), D in D16.items() if s != 'estimate']


def sps_at(t, bs):
    r = next((r for r in t['by_batch_size'] if r['batch_size'] == bs), None)
    return None if r is None else (r.get('samples_per_s') if r.get('ok') else 'OOM' if r.get('oom') else 'fail')


out = ['#### 估算汇总（主口径；外推按 extrapolated_with 的批大小；GPU 小时 = n / 样本每秒，由 run_e1 外推）', '',
       '| 模型 | 任务 | 开发样本 | max_new_tokens | 外推用的批大小 | 样本/秒 @16 / @32 / @64 | 推荐批大小 | n(eval) | GPU h(eval) | n(free) | GPU h(free) | 思考实测批大小 | 思考样本/秒 | n(think) | GPU h(think) | n(8192) | GPU h(8192, 上界) |',
       '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
for m, D in default:
    for t in D['tasks']:
        g, n, th, ex = t['gpu_hours'], t['n'], t.get('think_measurement') or {}, t.get('extrapolated_with') or {}
        s3 = ' / '.join(f(sps_at(t, b), 2) if not isinstance(sps_at(t, b), str) else sps_at(t, b) for b in (16, 32, 64))
        out.append(f"| {m} | {t['task']} | {t['n_dev_used']} | {t['max_new_tokens']} | {f(ex.get('batch_size'))} | {s3} | {f(t.get('best_batch_size'))} | "
                   f"{n.get('eval')} | {f(g.get('eval'))} | {n.get('free', '—')} | {f(g.get('free'))} | {f(th.get('batch_size'))} | {f(th.get('samples_per_s'), 2)} | "
                   f"{n.get('think')} | {f(g.get('think'))} | {n.get('think8192')} | {f(g.get('think8192'))} |")
    T = D['gpu_hours_total']
    out.append(f"| {m} | **合计** | | | | | | | **{f(T.get('eval'))}** | | **{f(T.get('free'))}** | | | | **{f(T.get('think'))}** | | **{f(T.get('think8192'))}** |")
out += ['', '#### 每个模型的合计（GPU 小时，主口径）', '', '| 模型 | eval | free | think | think8192（上界） | 合计 |', '|---|---|---|---|---|---|']
S = {'eval': 0.0, 'free': 0.0, 'think': 0.0, 'think8192': 0.0}
for m, D in default:
    T = D['gpu_hours_total']; tot = sum(T.get(k) or 0.0 for k in S)
    for k in S:
        S[k] += T.get(k) or 0.0
    out.append(f"| {m} | {f(T.get('eval'))} | {f(T.get('free'))} | {f(T.get('think'))} | {f(T.get('think8192'))} | {f(tot)} |")
out.append(f"| 已测 {len(default)} 个模型之和 | {f(S['eval'])} | {f(S['free'])} | {f(S['think'])} | {f(S['think8192'])} | {f(sum(S.values()))} |")
missing = [m for m in ORDER if m not in dict(default)]
out.append(f"\n未测的模型：{', '.join(missing) if missing else '无'}")
out += ['', '#### 附加口径（只跑 refexp_box、ovd；--batch-sizes 16 32 64；只外推 extra 子集）', '',
        '| 模型 | 口径 | 任务 | 外推用的批大小 | 样本/秒 @16 / @32 / @64 | n(extra) | GPU h(extra) | 主口径同任务样本/秒（外推用的批大小） |', '|---|---|---|---|---|---|---|---|']
for m, s, D in extra:
    base = {t['task']: t for t in D16.get((m, 'estimate'), {'tasks': []})['tasks']}
    for t in D['tasks']:
        s3 = ' / '.join(f(sps_at(t, b), 2) if not isinstance(sps_at(t, b), str) else sps_at(t, b) for b in (16, 32, 64))
        out.append(f"| {m} | {D['header']['policy']} | {t['task']} | {f((t.get('extrapolated_with') or {}).get('batch_size'))} | {s3} | {t['n'].get('extra')} | "
                   f"{f(t['gpu_hours'].get('extra'))} | {f(base.get(t['task'], {}).get('samples_per_s'), 2)} |")
    out.append(f"| {m} | {D['header']['policy']} | **合计** | | | | **{f(D['gpu_hours_total'].get('extra'))}** | |")
out += ['', '#### 开思考的实测（预算 4096；--estimate-think-n 64）', '', '| 模型 | 任务 | 批大小 | 样本数 | ok | 样本/秒 | 平均生成 token | 截断率 | 峰值显存 GB |', '|---|---|---|---|---|---|---|---|---|']
for m, D in default:
    for t in D['tasks']:
        r = t.get('think_measurement')
        if r:
            pk = r.get('peak_mem_bytes')
            out.append(f"| {m} | {t['task']} | {r['batch_size']} | {r.get('n_rows')} | {r['ok']}{' OOM' if r.get('oom') else ''} | {f(r.get('samples_per_s'), 3)} | "
                       f"{f(r.get('mean_generated_tokens'), 1)} | {f(r.get('truncation_rate'), 3)} | {f(pk / 1e9 if pk else None, 2)} |")
out += ['', '#### free 子集的实测（关思考，上限 1024；只有选择题与点对四个任务）', '', '| 模型 | 任务 | 批大小 | 样本数 | ok | 样本/秒 | 平均生成 token | 截断率 |', '|---|---|---|---|---|---|---|---|']
for m, D in default:
    for t in D['tasks']:
        r = t.get('free_measurement')
        if r:
            out.append(f"| {m} | {t['task']} | {r['batch_size']} | {r.get('n_rows')} | {r['ok']} | {f(r.get('samples_per_s'), 2)} | {f(r.get('mean_generated_tokens'), 1)} | {f(r.get('truncation_rate'), 3)} |")
out += ['', '#### 同模型的思考吞吐对照（RUN-0013：批大小 8、8 条；本次：见批大小列、64 条）', '',
        '| 模型 | 任务 | RUN-0013 批大小 / 样本数 / 样本/秒 | 本次 批大小 / 样本数 / 样本/秒 | 本次 ÷ RUN-0013 |', '|---|---|---|---|---|']
cmp = []
for m, D in default:
    old = {t['task']: t.get('think_measurement') for t in D13.get((m, 'estimate'), {'tasks': []})['tasks']} if m in SMALL else {}
    for t in D['tasks']:
        r = t.get('think_measurement') or {}
        o = old.get(t['task']) if m in SMALL else None
        ratio = (r['samples_per_s'] / o['samples_per_s']) if (o and r.get('samples_per_s') and o.get('samples_per_s')) else None
        os_ = f"{o['batch_size']} / {o.get('n_rows')} / {f(o.get('samples_per_s'), 3)}" if o else '—'
        out.append(f"| {m} | {t['task']} | {os_} | {f(r.get('batch_size'))} / {f(r.get('n_rows'))} / {f(r.get('samples_per_s'), 3)} | {f(ratio, 2)} |")
        cmp.append({'model': m, 'task': t['task'], 'run0013': o and {k: o.get(k) for k in ('batch_size', 'n_rows', 'samples_per_s')}, 'run0016': {k: r.get(k) for k in ('batch_size', 'n_rows', 'samples_per_s')}, 'ratio': ratio})


def windows(log, start_pat, end_pats):
    """driver.log → {model: [(start, end)]}，时间取 ISO 里的 HH:MM:SS（本运行与 RUN-0013 都不跨午夜）。"""
    w, cur = {}, {}
    for line in open(log):
        m = re.match(r'=== (\S+) ' + start_pat + r' (\d{4}-\d\d-\d\dT(\d\d:\d\d:\d\d))', line)
        if m:
            cur[m.group(1)] = m.group(3); continue
        for ep in end_pats:
            m = re.match(r'=== (\S+) ' + ep + r'.* (\d{4}-\d\d-\d\dT(\d\d:\d\d:\d\d))', line)
            if m and m.group(1) in cur:
                w.setdefault(m.group(1), []).append((cur[m.group(1)], m.group(3)))
    return {k: (min(a for a, _ in v), max(b for _, b in v)) for k, v in w.items()}


def peaks(apps_csvs, win):
    rows = []
    for p in apps_csvs:
        if os.path.exists(p):
            rows += [(r[0], int(r[3].split()[0])) for r in csv.reader(open(p), skipinitialspace=True) if len(r) == 4]
    return {m: max([mem for t, mem in rows if a <= t <= b] or [None], key=lambda x: -1 if x is None else x) for m, (a, b) in win.items()}


w16 = windows('experiments/E1/0016_estimate/logs/driver.log', r'estimate start', [r'estimate rc=', r'estimate policy \S+ rc='])
p16 = peaks(['experiments/E1/0016_estimate/logs/est_apps.csv'], w16)
w13e = windows('experiments/E1/0013_estimate/logs/driver.log', r'estimate start', [r'estimate rc=', r'estimate policy \S+ rc='])
p13e = peaks(['experiments/E1/0013_estimate/logs/chain_apps.csv'], w13e)
w13a = windows('experiments/E1/0013_acceptance/logs/driver.log', r'accept start', [r'accept count/countbenchqa outcome \(resume\) rc='])
p13a = peaks(['experiments/E1/0013_acceptance/logs/chain_apps_upto_acceptance_end.csv'], w13a)
out += ['', '#### nvidia-smi 峰值（按进程的 used_memory，MiB；每 5 s 采样；本次设了 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True，RUN-0013 没设）', '',
        '| 模型 | 本次（估算，含附加口径） | RUN-0013 估算（含附加口径） | RUN-0013 验收 | 本次 torch max_memory_allocated 的最大值 GB |', '|---|---|---|---|---|']
for m in ORDER:
    D = D16.get((m, 'estimate'))
    tm = None
    for (mm, s), DD in D16.items():
        if mm == m:
            for t in DD['tasks']:
                for r in t['by_batch_size'] + [x for x in (t.get('think_measurement'), t.get('free_measurement')) if x]:
                    if r.get('peak_mem_bytes'):
                        tm = max(tm or 0, r['peak_mem_bytes'])
    out.append(f"| {m} | {f(p16.get(m))} | {f(p13e.get(m))} | {f(p13a.get(m))} | {f(tm / 1e9 if tm else None, 2)} |")
json.dump({'tables': {f'{m}/{s}': {'gpu_hours_total': D['gpu_hours_total'], 'header': D['header'],
                                   'tasks': [{k: t.get(k) for k in ('task', 'n_dev_used', 'max_new_tokens', 'best_batch_size', 'extrapolated_with', 'samples_per_s', 'n', 'gpu_hours')} for t in D['tasks']]}
                      for (m, s), D in D16.items()},
           'think_comparison': cmp, 'nvidia_smi_peak_mib': {'run0016': p16, 'run0013_estimate': p13e, 'run0013_acceptance': p13a},
           'windows': {'run0016': w16, 'run0013_estimate': w13e, 'run0013_acceptance': w13a}}, open(OUT, 'w'), indent=1, ensure_ascii=False)
print('\n'.join(out))
