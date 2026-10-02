#!/usr/bin/env python
"""RUN-0018 §5：从 run_e5 的输出（<out>/<系统>/<模型或基线>/<任务>__dev/ 的 metrics.json、compute.json、predictions.jsonl、
logs/run_*.json）生成 report 第 3 节的表。数字只从文件读，不重算指标。
- 样本/秒 = n / compute.json 的 wall_time_s（只含 generate）；批大小取 logs/run_*.json 的 batch_size。
- torch 的峰值显存：run_e5 不记录（输出里没有这个字段），表里记"未记录"。nvidia-smi 峰值：按 driver.log 里每个模型的起止时间，
  把 e5_apps.csv（每 5 s，按进程）的样本归到模型，取 used_memory 的最大值。
usage (在 $REPO 下): python p22_e5_summary.py <out_dir> <driver.log> <apps.csv> <out.json> > e5_summary.md
"""
import csv, glob, json, os, re, sys
OUT_DIR, DRIVER, APPS, OUT = sys.argv[1:5]
ORDER = ['qwen3.5-0.8b', 'qwen3.5-2b', 'gemma-4-e2b-it', 'internvl3.5-2b-hf', 'qwen3.5-4b', 'gemma-4-e4b-it', 'internvl3.5-4b-hf', 'qwen3.5-9b']
TASKS = ['rotation3d', 'viewpoint', 'occlusion', 'physics', 'imagery']
f = lambda v, d=3: '—' if v is None else (f'{v:.{d}f}' if isinstance(v, float) else str(v))


def leaf(system, who, task):
    d = os.path.join(OUT_DIR, system, who, f'{task}__dev')
    if not os.path.exists(os.path.join(d, 'metrics.json')):
        return None
    M = json.load(open(os.path.join(d, 'metrics.json')))
    C = json.load(open(os.path.join(d, 'compute.json'))) if os.path.exists(os.path.join(d, 'compute.json')) else None
    runs = sorted(glob.glob(os.path.join(d, 'logs', 'run_*.json')))
    bs = json.load(open(runs[-1])).get('batch_size') if runs else None
    P = [json.loads(l) for l in open(os.path.join(d, 'predictions.jsonl'))] if os.path.exists(os.path.join(d, 'predictions.jsonl')) else []
    first_fail = next((r for r in P if r.get('failure')), None)
    return {'M': M, 'C': C, 'bs': bs, 'first_fail': first_fail}


def windows():
    w, cur = {}, {}
    for line in open(DRIVER):
        m = re.match(r'=== (\S+) start (\d{4}-\d\d-\d\dT(\d\d:\d\d:\d\d))', line)
        if m:
            cur[m.group(1)] = m.group(3); continue
        m = re.match(r'=== (\S+) run_e5 rc=\d+ (\d{4}-\d\d-\d\dT(\d\d:\d\d:\d\d))', line)
        if m and m.group(1) in cur:
            w[m.group(1)] = (cur[m.group(1)], m.group(3))
    return w


rows = [(r[0], int(r[3].split()[0])) for r in csv.reader(open(APPS), skipinitialspace=True) if len(r) == 4] if os.path.exists(APPS) else []
W = windows()
smi = {m: max([mem for t, mem in rows if a <= t <= b] or [None], key=lambda x: -1 if x is None else x) for m, (a, b) in W.items()}

res = {'models': {}, 'baselines': {}, 'nvidia_smi_peak_mib': smi, 'windows': W}
out = ['#### 逐模型 × 逐系统 × 逐任务：条数、有效率、解析失败、截断、生成量与吞吐（开发集 val，每档 4 条）', '',
       '| 模型 | 系统 | 任务 | n | 有效率（mask_iou_grid 的 n_valid/n） | 解析失败（按原因） | 截断率 | 思考未闭合 | 平均生成 token | 批大小 | generate 墙钟 s | 样本/秒 |',
       '|---|---|---|---|---|---|---|---|---|---|---|---|']
for m in ORDER:
    for system in ('zeroshot', 'cot'):
        for t in TASKS:
            x = leaf(system, m, t)
            if x is None:
                continue
            M, C = x['M'], x['C']; ov = M['overall']['mask_iou_grid']
            wall = C['wall_time_s'] if C else None
            res['models'].setdefault(m, {}).setdefault(system, {})[t] = {
                'n': M['n'], 'n_valid': ov['n_valid'], 'parse': M['parse'], 'truncation': M.get('truncation'), 'thinking_unclosed': M.get('thinking_unclosed'),
                'generated_tokens': M.get('generated_tokens'), 'batch_size': x['bs'], 'wall_time_s': wall, 'compute_generated_tokens': C and C['generated_tokens'],
                'overall': M['overall'], 'headline': M['headline'], 'headline_scope': M.get('headline_scope'), 'refusal': M.get('refusal'),
                'first_failure': None if x['first_fail'] is None else {'id': x['first_fail']['id'], 'failure': x['first_fail']['failure'], 'raw_text_600': (x['first_fail'].get('raw_text') or '')[:600]}}
            tr = M.get('truncation') or {}; tu = M.get('thinking_unclosed') or {}; gt = M.get('generated_tokens') or {}
            out.append(f"| {m} | {system} | {t} | {M['n']} | {ov['n_valid']}/{ov['n']} | {json.dumps(M['parse']['failures'], ensure_ascii=False)} | {f(tr.get('rate'), 2)} | "
                       f"{tu.get('n', '—') if system == 'cot' else '—'} | {f(gt.get('mean'), 1)} | {f(x['bs'])} | {f(wall, 1)} | {f(M['n'] / wall if wall else None, 3)} |")
out += ['', '#### 峰值显存（torch：run_e5 不记录；nvidia-smi：按进程 used_memory，每 5 s 采样，按模型的起止时间归属）', '', '| 模型 | torch max_memory_allocated | nvidia-smi 峰值 MiB | 时间窗口（UTC） |', '|---|---|---|---|']
for m in ORDER:
    if m in W:
        out.append(f"| {m} | 未记录 | {f(smi.get(m))} | {W[m][0]}–{W[m][1]} |")
out += ['', '#### 严格读数与条件读数（全部档，另列标题范围的严格读数；a 类另列起伏误差；n 很小，只是格式核对）', '',
        '| 模型 | 系统 | 任务 | mask_iou_grid 严格 | 条件（n_valid） | mask_iou_full 严格 | 起伏误差 严格 | 起伏误差 条件（n_valid） | 标题范围的 mask_iou_grid 严格（范围） |', '|---|---|---|---|---|---|---|---|---|']
for m in ORDER:
    for system in ('zeroshot', 'cot'):
        for t, e in (res['models'].get(m, {}).get(system) or {}).items():
            o, h = e['overall'], e['headline']; g = o['mask_iou_grid']; full = o.get('mask_iou_full') or {}; rel = o.get('relief_error')
            out.append(f"| {m} | {system} | {t} | {f(g['strict'])} | {f(g['conditional'])}（{g['n_valid']}） | {f(full.get('strict'))} | "
                       f"{f(rel['strict']) if rel else '—'} | {(f(rel['conditional']) + '（' + str(rel['n_valid']) + '）') if rel else '—'} | "
                       f"{f((h.get('mask_iou_grid') or {}).get('strict'))}（{e['headline_scope'] or '全部'}） |")
out += ['', '#### 平凡基线（同一批开发样本；不加载模型；"全部档"与"标题范围"两种范围）', '', '| 基线 | 任务 | n | mask_iou_grid 严格：全部档 / 标题范围 | mask_iou_full 严格：全部档 | 起伏误差 严格：全部档 / 标题范围 | 标题范围 | 解析失败率 |', '|---|---|---|---|---|---|---|---|']
for b in sorted(os.listdir(os.path.join(OUT_DIR, 'baseline'))) if os.path.isdir(os.path.join(OUT_DIR, 'baseline')) else []:
    for t in TASKS:
        x = leaf('baseline', b, t)
        if x is None:
            continue
        o, h = x['M']['overall'], x['M']['headline']; rel, hrel = o.get('relief_error'), h.get('relief_error')
        res['baselines'].setdefault(b, {})[t] = {'n': x['M']['n'], 'overall': o, 'headline': h, 'headline_scope': x['M'].get('headline_scope'), 'parse': x['M'].get('parse')}
        out.append(f"| {b} | {t} | {x['M']['n']} | {f(o['mask_iou_grid']['strict'])} / {f((h.get('mask_iou_grid') or {}).get('strict'))} | {f((o.get('mask_iou_full') or {}).get('strict'))} | "
                   f"{(f(rel['strict']) + ' / ' + f(hrel['strict'])) if rel else '—'} | {x['M'].get('headline_scope') or '全部'} | {f((x['M'].get('parse') or {}).get('failure_rate'), 2)} |")
out += ['', '#### 原始输出举例（每个模型 × 系统 × 任务的第一条解析失败；raw_text 的前 600 个字符，原样）', '']
for m in ORDER:
    for system in ('zeroshot', 'cot'):
        for t, e in (res['models'].get(m, {}).get(system) or {}).items():
            ff = e['first_failure']
            if ff is None:
                out.append(f"- **{m} · {system} · {t}**：没有解析失败的样本"); continue
            out += [f"- **{m} · {system} · {t}** · `{ff['id']}` · failure = `{ff['failure']}`", '', '  ```', '  ' + ff['raw_text_600'].replace('```', "'''").replace('\n', '\n  '), '  ```', '']
out += ['', '#### 墙钟与用量合计（compute.json；墙钟只含 generate）', '', '| 模型 | zeroshot 生成 token / 墙钟 s | cot 生成 token / 墙钟 s | 合计生成 token / 墙钟 s |', '|---|---|---|---|']
T = {'tok': 0, 'wall': 0.0}
for m in ORDER:
    s = {}
    for system in ('zeroshot', 'cot'):
        es = (res['models'].get(m, {}).get(system) or {}).values()
        s[system] = (sum(e['compute_generated_tokens'] or 0 for e in es), sum(e['wall_time_s'] or 0 for e in es))
    if any(v[0] for v in s.values()):
        tok, wall = s['zeroshot'][0] + s['cot'][0], s['zeroshot'][1] + s['cot'][1]; T['tok'] += tok; T['wall'] += wall
        out.append(f"| {m} | {s['zeroshot'][0]} / {s['zeroshot'][1]:.1f} | {s['cot'][0]} / {s['cot'][1]:.1f} | {tok} / {wall:.1f} |")
out.append(f"| 合计 | | | {T['tok']} / {T['wall']:.1f} |")
json.dump(res, open(OUT, 'w'), indent=1, ensure_ascii=False)
print('\n'.join(out))
