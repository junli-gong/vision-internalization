#!/usr/bin/env python
"""RUN-0020 §5：从 run_e1 --acceptance（v3）的输出生成 report 第 3 节的表。数字只从文件读，不重算判据：
- <out>/<模型>/acceptance/acceptance.json：总判定、(i) 逐位重跑（tasks[].rerun）、(ii) 组（groups[]）、v2 三种一致口径（tasks[].reference）；
- <out>/<模型>/acceptance/<A1|A2|B>/<模型>/<任务>__dev/：metrics.json（三列 three_columns、valid、refusal_by_person）、compute.json；
- nvidia-smi 峰值：按 driver.log 里每个模型的起止时间，把 *_apps.csv 的样本归到模型。torch 的峰值显存 run_e1 的验收不记录。
usage (在 $REPO 下): python p25_accept_v3_summary.py <out_dir> <driver.log> <apps.csv> <out.json> > accept_v3_summary.md
"""
import csv, json, os, re, sys
OUT_DIR, DRIVER, APPS, OUT = sys.argv[1:5]
ORDER = ['qwen3.5-0.8b', 'gemma-4-e2b-it', 'internvl3.5-2b-hf', 'qwen3.5-9b', 'qwen3.5-2b', 'qwen3.5-4b', 'gemma-4-e4b-it', 'internvl3.5-4b-hf']
TASKS = ['refexp_box', 'refexp_poly', 'grid16', 'count', 'ovd', 'rel_depth_nyu', 'rel_depth_blink', 'spatial_cvbench', 'spatial_embspatial']
f = lambda v, d=4: '—' if v is None else (('是' if v else '否') if isinstance(v, bool) else (f'{v:.{d}f}' if isinstance(v, float) else str(v)))
ci = lambda c: '—' if not c else f"[{f(c.get('low'))}, {f(c.get('high'))}]"
res = {}
for m in ORDER:
    p = os.path.join(OUT_DIR, m, 'acceptance', 'acceptance.json')
    if not os.path.exists(p):
        continue
    A = json.load(open(p)); e = {'acceptance': A, 'passes': {}}
    for ps in ('A1', 'A2', 'B'):
        for t in TASKS:
            leaf = os.path.join(OUT_DIR, m, 'acceptance', ps, m, f'{t}__dev')
            if os.path.exists(os.path.join(leaf, 'metrics.json')):
                e['passes'].setdefault(ps, {})[t] = {'metrics': json.load(open(os.path.join(leaf, 'metrics.json'))),
                                                     'compute': json.load(open(os.path.join(leaf, 'compute.json')))}
    res[m] = e
out = ['#### 总判定（acceptance.json）', '', '| 模型 | all_pass | rerun_all_identical | rerun_not_identical | groups_failed | groups_not_evaluated | groups_below_n_min | 批大小 A / B | frozen |', '|---|---|---|---|---|---|---|---|---|']
for m, e in res.items():
    A = e['acceptance']; h = A['header']
    out.append(f"| {m} | {f(A['all_pass'])} | {f(A['rerun_all_identical'])} | {A['rerun_not_identical'] or '[]'} | {A['groups_failed'] or '[]'} | {A['groups_not_evaluated'] or '[]'} | "
               f"{A.get('groups_below_n_min') or '[]'} | {h['batch_size_a']} / {h['batch_size_b']} | {f(h['frozen'])} |")
out += ['', '#### (i) 逐位重跑：A1 与 A2 的 predictions.jsonl（tasks[].rerun）', '', '| 模型 | 任务 | n | 文件逐字节相同 | 逐条不同的条数 | raw_text 不同的条数 | A1 的 sha256（前 16 位） |', '|---|---|---|---|---|---|---|']
for m, e in res.items():
    for t in e['acceptance']['tasks']:
        r = t['rerun']
        out.append(f"| {m} | {t['task']} | {t['n']} | {f(r['file_identical'])} | {r['n_diff']} | {r['raw_text_diff']} | {r['predictions_sha256']['A1'][:16]} |")
out += ['', '#### (ii) 噪声底：A1 与 B 按组比较（groups[]；严格读数，配对 bootstrap 95% CI）', '', '| 模型 | 组 | 任务 | 指标 | n（n_min） | A1 | B | 差（A1−B） | 95% CI | δ | CI 在 ±δ 内 | 通过 |', '|---|---|---|---|---|---|---|---|---|---|---|---|']
for m, e in res.items():
    for g in e['acceptance']['groups']:
        out.append(f"| {m} | {g['group']} | {', '.join(g['tasks'])} | {g['metric']} | {g['n']}（{g['n_min']}） | {f(g['metric_a1'])} | {f(g['metric_b'])} | {f(g['diff'])} | {ci(g['diff_ci'])} | "
                   f"{g['delta']} | {f(g['within_delta'])} | {f(g['pass'])} |")
out += ['', '#### 三列（A1 的 metrics.json 的 three_columns：严格读数 / 有效率 / 条件读数）', '', '| 模型 | 任务 | 指标（范围） | 严格 | 严格 95% CI | 有效 n_valid/n | 条件 | 无效：拒答 / 其他失败 |', '|---|---|---|---|---|---|---|---|']
for m, e in res.items():
    for t in TASKS:
        x = e['passes'].get('A1', {}).get(t)
        if not x:
            continue
        M = x['metrics']; v = M.get('valid') or {}
        for c in M.get('three_columns') or []:
            out.append(f"| {m} | {t} | {c['name']}（{c['scope']}） | {f(c['strict']['value'])} | {ci(c['strict'].get('ci'))} | {c['n_valid']}/{c['n']} | {f((c.get('conditional') or {}).get('value'))} | "
                       f"{v.get('n_refusal', '—')} / {v.get('n_other_failure', '—')} |")
out += ['', '#### 拒答诊断：定位、多边形、网格按 person 与非 person 分列（A1 的 refusal_by_person）', '', '| 模型 | 任务 | 指标 | 类 | n | 拒答 | 拒答率 | 有效 | 严格 | 条件 |', '|---|---|---|---|---|---|---|---|---|---|']
for m, e in res.items():
    for t in ('refexp_box', 'refexp_poly', 'grid16'):
        x = e['passes'].get('A1', {}).get(t)
        rb = x and x['metrics'].get('refusal_by_person')
        if not rb:
            continue
        for cls in ('person', 'non_person'):
            s = rb[cls]
            out.append(f"| {m} | {t} | {rb['metric']} | {cls} | {s['n']} | {s['n_refusal']} | {f(s['refusal_rate'], 3)} | {s['n_valid']} | {f(s['strict']['value'])} | {f((s.get('conditional') or {}).get('value'))} |")
out += ['', '#### v2 的三种一致口径（A1 与 B 之间，只作参考；tasks[].reference 原样）', '', '| 模型 | 任务 | 逐位 | IoU≥0.9 | 判定结果 | raw_text 相同 | 指标 | A1 | B | 差 | 95% CI |', '|---|---|---|---|---|---|---|---|---|---|---|']
for m, e in res.items():
    for t in e['acceptance']['tasks']:
        r = t['reference']; a = r.get('agreement_rate') or {}
        out.append(f"| {m} | {t['task']} | {f(a.get('bitwise'))} | {f(a.get('iou90'))} | {f(a.get('outcome'))} | {f(r.get('raw_text_equal_rate'))} | {r.get('metric') or '—'} | "
                   f"{f(r.get('metric_a1'))} | {f(r.get('metric_b'))} | {f(r.get('diff'))} | {ci(r.get('diff_ci'))} |")


def windows():
    w, cur = {}, {}
    for line in open(DRIVER):
        mm = re.match(r'=== (\S+) start (\d{4}-\d\d-\d\dT(\d\d:\d\d:\d\d))', line)
        if mm:
            cur[mm.group(1)] = mm.group(3); continue
        mm = re.match(r'=== (\S+) accept rc=\d+ (\d{4}-\d\d-\d\dT(\d\d:\d\d:\d\d))', line)
        if mm and mm.group(1) in cur:
            w[mm.group(1)] = (cur[mm.group(1)], mm.group(3))
    return w


rows = [(r[0], int(r[3].split()[0])) for r in csv.reader(open(APPS), skipinitialspace=True) if len(r) == 4] if os.path.exists(APPS) else []
W = windows(); smi = {k: max([mem for t, mem in rows if a <= t <= b] or [None], key=lambda x: -1 if x is None else x) for k, (a, b) in W.items()}
out += ['', '#### 计算量（compute.json，9 个任务合计；墙钟只含 generate）与显存峰值（torch：验收不记录；nvidia-smi：按进程，每 5 s）', '',
        '| 模型 | A1 生成 token / 墙钟 s | A2 生成 token / 墙钟 s | B 生成 token / 墙钟 s | 三遍合计墙钟 s | torch 峰值 | nvidia-smi 峰值 MiB |', '|---|---|---|---|---|---|---|']
for m, e in res.items():
    cells, tw = [], 0.0
    for ps in ('A1', 'A2', 'B'):
        xs = e['passes'].get(ps, {}).values(); tok = sum(x['compute']['generated_tokens'] for x in xs); wall = sum(x['compute']['wall_time_s'] for x in xs); tw += wall
        cells.append(f"{tok} / {wall:.1f}")
    out.append(f"| {m} | {' | '.join(cells)} | {tw:.1f} | 未记录 | {f(smi.get(m))} |")
json.dump({m: {'acceptance': e['acceptance'], 'compute': {ps: {t: x['compute'] for t, x in d.items()} for ps, d in e['passes'].items()},
               'a1_three_columns': {t: x['metrics'].get('three_columns') for t, x in e['passes'].get('A1', {}).items()},
               'a1_refusal_by_person': {t: x['metrics'].get('refusal_by_person') for t, x in e['passes'].get('A1', {}).items()}} for m, e in res.items()} |
          {'nvidia_smi_peak_mib': smi, 'windows': W}, open(OUT, 'w'), indent=1, ensure_ascii=False)
print('\n'.join(out))
