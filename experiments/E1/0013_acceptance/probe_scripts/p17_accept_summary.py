#!/usr/bin/env python
"""RUN-0013 §3：从 run_e1 --acceptance 的输出生成 report 第 3 节的表（所有数字只从 acceptance__<口径>.json、metrics.json、
compute.json 读；判据与诊断都原样取自这些文件，不重算）。
usage (在 $REPO 下): python p17_accept_summary.py <acceptance_root> <countbenchqa_root> <out.json> > accept_summary.md
"""
import json, os, sys
ROOT, ROOTC, OUT = sys.argv[1:4]
ORDER = ['qwen3.5-0.8b', 'qwen3.5-2b', 'gemma-4-e2b-it', 'internvl3.5-2b-hf', 'qwen3.5-4b', 'gemma-4-e4b-it', 'internvl3.5-4b-hf', 'qwen3.5-9b']
MODES = ['bitwise', 'iou90', 'outcome']
MN = {'bitwise': '(a) 逐位', 'iou90': '(b) IoU≥0.9', 'outcome': '(c) 判定结果'}
f = lambda v, k=4: '—' if v is None else (('是' if v else '否') if isinstance(v, bool) else (f'{v:.{k}f}' if isinstance(v, float) else str(v)))


def load(root, m):
    base = os.path.join(root, m, 'acceptance')
    p = os.path.join(base, 'acceptance__outcome.json')
    if not os.path.exists(p):
        return None
    A = json.load(open(p)); bsz = A['header']['batch_size']; e = {'header': A['header'], 'all_pass_by_agreement': A['all_pass_by_agreement'], 'tasks': {}}
    for row in A['tasks']:
        t = row['task']; d = {'row': row, 'passes': {}}
        for bs in (1, bsz):
            leaf = os.path.join(base, f'bs{bs}', m, f'{t}__dev')
            M = json.load(open(os.path.join(leaf, 'metrics.json'))); C = json.load(open(os.path.join(leaf, 'compute.json')))
            dg = M['diagnostics']
            d['passes'][bs] = {'max_new_tokens': M['generation']['max_new_tokens'], 'parse': M['parse'], 'truncation': M['truncation'],
                               'thinking_unclosed': M['thinking_unclosed'], 'failure_detail_counts': dg['failure_detail']['counts'],
                               'think_marker_when_off': {k: dg['think_marker_when_off'][k] for k in ('n', 'rate', 'n_thinking_unclosed')},
                               'generated_tokens': C['generated_tokens'], 'mean_generated_tokens': C['per_sample_mean']['generated_tokens'],
                               'wall_time_s': C['wall_time_s']}
        e['tasks'][t] = d
    return e


res = {'heuristic': {m: x for m in ORDER if (x := load(ROOT, m))}, 'countbenchqa': {m: x for m in ORDER if (x := load(ROOTC, m))}}
json.dump(res, open(OUT, 'w'), indent=1, ensure_ascii=False)


def acc_rows(R, title):
    o = [title, '', '| 模型 | 任务 | n | ' + ' | '.join(f'{MN[k]} 一致率 / 通过' for k in MODES) + ' | 指标 | 逐条 | 批量 | 差（逐条−批量） | 配对差 95% CI | CI 含 0 | raw_text 逐字相同 |',
         '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for m, e in R.items():
        for t, d in e['tasks'].items():
            r = d['row']; ci = r['diff_ci']; bm = r['agreement_by_mode']
            o.append(f"| {m} | {t} | {r['n']} | " + ' | '.join(f"{f(bm[k]['rate'])} / {f(bm[k]['pass'])}" for k in MODES) +
                     f" | {r['metric'] or '—'} | {f(r['metric_single'])} | {f(r['metric_batched'])} | {f(r['diff'])} | "
                     f"{'[%s, %s]' % (f(ci['low']), f(ci['high'])) if ci else '—'} | {f(r['diff_ci_contains_0'])} | {f(r['raw_text_equal_rate'])} |")
    return o


out = acc_rows(res['heuristic'], '#### 验收汇总（三种一致口径；判据：一致率 ≥ 0.98 且配对差 95% CI 含 0；计数用默认的启发式问句）')
out += ['', '#### 每种口径的总判定', '', '| 模型 | ' + ' | '.join(f'{MN[k]} all_pass（不判的任务）' for k in MODES) + ' | ' + ' | '.join(f'{MN[k]} 不通过的任务' for k in MODES) + ' |', '|---|---|---|---|---|---|---|']
for m, e in res['heuristic'].items():
    ap = e['all_pass_by_agreement']
    fails = {k: [t for t, d in e['tasks'].items() if d['row']['agreement_by_mode'][k]['pass'] is False] for k in MODES}
    out.append(f"| {m} | " + ' | '.join(f"{f(ap[k]['all_pass'])}（{', '.join(ap[k]['not_evaluated']) or '—'}）" for k in MODES) + ' | ' +
               ' | '.join(', '.join(fails[k]) or '无' for k in MODES) + ' |')
out += [''] + acc_rows(res['countbenchqa'], '#### 计数：CountBenchQA 人工问句（--task count --count-questions countbenchqa）')
out += ['', '| 模型 | ' + ' | '.join(f'{MN[k]} all_pass' for k in MODES) + ' |', '|---|---|---|---|']
for m, e in res['countbenchqa'].items():
    out.append(f"| {m} | " + ' | '.join(f(e['all_pass_by_agreement'][k]['all_pass']) for k in MODES) + ' |')
for qname, R in (('启发式问句', res['heuristic']), ('CountBenchQA 问句（仅计数）', res['countbenchqa'])):
    out += ['', f'#### 截断率与解析失败（逐条 bs=1 / 批量；截断 = stopped=False；{qname}）', '',
            '| 模型 | 任务 | max_new_tokens | 截断率 bs1 / 批量 | 解析失败 bs1 | 解析失败 批量 | 平均生成 token bs1 / 批量 |', '|---|---|---|---|---|---|---|']
    for m, e in R.items():
        bsz = e['header']['batch_size']
        for t, d in e['tasks'].items():
            p1, pb = d['passes'][1], d['passes'][bsz]
            fl = lambda p: f"{p['parse']['n_failed']}/{p['parse']['n']} {json.dumps(p['parse']['failures'], ensure_ascii=False)}"
            out.append(f"| {m} | {t} | {p1['max_new_tokens']} | {f(p1['truncation']['rate'], 2)} / {f(pb['truncation']['rate'], 2)} | {fl(p1)} | {fl(pb)} | "
                       f"{p1['mean_generated_tokens']:.1f} / {pb['mean_generated_tokens']:.1f} |")
    out += ['', f'#### 诊断（failure_detail：wrong_shape、refusal；关思考时自发的思考标记；思考未闭合；{qname}）', '',
            '| 模型 | 任务 | failure_detail bs1 | failure_detail 批量 | 自发思考标记 bs1 / 批量 | 思考未闭合 bs1 / 批量 |', '|---|---|---|---|---|---|']
    for m, e in R.items():
        bsz = e['header']['batch_size']
        for t, d in e['tasks'].items():
            p1, pb = d['passes'][1], d['passes'][bsz]
            fd = lambda p: json.dumps(p['failure_detail_counts'], ensure_ascii=False) if p['failure_detail_counts'] else '{}'
            out.append(f"| {m} | {t} | {fd(p1)} | {fd(pb)} | {p1['think_marker_when_off']['n']} / {pb['think_marker_when_off']['n']} | "
                       f"{p1['thinking_unclosed']['n']} / {pb['thinking_unclosed']['n']} |")
out += ['', '#### 每遍的墙钟与生成 token（compute.json；墙钟只含 generate；启发式问句）', '', '| 模型 | 任务 | 墙钟 s bs1 / 批量 | 生成 token bs1 / 批量 |', '|---|---|---|---|']
for m, e in res['heuristic'].items():
    bsz = e['header']['batch_size']; s1 = sb = 0.0
    for t, d in e['tasks'].items():
        p1, pb = d['passes'][1], d['passes'][bsz]; s1 += p1['wall_time_s']; sb += pb['wall_time_s']
        out.append(f"| {m} | {t} | {p1['wall_time_s']:.1f} / {pb['wall_time_s']:.1f} | {p1['generated_tokens']} / {pb['generated_tokens']} |")
    out.append(f"| {m} | 合计 | {s1:.1f} / {sb:.1f} | |")
print('\n'.join(out))
