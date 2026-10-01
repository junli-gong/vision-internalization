#!/usr/bin/env python
"""RUN-0012 §3：从 run_e1 --acceptance 的输出生成 report 第 3 节的表（所有数字只从 acceptance.json、metrics.json、
compute.json、predictions.jsonl 读）。不改任何判据；"不一致"的细目只是对两遍 parsed 的逐条描述：
- refexp_box：两遍都解析成功但框不同的条数、两框的 IoU（原图像素）、四个坐标的最大绝对差（像素）；
- ovd：两遍都解析成功时，标签序列相同 / 框数相同 / 逐框 IoU 的最小值；
- 其余：一遍失败一遍成功的条数、两遍都成功但值不同的条数。
usage (在 $REPO 下): python p15_accept_summary.py <acceptance_root> <out.json> > accept_summary.md
"""
import json, os, statistics, sys
ROOT, OUT = sys.argv[1:3]
ORDER = ['qwen3.5-0.8b', 'qwen3.5-2b', 'gemma-4-e2b-it', 'internvl3.5-2b-hf', 'qwen3.5-4b', 'gemma-4-e4b-it', 'internvl3.5-4b-hf', 'qwen3.5-9b']
TASKS = ['refexp_box', 'refexp_poly', 'grid16', 'count', 'ovd', 'rel_depth_nyu', 'rel_depth_blink', 'spatial_cvbench', 'spatial_embspatial']


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0.0


def preds(m, bs, t):
    p = os.path.join(ROOT, m, 'acceptance', f'bs{bs}', m, f'{t}__dev', 'predictions.jsonl')
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else None


def stat(xs):
    return None if not xs else {'n': len(xs), 'min': round(min(xs), 4), 'median': round(statistics.median(xs), 4), 'max': round(max(xs), 4)}


res = {}
for m in ORDER:
    ap = os.path.join(ROOT, m, 'acceptance', 'acceptance.json')
    if not os.path.exists(ap):
        continue
    A = json.load(open(ap)); bsz = A['header']['batch_size']
    e = {'all_pass': A['all_pass'], 'header': A['header'], 'tasks': {}}
    for row in A['tasks']:
        t = row['task']; d = {'acceptance': row, 'passes': {}}
        for bs in (1, bsz):
            leaf = os.path.join(ROOT, m, 'acceptance', f'bs{bs}', m, f'{t}__dev')
            M = json.load(open(os.path.join(leaf, 'metrics.json'))); C = json.load(open(os.path.join(leaf, 'compute.json')))
            d['passes'][bs] = {'max_new_tokens': M['generation']['max_new_tokens'], 'parse': M['parse'], 'truncation': M['truncation'],
                               'thinking_unclosed': M['thinking_unclosed'], 'generated_tokens': C['generated_tokens'],
                               'mean_generated_tokens': C['per_sample_mean']['generated_tokens'], 'wall_time_s': C['wall_time_s'],
                               'headline': M.get('headline')}
        a, b = preds(m, 1, t), preds(m, bsz, t)
        one_fail = sum((x['failure'] is None) != (y['failure'] is None) for x, y in zip(a, b))
        both_ok = [(x, y) for x, y in zip(a, b) if x['failure'] is None and y['failure'] is None]
        dz = {'n_one_pass_failed': one_fail, 'n_both_failed': sum(x['failure'] is not None and y['failure'] is not None for x, y in zip(a, b)),
              'n_both_parsed': len(both_ok), 'n_both_parsed_parsed_differs': sum(x['parsed'] != y['parsed'] for x, y in both_ok),
              'n_failure_reason_differs': sum(x['failure'] != y['failure'] for x, y in zip(a, b)),
              'n_generated_tokens_equal': sum(x['n_generated_tokens'] == y['n_generated_tokens'] for x, y in zip(a, b))}
        if t == 'refexp_box':
            diff = [(x, y) for x, y in both_ok if x['parsed']['box'] != y['parsed']['box']]
            dz['box_iou_between_passes'] = stat([iou(x['parsed']['box'], y['parsed']['box']) for x, y in diff])
            dz['box_max_abs_coord_diff_px'] = stat([max(abs(p - q) for p, q in zip(x['parsed']['box'], y['parsed']['box'])) for x, y in diff])
            dz['n_box_iou_lt_0.9'] = sum(iou(x['parsed']['box'], y['parsed']['box']) < 0.9 for x, y in diff)
        if t == 'ovd':
            diff = [(x, y) for x, y in both_ok if x['parsed'] != y['parsed']]
            dz['n_labels_equal'] = sum(x['parsed']['labels'] == y['parsed']['labels'] for x, y in diff)
            dz['n_box_count_equal'] = sum(len(x['parsed']['boxes']) == len(y['parsed']['boxes']) for x, y in diff)
            same = [(x, y) for x, y in diff if x['parsed']['labels'] == y['parsed']['labels'] and len(x['parsed']['boxes']) == len(y['parsed']['boxes'])]
            dz['min_box_iou_when_labels_and_count_equal'] = stat([min(iou(p, q) for p, q in zip(x['parsed']['boxes'], y['parsed']['boxes'])) for x, y in same if x['parsed']['boxes']])
        d['disagreement_detail'] = dz
        e['tasks'][t] = d
    res[m] = e
json.dump(res, open(OUT, 'w'), indent=1, ensure_ascii=False)

f = lambda v, k=4: '—' if v is None else (('是' if v else '否') if isinstance(v, bool) else (f'{v:.{k}f}' if isinstance(v, float) else str(v)))
out = ['#### 验收汇总（模型 × 任务；判据：答案一致率 ≥ 0.98 且配对差 95% CI 含 0；检测无 GT，只判一致率）', '',
       '| 模型 | 任务 | n | 答案一致率 | 不一致条数 | 指标 | 逐条 | 批量 | 差（逐条−批量） | 配对差 95% CI | CI 含 0 | raw_text 逐字相同 | 通过 |', '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
for m, e in res.items():
    for t in TASKS:
        if t not in e['tasks']:
            continue
        r = e['tasks'][t]['acceptance']; ci = r['diff_ci']
        out.append(f"| {m} | {t} | {r['n']} | {f(r['answer_agreement'])} | {r['n_disagree']} | {r['metric'] or '—'} | {f(r['metric_single'])} | {f(r['metric_batched'])} | "
                   f"{f(r['diff'])} | {'[%s, %s]' % (f(ci['low']), f(ci['high'])) if ci else '—'} | {f(r['diff_ci_contains_0'])} | {f(r['raw_text_equal_rate'])} | {f(r['pass'])} |")
out += ['', '| 模型 | all_pass | 不通过的任务 |', '|---|---|---|']
for m, e in res.items():
    out.append(f"| {m} | {f(e['all_pass'])} | {', '.join(t for t in TASKS if t in e['tasks'] and not e['tasks'][t]['acceptance']['pass']) or '无'} |")
out += ['', '#### 截断率与解析失败分布（逐条 bs=1 / 批量；截断 = stopped=False）', '',
        '| 模型 | 任务 | max_new_tokens | 截断率 bs1 / 批量 | 解析失败 bs1 | 解析失败 批量 | 平均生成 token bs1 / 批量 | 思考未闭合 bs1 / 批量 |', '|---|---|---|---|---|---|---|---|']
for m, e in res.items():
    bsz = e['header']['batch_size']
    for t in TASKS:
        if t not in e['tasks']:
            continue
        p1, pb = e['tasks'][t]['passes'][1], e['tasks'][t]['passes'][bsz]
        fl = lambda p: f"{p['parse']['n_failed']}/{p['parse']['n']} {json.dumps(p['parse']['failures'], ensure_ascii=False)}"
        out.append(f"| {m} | {t} | {p1['max_new_tokens']} | {f(p1['truncation']['rate'], 2)} / {f(pb['truncation']['rate'], 2)} | {fl(p1)} | {fl(pb)} | "
                   f"{p1['mean_generated_tokens']:.1f} / {pb['mean_generated_tokens']:.1f} | {p1['thinking_unclosed']['n']} / {pb['thinking_unclosed']['n']} |")
out += ['', '#### 两遍不一致的细目（只描述两遍 parsed 的差别）', '',
        '| 模型 | 任务 | 一遍失败一遍成功 | 两遍都失败 | 两遍都成功但 parsed 不同 | 失败原因不同 | 生成 token 数相同 | 框/检测的细目 |', '|---|---|---|---|---|---|---|---|']
for m, e in res.items():
    for t in TASKS:
        if t not in e['tasks']:
            continue
        z = e['tasks'][t]['disagreement_detail']; n = e['tasks'][t]['acceptance']['n']
        extra = ''
        if t == 'refexp_box':
            extra = f"两框 IoU {json.dumps(z['box_iou_between_passes'])}；坐标最大绝对差 px {json.dumps(z['box_max_abs_coord_diff_px'])}；IoU<0.9 的 {z['n_box_iou_lt_0.9']} 条"
        if t == 'ovd':
            extra = f"标签序列相同 {z['n_labels_equal']}；框数相同 {z['n_box_count_equal']}；标签与框数都相同时逐框 IoU 的最小值 {json.dumps(z['min_box_iou_when_labels_and_count_equal'])}"
        out.append(f"| {m} | {t} | {z['n_one_pass_failed']} | {z['n_both_failed']} | {z['n_both_parsed_parsed_differs']}/{z['n_both_parsed']} | {z['n_failure_reason_differs']} | {z['n_generated_tokens_equal']}/{n} | {extra or '—'} |")
out += ['', '#### 每遍的墙钟与生成 token（compute.json；墙钟只含 generate）', '', '| 模型 | 任务 | 墙钟 s bs1 / 批量 | 生成 token bs1 / 批量 |', '|---|---|---|---|']
for m, e in res.items():
    bsz = e['header']['batch_size']; s1 = sb = 0.0
    for t in TASKS:
        if t not in e['tasks']:
            continue
        p1, pb = e['tasks'][t]['passes'][1], e['tasks'][t]['passes'][bsz]; s1 += p1['wall_time_s']; sb += pb['wall_time_s']
        out.append(f"| {m} | {t} | {p1['wall_time_s']:.1f} / {pb['wall_time_s']:.1f} | {p1['generated_tokens']} / {pb['generated_tokens']} |")
    out.append(f"| {m} | 合计 | {s1:.1f} / {sb:.1f} | |")
print('\n'.join(out))
