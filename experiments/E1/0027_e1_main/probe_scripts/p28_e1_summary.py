#!/usr/bin/env python
"""RUN-0027 §5：从 run_e1 的输出（<out>/<模型>/<任务>[__free|__extra__<口径>]/ 的 metrics.json、compute.json、predictions.jsonl）
生成 report 第 3 节的表。主表、解析失败、拒答、坐标诊断都原样取自 metrics.json；两处"同样本对照"调用仓库现有的评分代码：
- free 对字母读数：src.e1.tasks.select(task, "free") 的样本，取主评测（eval）predictions 里同 id 的记录，按 free 的顺序排好，
  交给 src.e1.scoring.aggregate（has_gt=True）算三列；另报两边逐条对错的配对计数。
- extra 对默认口径：select(task, "extra") 的样本，取 eval predictions 里同 id 的记录，同样交给 aggregate。
在固定于运行 commit 的工作树里运行（REPO = 本文件往上 4 层），保证评分代码与运行时相同。
usage (在仓库根下): python p28_e1_summary.py <out_dir> <driver.log> <out.json> > e1_summary.md
"""
import argparse, json, os, re, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
import run_e1 as R  # noqa: E402
from src.e1.scoring import aggregate  # noqa: E402
from src.e1.tasks import select  # noqa: E402

OUT_DIR, DRIVER, OUT = sys.argv[1:4]
ORDER = ['qwen3.5-0.8b', 'qwen3.5-2b', 'qwen3.5-4b', 'qwen3.5-9b', 'internvl3.5-2b-hf', 'internvl3.5-4b-hf', 'gemma-4-e2b-it', 'gemma-4-e4b-it']
TASKS = ['refexp_box', 'refexp_poly', 'grid16', 'count', 'ovd', 'rel_depth_nyu', 'rel_depth_blink', 'spatial_cvbench', 'spatial_embspatial']
LETTER = ['rel_depth_nyu', 'rel_depth_blink', 'spatial_cvbench', 'spatial_embspatial']
EST = {'qwen3.5-0.8b': (1.21, 0.10, 0.06), 'qwen3.5-2b': (1.00, 0.10, 0.00), 'qwen3.5-4b': (3.13, 0.14, 0.00), 'qwen3.5-9b': (2.34, 0.12, 0.09),
       'gemma-4-e2b-it': (1.93, 0.18, 0.00), 'gemma-4-e4b-it': (3.06, 0.21, 0.00), 'internvl3.5-2b-hf': (1.69, 0.07, 0.11), 'internvl3.5-4b-hf': (2.68, 0.08, 0.24)}
f = lambda v, d=4: '—' if v is None else (f'{v:.{d}f}' if isinstance(v, float) else str(v))
ci = lambda c: '—' if not c else f"[{f(c.get('low'))}, {f(c.get('high'))}]"
paths = R.data_paths(argparse.Namespace(tiny=False, data_root=os.environ['VISINT_DATA_ROOT'], nyu_cache=None, dev_subsets=None, eval_subsets=None, nyu_pairs=None))
J = lambda p: json.load(open(p)) if os.path.exists(p) else None
L = lambda p: [json.loads(l) for l in open(p)] if os.path.exists(p) else None


def leaf(m, name):
    d = os.path.join(OUT_DIR, m, name)
    return {'M': J(os.path.join(d, 'metrics.json')), 'C': J(os.path.join(d, 'compute.json')), 'P': os.path.join(d, 'predictions.jsonl')}


models = [m for m in ORDER if os.path.isdir(os.path.join(OUT_DIR, m))]
res = {m: {} for m in models}
out = ['#### 主表：三列（严格读数 / 有效率 / 条件读数）与 95% CI（metrics.json 的 three_columns）', '',
       '| 模型 | 任务 | 指标（范围） | n | 严格 | 严格 95% CI | 有效 n_valid/n（率） | 条件 | 条件 95% CI |', '|---|---|---|---|---|---|---|---|---|']
for m in models:
    for t in TASKS:
        x = leaf(m, t)
        if not x['M']:
            out.append(f"| {m} | {t} | （没有输出） | | | | | | |"); continue
        res[m][t] = {'three_columns': x['M']['three_columns']}
        for c in x['M']['three_columns']:
            cd = c.get('conditional') or {}
            out.append(f"| {m} | {t} | {c['name']}（{c['scope']}） | {c['n']} | {f(c['strict']['value'])} | {ci(c['strict'].get('ci'))} | {c['n_valid']}/{c['n']}（{f(c['valid_rate'], 3)}） | {f(cd.get('value'))} | {ci(cd.get('ci'))} |")
out += ['', '#### 定位、多边形、网格按划分分列（metrics.json 的 breakdown.split：严格读数、95% CI、n、解析失败率）', '',
        '| 模型 | 任务 | 划分 | 指标 | n | 严格 | 95% CI | 解析失败率 |', '|---|---|---|---|---|---|---|---|']
for m in models:
    for t in ('refexp_box', 'refexp_poly', 'grid16'):
        x = leaf(m, t)
        if not x['M']:
            continue
        for sp, v in sorted((x['M'].get('breakdown') or {}).get('split', {}).items()):
            for k, mv in v.items():
                if isinstance(mv, dict) and 'value' in mv:
                    out.append(f"| {m} | {t} | {sp} | {k} | {mv.get('n')} | {f(mv['value'])} | {ci(mv.get('ci'))} | {f(v.get('parse_failure_rate'), 4)} |")
out += ['', '#### 解析失败（ParseTally，按原因）、截断率、生成 token 与墙钟（eval；compute.json 的墙钟只含 generate）', '',
        '| 模型 | 任务 | n | 解析失败 n（率） | 按原因 | 截断率 | 平均生成 token | 生成 token 合计 | generate 墙钟 s |', '|---|---|---|---|---|---|---|---|---|']
for m in models:
    for t in TASKS:
        x = leaf(m, t)
        if not x['M']:
            continue
        p, gt, C = x['M']['parse'], x['M']['diagnostics']['generated_tokens'] or {}, x['C'] or {}
        out.append(f"| {m} | {t} | {x['M']['n']} | {p['n_failed']}（{f(p['failure_rate'], 4)}） | {json.dumps(p['failures'], ensure_ascii=False)} | {f(x['M']['truncation']['rate'], 4)} | "
                   f"{f(gt.get('mean'), 1)} | {C.get('generated_tokens')} | {f(C.get('wall_time_s'), 1)} |")
out += ['', '#### 拒答：定位、多边形、网格按 person 与非 person 分列（refusal_by_person）', '', '| 模型 | 任务 | 指标 | 类 | n | 拒答 | 拒答率 | 有效 | 严格 | 条件 |', '|---|---|---|---|---|---|---|---|---|---|']
for m in models:
    for t in ('refexp_box', 'refexp_poly', 'grid16'):
        x = leaf(m, t); rb = x['M'] and x['M'].get('refusal_by_person')
        if not rb:
            continue
        for cls in ('person', 'non_person'):
            s = rb[cls]
            out.append(f"| {m} | {t} | {rb['metric']} | {cls} | {s['n']} | {s['n_refusal']} | {f(s['refusal_rate'], 4)} | {s['n_valid']} | {f(s['strict']['value'])} | {f((s.get('conditional') or {}).get('value'))} |")
out += ['', '#### free：自由作答 vs 同样本上的字母读数（eval）', '',
        '| 模型 | 任务 | n | free 严格 [CI] | free 有效率 | 字母读数 严格 [CI]（同样本） | 差（free − 字母） | 配对：都对 / 只字母对 / 只 free 对 / 都错 | free 截断率 | free 平均生成 token |', '|---|---|---|---|---|---|---|---|---|---|']
for m in models:
    for t in LETTER:
        fr = leaf(m, f'{t}__free'); ev = leaf(m, t)
        if not fr['M'] or not ev['M']:
            continue
        sel = select(t, 'free', paths)
        evr = {r['id']: r for r in L(ev['P'])}; recs = [evr[s.id] for s in sel.samples]
        agg = aggregate(t, sel.samples, recs, has_gt=True)
        a, b = fr['M']['three_columns'][0], agg['three_columns'][0]
        frr = {r['id']: r for r in L(fr['P'])}
        c = lambda r: bool(r.get('score') and r['score'].get('correct'))
        pairs = [(c(evr[s.id]), c(frr[s.id])) for s in sel.samples]
        both, lo, fo, nei = sum(x and y for x, y in pairs), sum(x and not y for x, y in pairs), sum(y and not x for x, y in pairs), sum(not x and not y for x, y in pairs)
        res[m][f'{t}__free_vs_letter'] = {'free': a, 'letter_same_ids': b, 'pairs': [both, lo, fo, nei]}
        out.append(f"| {m} | {t} | {a['n']} | {f(a['strict']['value'])} {ci(a['strict'].get('ci'))} | {f(a['valid_rate'], 3)} | {f(b['strict']['value'])} {ci(b['strict'].get('ci'))} | "
                   f"{f(a['strict']['value'] - b['strict']['value'])} | {both} / {lo} / {fo} / {nei} | {f(fr['M']['truncation']['rate'], 3)} | {f((fr['M']['diagnostics']['generated_tokens'] or {}).get('mean'), 1)} |")
out += ['', '#### extra：附加口径 vs 同样本上的主口径（eval）', '',
        '| 模型 | 口径 | 任务 | 指标（范围） | n | extra 严格 [CI] | extra 有效率 | 主口径 严格 [CI]（同样本） | 主口径有效率 | 差（extra − 主口径） | extra 视觉 token 均值 / 主口径 |', '|---|---|---|---|---|---|---|---|---|---|---|']
for m in models:
    for d in sorted(os.listdir(os.path.join(OUT_DIR, m))):
        if '__extra__' not in d:
            continue
        t, pol = d.split('__extra__')
        ex, ev = leaf(m, d), leaf(m, t)
        if not ex['M'] or not ev['M']:
            continue
        sel = select(t, 'extra', paths)
        evr = {r['id']: r for r in L(ev['P'])}; recs = [evr[s.id] for s in sel.samples]
        agg = aggregate(t, sel.samples, recs, has_gt=True)
        vt_x = (ex['M']['diagnostics'].get('visual_tokens') or {}).get('mean'); vt_e = (agg['diagnostics'].get('visual_tokens') or {}).get('mean')
        res[m][d] = {'extra': ex['M']['three_columns'], 'default_same_ids': agg['three_columns']}
        for a, b in zip(ex['M']['three_columns'], agg['three_columns']):
            out.append(f"| {m} | {pol} | {t} | {a['name']}（{a['scope']}） | {a['n']} | {f(a['strict']['value'])} {ci(a['strict'].get('ci'))} | {f(a['valid_rate'], 3)} | "
                       f"{f(b['strict']['value'])} {ci(b['strict'].get('ci'))} | {f(b['valid_rate'], 3)} | {f(a['strict']['value'] - b['strict']['value'])} | {f(vt_x, 1)} / {f(vt_e, 1)} |")
out += ['', '#### 坐标诊断（D22-2）：全部坐标 ≤ 1 且含小数的条数（diagnostics.coord_all_le1_with_decimal）', '', '| 模型 | 任务 | 子集 | 条数 | 比例 |', '|---|---|---|---|---|']
for m in models:
    for d in sorted(os.listdir(os.path.join(OUT_DIR, m))):
        x = leaf(m, d)
        cd = x['M'] and x['M']['diagnostics'].get('coord_all_le1_with_decimal')
        if cd:
            out.append(f"| {m} | {d.split('__')[0]} | {('__'.join(d.split('__')[1:]) or 'eval')} | {cd['n']} | {f(cd['rate'], 5)} |")
steps = {}
cur = {}
for line in open(DRIVER):
    mm = re.match(r'=== (\S+?)__(eval|free|extra) start (\S+)', line)
    if mm:
        cur[(mm.group(1), mm.group(2))] = mm.group(3); continue
    mm = re.match(r'=== (\S+?)__(eval|free|extra) rc=(\d+) (\S+)', line)
    if mm and (mm.group(1), mm.group(2)) in cur:
        from datetime import datetime
        a = datetime.fromisoformat(cur[(mm.group(1), mm.group(2))]); b = datetime.fromisoformat(mm.group(4))
        steps[(mm.group(1), mm.group(2))] = ((b - a).total_seconds(), int(mm.group(3)))
out += ['', '#### 用量：每个模型每部分的墙钟（驱动日志里每条 run_e1 命令的起止，含加载与预热；GPU 小时）与预注册 §9 的估算', '',
        '| 模型 | eval 实际 / 估算 | free 实际 / 估算 | extra 实际 / 估算 | 合计实际 / 估算 | rc（eval / free / extra） |', '|---|---|---|---|---|---|']
T = [0.0, 0.0]
for m in models:
    act = [steps.get((m, p), (None, None)) for p in ('eval', 'free', 'extra')]
    hrs = [None if a[0] is None else a[0] / 3600 for a in act]
    e = EST[m]; tot = sum(h for h in hrs if h)
    if hrs[0] is not None:  # 只把 eval 已跑完的模型计进合计
        T[0] += tot; T[1] += sum(e)
    out.append(f"| {m} | {f(hrs[0], 2)} / {e[0]:.2f} | {f(hrs[1], 2)} / {e[1]:.2f} | {f(hrs[2], 2)} / {e[2]:.2f} | {tot:.2f} / {sum(e):.2f} | {' / '.join(str(a[1]) if a[1] is not None else '—' for a in act)} |")
out.append(f"| **合计（已完成的模型）** | | | | **{T[0]:.2f} / {T[1]:.2f}** | |")
res['_usage'] = {f'{k[0]}__{k[1]}': v for k, v in steps.items()}
json.dump(res, open(OUT, 'w'), indent=1, ensure_ascii=False, default=str)
print('\n'.join(out))
