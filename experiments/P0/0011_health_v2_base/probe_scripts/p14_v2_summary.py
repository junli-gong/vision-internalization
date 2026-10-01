#!/usr/bin/env python
"""RUN-0011 §4：从 health_<键>.json（v2）与 v1 的 JSON 生成 report 第 3 节的表（所有数字只从 JSON 读）。
逐题核对：按 id 配对，比较两边共有的字段（wall_time_s 除外；v2 新增的 stopped 字段 v1 没有，单列为"仅一侧有的字段"）。
- GSM8K / IFEval：v2 对 RUN-0009 重评分后的 v1（experiments/P0/0009_ifeval_rescore/health_<键>.json）；
- mmlu_pro_fmt32：v2 截出的 32 token 记录对 RUN-0007 的 mmlu_pro（32 token）记录。
usage (在 $REPO 下): python p14_v2_summary.py <v2_dir> <v1_rescored_dir> <run0007_dir> <out.json> > summary.md
"""
import json, os, sys
V2, V1, R7, OUT = sys.argv[1:5]
ORDER = ['qwen3.5-0.8b', 'qwen3.5-2b', 'gemma-4-e2b-it', 'internvl3.5-2b-hf', 'qwen3.5-4b', 'gemma-4-e4b-it', 'internvl3.5-4b-hf', 'qwen3.5-9b']
load = lambda d, m: json.load(open(os.path.join(d, f'health_{m}.json'))) if os.path.exists(os.path.join(d, f'health_{m}.json')) else None


def item_check(new, old):
    o = {r['id']: r for r in old}
    common_fields = set.intersection(*(set(r) for r in new + old)) - {'wall_time_s'}
    only = sorted((set().union(*(set(r) for r in new)) ^ set().union(*(set(r) for r in old))) - {'wall_time_s'})
    n_text = n_all = 0; diff_fields = {}; diff_ids = []
    for r in new:
        q = o.get(r['id'])
        if q is None:
            diff_ids.append(r['id']); continue
        n_text += r['raw_text'] == q['raw_text']
        d = sorted(k for k in common_fields if r.get(k) != q.get(k))
        n_all += not d
        if d:
            diff_ids.append(r['id'])
            for k in d:
                diff_fields[k] = diff_fields.get(k, 0) + 1
    return {'n': len(new), 'n_old': len(old), 'n_raw_text_identical': n_text, 'n_all_common_fields_identical': n_all,
            'differing_field_counts': diff_fields, 'differing_ids': diff_ids, 'fields_only_one_side': only, 'compared_fields': sorted(common_fields)}


res = {}
for m in ORDER:
    R, B, S = load(V2, m), load(V1, m), load(R7, m)
    if R is None:
        continue
    e = {'ok': R.get('ok'), 'protocol_version': R.get('protocol_version'), 'wall_time_s': R.get('wall_time_s')}
    if R.get('ok'):
        e['scores'] = R['scores']; e['compute'] = {k: {'generated_tokens': v['generated_tokens'], 'wall_time_s': v['wall_time_s']} for k, v in R['compute'].items()}
        e['truncation'] = {k: {'n': v['n'], 'n_reached_max_new_tokens': v['n_reached_max_new_tokens']} for k, v in R['diagnostics'].items()}
        e['prefix_audit'] = {k: {kk: v[kk] for kk in ('max_new_tokens', 'n', 'n_identical', 'mismatches')} | {'audit_generated_tokens': v['compute']['generated_tokens'], 'audit_wall_time_s': v['compute']['wall_time_s']}
                             for k, v in R.get('prefix_audit', {}).items()}
        e['parse_tally'] = R['parse_tally']
        e['item_check'] = {'gsm8k_vs_run0009': item_check(R['records']['gsm8k'], B['records']['gsm8k']),
                           'ifeval_vs_run0009': item_check(R['records']['ifeval'], B['records']['ifeval']),
                           'mmlu_pro_fmt32_vs_run0007_mmlu_pro': item_check(R['records']['mmlu_pro_fmt32'], S['records']['mmlu_pro'])}
    else:
        e['error'] = {k: R[k] for k in R if 'error' in k or 'traceback' in k}
    res[m] = e
json.dump(res, open(OUT, 'w'), indent=1, ensure_ascii=False)

pct = lambda s: f"{s['pct']:.1f}（{s['n_correct']}/{s['n']}）"
out = ['### 分数表（协议 v2）', '', '| 模型 | ok | MMLU-Pro 256（标题） | MMLU-Pro fmt32（诊断） | GSM8K flexible（标题） | GSM8K strict | IFEval prompt strict（标题） | IFEval inst strict | IFEval prompt loose | IFEval inst loose |', '|---|---|---|---|---|---|---|---|---|---|']
for m, e in res.items():
    if not e['ok']:
        out.append(f"| {m} | False | — | — | — | — | — | — | — | — |"); continue
    s = e['scores']; i = s['ifeval']
    out.append(f"| {m} | True | {pct(s['mmlu_pro']['accuracy'])} | {pct(s['mmlu_pro_fmt32']['accuracy'])} | {pct(s['gsm8k']['flexible'])} | {pct(s['gsm8k']['strict'])} | "
               f"{pct(i['prompt_level_strict'])} | {pct(i['instruction_level_strict'])} | {pct(i['prompt_level_loose'])} | {pct(i['instruction_level_loose'])} |")
out += ['', '### 与 v1 的逐题核对（共有字段逐字段比较，wall_time_s 除外）', '',
        '| 模型 | GSM8K vs RUN-0009：raw_text 相同 / 共有字段全同 | IFEval vs RUN-0009：raw_text 相同 / 共有字段全同 | fmt32 vs RUN-0007 32 token：raw_text 相同 / 共有字段全同 | 不同的字段（计数） | 仅一侧有的字段 |', '|---|---|---|---|---|---|']
for m, e in res.items():
    if not e['ok']:
        continue
    c = e['item_check']; g, f, p = c['gsm8k_vs_run0009'], c['ifeval_vs_run0009'], c['mmlu_pro_fmt32_vs_run0007_mmlu_pro']
    df = {k: v['differing_field_counts'] for k, v in c.items() if v['differing_field_counts']}
    only = sorted(set().union(*(set(v['fields_only_one_side']) for v in c.values())))
    out.append(f"| {m} | {g['n_raw_text_identical']}/{g['n']} / {g['n_all_common_fields_identical']}/{g['n']} | {f['n_raw_text_identical']}/{f['n']} / {f['n_all_common_fields_identical']}/{f['n']} | "
               f"{p['n_raw_text_identical']}/{p['n']} / {p['n_all_common_fields_identical']}/{p['n']} | {json.dumps(df, ensure_ascii=False) if df else '无'} | {', '.join(only) or '无'} |")
out += ['', '### `--audit-fmt32 100`（截出的 fmt32 记录 vs 真的 max_new_tokens=32 生成，逐字段，wall_time_s 除外）', '',
        '| 模型 | 逐字段相同 | 不同的 id（字段） | 审计生成 token / 墙钟 s（不计入 compute） |', '|---|---|---|---|']
for m, e in res.items():
    if not e['ok']:
        continue
    for k, a in e['prefix_audit'].items():
        mm = '; '.join(f"{x['id']}({','.join(x['fields'])})" for x in a['mismatches']) or '无'
        out.append(f"| {m} | {a['n_identical']}/{a['n']} | {mm} | {a['audit_generated_tokens']} / {a['audit_wall_time_s']:.1f} |")
out += ['', '### 截断率（写满 max_new_tokens 的条数 / n）', '', '| 模型 | MMLU-Pro @256 | MMLU-Pro fmt32 @32 | GSM8K @512 | IFEval @1024 |', '|---|---|---|---|---|']
for m, e in res.items():
    if not e['ok']:
        continue
    t = e['truncation']; fm = lambda k: f"{t[k]['n_reached_max_new_tokens']}/{t[k]['n']}（{100 * t[k]['n_reached_max_new_tokens'] / t[k]['n']:.1f}%）"
    out.append(f"| {m} | {fm('mmlu_pro')} | {fm('mmlu_pro_fmt32')} | {fm('gsm8k')} | {fm('ifeval')} |")
out += ['', '### 计算量（compute 字段；fmt32 从同一次生成截出，不单独记账）', '', '| 模型 | 生成 token：MMLU-Pro / GSM8K / IFEval / 合计 | 墙钟 s：MMLU-Pro / GSM8K / IFEval / 合计 | 运行墙钟 s（wall_time_s，含加载与审计） |', '|---|---|---|---|']
for m, e in res.items():
    if not e['ok']:
        continue
    c = e['compute']; ks = [k for k in ('mmlu_pro', 'gsm8k', 'ifeval', 'total') if k in c]
    toks = ' / '.join(str(c[k]['generated_tokens']) for k in ks)
    walls = ' / '.join('%.1f' % c[k]['wall_time_s'] for k in ks)
    out.append(f"| {m} | {toks} | {walls} | {e['wall_time_s']} |")
out += ['', '### 解析失败', '', '| 模型 | MMLU-Pro 256 | MMLU-Pro fmt32 | GSM8K strict | GSM8K flexible |', '|---|---|---|---|---|']
for m, e in res.items():
    if not e['ok']:
        continue
    pt = e['parse_tally']; f = lambda k: f"{pt[k]['n_failed']}/{pt[k]['n']} {json.dumps(pt[k]['failures'], ensure_ascii=False)}" if k in pt else '—'
    out.append(f"| {m} | {f('mmlu_pro')} | {f('mmlu_pro_fmt32')} | {f('gsm8k_strict')} | {f('gsm8k_flexible')} |")
errs = {m: e for m, e in res.items() if not e['ok']}
if errs:
    out += ['', '### 出错的模型（原样）', ''] + [f"- {m}: `{json.dumps(e['error'], ensure_ascii=False)[:3000]}`" for m, e in errs.items()]
print('\n'.join(out))
