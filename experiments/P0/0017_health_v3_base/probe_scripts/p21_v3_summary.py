#!/usr/bin/env python
"""RUN-0017 §4：从 health_<键>.json（v3）、RUN-0011 的 v2 JSON、audit_letter_<键>.json 生成 report 第 3 节的表（数字只从文件读）。
逐题核对：按 id 配对，比两边共有的字段（wall_time_s 除外）：
- mmlu_pro_free256（v3）对 v2 的 mmlu_pro（256 token 自由作答）；mmlu_pro_fmt32 对 v2 的 mmlu_pro_fmt32；gsm8k；ifeval。
usage (在 $REPO 下): python p21_v3_summary.py <v3_dir> <v2_dir> <out.json> > health_summary_v3.md
"""
import json, os, sys
V3, V2, OUT = sys.argv[1:4]
ORDER = ['qwen3.5-0.8b', 'qwen3.5-2b', 'gemma-4-e2b-it', 'internvl3.5-2b-hf', 'qwen3.5-4b', 'gemma-4-e4b-it', 'internvl3.5-4b-hf', 'qwen3.5-9b']
load = lambda p: json.load(open(p)) if os.path.exists(p) else None


def item_check(new, old):
    o = {r['id']: r for r in old}
    common = set.intersection(*(set(r) for r in new + old)) - {'wall_time_s'}
    only = sorted((set().union(*(set(r) for r in new)) ^ set().union(*(set(r) for r in old))) - {'wall_time_s'})
    n_raw = n_all = 0; diff = {}; ids = []
    for r in new:
        q = o.get(r['id'])
        if q is None:
            ids.append(r['id']); continue
        n_raw += r.get('raw_text') == q.get('raw_text')
        d = sorted(k for k in common if r.get(k) != q.get(k))
        n_all += not d
        if d:
            ids.append(r['id'])
            for k in d:
                diff[k] = diff.get(k, 0) + 1
    return {'n': len(new), 'n_raw_text_identical': n_raw, 'n_all_common_fields_identical': n_all, 'differing_field_counts': diff,
            'differing_ids': ids, 'fields_only_one_side': only}


res = {}
for m in ORDER:
    R = load(f'{V3}/health_{m}.json')
    if R is None:
        continue
    e = {'ok': R.get('ok'), 'protocol_version': R.get('protocol_version'), 'wall_time_s': R.get('wall_time_s')}
    if R.get('ok'):
        B = load(f'{V2}/health_{m}.json'); A = load(f'{V3}/audit_letter_{m}.json')
        e.update(scores=R['scores'], option_spellings=R.get('option_spellings'),
                 compute={k: {'generated_tokens': v['generated_tokens'], 'wall_time_s': v['wall_time_s'], 'n_calls': v.get('n_calls')} for k, v in R['compute'].items()},
                 truncation={k: {'n': v['n'], 'n_reached_max_new_tokens': v.get('n_reached_max_new_tokens')} for k, v in R['diagnostics'].items()},
                 item_check={'mmlu_pro_free256_vs_v2_mmlu_pro': item_check(R['records']['mmlu_pro_free256'], B['records']['mmlu_pro']),
                             'mmlu_pro_fmt32_vs_v2_mmlu_pro_fmt32': item_check(R['records']['mmlu_pro_fmt32'], B['records']['mmlu_pro_fmt32']),
                             'gsm8k_vs_v2': item_check(R['records']['gsm8k'], B['records']['gsm8k']),
                             'ifeval_vs_v2': item_check(R['records']['ifeval'], B['records']['ifeval'])},
                 audit=None if A is None else {k: A[k] for k in ('n', 'n_same_token', 'n_same_letter', 'n_same_as_health', 'load_s', 'audit_wall_s')} |
                 {'mismatch_ids': [r['id'] for r in A['rows'] if not (r['same_token'] and r['same_letter'] and r['same_as_health'])]})
    else:
        e['error'] = {k: R[k] for k in R if 'error' in k or 'traceback' in k}
    res[m] = e
json.dump(res, open(OUT, 'w'), indent=1, ensure_ascii=False)

pct = lambda s: '—' if s is None else f"{s['pct']:.1f}（{s['n_correct']}/{s['n']}）"
out = ['#### 分数表（协议 v3）', '', '| 模型 | ok | MMLU-Pro letter（标题） | 正确选项平均概率 % | MMLU-Pro free256（诊断） | MMLU-Pro fmt32（诊断） | GSM8K flexible（标题） | GSM8K strict | IFEval prompt strict（标题） | IFEval inst strict | IFEval prompt loose | IFEval inst loose |', '|---|---|---|---|---|---|---|---|---|---|---|---|']
for m, e in res.items():
    if not e['ok']:
        out.append(f"| {m} | False |" + ' — |' * 10); continue
    s = e['scores']; i = s['ifeval']
    out.append(f"| {m} | True | {pct(s['mmlu_pro_letter']['accuracy'])} | {s['mmlu_pro_letter']['mean_p_gold']['pct']:.1f} | {pct(s['mmlu_pro_free256']['accuracy'])} | "
               f"{pct(s['mmlu_pro_fmt32']['accuracy'])} | {pct(s['gsm8k']['flexible'])} | {pct(s['gsm8k']['strict'])} | {pct(i['prompt_level_strict'])} | "
               f"{pct(i['instruction_level_strict'])} | {pct(i['prompt_level_loose'])} | {pct(i['instruction_level_loose'])} |")
out += ['', '#### 与 v2（RUN-0011）的逐题核对（共有字段逐字段比较，wall_time_s 除外；格式：raw_text 相同 / 共有字段全同）', '',
        '| 模型 | free256 vs v2 mmlu_pro | fmt32 vs v2 fmt32 | GSM8K | IFEval | 不同的字段（计数） | 仅一侧有的字段 |', '|---|---|---|---|---|---|---|']
for m, e in res.items():
    if not e['ok']:
        continue
    c = e['item_check']; fm = lambda x: f"{x['n_raw_text_identical']}/{x['n']} / {x['n_all_common_fields_identical']}/{x['n']}"
    df = {k: v['differing_field_counts'] for k, v in c.items() if v['differing_field_counts']}
    only = sorted(set().union(*(set(v['fields_only_one_side']) for v in c.values())))
    out.append(f"| {m} | {fm(c['mmlu_pro_free256_vs_v2_mmlu_pro'])} | {fm(c['mmlu_pro_fmt32_vs_v2_mmlu_pro_fmt32'])} | {fm(c['gsm8k_vs_v2'])} | {fm(c['ifeval_vs_v2'])} | "
               f"{json.dumps(df, ensure_ascii=False) if df else '无'} | {', '.join(only) or '无'} |")
out += ['', '#### 字母读数核对（audit_letter，MMLU-Pro 前 100 题；不计入记账）', '',
        '| 模型 | 生成的 token = option_logits 选中的 token | 字母相同 | option_logits 的字母 = health JSON 的 mmlu_pro_letter | 不一致的 id | 探针加载 s / 核对 s |', '|---|---|---|---|---|---|']
for m, e in res.items():
    a = e.get('audit')
    if not a:
        continue
    out.append(f"| {m} | {a['n_same_token']}/{a['n']} | {a['n_same_letter']}/{a['n']} | {a['n_same_as_health']}/{a['n']} | {', '.join(map(str, a['mismatch_ids'])) or '无'} | {a['load_s']} / {a['audit_wall_s']} |")
out += ['', '#### 各家族实际用到的字母写法与 token id（health JSON 的 option_spellings；"文本"=token id）', '', '| 模型 | A–J 的写法 | 跳过的写法 |', '|---|---|---|']
for m, e in res.items():
    sp = e.get('option_spellings')
    if not sp:
        continue
    cells = '; '.join(f"{L}: " + ', '.join(f"{json.dumps(x['text'])}={x['token_id']}" for x in v['spellings']) for L, v in sp.items())
    sk = {L: v['skipped'] for L, v in sp.items() if v['skipped']}
    out.append(f"| {m} | {cells} | {json.dumps(sk, ensure_ascii=False) if sk else '无'} |")
out += ['', '#### 截断率（写满 max_new_tokens 的条数 / n）', '', '| 模型 | free256 @256 | fmt32 @32 | GSM8K @512 | IFEval @1024 |', '|---|---|---|---|---|']
for m, e in res.items():
    if not e['ok']:
        continue
    t = e['truncation']; fm = lambda k: f"{t[k]['n_reached_max_new_tokens']}/{t[k]['n']}" if k in t and t[k]['n_reached_max_new_tokens'] is not None else '—'
    out.append(f"| {m} | {fm('mmlu_pro_free256')} | {fm('mmlu_pro_fmt32')} | {fm('gsm8k')} | {fm('ifeval')} |")
out += ['', '#### 计算量（compute 字段；mmlu_pro 含 letter 的 100 次调用与 free256 的 100 次生成；fmt32 从 free256 截出，不另记账）', '',
        '| 模型 | 生成 token：MMLU-Pro / GSM8K / IFEval / 合计 | 调用次数：MMLU-Pro / 合计 | 墙钟 s：MMLU-Pro / GSM8K / IFEval / 合计 | 运行墙钟 s（wall_time_s） |', '|---|---|---|---|---|']
for m, e in res.items():
    if not e['ok']:
        continue
    c = e['compute']; ks = [k for k in ('mmlu_pro', 'gsm8k', 'ifeval', 'total') if k in c]
    out.append(f"| {m} | {' / '.join(str(c[k]['generated_tokens']) for k in ks)} | {c['mmlu_pro']['n_calls']} / {c['total']['n_calls']} | "
               f"{' / '.join('%.1f' % c[k]['wall_time_s'] for k in ks)} | {e['wall_time_s']} |")
errs = {m: e for m, e in res.items() if not e['ok']}
if errs:
    out += ['', '#### 出错的模型（原样）', ''] + [f"- {m}: `{json.dumps(e['error'], ensure_ascii=False)[:3000]}`" for m, e in errs.items()]
print('\n'.join(out))
