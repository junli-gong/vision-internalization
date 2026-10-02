#!/usr/bin/env python
"""RUN-0025 §5：汇总探针输出（probe_out/*.json）、两次完整开发子集运行（0025_full_S1、0025_full_S3），并与 RUN-0024 的
qwen3.5-2b refexp_box（A1、A2、B）逐条对照。数字只从文件读。
usage (在 $REPO 下): python p27_det_summary.py <run_dir> <out.json> > det_summary.md
"""
import glob, hashlib, json, os, sys
RUN, OUT = sys.argv[1:3]
R24 = 'experiments/E1/0024_acceptance_v3_ext/qwen3.5-2b/acceptance'
SET = ['S1', 'S2a', 'S2b', 'S2c', 'S3', 'S4']
P = {s: json.load(open(os.path.join(RUN, 'probe_out', f'{s}.json'))) for s in SET if os.path.exists(os.path.join(RUN, 'probe_out', f'{s}.json'))}
L = lambda p: [json.loads(l) for l in open(p)]
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
res = {}
out = ['#### 每种设置：同一进程里与第 1 遍不同的条数（第 4 批，64 条；每遍一列）', '',
       '| 设置 | pid | 环境 | deterministic | 遍数 | 每遍与第 1 遍不同的条数 | 不同的位置 | 每遍 generate 墙钟 s：中位 / 最小 / 最大 | 模型加载 s |', '|---|---|---|---|---|---|---|---|---|']
for s, j in P.items():
    w = sorted(p['wall_s'] for p in j['passes']); med = w[len(w) // 2]
    env = ', '.join(f"{k}={v}" for k, v in j['env'].items() if v and k in ('CUBLAS_WORKSPACE_CONFIG', 'FLA_CACHE_MODE')) or '（同 RUN-0024）'
    pos = sorted({(d['batch_pos'], d['id']) for p in j['passes'] for d in p['diffs']})
    out.append(f"| {s} | {j['pid']} | {env} | {j['deterministic_algorithms']} | {len(j['passes'])} | {[p['n_rows_diff_vs_pass1'] for p in j['passes']]} | {pos or '无'} | "
               f"{med:.3f} / {w[0]:.3f} / {w[-1]:.3f} | {j['load_s']} |")
out += ['', '#### 跨进程与跨设置：各进程第 1 遍的 64 条 token 序列（与 S1 第 1 遍比较；sha256 是 64 行 token id 的 JSON）', '',
        '| 设置 | 第 1 遍 sha256（前 16 位） | 与 S1 第 1 遍不同的条数 | 不同的 id |', '|---|---|---|---|']
base = P['S1']['rows_by_pass'][0]
cross = {}
for s, j in P.items():
    r = j['rows_by_pass'][0]; d = [x['id'] for x, y in zip(base, r) if x['token_ids'] != y['token_ids']]
    cross[s] = d
    out.append(f"| {s} | {j['passes'][0]['sha256'][:16]} | {len(d)} | {', '.join(d) or '无'} |")
# 与 RUN-0024 对照
lo, hi = P['S1']['batch_range']
a1, a2, b = L(f'{R24}/A1/qwen3.5-2b/refexp_box__dev/predictions.jsonl'), L(f'{R24}/A2/qwen3.5-2b/refexp_box__dev/predictions.jsonl'), L(f'{R24}/B/qwen3.5-2b/refexp_box__dev/predictions.jsonl')
out += ['', f'#### 与 RUN-0024 的对照：第 4 批（第 {lo}–{hi - 1} 条）的 raw_text', '', '| 比较 | 不同的条数（/64） | 不同的 id |', '|---|---|---|']
cmp24 = {}
for name, ref in (('RUN-0024 A1', a1), ('RUN-0024 A2', a2), ('RUN-0024 B', b)):
    d = [x['id'] for x, y in zip(base, ref[lo:hi]) if x['raw_text'] != y['raw_text']]
    cmp24[name] = d
    out.append(f"| S1 第 1 遍 vs {name} | {len(d)} | {', '.join(d) or '无'} |")
i7909 = [k for k, x in enumerate(base) if x['id'] == 'refcoco+_unc/val/7909']
if i7909:
    k = i7909[0]
    out += ['', '`refcoco+_unc/val/7909` 的 raw_text：', '', '| 来源 | raw_text |', '|---|---|']
    for name, txt in (('RUN-0024 A1', a1[lo + k]['raw_text']), ('RUN-0024 A2', a2[lo + k]['raw_text']), ('RUN-0024 B（同 id）', next(r['raw_text'] for r in b if r['id'] == 'refcoco+_unc/val/7909'))) + tuple((f'本次 {s} 第 1 遍', j['rows_by_pass'][0][k]['raw_text']) for s, j in P.items()):
        out.append(f"| {name} | `{json.dumps(txt, ensure_ascii=False)[1:-1]}` |")
out += ['', '#### 完整开发子集（run_e1.py --acceptance --task refexp_box，600 条；A1 与 A2，另有 B）', '',
        '| 设置 | A1 vs A2 文件逐字节相同 | A1 vs A2 不同的条数 | A1 sha256（前 16 位） | A1 = RUN-0024 A1 | A1 = RUN-0024 A2 | 与 RUN-0024 A1 不同的条数 | 墙钟 s：A1 / A2 / B（compute.json） | all_pass |', '|---|---|---|---|---|---|---|---|---|']
s24a1, s24a2 = sha(f'{R24}/A1/qwen3.5-2b/refexp_box__dev/predictions.jsonl'), sha(f'{R24}/A2/qwen3.5-2b/refexp_box__dev/predictions.jsonl')
full = {}
for s in ('S1', 'S3'):
    d = os.path.join(RUN, f'0025_full_{s}', 'qwen3.5-2b', 'acceptance')
    if not os.path.exists(os.path.join(d, 'acceptance.json')):
        continue
    A = json.load(open(os.path.join(d, 'acceptance.json'))); t = A['tasks'][0]['rerun']
    pa1 = os.path.join(d, 'A1', 'qwen3.5-2b', 'refexp_box__dev', 'predictions.jsonl'); f1 = L(pa1)
    n24 = sum(x['raw_text'] != y['raw_text'] for x, y in zip(f1, a1))
    walls = [json.load(open(os.path.join(d, ps, 'qwen3.5-2b', 'refexp_box__dev', 'compute.json')))['wall_time_s'] for ps in ('A1', 'A2', 'B')]
    full[s] = {'rerun': t, 'a1_sha256': sha(pa1), 'eq_r24_a1': sha(pa1) == s24a1, 'eq_r24_a2': sha(pa1) == s24a2, 'n_diff_vs_r24_a1': n24, 'walls': walls, 'all_pass': A['all_pass']}
    out.append(f"| {s} | {t['file_identical']} | {t['n_diff']} | {sha(pa1)[:16]} | {sha(pa1) == s24a1} | {sha(pa1) == s24a2} | {n24} | {' / '.join('%.1f' % w for w in walls)} | {A['all_pass']} |")
out += ['', '#### 第 1 个不同位置的前两名 token 与 logit 差', '']
firsts = {s: j['first_divergence'] for s, j in P.items() if j['first_divergence']}
out.append('本次所有设置里都没有出现与第 1 遍不同的条，没有可记录的分歧位置。' if not firsts else json.dumps(firsts, ensure_ascii=False, indent=1))
out += ['', '#### warn_only 列出的非确定性算子（torch.use_deterministic_algorithms(True, warn_only=True) 期间的 warnings，去重）', '']
for s in ('S3', 'S4'):
    if s in P:
        ws = [w for w in P[s]['warnings_unique'] if 'determin' in w[1].lower()]
        out.append(f"- {s}（{len(ws)} 条与确定性相关；全部 warnings 去重后 {len(P[s]['warnings_unique'])} 条）：" + ('；'.join(f"`{w[1][:300]}`（{w[2]}:{w[3]}）" for w in ws) or '无'))
fw = os.path.join(RUN, 'probe_out', 'full_S3_warnings.json')
if os.path.exists(fw):
    W = json.load(open(fw)); ws = [w for w in W['warnings_unique'] if 'determin' in w[1].lower()]
    out.append(f"- full_S3（{len(ws)} 条与确定性相关；全部去重后 {len(W['warnings_unique'])} 条；deterministic_algorithms={W['deterministic_algorithms']}，rc={W['rc']}）：" + ('；'.join(f"`{w[1][:300]}`（{w[2]}:{w[3]}）" for w in ws) or '无'))
json.dump({'probe': {s: {k: j[k] for k in ('setting', 'pid', 'env', 'deterministic_algorithms', 'versions', 'batch_range', 'batch_ids_sha256', 'contains_7909', 'passes', 'first_divergence', 'warnings_unique', 'load_s', 'warmup_wall_s')} for s, j in P.items()},
           'cross_process_vs_S1_pass1': cross, 'vs_run0024_batch4': cmp24, 'full': full}, open(OUT, 'w'), indent=1, ensure_ascii=False, default=str)
print('\n'.join(out))
