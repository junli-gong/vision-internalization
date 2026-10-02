#!/usr/bin/env python
"""RUN-0021 §5：从 check_training_interfaces.py 的 <模型>.json 生成 report 第 3 节的表（数字只从 JSON 读，判据不重算）。
显存峰值：脚本的 JSON 里没有这个字段；nvidia-smi 峰值按 driver.log 里每个模型的起止时间，把 *_apps.csv 的样本归到模型。
usage (在 $REPO 下): python p26_iface_summary.py <run_dir> <driver.log> <apps.csv> <out.json> > iface_summary.md
"""
import csv, json, os, re, sys
RUN, DRIVER, APPS, OUT = sys.argv[1:5]
ORDER = ['qwen3.5-0.8b', 'gemma-4-e2b-it', 'internvl3.5-2b-hf', 'qwen3.5-9b']
MODES = ['thinking_off', 'thinking_on']
f = lambda v, d=4: '—' if v is None else (('是' if v else '否') if isinstance(v, bool) else (f'{v:.{d}f}' if isinstance(v, float) else str(v)))
gs = lambda g: '—' if not g or not g.get('n') else f"n {g['n']}；中位 {f(g['median'])}；最大 {f(g['max'])}；<{g['threshold']} 的比例 {f(g['frac_lt_threshold'], 3)}"
fd = lambda d: '—' if not d or not d.get('n') else f"n {d['n']}；p25 {f(d['p25'], 1)} / 中位 {f(d['median'], 1)} / p75 {f(d['p75'], 1)} / 最大 {d['max']}"
J = {m: json.load(open(os.path.join(RUN, f'{m}.json'))) for m in ORDER if os.path.exists(os.path.join(RUN, f'{m}.json'))}
out = ['#### 总判定', '', '| 模型 | pass | restricted pass | merge pass | failures（原样） | error |', '|---|---|---|---|---|---|']
for m, j in J.items():
    C = j.get('checks') or {}
    fl = '<br>'.join(x for k in ('restricted', 'merge') for x in (j.get('failures') or {}).get(k, [])) or '无'
    out.append(f"| {m} | {f(j.get('pass'))} | {f((C.get('restricted') or {}).get('pass'))} | {f((C.get('merge') or {}).get('pass'))} | {fl} | {j.get('error') or '—'} |")
out += ['', '#### restricted（在 base 上；受限解码 vs generate_batch(allowed)；教师强制）', '',
        '| 模型 | 思考 | 行数 | exact_rows | token_agreement | 分歧行数 | 第一个分歧位置 | 生成分歧处 gap | 生成分歧处 top2_gap | 教师强制 agreement | 教师强制不一致处 gap | 教师强制 top2_gap |',
        '|---|---|---|---|---|---|---|---|---|---|---|---|']
for m, j in J.items():
    R = ((j.get('checks') or {}).get('restricted') or {}).get('by_thinking') or {}
    for md in MODES:
        x = R.get(md)
        if not x:
            continue
        d, t = x['decode_vs_generate_batch'], x['teacher_forcing']
        out.append(f"| {m} | {md} | {x['n_rows']} | {f(d['exact_rows'])} | {f(d['token_agreement'])} | {d['n_divergent_rows']} | {fd(d['first_divergence'])} | {gs(d['gap'])} | {gs(d['top2_gap'])} | "
                   f"{f(t['agreement'])} | {gs(t['gap'])} | {gs(t['top2_gap'])} |")
out += ['', '#### merge（随机 LoRA；合并前 vs 合并后；生成全词表 greedy）', '',
        '| 模型 | 思考 | effect_vs_base：生成 exact_rows / token_agreement | effect_vs_base：教师强制 argmax_agreement / max_abs_logit / max_abs_hidden | 合并前后：exact_rows | token_agreement | 第一个分歧位置 | 生成分歧处 gap | 教师强制 argmax_agreement | max_abs_logit | max_abs_hidden | 教师强制不一致处 gap |',
        '|---|---|---|---|---|---|---|---|---|---|---|---|']
for m, j in J.items():
    M = (j.get('checks') or {}).get('merge') or {}
    for md in MODES:
        x = (M.get('by_thinking') or {}).get(md); e = (M.get('effect_vs_base') or {}).get(md)
        if not x:
            continue
        g, t = x['generate'], x['teacher_forcing']
        es = '—' if not e else f"{f(e['generate']['exact_rows'])} / {f(e['generate']['token_agreement'])}"
        et = '—' if not e else f"{f(e['teacher_forcing']['argmax_agreement'])} / {f(e['teacher_forcing']['max_abs_logit'], 3)} / {f(e['teacher_forcing']['max_abs_hidden'], 3)}"
        out.append(f"| {m} | {md} | {es} | {et} | {f(g['exact_rows'])} | {f(g['token_agreement'])} | {fd(g['first_divergence'])} | {gs(g['gap'])} | {f(t['argmax_agreement'])} | "
                   f"{f(t.get('max_abs_logit'), 3)} | {f(t.get('max_abs_hidden'), 3)} | {gs(t['gap'])} |")
out += ['', '#### 随机 LoRA 的设置（merge.lora）', '', '| 模型 | rel_scale | seed | r / alpha | 模块数 | 参数量 | 实际 ‖ΔW‖/‖W‖：最小 / 中位 / 最大 | targets_sha256（前 16 位） |', '|---|---|---|---|---|---|---|---|']
for m, j in J.items():
    lo = ((j.get('checks') or {}).get('merge') or {}).get('lora')
    if lo:
        r = lo['realized_rel_frobenius']
        out.append(f"| {m} | {lo['rel_scale']} | {lo['seed']} | {lo['r']} / {lo['lora_alpha']} | {lo['n_modules']} | {lo['n_parameters']:,} | {f(r['min'], 6)} / {f(r['median'], 6)} / {f(r['max'], 6)} | {lo['targets_sha256'][:16]} |")
out += ['', '#### allowed 集合的 token 表（allowed）', '', '| 模型 | token（角色：文本 → id） | 停止 token | 16×16 网格编码的 token 数 / 用到的 id |', '|---|---|---|---|']
for m, j in J.items():
    a = j.get('allowed') or {}
    toks = '; '.join(f"{t['role']}: {json.dumps(t['text'])} → {t['id']}（{json.dumps(t['token'], ensure_ascii=False)}）" for t in a.get('tokens', []))
    out.append(f"| {m} | {toks} | {a.get('stop_ids')} | {(a.get('probe') or {}).get('n_tokens')} / {(a.get('probe') or {}).get('distinct_ids')} |")


def windows():
    w, cur = {}, {}
    for line in open(DRIVER):
        mm = re.match(r'=== (\S+) start (\d{4}-\d\d-\d\dT(\d\d:\d\d:\d\d))', line)
        if mm:
            cur[mm.group(1)] = mm.group(3); continue
        mm = re.match(r'=== (\S+) check rc=\d+ (\d{4}-\d\d-\d\dT(\d\d:\d\d:\d\d))', line)
        if mm and mm.group(1) in cur:
            w[mm.group(1)] = (cur[mm.group(1)], mm.group(3))
    return w


rows = [(r[0], int(r[3].split()[0])) for r in csv.reader(open(APPS), skipinitialspace=True) if len(r) == 4] if os.path.exists(APPS) else []
W = windows(); smi = {k: max([mem for t, mem in rows if a <= t <= b] or [None], key=lambda x: -1 if x is None else x) for k, (a, b) in W.items()}
out += ['', '#### 批配置、墙钟与显存峰值', '', '| 模型 | 批大小（frozen） | 批的划分 | 数据 n / 总体 | timing_s：加载 / restricted / merge 前 / merge 后 / 合计 | nvidia-smi 峰值 MiB | torch 峰值 |', '|---|---|---|---|---|---|---|']
for m, j in J.items():
    bp, d, tm = j.get('batch_protocol') or {}, j.get('data') or {}, j.get('timing_s') or {}
    out.append(f"| {m} | {bp.get('batch_size')}（{f(bp.get('frozen'))}） | {bp.get('batches')} | {d.get('n')} / {d.get('n_population')} | "
               f"{f(tm.get('load_and_inputs_s'), 1)} / {f(tm.get('restricted_s'), 1)} / {f(tm.get('merge_before_s'), 1)} / {f(tm.get('merge_after_s'), 1)} / {f(tm.get('total_s'), 1)} | {f(smi.get(m))} | 脚本不记录 |")
json.dump({'per_model': {m: {k: j.get(k) for k in ('pass', 'failures', 'criteria', 'allowed', 'batch_protocol', 'timing_s', 'code', 'model', 'error')} for m, j in J.items()},
           'nvidia_smi_peak_mib': smi, 'windows': W}, open(OUT, 'w'), indent=1, ensure_ascii=False)
print('\n'.join(out))
