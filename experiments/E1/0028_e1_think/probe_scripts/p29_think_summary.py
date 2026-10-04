#!/usr/bin/env python
"""RUN-0028 §5：思考子集（预算强制收尾）的汇总。数字从 run_e1 的输出读；配对对照调用仓库现有的函数：
- 结束方式四类的条数、每类的严格读数与有效率（= 1 − 解析失败率）、思考与作答 token 的中位数：metrics.json 的 thinking_end；
- 全体的严格读数：three_columns；
- 与关思考的对照：select(task, "think") 的样本，开思考取本运行的记录，关思考取 RUN-0027 主评测（eval）同 id 的记录；字母读数四个任务
  另取 RUN-0027 的 free 同 id 的记录。逐样本取值用 src.e1.scoring.per_sample_metric（检测用 ovd_per_image_counts + f1_statistic），
  差值与 95% CI 用 src.stats.paired_bootstrap_ci（开思考 − 关思考），全体与按结束方式各算一次。
在固定于运行 commit 的工作树里运行（REPO = 本文件往上 4 层）。
usage: python p29_think_summary.py <think_out_dir> <run0027_dir> <driver.log> <out.json> > think_summary.md
"""
import argparse, gzip, json, os, re, sys
from datetime import datetime
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
import run_e1 as R  # noqa: E402
from src.e1.scoring import ovd_per_image_counts, per_sample_metric  # noqa: E402
from src.e1.tasks import select  # noqa: E402
from src.stats import f1_statistic, paired_bootstrap_ci  # noqa: E402

TH, R27, DRIVER, OUT = sys.argv[1:5]
ORDER = ['qwen3.5-0.8b', 'qwen3.5-2b', 'qwen3.5-4b', 'qwen3.5-9b', 'internvl3.5-2b-hf', 'internvl3.5-4b-hf', 'gemma-4-e2b-it', 'gemma-4-e4b-it']
TASKS = ['refexp_box', 'refexp_poly', 'grid16', 'count', 'ovd', 'rel_depth_nyu', 'rel_depth_blink', 'spatial_cvbench', 'spatial_embspatial']
LETTER = {'rel_depth_nyu', 'rel_depth_blink', 'spatial_cvbench', 'spatial_embspatial'}
KINDS = ['natural', 'forced', 'not_opened', 'unclosed']
EST = {'qwen3.5-0.8b': 0.67, 'qwen3.5-2b': 0.63, 'qwen3.5-4b': 1.40, 'qwen3.5-9b': 1.08, 'internvl3.5-2b-hf': 2.12, 'internvl3.5-4b-hf': 3.36, 'gemma-4-e2b-it': 0.32, 'gemma-4-e4b-it': 0.61}
f = lambda v, d=4: '—' if v is None or (isinstance(v, float) and np.isnan(v)) else (f'{v:.{d}f}' if isinstance(v, float) else str(v))
paths = R.data_paths(argparse.Namespace(tiny=False, data_root=os.environ['VISINT_DATA_ROOT'], nyu_cache=None, dev_subsets=None, eval_subsets=None, nyu_pairs=None))


def load_jsonl(p):
    if os.path.exists(p):
        return [json.loads(l) for l in open(p)]
    if os.path.exists(p + '.gz'):
        return [json.loads(l) for l in gzip.open(p + '.gz', 'rt')]
    return None


def arr(task, samples, recs):
    if task == 'ovd':
        return np.array(ovd_per_image_counts(samples, recs), dtype=np.float64), f1_statistic
    name, a, stat = per_sample_metric(task, recs)
    return a, stat


def paired(task, samples, think, off):
    if not samples:
        return None
    a, stat = arr(task, samples, think); b, _ = arr(task, samples, off)
    c = paired_bootstrap_ci(a, b, statistic=stat)
    sf = stat or (lambda x: float(np.mean(x)))
    return {'n': len(samples), 'think': float(sf(a)), 'off': float(sf(b)), 'diff': None if c is None else c.estimate,
            'ci': None if c is None else [c.low, c.high]}


models = [m for m in ORDER if os.path.isdir(os.path.join(TH, m))]
res = {}
o1 = ['#### 结束方式四类的条数、每类的严格读数与有效率（metrics.json 的 thinking_end；检测的每类读数是这一类全部图合在一起的 F1@0.5）', '',
      '| 模型 | 任务 | n | 自然收尾 | 强制收尾 | 没开思考段 | 思考段里就停 | 每类：严格读数 / 有效率 | 全体严格读数 [CI]（three_columns） | 思考 token 中位数（全体） | 作答 token 中位数（全体） |', '|---|---|---|---|---|---|---|---|---|---|---|']
o2 = ['#### 与关思考的对照（同一批 id；开思考 − 关思考；配对 bootstrap 95% CI；关思考 = RUN-0027 eval）', '',
      '| 模型 | 任务 | 范围 | n | 开思考 | 关思考（eval） | 差 [95% CI] | 关思考（free，字母读数任务） | 差 vs free [95% CI] |', '|---|---|---|---|---|---|---|---|---|']
for m in models:
    res[m] = {}
    for t in TASKS:
        d = os.path.join(TH, m, f'{t}__think')
        M = json.load(open(os.path.join(d, 'metrics.json'))) if os.path.exists(os.path.join(d, 'metrics.json')) else None
        if M is None:
            o1.append(f"| {m} | {t} | （没有输出） | | | | | | | | |"); continue
        te = M.get('thinking_end') or {}; cnt = te.get('counts', {})
        per = []
        for k in KINDS:
            b = (te.get('by_kind') or {}).get(k)
            if not b:
                continue
            sv = b['F1@0.5_pooled'].get('value') if isinstance(b.get('F1@0.5_pooled'), dict) else ((b.get('strict') or [{}])[0].get('value'))
            per.append(f"{k} {f(sv)} / {f(1 - b['parse_failure_rate'], 3)}")
        recs = load_jsonl(os.path.join(d, 'predictions.jsonl'))
        tt = float(np.median([r['n_thinking_tokens'] for r in recs])); at = float(np.median([r['n_answer_tokens'] for r in recs]))
        tc = M['three_columns'][0]
        o1.append(f"| {m} | {t} | {M['n']} | {cnt.get('natural', 0)} | {cnt.get('forced', 0)} | {cnt.get('not_opened', 0)} | {cnt.get('unclosed', 0)} | {'；'.join(per)} | "
                  f"{tc['name']}（{tc['scope']}） {f(tc['strict']['value'])} [{f((tc['strict'].get('ci') or {}).get('low'))}, {f((tc['strict'].get('ci') or {}).get('high'))}] | {tt:.0f} | {at:.0f} |")
        res[m][t] = {'counts': cnt, 'three_columns': M['three_columns'], 'median_thinking_tokens': tt, 'median_answer_tokens': at}
        sel = select(t, 'think', paths); samples = sel.samples
        if [s.id for s in samples] != [r['id'] for r in recs]:
            o2.append(f"| {m} | {t} | （样本顺序与预测不一致，未比较） | | | | | | |"); continue
        off = load_jsonl(os.path.join(R27, m, t, 'predictions.jsonl')); offd = {r['id']: r for r in off} if off else {}
        fr = load_jsonl(os.path.join(R27, m, f'{t}__free', 'predictions.jsonl')) if t in LETTER else None; frd = {r['id']: r for r in fr} if fr else {}
        if any(s.id not in offd for s in samples):
            o2.append(f"| {m} | {t} | （RUN-0027 eval 缺同 id 的记录） | | | | | | |"); continue
        scopes = [('全体', list(range(len(samples))))] + [(k, [i for i, r in enumerate(recs) if r['thinking_end'] == k]) for k in KINDS]
        res[m][t]['paired'] = {}
        for name, idx in scopes:
            if not idx:
                continue
            ss = [samples[i] for i in idx]; th = [recs[i] for i in idx]; ev = [offd[samples[i].id] for i in idx]
            p_e = paired(t, ss, th, ev)
            p_f = paired(t, ss, th, [frd[samples[i].id] for i in idx]) if frd and all(samples[i].id in frd for i in idx) else None
            res[m][t]['paired'][name] = {'vs_eval': p_e, 'vs_free': p_f}
            cs = lambda p: '—' if p is None or p['ci'] is None else f"{f(p['diff'])} [{f(p['ci'][0])}, {f(p['ci'][1])}]"
            o2.append(f"| {m} | {t} | {name} | {p_e['n']} | {f(p_e['think'])} | {f(p_e['off'])} | {cs(p_e)} | {f(p_f['off']) if p_f else '—'} | {cs(p_f) if p_f else '—'} |")
steps = {}
cur = {}
for line in open(DRIVER):
    mm = re.match(r'=== (\S+?)__think start (\S+)', line)
    if mm:
        cur[mm.group(1)] = mm.group(2); continue
    mm = re.match(r'=== (\S+?)__think rc=(\d+) (\S+)', line)
    if mm and mm.group(1) in cur:
        steps[mm.group(1)] = ((datetime.fromisoformat(mm.group(3)) - datetime.fromisoformat(cur[mm.group(1)])).total_seconds(), int(mm.group(2)))
o3 = ['#### 用量（驱动日志里每个模型 run_e1 命令的起止，含加载与预热；GPU 小时）与估算（E1 预注册 §9 的"思考 4096"列 × 5,120/4,096）', '',
      '| 模型 | 实际 | 估算 | rc |', '|---|---|---|---|']
ta = te_ = 0.0
for m in models:
    if m in steps:
        h = steps[m][0] / 3600; e = EST[m] * 5120 / 4096; ta += h; te_ += e
        o3.append(f"| {m} | {h:.2f} | {e:.2f} | {steps[m][1]} |")
o3.append(f"| **合计（已完成的模型）** | **{ta:.2f}** | **{te_:.2f}** | |")
res['_usage'] = steps
json.dump(res, open(OUT, 'w'), indent=1, ensure_ascii=False, default=str)
print('\n'.join(o1 + [''] + o2 + [''] + o3))
