#!/usr/bin/env python
"""RUN-0019 §5：从 LISA 外部预测目录（lisa_predictions/）、评分输出（external/lisa-7b-v1/<任务>__dev/metrics.json）与平凡基线
（baseline/<基线>/<任务>__dev/metrics.json）生成 report 第 3 节的表。数字只从文件读，不重算指标。
usage (在 $REPO 下): python p24_lisa_summary.py <run_dir> <out.json> > lisa_summary.md
"""
import collections, glob, json, os, statistics, sys
RUN, OUT = sys.argv[1:3]
TASKS = ['rotation3d', 'viewpoint', 'occlusion', 'physics']
f = lambda v, d=3: '—' if v is None else (f'{v:.{d}f}' if isinstance(v, float) else str(v))
pred = os.path.join(RUN, 'lisa_predictions')
runlog = json.load(open(sorted(glob.glob(os.path.join(pred, 'logs', 'run_*.json')))[-1]))
marker = json.load(open(os.path.join(pred, 'EXTERNAL_PREDICTIONS.json')))
ri = runlog['runner_info']
res = {'load': {'fixes_applied': marker['fixes_applied'], 'load_s': ri['load_s'], 'peak_mem_after_load_bytes': ri['peak_mem_after_load_bytes'],
                'peak_mem_inference_bytes': runlog.get('runner_stats', {}).get('peak_mem_inference_bytes'), 'tokenizer': ri['tokenizer'],
                'embed_tokens_rows': ri['embed_tokens_rows'], 'seg_token_idx': ri['seg_token_idx'], 'generation_config': ri['generation_config'],
                'tokenizers_available': ri['tokenizers_available'], 'versions': ri['versions'], 'n_params': ri['n_params']}, 'tasks': {}}
out = ['#### 加载（lisa_predictions/logs/run_*.json 的 runner_info；入口检查不通过时脚本会报错退出，本次通过）', '', '| 项 | 值 |', '|---|---|',
       f"| fixes_applied | {', '.join(marker['fixes_applied'])} |", f"| tokenizers 是否可用 | {ri['tokenizers_available']} |",
       f"| 入口检查：[SEG] | 单 token，id {ri['seg_token_idx']} |",
       f"| 入口检查：分词器长度 / embed_tokens 行数 | {ri['tokenizer']['len']} / {ri['embed_tokens_rows']} |",
       f"| 入口检查：bos / eos / unk / pad 的 id | {ri['tokenizer']['bos_token']} / {ri['tokenizer']['eos_token']} / {ri['tokenizer']['unk_token']} / {ri['tokenizer']['pad_token']} |",
       f"| 入口检查：生成配置 | {json.dumps(ri['generation_config'])} |",
       f"| 加载用时 | {ri['load_s']} s |", f"| 显存峰值（torch max_memory_allocated）：加载后 / 推理 | {ri['peak_mem_after_load_bytes']} B / {runlog.get('runner_stats', {}).get('peak_mem_inference_bytes')} B |",
       f"| 参数量 | {ri['n_params']:,} |", f"| 版本 | {json.dumps(ri['versions'], ensure_ascii=False)} |", '',
       '#### 逐任务（lisa_predictions/<任务>.jsonl）', '', '| 任务 | 条数 | 成功 | 失败（按原因） | [SEG] 个数分布 | 平均生成 token | 每条墙钟 s：均值 / 中位 / 最大 | 掩码像素：中位 / 最小 / 最大 |', '|---|---|---|---|---|---|---|---|']
for t in TASKS:
    R = [json.loads(l) for l in open(os.path.join(pred, f'{t}.jsonl'))]
    ok = [r for r in R if 'failure' not in r]; fails = collections.Counter(r['failure'] for r in R if 'failure' in r)
    segs = collections.Counter(r.get('n_seg') for r in R); w = [r['wall_s'] for r in R]; px = [r['n_mask_pixels'] for r in ok]
    res['tasks'][t] = {'n': len(R), 'n_ok': len(ok), 'failures': dict(fails), 'n_seg': {str(k): v for k, v in segs.items()},
                       'mean_generated_tokens': statistics.mean(r['n_generated_tokens'] for r in R), 'wall_s': {'mean': statistics.mean(w), 'median': statistics.median(w), 'max': max(w)},
                       'texts_first3': [r['text'] for r in R[:3]], 'mask_pixels': px}
    out.append(f"| {t} | {len(R)} | {len(ok)} | {json.dumps(dict(fails)) if fails else '无'} | {json.dumps({str(k): v for k, v in sorted(segs.items())})} | "
               f"{statistics.mean(r['n_generated_tokens'] for r in R):.1f} | {statistics.mean(w):.3f} / {statistics.median(w):.3f} / {max(w):.3f} | "
               f"{(statistics.median(px) if px else '—')} / {(min(px) if px else '—')} / {(max(px) if px else '—')} |")


def m(system, who, t):
    p = os.path.join(RUN, system, who, f'{t}__dev', 'metrics.json')
    return json.load(open(p)) if os.path.exists(p) else None


out += ['', '#### 评分（external/lisa-7b-v1；严格 = 失败计 0，条件 = 只算有效条；n 很小，只是核对）', '',
        '| 任务 | 范围 | n | 有效 | mask_iou_grid 严格 | 条件 | mask_iou_full 严格 | 起伏误差 |', '|---|---|---|---|---|---|---|---|']
for t in TASKS:
    M = m('external', 'lisa-7b-v1', t)
    if M is None:
        continue
    res['tasks'][t]['metrics'] = {'overall': M['overall'], 'headline': M['headline'], 'by_bin': M['by_bin'], 'headline_scope': M.get('headline_scope')}
    scopes = [('全部档', M['overall'])] + [(f'档 {b}', v) for b, v in sorted(M['by_bin'].items(), key=lambda x: int(x[0]))] + [(f"标题范围：{M.get('headline_scope')}", M['headline'])]
    for name, s in scopes:
        g = s['mask_iou_grid']; full = s.get('mask_iou_full') or {}; rel = s.get('relief_error')
        relt = '—' if rel is None else (f"严格 {f(rel['strict'])}，有效 {rel['n_valid']}/{rel['n']}，原因 {json.dumps(rel.get('reasons') or {}, ensure_ascii=False)}")
        out.append(f"| {t} | {name} | {g['n']} | {g['n_valid']} | {f(g['strict'])} | {f(g['conditional'])} | {f(full.get('strict'))} | {relt} |")
out += ['', '#### 同一批样本上的平凡基线（baseline/；全部档 / 标题范围的 mask_iou_grid 严格）', '', '| 任务 | 基线 | 全部档 | 标题范围 | 起伏误差 严格：全部档 / 标题范围 |', '|---|---|---|---|---|']
for t in TASKS:
    for b in sorted(os.listdir(os.path.join(RUN, 'baseline'))):
        M = m('baseline', b, t)
        if M is None:
            continue
        rel, hrel = M['overall'].get('relief_error'), M['headline'].get('relief_error')
        out.append(f"| {t} | {b} | {f(M['overall']['mask_iou_grid']['strict'])} | {f(M['headline']['mask_iou_grid']['strict'])} | {(f(rel['strict']) + ' / ' + f(hrel['strict'])) if rel else '—'} |")
    L = res['tasks'][t].get('metrics')
    if L:
        out.append(f"| {t} | **lisa-7b-v1** | {f(L['overall']['mask_iou_grid']['strict'])} | {f(L['headline']['mask_iou_grid']['strict'])} | "
                   f"{(f(L['overall']['relief_error']['strict']) + ' / ' + f(L['headline']['relief_error']['strict'])) if L['overall'].get('relief_error') else '—'} |")
out += ['', '#### 文本举例（每个任务前 3 条的 text，原样）', '']
for t in TASKS:
    for i, x in enumerate(res['tasks'][t]['texts_first3']):
        out.append(f"- {t} #{i}: `{x}`")
json.dump(res, open(OUT, 'w'), indent=1, ensure_ascii=False)
print('\n'.join(out))
