"""RUN-0076A 报告用：训练每 500 步的 loss、各流损失、每步墙钟中位数、显存峰值（读 train_log.jsonl 与 logs/timing_run_*.jsonl）。
用法：python p76_train.py <训练叶目录> <到第几步>"""
import json, statistics, sys, glob, os
leaf = sys.argv[1]; upto = int(sys.argv[2])
rows = [json.loads(l) for l in open(os.path.join(leaf, 'train_log.jsonl')) if l.strip()]
tim = []
for p in sorted(glob.glob(os.path.join(leaf, 'logs/timing_run_*.jsonl'))):
    tim += [json.loads(l) for l in open(p) if l.strip()]
print('leaf', leaf, 'logged steps', len(rows), 'last step', rows[-1]['step'], 'timing rows', len(tim))
for lo in range(0, upto, 500):
    hi = lo + 500
    w = [r for r in rows if lo <= r['step'] < hi]
    if not w: continue
    streams = sorted({k for r in w for k in (r.get('loss_by_stream') or {})})
    ls = {s: statistics.mean([r['loss_by_stream'][s] for r in w if s in (r.get('loss_by_stream') or {})]) for s in streams}
    t = [r for r in tim if lo <= r['step'] < hi]
    print(f"steps {lo}-{w[-1]['step']}: n {len(w)}, loss mean {statistics.mean(r['loss'] for r in w):.4f}, last {w[-1]['loss']:.4f}; by stream mean " +
          ", ".join(f"{s} {v:.4f}" for s, v in ls.items()) +
          (f"; step_s median {statistics.median(r['step_s'] for r in t):.2f}, peak_mem max {max(r['peak_mem_bytes'] for r in t):,}" if t else ''))
print('keys', list(rows[-1])[:20])
