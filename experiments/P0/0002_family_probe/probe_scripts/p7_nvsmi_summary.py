"""Summarize nvidia-smi csv per window from a marks file: python p7_nvsmi_summary.py <csv> <marks>"""
import csv, sys, datetime as dt
rows = list(csv.reader(open(sys.argv[1])))[1:]
p = lambda s: dt.datetime.strptime(s.strip(), '%Y/%m/%d %H:%M:%S.%f')
data = [(p(r[0]), float(r[1].split()[0]), float(r[2].split()[0]), float(r[3].split()[0])) for r in rows if len(r) >= 4 and 'N/A' not in r[3]]
marks = [l.strip() for l in open(sys.argv[2]) if l.strip()]
print('| 窗口 | 采样数 | util 最大 | util 均值 | nvidia-smi 显存最大 | 功耗最大 |'); print('|---|---|---|---|---|---|')
for i in range(0, len(marks), 2):
    name = marks[i].split(' ', 1)[1].rsplit(' ', 1)[0]
    t0 = dt.datetime.fromisoformat(marks[i].split()[-1]).replace(tzinfo=None); t1 = dt.datetime.fromisoformat(marks[i + 1].split()[2]).replace(tzinfo=None)
    sel = [d for d in data if t0 <= d[0] <= t1]
    if sel:
        print(f'| {name} | {len(sel)} | {max(d[1] for d in sel):.0f}% | {sum(d[1] for d in sel)/len(sel):.1f}% | {max(d[2] for d in sel):.0f} MiB | {max(d[3] for d in sel):.0f} W |')
