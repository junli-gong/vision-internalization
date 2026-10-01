#!/usr/bin/env python
"""RUN-0010：逐源汇总 MANIFEST.json（来源、核对结果、落地字节、耗时、不可达 URL）为 markdown。数字只从清单与驱动日志读。
usage: python p13_manifest_summary.py <manifests_dir> <land_driver.log>"""
import json, re, sys
from pathlib import Path
D = Path(sys.argv[1]); log = Path(sys.argv[2]).read_text()
wall = {m.group(1): (int(m.group(3)), int(m.group(2))) for m in re.finditer(r"=== (\S+) rc=(\d+) wall=(\d+)s", log)}
ORDER = ['refcoco', 'coco_val2017', 'odinw13', 'countbench', 'nyuv2', 'blink_reldepth', 'cvbench', 'embspatial']
out = ['| 源 | 退出码 | 墙钟 s | 文件数 | 落地字节 | 估算字节 | 核对项（通过/总数） | 不可达 URL | 上游 |', '|---|---|---|---|---|---|---|---|---|']
fails = []
for s in ORDER:
    m = json.loads((D / f'{s}.MANIFEST.json').read_text())
    ver = m.get('verification', []); ok = sum(1 for v in ver if v.get('ok')); bad = [v for v in ver if not v.get('ok')]
    ups = []
    for u in m.get('upstream', []):
        ups.append(u.get('url') or u.get('repo') or u.get('dataset') or json.dumps(u)[:80])
    w, rc = wall.get(s, (None, None))
    out.append(f"| {s} | {rc} | {w} | {m.get('n_files')} | {m.get('total_bytes'):,} | {m.get('estimate', {}).get('estimate_bytes', 0):,} | {ok}/{len(ver)} | {len(m.get('failures', []))} | {'; '.join(str(x) for x in ups[:3])}{' …' if len(ups) > 3 else ''} |")
    for v in bad:
        fails.append(f"- {s}: {json.dumps(v, ensure_ascii=False)[:500]}")
    for f in m.get('failures', []):
        fails.append(f"- {s} 不可达: {json.dumps(f, ensure_ascii=False)[:300]}")
print('\n'.join(out))
print('\n未通过的核对项与不可达 URL：' + ('\n' + '\n'.join(fails) if fails else '无'))
print('\n各源核对项明细（what → ok）：')
for s in ORDER:
    m = json.loads((D / f'{s}.MANIFEST.json').read_text())
    print(f"\n**{s}**（landed_at {m.get('landed_at_utc')}，script commit {m.get('script', {}).get('git', {}).get('commit', '')[:12]}）")
    for v in m.get('verification', []):
        print(f"- {v.get('what')} → {v.get('ok')}")
    for n in m.get('notes', []):
        print(f"- note: {n}")
