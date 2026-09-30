"""RUN-0001 §6 group C: paginate datasets-server /rows for nyu-visionx/CV-Bench (config=default, split=test),
keep only metadata columns (drop image URL), save jsonl to logs; images are never fetched."""
import json, time, urllib.request, concurrent.futures as cf
from huggingface_hub.utils import build_hf_headers
OUT = '/home/ubuntu/workspace/wzg/gjl/vision-internalization/logs/0001/p6/C/cvbench_rows_meta.jsonl'
hdr = build_hf_headers(); N = 2638
def page(off):
    url = f'https://datasets-server.huggingface.co/rows?dataset=nyu-visionx%2FCV-Bench&config=default&split=test&offset={off}&length=100'
    for k in range(6):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=hdr), timeout=120) as r: d = json.loads(r.read())
            return off, d
        except Exception as e: time.sleep(2 ** k); err = e
    raise RuntimeError(f'offset {off}: {err}')
with cf.ThreadPoolExecutor(8) as ex: res = dict(ex.map(page, range(0, N, 100)))
rows = []
for off in sorted(res):
    d = res[off]
    for r in d['rows']:
        row = {k: v for k, v in r['row'].items() if k != 'image'}; row['_row_idx'] = r['row_idx']; rows.append(row)
print('num_rows_total reported:', res[0].get('num_rows_total'), 'fetched:', len(rows), 'partial:', res[0].get('partial'))
print('features:', [(f['name'], f['type']) for f in res[0]['features']])
with open(OUT, 'w') as f:
    for r in rows: f.write(json.dumps(r) + '\n')
