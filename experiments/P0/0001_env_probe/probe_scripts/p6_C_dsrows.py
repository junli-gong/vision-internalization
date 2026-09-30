"""RUN-0001 §6 group C: datasets-server helper. Fetch /first-rows or /filter or /rows JSON, truncate long string cells
(image URLs / base64) to 120 chars, print features + rows. Nothing is written except the truncated JSON to logs.
usage: p6_C_dsrows.py <out_name> <endpoint> <dataset> <config> <split> [where] [offset] [length]"""
import sys, json, urllib.request, urllib.parse, time
from huggingface_hub.utils import build_hf_headers
out, ep, ds, cfg, sp = sys.argv[1:6]
where = sys.argv[6] if len(sys.argv) > 6 and sys.argv[6] != '-' else None
off = int(sys.argv[7]) if len(sys.argv) > 7 else 0; ln = int(sys.argv[8]) if len(sys.argv) > 8 else 3
q = dict(dataset=ds, config=cfg, split=sp)
if ep in ('rows', 'filter'): q.update(offset=off, length=ln)
if where: q['where'] = where
url = f'https://datasets-server.huggingface.co/{ep}?' + urllib.parse.urlencode(q)
for k in range(5):
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=build_hf_headers()), timeout=120) as r: d = json.loads(r.read()); break
    except urllib.error.HTTPError as e:
        print('HTTP', e.code, e.read()[:400]); sys.exit(1)
    except Exception as e: time.sleep(2 ** k)
def trunc(x):
    if isinstance(x, str): return x if len(x) <= 120 else x[:120] + f'...<{len(x)} chars>'
    if isinstance(x, list): return [trunc(y) for y in x]
    if isinstance(x, dict): return {k: trunc(v) for k, v in x.items()}
    return x
d = trunc(d); d['_url'] = url
json.dump(d, open(f'/home/ubuntu/workspace/wzg/gjl/vision-internalization/logs/0001/p6/C/{out}.json', 'w'), indent=1)
print('url:', url); print('num_rows_total:', d.get('num_rows_total'), 'partial:', d.get('partial'))
print('features:'); [print('  ', f['name'], json.dumps(f['type'])) for f in d.get('features', [])]
for r in d.get('rows', [])[:ln]: print('row', r.get('row_idx'), json.dumps(r['row'])[:1500])
