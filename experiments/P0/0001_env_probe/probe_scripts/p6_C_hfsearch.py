"""RUN-0001 §6 group C: search HF datasets by keyword, print id/downloads/likes/license tags. usage: p6_C_hfsearch.py kw1 kw2 ..."""
import sys, json
from huggingface_hub import HfApi
api = HfApi(); out = {}
for kw in sys.argv[1:]:
    res = []
    for d in api.list_datasets(search=kw, limit=40, sort='downloads', full=True):
        lic = [t for t in (d.tags or []) if t.startswith('license:')]
        res.append(dict(id=d.id, downloads=d.downloads, likes=d.likes, license=lic, gated=getattr(d, 'gated', None), last_modified=str(d.last_modified)))
    out[kw] = res
    print(f'== {kw}')
    for r in res: print(f"  {r['id']:<55} dl={r['downloads']} likes={r['likes']} {r['license']} gated={r['gated']}")
