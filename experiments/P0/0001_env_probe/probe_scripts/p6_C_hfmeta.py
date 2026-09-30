"""RUN-0001 §6 group C: HF metadata only (no data download).
For each dataset repo: HfApi.dataset_info(files_metadata=True) -> license/tags/gated/files+sizes;
README card text (small); datasets-server /splits /size /info JSON.
usage: p6_C_hfmeta.py repo1 repo2 ...   writes $PROJ/logs/0001/p6/C/hf/<repo__id>.json and .README.md"""
import sys, json, os, time, urllib.request, urllib.parse
from huggingface_hub import HfApi, hf_hub_url
from huggingface_hub.utils import build_hf_headers
OUT = '/home/ubuntu/workspace/wzg/gjl/vision-internalization/logs/0001/p6/C/hf'; os.makedirs(OUT, exist_ok=True)
api = HfApi(); hdr = build_hf_headers()
def get(url, tries=4):
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=hdr), timeout=90) as r: return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (400, 401, 403, 404, 501): return json.dumps({'_error': e.code, '_body': e.read()[:500].decode('utf-8', 'ignore')}).encode()
            time.sleep(2 ** k)
        except Exception as e: time.sleep(2 ** k); err = str(e)
    return json.dumps({'_error': 'failed'}).encode()
for repo in sys.argv[1:]:
    rec = {'repo': repo}
    try:
        d = api.dataset_info(repo, files_metadata=True)
        files = [(s.rfilename, s.size) for s in d.siblings]
        rec.update(sha=d.sha, last_modified=str(d.last_modified), gated=d.gated, private=d.private, tags=d.tags,
                   card_license=(d.card_data or {}).get('license') if d.card_data else None,
                   card_configs=(d.card_data or {}).get('configs') if d.card_data else None,
                   n_files=len(files), total_bytes=sum(s or 0 for _, s in files), files=files[:400], files_truncated=len(files) > 400)
    except Exception as e:
        rec['dataset_info_error'] = repr(e)[:500]
    q = urllib.parse.quote(repo, safe='')
    for ep in ('splits', 'size', 'info'):
        try: rec['ds_server_' + ep] = json.loads(get(f'https://datasets-server.huggingface.co/{ep}?dataset={q}'))
        except Exception as e: rec['ds_server_' + ep] = {'_error': repr(e)}
    fn = repo.replace('/', '__')
    json.dump(rec, open(f'{OUT}/{fn}.json', 'w'), indent=1, default=str)
    readme = get(f'https://huggingface.co/datasets/{repo}/raw/main/README.md')
    open(f'{OUT}/{fn}.README.md', 'wb').write(readme[:2_000_000])
    sp = rec.get('ds_server_size', {}).get('size', {}).get('splits', [])
    print(f"== {repo} sha={rec.get('sha')} gated={rec.get('gated')} license={rec.get('card_license')} n_files={rec.get('n_files')} total_bytes={rec.get('total_bytes')} err={rec.get('dataset_info_error','')}")
    for s in sp: print(f"   split {s['config']}/{s['split']}: rows={s['num_rows']} parquet_bytes={s['num_bytes_parquet_files']}")
    if not sp: print('   ds-server splits:', json.dumps(rec.get('ds_server_splits'))[:300])
