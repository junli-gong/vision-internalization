"""RUN-0001 §6 group B: HF dataset repo metadata (no data download): card license/tags, file list with sizes, total bytes,
README (card) text saved for evidence. usage: p6_B_hfinfo.py <out_dir> repo_id [repo_id ...]"""
import sys, json, os, collections
from huggingface_hub import HfApi
api = HfApi(); out_dir = sys.argv[1]
for rid in sys.argv[2:]:
    try:
        info = api.dataset_info(rid, files_metadata=True)
    except Exception as e:
        print(json.dumps({'repo': rid, 'error': repr(e)[:300]})); continue
    files = [(s.rfilename, s.size or 0) for s in info.siblings]
    tot = sum(sz for _, sz in files)
    ext = collections.Counter(os.path.splitext(f)[1] or f for f, _ in files)
    card = info.card_data.to_dict() if info.card_data else {}
    rec = {'repo': rid, 'sha': info.sha, 'last_modified': str(info.last_modified), 'gated': info.gated, 'private': info.private,
           'downloads': info.downloads, 'card_license': card.get('license'), 'card_license_name': card.get('license_name'),
           'card_license_link': card.get('license_link'),
           'tags_license': [t for t in (info.tags or []) if t.startswith('license:')], 'n_files': len(files), 'total_bytes': tot,
           'ext_counts': dict(ext), 'card_configs': card.get('configs'), 'card_dataset_info': card.get('dataset_info'),
           'files_top': sorted(files, key=lambda x: -x[1])[:15], 'files_all_if_small': files if len(files) <= 200 else None}
    safe = rid.replace('/', '__')
    json.dump(rec, open(f'{out_dir}/hfinfo_{safe}.json', 'w'), indent=1, default=str)
    try:
        p = api.hf_hub_download(rid, 'README.md', repo_type='dataset', local_dir=f'{out_dir}/cards/{safe}')
        rec['readme_bytes'] = os.path.getsize(p)
    except Exception as e:
        rec['readme_bytes'] = f'no README ({type(e).__name__})'
    print(json.dumps({k: rec[k] for k in ('repo', 'gated', 'card_license', 'tags_license', 'n_files', 'total_bytes', 'ext_counts', 'last_modified', 'readme_bytes')}, default=str))
