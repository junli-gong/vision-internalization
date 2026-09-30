"""RUN-0001 §6 group C: check CV-Bench Omni3D_SUNRGBD source_filename against SUN RGB-D official split (allsplit.mat)
and Omni3D SUNRGBD_val.json membership. Also Omni3D stats.json per-split image counts."""
import json, re, scipy.io as sio, collections as C
T = '/home/ubuntu/workspace/wzg/gjl/vision-internalization/tmp/p6_C'; L = '/home/ubuntu/workspace/wzg/gjl/vision-internalization/logs/0001/p6/C'
m = sio.loadmat(f'{T}/allsplit.mat', squeeze_me=True, struct_as_record=False)
print({k: (type(v).__name__, getattr(v, 'shape', None)) for k, v in m.items() if not k.startswith('__')})
def paths(v):
    out = []
    for x in (v.ravel() if hasattr(v, 'ravel') else v): out.append(str(x))
    return out
def norm(p):  # strip machine prefix up to 'SUNRGBD/', drop trailing slash and double slashes
    p = p.replace('\\', '/'); i = p.find('SUNRGBD/'); p = p[i:] if i >= 0 else p
    return re.sub(r'/+', '/', p).rstrip('/')
sets = {}
for k in ('alltrain', 'alltest', 'trainvalsplit'):
    if k in m:
        v = m[k]
        if k == 'trainvalsplit':
            for f in v._fieldnames: sets[f'trainvalsplit.{f}'] = {norm(x) for x in paths(getattr(v, f))}
        else: sets[k] = {norm(x) for x in paths(v)}
for k, s in sets.items(): print(k, len(s), sorted(s)[:2])
rows = [json.loads(l) for l in open(f'{L}/cvbench_rows_meta.jsonl')]
sun = [r for r in rows if r['source_dataset'] == 'Omni3D_SUNRGBD']
def seq(fn):  # 'SUNRGBD/kv1/NYUdata/NYU0021//image/NYU0021.jpg' -> 'SUNRGBD/kv1/NYUdata/NYU0021'
    return re.sub(r'/+', '/', fn.split('/image/')[0]).rstrip('/')
res = {'n_rows': len(sun), 'n_unique_seq': len({seq(r['source_filename']) for r in sun})}
for k, s in sets.items():
    res[f'unique_in_{k}'] = sum(1 for q in {seq(r['source_filename']) for r in sun} if q in s)
    res[f'rows_in_{k}'] = sum(1 for r in sun if seq(r['source_filename']) in s)
# Omni3D val json
v = json.load(open(f'{T}/SUNRGBD_val.json'))
print('Omni3D SUNRGBD_val keys:', list(v.keys()), 'n_images', len(v['images']), 'info', v.get('info'))
vp = {re.sub(r'/+', '/', im['file_path']) for im in v['images']}
print('val file_path e.g.', sorted(vp)[:2])
res['omni3d_val_n_images'] = len(vp)
res['rows_in_omni3d_val'] = sum(1 for r in sun if re.sub(r'/+', '/', r['source_filename']) in vp)
res['unique_in_omni3d_val'] = len({r['source_filename'] for r in sun if re.sub(r'/+', '/', r['source_filename']) in vp})
st = json.load(open(f'{T}/stats.json'))
res['omni3d_stats_json_head'] = str(st)[:1500]
print(json.dumps(res, indent=1))
json.dump(res, open(f'{L}/cvbench_sunrgbd_split_check.json', 'w'), indent=1)
