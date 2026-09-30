"""RUN-0001 §6 group C: tabulate CV-Bench per-row source fields from datasets-server rows dump (parquet view),
cross-check against the repo's test_2d/3d.jsonl. Output: logs/0001/p6/C/cvbench_source_tab.json + stdout."""
import json, collections as C, re
L = '/home/ubuntu/workspace/wzg/gjl/vision-internalization/logs/0001/p6/C'
T = '/home/ubuntu/workspace/wzg/gjl/vision-internalization/tmp/p6_C'
rows = [json.loads(l) for l in open(f'{L}/cvbench_rows_meta.jsonl')]
jl = [json.loads(l) for f in ('cvb_test_2d.jsonl', 'cvb_test_3d.jsonl') for l in open(f'{T}/{f}')]
out = {}
def tab(name, keyf, rs=rows):
    c = C.Counter(keyf(r) for r in rs); out[name] = {' | '.join(map(str, k)) if isinstance(k, tuple) else str(k): v for k, v in sorted(c.items(), key=lambda x: str(x[0]))}
    print(f'== {name}'); [print(f'   {k}: {v}') for k, v in out[name].items()]
tab('type_task', lambda r: (r['type'], r['task']))
tab('source', lambda r: r['source'])
tab('source_x_source_dataset', lambda r: (r['source'], r['source_dataset']))
tab('source_dataset_x_task', lambda r: (r['source_dataset'], r['task']))
# Omni3D sub-dataset from source_filename prefix
omni = [r for r in rows if r['source'] == 'Omni3D']
tab('omni3d_source_filename_top_dir', lambda r: r['source_filename'].split('/')[0], omni)
tab('omni3d_source_filename_2nd_dir', lambda r: '/'.join(r['source_filename'].split('/')[:2]) if r['source_filename'].startswith('SUNRGBD') or 'sunrgbd' in r['source_filename'].lower() else r['source_filename'].split('/')[0], omni)
sun = [r for r in omni if 'sunrgbd' in (r['source_dataset'] + r['source_filename']).lower()]
tab('sunrgbd_sensor_subdir(3rd level)', lambda r: '/'.join(r['source_filename'].split('/')[:3]), sun)
nyu = [r for r in omni if re.search(r'nyu', r['source_filename'], re.I)]
out['nyu_rows_count'] = len(nyu); out['nyu_unique_images'] = len({r['source_filename'] for r in nyu})
out['nyu_examples'] = sorted({r['source_filename'] for r in nyu})[:15]
out['sun_non_nyu_examples'] = sorted({r['source_filename'] for r in sun if r not in nyu})[:10]
print('NYU-pattern rows:', len(nyu), 'unique images:', out['nyu_unique_images']); [print('   ', x) for x in out['nyu_examples']]
print('SUN non-NYU examples:'); [print('   ', x) for x in out['sun_non_nyu_examples']]
for sd in sorted({r['source_dataset'] for r in rows}):
    ex = sorted({r['source_filename'] for r in rows if r['source_dataset'] == sd})
    out.setdefault('examples_per_source_dataset', {})[sd] = {'unique_source_files': len(ex), 'first3': ex[:3]}
    print(f'   {sd}: unique source files={len(ex)} e.g. {ex[:2]}')
# images reused across rows
out['unique_images_total'] = len({(r['source_dataset'], r['source_filename']) for r in rows})
out['unique_cvbench_filenames'] = len({r['filename'] for r in rows})
print('unique (source_dataset,source_filename):', out['unique_images_total'], ' unique CV-Bench filename:', out['unique_cvbench_filenames'])
# cross-check parquet rows vs jsonl
key = lambda r: (r['type'], r['task'], r['filename'], r['source'], r['source_dataset'], r['source_filename'], r['question'], r['answer'])
a, b = C.Counter(map(key, rows)), C.Counter(map(key, jl))
out['crosscheck'] = dict(parquet_rows=len(rows), jsonl_rows=len(jl), only_in_parquet=sum((a - b).values()), only_in_jsonl=sum((b - a).values()))
print('crosscheck parquet vs jsonl:', out['crosscheck'])
# COCO split hints
coco = [r for r in rows if r['source'] == 'COCO']
tab('coco_filename_prefix', lambda r: re.sub(r'\d+', '#', r['source_filename']), coco)
ade = [r for r in rows if r['source'] == 'ADE20K']
tab('ade_filename_prefix', lambda r: re.sub(r'\d+', '#', r['source_filename']), ade)
json.dump(out, open(f'{L}/cvbench_source_tab.json', 'w'), indent=1)
