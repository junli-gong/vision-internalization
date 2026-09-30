"""RUN-0001 §6 group C: render markdown tables for facts.md from the raw evidence JSONs in logs/0001/p6/C."""
import json, re
L = '/home/ubuntu/workspace/wzg/gjl/vision-internalization/logs/0001/p6/C'
st = json.load(open(f'{L}/odinw13_ann_stats.json'))
ZS = {}
for line in open(f'{L}/odinw_glip_hf_tree.txt'):
    m = re.match(r'\s*(\d+) odinw_35/(\S+)\.zip', line)
    if m: ZS[m.group(2)] = int(m.group(1))
print('| # | GLIP config (configs/odinw_13) | zip (GLIPModel/GLIP odinw_35) | zip bytes (HF API) | train img/box | valid img/box | test img/box | minival img/box | GLIP DATASETS.TEST | #cls | license (README.dataset.txt) |')
print('|---|---|---|---:|---|---|---|---|---|---:|---|')
for i, (cfg, r) in enumerate(st.items(), 1):
    sp = r['splits']; f = lambda k: f"{sp[k]['n_images']}/{sp[k]['n_annotations']}" if k in sp else '—'
    lic = [l.split(':', 1)[1].strip() for t in r['readmes'].values() for l in t.splitlines() if l.strip().startswith('License:')]
    print(f"| {i} | {cfg} | {r['zip']}.zip | {ZS[r['zip']]:,} | {f('train')} | {f('val')} | {f('test')} | {f('minival')} | {r['TEST']} | {sp['train']['n_categories']} | {lic[0] if lic else '?'} |")
tot = {k: (sum(r['splits'][k]['n_images'] for r in st.values() if k in r['splits']), sum(r['splits'][k]['n_annotations'] for r in st.values() if k in r['splits'])) for k in ('train', 'val', 'test', 'minival')}
print(f"| Σ | | 13 zips | {sum(ZS[r['zip']] for r in st.values()):,} | {tot['train'][0]}/{tot['train'][1]} | {tot['val'][0]}/{tot['val'][1]} | {tot['test'][0]}/{tot['test'][1]} | {tot['minival'][0]}/{tot['minival'][1]} | | | |")
print()
b = json.load(open(f'{L}/hf/BLINK-Benchmark__BLINK.json'))['ds_server_size']['size']['splits']
cfgs = sorted({s['config'] for s in b}); g = {(s['config'], s['split']): s for s in b}
print('| BLINK subtask | val rows | test rows | val+test parquet bytes |'); print('|---|---:|---:|---:|')
for c in cfgs: print(f"| {c} | {g[(c,'val')]['num_rows']} | {g[(c,'test')]['num_rows']} | {g[(c,'val')]['num_bytes_parquet_files'] + g[(c,'test')]['num_bytes_parquet_files']:,} |")
print(f"| Σ (14) | {sum(g[(c,'val')]['num_rows'] for c in cfgs)} | {sum(g[(c,'test')]['num_rows'] for c in cfgs)} | {sum(s['num_bytes_parquet_files'] for s in b):,} |")
