"""RUN-0001 §6 group C: fetch ODinW-13 annotation jsons (GLIP-registered train/val/test/minival) + README*.txt via HTTP Range
from GLIPModel/GLIP odinw_35/*.zip (pinned sha), compute stats, save stats + README text to logs, log downloads. Files go to
tmp/p6_C/odinw/ and are deleted by the caller afterwards."""
import json, os, subprocess, datetime, collections as C
PROJ = '/home/ubuntu/workspace/wzg/gjl/vision-internalization'; L = f'{PROJ}/logs/0001/p6/C'; T = f'{PROJ}/tmp/p6_C/odinw'
SHA = '66ee3ae9a3b8cee0cf78f10ef5fc9a3725db02a1'
mem = json.load(open(f'{L}/odinw13_members.json')); stats = {}
def resolve(u):
    out = subprocess.run(['curl', '-sI', '--max-time', '60', u], capture_output=True, text=True).stdout
    return [l.split(' ', 1)[1].strip() for l in out.splitlines() if l.lower().startswith('location:')][0]
for cfg, rec in mem.items():
    u = f"https://huggingface.co/GLIPModel/GLIP/resolve/{SHA}/odinw_35/{rec['zip']}.zip"; loc = resolve(u)
    od = f'{T}/{cfg}'; os.makedirs(od, exist_ok=True)
    members = sorted({v['member'] for v in rec['ann'].values()}) + rec['readmes']
    if cfg == 'pistols_export': members += ['pistols/README.dataset.txt', 'pistols/README.roboflow.txt']
    s = {}
    for m in members:
        sub = os.path.join(od, m.replace('/', '__'))
        r = subprocess.run(['python', f'{PROJ}/scripts/p6_C_zip.py', loc, 'get', od, m], capture_output=True, text=True)
        src = os.path.join(od, os.path.basename(m)); os.replace(src, sub)
        nb = os.path.getsize(sub)
        with open(f'{L}/downloads.tsv', 'a') as f: f.write(f"{datetime.datetime.now().isoformat()}\t{u}\t{m}\t{nb}\tODinW13 annotation/README stats\n")
        s[m] = {'bytes': nb, 'net': r.stderr.strip()}
    rs = {}
    for sp, v in rec['ann'].items():
        j = json.load(open(os.path.join(od, v['member'].replace('/', '__'))))
        cats = j.get('categories', []); anns = j.get('annotations', []); ims = j.get('images', [])
        wh = C.Counter((im.get('width'), im.get('height')) for im in ims)
        rs[sp] = dict(member=v['member'], n_images=len(ims), n_annotations=len(anns), n_categories=len(cats),
                      categories=[c.get('name') for c in cats], top_image_sizes=wh.most_common(3), keys=sorted(j.keys()),
                      image_keys=sorted(ims[0].keys()) if ims else [], ann_keys=sorted(anns[0].keys()) if anns else [],
                      ann_example=anns[0] if anns else None, info=j.get('info'), licenses=j.get('licenses'))
    readme_txt = {}
    for m in members:
        if 'README' in m: readme_txt[m] = open(os.path.join(od, m.replace('/', '__')), errors='ignore').read()
    stats[cfg] = dict(zip=rec['zip'], subdir=rec['subdir'], TRAIN=rec['TRAIN'], TEST=rec['TEST'], splits=rs, readmes=readme_txt, fetched=s)
    print(cfg, {sp: (x['n_images'], x['n_annotations'], x['n_categories']) for sp, x in rs.items()})
json.dump(stats, open(f'{L}/odinw13_ann_stats.json', 'w'), indent=1, default=str)
