"""RUN-0001 §6 group C: from GLIP odinw_13 yaml configs + zip listings, pick the annotation jsons each config registers
(train / val / test / minival, 'annotations_without_background') plus README*.txt of the subset dir; print members + sizes.
Writes logs/0001/p6/C/odinw13_members.json"""
import yaml, json, os, glob
L = '/home/ubuntu/workspace/wzg/gjl/vision-internalization/logs/0001/p6/C'
ZIP = {'AerialMaritimeDrone_large': 'AerialMaritimeDrone', 'Aquarium_Aquarium_Combined.v2-raw-1024.coco': 'Aquarium', 'CottontailRabbits': 'CottontailRabbits',
       'EgoHands_generic': 'EgoHands', 'NorthAmericaMushrooms_North_American_Mushrooms.v1-416x416.coco': 'NorthAmericaMushrooms', 'Packages_Raw': 'Packages',
       'PascalVOC': 'PascalVOC', 'Raccoon_Raccoon.v2-raw.coco': 'Raccoon', 'ShellfishOpenImages_raw': 'ShellfishOpenImages', 'VehiclesOpenImages_416x416': 'VehiclesOpenImages',
       'pistols_export': 'pistols', 'pothole': 'pothole', 'thermalDogsAndPeople': 'thermalDogsAndPeople'}
out = {}; grand = 0
for cfg, z in ZIP.items():
    d = yaml.safe_load(open(f'{L}/odinw_glip_configs13/{cfg}.yaml'))['DATASETS']
    listing = {}
    for line in open(f'{L}/odinw_ziplists/{z}.txt'):
        if line.startswith('zip_size'): continue
        p = line.rstrip('\n').split(None, 2)
        if len(p) == 3: listing[p[2]] = int(p[0])
    reg = d['REGISTER']; want = {}
    for sp in ('train', 'val', 'test', 'minival'):
        if sp in reg:
            m = reg[sp]['ann_file'].removeprefix('odinw/'); want[sp] = m
    subdir = os.path.dirname(os.path.dirname(want['train'])) if '/train/' in want['train'] else os.path.dirname(want['train'])
    readmes = [k for k in listing if k.startswith(subdir + '/README') or (k.startswith(subdir) and k.count('/') == subdir.count('/') + 1 and 'README' in k)]
    rec = {'zip': z, 'subdir': subdir, 'TRAIN': d.get('TRAIN'), 'TEST': d.get('TEST'), 'ann': {}, 'readmes': sorted(set(readmes))}
    for sp, m in want.items(): rec['ann'][sp] = {'member': m, 'size': listing.get(m, 'MISSING')}
    tot = sum(v['size'] for v in rec['ann'].values() if isinstance(v['size'], int)) + sum(listing[r] for r in rec['readmes'])
    rec['bytes_needed'] = tot; grand += tot; out[cfg] = rec
    print(f"{cfg:<62} zip={z:<22} subdir={subdir!r} bytes={tot} " + ' '.join(f"{sp}:{v['size']}" for sp, v in rec['ann'].items()) + f" readmes={rec['readmes']}")
print('grand total bytes', grand)
json.dump(out, open(f'{L}/odinw13_members.json', 'w'), indent=1)
