"""RUN-0001 §6 group B: via HTTP Range reads (byte budget, in-memory only), decode small MATLAB cell-array variables
(scenes, sceneTypes, names) of nyu_depth_v2_labeled.mat and cross them with the official splits.mat. usage: p6_B_h5cells.py <budget_MB>"""
import sys, os, json, collections, h5py, numpy as np, scipy.io
sys.argv = [sys.argv[0], 'http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat', sys.argv[1]]
exec(open('/home/ubuntu/workspace/wzg/gjl/vision-internalization/scripts/p6_B_h5range.py').read().split('f = RangeCache(url)')[0])
f = RangeCache(url); h = h5py.File(f, 'r')
def cellstr(name):
    refs = h[name][()].ravel()
    return [''.join(chr(c) for c in h[r][()].ravel()) for r in refs]
res = {}
try:
    scenes = cellstr('scenes'); res['bytes_after_scenes'] = f.fetched
    types = cellstr('sceneTypes'); res['bytes_after_sceneTypes'] = f.fetched
    names = cellstr('names'); res['bytes_after_names'] = f.fetched
    sp = scipy.io.loadmat('/home/ubuntu/workspace/wzg/gjl/datasets/nyuv2/splits.mat')
    tr = sp['trainNdxs'].ravel() - 1; te = sp['testNdxs'].ravel() - 1
    str_ = set(scenes[i] for i in tr); ste = set(scenes[i] for i in te)
    res.update(n_images=len(scenes), n_unique_scenes=len(set(scenes)), n_train_scenes=len(str_), n_test_scenes=len(ste),
               scene_overlap_train_test=sorted(str_ & ste), n_classes_names=len(names), names_first10=names[:10],
               sceneTypes_counts=dict(collections.Counter(types).most_common()),
               sceneTypes_train=dict(collections.Counter(types[i] for i in tr).most_common()),
               sceneTypes_test=dict(collections.Counter(types[i] for i in te).most_common()),
               scenes_first5=scenes[:5], test_scenes_sorted=sorted(ste), train_scenes_sorted=sorted(str_))
except Exception as e:
    res['error'] = repr(e)
res['bytes_fetched'] = f.fetched; res['n_range_requests'] = f.nreq
print(json.dumps(res, indent=1)); sys.stdout.flush()
h.close(); os._exit(0)
