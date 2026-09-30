"""RUN-0001 §6 group B: read NYUv2 official splits.mat (read-only) and dump facts."""
import json, hashlib, os, scipy.io, numpy as np
P = '/home/ubuntu/workspace/wzg/gjl/datasets/nyuv2/splits.mat'
raw = open(P, 'rb').read()
out = {'path': P, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(), 'header': raw[:116].decode('latin1').strip()}
m = scipy.io.loadmat(P)
out['keys'] = [k for k in m.keys()]
vars_ = {}
for k, v in m.items():
    if k.startswith('__'): continue
    a = np.asarray(v)
    vars_[k] = {'dtype': str(a.dtype), 'shape': list(a.shape), 'n': int(a.size), 'min': int(a.min()), 'max': int(a.max()),
                'unique': int(np.unique(a).size), 'sorted_ascending': bool(np.all(np.diff(a.ravel()) > 0)),
                'first10': a.ravel()[:10].tolist(), 'last5': a.ravel()[-5:].tolist()}
out['vars'] = vars_
tr = set(np.asarray(m['trainNdxs']).ravel().tolist()); te = set(np.asarray(m['testNdxs']).ravel().tolist())
u = tr | te
out['check'] = {'n_train': len(tr), 'n_test': len(te), 'intersection': len(tr & te), 'union': len(u),
                'union_equals_1..1449': u == set(range(1, 1450)), 'union_equals_0..1448': u == set(range(0, 1449)),
                'test_first20': sorted(te)[:20]}
print(json.dumps(out, indent=1))
