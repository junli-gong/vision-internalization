"""RUN-0001 §6 group B: read the first N members of a remote (uncompressed) tar via HTTP Range (in memory), and if a member is .h5,
open it with h5py and report keys/shapes/dtypes. usage: p6_B_tarpeek.py <url> <n_members> [max_member_MB]"""
import sys, io, json, tarfile, httpx, h5py, numpy as np
url, n = sys.argv[1], int(sys.argv[2]); mx = float(sys.argv[3]) * 1e6 if len(sys.argv) > 3 else 5e6
c = httpx.Client(follow_redirects=True, timeout=120)
fetched = 0
def rng(a, b):
    global fetched
    d = c.get(url, headers={'Range': f'bytes={a}-{b}'}).content; fetched += len(d); return d
off = 0; out = []
for _ in range(n):
    hdr = rng(off, off + 511)
    ti = tarfile.TarInfo.frombuf(hdr, 'utf-8', 'surrogateescape')
    rec = {'name': ti.name, 'size': ti.size, 'offset': off}
    data_off = off + 512
    if ti.name.endswith('.h5') and ti.size <= mx:
        blob = rng(data_off, data_off + ti.size - 1)
        h = h5py.File(io.BytesIO(blob), 'r')
        rec['h5'] = {k: {'shape': list(h[k].shape), 'dtype': str(h[k].dtype)} for k in h.keys()}
        if 'depth' in h:
            dd = np.array(h['depth']); rec['depth_stats'] = {'min': float(dd.min()), 'max': float(dd.max()), 'zeros': int((dd == 0).sum())}
        h.close()
    out.append(rec)
    off = data_off + ((ti.size + 511) // 512) * 512
print(json.dumps({'url': url, 'members': out, 'range_bytes_fetched': fetched}, indent=1))
