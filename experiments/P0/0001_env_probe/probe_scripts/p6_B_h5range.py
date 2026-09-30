"""RUN-0001 §6 group B: open a remote MATLAB v7.3 (HDF5) file via HTTP Range reads with a hard byte budget,
list top-level variables (shape/dtype/chunks/MATLAB_class) without downloading the payload. usage: p6_B_h5range.py <url> <budget_MB>"""
import sys, io, json, h5py, httpx
url, budget = sys.argv[1], float(sys.argv[2]) * 1e6
BS = 1 << 16
class RangeCache(io.RawIOBase):
    def __init__(self, url):
        self.c = httpx.Client(follow_redirects=True, timeout=120)
        r = self.c.head(url); self.url = str(r.url); self.size = int(r.headers['content-length'])
        self.pos = 0; self.blocks = {}; self.fetched = 0; self.nreq = 0
    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos
    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off); return self.pos
    def _block(self, i):
        if i not in self.blocks:
            if self.fetched + BS > budget: raise RuntimeError(f'byte budget exceeded ({self.fetched} B fetched)')
            s = i * BS; e = min(s + BS, self.size) - 1
            d = self.c.get(self.url, headers={'Range': f'bytes={s}-{e}'}).content
            self.blocks[i] = d; self.fetched += len(d); self.nreq += 1
        return self.blocks[i]
    def readinto(self, b):
        n = min(len(b), self.size - self.pos)
        if n <= 0: return 0
        out = bytearray(); p = self.pos
        while len(out) < n:
            blk = self._block(p // BS); off = p % BS; take = blk[off:off + n - len(out)]; out += take; p += len(take)
        b[:n] = out; self.pos += n; return n
f = RangeCache(url)
res = {'url': url, 'size': f.size, 'mat_header': None, 'vars': {}}
res['mat_header'] = bytes(f._block(0)[:116]).decode('latin1').strip()
try:
    h = h5py.File(f, 'r')
    for k in h.keys():
        o = h[k]
        d = {'h5type': type(o).__name__, 'attrs': {a: (o.attrs[a].decode() if isinstance(o.attrs[a], bytes) else str(o.attrs[a])) for a in o.attrs}}
        if isinstance(o, h5py.Dataset):
            d.update(shape=list(o.shape), dtype=str(o.dtype), chunks=list(o.chunks) if o.chunks else None, compression=o.compression)
        else:
            d['n_members'] = len(o)
        res['vars'][k] = d
except Exception as e:
    res['error'] = repr(e)
res['bytes_fetched'] = f.fetched; res['n_range_requests'] = f.nreq
print(json.dumps(res, indent=1))
