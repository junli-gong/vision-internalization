"""RUN-0001 §6 group C: remote zip via HTTP Range (wraps $PROJ/scripts/remote_zip.py's RangeFile idea) with network-byte accounting.
usage: p6_C_zip.py <url> list [grep_substr]        -> prints size/compressed/name (+ network bytes read)
       p6_C_zip.py <url> get <outdir> <member>...  -> extracts members (each must be < 50 MB) in one central-directory read"""
import sys, io, os, zipfile, urllib.request, time
class RangeFile(io.RawIOBase):
    def __init__(self, url):
        self.url, self.pos, self.fetched = url, 0, 0
        with urllib.request.urlopen(urllib.request.Request(url, method='HEAD'), timeout=60) as r:
            self.size = int(r.headers['Content-Length']); self.accept = r.headers.get('Accept-Ranges')
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off); return self.pos
    def read(self, n=-1):
        if n is None or n < 0: n = self.size - self.pos
        if n == 0 or self.pos >= self.size: return b''
        end = min(self.pos + n, self.size) - 1
        for k in range(5):
            try:
                req = urllib.request.Request(self.url, headers={'Range': f'bytes={self.pos}-{end}'})
                with urllib.request.urlopen(req, timeout=120) as r: data = r.read()
                break
            except Exception:
                if k == 4: raise
                time.sleep(2 ** k)
        self.pos += len(data); self.fetched += len(data); return data
    def readinto(self, b):
        d = self.read(len(b)); b[:len(d)] = d; return len(d)
url, cmd = sys.argv[1], sys.argv[2]
f = RangeFile(url); z = zipfile.ZipFile(io.BufferedReader(f, buffer_size=1 << 16))
if cmd == 'list':
    sub = sys.argv[3] if len(sys.argv) > 3 else ''
    print(f'zip_size={f.size} accept_ranges={f.accept} n_entries={len(z.infolist())}')
    for i in z.infolist():
        if sub in i.filename: print(f'{i.file_size:>12d} {i.compress_size:>12d} {i.filename}')
else:
    outdir = sys.argv[3]
    for m in sys.argv[4:]:
        info = z.getinfo(m); assert info.file_size < 50e6, f'{m} is {info.file_size} bytes >= 50 MB'
        op = os.path.join(outdir, os.path.basename(m))
        with z.open(m) as src, open(op, 'wb') as dst: dst.write(src.read())
        print(f'wrote {op} uncompressed={info.file_size} compressed={info.compress_size}')
print(f'network_bytes_range_read={f.fetched}', file=sys.stderr)
