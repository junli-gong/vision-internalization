"""Read members of a remote zip via HTTP Range requests (central directory + one member), no full download.
usage: python remote_zip.py <url> list | get <member> <out_path>"""
import sys, io, zipfile, urllib.request
class RangeFile(io.RawIOBase):
    def __init__(self, url):
        self.url, self.pos = url, 0
        req = urllib.request.Request(url, method='HEAD')
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size = int(r.headers['Content-Length']); self.accept = r.headers.get('Accept-Ranges')
        self.fetched = 0
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off); return self.pos
    def read(self, n=-1):
        if n is None or n < 0: n = self.size - self.pos
        if n == 0 or self.pos >= self.size: return b''
        end = min(self.pos + n, self.size) - 1
        req = urllib.request.Request(self.url, headers={'Range': f'bytes={self.pos}-{end}'})
        with urllib.request.urlopen(req, timeout=120) as r: data = r.read()
        self.pos += len(data); self.fetched += len(data); return data
    def readinto(self, b):
        d = self.read(len(b)); b[:len(d)] = d; return len(d)
url = sys.argv[1]
f = RangeFile(url)
z = zipfile.ZipFile(io.BufferedReader(f, buffer_size=1 << 20))
if sys.argv[2] == 'list':
    print(f'zip_size={f.size} accept_ranges={f.accept}')
    for i in z.infolist(): print(f'{i.file_size:>12d} {i.compress_size:>12d} {i.filename}')
else:
    m, outp = sys.argv[3], sys.argv[4]
    info = z.getinfo(m)
    assert info.file_size < 50e6, f'{m} is {info.file_size} bytes >= 50 MB limit'
    with z.open(m) as src, open(outp, 'wb') as dst: dst.write(src.read())
    print(f'wrote {outp} uncompressed={info.file_size} compressed={info.compress_size} bytes_fetched_total={f.fetched}')
