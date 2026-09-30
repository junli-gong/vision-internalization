"""RUN-0001 §6 group A: peek at the first N decompressed bytes of one member of a remote zip (Range reads, small buffer,
in memory only) to see its top-level JSON structure without downloading the member.
usage: python p6_A_zip_head.py <url> <member> <n_bytes> <downloads.tsv>"""
import sys, io, zipfile, time, urllib.request
sys.path.insert(0, "/home/ubuntu/workspace/wzg/gjl/vision-internalization/scripts")
url, member, n, dl = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
class RangeFile(io.RawIOBase):
    def __init__(self, url):
        self.url, self.pos, self.fetched = url, 0, 0
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as r: self.size = int(r.headers["Content-Length"])
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off); return self.pos
    def readinto(self, b):
        if self.pos >= self.size: return 0
        end = min(self.pos + len(b), self.size) - 1
        with urllib.request.urlopen(urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"}), timeout=120) as r: d = r.read()
        b[:len(d)] = d; self.pos += len(d); self.fetched += len(d); return len(d)
f = RangeFile(url)
z = zipfile.ZipFile(io.BufferedReader(f, buffer_size=128 << 10))
with z.open(member) as m: head = m.read(n)
print(f"member={member} uncompressed={z.getinfo(member).file_size} bytes_fetched_total={f.fetched}")
print(head.decode("utf-8", "replace"))
with open(dl, "a") as g:
    g.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')}\t{url}#{member}[head {n}B]\t{f.fetched}\tpeek top-level JSON structure; in memory only\tin-memory-only\n")
