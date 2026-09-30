"""RUN-0001 §6 group B: summarize a remote zip's central directory via HTTP Range reads (in memory; nothing written but the
listing). usage: p6_B_zipsum.py <url> <listing_out.tsv> [budget_MB]"""
import sys, io, zipfile, collections, json, httpx
url, outp = sys.argv[1], sys.argv[2]; budget = float(sys.argv[3]) * 1e6 if len(sys.argv) > 3 else 30e6
class RF(io.RawIOBase):
    def __init__(s, u):
        s.c = httpx.Client(follow_redirects=True, timeout=120, headers={'User-Agent': 'Mozilla/5.0'})
        r = s.c.head(u); s.u = str(r.url); s.size = int(r.headers['content-length']); s.pos = 0; s.fetched = 0; s.nreq = 0
    def seekable(s): return True
    def readable(s): return True
    def tell(s): return s.pos
    def seek(s, o, w=0): s.pos = o if w == 0 else (s.pos + o if w == 1 else s.size + o); return s.pos
    def readinto(s, b):
        n = min(len(b), s.size - s.pos)
        if n <= 0: return 0
        if s.fetched + n > budget: raise RuntimeError(f'budget exceeded at {s.fetched}')
        d = s.c.get(s.u, headers={'Range': f'bytes={s.pos}-{s.pos + n - 1}'}).content
        b[:len(d)] = d; s.pos += len(d); s.fetched += len(d); s.nreq += 1; return len(d)
f = RF(url)
z = zipfile.ZipFile(io.BufferedReader(f, buffer_size=1 << 20))
infos = z.infolist()
with open(outp, 'w') as o:
    for i in infos: o.write(f'{i.file_size}\t{i.compress_size}\t{i.filename}\n')
top = collections.Counter(); topb = collections.Counter(); ext = collections.Counter(); lvl2 = collections.Counter()
for i in infos:
    parts = i.filename.split('/'); top[parts[0]] += 1; topb[parts[0]] += i.file_size
    if len(parts) > 2: lvl2['/'.join(parts[:2])] += 1
    ext[i.filename.rsplit('.', 1)[-1] if '.' in parts[-1] else ('<dir>' if i.is_dir() else '<noext>')] += 1
print(json.dumps({'url': url, 'zip_bytes': f.size, 'n_entries': len(infos), 'n_files': sum(not i.is_dir() for i in infos),
                  'uncompressed_total': sum(i.file_size for i in infos), 'top_level_counts': dict(top.most_common(20)),
                  'top_level_bytes': dict(topb.most_common(20)), 'ext_counts': dict(ext.most_common(20)),
                  'n_level2_dirs': len(lvl2), 'level2_sample': dict(list(lvl2.most_common())[:8]),
                  'first_entries': [i.filename for i in infos[:12]],
                  'range_bytes_fetched': f.fetched, 'n_range_requests': f.nreq}, indent=1))
