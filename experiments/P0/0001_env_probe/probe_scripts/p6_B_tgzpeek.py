"""RUN-0001 §6 group B: range-read the first K bytes of a remote .tar.gz, stream-decompress in memory, list the tar member
headers that fit (names/sizes only). usage: p6_B_tgzpeek.py <url> <K_bytes>"""
import sys, zlib, tarfile, json, httpx
url, K = sys.argv[1], int(sys.argv[2])
d = httpx.Client(follow_redirects=True, timeout=180).get(url, headers={'Range': f'bytes=0-{K-1}'}).content
raw = zlib.decompressobj(16 + zlib.MAX_WBITS).decompress(d)
off, mem = 0, []
while off + 512 <= len(raw):
    try: ti = tarfile.TarInfo.frombuf(raw[off:off+512], 'utf-8', 'surrogateescape')
    except Exception as e: break
    if not ti.name: break
    mem.append({'name': ti.name, 'size': ti.size, 'type': ti.type.decode() if isinstance(ti.type, bytes) else ti.type})
    off += 512 + ((ti.size + 511) // 512) * 512
print(json.dumps({'url': url, 'range_bytes': len(d), 'decompressed_bytes': len(raw), 'members_seen': mem[:60], 'n_seen': len(mem)}, indent=1))
