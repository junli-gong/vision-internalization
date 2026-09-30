"""RUN-0001 §6 group B: list variables of a remote MATLAB v5 .mat via small HTTP Range reads (element tags + array headers only).
usage: p6_B_mat5vars.py <url>"""
import sys, struct, zlib, json, httpx
url = sys.argv[1]; c = httpx.Client(follow_redirects=True, timeout=120); fetched = 0
MI = {1: 'int8', 2: 'uint8', 3: 'int16', 4: 'uint16', 5: 'int32', 6: 'uint32', 7: 'single', 9: 'double', 12: 'int64', 13: 'uint64', 14: 'miMATRIX', 15: 'miCOMPRESSED'}
MX = {1: 'cell', 2: 'struct', 3: 'object', 4: 'char', 5: 'sparse', 6: 'double', 7: 'single', 8: 'int8', 9: 'uint8', 10: 'int16', 11: 'uint16', 12: 'int32', 13: 'uint32', 14: 'int64', 15: 'uint64'}
def rng(a, b):
    global fetched
    d = c.get(url, headers={'Range': f'bytes={a}-{b}'}).content; fetched += len(d); return d
size = int(c.head(url).headers['content-length'])
hdr = rng(0, 127); endian = hdr[126:128]
off = 128; out = []
def parse_matrix(buf):
    # buf starts at array-flags subelement
    p = 0; res = {}
    def sub():
        nonlocal p
        t, n = struct.unpack('<II', buf[p:p+8])
        if t >> 16:  # small element format
            n = t >> 16; t = t & 0xffff; data = buf[p+4:p+4+n]; p += 8; return t, data
        data = buf[p+8:p+8+n]; p += 8 + ((n + 7) // 8) * 8; return t, data
    _, flags = sub(); cls = flags[0]; res['class'] = MX.get(cls, cls); res['complex'] = bool(flags[1] & 8)
    _, dims = sub(); res['dims'] = list(struct.unpack('<' + 'i' * (len(dims) // 4), dims))
    _, name = sub(); res['name'] = name.decode()
    try:
        t, _ = struct.unpack('<II', buf[p:p+8]); res['data_type'] = MI.get(t & 0xffff, t)
    except Exception: pass
    return res
while off < size:
    t, n = struct.unpack('<II', rng(off, off + 7))
    if t == 14:
        rec = parse_matrix(rng(off + 8, off + 8 + 255)); rec['stored'] = 'miMATRIX'; rec['nbytes'] = n
    elif t == 15:
        raw = zlib.decompressobj().decompress(rng(off + 8, off + 8 + min(n, 4095)))
        t2, n2 = struct.unpack('<II', raw[:8]); rec = parse_matrix(raw[8:]); rec['stored'] = 'miCOMPRESSED'; rec['nbytes'] = n
    else:
        rec = {'type': t, 'nbytes': n}
    out.append(rec); off += 8 + n + ((8 - n % 8) % 8 if t != 15 else 0)
print(json.dumps({'url': url, 'size': size, 'header': hdr[:116].decode('latin1').strip(), 'vars': out, 'range_bytes_fetched': fetched}, indent=1))
