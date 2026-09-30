"""RUN-0001 §6 group A: read only the tail (End Of Central Directory, incl. zip64) of remote zips via one Range
request, to get the total number of entries and central-directory size without listing the zip.
usage: python p6_A_zip_eocd.py <url> [<url> ...]"""
import sys, struct, urllib.request, json
for url in sys.argv[1:]:
    with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as r:
        size = int(r.headers["Content-Length"])
    n = 1024
    with urllib.request.urlopen(urllib.request.Request(url, headers={"Range": f"bytes={size-n}-{size-1}"}), timeout=60) as r:
        tail = r.read()
    rec = {"url": url, "zip_size": size, "tail_bytes_fetched": len(tail)}
    i = tail.rfind(b"PK\x05\x06")
    (_, disk, cd_disk, n_disk, n_total, cd_size, cd_off, clen) = struct.unpack("<IHHHHIIH", tail[i:i + 22])
    rec.update({"eocd_entries_total": n_total, "eocd_cd_size": cd_size, "eocd_cd_offset": cd_off})
    j = tail.rfind(b"PK\x06\x06")
    if j >= 0:
        (_, _, _, _, _, _, n_disk64, n_total64, cd_size64, cd_off64) = struct.unpack("<IQHHIIQQQQ", tail[j:j + 56])
        rec.update({"zip64_entries_total": n_total64, "zip64_cd_size": cd_size64, "zip64_cd_offset": cd_off64})
    print(json.dumps(rec))
