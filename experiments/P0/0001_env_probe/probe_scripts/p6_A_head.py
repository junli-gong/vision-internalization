"""RUN-0001 §6 group A: HTTP HEAD probe of many URLs (no body download).
<=16 concurrent, retries with exponential backoff on network errors / 429 / 5xx.
usage: python p6_A_head.py <urls.txt> <out_prefix> [concurrency<=16]
writes <out_prefix>.tsv (url, status, content_length, accept_ranges, content_type, final_url, error)
and <out_prefix>_summary.json (status histogram, failures, byte totals/mean/median)."""
import sys, json, time, statistics, collections, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor

urls = [l.strip() for l in open(sys.argv[1]) if l.strip()]
prefix = sys.argv[2]
conc = min(int(sys.argv[3]) if len(sys.argv) > 3 else 16, 16)
UA = {"User-Agent": "Mozilla/5.0 (research data-availability probe; HEAD only)"}


class NoRedirect(urllib.request.HTTPRedirectHandler):
    pass  # default: follow redirects (HEAD preserved by urllib for 301/302/303/307/308)


def head(url):
    err, last = None, None
    for i in range(4):
        try:
            req = urllib.request.Request(url, method="HEAD", headers=UA)
            with urllib.request.urlopen(req, timeout=30) as r:
                h = r.headers
                return (url, r.status, h.get("Content-Length"), h.get("Accept-Ranges"), h.get("Content-Type"), r.geturl(), "")
        except urllib.error.HTTPError as e:
            last = (url, e.code, e.headers.get("Content-Length") if e.headers else None, None,
                    e.headers.get("Content-Type") if e.headers else None, url, f"HTTPError {e.code}")
            if e.code in (429, 500, 502, 503, 504) and i < 3:
                time.sleep(2 ** i); continue
            return last
        except Exception as e:
            err = f"{type(e).__name__}: {str(e)[:150]}"
            last = (url, None, None, None, None, url, err)
            if i < 3:
                time.sleep(2 ** i); continue
    return last


t0 = time.time()
with ThreadPoolExecutor(conc) as ex:
    res = list(ex.map(head, urls))
dt = time.time() - t0
with open(prefix + ".tsv", "w") as f:
    f.write("url\tstatus\tcontent_length\taccept_ranges\tcontent_type\tfinal_url\terror\n")
    for r in res:
        f.write("\t".join("" if x is None else str(x).replace("\t", " ") for x in r) + "\n")
ok = [r for r in res if r[1] == 200]
sizes = [int(r[2]) for r in ok if r[2] and r[2].isdigit()]
summ = {
    "n_urls": len(urls), "concurrency": conc, "elapsed_s": round(dt, 1),
    "status_hist": dict(collections.Counter(str(r[1]) for r in res)),
    "n_ok_200": len(ok), "n_fail": len(res) - len(ok),
    "failure_ratio": round((len(res) - len(ok)) / max(1, len(res)), 4),
    "errors_hist": dict(collections.Counter(r[6] for r in res if r[6]).most_common(20)),
    "content_type_hist": dict(collections.Counter(str(r[4]) for r in ok).most_common(10)),
    "accept_ranges_hist": dict(collections.Counter(str(r[3]) for r in ok)),
    "redirected": sum(1 for r in ok if r[5] != r[0]),
    "n_with_length": len(sizes),
    "total_bytes": sum(sizes),
    "mean_bytes": round(statistics.mean(sizes), 1) if sizes else None,
    "median_bytes": statistics.median(sizes) if sizes else None,
    "min_bytes": min(sizes) if sizes else None, "max_bytes": max(sizes) if sizes else None,
    "failed_examples": [r for r in res if r[1] != 200][:20],
}
json.dump(summ, open(prefix + "_summary.json", "w"), indent=1)
print(json.dumps({k: v for k, v in summ.items() if k != "failed_examples"}, indent=1))
