"""RUN-0001 §6 group B: HEAD a list of URLs (follow redirects), print JSON lines. usage: p6_B_head.py <urls.txt|-> [out.jsonl]"""
import sys, json, time, httpx, concurrent.futures as cf
urls = [l.strip() for l in (open(sys.argv[1]) if sys.argv[1] != '-' else sys.stdin) if l.strip() and not l.startswith('#')]
def head(u):
    for attempt in range(4):
        try:
            with httpx.Client(follow_redirects=True, timeout=60, headers={'User-Agent': 'Mozilla/5.0 (research metadata probe)'}) as c:
                r = c.head(u)
                if r.status_code in (403, 405) or ('content-length' not in r.headers and r.status_code == 200):
                    # some servers reject HEAD; use a 1-byte range GET to learn total size
                    r2 = c.get(u, headers={'Range': 'bytes=0-0'})
                    cr = r2.headers.get('content-range')
                    return {'url': u, 'final_url': str(r2.url), 'status': r2.status_code, 'method': 'GET range 0-0',
                            'content_range': cr, 'bytes': int(cr.split('/')[-1]) if cr and cr.split('/')[-1].isdigit() else None,
                            'content_type': r2.headers.get('content-type'), 'last_modified': r2.headers.get('last-modified')}
                cl = r.headers.get('content-length')
                return {'url': u, 'final_url': str(r.url), 'status': r.status_code, 'method': 'HEAD', 'bytes': int(cl) if cl else None,
                        'content_type': r.headers.get('content-type'), 'last_modified': r.headers.get('last-modified'),
                        'accept_ranges': r.headers.get('accept-ranges')}
        except Exception as e:
            err = repr(e); time.sleep(2 ** attempt)
    return {'url': u, 'error': err}
with cf.ThreadPoolExecutor(12) as ex:
    res = list(ex.map(head, urls))
out = open(sys.argv[2], 'w') if len(sys.argv) > 2 else None
for r in res:
    s = json.dumps(r); print(s)
    if out: out.write(s + '\n')
