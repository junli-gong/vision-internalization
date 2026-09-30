"""Sum download sizes of a pip --report via HTTP HEAD (no download). Also list torch-related requirements."""
import json, sys, concurrent.futures as cf, urllib.request
rep = json.load(open(sys.argv[1]))
ins = rep['install']
def head(u):
    for _ in range(3):
        try:
            req = urllib.request.Request(u, method='HEAD')
            with urllib.request.urlopen(req, timeout=30) as r:
                return int(r.headers.get('Content-Length') or -1)
        except Exception as e:
            err = e
    return -1
urls = [i['download_info']['url'] for i in ins]
with cf.ThreadPoolExecutor(16) as ex:
    sizes = list(ex.map(head, urls))
rows = sorted(zip(sizes, [i['metadata']['name'] + '==' + i['metadata']['version'] for i in ins], urls), reverse=True)
tot = sum(s for s in sizes if s > 0)
print(f'packages={len(ins)} total_download_bytes={tot} ({tot/1e9:.3f} GB) unknown={sum(1 for s in sizes if s<0)}')
for s, n, u in rows[:15]:
    print(f'  {s/1e6:9.1f} MB  {n}')
for u in [x for x in rows if x[0] < 0]:
    print('  UNKNOWN', u[1], u[2][:120])
print('torch-related requires_dist:')
for i in ins:
    m = i['metadata']
    for rq in m.get('requires_dist') or []:
        if rq.split(';')[0].replace(' ', '').lower().startswith(('torch=', 'torch>', 'torch<', 'torch!', 'torch~', 'torch(', 'torch')) and 'torchvision' not in rq and 'torchaudio' not in rq and 'torch_' not in rq and 'torchcodec' not in rq:
            print('  ', m['name'], m['version'], '->', rq)
