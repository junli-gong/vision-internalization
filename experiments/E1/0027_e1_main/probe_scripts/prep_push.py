#!/usr/bin/env python
"""RUN-0027 §5：一个模型跑完后准备入库的文件清单。
- 单个文件 > 20 MB（20,000,000 B）：在同目录写 <文件>.gz（gzip -9，确定性：mtime=0、不存文件名），并在同目录的 LARGE_FILES.sha256
  里追加一行 "<原文件 sha256>  <原文件字节数>  <文件名>  <gz 的 sha256>  <gz 字节数>"（已有同名行则替换）。原文件留在服务器上，不入库。
- 输出：要 git add 的路径（每行一个），即该模型目录下除了大文件原件以外的全部文件（含 .gz 与 LARGE_FILES.sha256）。
usage (在 $REPO 下): python prep_push.py <model_dir> <add_list_out>
"""
import gzip, hashlib, os, sys
LIMIT = 20_000_000
root, out = sys.argv[1], sys.argv[2]


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


add, large = [], []
for d, _, files in os.walk(root):
    for fn in sorted(files):
        p = os.path.join(d, fn)
        if fn.endswith('.gz') or fn == 'LARGE_FILES.sha256':
            continue
        if os.path.getsize(p) > LIMIT:
            gz = p + '.gz'
            with open(p, 'rb') as fi, open(gz, 'wb') as fo:
                with gzip.GzipFile(filename='', mode='wb', fileobj=fo, compresslevel=9, mtime=0) as g:
                    for b in iter(lambda: fi.read(1 << 24), b''):
                        g.write(b)
            line = f"{sha(p)}  {os.path.getsize(p)}  {fn}  {sha(gz)}  {os.path.getsize(gz)}\n"
            lf = os.path.join(d, 'LARGE_FILES.sha256')
            old = [l for l in open(lf)] if os.path.exists(lf) else []
            old = [l for l in old if l.split()[2] != fn] + [line]
            open(lf, 'w').writelines(old)
            large.append((p, os.path.getsize(p), os.path.getsize(gz)))
        else:
            add.append(p)
for d, _, files in os.walk(root):
    for fn in files:
        if fn.endswith('.gz') or fn == 'LARGE_FILES.sha256':
            add.append(os.path.join(d, fn))
too_big = [p for p in add if os.path.getsize(p) > LIMIT]
open(out, 'w').write('\n'.join(sorted(set(add))) + '\n')
for p, a, b in large:
    print(f"gzip {p}: {a} -> {b} B")
print(f"{len(set(add))} files to add; files still > 20 MB after gzip: {too_big}")
