#!/usr/bin/env python
"""按 weights_manifest.json 恢复并核对一个模型的权重（重启内存盘丢失后用）。
usage (在 $REPO 下，先 source env.sh 让 HF_HUB_CACHE 指向 /dev/shm/visint/hf-hub):
  python experiments/P0/0009_ifeval_rescore/probe_scripts/p12_verify_weights.py <模型键> [--download]
--download：先 snapshot_download(repo, revision=清单里的 revision)；不加则只核对本地已有的快照。
逐个文件核对字节数与哈希（LFS 文件比 sha256，小文件比 git blob sha1）；全部一致退出码 0，否则 1。
"""
import hashlib
import json
import os
import sys
from pathlib import Path

MANIFEST = Path(__file__).resolve().parents[1] / "weights_manifest.json"


def file_hash(path, kind):
    if kind == "lfs_sha256":
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for b in iter(lambda: f.read(1 << 24), b""):
                h.update(b)
        return h.hexdigest()
    data = Path(path).read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def main():
    key = sys.argv[1]
    m = json.loads(MANIFEST.read_text())["models"][key]
    from huggingface_hub import snapshot_download
    snap = snapshot_download(m["repo"], revision=m["revision"], local_files_only="--download" not in sys.argv)
    bad = 0
    for f in m["files"]:
        p = os.path.join(snap, f["path"])
        if not os.path.exists(p):
            print("MISSING", f["path"]); bad += 1; continue
        size_ok = os.path.getsize(p) == f["bytes"]
        hash_ok = file_hash(p, f["hash_kind"]) == f["hash"]
        if not (size_ok and hash_ok):
            print("MISMATCH", f["path"], "size_ok", size_ok, "hash_ok", hash_ok); bad += 1
    print(f"{key} @ {m['revision']}: {len(m['files']) - bad}/{len(m['files'])} files verified ({snap})")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
