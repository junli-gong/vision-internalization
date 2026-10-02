#!/usr/bin/env python
"""RUN-0015 §2.4：把 LISA-7B-v1 与 CLIP 视觉塔下载到内存盘（HF_HUB_CACHE），固定 revision；CLIP 只取 safetensors 一种格式。
用项目 venv（huggingface_hub 1.33）运行；之后 LISA 的 venv 只按本地快照路径加载，不再访问 Hub。下载后逐文件核对字节数与哈希。"""
import hashlib, json, os, sys, time
from huggingface_hub import HfApi, snapshot_download
SPECS = [("xinlai/LISA-7B-v1", "43c754eef75871fcc11c84d3930a402f7b0a754f", None),
         ("openai/clip-vit-large-patch14", "32bd64288804d66eefd0ccbe215aa642df71cc41", ["*.json", "*.txt", "model.safetensors"])]
api = HfApi(); out = {}
for repo, rev, pats in SPECS:
    t0 = time.time(); snap = snapshot_download(repo, revision=rev, allow_patterns=pats); dt = time.time() - t0
    mi = api.model_info(repo, revision=rev, files_metadata=True); files = []
    for s in mi.siblings:
        p = os.path.join(snap, s.rfilename)
        if not os.path.exists(p):
            continue
        if s.lfs is not None:
            h = hashlib.sha256()
            with open(p, 'rb') as f:
                for b in iter(lambda: f.read(1 << 24), b''):
                    h.update(b)
            want = (s.lfs if isinstance(s.lfs, dict) else s.lfs.__dict__).get('sha256'); got = h.hexdigest(); kind = 'lfs_sha256'
        else:
            data = open(p, 'rb').read(); got = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest(); want = s.blob_id; kind = 'git_blob_sha1'
        files.append({"path": s.rfilename, "bytes": os.path.getsize(p), "expected_bytes": s.size, "hash_kind": kind, "hash": got, "ok": got == want and os.path.getsize(p) == s.size})
    out[repo] = {"revision": rev, "allow_patterns": pats, "snapshot": snap, "download_s": round(dt, 1), "n_files": len(files),
                 "n_ok": sum(f["ok"] for f in files), "bytes": sum(f["bytes"] for f in files), "files": files}
    print(repo, rev[:12], f"{len(files)} files, ok {out[repo]['n_ok']}, {out[repo]['bytes']/1e9:.3f} GB, {dt:.1f}s -> {snap}", flush=True)
json.dump(out, open(sys.argv[1], 'w'), indent=1)
sys.exit(0 if all(v["n_ok"] == v["n_files"] for v in out.values()) else 1)
