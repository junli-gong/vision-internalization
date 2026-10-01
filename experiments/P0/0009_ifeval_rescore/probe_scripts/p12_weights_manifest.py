#!/usr/bin/env python
"""RUN-0009 §4：8 个注册模型在固定 revision 下的文件清单（路径、字节数、哈希）。只读 HF API，不下载任何文件。

哈希的种类（每个文件的 hash_kind 字段注明）：
- LFS 文件（权重 safetensors、大 tokenizer 等）：lfs.sha256 = 文件内容的 sha256，可直接与本地文件的 sha256 比对；
- 非 LFS 小文件：blob_id = git blob 的 sha1，即 sha1(b"blob <字节数>\\0" + 内容)，比对时按这个公式算。
usage (在 $REPO 下): python experiments/P0/0009_ifeval_rescore/probe_scripts/p12_weights_manifest.py <out.json>
"""
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

from huggingface_hub import HfApi  # noqa: E402
import huggingface_hub  # noqa: E402

from src.adapters import registry  # noqa: E402

api = HfApi()
out = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "huggingface_hub": huggingface_hub.__version__,
       "hash_kinds": {"lfs_sha256": "sha256 of file content (compare with sha256sum of the local file)",
                      "git_blob_sha1": "sha1(b'blob ' + str(size).encode() + b'\\0' + content) (git blob id)"},
       "models": {}}
for key, e in registry.MODELS.items():
    mi = api.model_info(e.repo, revision=e.revision, files_metadata=True)
    files = []
    for s in sorted(mi.siblings, key=lambda x: x.rfilename):
        if s.lfs is not None:
            lfs = s.lfs if isinstance(s.lfs, dict) else s.lfs.__dict__
            files.append({"path": s.rfilename, "bytes": s.size, "hash_kind": "lfs_sha256", "hash": lfs.get("sha256"), "lfs_size": lfs.get("size")})
        else:
            files.append({"path": s.rfilename, "bytes": s.size, "hash_kind": "git_blob_sha1", "hash": s.blob_id})
    tot = sum(f["bytes"] or 0 for f in files)
    st = sum(f["bytes"] or 0 for f in files if f["path"].endswith(".safetensors"))
    out["models"][key] = {"repo": e.repo, "revision": e.revision, "family": e.family, "api_sha": mi.sha, "revision_matches": mi.sha == e.revision,
                          "n_files": len(files), "total_bytes": tot, "safetensors_bytes": st, "files": files}
    print(f"{key:20s} {e.repo:32s} {e.revision[:12]} match={mi.sha == e.revision} files={len(files)} total={tot/1e9:.3f} GB safetensors={st/1e9:.3f} GB")
Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
