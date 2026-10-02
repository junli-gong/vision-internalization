#!/usr/bin/env python
"""按 B200 的权重清单（RUN-0009 weights_manifest.json）逐文件核对本机 HF 缓存里的权重。

用法：python verify_weights.py <manifest.json> <模型键> [<模型键> ...]
每个文件：safetensors 等 LFS 文件比 sha256，小文件比 git blob sha1；缺文件、字节数不同、哈希不同都算失败。
退出码 0 = 全部相同，1 = 有不同或缺失。
"""
import hashlib
import json
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


def git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    manifest = json.loads(Path(sys.argv[1]).read_text())
    bad = 0
    for key in sys.argv[2:]:
        m = manifest["models"][key]
        root = Path(snapshot_download(m["repo"], revision=m["revision"], local_files_only=True))
        ok = 0
        for f in m["files"]:
            p = root / f["path"]
            if not p.exists():
                print(f"MISSING {key} {f['path']}"); bad += 1; continue
            if p.stat().st_size != f["bytes"]:
                print(f"SIZE {key} {f['path']} {p.stat().st_size} != {f['bytes']}"); bad += 1; continue
            got = sha256(p) if f["hash_kind"] == "lfs_sha256" else git_blob_sha1(p)
            if got != f["hash"]:
                print(f"HASH {key} {f['path']} {got} != {f['hash']}"); bad += 1; continue
            ok += 1
        print(f"{key}: {ok}/{len(m['files'])} files identical to the B200 manifest", flush=True)
    print("ALL_OK" if bad == 0 else f"FAILURES {bad}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
