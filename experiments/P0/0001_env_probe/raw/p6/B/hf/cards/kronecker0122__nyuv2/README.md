---
license: other
---

# NYUv2

Archive of the local `NYUv2` directory.

## Contents

- `nyuv2.tar.zst.part-00` (2,963,987,636 bytes)

## Restore

```bash
cat nyuv2.tar.zst.part-* | tar --zstd -xf -
```

The extracted top-level directory is `NYUv2/`, containing `nyu_depth_v2_labeled.mat`.

Archive format: `tar` compressed with `zstd`; split size: 5 GiB (this upload required one part).
