# RUN-0044 盘上文件重算 sha256，对照各源自己的 MANIFEST.json

| 源 | MANIFEST 文件数 | 盘上文件数（不含 MANIFEST.json） | sha256 与字节都相同 | 不同 | MANIFEST 有、盘上缺 | 盘上有、MANIFEST 没有 |
|---|---|---|---|---|---|---|
| refcoco_train | 24411 | 24411 | 24411 | 0 | 0 | 0 |
| nyu_depth_train | 45698 | 45698 | 45698 | 0 | 0 | 0 |
| nyu_normals | 2899 | 2899 | 2899 | 0 | 0 | 0 |
| ade20k | 44424 | 44424 | 44424 | 0 | 0 | 0 |

全部相同：True
