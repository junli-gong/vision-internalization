# RUN-0047 — B200 归档从开发机传到 HPC 的 Lustre，并逐文件核对 sha256（完成）

服务器 CC（HPC），在用户的 Mac 上操作。2026-10-11 05:46–08:54 UTC。只贴数字与事实，不做解读。原始输出：`experiments/E3/0047_hpc_relay_b200_archive/raw/`。

## 0. 结果

- 16,163 个文件，即清单去掉 `logs/` 之后的全部，传到 `$VI/b200/`。
- `sha256sum -c` 结果：**OK 16,163、FAILED 0、缺失 0**（rc=0）。文件数与路径清单都对得上。
- 没有断开，没有补传；没有带进 `._*` 或 `.DS_Store`。
- 没有用 GPU，没有碰 RUN-0045 的驱动与推送循环。前后 squeue 两条都是 RUNNING。

## 1. squeue

```
## squeue 2026-10-11T05:46:53Z [0047 before]
157296 RUNNING 56-17:12:39 hpcgpu06
157297 RUNNING 58-14:44:48 hpcgpu07
## squeue 2026-10-11T08:53:43Z [0047 after]
157296 RUNNING 56-20:19:29 hpcgpu06
157297 RUNNING 58-17:51:38 hpcgpu07
```

## 2. 源与清单（开发机，只读）

- 源目录 `~/Research/vision-internalization-archive/b200_20261011/`，没有符号链接，没有 `._*`。

  | 部分 | 文件数 | 字节 |
  |---|---|---|
  | `ckpt/0036` | 189 | 1,767,969,217 |
  | `ckpt/0040` | 48 | 1,213,196,790 |
  | `data` | 15,848 | 3,130,969,434 |
  | `nltk_data` | 78 | 15,204,406 |

- `SHA256SUMS.b200` 共 18,042 行，去掉 `logs/` 后 16,163 行，等于上面四部分之和。
- 源目录里还有一个 `sha.err`（0 B），不在清单里，没有传。`logs/` 也没有传。

## 3. 中转（`probe_scripts/relay47.sh`）

- **方式：** 源目录 → `tar -cf - -T <清单>`（`COPYFILE_DISABLE=1`）→ `ssh a100` → `ssh gpu` → `tar -xf - -C $VI/b200`。lab5 上不落地，没有动 B200。
- **三条流并行**，文件清单按 `SHA256SUMS.b200` 去掉 `logs/` 后的路径划分（`raw/s_*.txt`）：

  | 流 | 内容 | 文件数 | 字节（tee 计数） | 用时 | 平均速度 | rc |
  |---|---|---|---|---|---|---|
  | s_ckpt | ckpt/0036 + ckpt/0040 | 237 | 2,981,601,280 | 10,875 s | 0.274 MB/s | 0 0 0 |
  | s_data_a | data 单号行 | 7,924 | 1,570,396,160 | 5,866 s | 0.268 MB/s | 0 0 0 |
  | s_data_b | data 双号行 + nltk_data + SHA256SUMS.b200 + README.md | 8,004 | 1,606,400,000 | 7,376 s | 0.218 MB/s | 0 0 0 |
  | 合计 | | 16,165 | 6,158,397,440 | 10,875 s（墙钟） | 0.566 MB/s | |

  - 字节数是 tar 流的大小，含 tar 头。
- **错误输出：** 三条流的 err 里每个文件各有一行 `tar: Ignoring unknown extended header keyword 'LIBARCHIVE.xattr.com.apple.provenance'`（GNU tar 忽略 macOS 的扩展属性），没有别的内容（`raw/*.err.summary.txt`）。
- **没有断开，没有补传。**

## 4. 核对（hpcmgt，`probe_scripts/verify47.sh`，`raw/verify47.out`）

- **删掉的 `._*`、`.DS_Store`：** 0 个。
- **`sha256sum -c --quiet`**（`SHA256SUMS.b200` 去掉 `logs/`，16,163 行）：rc=0，258 s；FAILED 0，No such file 0。
- **文件数：** `$VI/b200` 下的普通文件 16,165 个，等于 16,163 + `SHA256SUMS.b200` + `README.md`。
- **路径清单：** 盘上的路径集合（不含那 2 个）与清单路径逐行相同，diff 0 行。
  - 第一次比对时我用 `awk '{print $2}'` 取路径，把带空格的 68 个路径截断了，显示 156 行差异；改用 `cut -c67-` 重比是 0 行。这是比对写法的问题，不是数据。
- **大小：** `du -s --apparent-size $VI/b200` = 6,131,510,465 B。

## 5. 份额

- `lfs quota -u chenyuma`（KB）：08:53Z 是 7,115,223,232，比 RUN-0044 开始时（7,059,403,936）多 **55.8 GB**，上限 80 GB。
  - 其中有这次的 6.13 GB，以及 RUN-0045 截至此时的检查点。
- RUN-0045 驱动的份额命令按"固定 34 GB + `$VI/ckpt`"算，不含 `$VI/b200`，会少算约 6.1 GB（任务书 §1 已注明）。
