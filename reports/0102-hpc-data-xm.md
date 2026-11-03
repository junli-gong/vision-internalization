# RUN-0102 报告：hpcmgt（网络与 CPU），X-M 的数据落地——COCO 2017、LVIS v1 val、ReasonSeg val（D165；D166、D169、D172）

任务书 `prompts/server_hpc/0102-hpc-data-xm.md`。不进 srun，不碰 GPU 节点。数据只在 HPC 的 `$VI/data/`，不进 git（D28-9）；本报告只录清单。只贴数字，不解读。

## 经过

- **SSH 中断**：任务发出后，Mac→a100 的跳板连不上（`Connection timed out during banner exchange`／`Connection closed by UNKNOWN port 65535`，11-03 02:17Z 起每 5 分钟重试 7 次都失败）。用户修好路由（Tailscale → 公网 → 内网，按 SSH 横幅判断）后恢复；HPC 本身一直在跑。
- 03:20Z 前 squeue：157296 RUNNING 79-14:45:51 hpcgpu06；157297 RUNNING 81-12:18:00 hpcgpu07。Lustre 余量 7,272 GB（只记录，D95）。
- 从 hpcmgt 的可达性（HEAD，25 s 超时）：`images.cocodataset.org` 200（0.92 s）；`dl.fbaipublicfiles.com` 200（0.66 s）；`drive.google.com`、`www.google.com` 都超时、无响应。
- **COCO 直连太慢**：`wget -c` 官方地址约 33 KB/s（`annotations_trainval2017.zip` 66 s 内 2.16 MB）；照此 annotations 约 2 小时、val2017 约 7 小时、train2017 约 6.8 天。03:44Z 停掉自己的直连 wget（部分文件 47,902,560 字节移到 `$VI/tmp`），按 **D169** 改用 hf-mirror：只有字节数等于官方 Content-Length 且 MD5 等于官方 MD5 才接受。
  - 官方字节数取自 `images.cocodataset.org` 的 HEAD；annotations 的官方 MD5 即其单段 S3 ETag `f4bbac642086de4f52a3fdda2de5fa2c`；val2017、train2017 的 ETag 是分段的，用公开的官方 MD5（`442b8da7…`、`cced6f7f…`）。
  - 镜像仓库 `pcuenq/coco-2017-mirror@a4cd8b69bd45a35e9a0a9c8692f3a7f4f23321fe`（三个 zip 的大小都等于官方）。另有 danasone、g292725651、chen132300、BoyangZ、manh6054、nyanko7/coco-hosted 的 train2017.zip 大小也相同，未用。
- **ReasonSeg**：LISA 官方 README 只给一个来源，Google Drive 文件夹 `125mewyg5Ao6tZ3ZdJ-1-E3n04LGVELqy`（内有 explanatory/train.json、test.zip、train.zip、val.zip）。hpcmgt 连不上 Drive。
  - 按 **D166** 在 Mac 的会话临时目录用 gdown 下官方 val.zip（id `167eq2ZW7rdbJ1OeB91fM7rFe0CTrLBM9`）：277,101,832 字节，sha256 `9248ec018f5896b91c09089d9c7cf3229102063985e6af6884a21bfea70d199c`；经跳板转到 HPC 只有约 9–10 KB/s（约 8 小时）。
  - 在 HF 上找到 `Myszz/ReasonSeg` 的 val.zip，LFS sha256 与上面的官方文件相同。按 **D172** 从 hf-mirror 下（钉住 `9a7e37ea86e5adf8d3c69665710d731414f26846`），54 s，字节数与 sha256 都与官方相同，接受；随即停掉转运（转了 17,727,488 字节的部分文件移到 `$VI/tmp`），删除前再核对 Mac 副本的 sha256（`9248ec01…`，相同）后删除。

## 下载与核对（`$VI/logs/mirror0102.txt` 原文摘录）

```
GET annotations_trainval2017.zip rc=0 bytes=252907541 (official 252907541) md5=f4bbac642086de4f52a3fdda2de5fa2c (official f4bbac642086de4f52a3fdda2de5fa2c) seconds=67 MB/s=3.8 2026-11-03T03:45:36Z
ACCEPT annotations_trainval2017.zip
GET val2017.zip rc=0 bytes=815585330 (official 815585330) md5=442b8da7639aecaf257c1dceb8ba8c80 (official 442b8da7639aecaf257c1dceb8ba8c80) seconds=205 MB/s=4.0 2026-11-03T03:49:02Z
ACCEPT val2017.zip
unzip annotations rc=0
unzip val2017 rc=0
val2017 images: 5000
GET train2017.zip rc=0 bytes=19336861798 (official 19336861798) md5=cced6f7f71b7629ddf16f17bbcfab6b2 (official cced6f7f71b7629ddf16f17bbcfab6b2) seconds=8366 MB/s=2.3 2026-11-03T06:09:29Z
ACCEPT train2017.zip
unzip train2017 rc=0
train2017 images: 118287
```

- LVIS：官方地址直连，64,026,968 字节，13 s；解压 rc=0。
- ReasonSeg：解压 rc=0，`val/` 下 200 个 .jpg、200 个 .json（zip 的 401 条含目录）。
- 张数：train2017 118,287、val2017 5,000，与预期相同。
- 结束时 Lustre 余量 7,031 GB。后 squeue（06:32:25Z）：157296 RUNNING 79-17:58:11 hpcgpu06；157297 RUNNING 81-15:30:20 hpcgpu07。

## 清单原文

### `$VI/data/coco2017/MANIFEST.json`

```json
{
 "dataset": "COCO 2017",
 "written_utc": "2026-11-03T06:30:21.653230Z",
 "license_note": "non-commercial research use, not redistributed (D28-9)",
 "zips": [
  {
   "path": "coco2017/zips/annotations_trainval2017.zip",
   "bytes": 252907541,
   "md5": "f4bbac642086de4f52a3fdda2de5fa2c",
   "sha256": "113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268",
   "source": "hf-mirror (D169)",
   "mirror_repo": "pcuenq/coco-2017-mirror",
   "mirror_revision": "a4cd8b69bd45a35e9a0a9c8692f3a7f4f23321fe",
   "url": "https://hf-mirror.com/datasets/pcuenq/coco-2017-mirror/resolve/a4cd8b69bd45a35e9a0a9c8692f3a7f4f23321fe/annotations_trainval2017.zip",
   "official_url": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
   "official_bytes": 252907541,
   "official_md5": "f4bbac642086de4f52a3fdda2de5fa2c",
   "official_bytes_match": true,
   "official_md5_match": true
  },
  {
   "path": "coco2017/zips/train2017.zip",
   "bytes": 19336861798,
   "md5": "cced6f7f71b7629ddf16f17bbcfab6b2",
   "sha256": "69a8bb58ea5f8f99d24875f21416de2e9ded3178e903f1f7603e283b9e06d929",
   "source": "hf-mirror (D169)",
   "mirror_repo": "pcuenq/coco-2017-mirror",
   "mirror_revision": "a4cd8b69bd45a35e9a0a9c8692f3a7f4f23321fe",
   "url": "https://hf-mirror.com/datasets/pcuenq/coco-2017-mirror/resolve/a4cd8b69bd45a35e9a0a9c8692f3a7f4f23321fe/train2017.zip",
   "official_url": "http://images.cocodataset.org/zips/train2017.zip",
   "official_bytes": 19336861798,
   "official_md5": "cced6f7f71b7629ddf16f17bbcfab6b2",
   "official_bytes_match": true,
   "official_md5_match": true
  },
  {
   "path": "coco2017/zips/val2017.zip",
   "bytes": 815585330,
   "md5": "442b8da7639aecaf257c1dceb8ba8c80",
   "sha256": "4f7e2ccb2866ec5041993c9cf2a952bbed69647b115d0f74da7ce8f4bef82f05",
   "source": "hf-mirror (D169)",
   "mirror_repo": "pcuenq/coco-2017-mirror",
   "mirror_revision": "a4cd8b69bd45a35e9a0a9c8692f3a7f4f23321fe",
   "url": "https://hf-mirror.com/datasets/pcuenq/coco-2017-mirror/resolve/a4cd8b69bd45a35e9a0a9c8692f3a7f4f23321fe/val2017.zip",
   "official_url": "http://images.cocodataset.org/zips/val2017.zip",
   "official_bytes": 815585330,
   "official_md5": "442b8da7639aecaf257c1dceb8ba8c80",
   "official_bytes_match": true,
   "official_md5_match": true
  }
 ],
 "annotations": [
  {
   "path": "coco2017/annotations/captions_train2017.json",
   "bytes": 91865115,
   "sha256": "4b62086319480e0739ef390d04084515defb9c213ff13605a036061e33314317"
  },
  {
   "path": "coco2017/annotations/captions_val2017.json",
   "bytes": 3872473,
   "sha256": "afe3b30e403dd7f228e2373023abbd60042a6e10ec6874d3652df034d289ebb9"
  },
  {
   "path": "coco2017/annotations/instances_train2017.json",
   "bytes": 469785474,
   "sha256": "610fce4944abdeb15354cc765333805529359d12d88f2f711393ca586901d01d"
  },
  {
   "path": "coco2017/annotations/instances_val2017.json",
   "bytes": 19987840,
   "sha256": "e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f"
  },
  {
   "path": "coco2017/annotations/person_keypoints_train2017.json",
   "bytes": 238884731,
   "sha256": "7fc1549d934547c470384d8a207c38707ca33fc016d00e33b795e408603af83e"
  },
  {
   "path": "coco2017/annotations/person_keypoints_val2017.json",
   "bytes": 10020657,
   "sha256": "788e2dae83c86bd547be7fab269d6399df5671063d29a61360cdb2cc370d2b14"
  }
 ],
 "image_counts": {
  "train2017": 118287,
  "val2017": 5000
 },
 "expected_counts": {
  "train2017": 118287,
  "val2017": 5000
 }
}
```

### `$VI/data/lvis/MANIFEST.json`

```json
{
 "dataset": "LVIS v1 val annotations",
 "source": "https://dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip (official, direct)",
 "images": "COCO 2017 (not downloaded separately)",
 "written_utc": "2026-11-03T06:30:21.653230Z",
 "files": [
  {
   "path": "lvis/lvis_v1_val.json",
   "bytes": 201235232,
   "sha256": "2bf946b92c3037f53c172d80017f5b74ea035f00a21b20e0766b3b638b2363f9"
  },
  {
   "path": "lvis/lvis_v1_val.json.zip",
   "bytes": 64026968,
   "sha256": "5cae9a3c79aadb667550c2b5dcf7f4d86e059a41ec91ef690225b667e28e9ba5"
  }
 ]
}
```

### `$VI/data/reasonseg/MANIFEST.json`（`files` 一项在 HPC 上是 400 条逐文件 sha256，这里只写条数）

```json
{
 "dataset": "ReasonSeg val (LISA)",
 "written_utc": "2026-11-03T03:56:59.070539Z",
 "official_source": "LISA README Google Drive folder https://drive.google.com/drive/folders/125mewyg5Ao6tZ3ZdJ-1-E3n04LGVELqy, file val.zip id 167eq2ZW7rdbJ1OeB91fM7rFe0CTrLBM9",
 "delivery": "bytes delivered via HF mirror Myszz/ReasonSeg@9a7e37ea86e5adf8d3c69665710d731414f26846 on hf-mirror.com (D172); drive.google.com unreachable from hpcmgt; a Mac relay of the official Drive file (D166) was started and stopped once the mirror copy was verified",
 "url": "https://hf-mirror.com/datasets/Myszz/ReasonSeg/resolve/9a7e37ea86e5adf8d3c69665710d731414f26846/val.zip",
 "zip": {
  "path": "reasonseg/val.zip",
  "bytes": 277101832,
  "sha256": "9248ec018f5896b91c09089d9c7cf3229102063985e6af6884a21bfea70d199c"
 },
 "official_sha256_downloaded_on_mac": "9248ec018f5896b91c09089d9c7cf3229102063985e6af6884a21bfea70d199c",
 "official_bytes": 277101832,
 "counts": {
  "jpg": 200,
  "json": 200
 },
 "files": "400 entries (path, bytes, sha256) in the HPC copy of this file",
 "license_note": "non-commercial research use, not redistributed (D28-9)",
 "sha256_identical_to_official": true
}
```

## 之后

- RUN-0103 由 hpcmgt 上的排队脚本接着跑（06:32Z 起：train2017 通过后建 wt0103）。
