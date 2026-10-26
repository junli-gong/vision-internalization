# RUN-0072 报告：hpcgpu06，W2 的 test 主实验，9B 的全部设置（5 个 test 划分 × 1,000 句）

任务书 `prompts/server_hpc/0072-hpc-06-w2-test-9b.md` + 补充 1 `0072-addendum-1.md`（W2-c 补跑）。test 只报告、不判定。没有碰 07。只贴数字，不解读。

## 检查点恢复

### 起因：W2-c 的 adapter 被 shrink_large 压掉

- RUN-0069 的检查点在它自己的结果目录 `experiments/W2/0069_w2c_hpcgpu06/0069_train/w2c/qwen3.5-9b/seed0/ckpt/final/model` 下。
- 旧版 `eval_lib.sh` 的 `shrink_large` 在驱动收尾时把 `$OUT` 下大于 20 MB 的文件压成 `.gz` 并删掉原文件。时间线（10-25，UTC）：
  - 01:23:51Z RUN-0069 done；
  - 01:24:06Z `lora/adapter_model.safetensors.gz` 写出（320,666,444 字节），原文件不在了；01:24:09Z `lora/LARGE_FILES.sha256` 写出；
  - 01:24:11Z RUN-0069 的 push_out 一行。
- RUN-0069 自己的健康检查与 T 文本都在这之前用过检查点。
- `LARGE_FILES.sha256` 记的原文件：sha256 `8c94c11c5cbc81a08a515058714d53f41a722bea69e57ff16341f42838724f37`，346,302,176 字节，与 manifest 里 `lora/adapter_model.safetensors` 一行相同。

### 第一步：原地解压（管理员选 (a)，`$VI/tmp/restore0072.sh`，10-26 02:31Z）

```
## restore start 2026-10-26T02:31:23Z
-- before:
total 313172
drwxr-xr-x 2 chenyuma domain users      4096 2026-10-25 09:24:09.000000000 +0800 .
drwxr-xr-x 3 chenyuma domain users      4096 2026-10-25 04:36:10.000000000 +0800 ..
-rw-r--r-- 1 chenyuma domain users      1291 2026-10-25 04:36:09.000000000 +0800 adapter_config.json
-rw-r--r-- 1 chenyuma domain users 320666444 2026-10-25 09:24:06.000000000 +0800 adapter_model.safetensors.gz
-rw-r--r-- 1 chenyuma domain users       180 2026-10-25 09:24:09.000000000 +0800 LARGE_FILES.sha256
gz sha256: 17dc9064e92de62327f9d517e2a004fa359012dc6e9a23ce8fddc6de85b48591
8c94c11c5cbc81a08a515058714d53f41a722bea69e57ff16341f42838724f37  346302176  adapter_model.safetensors  17dc9064e92de62327f9d517e2a004fa359012dc6e9a23ce8fddc6de85b48591  320666444
gunzip+mv rc=0
restored: sha256 8c94c11c5cbc81a08a515058714d53f41a722bea69e57ff16341f42838724f37 bytes 346302176
restored file matches 8c94c11c… / 346,302,176
manifest.json sha256 c605408fbb92cceafbc59dff4d85c9c912b246d05dc5e29235db0cff389d93b7
manifest.json unchanged (c605408f…93b7)
lora/adapter_config.json  1291  9805c8ab8f5a155c0bf10589977fd8f1a25b28646f24fc08f3ceabbc15054de0  OK
lora/adapter_model.safetensors  346302176  8c94c11c5cbc81a08a515058714d53f41a722bea69e57ff16341f42838724f37  OK
token_rows.safetensors  168  3948c1fb6b04aa7c5092487e64293995629b2f0e7e0d62f8d276b61a5a2cfc94  OK
tokens.json  719  c227930849413911c7ba00ebc0ee0a7196e143b275b88915a31d02ce55ffbec1  OK
manifest files: 4, mismatches: 0
chmod a-w rc=0
-- after:
total 649044
drwxr-xr-x 2 chenyuma domain users      4096 2026-10-26 10:31:26.000000000 +0800 .
drwxr-xr-x 3 chenyuma domain users      4096 2026-10-25 04:36:10.000000000 +0800 ..
-rw-r--r-- 1 chenyuma domain users      1291 2026-10-25 04:36:09.000000000 +0800 adapter_config.json
-r--r--r-- 1 chenyuma domain users 346302176 2026-10-26 10:31:26.000000000 +0800 adapter_model.safetensors
-rw-r--r-- 1 chenyuma domain users 320666444 2026-10-25 09:24:06.000000000 +0800 adapter_model.safetensors.gz
-rw-r--r-- 1 chenyuma domain users       180 2026-10-25 09:24:09.000000000 +0800 LARGE_FILES.sha256
== restore OK 2026-10-26T02:31:28Z
```

### testA 的 w2c 失败（04:29Z）

- `steps.log`：`2026-10-26T04:29:12Z w2c_qwen3.5-9b_test_refcoco_unc_testA rc=2 wall_s=27`。
- `logs/w2c_qwen3.5-9b_test_refcoco_unc_testA.log` 原文：

```
失败：CheckpointError("目录里的文件与 manifest 不符：多出 ['lora/LARGE_FILES.sha256', 'lora/adapter_model.safetensors.gz']，缺少 []")（已写出 /home/mail/chenyuma/visint/repo/experiments/W2/0072_w2_test_hpcgpu06/qwen3.5-9b/errors/error_20261026T042911935199Z.json）
```

- 加载器 `src.adapters.checkpoint.read_manifest` 要求模型目录里除 manifest 外的文件与 manifest 登记的恰好一致；第一步按指示保留在 `lora/` 里的 `.gz` 与 `LARGE_FILES.sha256` 不在 manifest 里。

### 第二步：把多出的两个文件移出模型目录（管理员选 (a)，`$VI/tmp/archive0072.sh`，10-26 04:40Z）

```
## start 2026-10-26T04:40:51Z
-- before:
/home/mail/chenyuma/visint/repo/experiments/W2/0069_w2c_hpcgpu06/0069_train/w2c/qwen3.5-9b/seed0/ckpt/final/model:
total 24
drwxr-xr-x 3 chenyuma domain users 4096 2026-10-25 04:36:10.000000000 +0800 .
drwxr-xr-x 3 chenyuma domain users 4096 2026-10-25 04:36:09.000000000 +0800 ..
drwxr-xr-x 2 chenyuma domain users 4096 2026-10-26 10:31:26.000000000 +0800 lora
-rw-r--r-- 1 chenyuma domain users 1427 2026-10-25 04:36:10.000000000 +0800 manifest.json
-rw------- 1 chenyuma domain users  168 2026-10-25 04:36:09.000000000 +0800 token_rows.safetensors
-rw-r--r-- 1 chenyuma domain users  719 2026-10-25 04:36:09.000000000 +0800 tokens.json

/home/mail/chenyuma/visint/repo/experiments/W2/0069_w2c_hpcgpu06/0069_train/w2c/qwen3.5-9b/seed0/ckpt/final/model/lora:
total 651364
drwxr-xr-x 2 chenyuma domain users      4096 2026-10-26 10:31:26.000000000 +0800 .
drwxr-xr-x 3 chenyuma domain users      4096 2026-10-25 04:36:10.000000000 +0800 ..
-rw-r--r-- 1 chenyuma domain users      1291 2026-10-25 04:36:09.000000000 +0800 adapter_config.json
-r--r--r-- 1 chenyuma domain users 346302176 2026-10-26 10:31:26.000000000 +0800 adapter_model.safetensors
-rw-r--r-- 1 chenyuma domain users 320666444 2026-10-25 09:24:06.000000000 +0800 adapter_model.safetensors.gz
-rw-r--r-- 1 chenyuma domain users       180 2026-10-25 09:24:09.000000000 +0800 LARGE_FILES.sha256
before move: .gz 17dc9064e92de62327f9d517e2a004fa359012dc6e9a23ce8fddc6de85b48591, LARGE_FILES.sha256 ba8677c323dc2354d5aaa7b595ad687956f439920801adbe43f34454164d98f6
mkdir+mv rc=0
after move:  .gz 17dc9064e92de62327f9d517e2a004fa359012dc6e9a23ce8fddc6de85b48591, LARGE_FILES.sha256 ba8677c323dc2354d5aaa7b595ad687956f439920801adbe43f34454164d98f6
archived files unchanged
-- after:
/home/mail/chenyuma/visint/repo/experiments/W2/0069_w2c_hpcgpu06/0069_train/w2c/qwen3.5-9b/seed0/ckpt/final/lora_gz_archive:
total 313168
drwxr-xr-x 2 chenyuma domain users      4096 2026-10-26 12:40:52.000000000 +0800 .
drwxr-xr-x 4 chenyuma domain users      4096 2026-10-26 12:40:52.000000000 +0800 ..
-rw-r--r-- 1 chenyuma domain users 320666444 2026-10-25 09:24:06.000000000 +0800 adapter_model.safetensors.gz
-rw-r--r-- 1 chenyuma domain users       180 2026-10-25 09:24:09.000000000 +0800 LARGE_FILES.sha256

/home/mail/chenyuma/visint/repo/experiments/W2/0069_w2c_hpcgpu06/0069_train/w2c/qwen3.5-9b/seed0/ckpt/final/model:
total 24
drwxr-xr-x 3 chenyuma domain users 4096 2026-10-25 04:36:10.000000000 +0800 .
drwxr-xr-x 4 chenyuma domain users 4096 2026-10-26 12:40:52.000000000 +0800 ..
drwxr-xr-x 2 chenyuma domain users 4096 2026-10-26 12:40:52.000000000 +0800 lora
-rw-r--r-- 1 chenyuma domain users 1427 2026-10-25 04:36:10.000000000 +0800 manifest.json
-rw------- 1 chenyuma domain users  168 2026-10-25 04:36:09.000000000 +0800 token_rows.safetensors
-rw-r--r-- 1 chenyuma domain users  719 2026-10-25 04:36:09.000000000 +0800 tokens.json

/home/mail/chenyuma/visint/repo/experiments/W2/0069_w2c_hpcgpu06/0069_train/w2c/qwen3.5-9b/seed0/ckpt/final/model/lora:
total 338204
drwxr-xr-x 2 chenyuma domain users      4096 2026-10-26 12:40:52.000000000 +0800 .
drwxr-xr-x 3 chenyuma domain users      4096 2026-10-25 04:36:10.000000000 +0800 ..
-rw-r--r-- 1 chenyuma domain users      1291 2026-10-25 04:36:09.000000000 +0800 adapter_config.json
-r--r--r-- 1 chenyuma domain users 346302176 2026-10-26 10:31:26.000000000 +0800 adapter_model.safetensors
manifest.json sha256 c605408fbb92cceafbc59dff4d85c9c912b246d05dc5e29235db0cff389d93b7
manifest.json unchanged (c605408f…93b7)
files on disk: ['lora/adapter_config.json', 'lora/adapter_model.safetensors', 'manifest.json', 'token_rows.safetensors', 'tokens.json']
manifest + manifest.json: ['lora/adapter_config.json', 'lora/adapter_model.safetensors', 'manifest.json', 'token_rows.safetensors', 'tokens.json']
lora/adapter_config.json  1291  9805c8ab8f5a155c0bf10589977fd8f1a25b28646f24fc08f3ceabbc15054de0  OK
lora/adapter_model.safetensors  346302176  8c94c11c5cbc81a08a515058714d53f41a722bea69e57ff16341f42838724f37  OK
token_rows.safetensors  168  3948c1fb6b04aa7c5092487e64293995629b2f0e7e0d62f8d276b61a5a2cfc94  OK
tokens.json  719  c227930849413911c7ba00ebc0ee0a7196e143b275b88915a31d02ce55ffbec1  OK
mismatches: 0
-- loader check (src.adapters.checkpoint.read_manifest, from wt0072, CPU only):
read_manifest OK: 4 files, format visint-e3-checkpoint/1
== OK 2026-10-26T04:40:57Z
```

- 之后 testB 起每个划分的 w2c 都 rc=0（见 steps.log）；testA 由补充 1 补跑（见"W2-c 补跑"）。

## 启动

```
## start 2026-10-26T02:32:36Z
## squeue [before]
157296 RUNNING 71-13:58:22 hpcgpu06
157297 RUNNING 73-11:30:31 hpcgpu07
fetch try 1 rc=0 02:32:40
Preparing worktree (detached HEAD d38a5205)
add_rc=0
sparse_rc=0
readtree_rc=0
HEAD d38a520525c6208a3f5ddb1062ca09a9646e85b6 status_lines=0
experiments: P0 
contains d77ed1fd: yes
12/12 sha256 match
driver uses per-node pause file
w2.json frozen: True
restore0072: == restore OK 2026-10-26T02:31:28Z
W2-c adapter sha256 8c94c11c5cbc81a08a515058714d53f41a722bea69e57ff16341f42838724f37
W2-c manifest.json c605408f… ok
hpcgpu06 GPU idle
== launching with -c 8 2026-10-26T02:32:54Z
launched hpcgpu06 srun pid 2855131
== driver log:
status: 2026-10-26T02:32:56Z running e1_box_qwen3.5-9b_test_refcoco_unc_testA
== steps.log:
2026-10-26T02:32:55Z start commit d38a520525c6208a3f5ddb1062ca09a9646e85b6 node hpcgpu06 models [qwen3.5-9b] w2.json c4d083489783039d46718e19ede810370bd4b9272ef94e99f406104c2e0f9b29
sync loop 0072 pid 535295
## squeue [after launch]
157296 RUNNING 71-13:59:41 hpcgpu06
157297 RUNNING 73-11:31:50 hpcgpu07
== done 2026-10-26T02:33:55Z
```

- 推送循环 TAG 0072（PID 535295，1800 s）。

## steps.log 原文（含补跑部分）

```
2026-10-26T02:32:55Z start commit d38a520525c6208a3f5ddb1062ca09a9646e85b6 node hpcgpu06 models [qwen3.5-9b] w2.json c4d083489783039d46718e19ede810370bd4b9272ef94e99f406104c2e0f9b29
2026-10-26T02:36:43Z e1_box_qwen3.5-9b_test_refcoco_unc_testA rc=0 wall_s=227
2026-10-26T02:36:48Z ownbox_qwen3.5-9b_test_refcoco_unc_testA rc=0 wall_s=5
2026-10-26T02:42:05Z w1f_qwen3.5-9b_test_refcoco_unc_testA rc=0 wall_s=317
2026-10-26T02:42:15Z random_pick_qwen3.5-9b_test_refcoco_unc_testA rc=0 wall_s=10
2026-10-26T02:42:24Z oracle_pick_qwen3.5-9b_test_refcoco_unc_testA rc=0 wall_s=9
2026-10-26T02:48:30Z baseline_e1_qwen3.5-9b_test_refcoco_unc_testA rc=0 wall_s=366
2026-10-26T04:28:45Z w2a_qwen3.5-9b_test_refcoco_unc_testA rc=0 wall_s=6015
2026-10-26T04:29:12Z w2c_qwen3.5-9b_test_refcoco_unc_testA rc=2 wall_s=27
2026-10-26T04:32:32Z e1_box_qwen3.5-9b_test_refcoco_unc_testB rc=0 wall_s=200
2026-10-26T04:32:37Z ownbox_qwen3.5-9b_test_refcoco_unc_testB rc=0 wall_s=5
2026-10-26T04:37:55Z w1f_qwen3.5-9b_test_refcoco_unc_testB rc=0 wall_s=318
2026-10-26T04:38:05Z random_pick_qwen3.5-9b_test_refcoco_unc_testB rc=0 wall_s=10
2026-10-26T04:38:14Z oracle_pick_qwen3.5-9b_test_refcoco_unc_testB rc=0 wall_s=9
2026-10-26T04:49:29Z baseline_e1_qwen3.5-9b_test_refcoco_unc_testB rc=0 wall_s=675
2026-10-26T06:28:53Z w2a_qwen3.5-9b_test_refcoco_unc_testB rc=0 wall_s=5964
2026-10-26T06:45:56Z w2c_qwen3.5-9b_test_refcoco_unc_testB rc=0 wall_s=1023
2026-10-26T06:48:22Z e1_box_qwen3.5-9b_test_refcoco+_unc_testA rc=0 wall_s=146
2026-10-26T06:48:26Z ownbox_qwen3.5-9b_test_refcoco+_unc_testA rc=0 wall_s=4
2026-10-26T06:53:37Z w1f_qwen3.5-9b_test_refcoco+_unc_testA rc=0 wall_s=311
2026-10-26T06:53:46Z random_pick_qwen3.5-9b_test_refcoco+_unc_testA rc=0 wall_s=9
2026-10-26T06:53:56Z oracle_pick_qwen3.5-9b_test_refcoco+_unc_testA rc=0 wall_s=10
2026-10-26T06:59:20Z baseline_e1_qwen3.5-9b_test_refcoco+_unc_testA rc=0 wall_s=324
2026-10-26T08:38:46Z w2a_qwen3.5-9b_test_refcoco+_unc_testA rc=0 wall_s=5966
2026-10-26T08:52:07Z w2c_qwen3.5-9b_test_refcoco+_unc_testA rc=0 wall_s=801
2026-10-26T08:55:33Z e1_box_qwen3.5-9b_test_refcoco+_unc_testB rc=0 wall_s=206
2026-10-26T08:55:37Z ownbox_qwen3.5-9b_test_refcoco+_unc_testB rc=0 wall_s=4
2026-10-26T09:00:44Z w1f_qwen3.5-9b_test_refcoco+_unc_testB rc=0 wall_s=307
2026-10-26T09:00:54Z random_pick_qwen3.5-9b_test_refcoco+_unc_testB rc=0 wall_s=10
2026-10-26T09:01:03Z oracle_pick_qwen3.5-9b_test_refcoco+_unc_testB rc=0 wall_s=9
2026-10-26T09:11:44Z baseline_e1_qwen3.5-9b_test_refcoco+_unc_testB rc=0 wall_s=641
2026-10-26T10:50:15Z w2a_qwen3.5-9b_test_refcoco+_unc_testB rc=0 wall_s=5911
2026-10-26T11:05:45Z w2c_qwen3.5-9b_test_refcoco+_unc_testB rc=0 wall_s=930
2026-10-26T11:08:26Z e1_box_qwen3.5-9b_test_refcocog_umd_test rc=0 wall_s=161
2026-10-26T11:08:30Z ownbox_qwen3.5-9b_test_refcocog_umd_test rc=0 wall_s=4
2026-10-26T11:13:33Z w1f_qwen3.5-9b_test_refcocog_umd_test rc=0 wall_s=303
2026-10-26T11:13:42Z random_pick_qwen3.5-9b_test_refcocog_umd_test rc=0 wall_s=9
2026-10-26T11:13:51Z oracle_pick_qwen3.5-9b_test_refcocog_umd_test rc=0 wall_s=9
2026-10-26T11:25:26Z baseline_e1_qwen3.5-9b_test_refcocog_umd_test rc=0 wall_s=695
2026-10-26T13:05:01Z w2a_qwen3.5-9b_test_refcocog_umd_test rc=0 wall_s=5975
2026-10-26T13:18:15Z w2c_qwen3.5-9b_test_refcocog_umd_test rc=0 wall_s=794
2026-10-26T13:18:15Z done; failed steps: 1
2026-10-26T13:18:15Z （不推送，EV_NO_GIT=1：由登录节点的同步循环推）W2 test 0072 done (failed steps: 1)
2026-10-26T19:04:45Z rerun_w2c start code d38a520525c6208a3f5ddb1062ca09a9646e85b6 node hpcgpu06 checkpoint manifest c605408fbb92cceafbc59dff4d85c9c912b246d05dc5e29235db0cff389d93b7
2026-10-26T19:18:08Z w2c_rerun_qwen3.5-9b_test_refcoco_unc_testA rc=0 wall_s=803
2026-10-26T19:18:08Z w2c_rerun_qwen3.5-9b_test_refcoco_unc_testB skip（RUN-0072 已跑完这个划分）
2026-10-26T19:18:08Z w2c_rerun_qwen3.5-9b_test_refcoco+_unc_testA skip（RUN-0072 已跑完这个划分）
2026-10-26T19:18:08Z w2c_rerun_qwen3.5-9b_test_refcoco+_unc_testB skip（RUN-0072 已跑完这个划分）
2026-10-26T19:18:09Z w2c_rerun_qwen3.5-9b_test_refcocog_umd_test skip（RUN-0072 已跑完这个划分）
2026-10-26T19:18:09Z rerun_w2c done; failed steps: 0
2026-10-26T19:18:09Z （不推送，EV_NO_GIT=1：由登录节点的同步循环推）W2 test 0072 w2c rerun done (failed steps: 0)
```

- 主驱动心跳 `2026-10-26T13:18:15Z done 0072 (failed steps: 1)`（失败的一步是 testA 的 w2c）；主驱动墙钟 02:32:55Z – 13:18:15Z（38,720 s，10.76 h），上限 16 h。
- 每个划分里墙钟最长的是 w2a（5,964–6,015 s）；w2c 794–1,023 s；baseline_e1（多边形）324–695 s；e1_box 146–227 s；w1f 303–318 s。

## 各划分的读数（由 `experiments/W2/0072_w2_test_probe/p72_tables.py` 读出；testA 的 w2c 为补跑结果）

### test_refcoco_unc_testA

| 设置 | cIoU [95% CI] | gIoU [95% CI] | n | complete |
|---|---|---|---|---|
| w1f | 0.5514 [0.5286, 0.5743] | 0.5419 [0.5243, 0.5593] | 1000 | True |
| w2a | 0.5832 [0.5618, 0.6018] | 0.6038 [0.5914, 0.6160] | 1000 | True |
| w2c | 0.6423 [0.6246, 0.6574] | 0.6531 [0.6411, 0.6648] | 1000 | True |
| ownbox | 0.4913 [0.4765, 0.5042] | 0.5048 [0.4950, 0.5140] | 1000 | True |
| random_pick | 0.5483 [0.5280, 0.5688] | 0.5508 [0.5361, 0.5661] | 1000 | True |
| oracle_pick | 0.6324 [0.6119, 0.6506] | 0.6499 [0.6369, 0.6617] | 1000 | True |
| baseline_e1 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 1000 | True |
| 出框 baseline_e1 acc@0.5 | 0.9360 [0.9210, 0.9500] | | 1000 | True |
| 出框 propose_w2c acc@0.5 | 0.9550 [0.9410, 0.9670] | | 1000 | True |

- baseline_e1（多边形）parse：{"n": 1000, "n_ok": 0, "n_failed": 1000, "failure_rate": 1.0, "failures": {"malformed": 993, "no_polygon": 3, "truncated": 4}}

### test_refcoco_unc_testB

| 设置 | cIoU [95% CI] | gIoU [95% CI] | n | complete |
|---|---|---|---|---|
| w1f | 0.5705 [0.5473, 0.5924] | 0.5535 [0.5337, 0.5740] | 1000 | True |
| w2a | 0.5952 [0.5727, 0.6164] | 0.6183 [0.6017, 0.6360] | 1000 | True |
| w2c | 0.6642 [0.6421, 0.6858] | 0.6842 [0.6683, 0.7007] | 1000 | True |
| ownbox | 0.5300 [0.5129, 0.5466] | 0.5549 [0.5411, 0.5692] | 1000 | True |
| random_pick | 0.5632 [0.5403, 0.5848] | 0.5617 [0.5423, 0.5815] | 1000 | True |
| oracle_pick | 0.6452 [0.6229, 0.6668] | 0.6660 [0.6498, 0.6833] | 1000 | True |
| baseline_e1 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 1000 | True |
| 出框 baseline_e1 acc@0.5 | 0.8680 [0.8450, 0.8880] | | 1000 | True |
| 出框 propose_w2c acc@0.5 | 0.9110 [0.8930, 0.9300] | | 1000 | True |

- baseline_e1（多边形）parse：{"n": 1000, "n_ok": 0, "n_failed": 1000, "failure_rate": 1.0, "failures": {"malformed": 988, "no_polygon": 2, "truncated": 10}}

### test_refcoco+_unc_testA

| 设置 | cIoU [95% CI] | gIoU [95% CI] | n | complete |
|---|---|---|---|---|
| w1f | 0.5416 [0.5197, 0.5609] | 0.5220 [0.5035, 0.5396] | 1000 | True |
| w2a | 0.5630 [0.5419, 0.5829] | 0.5753 [0.5594, 0.5903] | 1000 | True |
| w2c | 0.6047 [0.5859, 0.6237] | 0.6291 [0.6150, 0.6432] | 1000 | True |
| ownbox | 0.4779 [0.4631, 0.4923] | 0.4823 [0.4704, 0.4935] | 1000 | True |
| random_pick | 0.5390 [0.5186, 0.5567] | 0.5372 [0.5203, 0.5528] | 1000 | True |
| oracle_pick | 0.6174 [0.5965, 0.6370] | 0.6235 [0.6086, 0.6384] | 1000 | True |
| baseline_e1 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 1000 | True |
| 出框 baseline_e1 acc@0.5 | 0.8860 [0.8660, 0.9040] | | 1000 | True |
| 出框 propose_w2c acc@0.5 | 0.9180 [0.9010, 0.9340] | | 1000 | True |

- baseline_e1（多边形）parse：{"n": 1000, "n_ok": 0, "n_failed": 1000, "failure_rate": 1.0, "failures": {"malformed": 996, "truncated": 4}}

### test_refcoco+_unc_testB

| 设置 | cIoU [95% CI] | gIoU [95% CI] | n | complete |
|---|---|---|---|---|
| w1f | 0.4966 [0.4711, 0.5216] | 0.4889 [0.4680, 0.5098] | 1000 | True |
| w2a | 0.5208 [0.4959, 0.5441] | 0.5407 [0.5214, 0.5588] | 1000 | True |
| w2c | 0.6104 [0.5872, 0.6320] | 0.6290 [0.6103, 0.6455] | 1000 | True |
| ownbox | 0.4709 [0.4502, 0.4906] | 0.4874 [0.4708, 0.5032] | 1000 | True |
| random_pick | 0.4941 [0.4695, 0.5178] | 0.4931 [0.4740, 0.5139] | 1000 | True |
| oracle_pick | 0.5627 [0.5359, 0.5871] | 0.5810 [0.5606, 0.6006] | 1000 | True |
| baseline_e1 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 1000 | True |
| 出框 baseline_e1 acc@0.5 | 0.7580 [0.7320, 0.7830] | | 1000 | True |
| 出框 propose_w2c acc@0.5 | 0.8340 [0.8110, 0.8580] | | 1000 | True |

- baseline_e1（多边形）parse：{"n": 1000, "n_ok": 0, "n_failed": 1000, "failure_rate": 1.0, "failures": {"malformed": 987, "truncated": 13}}

### test_refcocog_umd_test

| 设置 | cIoU [95% CI] | gIoU [95% CI] | n | complete |
|---|---|---|---|---|
| w1f | 0.5505 [0.5259, 0.5733] | 0.5337 [0.5147, 0.5524] | 1000 | True |
| w2a | 0.5751 [0.5528, 0.5969] | 0.5778 [0.5619, 0.5941] | 1000 | True |
| w2c | 0.6351 [0.6150, 0.6537] | 0.6347 [0.6199, 0.6483] | 1000 | True |
| ownbox | 0.4808 [0.4642, 0.4977] | 0.4852 [0.4729, 0.4984] | 1000 | True |
| random_pick | 0.5568 [0.5337, 0.5791] | 0.5408 [0.5240, 0.5585] | 1000 | True |
| oracle_pick | 0.6247 [0.6016, 0.6465] | 0.6245 [0.6080, 0.6408] | 1000 | True |
| baseline_e1 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 1000 | True |
| 出框 baseline_e1 acc@0.5 | 0.8780 [0.8590, 0.8960] | | 1000 | True |
| 出框 propose_w2c acc@0.5 | 0.9160 [0.8970, 0.9330] | | 1000 | True |

- baseline_e1（多边形）parse：{"n": 1000, "n_ok": 0, "n_failed": 1000, "failure_rate": 1.0, "failures": {"ambiguous": 1, "malformed": 986, "no_polygon": 4, "truncated": 9}}

### w2a 与 w2c 的挑选统计（分母 = 有候选的句子；都不对 = pick.none）

| 划分 | 设置 | 挑中最优 | 挑 1 号 | 都不对 | 解析失败 | 出框失败 | thinking_end | parse_failure |
|---|---|---|---|---|---|---|---|---|
| test_refcoco_unc_testA | w2a | 0.346（345/997） | 0.469（468/997） | 0.005（5/997） | 0.021（21/997） | {'multiple_boxes': 2, 'truncated': 1} | {'forced': 802, 'natural': 195} | {'no_choice': 3, 'truncated': 18} |
| test_refcoco_unc_testA | w2c | 0.659（659/1000） | 0.123（123/1000） | 0.000（0/1000） | 0.000（0/1000） | {} | {'None': 1000} | {} |
| test_refcoco_unc_testB | w2a | 0.348（342/983） | 0.464（456/983） | 0.010（10/983） | 0.028（28/983） | {'multiple_boxes': 12, 'truncated': 5} | {'forced': 817, 'natural': 166} | {'no_choice': 4, 'truncated': 24} |
| test_refcoco_unc_testB | w2c | 0.653（653/1000） | 0.089（89/1000） | 0.000（0/1000） | 0.000（0/1000） | {} | {'None': 1000} | {} |
| test_refcoco+_unc_testA | w2a | 0.357（351/984） | 0.526（518/984） | 0.019（19/984） | 0.028（28/984） | {'multiple_boxes': 15, 'truncated': 1} | {'forced': 813, 'natural': 171} | {'no_choice': 3, 'truncated': 25} |
| test_refcoco+_unc_testA | w2c | 0.646（646/1000） | 0.111（111/1000） | 0.000（0/1000） | 0.000（0/1000） | {} | {'None': 1000} | {} |
| test_refcoco+_unc_testB | w2a | 0.429（410/955） | 0.426（407/955） | 0.012（11/955） | 0.029（28/955） | {'multiple_boxes': 40, 'truncated': 5} | {'forced': 793, 'natural': 162} | {'no_choice': 2, 'out_of_range': 1, 'truncated': 25} |
| test_refcoco+_unc_testB | w2c | 0.658（658/1000） | 0.130（130/1000） | 0.000（0/1000） | 0.000（0/1000） | {} | {'None': 1000} | {} |
| test_refcocog_umd_test | w2a | 0.366（358/978） | 0.496（485/978） | 0.022（22/978） | 0.040（39/978） | {'multiple_boxes': 22} | {'forced': 812, 'natural': 166} | {'no_choice': 4, 'truncated': 35} |
| test_refcocog_umd_test | w2c | 0.652（652/1000） | 0.125（125/1000） | 0.000（0/1000） | 0.000（0/1000） | {} | {'None': 1000} | {} |

## W2-c 补跑（RUN-0072 补充 1）

- 工作树 wt0072r（稀疏）HEAD `cdd489c7b621245a5f2eead15ba245fc1596d91a`，`rerun_w2c.sh` sha256 `b610f00841eb5563ac36ad2652c2a465e7694c3ef7feef7324bedd22f7b61ca7` 与补充 1 相同；CODE 用 wt0072（HEAD `d38a5205…`），其 12 个文件的 sha256 再核对一次，都与 RUN-0072 任务书相同。
- testA 失败留下的只有 `qwen3.5-9b/errors/error_20261026T042911935199Z.json`；`refseg/w2c/`、`refexp_box/propose_w2c/` 当时都不存在，没有半成品，没有删任何东西。
- 启动：`$VI/tmp/start0072r.sh` 等到 RUN-0074 心跳 `13:18:40Z running lb_base`，13:18:43Z 按补充 1 的命令启动（`srun --jobid=157296 --overlap -n1 -c 8`，`TINY=0 EV_CAP_H=4`）；前后 squeue 两条 RUNNING。驱动日志 `manifest files 4; extra []; missing []`；等 RUN-0074 期间心跳 "waiting for gpu (running: eval_0074.status …)"。
- 补跑与跳过：testA 补跑（19:04:45Z – 19:18:08Z，rc=0，803 s）；testB、refcoco+ testA、refcoco+ testB、refcocog test 跳过（RUN-0072 自己的 w2c 已 complete）。心跳 `2026-10-26T19:18:09Z done 0072r (failed steps: 0)`。
- 5 个划分的 w2c：

| 划分 | 来源 | cIoU [95% CI] | gIoU [95% CI] |
|---|---|---|---|
| refcoco testA | 补跑 | 0.6423 [0.6246, 0.6574] | 0.6531 [0.6411, 0.6648] |
| refcoco testB | RUN-0072 | 0.6642 [0.6421, 0.6858] | 0.6842 [0.6683, 0.7007] |
| refcoco+ testA | RUN-0072 | 0.6047 [0.5859, 0.6237] | 0.6291 [0.6150, 0.6432] |
| refcoco+ testB | RUN-0072 | 0.6104 [0.5872, 0.6320] | 0.6290 [0.6103, 0.6455] |
| refcocog test | RUN-0072 | 0.6351 [0.6150, 0.6537] | 0.6347 [0.6199, 0.6483] |

## 显存与 traceback

- 显存峰值：test_w2.sh 与 rerun_w2c.sh 都不采样显存，结果目录里没有 gpu_mem.csv，日志里也没有。
- `logs/` 下 41 个步日志 grep `Traceback`、`out of memory`：都是 0。testA 的 w2c 日志只有上面那一行 CheckpointError。

## 收尾

- 推送循环按管理员要求开到补跑推完：19:11:42Z 一轮推送成功；19:20:26Z 在 git.lock 下按 PID 停掉循环 535295（当时在 sleep），随后 `ONCE=1` 推一轮：`2026-10-26T19:20:41Z ok`。
- 后 squeue：157296、157297 RUNNING。06 接着跑 RUN-0078（19:18:29Z running llmseg）。
