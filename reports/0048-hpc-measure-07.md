# RUN-0048 — hpcgpu07 训完 0.8B 后的实测（第二版，D93；含补充 1）

服务器 CC（HPC）。只贴数字与事实，不做解读。结果在 `experiments/E3/0048_hpc_measure_hpcgpu07/`、`experiments/E5/0048_hpc_measure_hpcgpu07/`、`experiments/E3/0048_hpc_measure_e1_hpcgpu07/`。下面的表都由 `experiments/E3/0048_hpc_measure_hpcgpu07_probe/p48_tables.py` 从这些文件读出，原始命令输出在同目录的 `raw/`。

## 0. 结果

- measure.sh：`done 0048 (failed steps: 3)`。三个失败步都是 T 文本，rc=2，原因是 `--out` 的目录名不带运行编号（管理员已确认）；由补充 1 补测，`done 0048e1 (failed steps: 0)`。
- **E3b 种子 1 起点在 A100 上的合并核对没过**：`restricted/thinking_off：关思考时受限解码与 generate_batch 的 exact_rows 0.9400 < 0.95`，训练没有开始（`e5_train20_e3b_s1 rc=1`）。按 E5 预注册 §8b，E3b 起点只用种子 0、2。
- 其余三个 E5 起点（e3b_c3、c3_warmup、base_warmup）的合并核对都 pass，各跑了 20 步。
- GPU 用量：measure.sh 12:36:26–13:30:36Z（3,250 s），measure_e1.sh 14:26:14–14:44:06Z（1,072 s），合计约 **1.20 h**，在上限 4 h + 2 h 以内。
- `$VI/ckpt/hpcgpu07/tmp0048` 已不存在。份额差值（`lfs quota` 减基线）是 59,299,376 KB，约 59.3 GB。
- 只用了 hpcgpu07，没有写 `$VI/logs/train.pause`。06 的 RUN-0045 一直在训练。

## 1. 校验与启动

- **工作树：** `$VI/wt/wt0048` 是 `bb14b3aa1811be62efbc1cf923e620c70a100c46`，status 为空（`raw/s1b.out`）。
  - `measure.sh` 的 sha256 是 `c00f16550b480cad19ca15f9d632861188ee05ffa11aab35cf80c9740f25f3df`（第二版），`eval_lib.sh` 是 `2380464b…4a59`，都与任务书相同。
  - `train_0045-hpcgpu07.done` 存在；`train.pause`、`train.pause.hpcgpu07`、`HOLD` 都不存在。RUN-0047 已核对通过（报告 9e1b0cc）。
  - RUN-0049 的 07 部分先跑完（10:43Z）。
- **启动：** 12:36:26Z 用方案 B（`launch.sh`，srun 那一行与任务书相同），步骤 157297.81，拿到 GPU 的时刻即 12:36:26Z（`raw/s3.out`）。
- **squeue：** 启动前 `12:36:10Z 157296 RUNNING 58-00:01:56 / 157297 RUNNING 59-21:34:05`；启动后 `12:37:11Z 157296 RUNNING 58-00:02:57 / 157297 RUNNING 59-21:35:06`。
- **推送循环（TAG 0048，600 s，v2）：** PID 2536554。
  - 12:37Z 与 12:47Z 两轮报 `push failed`：`experiments/E5/0048_hpc_measure_hpcgpu07` 当时只有空目录，git 不认这个路径，整轮都没推上去。目录里有文件之后，13:28Z 起每轮都是 `ok`。
  - 14:26Z 按补充 1 在 sleep 时按 PID 停掉，加上 e1 路径重启（PID 2918712）。
  - 14:49Z 核对三个结果目录与推送工作树逐文件一致（差异 0）后，在 sleep 时按 PID 停掉（`raw/fin.out`）。RUN-0045 的循环不受影响。
- **steps.log**（measure.sh，原文）：

```
2026-10-12T12:36:26Z start commit bb14b3aa1811be62efbc1cf923e620c70a100c46 node hpcgpu07
2026-10-12T12:50:28Z e5_train20_e3b_c3 rc=0 wall_s=842
2026-10-12T12:59:52Z e5_train20_c3_warmup rc=0 wall_s=564
2026-10-12T13:09:06Z e5_train20_base_warmup rc=0 wall_s=554
2026-10-12T13:09:06Z （不推送，EV_NO_GIT=1：由登录节点的同步循环推）E5 train 20 steps x 3 starts
2026-10-12T13:12:18Z e5_val64_e3b_c3 rc=0 wall_s=192
2026-10-12T13:12:18Z （不推送，EV_NO_GIT=1：由登录节点的同步循环推）E5 val pass on e3b_c3 start
2026-10-12T13:14:20Z e3b_train20_qwen3.5-0.8b rc=0 wall_s=122
2026-10-12T13:21:27Z e5_train20_e3b_s1 rc=1 wall_s=427
2026-10-12T13:21:27Z e5_train20_e3b_s1 没有开训（见日志：合并核对或其他原因）
2026-10-12T13:21:27Z （不推送，EV_NO_GIT=1：由登录节点的同步循环推）E3b 0.8B train 20 steps; E3b s1 start check
2026-10-12T13:21:31Z ttext_e3b_c3_s0_dev20 rc=2 wall_s=4
2026-10-12T13:21:34Z ttext_e3b_c3_s0_free10 rc=2 wall_s=3
2026-10-12T13:29:18Z grid_e3b_c3_s0_refseg100 rc=0 wall_s=464
2026-10-12T13:30:33Z grid_e3b_c3_s0_depth50 rc=0 wall_s=75
2026-10-12T13:30:33Z （不推送，EV_NO_GIT=1：由登录节点的同步循环推）0.8B E3b+C3 s0 eval timing
2026-10-12T13:30:35Z e1_base_dev20 rc=2 wall_s=2
2026-10-12T13:30:36Z done; failed steps: 3
2026-10-12T13:30:36Z （不推送，EV_NO_GIT=1：由登录节点的同步循环推）measurement done (failed steps: 3)
```

## 2. 实测
### E5 训练 20 步（2B，各起点）

| 起点 | 命令 rc | 命令墙钟 s | 合并核对 pass | 合并核对失败项 | 合并核对耗时 s（merge 前 / 后 / restricted / across_merge / 合计） | 命令开始到训练开始 s | 每步中位 s | 每步平均 s | 显存峰值 GB |
|---|---|---|---|---|---|---|---|---|---|
| e3b_c3 | 0 | 842 | True | 无 | 152.4 / 58.6 / 285.2 / 20.8 / 517.8 | 719 | 5.034 | 5.348 | 9.35 |
| c3_warmup | 0 | 564 | True | 无 | 91.9 / 59.7 / 254.7 / 22.3 / 429.2 | 461 | 5.064 | 5.066 | 9.35 |
| base_warmup | 0 | 554 | True | 无 | 82.2 / 60.3 / 255.4 / 20.7 / 419.2 | 451 | 5.060 | 5.061 | 9.35 |
| e3b（种子 1） | 1 | 427 | False | restricted/thinking_off：关思考时受限解码与 generate_batch 的 exact_rows 0.9400 < 0.95 | 81.5 / 59.4 / 252.7 / 20.5 / 414.8 | 没有开训 | — | — | — |

#### 合并核对原文摘要：e3b_c3（`experiments/E5/0048_hpc_measure_hpcgpu07/train/e3b_c3/qwen3.5-2b/start_checks/merge_check_bf73188b8c88c760.json`）

- checkpoint：`/home/mail/chenyuma/visint/b200/ckpt/0040/e3b_c3/qwen3.5-2b/seed0/ckpt/final/model`，manifest_sha256 `bf73188b8c88c760…`；pass = True；failures = {"restricted": [], "merge": []}
- merge：`by_thinking.thinking_off.generate.exact_rows`=0.25，`by_thinking.thinking_off.generate.token_agreement`=0.3618，`by_thinking.thinking_off.generate.first_divergence.median`=15.0，`by_thinking.thinking_off.generate.first_divergence.max`=87，`by_thinking.thinking_off.generate.gap.median`=0.0733，`by_thinking.thinking_off.generate.gap.max`=0.5017，`by_thinking.thinking_off.generate.top2_gap.median`=0.059，`by_thinking.thinking_off.generate.top2_gap.max`=0.5017，`by_thinking.thinking_off.teacher_forcing.argmax_agreement`=0.9912，`by_thinking.thinking_off.teacher_forcing.gap.median`=0.0131，`by_thinking.thinking_off.teacher_forcing.gap.max`=0.3092，`by_thinking.thinking_off.teacher_forcing.top2_gap.median`=0.0127，`by_thinking.thinking_off.teacher_forcing.top2_gap.max`=0.3092，`by_thinking.thinking_on.generate.exact_rows`=0.44，`by_thinking.thinking_on.generate.token_agreement`=0.5768，`by_thinking.thinking_on.generate.first_divergence.median`=19.0，`by_thinking.thinking_on.generate.first_divergence.max`=48，`by_thinking.thinking_on.generate.gap.median`=0.0682，`by_thinking.thinking_on.generate.gap.max`=0.8432，`by_thinking.thinking_on.generate.top2_gap.median`=0.0497，`by_thinking.thinking_on.generate.top2_gap.max`=0.8432，`by_thinking.thinking_on.teacher_forcing.argmax_agreement`=0.9924，`by_thinking.thinking_on.teacher_forcing.gap.median`=0.0247，`by_thinking.thinking_on.teacher_forcing.gap.max`=0.8432
- restricted：`by_thinking.thinking_off.decode_vs_generate_batch.exact_rows`=0.985，`by_thinking.thinking_off.decode_vs_generate_batch.token_agreement`=0.9944，`by_thinking.thinking_off.decode_vs_generate_batch.first_divergence.median`=570.0，`by_thinking.thinking_off.decode_vs_generate_batch.first_divergence.max`=963，`by_thinking.thinking_off.decode_vs_generate_batch.gap.median`=0.0141，`by_thinking.thinking_off.decode_vs_generate_batch.gap.max`=0.0261，`by_thinking.thinking_off.decode_vs_generate_batch.top2_gap.median`=0.0141，`by_thinking.thinking_off.decode_vs_generate_batch.top2_gap.max`=0.0261，`by_thinking.thinking_off.teacher_forcing.agreement`=0.9999，`by_thinking.thinking_off.teacher_forcing.gap.median`=0.0035，`by_thinking.thinking_off.teacher_forcing.gap.max`=0.0607，`by_thinking.thinking_off.teacher_forcing.top2_gap.median`=0.0035，`by_thinking.thinking_off.teacher_forcing.top2_gap.max`=0.0607，`across_merge.by_thinking.thinking_off.teacher_forcing.argmax_agreement`=0.9999，`across_merge.by_thinking.thinking_off.teacher_forcing.gap.median`=0.0164，`across_merge.by_thinking.thinking_off.teacher_forcing.gap.max`=0.038，`across_merge.by_thinking.thinking_off.teacher_forcing.top2_gap.median`=0.0164，`across_merge.by_thinking.thinking_off.teacher_forcing.top2_gap.max`=0.038，`pass`=True

#### 合并核对原文摘要：c3_warmup（`experiments/E5/0048_hpc_measure_hpcgpu07/train/c3_warmup/qwen3.5-2b/start_checks/merge_check_dca6a89d9a3e920c.json`）

- checkpoint：`/home/mail/chenyuma/visint/b200/ckpt/0040/c3_warmup/qwen3.5-2b/seed0/ckpt/final/model`，manifest_sha256 `dca6a89d9a3e920c…`；pass = True；failures = {"restricted": [], "merge": []}
- merge：`by_thinking.thinking_off.generate.exact_rows`=0.48，`by_thinking.thinking_off.generate.token_agreement`=0.5417，`by_thinking.thinking_off.generate.first_divergence.median`=11.0，`by_thinking.thinking_off.generate.first_divergence.max`=118，`by_thinking.thinking_off.generate.gap.median`=0.0822，`by_thinking.thinking_off.generate.gap.max`=0.2999，`by_thinking.thinking_off.generate.top2_gap.median`=0.0662，`by_thinking.thinking_off.generate.top2_gap.max`=0.2999，`by_thinking.thinking_off.teacher_forcing.argmax_agreement`=0.9969，`by_thinking.thinking_off.teacher_forcing.gap.median`=0.0256，`by_thinking.thinking_off.teacher_forcing.gap.max`=0.2605，`by_thinking.thinking_off.teacher_forcing.top2_gap.median`=0.0254，`by_thinking.thinking_off.teacher_forcing.top2_gap.max`=0.2605，`by_thinking.thinking_on.generate.exact_rows`=0.775，`by_thinking.thinking_on.generate.token_agreement`=0.7918，`by_thinking.thinking_on.generate.first_divergence.median`=3.0，`by_thinking.thinking_on.generate.first_divergence.max`=58，`by_thinking.thinking_on.generate.gap.median`=0.0734，`by_thinking.thinking_on.generate.gap.max`=0.1878，`by_thinking.thinking_on.generate.top2_gap.median`=0.0793，`by_thinking.thinking_on.generate.top2_gap.max`=0.1878，`by_thinking.thinking_on.teacher_forcing.argmax_agreement`=0.9982，`by_thinking.thinking_on.teacher_forcing.gap.median`=0.0404，`by_thinking.thinking_on.teacher_forcing.gap.max`=0.1837
- restricted：`by_thinking.thinking_off.decode_vs_generate_batch.exact_rows`=1.0，`by_thinking.thinking_off.decode_vs_generate_batch.token_agreement`=1.0，`by_thinking.thinking_off.teacher_forcing.agreement`=1.0，`across_merge.by_thinking.thinking_off.teacher_forcing.argmax_agreement`=1.0，`pass`=True

#### 合并核对原文摘要：base_warmup（`experiments/E5/0048_hpc_measure_hpcgpu07/train/base_warmup/qwen3.5-2b/start_checks/merge_check_58ba382e9cd0b5bf.json`）

- checkpoint：`/home/mail/chenyuma/visint/b200/ckpt/0036/warmup/qwen3.5-2b/seed0/ckpt/final/model`，manifest_sha256 `58ba382e9cd0b5bf…`；pass = True；failures = {"restricted": [], "merge": []}
- merge：`by_thinking.thinking_off.generate.exact_rows`=0.505，`by_thinking.thinking_off.generate.token_agreement`=0.6581，`by_thinking.thinking_off.generate.first_divergence.median`=30.0，`by_thinking.thinking_off.generate.first_divergence.max`=121，`by_thinking.thinking_off.generate.gap.median`=0.0843，`by_thinking.thinking_off.generate.gap.max`=0.2423，`by_thinking.thinking_off.generate.top2_gap.median`=0.0814，`by_thinking.thinking_off.generate.top2_gap.max`=0.2423，`by_thinking.thinking_off.teacher_forcing.argmax_agreement`=0.9954，`by_thinking.thinking_off.teacher_forcing.gap.median`=0.0409，`by_thinking.thinking_off.teacher_forcing.gap.max`=0.1625，`by_thinking.thinking_off.teacher_forcing.top2_gap.median`=0.0395，`by_thinking.thinking_off.teacher_forcing.top2_gap.max`=0.1625，`by_thinking.thinking_on.generate.exact_rows`=0.335，`by_thinking.thinking_on.generate.token_agreement`=0.6027，`by_thinking.thinking_on.generate.first_divergence.median`=47.0，`by_thinking.thinking_on.generate.first_divergence.max`=126，`by_thinking.thinking_on.generate.gap.median`=0.0693，`by_thinking.thinking_on.generate.gap.max`=0.2187，`by_thinking.thinking_on.generate.top2_gap.median`=0.0674，`by_thinking.thinking_on.generate.top2_gap.max`=0.2187，`by_thinking.thinking_on.teacher_forcing.argmax_agreement`=0.993，`by_thinking.thinking_on.teacher_forcing.gap.median`=0.0369，`by_thinking.thinking_on.teacher_forcing.gap.max`=0.2026
- restricted：`by_thinking.thinking_off.decode_vs_generate_batch.exact_rows`=1.0，`by_thinking.thinking_off.decode_vs_generate_batch.token_agreement`=1.0，`by_thinking.thinking_off.teacher_forcing.agreement`=1.0，`across_merge.by_thinking.thinking_off.teacher_forcing.argmax_agreement`=1.0，`pass`=True

#### 合并核对原文摘要：e3b（`experiments/E5/0048_hpc_measure_hpcgpu07/train/e3b/qwen3.5-2b/start_checks/merge_check_7c3d293818e37788.json`）

- checkpoint：`/home/mail/chenyuma/visint/b200/ckpt/0036/e3b/qwen3.5-2b/seed1/ckpt/final/model`，manifest_sha256 `7c3d293818e37788…`；pass = False；failures = {"restricted": ["restricted/thinking_off：关思考时受限解码与 generate_batch 的 exact_rows 0.9400 < 0.95"], "merge": []}
- merge：`by_thinking.thinking_off.generate.exact_rows`=0.445，`by_thinking.thinking_off.generate.token_agreement`=0.5365，`by_thinking.thinking_off.generate.first_divergence.median`=15.0，`by_thinking.thinking_off.generate.first_divergence.max`=126，`by_thinking.thinking_off.generate.gap.median`=0.082，`by_thinking.thinking_off.generate.gap.max`=0.6038，`by_thinking.thinking_off.generate.top2_gap.median`=0.0814，`by_thinking.thinking_off.generate.top2_gap.max`=0.343，`by_thinking.thinking_off.teacher_forcing.argmax_agreement`=0.9924，`by_thinking.thinking_off.teacher_forcing.gap.median`=0.0772，`by_thinking.thinking_off.teacher_forcing.gap.max`=0.9902，`by_thinking.thinking_off.teacher_forcing.top2_gap.median`=0.0726，`by_thinking.thinking_off.teacher_forcing.top2_gap.max`=0.3214，`by_thinking.thinking_on.generate.exact_rows`=0.225，`by_thinking.thinking_on.generate.token_agreement`=0.4011，`by_thinking.thinking_on.generate.first_divergence.median`=22.0，`by_thinking.thinking_on.generate.first_divergence.max`=122，`by_thinking.thinking_on.generate.gap.median`=0.089，`by_thinking.thinking_on.generate.gap.max`=0.8202，`by_thinking.thinking_on.generate.top2_gap.median`=0.0853，`by_thinking.thinking_on.generate.top2_gap.max`=0.8202，`by_thinking.thinking_on.teacher_forcing.argmax_agreement`=0.9904，`by_thinking.thinking_on.teacher_forcing.gap.median`=0.0637，`by_thinking.thinking_on.teacher_forcing.gap.max`=0.5363
- restricted：`by_thinking.thinking_off.decode_vs_generate_batch.exact_rows`=0.94，`by_thinking.thinking_off.decode_vs_generate_batch.token_agreement`=0.9791，`by_thinking.thinking_off.decode_vs_generate_batch.first_divergence.median`=704.0，`by_thinking.thinking_off.decode_vs_generate_batch.first_divergence.max`=1044，`by_thinking.thinking_off.decode_vs_generate_batch.gap.median`=0.0391，`by_thinking.thinking_off.decode_vs_generate_batch.gap.max`=0.0935，`by_thinking.thinking_off.decode_vs_generate_batch.top2_gap.median`=0.0391，`by_thinking.thinking_off.decode_vs_generate_batch.top2_gap.max`=0.0935，`by_thinking.thinking_off.teacher_forcing.agreement`=0.9998，`by_thinking.thinking_off.teacher_forcing.gap.median`=0.0334，`by_thinking.thinking_off.teacher_forcing.gap.max`=0.1665，`by_thinking.thinking_off.teacher_forcing.top2_gap.median`=0.0334，`by_thinking.thinking_off.teacher_forcing.top2_gap.max`=0.1665，`across_merge.by_thinking.thinking_off.teacher_forcing.argmax_agreement`=0.9998，`across_merge.by_thinking.thinking_off.teacher_forcing.gap.median`=0.0462，`across_merge.by_thinking.thinking_off.teacher_forcing.gap.max`=0.237，`across_merge.by_thinking.thinking_off.teacher_forcing.top2_gap.median`=0.0462，`across_merge.by_thinking.thinking_off.teacher_forcing.top2_gap.max`=0.237，`pass`=False

### E5 val 一次（e3b_c3 起点，只评起点，结构强制）

- 命令 rc=0，命令墙钟 192 s。起点合并核对：复用 e3b_c3 训练时写的 `merge_check_bf73188b….json`，没有重算（日志第 1 行）。

| 类 | 条数 | 生成墙钟 s（compute.json wall_time_s，不含加载） | 每条 s | 批数 | 每批墙钟 s |
|---|---|---|---|---|---|
| imagery | 64 | 30.7 | 0.479 | 1 | 30.7 |
| occlusion | 64 | 32.1 | 0.502 | 1 | 32.1 |
| physics | 64 | 31.7 | 0.495 | 1 | 31.7 |
| rotation3d | 64 | 33.1 | 0.517 | 1 | 33.1 |
| viewpoint | 64 | 32.3 | 0.505 | 1 | 32.3 |

### 0.8B 的 E3b 训练 20 步

- 命令 rc=0，墙钟 122 s；accounting wall_s 80.2；每步中位 3.851 s，输入构建中位 0.104 s，前向中位 1.093 s，反向中位 2.587 s；显存峰值 4.86 GB。

### 0.8B 网格评测（E3b+C3 种子 0，结构强制，单轮）

- 合并核对（两条命令共用一份，`eval/e3b_c3/qwen3.5-0.8b/seed0/merge_check.json`）：pass = True，failures = {"restricted": [], "merge": []}；耗时 merge_before_s 72.9 s、merge_after_s 46.1 s、restricted_s 243.2 s、restricted_across_merge_s 13.5 s、total_s 376.6 s

| 命令 | rc | 命令墙钟 s | 评测运行 wall_s（run_001） | 加载与合并核对 s（命令墙钟 − 评测运行） | 条数 | 生成墙钟 s | 每条 s | 评分 s |
|---|---|---|---|---|---|---|---|---|
| grid_e3b_c3_s0_refseg100 | 0 | 464 | 77.9 | 386 | 100 | 69.8 | 0.698 | 6.855 |
| grid_e3b_c3_s0_depth50 | 0 | 75 | 47.4 | 28 | 50 | 38.8 | 0.777 | 6.225 |




## 3. 补充 1：T 文本与基座 E1（0048-addendum-1）

- **工作树：** `$VI/wt/wt0048a` 是 `3d237fa7d9be01e9eee84009fb95ed00dc24c489`，status 为空；`measure_e1.sh` 的 sha256 是 `6cf0f673…77b2`，`eval_lib.sh` 是 `2380464b…4a59`，都与补充相同（`raw/a1_wt0048a.out`）。
- **启动：** 14:26:13Z，步骤 157297.82；前后 squeue 都是 RUNNING（`raw/a4.out`）。跑完后已删 `wt0048a` 与 `wt0048`。
- **steps.log**（原文）：

```
2026-10-12T14:26:14Z start commit 3d237fa7d9be01e9eee84009fb95ed00dc24c489 node hpcgpu07
2026-10-12T14:33:44Z ttext_e3b_c3_s0_dev20 rc=0 wall_s=450
2026-10-12T14:42:12Z ttext_e3b_c3_s0_free10 rc=0 wall_s=508
2026-10-12T14:44:06Z e1_base_dev20 rc=0 wall_s=114
2026-10-12T14:44:06Z done; failed steps: 0
2026-10-12T14:44:06Z （不推送，EV_NO_GIT=1：由登录节点的同步循环推）T-text timing (failed steps: 0)
```

- 每条命令拆成三段：加载与合并核对 = 命令墙钟减去各任务评测运行的合计；每条样本的耗时按任务；合计墙钟就是命令墙钟。

#### ttext_e3b_c3_s0_dev20：rc=0，命令墙钟 450 s；各任务评测运行 wall_s 合计 146.9 s；加载与核对等 303 s（命令墙钟 − 各任务合计）；合并核对 pass = True，耗时 118.8 s（`experiments/E3/0048_hpc_measure_e1_hpcgpu07/t_text/e3b_c3/seed0/qwen3.5-0.8b/merge_check.json`，dev 与 free 共用）

| 任务 | 条数 | 评测运行 wall_s | 每条 s（compute.json） | 生成 token |
|---|---|---|---|---|
| count__dev | 20 | 8.1 | 0.350 | 3951 |
| grid16__dev | 20 | 28.7 | 1.407 | 19498 |
| ovd__dev | 20 | 57.1 | 2.761 | 40960 |
| refexp_box__dev | 20 | 3.1 | 0.077 | 896 |
| refexp_poly__dev | 20 | 41.7 | 1.409 | 19495 |
| rel_depth_blink__dev | 12 | 1.5 | 0.013 | 12 |
| rel_depth_nyu__dev | 20 | 0.8 | 0.011 | 20 |
| spatial_cvbench__dev | 20 | 4.8 | 0.185 | 20 |
| spatial_embspatial__dev | 20 | 1.0 | 0.013 | 20 |

#### ttext_e3b_c3_s0_free10：rc=0，命令墙钟 508 s；各任务评测运行 wall_s 合计 32.4 s；加载与核对等 476 s（命令墙钟 − 各任务合计）；合并核对 pass = True，耗时 118.8 s（`experiments/E3/0048_hpc_measure_e1_hpcgpu07/t_text/e3b_c3/seed0/qwen3.5-0.8b/merge_check.json`，dev 与 free 共用）

| 任务 | 条数 | 评测运行 wall_s | 每条 s（compute.json） | 生成 token |
|---|---|---|---|---|
| rel_depth_blink__free | 10 | 0.8 | 0.032 | 64 |
| rel_depth_nyu__free | 10 | 2.2 | 0.030 | 68 |
| spatial_cvbench__free | 10 | 1.3 | 0.091 | 190 |
| spatial_embspatial__free | 10 | 28.0 | 2.768 | 2247 |

#### e1_base_dev20：rc=0，命令墙钟 114 s；各任务评测运行 wall_s 合计 95.9 s；加载与核对等 18 s（命令墙钟 − 各任务合计）

| 任务 | 条数 | 评测运行 wall_s | 每条 s（compute.json） | 生成 token |
|---|---|---|---|---|
| count__dev | 20 | 0.7 | 0.014 | 42 |
| grid16__dev | 20 | 28.3 | 1.393 | 20480 |
| ovd__dev | 20 | 57.9 | 2.792 | 22397 |
| refexp_box__dev | 20 | 3.0 | 0.075 | 894 |
| refexp_poly__dev | 20 | 3.0 | 0.128 | 1030 |
| rel_depth_blink__dev | 12 | 0.4 | 0.012 | 12 |
| rel_depth_nyu__dev | 20 | 0.8 | 0.011 | 20 |
| spatial_cvbench__dev | 20 | 1.2 | 0.034 | 20 |
| spatial_embspatial__dev | 20 | 0.6 | 0.013 | 20 |


- free10 那条命令墙钟 508 s，其中各任务评测运行合计 32.4 s。日志里没有把其余 476 s 再分段（只陈述）。
