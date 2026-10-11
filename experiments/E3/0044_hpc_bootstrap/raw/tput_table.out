# RUN-0044 吞吐与一致性：每条训练命令（HPC A100-PCIE-40GB）

| 节点 | 运行 | 模型 | rc | 命令墙钟 s（driver，含加载） | accounting wall_s | 步数（计时 / 记录） | 每步中位 s | 输入构建中位 s | 前向中位 s | 反向中位 s | 每步平均 s | 显存峰值 GB | 目录 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hpcgpu06 | consistency_e3b_qwen3.5-2b | qwen3.5-2b | 0 | 611 | 386.5 | 50 / 50 | 5.874 | 0.281 | 1.742 | 3.805 | 7.698 | 9.83 | `0044_hpc_consistency_hpcgpu06/e3b/qwen3.5-2b/seed0` |
| hpcgpu06 | consistency_e3a_qwen3.5-2b | qwen3.5-2b | 0 | 251 | 188.3 | 50 / 50 | 3.314 | 0.547 | 1.139 | 1.577 | 3.734 | 7.07 | `0044_hpc_consistency_hpcgpu06/e3a/full/qwen3.5-2b/seed0` |
| hpcgpu06 | tput_c3_qwen3.5-4b | qwen3.5-4b | 0 | 247 | 200.0 | 20 / 20 | 5.562 | 0.211 | 1.723 | 3.609 | 9.910 | 16.53 | `0044_hpc_throughput_hpcgpu06/c3/qwen3.5-4b/seed0` |
| hpcgpu06 | tput_e3b_c3_qwen3.5-4b | qwen3.5-4b | 0 | 224 | 192.4 | 20 / 20 | 8.636 | 0.213 | 2.559 | 5.683 | 9.525 | 19.21 | `0044_hpc_throughput_hpcgpu06/e3b_c3/qwen3.5-4b/seed0` |
| hpcgpu06 | tput_shm_e3b_c3_qwen3.5-4b | qwen3.5-4b | 0 | 265 | 180.9 | 20 / 20 | 8.586 | 0.217 | 2.554 | 5.701 | 8.949 | 19.21 | `0044_hpc_throughput_shm_hpcgpu06/e3b_c3/qwen3.5-4b/seed0` |
| hpcgpu07 | consistency_e3b_qwen3.5-2b | qwen3.5-2b | 0 | 612 | 402.1 | 50 / 50 | 5.812 | 0.119 | 1.766 | 3.838 | 7.997 | 9.83 | `0044_hpc_consistency_hpcgpu07/e3b/qwen3.5-2b/seed0` |
| hpcgpu07 | consistency_e3a_qwen3.5-2b | qwen3.5-2b | 0 | 227 | 191.4 | 50 / 50 | 3.315 | 0.543 | 1.138 | 1.592 | 3.798 | 7.07 | `0044_hpc_consistency_hpcgpu07/e3a/full/qwen3.5-2b/seed0` |
| hpcgpu07 | tput_c3_qwen3.5-0.8b | qwen3.5-0.8b | 0 | 75 | 44.5 | 20 / 20 | 2.102 | 0.186 | 0.622 | 1.248 | 2.172 | 6.40 | `0044_hpc_throughput_hpcgpu07/c3/qwen3.5-0.8b/seed0` |
| hpcgpu07 | tput_e3b_c3_qwen3.5-0.8b | qwen3.5-0.8b | 0 | 90 | 63.7 | 20 / 20 | 2.986 | 0.211 | 0.834 | 1.912 | 3.137 | 6.43 | `0044_hpc_throughput_hpcgpu07/e3b_c3/qwen3.5-0.8b/seed0` |
| hpcgpu07 | tput_shm_e3b_c3_qwen3.5-0.8b | qwen3.5-0.8b | 0 | 513 | 115.6 | 20 / 20 | 2.971 | 0.217 | 0.832 | 1.872 | 5.609 | 6.43 | `0044_hpc_throughput_shm_hpcgpu07/e3b_c3/qwen3.5-0.8b/seed0` |

## 正式训练的逐项预估（每步中位 × 配置步数；3 个种子；另列 +10 % 的加载、检查点与分段重启余量）

步数：C3 = e3b_train.json `steps` = 4000；E3b+C3 = `mixed.steps` = 8000。

| 节点 | 模型 | 变体 | 每步中位 s（Lustre） | 单个种子 h | 3 个种子 h | 3 个种子 h（+10 %） |
|---|---|---|---|---|---|---|
| hpcgpu06 | qwen3.5-4b | c3 | 5.562 | 6.18 | 18.54 | 20.39 |
| hpcgpu06 | qwen3.5-4b | e3b_c3 | 8.636 | 19.19 | 57.58 | 63.33 |
| hpcgpu07 | qwen3.5-0.8b | c3 | 2.102 | 2.34 | 7.01 | 7.71 |
| hpcgpu07 | qwen3.5-0.8b | e3b_c3 | 2.986 | 6.64 | 19.91 | 21.90 |
| hpcgpu06 | 合计（C3 × 3 + E3b+C3 × 3） | | | | 76.12 | 83.73 |
| hpcgpu07 | 合计（C3 × 3 + E3b+C3 × 3） | | | | 26.91 | 29.60 |
