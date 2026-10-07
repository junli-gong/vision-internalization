# RUN-0034 吞吐与一致性：每条训练命令

| 运行 | 模型 | rc | 命令墙钟 s（driver，含加载） | accounting wall_s | 步数（计时 / 记录） | 每步中位 s（timing） | 每步平均 s（accounting） | 前向中位 s | 反向中位 s | 显存峰值 GB（accounting） | gpu_hours（accounting） | 目录 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| consistency_e3b | qwen3.5-2b | 0 | 125 | 108.8 | 50 / 50 | 1.925 | 2.160 | 0.596 | 1.229 | 9.88 | 0.0302 | `0034_e3_consistency/e3b/qwen3.5-2b/seed0` |
| consistency_e3a | qwen3.5-2b | 0 | 84 | 67.1 | 50 / 50 | 1.136 | 1.326 | 0.316 | 0.582 | 7.12 | 0.0186 | `0034_e3_consistency/e3a/full/qwen3.5-2b/seed0` |
| tput_e3a_full | qwen3.5-2b | 0 | 42 | 24.8 | 20 / 20 | 1.155 | 1.206 | 0.339 | 0.583 | 7.12 | 0.0069 | `0034_e3_throughput/e3a/full/qwen3.5-2b/seed0` |
| tput_e3a_full | qwen3.5-9b | 0 | 69 | 50.4 | 20 / 20 | 2.291 | 2.472 | 0.678 | 1.374 | 25.25 | 0.0140 | `0034_e3_throughput/e3a/full/qwen3.5-9b/seed0` |
| tput_e3a_c1 | qwen3.5-2b | 0 | 33 | 18.2 | 20 / 20 | 0.779 | 0.886 | 0.232 | 0.319 | 6.69 | 0.0050 | `0034_e3_throughput/e3a/c1/qwen3.5-2b/seed0` |
| tput_e3a_c1 | qwen3.5-9b | 0 | 48 | 30.6 | 20 / 20 | 1.388 | 1.503 | 0.399 | 0.758 | 24.21 | 0.0085 | `0034_e3_throughput/e3a/c1/qwen3.5-9b/seed0` |
| tput_e3a_c2 | qwen3.5-2b | 0 | 25 | 9.0 | 20 / 20 | 0.382 | 0.419 | 0.153 | 0.013 | 4.93 | 0.0025 | `0034_e3_throughput/e3a/c2/qwen3.5-2b/seed0` |
| tput_e3a_c2 | qwen3.5-9b | 0 | 25 | 9.3 | 20 / 20 | 0.409 | 0.440 | 0.164 | 0.012 | 19.37 | 0.0026 | `0034_e3_throughput/e3a/c2/qwen3.5-9b/seed0` |
| tput_e3b | qwen3.5-2b | 0 | 56 | 41.0 | 20 / 20 | 1.922 | 2.016 | 0.600 | 1.227 | 9.87 | 0.0114 | `0034_e3_throughput/e3b/qwen3.5-2b/seed0` |
| tput_e3b | qwen3.5-9b | 0 | 120 | 102.2 | 20 / 20 | 4.699 | 5.063 | 1.404 | 3.182 | 30.54 | 0.0284 | `0034_e3_throughput/e3b/qwen3.5-9b/seed0` |
| tput_c1prime | qwen3.5-2b | 0 | 40 | 24.8 | 20 / 20 | 1.121 | 1.219 | 0.351 | 0.681 | 8.42 | 0.0069 | `0034_e3_throughput/c1prime/qwen3.5-2b/seed0` |
| tput_c1prime | qwen3.5-9b | 0 | 71 | 55.3 | 20 / 20 | 2.673 | 2.743 | 0.778 | 1.796 | 29.16 | 0.0153 | `0034_e3_throughput/c1prime/qwen3.5-9b/seed0` |
| tput_c3 | qwen3.5-2b | 0 | 39 | 24.6 | 20 / 20 | 1.103 | 1.198 | 0.381 | 0.657 | 10.62 | 0.0068 | `0034_e3_throughput/c3/qwen3.5-2b/seed0` |
| tput_c3 | qwen3.5-9b | 0 | 66 | 50.5 | 20 / 20 | 2.350 | 2.410 | 0.760 | 1.515 | 28.66 | 0.0140 | `0034_e3_throughput/c3/qwen3.5-9b/seed0` |

命令墙钟合计：843 s（0.234 h）。
