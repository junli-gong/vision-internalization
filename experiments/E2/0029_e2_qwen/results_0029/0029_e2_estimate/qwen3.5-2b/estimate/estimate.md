# E2 估算 · qwen3.5-2b

- 设备：NVIDIA A800-SXM4-80GB；dtype bf16
- 层：all（25 个，另加 L24_pre）
- 参考线：post_proj、pre_proj（pre_proj 那组探针与视觉塔的额外前向都在实测里）
- 含法向：是；任务：depth、semseg、normals、refseg

| 任务 | 上采样实现 | 探针数（pre_proj） | 步 × 批 | 前向 s | 一步 s | 评测一条 s | 评测集 | GPU 小时 | 显存峰值 GB |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| depth | separable_matmul | 54（2） | 3000 × 8 | 0.980 | 1.269 | 1.470 | 654 | 1.73 | 6.6 |
| semseg | separable_matmul | 54（2） | 3000 × 8 | 0.121 | 1.565 | 0.893 | 2000 | 2.02 | 8.9 |
| normals | separable_matmul | 54（2） | 1000 × 8 | 0.145 | 1.439 | 2.902 | 654 | 1.22 | 6.7 |
| refseg | separable_matmul | 54（2） | 3000 × 8 | 0.137 | 1.077 | 0.141 | 10634 | 1.39 | 6.7 |

合计 6.36 GPU 小时（不含 bootstrap 的 CPU 时间与模型加载）。
