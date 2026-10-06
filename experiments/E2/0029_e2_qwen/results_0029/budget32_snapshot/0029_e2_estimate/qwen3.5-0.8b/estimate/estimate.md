# E2 估算 · qwen3.5-0.8b

- 设备：NVIDIA A800-SXM4-80GB；dtype bf16
- 层：all（25 个，另加 L24_pre）
- 参考线：post_proj、pre_proj（pre_proj 那组探针与视觉塔的额外前向都在实测里）
- 含法向：是；任务：depth、semseg、normals、refseg

| 任务 | 上采样实现 | 探针数（pre_proj） | 步 × 批 | 前向 s | 一步 s | 评测一条 s | 评测集 | GPU 小时 | 显存峰值 GB |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| depth | separable_matmul | 54（2） | 3000 × 8 | 0.698 | 0.998 | 1.475 | 654 | 1.49 | 2.8 |
| semseg | separable_matmul | 54（2） | 3000 × 8 | 0.100 | 1.379 | 0.836 | 2000 | 1.81 | 5.1 |
| normals | separable_matmul | 54（2） | 1000 × 8 | 0.111 | 1.392 | 2.873 | 654 | 1.20 | 3.0 |
| refseg | separable_matmul | 54（2） | 3000 × 8 | 0.106 | 1.050 | 0.118 | 10634 | 1.29 | 2.9 |

合计 5.79 GPU 小时（不含 bootstrap 的 CPU 时间与模型加载）。
