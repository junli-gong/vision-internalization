# E2 估算 · qwen3.5-0.8b

- 设备：NVIDIA A800-SXM4-80GB；dtype bf16
- 层：all（25 个，另加 L24_pre）
- 参考线：post_proj、pre_proj（pre_proj 那组探针与视觉塔的额外前向都在实测里）
- 含法向：是；任务：depth、semseg、normals、refseg

| 任务 | 上采样实现 | 探针数（pre_proj） | 步 × 批 | 前向 s | 一步 s | 评测一条 s | 评测集 | GPU 小时 | 显存峰值 GB |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| depth | separable_matmul | 54（2） | 3000 × 8 | 0.666 | 0.963 | 1.515 | 654 | 1.48 | 2.8 |
| semseg | separable_matmul | 54（2） | 3000 × 8 | 0.106 | 1.403 | 0.839 | 2000 | 1.84 | 5.1 |
| normals | separable_matmul | 54（2） | 1000 × 8 | 0.107 | 1.283 | 2.846 | 654 | 1.16 | 3.0 |
| refseg | separable_matmul | 54（2） | 3000 × 8 | 0.099 | 1.043 | 0.117 | 10634 | 1.28 | 2.9 |

合计 5.75 GPU 小时（不含 bootstrap 的 CPU 时间与模型加载）。
