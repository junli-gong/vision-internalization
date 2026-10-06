# E2 估算 · qwen3.5-2b

- 设备：NVIDIA A800-SXM4-80GB；dtype bf16
- 层：all（25 个，另加 L24_pre）
- 参考线：post_proj、pre_proj（pre_proj 那组探针与视觉塔的额外前向都在实测里）
- 含法向：是；任务：depth、semseg、normals、refseg

| 任务 | 上采样实现 | 探针数（pre_proj） | 步 × 批 | 前向 s | 一步 s | 评测一条 s | 评测集 | GPU 小时 | 显存峰值 GB |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| depth | separable_matmul | 54（2） | 3000 × 8 | 0.691 | 1.147 | 1.451 | 654 | 1.60 | 6.6 |
| semseg | separable_matmul | 54（2） | 3000 × 8 | 0.131 | 1.534 | 0.847 | 2000 | 1.95 | 8.9 |
| normals | separable_matmul | 54（2） | 1000 × 8 | 0.143 | 1.468 | 2.895 | 654 | 1.23 | 6.7 |
| refseg | separable_matmul | 54（2） | 3000 × 8 | 0.131 | 1.163 | 0.133 | 10634 | 1.43 | 6.7 |

合计 6.22 GPU 小时（不含 bootstrap 的 CPU 时间与模型加载）。
