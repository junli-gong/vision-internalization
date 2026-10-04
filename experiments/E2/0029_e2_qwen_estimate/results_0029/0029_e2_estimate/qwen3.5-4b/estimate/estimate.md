# E2 估算 · qwen3.5-4b

- 设备：NVIDIA A800-SXM4-80GB；dtype bf16
- 层：all（33 个，另加 L32_pre）
- 参考线：post_proj、pre_proj（pre_proj 那组探针与视觉塔的额外前向都在实测里）
- 含法向：是；任务：depth、semseg、normals、refseg

| 任务 | 上采样实现 | 探针数（pre_proj） | 步 × 批 | 前向 s | 一步 s | 评测一条 s | 评测集 | GPU 小时 | 显存峰值 GB |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| depth | separable_matmul | 70（2） | 3000 × 8 | 0.738 | 1.209 | 1.948 | 654 | 1.86 | 12.6 |
| semseg | separable_matmul | 70（2） | 3000 × 8 | 0.156 | 1.925 | 1.158 | 2000 | 2.53 | 15.0 |
| normals | separable_matmul | 70（2） | 1000 × 8 | 0.183 | 1.537 | 3.846 | 654 | 1.51 | 12.7 |
| refseg | separable_matmul | 70（2） | 3000 × 8 | 0.161 | 1.313 | 0.162 | 10634 | 1.66 | 12.7 |

合计 7.56 GPU 小时（不含 bootstrap 的 CPU 时间与模型加载）。
