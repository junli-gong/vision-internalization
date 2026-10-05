# E2 估算 · qwen3.5-9b

- 设备：NVIDIA A800-SXM4-80GB；dtype bf16
- 层：every 2（17 个，另加 L32_pre）
- 参考线：post_proj、pre_proj（pre_proj 那组探针与视觉塔的额外前向都在实测里）
- 含法向：是；任务：depth、semseg、normals、refseg

| 任务 | 上采样实现 | 探针数（pre_proj） | 步 × 批 | 前向 s | 一步 s | 评测一条 s | 评测集 | GPU 小时 | 显存峰值 GB |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| depth | separable_matmul | 38（2） | 3000 × 8 | 0.730 | 1.218 | 1.102 | 654 | 1.52 | 21.9 |
| semseg | separable_matmul | 38（2） | 3000 × 8 | 0.166 | 1.542 | 0.666 | 2000 | 1.82 | 24.1 |
| normals | separable_matmul | 38（2） | 1000 × 8 | 0.162 | 1.473 | 2.246 | 654 | 1.05 | 21.9 |
| refseg | separable_matmul | 38（2） | 3000 × 8 | 0.215 | 1.269 | 0.150 | 10634 | 1.59 | 21.9 |

合计 5.98 GPU 小时（不含 bootstrap 的 CPU 时间与模型加载）。
