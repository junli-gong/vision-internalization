# E2 估算 · qwen3.5-9b

- 设备：NVIDIA A800-SXM4-80GB；dtype bf16
- 层：every 2（17 个，另加 L32_pre）
- 参考线：post_proj、pre_proj（pre_proj 那组探针与视觉塔的额外前向都在实测里）
- 含法向：是；任务：depth、semseg、normals、refseg

| 任务 | 上采样实现 | 探针数（pre_proj） | 步 × 批 | 前向 s | 一步 s | 评测一条 s | 评测集 | GPU 小时 | 显存峰值 GB |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| depth | separable_matmul | 38（2） | 3000 × 8 | 0.714 | 1.241 | 1.106 | 654 | 1.54 | 21.9 |
| semseg | separable_matmul | 38（2） | 3000 × 8 | 0.161 | 1.526 | 0.676 | 2000 | 1.82 | 24.1 |
| normals | separable_matmul | 38（2） | 1000 × 8 | 0.157 | 1.405 | 2.129 | 654 | 1.00 | 21.9 |
| refseg | separable_matmul | 38（2） | 3000 × 8 | 0.163 | 1.257 | 0.149 | 10634 | 1.57 | 21.9 |

合计 5.92 GPU 小时（不含 bootstrap 的 CPU 时间与模型加载）。
