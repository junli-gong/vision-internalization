# W2 · qwen3.5-9b · 候选缓存 · candidates_w2c · devcheck2

- 提出：<out>/qwen3.5-9b/refexp_box/propose_w2c/devcheck2（mode = box）
- 生成器（src.w2.candidates，K = 4）：['gc_default', 'gc_soft_box_minus10', 'gc_default+lcc+fill', 'ws_pts']
- n = 500，有候选 500；失败（ok = false，用框代替）按编号 {}；与更小编号完全相同的候选 131（0.0655）
- candidates.jsonl 的 info 含耗时，不逐字节可复现；下游设置的运行身份用 content_sha256（不含 info）。
