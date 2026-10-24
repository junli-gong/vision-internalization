# W2 · qwen3.5-9b · 候选缓存 · candidates_box_points · devtune800

- 提出：<out>/qwen3.5-9b/refseg/propose_box_points/devtune800（mode = box_points）
- 生成器（src.w2.candidates，K = 4）：['gc_default', 'gc_pts', 'gc_default+lcc+fill', 'ws_pts']
- n = 800，有候选 94；失败（ok = false，用框代替）按编号 {'2': 13}；与更小编号完全相同的候选 37（0.0984）
- candidates.jsonl 的 info 含耗时，不逐字节可复现；下游设置的运行身份用 content_sha256（不含 info）。
