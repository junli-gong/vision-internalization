# W2 · qwen3.5-2b · 候选缓存 · candidates · test_refcocog_umd_test

- 提出：<out>/qwen3.5-2b/refexp_box/baseline_e1/test_refcocog_umd_test（mode = box）
- 生成器（src.w2.candidates，K = 4）：['gc_default', 'gc_soft_box_minus10', 'gc_default+lcc+fill', 'ws_pts']
- n = 1000，有候选 989；失败（ok = false，用框代替）按编号 {'1': 4, '3': 4, '4': 4}；与更小编号完全相同的候选 294（0.0743）
- candidates.jsonl 的 info 含耗时，不逐字节可复现；下游设置的运行身份用 content_sha256（不含 info）。
