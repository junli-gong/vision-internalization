# W2 · qwen3.5-9b · 候选缓存 · candidates · test_refcoco+_unc_testB

- 提出：<out>/qwen3.5-9b/refexp_box/baseline_e1/test_refcoco+_unc_testB（mode = box）
- 生成器（src.w2.candidates，K = 4）：['gc_default', 'gc_soft_box_minus10', 'gc_default+lcc+fill', 'ws_pts']
- n = 1000，有候选 955；失败（ok = false，用框代替）按编号 {'1': 1, '3': 1, '4': 1}；与更小编号完全相同的候选 310（0.0812）
- candidates.jsonl 的 info 含耗时，不逐字节可复现；下游设置的运行身份用 content_sha256（不含 info）。
