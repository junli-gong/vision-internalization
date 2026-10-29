# W2 · qwen3.5-9b · 候选缓存 · candidates_w2c · explore_val_refcoco+_unc_val

- 提出：<out>/qwen3.5-9b/refexp_box/propose_w2c/explore_val_refcoco+_unc_val（mode = box）
- 生成器（src.w2.candidates，K = 4）：['gc_default', 'gc_soft_box_minus10', 'gc_default+lcc+fill', 'ws_pts']
- n = 400，有候选 400；失败（ok = false，用框代替）按编号 {}；与更小编号完全相同的候选 92（0.0575）
- candidates.jsonl 的 info 含耗时，不逐字节可复现；下游设置的运行身份用 content_sha256（不含 info）。
