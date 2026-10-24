# W2 · qwen3.5-9b · 候选缓存 · candidates · devtune800

- 提出：<out>/qwen3.5-9b/refexp_box/baseline_e1/devtune800（mode = box）
- 生成器（src.w2.candidates，K = 8）：['gc_default', 'gc_box_plus10', 'gc_soft_box_minus10', 'gc_default+lcc+fill', 'slic_s30_t50', 'slic_s100_t50', 'ws_pts', 'box']
- n = 800，有候选 786；失败（ok = false，用框代替）按编号 {'2': 1}；与更小编号完全相同的候选 271（0.0431）
- candidates.jsonl 的 info 含耗时，不逐字节可复现；下游设置的运行身份用 content_sha256（不含 info）。
