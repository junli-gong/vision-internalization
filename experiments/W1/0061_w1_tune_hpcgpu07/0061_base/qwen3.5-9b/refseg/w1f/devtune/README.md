# W1 · qwen3.5-9b · refseg · w1f · devtune

- W1-F：<out>/qwen3.5-9b/refexp_box/baseline_e1/devtune 的框 → src.w1.workbench.grabcut_mask（迭代 5 次）；没有 LLM（compute.json 只有 GrabCut 的墙钟）
- n = 96；失败 {'multiple_boxes': 3, 'ok': 92, 'truncated': 1}

## 标题指标

- cIoU：0.5222，95% CI [0.4572, 0.5948]，n = 96
- gIoU：0.5304，95% CI [0.4729, 0.5895]，n = 96
