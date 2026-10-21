# W1 · qwen3.5-9b · refseg · w1f · devtune

- W1-F：<out>/qwen3.5-9b/refexp_box/baseline_e1/devtune 的框 → src.w1.workbench.grabcut_mask（迭代 5 次）；没有 LLM（compute.json 只有 GrabCut 的墙钟）
- n = 32；失败 {'multiple_boxes': 1, 'ok': 31}

## 标题指标

- cIoU：0.5177，95% CI [0.3818, 0.6281]，n = 32
- gIoU：0.5785，95% CI [0.4686, 0.6677]，n = 32
