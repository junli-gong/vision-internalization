# W1 · qwen3.5-9b · refseg · w1f · devcheck

- W1-F：<out>/qwen3.5-9b/refexp_box/baseline_e1/devcheck 的框 → src.w1.workbench.grabcut_mask（迭代 5 次）；没有 LLM（compute.json 只有 GrabCut 的墙钟）
- n = 500；失败 {'multiple_boxes': 7, 'ok': 493}

## 标题指标

- cIoU：0.5757，95% CI [0.5435, 0.6084]，n = 500
- gIoU：0.5512，95% CI [0.5250, 0.5791]，n = 500
