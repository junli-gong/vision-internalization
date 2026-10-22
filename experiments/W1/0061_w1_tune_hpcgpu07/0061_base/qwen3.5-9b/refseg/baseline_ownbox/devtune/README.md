# W1 · qwen3.5-9b · refseg · baseline_ownbox · devtune

- "自己的框"：<out>/qwen3.5-9b/refexp_box/baseline_e1/devtune 的框（W1-0 refexp_box，E1 的协议）光栅化（框 = 框记录的 parsed.box（parse_one_box 换算好的原图绝对像素 xyxy，同 E1 定位）；解…）
- n = 96；框解析失败 4 条（空 mask）；框有效 92 条

## 标题指标

- cIoU：0.4670，95% CI [0.4081, 0.5252]，n = 96
- gIoU：0.5013，95% CI [0.4601, 0.5433]，n = 96

- compute.json 是框那次运行的计算量（这里没有模型调用）。
