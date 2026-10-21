# W1 · qwen3.5-9b · refseg · baseline_ownbox · devtune

- "自己的框"：<out>/qwen3.5-9b/refexp_box/baseline_e1/devtune 的框（W1-0 refexp_box，E1 的协议）光栅化（框 = 框记录的 parsed.box（parse_one_box 换算好的原图绝对像素 xyxy，同 E1 定位）；解…）
- n = 32；框解析失败 1 条（空 mask）；框有效 31 条

## 标题指标

- cIoU：0.4869，95% CI [0.3718, 0.6095]，n = 32
- gIoU：0.5531，95% CI [0.4826, 0.6146]，n = 32

- compute.json 是框那次运行的计算量（这里没有模型调用）。
