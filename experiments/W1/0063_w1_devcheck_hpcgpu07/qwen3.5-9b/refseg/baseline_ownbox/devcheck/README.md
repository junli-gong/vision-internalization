# W1 · qwen3.5-9b · refseg · baseline_ownbox · devcheck

- "自己的框"：<out>/qwen3.5-9b/refexp_box/baseline_e1/devcheck 的框（W1-0 refexp_box，E1 的协议）光栅化（框 = 框记录的 parsed.box（parse_one_box 换算好的原图绝对像素 xyxy，同 E1 定位）；解…）
- n = 500；框解析失败 7 条（空 mask）；框有效 493 条

## 标题指标

- cIoU：0.5074，95% CI [0.4858, 0.5289]，n = 500
- gIoU：0.5229，95% CI [0.5057, 0.5409]，n = 500

- compute.json 是框那次运行的计算量（这里没有模型调用）。
