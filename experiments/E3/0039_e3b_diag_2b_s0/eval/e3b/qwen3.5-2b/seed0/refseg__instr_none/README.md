# E3b 指令检查 · none · e3b · qwen3.5-2b · seed 0 · refseg

- 检查点 manifest sha256：acb7f58d1ae3aa0c66ed7ba3c57b03d45131b6a3476b5f87b5211f451ceded30
- 规则：去掉指令：指令换成 E3a 的 no_instruction（configs/e3a_train.json 的 eval.leak_checks.no_instruction.instruction，由 src/readout/evaluate.resolve_eval 读出），user 轮里只剩图像；允许集合与解码与正常评测相同
- 样本：前 2048 条，送进模型 2048 条，跳过 0 条；配对 2048 条
- 正常评测一侧：predictions.jsonl（sha256 66cdc017c3a0f6d4…；解码 restricted）；与它的 scores.jsonl 核对一致 2048 条
- 上采样：guided；配对差 = 正常 − 检查（同一组 id、同一套评分，配对 bootstrap）。original：两侧都对原来的物体评分（模型跟着指令走时差为正）；swapped（只有 swap）：两侧都对换上去的物体评分（跟着指令走时检查一侧更高，差为负）。只用默认上采样

| 目标 | 检查 cIoU | 正常 cIoU | 检查 gIoU | 正常 gIoU | 正常 − 检查 cIoU | 正常 − 检查 gIoU |
|---|---|---|---|---|---|---|
| original | 0.1378 [0.1215, 0.1548] | 0.5314 [0.5095, 0.5498] | 0.1162 [0.1055, 0.1283] | 0.5222 [0.5064, 0.5375] | 0.3937 [0.3717, 0.4138] | 0.4060 [0.3889, 0.4222] |

## 文件

- predictions.jsonl：本次的单轮输出；scores.jsonl：每条在各目标上两侧的统计；metrics.json：汇总；
  swap_pairs.jsonl（swap）：每条样本的候选或跳过；compute.json；config.yaml：运行身份；logs/。
