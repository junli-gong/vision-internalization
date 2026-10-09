# E3b 指令检查 · swap · e3b · qwen3.5-2b · seed 0 · refseg

- 检查点 manifest sha256：acb7f58d1ae3aa0c66ed7ba3c57b03d45131b6a3476b5f87b5211f451ceded30
- 规则：候选 = 评测总体（load_eval_set 的全部样本，不受 --limit 影响）里同一张图（fields.image_id）、不同标注 id（fields.ann_id）的全部指代句，按评测总体的顺序排列；取第 k 个，k = sha256('visint-e3b-instr-swap|<SWAP_SEED>|<样本 id>') 的前 8 字节（大端整数）mod 候选数；没有候选的样本跳过并计数。指令 = 同一模板（tasks.refseg.grid_template）填候选的指代句
- 样本：前 2048 条，送进模型 2014 条，跳过 34 条；配对 2014 条
- 正常评测一侧：predictions.jsonl（sha256 66cdc017c3a0f6d4…；解码 restricted）；与它的 scores.jsonl 核对一致 2014 条
- 上采样：guided；配对差 = 正常 − 检查（同一组 id、同一套评分，配对 bootstrap）。original：两侧都对原来的物体评分（模型跟着指令走时差为正）；swapped（只有 swap）：两侧都对换上去的物体评分（跟着指令走时检查一侧更高，差为负）。只用默认上采样

| 目标 | 检查 cIoU | 正常 cIoU | 检查 gIoU | 正常 gIoU | 正常 − 检查 cIoU | 正常 − 检查 gIoU |
|---|---|---|---|---|---|---|
| original | 0.0199 [0.0157, 0.0245] | 0.5340 [0.5148, 0.5538] | 0.0250 [0.0196, 0.0306] | 0.5222 [0.5065, 0.5371] | 0.5141 [0.4944, 0.5347] | 0.4973 [0.4813, 0.5138] |
| swapped | 0.5334 [0.5138, 0.5530] | 0.0193 [0.0157, 0.0238] | 0.5217 [0.5071, 0.5378] | 0.0251 [0.0200, 0.0306] | -0.5141 [-0.5342, -0.4948] | -0.4966 [-0.5138, -0.4816] |

## 文件

- predictions.jsonl：本次的单轮输出；scores.jsonl：每条在各目标上两侧的统计；metrics.json：汇总；
  swap_pairs.jsonl（swap）：每条样本的候选或跳过；compute.json；config.yaml：运行身份；logs/。
