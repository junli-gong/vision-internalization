# E1 · internvl3.5-2b-hf · spatial_embspatial · dev

- 模型：OpenGVLab/InternVL3_5-2B-HF@3f301ffcf3dcbb47893afae6650ea3e78d96fb6d（internvl3.5；适配器 internvl3.5-2b-hf）
- 图像口径：{'tag': 'default', 'type': 'TilePolicy', 'crop_to_patches': False, 'max_patches': 1}
- 生成：enable_thinking=False，max_new_tokens=64，greedy，image_first
- 提示配置 sha256：974d3d730e903a6f0a0727a13fceaa52437b3a048914b90836f8b918afcdc0fa；模板：`'Question: {question}\nOptions:\n{options}\nPlease select the correct answer from the options above.'`
- 样本数 n = 50（id 指纹 a4453369d042eb01…）；有 GT：True

## 标题指标

- accuracy（all）：0.1400，95% CI [0.0600, 0.2400]，n = 50

## 解析与截断

- 解析失败率：0.6200（31/50）；分原因：{'truncated': 31}
- 截断率：0.6200；思考段未闭合：0.0000

## 文件

- predictions.jsonl：逐条输出、解析与得分；metrics.json：全部指标（由 predictions.jsonl 算出）；
- compute.json：计算记账（src/accounting.py）；config.yaml：运行身份；meta.json：git、主机、包版本；logs/：墙钟与每次调用的记录。
