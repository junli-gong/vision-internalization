# E1 · internvl3.5-2b-hf · count · dev

- 模型：OpenGVLab/InternVL3_5-2B-HF@3f301ffcf3dcbb47893afae6650ea3e78d96fb6d（internvl3.5；适配器 internvl3.5-2b-hf）
- 图像口径：{'tag': 'default', 'type': 'TilePolicy', 'crop_to_patches': False, 'max_patches': 1}
- 生成：enable_thinking=False，max_new_tokens=64，greedy，image_first
- 提示配置 sha256：974d3d730e903a6f0a0727a13fceaa52437b3a048914b90836f8b918afcdc0fa；模板：`'How many {object} are there in the image? Answer with a single number.'`
- 样本数 n = 49（id 指纹 9cc83f756324ac38…）；有 GT：True

## 标题指标

- accuracy（all）：0.4490，95% CI [0.3061, 0.5918]，n = 49

## 解析与截断

- 解析失败率：0.2857（14/49）；分原因：{'thinking_unclosed': 3, 'truncated': 11}
- 截断率：0.2857；思考段未闭合：0.0612

## 文件

- predictions.jsonl：逐条输出、解析与得分；metrics.json：全部指标（由 predictions.jsonl 算出）；
- compute.json：计算记账（src/accounting.py）；config.yaml：运行身份；meta.json：git、主机、包版本；logs/：墙钟与每次调用的记录。
