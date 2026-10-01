# E1 · internvl3.5-4b-hf · rel_depth_nyu · dev

- 模型：OpenGVLab/InternVL3_5-4B-HF@6bd4487402110ef9889ba50eb7aefeb302526fed（internvl3.5；适配器 internvl3.5-4b-hf）
- 图像口径：{'tag': 'default', 'type': 'TilePolicy', 'crop_to_patches': False, 'max_patches': 1}
- 生成：enable_thinking=False，max_new_tokens=64，greedy，image_first
- 提示配置 sha256：974d3d730e903a6f0a0727a13fceaa52437b3a048914b90836f8b918afcdc0fa；模板：`'Two points are circled on the image, labeled by A and B beside each circle. Which point is closer to the camera?\nSelect from the following choices.\n(A) A is closer\n(B) B is closer'`
- 样本数 n = 100（id 指纹 b1605ec56f9cd8d7…）；有 GT：True

## 标题指标

- accuracy（all）：0.0000，95% CI [0.0000, 0.0000]，n = 100

## 解析与截断

- 解析失败率：1.0000（100/100）；分原因：{'truncated': 100}
- 截断率：1.0000；思考段未闭合：0.0000

## 文件

- predictions.jsonl：逐条输出、解析与得分；metrics.json：全部指标（由 predictions.jsonl 算出）；
- compute.json：计算记账（src/accounting.py）；config.yaml：运行身份；meta.json：git、主机、包版本；logs/：墙钟与每次调用的记录。
