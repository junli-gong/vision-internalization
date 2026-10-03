# E1 · internvl3.5-4b-hf · ovd · extra

- 模型：OpenGVLab/InternVL3_5-4B-HF@6bd4487402110ef9889ba50eb7aefeb302526fed（internvl3.5；适配器 internvl3.5-4b-hf）
- 图像口径：{'tag': 'internvl:native', 'type': 'TilePolicy', 'crop_to_patches': True, 'max_patches': 12}
- 生成：enable_thinking=False，max_new_tokens=2048，greedy，image_first；读法 text（解析 boxes）
- 批配置：batch_size=64、left 填充、顺序 selection、same_letters；冻结的协议配置
- 提示配置 sha256：5de1aec711fad0140db0cf27094d66e460cf2999ffcd5456f1964bc5b9851158；模板：`'Please detect and label all objects in the following image that belong to these categories, and mark their positions: {categories}.'`
- 样本数 n = 100（id 指纹 5278c31692ebde37…）；有 GT：True

## 标题指标（严格读数）

- F1@0.5（coco_val2017）：0.1151，95% CI [0.0873, 0.1437]，n = 100

## 三列：严格读数 / 有效率 / 条件读数（规则见 configs/e1_prompts.json 的 report_columns）

| 指标 | 范围 | 严格读数 | 95% CI | 有效率 | 有效 / 全部 | 条件读数 | 95% CI |
|---|---|---|---|---|---|---|---|
| F1@0.5 | coco_val2017 | 0.1151 | [0.0873, 0.1437] | 0.8000 | 80 / 100 | 0.1246 | [0.0963, 0.1549] |

- 无效 20 条 = 拒答 0 + 其他解析失败 20

## 解析与截断

- 解析失败率：0.2000（20/100）；分原因：{'no_box': 16, 'truncated': 4}
- 截断率：0.0400；思考段未闭合：0.0000

## 失败的诊断（只作诊断，失败原因不变；规则见 src/e1/scoring.failure_detail）

- 失败的样子：（无）
- 关思考时自发写出思考标记：0 条（其中思考段未闭合 0 条）

## 文件

- predictions.jsonl：逐条输出、解析与得分；metrics.json：全部指标（由 predictions.jsonl 算出）；
- compute.json：计算记账（src/accounting.py）；config.yaml：运行身份；meta.json：git、主机、包版本；logs/：墙钟与每次调用的记录。
