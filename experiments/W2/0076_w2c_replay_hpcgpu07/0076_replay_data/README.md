# W2-c 加回放的回放数据（DEV-0065）· complete = True

- general：2000/2000 条，截断 763（0.3815），可用 1237；生成长度 p50 344.5000
- count：2000/2000 条，截断 0（0.0000），可用 2000；生成长度 p50 2.0000
- 通用：databricks/databricks-dolly-15k@bdd27f4d94b9（CC-BY-SA-3.0）；剔除 {'too_long': 157, 'structural_marker': 0, 'eval_overlap': 15, 'duplicate': 209}；与评测重叠的按参照集 {'health_check': 3, 'mmlu_pro': 12}；类别 {'brainstorming': 229, 'classification': 281, 'closed_qa': 197, 'creative_writing': 91, 'general_qa': 307, 'information_extraction': 194, 'open_qa': 541, 'summarization': 160}
- 计数：W2-c 的 10438 张图，(图, 类别) 对 13108；与 CountBench 重合的图 0；选中 2000 条、1934 张图

## 文件

- replay/manifest.json：选择与去重的记录；replay/general.jsonl、replay/count.jsonl：每条一行（提示、图、回答、生成的 token、是否截断）；config.yaml：运行身份；metrics.json：由各行算；compute.json、logs/：记账与计时。训练：scripts/train_w2c.py --config configs/w2c_replay_train.json --replay-data <本目录>。
