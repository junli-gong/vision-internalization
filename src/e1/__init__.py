"""E1 评测框架（T 文本通道零样本基线；简报 §8.2；DEV-0014）。

- prompts：提示配置 configs/e1_prompts.json 的读取、校验、sha256 与逐条渲染（含 CountBench 问句的生成规则）；
- tasks：9 个任务的登记（任务 → 解析方式、数据与各子集的样本选择）；
- scoring：逐条解析与评分、诊断量、metrics.json 的汇总（指标一律用 src/metrics 与 src/stats）；
- runner：逐模型、逐任务的运行（按批写 predictions.jsonl、断点续传、metrics/compute/config/README）；
- acceptance：批处理验收（D28-5：batch=1 与批量各跑一遍，比较解析后的答案、指标差与 raw_text；答案一致按逐位、IoU ≥ 0.9、
  按判定结果三种口径都算，N15 / DEV-0015）；
- estimate：吞吐与算力估算（按真实提示与生成上限测样本/秒，外推到预注册的 n）。
事后读数（DEV-0033；D66；E1 预注册 §10；口径在 configs/e1_posthoc.json，主评测的代码路径与输出不变）：
- posthoc：事后配置的读取、校验与对冻结提示配置的核对；
- debiased：去位置偏好的字母读数（选项拆分与循环移位、k 种顺序的合成、指标）；
- forced：格式强制诊断（0/1/换行的允许集合、结构强制的 logits 处理器、固定长度的入口断言）；
- posthoc_runner：两项事后读数的运行（批、续跑、输出）。汇总：scripts/summarize_e1_posthoc.py。
入口：scripts/run_e1.py。
"""
