# E3b 评测 · e3b · qwen3.5-4b · seed 2 · depth

- 检查点 manifest sha256：dca97e758611baef5cda924b359ddae503128e78a2f7380e2ac42286a5828f12
- 模型：Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a（qwen3.5；bf16；tiny = False）
- 评测集：n = 654（id 指纹 4f06d8a06cfd3234…）
- 网格 32×32，max_new_tokens = 1057，批大小 64，允许集合 66 个 token
- 上采样：默认 guided；另报 bilinear, cells

## 主要数字：单轮（guided）

- abs_rel：0.3384 [0.3209, 0.3555]
- d1：0.5126 [0.4939, 0.5327]
- d1_strict：0.5126 [0.4939, 0.5327]

## 多轮与匹配计算量的单轮（guided；95% CI）

| 流程 | abs_rel | d1 | d1_strict | 有效率 | 调用/条 | 轮/条 | 生成 token/条 | 视觉 token/条 | 记账文件 |
|---|---|---|---|---|---|---|---|---|---|
| 多轮 K = 1 | 0.3384 [0.3209, 0.3555] | 0.5126 [0.4939, 0.5327] | 0.5126 [0.4939, 0.5327] | 1.0000 | 1.0000 | 1.0000 | 1057.0000 | 300.0000 | compute_multi_K1.json |

- 本次只跑 K = [1]（--k-values 覆盖配置的 [1, 2, 4]；D72-4 (c)：多轮与匹配计算量的评测每个尺寸只做 1 个种子，单轮的主读数保留全部种子；其余种子用 scripts/eval_e3b.py --k-values 只跑配置那一列的前缀（第 1 轮 = 单轮与跑全部 K 时逐字节相同））

- 解码：structured（结构强制（D83，DEV-0041）：第 r 行的格子位置只允许第 r 行的值 token，行末只允许行分隔，最后一个位置只允许停止 token；长度恒为 n(n + 1) + 1；只强制网格的形状，不注入任务知识。经 generate_batch(allowed_token_ids = 任务的允许集合, logits_processor = [StructuredGridProcessor]) 实现（src/grid/structured_decode.py））

- 评测前合并核对的判据：e5-v2（评测前的合并核对按 e5-v2 判（src/e3/merge_check.rejudge_e5v2 → src/e5/trained_eval.judge_e5v2，同一个函数）；读数照旧来自 scripts/check_training_interfaces.py --checkpoint；v1 的判定只报告（merge_check.json 的 v1_report_only）；D102-2（N37(b)，项目负责人 2026-10-19）：E3 的结构强制网格评测统一到 e5-v2，补评被拒的种子（4B E3b 种子 2、2B E3b 种子 1，连同机内配对所需的重评）；按 N34 统一判据，决定于看到被拒种子的读数之前）

- K = 1 的多轮与单轮是同一次调用（逐个相同）；匹配的单轮 = 贪心 + K − 1 次受限采样，逐格合成（规则见 src/e3/evaluate.py）
- GPU 墙钟（call_time_s）与端到端墙钟（wall_time_s）在记账文件里（compute*.json）；本表只列由输出决定的计数

## 文件

- predictions.jsonl（第 1 轮 = 单轮）、predictions_round<r>.jsonl、predictions_sample<s>.jsonl：逐条输出；scores.jsonl：逐条指标；
- metrics.json：全部汇总；compute*.json：各流程的记账（compute_by_stage.json 是每轮单独的）；config.yaml：运行身份；
  meta.json；logs/：墙钟与每次调用的记录。
