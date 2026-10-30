# 统一主表的"手" · qwen3.5-9b · refseg · u_e3b_grid · test_refcoco_unc_testB

- 系统：E3b 像素 token 网格（检查点的单轮、结构强制解码、guided 上采样；D83-3；同 E3 表的 refseg 读法）
- 句子：W2 test 子集 300 句（configs/w2_llmseg_subsets.json，sha256 9b35912cbe696756…）；本次 n = 300（limit = None）
- 放行：D154（D154-2(2)：统一主表的新跑基线在 W2 test 的 5 × 300 句子集（configs/w2_llmseg_subsets.json）上运行；预注册 U-2）
- 原运行器的输出：raw/eval/e3b/qwen3.5-9b/seed0/refseg__structured/（E3b 评测（src/e3/evaluate.run_eval_task））
- 检查点：manifest sha256 316d25c7bf0b21edd6355a55bc0201249078b7c59305563dd3e25082c7c088db（变体 e3b，seed 0）

## 标题指标（W2 的计分）

- cIoU：0.5760，95% CI [0.5165, 0.6335]，n = 300
- gIoU：0.5604，95% CI [0.5165, 0.6072]，n = 300

- 失败（按空 mask 计）：{'ok': 300}；解析成功但 mask 为空：63
- 每句计算量：{'n_model_calls': 1.0, 'prompt_tokens': 344.87333333333333, 'generated_tokens': 1057.0, 'visual_tokens': 310.9766666666667}（compute.json 是 raw 那次运行的记账）

## 文件

- predictions.jsonl（逐句：失败、mask 的 RLE、score、调用与 token）；metrics.json；compute.json；config.yaml；meta.json；logs/derive_<k>.json；raw/（原运行器的输出原样）。
