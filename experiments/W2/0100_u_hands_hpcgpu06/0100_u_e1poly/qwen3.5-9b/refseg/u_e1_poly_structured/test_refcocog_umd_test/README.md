# 统一主表的"手" · qwen3.5-9b · refseg · u_e1_poly_structured · test_refcocog_umd_test

- 系统：E1 结构强制多边形（9B 零样本；configs/e1_posthoc.json 的 format_forced_poly，变体 poly_structured：至少 3 个点；关思考、贪心、E1 冻结的图像口径与批大小；D72-3）
- 句子：W2 test 子集 300 句（configs/w2_llmseg_subsets.json，sha256 8c8a1d134ab324e0…）；本次 n = 300（limit = None）
- 放行：D154（D154-2(2)：统一主表的新跑基线在 W2 test 的 5 × 300 句子集（configs/w2_llmseg_subsets.json）上运行；预注册 U-2）
- 原运行器的输出：raw/（E1 事后读数（src/e1/posthoc_runner.run_forced_poly_task））

## 标题指标（W2 的计分）

- cIoU：0.3454，95% CI [0.3132, 0.3755]，n = 300
- gIoU：0.3355，95% CI [0.3092, 0.3595]，n = 300

- 失败（按空 mask 计）：{'ok': 300}；解析成功但 mask 为空：18
- 每句计算量：{'n_model_calls': 1.0, 'prompt_tokens': 406.86, 'generated_tokens': 66.51, 'visual_tokens': 312.4066666666667}（compute.json 是 raw 那次运行的记账）

## 文件

- predictions.jsonl（逐句：失败、mask 的 RLE、score、调用与 token）；metrics.json；compute.json；config.yaml；meta.json；logs/derive_<k>.json；raw/（原运行器的输出原样）。
