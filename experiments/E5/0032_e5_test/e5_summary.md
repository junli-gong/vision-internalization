# RUN-0032 E5 test 汇总（取自各 metrics.json 原值）

## 1. 逐模型 × 系统 × 任务（标题口径：a 只算 30° 以上三档，b 只算环绕 [20,180] 三档，c、d、e 全部样本）

| 模型 | 系统 | 任务 | n | 有效率 | 解析失败（原因：条数） | 截断率 | thinking_end（自然/强制/没开/未闭合） | 标题 n | IoU 严格 | 95% CI | IoU 条件 | 起伏 严格 | 95% CI | 起伏 条件 | 起伏 条件中位数 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3.5-2b | zeroshot | a rotation3d | 400 | 0.0000 | truncated: 400 | 1.0000 | — | 300 | 0.0000 | [0.0000, 0.0000] | — | 2.0000 | [2.0000, 2.0000] | — | — |
| qwen3.5-2b | zeroshot | b viewpoint | 800 | 0.0000 | truncated: 800 | 1.0000 | — | 315 | 0.0000 | [0.0000, 0.0000] | — | 2.0000 | [2.0000, 2.0000] | — | — |
| qwen3.5-2b | zeroshot | c occlusion | 300 | 0.0000 | truncated: 299, illegal_char: 1 | 0.9967 | — | 300 | 0.0000 | [0.0000, 0.0000] | — | — | — | — | — |
| qwen3.5-2b | zeroshot | d physics | 400 | 0.0000 | truncated: 400 | 1.0000 | — | 400 | 0.0000 | [0.0000, 0.0000] | — | — | — | — | — |
| qwen3.5-2b | zeroshot | e imagery | 400 | 0.0000 | truncated: 400 | 1.0000 | — | 400 | 0.0000 | [0.0000, 0.0000] | — | — | — | — | — |
| qwen3.5-2b | cot | a rotation3d | 400 | 0.0000 | truncated: 400 | 1.0000 | 0/400/0/0 | 300 | 0.0000 | [0.0000, 0.0000] | — | 2.0000 | [2.0000, 2.0000] | — | — |
| qwen3.5-2b | cot | b viewpoint | 800 | 0.0000 | truncated: 797, no_grid: 2, wrong_row_count: 1 | 0.9962 | 0/800/0/0 | 315 | 0.0000 | [0.0000, 0.0000] | — | 2.0000 | [2.0000, 2.0000] | — | — |
| qwen3.5-2b | cot | c occlusion | 300 | 0.6400 | truncated: 105, malformed: 2, ambiguous: 1 | 0.3500 | 2/298/0/0 | 300 | 0.1024 | [0.0851, 0.1209] | 0.1600 | — | — | — | — |
| qwen3.5-2b | cot | d physics | 400 | 0.1550 | truncated: 293, too_few_points: 44, ambiguous: 1 | 0.7325 | 4/396/0/0 | 400 | 0.0029 | [0.0006, 0.0060] | 0.0189 | — | — | — | — |
| qwen3.5-2b | cot | e imagery | 400 | 0.0000 | truncated: 399, too_few_points: 1 | 0.9975 | 0/400/0/0 | 400 | 0.0000 | [0.0000, 0.0000] | — | — | — | — | — |
| qwen3.5-0.8b | zeroshot | a rotation3d | 400 | 0.0000 | truncated: 400 | 1.0000 | — | 300 | 0.0000 | [0.0000, 0.0000] | — | 2.0000 | [2.0000, 2.0000] | — | — |
| qwen3.5-0.8b | zeroshot | b viewpoint | 800 | 0.0000 | truncated: 798, no_grid: 2 | 0.9975 | — | 315 | 0.0000 | [0.0000, 0.0000] | — | 2.0000 | [2.0000, 2.0000] | — | — |
| qwen3.5-0.8b | zeroshot | c occlusion | 300 | 0.0000 | truncated: 300 | 1.0000 | — | 300 | 0.0000 | [0.0000, 0.0000] | — | — | — | — | — |
| qwen3.5-0.8b | zeroshot | d physics | 400 | 0.0000 | truncated: 400 | 1.0000 | — | 400 | 0.0000 | [0.0000, 0.0000] | — | — | — | — | — |
| qwen3.5-0.8b | zeroshot | e imagery | 400 | 0.0000 | truncated: 400 | 1.0000 | — | 400 | 0.0000 | [0.0000, 0.0000] | — | — | — | — | — |
| qwen3.5-0.8b | cot | a rotation3d | 400 | 0.0000 | truncated: 400 | 1.0000 | 0/400/0/0 | 300 | 0.0000 | [0.0000, 0.0000] | — | 2.0000 | [2.0000, 2.0000] | — | — |
| qwen3.5-0.8b | cot | b viewpoint | 800 | 0.0000 | truncated: 800 | 1.0000 | 0/800/0/0 | 315 | 0.0000 | [0.0000, 0.0000] | — | 2.0000 | [2.0000, 2.0000] | — | — |
| qwen3.5-0.8b | cot | c occlusion | 300 | 0.2833 | truncated: 155, malformed: 28, ambiguous: 22, too_few_points: 9, out_of_range: 1 | 0.5167 | 23/277/0/0 | 300 | 0.0160 | [0.0092, 0.0234] | 0.0565 | — | — | — | — |
| qwen3.5-0.8b | cot | d physics | 400 | 0.1100 | truncated: 270, too_few_points: 51, ambiguous: 21, out_of_range: 11, malformed: 3 | 0.6750 | 58/342/0/0 | 400 | 0.0001 | [0.0000, 0.0002] | 0.0007 | — | — | — | — |
| qwen3.5-0.8b | cot | e imagery | 400 | 0.0025 | truncated: 398, ambiguous: 1 | 0.9950 | 0/400/0/0 | 400 | 0.0000 | [0.0000, 0.0000] | 0.0000 | — | — | — | — |
| qwen3.5-4b | zeroshot | a rotation3d | 400 | 0.0000 | truncated: 307, wrong_row_count: 93 | 0.7675 | — | 300 | 0.0000 | [0.0000, 0.0000] | — | 2.0000 | [2.0000, 2.0000] | — | — |
| qwen3.5-4b | zeroshot | b viewpoint | 800 | 0.0000 | truncated: 672, wrong_row_count: 120, wrong_col_count: 8 | 0.8400 | — | 315 | 0.0000 | [0.0000, 0.0000] | — | 2.0000 | [2.0000, 2.0000] | — | — |
| qwen3.5-4b | zeroshot | c occlusion | 300 | 0.0000 | wrong_row_count: 232, truncated: 68 | 0.2267 | — | 300 | 0.0000 | [0.0000, 0.0000] | — | — | — | — | — |
| qwen3.5-4b | zeroshot | d physics | 400 | 0.0000 | truncated: 271, wrong_row_count: 129 | 0.6775 | — | 400 | 0.0000 | [0.0000, 0.0000] | — | — | — | — | — |
| qwen3.5-4b | zeroshot | e imagery | 400 | 0.0700 | wrong_row_count: 359, truncated: 13 | 0.0325 | — | 400 | 0.0001 | [0.0000, 0.0004] | 0.0018 | — | — | — | — |
| qwen3.5-4b | cot | a rotation3d | 400 | 0.0000 | truncated: 400 | 1.0000 | 0/400/0/0 | 300 | 0.0000 | [0.0000, 0.0000] | — | 2.0000 | [2.0000, 2.0000] | — | — |
| qwen3.5-4b | cot | b viewpoint | 800 | 0.0000 | truncated: 800 | 1.0000 | 0/800/0/0 | 315 | 0.0000 | [0.0000, 0.0000] | — | 2.0000 | [2.0000, 2.0000] | — | — |
| qwen3.5-4b | cot | c occlusion | 300 | 0.4167 | truncated: 154, ambiguous: 11, malformed: 10 | 0.5133 | 1/299/0/0 | 300 | 0.1562 | [0.1308, 0.1815] | 0.3749 | — | — | — | — |
| qwen3.5-4b | cot | d physics | 400 | 0.3225 | truncated: 268, ambiguous: 2, out_of_range: 1 | 0.6700 | 0/400/0/0 | 400 | 0.0435 | [0.0330, 0.0554] | 0.1350 | — | — | — | — |
| qwen3.5-4b | cot | e imagery | 400 | 0.0000 | truncated: 400 | 1.0000 | 0/400/0/0 | 400 | 0.0000 | [0.0000, 0.0000] | — | — | — | — | — |
| qwen3.5-9b | zeroshot | a rotation3d | 400 | 0.0050 | wrong_row_count: 156, wrong_col_count: 132, truncated: 105, multiple_blocks: 5 | 0.2625 | — | 300 | 0.0000 | [0.0000, 0.0000] | 0.0000 | 2.0000 | [2.0000, 2.0000] | — | — |
| qwen3.5-9b | zeroshot | b viewpoint | 800 | 0.0100 | truncated: 321, wrong_col_count: 241, wrong_row_count: 224, multiple_blocks: 6 | 0.4012 | — | 315 | 0.0000 | [0.0000, 0.0000] | 0.0000 | 2.0000 | [2.0000, 2.0000] | — | — |
| qwen3.5-9b | zeroshot | c occlusion | 300 | 0.0033 | wrong_row_count: 230, truncated: 69 | 0.2300 | — | 300 | 0.0000 | [0.0000, 0.0000] | 0.0000 | — | — | — | — |
| qwen3.5-9b | zeroshot | d physics | 400 | 0.0000 | truncated: 364, wrong_row_count: 30, no_grid: 5, multiple_blocks: 1 | 0.9100 | — | 400 | 0.0000 | [0.0000, 0.0000] | — | — | — | — | — |
| qwen3.5-9b | zeroshot | e imagery | 400 | 0.0000 | wrong_row_count: 400 | 0.0000 | — | 400 | 0.0000 | [0.0000, 0.0000] | — | — | — | — | — |
| qwen3.5-9b | cot | a rotation3d | 400 | 0.0000 | truncated: 338, wrong_row_count: 55, wrong_col_count: 7 | 0.8450 | 0/400/0/0 | 300 | 0.0000 | [0.0000, 0.0000] | — | 2.0000 | [2.0000, 2.0000] | — | — |
| qwen3.5-9b | cot | b viewpoint | 800 | 0.0000 | truncated: 689, wrong_row_count: 103, wrong_col_count: 8 | 0.8612 | 0/800/0/0 | 315 | 0.0000 | [0.0000, 0.0000] | — | 2.0000 | [2.0000, 2.0000] | — | — |
| qwen3.5-9b | cot | c occlusion | 300 | 0.6167 | truncated: 108, ambiguous: 7 | 0.3600 | 0/300/0/0 | 300 | 0.3099 | [0.2762, 0.3450] | 0.5026 | — | — | — | — |
| qwen3.5-9b | cot | d physics | 400 | 0.3750 | truncated: 246, ambiguous: 3, out_of_range: 1 | 0.6150 | 1/399/0/0 | 400 | 0.0509 | [0.0372, 0.0646] | 0.1356 | — | — | — | — |
| qwen3.5-9b | cot | e imagery | 400 | 0.0175 | truncated: 393 | 0.9825 | 0/400/0/0 | 400 | 0.0152 | [0.0046, 0.0284] | 0.8706 | — | — | — | — |
| internvl3.5-2b-hf | zeroshot | a rotation3d | 400 | 0.0000 | truncated: 400 | 1.0000 | — | 300 | 0.0000 | [0.0000, 0.0000] | — | 2.0000 | [2.0000, 2.0000] | — | — |
| internvl3.5-2b-hf | zeroshot | b viewpoint | 800 | 0.0000 | truncated: 796, wrong_row_count: 4 | 0.9950 | — | 315 | 0.0000 | [0.0000, 0.0000] | — | 2.0000 | [2.0000, 2.0000] | — | — |
| internvl3.5-2b-hf | zeroshot | c occlusion | 300 | 0.0000 | wrong_row_count: 272, truncated: 28 | 0.0933 | — | 300 | 0.0000 | [0.0000, 0.0000] | — | — | — | — | — |
| internvl3.5-2b-hf | zeroshot | d physics | 400 | 0.0000 | truncated: 319, wrong_row_count: 79, multiple_blocks: 2 | 0.7975 | — | 400 | 0.0000 | [0.0000, 0.0000] | — | — | — | — | — |
| internvl3.5-2b-hf | zeroshot | e imagery | 400 | 0.0000 | truncated: 344, no_grid: 53, wrong_row_count: 3 | 0.8600 | — | 400 | 0.0000 | [0.0000, 0.0000] | — | — | — | — | — |
| internvl3.5-2b-hf | cot | a rotation3d | 400 | 0.0000 | truncated: 351, wrong_row_count: 30, no_grid: 19 | 0.8775 | 3/397/0/0 | 300 | 0.0000 | [0.0000, 0.0000] | — | 2.0000 | [2.0000, 2.0000] | — | — |
| internvl3.5-2b-hf | cot | b viewpoint | 800 | 0.0000 | truncated: 709, no_grid: 70, wrong_row_count: 18, wrong_col_count: 2, multiple_blocks: 1 | 0.8862 | 0/800/0/0 | 315 | 0.0000 | [0.0000, 0.0000] | — | 2.0000 | [2.0000, 2.0000] | — | — |
| internvl3.5-2b-hf | cot | c occlusion | 300 | 0.5600 | malformed: 54, truncated: 43, ambiguous: 32, too_few_points: 3 | 0.1433 | 0/300/0/0 | 300 | 0.0311 | [0.0236, 0.0388] | 0.0555 | — | — | — | — |
| internvl3.5-2b-hf | cot | d physics | 400 | 0.1775 | truncated: 184, too_few_points: 101, ambiguous: 33, malformed: 9, out_of_range: 2 | 0.4600 | 0/400/0/0 | 400 | 0.0002 | [0.0000, 0.0004] | 0.0011 | — | — | — | — |
| internvl3.5-2b-hf | cot | e imagery | 400 | 0.2000 | truncated: 241, malformed: 56, ambiguous: 18, out_of_range: 5 | 0.6025 | 0/400/0/0 | 400 | 0.0001 | [0.0000, 0.0003] | 0.0005 | — | — | — | — |
| internvl3.5-4b-hf | zeroshot | a rotation3d | 400 | 0.0000 | wrong_row_count: 284, truncated: 113, wrong_col_count: 2, multiple_blocks: 1 | 0.2825 | — | 300 | 0.0000 | [0.0000, 0.0000] | — | 2.0000 | [2.0000, 2.0000] | — | — |
| internvl3.5-4b-hf | zeroshot | b viewpoint | 800 | 0.0025 | truncated: 415, wrong_row_count: 383 | 0.5188 | — | 315 | 0.0000 | [0.0000, 0.0000] | — | 2.0000 | [2.0000, 2.0000] | — | — |
| internvl3.5-4b-hf | zeroshot | c occlusion | 300 | 0.0000 | wrong_row_count: 255, truncated: 44, wrong_col_count: 1 | 0.1467 | — | 300 | 0.0000 | [0.0000, 0.0000] | — | — | — | — | — |
| internvl3.5-4b-hf | zeroshot | d physics | 400 | 0.0250 | wrong_row_count: 201, truncated: 178, wrong_col_count: 7, thinking_unclosed: 3, no_grid: 1 | 0.4525 | — | 400 | 0.0000 | [0.0000, 0.0000] | 0.0000 | — | — | — | — |
| internvl3.5-4b-hf | zeroshot | e imagery | 400 | 0.0100 | wrong_row_count: 199, truncated: 173, no_grid: 24 | 0.4325 | — | 400 | 0.0000 | [0.0000, 0.0000] | 0.0000 | — | — | — | — |
| internvl3.5-4b-hf | cot | a rotation3d | 400 | 0.0025 | truncated: 367, wrong_row_count: 24, no_grid: 3, multiple_blocks: 2, wrong_col_count: 2, illegal_char: 1 | 0.9175 | 7/393/0/0 | 300 | 0.0001 | [0.0000, 0.0004] | 0.0417 | 1.9976 | [1.9928, 2.0000] | 1.2789 | 1.2789 |
| internvl3.5-4b-hf | cot | b viewpoint | 800 | 0.0000 | truncated: 760, wrong_row_count: 29, no_grid: 6, multiple_blocks: 3, illegal_char: 2 | 0.9500 | 40/760/0/0 | 315 | 0.0000 | [0.0000, 0.0000] | — | 2.0000 | [2.0000, 2.0000] | — | — |
| internvl3.5-4b-hf | cot | c occlusion | 300 | 0.5867 | truncated: 116, ambiguous: 7, out_of_range: 1 | 0.3867 | 7/293/0/0 | 300 | 0.0697 | [0.0567, 0.0849] | 0.1189 | — | — | — | — |
| internvl3.5-4b-hf | cot | d physics | 400 | 0.6075 | truncated: 134, ambiguous: 11, out_of_range: 10, too_few_points: 2 | 0.3350 | 7/393/0/0 | 400 | 0.0035 | [0.0006, 0.0075] | 0.0058 | — | — | — | — |
| internvl3.5-4b-hf | cot | e imagery | 400 | 0.3975 | truncated: 201, ambiguous: 18, malformed: 15, out_of_range: 7 | 0.5025 | 1/399/0/0 | 400 | 0.0770 | [0.0590, 0.0988] | 0.1938 | — | — | — | — |

## 2. 门槛（metrics 的 gate；差 = 系统 − 基线，配对 bootstrap 95% CI；IoU 下界 > 0、起伏上界 < 0 才算胜过）

| 模型 | 系统 | 任务 | 标题 n | 先决：最强非真值基线 | 系统 | 差 [CI] | 先决通过 | 判定：平面 差 [CI] | 判定：真值仿射 差 [CI] | 判定：凸起先验 差 [CI] | 判定 状态 / 通过 | imagining |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3.5-2b | zeroshot | a rotation3d | 300 | copy 0.6878 | 0.0000 | -0.6878 [-0.7046, -0.6710] | False | +0.9985 [0.9865, 1.0089] 否 | +0.7640 [0.6067, 0.8989] 否 | +1.0637 [1.0457, 1.0789] 否 | evaluated / False | False |
| qwen3.5-2b | zeroshot | b viewpoint | 315 | translate 0.2630 | 0.0000 | -0.2630 [-0.2864, -0.2393] | False | +0.8617 [0.8157, 0.9025] 否 | +1.2296 [1.1736, 1.2848] 否 | +0.8597 [0.8144, 0.9007] 否 | evaluated / False | False |
| qwen3.5-2b | zeroshot | c occlusion | 300 | convex_hull 0.7081 | 0.0000 | -0.7081 [-0.7265, -0.6893] | False | — | — | — | not_applicable / None | False |
| qwen3.5-2b | zeroshot | d physics | 400 | copy 0.2906 | 0.0000 | -0.2906 [-0.3157, -0.2651] | False | — | — | — | not_applicable / None | False |
| qwen3.5-2b | zeroshot | e imagery | 400 | no_transform 0.2146 | 0.0000 | -0.2146 [-0.2377, -0.1923] | False | — | — | — | not_applicable / None | False |
| qwen3.5-2b | cot | a rotation3d | 300 | copy 0.6878 | 0.0000 | -0.6878 [-0.7046, -0.6710] | False | +0.9985 [0.9865, 1.0089] 否 | +0.7640 [0.6067, 0.8989] 否 | +1.0637 [1.0457, 1.0789] 否 | evaluated / False | False |
| qwen3.5-2b | cot | b viewpoint | 315 | translate 0.2630 | 0.0000 | -0.2630 [-0.2864, -0.2393] | False | +0.8617 [0.8157, 0.9025] 否 | +1.2296 [1.1736, 1.2848] 否 | +0.8597 [0.8144, 0.9007] 否 | evaluated / False | False |
| qwen3.5-2b | cot | c occlusion | 300 | convex_hull 0.7081 | 0.1024 | -0.6058 [-0.6297, -0.5820] | False | — | — | — | not_applicable / None | False |
| qwen3.5-2b | cot | d physics | 400 | copy 0.2906 | 0.0029 | -0.2876 [-0.3127, -0.2618] | False | — | — | — | not_applicable / None | False |
| qwen3.5-2b | cot | e imagery | 400 | no_transform 0.2146 | 0.0000 | -0.2146 [-0.2377, -0.1923] | False | — | — | — | not_applicable / None | False |
| qwen3.5-0.8b | zeroshot | a rotation3d | 300 | copy 0.6878 | 0.0000 | -0.6878 [-0.7046, -0.6710] | False | +0.9985 [0.9865, 1.0089] 否 | +0.7640 [0.6067, 0.8989] 否 | +1.0637 [1.0457, 1.0789] 否 | evaluated / False | False |
| qwen3.5-0.8b | zeroshot | b viewpoint | 315 | translate 0.2630 | 0.0000 | -0.2630 [-0.2864, -0.2393] | False | +0.8617 [0.8157, 0.9025] 否 | +1.2296 [1.1736, 1.2848] 否 | +0.8597 [0.8144, 0.9007] 否 | evaluated / False | False |
| qwen3.5-0.8b | zeroshot | c occlusion | 300 | convex_hull 0.7081 | 0.0000 | -0.7081 [-0.7265, -0.6893] | False | — | — | — | not_applicable / None | False |
| qwen3.5-0.8b | zeroshot | d physics | 400 | copy 0.2906 | 0.0000 | -0.2906 [-0.3157, -0.2651] | False | — | — | — | not_applicable / None | False |
| qwen3.5-0.8b | zeroshot | e imagery | 400 | no_transform 0.2146 | 0.0000 | -0.2146 [-0.2377, -0.1923] | False | — | — | — | not_applicable / None | False |
| qwen3.5-0.8b | cot | a rotation3d | 300 | copy 0.6878 | 0.0000 | -0.6878 [-0.7046, -0.6710] | False | +0.9985 [0.9865, 1.0089] 否 | +0.7640 [0.6067, 0.8989] 否 | +1.0637 [1.0457, 1.0789] 否 | evaluated / False | False |
| qwen3.5-0.8b | cot | b viewpoint | 315 | translate 0.2630 | 0.0000 | -0.2630 [-0.2864, -0.2393] | False | +0.8617 [0.8157, 0.9025] 否 | +1.2296 [1.1736, 1.2848] 否 | +0.8597 [0.8144, 0.9007] 否 | evaluated / False | False |
| qwen3.5-0.8b | cot | c occlusion | 300 | convex_hull 0.7081 | 0.0160 | -0.6921 [-0.7125, -0.6715] | False | — | — | — | not_applicable / None | False |
| qwen3.5-0.8b | cot | d physics | 400 | copy 0.2906 | 0.0001 | -0.2905 [-0.3156, -0.2649] | False | — | — | — | not_applicable / None | False |
| qwen3.5-0.8b | cot | e imagery | 400 | no_transform 0.2146 | 0.0000 | -0.2146 [-0.2377, -0.1923] | False | — | — | — | not_applicable / None | False |
| qwen3.5-4b | zeroshot | a rotation3d | 300 | copy 0.6878 | 0.0000 | -0.6878 [-0.7046, -0.6710] | False | +0.9985 [0.9865, 1.0089] 否 | +0.7640 [0.6067, 0.8989] 否 | +1.0637 [1.0457, 1.0789] 否 | evaluated / False | False |
| qwen3.5-4b | zeroshot | b viewpoint | 315 | translate 0.2630 | 0.0000 | -0.2630 [-0.2864, -0.2393] | False | +0.8617 [0.8157, 0.9025] 否 | +1.2296 [1.1736, 1.2848] 否 | +0.8597 [0.8144, 0.9007] 否 | evaluated / False | False |
| qwen3.5-4b | zeroshot | c occlusion | 300 | convex_hull 0.7081 | 0.0000 | -0.7081 [-0.7265, -0.6893] | False | — | — | — | not_applicable / None | False |
| qwen3.5-4b | zeroshot | d physics | 400 | copy 0.2906 | 0.0000 | -0.2906 [-0.3157, -0.2651] | False | — | — | — | not_applicable / None | False |
| qwen3.5-4b | zeroshot | e imagery | 400 | no_transform 0.2146 | 0.0001 | -0.2144 [-0.2374, -0.1922] | False | — | — | — | not_applicable / None | False |
| qwen3.5-4b | cot | a rotation3d | 300 | copy 0.6878 | 0.0000 | -0.6878 [-0.7046, -0.6710] | False | +0.9985 [0.9865, 1.0089] 否 | +0.7640 [0.6067, 0.8989] 否 | +1.0637 [1.0457, 1.0789] 否 | evaluated / False | False |
| qwen3.5-4b | cot | b viewpoint | 315 | translate 0.2630 | 0.0000 | -0.2630 [-0.2864, -0.2393] | False | +0.8617 [0.8157, 0.9025] 否 | +1.2296 [1.1736, 1.2848] 否 | +0.8597 [0.8144, 0.9007] 否 | evaluated / False | False |
| qwen3.5-4b | cot | c occlusion | 300 | convex_hull 0.7081 | 0.1562 | -0.5519 [-0.5821, -0.5226] | False | — | — | — | not_applicable / None | False |
| qwen3.5-4b | cot | d physics | 400 | copy 0.2906 | 0.0435 | -0.2471 [-0.2725, -0.2207] | False | — | — | — | not_applicable / None | False |
| qwen3.5-4b | cot | e imagery | 400 | no_transform 0.2146 | 0.0000 | -0.2146 [-0.2377, -0.1923] | False | — | — | — | not_applicable / None | False |
| qwen3.5-9b | zeroshot | a rotation3d | 300 | copy 0.6878 | 0.0000 | -0.6878 [-0.7046, -0.6710] | False | +0.9985 [0.9865, 1.0089] 否 | +0.7640 [0.6067, 0.8989] 否 | +1.0637 [1.0457, 1.0789] 否 | evaluated / False | False |
| qwen3.5-9b | zeroshot | b viewpoint | 315 | translate 0.2630 | 0.0000 | -0.2630 [-0.2864, -0.2393] | False | +0.8617 [0.8157, 0.9025] 否 | +1.2296 [1.1736, 1.2848] 否 | +0.8597 [0.8144, 0.9007] 否 | evaluated / False | False |
| qwen3.5-9b | zeroshot | c occlusion | 300 | convex_hull 0.7081 | 0.0000 | -0.7081 [-0.7265, -0.6893] | False | — | — | — | not_applicable / None | False |
| qwen3.5-9b | zeroshot | d physics | 400 | copy 0.2906 | 0.0000 | -0.2906 [-0.3157, -0.2651] | False | — | — | — | not_applicable / None | False |
| qwen3.5-9b | zeroshot | e imagery | 400 | no_transform 0.2146 | 0.0000 | -0.2146 [-0.2377, -0.1923] | False | — | — | — | not_applicable / None | False |
| qwen3.5-9b | cot | a rotation3d | 300 | copy 0.6878 | 0.0000 | -0.6878 [-0.7046, -0.6710] | False | +0.9985 [0.9865, 1.0089] 否 | +0.7640 [0.6067, 0.8989] 否 | +1.0637 [1.0457, 1.0789] 否 | evaluated / False | False |
| qwen3.5-9b | cot | b viewpoint | 315 | translate 0.2630 | 0.0000 | -0.2630 [-0.2864, -0.2393] | False | +0.8617 [0.8157, 0.9025] 否 | +1.2296 [1.1736, 1.2848] 否 | +0.8597 [0.8144, 0.9007] 否 | evaluated / False | False |
| qwen3.5-9b | cot | c occlusion | 300 | convex_hull 0.7081 | 0.3099 | -0.3982 [-0.4341, -0.3620] | False | — | — | — | not_applicable / None | False |
| qwen3.5-9b | cot | d physics | 400 | copy 0.2906 | 0.0509 | -0.2397 [-0.2669, -0.2125] | False | — | — | — | not_applicable / None | False |
| qwen3.5-9b | cot | e imagery | 400 | no_transform 0.2146 | 0.0152 | -0.1993 [-0.2233, -0.1752] | False | — | — | — | not_applicable / None | False |
| internvl3.5-2b-hf | zeroshot | a rotation3d | 300 | copy 0.6878 | 0.0000 | -0.6878 [-0.7046, -0.6710] | False | +0.9985 [0.9865, 1.0089] 否 | +0.7640 [0.6067, 0.8989] 否 | +1.0637 [1.0457, 1.0789] 否 | evaluated / False | False |
| internvl3.5-2b-hf | zeroshot | b viewpoint | 315 | translate 0.2630 | 0.0000 | -0.2630 [-0.2864, -0.2393] | False | +0.8617 [0.8157, 0.9025] 否 | +1.2296 [1.1736, 1.2848] 否 | +0.8597 [0.8144, 0.9007] 否 | evaluated / False | False |
| internvl3.5-2b-hf | zeroshot | c occlusion | 300 | convex_hull 0.7081 | 0.0000 | -0.7081 [-0.7265, -0.6893] | False | — | — | — | not_applicable / None | False |
| internvl3.5-2b-hf | zeroshot | d physics | 400 | copy 0.2906 | 0.0000 | -0.2906 [-0.3157, -0.2651] | False | — | — | — | not_applicable / None | False |
| internvl3.5-2b-hf | zeroshot | e imagery | 400 | no_transform 0.2146 | 0.0000 | -0.2146 [-0.2377, -0.1923] | False | — | — | — | not_applicable / None | False |
| internvl3.5-2b-hf | cot | a rotation3d | 300 | copy 0.6878 | 0.0000 | -0.6878 [-0.7046, -0.6710] | False | +0.9985 [0.9865, 1.0089] 否 | +0.7640 [0.6067, 0.8989] 否 | +1.0637 [1.0457, 1.0789] 否 | evaluated / False | False |
| internvl3.5-2b-hf | cot | b viewpoint | 315 | translate 0.2630 | 0.0000 | -0.2630 [-0.2864, -0.2393] | False | +0.8617 [0.8157, 0.9025] 否 | +1.2296 [1.1736, 1.2848] 否 | +0.8597 [0.8144, 0.9007] 否 | evaluated / False | False |
| internvl3.5-2b-hf | cot | c occlusion | 300 | convex_hull 0.7081 | 0.0311 | -0.6770 [-0.6958, -0.6582] | False | — | — | — | not_applicable / None | False |
| internvl3.5-2b-hf | cot | d physics | 400 | copy 0.2906 | 0.0002 | -0.2904 [-0.3156, -0.2648] | False | — | — | — | not_applicable / None | False |
| internvl3.5-2b-hf | cot | e imagery | 400 | no_transform 0.2146 | 0.0001 | -0.2145 [-0.2376, -0.1920] | False | — | — | — | not_applicable / None | False |
| internvl3.5-4b-hf | zeroshot | a rotation3d | 300 | copy 0.6878 | 0.0000 | -0.6878 [-0.7046, -0.6710] | False | +0.9985 [0.9865, 1.0089] 否 | +0.7640 [0.6067, 0.8989] 否 | +1.0637 [1.0457, 1.0789] 否 | evaluated / False | False |
| internvl3.5-4b-hf | zeroshot | b viewpoint | 315 | translate 0.2630 | 0.0000 | -0.2630 [-0.2864, -0.2393] | False | +0.8617 [0.8157, 0.9025] 否 | +1.2296 [1.1736, 1.2848] 否 | +0.8597 [0.8144, 0.9007] 否 | evaluated / False | False |
| internvl3.5-4b-hf | zeroshot | c occlusion | 300 | convex_hull 0.7081 | 0.0000 | -0.7081 [-0.7265, -0.6893] | False | — | — | — | not_applicable / None | False |
| internvl3.5-4b-hf | zeroshot | d physics | 400 | copy 0.2906 | 0.0000 | -0.2906 [-0.3157, -0.2651] | False | — | — | — | not_applicable / None | False |
| internvl3.5-4b-hf | zeroshot | e imagery | 400 | no_transform 0.2146 | 0.0000 | -0.2146 [-0.2377, -0.1923] | False | — | — | — | not_applicable / None | False |
| internvl3.5-4b-hf | cot | a rotation3d | 300 | copy 0.6878 | 0.0001 | -0.6876 [-0.7046, -0.6709] | False | +0.9961 [0.9839, 1.0075] 否 | +0.7616 [0.6024, 0.8958] 否 | +1.0613 [1.0421, 1.0773] 否 | evaluated / False | False |
| internvl3.5-4b-hf | cot | b viewpoint | 315 | translate 0.2630 | 0.0000 | -0.2630 [-0.2864, -0.2393] | False | +0.8617 [0.8157, 0.9025] 否 | +1.2296 [1.1736, 1.2848] 否 | +0.8597 [0.8144, 0.9007] 否 | evaluated / False | False |
| internvl3.5-4b-hf | cot | c occlusion | 300 | convex_hull 0.7081 | 0.0697 | -0.6384 [-0.6613, -0.6156] | False | — | — | — | not_applicable / None | False |
| internvl3.5-4b-hf | cot | d physics | 400 | copy 0.2906 | 0.0035 | -0.2870 [-0.3112, -0.2625] | False | — | — | — | not_applicable / None | False |
| internvl3.5-4b-hf | cot | e imagery | 400 | no_transform 0.2146 | 0.0770 | -0.1375 [-0.1680, -0.1078] | False | — | — | — | not_applicable / None | False |
| lisa-7b-v1 | external | a rotation3d | 300 | copy 0.6878 | 0.6489 | -0.0389 [-0.0500, -0.0281] | False | — | — | — | not_applicable / None | False |
| lisa-7b-v1 | external | b viewpoint | 315 | translate 0.2630 | 0.2132 | -0.0498 [-0.0662, -0.0334] | False | — | — | — | not_applicable / None | False |
| lisa-7b-v1 | external | c occlusion | 300 | convex_hull 0.7081 | 0.4891 | -0.2191 [-0.2473, -0.1901] | False | — | — | — | not_applicable / None | False |
| lisa-7b-v1 | external | d physics | 400 | copy 0.2906 | 0.1828 | -0.1078 [-0.1251, -0.0903] | False | — | — | — | not_applicable / None | False |

## 3. LISA-7B-v1（外部对照，a–d；标题口径同上）

| 任务 | n | 解析失败（原因：条数） | 标题 n | IoU 严格 | 95% CI | IoU 条件 | 对当前 mask 的 IoU 严格 | 95% CI | 对当前 mask 的 IoU 条件 |
|---|---|---|---|---|---|---|---|---|---|
| a rotation3d | 400 | — | 300 | 0.6489 | [0.6293, 0.6671] | 0.6489 | 0.9206 | [0.9042, 0.9352] | 0.9206 |
| b viewpoint | 800 | external_failure: 1 | 315 | 0.2132 | [0.1923, 0.2359] | 0.2132 | 0.6878 | [0.6532, 0.7224] | 0.6878 |
| c occlusion | 300 | — | 300 | 0.4891 | [0.4599, 0.5194] | 0.4891 | 0.4334 | [0.3979, 0.4701] | 0.4334 |
| d physics | 400 | external_failure: 1 | 400 | 0.1828 | [0.1639, 0.2025] | 0.1833 | 0.6088 | [0.5709, 0.6439] | 0.6104 |

## 4. 平凡基线（`baseline/<基线>/<任务>/metrics.json` 的 by_bin；档号从 0 数，档界见 E5 预注册 §2）

| 基线 | 任务 | 档 | n | IoU（网格） | 起伏误差 严格 | 起伏误差 条件中位数 |
|---|---|---|---|---|---|---|
| const_velocity | d physics | 档 0 | 100 | 0.6867 | — | — |
| const_velocity | d physics | 档 1 | 100 | 0.2528 | — | — |
| const_velocity | d physics | 档 2 | 100 | 0.0410 | — | — |
| const_velocity | d physics | 档 3 | 100 | 0.0036 | — | — |
| const_velocity | d physics | 标题（全部样本） | 400 | 0.2460 | — | — |
| convex_hull | c occlusion | 档 0 | 100 | 0.8092 | — | — |
| convex_hull | c occlusion | 档 1 | 100 | 0.6980 | — | — |
| convex_hull | c occlusion | 档 2 | 100 | 0.6172 | — | — |
| convex_hull | c occlusion | 标题（全部样本） | 300 | 0.7081 | — | — |
| copy | a rotation3d | 档 0 | 100 | 0.8240 | 0.7214 | 0.6269 |
| copy | a rotation3d | 档 1 | 100 | 0.7567 | 1.2909 | 1.0168 |
| copy | a rotation3d | 档 2 | 100 | 0.6445 | 1.8520 | 1.4393 |
| copy | a rotation3d | 档 3 | 100 | 0.6622 | 1.5156 | 1.3020 |
| copy | a rotation3d | 标题（档 [1, 2, 3]（下限 ≥ 30°）） | 300 | 0.6878 | 1.5528 | 1.3020 |
| copy | b viewpoint | 档 0 | 200 | 0.7332 | 0.5983 | 0.5762 |
| copy | b viewpoint | 档 1 | 200 | 0.5554 | 0.9188 | 0.8750 |
| copy | b viewpoint | 档 2 | 200 | 0.3198 | 1.2591 | 1.1724 |
| copy | b viewpoint | 档 3 | 200 | 0.1896 | 1.5326 | 1.4671 |
| copy | b viewpoint | move=backward | 59 | 0.7604 | 0.3548 | 0.2953 |
| copy | b viewpoint | move=down | 64 | 0.4835 | 1.1398 | 0.9234 |
| copy | b viewpoint | move=forward | 70 | 0.6983 | 0.4578 | 0.4105 |
| copy | b viewpoint | move=left | 51 | 0.4527 | 1.0172 | 0.9679 |
| copy | b viewpoint | move=orbit | 426 | 0.3680 | 1.2505 | 1.0663 |
| copy | b viewpoint | move=right | 71 | 0.3867 | 1.1576 | 1.1465 |
| copy | b viewpoint | move=up | 59 | 0.4680 | 1.1701 | 1.1214 |
| copy | b viewpoint | 标题（环绕移动，档 [1, 2, 3]（下限 ≥ 20°）） | 315 | 0.2621 | 1.4242 | 1.2338 |
| copy | d physics | 档 0 | 100 | 0.4836 | — | — |
| copy | d physics | 档 1 | 100 | 0.2522 | — | — |
| copy | d physics | 档 2 | 100 | 0.1961 | — | — |
| copy | d physics | 档 3 | 100 | 0.2304 | — | — |
| copy | d physics | 标题（全部样本） | 400 | 0.2906 | — | — |
| copy_convex | a rotation3d | 档 0 | 100 | 0.8240 | 0.9056 | 0.9168 |
| copy_convex | a rotation3d | 档 1 | 100 | 0.7567 | 0.9029 | 0.9085 |
| copy_convex | a rotation3d | 档 2 | 100 | 0.6445 | 0.9670 | 0.9443 |
| copy_convex | a rotation3d | 档 3 | 100 | 0.6622 | 0.9390 | 0.9312 |
| copy_convex | a rotation3d | 标题（档 [1, 2, 3]（下限 ≥ 30°）） | 300 | 0.6878 | 0.9363 | 0.9258 |
| copy_convex | b viewpoint | 档 0 | 200 | 0.7332 | 0.9442 | 0.9571 |
| copy_convex | b viewpoint | 档 1 | 200 | 0.5554 | 0.9649 | 0.9616 |
| copy_convex | b viewpoint | 档 2 | 200 | 0.3198 | 1.0485 | 0.9794 |
| copy_convex | b viewpoint | 档 3 | 200 | 0.1896 | 1.1835 | 0.9820 |
| copy_convex | b viewpoint | move=backward | 59 | 0.7604 | 0.9428 | 0.9510 |
| copy_convex | b viewpoint | move=down | 64 | 0.4835 | 0.9961 | 0.9631 |
| copy_convex | b viewpoint | move=forward | 70 | 0.6983 | 0.9100 | 0.9099 |
| copy_convex | b viewpoint | move=left | 51 | 0.4527 | 0.9909 | 0.9441 |
| copy_convex | b viewpoint | move=orbit | 426 | 0.3680 | 1.0911 | 0.9800 |
| copy_convex | b viewpoint | move=right | 71 | 0.3867 | 1.0064 | 0.9836 |
| copy_convex | b viewpoint | move=up | 59 | 0.4680 | 0.9894 | 0.9658 |
| copy_convex | b viewpoint | 标题（环绕移动，档 [1, 2, 3]（下限 ≥ 20°）） | 315 | 0.2621 | 1.1403 | 0.9910 |
| copy_flat | a rotation3d | 档 0 | 100 | 0.8240 | 0.9913 | 1.0039 |
| copy_flat | a rotation3d | 档 1 | 100 | 0.7567 | 0.9983 | 1.0086 |
| copy_flat | a rotation3d | 档 2 | 100 | 0.6445 | 1.0066 | 1.0134 |
| copy_flat | a rotation3d | 档 3 | 100 | 0.6622 | 0.9996 | 1.0175 |
| copy_flat | a rotation3d | 标题（档 [1, 2, 3]（下限 ≥ 30°）） | 300 | 0.6878 | 1.0015 | 1.0126 |
| copy_flat | b viewpoint | 档 0 | 200 | 0.7332 | 0.9911 | 1.0072 |
| copy_flat | b viewpoint | 档 1 | 200 | 0.5554 | 0.9936 | 1.0055 |
| copy_flat | b viewpoint | 档 2 | 200 | 0.3198 | 1.0503 | 1.0028 |
| copy_flat | b viewpoint | 档 3 | 200 | 0.1896 | 1.1804 | 1.0059 |
| copy_flat | b viewpoint | move=backward | 59 | 0.7604 | 1.0067 | 1.0080 |
| copy_flat | b viewpoint | move=down | 64 | 0.4835 | 1.0133 | 1.0026 |
| copy_flat | b viewpoint | move=forward | 70 | 0.6983 | 0.9514 | 0.9602 |
| copy_flat | b viewpoint | move=left | 51 | 0.4527 | 1.0196 | 0.9870 |
| copy_flat | b viewpoint | move=orbit | 426 | 0.3680 | 1.1008 | 1.0061 |
| copy_flat | b viewpoint | move=right | 71 | 0.3867 | 1.0091 | 1.0084 |
| copy_flat | b viewpoint | move=up | 59 | 0.4680 | 1.0111 | 1.0112 |
| copy_flat | b viewpoint | 标题（环绕移动，档 [1, 2, 3]（下限 ≥ 20°）） | 315 | 0.2621 | 1.1383 | 1.0039 |
| gt_affine_upper | a rotation3d | 档 0 | 100 | 0.9350 | 0.7625 | 0.6330 |
| gt_affine_upper | a rotation3d | 档 1 | 100 | 0.9260 | 1.1464 | 0.8903 |
| gt_affine_upper | a rotation3d | 档 2 | 100 | 0.8831 | 1.4416 | 1.1045 |
| gt_affine_upper | a rotation3d | 档 3 | 100 | 0.9127 | 1.1200 | 0.9544 |
| gt_affine_upper | a rotation3d | 标题（档 [1, 2, 3]（下限 ≥ 30°）） | 300 | 0.9073 | 1.2360 | 0.9639 |
| gt_affine_upper | b viewpoint | 档 0 | 200 | 0.9757 | 0.3283 | 0.1887 |
| gt_affine_upper | b viewpoint | 档 1 | 200 | 0.9638 | 0.4534 | 0.3061 |
| gt_affine_upper | b viewpoint | 档 2 | 200 | 0.9477 | 0.6009 | 0.4263 |
| gt_affine_upper | b viewpoint | 档 3 | 200 | 0.9545 | 0.6000 | 0.4794 |
| gt_affine_upper | b viewpoint | move=backward | 59 | 0.9902 | 0.0943 | 0.0815 |
| gt_affine_upper | b viewpoint | move=down | 64 | 0.9769 | 0.2843 | 0.1955 |
| gt_affine_upper | b viewpoint | move=forward | 70 | 0.9831 | 0.1376 | 0.1201 |
| gt_affine_upper | b viewpoint | move=left | 51 | 0.9806 | 0.4011 | 0.2425 |
| gt_affine_upper | b viewpoint | move=orbit | 426 | 0.9426 | 0.7023 | 0.5647 |
| gt_affine_upper | b viewpoint | move=right | 71 | 0.9769 | 0.3671 | 0.3063 |
| gt_affine_upper | b viewpoint | move=up | 59 | 0.9773 | 0.2951 | 0.1979 |
| gt_affine_upper | b viewpoint | 标题（环绕移动，档 [1, 2, 3]（下限 ≥ 20°）） | 315 | 0.9355 | 0.7704 | 0.6551 |
| instr_affine | a rotation3d | 档 0 | 100 | 0.8184 | 0.7319 | 0.6210 |
| instr_affine | a rotation3d | 档 1 | 100 | 0.6456 | 1.2516 | 1.0000 |
| instr_affine | a rotation3d | 档 2 | 100 | 0.2266 | 1.6298 | 1.1850 |
| instr_affine | a rotation3d | 档 3 | 100 | 0.5263 | 1.5302 | 1.4258 |
| instr_affine | a rotation3d | 标题（档 [1, 2, 3]（下限 ≥ 30°）） | 300 | 0.4662 | 1.4705 | 1.1903 |
| no_transform | e imagery | 档 0 | 100 | 0.4947 | — | — |
| no_transform | e imagery | 档 1 | 100 | 0.2910 | — | — |
| no_transform | e imagery | 档 2 | 100 | 0.0707 | — | — |
| no_transform | e imagery | 档 3 | 100 | 0.0018 | — | — |
| no_transform | e imagery | 标题（全部样本） | 400 | 0.2146 | — | — |
| translate | b viewpoint | 档 0 | 200 | 0.7850 | 0.5043 | 0.4538 |
| translate | b viewpoint | 档 1 | 200 | 0.6366 | 0.7570 | 0.6958 |
| translate | b viewpoint | 档 2 | 200 | 0.4512 | 1.0629 | 0.8792 |
| translate | b viewpoint | 档 3 | 200 | 0.3974 | 1.2107 | 0.8949 |
| translate | b viewpoint | move=backward | 59 | 0.7569 | 0.3426 | 0.2990 |
| translate | b viewpoint | move=down | 64 | 0.8308 | 0.4986 | 0.3485 |
| translate | b viewpoint | move=forward | 70 | 0.6952 | 0.4752 | 0.4292 |
| translate | b viewpoint | move=left | 51 | 0.8662 | 0.4263 | 0.3675 |
| translate | b viewpoint | move=orbit | 426 | 0.3692 | 1.2552 | 1.0433 |
| translate | b viewpoint | move=right | 71 | 0.8124 | 0.5370 | 0.4624 |
| translate | b viewpoint | move=up | 59 | 0.8211 | 0.4580 | 0.3721 |
| translate | b viewpoint | 标题（环绕移动，档 [1, 2, 3]（下限 ≥ 20°）） | 315 | 0.2630 | 1.4303 | 1.2578 |
| visible | c occlusion | 档 0 | 100 | 0.7797 | — | — |
| visible | c occlusion | 档 1 | 100 | 0.5980 | — | — |
| visible | c occlusion | 档 2 | 100 | 0.3907 | — | — |
| visible | c occlusion | 标题（全部样本） | 300 | 0.5895 | — | — |

## 5. 用量（`logs/driver.log`；GPU 时间 = 零样本/思维链两条命令与 LISA 推理的墙钟，含加载模型；导出、评分、基线是 CPU，不计）

| 步骤 | rc | 墙钟 s | 计入 GPU | 累计 GPU s |
|---|---|---|---|---|
| baseline__acde | 0 | 74 | 否 | 0 |
| baseline__b | 0 | 187 | 否 | 0 |
| qwen3.5-2b__acde | 0 | 3604 | 是 | 3604 |
| qwen3.5-2b__b | 0 | 1963 | 是 | 5567 |
| lisa__export | 0 | 25 | 否 | 5567 |
| lisa__export_b200 | 0 | 26 | 否 | 5567 |
| lisa__limit5 | 0 | 13 | 是 | 5580 |
| lisa__overlays | 0 | 0 | 否 | 5580 |
| lisa__full | 0 | 380 | 是 | 5960 |
| lisa__full_b200 | 0 | 206 | 是 | 6166 |
| lisa__score_acd | 0 | 23 | 否 | 6166 |
| lisa__score_b | 0 | 40 | 否 | 6166 |
| qwen3.5-0.8b__acde | 0 | 3488 | 是 | 9654 |
| qwen3.5-0.8b__b | 0 | 1897 | 是 | 11551 |
| qwen3.5-4b__acde | 0 | 7155 | 是 | 18706 |
| qwen3.5-4b__b | 0 | 3914 | 是 | 22620 |
| qwen3.5-9b__acde | 0 | 7148 | 是 | 29768 |
| qwen3.5-9b__b | 0 | 4043 | 是 | 33811 |
| internvl3.5-2b-hf__acde | 0 | 10567 | 是 | 44378 |
| internvl3.5-2b-hf__b | 0 | 5893 | 是 | 50271 |
| internvl3.5-4b-hf__acde | 0 | 21638 | 是 | 71909 |
| internvl3.5-4b-hf__b | 0 | 12044 | 是 | 83953 |

| 项目 | GPU 墙钟 s | GPU 小时 |
|---|---|---|
| qwen3.5-2b | 5567 | 1.546 |
| lisa | 599 | 0.166 |
| qwen3.5-0.8b | 5385 | 1.496 |
| qwen3.5-4b | 11069 | 3.075 |
| qwen3.5-9b | 11191 | 3.109 |
| internvl3.5-2b-hf | 16460 | 4.572 |
| internvl3.5-4b-hf | 33682 | 9.356 |
| **合计** | **83953** | **23.320** |

