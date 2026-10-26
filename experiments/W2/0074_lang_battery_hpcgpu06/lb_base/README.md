# 语言读数（大题量）：base

- 脚本：`scripts/lang_battery.py`（DEV-0064），battery v1；commit `06557387cc6711b4cfa4e154a4308861b3e6633d`；模式 real；--limit None
- 模型：`qwen3.5-9b`（Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a，cuda:0，torch.bfloat16）
- MMLU-Pro：TIGER-Lab/MMLU-Pro@b189ec765aa7ed75c8acfea42df31fdae71f97be test（MIT），文件 sha256 `0e24a191921c2f453518a537a8b2117bd137e7714d4ef1565e9ba06c1ecb9ad8`（与固定值相同：True）。总体 12032 题去掉健康检查的 100 题，按 `lang_battery_v1` 分层抽 2000 题；与健康检查重叠 0 题。
- IFEval：google/IFEval@966cd89545d6b6acfd7638bc708b98261ca58e84（Apache-2.0），全部 541 条；其中 50 条也是健康检查的题（逐题 meta.in_health_check）。
- 数据只限非商业研究，不再分发；本目录只存题目 id、提示的 sha256 与模型输出，不存题面。
- 口径与健康检查 v3 相同：MMLU-Pro 受限字母读数（option_logits）、IFEval 官方 checker；关思考、greedy、零张图。

| 读数 | 口径 | n | 点估计（%） | bootstrap 95% CI |
|---|---|---:|---:|---|
| mmlu_pro_letter | accuracy | 2000 | 54.2 | [52.1, 56.3] |
| mmlu_pro_letter | mean_p_gold | 2000 | 49.4 | [47.7, 51.0] |
| ifeval | prompt_level_strict | 541 | 83.5 | [80.4, 86.5] |
| ifeval | instruction_level_strict | 834 | 88.2 | [85.9, 90.3] |
| ifeval | prompt_level_loose | 541 | 87.6 | [84.7, 90.4] |
| ifeval | instruction_level_loose | 834 | 91.2 | [89.1, 93.0] |

MMLU-Pro 分学科（准确率）：

| 学科 | n | % | 95% CI |
|---|---:|---:|---|
| biology | 143 | 82.5 | [75.5, 88.8] |
| business | 143 | 36.4 | [28.0, 44.8] |
| chemistry | 144 | 36.1 | [28.5, 43.8] |
| computer science | 142 | 52.1 | [44.4, 60.6] |
| economics | 143 | 69.9 | [62.9, 76.9] |
| engineering | 142 | 38.0 | [30.3, 46.5] |
| health | 142 | 64.8 | [57.0, 73.2] |
| history | 142 | 68.3 | [60.6, 76.1] |
| law | 144 | 42.4 | [34.7, 50.0] |
| math | 144 | 34.0 | [26.4, 41.7] |
| other | 142 | 50.7 | [42.3, 59.2] |
| philosophy | 143 | 62.2 | [54.5, 70.6] |
| physics | 143 | 44.1 | [36.4, 52.4] |
| psychology | 143 | 76.9 | [70.6, 83.2] |

- 计算记账（预热不计）：2541 题，2541 次调用，生成 164948 token，提示 464410 token，调用耗时 5537.5 s；整次墙钟 5578.36 s
- IFEval 诊断：思考段未闭合 0，空回答 0，写满 1024 token 31
- 命令：`python scripts/lang_battery.py --model qwen3.5-9b --raw-dir /home/mail/chenyuma/visint/data/lang_battery_raw --label base --out /home/mail/chenyuma/visint/repo/experiments/W2/0074_lang_battery_hpcgpu06/lb_base`
