# 语言读数（大题量）：base_4b

- 脚本：`scripts/lang_battery.py`（DEV-0064），battery v1；commit `ea3fd8548a486a1c8f0506808f72f6c3090e81a0`；模式 real；--limit None
- 模型：`qwen3.5-4b`（Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a，cuda:0，torch.bfloat16）
- MMLU-Pro：TIGER-Lab/MMLU-Pro@b189ec765aa7ed75c8acfea42df31fdae71f97be test（MIT），文件 sha256 `0e24a191921c2f453518a537a8b2117bd137e7714d4ef1565e9ba06c1ecb9ad8`（与固定值相同：True）。总体 12032 题去掉健康检查的 100 题，按 `lang_battery_v1` 分层抽 2000 题；与健康检查重叠 0 题。
- IFEval：google/IFEval@966cd89545d6b6acfd7638bc708b98261ca58e84（Apache-2.0），全部 541 条；其中 50 条也是健康检查的题（逐题 meta.in_health_check）。
- 数据只限非商业研究，不再分发；本目录只存题目 id、提示的 sha256 与模型输出，不存题面。
- 口径与健康检查 v3 相同：MMLU-Pro 受限字母读数（option_logits）、IFEval 官方 checker；关思考、greedy、零张图。

| 读数 | 口径 | n | 点估计（%） | bootstrap 95% CI |
|---|---|---:|---:|---|
| mmlu_pro_letter | accuracy | 2000 | 47.4 | [45.4, 49.5] |
| mmlu_pro_letter | mean_p_gold | 2000 | 42.3 | [40.6, 43.8] |
| ifeval | prompt_level_strict | 541 | 83.0 | [79.9, 86.1] |
| ifeval | instruction_level_strict | 834 | 88.0 | [85.7, 90.2] |
| ifeval | prompt_level_loose | 541 | 86.7 | [83.7, 89.6] |
| ifeval | instruction_level_loose | 834 | 90.8 | [88.7, 92.7] |

MMLU-Pro 分学科（准确率）：

| 学科 | n | % | 95% CI |
|---|---:|---:|---|
| biology | 143 | 78.3 | [70.6, 85.3] |
| business | 143 | 24.5 | [17.5, 32.2] |
| chemistry | 144 | 30.6 | [22.9, 37.5] |
| computer science | 142 | 49.3 | [40.8, 57.0] |
| economics | 143 | 65.7 | [58.0, 73.4] |
| engineering | 142 | 32.4 | [25.4, 40.1] |
| health | 142 | 61.3 | [53.5, 69.0] |
| history | 142 | 51.4 | [43.7, 59.9] |
| law | 144 | 38.2 | [30.6, 46.5] |
| math | 144 | 29.2 | [22.2, 36.8] |
| other | 142 | 42.3 | [34.5, 50.7] |
| philosophy | 143 | 53.8 | [45.5, 61.5] |
| physics | 143 | 35.0 | [28.0, 42.7] |
| psychology | 143 | 72.0 | [65.0, 79.0] |

- 计算记账（预热不计）：2541 题，2541 次调用，生成 159813 token，提示 464410 token，调用耗时 5366.9 s；整次墙钟 5877.36 s
- IFEval 诊断：思考段未闭合 0，空回答 0，写满 1024 token 32
- 命令：`python scripts/lang_battery.py --model qwen3.5-4b --raw-dir /home/mail/chenyuma/visint/data/lang_battery_raw --label base_4b --out /home/mail/chenyuma/visint/repo/experiments/X/0101_lb_4b_hpcgpu07/lb_base`
