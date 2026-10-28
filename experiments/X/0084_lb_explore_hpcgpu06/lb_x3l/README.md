# 语言读数（大题量）：x3l

- 脚本：`scripts/lang_battery.py`（DEV-0064），battery v1；commit `46dff4ae14fa8c07583883b05fa4f05f5c24a4c1`；模式 real；--limit None
- 模型：`qwen3.5-9b`（Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a，cuda:0，torch.bfloat16）
- 检查点：`/home/mail/chenyuma/visint/repo/experiments/X/0083_x3l_hpcgpu06/0083_train/w2c/qwen3.5-9b/seed0/ckpt/final/model`，manifest sha256 `eb200b245cc620c7f28b7c8051967cb29409b0d36e7ce7de9d20235e1d628f6a`；评测前的合并核对 pass=True；之后 load_trained(merge=True)
- MMLU-Pro：TIGER-Lab/MMLU-Pro@b189ec765aa7ed75c8acfea42df31fdae71f97be test（MIT），文件 sha256 `0e24a191921c2f453518a537a8b2117bd137e7714d4ef1565e9ba06c1ecb9ad8`（与固定值相同：True）。总体 12032 题去掉健康检查的 100 题，按 `lang_battery_v1` 分层抽 2000 题；与健康检查重叠 0 题。
- IFEval：google/IFEval@966cd89545d6b6acfd7638bc708b98261ca58e84（Apache-2.0），全部 541 条；其中 50 条也是健康检查的题（逐题 meta.in_health_check）。
- 数据只限非商业研究，不再分发；本目录只存题目 id、提示的 sha256 与模型输出，不存题面。
- 口径与健康检查 v3 相同：MMLU-Pro 受限字母读数（option_logits）、IFEval 官方 checker；关思考、greedy、零张图。

| 读数 | 口径 | n | 点估计（%） | bootstrap 95% CI |
|---|---|---:|---:|---|
| mmlu_pro_letter | accuracy | 2000 | 52.8 | [50.5, 54.9] |
| mmlu_pro_letter | mean_p_gold | 2000 | 48.8 | [47.1, 50.4] |
| ifeval | prompt_level_strict | 541 | 79.5 | [76.2, 83.0] |
| ifeval | instruction_level_strict | 834 | 85.4 | [82.9, 87.8] |
| ifeval | prompt_level_loose | 541 | 83.4 | [80.0, 86.5] |
| ifeval | instruction_level_loose | 834 | 88.2 | [85.9, 90.4] |

MMLU-Pro 分学科（准确率）：

| 学科 | n | % | 95% CI |
|---|---:|---:|---|
| biology | 143 | 82.5 | [75.5, 88.8] |
| business | 143 | 28.7 | [21.7, 36.4] |
| chemistry | 144 | 34.7 | [27.8, 42.4] |
| computer science | 142 | 52.8 | [43.7, 59.9] |
| economics | 143 | 67.8 | [60.1, 75.5] |
| engineering | 142 | 33.8 | [26.7, 41.5] |
| health | 142 | 67.6 | [59.2, 75.4] |
| history | 142 | 66.2 | [58.5, 73.3] |
| law | 144 | 41.0 | [34.0, 49.3] |
| math | 144 | 34.7 | [27.1, 43.1] |
| other | 142 | 49.3 | [40.8, 57.7] |
| philosophy | 143 | 61.5 | [53.8, 69.2] |
| physics | 143 | 44.8 | [36.4, 53.1] |
| psychology | 143 | 73.4 | [66.4, 79.7] |

- 计算记账（预热不计）：2541 题，2541 次调用，生成 166220 token，提示 464410 token，调用耗时 5959.0 s；整次墙钟 6351.21 s
- IFEval 诊断：思考段未闭合 0，空回答 0，写满 1024 token 27
- 命令：`python scripts/lang_battery.py --model qwen3.5-9b --raw-dir /home/mail/chenyuma/visint/data/lang_battery_raw --label x3l --data-root /home/mail/chenyuma/visint/b200/data --checkpoint /home/mail/chenyuma/visint/repo/experiments/X/0083_x3l_hpcgpu06/0083_train/w2c/qwen3.5-9b/seed0/ckpt/final/model --out /home/mail/chenyuma/visint/repo/experiments/X/0084_lb_explore_hpcgpu06/lb_x3l`
