# 语言读数（大题量）：x2s1

- 脚本：`scripts/lang_battery.py`（DEV-0064），battery v1；commit `b5adf2b4638cf7a9b051c61d88ab66312b2daa75`；模式 real；--limit None
- 模型：`qwen3.5-9b`（Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a，cuda:0，torch.bfloat16）
- 检查点：`/home/mail/chenyuma/visint/repo/experiments/X/0093_x2s1_hpcgpu07/0093_train/w2c/qwen3.5-9b/seed1/ckpt/final/model`，manifest sha256 `333f446b3a3ca77a31a9fdba0968d06c02dbe89c442c41e6bf91bdb197480816`；评测前的合并核对 pass=True；之后 load_trained(merge=True)
- MMLU-Pro：TIGER-Lab/MMLU-Pro@b189ec765aa7ed75c8acfea42df31fdae71f97be test（MIT），文件 sha256 `0e24a191921c2f453518a537a8b2117bd137e7714d4ef1565e9ba06c1ecb9ad8`（与固定值相同：True）。总体 12032 题去掉健康检查的 100 题，按 `lang_battery_v1` 分层抽 2000 题；与健康检查重叠 0 题。
- IFEval：google/IFEval@966cd89545d6b6acfd7638bc708b98261ca58e84（Apache-2.0），全部 541 条；其中 50 条也是健康检查的题（逐题 meta.in_health_check）。
- 数据只限非商业研究，不再分发；本目录只存题目 id、提示的 sha256 与模型输出，不存题面。
- 口径与健康检查 v3 相同：MMLU-Pro 受限字母读数（option_logits）、IFEval 官方 checker；关思考、greedy、零张图。

| 读数 | 口径 | n | 点估计（%） | bootstrap 95% CI |
|---|---|---:|---:|---|
| mmlu_pro_letter | accuracy | 2000 | 51.9 | [49.8, 54.1] |
| mmlu_pro_letter | mean_p_gold | 2000 | 46.9 | [45.2, 48.6] |
| ifeval | prompt_level_strict | 541 | 83.7 | [80.6, 86.7] |
| ifeval | instruction_level_strict | 834 | 88.7 | [86.6, 90.7] |
| ifeval | prompt_level_loose | 541 | 87.1 | [84.1, 89.8] |
| ifeval | instruction_level_loose | 834 | 90.9 | [88.7, 92.8] |

MMLU-Pro 分学科（准确率）：

| 学科 | n | % | 95% CI |
|---|---:|---:|---|
| biology | 143 | 81.8 | [74.8, 88.1] |
| business | 143 | 29.4 | [22.4, 37.1] |
| chemistry | 144 | 31.9 | [24.3, 39.6] |
| computer science | 142 | 49.3 | [40.8, 57.7] |
| economics | 143 | 65.0 | [58.0, 72.7] |
| engineering | 142 | 29.6 | [22.5, 37.3] |
| health | 142 | 62.0 | [54.2, 70.4] |
| history | 142 | 69.0 | [62.0, 76.8] |
| law | 144 | 43.1 | [35.4, 51.4] |
| math | 144 | 32.6 | [25.0, 40.3] |
| other | 142 | 50.0 | [40.8, 58.5] |
| philosophy | 143 | 62.2 | [53.8, 70.6] |
| physics | 143 | 42.7 | [34.3, 51.1] |
| psychology | 143 | 78.3 | [71.3, 84.6] |

- 计算记账（预热不计）：2541 题，2541 次调用，生成 165005 token，提示 464410 token，调用耗时 5728.4 s；整次墙钟 6198.98 s
- IFEval 诊断：思考段未闭合 0，空回答 0，写满 1024 token 24
- 命令：`python scripts/lang_battery.py --model qwen3.5-9b --raw-dir /home/mail/chenyuma/visint/data/lang_battery_raw --label x2s1 --data-root /home/mail/chenyuma/visint/b200/data --checkpoint /home/mail/chenyuma/visint/repo/experiments/X/0093_x2s1_hpcgpu07/0093_train/w2c/qwen3.5-9b/seed1/ckpt/final/model --out /home/mail/chenyuma/visint/repo/experiments/X/0096_lang_seeds_hpcgpu07/lb_x2s1`
