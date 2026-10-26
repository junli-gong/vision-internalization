# 语言读数（大题量）：w2c

- 脚本：`scripts/lang_battery.py`（DEV-0064），battery v1；commit `06557387cc6711b4cfa4e154a4308861b3e6633d`；模式 real；--limit None
- 模型：`qwen3.5-9b`（Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a，cuda:0，torch.bfloat16）
- 检查点：`/home/mail/chenyuma/visint/repo/experiments/W2/0069_w2c_hpcgpu06/0069_train/w2c/qwen3.5-9b/seed0/ckpt/final/model`，manifest sha256 `c605408fbb92cceafbc59dff4d85c9c912b246d05dc5e29235db0cff389d93b7`；评测前的合并核对 pass=True；之后 load_trained(merge=True)
- MMLU-Pro：TIGER-Lab/MMLU-Pro@b189ec765aa7ed75c8acfea42df31fdae71f97be test（MIT），文件 sha256 `0e24a191921c2f453518a537a8b2117bd137e7714d4ef1565e9ba06c1ecb9ad8`（与固定值相同：True）。总体 12032 题去掉健康检查的 100 题，按 `lang_battery_v1` 分层抽 2000 题；与健康检查重叠 0 题。
- IFEval：google/IFEval@966cd89545d6b6acfd7638bc708b98261ca58e84（Apache-2.0），全部 541 条；其中 50 条也是健康检查的题（逐题 meta.in_health_check）。
- 数据只限非商业研究，不再分发；本目录只存题目 id、提示的 sha256 与模型输出，不存题面。
- 口径与健康检查 v3 相同：MMLU-Pro 受限字母读数（option_logits）、IFEval 官方 checker；关思考、greedy、零张图。

| 读数 | 口径 | n | 点估计（%） | bootstrap 95% CI |
|---|---|---:|---:|---|
| mmlu_pro_letter | accuracy | 2000 | 45.2 | [42.9, 47.2] |
| mmlu_pro_letter | mean_p_gold | 2000 | 38.2 | [36.6, 39.6] |
| ifeval | prompt_level_strict | 541 | 78.9 | [75.4, 82.3] |
| ifeval | instruction_level_strict | 834 | 85.0 | [82.5, 87.3] |
| ifeval | prompt_level_loose | 541 | 81.9 | [78.2, 84.8] |
| ifeval | instruction_level_loose | 834 | 87.3 | [84.8, 89.4] |

MMLU-Pro 分学科（准确率）：

| 学科 | n | % | 95% CI |
|---|---:|---:|---|
| biology | 143 | 75.5 | [67.8, 82.5] |
| business | 143 | 22.4 | [16.1, 29.4] |
| chemistry | 144 | 30.6 | [23.6, 37.5] |
| computer science | 142 | 39.4 | [31.7, 47.9] |
| economics | 143 | 60.1 | [51.7, 68.5] |
| engineering | 142 | 25.4 | [18.3, 33.1] |
| health | 142 | 59.2 | [51.4, 66.9] |
| history | 142 | 54.9 | [46.5, 62.7] |
| law | 144 | 32.6 | [26.4, 39.6] |
| math | 144 | 26.4 | [19.4, 34.0] |
| other | 142 | 43.7 | [35.9, 52.1] |
| philosophy | 143 | 58.7 | [51.0, 67.1] |
| physics | 143 | 34.3 | [26.6, 42.7] |
| psychology | 143 | 69.2 | [61.5, 76.2] |

- 计算记账（预热不计）：2541 题，2541 次调用，生成 133181 token，提示 464410 token，调用耗时 4581.9 s；整次墙钟 4890.41 s
- IFEval 诊断：思考段未闭合 0，空回答 0，写满 1024 token 27
- 命令：`python scripts/lang_battery.py --model qwen3.5-9b --raw-dir /home/mail/chenyuma/visint/data/lang_battery_raw --label w2c --data-root /home/mail/chenyuma/visint/b200/data --checkpoint /home/mail/chenyuma/visint/repo/experiments/W2/0069_w2c_hpcgpu06/0069_train/w2c/qwen3.5-9b/seed0/ckpt/final/model --out /home/mail/chenyuma/visint/repo/experiments/W2/0074_lang_battery_hpcgpu06/lb_w2c`
