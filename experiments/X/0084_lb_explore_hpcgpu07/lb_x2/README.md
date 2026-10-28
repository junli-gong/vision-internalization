# 语言读数（大题量）：x2

- 脚本：`scripts/lang_battery.py`（DEV-0064），battery v1；commit `46dff4ae14fa8c07583883b05fa4f05f5c24a4c1`；模式 real；--limit None
- 模型：`qwen3.5-9b`（Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a，cuda:0，torch.bfloat16）
- 检查点：`/home/mail/chenyuma/visint/repo/experiments/X/0080_x2_hpcgpu07/0080_train/w2c/qwen3.5-9b/seed0/ckpt/final/model`，manifest sha256 `a49cf4e5a81fd8f7f57ca5c86a879cf24cdd1449daa43cc355e6d3a074a6fd0a`；评测前的合并核对 pass=True；之后 load_trained(merge=True)
- MMLU-Pro：TIGER-Lab/MMLU-Pro@b189ec765aa7ed75c8acfea42df31fdae71f97be test（MIT），文件 sha256 `0e24a191921c2f453518a537a8b2117bd137e7714d4ef1565e9ba06c1ecb9ad8`（与固定值相同：True）。总体 12032 题去掉健康检查的 100 题，按 `lang_battery_v1` 分层抽 2000 题；与健康检查重叠 0 题。
- IFEval：google/IFEval@966cd89545d6b6acfd7638bc708b98261ca58e84（Apache-2.0），全部 541 条；其中 50 条也是健康检查的题（逐题 meta.in_health_check）。
- 数据只限非商业研究，不再分发；本目录只存题目 id、提示的 sha256 与模型输出，不存题面。
- 口径与健康检查 v3 相同：MMLU-Pro 受限字母读数（option_logits）、IFEval 官方 checker；关思考、greedy、零张图。

| 读数 | 口径 | n | 点估计（%） | bootstrap 95% CI |
|---|---|---:|---:|---|
| mmlu_pro_letter | accuracy | 2000 | 54.0 | [51.9, 56.0] |
| mmlu_pro_letter | mean_p_gold | 2000 | 48.4 | [46.8, 50.1] |
| ifeval | prompt_level_strict | 541 | 80.2 | [76.9, 83.4] |
| ifeval | instruction_level_strict | 834 | 86.1 | [83.8, 88.5] |
| ifeval | prompt_level_loose | 541 | 83.2 | [80.0, 86.1] |
| ifeval | instruction_level_loose | 834 | 88.5 | [86.4, 90.4] |

MMLU-Pro 分学科（准确率）：

| 学科 | n | % | 95% CI |
|---|---:|---:|---|
| biology | 143 | 81.8 | [74.8, 88.1] |
| business | 143 | 32.9 | [25.9, 41.3] |
| chemistry | 144 | 36.8 | [28.5, 44.4] |
| computer science | 142 | 54.2 | [45.1, 62.0] |
| economics | 143 | 67.1 | [59.4, 74.8] |
| engineering | 142 | 34.5 | [27.5, 43.0] |
| health | 142 | 66.9 | [59.1, 74.7] |
| history | 142 | 66.9 | [59.2, 74.6] |
| law | 144 | 43.1 | [35.4, 51.4] |
| math | 144 | 36.1 | [28.5, 44.4] |
| other | 142 | 52.8 | [44.4, 61.3] |
| philosophy | 143 | 58.7 | [50.3, 66.4] |
| physics | 143 | 49.7 | [41.3, 58.0] |
| psychology | 143 | 74.8 | [67.8, 81.1] |

- 计算记账（预热不计）：2541 题，2541 次调用，生成 167484 token，提示 464410 token，调用耗时 5854.4 s；整次墙钟 6986.83 s
- IFEval 诊断：思考段未闭合 0，空回答 0，写满 1024 token 31
- 命令：`python scripts/lang_battery.py --model qwen3.5-9b --raw-dir /home/mail/chenyuma/visint/data/lang_battery_raw --label x2 --data-root /home/mail/chenyuma/visint/b200/data --checkpoint /home/mail/chenyuma/visint/repo/experiments/X/0080_x2_hpcgpu07/0080_train/w2c/qwen3.5-9b/seed0/ckpt/final/model --out /home/mail/chenyuma/visint/repo/experiments/X/0084_lb_explore_hpcgpu07/lb_x2`
