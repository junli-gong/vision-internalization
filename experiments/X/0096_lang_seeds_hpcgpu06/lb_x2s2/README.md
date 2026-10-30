# 语言读数（大题量）：x2s2

- 脚本：`scripts/lang_battery.py`（DEV-0064），battery v1；commit `b5adf2b4638cf7a9b051c61d88ab66312b2daa75`；模式 real；--limit None
- 模型：`qwen3.5-9b`（Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a，cuda:0，torch.bfloat16）
- 检查点：`/home/mail/chenyuma/visint/repo/experiments/X/0094_x2s2_hpcgpu06/0094_train/w2c/qwen3.5-9b/seed2/ckpt/final/model`，manifest sha256 `836e69235e33b1c97809d76d8d475eab38e53a4a4b0a12b3e0860611a683f1c8`；评测前的合并核对 pass=True；之后 load_trained(merge=True)
- MMLU-Pro：TIGER-Lab/MMLU-Pro@b189ec765aa7ed75c8acfea42df31fdae71f97be test（MIT），文件 sha256 `0e24a191921c2f453518a537a8b2117bd137e7714d4ef1565e9ba06c1ecb9ad8`（与固定值相同：True）。总体 12032 题去掉健康检查的 100 题，按 `lang_battery_v1` 分层抽 2000 题；与健康检查重叠 0 题。
- IFEval：google/IFEval@966cd89545d6b6acfd7638bc708b98261ca58e84（Apache-2.0），全部 541 条；其中 50 条也是健康检查的题（逐题 meta.in_health_check）。
- 数据只限非商业研究，不再分发；本目录只存题目 id、提示的 sha256 与模型输出，不存题面。
- 口径与健康检查 v3 相同：MMLU-Pro 受限字母读数（option_logits）、IFEval 官方 checker；关思考、greedy、零张图。

| 读数 | 口径 | n | 点估计（%） | bootstrap 95% CI |
|---|---|---:|---:|---|
| mmlu_pro_letter | accuracy | 2000 | 51.9 | [49.8, 54.2] |
| mmlu_pro_letter | mean_p_gold | 2000 | 47.7 | [46.0, 49.4] |
| ifeval | prompt_level_strict | 541 | 81.9 | [78.6, 85.0] |
| ifeval | instruction_level_strict | 834 | 87.4 | [85.0, 89.6] |
| ifeval | prompt_level_loose | 541 | 85.2 | [81.9, 88.0] |
| ifeval | instruction_level_loose | 834 | 89.8 | [87.5, 91.6] |

MMLU-Pro 分学科（准确率）：

| 学科 | n | % | 95% CI |
|---|---:|---:|---|
| biology | 143 | 81.8 | [75.5, 88.1] |
| business | 143 | 27.3 | [20.3, 35.0] |
| chemistry | 144 | 34.0 | [26.4, 41.7] |
| computer science | 142 | 49.3 | [40.8, 57.0] |
| economics | 143 | 69.9 | [62.2, 76.9] |
| engineering | 142 | 33.1 | [25.4, 41.5] |
| health | 142 | 63.4 | [54.9, 71.1] |
| history | 142 | 66.2 | [58.5, 73.9] |
| law | 144 | 40.3 | [33.3, 47.9] |
| math | 144 | 32.6 | [25.0, 41.0] |
| other | 142 | 46.5 | [37.3, 54.9] |
| philosophy | 143 | 62.9 | [55.2, 70.6] |
| physics | 143 | 41.3 | [32.2, 49.7] |
| psychology | 143 | 78.3 | [72.0, 84.6] |

- 计算记账（预热不计）：2541 题，2541 次调用，生成 162823 token，提示 464410 token，调用耗时 5722.8 s；整次墙钟 6211.4 s
- IFEval 诊断：思考段未闭合 0，空回答 0，写满 1024 token 25
- 命令：`python scripts/lang_battery.py --model qwen3.5-9b --raw-dir /home/mail/chenyuma/visint/data/lang_battery_raw --label x2s2 --data-root /home/mail/chenyuma/visint/b200/data --checkpoint /home/mail/chenyuma/visint/repo/experiments/X/0094_x2s2_hpcgpu06/0094_train/w2c/qwen3.5-9b/seed2/ckpt/final/model --out /home/mail/chenyuma/visint/repo/experiments/X/0096_lang_seeds_hpcgpu06/lb_x2s2`
