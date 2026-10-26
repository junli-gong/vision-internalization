# 语言读数（大题量）：e3b_c3

- 脚本：`scripts/lang_battery.py`（DEV-0064），battery v1；commit `06557387cc6711b4cfa4e154a4308861b3e6633d`；模式 real；--limit None
- 模型：`qwen3.5-9b`（Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a，cuda:0，torch.bfloat16）
- 检查点：`/home/mail/chenyuma/visint/ckpt/hpcgpu06/0056/e3b_c3/qwen3.5-9b/seed0/ckpt/final/model`，manifest sha256 `02f416c93689a72da0f7c941fbfaf3d2320e6afe4448e38248710630fc9de3f6`；评测前的合并核对 pass=True；之后 load_trained(merge=True)
- MMLU-Pro：TIGER-Lab/MMLU-Pro@b189ec765aa7ed75c8acfea42df31fdae71f97be test（MIT），文件 sha256 `0e24a191921c2f453518a537a8b2117bd137e7714d4ef1565e9ba06c1ecb9ad8`（与固定值相同：True）。总体 12032 题去掉健康检查的 100 题，按 `lang_battery_v1` 分层抽 2000 题；与健康检查重叠 0 题。
- IFEval：google/IFEval@966cd89545d6b6acfd7638bc708b98261ca58e84（Apache-2.0），全部 541 条；其中 50 条也是健康检查的题（逐题 meta.in_health_check）。
- 数据只限非商业研究，不再分发；本目录只存题目 id、提示的 sha256 与模型输出，不存题面。
- 口径与健康检查 v3 相同：MMLU-Pro 受限字母读数（option_logits）、IFEval 官方 checker；关思考、greedy、零张图。

| 读数 | 口径 | n | 点估计（%） | bootstrap 95% CI |
|---|---|---:|---:|---|
| mmlu_pro_letter | accuracy | 2000 | 42.0 | [39.8, 44.1] |
| mmlu_pro_letter | mean_p_gold | 2000 | 36.3 | [34.8, 37.8] |
| ifeval | prompt_level_strict | 541 | 59.3 | [55.1, 63.6] |
| ifeval | instruction_level_strict | 834 | 68.6 | [65.3, 71.6] |
| ifeval | prompt_level_loose | 541 | 61.4 | [57.3, 65.4] |
| ifeval | instruction_level_loose | 834 | 70.9 | [67.7, 73.9] |

MMLU-Pro 分学科（准确率）：

| 学科 | n | % | 95% CI |
|---|---:|---:|---|
| biology | 143 | 72.7 | [65.0, 79.7] |
| business | 143 | 22.4 | [16.1, 29.4] |
| chemistry | 144 | 27.1 | [20.1, 34.7] |
| computer science | 142 | 39.4 | [31.0, 47.2] |
| economics | 143 | 59.4 | [51.0, 67.8] |
| engineering | 142 | 25.4 | [19.0, 32.4] |
| health | 142 | 51.4 | [43.0, 59.2] |
| history | 142 | 47.2 | [39.4, 55.6] |
| law | 144 | 31.9 | [25.0, 40.3] |
| math | 144 | 26.4 | [19.4, 33.4] |
| other | 142 | 40.8 | [32.4, 49.3] |
| philosophy | 143 | 46.9 | [38.5, 55.2] |
| physics | 143 | 32.2 | [24.5, 40.6] |
| psychology | 143 | 64.3 | [56.6, 72.0] |

- 计算记账（预热不计）：2541 题，2541 次调用，生成 204258 token，提示 464410 token，调用耗时 7060.2 s；整次墙钟 7503.52 s
- IFEval 诊断：思考段未闭合 0，空回答 0，写满 1024 token 120
- 命令：`python scripts/lang_battery.py --model qwen3.5-9b --raw-dir /home/mail/chenyuma/visint/data/lang_battery_raw --label e3b_c3 --data-root /home/mail/chenyuma/visint/b200/data --checkpoint /home/mail/chenyuma/visint/ckpt/hpcgpu06/0056/e3b_c3/qwen3.5-9b/seed0/ckpt/final/model --out /home/mail/chenyuma/visint/repo/experiments/W2/0074_lang_battery_hpcgpu06/lb_e3b_c3`
