# 语言读数（大题量）：x2_4b

- 脚本：`scripts/lang_battery.py`（DEV-0064），battery v1；commit `ea3fd8548a486a1c8f0506808f72f6c3090e81a0`；模式 real；--limit None
- 模型：`qwen3.5-4b`（Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a，cuda:0，torch.bfloat16）
- 检查点：`/home/mail/chenyuma/visint/repo/experiments/X/0098_x2_4b_hpcgpu07/0098_train/w2c/qwen3.5-4b/seed0/ckpt/final/model`，manifest sha256 `1ce5dc51c6fdc04f39264a84eb3d8bda4829e5cb3bc7783035c5958546c57ee9`；评测前的合并核对 pass=True；之后 load_trained(merge=True)
- MMLU-Pro：TIGER-Lab/MMLU-Pro@b189ec765aa7ed75c8acfea42df31fdae71f97be test（MIT），文件 sha256 `0e24a191921c2f453518a537a8b2117bd137e7714d4ef1565e9ba06c1ecb9ad8`（与固定值相同：True）。总体 12032 题去掉健康检查的 100 题，按 `lang_battery_v1` 分层抽 2000 题；与健康检查重叠 0 题。
- IFEval：google/IFEval@966cd89545d6b6acfd7638bc708b98261ca58e84（Apache-2.0），全部 541 条；其中 50 条也是健康检查的题（逐题 meta.in_health_check）。
- 数据只限非商业研究，不再分发；本目录只存题目 id、提示的 sha256 与模型输出，不存题面。
- 口径与健康检查 v3 相同：MMLU-Pro 受限字母读数（option_logits）、IFEval 官方 checker；关思考、greedy、零张图。

| 读数 | 口径 | n | 点估计（%） | bootstrap 95% CI |
|---|---|---:|---:|---|
| mmlu_pro_letter | accuracy | 2000 | 46.2 | [44.1, 48.4] |
| mmlu_pro_letter | mean_p_gold | 2000 | 42.2 | [40.5, 44.0] |
| ifeval | prompt_level_strict | 541 | 77.3 | [73.9, 80.6] |
| ifeval | instruction_level_strict | 834 | 84.2 | [81.8, 86.5] |
| ifeval | prompt_level_loose | 541 | 81.3 | [78.0, 84.7] |
| ifeval | instruction_level_loose | 834 | 87.2 | [84.9, 89.4] |

MMLU-Pro 分学科（准确率）：

| 学科 | n | % | 95% CI |
|---|---:|---:|---|
| biology | 143 | 74.8 | [67.1, 81.8] |
| business | 143 | 23.1 | [16.8, 30.1] |
| chemistry | 144 | 29.9 | [22.9, 36.8] |
| computer science | 142 | 48.6 | [40.1, 57.0] |
| economics | 143 | 61.5 | [53.1, 69.9] |
| engineering | 142 | 31.0 | [23.9, 38.7] |
| health | 142 | 57.0 | [49.3, 64.8] |
| history | 142 | 50.7 | [42.9, 58.5] |
| law | 144 | 36.1 | [29.1, 43.8] |
| math | 144 | 28.5 | [21.5, 36.1] |
| other | 142 | 44.4 | [35.9, 52.8] |
| philosophy | 143 | 51.7 | [44.1, 59.4] |
| physics | 143 | 41.3 | [33.6, 49.7] |
| psychology | 143 | 68.5 | [60.8, 75.5] |

- 计算记账（预热不计）：2541 题，2541 次调用，生成 175136 token，提示 464410 token，调用耗时 6102.5 s；整次墙钟 6531.76 s
- IFEval 诊断：思考段未闭合 0，空回答 0，写满 1024 token 32
- 命令：`python scripts/lang_battery.py --model qwen3.5-4b --raw-dir /home/mail/chenyuma/visint/data/lang_battery_raw --label x2_4b --data-root /home/mail/chenyuma/visint/b200/data --checkpoint /home/mail/chenyuma/visint/repo/experiments/X/0098_x2_4b_hpcgpu07/0098_train/w2c/qwen3.5-4b/seed0/ckpt/final/model --out /home/mail/chenyuma/visint/repo/experiments/X/0101_lb_4b_hpcgpu07/lb_x2_4b`
