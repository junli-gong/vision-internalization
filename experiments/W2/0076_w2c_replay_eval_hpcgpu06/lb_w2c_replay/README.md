# 语言读数（大题量）：w2c_replay

- 脚本：`scripts/lang_battery.py`（DEV-0064），battery v1；commit `237fbf453a6936381d5da699cec52fb7f6d53895`；模式 real；--limit None
- 模型：`qwen3.5-9b`（Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a，cuda:0，torch.bfloat16）
- 检查点：`/home/mail/chenyuma/visint/repo/experiments/W2/0076_w2c_replay_hpcgpu07/0076_train/w2c/qwen3.5-9b/seed0/ckpt/final/model`，manifest sha256 `ecad7667b4588f4840ba13f79b71b8c52d475e26a99c79a0e6d3999c99dcecfd`；评测前的合并核对 pass=True；之后 load_trained(merge=True)
- MMLU-Pro：TIGER-Lab/MMLU-Pro@b189ec765aa7ed75c8acfea42df31fdae71f97be test（MIT），文件 sha256 `0e24a191921c2f453518a537a8b2117bd137e7714d4ef1565e9ba06c1ecb9ad8`（与固定值相同：True）。总体 12032 题去掉健康检查的 100 题，按 `lang_battery_v1` 分层抽 2000 题；与健康检查重叠 0 题。
- IFEval：google/IFEval@966cd89545d6b6acfd7638bc708b98261ca58e84（Apache-2.0），全部 541 条；其中 50 条也是健康检查的题（逐题 meta.in_health_check）。
- 数据只限非商业研究，不再分发；本目录只存题目 id、提示的 sha256 与模型输出，不存题面。
- 口径与健康检查 v3 相同：MMLU-Pro 受限字母读数（option_logits）、IFEval 官方 checker；关思考、greedy、零张图。

| 读数 | 口径 | n | 点估计（%） | bootstrap 95% CI |
|---|---|---:|---:|---|
| mmlu_pro_letter | accuracy | 2000 | 52.2 | [50.1, 54.3] |
| mmlu_pro_letter | mean_p_gold | 2000 | 46.7 | [45.1, 48.3] |
| ifeval | prompt_level_strict | 541 | 80.0 | [76.5, 83.4] |
| ifeval | instruction_level_strict | 834 | 86.1 | [83.7, 88.3] |
| ifeval | prompt_level_loose | 541 | 83.2 | [79.9, 86.1] |
| ifeval | instruction_level_loose | 834 | 88.2 | [86.0, 90.2] |

MMLU-Pro 分学科（准确率）：

| 学科 | n | % | 95% CI |
|---|---:|---:|---|
| biology | 143 | 78.3 | [71.3, 85.3] |
| business | 143 | 32.9 | [25.2, 40.6] |
| chemistry | 144 | 34.0 | [26.4, 41.7] |
| computer science | 142 | 51.4 | [43.0, 59.2] |
| economics | 143 | 63.6 | [55.9, 71.3] |
| engineering | 142 | 35.9 | [28.2, 43.7] |
| health | 142 | 62.7 | [54.2, 70.4] |
| history | 142 | 64.1 | [56.3, 71.8] |
| law | 144 | 41.0 | [34.0, 48.6] |
| math | 144 | 32.6 | [25.0, 41.0] |
| other | 142 | 51.4 | [42.3, 60.6] |
| philosophy | 143 | 62.2 | [53.8, 70.6] |
| physics | 143 | 46.9 | [38.5, 55.2] |
| psychology | 143 | 73.4 | [66.4, 79.7] |

- 计算记账（预热不计）：2541 题，2541 次调用，生成 169015 token，提示 464410 token，调用耗时 5954.8 s；整次墙钟 6287.36 s
- IFEval 诊断：思考段未闭合 0，空回答 0，写满 1024 token 30
- 命令：`python scripts/lang_battery.py --model qwen3.5-9b --raw-dir /home/mail/chenyuma/visint/data/lang_battery_raw --label w2c_replay --data-root /home/mail/chenyuma/visint/b200/data --checkpoint /home/mail/chenyuma/visint/repo/experiments/W2/0076_w2c_replay_hpcgpu07/0076_train/w2c/qwen3.5-9b/seed0/ckpt/final/model --out /home/mail/chenyuma/visint/repo/experiments/W2/0076_w2c_replay_eval_hpcgpu06/lb_w2c_replay`
