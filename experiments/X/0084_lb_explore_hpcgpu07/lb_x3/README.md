# 语言读数（大题量）：x3

- 脚本：`scripts/lang_battery.py`（DEV-0064），battery v1；commit `46dff4ae14fa8c07583883b05fa4f05f5c24a4c1`；模式 real；--limit None
- 模型：`qwen3.5-9b`（Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a，cuda:0，torch.bfloat16）
- 检查点：`/home/mail/chenyuma/visint/repo/experiments/X/0082_x3_hpcgpu07/0082_train/w2c/qwen3.5-9b/seed0/ckpt/final/model`，manifest sha256 `837bb679cfc172ecfa4bce3af1ae21c4e1774d3d9fdf0e16fc4e5d2c34ed7f5c`；评测前的合并核对 pass=True；之后 load_trained(merge=True)
- MMLU-Pro：TIGER-Lab/MMLU-Pro@b189ec765aa7ed75c8acfea42df31fdae71f97be test（MIT），文件 sha256 `0e24a191921c2f453518a537a8b2117bd137e7714d4ef1565e9ba06c1ecb9ad8`（与固定值相同：True）。总体 12032 题去掉健康检查的 100 题，按 `lang_battery_v1` 分层抽 2000 题；与健康检查重叠 0 题。
- IFEval：google/IFEval@966cd89545d6b6acfd7638bc708b98261ca58e84（Apache-2.0），全部 541 条；其中 50 条也是健康检查的题（逐题 meta.in_health_check）。
- 数据只限非商业研究，不再分发；本目录只存题目 id、提示的 sha256 与模型输出，不存题面。
- 口径与健康检查 v3 相同：MMLU-Pro 受限字母读数（option_logits）、IFEval 官方 checker；关思考、greedy、零张图。

| 读数 | 口径 | n | 点估计（%） | bootstrap 95% CI |
|---|---|---:|---:|---|
| mmlu_pro_letter | accuracy | 2000 | 53.5 | [51.4, 55.6] |
| mmlu_pro_letter | mean_p_gold | 2000 | 47.6 | [46.0, 49.3] |
| ifeval | prompt_level_strict | 541 | 81.1 | [77.8, 84.5] |
| ifeval | instruction_level_strict | 834 | 87.1 | [84.7, 89.2] |
| ifeval | prompt_level_loose | 541 | 85.0 | [81.9, 88.0] |
| ifeval | instruction_level_loose | 834 | 89.6 | [87.4, 91.7] |

MMLU-Pro 分学科（准确率）：

| 学科 | n | % | 95% CI |
|---|---:|---:|---|
| biology | 143 | 81.1 | [74.8, 87.4] |
| business | 143 | 33.6 | [25.9, 42.0] |
| chemistry | 144 | 34.7 | [27.1, 42.4] |
| computer science | 142 | 51.4 | [43.0, 59.9] |
| economics | 143 | 69.9 | [62.2, 76.9] |
| engineering | 142 | 35.2 | [27.5, 43.7] |
| health | 142 | 63.4 | [55.6, 71.1] |
| history | 142 | 69.0 | [61.3, 76.1] |
| law | 144 | 39.6 | [32.6, 47.9] |
| math | 144 | 33.3 | [25.7, 41.7] |
| other | 142 | 51.4 | [43.0, 59.9] |
| philosophy | 143 | 63.6 | [55.9, 72.0] |
| physics | 143 | 46.2 | [37.8, 54.5] |
| psychology | 143 | 76.2 | [69.2, 83.2] |

- 计算记账（预热不计）：2541 题，2541 次调用，生成 159592 token，提示 464410 token，调用耗时 5631.1 s；整次墙钟 8053.01 s
- IFEval 诊断：思考段未闭合 0，空回答 0，写满 1024 token 20
- 命令：`python scripts/lang_battery.py --model qwen3.5-9b --raw-dir /home/mail/chenyuma/visint/data/lang_battery_raw --label x3 --data-root /home/mail/chenyuma/visint/b200/data --checkpoint /home/mail/chenyuma/visint/repo/experiments/X/0082_x3_hpcgpu07/0082_train/w2c/qwen3.5-9b/seed0/ckpt/final/model --out /home/mail/chenyuma/visint/repo/experiments/X/0084_lb_explore_hpcgpu07/lb_x3`
