# 语言读数（大题量）：xm

- 脚本：`scripts/lang_battery.py`（DEV-0064），battery v1；commit `b3e96ee4a090167fa2c9345719d586ca08cbf861`；模式 real；--limit None
- 模型：`qwen3.5-9b`（Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a，cuda:0，torch.bfloat16）
- 检查点：`/home/mail/chenyuma/visint/repo/experiments/XM/0104_xm_hpcgpu07/0104_xm_train/w2c/qwen3.5-9b/seed0/ckpt/final/model`，manifest sha256 `29e26667c7810ef2cf6f1704e72e4d025697c0ab974fe7a74f121ded972d4e8e`；评测前的合并核对 pass=True；之后 load_trained(merge=True)
- MMLU-Pro：TIGER-Lab/MMLU-Pro@b189ec765aa7ed75c8acfea42df31fdae71f97be test（MIT），文件 sha256 `0e24a191921c2f453518a537a8b2117bd137e7714d4ef1565e9ba06c1ecb9ad8`（与固定值相同：True）。总体 12032 题去掉健康检查的 100 题，按 `lang_battery_v1` 分层抽 2000 题；与健康检查重叠 0 题。
- IFEval：google/IFEval@966cd89545d6b6acfd7638bc708b98261ca58e84（Apache-2.0），全部 541 条；其中 50 条也是健康检查的题（逐题 meta.in_health_check）。
- 数据只限非商业研究，不再分发；本目录只存题目 id、提示的 sha256 与模型输出，不存题面。
- 口径与健康检查 v3 相同：MMLU-Pro 受限字母读数（option_logits）、IFEval 官方 checker；关思考、greedy、零张图。

| 读数 | 口径 | n | 点估计（%） | bootstrap 95% CI |
|---|---|---:|---:|---|
| mmlu_pro_letter | accuracy | 2000 | 52.9 | [50.7, 55.0] |
| mmlu_pro_letter | mean_p_gold | 2000 | 47.3 | [45.7, 49.0] |
| ifeval | prompt_level_strict | 541 | 80.6 | [77.1, 83.7] |
| ifeval | instruction_level_strict | 834 | 86.2 | [83.8, 88.5] |
| ifeval | prompt_level_loose | 541 | 83.0 | [79.9, 86.1] |
| ifeval | instruction_level_loose | 834 | 88.1 | [85.9, 90.2] |

MMLU-Pro 分学科（准确率）：

| 学科 | n | % | 95% CI |
|---|---:|---:|---|
| biology | 143 | 81.8 | [74.8, 88.1] |
| business | 143 | 30.1 | [23.1, 38.5] |
| chemistry | 144 | 34.0 | [26.4, 41.7] |
| computer science | 142 | 49.3 | [40.1, 57.1] |
| economics | 143 | 67.8 | [60.1, 75.5] |
| engineering | 142 | 34.5 | [27.4, 42.3] |
| health | 142 | 65.5 | [57.0, 73.9] |
| history | 142 | 68.3 | [60.6, 75.4] |
| law | 144 | 43.8 | [36.1, 52.1] |
| math | 144 | 31.9 | [24.3, 39.6] |
| other | 142 | 50.7 | [42.3, 59.2] |
| philosophy | 143 | 62.9 | [55.2, 70.6] |
| physics | 143 | 42.7 | [34.2, 51.0] |
| psychology | 143 | 76.9 | [70.6, 83.2] |

- 计算记账（预热不计）：2541 题，2541 次调用，生成 163899 token，提示 464410 token，调用耗时 5761.3 s；整次墙钟 6114.79 s
- IFEval 诊断：思考段未闭合 0，空回答 0，写满 1024 token 33
- 命令：`python scripts/lang_battery.py --model qwen3.5-9b --raw-dir /home/mail/chenyuma/visint/data/lang_battery_raw --label xm --data-root /home/mail/chenyuma/visint/b200/data --checkpoint /home/mail/chenyuma/visint/repo/experiments/XM/0104_xm_hpcgpu07/0104_xm_train/w2c/qwen3.5-9b/seed0/ckpt/final/model --out /home/mail/chenyuma/visint/repo/experiments/XM/0107_xm_refseg_lb_hpcgpu07/0107_lb_xm`
