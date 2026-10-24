# W2 去留判定：W2-c（D113 第六条，规则写死）

- 模型 qwen3.5-9b@c20223623576…，划分 devcheck2，n = 500；判定点核对：{'model_is_qwen3.5-9b': True, 'split_is_devcheck2': True, 'n_is_500': True, 'not_tiny': True, 'no_limit': True, 'w2_config_frozen': True, 'same_w2_config': True}

| 读数 | 值 |
|---|---|
| W2-c（w2c）cIoU | 0.6419 |
| W1-F（w1f）cIoU | 0.5777 |

- 主配置 − W1-F 的 cIoU 差（Σinter / Σunion），配对 bootstrap 97.5% CI（重抽样 10,000 次，种子 src/constants.BOOTSTRAP_SEED），下界 > 0 才算通过：差 0.0642（29157191865223/453995507587920），97.5% CI [0.0434, 0.0870]（10000 次，种子 20260929）→ **通过**

**结论：通过**

两个主配置（W2-a、W2-c）各判一次；"至少一个通过"由管理员在门报告里合起来写。
