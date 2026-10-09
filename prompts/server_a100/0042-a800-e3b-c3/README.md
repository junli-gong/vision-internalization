# RUN-0042：A800 上的 E3b+C3（0.8B 与 4B，各 3 个种子）

**给协助运行的合作者。** 这是补充 1 里预告的新包。

另一台机器的结果显示，原来的 E3b 训练会冲掉语言能力。负责人改用 **E3b+C3**：每一步同时学网格输出与普通文本回答，用来保住语言。A800 负责 0.8B 与 4B。

- **沿用工作目录** `/root/autodl-tmp/visint-0022`：venv、权重、`data/` 都不动，只换代码快照。
- **先跑完补充 1 的第 2c 步（C3）**，再开始本包。第 2c 步还在跑的话，等它跑完；结果与检查点照常用 RUN-0035 包的 `step4_collect.sh` 回传。
- **用时**（A800 全程要 GPU，建议放在 tmux 里）：
  - 第 1 步约 20 分钟；
  - 第 2 步（0.8B）约 18 小时；
  - 第 3 步（4B）约 39 小时，每个种子约 13 小时。
- **不需要改任何代码**。遇到报错不要自己修，原样记下来，写进 notes.txt。

## 包里有什么

| 文件 | 作用 |
|---|---|
| `vision-internalization-<commit>.tar.gz` | 新的代码快照，完整 commit 见 `VERSION.txt` |
| `common.sh` | 各步共用的环境设置，不单独运行 |
| `step1_update_code.sh` | 第 1 步：换代码；E3b+C3 的吞吐实测（0.8B、4B 各 20 步） |
| `step2_e3b_c3_0p8b.sh` | 第 2 步：0.8B 的 E3b+C3 × 3 个种子 |
| `step3_e3b_c3_4b.sh` | 第 3 步：4B 的 E3b+C3 × 3 个种子 |
| `step4_collect.sh` | 打包结果与最终检查点，随时可以运行 |

## 步骤

```bash
W=/root/autodl-tmp/visint-0022
bash step1_update_code.sh $W          # 约 20 分钟
bash step4_collect.sh $W              # 回传一次（吞吐）
bash step2_e3b_c3_0p8b.sh $W          # 约 18 小时，放 tmux
bash step4_collect.sh $W              # 0.8B 跑完立即回传（结果 + 检查点）
bash step3_e3b_c3_4b.sh $W            # 约 39 小时，放 tmux
bash step4_collect.sh $W              # 全部跑完再回传一次
```

- 每一步的退出码记在 `<工作目录>/results_0042/steps.log`；当前在跑哪一个，写在 `results_0042/progress.txt`。
- **中断了**：重新运行同一条命令即可。已完成的运行自动跳过，没完成的从检查点接着跑。
- 某个运行报错（`steps.log` 里 rc ≠ 0）：脚本会继续跑下一个。把 `results_0042/logs/<运行名>.log` 原样回传。

## 之后（不急，等负责人另行通知）

本包跑完后，原 RUN-0035 包的第 2、3 步（其余变体，作为消融）可以照常跑。它们会用本包换上的新代码；补充 1 里已经训完的 C3 会自动跳过。

## 回传与规则

- `step4_collect.sh` 生成 `a800_results_0042_<时间>.tar.gz` 与 `a800_ckpt_0042_<时间>.tar.gz`。连同 notes.txt 上传到 Google Drive 的 `vision-internalization-a100/`。
- 不修改代码、配置与测试；模型与数据只用于本项目的非商业研究，不再转发给第三方；只在工作目录里读写，不改机器上的全局配置；跑完之后检查点留着。
