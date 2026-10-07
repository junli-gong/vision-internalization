[RUN-TASK 0032 补充 1] RUN-0032 在 6/9 收尾：驱动在下载 InternVL-4B 时中断，按 15 小时上限不再续跑；写报告并标完成，然后依次做 0033、0034

## 情况（管理员 2026-10-07 02:50 UTC 核对）

- 驱动 `run_e5_32.sh` 在 2026-10-06 14:34 UTC 开始下载 internvl3.5-4b-hf 权重之后中断：之后没有日志，也没有进程。很可能是服务器 CC 的会话结束时，挂在它后台任务下的驱动被一起杀掉了。GPU 从那时起空闲。
- 已完成 6/9：qwen3.5-2b、LISA、qwen3.5-0.8b、qwen3.5-4b、qwen3.5-9b、internvl3.5-2b-hf，都已推回。累计约 14.3 GPU 小时，在 15 小时的上限之内。

## 要做的

1. **不续跑** internvl3.5-4b-hf、gemma-4-e2b-it、gemma-4-e4b-it。按任务书 §4 的上限，本次到此为止，报告里写明"在 6/9 收尾，原因：驱动中断时累计已近上限"。
2. 清理：内存盘上若有 internvl3.5-4b-hf 下载了一半的权重，删掉并记进 `disk_ledger.tsv`；只保留 qwen3.5-0.8b。
3. 按任务书 §5 写 `reports/0032-e5-test.md`，范围是已完成的 6 个系统，平凡基线全量。推回。
4. 把 `0032-e5-test-zeroshot.md` 与 `0032-addendum-1.md` 都追加进完成清单。
5. 依次做 `0033-e1-posthoc-followup.md` 与 `0034-b200-e3-prep.md`。
6. **新规则**（已写进 STANDBY.md）：超过 30 分钟的驱动脚本，一律用 `setsid nohup … &` 或 tmux 启动，与 CC 的进程脱钩。CC 自己的后台任务只用来等待和轮询，免得会话结束时把实验一起带走。
7. 你自己的仓库里有 `experiments/E1/0027_e1_main/*/refexp_box/predictions.jsonl` 这类未跟踪文件，大概是解压出来给 RUN-0033 用的。不要提交，用完删掉。
