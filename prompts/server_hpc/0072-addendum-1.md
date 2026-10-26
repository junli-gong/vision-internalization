[RUN-TASK 0072 补充 1] hpcgpu06：补跑 W2 test 里 W2-c 一项（`--setting w2c --checkpoint`），只跑 RUN-0072 没有跑完的划分。排在 RUN-0074 之后，约 0.5–2.5 GPU 小时，上限 4 小时

## 铁律

`~/a100-hpc-guide.md` 第 0 节：**任何情况下都不能取消或停止 Slurm 任务 157296 和 157297。**
- 前后各跑一次 `squeue`。
- 不碰 RUN-0072 主驱动与 RUN-0074、0076、0077。
- 只用 06 自己的暂停文件。

## 为什么

- RUN-0072 的 testA 上，w2c 一步 rc=2：W2-c 的模型目录里多了 `adapter_model.safetensors.gz` 与 `LARGE_FILES.sha256`，是 D127 恢复时我让保留的，而加载器要求目录与 manifest 恰好一致。
- 04:40Z 两份文件已移到 `ckpt/final/lora_gz_archive/`，模型目录与 manifest 一致，`read_manifest` 通过。
- RUN-0072 后面几个划分的 w2c 步骤会重新加载检查点，应当能跑通。这里补跑那时仍没有完整结果的划分，至少有 testA。

## 文件（工作树 wt0072r 建在 origin/main，只用来取驱动；python 的代码用 RUN-0072 自己的工作树 wt0072）

| 文件 | sha256 |
|---|---|
| `prompts/server_hpc/0072-addendum-1/rerun_w2c.sh` | `b610f00841eb5563ac36ad2652c2a465e7694c3ef7feef7324bedd22f7b61ca7` |

- 驱动里 `CODE=$VI/wt/wt0072`：`run_w2.py`、`eval_lib.sh` 都与 RUN-0072 的其余设置同一个 commit（d38a5205），即 RUN-0072 任务书哈希表里的那些版本。开跑前再核对一次 wt0072 里那几个文件的 sha256。

## 内容（`rerun_w2c.sh`）

- 开头核对 W2-c 模型目录与 manifest 恰好一致，不一致就拒跑。
- `claim_gpu` 等 `eval_0074.status` 不再是 running，且 GPU 上没有进程。
- 对 5 个 test 划分逐个处理：`$OUT/qwen3.5-9b/refseg/w2c/<划分>/metrics.json` 已有 `complete: true` 的跳过，并在 `steps.log` 里写明；其余跑 `run_w2.py --task refseg --setting w2c --checkpoint <W2-c>`，参数与 RUN-0072 相同，OUT 也是 RUN-0072 的 OUT。
- 心跳在 `$VI/logs/eval_0072r.status`，`steps.log` 接着写在 RUN-0072 的 `steps.log` 里。

## 做法

1. hpcmgt 上 `squeue`。建工作树 wt0072r（稀疏），建在 origin/main，核对上表的 sha256。
2. **先看 testA 失败时留下了什么**：
   - 列出 `$OUT/qwen3.5-9b/refseg/w2c/` 与 `$OUT/qwen3.5-9b/refexp_box/` 下 testA 相关的目录，以及 `$OUT/qwen3.5-9b/errors/`。
   - 如果 testA 的 w2c 或 propose_w2c 有写了一半的目录（有文件，但没有 complete 的 metrics.json），**不要删**，报告给我，等我回复。
   - 只有错误记录、没有半成品目录，就照常发。
3. **启动时机**：RUN-0074 的心跳显示 "running lb_base" 之后再启动，避免两个驱动同时抢 GPU：
   ```bash
   ssh -n hpcgpu06 'setsid nohup srun --jobid=157296 --overlap -n1 -c 8 \
     env PATH=/home/mail/chenyuma/visint/env/bin:/usr/bin:/bin TINY=0 EV_CAP_H=4 \
     bash /home/mail/chenyuma/visint/wt/wt0072r/prompts/server_hpc/0072-addendum-1/rerun_w2c.sh \
     > /home/mail/chenyuma/visint/logs/eval_0072r.driver.log 2>&1 < /dev/null &'
   ```
   - 等待期间心跳是 "waiting for gpu"。
   - 0072 的推送循环（PID 535295）一直开到补跑结束、推完。
4. 每个补跑的划分报 cIoU 与 95% CI，跳过的划分报 RUN-0072 自己那次的 cIoU。
5. 全部跑完：在 `reports/0072-hpc-06-w2-test-9b.md` 里加一节"W2-c 补跑"，再跑一次 `squeue`。

## 报告

- testA 失败的原文与模型目录的修复记录（04:40Z 的移动、核对、`read_manifest`）；
- 补跑与跳过的划分清单；
- `steps.log` 里补跑部分的原文；
- 5 个划分的 w2c cIoU、gIoU；
- 显存峰值、任何 traceback。
