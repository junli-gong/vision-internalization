[RUN-TASK 0061] hpcgpu07：W1 开发-调参，四个臂 T1–T4（W1 预注册 §10b；D108、D109；W1 预算内，约 4 GPU 小时）。**在 RUN-0057 的段边界让路**：claim_gpu 写 07 自己的暂停文件，RUN-0057 跑完当前段后停住；本任务跑完删掉暂停文件，RUN-0057 自己接着跑。上限 6 小时

## 铁律

`~/a100-hpc-guide.md` 第 0 节：**任何情况下都不能取消或停止 Slurm 任务 157296 和 157297。**
- 前后各跑一次 `squeue`。
- 不碰 06 上的 RUN-0056。
- **RUN-0057 不停、不杀**：只靠暂停文件 `$VI/logs/train.pause.hpcgpu07` 让它在段边界自己等着。脚本的 `claim_gpu` 写这个文件，退出时（含出错）自动删掉。
- 不写全局的 `$VI/logs/train.pause`。

## 为什么

- RUN-0060 冒烟里，9B 的完整工作台有 30/32 句到第 6 轮还没提交。原因：
  - 模型每轮只发 1 个调用；
  - 最后一轮还在调 `show` 等工具；
  - 写进工具的框偏粗。
- DEV-0053、DEV-0054 加了与任务无关的开关。四个臂逐级累加：

| 臂 | 加什么 |
|---|---|
| T1 | 最后一轮只给提交工具；调用满额后也只给提交工具；最后一轮的提示改写 |
| T2 | T1，加每轮报剩余预算 |
| T3 | T2，加系统提示两句：一轮可以发几个调用；坐标尽量精确 |
| T4 | T3，加视图坐标（`view_frames`），系统提示的坐标段改写 |

- 每个臂都跑 9B、指代分割、开发-调参集前 96 句。四种基线也在同样 96 句上跑。
- 选法事先写在 W1 预注册 §10b。只用开发-调参集，**不碰开发-检查集**：未冻结的配置碰开发-检查集，`run_w1.py` 会拒绝，退出码 2。

## 文件（commit `6038d755` 或更新；工作树 wt0061 建在 origin/main）

| 文件 | sha256 |
|---|---|
| `prompts/server_hpc/0061-hpc-07-w1-tune/tune.sh` | `3e2a7851f49a1c5b4b20f9899da185507bc5397f004d57968dc6bd6ced165661` |
| `configs/w1.json` | `5102fa228367d48dfdea8e29b58933536ef8c99ebefbc4983bcd64213503006f`（基线用；`frozen: false`） |
| `configs/w1_tune/T1.json` | `37e5a7546b1bf849d0093831f20c1ff068bcb8e716ea77131e3443bc8dadd82e` |
| `configs/w1_tune/T2.json` | `36c06a204cd359a119ff0f6f01db4d660ffe159970d50954b873ceec8c000bb4` |
| `configs/w1_tune/T3.json` | `16e7a6f3ae66355b8d62118f2acc88fbc37cbad0a292396a832c51d57e18b142` |
| `configs/w1_tune/T4.json` | `98548d1a77687b33bd8b93d0200ff236fcbe4b911b2a772f710f34930faaf15a` |
| `configs/w1_splits.json` | `a1371851555e854fac6f9d0e267f19a23602ee9cf31d5c303ef08432d1f7f0f7` |
| `scripts/run_w1.py` | `23770d71a72ff38ae80388ab906d8ad2b04f022a32c6617ca27b6eae5a6904b4` |
| `scripts/w1_gate.py` | `0826f7822886173913fad93db86be7901f752c384a2eae1b98d58ce24c83c208` |
| `scripts/server/eval_lib.sh` | `2380464beeabe300f2cfdd17ab1d79921b8d978b454b2d9f16ee42e4ca074a59` |
| `src/w1/loop.py` | `343089b833e23ed03bcec3de7f0eb595b747abcf2f5b487fc499d3901375833d` |
| `src/w1/workbench.py` | `775e0dca6901662ddd2e1880c01dbf848efaad73127f6d594308fcf797d46591` |

开发机 tiny 彩排：12 步都 rc=0。T4 的运行身份里 `view_frames: true`，`tools_sha256` 是 `0bdcaf73…`；T1 是 `dcc91402…`。

## 内容（全部在 tune.sh 里）

| 步 | 内容 |
|---|---|
| `base_e1_refexp_box_devtune96` | W1-0 的定位（"自己的框"与 W1-F 要用） |
| `base_ownbox_devtune96` | 自己的框光栅化，只用 CPU |
| `base_poly_structured_devtune96` | E1 结构强制多边形 |
| `base_w1f_devtune96` | 自己的框交给 GrabCut，只用 CPU |
| `arm_T1_refseg_devtune96` … `arm_T4_refseg_devtune96` | 四个臂，9B 完整工作台 |
| `gate_tune_T1` … `gate_tune_T4` | 每个臂对两条基线算一次 `w1_gate.py`。devtune 不是判定点，脚本自己会标"彩排，不作数" |

- 结果目录：`experiments/W1/0061_w1_tune_hpcgpu07/`，子目录 `base/`、`T1/` … `T4/`。
- 全程每秒采样一次显存，写 `gpu_mem.csv`。
- 预计：基线约 10 分钟；每个臂约 40–60 分钟（RUN-0060 的 32 句用了 995 s）。合计约 4 小时，另加最多一段（约 2 小时）等 RUN-0057 到段边界。

## 做法

1. hpcmgt 上 `squeue`。
2. 建工作树 wt0061（稀疏，同 RUN-0054 的规则），建在 origin/main。核对上表 12 个文件的 sha256。
3. 启动（方案 B），**现在就可以发**。claim_gpu 会等 RUN-0057 到段边界，最多 3 小时：

   ```bash
   ssh -n hpcgpu07 'setsid nohup srun --jobid=157297 --overlap -n1 \
     env PATH=/home/mail/chenyuma/visint/env/bin:/usr/bin:/bin EV_CAP_H=6 \
     bash /home/mail/chenyuma/visint/wt/wt0061/prompts/server_hpc/0061-hpc-07-w1-tune/tune.sh \
     > /home/mail/chenyuma/visint/logs/eval_0061.driver.log 2>&1 < /dev/null &'
   ```

   - 心跳在 `$VI/logs/eval_0061.status`。先是 `waiting for gpu …`，接着 `gpu claimed`，然后 `running <步名>`。
   - 另起一个 v3 推送循环，TAG 0061，路径 `experiments/W1/0061_w1_tune_hpcgpu07`。跑完、推完后按 PID 停掉。
4. 拿到 GPU 后报一次：时间，以及 RUN-0057 停在哪一段之后（`train_0057-hpcgpu07.status` 原文）。
5. `arm_T1` 跑完先报一次：
   - 墙钟；
   - 96 句里提交的条数，以及各失败码的条数。
6. 全部跑完：
   - 确认暂停文件已删、RUN-0057 已接着跑（心跳原文）；
   - 写报告；
   - 再跑一次 `squeue`。
   - **出错或触线**：暂停文件由 `claim_gpu` 的 trap 删除。要核对它确实没了，RUN-0057 恢复了。

## 报告

推送 `reports/0061-hpc-07-w1-tune.md`，内容：
- `steps.log` 原文与每步的墙钟。
- 每个臂各一张表：
  - 提交数；失败码分布；
  - 轮数与计数调用次数的分布；
  - **每轮调用个数的分布**（1、2、3……个的轮数）；
  - 被拒调用按原因（`final_round`、`tool_cap`、`after_submission`）计数；
  - 每句生成 token 数的中位与最大；
  - 指代分割 cIoU（`metrics.json` 的标题读数与 CI）。
- 基线四步的标题读数。
- `gate_tune_T1.md` … `gate_tune_T4.md` 原文。
- 显存峰值（`gpu_mem.csv`）。
- 任何报错的 traceback。

devtune 的分数可以照贴，这是调参集。只贴数字，不解读。
