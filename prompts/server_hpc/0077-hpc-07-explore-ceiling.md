[RUN-TASK 0077] hpcgpu07，只用 CPU：探索 X-1 的上限测量，first 网格（D126-3、D133；探索日志 X-1）。约 30 CPU 小时，12 核约 3 小时，不占 GPU，现在就可以发

## 铁律

`~/a100-hpc-guide.md` 第 0 节：**任何情况下都不能取消或停止 Slurm 任务 157296 和 157297。**
- 前后各跑一次 `squeue`。
- 不碰 RUN-0072、0073、0074。
- 不用 GPU，不写暂停文件，不调 `claim_gpu`。

## 为什么

- 探索 X-0 发现，W2-c 的损失 58% 出在框已经很好的句子上；瓶颈是从框到 mask。
- X-1 用真值模拟一个"理想 LLM"，测四种 LLM 输出通道配上确定性工具，最高能把 mask 推到多少：
  - 前景/背景点；
  - B 样条轮廓；
  - 带符号的局部修正；
  - 更大的生成器库。
- 结果决定探索的 GPU 先投在哪个通道。
- 只用不含学习的工具，不跑任何模型；tokenizer 只用来数 token。
- 数据只用开发集 devtune800 与 devcheck2，**不碰 test 与 devcheck3**，脚本也会拒绝。

## 文件（commit 见发出时的启动语；工作树 wt0077 建在 origin/main）

| 文件 | sha256 |
|---|---|
| `src/explore/__init__.py` | `c8d4accca2e9123fc3f1476d7173c2fa7f46e55c9ecc8f5750600bed76a3a2ad` |
| `src/explore/guidance.py` | `40c52ebc0cd6eabe0e6d51c9f6906530fb8fb4a0c5a63b8cef5d912de7ee5ef5` |
| `src/explore/oracle_sim.py` | `5ada0675c52384b67c2c463deacde7c9209cff057ee847eb0424259c3499cbbd` |
| `src/explore/ceiling.py` | `dd5bf4d9eda0f512efcc7001b7dcc3c2c623b7f69bed419010155b1f42041c4a` |
| `scripts/explore_ceiling.py` | `65a5f6da86bad434e22f32e72c7dd3f74ed2f159ef0a20005678ca100fd8c17e` |
| `configs/explore_ceiling.json` | `7d9693e9c1402fe0dbfce25b5a8f497ac324441e83605c0c60e50c018b0fdce8` |

- 框与起点 mask 读 `$VI/repo` 里的已有结果，路径写在 `configs/explore_ceiling.json` 的 `box_sources`、`start_masks`：
  - RUN-0069 的 `0069_eval/qwen3.5-9b/refexp_box/{baseline_e1,propose_w2c}/devcheck2` 与 `refseg/w2c/devcheck2`；
  - RUN-0070 的 `0070_sep_on_k4/qwen3.5-9b/refexp_box/baseline_e1/devtune800`。

## 公共环境

```bash
VI=/home/mail/chenyuma/visint
ENVV="PATH=$VI/env/bin:/usr/bin:/bin CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONHASHSEED=0 \
  HF_HOME=$VI/hf HF_HUB_CACHE=$VI/hf/hub HF_HUB_OFFLINE=1 TMPDIR=$VI/tmp PIP_NO_CACHE_DIR=1"
```

## 做法

1. hpcmgt 上 `squeue`。
2. 建工作树 wt0077（稀疏，规则同 RUN-0072），建在 origin/main。fetch 失败就重试。核对上表 6 个 sha256。
3. **探测与换算**：在 07 上用 `srun --jobid=157297 --overlap -n1 -c 1`。
   - 先记 `nproc`、`uptime`、`free -g`。
   - 再跑 `python scripts/explore_ceiling.py --bench --bench-grids first --out $VI/tmp/bench0077`，单进程约 8 分钟。
   - 报 bench 打印的每 job 用时与估计墙钟，与开发机 M4 的 36 s / job 比。
4. **正式运行**：工作树是稀疏检出，没有 experiments/，所以框与起点 mask 用 `--box-run`、`--start-mask` 指向 `$VI/repo` 里的绝对路径。覆盖对一次运行里的所有划分生效，所以两个划分分开跑，写进同一个 `--out`。

   ```bash
   R=$VI/repo/experiments/W2
   cat > $VI/tmp/run0077.sh <<EOS
   set -e
   cd $VI/wt/wt0077
   C="python scripts/explore_ceiling.py --data-root $VI/b200/data --grid first --workers 12 --out $VI/repo/experiments/X/0077_ceiling_hpcgpu07"
   \$C --splits devtune800 --box-run base=$R/0070_w2_tune_r2_hpcgpu06/0070_sep_on_k4/qwen3.5-9b/refexp_box/baseline_e1/devtune800 \$RES
   \$C --splits devcheck2 --box-run base=$R/0069_w2c_hpcgpu06/0069_eval/qwen3.5-9b/refexp_box/baseline_e1/devcheck2 \\
     --box-run w2c=$R/0069_w2c_hpcgpu06/0069_eval/qwen3.5-9b/refexp_box/propose_w2c/devcheck2 \\
     --start-mask w2c_final=w2c:$R/0069_w2c_hpcgpu06/0069_eval/qwen3.5-9b/refseg/w2c/devcheck2 \$RES
   EOS
   ssh -n hpcgpu07 "setsid nohup srun --jobid=157297 --overlap -n1 -c 12 env $ENVV RES= bash $VI/tmp/run0077.sh \
     > $VI/logs/run_0077.log 2>&1 < /dev/null &"
   ```

   - 写好 `run0077.sh` 后先 `cat` 一遍，核对路径都展开成绝对路径、四个结果目录都存在。
   - 中断了就用 `RES=--resume` 重启同一个脚本接着跑。
   - **与 RUN-0073 共用节点**：开跑后 10 分钟内看一次 `uptime`，以及 RUN-0073 当前步的进度。如果负载明显超过核数，或 RUN-0073 的候选生成明显变慢，就停掉本任务的进程（只停 python，不碰 Slurm 任务本身），把脚本里的 12 改成 6、`-c 6`、`RES=--resume` 重启，并报告。
   - 墙钟超过 6 小时还没跑完，报一次进度。
5. **推送**：跑完后用 `sync_push.sh` 的 ONCE=1 推 `experiments/X/0077_ceiling_hpcgpu07`。`predictions.jsonl` 若超过推送的大文件线，按 sync 的约定压缩，原文件的 sha256 记进 `LARGE_FILES.sha256`。再推送报告。
6. 再跑一次 `squeue`。

## 报告

推送 `reports/0077-hpc-07-explore-ceiling.md`，内容：
- 第 3 步的 nproc、负载、bench 输出原文；
- 正式运行的命令、退出码、墙钟；有没有因负载降过核数；
- 输出目录的文件清单与大小；
- 两个划分各自的 `summary.md` 原文（脚本生成的）；
- 任何警告，以及局部 GrabCut 退回 direct 的计数。

只贴数字，不解读。
