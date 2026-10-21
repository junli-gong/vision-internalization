[RUN-TASK 0059] 在 HPC 环境里跑 W1 工作台、沙箱与多轮对话接口的测试（D104、D106）。只用 CPU；约 5 分钟

## 铁律

`~/a100-hpc-guide.md` 第 0 节：**任何情况下都不能取消或停止 Slurm 任务 157296 和 157297。**
- 前后各跑一次 `squeue`。
- 不碰 06 上的 RUN-0056、07 上的 RUN-0054。
- 不往 `$VI/repo` 里 pull。

## 为什么

- W1 的工作台与沙箱（DEV-0050）是在开发机上写的。开发机是 macOS，cv2 是 5.0。
- HPC 上要验证三件事：
  - Linux 的内存上限路径：`RLIMIT_AS` 设为 4 GB 时，numpy、cv2、skimage 能否正常启动；
  - OpenCV 4.10 下的彩色 Canny 与 GrabCut；
  - cv2 4.10 初始化时，属性下毒是否正常。
- 多轮对话接口（DEV-0049）的测试也一起跑。

## 做法

1. hpcmgt 上 `squeue`。
2. 建临时工作树（稀疏，同 RUN-0054 的规则；`--no-checkout` 之后要 `read-tree`）：

   ```bash
   VI=/home/mail/chenyuma/visint
   flock $VI/logs/git.lock -c "git -C $VI/repo fetch -q origin main"
   git -C $VI/repo worktree add --no-checkout --detach $VI/wt/wt0059 origin/main
   git -C $VI/wt/wt0059 sparse-checkout set --no-cone '/*' '!/experiments/*/' '/experiments/P0/'
   git -C $VI/wt/wt0059 read-tree -mu HEAD
   git -C $VI/wt/wt0059 rev-parse HEAD; git -C $VI/wt/wt0059 status --short
   ```

   origin/main 必须包含 a5f1fc28。
3. 在 06 上经 srun 跑测试（CPU，环境变量照抄）：

   ```bash
   ssh -n hpcgpu06 'srun --jobid=157296 --overlap -n1 env PATH=/home/mail/chenyuma/visint/env/bin:/usr/bin:/bin \
     HF_HOME=/home/mail/chenyuma/visint/hf HF_HUB_CACHE=/home/mail/chenyuma/visint/hf/hub HF_HUB_OFFLINE=1 TMPDIR=/home/mail/chenyuma/visint/tmp \
     bash -c "cd /home/mail/chenyuma/visint/wt/wt0059 && CUDA_VISIBLE_DEVICES= python -m pytest tests/test_w1_workbench.py tests/test_w1_sandbox.py tests/test_adapter_chat.py tests/test_constants.py -q -p no:cacheprovider --basetemp=/home/mail/chenyuma/visint/tmp/pt_0059"' < /dev/null
   ```

   - tiny 模型的 config 与 tokenizer 在 `$VI/hf/hub` 里有缓存，离线可用。如果因为缺缓存而失败，照实报告，不要联网下载。
4. 跑完删掉 `$VI/tmp/pt_0059` 与工作树 `wt0059`（`git -C $VI/repo worktree remove $VI/wt/wt0059`）。
5. 再跑一次 `squeue`。

## 报告

推送 `reports/0059-hpc-w1-tests.md`：
- commit；
- pytest 输出的最后 30 行原文；
- 失败的贴完整的报错。
