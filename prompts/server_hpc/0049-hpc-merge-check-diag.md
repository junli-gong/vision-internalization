[RUN-TASK 0049] 在 HPC 上复核合并核对：DEV-0045 的诊断加两组测试（D92），只用 CPU，约 5 分钟

## 铁律

`~/a100-hpc-guide.md` 第 0 节：**任何情况下都不能取消或停止 Slurm 任务 157296 和 157297。** 前后各跑一次 `squeue`，两条都必须是 RUNNING。

## 为什么做

- RUN-0044 在 HPC 上有一条测试 4 次都失败：`test_grid_checkpoint_split_across_merge_fails_restricted`。DEV-0045 查清了原因（D92）：
  - 测试里的病态检查点只靠 fp32 舍入分家，HPC 的舍入恰好没让它分家；
  - 核对本身没有变钝，判据不改。
- 修过的测试换了一个在任何 fp32 实现上都大量分家的构造，另加了一条与舍入无关的对照。
- 后面 E5 和 0.8B、4B 的评测都在 A100 上做，先在 HPC 上确认一遍。

## 何时、在哪跑

- **hpcgpu07**：等 RUN-0045 的 07 队列跑完（`$VI/logs/train_0045-hpcgpu07.done`），**在 RUN-0048 之前**跑，那时 07 空闲。
- **hpcgpu06**：等 RUN-0045 的 06 队列跑完再跑一次。RUN-0044 的失败就出在这台。
- 不用 GPU，不写任何暂停文件。

## 做法

1. `flock $VI/logs/git.lock -c 'git -C $VI/repo fetch -q origin main'`。origin/main 必须包含 71e2ab3（DEV-0045 合入）与本任务书所在的 commit。
2. `git -C $VI/repo worktree add --detach $VI/wt/wt0049 origin/main`，确认 `status --short` 为空。
3. 在节点上经 srun 跑（以 07 为例；06 换成 `--jobid=157296`，输出文件名里的主机名会自己变）：

   ```bash
   ssh -n hpcgpu07 'srun --jobid=157297 --overlap -n1 bash -c '"'"'export VI=/home/mail/chenyuma/visint TMPDIR=$VI/tmp HF_HOME=$VI/hf HF_HUB_CACHE=$VI/hf/hub HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES= PATH=$VI/env/bin:/usr/bin:/bin; cd $VI/wt/wt0049 && python scripts/diag/merge_split_probe.py --out $VI/tmp/msp_0049_$(hostname -s).json --compare scripts/diag/merge_split_probe.dev.json; echo diag_rc=$?; python -m pytest -q -p no:cacheprovider --basetemp=$VI/tmp/pt_0049_$(hostname -s) -rfEs tests/test_check_training_interfaces.py "tests/test_e3_eval.py::test_bad_checkpoint_refused"; echo pytest_rc=$?'"'"' < /dev/null' \
     > $VI/logs/0049_$(date -u +%H%M)_07.log 2>&1
   ```

   引号嵌套照你在 RUN-0044 用过的写法调整即可，命令内容不变。
4. 把 `$VI/tmp/msp_0049_<主机>.json` 拷进 `experiments/E3/0049_hpc_merge_check_diag/`，与 pytest 输出一起推送。推送用开发机上的 `A100/repo`，或 v2 的推送工作树，不要在 `$VI/repo` 里提交。

## 预期（D92）

- `diag_rc=0`。
- `split_run0044` 的输入与 LoRA 指纹和开发机相同。
  - hpcgpu06 上应复现 RUN-0044 的读数：0 处不同，max_abs_logit 8.57；
  - hpcgpu07 的 CPU 或 BLAS 可能不同，读数照实记下。
- `split` 有大量位置不同；`control` 判不过（退出码 1，失败只在 across_merge）。
- pytest 全过，没有 skip。

## 报告

推送 `reports/0049-hpc-merge-check-diag.md`，只贴事实：
- 每台节点的 `diag_rc`；
- 诊断 JSON 里 `verdict` 与四个情形的 across_merge 读数（argmax 不同的位置数、max_abs_logit）；
- 与开发机输出对照的结果（`--compare` 打印的部分）；
- pytest 的汇总行。

和预期不符的，原样贴出来，不要解读。06 的那一次跑完后，追加到同一份报告里。
