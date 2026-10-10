# RUN-0044 — ITSC HPC A100 开训前准备的产物

- 运行：ITSC HPC 的 hpcgpu06（Slurm 任务 157296）与 hpcgpu07（157297），各 1 张 A100-PCIE-40GB；由用户 Mac 上的服务器 CC 经 `ssh a100 → gpu(hpcmgt)` 操作，节点上的进程一律在任务的分配里（`srun --overlap`）。
- 本目录由 hpcmgt 上的仓库 `$VI/repo`（`VI=/home/mail/chenyuma/visint`）推送；这个 README 是 deploy key 的第一次试推（RUN-0044 §3）。
- 报告：`reports/0044-hpc-bootstrap.md`。原始输出在 `raw/`，探针与驱动脚本在 `probe_scripts/`。
