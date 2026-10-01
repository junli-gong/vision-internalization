#!/usr/bin/env bash
# 集群作业示例（SLURM）。分区名、账户、GPU 写法请按你们集群改。
#SBATCH --job-name=visint-0014
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=visint-0014-%j.out

WORK=/path/to/visint        # 改成 setup_env.sh 用的同一个工作目录
PKG=/path/to/this/package   # 改成解压本包的目录
export HF_HUB_OFFLINE=1     # GPU 节点通常不能联网，权重已由 prefetch.sh 在登录节点下好
bash "$PKG/run_all.sh" "$WORK"
