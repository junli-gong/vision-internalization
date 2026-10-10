#!/usr/bin/env bash
# RUN-0045 的训练队列（ITSC HPC A100；D83、D85、D88、D89）：每台节点一个尺寸（同尺寸同机，D72-4）。
#   hpcgpu06（Slurm 157296）：4B；hpcgpu07（Slurm 157297）：0.8B。
#   每个种子先 C3、再 E3b+C3（H1 修订版的主臂），种子 0 → 1 → 2，最早拿到成对的比较。消融（E3b、E3a、C1、C2、C1′）另发任务。
# 驱动用 TQ_NO_GIT=1：计算节点不碰 git，结果由 hpcmgt 上的 scripts/server/sync_push.sh 推送；HOLD 看 $VI/logs/HOLD。
# 铁律：任何情况下都不能取消或停止 Slurm 任务 157296、157297（~/a100-hpc-guide.md 第 0 节）。本驱动只管自己的子进程。
#
# 用法（在节点上经 srun 启动，整个驱动与它的子进程都在任务的资源范围里；见任务书 §3）：
#   srun --jobid=<任务> --overlap -n1 env NODE=<hpcgpu06|hpcgpu07> bash $VI/wt/wt0045/prompts/server_hpc/0045-hpc-train/queue.sh < /dev/null
# 彩排（开发机）：TINY=1 NODE=hpcgpu07 WT=… REPO=… OUT=… CK_TMP=… CK_FINAL=… LOGS=… PY=… bash queue.sh
NODE=${NODE:?NODE=hpcgpu06 或 hpcgpu07}
case "$NODE" in
  hpcgpu06) M=qwen3.5-4b ;;
  hpcgpu07) M=qwen3.5-0.8b ;;
  *) echo "未知节点 $NODE" >&2; exit 2 ;;
esac
RUN_ID=0045-$NODE
if [ "${TINY:-0}" = 1 ]; then
  : "${WT:?}" "${REPO:?}" "${OUT:?}" "${CK_TMP:?}" "${CK_FINAL:?}" "${LOGS:?}" "${PY:?}"
  CHUNK=${CHUNK:-3}; EXTRA=(--tiny); DATA=()
  M=qwen3.5-0.8b
  SEEDS=(${SEEDS:-0})
  export TQ_NO_GIT=1
else
  VI=/home/mail/chenyuma/visint
  WT=$VI/wt/wt0045
  REPO=$VI/repo
  LOGS=$VI/logs
  TRAIN=${TRAIN:-$VI/data/train_data}          # 任务书 §2：RUN-0044 的 A/B 显示 Lustre 明显慢时，改给 /dev/shm 的副本
  OUT=$REPO/experiments/E3/0045_hpc_train_$NODE
  CK_TMP=$VI/ckpt/$NODE/tmp0045
  CK_FINAL=$VI/ckpt/$NODE/0045
  PY=$VI/env/bin/python
  CHUNK=500                                      # 每段 500 步 = ckpt_every；hpcmgt 或节点出事时最多丢一段
  export HF_HOME=$VI/hf HF_HUB_CACHE=$VI/hf/hub HF_HUB_OFFLINE=1
  export TMPDIR=$VI/tmp PIP_NO_CACHE_DIR=1
  export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 PYTHONHASHSEED=0
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export TRITON_CACHE_DIR=$VI/cache/$NODE/triton TORCHINDUCTOR_CACHE_DIR=$VI/cache/$NODE/inductor XDG_CACHE_HOME=$VI/cache/$NODE/xdg
  export TQ_NO_GIT=1
  # 份额（D89-7，上限 80 GB）：Lustre 上不能对大目录 du；数据、权重、环境的大小是固定的（约 34 GB，任务书 §2），只量检查点目录
  export TQ_QUOTA_CMD="echo \$(( 34 + \$(du -sk $VI/ckpt 2>/dev/null | awk '{print int(\$1/1048576)}') ))"
  SHARE_STOP_GB=80
  MEM_STOP_GB=100                                # 节点 MemAvailable 低于 100 GB 就停（RUN-0043 §3.7）
  EXTRA=(); DATA=(--data-root "$TRAIN"); TQ_TRAIN_DATA=""   # 训练数据在 Lustre 上，内存不足时不删
  SEEDS=(0 1 2)
  QUEUE_CAP_H=${QUEUE_CAP_H:?任务书 §4 给每台节点的上限}
fi
export SHARE_STOP_GB MEM_STOP_GB 2>/dev/null
# shellcheck disable=SC1091
source "$WT/scripts/server/train_queue.sh"
cd "$WT" || die "工作树不存在：$WT"
[ "$TINY" = 1 ] || [ -d "$TRAIN" ] || die "训练数据不在：$TRAIN"
[ "$TINY" = 1 ] || [ "${CUDA_VISIBLE_DEVICES:-}" = 0 ] || die "不在 srun 的资源范围里（CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-未设}）"
note "队列开始：$NODE，$M，commit $(git -C "$WT" rev-parse HEAD 2>/dev/null)，种子 ${SEEDS[*]}，每段 $CHUNK 步"

for s in "${SEEDS[@]}"; do
  EST_H=${EST_C3_H:-0.01}
  run "c3_s$s"     scripts/train_e3b.py --model $M --supervision text --seed "$s" ${DATA[@]+"${DATA[@]}"}
  EST_H=${EST_MIX_H:-0.01}
  run "e3b_c3_s$s" scripts/train_e3b.py --model $M --supervision grid+text --seed "$s" ${DATA[@]+"${DATA[@]}"}
  note "$M 种子 $s：C3 与 E3b+C3 都过了一遍"
done
finish
