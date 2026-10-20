#!/usr/bin/env bash
# RUN-0056（ITSC HPC hpcgpu06）的训练部分：9B 的方向性检查（E3 预注册 §3：每个变体 1 个种子），臂同 0.8B、4B 的 HPC 运行。
#   阶段 1：c3_s0（--supervision text）→ e3b_c3_s0（--supervision grid+text）：给语言干扰 L 与 T 文本（§12、§13）；
#   阶段 2：e3b_s0（--supervision grid）：给像素干扰 P（§13）。
# 配置是 E3 的正式配置 configs/e3b_train.json（微批 4 × 累积 4），RUN-0055 实测 9B 在 A100-40GB 上放得下（E3b+C3 峰值 36.35 GB）。
# 驱动规则见 scripts/server/train_queue.sh；HPC 模式：TQ_NO_GIT=1，本节点自己的暂停文件；不查份额（D95），只看 Lustre 的剩余空间。
# gpu_seconds 在两个阶段之间累计（同一个驱动目录），所以阶段 2 的 QUEUE_CAP_H 要写累计值（任务书 §3）。
# 铁律：任何情况下都不能取消或停止 Slurm 任务 157296、157297。
#
# 用法：由 run.sh 在节点上经 srun 调用：bash queue.sh 1|2
# 彩排（开发机）：TINY=1 WT=… REPO=… OUT=… CK_TMP=… CK_FINAL=… LOGS=… PY=… bash queue.sh 1|2
PHASE=${1:?1 或 2}
NODE=hpcgpu06
M=qwen3.5-9b
RUN_ID=0056-$NODE
if [ "${TINY:-0}" = 1 ]; then
  : "${WT:?}" "${REPO:?}" "${OUT:?}" "${CK_TMP:?}" "${CK_FINAL:?}" "${LOGS:?}" "${PY:?}"
  M=qwen3.5-0.8b
  CHUNK=${CHUNK:-3}; EXTRA=(--tiny); DATA=()
  export TQ_NO_GIT=1
else
  VI=/home/mail/chenyuma/visint
  WT=$VI/wt/wt0056
  REPO=$VI/repo
  LOGS=$VI/logs
  TRAIN=$VI/data/train_data
  OUT=$REPO/experiments/E3/0056_hpc_9b_train_$NODE
  CK_TMP=$VI/ckpt/$NODE/tmp0056
  CK_FINAL=$VI/ckpt/$NODE/0056
  PY=$VI/env/bin/python
  CHUNK=500
  export HF_HOME=$VI/hf HF_HUB_CACHE=$VI/hf/hub HF_HUB_OFFLINE=1
  export TMPDIR=$VI/tmp PIP_NO_CACHE_DIR=1
  export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 PYTHONHASHSEED=0
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export TRITON_CACHE_DIR=$VI/cache/$NODE/triton TORCHINDUCTOR_CACHE_DIR=$VI/cache/$NODE/inductor XDG_CACHE_HOME=$VI/cache/$NODE/xdg
  export TQ_NO_GIT=1
  export TQ_PAUSE=$LOGS/train.pause.$NODE   # 按节点分开（D91）；不碰 $LOGS/train.pause
  export TQ_DF_PATH=$VI                     # D95：不查份额，只在 Lustre 剩余空间低于 3 GB 时停
  MEM_STOP_GB=100
  EXTRA=(); DATA=(--data-root "$TRAIN"); TQ_TRAIN_DATA=""
  QUEUE_CAP_H=${QUEUE_CAP_H:?任务书 §3 的上限（累计）}
fi
export MEM_STOP_GB 2>/dev/null
# shellcheck disable=SC1091
source "$WT/scripts/server/train_queue.sh"
cd "$WT" || die "工作树不存在：$WT"
[ "$TINY" = 1 ] || [ -d "$TRAIN" ] || die "训练数据不在：$TRAIN"
[ "$TINY" = 1 ] || [ "${CUDA_VISIBLE_DEVICES:-}" = 0 ] || die "不在 srun 的资源范围里（CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-未设}）"
note "队列开始：$NODE，$M，阶段 $PHASE，种子 0，commit $(git -C "$WT" rev-parse HEAD 2>/dev/null)，每段 $CHUNK 步"

case "$PHASE" in
  1)
    EST_H=${EST_C3_H:-0.01}
    run "c3_s0" scripts/train_e3b.py --model $M --supervision text --seed 0 ${DATA[@]+"${DATA[@]}"}
    EST_H=${EST_MIX_H:-0.01}
    run "e3b_c3_s0" scripts/train_e3b.py --model $M --supervision grid+text --seed 0 ${DATA[@]+"${DATA[@]}"}
    ;;
  2)
    EST_H=${EST_E3B_H:-0.01}
    run "e3b_s0" scripts/train_e3b.py --model $M --supervision grid --seed 0 ${DATA[@]+"${DATA[@]}"}
    ;;
  *) die "未知阶段 $PHASE" ;;
esac
finish
