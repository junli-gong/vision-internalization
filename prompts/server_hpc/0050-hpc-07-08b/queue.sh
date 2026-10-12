#!/usr/bin/env bash
# RUN-0050（ITSC HPC hpcgpu07）的训练部分：0.8B 的 E3b 单独训练 × 3 个种子（--supervision grid），给像素干扰（E3 预注册 §13；D93）。
# 这一臂在 E3 v1 的消融里，原定往后排（D83-2），D93 提前。配置与 2B 的 E3b（RUN-0036）相同，只换尺寸。
# 驱动规则见 scripts/server/train_queue.sh；HPC 模式：TQ_NO_GIT=1，本节点自己的暂停文件，份额用 lfs quota 的差值。
# 铁律：任何情况下都不能取消或停止 Slurm 任务 157296、157297。
#
# 用法：由 run.sh 在节点上经 srun 调用。
# 彩排（开发机）：TINY=1 WT=… REPO=… OUT=… CK_TMP=… CK_FINAL=… LOGS=… PY=… bash queue.sh
NODE=hpcgpu07
M=qwen3.5-0.8b
RUN_ID=0050-$NODE
if [ "${TINY:-0}" = 1 ]; then
  : "${WT:?}" "${REPO:?}" "${OUT:?}" "${CK_TMP:?}" "${CK_FINAL:?}" "${LOGS:?}" "${PY:?}"
  CHUNK=${CHUNK:-3}; EXTRA=(--tiny); DATA=()
  SEEDS=(${SEEDS:-0})
  export TQ_NO_GIT=1
else
  VI=/home/mail/chenyuma/visint
  WT=$VI/wt/wt0050
  REPO=$VI/repo
  LOGS=$VI/logs
  TRAIN=$VI/data/train_data
  OUT=$REPO/experiments/E3/0050_hpc_e3b_$NODE
  CK_TMP=$VI/ckpt/$NODE/tmp0050
  CK_FINAL=$VI/ckpt/$NODE/0050
  PY=$VI/env/bin/python
  CHUNK=500
  export HF_HOME=$VI/hf HF_HUB_CACHE=$VI/hf/hub HF_HUB_OFFLINE=1
  export TMPDIR=$VI/tmp PIP_NO_CACHE_DIR=1
  export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 PYTHONHASHSEED=0
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export TRITON_CACHE_DIR=$VI/cache/$NODE/triton TORCHINDUCTOR_CACHE_DIR=$VI/cache/$NODE/inductor XDG_CACHE_HOME=$VI/cache/$NODE/xdg
  export TQ_NO_GIT=1
  export TQ_PAUSE=$LOGS/train.pause.$NODE   # 按节点分开（D91、0045-addendum-1）；不碰 06 驱动看的 $LOGS/train.pause
  export TQ_QUOTA_CMD="timeout 60 lfs quota -u chenyuma /home | awk 'NR==3{u=\$2; gsub(/\\*/,\"\",u); v=int((u-7059403936)/1048576)} END{print v+0}'"
  SHARE_STOP_GB=80
  MEM_STOP_GB=100
  EXTRA=(); DATA=(--data-root "$TRAIN"); TQ_TRAIN_DATA=""
  SEEDS=(0 1 2)
  QUEUE_CAP_H=${QUEUE_CAP_H:?任务书 §3 的上限}
fi
export SHARE_STOP_GB MEM_STOP_GB 2>/dev/null
# shellcheck disable=SC1091
source "$WT/scripts/server/train_queue.sh"
cd "$WT" || die "工作树不存在：$WT"
[ "$TINY" = 1 ] || [ -d "$TRAIN" ] || die "训练数据不在：$TRAIN"
[ "$TINY" = 1 ] || [ "${CUDA_VISIBLE_DEVICES:-}" = 0 ] || die "不在 srun 的资源范围里（CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-未设}）"
note "队列开始：$NODE，$M，E3b × 种子 ${SEEDS[*]}，commit $(git -C "$WT" rev-parse HEAD 2>/dev/null)，每段 $CHUNK 步"

for s in "${SEEDS[@]}"; do
  EST_H=${EST_E3B_H:-0.01}
  run "e3b_s$s" scripts/train_e3b.py --model $M --supervision grid --seed "$s" ${DATA[@]+"${DATA[@]}"}
done
finish
