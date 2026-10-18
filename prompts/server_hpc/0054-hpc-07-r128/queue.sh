#!/usr/bin/env bash
# RUN-0054 的训练队列（ITSC HPC hpcgpu07；D99-1、E3 预注册 §13a 的修订）：0.8B 上把 LoRA 的秩提到 128（alpha 256），
# C3 与 E3b+C3 各 3 个种子；每个种子先 C3、再 E3b+C3，同 RUN-0045。配置 configs/e3b_train_r128.json（sha256 钉在 §13a，开跑前核对）。
# 驱动规则见 scripts/server/train_queue.sh；HPC 模式：TQ_NO_GIT=1，本节点自己的暂停文件；不查份额（D95），只看 Lustre 的剩余空间。
# 铁律：任何情况下都不能取消或停止 Slurm 任务 157296、157297。
#
# 用法（hpcgpu07 上经 run.sh 启动，见任务书）；彩排（开发机）：TINY=1 WT=… REPO=… OUT=… CK_TMP=… CK_FINAL=… LOGS=… PY=… bash queue.sh
NODE=hpcgpu07
M=qwen3.5-0.8b
RUN_ID=0054-$NODE
CFG_SHA=b57409372188d236f68e657ccf026ebf6316119fcf27cc361d98c647ae928f02
if [ "${TINY:-0}" = 1 ]; then
  : "${WT:?}" "${REPO:?}" "${OUT:?}" "${CK_TMP:?}" "${CK_FINAL:?}" "${LOGS:?}" "${PY:?}"
  CHUNK=${CHUNK:-3}; EXTRA=(--tiny); DATA=()
  SEEDS=(${SEEDS:-0})
  export TQ_NO_GIT=1
else
  VI=/home/mail/chenyuma/visint
  WT=$VI/wt/wt0054
  REPO=$VI/repo
  LOGS=$VI/logs
  TRAIN=$VI/data/train_data
  OUT=$REPO/experiments/E3/0054_r128_train_$NODE
  CK_TMP=$VI/ckpt/$NODE/tmp0054
  CK_FINAL=$VI/ckpt/$NODE/0054
  PY=$VI/env/bin/python
  CHUNK=500
  export HF_HOME=$VI/hf HF_HUB_CACHE=$VI/hf/hub HF_HUB_OFFLINE=1
  export TMPDIR=$VI/tmp PIP_NO_CACHE_DIR=1
  export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 PYTHONHASHSEED=0
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export TRITON_CACHE_DIR=$VI/cache/$NODE/triton TORCHINDUCTOR_CACHE_DIR=$VI/cache/$NODE/inductor XDG_CACHE_HOME=$VI/cache/$NODE/xdg
  export TQ_NO_GIT=1
  export TQ_PAUSE=$LOGS/train.pause.$NODE
  export TQ_DF_PATH=$VI
  MEM_STOP_GB=100
  EXTRA=(); DATA=(--data-root "$TRAIN"); TQ_TRAIN_DATA=""
  SEEDS=(0 1 2)
  QUEUE_CAP_H=${QUEUE_CAP_H:?任务书的上限}
fi
export MEM_STOP_GB 2>/dev/null
# shellcheck disable=SC1091
source "$WT/scripts/server/train_queue.sh"
cd "$WT" || die "工作树不存在：$WT"
CFG=$WT/configs/e3b_train_r128.json   # --config 要绝对路径
[ "$(sha "$CFG" | cut -d' ' -f1)" = "$CFG_SHA" ] || die "r128 配置的 sha256 与 §13a 钉住的不同：$CFG"
[ "$TINY" = 1 ] || [ -d "$TRAIN" ] || die "训练数据不在：$TRAIN"
[ "$TINY" = 1 ] || [ "${CUDA_VISIBLE_DEVICES:-}" = 0 ] || die "不在 srun 的资源范围里（CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-未设}）"
note "队列开始：$NODE，$M，秩 128（$CFG），commit $(git -C "$WT" rev-parse HEAD 2>/dev/null)，种子 ${SEEDS[*]}，每段 $CHUNK 步"

for s in "${SEEDS[@]}"; do
  EST_H=${EST_C3_H:-0.01}
  run "c3_r128_s$s"     scripts/train_e3b.py --model $M --supervision text --seed "$s" --config "$CFG" ${DATA[@]+"${DATA[@]}"}
  EST_H=${EST_MIX_H:-0.01}
  run "e3b_c3_r128_s$s" scripts/train_e3b.py --model $M --supervision grid+text --seed "$s" --config "$CFG" ${DATA[@]+"${DATA[@]}"}
  note "$M 秩 128 种子 $s：C3 与 E3b+C3 都过了一遍"
done
finish
