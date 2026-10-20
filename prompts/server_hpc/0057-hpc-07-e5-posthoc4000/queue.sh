#!/usr/bin/env bash
# RUN-0057（ITSC HPC hpcgpu07）：E5 的事后探索，4K 档延长到 4000 步（D103 → N40 (a)；E5 预注册 §8e；标为事后，不改 H5 的结论）。
# E3b+C3 与 C3+预热 × 种子 0–2，共 6 个运行；与原 4K 运行（RUN-0051，hpcgpu07）同机。
# 配置 configs/e5_train_posthoc4000.json：由 configs/e5_train.json 派生，只把 training.min_steps.steps 从 1000 改为 4000，其余逐字段相同。
# 每个运行训完，紧接着在 test 上评一次（选定的检查点只评这一次），评测同一份配置（eval 节与原配置相同）。
# 驱动规则见 scripts/server/train_queue.sh；HPC 模式：TQ_NO_GIT=1，本节点自己的暂停文件；D95：只看 Lustre 的剩余空间。
# 铁律：任何情况下都不能取消或停止 Slurm 任务 157296、157297。
#
# 用法（在节点上经 srun 启动，见任务书）：srun --jobid=157297 --overlap -n1 env PATH=… QUEUE_CAP_H=… bash …/queue.sh < /dev/null
# 彩排（开发机）：TINY=1 WT=… REPO=… OUT=… CK_TMP=… CK_FINAL=… LOGS=… PY=… ROOT40=<有 e3b_c3、c3_warmup> bash queue.sh
NODE=hpcgpu07
RUN_ID=0057-$NODE
M=qwen3.5-2b
if [ "${TINY:-0}" = 1 ]; then
  : "${WT:?}" "${REPO:?}" "${OUT:?}" "${CK_TMP:?}" "${CK_FINAL:?}" "${LOGS:?}" "${PY:?}" "${ROOT40:?}"
  CHUNK=${CHUNK:-3}; EXTRA=(--tiny); DATA=()
  M=qwen3.5-0.8b
  TIER=10; SEEDS=(${SEEDS:-0}); EVAL_LIMIT=(--limit 2)
  export TQ_NO_GIT=1
else
  VI=/home/mail/chenyuma/visint
  WT=$VI/wt/wt0057
  REPO=$VI/repo
  LOGS=$VI/logs
  OUT=$REPO/experiments/E5/0057_e5_posthoc4000_$NODE
  CK_TMP=$VI/ckpt/$NODE/tmp0057
  CK_FINAL=$VI/ckpt/$NODE/0057
  ROOT40=$VI/b200/ckpt/0040
  PY=$VI/env/bin/python
  CHUNK=1000
  export HF_HOME=$VI/hf HF_HUB_CACHE=$VI/hf/hub HF_HUB_OFFLINE=1
  export TMPDIR=$VI/tmp PIP_NO_CACHE_DIR=1
  export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 PYTHONHASHSEED=0
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export TRITON_CACHE_DIR=$VI/cache/$NODE/triton TORCHINDUCTOR_CACHE_DIR=$VI/cache/$NODE/inductor XDG_CACHE_HOME=$VI/cache/$NODE/xdg
  export TQ_NO_GIT=1
  export TQ_PAUSE=$LOGS/train.pause.$NODE
  export TQ_DF_PATH=$VI
  MEM_STOP_GB=100
  EXTRA=(); DATA=(--data-root "$VI/b200/data")  # 起点检查点的合并核对用 E1 的 grid16 开发子集
  TIER=4000; SEEDS=(0 1 2); EVAL_LIMIT=()
  QUEUE_CAP_H=${QUEUE_CAP_H:?任务书的上限}
fi
CFG=$WT/configs/e5_train_posthoc4000.json
export MEM_STOP_GB 2>/dev/null
# shellcheck disable=SC1091
source "$WT/scripts/server/train_queue.sh"
cd "$WT" || die "工作树不存在：$WT"
[ "$TINY" = 1 ] || [ "${CUDA_VISIBLE_DEVICES:-}" = 0 ] || die "不在 srun 的资源范围里（CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-未设}）"
[ -f "$CFG" ] || die "配置不在：$CFG"
note "队列开始：$NODE，$M，4K 延长到 4000 步（§8e），配置 $(sha256sum "$CFG" 2>/dev/null | cut -c1-12 || shasum -a 256 "$CFG" | cut -c1-12)，commit $(git -C "$WT" rev-parse HEAD 2>/dev/null)，每段 $CHUNK 步"

start_ckpt() {  # $1 = 起点，$2 = 种子 → 起点的最终检查点（与 RUN-0051 相同）
  case "$1" in
    e3b_c3)    echo "$ROOT40/e3b_c3/$M/seed$2/ckpt/final/model" ;;
    c3_warmup) echo "$ROOT40/c3_warmup/$M/seed$2/ckpt/final/model" ;;
  esac
}
e5() {  # 训一个运行，跑完紧接着在 test 上评一次。$1 = 起点，$2 = 种子
  local init=$1 s=$2 tag ck
  tag="${init}_n${TIER}_s${s}"
  ck=$(start_ckpt "$init" "$s")
  [ -d "$ck" ] || { note "$tag：起点检查点不在（$ck），跳过"; echo "$(now) $tag missing-start $ck" >> "$DRV/failed.txt"; return 1; }
  EST_H=${EST_TRAIN_H:-0.01}
  run "$tag" scripts/train_e5.py --init "$init" --start-checkpoint "$ck" --model $M --n-train "$TIER" --seed "$s" \
    --config "$CFG" ${DATA[@]+"${DATA[@]}"} || return 1
  EST_H=0.5
  run_cmd "eval_$tag" "$PY" scripts/eval_e5_trained.py --checkpoint "$(ckpt_of "$tag")" --model $M --out "$OUT" --task all \
    --config "$CFG" ${EXTRA[@]+"${EXTRA[@]}"} ${EVAL_LIMIT[@]+"${EVAL_LIMIT[@]}"}
}

for s in "${SEEDS[@]}"; do   # 同一个种子的两个起点相邻，最早拿到一对完整的比较
  e5 e3b_c3 "$s"
  e5 c3_warmup "$s"
done
finish
