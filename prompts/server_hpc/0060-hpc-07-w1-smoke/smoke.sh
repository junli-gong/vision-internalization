#!/usr/bin/env bash
# RUN-0060（ITSC HPC hpcgpu07）：W1 的冒烟测试与单位用时实测（D103、D107；W1 预算内）。只用开发-调参集（devtune），不碰开发-检查集。
#   1. 9B 完整工作台（W1-1，setting=full），refseg devtune 前 32 句：回路、工具、记账在真模型上跑通；每句用时、轮数、工具调用、失败码。
#   2. 四种基线在同样 32 句上：W1-0（refexp_box，"自己的框"要用）、自己的框、E1 结构强制多边形、W1-F。
#   3. 完整工作台的定位（refexp_box devtune 前 16 句）与计数（devtune_countbench 前 16 条）：另两种任务也跑通。
#   4. w1_gate.py 在 devtune 的 32 句上演练一次（不是判定点，结论标"彩排，不作数"）。
# 全程每秒采样一次显存（nvidia-smi），写 <OUT>/gpu_mem.csv。结果只作计时、显存与排错用。
# 铁律：任何情况下都不能取消或停止 Slurm 任务 157296、157297。只用本节点自己的暂停文件。
#
# 用法（hpcgpu07 上经 srun 启动，见任务书）：srun --jobid=157297 --overlap -n1 env PATH=… EV_CAP_H=3 bash …/smoke.sh < /dev/null
# 彩排（开发机）：TINY=1 WT=… REPO=… LOGS=… PY=… bash smoke.sh
RUN_ID=0060
NODE=hpcgpu07
M=qwen3.5-9b
if [ "${TINY:-0}" = 1 ]; then
  : "${WT:?}" "${REPO:?}" "${LOGS:?}" "${PY:?}"
  M=qwen3.5-0.8b
  T=(--tiny); D=(); L_SEG=4; L_BOX=2; L_CNT=2
  SPLIT_SEG=devtune; SPLIT_CNT=devtune_countbench
  export EV_NO_GIT=1
else
  VI=/home/mail/chenyuma/visint
  WT=$VI/wt/wt0060
  REPO=$VI/repo
  LOGS=$VI/logs
  PY=$VI/env/bin/python
  export HF_HOME=$VI/hf HF_HUB_CACHE=$VI/hf/hub HF_HUB_OFFLINE=1
  export TMPDIR=$VI/tmp PIP_NO_CACHE_DIR=1 NLTK_DATA=$VI/b200/nltk_data
  export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 PYTHONHASHSEED=0
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export TRITON_CACHE_DIR=$VI/cache/$NODE/triton TORCHINDUCTOR_CACHE_DIR=$VI/cache/$NODE/inductor XDG_CACHE_HOME=$VI/cache/$NODE/xdg
  T=(); D=(--data-root "$VI/b200/data"); L_SEG=32; L_BOX=16; L_CNT=16
  SPLIT_SEG=devtune; SPLIT_CNT=devtune_countbench
  export EV_PAUSE=$LOGS/train.pause.$NODE EV_STATUS_GLOB="train_0057-$NODE.status" EV_NO_GIT=1
  export EV_DF_PATH=$VI
  EV_CAP_S=$(awk -v h="${EV_CAP_H:?任务书的上限（小时）}" 'BEGIN{print int(h * 3600)}')
fi
OUT=$REPO/experiments/W1/0060_w1_smoke_$NODE
# shellcheck disable=SC1091
source "$WT/scripts/server/eval_lib.sh"
cd "$WT" || exit 1
[ "$TINY" = 1 ] || [ "${CUDA_VISIBLE_DEVICES:-}" = 0 ] || { progress "blocked: 不在 srun 的资源范围里"; exit 1; }
NOK=0
w1() {  # $1 = 步名，其余 = run_w1.py 的参数
  local name=$1; shift
  step "$name" "$PY" scripts/run_w1.py --model $M ${T[@]+"${T[@]}"} ${D[@]+"${D[@]}"} --out "$OUT" "$@" || NOK=$((NOK + 1))
}

claim_gpu "0060-hpc-07-w1-smoke.md" || { progress "blocked: gpu not free after ${CLAIM_MAX_S}s"; exit 1; }
echo "$(now) start commit $(git -C "$WT" rev-parse HEAD 2>/dev/null || echo unknown) node $NODE model $M" >> "$OUT/steps.log"
MEMPID=""
if [ "$TINY" != 1 ] && command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -lms 1000 > "$OUT/gpu_mem.csv" 2>/dev/null &
  MEMPID=$!
fi

# 1. 9B 完整工作台，refseg
w1 "full_refseg_${SPLIT_SEG}${L_SEG}" --task refseg --split $SPLIT_SEG --setting full --limit $L_SEG
# 2. 基线（同样的句子）
w1 "baseline_e1_refexp_box_${SPLIT_SEG}${L_SEG}" --task refexp_box --split $SPLIT_SEG --setting baseline_e1 --limit $L_SEG
w1 "baseline_ownbox_${SPLIT_SEG}${L_SEG}" --task refseg --split $SPLIT_SEG --setting baseline_ownbox --limit $L_SEG
w1 "baseline_poly_structured_${SPLIT_SEG}${L_SEG}" --task refseg --split $SPLIT_SEG --setting baseline_poly_structured --limit $L_SEG
w1 "w1f_${SPLIT_SEG}${L_SEG}" --task refseg --split $SPLIT_SEG --setting w1f --limit $L_SEG
# 3. 另两种任务
w1 "full_refexp_box_${SPLIT_SEG}${L_BOX}" --task refexp_box --split $SPLIT_SEG --setting full --limit $L_BOX
w1 "full_count_${SPLIT_CNT}${L_CNT}" --task count --split $SPLIT_CNT --setting full --limit $L_CNT
# 4. 去留计算的演练（devtune 不是判定点）
G=$OUT/$M/refseg
step "gate_rehearsal" "$PY" scripts/w1_gate.py --w1 "$G/full/$SPLIT_SEG" --ownbox "$G/baseline_ownbox/$SPLIT_SEG" \
  --poly "$G/baseline_poly_structured/$SPLIT_SEG" --out-json "$OUT/gate_rehearsal.json" --out-md "$OUT/gate_rehearsal.md" || NOK=$((NOK + 1))
[ -n "$MEMPID" ] && kill "$MEMPID" 2>/dev/null

echo "$(now) done; failed steps: $NOK" >> "$OUT/steps.log"
progress "done $RUN_ID (failed steps: $NOK)"
push_out "W1 smoke done (failed steps: $NOK)"
