#!/usr/bin/env bash
# RUN-0061（ITSC HPC hpcgpu07）：W1 开发-调参（W1 预注册 §10b；D103 §二.4；W1 预算内）。只用开发-调参集（devtune），不碰开发-检查集。
#   1. 基线（configs/w1.json）：W1-0 定位、自己的框、E1 结构强制多边形、W1-F，devtune 前 N 句。
#   2. 四个臂 T1–T4（configs/w1_tune/T<k>.json）：9B 完整工作台，指代分割，devtune 前 N 句。
#   3. 每个臂对两条基线各算一次 w1_gate.py（调参集上的读数，不是判定点，结论标"调参，不作数"）。
# 全程每秒采样一次显存，写 <OUT>/gpu_mem.csv。GPU 由 claim_gpu 从 RUN-0057 的段边界接过来（本节点自己的暂停文件），退出时交还。
# 铁律：任何情况下都不能取消或停止 Slurm 任务 157296、157297。只用本节点自己的暂停文件。
#
# 用法（hpcgpu07 上经 srun 启动，见任务书）：srun --jobid=157297 --overlap -n1 env PATH=… EV_CAP_H=6 bash …/tune.sh < /dev/null
# 彩排（开发机）：TINY=1 WT=… REPO=… LOGS=… PY=… bash tune.sh
RUN_ID=0061
NODE=hpcgpu07
M=qwen3.5-9b
ARMS=(T1 T2 T3 T4)
if [ "${TINY:-0}" = 1 ]; then
  : "${WT:?}" "${REPO:?}" "${LOGS:?}" "${PY:?}"
  M=qwen3.5-0.8b
  T=(--tiny); D=(); N=4
  export EV_NO_GIT=1
else
  VI=/home/mail/chenyuma/visint
  WT=$VI/wt/wt0061
  REPO=$VI/repo
  LOGS=$VI/logs
  PY=$VI/env/bin/python
  export HF_HOME=$VI/hf HF_HUB_CACHE=$VI/hf/hub HF_HUB_OFFLINE=1
  export TMPDIR=$VI/tmp PIP_NO_CACHE_DIR=1 NLTK_DATA=$VI/b200/nltk_data
  export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 PYTHONHASHSEED=0
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export TRITON_CACHE_DIR=$VI/cache/$NODE/triton TORCHINDUCTOR_CACHE_DIR=$VI/cache/$NODE/inductor XDG_CACHE_HOME=$VI/cache/$NODE/xdg
  T=(); D=(--data-root "$VI/b200/data"); N=96
  export EV_PAUSE=$LOGS/train.pause.$NODE EV_STATUS_GLOB="train_0057-$NODE.status" EV_NO_GIT=1
  export EV_DF_PATH=$VI
  EV_CAP_S=$(awk -v h="${EV_CAP_H:?任务书的上限（小时）}" 'BEGIN{print int(h * 3600)}')
fi
OUT=$REPO/experiments/W1/0061_w1_tune_$NODE
# shellcheck disable=SC1091
source "$WT/scripts/server/eval_lib.sh"
cd "$WT" || exit 1
[ "$TINY" = 1 ] || [ "${CUDA_VISIBLE_DEVICES:-}" = 0 ] || { progress "blocked: 不在 srun 的资源范围里"; exit 1; }
NOK=0
w1() {  # $1 = 步名，$2 = 输出子目录，$3 = 配置，其余 = run_w1.py 的参数
  local name=$1 sub=$2 cfg=$3; shift 3
  step "$name" "$PY" scripts/run_w1.py --model $M ${T[@]+"${T[@]}"} ${D[@]+"${D[@]}"} --w1-config "$cfg" --out "$OUT/$sub" "$@" || NOK=$((NOK + 1))
}

claim_gpu "0061-hpc-07-w1-tune.md" || { progress "blocked: gpu not free after ${CLAIM_MAX_S}s"; exit 1; }
echo "$(now) start commit $(git -C "$WT" rev-parse HEAD 2>/dev/null || echo unknown) node $NODE model $M n $N" >> "$OUT/steps.log"
MEMPID=""
if [ "$TINY" != 1 ] && command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -lms 1000 > "$OUT/gpu_mem.csv" 2>/dev/null &
  MEMPID=$!
fi

# 1. 基线（协议无关，用 configs/w1.json）
B=configs/w1.json
w1 "base_e1_refexp_box_devtune$N" base $B --task refexp_box --split devtune --setting baseline_e1 --limit $N
w1 "base_ownbox_devtune$N" base $B --task refseg --split devtune --setting baseline_ownbox --limit $N
w1 "base_poly_structured_devtune$N" base $B --task refseg --split devtune --setting baseline_poly_structured --limit $N
w1 "base_w1f_devtune$N" base $B --task refseg --split devtune --setting w1f --limit $N
# 2. 四个臂
for k in "${ARMS[@]}"; do
  w1 "arm_${k}_refseg_devtune$N" "$k" "configs/w1_tune/$k.json" --task refseg --split devtune --setting full --limit $N
done
# 3. 每个臂对两条基线（调参集上的读数，不作数）
G=$OUT/base/$M/refseg
for k in "${ARMS[@]}"; do
  step "gate_tune_$k" "$PY" scripts/w1_gate.py --w1 "$OUT/$k/$M/refseg/full/devtune" --ownbox "$G/baseline_ownbox/devtune" \
    --poly "$G/baseline_poly_structured/devtune" --out-json "$OUT/gate_tune_$k.json" --out-md "$OUT/gate_tune_$k.md" || NOK=$((NOK + 1))
done
[ -n "$MEMPID" ] && kill "$MEMPID" 2>/dev/null

echo "$(now) done; failed steps: $NOK" >> "$OUT/steps.log"
progress "done $RUN_ID (failed steps: $NOK)"
push_out "W1 tuning arms done (failed steps: $NOK)"
