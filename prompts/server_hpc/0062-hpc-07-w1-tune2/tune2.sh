#!/usr/bin/env bash
# RUN-0062（ITSC HPC hpcgpu07）：W1 开发-调参第 2 轮（W1 预注册 §10b"第 2 轮"；D110；W1 预算内）。只用开发-调参集（devtune），不碰开发-检查集。
#   1. 第 2 轮的臂（configs/w1_tune/U<k>.json，ARMS 给出，例如 "U1 U2 U3"）：9B 完整工作台，指代分割，devtune 前 N 句。
#   2. 每个臂对 RUN-0061 的两条基线（同样的 96 句，<REPO>/experiments/W1/0061_w1_tune_hpcgpu07/0061_base）各算一次 w1_gate.py
#      （调参集上的读数，不是判定点）。基线与回路协议无关（D108-3），不重跑。
# 全程每秒采样一次显存，写 <OUT>/gpu_mem.csv。GPU 由 claim_gpu 从 RUN-0057 的段边界接过来（本节点自己的暂停文件），退出时交还。
# 铁律：任何情况下都不能取消或停止 Slurm 任务 157296、157297。只用本节点自己的暂停文件。
#
# 用法（hpcgpu07 上经 srun 启动，见任务书）：srun --jobid=157297 --overlap -n1 env PATH=… EV_CAP_H=… ARMS="U1 U2 U3" bash …/tune2.sh < /dev/null
# 彩排（开发机）：TINY=1 WT=… REPO=… LOGS=… PY=… ARMS="U1 U2 U3" BASE_DIR=<tiny 的 0061_base> bash tune2.sh
RUN_ID=0062
NODE=hpcgpu07
M=qwen3.5-9b
read -r -a ARMS <<< "${ARMS:?第 2 轮的臂，例如 U1 U2 U3}"
if [ "${TINY:-0}" = 1 ]; then
  : "${WT:?}" "${REPO:?}" "${LOGS:?}" "${PY:?}" "${BASE_DIR:?}"
  M=qwen3.5-0.8b
  T=(--tiny); D=(); N=4
  export EV_NO_GIT=1
else
  VI=/home/mail/chenyuma/visint
  WT=$VI/wt/wt0062
  REPO=$VI/repo
  LOGS=$VI/logs
  PY=$VI/env/bin/python
  export HF_HOME=$VI/hf HF_HUB_CACHE=$VI/hf/hub HF_HUB_OFFLINE=1
  export TMPDIR=$VI/tmp PIP_NO_CACHE_DIR=1 NLTK_DATA=$VI/b200/nltk_data
  export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 PYTHONHASHSEED=0
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export TRITON_CACHE_DIR=$VI/cache/$NODE/triton TORCHINDUCTOR_CACHE_DIR=$VI/cache/$NODE/inductor XDG_CACHE_HOME=$VI/cache/$NODE/xdg
  T=(); D=(--data-root "$VI/b200/data"); N=96
  BASE_DIR=$REPO/experiments/W1/0061_w1_tune_$NODE/0061_base
  export EV_PAUSE=$LOGS/train.pause.$NODE EV_STATUS_GLOB="train_0057-$NODE.status" EV_NO_GIT=1
  export EV_DF_PATH=$VI
  EV_CAP_S=$(awk -v h="${EV_CAP_H:?任务书的上限（小时）}" 'BEGIN{print int(h * 3600)}')
fi
OUT=$REPO/experiments/W1/0062_w1_tune2_$NODE
# shellcheck disable=SC1091
source "$WT/scripts/server/eval_lib.sh"
cd "$WT" || exit 1
[ "$TINY" = 1 ] || [ "${CUDA_VISIBLE_DEVICES:-}" = 0 ] || { progress "blocked: 不在 srun 的资源范围里"; exit 1; }
G=$BASE_DIR/$M/refseg
for f in "$G/baseline_ownbox/devtune/predictions.jsonl" "$G/baseline_poly_structured/devtune/predictions.jsonl"; do
  [ -s "$f" ] || { progress "blocked: 缺 RUN-0061 的基线 $f"; exit 1; }
done
for k in "${ARMS[@]}"; do [ -f "configs/w1_tune/$k.json" ] || { progress "blocked: 缺 configs/w1_tune/$k.json"; exit 1; }; done
NOK=0

claim_gpu "0062-hpc-07-w1-tune2.md" || { progress "blocked: gpu not free after ${CLAIM_MAX_S}s"; exit 1; }
echo "$(now) start commit $(git -C "$WT" rev-parse HEAD 2>/dev/null || echo unknown) node $NODE model $M n $N arms ${ARMS[*]}" >> "$OUT/steps.log"
MEMPID=""
if [ "$TINY" != 1 ] && command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -lms 1000 > "$OUT/gpu_mem.csv" 2>/dev/null &
  MEMPID=$!
fi
# 1. 臂（--out 的目录名必须带运行编号：0062_U1 …）
for k in "${ARMS[@]}"; do
  step "arm_${k}_refseg_devtune$N" "$PY" scripts/run_w1.py --model $M ${T[@]+"${T[@]}"} ${D[@]+"${D[@]}"} \
    --w1-config "configs/w1_tune/$k.json" --out "$OUT/${RUN_ID}_$k" --task refseg --split devtune --setting full --limit $N || NOK=$((NOK + 1))
done
# 2. 每个臂对 RUN-0061 的两条基线
for k in "${ARMS[@]}"; do
  step "gate_tune_$k" "$PY" scripts/w1_gate.py --w1 "$OUT/${RUN_ID}_$k/$M/refseg/full/devtune" --ownbox "$G/baseline_ownbox/devtune" \
    --poly "$G/baseline_poly_structured/devtune" --out-json "$OUT/gate_tune_$k.json" --out-md "$OUT/gate_tune_$k.md" || NOK=$((NOK + 1))
done
[ -n "$MEMPID" ] && kill "$MEMPID" 2>/dev/null

echo "$(now) done; failed steps: $NOK" >> "$OUT/steps.log"
progress "done $RUN_ID (failed steps: $NOK)"
push_out "W1 tuning round 2 done (failed steps: $NOK)"
