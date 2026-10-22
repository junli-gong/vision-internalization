#!/usr/bin/env bash
# RUN-0066（ITSC HPC）：W2 在 devtune800 上调参（W2 预注册 §5 与"补记"；D113；W2 预算内）。只用 devtune800，不碰 devcheck2。
#   第 0 步：E1 的框（baseline_e1 refexp_box）、候选（DEFAULT_SPEC）、w1f、ownbox、random_pick、oracle_pick；按 oracle 的 best_index 给生成器排序。
#   第 1 步：布局 × 挑选时思考，4 个臂（grid_off < sep_off < grid_on < sep_on，越前越简单），K = 8。
#   第 2 步：第 1 步选中的臂，K ∈ {4, 6}（K = 8 即第 1 步的臂），简单顺序 K4 < K6 < K8。
#   第 3 步：第 2 步选中的臂，提出 box（即第 2 步的臂）< box_points。
#   每一步按"减 W1-F 的配对 cIoU 差最大、相差不到 0.005 取最简单"选（scripts/mgr/w2_tune.py select），全部用第 0 步那一份 w1f 配对。
#   结果：<OUT>/final.json 与 <OUT>/cfg/<选中的臂>.json；冻结由管理员做（不在本脚本里）。
# 各臂共用第 0 步的框（复制目录；运行器按身份核对）。候选缓存的身份含整份配置的 sha256，各臂的配置不同，所以候选每个臂各自重新生成（彩排实测：复制候选会被拒收）。
# 铁律：任何情况下都不能取消或停止 Slurm 任务 157296、157297。只用本节点自己的暂停文件。
#
# 用法（经 srun 启动，见任务书）：srun --jobid=<本节点的任务> --overlap -n1 -c 8 env PATH=… NODE=… TQ_STATUS=… EV_CAP_H=… bash …/tune_w2.sh
# 彩排（开发机）：TINY=1 WT=… REPO=… LOGS=… PY=… bash tune_w2.sh
RUN_ID=0066
M=qwen3.5-9b
SPLIT=devtune800
if [ "${TINY:-0}" = 1 ]; then
  : "${WT:?}" "${REPO:?}" "${LOGS:?}" "${PY:?}"
  NODE=${NODE:-tiny}
  M=qwen3.5-0.8b
  T=(--tiny); D=(); CW=1
  export EV_NO_GIT=1
else
  : "${NODE:?hpcgpu06 或 hpcgpu07}" "${TQ_STATUS:?本节点训练队列的心跳文件名（\$LOGS 下）}"
  VI=/home/mail/chenyuma/visint
  WT=$VI/wt/wt0066
  REPO=$VI/repo
  LOGS=$VI/logs
  PY=$VI/env/bin/python
  export HF_HOME=$VI/hf HF_HUB_CACHE=$VI/hf/hub HF_HUB_OFFLINE=1
  export TMPDIR=$VI/tmp PIP_NO_CACHE_DIR=1 NLTK_DATA=$VI/b200/nltk_data
  export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONHASHSEED=0
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export TRITON_CACHE_DIR=$VI/cache/$NODE/triton TORCHINDUCTOR_CACHE_DIR=$VI/cache/$NODE/inductor XDG_CACHE_HOME=$VI/cache/$NODE/xdg
  T=(); D=(--data-root "$VI/b200/data"); CW=${CW:-8}
  export EV_PAUSE=$LOGS/train.pause.$NODE EV_STATUS_GLOB="$TQ_STATUS" EV_NO_GIT=1
  export EV_DF_PATH=$VI
  EV_CAP_S=$(awk -v h="${EV_CAP_H:?任务书的上限（小时）}" 'BEGIN{print int(h * 3600)}')
fi
OUT=$REPO/experiments/W2/0066_w2_tune_$NODE
# shellcheck disable=SC1091
source "$WT/scripts/server/eval_lib.sh"
cd "$WT" || exit 1
[ "$TINY" = 1 ] || [ "${CUDA_VISIBLE_DEVICES:-}" = 0 ] || { progress "blocked: 不在 srun 的资源范围里"; exit 1; }
H=("$PY" scripts/mgr/w2_tune.py)
CFG=$OUT/cfg
mkdir -p "$CFG"
NOK=0
w2() {  # $1 = 步名，$2 = 输出目录，$3 = 配置，其余 = run_w2.py 的参数
  local name=$1 out=$2 cfg=$3; shift 3
  step "$name" "$PY" scripts/run_w2.py --model $M ${T[@]+"${T[@]}"} ${D[@]+"${D[@]}"} --split $SPLIT --w2-config "$cfg" \
    --out "$out" --candidate-workers "$CW" "$@" || { NOK=$((NOK + 1)); return 1; }
}
leaf() { echo "$1/$M/$2/$3/$SPLIT"; }  # <out> <任务> <设置>
share() {  # 把第 0 步的框复制到臂的输出目录：$1 = 臂的输出目录
  mkdir -p "$1/$M/refexp_box"
  cp -R "$S0/$M/refexp_box/baseline_e1" "$1/$M/refexp_box/"
}
die() { echo "$(now) ABORT: $*" | tee -a "$OUT/steps.log"; progress "blocked: $*"; push_out "W2 tuning aborted: $*"; exit 1; }

claim_gpu "0066-hpc-w2-tune.md" || { progress "blocked: gpu not free after ${CLAIM_MAX_S}s"; exit 1; }
echo "$(now) start commit $(git -C "$WT" rev-parse HEAD 2>/dev/null || echo unknown) node $NODE model $M split $SPLIT cw $CW" >> "$OUT/steps.log"
MEMPID=""
if [ "$TINY" != 1 ] && command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -lms 1000 > "$OUT/gpu_mem.csv" 2>/dev/null &
  MEMPID=$!
fi

# ---- 第 0 步 ----
B=configs/w2.json
S0=$OUT/${RUN_ID}_s0
w2 s0_e1_box "$S0" "$B" --task refexp_box --setting baseline_e1 || die "第 0 步的框失败"
if [ "$TINY" = 1 ] && [ -n "${INJECT:-}" ]; then  # 彩排：随机小模型给不出合法的框，换成 GT 内缩的框（tests/test_w2_cli.inject_boxes）
  "$PY" -c "import sys; sys.path.insert(0, '.'); from pathlib import Path; from tests.test_w2_cli import inject_boxes; inject_boxes(Path('$(leaf "$S0" refexp_box baseline_e1)'), '$SPLIT')"
fi
for s in w1f ownbox random_pick oracle_pick; do
  w2 "s0_$s" "$S0" "$B" --task refseg --setting $s || die "第 0 步 $s 失败"
done
step s0_rank "${H[@]}" rank --oracle "$(leaf "$S0" refseg oracle_pick)/predictions.jsonl" --base "$B" --out "$OUT/ranking.json" || die "排序失败"
W1F=$(leaf "$S0" refseg w1f)/predictions.jsonl
REFS=(--random "$(leaf "$S0" refseg random_pick)/predictions.jsonl" --oracle "$(leaf "$S0" refseg oracle_pick)/predictions.jsonl")

# ---- 第 1 步：布局 × 思考 ----
ARGS=()
for arm in grid_off sep_off grid_on sep_on; do
  lay=grid; [ "${arm%_*}" = sep ] && lay=separate
  "${H[@]}" arm --base "$B" --out "$CFG/$arm.json" --name "$arm" --layout $lay --thinking "${arm#*_}" >> "$OUT/steps.log" 2>&1 || die "写臂 $arm 失败"
  share "$OUT/${RUN_ID}_$arm"
  w2 "s1_$arm" "$OUT/${RUN_ID}_$arm" "$CFG/$arm.json" --task refseg --setting w2a || die "臂 $arm 失败"
  ARGS+=(--arm "$arm=$(leaf "$OUT/${RUN_ID}_$arm" refseg w2a)/predictions.jsonl")
done
C1=$("${H[@]}" select --w1f "$W1F" "${ARGS[@]}" "${REFS[@]}" --out "$OUT/step1.json" --md "$OUT/step1.md" --step "第 1 步") || die "第 1 步选臂失败"
echo "$(now) step1 chosen $C1" >> "$OUT/steps.log"

# ---- 第 2 步：K ----
ARGS=()
for k in 4 6; do
  arm=${C1}_k$k
  "${H[@]}" arm --base "$CFG/$C1.json" --out "$CFG/$arm.json" --name "$arm" --ranking "$OUT/ranking.json" --k $k >> "$OUT/steps.log" 2>&1 || die "写臂 $arm 失败"
  share "$OUT/${RUN_ID}_$arm"
  w2 "s2_$arm" "$OUT/${RUN_ID}_$arm" "$CFG/$arm.json" --task refseg --setting w2a || die "臂 $arm 失败"
  ARGS+=(--arm "$arm=$(leaf "$OUT/${RUN_ID}_$arm" refseg w2a)/predictions.jsonl")
done
ARGS+=(--arm "$C1=$(leaf "$OUT/${RUN_ID}_$C1" refseg w2a)/predictions.jsonl")  # K = 8
C2=$("${H[@]}" select --w1f "$W1F" "${ARGS[@]}" "${REFS[@]}" --out "$OUT/step2.json" --md "$OUT/step2.md" --step "第 2 步") || die "第 2 步选臂失败"
echo "$(now) step2 chosen $C2" >> "$OUT/steps.log"
D2=$OUT/${RUN_ID}_$C2

# ---- 第 3 步：box < box_points ----
arm=${C2}_pts
"${H[@]}" arm --base "$CFG/$C2.json" --out "$CFG/$arm.json" --name "$arm" --ranking "$OUT/ranking.json" --propose box_points >> "$OUT/steps.log" 2>&1 || die "写臂 $arm 失败"
w2 "s3_$arm" "$OUT/${RUN_ID}_$arm" "$CFG/$arm.json" --task refseg --setting w2a || die "臂 $arm 失败"
C3=$("${H[@]}" select --w1f "$W1F" --arm "$C2=$(leaf "$D2" refseg w2a)/predictions.jsonl" \
  --arm "$arm=$(leaf "$OUT/${RUN_ID}_$arm" refseg w2a)/predictions.jsonl" "${REFS[@]}" --out "$OUT/step3.json" --md "$OUT/step3.md" --step "第 3 步") \
  || die "第 3 步选臂失败"
echo "$(now) step3 chosen $C3" >> "$OUT/steps.log"
"$PY" -c "import json, hashlib; p='$CFG/$C3.json'; print(json.dumps({'chosen': '$C3', 'config': p, 'sha256': hashlib.sha256(open(p,'rb').read()).hexdigest(), 'step1': '$C1', 'step2': '$C2'}, ensure_ascii=False, indent=1))" > "$OUT/final.json"
[ -n "$MEMPID" ] && kill "$MEMPID" 2>/dev/null

echo "$(now) done; failed steps: $NOK; chosen $C3" >> "$OUT/steps.log"
progress "done $RUN_ID (failed steps: $NOK; chosen $C3)"
push_out "W2 tuning done (chosen $C3)"
