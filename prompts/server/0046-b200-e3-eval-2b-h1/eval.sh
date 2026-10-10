#!/usr/bin/env bash
# RUN-0046（B200）：2B 的 H1 修订版核心评测（E3 预注册 §12；D83、D85）。单轮主读数加 T 文本，三组：
#   1. E3b+C3 种子 0–2（RUN-0040）：网格 refseg、depth 用结构强制（D83-3），加 T 文本全套（eval、debiased、free、letter_swap）。
#      种子 0 排最前：它是 E5 的起点，要尽早确认语言保住、网格能力还在。
#   2. C3 种子 1、2（RUN-0036）：T 文本全套；种子 0 在 RUN-0038。
#   3. E3b 种子 0–2（RUN-0036）：网格 refseg、depth 的结构强制读数，与 E3b+C3 同一种解码（D83-3"新旧臂比较用同一种解码"）；
#      种子 0 的 depth 已在 RUN-0039，这里跳过。
# 检查点还没出来的（E3b+C3 种子 2 在 RUN-0040 的最后），等它出现，最多等 WAIT_MAX_S（默认 8 小时）。
#
# 用法（B200，服务器 CC 在固定 commit 的工作树里、与 CC 脱钩地启动）：
#   setsid nohup bash /dev/shm/visint/tmp/wt0046/prompts/server/0046-b200-e3-eval-2b-h1/eval.sh > $PROJ/logs/eval_0046.driver.log 2>&1 < /dev/null &
# 彩排（开发机）：TINY=1 WT=… REPO=… OUT=… LOGS=… PY=… CK36=<有 c3、e3b 的 CK_FINAL> CK40=<有 e3b_c3 的 CK_FINAL> bash eval.sh
RUN_ID=0046
M=qwen3.5-2b
WAIT_MAX_S=${WAIT_MAX_S:-28800}
if [ "${TINY:-0}" = 1 ]; then
  : "${WT:?}" "${REPO:?}" "${OUT:?}" "${LOGS:?}" "${PY:?}" "${CK36:?}" "${CK40:?}"
  M=qwen3.5-0.8b
  T=(--tiny); D=(); NOK=0; SEEDS=(${SEEDS:-0}); WAIT_MAX_S=5
else
  export ROOT=/home/ubuntu/workspace/wzg/gjl
  export PROJ=$ROOT/vision-internalization
  WT=/dev/shm/visint/tmp/wt0046
  REPO=$PROJ/repo
  LOGS=$PROJ/logs
  OUT=$REPO/experiments/E3/0046_e3_eval_2b_h1
  CK36=$PROJ/ckpt/0036
  CK40=$PROJ/ckpt/0040
  PY=$PROJ/venv/bin/python
  export HF_HOME=$ROOT/.hf HF_HUB_CACHE=/dev/shm/visint/hf-hub HF_HUB_OFFLINE=1 TMPDIR=/dev/shm/visint/tmp PIP_NO_CACHE_DIR=1
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  T=(); D=(--data-root "$PROJ/data"); NOK=0; SEEDS=(0 1 2)
  EV_CAP_S=$((20 * 3600))  # 任务书 §3 的上限：拿到 GPU 之后 20 小时
fi
# shellcheck disable=SC1091
source "$WT/scripts/server/eval_lib.sh"
cd "$WT" || exit 1

wait_ckpt() {  # $1 = 检查点目录；等它出现（训练队列跑完那一条才会搬到根盘）
  local t0; t0=$(date +%s)
  while [ ! -d "$1" ]; do
    [ $(( $(date +%s) - t0 )) -ge "$WAIT_MAX_S" ] && { echo "$(now) 等不到 $1" >> "$OUT/steps.log"; return 1; }
    release_gpu; progress "waiting for checkpoint $1"; sleep 300
    claim_gpu "0046-b200-e3-eval-2b-h1.md (resume)" || return 1
  done
}
grid() {  # $1 = 名字，$2 = 检查点，$3 = 变体，$4 = 种子，其余 = 额外参数
  local name=$1 ck=$2 v=$3 s=$4; shift 4
  step "$name" "$PY" scripts/eval_e3b.py --checkpoint "$ck" --model "$M" --variant "$v" --seed "$s" ${T[@]+"${T[@]}"} ${D[@]+"${D[@]}"} \
    --out "$OUT" --k-values 1 --decode structured "$@" || NOK=$((NOK + 1))
}
ttext() {  # T 文本全套：$1 = 变体，$2 = 种子，$3 = 检查点（model/）
  local v=$1 s=$2 ck=$3 lab="t_text/$1/seed$2" sub
  for sub in eval debiased free letter_swap; do
    step "ttext_${v}_s${s}_$sub" "$PY" scripts/run_e1.py --model "$M" ${T[@]+"${T[@]}"} ${D[@]+"${D[@]}"} --checkpoint "$ck" \
      --checkpoint-label "$lab" --out "$OUT" --task all --subset "$sub" || NOK=$((NOK + 1))
  done
}

claim_gpu "0046-b200-e3-eval-2b-h1.md" || { progress "blocked: gpu not free after ${CLAIM_MAX_S}s"; exit 1; }
echo "$(now) start commit $(git -C "$WT" rev-parse HEAD)" >> "$OUT/steps.log"

# 1. E3b+C3（种子 0 最先）
for s in "${SEEDS[@]}"; do
  ck=$CK40/e3b_c3/$M/seed$s/ckpt/final/model
  wait_ckpt "$ck" || { NOK=$((NOK + 1)); continue; }
  ttext e3b_c3 "$s" "$ck"
  grid "grid_e3b_c3_s${s}_refseg" "$ck" e3b_c3 "$s" --task refseg
  grid "grid_e3b_c3_s${s}_depth"  "$ck" e3b_c3 "$s" --task depth
  push_out "2B E3b+C3 s$s: T-text battery + structured grid (refseg, depth)"
done
# 2. C3 种子 1、2 的 T 文本
for s in "${SEEDS[@]}"; do
  [ "$s" = 0 ] && [ "$TINY" != 1 ] && continue  # 种子 0 在 RUN-0038
  ttext c3 "$s" "$CK36/c3/$M/seed$s/ckpt/final/model"
  push_out "2B C3 s$s: T-text battery"
done
# 3. E3b 的结构强制网格读数
for s in "${SEEDS[@]}"; do
  ck=$CK36/e3b/$M/seed$s/ckpt/final/model
  grid "grid_e3b_s${s}_refseg" "$ck" e3b "$s" --task refseg
  if [ "$s" != 0 ] || [ "$TINY" = 1 ]; then grid "grid_e3b_s${s}_depth" "$ck" e3b "$s" --task depth; fi  # 种子 0 的 depth 在 RUN-0039
  push_out "2B E3b s$s: structured grid"
done

echo "$(now) done; failed steps: $NOK" >> "$OUT/steps.log"
progress "done 0046 (failed steps: $NOK)"
push_out "2B H1 core eval done (failed steps: $NOK)"
