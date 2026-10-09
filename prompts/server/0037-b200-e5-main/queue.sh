#!/usr/bin/env bash
# RUN-0037 的训练队列（B200）：E5 主比较 v3（2B；E5 预注册 §8a v3；D83、D87），含 N = 0 的读数与各检查点的 test 评测。
# 三个起点（配种子：E5 的种子 i 用起点的种子 i）：
#   e3b_c3      = E3b+C3（RUN-0040，$PROJ/ckpt/0040/e3b_c3）：内化，并且保住了语言；
#   c3_warmup   = C3+预热（RUN-0040，$PROJ/ckpt/0040/c3_warmup）：文本曝光相同，只差稠密监督；
#   base_warmup = 基座+预热（RUN-0036，$PROJ/ckpt/0036/warmup）：参照。
# H5 的主判据是 e3b_c3 对 c3_warmup。网格一律结构强制解码（configs/e5_train.json 的 decode = structured，D83-3）。
# 顺序：N = 0（3 起点 × 3 种子）→ 1K、4K 的三个种子 → 16K 的三个种子 → 64K 的种子 0 → a–d 迁移（4K，三个起点，test 只评 e）。
# 每个 E5 运行训完，紧接着在 test 上评一次（选定的检查点只评这一次，D77-3）。
# 驱动的规则见 scripts/server/train_queue.sh；RUN-0040 的队列跑完才开始（wait_for_queue 0040）。
#
# 用法（B200）：setsid nohup bash /dev/shm/visint/tmp/wt0037/prompts/server/0037-b200-e5-main/queue.sh \
#   > $PROJ/logs/train_0037.driver.log 2>&1 < /dev/null &
# 彩排（开发机）：TINY=1 WT=… REPO=… OUT=… CK_TMP=… CK_FINAL=… LOGS=… PY=… ROOT36=<有 warmup 的 CK_FINAL> ROOT40=<有 e3b_c3、c3_warmup 的 CK_FINAL> bash queue.sh
RUN_ID=0037
M=qwen3.5-2b
if [ "${TINY:-0}" = 1 ]; then
  : "${WT:?}" "${REPO:?}" "${OUT:?}" "${CK_TMP:?}" "${CK_FINAL:?}" "${LOGS:?}" "${PY:?}" "${ROOT36:?}" "${ROOT40:?}"
  CHUNK=${CHUNK:-3}; EXTRA=(--tiny); DATA=()
  M=qwen3.5-0.8b
  SMALL=(10); MID=(20); BIG=(); SEEDS=(${SEEDS:-0}); TIER_AD=10; EVAL_LIMIT=(--limit 2)
else
  export ROOT=/home/ubuntu/workspace/wzg/gjl
  export PROJ=$ROOT/vision-internalization
  WT=/dev/shm/visint/tmp/wt0037
  REPO=$PROJ/repo
  LOGS=$PROJ/logs
  OUT=$REPO/experiments/E5/0037_e5_main
  CK_TMP=/dev/shm/visint/ckpt_0037
  CK_FINAL=$PROJ/ckpt/0037
  ROOT36=$PROJ/ckpt/0036
  ROOT40=$PROJ/ckpt/0040
  PY=$PROJ/venv/bin/python
  CHUNK=1000
  export HF_HOME=$ROOT/.hf HF_HUB_CACHE=/dev/shm/visint/hf-hub HF_HUB_OFFLINE=1 TMPDIR=/dev/shm/visint/tmp PIP_NO_CACHE_DIR=1
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  EXTRA=(); DATA=(--data-root "$PROJ/data")  # 起点检查点的合并核对用 E1 的 grid16 开发子集（根盘上的 E1 落地）
  SMALL=(1000 4000); MID=(16000); BIG=(64000); SEEDS=(0 1 2); TIER_AD=4000; EVAL_LIMIT=()
  QUEUE_CAP_H=70  # 任务书 §2：累计跑命令的墙钟上限；每条开始前按"已用 + 这一条的预估"判断（D81-2）
fi
# shellcheck disable=SC1091
source "$WT/scripts/server/train_queue.sh"
cd "$WT" || die "工作树不存在：$WT"

STARTS=(e3b_c3 c3_warmup base_warmup)
start_ckpt() {  # $1 = 起点，$2 = 种子 → 起点的最终检查点（model/；驱动按训练叶目录的结构存放，旁边有叶目录的 config.yaml）
  case "$1" in
    e3b_c3)      echo "$ROOT40/e3b_c3/$M/seed$2/ckpt/final/model" ;;
    c3_warmup)   echo "$ROOT40/c3_warmup/$M/seed$2/ckpt/final/model" ;;
    base_warmup) echo "$ROOT36/warmup/$M/seed$2/ckpt/final/model" ;;
  esac
}
est_train() {  # 每条训练的预估（小时，任务书 §2；含 val 选择，结构强制解码）
  case "$1" in 1000|4000) echo 0.75 ;; 16000) echo 1.15 ;; 64000) echo 4.5 ;; *) echo 0.01 ;; esac
}
e5() {  # 训一个 E5 运行，跑完紧接着在 test 上评一次。$1 = 起点，$2 = 数据量档，$3 = 种子，其余 = 额外参数（--tasks …）
  local init=$1 n=$2 s=$3; shift 3
  local tag="${init}_n${n}_s${s}" ck task=(--task all)
  [ $# -gt 0 ] && tag="${tag}_$(echo "$*" | sed 's/--tasks //; s/ //g')"
  case "$tag" in *_abcd) task=(--task e) ;; esac  # a–d 迁移：test 只评 e（E5 预注册 §8a 第 4 项）
  ck=$(start_ckpt "$init" "$s")
  [ -d "$ck" ] || { note "$tag：起点检查点不在（$ck），跳过"; echo "$(now) $tag missing-start $ck" >> "$DRV/failed.txt"; return 1; }
  EST_H=$(est_train "$n")
  run "$tag" scripts/train_e5.py --init "$init" --start-checkpoint "$ck" --model $M --n-train "$n" --seed "$s" \
    ${DATA[@]+"${DATA[@]}"} "$@" || return 1
  case "${task[*]}" in "--task e") EST_H=0.1 ;; *) EST_H=0.35 ;; esac
  run_cmd "eval_$tag" "$PY" scripts/eval_e5_trained.py --checkpoint "$(ckpt_of "$tag")" --model $M --out "$OUT" "${task[@]}" \
    ${EXTRA[@]+"${EXTRA[@]}"} ${EVAL_LIMIT[@]+"${EVAL_LIMIT[@]}"}
}
n0() {  # N = 0 的读数（D81-1、D83-5；DEV-0040）：起点不经 E5 训练直接评 test；与训练用同一个 --out，起点的合并核对互相复用
  local init=$1 s=$2 ck
  ck=$(start_ckpt "$init" "$s")
  [ -d "$ck" ] || { note "n0_${init}_s$s：起点检查点不在（$ck），跳过"; echo "$(now) n0_${init}_s$s missing-start $ck" >> "$DRV/failed.txt"; return 1; }
  EST_H=0.4
  run_cmd "n0_${init}_s$s" "$PY" scripts/eval_e5_trained.py --start-only --init "$init" --start-checkpoint "$ck" --model $M --seed "$s" \
    ${DATA[@]+"${DATA[@]}"} --out "$OUT" --task all ${EXTRA[@]+"${EXTRA[@]}"} ${EVAL_LIMIT[@]+"${EVAL_LIMIT[@]}"}
}

[ "$TINY" = 1 ] || wait_for_queue 0040
note "队列开始：commit $(git -C "$WT" rev-parse HEAD)，起点根 $ROOT40、$ROOT36，每段 $CHUNK 步"
# 0. N = 0（最先跑，顺带检验评测入口）
for s in "${SEEDS[@]}"; do for init in "${STARTS[@]}"; do n0 "$init" "$s"; done; done
# 1. 1K、4K 的三个种子 → 2. 16K 的三个种子 → 3. 64K 的种子 0 → 4. a–d 迁移
for n in "${SMALL[@]}"; do for s in "${SEEDS[@]}"; do for init in "${STARTS[@]}"; do e5 "$init" "$n" "$s"; done; done; done
for n in ${MID[@]+"${MID[@]}"}; do for s in "${SEEDS[@]}"; do for init in "${STARTS[@]}"; do e5 "$init" "$n" "$s"; done; done; done
for n in ${BIG[@]+"${BIG[@]}"}; do for init in "${STARTS[@]}"; do e5 "$init" "$n" 0; done; done
for s in "${SEEDS[@]}"; do for init in "${STARTS[@]}"; do e5 "$init" "$TIER_AD" "$s" --tasks a b c d; done; done
finish
