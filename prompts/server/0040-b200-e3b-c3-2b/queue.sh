#!/usr/bin/env bash
# RUN-0040 的训练队列（B200）：2B 的 E3b+C3 × 3 个种子，加 C3+预热 × 3 个种子（D83；E3 预注册 §12；E5 §8a v3 的两个新起点）。
# 实现：DEV-0042（D85）。E3b+C3 = 每步 8 条网格（与 E3b 同种子的样本流）+ 8 条文本（与 C3 同种子的样本流），8,000 步，损失 1:1。
# C3+预热 = 在 RUN-0036 的 C3（同种子）之上做网格格式预热，存成一个秩 64 的 LoRA。
# 按种子分批：每个种子先 E3b+C3，再 C3+预热。RUN-0036 的队列跑完才开始（C3 的种子 2 在它里面）。
# 驱动的规则见 scripts/server/train_queue.sh。
#
# 用法（B200）：setsid nohup bash /dev/shm/visint/tmp/wt0040/prompts/server/0040-b200-e3b-c3-2b/queue.sh \
#   > $PROJ/logs/train_0040.driver.log 2>&1 < /dev/null &
# 彩排（开发机）：TINY=1 WT=… REPO=… OUT=… CK_TMP=… CK_FINAL=… LOGS=… PY=… C3_ROOT=<有 c3 检查点的 CK_FINAL> bash queue.sh
RUN_ID=0040
M=qwen3.5-2b
if [ "${TINY:-0}" = 1 ]; then
  : "${WT:?}" "${REPO:?}" "${OUT:?}" "${CK_TMP:?}" "${CK_FINAL:?}" "${LOGS:?}" "${PY:?}" "${C3_ROOT:?}"
  CHUNK=${CHUNK:-3}; EXTRA=(--tiny); DATA=()
  M=qwen3.5-0.8b
  SEEDS=(${SEEDS:-0})
else
  export ROOT=/home/ubuntu/workspace/wzg/gjl
  export PROJ=$ROOT/vision-internalization
  WT=/dev/shm/visint/tmp/wt0040
  REPO=$PROJ/repo
  LOGS=$PROJ/logs
  TRAIN=/dev/shm/visint/train_data
  OUT=$REPO/experiments/E3/0040_e3b_c3_2b
  CK_TMP=/dev/shm/visint/ckpt_0040
  CK_FINAL=$PROJ/ckpt/0040
  C3_ROOT=$PROJ/ckpt/0036
  PY=$PROJ/venv/bin/python
  CHUNK=1000
  export HF_HOME=$ROOT/.hf HF_HUB_CACHE=/dev/shm/visint/hf-hub HF_HUB_OFFLINE=1 TMPDIR=/dev/shm/visint/tmp PIP_NO_CACHE_DIR=1
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  EXTRA=(); DATA=(--data-root "$TRAIN"); TQ_TRAIN_DATA=$TRAIN
  SEEDS=(0 1 2)
  QUEUE_CAP_H=16  # 任务书 §2：E3b+C3 每个约 3.5 小时，C3+预热每个约 0.2 小时，合计约 11 小时
fi
# shellcheck disable=SC1091
source "$WT/scripts/server/train_queue.sh"
cd "$WT" || die "工作树不存在：$WT"
[ "$TINY" = 1 ] || [ -d "$TRAIN" ] || die "训练数据不在：$TRAIN（内存盘可能被清空，按 RUN-0034 §3 重新落地并核对）"
[ "$TINY" = 1 ] || wait_for_queue 0036
note "队列开始：commit $(git -C "$WT" rev-parse HEAD)，种子 ${SEEDS[*]}，每段 $CHUNK 步"

for s in "${SEEDS[@]}"; do
  EST_H=3.6
  run "e3b_c3_s$s" scripts/train_e3b.py --model $M --supervision grid+text --seed "$s" ${DATA[@]+"${DATA[@]}"}
  c3=$C3_ROOT/c3/$M/seed$s/ckpt/final/model
  if [ -d "$c3" ]; then
    EST_H=0.3
    run "c3_warmup_s$s" scripts/train_e3b.py --model $M --supervision warmup --start-checkpoint "$c3" --seed "$s"
  else
    note "c3_warmup_s$s：C3 的检查点不在（$c3），跳过"; echo "$(now) c3_warmup_s$s missing-parent $c3" >> "$DRV/failed.txt"
  fi
  note "2B seed $s：E3b+C3 与 C3+预热都过了一遍"
done
finish
