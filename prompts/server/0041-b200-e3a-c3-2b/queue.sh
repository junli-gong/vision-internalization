#!/usr/bin/env bash
# RUN-0041 的训练队列（B200）：2B 的 E3a+C3 × 3 个种子（D83：机制比较，E3b 的迁移是否大于 E3a；E3 预注册 §10a、§12）。
# 实现：DEV-0044（D86）。每步 8 条稠密（与 E3a full 同种子的样本流）+ 8 条文本（与 C3 同种子的样本流），8,000 步，损失 1:1。
# 前一个队列跑完才开始：默认等 RUN-0040（AFTER=0040）；任务书可以另给 AFTER。
# 驱动的规则见 scripts/server/train_queue.sh。
#
# 用法（B200）：setsid nohup bash /dev/shm/visint/tmp/wt0041/prompts/server/0041-b200-e3a-c3-2b/queue.sh \
#   > $PROJ/logs/train_0041.driver.log 2>&1 < /dev/null &
# 彩排（开发机）：TINY=1 WT=… REPO=… OUT=… CK_TMP=… CK_FINAL=… LOGS=… PY=… bash queue.sh
RUN_ID=0041
M=qwen3.5-2b
AFTER=${AFTER:-0040}
if [ "${TINY:-0}" = 1 ]; then
  : "${WT:?}" "${REPO:?}" "${OUT:?}" "${CK_TMP:?}" "${CK_FINAL:?}" "${LOGS:?}" "${PY:?}"
  CHUNK=${CHUNK:-3}; EXTRA=(--tiny); DATA=()
  M=qwen3.5-0.8b
  SEEDS=(${SEEDS:-0})
else
  export ROOT=/home/ubuntu/workspace/wzg/gjl
  export PROJ=$ROOT/vision-internalization
  WT=/dev/shm/visint/tmp/wt0041
  REPO=$PROJ/repo
  LOGS=$PROJ/logs
  TRAIN=/dev/shm/visint/train_data
  OUT=$REPO/experiments/E3/0041_e3a_c3_2b
  CK_TMP=/dev/shm/visint/ckpt_0041
  CK_FINAL=$PROJ/ckpt/0041
  PY=$PROJ/venv/bin/python
  CHUNK=1000
  export HF_HOME=$ROOT/.hf HF_HUB_CACHE=/dev/shm/visint/hf-hub HF_HUB_OFFLINE=1 TMPDIR=/dev/shm/visint/tmp PIP_NO_CACHE_DIR=1
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  EXTRA=(); DATA=(--data-root "$TRAIN"); TQ_TRAIN_DATA=$TRAIN
  SEEDS=(0 1 2)
  QUEUE_CAP_H=15  # 任务书 §2：每个约 2.9–3.7 小时，3 个种子约 9–11 小时
fi
# shellcheck disable=SC1091
source "$WT/scripts/server/train_queue.sh"
cd "$WT" || die "工作树不存在：$WT"
[ "$TINY" = 1 ] || [ -d "$TRAIN" ] || die "训练数据不在：$TRAIN（内存盘可能被清空，按 RUN-0034 §3 重新落地并核对）"
[ "$TINY" = 1 ] || wait_for_queue "$AFTER"
note "队列开始：commit $(git -C "$WT" rev-parse HEAD)，种子 ${SEEDS[*]}，每段 $CHUNK 步"

for s in "${SEEDS[@]}"; do
  EST_H=3.7
  run "e3a_c3_s$s" scripts/train_e3a.py --model $M --variant full --with-text --seed "$s" ${DATA[@]+"${DATA[@]}"}
done
finish
