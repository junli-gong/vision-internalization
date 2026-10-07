#!/usr/bin/env bash
# RUN-0036 的训练队列（B200）：2B 的 E3 全部变体 × 3 个种子，加 2B 的网格格式预热 × 3 个种子（D77-4、D78-5）。
# 按种子分批；每个种子先跑 E5 主比较要用的三个起点（预热、E3b、E3a），再跑对照（C3、C1′、C1、C2）。
# 驱动的规则见 scripts/server/train_queue.sh。
#
# 用法（B200，服务器 CC 按任务书在固定 commit 的工作树里启动，与 CC 脱钩）：
#   setsid nohup bash /dev/shm/visint/tmp/wt0036/prompts/server/0036-b200-e3-train-2b/queue.sh \
#     > $PROJ/logs/train_0036.driver.log 2>&1 < /dev/null &
# 彩排（开发机）：TINY=1 WT=<仓库根> REPO=<彩排用的 clone> OUT=<REPO 里的目录> CK_TMP=… CK_FINAL=… LOGS=… PY=<python> bash queue.sh
RUN_ID=0036
M=qwen3.5-2b
if [ "${TINY:-0}" = 1 ]; then
  : "${WT:?}" "${REPO:?}" "${OUT:?}" "${CK_TMP:?}" "${CK_FINAL:?}" "${LOGS:?}" "${PY:?}"
  CHUNK=${CHUNK:-2}; EXTRA=(--tiny); DATA=()
  M=qwen3.5-0.8b  # 开发机只缓存了 0.8B 的分词器与配置；小型随机模型的结构与尺寸无关
  SEEDS=(${SEEDS:-0})
else
  export ROOT=/home/ubuntu/workspace/wzg/gjl
  export PROJ=$ROOT/vision-internalization
  WT=/dev/shm/visint/tmp/wt0036
  REPO=$PROJ/repo
  LOGS=$PROJ/logs
  TRAIN=/dev/shm/visint/train_data
  OUT=$REPO/experiments/E3/0036_e3_train_2b
  CK_TMP=/dev/shm/visint/ckpt_0036
  CK_FINAL=$PROJ/ckpt/0036
  PY=$PROJ/venv/bin/python
  CHUNK=1000
  export HF_HOME=$ROOT/.hf HF_HUB_CACHE=/dev/shm/visint/hf-hub HF_HUB_OFFLINE=1 TMPDIR=/dev/shm/visint/tmp PIP_NO_CACHE_DIR=1
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  EXTRA=(); DATA=(--data-root "$TRAIN"); TQ_TRAIN_DATA=$TRAIN
  SEEDS=(0 1 2)
fi
# shellcheck disable=SC1091
source "$WT/scripts/server/train_queue.sh"
cd "$WT" || die "工作树不存在：$WT"
[ "$TINY" = 1 ] || [ -d "$TRAIN" ] || die "训练数据不在：$TRAIN（内存盘可能被清空，按 RUN-0034 §3 重新落地并核对）"
note "队列开始：commit $(git -C "$WT" rev-parse HEAD)，种子 ${SEEDS[*]}，每段 $CHUNK 步"

for s in "${SEEDS[@]}"; do
  # E5 主比较的三个起点（D77-3、D78-5：E5 的种子 i 配这里的种子 i）
  run "warmup_s$s"   scripts/train_e3b.py --model $M --supervision warmup --seed "$s"
  run "e3b_s$s"      scripts/train_e3b.py --model $M --supervision grid --seed "$s" ${DATA[@]+"${DATA[@]}"}
  run "e3a_full_s$s" scripts/train_e3a.py --model $M --variant full --seed "$s" ${DATA[@]+"${DATA[@]}"}
  # 对照
  run "c3_s$s"       scripts/train_e3b.py --model $M --supervision text --seed "$s" ${DATA[@]+"${DATA[@]}"}
  run "c1prime_s$s"  scripts/train_e3b.py --model $M --supervision grid --variant c1prime --seed "$s" ${DATA[@]+"${DATA[@]}"}
  run "e3a_c1_s$s"   scripts/train_e3a.py --model $M --variant c1 --seed "$s" ${DATA[@]+"${DATA[@]}"}
  run "e3a_c2_s$s"   scripts/train_e3a.py --model $M --variant c2 --seed "$s" ${DATA[@]+"${DATA[@]}"}
  note "2B seed $s 的 7 条都过了一遍"
done
finish
