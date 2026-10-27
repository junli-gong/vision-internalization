#!/usr/bin/env bash
# RUN-0082（ITSC HPC hpcgpu07）：探索 X-3（D142；探索日志 X-2、X-3）。单一 LoRA：W2-c 的提出、挑选（K = 4），描样条（每步 10 条），回放；
#   没有修正流。训练数据复用 RUN-0080 的 0080_x2_data（只读样条部分）与 RUN-0076 的回放数据。
#   1. 排队：等 $LOGS/eval_0080.status 以 "done 0080" 开头（X-2 的评测跑完），再 claim_gpu。
#   2. train：train_w2c.py --config configs/x3_train.json，2,000 步（RUN-0080 的每步 17.5 s 折算约 16.4–16.9 s，约 9.3 小时）。
#   3. eval（都在 07，与 RUN-0080 同节点配对）：run_x2 在 explore_val 三层、devtune800、devcheck2（修正类设置自动跳过）；
#      健康检查 X-3，与 RUN-0080 在 07 上的 health_base 配对。W2-c 参照沿用 RUN-0080 的 w2c_ref（同节点）。
#   test 一律不碰；devcheck3 不碰（判定前）。
# 铁律：任何情况下都不能取消或停止 Slurm 任务 157296、157297。只用本节点自己的暂停文件。
# 用法：srun --jobid=157297 --overlap -n1 -c 12 env PATH=… TINY=0 EV_CAP_H=… bash …/x3.sh < /dev/null
# 彩排（开发机）：TINY=1 WT=… REPO=… LOGS=… PY=… X2DATA=… REPLAY=… W2C_DATA=… BASE_HEALTH=… bash x3.sh
set -uo pipefail
TINY=${TINY:-0}
RUN_ID=0082
M=qwen3.5-9b
EVSPLITS=(explore_val_refcoco_unc_val "explore_val_refcoco+_unc_val" explore_val_refcocog_umd_val)
W2SPLITS=(devtune800 devcheck2)
if [ "$TINY" = 1 ]; then
  : "${WT:?}" "${REPO:?}" "${LOGS:?}" "${PY:?}" "${X2DATA:?}" "${REPLAY:?}" "${W2C_DATA:?}" "${BASE_HEALTH:?}"
  NODE=tiny; M=qwen3.5-0.8b
  T=(--tiny); D=(); TD=(); CW=1; TW=1; HT=(--tiny --skip-ifeval-scoring --limit 2); LIMIT=(--limit 3)
  EVSPLITS=(explore_val_refcoco_unc_val); W2SPLITS=(devtune800)
  export NLTK_DATA=$WT/tests/fixtures/health/nltk_data EV_NO_GIT=1
else
  NODE=hpcgpu07
  VI=/home/mail/chenyuma/visint
  WT=$VI/wt/wt$RUN_ID
  REPO=$VI/repo
  LOGS=$VI/logs
  PY=$VI/env/bin/python
  export HF_HOME=$VI/hf HF_HUB_CACHE=$VI/hf/hub HF_HUB_OFFLINE=1
  export TMPDIR=$VI/tmp PIP_NO_CACHE_DIR=1 NLTK_DATA=$VI/b200/nltk_data
  export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONHASHSEED=0
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export TRITON_CACHE_DIR=$VI/cache/$NODE/triton TORCHINDUCTOR_CACHE_DIR=$VI/cache/$NODE/inductor XDG_CACHE_HOME=$VI/cache/$NODE/xdg
  T=(); D=(--data-root "$VI/b200/data"); TD=(--data-root "$VI/data/train_data"); CW=8; TW=8; HT=(); LIMIT=()
  W2C_DATA=$REPO/experiments/W2/0067_w2c_data_hpcgpu06/w2c_data
  X2DATA=$REPO/experiments/X/0080_x2_hpcgpu07/0080_x2_data
  REPLAY=$REPO/experiments/W2/0076_w2c_replay_hpcgpu07/0076_replay_data
  BASE_HEALTH=$REPO/experiments/X/0080_x2_hpcgpu07/0080_eval/health/health_base.json
  export EV_PAUSE=$LOGS/train.pause.$NODE EV_STATUS_GLOB="nonexistent_queue.status" EV_NO_GIT=1
  export EV_DF_PATH=$VI
  CLAIM_MAX_S=${CLAIM_MAX_S:-3600}
  EV_CAP_S=$(awk -v h="${EV_CAP_H:?任务书的上限（小时）}" 'BEGIN{print int(h * 3600)}')
fi
OUT=$REPO/experiments/X/${RUN_ID}_x3_$NODE
TRAIN=$OUT/0082_train
EVAL=$OUT/0082_eval
LEAF=$TRAIN/w2c/$M/seed0
CK=$LEAF/ckpt/final/model
# --out 的目录名必须以运行编号开头（CLAUDE.md"服务器驱动的写法"）：训练与评测的 --out 都查
for o in "$TRAIN" "$EVAL"; do case "$(basename "$o")" in [0-9][0-9][0-9][0-9]_*) ;; *) echo "目录名没有运行编号：$o" >&2; exit 2;; esac; done
# shellcheck disable=SC1091
source "$WT/scripts/server/eval_lib.sh"
cd "$WT" || exit 1
[ "$TINY" = 1 ] || [ "${CUDA_VISIBLE_DEVICES:-}" = 0 ] || { progress "blocked: 不在 srun 的资源范围里"; exit 1; }
if [ "$TINY" != 1 ] && ! "$PY" -c "import json,sys; sys.exit(0 if json.load(open('configs/w2.json'))['frozen'] is True else 1)"; then
  progress "blocked: configs/w2.json 没有冻结"; exit 1
fi
die() { echo "$(now) ABORT: $*" | tee -a "$OUT/steps.log"; progress "blocked: $*"; push_out "X-3 aborted: $*"; exit 1; }
if [ "$TINY" != 1 ]; then
  t0=$(date +%s)
  while :; do
    st=$(cut -d' ' -f2- "$LOGS/eval_0080.status" 2>/dev/null || true)
    case "$st" in "done 0080"*) break;; blocked*) progress "blocked: RUN-0080 是 blocked：$st"; exit 1;; esac
    progress "waiting for eval_0080.status to start with 'done 0080' (now: ${st:-missing})"
    [ $(( $(date +%s) - t0 )) -ge 172800 ] && { progress "blocked: 等 RUN-0080 超时"; exit 1; }
    sleep 60
  done
  [ -f "$BASE_HEALTH" ] || { progress "blocked: 没有 RUN-0080 的 health_base（$BASE_HEALTH）"; exit 1; }
fi
NOK=0
claim_gpu "0082-hpc-07-x3.md" || { progress "blocked: gpu not free after ${CLAIM_MAX_S}s"; exit 1; }
mkdir -p "$OUT"
echo "$(now) start commit $(git -C "$WT" rev-parse HEAD 2>/dev/null || echo unknown) node $NODE model $M x3 cfg $(sha "$WT/configs/x3_train.json" | cut -d' ' -f1)" >> "$OUT/steps.log"
MEMPID=""
if [ "$TINY" != 1 ] && command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -lms 1000 >> "$OUT/gpu_mem.csv" 2>/dev/null &
  MEMPID=$!
fi
# 训练（叶目录已有 latest 时自动 --resume；已有最终检查点就跳过）
if [ ! -f "$CK/manifest.json" ]; then
  RES=(); [ -d "$LEAF/ckpt/latest" ] && RES=(--resume)
  step train "$PY" scripts/train_w2c.py --config configs/x3_train.json --model $M ${T[@]+"${T[@]}"} --w2c-data "$W2C_DATA" \
    --x2-data "$X2DATA" --replay-data "$REPLAY" ${TD[@]+"${TD[@]}"} --out "$TRAIN" ${RES[@]+"${RES[@]}"} || die "训练失败"
fi
[ -f "$CK/manifest.json" ] || die "训练后没有检查点 $CK"
echo "$(now) checkpoint $CK manifest $(sha "$CK/manifest.json" | cut -d' ' -f1)" >> "$OUT/steps.log"
X=(--model $M ${T[@]+"${T[@]}"} ${D[@]+"${D[@]}"} --checkpoint "$CK" --candidate-workers "$CW" --tool-workers "$TW" ${LIMIT[@]+"${LIMIT[@]}"})
for SP in "${EVSPLITS[@]}"; do
  step "x3_$SP" "$PY" scripts/run_x2.py "${X[@]}" --splits-file configs/explore_splits.json --split "$SP" --out "$EVAL" || NOK=$((NOK + 1))
done
for SP in "${W2SPLITS[@]}"; do
  step "x3_$SP" "$PY" scripts/run_x2.py "${X[@]}" --split "$SP" --out "$EVAL" || NOK=$((NOK + 1))
done
HC=(--model $M ${HT[@]+"${HT[@]}"})
step health_x3 "$PY" scripts/health_check.py "${HC[@]}" --checkpoint "$CK" ${D[@]+"${D[@]}"} --out "$EVAL/health/health_x3.json" || NOK=$((NOK + 1))
step health_compare "$PY" scripts/health_check.py --compare "$BASE_HEALTH" "$EVAL/health/health_x3.json" --out "$EVAL/health/health_compare.json" || NOK=$((NOK + 1))
[ -n "$MEMPID" ] && kill "$MEMPID" 2>/dev/null
echo "$(now) done; failed steps: $NOK" >> "$OUT/steps.log"
progress "done $RUN_ID (failed steps: $NOK)"
push_out "X-3 ${RUN_ID} done (failed steps: $NOK)"
