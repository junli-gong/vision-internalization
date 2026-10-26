#!/usr/bin/env bash
# RUN-0080（ITSC HPC hpcgpu07）：探索 X-2（D126-3、D136；探索日志 X-2）。单一 LoRA：W2-c 的提出、挑选，加描样条、指修正框，加回放。
#   1. data（CPU，不等 GPU）：build_x2_data.py，从 RUN-0067 的 w2c_data 行算样条目标与两轮修正目标。
#   2. 排队：等 WAIT_DONE（缺省 eval_0076.status 以 "done 0076 [train]" 开头，即回放变体训练完），再 claim_gpu。
#   3. train：先跑 50 步（--stop-after-steps 50），取第 5 步以后 step_s 的中位数：≤ 21.6 s 续跑到 2,000 步，否则另起 1,500 步（D136-7）。
#   4. eval（都在 07，配对）：
#      - run_x2：explore_val 三层、devtune800、devcheck2，全部设置；
#      - 参照：W2-c（RUN-0069 的检查点）在 explore_val 三层与 devtune800 上（run_w2 --setting w2c）；
#      - 健康检查：基座与 X-2，同节点配对。
#   test 一律不碰；devcheck3 不碰（判定前）。
# 铁律：任何情况下都不能取消或停止 Slurm 任务 157296、157297。只用本节点自己的暂停文件。
# 用法：srun --jobid=157297 --overlap -n1 -c 12 env PATH=… TINY=0 EV_CAP_H=… [STAGES=…] [WAIT_DONE=…] bash …/x2.sh < /dev/null
# 彩排（开发机）：TINY=1 WT=… REPO=… LOGS=… PY=… X2_TINY_CK=<tiny 检查点> bash x2.sh（tiny 跳过训练，用给的小检查点走评测）
set -uo pipefail
TINY=${TINY:-0}
RUN_ID=0080
STAGES=${STAGES:-"data train eval"}
M=qwen3.5-9b
EVSPLITS=(explore_val_refcoco_unc_val "explore_val_refcoco+_unc_val" explore_val_refcocog_umd_val)
W2SPLITS=(devtune800 devcheck2)
if [ "$TINY" = 1 ]; then
  : "${WT:?}" "${REPO:?}" "${LOGS:?}" "${PY:?}" "${X2_TINY_CK:?}"
  NODE=tiny; M=qwen3.5-0.8b
  T=(--tiny); D=(); TD=(); CW=1; TW=1
  EVSPLITS=(explore_val_refcoco_unc_val); W2SPLITS=(devtune800)
  HT=(--tiny --skip-ifeval-scoring --limit 2)
  export NLTK_DATA=$WT/tests/fixtures/health/nltk_data EV_NO_GIT=1
  W2C_DATA=$REPO/experiments/W2/0067_w2c_data_tiny/w2c_data
  W2C_CK=$X2_TINY_CK
  WAIT_DONE=${WAIT_DONE:-}
  LIMIT=(--limit 3)
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
  W2C_CK=$REPO/experiments/W2/0069_w2c_hpcgpu06/0069_train/w2c/qwen3.5-9b/seed0/ckpt/final/model
  REPLAY=$REPO/experiments/W2/0076_w2c_replay_hpcgpu07/0076_replay_data
  WAIT_DONE=${WAIT_DONE:-"eval_0076.status:done 0076 [train]"}
  export EV_PAUSE=$LOGS/train.pause.$NODE EV_STATUS_GLOB="nonexistent_queue.status" EV_NO_GIT=1
  export EV_DF_PATH=$VI
  CLAIM_MAX_S=${CLAIM_MAX_S:-3600}
  WAIT_MAX_S=${WAIT_MAX_S:-172800}
  EV_CAP_S=$(awk -v h="${EV_CAP_H:?任务书的上限（小时）}" 'BEGIN{print int(h * 3600)}')
fi
OUT=$REPO/experiments/X/${RUN_ID}_x2_$NODE
X2DATA=$OUT/0080_x2_data
TRAIN=$OUT/0080_train
TRAIN1500=$OUT/0080_train_1500
EVAL=$OUT/0080_eval
# shellcheck disable=SC1091
source "$WT/scripts/server/eval_lib.sh"
cd "$WT" || exit 1
[ "$TINY" = 1 ] || [ "${CUDA_VISIBLE_DEVICES:-}" = 0 ] || { progress "blocked: 不在 srun 的资源范围里"; exit 1; }
if [ "$TINY" != 1 ] && ! "$PY" -c "import json,sys; sys.exit(0 if json.load(open('configs/w2.json'))['frozen'] is True else 1)"; then
  progress "blocked: configs/w2.json 没有冻结"; exit 1
fi
has() { case " $STAGES " in *" $1 "*) return 0;; esac; return 1; }
die() { echo "$(now) ABORT: $*" | tee -a "$OUT/steps.log"; progress "blocked: $*"; push_out "X-2 aborted: $*"; exit 1; }
leaf_of() { echo "$1/w2c/$M/seed0"; }
median_step_s() {  # 训练叶目录 logs/timing_run_*.jsonl 里第 5 步以后 step_s 的中位数
  "$PY" - "$1" <<'PYEOF'
import glob, json, statistics, sys
xs = []
for f in sorted(glob.glob(sys.argv[1] + "/logs/timing_run_*.jsonl")):
    for line in open(f, encoding="utf-8"):
        r = json.loads(line)
        if int(r.get("step", 0)) >= 5 and r.get("step_s") is not None:
            xs.append(float(r["step_s"]))
print(f"{statistics.median(xs):.3f}" if xs else "nan")
PYEOF
}
NOK=0
mkdir -p "$OUT"
echo "$(now) start commit $(git -C "$WT" rev-parse HEAD 2>/dev/null || echo unknown) node $NODE model $M stages [$STAGES] x2 cfg $(sha "$WT/configs/x2_train.json" | cut -d' ' -f1)" >> "$OUT/steps.log"

# 1. 数据（CPU）
if has data; then
  if [ "$TINY" = 1 ]; then
    step data_fakes "$PY" -m tests._x2_fakes "$W2C_DATA" || die "造假数据失败"
    step x2_data "$PY" scripts/build_x2_data.py --tiny --w2c-data "$W2C_DATA" --out "$X2DATA" --workers 1 || die "X-2 数据失败"
  else
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 step x2_data "$PY" scripts/build_x2_data.py --w2c-data "$W2C_DATA" \
      "${TD[@]}" --out "$X2DATA" --workers 12 $( [ -f "$X2DATA/manifest.json" ] && echo --resume ) || die "X-2 数据失败"
  fi
fi
# 2. 排队：前一个任务的心跳以给定前缀开头
if has train || has eval; then
  t0=$(date +%s)
  if [ -n "$WAIT_DONE" ]; then
    wf=${WAIT_DONE%%:*}; wp=${WAIT_DONE#*:}
    while :; do
      st=$(cut -d' ' -f2- "$LOGS/$wf" 2>/dev/null || true)
      case "$st" in "$wp"*) break;; blocked*) die "前一个任务 $wf 是 blocked：$st";; esac
      progress "waiting for $wf to start with '$wp' (now: ${st:-missing})"
      [ $(( $(date +%s) - t0 )) -ge "${WAIT_MAX_S:-172800}" ] && die "等 $wf 超时"
      sleep 60
    done
  fi
  claim_gpu "0080-hpc-07-x2.md" || { progress "blocked: gpu not free after ${CLAIM_MAX_S}s"; exit 1; }
fi
MEMPID=""
if [ "$TINY" != 1 ] && command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -lms 1000 >> "$OUT/gpu_mem.csv" 2>/dev/null &
  MEMPID=$!
fi
# 3. 训练：50 步探测 → 2,000 步续跑或 1,500 步另起
CK=""
if has train; then
  if [ "$TINY" = 1 ]; then
    CK=$X2_TINY_CK; echo "$(now) tiny：跳过训练，用 $CK" >> "$OUT/steps.log"
  else
    TR=(--config configs/x2_train.json --model $M --w2c-data "$W2C_DATA" --x2-data "$X2DATA" --replay-data "$REPLAY" "${TD[@]}")
    L=$(leaf_of "$TRAIN"); L15=$(leaf_of "$TRAIN1500")
    if [ -f "$L15/ckpt/final/model/manifest.json" ]; then CK=$L15/ckpt/final/model
    elif [ -f "$L/ckpt/final/model/manifest.json" ]; then CK=$L/ckpt/final/model
    elif [ -d "$L15" ]; then  # 1,500 步那一支已经开了：接着跑
      step train_1500 "$PY" scripts/train_w2c.py "${TR[@]}" --steps 1500 --out "$TRAIN1500" $( [ -d "$L15/ckpt/latest" ] && echo --resume ) || die "训练（1,500 步）失败"
      CK=$L15/ckpt/final/model
    else
      if [ ! -d "$L/ckpt/latest" ]; then
        step train_probe "$PY" scripts/train_w2c.py "${TR[@]}" --out "$TRAIN" --stop-after-steps 50 || die "训练探测失败"
      fi
      MS=$(median_step_s "$L")
      echo "$(now) probe median step_s (step >= 5) = $MS" | tee -a "$OUT/steps.log"
      if awk -v m="$MS" 'BEGIN{exit !(m != "nan" && m + 0 <= 21.6)}'; then
        step train "$PY" scripts/train_w2c.py "${TR[@]}" --out "$TRAIN" --resume || die "训练失败"
        CK=$L/ckpt/final/model
      else
        echo "$(now) median $MS > 21.6 s：改为 1,500 步另起（D136-7）" | tee -a "$OUT/steps.log"
        step train_1500 "$PY" scripts/train_w2c.py "${TR[@]}" --steps 1500 --out "$TRAIN1500" || die "训练（1,500 步）失败"
        CK=$L15/ckpt/final/model
      fi
    fi
  fi
  [ -f "$CK/manifest.json" ] || die "训练后没有检查点 $CK"
  echo "$(now) checkpoint $CK manifest $(sha "$CK/manifest.json" | cut -d' ' -f1)" >> "$OUT/steps.log"
fi
# 4. 评测
if has eval; then
  if [ -z "$CK" ]; then
    if [ "$TINY" = 1 ]; then CK=$X2_TINY_CK
    elif [ -f "$(leaf_of "$TRAIN1500")/ckpt/final/model/manifest.json" ]; then CK=$(leaf_of "$TRAIN1500")/ckpt/final/model
    else CK=$(leaf_of "$TRAIN")/ckpt/final/model; fi
  fi
  [ -f "$CK/manifest.json" ] || die "没有检查点 $CK"
  X=(--model $M ${T[@]+"${T[@]}"} ${D[@]+"${D[@]}"} --checkpoint "$CK" --candidate-workers "$CW" --tool-workers "$TW" ${LIMIT[@]+"${LIMIT[@]}"})
  for SP in "${EVSPLITS[@]}"; do
    step "x2_$SP" "$PY" scripts/run_x2.py "${X[@]}" --splits-file configs/explore_splits.json --split "$SP" --out "$EVAL" || NOK=$((NOK + 1))
  done
  for SP in "${W2SPLITS[@]}"; do
    step "x2_$SP" "$PY" scripts/run_x2.py "${X[@]}" --split "$SP" --out "$EVAL" || NOK=$((NOK + 1))
  done
  # 参照：W2-c 在 explore_val 与 devtune800 上（devcheck2 上已有 RUN-0069 的读数，但在 06；这里同节点再跑一次）
  W=(--model $M ${T[@]+"${T[@]}"} ${D[@]+"${D[@]}"} --w2-config configs/w2.json --out "$EVAL/w2c_ref" --candidate-workers "$CW" ${LIMIT[@]+"${LIMIT[@]}"})
  for SP in "${EVSPLITS[@]}"; do
    step "w2c_ref_$SP" "$PY" scripts/run_w2.py "${W[@]}" --splits-file configs/explore_splits.json --split "$SP" --task refseg --setting w2c \
      --checkpoint "$W2C_CK" || NOK=$((NOK + 1))
  done
  for SP in "${W2SPLITS[@]}"; do
    step "w2c_ref_$SP" "$PY" scripts/run_w2.py "${W[@]}" --split "$SP" --task refseg --setting w2c --checkpoint "$W2C_CK" || NOK=$((NOK + 1))
  done
  HC=(--model $M ${HT[@]+"${HT[@]}"})
  step health_base "$PY" scripts/health_check.py "${HC[@]}" --out "$EVAL/health/health_base.json" || NOK=$((NOK + 1))
  step health_x2 "$PY" scripts/health_check.py "${HC[@]}" --checkpoint "$CK" ${D[@]+"${D[@]}"} --out "$EVAL/health/health_x2.json" || NOK=$((NOK + 1))
  step health_compare "$PY" scripts/health_check.py --compare "$EVAL/health/health_base.json" "$EVAL/health/health_x2.json" \
    --out "$EVAL/health/health_compare.json" || NOK=$((NOK + 1))
fi
[ -n "$MEMPID" ] && kill "$MEMPID" 2>/dev/null
echo "$(now) done; failed steps: $NOK" >> "$OUT/steps.log"
progress "done $RUN_ID [$STAGES] (failed steps: $NOK)"
push_out "X-2 ${RUN_ID} [$STAGES] done (failed steps: $NOK)"
