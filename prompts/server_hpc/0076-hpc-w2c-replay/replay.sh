#!/usr/bin/env bash
# RUN-0076（ITSC HPC）：N44 (b)，W2-c 加自蒸馏回放（W2 预注册 §12、§12a；D125-2 (b)、D132）。事后描述，不判定。
#   07（STAGES="data train"，NODE=hpcgpu07）：
#     1. data：build_w2c_replay.py，基座 9B 对 2,000 条通用指令与 2,000 条计数提示的回答（贪心、关思考）；
#     2. train：train_w2c.py --config configs/w2c_replay_train.json --replay-data …，9B，2,000 步，每步 8 + 8 + 3 + 2。
#   06（STAGES="eval"，NODE=hpcgpu06）：与 06 上已有的 W2-c 与基座读数同节点配对（D93-1）：
#     3. test：5 个 test 划分，run_w2 --setting w2c --checkpoint（回放变体的检查点），新的 --out；
#     4. 健康检查（与 RUN-0069 的 health_base 配对）、T 文本读数（与 RUN-0069 的 ttext 基座配对）、大题量语言读数（与 RUN-0074 的 lb_base 配对）。
# 每步可续跑（各脚本已完成即跳过；训练在 latest 存在时自动 --resume）。
# 铁律：任何情况下都不能取消或停止 Slurm 任务 157296、157297。只用本节点自己的暂停文件。
# 用法：srun --jobid=… --overlap -n1 -c 8 env PATH=… NODE=… STAGES=… WAIT_GLOB=… EV_CAP_H=… bash …/replay.sh < /dev/null
# 彩排（开发机）：TINY=1 WT=… REPO=… LOGS=… PY=… [STAGES=…] bash replay.sh
#   tiny 的回放数据由随机模型生成，回答不以停止 token 结尾，训练会按规则拒收（见 tests/test_w2c_replay.py 的做法）；
#   所以 tiny 的 train 用默认配置（不带回放）训出一个小检查点，只为让 eval 这一段走通。回放训练本身由 tests 覆盖。
set -uo pipefail
RUN_ID=0076
STAGES=${STAGES:?"data train"（07）或 "eval"（06）}
M=qwen3.5-9b
SPLITS=(test_refcoco_unc_testA test_refcoco_unc_testB "test_refcoco+_unc_testA" "test_refcoco+_unc_testB" test_refcocog_umd_test)
if [ "${TINY:-0}" = 1 ]; then
  : "${WT:?}" "${REPO:?}" "${LOGS:?}" "${PY:?}" "${TINY_LB_RAW:?lang_battery 的 fixtures 目录（python -m tests._lang_battery_fixtures <目录>）}"
  NODE=tiny; M=qwen3.5-0.8b
  T=(--tiny); D=(); TD=(); CW=1; RAW=(); SPLITS=(devtune800)
  HT=(--tiny --skip-ifeval-scoring --limit 2)
  export NLTK_DATA=$WT/tests/fixtures/health/nltk_data EV_NO_GIT=1
  DATA_RUN=$REPO/experiments/W2/0067_w2c_data_tiny
  BASE_HEALTH=$REPO/experiments/W2/0076_tiny_base/health_base.json
  BASE_LB=$REPO/experiments/W2/0076_tiny_base/lb_base
else
  : "${NODE:?hpcgpu07（data train）或 hpcgpu06（eval）}"
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
  T=(); D=(--data-root "$VI/b200/data"); TD=(--data-root "$VI/data/train_data"); CW=${CW:-8}
  RAW=(--raw-dir "$VI/data/raw/w2c_replay"); HT=()
  export EV_PAUSE=$LOGS/train.pause.$NODE EV_STATUS_GLOB="${WAIT_GLOB:?要等的心跳文件（07：eval_0073.status；06：eval_0074.status 等）}" EV_NO_GIT=1
  export EV_DF_PATH=$VI
  CLAIM_MAX_S=${CLAIM_MAX_S:-86400}
  EV_CAP_S=$(awk -v h="${EV_CAP_H:?任务书的上限（小时）}" 'BEGIN{print int(h * 3600)}')
  DATA_RUN=$REPO/experiments/W2/0067_w2c_data_hpcgpu06
  BASE_HEALTH=$REPO/experiments/W2/0069_w2c_hpcgpu06/health/health_base.json
  BASE_LB=$REPO/experiments/W2/0074_lang_battery_hpcgpu06/lb_base
fi
TRAIN_OUT=$REPO/experiments/W2/0076_w2c_replay_hpcgpu07
[ "$TINY" = 1 ] && TRAIN_OUT=$REPO/experiments/W2/0076_w2c_replay_tiny
REPLAY=$TRAIN_OUT/0076_replay_data
TRAIN=$TRAIN_OUT/0076_train
LEAF=$TRAIN/w2c/$M/seed0
CK=$LEAF/ckpt/final/model
EVAL_OUT=$REPO/experiments/W2/0076_w2c_replay_eval_hpcgpu06
[ "$TINY" = 1 ] && EVAL_OUT=$REPO/experiments/W2/0076_w2c_replay_eval_tiny
case " $STAGES " in *" eval "*) OUT=$EVAL_OUT;; *) OUT=$TRAIN_OUT;; esac
# shellcheck disable=SC1091
source "$WT/scripts/server/eval_lib.sh"
cd "$WT" || exit 1
[ "$TINY" = 1 ] || [ "${CUDA_VISIBLE_DEVICES:-}" = 0 ] || { progress "blocked: 不在 srun 的资源范围里"; exit 1; }
if [ "$TINY" != 1 ] && ! "$PY" -c "import json,sys; sys.exit(0 if json.load(open('configs/w2.json'))['frozen'] is True else 1)"; then
  progress "blocked: configs/w2.json 没有冻结"; exit 1
fi
has() { case " $STAGES " in *" $1 "*) return 0;; esac; return 1; }
die() { echo "$(now) ABORT: $*" | tee -a "$OUT/steps.log"; progress "blocked: $*"; push_out "W2-c replay aborted: $*"; exit 1; }
NOK=0
claim_gpu "0076-hpc-w2c-replay.md" || { progress "blocked: gpu not free after ${CLAIM_MAX_S}s"; exit 1; }
echo "$(now) start commit $(git -C "$WT" rev-parse HEAD 2>/dev/null || echo unknown) node $NODE model $M stages [$STAGES] replay cfg $(sha "$WT/configs/w2c_replay_train.json" | cut -d' ' -f1)" >> "$OUT/steps.log"
MEMPID=""
if [ "$TINY" != 1 ] && command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -lms 1000 >> "$OUT/gpu_mem.csv" 2>/dev/null &
  MEMPID=$!
fi

# 1. 回放数据
if has data; then
  if [ "$TINY" = 1 ]; then
    step data_fakes "$PY" -m tests._w2c_fakes "$DATA_RUN/w2c_data" || die "造假数据失败"
    step data "$PY" scripts/build_w2c_replay.py --tiny --w2c-data "$DATA_RUN/w2c_data" --out "$REPLAY" || die "回放数据失败"
  else
    step data "$PY" scripts/build_w2c_replay.py --model $M "${RAW[@]}" --w2c-data "$DATA_RUN/w2c_data" \
      --train-data-root "$VI/data/train_data" --eval-data-root "$VI/b200/data" --out "$REPLAY" || die "回放数据失败"
  fi
fi
# 2. 训练（叶目录已有 latest 时自动 --resume）
if has train; then
  RES=(); [ -d "$LEAF/ckpt/latest" ] && RES=(--resume)
  if [ "$TINY" = 1 ]; then  # 见文件头：tiny 只训一个不带回放的小检查点
    step train "$PY" scripts/train_w2c.py --tiny --model $M --w2c-data "$DATA_RUN/w2c_data" --out "$TRAIN" ${RES[@]+"${RES[@]}"} || die "训练失败"
  else
    step train "$PY" scripts/train_w2c.py --config configs/w2c_replay_train.json --model $M --w2c-data "$DATA_RUN/w2c_data" \
      --replay-data "$REPLAY" "${TD[@]}" --out "$TRAIN" ${RES[@]+"${RES[@]}"} || die "训练失败"
  fi
  [ -f "$CK/manifest.json" ] || die "训练后没有检查点 $CK"
fi
# 3–4. 评测（06）
if has eval; then
  [ -f "$CK/manifest.json" ] || die "没有检查点 $CK（先跑 07 的 data train）"
  echo "$(now) checkpoint $CK manifest $(sha "$CK/manifest.json" | cut -d' ' -f1)" >> "$OUT/steps.log"
  for SP in "${SPLITS[@]}"; do
    step "w2c_replay_$SP" "$PY" scripts/run_w2.py --model $M ${T[@]+"${T[@]}"} ${D[@]+"${D[@]}"} --split "$SP" --w2-config configs/w2.json \
      --out "$OUT/test" --candidate-workers "$CW" --task refseg --setting w2c --checkpoint "$CK" || NOK=$((NOK + 1))
  done
  HC=(--model $M ${HT[@]+"${HT[@]}"})
  if [ "$TINY" = 1 ]; then  # tiny 没有现成的基座读数：先在这里造
    step tiny_health_base "$PY" scripts/health_check.py "${HC[@]}" --out "$BASE_HEALTH" || NOK=$((NOK + 1))
    step tiny_lb_base "$PY" scripts/lang_battery.py --model $M --tiny --limit 3 --raw-dir "$TINY_LB_RAW" --label base --out "$BASE_LB" || NOK=$((NOK + 1))
  fi
  step health_w2c_replay "$PY" scripts/health_check.py "${HC[@]}" --checkpoint "$CK" ${D[@]+"${D[@]}"} \
    --out "$OUT/health/health_w2c_replay.json" || NOK=$((NOK + 1))
  step health_compare "$PY" scripts/health_check.py --compare "$BASE_HEALTH" "$OUT/health/health_w2c_replay.json" \
    --out "$OUT/health/health_compare.json" || NOK=$((NOK + 1))
  if [ "$TINY" != 1 ]; then
    step ttext_w2c_replay "$PY" scripts/run_e1.py --model $M --task refexp_box count --subset eval ${D[@]+"${D[@]}"} \
      --checkpoint "$CK" --checkpoint-label t_text/w2c_replay/seed0 --out "$OUT/ttext" || NOK=$((NOK + 1))
    LBR=(--raw-dir "$VI/data/lang_battery_raw")
  else
    LBR=(--tiny --limit 3 --raw-dir "$TINY_LB_RAW")
  fi
  step lb_w2c_replay "$PY" scripts/lang_battery.py --model $M "${LBR[@]}" --label w2c_replay ${D[@]+"${D[@]}"} --checkpoint "$CK" \
    --out "$OUT/lb_w2c_replay" || NOK=$((NOK + 1))
  step lb_compare "$PY" scripts/lang_battery.py --compare "$BASE_LB" "$OUT/lb_w2c_replay" --out "$OUT/lb_compare_w2c_replay.json" || NOK=$((NOK + 1))
fi
[ -n "$MEMPID" ] && kill "$MEMPID" 2>/dev/null
echo "$(now) done; failed steps: $NOK" >> "$OUT/steps.log"
progress "done $RUN_ID [$STAGES] (failed steps: $NOK)"
push_out "W2-c replay ${RUN_ID} [$STAGES] done (failed steps: $NOK)"
