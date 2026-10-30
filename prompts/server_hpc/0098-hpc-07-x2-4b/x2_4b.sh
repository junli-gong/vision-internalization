#!/usr/bin/env bash
# RUN-0098（ITSC HPC hpcgpu07）：补充实验，4B 上的 X-2（预注册 X-7；D154-2(6)）。配方同 C1（configs/x2_train.json，2,000 步，seed 0），
#   四份数据都用 4B 重做（训练器要求数据由同一个模型生成，D150）。只在开发数据上评，不碰 test 与 devcheck3。
#   阶段（STAGES，缺省全跑；每步可续跑，重启时同一条命令再跑一遍）：
#     propose    build_w2c_data --stage propose --model qwen3.5-4b（GPU；同 RUN-0067 的参数）
#     candidates build_w2c_data --stage candidates（CPU；同 RUN-0069）
#     replay     build_w2c_replay --model qwen3.5-4b（GPU；同 RUN-0076）
#     x2data     build_x2_data（CPU；同 RUN-0080）
#     train      train_w2c --config configs/x2_train.json --model qwen3.5-4b --seed 0（同 RUN-0093 的写法）
#     eval       run_x2：explore_val 三层、devtune800、devcheck2（同 RUN-0093）
#     health     4B 基座与 4B X-2 的健康检查，同节点配对
# 铁律：任何情况下都不能取消或停止 Slurm 任务 157296、157297。只用本节点自己的暂停文件。
# 用法：srun --jobid=157297 --overlap -n1 -c 12 env PATH=… TINY=0 EV_CAP_H=… [STAGES=…] bash …/x2_4b.sh < /dev/null
# 彩排（开发机）：TINY=1 WT=… REPO=… LOGS=… PY=… X2DATA=… REPLAY=… W2C_DATA=… bash x2_4b.sh
#   tiny 只把四个数据阶段的命令写进 cmd.txt 核对，用给的小数据真跑 train、eval、health。
set -uo pipefail
TINY=${TINY:-0}
RUN_ID=0098
STAGES=${STAGES:-"propose candidates replay x2data train eval health"}
M=qwen3.5-4b
EVSPLITS=(explore_val_refcoco_unc_val "explore_val_refcoco+_unc_val" explore_val_refcocog_umd_val)
W2SPLITS=(devtune800 devcheck2)
if [ "$TINY" = 1 ]; then
  : "${WT:?}" "${REPO:?}" "${LOGS:?}" "${PY:?}" "${X2DATA:?}" "${REPLAY:?}" "${W2C_DATA:?}"
  NODE=tiny; M=qwen3.5-0.8b
  T=(--tiny); D=(); TD=(); DD=(); RAW=(); CW=1; TW=1; HT=(--tiny --skip-ifeval-scoring --limit 2); LIMIT=(--limit 3)
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
  T=(); D=(--data-root "$VI/b200/data"); TD=(--data-root "$VI/data/train_data")
  DD=(--train-data-root "$VI/data/train_data" --eval-data-root "$VI/b200/data")
  RAW=(--raw-dir "$VI/data/raw/w2c_replay"); CW=8; TW=8; HT=(); LIMIT=()
  export EV_PAUSE=$LOGS/train.pause.$NODE EV_STATUS_GLOB="nonexistent_queue.status" EV_NO_GIT=1
  export EV_DF_PATH=$VI
  CLAIM_MAX_S=${CLAIM_MAX_S:-3600}
  EV_CAP_S=$(awk -v h="${EV_CAP_H:?任务书的上限（小时）}" 'BEGIN{print int(h * 3600)}')
fi
OUT=$REPO/experiments/X/${RUN_ID}_x2_4b_$NODE
DATA_RUN=$OUT/0098_w2c_data            # build_w2c_data 的 --out（里面是 propose/ 与 w2c_data/）
[ "$TINY" = 1 ] || W2C_DATA=$DATA_RUN/w2c_data
[ "$TINY" = 1 ] || REPLAY=$OUT/0098_replay_data
[ "$TINY" = 1 ] || X2DATA=$OUT/0098_x2_data
TRAIN=$OUT/0098_train
EVAL=$OUT/0098_eval
LEAF=$TRAIN/w2c/$M/seed0
CK=$LEAF/ckpt/final/model
# --out 的目录名必须以运行编号开头（CLAUDE.md"服务器驱动的写法"）：每个 --out 都查
for o in "$OUT" "$DATA_RUN" "$OUT/0098_replay_data" "$OUT/0098_x2_data" "$TRAIN" "$EVAL"; do
  case "$(basename "$o")" in [0-9][0-9][0-9][0-9]_*) ;; *) echo "目录名没有运行编号：$o" >&2; exit 2;; esac
done
has() { case " $STAGES " in *" $1 "*) return 0;; esac; return 1; }
if [ "$TINY" = 1 ]; then  # 四个数据阶段只核对命令行（正式参数同 RUN-0067、0069、0076、0080）
  mkdir -p "$OUT"
  { echo "$PY scripts/build_w2c_data.py --stage propose --model qwen3.5-4b --train-data-root \$VI/data/train_data --eval-data-root \$VI/b200/data --out $DATA_RUN"
    echo "$PY scripts/build_w2c_data.py --stage candidates --model qwen3.5-4b --train-data-root \$VI/data/train_data --eval-data-root \$VI/b200/data --out $DATA_RUN --candidate-workers 8"
    echo "$PY scripts/build_w2c_replay.py --model qwen3.5-4b --raw-dir \$VI/data/raw/w2c_replay --w2c-data $DATA_RUN/w2c_data --train-data-root \$VI/data/train_data --eval-data-root \$VI/b200/data --out $OUT/0098_replay_data"
    echo "$PY scripts/build_x2_data.py --w2c-data $DATA_RUN/w2c_data --data-root \$VI/data/train_data --out $OUT/0098_x2_data --workers 12"; } > "$OUT/cmd.txt"
fi
# shellcheck disable=SC1091
source "$WT/scripts/server/eval_lib.sh"
cd "$WT" || exit 1
[ "$TINY" = 1 ] || [ "${CUDA_VISIBLE_DEVICES:-}" = 0 ] || { progress "blocked: 不在 srun 的资源范围里"; exit 1; }
if [ "$TINY" != 1 ] && ! "$PY" -c "import json,sys; sys.exit(0 if json.load(open('configs/w2.json'))['frozen'] is True else 1)"; then
  progress "blocked: configs/w2.json 没有冻结"; exit 1
fi
die() { echo "$(now) ABORT: $*" | tee -a "$OUT/steps.log"; progress "blocked: $*"; push_out "X-2 4B aborted: $*"; exit 1; }
NOK=0
claim_gpu "0098-hpc-07-x2-4b.md" || { progress "blocked: gpu not free after ${CLAIM_MAX_S}s"; exit 1; }
mkdir -p "$OUT"
echo "$(now) start commit $(git -C "$WT" rev-parse HEAD 2>/dev/null || echo unknown) node $NODE model $M stages [$STAGES] x2 cfg $(sha "$WT/configs/x2_train.json" | cut -d' ' -f1)" >> "$OUT/steps.log"
MEMPID=""
if [ "$TINY" != 1 ] && command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -lms 1000 >> "$OUT/gpu_mem.csv" 2>/dev/null &
  MEMPID=$!
fi
if [ "$TINY" != 1 ]; then
  has propose && { step propose "$PY" scripts/build_w2c_data.py --stage propose --model $M "${DD[@]}" --out "$DATA_RUN" || die "出框失败"; }
  has candidates && { OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 step candidates "$PY" scripts/build_w2c_data.py --stage candidates \
      --model $M "${DD[@]}" --out "$DATA_RUN" --candidate-workers "$CW" || die "候选与标签失败"; }
  has replay && { step replay "$PY" scripts/build_w2c_replay.py --model $M "${RAW[@]}" --w2c-data "$W2C_DATA" "${DD[@]}" \
      --out "$REPLAY" || die "回放数据失败"; }
  has x2data && { OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 step x2data "$PY" scripts/build_x2_data.py --w2c-data "$W2C_DATA" \
      "${TD[@]}" --out "$X2DATA" --workers 12 $( [ -f "$X2DATA/manifest.json" ] && echo --resume ) || die "X-2 数据失败"; }
fi
if has train && [ ! -f "$CK/manifest.json" ]; then
  RES=(); [ -d "$LEAF/ckpt/latest" ] && RES=(--resume)
  step train "$PY" scripts/train_w2c.py --config configs/x2_train.json --seed 0 --model $M ${T[@]+"${T[@]}"} --w2c-data "$W2C_DATA" \
    --x2-data "$X2DATA" --replay-data "$REPLAY" ${TD[@]+"${TD[@]}"} --out "$TRAIN" ${RES[@]+"${RES[@]}"} || die "训练失败"
fi
if has eval || has health; then
  [ -f "$CK/manifest.json" ] || die "没有检查点 $CK"
  echo "$(now) checkpoint $CK manifest $(sha "$CK/manifest.json" | cut -d' ' -f1)" >> "$OUT/steps.log"
fi
if has eval; then
  X=(--model $M ${T[@]+"${T[@]}"} ${D[@]+"${D[@]}"} --checkpoint "$CK" --candidate-workers "$CW" --tool-workers "$TW" ${LIMIT[@]+"${LIMIT[@]}"})
  for SP in "${EVSPLITS[@]}"; do
    step "x2_4b_$SP" "$PY" scripts/run_x2.py "${X[@]}" --splits-file configs/explore_splits.json --split "$SP" --out "$EVAL" || NOK=$((NOK + 1))
  done
  for SP in "${W2SPLITS[@]}"; do
    step "x2_4b_$SP" "$PY" scripts/run_x2.py "${X[@]}" --split "$SP" --out "$EVAL" || NOK=$((NOK + 1))
  done
fi
if has health; then
  HC=(--model $M ${HT[@]+"${HT[@]}"})
  # 补充 1（D161）：基座不带 --data-root（health_check.py 只许它配 --checkpoint；同 RUN-0080 的 health_base）；已有的读数不重跑
  [ -f "$EVAL/health/health_base_4b.json" ] || \
    step health_base_4b "$PY" scripts/health_check.py "${HC[@]}" --out "$EVAL/health/health_base_4b.json" || NOK=$((NOK + 1))
  [ -f "$EVAL/health/health_x2_4b.json" ] || \
    step health_x2_4b "$PY" scripts/health_check.py "${HC[@]}" --checkpoint "$CK" ${D[@]+"${D[@]}"} --out "$EVAL/health/health_x2_4b.json" || NOK=$((NOK + 1))
  step health_compare "$PY" scripts/health_check.py --compare "$EVAL/health/health_base_4b.json" "$EVAL/health/health_x2_4b.json" \
    --out "$EVAL/health/health_compare.json" || NOK=$((NOK + 1))
fi
[ -n "$MEMPID" ] && kill "$MEMPID" 2>/dev/null
echo "$(now) done; failed steps: $NOK" >> "$OUT/steps.log"
progress "done $RUN_ID (failed steps: $NOK)"
push_out "X-2 4B ${RUN_ID} done (failed steps: $NOK)"
