#!/usr/bin/env bash
# RUN-0069（ITSC HPC hpcgpu06）：W2-c 的数据（候选与标签）→ 训练 → devcheck2 判定 → 语言健康检查与 T 文本读数（W2 预注册 §9；D118、D119）。
#   1. data：build_w2c_data.py --stage candidates，接着 RUN-0067 的出框写在同一个数据目录里（要求 configs/w2.json 已冻结）。
#   2. train：train_w2c.py，9B，2,000 步（configs/w2c_train.json；w2_config.sha256 已由管理员填）。
#   3. gate：同一节点上 base 9B 的 E1 框与 w1f，再跑 w2c（--checkpoint），w2_gate.py --main-name W2-c（与同节点的 w1f 配对，D93-1）。
#   4. health：健康检查（基座与 W2-c，同节点）与 --compare；定位、计数的 T 文本读数（基座与 W2-c，同节点）。只报告。
# STAGES（默认 "data train gate health"）可以只跑其中几步；每步可续跑（各脚本的 --resume / 已完成即跳过）。
# 铁律：任何情况下都不能取消或停止 Slurm 任务 157296、157297。只用本节点自己的暂停文件。
# 用法：srun --jobid=157296 --overlap -n1 -c 8 env PATH=… EV_CAP_H=… [STAGES=…] bash …/w2c.sh < /dev/null
# 彩排（开发机）：TINY=1 WT=… REPO=… LOGS=… PY=… bash w2c.sh（数据用 tests._w2c_fakes 造，跳过 T 文本读数：run_e1 的 --checkpoint 只用于正式运行）
RUN_ID=0069
NODE=hpcgpu06
STAGES=${STAGES:-"data train gate health"}
M=qwen3.5-9b
if [ "${TINY:-0}" = 1 ]; then
  : "${WT:?}" "${REPO:?}" "${LOGS:?}" "${PY:?}"
  M=qwen3.5-0.8b
  T=(--tiny); D=(); TD=(); CW=1
  export EV_NO_GIT=1
else
  VI=/home/mail/chenyuma/visint
  WT=$VI/wt/wt0069
  REPO=$VI/repo
  LOGS=$VI/logs
  PY=$VI/env/bin/python
  export HF_HOME=$VI/hf HF_HUB_CACHE=$VI/hf/hub HF_HUB_OFFLINE=1
  export TMPDIR=$VI/tmp PIP_NO_CACHE_DIR=1 NLTK_DATA=$VI/b200/nltk_data
  export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONHASHSEED=0
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export TRITON_CACHE_DIR=$VI/cache/$NODE/triton TORCHINDUCTOR_CACHE_DIR=$VI/cache/$NODE/inductor XDG_CACHE_HOME=$VI/cache/$NODE/xdg
  T=(); D=(--data-root "$VI/b200/data"); TD=(--data-root "$VI/data/train_data"); CW=${CW:-8}
  export EV_PAUSE=$LOGS/train.pause.$NODE EV_STATUS_GLOB="train_0056-$NODE.status" EV_NO_GIT=1
  export EV_DF_PATH=$VI
  EV_CAP_S=$(awk -v h="${EV_CAP_H:?任务书的上限（小时）}" 'BEGIN{print int(h * 3600)}')
fi
OUT=$REPO/experiments/W2/0069_w2c_hpcgpu06
DATA_RUN=$REPO/experiments/W2/0067_w2c_data_hpcgpu06
TRAIN=$OUT/0069_train
EVAL=$OUT/0069_eval
TTEXT=$OUT/0069_ttext
LEAF=$TRAIN/w2c/$M/seed0
CK=$LEAF/ckpt/final/model
# shellcheck disable=SC1091
source "$WT/scripts/server/eval_lib.sh"
cd "$WT" || exit 1
[ "$TINY" = 1 ] || [ "${CUDA_VISIBLE_DEVICES:-}" = 0 ] || { progress "blocked: 不在 srun 的资源范围里"; exit 1; }
if [ "$TINY" != 1 ] && ! "$PY" -c "import json,sys; sys.exit(0 if json.load(open('configs/w2.json'))['frozen'] is True else 1)"; then
  progress "blocked: configs/w2.json 没有冻结"; exit 1
fi
has() { case " $STAGES " in *" $1 "*) return 0;; esac; return 1; }
die() { echo "$(now) ABORT: $*" | tee -a "$OUT/steps.log"; progress "blocked: $*"; push_out "W2-c aborted: $*"; exit 1; }
NOK=0
claim_gpu "0069-hpc-06-w2c.md" || { progress "blocked: gpu not free after ${CLAIM_MAX_S}s"; exit 1; }
echo "$(now) start commit $(git -C "$WT" rev-parse HEAD 2>/dev/null || echo unknown) node $NODE model $M stages [$STAGES] w2.json $(sha "$WT/configs/w2.json" | cut -d' ' -f1)" >> "$OUT/steps.log"
MEMPID=""
if [ "$TINY" != 1 ] && command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -lms 1000 >> "$OUT/gpu_mem.csv" 2>/dev/null &
  MEMPID=$!
fi

# 1. 数据：候选与标签
if has data; then
  if [ "$TINY" = 1 ]; then
    step data "$PY" -m tests._w2c_fakes "$DATA_RUN/w2c_data" || die "造假数据失败"
  else
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 step data "$PY" scripts/build_w2c_data.py --stage candidates --model $M \
      --train-data-root "$VI/data/train_data" --eval-data-root "$VI/b200/data" --out "$DATA_RUN" --candidate-workers "$CW" || die "候选与标签失败"
  fi
fi
# 2. 训练（中途失败可以加 --resume 接着跑：同一条命令再执行一次，脚本在叶目录已有 latest 时自动带 --resume）
if has train; then
  RES=(); [ -d "$LEAF/ckpt/latest" ] && RES=(--resume)
  if [ "$TINY" = 1 ]; then
    step train "$PY" scripts/train_w2c.py --tiny --model $M --w2c-data "$DATA_RUN/w2c_data" --out "$TRAIN" ${RES[@]+"${RES[@]}"} || die "训练失败"
  else
    step train "$PY" scripts/train_w2c.py --model $M --w2c-data "$DATA_RUN/w2c_data" ${TD[@]+"${TD[@]}"} --out "$TRAIN" ${RES[@]+"${RES[@]}"} || die "训练失败"
  fi
  [ -f "$CK/manifest.json" ] || die "训练后没有检查点 $CK"
fi
# 3. 判定：同节点的 base 框与 w1f，再 w2c，再 gate
if has gate; then
  W2=(--model $M ${T[@]+"${T[@]}"} ${D[@]+"${D[@]}"} --split devcheck2 --w2-config configs/w2.json --out "$EVAL" --candidate-workers "$CW")
  step e1_box "$PY" scripts/run_w2.py "${W2[@]}" --task refexp_box --setting baseline_e1 || NOK=$((NOK + 1))
  step w1f "$PY" scripts/run_w2.py "${W2[@]}" --task refseg --setting w1f || NOK=$((NOK + 1))
  step w2c "$PY" scripts/run_w2.py "${W2[@]}" --task refseg --setting w2c --checkpoint "$CK" || NOK=$((NOK + 1))
  G=$EVAL/$M/refseg
  step gate "$PY" scripts/w2_gate.py --main-name W2-c --main "$G/w2c/devcheck2" --w1f "$G/w1f/devcheck2" --out-dir "$OUT/gate_w2c" || NOK=$((NOK + 1))
fi
# 4. 语言：健康检查与 T 文本读数（只报告）
if has health; then
  HC=(--model $M ${T[@]+"${T[@]}"}); [ "$TINY" = 1 ] && HC+=(--skip-ifeval-scoring --limit 2)  # 开发机没有 NLTK 的 punkt_tab；服务器上 NLTK_DATA 已设
  step health_base "$PY" scripts/health_check.py "${HC[@]}" --out "$OUT/health/health_base.json" || NOK=$((NOK + 1))
  step health_w2c "$PY" scripts/health_check.py "${HC[@]}" --checkpoint "$CK" ${D[@]+"${D[@]}"} --out "$OUT/health/health_w2c.json" || NOK=$((NOK + 1))
  step health_compare "$PY" scripts/health_check.py --compare "$OUT/health/health_base.json" "$OUT/health/health_w2c.json" \
    --out "$OUT/health/health_compare.json" || NOK=$((NOK + 1))
  if [ "$TINY" != 1 ]; then
    step ttext_base "$PY" scripts/run_e1.py --model $M --task refexp_box count --subset eval ${D[@]+"${D[@]}"} --out "$TTEXT" || NOK=$((NOK + 1))
    step ttext_w2c "$PY" scripts/run_e1.py --model $M --task refexp_box count --subset eval ${D[@]+"${D[@]}"} \
      --checkpoint "$CK" --checkpoint-label t_text/w2c/seed0 --out "$TTEXT" || NOK=$((NOK + 1))
  fi
fi
[ -n "$MEMPID" ] && kill "$MEMPID" 2>/dev/null
echo "$(now) done; failed steps: $NOK" >> "$OUT/steps.log"
progress "done $RUN_ID (failed steps: $NOK)"
push_out "W2-c done (failed steps: $NOK)"
