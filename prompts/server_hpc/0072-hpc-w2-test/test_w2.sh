#!/usr/bin/env bash
# RUN-0072 / RUN-0073（ITSC HPC）：W2 的 test 主实验（W2 预注册 §11；D113 第三节"至少一个主配置通过"；冻结的配置 D122）。
#   5 个 test 划分（E1 的 poly 子集，各 1,000 句）× 一个模型。每个划分：
#     E1 的框 → 自己的框、W1-F →（9B 另有）随机挑选、最优候选、E1 零样本多边形 → W2-a →（9B 另有）W2-c（训练好的检查点）。
#   MODEL=qwen3.5-9b：全部设置（RUN-0072，06）；MODEL=qwen3.5-2b|qwen3.5-4b：W2-S，只跑自己的框、W1-F、W2-a（RUN-0073，07）。
#   test 只报告、不判定（去留已在 devcheck2 上定）。只用冻结的 configs/w2.json 与 RUN-0069 的 W2-c 检查点。
# 铁律：任何情况下都不能取消或停止 Slurm 任务 157296、157297。只用本节点自己的暂停文件。
# 用法：srun --jobid=… --overlap -n1 -c 8 env PATH=… NODE=… RUN_ID=0072|0073 MODELS="…" EV_CAP_H=… bash …/test_w2.sh < /dev/null
# 彩排（开发机）：TINY=1 WT=… REPO=… LOGS=… PY=… RUN_ID=0072 MODELS=qwen3.5-0.8b [W2C_CK=<tiny 检查点>] bash test_w2.sh
: "${RUN_ID:?0072 或 0073}" "${MODELS:?例如 qwen3.5-9b，或 \"qwen3.5-2b qwen3.5-4b\"}"
SPLITS=(test_refcoco_unc_testA test_refcoco_unc_testB "test_refcoco+_unc_testA" "test_refcoco+_unc_testB" test_refcocog_umd_test)
if [ "${TINY:-0}" = 1 ]; then
  : "${WT:?}" "${REPO:?}" "${LOGS:?}" "${PY:?}"
  NODE=${NODE:-tiny}
  T=(--tiny); D=(); CW=1
  export EV_NO_GIT=1
else
  : "${NODE:?hpcgpu06 或 hpcgpu07}"
  VI=/home/mail/chenyuma/visint
  WT=$VI/wt/wt$RUN_ID
  REPO=$VI/repo
  LOGS=$VI/logs
  PY=$VI/env/bin/python
  W2C_CK=${W2C_CK:-$REPO/experiments/W2/0069_w2c_hpcgpu06/0069_train/w2c/qwen3.5-9b/seed0/ckpt/final/model}
  export HF_HOME=$VI/hf HF_HUB_CACHE=$VI/hf/hub HF_HUB_OFFLINE=1
  export TMPDIR=$VI/tmp PIP_NO_CACHE_DIR=1 NLTK_DATA=$VI/b200/nltk_data
  export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONHASHSEED=0
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export TRITON_CACHE_DIR=$VI/cache/$NODE/triton TORCHINDUCTOR_CACHE_DIR=$VI/cache/$NODE/inductor XDG_CACHE_HOME=$VI/cache/$NODE/xdg
  T=(); D=(--data-root "$VI/b200/data"); CW=${CW:-8}
  export EV_PAUSE=$LOGS/train.pause.$NODE EV_STATUS_GLOB="nonexistent_queue.status" EV_NO_GIT=1
  export EV_DF_PATH=$VI
  EV_CAP_S=$(awk -v h="${EV_CAP_H:?任务书的上限（小时）}" 'BEGIN{print int(h * 3600)}')
fi
OUT=$REPO/experiments/W2/${RUN_ID}_w2_test_$NODE
# shellcheck disable=SC1091
source "$WT/scripts/server/eval_lib.sh"
cd "$WT" || exit 1
[ "$TINY" = 1 ] || [ "${CUDA_VISIBLE_DEVICES:-}" = 0 ] || { progress "blocked: 不在 srun 的资源范围里"; exit 1; }
if [ "$TINY" != 1 ] && ! "$PY" -c "import json,sys; sys.exit(0 if json.load(open('configs/w2.json'))['frozen'] is True else 1)"; then
  progress "blocked: configs/w2.json 没有冻结"; exit 1
fi
NOK=0
claim_gpu "${RUN_ID}-hpc-w2-test.md" || { progress "blocked: gpu not free after ${CLAIM_MAX_S}s"; exit 1; }
echo "$(now) start commit $(git -C "$WT" rev-parse HEAD 2>/dev/null || echo unknown) node $NODE models [$MODELS] w2.json $(sha "$WT/configs/w2.json" | cut -d' ' -f1)" >> "$OUT/steps.log"
for M in $MODELS; do
  full=0; [ "$M" = qwen3.5-9b ] && full=1; [ "$TINY" = 1 ] && full=1
  for SP in "${SPLITS[@]}"; do
    W2=(--model "$M" ${T[@]+"${T[@]}"} ${D[@]+"${D[@]}"} --split "$SP" --w2-config configs/w2.json --out "$OUT" --candidate-workers "$CW")
    tag="${M}_${SP}"
    step "e1_box_$tag" "$PY" scripts/run_w2.py "${W2[@]}" --task refexp_box --setting baseline_e1 || NOK=$((NOK + 1))
    for s in ownbox w1f; do step "${s}_$tag" "$PY" scripts/run_w2.py "${W2[@]}" --task refseg --setting $s || NOK=$((NOK + 1)); done
    if [ $full = 1 ]; then
      for s in random_pick oracle_pick baseline_e1; do step "${s}_$tag" "$PY" scripts/run_w2.py "${W2[@]}" --task refseg --setting $s || NOK=$((NOK + 1)); done
    fi
    step "w2a_$tag" "$PY" scripts/run_w2.py "${W2[@]}" --task refseg --setting w2a || NOK=$((NOK + 1))
    if [ $full = 1 ] && [ -n "${W2C_CK:-}" ]; then
      step "w2c_$tag" "$PY" scripts/run_w2.py "${W2[@]}" --task refseg --setting w2c --checkpoint "$W2C_CK" || NOK=$((NOK + 1))
    fi
  done
done
echo "$(now) done; failed steps: $NOK" >> "$OUT/steps.log"
progress "done $RUN_ID (failed steps: $NOK)"
push_out "W2 test ${RUN_ID} done (failed steps: $NOK)"
