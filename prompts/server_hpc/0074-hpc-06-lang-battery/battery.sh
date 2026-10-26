#!/usr/bin/env bash
# RUN-0074（ITSC HPC，hpcgpu06）：N44 (c) 与 D125-3 的语言代价读数（W2 预注册 §2 的主张修改、§12）。
#   9B 的基座、W2-c（RUN-0069）、E3b+C3（RUN-0056 种子 0）三个模型，同节点配对：
#   1. 大题量的语言读数（scripts/lang_battery.py，DEV-0064）：MMLU-Pro 字母作答 2,000 题、IFEval 541 条；--compare 基座对另两个；
#   2. 健康检查（简报 8.11）：基座重跑一次（与 RUN-0069 的 health_base 逐题核对可复现）与 E3b+C3（RUN-0056 没有做）；--compare。
#      W2-c 的健康检查已在 RUN-0069 做过，不重跑。
#   只报告、不判定。排在 RUN-0072 之后：claim_gpu 等 $LOGS/eval_0072.status 不再是 running，且 GPU 上没有进程。
# 铁律：任何情况下都不能取消或停止 Slurm 任务 157296、157297。只用本节点自己的暂停文件。
# 用法：srun --jobid=157296 --overlap -n1 -c 8 env PATH=… EV_CAP_H=… bash …/battery.sh < /dev/null
# 彩排（开发机）：TINY=1 WT=… REPO=… LOGS=… PY=… RAW=<fixtures 目录> bash battery.sh（只跑基座与基座对基座的 --compare）
set -uo pipefail
RUN_ID=0074
if [ "${TINY:-0}" = 1 ]; then
  : "${WT:?}" "${REPO:?}" "${LOGS:?}" "${PY:?}" "${RAW:?}"
  NODE=tiny
  M=qwen3.5-0.8b
  T=(--tiny --limit 3); HT=(--tiny --skip-ifeval-scoring --limit 2); D=()
  export NLTK_DATA=$WT/tests/fixtures/health/nltk_data EV_NO_GIT=1
else
  NODE=hpcgpu06
  VI=/home/mail/chenyuma/visint
  WT=$VI/wt/wt$RUN_ID
  REPO=$VI/repo
  LOGS=$VI/logs
  PY=$VI/env/bin/python
  RAW=$VI/data/lang_battery_raw
  M=qwen3.5-9b
  W2C_CK=$REPO/experiments/W2/0069_w2c_hpcgpu06/0069_train/w2c/qwen3.5-9b/seed0/ckpt/final/model
  E3B_CK=$VI/ckpt/hpcgpu06/0056/e3b_c3/qwen3.5-9b/seed0/ckpt/final/model
  W2C_MANIFEST=c605408fbb92cceafbc59dff4d85c9c912b246d05dc5e29235db0cff389d93b7
  E3B_MANIFEST=${E3B_MANIFEST:?任务书第 2 步核对后填入的 E3b+C3 manifest.json 的 sha256}
  export HF_HOME=$VI/hf HF_HUB_CACHE=$VI/hf/hub HF_HUB_OFFLINE=1
  export TMPDIR=$VI/tmp PIP_NO_CACHE_DIR=1 NLTK_DATA=$VI/b200/nltk_data
  export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONHASHSEED=0
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export TRITON_CACHE_DIR=$VI/cache/$NODE/triton TORCHINDUCTOR_CACHE_DIR=$VI/cache/$NODE/inductor XDG_CACHE_HOME=$VI/cache/$NODE/xdg
  T=(); HT=(); D=(--data-root "$VI/b200/data")
  export EV_PAUSE=$LOGS/train.pause.$NODE EV_STATUS_GLOB="eval_0072.status" EV_NO_GIT=1
  export EV_DF_PATH=$VI
  CLAIM_MAX_S=${CLAIM_MAX_S:-86400}  # RUN-0072 约 10 小时，10-26 02:33Z 开跑
  EV_CAP_S=$(awk -v h="${EV_CAP_H:?任务书的上限（小时）}" 'BEGIN{print int(h * 3600)}')
fi
OUT=$REPO/experiments/W2/${RUN_ID}_lang_battery_$NODE
# shellcheck disable=SC1091
source "$WT/scripts/server/eval_lib.sh"
cd "$WT" || exit 1
[ "$TINY" = 1 ] || [ "${CUDA_VISIBLE_DEVICES:-}" = 0 ] || { progress "blocked: 不在 srun 的资源范围里"; exit 1; }
if [ "$TINY" != 1 ]; then
  for pair in "$W2C_CK:$W2C_MANIFEST" "$E3B_CK:$E3B_MANIFEST"; do
    ck=${pair%%:*}; want=${pair##*:}
    have=$(sha "$ck/manifest.json" 2>/dev/null | cut -d' ' -f1)
    [ "$have" = "$want" ] || { progress "blocked: $ck/manifest.json sha256 ${have:-缺} ≠ $want"; exit 1; }
  done
  [ -d "$RAW" ] || { progress "blocked: 没有 $RAW（先在 hpcmgt 上 --fetch）"; exit 1; }
fi
NOK=0
claim_gpu "${RUN_ID}-hpc-06-lang-battery.md" || { progress "blocked: gpu not free after ${CLAIM_MAX_S}s"; exit 1; }
echo "$(now) start commit $(git -C "$WT" rev-parse HEAD 2>/dev/null || echo unknown) node $NODE model $M" >> "$OUT/steps.log"
LB=(--model "$M" ${T[@]+"${T[@]}"} --raw-dir "$RAW")
step lb_base "$PY" scripts/lang_battery.py "${LB[@]}" --label base --out "$OUT/lb_base" || NOK=$((NOK + 1))
if [ "$TINY" != 1 ]; then
  step lb_w2c "$PY" scripts/lang_battery.py "${LB[@]}" --label w2c "${D[@]}" --checkpoint "$W2C_CK" --out "$OUT/lb_w2c" || NOK=$((NOK + 1))
  step lb_e3b_c3 "$PY" scripts/lang_battery.py "${LB[@]}" --label e3b_c3 "${D[@]}" --checkpoint "$E3B_CK" --out "$OUT/lb_e3b_c3" || NOK=$((NOK + 1))
  step lb_compare_w2c "$PY" scripts/lang_battery.py --compare "$OUT/lb_base" "$OUT/lb_w2c" --out "$OUT/lb_compare_w2c.json" || NOK=$((NOK + 1))
  step lb_compare_e3b_c3 "$PY" scripts/lang_battery.py --compare "$OUT/lb_base" "$OUT/lb_e3b_c3" --out "$OUT/lb_compare_e3b_c3.json" || NOK=$((NOK + 1))
else
  step lb_compare_self "$PY" scripts/lang_battery.py --compare "$OUT/lb_base" "$OUT/lb_base" --out "$OUT/lb_compare_self.json" || NOK=$((NOK + 1))
fi
HC=(--model "$M" ${HT[@]+"${HT[@]}"})
step health_base "$PY" scripts/health_check.py "${HC[@]}" --out "$OUT/health/health_base.json" || NOK=$((NOK + 1))
if [ "$TINY" != 1 ]; then
  step health_e3b_c3 "$PY" scripts/health_check.py "${HC[@]}" --checkpoint "$E3B_CK" "${D[@]}" --out "$OUT/health/health_e3b_c3.json" || NOK=$((NOK + 1))
  step health_compare_e3b_c3 "$PY" scripts/health_check.py --compare "$OUT/health/health_base.json" "$OUT/health/health_e3b_c3.json" \
    --out "$OUT/health/health_compare_e3b_c3.json" || NOK=$((NOK + 1))
  # 基座的可复现核对：与 RUN-0069 在同一节点上的 health_base 配对（差值应为 0）
  step health_compare_base_0069 "$PY" scripts/health_check.py --compare "$REPO/experiments/W2/0069_w2c_hpcgpu06/health/health_base.json" \
    "$OUT/health/health_base.json" --out "$OUT/health/health_compare_base_0069.json" || NOK=$((NOK + 1))
fi
echo "$(now) done; failed steps: $NOK" >> "$OUT/steps.log"
progress "done $RUN_ID (failed steps: $NOK)"
push_out "lang battery ${RUN_ID} done (failed steps: $NOK)"
