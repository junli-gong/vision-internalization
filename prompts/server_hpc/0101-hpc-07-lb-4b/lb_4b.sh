#!/usr/bin/env bash
# RUN-0101（ITSC HPC hpcgpu07）：4B 基座与 4B X-2（RUN-0098 的检查点）各跑一次大题量（scripts/lang_battery.py：MMLU-Pro 2,000 题、
#   IFEval 541 条），同节点逐题配对 --compare（D162-2；同 9B 的 RUN-0084 口径）。替换 X-7 里"健康检查标红、不下结论"的写法。
#   参数规则（D161 的教训，逐条对过 lang_battery.py 的 p.error）：--data-root 只配 --checkpoint；--compare 不带模型参数。
# 铁律：任何情况下都不能取消或停止 Slurm 任务 157296、157297。只用本节点自己的暂停文件。
# 用法：srun --jobid=157297 --overlap -n1 -c 8 env PATH=… TINY=0 EV_CAP_H=… bash …/lb_4b.sh < /dev/null
# 彩排（开发机）：TINY=1 WT=… REPO=… LOGS=… PY=… RAW=… CK=<tiny 检查点> bash lb_4b.sh
set -uo pipefail
TINY=${TINY:-0}
RUN_ID=0101
if [ "$TINY" = 1 ]; then
  : "${WT:?}" "${REPO:?}" "${LOGS:?}" "${PY:?}" "${RAW:?}" "${CK:?}"
  NODE=tiny; M=qwen3.5-0.8b; T=(--tiny --limit 3); D=()
  export NLTK_DATA=$WT/tests/fixtures/health/nltk_data EV_NO_GIT=1
else
  NODE=hpcgpu07; M=qwen3.5-4b
  VI=/home/mail/chenyuma/visint
  WT=$VI/wt/wt$RUN_ID
  REPO=$VI/repo
  LOGS=$VI/logs
  PY=$VI/env/bin/python
  RAW=$VI/data/lang_battery_raw
  CK=$REPO/experiments/X/0098_x2_4b_hpcgpu07/0098_train/w2c/qwen3.5-4b/seed0/ckpt/final/model
  export HF_HOME=$VI/hf HF_HUB_CACHE=$VI/hf/hub HF_HUB_OFFLINE=1
  export TMPDIR=$VI/tmp PIP_NO_CACHE_DIR=1 NLTK_DATA=$VI/b200/nltk_data MPLCONFIGDIR=$VI/cache/$NODE/mpl
  export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONHASHSEED=0
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export TRITON_CACHE_DIR=$VI/cache/$NODE/triton TORCHINDUCTOR_CACHE_DIR=$VI/cache/$NODE/inductor XDG_CACHE_HOME=$VI/cache/$NODE/xdg
  T=(); D=(--data-root "$VI/b200/data")
  export EV_PAUSE=$LOGS/train.pause.$NODE EV_STATUS_GLOB="nonexistent_queue.status" EV_NO_GIT=1
  export EV_DF_PATH=$VI
  CLAIM_MAX_S=${CLAIM_MAX_S:-3600}
  EV_CAP_S=$(awk -v h="${EV_CAP_H:?任务书的上限（小时）}" 'BEGIN{print int(h * 3600)}')
fi
OUT=$REPO/experiments/X/${RUN_ID}_lb_4b_$NODE
case "$(basename "$OUT")" in [0-9][0-9][0-9][0-9]_*) ;; *) echo "目录名没有运行编号：$OUT" >&2; exit 2;; esac
# shellcheck disable=SC1091
source "$WT/scripts/server/eval_lib.sh"
cd "$WT" || exit 1
[ "$TINY" = 1 ] || [ "${CUDA_VISIBLE_DEVICES:-}" = 0 ] || { progress "blocked: 不在 srun 的资源范围里"; exit 1; }
[ -f "$CK/manifest.json" ] || { progress "blocked: 没有 4B X-2 的检查点 $CK"; exit 1; }
NOK=0
claim_gpu "0101-hpc-07-lb-4b.md" || { progress "blocked: gpu not free after ${CLAIM_MAX_S}s"; exit 1; }
mkdir -p "$OUT"
echo "$(now) start commit $(git -C "$WT" rev-parse HEAD 2>/dev/null || echo unknown) node $NODE model $M checkpoint manifest $(sha "$CK/manifest.json" | cut -d' ' -f1)" >> "$OUT/steps.log"
LB=(--model "$M" ${T[@]+"${T[@]}"} --raw-dir "$RAW")
[ -f "$OUT/lb_base/lang_battery.json" ] || \
  step lb_base_4b "$PY" scripts/lang_battery.py "${LB[@]}" --label base_4b --out "$OUT/lb_base" || NOK=$((NOK + 1))
[ -f "$OUT/lb_x2_4b/lang_battery.json" ] || \
  step lb_x2_4b "$PY" scripts/lang_battery.py "${LB[@]}" --label x2_4b ${D[@]+"${D[@]}"} --checkpoint "$CK" --out "$OUT/lb_x2_4b" || NOK=$((NOK + 1))
step lb_compare_x2_4b "$PY" scripts/lang_battery.py --compare "$OUT/lb_base" "$OUT/lb_x2_4b" --out "$OUT/lb_compare_x2_4b.json" || NOK=$((NOK + 1))
echo "$(now) done; failed steps: $NOK" >> "$OUT/steps.log"
progress "done $RUN_ID (failed steps: $NOK)"
push_out "lang battery 4B ${RUN_ID} done (failed steps: $NOK)"
