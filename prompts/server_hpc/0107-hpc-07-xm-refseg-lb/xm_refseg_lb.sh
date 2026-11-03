#!/usr/bin/env bash
# RUN-0107（ITSC HPC hpcgpu07）：X-M 判定第 3、4 条的评测，都在 07，与 X-2 seed 0（RUN-0080 指代分割、RUN-0084 大题量）同节点配对（XM-4b）。
#   1. 指代分割：run_x2 用 X-M 的检查点，explore_val 三层、devtune800、devcheck2（共 2,300 句；同 RUN-0080 / 0093 的命令）；
#   2. 大题量：lang_battery 用 X-M 的检查点（同 RUN-0084 / 0101 的口径；--data-root 只配 --checkpoint，D161）。
#   等 WAIT_DONE（缺省 eval_0104.status 以 "done 0104" 开头；RUN-0106 也在 07 上时，任务书改成等 eval_0106.status）。
# 铁律：任何情况下都不能取消或停止 Slurm 任务 157296、157297。只用本节点自己的暂停文件。
# 用法：srun --jobid=157297 --overlap -n1 -c 12 env PATH=… TINY=0 EV_CAP_H=… [WAIT_DONE=…] bash …/xm_refseg_lb.sh < /dev/null
# 彩排（开发机）：TINY=1 WT=… REPO=… LOGS=… PY=… CK=<tiny X-M 检查点> RAW=<lang_battery 的原始数据> bash xm_refseg_lb.sh
set -uo pipefail
TINY=${TINY:-0}
RUN_ID=0107
EVSPLITS=(explore_val_refcoco_unc_val "explore_val_refcoco+_unc_val" explore_val_refcocog_umd_val)
W2SPLITS=(devtune800 devcheck2)
if [ "$TINY" = 1 ]; then
  : "${WT:?}" "${REPO:?}" "${LOGS:?}" "${PY:?}" "${CK:?}" "${RAW:?}"
  NODE=tiny; M=qwen3.5-0.8b; T=(--tiny); D=(); CW=1; TW=1; LIMIT=(--limit 3); LBT=(--tiny --limit 3); WAIT_DONE=""
  EVSPLITS=(explore_val_refcoco_unc_val); W2SPLITS=(devtune800)
  export NLTK_DATA=$WT/tests/fixtures/health/nltk_data EV_NO_GIT=1
else
  NODE=hpcgpu07; M=qwen3.5-9b
  VI=/home/mail/chenyuma/visint
  WT=$VI/wt/wt$RUN_ID
  REPO=$VI/repo
  LOGS=$VI/logs
  PY=$VI/env/bin/python
  RAW=$VI/data/lang_battery_raw
  export HF_HOME=$VI/hf HF_HUB_CACHE=$VI/hf/hub HF_HUB_OFFLINE=1
  export TMPDIR=$VI/tmp PIP_NO_CACHE_DIR=1 NLTK_DATA=$VI/b200/nltk_data MPLCONFIGDIR=$VI/cache/$NODE/mpl
  export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONHASHSEED=0
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export TRITON_CACHE_DIR=$VI/cache/$NODE/triton TORCHINDUCTOR_CACHE_DIR=$VI/cache/$NODE/inductor XDG_CACHE_HOME=$VI/cache/$NODE/xdg
  T=(); D=(--data-root "$VI/b200/data"); CW=8; TW=8; LIMIT=(); LBT=()
  CK=${CK:-$REPO/experiments/XM/0104_xm_hpcgpu07/0104_xm_train/w2c/qwen3.5-9b/seed0/ckpt/final/model}
  WAIT_DONE=${WAIT_DONE:-"eval_0104.status:done 0104"}
  export EV_PAUSE=$LOGS/train.pause.$NODE EV_STATUS_GLOB="nonexistent_queue.status" EV_NO_GIT=1
  export EV_DF_PATH=$VI
  CLAIM_MAX_S=${CLAIM_MAX_S:-3600}
  EV_CAP_S=$(awk -v h="${EV_CAP_H:?任务书的上限（小时）}" 'BEGIN{print int(h * 3600)}')
fi
OUT=$REPO/experiments/XM/${RUN_ID}_xm_refseg_lb_$NODE
EVAL=$OUT/${RUN_ID}_refseg
LB=$OUT/${RUN_ID}_lb_xm
for o in "$OUT" "$EVAL" "$LB"; do case "$(basename "$o")" in [0-9][0-9][0-9][0-9]_*) ;; *) echo "目录名没有运行编号：$o" >&2; exit 2;; esac; done
# shellcheck disable=SC1091
source "$WT/scripts/server/eval_lib.sh"
cd "$WT" || exit 1
if [ "$TINY" != 1 ]; then
  [ "${CUDA_VISIBLE_DEVICES:-}" = 0 ] || { progress "blocked: 不在 srun 的资源范围里"; exit 1; }
  "$PY" -c "import json,sys; sys.exit(0 if json.load(open('configs/w2.json'))['frozen'] is True else 1)" || { progress "blocked: configs/w2.json 没有冻结"; exit 1; }
  if [ -n "$WAIT_DONE" ]; then
    wf=${WAIT_DONE%%:*}; wp=${WAIT_DONE#*:}; t0=$(date +%s)
    while :; do
      st=$(cut -d' ' -f2- "$LOGS/$wf" 2>/dev/null || true)
      case "$st" in "$wp"*) break;; blocked*) progress "blocked: $wf 是 blocked：$st"; exit 1;; esac
      progress "waiting for $wf to start with '$wp' (now: ${st:-missing})"
      [ $(( $(date +%s) - t0 )) -ge 172800 ] && { progress "blocked: 等 $wf 超时"; exit 1; }
      sleep 60
    done
  fi
  [ -f "$CK/manifest.json" ] || { progress "blocked: 没有 X-M 的检查点 $CK"; exit 1; }
fi
NOK=0
claim_gpu "0107-hpc-07-xm-refseg-lb.md" || { progress "blocked: gpu not free after ${CLAIM_MAX_S}s"; exit 1; }
echo "$(now) start commit $(git -C "$WT" rev-parse HEAD 2>/dev/null || echo unknown) node $NODE model $M x-m checkpoint manifest $(sha "$CK/manifest.json" | cut -d' ' -f1)" >> "$OUT/steps.log"
X=(--model $M ${T[@]+"${T[@]}"} ${D[@]+"${D[@]}"} --checkpoint "$CK" --candidate-workers "$CW" --tool-workers "$TW" ${LIMIT[@]+"${LIMIT[@]}"})
for SP in "${EVSPLITS[@]}"; do
  RES=(); [ -d "$EVAL/$M/refseg/x2_spline_trimap/$SP" ] && RES=(--resume)
  step "xm_refseg_$SP" "$PY" scripts/run_x2.py "${X[@]}" --splits-file configs/explore_splits.json --split "$SP" --out "$EVAL" ${RES[@]+"${RES[@]}"} || NOK=$((NOK + 1))
done
for SP in "${W2SPLITS[@]}"; do
  RES=(); [ -d "$EVAL/$M/refseg/x2_spline_trimap/$SP" ] && RES=(--resume)
  step "xm_refseg_$SP" "$PY" scripts/run_x2.py "${X[@]}" --split "$SP" --out "$EVAL" ${RES[@]+"${RES[@]}"} || NOK=$((NOK + 1))
done
[ -f "$LB/lang_battery.json" ] || \
  step lb_xm "$PY" scripts/lang_battery.py --model "$M" ${LBT[@]+"${LBT[@]}"} --raw-dir "$RAW" --label xm ${D[@]+"${D[@]}"} --checkpoint "$CK" --out "$LB" || NOK=$((NOK + 1))
echo "$(now) done; failed steps: $NOK" >> "$OUT/steps.log"
progress "done $RUN_ID (failed steps: $NOK)"
push_out "X-M refseg + battery ${RUN_ID} done (failed steps: $NOK)"
