#!/usr/bin/env bash
# RUN-0110（ITSC HPC，节点可选：06 或 07）：X-M 合并后模型的最低限度健康检查（简报 §8.11：每个门都要；G-XM 门报告的第 5 项）。
#   health_xm：scripts/health_check.py --checkpoint <X-M>（约 200 题：MMLU-Pro 100、GSM8K 50、IFEval 50；评测前合并核对）；
#   health_compare：对同节点基座的 health_base.json（06 = RUN-0074，07 = RUN-0080；健康检查是确定性的固定协议，同节点重测与已有读数逐题相同，RUN-0074 核对过；G-EX 同此做法）。
#   命令行照抄 RUN-0080（prompts/server_hpc/0080-hpc-07-x2/x2.sh 的 HC 与 D）。07 上等 WAIT_DONE（缺省 eval_0109.status 以 "done 0109" 开头），06 上不等。
# 铁律：任何情况下都不能取消或停止 Slurm 任务 157296、157297。只用本节点自己的暂停文件。
# 用法：srun --jobid=<157296（06）或 157297（07）> --overlap -n1 -c 8 env PATH=… TINY=0 NODE=<hpcgpu06|hpcgpu07> EV_CAP_H=… bash …/xm_health.sh < /dev/null
# 彩排（开发机）：TINY=1 WT=… REPO=… LOGS=… PY=… CK=<tiny X-M 检查点> HB=<tiny 的 health_base.json，空则先跑一次 tiny 基座> bash xm_health.sh
set -uo pipefail
TINY=${TINY:-0}
RUN_ID=0110
if [ "$TINY" = 1 ]; then
  : "${WT:?}" "${REPO:?}" "${LOGS:?}" "${PY:?}" "${CK:?}"
  NODE=tiny; M=qwen3.5-0.8b; D=(); HT=(--tiny --skip-ifeval-scoring --limit 2); WAIT_DONE=""
  HB=${HB:-}
  export NLTK_DATA=$WT/tests/fixtures/health/nltk_data EV_NO_GIT=1
else
  : "${NODE:?hpcgpu06 或 hpcgpu07}"
  M=qwen3.5-9b
  VI=/home/mail/chenyuma/visint
  WT=$VI/wt/wt$RUN_ID
  REPO=$VI/repo
  LOGS=$VI/logs
  PY=$VI/env/bin/python
  export HF_HOME=$VI/hf HF_HUB_CACHE=$VI/hf/hub HF_HUB_OFFLINE=1
  export TMPDIR=$VI/tmp PIP_NO_CACHE_DIR=1 NLTK_DATA=$VI/b200/nltk_data MPLCONFIGDIR=$VI/cache/$NODE/mpl
  export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONHASHSEED=0
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export TRITON_CACHE_DIR=$VI/cache/$NODE/triton TORCHINDUCTOR_CACHE_DIR=$VI/cache/$NODE/inductor XDG_CACHE_HOME=$VI/cache/$NODE/xdg
  D=(--data-root "$VI/b200/data"); HT=()   # 合并核对读 E1 的 grid16 开发子集（同 RUN-0080）
  CK=$REPO/experiments/XM/0104_xm_hpcgpu07/0104_xm_train/w2c/qwen3.5-9b/seed0/ckpt/final/model
  case "$NODE" in
    hpcgpu06) HB=$REPO/experiments/W2/0074_lang_battery_hpcgpu06/health/health_base.json; WAIT_DONE=${WAIT_DONE:-} ;;
    hpcgpu07) HB=$REPO/experiments/X/0080_x2_hpcgpu07/0080_eval/health/health_base.json; WAIT_DONE=${WAIT_DONE:-"eval_0109.status:done 0109"} ;;
    *) echo "NODE 只能是 hpcgpu06 或 hpcgpu07：$NODE" >&2; exit 2;;
  esac
  export EV_PAUSE=$LOGS/train.pause.$NODE EV_STATUS_GLOB="nonexistent_queue.status" EV_NO_GIT=1
  export EV_DF_PATH=$VI
  CLAIM_MAX_S=${CLAIM_MAX_S:-3600}
  EV_CAP_S=$(awk -v h="${EV_CAP_H:?任务书的上限（小时）}" 'BEGIN{print int(h * 3600)}')
fi
OUT=$REPO/experiments/XM/${RUN_ID}_xm_health_$NODE
case "$(basename "$OUT")" in [0-9][0-9][0-9][0-9]_*) ;; *) echo "目录名没有运行编号：$OUT" >&2; exit 2;; esac
# shellcheck disable=SC1091
source "$WT/scripts/server/eval_lib.sh"
cd "$WT" || exit 1
if [ "$TINY" != 1 ]; then
  [ "${CUDA_VISIBLE_DEVICES:-}" = 0 ] || { progress "blocked: 不在 srun 的资源范围里"; exit 1; }
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
  [ -f "$HB" ] || { progress "blocked: 没有 07 上的基座健康检查 $HB"; exit 1; }
fi
NOK=0
claim_gpu "0110-hpc-xm-health.md ($NODE)" || { progress "blocked: gpu not free after ${CLAIM_MAX_S}s"; exit 1; }
mkdir -p "$OUT/health"
HC=(--model $M ${HT[@]+"${HT[@]}"})
if [ "$TINY" = 1 ] && [ -z "$HB" ]; then
  HB=$OUT/health/health_base_tiny.json
  step health_base_tiny "$PY" scripts/health_check.py "${HC[@]}" --out "$HB" || NOK=$((NOK + 1))
fi
echo "$(now) start commit $(git -C "$WT" rev-parse HEAD 2>/dev/null || echo unknown) node $NODE model $M x-m checkpoint manifest $(sha "$CK/manifest.json" | cut -d' ' -f1) health_base $(sha "$HB" | cut -d' ' -f1)" >> "$OUT/steps.log"
step health_xm "$PY" scripts/health_check.py "${HC[@]}" --checkpoint "$CK" ${D[@]+"${D[@]}"} --out "$OUT/health/health_xm.json" || NOK=$((NOK + 1))
step health_compare "$PY" scripts/health_check.py --compare "$HB" "$OUT/health/health_xm.json" \
  --out "$OUT/health/health_compare.json" || NOK=$((NOK + 1))
echo "$(now) done; failed steps: $NOK" >> "$OUT/steps.log"
progress "done $RUN_ID (failed steps: $NOK)"
push_out "X-M health ${RUN_ID} done (failed steps: $NOK)"
