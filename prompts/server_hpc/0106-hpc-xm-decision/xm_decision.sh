#!/usr/bin/env bash
# RUN-0106（ITSC HPC，节点可选）：X-M 判定集的评测（D165 第五节；预注册 XM-4b、XM-5）。三个叶目录必须在同一节点（gate_xm 核对）。
#   1. xm_tune：X-M 的 4 个 xm_* 设置，只作检查（XM-4b：不改推理设置）；
#   2. xm_decision（密封，--gate G-XM）：X-M 的 4 个 xm_* 设置；
#   3. xm_decision（密封，--gate G-XM）：基座零样本 base_det、base_box_grabcut（D168：判定时与 X-M 同节点跑）。
#   等 WAIT_DONE（缺省 eval_0104.status 以 "done 0104" 开头，即 X-M 训练完）。
# 铁律：任何情况下都不能取消或停止 Slurm 任务 157296、157297。只用本节点自己的暂停文件。
# 用法：srun --jobid=<157296（06）或 157297（07）> --overlap -n1 -c 8 env PATH=… TINY=0 NODE=<hpcgpu06|hpcgpu07> EV_CAP_H=… bash …/xm_decision.sh < /dev/null
# 彩排（开发机）：TINY=1 WT=… REPO=… LOGS=… PY=… CK=<tiny X-M 检查点> PAIRS_JSON=… bash xm_decision.sh
# 补充 1（D176）：X-M 两步补上 --merge-check-data-root（同 RUN-0105 的 xm_tune.sh；首次运行因此在合并核对处被拒）；
#   STAGES 只重跑其中几步（缺省三步都跑），SUF 给重跑用新的子目录名（如 _r2），不与失败的那次混在一起。
set -uo pipefail
TINY=${TINY:-0}
RUN_ID=0106
STAGES=${STAGES:-"tune decision base"}
SUF=${SUF:-}
XS=(xm_spline_trimap xm_box_grabcut xm_ellipse_trimap xm_spline_raster)
BS=(base_det base_box_grabcut)
if [ "$TINY" = 1 ]; then
  : "${WT:?}" "${REPO:?}" "${LOGS:?}" "${PY:?}" "${CK:?}" "${PAIRS_JSON:?}"
  NODE=tiny; M=qwen3.5-0.8b; T=(--tiny); DATA=(--pairs-json "$PAIRS_JSON"); TW=1; WAIT_DONE=""; MC=()
  export EV_NO_GIT=1
else
  : "${NODE:?hpcgpu06 或 hpcgpu07}"
  case "$NODE" in hpcgpu06|hpcgpu07) ;; *) echo "NODE 只能是 hpcgpu06 或 hpcgpu07：$NODE" >&2; exit 2;; esac
  M=qwen3.5-9b
  VI=/home/mail/chenyuma/visint
  WT=$VI/wt/wt$RUN_ID
  REPO=$VI/repo
  LOGS=$VI/logs
  PY=$VI/env/bin/python
  export HF_HOME=$VI/hf HF_HUB_CACHE=$VI/hf/hub HF_HUB_OFFLINE=1
  export TMPDIR=$VI/tmp PIP_NO_CACHE_DIR=1 MPLCONFIGDIR=$VI/cache/$NODE/mpl
  export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONHASHSEED=0
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export TRITON_CACHE_DIR=$VI/cache/$NODE/triton TORCHINDUCTOR_CACHE_DIR=$VI/cache/$NODE/inductor XDG_CACHE_HOME=$VI/cache/$NODE/xdg
  T=(); DATA=(--data-root "$VI/data"); TW=8; MC=(--merge-check-data-root "$VI/b200/data")   # 合并核对读 E1 的开发数据（D176）
  CK=${CK:-$REPO/experiments/XM/0104_xm_hpcgpu07/0104_xm_train/w2c/qwen3.5-9b/seed0/ckpt/final/model}
  WAIT_DONE=${WAIT_DONE:-"eval_0104.status:done 0104"}
  export EV_PAUSE=$LOGS/train.pause.$NODE EV_STATUS_GLOB="nonexistent_queue.status" EV_NO_GIT=1
  export EV_DF_PATH=$VI
  CLAIM_MAX_S=${CLAIM_MAX_S:-3600}
  EV_CAP_S=$(awk -v h="${EV_CAP_H:?任务书的上限（小时）}" 'BEGIN{print int(h * 3600)}')
fi
OUT=$REPO/experiments/XM/${RUN_ID}_xm_decision_$NODE
O_TUNE=$OUT/${RUN_ID}_xm_tune${SUF}
O_XM=$OUT/${RUN_ID}_xm_dec${SUF}
O_BASE=$OUT/${RUN_ID}_base_dec
for o in "$OUT" "$O_TUNE" "$O_XM" "$O_BASE"; do case "$(basename "$o")" in [0-9][0-9][0-9][0-9]_*) ;; *) echo "目录名没有运行编号：$o" >&2; exit 2;; esac; done
# shellcheck disable=SC1091
source "$WT/scripts/server/eval_lib.sh"
cd "$WT" || exit 1
if [ "$TINY" != 1 ]; then
  [ "${CUDA_VISIBLE_DEVICES:-}" = 0 ] || { progress "blocked: 不在 srun 的资源范围里"; exit 1; }
  "$PY" -c "import sys; sys.path.insert(0,'.'); from src.xm import prompts as P; sys.exit(0 if P.FROZEN else 1)" || { progress "blocked: 提示没有冻结"; exit 1; }
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
claim_gpu "0106-hpc-xm-decision.md ($NODE)" || { progress "blocked: gpu not free after ${CLAIM_MAX_S}s"; exit 1; }
has() { case " $STAGES " in *" $1 "*) return 0;; esac; return 1; }
echo "$(now) stages [$STAGES] suffix [${SUF}]" >> "$OUT/steps.log"
echo "$(now) start commit $(git -C "$WT" rev-parse HEAD 2>/dev/null || echo unknown) node $NODE model $M x-m checkpoint manifest $(sha "$CK/manifest.json" | cut -d' ' -f1)" >> "$OUT/steps.log"
res_of() { [ -d "$1/$M" ] && echo --resume; }
G=(); [ "$TINY" = 1 ] || G=(--gate G-XM)
has tune && { step xm_tune "$PY" scripts/run_xm.py --model $M ${T[@]+"${T[@]}"} --checkpoint "$CK" --split xm_tune --setting "${XS[@]}" \
  ${DATA[@]+"${DATA[@]}"} ${MC[@]+"${MC[@]}"} --out "$O_TUNE" --tool-workers "$TW" $(res_of "$O_TUNE") || NOK=$((NOK + 1)); }
DSPLIT=xm_decision; [ "$TINY" = 1 ] && DSPLIT=xm_tune   # tiny 的合成对只有一个划分
has decision && { step xm_decision "$PY" scripts/run_xm.py --model $M ${T[@]+"${T[@]}"} --checkpoint "$CK" --split $DSPLIT ${G[@]+"${G[@]}"} --setting "${XS[@]}" \
  ${DATA[@]+"${DATA[@]}"} ${MC[@]+"${MC[@]}"} --out "$O_XM" --tool-workers "$TW" $(res_of "$O_XM") || NOK=$((NOK + 1)); }
has base && { step base_decision "$PY" scripts/run_xm.py --model $M ${T[@]+"${T[@]}"} --split $DSPLIT ${G[@]+"${G[@]}"} --setting "${BS[@]}" \
  ${DATA[@]+"${DATA[@]}"} --out "$O_BASE" --tool-workers "$TW" $(res_of "$O_BASE") || NOK=$((NOK + 1)); }
echo "$(now) done; failed steps: $NOK" >> "$OUT/steps.log"
progress "done $RUN_ID (failed steps: $NOK)"
push_out "X-M decision eval ${RUN_ID} done (failed steps: $NOK)"
