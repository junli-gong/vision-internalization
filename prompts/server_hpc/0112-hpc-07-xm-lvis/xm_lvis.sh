#!/usr/bin/env bash
# RUN-0112（ITSC HPC hpcgpu07）：G-XM 通过之后的 LVIS 子集（D165 第五节，描述性；XM-2b 每图最多 3 类，DEV-0082；D177、D179）。不调任何推理设置（XM-4b）。
#   1. lvis_sub：X-M 的 4 个 xm_* 设置，--xm-gate = RUN-0108 的 gate_xm.json；
#   2. lvis_sub：基座零样本 base_det、base_box_grabcut（同一 gate 文件放行；与 X-M 同节点）。
#   命令行逐行照抄 RUN-0109（split 换成 lvis_sub）。等 WAIT_DONE（缺省 eval_0109.status 以 "done 0109" 开头）。
#   每图的类别上限的计数写进各叶目录 config.yaml 的 xm.lvis_cap（n_pairs_before_cap 必须是 3,134，XM-2a）。
# 铁律：任何情况下都不能取消或停止 Slurm 任务 157296、157297。只用本节点自己的暂停文件。
# 用法：srun --jobid=157297 --overlap -n1 -c 8 env PATH=… TINY=0 EV_CAP_H=… bash …/xm_lvis.sh < /dev/null
# 彩排（开发机）：TINY=1 WT=… REPO=… LOGS=… PY=… CK=<tiny X-M 检查点> PAIRS_JSON=… bash xm_lvis.sh（tiny 的合成对只有 xm_tune，不带 gate）
set -uo pipefail
TINY=${TINY:-0}
RUN_ID=0112
STAGES=${STAGES:-"xm base"}
XS=(xm_spline_trimap xm_box_grabcut xm_ellipse_trimap xm_spline_raster)
BS=(base_det base_box_grabcut)
if [ "$TINY" = 1 ]; then
  : "${WT:?}" "${REPO:?}" "${LOGS:?}" "${PY:?}" "${CK:?}" "${PAIRS_JSON:?}"
  NODE=tiny; M=qwen3.5-0.8b; T=(--tiny); DATA=(--pairs-json "$PAIRS_JSON"); TW=1; MC=()
  LSPLIT=xm_tune; XG=(); GATE_JSON=/nonexistent/gate_xm.json; WAIT_DONE=""
  export EV_NO_GIT=1
else
  NODE=hpcgpu07; M=qwen3.5-9b
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
  T=(); DATA=(--data-root "$VI/data"); TW=8; MC=(--merge-check-data-root "$VI/b200/data")   # 合并核对读 E1 的开发数据（D176）
  CK=$REPO/experiments/XM/0104_xm_hpcgpu07/0104_xm_train/w2c/qwen3.5-9b/seed0/ckpt/final/model
  GATE_JSON=$REPO/experiments/XM/0108_gate_xm_hpcgpu07/0108_gate/gate_xm.json
  LSPLIT=lvis_sub; XG=(--xm-gate "$GATE_JSON")
  WAIT_DONE=${WAIT_DONE:-"eval_0109.status:done 0109"}
  export EV_PAUSE=$LOGS/train.pause.$NODE EV_STATUS_GLOB="nonexistent_queue.status" EV_NO_GIT=1
  export EV_DF_PATH=$VI
  CLAIM_MAX_S=${CLAIM_MAX_S:-3600}
  EV_CAP_S=$(awk -v h="${EV_CAP_H:?任务书的上限（小时）}" 'BEGIN{print int(h * 3600)}')
fi
OUT=$REPO/experiments/XM/${RUN_ID}_xm_lvis_$NODE
O_XM=$OUT/${RUN_ID}_xm_lvis
O_BASE=$OUT/${RUN_ID}_base_lvis
for o in "$OUT" "$O_XM" "$O_BASE"; do case "$(basename "$o")" in [0-9][0-9][0-9][0-9]_*) ;; *) echo "目录名没有运行编号：$o" >&2; exit 2;; esac; done
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
  [ -f "$GATE_JSON" ] || { progress "blocked: 没有 G-XM 的判定文件 $GATE_JSON"; exit 1; }
fi
NOK=0
claim_gpu "0112-hpc-07-xm-lvis.md" || { progress "blocked: gpu not free after ${CLAIM_MAX_S}s"; exit 1; }
has() { case " $STAGES " in *" $1 "*) return 0;; esac; return 1; }
echo "$(now) stages [$STAGES]" >> "$OUT/steps.log"
echo "$(now) start commit $(git -C "$WT" rev-parse HEAD 2>/dev/null || echo unknown) node $NODE model $M x-m checkpoint manifest $(sha "$CK/manifest.json" | cut -d' ' -f1) gate $( [ -f "$GATE_JSON" ] && sha "$GATE_JSON" | cut -d' ' -f1 || echo none)" >> "$OUT/steps.log"
res_of() { [ -d "$1/$M" ] && echo --resume; }
has xm && { step xm_lvis "$PY" scripts/run_xm.py --model $M ${T[@]+"${T[@]}"} --checkpoint "$CK" --split $LSPLIT ${XG[@]+"${XG[@]}"} --setting "${XS[@]}" \
  ${DATA[@]+"${DATA[@]}"} ${MC[@]+"${MC[@]}"} --out "$O_XM" --tool-workers "$TW" $(res_of "$O_XM") || NOK=$((NOK + 1)); }
has base && { step base_lvis "$PY" scripts/run_xm.py --model $M ${T[@]+"${T[@]}"} --split $LSPLIT ${XG[@]+"${XG[@]}"} --setting "${BS[@]}" \
  ${DATA[@]+"${DATA[@]}"} --out "$O_BASE" --tool-workers "$TW" $(res_of "$O_BASE") || NOK=$((NOK + 1)); }
echo "$(now) done; failed steps: $NOK" >> "$OUT/steps.log"
progress "done $RUN_ID (failed steps: $NOK)"
push_out "X-M LVIS ${RUN_ID} done (failed steps: $NOK)"
