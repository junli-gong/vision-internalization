#!/usr/bin/env bash
# RUN-0105（ITSC HPC hpcgpu07）：X-M 的两项不依赖 X-M 检查点的运行（D165 第六节 D0、D1–D3；D168）：
#   1. 基座零样本在调参集 xm_tune 上：base_det、base_box_grabcut（检测与实例分割的起点；只调推理设置用，XM-4a）；
#   2. X-2（seed 0，RUN-0080 的检查点）在 ReasonSeg val 上的指代分割（--system x2，描述性，D165 第六节）。
#   不碰判定集 xm_decision（基座在判定集上的运行推迟到判定时，在 06，D168）。
# 铁律：任何情况下都不能取消或停止 Slurm 任务 157296、157297。只用本节点自己的暂停文件。
# 用法：srun --jobid=157297 --overlap -n1 -c 8 env PATH=… TINY=0 EV_CAP_H=… bash …/xm_tune.sh < /dev/null
# 彩排（开发机）：TINY=1 WT=… REPO=… LOGS=… PY=… PAIRS_JSON=<tests/_xm_run_fakes.write_fixture 的输出> bash xm_tune.sh
set -uo pipefail
TINY=${TINY:-0}
RUN_ID=0105
if [ "$TINY" = 1 ]; then
  : "${WT:?}" "${REPO:?}" "${LOGS:?}" "${PY:?}" "${PAIRS_JSON:?}"
  NODE=tiny; M=qwen3.5-0.8b; T=(--tiny); DATA=(--pairs-json "$PAIRS_JSON"); TW=1; CW=1
  X2CK=/nonexistent/x2_ck; RS_DATA=(--data-root /nonexistent/data); MC=()
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
  T=(); DATA=(--data-root "$VI/data"); TW=8; CW=8
  X2CK=$REPO/experiments/X/0080_x2_hpcgpu07/0080_train/w2c/qwen3.5-9b/seed0/ckpt/final/model
  RS_DATA=(--data-root "$VI/data"); MC=(--merge-check-data-root "$VI/b200/data")
  export EV_PAUSE=$LOGS/train.pause.$NODE EV_STATUS_GLOB="nonexistent_queue.status" EV_NO_GIT=1
  export EV_DF_PATH=$VI
  CLAIM_MAX_S=${CLAIM_MAX_S:-3600}
  EV_CAP_S=$(awk -v h="${EV_CAP_H:?任务书的上限（小时）}" 'BEGIN{print int(h * 3600)}')
fi
OUT=$REPO/experiments/XM/${RUN_ID}_xm_tune_$NODE
OUT_BASE=$OUT/${RUN_ID}_base_tune
OUT_RS=$OUT/${RUN_ID}_reasonseg_x2
for o in "$OUT" "$OUT_BASE" "$OUT_RS"; do case "$(basename "$o")" in [0-9][0-9][0-9][0-9]_*) ;; *) echo "目录名没有运行编号：$o" >&2; exit 2;; esac; done
# shellcheck disable=SC1091
source "$WT/scripts/server/eval_lib.sh"
cd "$WT" || exit 1
if [ "$TINY" != 1 ]; then
  [ "${CUDA_VISIBLE_DEVICES:-}" = 0 ] || { progress "blocked: 不在 srun 的资源范围里"; exit 1; }
  [ -f configs/xm_splits.json ] || { progress "blocked: configs/xm_splits.json 还没提交（RUN-0103 之后由管理员提交）"; exit 1; }
  [ -f "$X2CK/manifest.json" ] || { progress "blocked: 没有 X-2 seed 0 的检查点 $X2CK"; exit 1; }
fi
NOK=0
claim_gpu "0105-hpc-07-xm-tune.md" || { progress "blocked: gpu not free after ${CLAIM_MAX_S}s"; exit 1; }
echo "$(now) start commit $(git -C "$WT" rev-parse HEAD 2>/dev/null || echo unknown) node $NODE model $M" >> "$OUT/steps.log"
res_of() { [ -d "$1/$M" ] && echo --resume; }
step base_tune "$PY" scripts/run_xm.py --model $M ${T[@]+"${T[@]}"} --split xm_tune --setting base_det base_box_grabcut \
  ${DATA[@]+"${DATA[@]}"} --out "$OUT_BASE" --tool-workers "$TW" $(res_of "$OUT_BASE") || NOK=$((NOK + 1))
if [ "$TINY" = 1 ]; then
  echo "$PY scripts/run_xm.py --model qwen3.5-9b --checkpoint \$X2CK --split reasonseg_val --system x2 --data-root \$VI/data --out $OUT_RS --candidate-workers 8 --tool-workers 8 --merge-check-data-root \$VI/b200/data" > "$OUT/cmd_reasonseg.txt"
else
  echo "$(now) x2 checkpoint manifest $(sha "$X2CK/manifest.json" | cut -d' ' -f1)" >> "$OUT/steps.log"
  step reasonseg_x2 "$PY" scripts/run_xm.py --model $M --checkpoint "$X2CK" --split reasonseg_val --system x2 ${RS_DATA[@]+"${RS_DATA[@]}"} \
    --out "$OUT_RS" --candidate-workers "$CW" --tool-workers "$TW" ${MC[@]+"${MC[@]}"} $(res_of "$OUT_RS") || NOK=$((NOK + 1))
fi
echo "$(now) done; failed steps: $NOK" >> "$OUT/steps.log"
progress "done $RUN_ID (failed steps: $NOK)"
push_out "X-M tune/ReasonSeg ${RUN_ID} done (failed steps: $NOK)"
