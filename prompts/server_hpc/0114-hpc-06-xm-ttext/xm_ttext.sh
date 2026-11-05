#!/usr/bin/env bash
# RUN-0114（ITSC HPC hpcgpu06）：X-M 的定位与计数 T 文本读数（run_e1 --task refexp_box count --checkpoint …；D182 第 3 点：补进 X-M 的语言表）。
#   命令行照抄 RUN-0091（prompts/server_hpc/0091-hpc-06-x-ttext/x_ttext.sh），只换检查点与输出目录。与 RUN-0069 在 06 上的 T 文本基座
#   （0069_ttext/qwen3.5-9b）逐题配对：从 7f53f240 到现在，run_e1 / src/e1 / src/adapters 没有改动（管理员核对过 git log）。
#   另：lb_compare_xm（只用 CPU）：scripts/lang_battery.py --compare 07 的基座（RUN-0084 lb_base）与 X-M（RUN-0107 lb_xm），与四方表其余各列同格式。
# 铁律：任何情况下都不能取消或停止 Slurm 任务 157296、157297。只用本节点自己的暂停文件。
# 用法：srun --jobid=157296 --overlap -n1 -c 8 env PATH=… TINY=0 EV_CAP_H=… bash …/xm_ttext.sh < /dev/null
# 彩排（开发机）：TINY=1 WT=… REPO=… LOGS=… PY=… bash xm_ttext.sh（只核对目录名与命令行）
set -uo pipefail
TINY=${TINY:-0}
RUN_ID=0114
M=qwen3.5-9b
if [ "$TINY" = 1 ]; then
  : "${WT:?}" "${REPO:?}" "${LOGS:?}" "${PY:?}"
  NODE=tiny; D=(); XM_CK=/nonexistent/xm
  export EV_NO_GIT=1
  OUT=$REPO/experiments/XM/0114_xm_ttext_tiny
else
  NODE=hpcgpu06
  VI=/home/mail/chenyuma/visint
  WT=$VI/wt/wt0114
  REPO=$VI/repo
  LOGS=$VI/logs
  PY=$VI/env/bin/python
  XM_CK=$REPO/experiments/XM/0104_xm_hpcgpu07/0104_xm_train/w2c/qwen3.5-9b/seed0/ckpt/final/model
  export HF_HOME=$VI/hf HF_HUB_CACHE=$VI/hf/hub HF_HUB_OFFLINE=1
  export TMPDIR=$VI/tmp PIP_NO_CACHE_DIR=1 NLTK_DATA=$VI/b200/nltk_data
  export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONHASHSEED=0
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export TRITON_CACHE_DIR=$VI/cache/$NODE/triton TORCHINDUCTOR_CACHE_DIR=$VI/cache/$NODE/inductor XDG_CACHE_HOME=$VI/cache/$NODE/xdg
  D=(--data-root "$VI/b200/data")
  export EV_PAUSE=$LOGS/train.pause.$NODE EV_STATUS_GLOB="nonexistent_queue.status" EV_NO_GIT=1
  export EV_DF_PATH=$VI
  CLAIM_MAX_S=${CLAIM_MAX_S:-3600}
  EV_CAP_S=$(awk -v h="${EV_CAP_H:?任务书的上限（小时）}" 'BEGIN{print int(h * 3600)}')
  OUT=$REPO/experiments/XM/0114_xm_ttext_hpcgpu06
fi
LB_BASE=$REPO/experiments/X/0084_lb_explore_hpcgpu07/lb_base
LB_XM=$REPO/experiments/XM/0107_xm_refseg_lb_hpcgpu07/0107_lb_xm
case "$(basename "$OUT")" in [0-9][0-9][0-9][0-9]_*) ;; *) echo "OUT 的目录名没有运行编号：$OUT" >&2; exit 2;; esac
cmd() { echo "$PY" scripts/run_e1.py --model $M --task refexp_box count --subset eval ${D[@]+"${D[@]}"} --checkpoint "$2" --checkpoint-label "t_text/$1/seed0" --out "$OUT"; }
if [ "$TINY" = 1 ]; then
  mkdir -p "$OUT"; { cmd xm "$XM_CK"; echo "$PY scripts/lang_battery.py --compare $LB_BASE $LB_XM --out $OUT/lb_compare_xm.json"; } > "$OUT/cmd.txt"
  echo "tiny：只核对目录名与命令行 → $OUT/cmd.txt"; exit 0
fi
# shellcheck disable=SC1091
source "$WT/scripts/server/eval_lib.sh"
cd "$WT" || exit 1
[ "${CUDA_VISIBLE_DEVICES:-}" = 0 ] || { progress "blocked: 不在 srun 的资源范围里"; exit 1; }
[ -f "$XM_CK/manifest.json" ] || { progress "blocked: 没有 X-M 的检查点 $XM_CK"; exit 1; }
NOK=0
claim_gpu "0114-hpc-06-xm-ttext.md" || { progress "blocked: gpu not free after ${CLAIM_MAX_S}s"; exit 1; }
echo "$(now) start commit $(git -C "$WT" rev-parse HEAD 2>/dev/null || echo unknown) node $NODE" >> "$OUT/steps.log"
echo "$(now) xm checkpoint manifest $(sha "$XM_CK/manifest.json" | cut -d' ' -f1)" >> "$OUT/steps.log"
step ttext_xm "$PY" scripts/run_e1.py --model $M --task refexp_box count --subset eval ${D[@]+"${D[@]}"} \
  --checkpoint "$XM_CK" --checkpoint-label "t_text/xm/seed0" --out "$OUT" || NOK=$((NOK + 1))
step lb_compare_xm "$PY" scripts/lang_battery.py --compare "$LB_BASE" "$LB_XM" --out "$OUT/lb_compare_xm.json" || NOK=$((NOK + 1))
echo "$(now) done; failed steps: $NOK" >> "$OUT/steps.log"
progress "done $RUN_ID (failed steps: $NOK)"
push_out "X-M T-text ${RUN_ID} done (failed steps: $NOK)"
