#!/usr/bin/env bash
# RUN-0104（ITSC HPC hpcgpu06）：X-M 训练（D165；预注册 XM-4、XM-4a；DEV-0081）。9B，从基座开始，configs/xm_train.json，seed 0。
#   1. data：build_xm_data.py，从 RUN-0103 的训练目标建 COCO 两条流的样本（剔除带 crowd 的对，D167-4；提示 xm-prompts-v1，已冻结）。
#   2. probe：train_w2c.py --stop-after-steps 50，取第 5 步以后 step_s 的中位数（timing_summary）。
#   3. 判断（XM-4a）：中位数 × 2,000 ≤ 30 小时（≤ 54 s/步）就续跑到 2,000 步；否则停下报告，由管理员定退路（从 X-2 继续训，XM-4b）。
#   4. train：同一条命令 --resume 到 2,000 步。
# 铁律：任何情况下都不能取消或停止 Slurm 任务 157296、157297。只用本节点自己的暂停文件。
# 用法：srun --jobid=157296 --overlap -n1 -c 12 env PATH=… TINY=0 EV_CAP_H=… bash …/xm_train.sh < /dev/null
# 彩排（开发机）：TINY=1 WT=… REPO=… LOGS=… PY=… XM_TARGETS=<build_targets --tiny 的输出> XM_DATA_ROOT=<假数据根>
#   X2DATA=… REPLAY=… W2C_DATA=… bash xm_train.sh（tiny 训练 3 步代替 50 步、续跑到 4 步）
set -uo pipefail
TINY=${TINY:-0}
RUN_ID=0104
STEP_S_MAX=54   # XM-4a：30 小时 / 2,000 步
if [ "$TINY" = 1 ]; then
  : "${WT:?}" "${REPO:?}" "${LOGS:?}" "${PY:?}" "${XM_TARGETS:?}" "${XM_DATA_ROOT:?}" "${X2DATA:?}" "${REPLAY:?}" "${W2C_DATA:?}"
  NODE=tiny; M=qwen3.5-0.8b; T=(--tiny); TD=(); PROBE=3; FULL=(--steps 4)
  export EV_NO_GIT=1
else
  NODE=hpcgpu06; M=qwen3.5-9b
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
  T=(); TD=(--data-root "$VI/data/train_data"); PROBE=50; FULL=()
  XM_TARGETS=$REPO/experiments/XM/0103_xm_targets
  XM_DATA_ROOT=$VI/data
  W2C_DATA=$REPO/experiments/W2/0067_w2c_data_hpcgpu06/w2c_data
  X2DATA=$REPO/experiments/X/0080_x2_hpcgpu07/0080_x2_data
  REPLAY=$REPO/experiments/W2/0076_w2c_replay_hpcgpu07/0076_replay_data
  export EV_PAUSE=$LOGS/train.pause.$NODE EV_STATUS_GLOB="nonexistent_queue.status" EV_NO_GIT=1
  export EV_DF_PATH=$VI
  CLAIM_MAX_S=${CLAIM_MAX_S:-3600}
  EV_CAP_S=$(awk -v h="${EV_CAP_H:?任务书的上限（小时）}" 'BEGIN{print int(h * 3600)}')
fi
OUT=$REPO/experiments/XM/${RUN_ID}_xm_$NODE
XMDATA=$OUT/${RUN_ID}_xm_data
TRAIN=$OUT/${RUN_ID}_xm_train
LEAF=$TRAIN/w2c/$M/seed0
CK=$LEAF/ckpt/final/model
for o in "$OUT" "$XMDATA" "$TRAIN"; do case "$(basename "$o")" in [0-9][0-9][0-9][0-9]_*) ;; *) echo "目录名没有运行编号：$o" >&2; exit 2;; esac; done
# shellcheck disable=SC1091
source "$WT/scripts/server/eval_lib.sh"
cd "$WT" || exit 1
if [ "$TINY" != 1 ]; then
  [ "${CUDA_VISIBLE_DEVICES:-}" = 0 ] || { progress "blocked: 不在 srun 的资源范围里"; exit 1; }
  "$PY" -c "import json,sys; sys.exit(0 if json.load(open('configs/w2.json'))['frozen'] is True else 1)" || { progress "blocked: configs/w2.json 没有冻结"; exit 1; }
  "$PY" -c "import sys; sys.path.insert(0,'.'); from src.xm import prompts as P; sys.exit(0 if P.FROZEN else 1)" || { progress "blocked: 提示没有冻结"; exit 1; }
  # build_targets 的布局：<out>/xm_targets/manifest.json（src/xm/targets.DATA_DIR）；--targets 给的是 <out>（RUN-0104 补充 1）
  [ -f "$XM_TARGETS/xm_targets/manifest.json" ] || { progress "blocked: 没有 RUN-0103 的训练目标 $XM_TARGETS/xm_targets"; exit 1; }
fi
die() { echo "$(now) ABORT: $*" | tee -a "$OUT/steps.log"; progress "blocked: $*"; push_out "X-M train aborted: $*"; exit 1; }
NOK=0
claim_gpu "0104-hpc-06-xm-train.md" || { progress "blocked: gpu not free after ${CLAIM_MAX_S}s"; exit 1; }
mkdir -p "$OUT"
echo "$(now) start commit $(git -C "$WT" rev-parse HEAD 2>/dev/null || echo unknown) node $NODE model $M xm cfg $(sha "$WT/configs/xm_train.json" | cut -d' ' -f1)" >> "$OUT/steps.log"
MEMPID=""
if [ "$TINY" != 1 ] && command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -lms 1000 >> "$OUT/gpu_mem.csv" 2>/dev/null &
  MEMPID=$!
fi
# 1. COCO 训练样本（可续跑）
if [ ! -f "$XMDATA/xm_data/manifest.json" ]; then   # build_xm_data 的布局：<out>/xm_data/manifest.json
  step xm_data "$PY" scripts/build_xm_data.py ${T[@]+"${T[@]}"} --targets "$XM_TARGETS" --data-root "$XM_DATA_ROOT" --out "$XMDATA" || die "COCO 训练样本失败"
fi
TR=(--config configs/xm_train.json --seed 0 --model $M ${T[@]+"${T[@]}"} --w2c-data "$W2C_DATA" --x2-data "$X2DATA" --replay-data "$REPLAY"
    ${TD[@]+"${TD[@]}"} --xm-data "$XMDATA" --xm-data-root "$XM_DATA_ROOT" --out "$TRAIN")
if [ ! -f "$CK/manifest.json" ]; then
  # 2. 探测（已有 latest 时跳过）
  if [ ! -d "$LEAF/ckpt/latest" ]; then
    step xm_probe "$PY" scripts/train_w2c.py "${TR[@]}" ${FULL[@]+"${FULL[@]}"} --stop-after-steps "$PROBE" || die "50 步探测失败"
  fi
  med=$("$PY" -c "import json,sys; sys.path.insert(0,'.'); from src.w2.w2c_train import timing_summary as t; s=t(sys.argv[1]); print(s['step_s_median'])" "$LEAF" 2>/dev/null || echo nan)
  echo "$(now) probe median step_s $med (max $STEP_S_MAX for 2,000 steps within 30 h)" >> "$OUT/steps.log"
  if [ "$TINY" != 1 ]; then
    awk -v m="$med" -v x="$STEP_S_MAX" 'BEGIN{exit !(m+0 > 0 && m+0 <= x)}' || die "探测的每步用时 $med s 超过 $STEP_S_MAX s：按 XM-4a 停下，由管理员定退路"
  fi
  # 4. 正式训练：续跑到 2,000 步
  step xm_train "$PY" scripts/train_w2c.py "${TR[@]}" ${FULL[@]+"${FULL[@]}"} --resume || die "训练失败"
fi
[ -f "$CK/manifest.json" ] || die "训练后没有检查点 $CK"
echo "$(now) checkpoint $CK manifest $(sha "$CK/manifest.json" | cut -d' ' -f1)" >> "$OUT/steps.log"
[ -n "$MEMPID" ] && kill "$MEMPID" 2>/dev/null
echo "$(now) done; failed steps: $NOK" >> "$OUT/steps.log"
progress "done $RUN_ID (failed steps: $NOK)"
push_out "X-M train ${RUN_ID} done (failed steps: $NOK)"
