#!/usr/bin/env bash
# RUN-0052（ITSC HPC hpcgpu06）的评测部分：4B 在 A100 上的 H1 修订版读数与干扰随尺寸（E3 预注册 §12、§13；D91、D93）。
# 两个阶段，由 run.sh 依次调用：
#   a  RUN-0045 的检查点（训练已完成）：
#      - T 文本全套（eval、debiased、free、letter_swap）：C3 种子 0–2、E3b+C3 种子 0–2，以及 4B 基座（冻结对照在 A100 上的那一份，D93-1）；
#      - 网格（结构强制，D83-3）：E3b+C3 种子 0–2 的 refseg、depth。
#      种子 0 排最前，最早拿到一对完整的比较。
#   b  RUN-0052 的 E3b 检查点（queue.sh 训完之后）：网格（结构强制）E3b 种子 0–2 的 refseg、depth，给像素干扰（§13）。
# 效应都在机内算（D93-1）：4B 的各臂都在 hpcgpu06 上评。
# 铁律：任何情况下都不能取消或停止 Slurm 任务 157296、157297。只用本节点自己的暂停文件，不写 $LOGS/train.pause（RUN-0045 的驱动看它）。
#
# 用法：由 run.sh 在节点上经 srun 调用：bash eval.sh a|b
# 彩排（开发机）：TINY=1 WT=… REPO=… LOGS=… PY=… CK45=<有 c3、e3b_c3 的 0.8B（彩排用 0.8B）> CK52=<有 e3b 的 0.8B> bash eval.sh a|b
PHASE=${1:?a 或 b}
RUN_ID=0052-$PHASE
NODE=hpcgpu06
M=qwen3.5-4b
if [ "${TINY:-0}" = 1 ]; then
  : "${WT:?}" "${REPO:?}" "${LOGS:?}" "${PY:?}" "${CK45:?}" "${CK52:?}"
  M=qwen3.5-0.8b
  T=(--tiny); D=(); SEEDS=(${SEEDS:-0})
else
  VI=/home/mail/chenyuma/visint
  WT=$VI/wt/wt0052
  REPO=$VI/repo
  LOGS=$VI/logs
  PY=$VI/env/bin/python
  CK45=$VI/ckpt/$NODE/0045
  CK52=$VI/ckpt/$NODE/0052
  export HF_HOME=$VI/hf HF_HUB_CACHE=$VI/hf/hub HF_HUB_OFFLINE=1
  export TMPDIR=$VI/tmp PIP_NO_CACHE_DIR=1 NLTK_DATA=$VI/b200/nltk_data
  export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 PYTHONHASHSEED=0
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export TRITON_CACHE_DIR=$VI/cache/$NODE/triton TORCHINDUCTOR_CACHE_DIR=$VI/cache/$NODE/inductor XDG_CACHE_HOME=$VI/cache/$NODE/xdg
  T=(); D=(--data-root "$VI/b200/data"); SEEDS=(0 1 2)
  # eval_lib 的 HPC 模式（D91）；份额用 lfs quota 减 RUN-0044 开始时的基线（RUN-0047 报告 §5）
  export EV_PAUSE=$LOGS/train.pause.$NODE EV_STATUS_GLOB="train_0052-$NODE.status" EV_NO_GIT=1
  export EV_QUOTA_CMD="timeout 60 lfs quota -u chenyuma /home | awk 'NR==3{u=\$2; gsub(/\\*/,\"\",u); v=int((u-7059403936)/1048576)} END{print v+0}'"
  export EV_SHARE_STOP_GB=80 EV_DF_PATH=$VI
  EV_CAP_S=$(awk -v h="${EV_CAP_H:?任务书 §3 给每个阶段的上限（小时）}" 'BEGIN{print int(h * 3600)}')
fi
OUT=$REPO/experiments/E3/0052_hpc_eval_4b_$NODE
# shellcheck disable=SC1091
source "$WT/scripts/server/eval_lib.sh"
cd "$WT" || exit 1
[ "$TINY" = 1 ] || [ "${CUDA_VISIBLE_DEVICES:-}" = 0 ] || { progress "blocked: 不在 srun 的资源范围里"; exit 1; }
NOK=0

grid() {  # $1 = 名字，$2 = 检查点，$3 = 变体，$4 = 种子，其余 = 额外参数
  local name=$1 ck=$2 v=$3 s=$4; shift 4
  step "$name" "$PY" scripts/eval_e3b.py --checkpoint "$ck" --model "$M" --variant "$v" --seed "$s" ${T[@]+"${T[@]}"} ${D[@]+"${D[@]}"} \
    --out "$OUT" --k-values 1 --decode structured "$@" || NOK=$((NOK + 1))
}
ttext() {  # T 文本全套：$1 = 变体（base 时不给检查点），$2 = 种子，$3 = 检查点（model/；base 时为空）
  local v=$1 s=$2 ck=$3 sub
  local extra=()
  [ -n "$ck" ] && extra=(--checkpoint "$ck" --checkpoint-label "t_text/$v/seed$s")
  for sub in eval debiased free letter_swap; do
    step "ttext_${v}_s${s}_$sub" "$PY" scripts/run_e1.py --model "$M" ${T[@]+"${T[@]}"} ${D[@]+"${D[@]}"} ${extra[@]+"${extra[@]}"} \
      --out "$OUT" --task all --subset "$sub" || NOK=$((NOK + 1))
  done
}
need() { [ -d "$1" ] || { echo "$(now) 检查点不在：$1" >> "$OUT/steps.log"; NOK=$((NOK + 1)); return 1; }; }

claim_gpu "0052-hpc-06-4b.md ($PHASE)" || { progress "blocked: gpu not free after ${CLAIM_MAX_S}s"; exit 1; }
echo "$(now) start phase $PHASE commit $(git -C "$WT" rev-parse HEAD 2>/dev/null || echo unknown) node $NODE" >> "$OUT/steps.log"

if [ "$PHASE" = a ]; then
  for s in "${SEEDS[@]}"; do
    c3=$CK45/c3/$M/seed$s/ckpt/final/model; mix=$CK45/e3b_c3/$M/seed$s/ckpt/final/model
    need "$mix" && { ttext e3b_c3 "$s" "$mix"; grid "grid_e3b_c3_s${s}_refseg" "$mix" e3b_c3 "$s" --task refseg
                     grid "grid_e3b_c3_s${s}_depth" "$mix" e3b_c3 "$s" --task depth; }
    need "$c3" && ttext c3 "$s" "$c3"
    push_out "4B s$s: T-text C3 and E3b+C3, structured grid E3b+C3"
  done
  ttext base 0 ""   # 4B 基座：冻结对照在 A100 上的那一份（种子号只用于命名；基座没有种子）
  push_out "4B base T-text battery"
elif [ "$PHASE" = b ]; then
  for s in "${SEEDS[@]}"; do
    e3b=$CK52/e3b/$M/seed$s/ckpt/final/model
    need "$e3b" && { grid "grid_e3b_s${s}_refseg" "$e3b" e3b "$s" --task refseg
                     grid "grid_e3b_s${s}_depth" "$e3b" e3b "$s" --task depth; }
    push_out "4B E3b s$s: structured grid"
  done
else
  echo "未知阶段 $PHASE" >&2; exit 2
fi

echo "$(now) done phase $PHASE; failed steps: $NOK" >> "$OUT/steps.log"
progress "done $RUN_ID (failed steps: $NOK)"
push_out "phase $PHASE done (failed steps: $NOK)"
