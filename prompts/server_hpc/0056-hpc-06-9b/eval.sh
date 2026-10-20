#!/usr/bin/env bash
# RUN-0056（ITSC HPC hpcgpu06）的评测部分：9B 在 A100 上的 T 文本与像素读数（E3 预注册 §3、§12、§13；D91、D93-1），由 run.sh 依次调用：
#   a  阶段 1 的检查点：E3b+C3 种子 0 的 T 文本全套（eval、debiased、free、letter_swap）与结构强制网格（refseg、depth），
#      C3 种子 0 的 T 文本全套，以及 9B 基座的 T 文本全套（冻结对照在 A100 上的那一份，D93-1）；
#   b  阶段 2 的检查点：E3b 种子 0 的结构强制网格（refseg、depth），给像素干扰（§13）。
# 效应都在机内算（D93-1）：9B 的各臂都在 hpcgpu06 上训练与评测。评测前的合并核对照 E3 的 v1（D102-2）：被拒是规则，照实记，不绕过。
# 铁律：任何情况下都不能取消或停止 Slurm 任务 157296、157297。只用本节点自己的暂停文件。
#
# 用法：由 run.sh 在节点上经 srun 调用：bash eval.sh a|b
# 彩排（开发机）：TINY=1 WT=… REPO=… LOGS=… PY=… CK56=<有 c3、e3b_c3、e3b 的 0.8B> bash eval.sh a|b
PHASE=${1:?a 或 b}
RUN_ID=0056-$PHASE
NODE=hpcgpu06
M=qwen3.5-9b
if [ "${TINY:-0}" = 1 ]; then
  : "${WT:?}" "${REPO:?}" "${LOGS:?}" "${PY:?}" "${CK56:?}"
  M=qwen3.5-0.8b
  T=(--tiny); D=()
  export EV_NO_GIT=1
else
  VI=/home/mail/chenyuma/visint
  WT=$VI/wt/wt0056
  REPO=$VI/repo
  LOGS=$VI/logs
  PY=$VI/env/bin/python
  CK56=$VI/ckpt/$NODE/0056
  export HF_HOME=$VI/hf HF_HUB_CACHE=$VI/hf/hub HF_HUB_OFFLINE=1
  export TMPDIR=$VI/tmp PIP_NO_CACHE_DIR=1 NLTK_DATA=$VI/b200/nltk_data
  export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 PYTHONHASHSEED=0
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export TRITON_CACHE_DIR=$VI/cache/$NODE/triton TORCHINDUCTOR_CACHE_DIR=$VI/cache/$NODE/inductor XDG_CACHE_HOME=$VI/cache/$NODE/xdg
  T=(); D=(--data-root "$VI/b200/data")
  # eval_lib 的 HPC 模式（D91）：本节点自己的暂停文件，只等本节点 9B 队列的心跳；不碰 git；D95：只看 Lustre 的剩余空间
  export EV_PAUSE=$LOGS/train.pause.$NODE EV_STATUS_GLOB="train_0056-$NODE.status" EV_NO_GIT=1
  export EV_DF_PATH=$VI
  EV_CAP_S=$(awk -v h="${EV_CAP_H:?任务书 §3 给每个阶段的上限（小时）}" 'BEGIN{print int(h * 3600)}')
fi
OUT=$REPO/experiments/E3/0056_hpc_9b_eval_$NODE
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

claim_gpu "0056-hpc-06-9b.md ($PHASE)" || { progress "blocked: gpu not free after ${CLAIM_MAX_S}s"; exit 1; }
echo "$(now) start phase $PHASE commit $(git -C "$WT" rev-parse HEAD 2>/dev/null || echo unknown) node $NODE" >> "$OUT/steps.log"

if [ "$PHASE" = a ]; then
  mix=$CK56/e3b_c3/$M/seed0/ckpt/final/model; c3=$CK56/c3/$M/seed0/ckpt/final/model
  need "$mix" && { ttext e3b_c3 0 "$mix"; grid "grid_e3b_c3_s0_refseg" "$mix" e3b_c3 0 --task refseg
                   grid "grid_e3b_c3_s0_depth" "$mix" e3b_c3 0 --task depth; }
  need "$c3" && ttext c3 0 "$c3"
  push_out "9B s0: T-text C3 and E3b+C3, structured grid E3b+C3"
  ttext base 0 ""   # 9B 基座：冻结对照在 A100 上的那一份（种子号只用于命名；基座没有种子）
  push_out "9B base T-text battery"
elif [ "$PHASE" = b ]; then
  e3b=$CK56/e3b/$M/seed0/ckpt/final/model
  need "$e3b" && { grid "grid_e3b_s0_refseg" "$e3b" e3b 0 --task refseg
                   grid "grid_e3b_s0_depth" "$e3b" e3b 0 --task depth; }
  push_out "9B E3b s0: structured grid"
else
  echo "未知阶段 $PHASE" >&2; exit 2
fi

echo "$(now) done phase $PHASE; failed steps: $NOK" >> "$OUT/steps.log"
progress "done $RUN_ID (failed steps: $NOK)"
push_out "phase $PHASE done (failed steps: $NOK)"
