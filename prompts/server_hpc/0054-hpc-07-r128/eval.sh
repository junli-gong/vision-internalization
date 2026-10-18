#!/usr/bin/env bash
# RUN-0054 的评测（ITSC HPC hpcgpu07；D99-1、§13a）：秩 128 的 C3 与 E3b+C3 × 3 的 T 文本全套，E3b+C3 × 3 的结构强制网格。
# 与 RUN-0050 的 eval.sh a 同一套命令，只换检查点根与结果目录（结果目录里变体名仍是 c3、e3b_c3；秩 128 由目录区分）。
# 网格评测前的合并核对照 E3 的判据 v1（D96 未改 E3；N37 待定）。
# 铁律：任何情况下都不能取消或停止 Slurm 任务 157296、157297。只用本节点自己的暂停文件。
# 彩排（开发机）：TINY=1 WT=… REPO=… LOGS=… PY=… CK54=<有 c3、e3b_c3 的 0.8B> bash eval.sh
RUN_ID=0054-eval
NODE=hpcgpu07
M=qwen3.5-0.8b
if [ "${TINY:-0}" = 1 ]; then
  : "${WT:?}" "${REPO:?}" "${LOGS:?}" "${PY:?}" "${CK54:?}"
  T=(--tiny); D=(); SEEDS=(${SEEDS:-0})
else
  VI=/home/mail/chenyuma/visint
  WT=$VI/wt/wt0054
  REPO=$VI/repo
  LOGS=$VI/logs
  PY=$VI/env/bin/python
  CK54=$VI/ckpt/$NODE/0054
  export HF_HOME=$VI/hf HF_HUB_CACHE=$VI/hf/hub HF_HUB_OFFLINE=1
  export TMPDIR=$VI/tmp PIP_NO_CACHE_DIR=1 NLTK_DATA=$VI/b200/nltk_data
  export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 PYTHONHASHSEED=0
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export TRITON_CACHE_DIR=$VI/cache/$NODE/triton TORCHINDUCTOR_CACHE_DIR=$VI/cache/$NODE/inductor XDG_CACHE_HOME=$VI/cache/$NODE/xdg
  T=(); D=(--data-root "$VI/b200/data"); SEEDS=(0 1 2)
  export EV_PAUSE=$LOGS/train.pause.$NODE EV_STATUS_GLOB="train_0054-$NODE.status" EV_NO_GIT=1
  export EV_DF_PATH=$VI   # D95：不查份额，只看 Lustre 剩余空间
  EV_CAP_S=$(awk -v h="${EV_CAP_H:?任务书给的上限（小时）}" 'BEGIN{print int(h * 3600)}')
fi
OUT=$REPO/experiments/E3/0054_r128_eval_$NODE
# shellcheck disable=SC1091
source "$WT/scripts/server/eval_lib.sh"
cd "$WT" || exit 1
[ "$TINY" = 1 ] || [ "${CUDA_VISIBLE_DEVICES:-}" = 0 ] || { progress "blocked: 不在 srun 的资源范围里"; exit 1; }
NOK=0

grid() {  # $1 = 名字，$2 = 检查点，$3 = 变体，$4 = 种子，其余 = 额外参数（同 RUN-0050）
  local name=$1 ck=$2 v=$3 s=$4; shift 4
  step "$name" "$PY" scripts/eval_e3b.py --checkpoint "$ck" --model "$M" --variant "$v" --seed "$s" ${T[@]+"${T[@]}"} ${D[@]+"${D[@]}"} \
    --out "$OUT" --k-values 1 --decode structured "$@" || NOK=$((NOK + 1))
}
ttext() {  # T 文本全套（同 RUN-0050）：$1 = 变体，$2 = 种子，$3 = 检查点（model/）
  local v=$1 s=$2 ck=$3 sub
  for sub in eval debiased free letter_swap; do
    step "ttext_${v}_s${s}_$sub" "$PY" scripts/run_e1.py --model "$M" ${T[@]+"${T[@]}"} ${D[@]+"${D[@]}"} \
      --checkpoint "$ck" --checkpoint-label "t_text/$v/seed$s" --out "$OUT" --task all --subset "$sub" || NOK=$((NOK + 1))
  done
}
need() { [ -d "$1" ] || { echo "$(now) 检查点不在：$1" >> "$OUT/steps.log"; NOK=$((NOK + 1)); return 1; }; }

claim_gpu "0054-hpc-07-r128.md (eval)" || { progress "blocked: gpu not free after ${CLAIM_MAX_S}s"; exit 1; }
echo "$(now) start eval commit $(git -C "$WT" rev-parse HEAD 2>/dev/null || echo unknown) node $NODE" >> "$OUT/steps.log"
for s in "${SEEDS[@]}"; do
  c3=$CK54/c3/$M/seed$s/ckpt/final/model; mix=$CK54/e3b_c3/$M/seed$s/ckpt/final/model
  need "$c3" && ttext c3 "$s" "$c3"
  need "$mix" && { ttext e3b_c3 "$s" "$mix"; grid "grid_e3b_c3_s${s}_refseg" "$mix" e3b_c3 "$s" --task refseg
                   grid "grid_e3b_c3_s${s}_depth" "$mix" e3b_c3 "$s" --task depth; }
  push_out "0.8B r128 s$s: T-text C3 and E3b+C3, structured grid E3b+C3"
done
echo "$(now) done eval; failed steps: $NOK" >> "$OUT/steps.log"
progress "done $RUN_ID (failed steps: $NOK)"
push_out "eval done (failed steps: $NOK)"
