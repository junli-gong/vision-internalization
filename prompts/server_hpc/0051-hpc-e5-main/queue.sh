#!/usr/bin/env bash
# RUN-0051 的 E5 主比较队列（ITSC HPC，两台 A100 共用；E5 预注册 §8a v3、§8b；D83、D91、D93）。
# 起点（2B，B200 上训的，已传到 $VI/b200；E5 的种子 i 配起点的种子 i）：
#   e3b_c3 = E3b+C3（RUN-0040）；c3_warmup = C3+预热（RUN-0040）；base_warmup = 基座+预热（RUN-0036）；
#   e3b    = E3b（RUN-0036；§8b：探索性，只做 N = 0、1K、4K；种子 0、2，种子 1 在 A100 上没过合并核对，RUN-0048）。
# H5 的主判据是 e3b_c3 对 c3_warmup，网格一律结构强制解码（configs/e5_train.json）。
# **两台分摊**（D91-1）：两台节点跑同一个脚本，按组认领（Lustre 上原子的 mkdir），组按预注册的顺序排：
#   n0_all → small_s0 → small_s1 → small_s2 → mid_s0 → mid_s1 → mid_s2 → big_s0 → ad_s0 → ad_s1 → ad_s2
#   同一个数据量档、同一个种子的所有起点在同一组里，所以 H5 的比较都在机内完成。哪台先空就认领下一组。
#   N = 0 的 9 + 2 个读数放一组，最先跑（§8a 第 6 项）。
# 每个 E5 运行训完，紧接着在 test 上评一次（选定的检查点只评这一次，D77-3）。
# 驱动规则见 scripts/server/train_queue.sh；HPC 模式：TQ_NO_GIT=1，本节点自己的暂停文件，份额用 lfs quota 的差值。
# 铁律：任何情况下都不能取消或停止 Slurm 任务 157296、157297。
#
# 用法（在节点上经 srun 启动，见任务书）：
#   srun --jobid=<任务> --overlap -n1 env NODE=<hpcgpu06|hpcgpu07> PATH=… QUEUE_CAP_H=… bash $VI/wt/wt0051/prompts/server_hpc/0051-hpc-e5-main/queue.sh < /dev/null
# 彩排（开发机）：TINY=1 NODE=hpcgpu07 WT=… REPO=… OUT=… CK_TMP=… CK_FINAL=… LOGS=… PY=… ROOT36=<有 warmup、e3b> ROOT40=<有 e3b_c3、c3_warmup> bash queue.sh
NODE=${NODE:?NODE=hpcgpu06 或 hpcgpu07}
case "$NODE" in hpcgpu06|hpcgpu07) ;; *) echo "未知节点 $NODE" >&2; exit 2 ;; esac
RUN_ID=0051-$NODE
M=qwen3.5-2b
if [ "${TINY:-0}" = 1 ]; then
  : "${WT:?}" "${REPO:?}" "${OUT:?}" "${CK_TMP:?}" "${CK_FINAL:?}" "${LOGS:?}" "${PY:?}" "${ROOT36:?}" "${ROOT40:?}"
  CHUNK=${CHUNK:-3}; EXTRA=(--tiny); DATA=()
  M=qwen3.5-0.8b
  SMALL=(10); MID=(20); BIG=(); SEEDS=(${SEEDS:-0}); E3B_SEEDS=(${E3B_SEEDS:-0}); TIER_AD=10; EVAL_LIMIT=(--limit 2)
  export TQ_NO_GIT=1
else
  VI=/home/mail/chenyuma/visint
  WT=$VI/wt/wt0051
  REPO=$VI/repo
  LOGS=$VI/logs
  OUT=$REPO/experiments/E5/0051_e5_main_$NODE
  CK_TMP=$VI/ckpt/$NODE/tmp0051
  CK_FINAL=$VI/ckpt/$NODE/0051
  ROOT36=$VI/b200/ckpt/0036
  ROOT40=$VI/b200/ckpt/0040
  PY=$VI/env/bin/python
  CHUNK=1000
  export HF_HOME=$VI/hf HF_HUB_CACHE=$VI/hf/hub HF_HUB_OFFLINE=1
  export TMPDIR=$VI/tmp PIP_NO_CACHE_DIR=1
  export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 PYTHONHASHSEED=0
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export TRITON_CACHE_DIR=$VI/cache/$NODE/triton TORCHINDUCTOR_CACHE_DIR=$VI/cache/$NODE/inductor XDG_CACHE_HOME=$VI/cache/$NODE/xdg
  export TQ_NO_GIT=1
  export TQ_PAUSE=$LOGS/train.pause.$NODE
  export TQ_QUOTA_CMD="timeout 60 lfs quota -u chenyuma /home | awk 'NR==3{u=\$2; gsub(/\\*/,\"\",u); v=int((u-7059403936)/1048576)} END{print v+0}'"
  SHARE_STOP_GB=80
  MEM_STOP_GB=100
  EXTRA=(); DATA=(--data-root "$VI/b200/data")  # 起点检查点的合并核对用 E1 的 grid16 开发子集
  SMALL=(1000 4000); MID=(16000); BIG=(64000); SEEDS=(0 1 2); E3B_SEEDS=(0 2); TIER_AD=4000; EVAL_LIMIT=()
  QUEUE_CAP_H=${QUEUE_CAP_H:?任务书的上限}
fi
CLAIMS=$LOGS/e5_claims_0051
export SHARE_STOP_GB MEM_STOP_GB 2>/dev/null
# shellcheck disable=SC1091
source "$WT/scripts/server/train_queue.sh"
cd "$WT" || die "工作树不存在：$WT"
[ "$TINY" = 1 ] || [ "${CUDA_VISIBLE_DEVICES:-}" = 0 ] || die "不在 srun 的资源范围里（CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-未设}）"
mkdir -p "$CLAIMS"

STARTS=(e3b_c3 c3_warmup base_warmup)
start_ckpt() {  # $1 = 起点，$2 = 种子 → 起点的最终检查点（训练叶目录的结构，旁边有叶目录的 config.yaml）
  case "$1" in
    e3b_c3)      echo "$ROOT40/e3b_c3/$M/seed$2/ckpt/final/model" ;;
    c3_warmup)   echo "$ROOT40/c3_warmup/$M/seed$2/ckpt/final/model" ;;
    base_warmup) echo "$ROOT36/warmup/$M/seed$2/ckpt/final/model" ;;
    e3b)         echo "$ROOT36/e3b/$M/seed$2/ckpt/final/model" ;;
  esac
}
starts_for() {  # $1 = 种子 → 这个种子在 N = 0、1K、4K 上的起点（§8b 的 E3b 只在 E3B_SEEDS 里）
  local s=$1 e
  echo "${STARTS[@]}"
  for e in "${E3B_SEEDS[@]}"; do [ "$e" = "$s" ] && echo e3b; done
}
est_train() {  # 每条训练的预估（小时；RUN-0048 实测：每步 5.05 s，val 一次约 2.7 分钟，起点合并核对约 7 分钟；加 10%）
  case "$1" in 1000|4000) echo 2.05 ;; 16000) echo 3.25 ;; 64000) echo 12.6 ;; *) echo 0.01 ;; esac
}
e5() {  # 训一个 E5 运行，跑完紧接着在 test 上评一次。$1 = 起点，$2 = 数据量档，$3 = 种子，其余 = 额外参数（--tasks …）
  local init=$1 n=$2 s=$3; shift 3
  local tag="${init}_n${n}_s${s}" ck task=(--task all)
  [ $# -gt 0 ] && tag="${tag}_$(echo "$*" | sed 's/--tasks //; s/ //g')"
  case "$tag" in *_abcd) task=(--task e) ;; esac  # a–d 迁移：test 只评 e（§8a 第 4 项）
  ck=$(start_ckpt "$init" "$s")
  [ -d "$ck" ] || { note "$tag：起点检查点不在（$ck），跳过"; echo "$(now) $tag missing-start $ck" >> "$DRV/failed.txt"; return 1; }
  EST_H=$(est_train "$n")
  run "$tag" scripts/train_e5.py --init "$init" --start-checkpoint "$ck" --model $M --n-train "$n" --seed "$s" \
    ${DATA[@]+"${DATA[@]}"} "$@" || return 1
  case "${task[*]}" in "--task e") EST_H=0.2 ;; *) EST_H=0.5 ;; esac
  run_cmd "eval_$tag" "$PY" scripts/eval_e5_trained.py --checkpoint "$(ckpt_of "$tag")" --model $M --out "$OUT" "${task[@]}" \
    ${EXTRA[@]+"${EXTRA[@]}"} ${EVAL_LIMIT[@]+"${EVAL_LIMIT[@]}"}
}
n0() {  # N = 0（D81-1、D83-5）：起点不经 E5 训练直接评 test
  local init=$1 s=$2 ck
  ck=$(start_ckpt "$init" "$s")
  [ -d "$ck" ] || { note "n0_${init}_s$s：起点检查点不在（$ck），跳过"; echo "$(now) n0_${init}_s$s missing-start $ck" >> "$DRV/failed.txt"; return 1; }
  EST_H=0.5
  run_cmd "n0_${init}_s$s" "$PY" scripts/eval_e5_trained.py --start-only --init "$init" --start-checkpoint "$ck" --model $M --seed "$s" \
    ${DATA[@]+"${DATA[@]}"} --out "$OUT" --task all ${EXTRA[@]+"${EXTRA[@]}"} ${EVAL_LIMIT[@]+"${EVAL_LIMIT[@]}"}
}
group() {  # $1 = 组名：认领到就跑；别的节点已认领就跳过；本节点认领过、没跑完（驱动重启）就接着跑，跑过的运行由 run 跳过
  local g=$1 s n init owner
  if mkdir "$CLAIMS/$g" 2>/dev/null; then
    echo "$NODE $(now)" > "$CLAIMS/$g/node"
    note "认领组 $g"
  else
    [ -e "$CLAIMS/$g/done" ] && { note "组 $g 已跑完（$(cat "$CLAIMS/$g/done")），跳过"; return 0; }
    owner=$(cut -d' ' -f1 "$CLAIMS/$g/node" 2>/dev/null)
    if [ "$owner" != "$NODE" ]; then note "组 $g 已被 ${owner:-?} 认领，跳过"; return 0; fi
    note "组 $g 是本节点认领过、没跑完的，接着跑"
  fi
  case "$g" in
    n0_all)   for s in "${SEEDS[@]}"; do for init in $(starts_for "$s"); do n0 "$init" "$s"; done; done ;;
    small_s*) s=${g#small_s}; for n in "${SMALL[@]}"; do for init in $(starts_for "$s"); do e5 "$init" "$n" "$s"; done; done ;;
    mid_s*)   s=${g#mid_s};   for n in ${MID[@]+"${MID[@]}"}; do for init in "${STARTS[@]}"; do e5 "$init" "$n" "$s"; done; done ;;
    big_s*)   s=${g#big_s};   for n in ${BIG[@]+"${BIG[@]}"}; do for init in "${STARTS[@]}"; do e5 "$init" "$n" "$s"; done; done ;;
    ad_s*)    s=${g#ad_s};    for init in "${STARTS[@]}"; do e5 "$init" "$TIER_AD" "$s" --tasks a b c d; done ;;
  esac
  echo "$NODE $(now)" > "$CLAIMS/$g/done"
  note "组 $g 跑完"
}

note "队列开始：$NODE，commit $(git -C "$WT" rev-parse HEAD 2>/dev/null)，起点根 $ROOT40、$ROOT36，每段 $CHUNK 步，认领目录 $CLAIMS"
GRP_LIST=(n0_all)  # 不能叫 GROUPS：那是 bash 的内置变量（用户所属的组 ID），赋值会被忽略
for s in "${SEEDS[@]}"; do GRP_LIST+=("small_s$s"); done
for s in "${SEEDS[@]}"; do GRP_LIST+=("mid_s$s"); done
[ ${#BIG[@]} -gt 0 ] && GRP_LIST+=(big_s0)
for s in "${SEEDS[@]}"; do GRP_LIST+=("ad_s$s"); done
for g in "${GRP_LIST[@]}"; do group "$g"; done
finish
