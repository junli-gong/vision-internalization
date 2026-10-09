#!/usr/bin/env bash
# RUN-0039（B200）：E3b 的指令检查、结构强制重读、LoRA 缩放的折中曲线（2B 种子 0；D83-1；工具：DEV-0041）。
#   1. 指令检查（refseg，前 2,048 条，与 RUN-0038 的正常评测逐条配对）：none（去掉指令）、swap（换成同图另一物体的指代句）。
#   2. 结构强制重读（单轮）：E3b 与 C1′ 的 depth（全量 654）。
#   3. LoRA 缩放 s = 0.25、0.5、0.75：派生检查点 → 网格 refseg（受限解码，前 2,048 条）、网格 depth（结构强制，全量 654）、
#      T 文本（run_e1 全部任务，前 192 条：批边界，D80）。s = 1 的读数取 RUN-0038（同一批前 N 条），s = 0 的 T 文本取 RUN-0027。
# E3a 的诊断脚本要人选图与句子，由服务器 CC 在本脚本之后单独跑（任务书 §3）。
#
# 用法（B200，服务器 CC 在固定 commit 的工作树里、与 CC 脱钩地启动）：
#   setsid nohup bash /dev/shm/visint/tmp/wt0039/prompts/server/0039-b200-e3b-diag/eval.sh > $PROJ/logs/eval_0039.driver.log 2>&1 < /dev/null &
# 彩排（开发机）：TINY=1 WT=<仓库根> REPO=<彩排 clone> OUT=<REPO 里的目录> LOGS=… PY=… CK=<0036 彩排的 CK_FINAL> CK39=<派生检查点目录>
#   NORMAL=<彩排的正常评测 refseg 叶目录> bash eval.sh
RUN_ID=0039
M=qwen3.5-2b; S=0
if [ "${TINY:-0}" = 1 ]; then
  : "${WT:?}" "${REPO:?}" "${OUT:?}" "${LOGS:?}" "${PY:?}" "${CK:?}" "${CK39:?}" "${NORMAL:?}"
  M=qwen3.5-0.8b
  T=(--tiny); D=(); LIMG=8; LIMT=4; NOK=0
else
  export ROOT=/home/ubuntu/workspace/wzg/gjl
  export PROJ=$ROOT/vision-internalization
  WT=/dev/shm/visint/tmp/wt0039
  REPO=$PROJ/repo
  LOGS=$PROJ/logs
  OUT=$REPO/experiments/E3/0039_e3b_diag_2b_s0
  CK=$PROJ/ckpt/0036
  CK39=$PROJ/ckpt/0039
  PY=$PROJ/venv/bin/python
  NORMAL=$REPO/experiments/E3/0038_e3_eval_2b_s0/eval/e3b/$M/seed$S/refseg
  export HF_HOME=$ROOT/.hf HF_HUB_CACHE=/dev/shm/visint/hf-hub HF_HUB_OFFLINE=1 TMPDIR=/dev/shm/visint/tmp PIP_NO_CACHE_DIR=1
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  T=(); D=(--data-root "$PROJ/data"); LIMG=2048; LIMT=192; NOK=0
  EV_CAP_S=$((4 * 3600))  # 任务书 §4 的上限：拿到 GPU 之后 4 小时
fi
# shellcheck disable=SC1091
source "$WT/scripts/server/eval_lib.sh"
cd "$WT" || exit 1

E3B=$CK/e3b/$M/seed$S/ckpt/final/model
C1P=$CK/c1prime/$M/seed$S/ckpt/final/model
for d in "$E3B" "$C1P" "$NORMAL"; do [ -e "$d" ] || { echo "不在：$d"; exit 2; }; done

claim_gpu "0039-b200-e3b-diag.md" || { progress "blocked: gpu not free after ${CLAIM_MAX_S}s"; exit 1; }
echo "$(now) start commit $(git -C "$WT" rev-parse HEAD)" >> "$OUT/steps.log"

e3b() {  # $1 = 名字，其余 = eval_e3b 的参数
  local name=$1; shift
  step "$name" "$PY" scripts/eval_e3b.py --model "$M" --seed $S ${T[@]+"${T[@]}"} ${D[@]+"${D[@]}"} --out "$OUT" "$@" || NOK=$((NOK + 1))
}

# 1. 指令检查（E3b，refseg，受限解码，与 RUN-0038 逐条配对）
e3b instr_none --checkpoint "$E3B" --variant e3b --instruction-check none --limit $LIMG --normal-leaf "$NORMAL"
e3b instr_swap --checkpoint "$E3B" --variant e3b --instruction-check swap --limit $LIMG --normal-leaf "$NORMAL"
push_out "E3b 2B s$S instruction checks (none, swap)"

# 2. 结构强制重读（单轮）
e3b struct_e3b_depth    --checkpoint "$E3B" --variant e3b     --task depth  --k-values 1 --decode structured
e3b struct_c1p_depth    --checkpoint "$C1P" --variant c1prime --task depth  --k-values 1 --decode structured
push_out "2B s$S structured re-read (e3b, c1prime; depth)"
# refseg 的结构强制全量重读（E3b、C1′ 各约 50 分钟）留到 E3b+C3 的评测里一起做：新旧臂的网格比较两边都用结构强制（D83-3）

# 3. LoRA 缩放的折中曲线
for sc in 0.25 0.5 0.75; do
  tag=e3b_s$(echo "$sc" | tr -d .)
  leaf=$CK39/$tag/$M/seed$S
  if [ ! -d "$leaf/ckpt/final/model" ]; then
    step "scale_$tag" "$PY" scripts/scale_lora_ckpt.py --checkpoint "$E3B" --scale "$sc" --out "$leaf/ckpt/final/model" || { NOK=$((NOK + 1)); continue; }
  fi
  ck=$leaf/ckpt/final/model
  e3b "grid_${tag}_refseg" --checkpoint "$ck" --variant "$tag" --task refseg --k-values 1 --limit $LIMG
  e3b "grid_${tag}_depth"  --checkpoint "$ck" --variant "$tag" --task depth  --k-values 1 --decode structured
  step "ttext_${tag}" "$PY" scripts/run_e1.py --model "$M" ${T[@]+"${T[@]}"} ${D[@]+"${D[@]}"} --checkpoint "$ck" \
    --checkpoint-label "t_text/$tag/seed$S" --out "$OUT" --task all --subset eval --limit $LIMT || NOK=$((NOK + 1))
  push_out "2B s$S LoRA scale $sc (grid refseg, structured depth, T-text first $LIMT)"
done

echo "$(now) done; failed steps: $NOK" >> "$OUT/steps.log"
progress "done 0039 (failed steps: $NOK)"
push_out "2B s$S diag done (failed steps: $NOK)"
