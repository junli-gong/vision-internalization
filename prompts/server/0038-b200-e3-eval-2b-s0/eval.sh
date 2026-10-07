#!/usr/bin/env bash
# RUN-0038（B200）：2B 种子 0 的 E3 检查点评测（RUN-0036 的最终检查点；E3 预注册 v1 §4、D72-4）。
# 只做单轮的主读数与 T 文本：
#   - 网格（E3b、C1′）：eval_e3b --k-values 1；
#   - 稠密（E3a、C1、C2）：eval_e3a，四个任务加两项泄漏检查；
#   - T 文本（E3b、C3、E3a）：run_e1 正式运行（全部任务）、去偏字母读数、free 200 条、NYU 对调标记；
#   - 冻结对照（C1、C2、C1′）：引用基座的读数，只抽查前 192 条（批边界，DEV-0039），与 RUN-0027 逐条比对；C1′ 加 --mask-grid-tokens。
# 多轮与匹配计算量的评测另发任务，用本次的单轮墙钟估算用量。
#
# 用法（B200，服务器 CC 在固定 commit 的工作树里、与 CC 脱钩地启动）：
#   setsid nohup bash /dev/shm/visint/tmp/wt0038/prompts/server/0038-b200-e3-eval-2b-s0/eval.sh > $PROJ/logs/eval_0038.driver.log 2>&1 < /dev/null &
# 彩排（开发机）：TINY=1 WT=<仓库根> REPO=<彩排 clone> OUT=<REPO 里的目录> LOGS=… PY=… CK=<0036 彩排的 CK_FINAL> BASE_E1=<tiny base 的 run_e1 输出>/<模型> bash eval.sh
RUN_ID=0038
M=qwen3.5-2b; S=0
if [ "${TINY:-0}" = 1 ]; then
  : "${WT:?}" "${REPO:?}" "${OUT:?}" "${LOGS:?}" "${PY:?}" "${CK:?}" "${BASE_E1:?}"
  M=qwen3.5-0.8b
  T=(--tiny); D=(); TD=(); LIM=4; NOK=0
else
  export ROOT=/home/ubuntu/workspace/wzg/gjl
  export PROJ=$ROOT/vision-internalization
  WT=/dev/shm/visint/tmp/wt0038
  REPO=$PROJ/repo
  LOGS=$PROJ/logs
  OUT=$REPO/experiments/E3/0038_e3_eval_2b_s0
  CK=$PROJ/ckpt/0036
  PY=$PROJ/venv/bin/python
  BASE_E1=$REPO/experiments/E1/0027_e1_main/$M
  export HF_HOME=$ROOT/.hf HF_HUB_CACHE=/dev/shm/visint/hf-hub HF_HUB_OFFLINE=1 TMPDIR=/dev/shm/visint/tmp PIP_NO_CACHE_DIR=1
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  T=(); D=(--data-root "$PROJ/data"); TD=(--train-data-root /dev/shm/visint/train_data); LIM=192; NOK=0
  EV_CAP_S=$((16 * 3600))  # 任务书 §3 的上限：拿到 GPU 之后 16 小时
fi
# shellcheck disable=SC1091
source "$WT/scripts/server/eval_lib.sh"
cd "$WT" || exit 1

# 检查点（RUN-0036 的驱动按训练叶目录的结构存放；路径同 experiments/E3/0036_e3_train_2b/ckpt_ledger.tsv）
E3B=$CK/e3b/$M/seed$S/ckpt/final/model
C1P=$CK/c1prime/$M/seed$S/ckpt/final/model
C3=$CK/c3/$M/seed$S/ckpt/final/model
E3A=$CK/e3a/full/$M/seed$S/ckpt/final
C1=$CK/e3a/c1/$M/seed$S/ckpt/final
C2=$CK/e3a/c2/$M/seed$S/ckpt/final
for d in "$E3B" "$C1P" "$C3" "$E3A" "$C1" "$C2"; do [ -d "$d" ] || { echo "检查点不在：$d"; exit 2; }; done

claim_gpu "0038-b200-e3-eval-2b-s0.md" || { progress "blocked: gpu not free after ${CLAIM_MAX_S}s"; exit 1; }
echo "$(now) start commit $(git -C "$WT" rev-parse HEAD)" >> "$OUT/steps.log"

e1() {  # $1 = 名字，$2 = 检查点，$3 = 标签，其余 = run_e1 的参数
  local name=$1 ck=$2 lab=$3; shift 3
  step "$name" "$PY" scripts/run_e1.py --model "$M" ${T[@]+"${T[@]}"} ${D[@]+"${D[@]}"} --checkpoint "$ck" --checkpoint-label "$lab" --out "$OUT" "$@" \
    || NOK=$((NOK + 1))
}
ttext() {  # T 文本全套：$1 = 变体，$2 = 检查点（model/）
  local v=$1 ck=$2 lab="t_text/$1/seed$S"
  e1 "ttext_${v}_eval"        "$ck" "$lab" --task all --subset eval
  e1 "ttext_${v}_debiased"    "$ck" "$lab" --task all --subset debiased
  e1 "ttext_${v}_free"        "$ck" "$lab" --task all --subset free
  e1 "ttext_${v}_letter_swap" "$ck" "$lab" --task all --subset letter_swap
  push_out "2B s$S T-text $v (eval, debiased, free, letter_swap)"
}
spot() {  # 冻结对照的抽查：$1 = 变体，$2 = 检查点（model/），其余 = run_e1 的额外参数
  local v=$1 ck=$2 lab="t_text_spot/$1/seed$S"; shift 2
  e1 "spot_${v}" "$ck" "$lab" --task all --subset eval --limit "$LIM" "$@"
  mkdir -p "$OUT/spotcheck"
  step "spotcmp_${v}" "$PY" scripts/compare_ttext_spotcheck.py --base "$BASE_E1" --spot "$OUT/$lab/$M" \
    --out "$OUT/spotcheck/${v}_seed$S.json" ${T[@]+"${T[@]}"} ${D[@]+"${D[@]}"}  # 退出码 1 = 不完全相同，照记，由管理员看
  push_out "2B s$S spot check $v"
}

# 1. 网格单轮：E3b、C1′
step grid_e3b     "$PY" scripts/eval_e3b.py --checkpoint "$E3B" --model "$M" --variant e3b     --seed $S ${T[@]+"${T[@]}"} ${D[@]+"${D[@]}"} --out "$OUT" --k-values 1 || NOK=$((NOK + 1))
step grid_c1prime "$PY" scripts/eval_e3b.py --checkpoint "$C1P" --model "$M" --variant c1prime --seed $S ${T[@]+"${T[@]}"} ${D[@]+"${D[@]}"} --out "$OUT" --k-values 1 || NOK=$((NOK + 1))
push_out "2B s$S grid single-round (e3b, c1prime)"
# 2. 稠密：E3a、C1、C2
for pair in "full:$E3A" "c1:$C1" "c2:$C2"; do
  v=${pair%%:*}; ck=${pair#*:}
  step "dense_$v" "$PY" scripts/eval_e3a.py --checkpoint "$ck" --model "$M" --variant "$v" --seed $S ${T[@]+"${T[@]}"} ${D[@]+"${D[@]}"} ${TD[@]+"${TD[@]}"} --out "$OUT" \
    || NOK=$((NOK + 1))
done
push_out "2B s$S dense (e3a full, c1, c2)"
# 3. T 文本：E3b、C3、E3a
ttext e3b "$E3B"
ttext c3 "$C3"
ttext e3a "$E3A/model"
# 4. 冻结对照的抽查
spot c1 "$C1/model"
spot c2 "$C2/model"
spot c1prime "$C1P" --mask-grid-tokens

echo "$(now) done; failed steps: $NOK" >> "$OUT/steps.log"
progress "done 0038 (failed steps: $NOK)"
push_out "2B s$S eval done (failed steps: $NOK)"
