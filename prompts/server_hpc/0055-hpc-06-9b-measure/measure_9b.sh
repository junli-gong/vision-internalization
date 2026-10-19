#!/usr/bin/env bash
# RUN-0055 第二部分（ITSC HPC hpcgpu06）：9B 在 A100-40GB 上的实测（N38 答复前的默认动作）。9B 在这张卡上能不能训、每步多久、评测的
# 批配置放不放得下，都还没有实测（B200 上 9B 的 LoRA 训练批 8 峰值 47.8 GB，docs/env_report.md §7）。这里每类跑一小段，量显存与单位耗时，
# 管理员据此写 9B 的用量估计与任务书（按 12 小时线报批）。结果只作计时与显存用，不进任何结论。
#   1. 训练 20 步 × 3 臂（E3b+C3、E3b、C3；种子 0；E3 的正式配置 configs/e3b_train.json，微批 4 × 累积 4）。
#      某一臂显存不够（日志里有 out of memory）时，同一臂再用"微批 2 × 累积 8"跑 20 步（每步仍是 16 条；配置在 $CKM 里现生成，
#      只改这两项与 version，只作实测，不进正式运行）。
#   2. 网格评测（结构强制，E3 的正式评测设置；含评测前的合并核对 v1）：用第 1 步 E3b+C3（没有就用 E3b）的 20 步检查点，refseg 前 100 条、depth 前 50 条。
#   3. T 文本（run_e1，E1 冻结的批配置）：9B 基座 dev 前 20 条、free 前 10 条；E3b+C3（没有就用 C3）的 20 步检查点 dev 前 20 条（含加载与合并核对）。
# 只在 RUN-0051 的 06 队列跑完之后启动（$LOGS/train_0051-hpcgpu06.done，big_s0 有 done）。检查点放 $CKM，跑完删掉。
# 铁律：任何情况下都不能取消或停止 Slurm 任务 157296、157297（~/a100-hpc-guide.md 第 0 节）。只用本节点自己的暂停文件。
#
# 用法（hpcgpu06 上经 srun 启动，见任务书 §2）：
#   srun --jobid=157296 --overlap -n1 env PATH=… EV_CAP_H=5 bash $VI/wt/wt0055/prompts/server_hpc/0055-hpc-06-9b-measure/measure_9b.sh < /dev/null
# 彩排（开发机）：TINY=1 WT=… REPO=… LOGS=… PY=… CKM=… [FORCE_FALLBACK=1] bash measure_9b.sh
RUN_ID=0055
NODE=hpcgpu06
M=qwen3.5-9b
if [ "${TINY:-0}" = 1 ]; then
  : "${WT:?}" "${REPO:?}" "${LOGS:?}" "${PY:?}" "${CKM:?}"
  M=qwen3.5-0.8b
  T=(--tiny); D=(); TD=(); STEPS=3; L1=2; LF=1; LR=2; LD=1
  export EV_NO_GIT=1
else
  VI=/home/mail/chenyuma/visint
  WT=$VI/wt/wt0055
  REPO=$VI/repo
  LOGS=$VI/logs
  PY=$VI/env/bin/python
  CKM=$VI/ckpt/$NODE/tmp0055
  TRAIN=$VI/data/train_data
  export HF_HOME=$VI/hf HF_HUB_CACHE=$VI/hf/hub HF_HUB_OFFLINE=1
  export TMPDIR=$VI/tmp PIP_NO_CACHE_DIR=1 NLTK_DATA=$VI/b200/nltk_data
  export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 PYTHONHASHSEED=0
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export TRITON_CACHE_DIR=$VI/cache/$NODE/triton TORCHINDUCTOR_CACHE_DIR=$VI/cache/$NODE/inductor XDG_CACHE_HOME=$VI/cache/$NODE/xdg
  T=(); D=(--data-root "$VI/b200/data"); TD=(--data-root "$TRAIN")
  STEPS=20; L1=20; LF=10; LR=100; LD=50
  # eval_lib 的 HPC 模式（D91）：本节点自己的暂停文件，只等本节点 E5 队列的心跳；不碰 git；D95：不查份额，只看 Lustre 的剩余空间
  export EV_PAUSE=$LOGS/train.pause.$NODE EV_STATUS_GLOB="train_0051-$NODE.status" EV_NO_GIT=1
  export EV_DF_PATH=$VI
  EV_CAP_S=$(awk -v h="${EV_CAP_H:?任务书 §2 的上限（小时）}" 'BEGIN{print int(h * 3600)}')
fi
OUT=$REPO/experiments/E3/0055_hpc_measure_9b_$NODE
OUTE1=$REPO/experiments/E3/0055_hpc_measure_9b_e1_$NODE
# shellcheck disable=SC1091
source "$WT/scripts/server/eval_lib.sh"
cd "$WT" || exit 1
mkdir -p "$CKM" "$OUTE1"
if [ "$TINY" != 1 ]; then
  [ -e "$LOGS/train_0051-$NODE.done" ] || { progress "blocked: RUN-0051 的 $NODE 队列还没跑完"; exit 1; }
  [ -e "$LOGS/e5_claims_0051/big_s0/done" ] || { progress "blocked: e5_claims_0051/big_s0 没有 done"; exit 1; }
  [ "${CUDA_VISIBLE_DEVICES:-}" = 0 ] || { progress "blocked: 不在 srun 的资源范围里"; exit 1; }
  [ -d "$TRAIN" ] || { progress "blocked: 训练数据不在：$TRAIN"; exit 1; }
  # 权重：第一部分（fetch_9b.sh）下载并核对过；这里只读本地快照再核一次（local_files_only）
  "$PY" experiments/P0/0009_ifeval_rescore/probe_scripts/p12_verify_weights.py $M >> "$OUT/weights_verify.log" 2>&1 \
    || { progress "blocked: $M 的权重核对没过（$OUT/weights_verify.log）"; exit 1; }
fi
NOK=0

claim_gpu "0055-hpc-06-9b-measure.md" || { progress "blocked: gpu not free after ${CLAIM_MAX_S}s"; exit 1; }
echo "$(now) start commit $(git -C "$WT" rev-parse HEAD 2>/dev/null || echo unknown) node $NODE model $M" >> "$OUT/steps.log"

ck_of() {  # $1 = 变体 → 它在 $CKM 下的检查点（--stop-after-steps 停下时是 latest/model；跑完了是 final/model）；没有就空
  local d=$CKM/$1/$M/seed0
  if [ -d "$d/latest/model" ]; then echo "$d/latest/model"; elif [ -d "$d/final/model" ]; then echo "$d/final/model"; fi
}
oom() { grep -qiE 'out of memory|OutOfMemoryError' "$OUT/logs/$1.log" 2>/dev/null; }
CFG2=$CKM/e3b_train_micro2.json
make_micro2() {  # 微批 2 × 累积 8（每步 16 条不变）；只改这两项与 version
  "$PY" - "$WT/configs/e3b_train.json" "$CFG2" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
d["batch"]["micro_batch_size"], d["batch"]["grad_accum_steps"] = 2, 8
d["version"] = str(d["version"]) + "-probe-micro2"
json.dump(d, open(sys.argv[2], "w"), ensure_ascii=False, indent=1)
PYEOF
}

# 1. 训练 20 步 × 3 臂（显存不够时同一臂再用微批 2 跑一次）
for pair in "e3b_c3:grid+text" "e3b:grid" "c3:text"; do
  v=${pair%%:*}; sup=${pair#*:}
  name="train${STEPS}_${v}_micro4"
  step "$name" "$PY" scripts/train_e3b.py --model $M --supervision "$sup" --seed 0 ${T[@]+"${T[@]}"} ${TD[@]+"${TD[@]}"} \
    --out "$OUT" --ckpt-root "$CKM" --stop-after-steps "$STEPS" && [ "${FORCE_FALLBACK:-0}" != 1 ] && continue
  if oom "$name" || [ "${FORCE_FALLBACK:-0}" = 1 ]; then
    echo "$(now) $name 显存不够（或彩排强制），同一臂改用微批 2 × 累积 8 再跑" >> "$OUT/steps.log"
    [ -f "$CFG2" ] || make_micro2 || { echo "$(now) 生成微批 2 的配置失败" >> "$OUT/steps.log"; NOK=$((NOK + 1)); continue; }
    rm -rf "${CKM:?}/$v"
    step "train${STEPS}_${v}_micro2" "$PY" scripts/train_e3b.py --model $M --supervision "$sup" --seed 0 ${T[@]+"${T[@]}"} ${TD[@]+"${TD[@]}"} \
      --config "$CFG2" --out "$OUT/micro2" --ckpt-root "$CKM" --stop-after-steps "$STEPS" || NOK=$((NOK + 1))
  else
    NOK=$((NOK + 1))
  fi
done
push_out "9B train ${STEPS} steps x 3 arms"

# 2. 网格评测（结构强制）：E3b+C3 的检查点，没有就用 E3b 的
CKG=$(ck_of e3b_c3); VG=e3b_c3
[ -n "$CKG" ] || { CKG=$(ck_of e3b); VG=e3b; }
if [ -n "$CKG" ]; then
  echo "$(now) 网格评测用 $VG 的检查点 $CKG" >> "$OUT/steps.log"
  step "grid_${VG}_refseg$LR" "$PY" scripts/eval_e3b.py --checkpoint "$CKG" --model $M --variant "$VG" --seed 0 \
    ${T[@]+"${T[@]}"} ${D[@]+"${D[@]}"} --out "$OUT" --k-values 1 --decode structured --task refseg --limit "$LR" || NOK=$((NOK + 1))
  step "grid_${VG}_depth$LD" "$PY" scripts/eval_e3b.py --checkpoint "$CKG" --model $M --variant "$VG" --seed 0 \
    ${T[@]+"${T[@]}"} ${D[@]+"${D[@]}"} --out "$OUT" --k-values 1 --decode structured --task depth --limit "$LD" || NOK=$((NOK + 1))
else
  echo "$(now) 没有可用的网格检查点（E3b+C3、E3b 都没训成），跳过网格评测" >> "$OUT/steps.log"; NOK=$((NOK + 1))
fi
push_out "9B grid eval timing"

# 3. T 文本：基座 dev、free；检查点 dev（E3b+C3，没有就用 C3）
step "e1_base_dev$L1" "$PY" scripts/run_e1.py --model $M ${T[@]+"${T[@]}"} ${D[@]+"${D[@]}"} \
  --out "$OUTE1" --task all --subset dev --limit "$L1" || NOK=$((NOK + 1))
step "e1_base_free$LF" "$PY" scripts/run_e1.py --model $M ${T[@]+"${T[@]}"} ${D[@]+"${D[@]}"} \
  --out "$OUTE1" --task all --subset free --limit "$LF" || NOK=$((NOK + 1))
CKT=$(ck_of e3b_c3); VT=e3b_c3
[ -n "$CKT" ] || { CKT=$(ck_of c3); VT=c3; }
if [ -n "$CKT" ]; then
  step "ttext_${VT}_dev$L1" "$PY" scripts/run_e1.py --model $M ${T[@]+"${T[@]}"} ${D[@]+"${D[@]}"} --checkpoint "$CKT" \
    --checkpoint-label "t_text/$VT/seed0" --out "$OUTE1" --task all --subset dev --limit "$L1" || NOK=$((NOK + 1))
else
  echo "$(now) 没有可用的文本检查点（E3b+C3、C3 都没训成），跳过检查点的 T 文本" >> "$OUT/steps.log"; NOK=$((NOK + 1))
fi

rm -rf "$CKM"
echo "$(now) done; failed steps: $NOK" >> "$OUT/steps.log"
progress "done 0055 (failed steps: $NOK)"
push_out "9B measurement done (failed steps: $NOK)"
