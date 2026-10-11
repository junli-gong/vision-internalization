#!/usr/bin/env bash
# RUN-0048（ITSC HPC hpcgpu07）：迁移后排期用的实测（D81-2、D91）。B200 退役后，E5、2B 的 E3a+C3、0.8B 的评测与基座 E1 都改在 A100 上做；
# A100 上没有这几类命令的实测，按 B200 推算的倍数不可靠（生成受显存带宽限制，与训练的倍数不同）。这里每类跑一小段，量墙钟与单位耗时，
# 管理员据此写后面的长任务（每条的预估与上限）。顺带核对 E5 的三个 2B 起点在 A100 上过得了合并核对（训练开头自动做）。
#   1. E5 训练 20 步 × 3 个起点（2B，种子 0，1K 档）：每步耗时；起点的合并核对。
#   2. E5 的 val 评测一次（e3b_c3 起点，种子 0，每类 64 条 = 一次 val 选择的量；结构强制解码）：test 评测与 val 选择的单位耗时。
#      用 val（dev 子集），不碰 test：起点的 test 读数（N = 0）留给正式任务，只评一次。
#   3. 2B 的 E3a+C3 训练 20 步：每步耗时。
#   4. 0.8B 的评测（RUN-0045 的 E3b+C3 种子 0）：T 文本（dev 前 20 条、free 前 10 条），结构强制网格（refseg 前 100 条、depth 前 50 条）。
#   5. 0.8B 基座的 E1（dev 前 20 条）。
# 只在 RUN-0045 的 hpcgpu07 队列跑完之后启动（$LOGS/train_0045-hpcgpu07.done）。这里写的检查点放 $CKM，跑完删掉；结果只作计时用。
# 铁律：任何情况下都不能取消或停止 Slurm 任务 157296、157297（~/a100-hpc-guide.md 第 0 节）。
#
# 用法（hpcgpu07 上经 srun 启动，见任务书 §3）：
#   srun --jobid=157297 --overlap -n1 env PATH=… bash $VI/wt/wt0048/prompts/server_hpc/0048-hpc-measure-07/measure.sh < /dev/null
# 彩排（开发机）：TINY=1 WT=… REPO=… LOGS=… PY=… CKM=… ROOT40=<有 e3b_c3、c3_warmup> ROOT36=<有 warmup> CK45=<有 e3b_c3 的 0.8B> bash measure.sh
RUN_ID=0048
NODE=hpcgpu07
M2=qwen3.5-2b
M08=qwen3.5-0.8b
if [ "${TINY:-0}" = 1 ]; then
  : "${WT:?}" "${REPO:?}" "${LOGS:?}" "${PY:?}" "${CKM:?}" "${ROOT40:?}" "${ROOT36:?}" "${CK45:?}"
  M2=qwen3.5-0.8b
  T=(--tiny); D=(); TD=(); STEPS=3; L5=1; L1=2; LF=1; LR=2; LD=1
else
  VI=/home/mail/chenyuma/visint
  WT=$VI/wt/wt0048
  REPO=$VI/repo
  LOGS=$VI/logs
  PY=$VI/env/bin/python
  CKM=$VI/ckpt/$NODE/tmp0048
  ROOT40=$VI/b200/ckpt/0040
  ROOT36=$VI/b200/ckpt/0036
  CK45=$VI/ckpt/$NODE/0045
  export HF_HOME=$VI/hf HF_HUB_CACHE=$VI/hf/hub HF_HUB_OFFLINE=1
  export TMPDIR=$VI/tmp PIP_NO_CACHE_DIR=1 NLTK_DATA=$VI/b200/nltk_data
  export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 PYTHONHASHSEED=0
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export TRITON_CACHE_DIR=$VI/cache/$NODE/triton TORCHINDUCTOR_CACHE_DIR=$VI/cache/$NODE/inductor XDG_CACHE_HOME=$VI/cache/$NODE/xdg
  T=(); D=(--data-root "$VI/b200/data"); TD=(--data-root "$VI/data/train_data")
  STEPS=20; L5=64; L1=20; LF=10; LR=100; LD=50
  # eval_lib 的 HPC 模式（D91）：本节点自己的暂停文件，只等本节点的训练心跳；不碰 git；份额用 du 量我们自己的小目录（Lustre 上不对大目录 du）
  export EV_PAUSE=$LOGS/train.pause.$NODE EV_STATUS_GLOB="train_0045-$NODE.status" EV_NO_GIT=1
  export EV_QUOTA_CMD="echo \$(( 40 + \$(du -sk $VI/ckpt 2>/dev/null | awk '{print int(\$1/1048576)}') ))" EV_SHARE_STOP_GB=80
  export EV_DF_PATH=$VI
  EV_CAP_S=$((4 * 3600))  # 任务书 §2 的上限：拿到 GPU 之后 4 小时
fi
OUT=$REPO/experiments/E3/0048_hpc_measure_$NODE
OUT5=$REPO/experiments/E5/0048_hpc_measure_$NODE
# shellcheck disable=SC1091
source "$WT/scripts/server/eval_lib.sh"
cd "$WT" || exit 1
mkdir -p "$OUT5" "$CKM"
if [ "$TINY" != 1 ]; then
  [ -e "$LOGS/train_0045-$NODE.done" ] || { progress "blocked: RUN-0045 的 $NODE 队列还没跑完"; exit 1; }
  [ "${CUDA_VISIBLE_DEVICES:-}" = 0 ] || { progress "blocked: 不在 srun 的资源范围里"; exit 1; }
fi
NOK=0
ck_start() {  # $1 = 起点 → 种子 0 的起点检查点（训练叶目录的结构，旁边有叶目录的 config.yaml）
  case "$1" in
    e3b_c3)      echo "$ROOT40/e3b_c3/$M2/seed0/ckpt/final/model" ;;
    c3_warmup)   echo "$ROOT40/c3_warmup/$M2/seed0/ckpt/final/model" ;;
    base_warmup) echo "$ROOT36/warmup/$M2/seed0/ckpt/final/model" ;;
  esac
}
CK08=$CK45/e3b_c3/$M08/seed0/ckpt/final/model
for p in "$(ck_start e3b_c3)" "$(ck_start c3_warmup)" "$(ck_start base_warmup)" "$CK08"; do
  [ -d "$p" ] || { progress "blocked: 检查点不在：$p"; exit 1; }
done

claim_gpu "0048-hpc-measure-07.md" || { progress "blocked: gpu not free after ${CLAIM_MAX_S}s"; exit 1; }
echo "$(now) start commit $(git -C "$WT" rev-parse HEAD 2>/dev/null || echo unknown) node $NODE" >> "$OUT/steps.log"

# 1. E5 训练 20 步 × 3 个起点
for init in e3b_c3 c3_warmup base_warmup; do
  step "e5_train${STEPS}_$init" "$PY" scripts/train_e5.py --init "$init" --start-checkpoint "$(ck_start "$init")" --model $M2 \
    --n-train "$( [ "$TINY" = 1 ] && echo 10 || echo 1000 )" --seed 0 ${T[@]+"${T[@]}"} ${D[@]+"${D[@]}"} \
    --out "$OUT5" --ckpt-root "$CKM/e5" --stop-after-steps "$STEPS" || NOK=$((NOK + 1))
done
push_out "E5 train ${STEPS} steps x 3 starts"
# 2. E5 的 val 评测一次（起点，结构强制解码，每类 L5 条）
step "e5_val${L5}_e3b_c3" "$PY" scripts/eval_e5_trained.py --start-only --init e3b_c3 --start-checkpoint "$(ck_start e3b_c3)" --model $M2 \
  --seed 0 ${T[@]+"${T[@]}"} ${D[@]+"${D[@]}"} --out "$OUT5" --task all --subset dev --limit "$L5" || NOK=$((NOK + 1))
push_out "E5 val pass on e3b_c3 start"
# 3. 2B 的 E3a+C3 训练 20 步
step "e3a_c3_train${STEPS}" "$PY" scripts/train_e3a.py --model $M2 --variant full --with-text --seed 0 ${T[@]+"${T[@]}"} ${TD[@]+"${TD[@]}"} \
  --out "$OUT" --ckpt-root "$CKM/e3a_c3" --stop-after-steps "$STEPS" || NOK=$((NOK + 1))
push_out "E3a+C3 2B train ${STEPS} steps"
# 4. 0.8B 的评测（RUN-0045 的 E3b+C3 种子 0）
step "ttext_e3b_c3_s0_dev${L1}" "$PY" scripts/run_e1.py --model $M08 ${T[@]+"${T[@]}"} ${D[@]+"${D[@]}"} --checkpoint "$CK08" \
  --checkpoint-label t_text/e3b_c3/seed0 --out "$OUT/e1" --task all --subset dev --limit "$L1" || NOK=$((NOK + 1))
step "ttext_e3b_c3_s0_free${LF}" "$PY" scripts/run_e1.py --model $M08 ${T[@]+"${T[@]}"} ${D[@]+"${D[@]}"} --checkpoint "$CK08" \
  --checkpoint-label t_text/e3b_c3/seed0 --out "$OUT/e1" --task all --subset free --limit "$LF" || NOK=$((NOK + 1))
step "grid_e3b_c3_s0_refseg${LR}" "$PY" scripts/eval_e3b.py --checkpoint "$CK08" --model $M08 --variant e3b_c3 --seed 0 \
  ${T[@]+"${T[@]}"} ${D[@]+"${D[@]}"} --out "$OUT" --k-values 1 --decode structured --task refseg --limit "$LR" || NOK=$((NOK + 1))
step "grid_e3b_c3_s0_depth${LD}" "$PY" scripts/eval_e3b.py --checkpoint "$CK08" --model $M08 --variant e3b_c3 --seed 0 \
  ${T[@]+"${T[@]}"} ${D[@]+"${D[@]}"} --out "$OUT" --k-values 1 --decode structured --task depth --limit "$LD" || NOK=$((NOK + 1))
push_out "0.8B E3b+C3 s0 eval timing"
# 5. 0.8B 基座的 E1
step "e1_base_dev${L1}" "$PY" scripts/run_e1.py --model $M08 ${T[@]+"${T[@]}"} ${D[@]+"${D[@]}"} \
  --out "$OUT/e1base" --task all --subset dev --limit "$L1" || NOK=$((NOK + 1))

rm -rf "$CKM"
echo "$(now) done; failed steps: $NOK" >> "$OUT/steps.log"
progress "done 0048 (failed steps: $NOK)"
push_out "measurement done (failed steps: $NOK)"
