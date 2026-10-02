#!/usr/bin/env bash
# RUN-0026 第 2 步（GPU，不需要联网）：去重报告、训练接口核对、E2 算力估算、E3b 训练吞吐的短实测。约 2–3 小时。
# 用法：bash step2_checks.sh <工作目录>       建议放进 tmux 或 nohup
set -uo pipefail
WORK=${1:?用法: bash step2_checks.sh <工作目录>}
OUT="$WORK/results_0026"; mkdir -p "$OUT"
step() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $1 rc=$2" | tee -a "$OUT/steps.log"; }
# shellcheck disable=SC1091
source "$WORK/venv/bin/activate"
export HF_HOME=${HF_HOME:-$WORK/hf} HF_HUB_OFFLINE=1 TMPDIR=$WORK/tmp PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NCPU=$(nproc 2>/dev/null || echo 8); [ "$NCPU" -gt 16 ] && NCPU=16; export OMP_NUM_THREADS=$NCPU MKL_NUM_THREADS=$NCPU
cd "$WORK/vision-internalization"
DATA="$WORK/data"
{ date -u; hostname; cat PACKAGE_VERSION.txt; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv; df -h "$WORK"; } > "$OUT/facts.txt" 2>&1

# 1. 去重（硬约束 7）：本机训练数据的哈希与 B200 评测数据的哈希清单（随代码快照）比对，剔除清单写进 $DATA/dedupe/（schema 2）
python scripts/data/dedupe_report.py --export-hashes "$DATA" --out "$OUT/train_hashes_a800.json" > "$OUT/dedupe_export.log" 2>&1
step dedupe_export $?
python scripts/data/dedupe_report.py --train-root "$DATA" --eval-manifest experiments/P0/0023_eval_hashes/eval_hashes_b200.json \
  --out "$DATA/dedupe" > "$OUT/dedupe_report.log" 2>&1
step dedupe_report $?
mkdir -p "$OUT/dedupe" && cp "$DATA"/dedupe/* "$OUT/dedupe/" 2>/dev/null

# 2. 训练接口核对（v1 判据，D50；要 E1 的 RefCOCO）
if [ -f "$DATA/refcoco/MANIFEST.json" ]; then
  for m in qwen3.5-0.8b qwen3.5-9b; do
    python scripts/check_training_interfaces.py --model "$m" --checks restricted merge --n 200 --data-root "$DATA" \
      --out "$OUT/train_iface_$m.json" > "$OUT/train_iface_$m.log" 2>&1; step "train_iface_$m" $?
  done
else
  step train_iface_skipped_no_refcoco 0
fi

# 3. E2 的算力估算（每个模型几分钟；法向的数据许可未定，不加 --enable-normals）
for m in qwen3.5-0.8b qwen3.5-2b qwen3.5-4b qwen3.5-9b gemma-4-e2b-it gemma-4-e4b-it internvl3.5-2b-hf internvl3.5-4b-hf; do
  python scripts/run_e2.py --estimate --model "$m" --task all --reference post_proj --data-root "$DATA" \
    --out "$OUT/e2_estimate" > "$OUT/e2_estimate_$m.log" 2>&1; step "e2_estimate_$m" $?
done

# 4. E3b 训练吞吐的短实测（每种 20 步；只为测每步耗时与显存，不是正式训练；检查点写在 $WORK/e3_probe_ckpt，不回传）
for m in qwen3.5-0.8b qwen3.5-9b; do
  for sup in grid text; do
    python scripts/train_e3b.py --model "$m" --supervision "$sup" --seed 0 --data-root "$DATA" \
      --out "$OUT/e3b_probe" --ckpt-root "$WORK/e3_probe_ckpt" --stop-after-steps 20 \
      > "$OUT/e3b_probe_${m}_${sup}.log" 2>&1; step "e3b_probe_${m}_${sup}" $?
  done
done
df -h "$WORK" >> "$OUT/facts.txt"
echo "[step2] 完成。下一步：bash step3_collect.sh $WORK"
