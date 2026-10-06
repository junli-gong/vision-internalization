#!/usr/bin/env bash
# RUN-0035 第 1 步（约 1 小时）：换代码快照；记精度设置；跨机器一致性（2B 各 50 步，D72-4）；吞吐实测（0.8B、4B × 6 个变体，各 20 步）。
PKG=$(cd "$(dirname "$0")" && pwd)
# shellcheck disable=SC1091
source "$PKG/common.sh" "$@"
if [ "$TINY" != 1 ]; then
  if [ -d "$CODE" ]; then mv "$CODE" "$CODE.before0035.$(date +%s)"; fi
  tar -xzf "$PKG"/vision-internalization-*.tar.gz -C "$WORK" && cp "$PKG/VERSION.txt" "$CODE/PACKAGE_VERSION.txt"
  step update_code $?
  { date -u; hostname; cat "$CODE/PACKAGE_VERSION.txt"; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv; df -h "$WORK"; free -g; } > "$OUT/facts.txt" 2>&1
fi
cd "$CODE" || exit 1
python - > "$OUT/torch_backends.json" 2>&1 <<'PY'
import json, torch
b = torch.backends
conv = getattr(b.cudnn, "conv", None)
print(json.dumps({"torch": torch.__version__, "cudnn.allow_tf32": b.cudnn.allow_tf32, "cudnn.deterministic": b.cudnn.deterministic,
                  "cudnn.benchmark": b.cudnn.benchmark, "cudnn.conv.fp32_precision": getattr(conv, "fp32_precision", None),
                  "cuda.matmul.fp32_precision": getattr(b.cuda.matmul, "fp32_precision", None), "cuda_available": torch.cuda.is_available()}, indent=1))
PY
step torch_backends $?

# 跨机器一致性：同一个种子、同一批数据，A800 与 B200 各跑 50 步，比对损失曲线（D72-4；B200 那一半在 RUN-0034）
M_CONS=qwen3.5-2b; [ "$TINY" = 1 ] && M_CONS=qwen3.5-0.8b
train "consistency_e3b_${M_CONS}"   scripts/train_e3b.py --model "$M_CONS" --supervision grid --seed 0 \
  --out "$OUT/0035_e3_consistency" --ckpt-root "$CKPT/consistency" --stop-after-steps 50
train "consistency_e3a_${M_CONS}"   scripts/train_e3a.py --model "$M_CONS" --variant full --seed 0 \
  --out "$OUT/0035_e3_consistency" --ckpt-root "$CKPT/consistency" --stop-after-steps 50

# 吞吐：每个尺寸 × 变体 20 步（只为每步耗时与显存；检查点写在单独的目录，之后删掉）
for m in qwen3.5-0.8b qwen3.5-4b; do
  [ "$TINY" = 1 ] && [ "$m" != qwen3.5-0.8b ] && continue
  T="$OUT/0035_e3_throughput"; TC="$CKPT/throughput"
  train "tput_e3a_full_$m" scripts/train_e3a.py --model "$m" --variant full --seed 0 --out "$T" --ckpt-root "$TC" --stop-after-steps 20
  train "tput_e3a_c1_$m"   scripts/train_e3a.py --model "$m" --variant c1   --seed 0 --out "$T" --ckpt-root "$TC" --stop-after-steps 20
  train "tput_e3a_c2_$m"   scripts/train_e3a.py --model "$m" --variant c2   --seed 0 --out "$T" --ckpt-root "$TC" --stop-after-steps 20
  train "tput_e3b_$m"      scripts/train_e3b.py --model "$m" --supervision grid --seed 0 --out "$T" --ckpt-root "$TC" --stop-after-steps 20
  train "tput_c1prime_$m"  scripts/train_e3b.py --model "$m" --supervision grid --variant c1prime --seed 0 --out "$T" --ckpt-root "$TC" --stop-after-steps 20
  train "tput_c3_$m"       scripts/train_e3b.py --model "$m" --supervision text --seed 0 --out "$T" --ckpt-root "$TC" --stop-after-steps 20
done
rm -rf "$CKPT/throughput"
echo "[step1] 完成。下一步：bash step2_train_0p8b.sh $WORK（约 2–3 天，放 tmux）"
