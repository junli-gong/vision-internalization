#!/usr/bin/env bash
# RUN-0042 第 1 步（约 20 分钟）：换代码快照（含 E3b+C3，D85）；记精度设置；吞吐实测（0.8B、4B 的 E3b+C3，各 20 步）。
# 跨机器一致性已在 RUN-0035 第 1 步做过，这里不重做。
PKG=$(cd "$(dirname "$0")" && pwd)
# shellcheck disable=SC1091
source "$PKG/common.sh" "$@"
if [ "$TINY" != 1 ]; then
  if [ -d "$CODE" ]; then mv "$CODE" "$CODE.before0042.$(date +%s)"; fi
  tar -xzf "$PKG"/vision-internalization-*.tar.gz -C "$WORK" && cp "$PKG/VERSION.txt" "$CODE/PACKAGE_VERSION.txt"
  step update_code $?
  { date -u; hostname; cat "$CODE/PACKAGE_VERSION.txt"; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv; df -h "$WORK"; free -g; } > "$OUT/facts.txt" 2>&1
fi
cd "$CODE" || exit 1
python -c "import json,torch;print(json.dumps({'torch':torch.__version__,'cuda':torch.cuda.is_available()}))" > "$OUT/torch_backends.json" 2>&1
step torch_backends $?
for m in qwen3.5-0.8b qwen3.5-4b; do
  [ "$TINY" = 1 ] && [ "$m" != qwen3.5-0.8b ] && continue
  train "tput_e3b_c3_$m" scripts/train_e3b.py --model "$m" --supervision grid+text --seed 0 \
    --out "$OUT/0042_e3_throughput" --ckpt-root "$CKPT/throughput" --stop-after-steps 20
done
rm -rf "$CKPT/throughput"
echo "[step1] 完成。请运行 bash step4_collect.sh $WORK 回传一次（吞吐），然后 bash step2_e3b_c3_0p8b.sh $WORK"
