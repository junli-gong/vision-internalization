#!/usr/bin/env bash
# RUN-0042 第 3 步（A800 约 39 小时，每个种子约 13 小时）：4B 的 E3b+C3 × 3 个种子（D83）。中断了重新运行同一条命令。
PKG=$(cd "$(dirname "$0")" && pwd)
# shellcheck disable=SC1091
source "$PKG/common.sh" "$@"
cd "$CODE" || exit 1
M=qwen3.5-4b; [ "$TINY" = 1 ] && M=qwen3.5-0.8b
for s in "${SEEDS[@]}"; do
  train "e3b_c3_qwen3.5-4b_s$s" scripts/train_e3b.py --model "$M" --supervision grid+text --seed "$s" \
    --out "$OUT/0042_e3_train" --ckpt-root "$CKPT/train"
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) E3b+C3 4B seed $s done" >> "$OUT/milestones.txt"
  [ "$TINY" = 1 ] && break
done
progress "step3 done"
echo "[step3] 完成。请运行 bash step4_collect.sh $WORK 回传。之后可以按原 RUN-0035 包的 step2、step3 跑消融（已完成的 C3 自动跳过）。"
