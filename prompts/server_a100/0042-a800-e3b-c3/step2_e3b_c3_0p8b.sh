#!/usr/bin/env bash
# RUN-0042 第 2 步（A800 约 18 小时）：0.8B 的 E3b+C3 × 3 个种子（D83：H1 修订版的主臂）。中断了重新运行同一条命令。
PKG=$(cd "$(dirname "$0")" && pwd)
# shellcheck disable=SC1091
source "$PKG/common.sh" "$@"
cd "$CODE" || exit 1
for s in "${SEEDS[@]}"; do
  train "e3b_c3_qwen3.5-0.8b_s$s" scripts/train_e3b.py --model qwen3.5-0.8b --supervision grid+text --seed "$s" \
    --out "$OUT/0042_e3_train" --ckpt-root "$CKPT/train"
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) E3b+C3 0.8B seed $s done" >> "$OUT/milestones.txt"
  [ "$TINY" = 1 ] && break
done
progress "step2 done"
echo "[step2] 完成。请运行 bash step4_collect.sh $WORK 回传一次（0.8B 的结果与检查点），然后 bash step3_e3b_c3_4b.sh $WORK"
