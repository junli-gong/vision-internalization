#!/usr/bin/env bash
# RUN-0035 第 2 步（A800 约 50 小时）：0.8B 全部变体 × 3 个种子，加同预算对照（E3a 只用指代分割与深度，0.8B）。
# 按种子分批：先跑完种子 0 的全部变体（最早出信号），再种子 1、2。中断了重新运行同一条命令，已完成的跳过、没完成的接着跑。
PKG=$(cd "$(dirname "$0")" && pwd)
# shellcheck disable=SC1091
source "$PKG/common.sh" "$@"
cd "$CODE" || exit 1
for s in "${SEEDS[@]}"; do
  size_seed qwen3.5-0.8b "$s" 1
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) 0.8B seed $s done" >> "$OUT/milestones.txt"
  [ "$TINY" = 1 ] && break
done
progress "step2 done"
echo "[step2] 完成。请运行 bash step4_collect.sh $WORK 回传一次（含 0.8B 的检查点），然后 bash step3_train_4b.sh $WORK"
