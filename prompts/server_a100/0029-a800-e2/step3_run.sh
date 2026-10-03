#!/usr/bin/env bash
# RUN-0029 第 3 步（GPU，约一天）：预跑 → 收敛判定 → Qwen 四个尺寸的正式运行 → 噪声底（E2 预注册 v1 §5）。
# 中断了重新运行同一条命令即可：每个任务从检查点接着跑，已完成的直接跳过。
PKG=$(cd "$(dirname "$0")" && pwd)
# shellcheck disable=SC1091
source "$PKG/common.sh" "$@"
cd "$CODE" || exit 1
if ! python -c "import json,sys; sys.exit(json.load(open('$OUT/budget_check.json'))['exit_code'])" 2>/dev/null; then
  echo "第 2 步的判定不是 GO（或没跑），不运行第 3 步。"; step run_refused_no_go 3; exit 3
fi
RUN="$OUT/0029_e2_qwen"; CKPT="$WORK/e2_ckpt_0029"
LIMIT_S=$(python -c "print(int($CAP * 1.2 * 3600))")   # 累计超过上限的 120% 就在当前模型跑完后停
T0_FILE="$OUT/step3_t0"; [ -f "$T0_FILE" ] || date +%s > "$T0_FILE"; T0=$(cat "$T0_FILE")
run_e2() {  # $1 = 模型，$2 = 任务，$3 = 输出目录，$4 = 检查点目录
  progress "running $1 $2"
  python scripts/run_e2.py --model "$1" --task "$2" "${REFS[@]}" ${DATA_ARGS[@]+"${DATA_ARGS[@]}"} ${TFLAG[@]+"${TFLAG[@]}"} \
    --out "$3" --ckpt-root "$4" --resume >> "$OUT/run_$(basename "$3")_$1.log" 2>&1
  local rc=$?; step "run_$(basename "$3")_$1_$2" $rc; return $rc
}
over_budget() { [ $(( $(date +%s) - T0 )) -gt "$LIMIT_S" ]; }

# 1. 预跑：0.8B 的深度（结果就是正式结果的一部分，不重跑）
run_e2 qwen3.5-0.8b depth "$RUN" "$CKPT" || { echo "预跑出错，停。"; exit 2; }
python scripts/mgr/e2_prerun_check.py "$RUN/qwen3.5-0.8b/depth" --out "$OUT/prerun_check.json" > "$OUT/prerun_check.log" 2>&1
rc=$?; step prerun_check $rc
if [ $rc -ne 0 ] && [ "$TINY" != 1 ]; then   # 彩排的小模型不会收敛，只核对流程
  echo "预跑没有通过收敛判定（见 prerun_check.json），停。请运行 step4_collect.sh 回传。"; progress "stopped prerun_check rc=$rc"; exit 3
fi

# 2. 正式运行：0.8B → 2B → 4B → 9B，每个模型四个任务
for m in "${QWEN[@]}"; do
  [ "$TINY" = 1 ] && [ "$m" != qwen3.5-0.8b ] && continue
  run_e2 "$m" all "$RUN" "$CKPT"
  if over_budget; then echo "累计超过上限的 120%，停。"; step blocked_over_budget 4; progress "stopped over budget after $m"; exit 4; fi
done

# 3. 噪声底：0.8B 四个任务，同配置重跑一次（另一个输出目录与检查点目录）
run_e2 qwen3.5-0.8b all "$OUT/0029_e2_qwen_noise" "$WORK/e2_ckpt_0029_noise"
progress "done"
echo "[step3] 完成。下一步：bash step4_collect.sh $WORK"
