#!/usr/bin/env bash
# RUN-0062（ITSC HPC）：W1 去留判定的开发-检查集运行（D103 §三；W1 预注册 §8、§10b；协议已冻结，configs/w1.json 的 frozen = true）。
#   1. 两条基线与 W1-0、W1-F：开发-检查集 500 句（同一模型、同一套协议，D103 §三"两条基线都要在这 500 句上补跑"）。
#   2. 9B 完整工作台（W1-1），开发-检查集 500 句。
#   3. w1_gate.py：去留判定（规则写死在脚本里；判定点核对全部为真时才是正式结论）。
#   4. 冻结的协议在开发-调参集上跑定位 64 条、计数 49 条（只报告，W1 预注册 §10b 第 5 条）。
# 全程每秒采样一次显存，写 <OUT>/gpu_mem.csv。GPU 由 claim_gpu 从本节点训练队列的段边界接过来（本节点自己的暂停文件），退出时交还。
# 铁律：任何情况下都不能取消或停止 Slurm 任务 157296、157297。只用本节点自己的暂停文件。
#
# 用法（经 srun 启动，见任务书）：srun --jobid=<本节点的任务> --overlap -n1 env PATH=… NODE=hpcgpu07 TQ_STATUS=train_0057-hpcgpu07.status EV_CAP_H=… bash …/devcheck.sh
# 彩排（开发机）：TINY=1 WT=… REPO=… LOGS=… PY=… bash devcheck.sh
RUN_ID=0062
M=qwen3.5-9b
if [ "${TINY:-0}" = 1 ]; then
  : "${WT:?}" "${REPO:?}" "${LOGS:?}" "${PY:?}"
  NODE=${NODE:-tiny}
  M=qwen3.5-0.8b
  T=(--tiny); D=(); L_BOX=2; L_CNT=2
  export EV_NO_GIT=1
else
  : "${NODE:?hpcgpu06 或 hpcgpu07}" "${TQ_STATUS:?本节点训练队列的心跳文件名（\$LOGS 下）}"
  VI=/home/mail/chenyuma/visint
  WT=$VI/wt/wt0062
  REPO=$VI/repo
  LOGS=$VI/logs
  PY=$VI/env/bin/python
  export HF_HOME=$VI/hf HF_HUB_CACHE=$VI/hf/hub HF_HUB_OFFLINE=1
  export TMPDIR=$VI/tmp PIP_NO_CACHE_DIR=1 NLTK_DATA=$VI/b200/nltk_data
  export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 PYTHONHASHSEED=0
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export TRITON_CACHE_DIR=$VI/cache/$NODE/triton TORCHINDUCTOR_CACHE_DIR=$VI/cache/$NODE/inductor XDG_CACHE_HOME=$VI/cache/$NODE/xdg
  T=(); D=(--data-root "$VI/b200/data"); L_BOX=64; L_CNT=49
  export EV_PAUSE=$LOGS/train.pause.$NODE EV_STATUS_GLOB="$TQ_STATUS" EV_NO_GIT=1
  export EV_DF_PATH=$VI
  EV_CAP_S=$(awk -v h="${EV_CAP_H:?任务书的上限（小时）}" 'BEGIN{print int(h * 3600)}')
fi
OUT=$REPO/experiments/W1/0062_w1_devcheck_$NODE
# shellcheck disable=SC1091
source "$WT/scripts/server/eval_lib.sh"
cd "$WT" || exit 1
[ "$TINY" = 1 ] || [ "${CUDA_VISIBLE_DEVICES:-}" = 0 ] || { progress "blocked: 不在 srun 的资源范围里"; exit 1; }
# 冻结核对：非 tiny 时 configs/w1.json 必须 frozen = true（run_w1.py 也会拒绝，这里提前报清楚）
if [ "$TINY" != 1 ] && ! "$PY" -c "import json,sys; sys.exit(0 if json.load(open('configs/w1.json'))['frozen'] is True else 1)"; then
  progress "blocked: configs/w1.json 没有冻结"; exit 1
fi
NOK=0
w1() {  # $1 = 步名，其余 = run_w1.py 的参数（配置一律 configs/w1.json）
  local name=$1; shift
  step "$name" "$PY" scripts/run_w1.py --model $M ${T[@]+"${T[@]}"} ${D[@]+"${D[@]}"} --w1-config configs/w1.json --out "$OUT" "$@" || NOK=$((NOK + 1))
}

claim_gpu "0062-hpc-w1-devcheck.md" || { progress "blocked: gpu not free after ${CLAIM_MAX_S}s"; exit 1; }
echo "$(now) start commit $(git -C "$WT" rev-parse HEAD 2>/dev/null || echo unknown) node $NODE model $M w1.json $(sha "$WT/configs/w1.json" | cut -d' ' -f1)" >> "$OUT/steps.log"
MEMPID=""
if [ "$TINY" != 1 ] && command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -lms 1000 > "$OUT/gpu_mem.csv" 2>/dev/null &
  MEMPID=$!
fi

# 1. 基线（开发-检查集全部 500 句，不加 --limit）
w1 "baseline_e1_refexp_box_devcheck" --task refexp_box --split devcheck --setting baseline_e1
w1 "baseline_ownbox_devcheck" --task refseg --split devcheck --setting baseline_ownbox
w1 "baseline_poly_structured_devcheck" --task refseg --split devcheck --setting baseline_poly_structured
w1 "w1f_devcheck" --task refseg --split devcheck --setting w1f
# 2. W1-1
w1 "full_refseg_devcheck" --task refseg --split devcheck --setting full
# 3. 去留判定
G=$OUT/$M/refseg
step "gate" "$PY" scripts/w1_gate.py --w1 "$G/full/devcheck" --ownbox "$G/baseline_ownbox/devcheck" \
  --poly "$G/baseline_poly_structured/devcheck" --out-json "$OUT/gate.json" --out-md "$OUT/gate.md" || NOK=$((NOK + 1))
# 4. 冻结协议在开发-调参集上的定位与计数（只报告）
w1 "full_refexp_box_devtune$L_BOX" --task refexp_box --split devtune --setting full --limit $L_BOX
w1 "full_count_devtune_countbench$L_CNT" --task count --split devtune_countbench --setting full --limit $L_CNT
[ -n "$MEMPID" ] && kill "$MEMPID" 2>/dev/null

echo "$(now) done; failed steps: $NOK" >> "$OUT/steps.log"
progress "done $RUN_ID (failed steps: $NOK)"
push_out "W1 devcheck done (failed steps: $NOK)"
