#!/usr/bin/env bash
# RUN-0068（ITSC HPC）：W2-a 的去留判定（D113 第三节；W2 预注册 §6）。devcheck2 500 句、Qwen3.5-9B、冻结的 configs/w2.json。
#   1. 出框（E1 协议）与候选；w1f（判定的对照）、w2a（主配置）；ownbox、random_pick、oracle_pick（参照，只报告）。
#   2. w2_gate.py --main-name W2-a：主配置 − W1-F 的 cIoU 差，配对 bootstrap 97.5% CI 下界 > 0 才算通过（规则写死在脚本里）。
# 铁律：任何情况下都不能取消或停止 Slurm 任务 157296、157297。只用本节点自己的暂停文件。
# 用法（经 srun 启动，见任务书）：srun --jobid=… --overlap -n1 -c 8 env PATH=… NODE=… TQ_STATUS=… EV_CAP_H=… bash …/gate_w2a.sh
# 彩排（开发机）：TINY=1 WT=… REPO=… LOGS=… PY=… bash gate_w2a.sh
RUN_ID=0068
M=qwen3.5-9b
SPLIT=devcheck2
if [ "${TINY:-0}" = 1 ]; then
  : "${WT:?}" "${REPO:?}" "${LOGS:?}" "${PY:?}"
  NODE=${NODE:-tiny}
  M=qwen3.5-0.8b
  T=(--tiny); D=(); CW=1
  export EV_NO_GIT=1
else
  : "${NODE:?hpcgpu06 或 hpcgpu07}" "${TQ_STATUS:?本节点训练队列的心跳文件名（\$LOGS 下）}"
  VI=/home/mail/chenyuma/visint
  WT=$VI/wt/wt0068
  REPO=$VI/repo
  LOGS=$VI/logs
  PY=$VI/env/bin/python
  export HF_HOME=$VI/hf HF_HUB_CACHE=$VI/hf/hub HF_HUB_OFFLINE=1
  export TMPDIR=$VI/tmp PIP_NO_CACHE_DIR=1 NLTK_DATA=$VI/b200/nltk_data
  export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONHASHSEED=0
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export TRITON_CACHE_DIR=$VI/cache/$NODE/triton TORCHINDUCTOR_CACHE_DIR=$VI/cache/$NODE/inductor XDG_CACHE_HOME=$VI/cache/$NODE/xdg
  T=(); D=(--data-root "$VI/b200/data"); CW=${CW:-8}
  export EV_PAUSE=$LOGS/train.pause.$NODE EV_STATUS_GLOB="$TQ_STATUS" EV_NO_GIT=1
  export EV_DF_PATH=$VI
  EV_CAP_S=$(awk -v h="${EV_CAP_H:?任务书的上限（小时）}" 'BEGIN{print int(h * 3600)}')
fi
OUT=$REPO/experiments/W2/0068_w2a_devcheck2_$NODE
# shellcheck disable=SC1091
source "$WT/scripts/server/eval_lib.sh"
cd "$WT" || exit 1
[ "$TINY" = 1 ] || [ "${CUDA_VISIBLE_DEVICES:-}" = 0 ] || { progress "blocked: 不在 srun 的资源范围里"; exit 1; }
if [ "$TINY" != 1 ] && ! "$PY" -c "import json,sys; sys.exit(0 if json.load(open('configs/w2.json'))['frozen'] is True else 1)"; then
  progress "blocked: configs/w2.json 没有冻结"; exit 1
fi
NOK=0
w2() {  # $1 = 步名，其余 = run_w2.py 的参数（配置一律 configs/w2.json）
  local name=$1; shift
  step "$name" "$PY" scripts/run_w2.py --model $M ${T[@]+"${T[@]}"} ${D[@]+"${D[@]}"} --split $SPLIT --w2-config configs/w2.json \
    --out "$OUT" --candidate-workers "$CW" "$@" || NOK=$((NOK + 1))
}

claim_gpu "0068-hpc-w2a-devcheck2.md" || { progress "blocked: gpu not free after ${CLAIM_MAX_S}s"; exit 1; }
echo "$(now) start commit $(git -C "$WT" rev-parse HEAD 2>/dev/null || echo unknown) node $NODE model $M w2.json $(sha "$WT/configs/w2.json" | cut -d' ' -f1)" >> "$OUT/steps.log"
MEMPID=""
if [ "$TINY" != 1 ] && command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -lms 1000 > "$OUT/gpu_mem.csv" 2>/dev/null &
  MEMPID=$!
fi
w2 e1_box --task refexp_box --setting baseline_e1
if [ "$TINY" = 1 ] && [ -n "${INJECT:-}" ]; then  # 彩排：随机小模型给不出合法的框，换成 GT 内缩的框
  "$PY" -c "import sys; sys.path.insert(0, '.'); from pathlib import Path; from tests.test_w2_cli import inject_boxes; inject_boxes(Path('$OUT/$M/refexp_box/baseline_e1/$SPLIT'), '$SPLIT')"
fi
w2 w1f --task refseg --setting w1f
w2 w2a --task refseg --setting w2a
G=$OUT/$M/refseg
step gate "$PY" scripts/w2_gate.py --main-name W2-a --main "$G/w2a/$SPLIT" --w1f "$G/w1f/$SPLIT" --out-dir "$OUT/gate_w2a" || NOK=$((NOK + 1))
for s in ownbox random_pick oracle_pick; do
  w2 "$s" --task refseg --setting $s
done
[ -n "$MEMPID" ] && kill "$MEMPID" 2>/dev/null
echo "$(now) done; failed steps: $NOK" >> "$OUT/steps.log"
progress "done $RUN_ID (failed steps: $NOK)"
push_out "W2-a devcheck2 gate done (failed steps: $NOK)"
