#!/usr/bin/env bash
# RUN-0070（ITSC HPC）：W2-a 调参的第 2 步重做（W2 预注册 §5 补记 2，D121）：以 RUN-0066 的 sep_on 为底，跑 K = $K 的一个臂。
# 两个臂（K = 4、6）分在 06、07 上各跑一个；选臂在 hpcmgt 上用 scripts/mgr/w2_tune.py select（带仪器有效性过滤）做，见任务书。
# 框复制自 RUN-0066 的第 0 步（同一份 E1 框）；候选按臂的清单重新生成；只用 devtune800。
# 铁律：任何情况下都不能取消或停止 Slurm 任务 157296、157297。只用本节点自己的暂停文件。
# 用法：srun --jobid=… --overlap -n1 -c 8 env PATH=… NODE=… K=4|6 EV_CAP_H=… bash …/arm.sh < /dev/null
# 彩排（开发机）：TINY=1 WT=… REPO=… LOGS=… PY=… K=4 R66=<tiny 的 RUN-0066 输出目录> bash arm.sh
RUN_ID=0070
M=qwen3.5-9b
SPLIT=devtune800
: "${K:?4 或 6}"
if [ "${TINY:-0}" = 1 ]; then
  : "${WT:?}" "${REPO:?}" "${LOGS:?}" "${PY:?}" "${R66:?}"
  NODE=${NODE:-tiny}
  M=qwen3.5-0.8b
  T=(--tiny); D=(); CW=1
  export EV_NO_GIT=1
else
  : "${NODE:?hpcgpu06 或 hpcgpu07}"
  VI=/home/mail/chenyuma/visint
  WT=$VI/wt/wt0070
  REPO=$VI/repo
  LOGS=$VI/logs
  PY=$VI/env/bin/python
  R66=$REPO/experiments/W2/0066_w2_tune_hpcgpu06
  export HF_HOME=$VI/hf HF_HUB_CACHE=$VI/hf/hub HF_HUB_OFFLINE=1
  export TMPDIR=$VI/tmp PIP_NO_CACHE_DIR=1 NLTK_DATA=$VI/b200/nltk_data
  export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONHASHSEED=0
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export TRITON_CACHE_DIR=$VI/cache/$NODE/triton TORCHINDUCTOR_CACHE_DIR=$VI/cache/$NODE/inductor XDG_CACHE_HOME=$VI/cache/$NODE/xdg
  T=(); D=(--data-root "$VI/b200/data"); CW=${CW:-8}
  export EV_PAUSE=$LOGS/train.pause.$NODE EV_STATUS_GLOB="nonexistent_queue.status" EV_NO_GIT=1
  export EV_DF_PATH=$VI
  EV_CAP_S=$(awk -v h="${EV_CAP_H:?任务书的上限（小时）}" 'BEGIN{print int(h * 3600)}')
fi
RUN_ID=0070_k$K
OUT=$REPO/experiments/W2/0070_w2_tune_r2_$NODE
ARM=sep_on_k$K
# shellcheck disable=SC1091
source "$WT/scripts/server/eval_lib.sh"
cd "$WT" || exit 1
[ "$TINY" = 1 ] || [ "${CUDA_VISIBLE_DEVICES:-}" = 0 ] || { progress "blocked: 不在 srun 的资源范围里"; exit 1; }
for f in "$R66/cfg/sep_on.json" "$R66/ranking.json" "$R66/0066_s0/$M/refexp_box/baseline_e1/$SPLIT/predictions.jsonl"; do
  [ -s "$f" ] || { progress "blocked: 缺 RUN-0066 的 $f"; exit 1; }
done
NOK=0
claim_gpu "0070-hpc-w2-tune-r2.md" || { progress "blocked: gpu not free after ${CLAIM_MAX_S}s"; exit 1; }
echo "$(now) start commit $(git -C "$WT" rev-parse HEAD 2>/dev/null || echo unknown) node $NODE model $M arm $ARM" >> "$OUT/steps.log"
mkdir -p "$OUT/cfg"
step "arm_cfg_$ARM" "$PY" scripts/mgr/w2_tune.py arm --base "$R66/cfg/sep_on.json" --out "$OUT/cfg/$ARM.json" --name "$ARM" \
  --ranking "$R66/ranking.json" --k "$K" || { echo "$(now) done; failed steps: 1" >> "$OUT/steps.log"; progress "done $RUN_ID (failed steps: 1)"; exit 1; }
A=$OUT/0070_$ARM
mkdir -p "$A/$M/refexp_box" && cp -R "$R66/0066_s0/$M/refexp_box/baseline_e1" "$A/$M/refexp_box/"
step "arm_$ARM" "$PY" scripts/run_w2.py --model $M ${T[@]+"${T[@]}"} ${D[@]+"${D[@]}"} --split $SPLIT --w2-config "$OUT/cfg/$ARM.json" \
  --out "$A" --candidate-workers "$CW" --task refseg --setting w2a || NOK=$((NOK + 1))
echo "$(now) done; failed steps: $NOK" >> "$OUT/steps.log"
progress "done $RUN_ID (failed steps: $NOK)"
push_out "W2 tuning round-2 arm $ARM done (failed steps: $NOK)"
