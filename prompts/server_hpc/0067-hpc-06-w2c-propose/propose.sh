#!/usr/bin/env bash
# RUN-0067（ITSC HPC hpcgpu06）：W2-c 训练数据的第一步——base 9B 按 E1 协议在 train 抽样的 16,000 句上出框（D118；W2 预注册 §9）。
# 只跑 --stage propose：不读候选清单，不要求 configs/w2.json 冻结（D118-3）。候选与标签一步等 W2-a 冻结后另发。
# 铁律：任何情况下都不能取消或停止 Slurm 任务 157296、157297。只用本节点自己的暂停文件。
# 用法：srun --jobid=157296 --overlap -n1 env PATH=… EV_CAP_H=… bash …/propose.sh < /dev/null
# 彩排（开发机）：TINY=1 WT=… REPO=… LOGS=… PY=… bash propose.sh
RUN_ID=0067
NODE=hpcgpu06
if [ "${TINY:-0}" = 1 ]; then
  : "${WT:?}" "${REPO:?}" "${LOGS:?}" "${PY:?}"
  T=(--tiny); D=()
  export EV_NO_GIT=1
else
  VI=/home/mail/chenyuma/visint
  WT=$VI/wt/wt0067
  REPO=$VI/repo
  LOGS=$VI/logs
  PY=$VI/env/bin/python
  export HF_HOME=$VI/hf HF_HUB_CACHE=$VI/hf/hub HF_HUB_OFFLINE=1
  export TMPDIR=$VI/tmp PIP_NO_CACHE_DIR=1 NLTK_DATA=$VI/b200/nltk_data
  export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 PYTHONHASHSEED=0
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export TRITON_CACHE_DIR=$VI/cache/$NODE/triton TORCHINDUCTOR_CACHE_DIR=$VI/cache/$NODE/inductor XDG_CACHE_HOME=$VI/cache/$NODE/xdg
  T=(--model qwen3.5-9b); D=(--train-data-root "$VI/data/train_data" --eval-data-root "$VI/b200/data")
  export EV_PAUSE=$LOGS/train.pause.$NODE EV_STATUS_GLOB="train_0056-$NODE.status" EV_NO_GIT=1
  export EV_DF_PATH=$VI
  EV_CAP_S=$(awk -v h="${EV_CAP_H:?任务书的上限（小时）}" 'BEGIN{print int(h * 3600)}')
fi
OUT=$REPO/experiments/W2/0067_w2c_data_hpcgpu06
# shellcheck disable=SC1091
source "$WT/scripts/server/eval_lib.sh"
cd "$WT" || exit 1
[ "$TINY" = 1 ] || [ "${CUDA_VISIBLE_DEVICES:-}" = 0 ] || { progress "blocked: 不在 srun 的资源范围里"; exit 1; }
claim_gpu "0067-hpc-06-w2c-propose.md" || { progress "blocked: gpu not free after ${CLAIM_MAX_S}s"; exit 1; }
echo "$(now) start commit $(git -C "$WT" rev-parse HEAD 2>/dev/null || echo unknown) node $NODE" >> "$OUT/steps.log"
NOK=0
step propose "$PY" scripts/build_w2c_data.py --stage propose ${T[@]+"${T[@]}"} ${D[@]+"${D[@]}"} --out "$OUT" || NOK=$((NOK + 1))
echo "$(now) done; failed steps: $NOK" >> "$OUT/steps.log"
progress "done $RUN_ID (failed steps: $NOK)"
push_out "W2-c propose done (failed steps: $NOK)"
