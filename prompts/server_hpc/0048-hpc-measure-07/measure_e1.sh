#!/usr/bin/env bash
# RUN-0048 补充 1（ITSC HPC hpcgpu07）：补测 T 文本的单位耗时。measure.sh 里这三步把 run_e1 的 --out 设成了 <OUT>/e1、<OUT>/e1base，
# 目录名不带运行编号，被 run_e1 拒绝（rc=2，非 tiny 才查，开发机彩排没查出来）。这里用带编号的目录重跑同样的三条，其余不变。
# 只在 measure.sh 跑完（eval_0048.status 是 "done 0048 …"）之后启动。结果只作计时用，不进任何结论。
# 铁律：任何情况下都不能取消或停止 Slurm 任务 157296、157297。只用本节点自己的暂停文件，不写 $LOGS/train.pause。
#
# 用法（hpcgpu07 上经 srun 启动，见 0048-addendum-1.md）：
#   srun --jobid=157297 --overlap -n1 env PATH=… bash $VI/wt/wt0048a/prompts/server_hpc/0048-hpc-measure-07/measure_e1.sh < /dev/null
# 彩排（开发机）：TINY=1 WT=… REPO=… LOGS=… PY=… CK45=<有 e3b_c3 的 0.8B> bash measure_e1.sh
RUN_ID=0048e1
NODE=hpcgpu07
M08=qwen3.5-0.8b
if [ "${TINY:-0}" = 1 ]; then
  : "${WT:?}" "${REPO:?}" "${LOGS:?}" "${PY:?}" "${CK45:?}"
  T=(--tiny); D=(); L1=2; LF=1
else
  VI=/home/mail/chenyuma/visint
  WT=$VI/wt/wt0048a
  REPO=$VI/repo
  LOGS=$VI/logs
  PY=$VI/env/bin/python
  CK45=$VI/ckpt/$NODE/0045
  export HF_HOME=$VI/hf HF_HUB_CACHE=$VI/hf/hub HF_HUB_OFFLINE=1
  export TMPDIR=$VI/tmp PIP_NO_CACHE_DIR=1 NLTK_DATA=$VI/b200/nltk_data
  export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 PYTHONHASHSEED=0
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export TRITON_CACHE_DIR=$VI/cache/$NODE/triton TORCHINDUCTOR_CACHE_DIR=$VI/cache/$NODE/inductor XDG_CACHE_HOME=$VI/cache/$NODE/xdg
  T=(); D=(--data-root "$VI/b200/data"); L1=20; LF=10
  export EV_PAUSE=$LOGS/train.pause.$NODE EV_STATUS_GLOB="train_0045-$NODE.status" EV_NO_GIT=1
  export EV_QUOTA_CMD="timeout 60 lfs quota -u chenyuma /home | awk 'NR==3{u=\$2; gsub(/\\*/,\"\",u); v=int((u-7059403936)/1048576)} END{print v+0}'"
  export EV_SHARE_STOP_GB=80 EV_DF_PATH=$VI
  EV_CAP_S=$((2 * 3600))  # 0048-addendum-1 的上限：拿到 GPU 之后 2 小时
fi
OUT=$REPO/experiments/E3/0048_hpc_measure_e1_$NODE
# shellcheck disable=SC1091
source "$WT/scripts/server/eval_lib.sh"
cd "$WT" || exit 1
if [ "$TINY" != 1 ]; then
  case "$(cut -d' ' -f2- "$LOGS/eval_0048.status" 2>/dev/null)" in "done 0048"*) ;; *) progress "blocked: measure.sh 还没跑完"; exit 1;; esac
  [ "${CUDA_VISIBLE_DEVICES:-}" = 0 ] || { progress "blocked: 不在 srun 的资源范围里"; exit 1; }
fi
CK08=$CK45/e3b_c3/$M08/seed0/ckpt/final/model
[ -d "$CK08" ] || { progress "blocked: 检查点不在：$CK08"; exit 1; }
NOK=0

claim_gpu "0048-addendum-1.md" || { progress "blocked: gpu not free after ${CLAIM_MAX_S}s"; exit 1; }
echo "$(now) start commit $(git -C "$WT" rev-parse HEAD 2>/dev/null || echo unknown) node $NODE" >> "$OUT/steps.log"
step "ttext_e3b_c3_s0_dev${L1}" "$PY" scripts/run_e1.py --model $M08 ${T[@]+"${T[@]}"} ${D[@]+"${D[@]}"} --checkpoint "$CK08" \
  --checkpoint-label t_text/e3b_c3/seed0 --out "$OUT" --task all --subset dev --limit "$L1" || NOK=$((NOK + 1))
step "ttext_e3b_c3_s0_free${LF}" "$PY" scripts/run_e1.py --model $M08 ${T[@]+"${T[@]}"} ${D[@]+"${D[@]}"} --checkpoint "$CK08" \
  --checkpoint-label t_text/e3b_c3/seed0 --out "$OUT" --task all --subset free --limit "$LF" || NOK=$((NOK + 1))
step "e1_base_dev${L1}" "$PY" scripts/run_e1.py --model $M08 ${T[@]+"${T[@]}"} ${D[@]+"${D[@]}"} \
  --out "$OUT" --task all --subset dev --limit "$L1" || NOK=$((NOK + 1))
echo "$(now) done; failed steps: $NOK" >> "$OUT/steps.log"
progress "done $RUN_ID (failed steps: $NOK)"
push_out "T-text timing (failed steps: $NOK)"
