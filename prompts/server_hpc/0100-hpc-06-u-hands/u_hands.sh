#!/usr/bin/env bash
# RUN-0100（ITSC HPC hpcgpu06）：统一主表的两只"手"在 W2 test 的 1,500 句子集上（D154、D159；预注册 U-2）：
#   1. E1 结构强制多边形，9B 零样本（run_u_hands --system e1_poly_structured）；
#   2. 9B 的 E3b 像素 token 网格，结构强制解码、单轮、guided（--system e3b_grid，RUN-0056 的 E3b seed 0 检查点，manifest 316d25c7…）。
#   排在 RUN-0097（LISA）之后：等 eval_0097.status 以 "done 0097" 开头。两个系统各一个 --out（目录名带运行编号）。
# 铁律：任何情况下都不能取消或停止 Slurm 任务 157296、157297。只用本节点自己的暂停文件。
# 用法：srun --jobid=157296 --overlap -n1 -c 8 env PATH=… TINY=0 EV_CAP_H=… bash …/u_hands.sh < /dev/null
# 彩排（开发机）：TINY=1 WT=… REPO=… LOGS=… PY=… E3B_TINY_CK=<train_e3b --tiny 的检查点> bash u_hands.sh
set -uo pipefail
TINY=${TINY:-0}
RUN_ID=0100
E3B_MANIFEST_SHA=316d25c7bf0b21edd6355a55bc0201249078b7c59305563dd3e25082c7c088db
if [ "$TINY" = 1 ]; then
  : "${WT:?}" "${REPO:?}" "${LOGS:?}" "${PY:?}" "${E3B_TINY_CK:?}"
  NODE=tiny; M=qwen3.5-0.8b; T=(--tiny); D=(); SW=0; CK=$E3B_TINY_CK; SPL=test_refcoco_unc_testA
  export EV_NO_GIT=1
else
  NODE=hpcgpu06; M=qwen3.5-9b
  VI=/home/mail/chenyuma/visint
  WT=$VI/wt/wt$RUN_ID
  REPO=$VI/repo
  LOGS=$VI/logs
  PY=$VI/env/bin/python
  export HF_HOME=$VI/hf HF_HUB_CACHE=$VI/hf/hub HF_HUB_OFFLINE=1
  export TMPDIR=$VI/tmp PIP_NO_CACHE_DIR=1 MPLCONFIGDIR=$VI/cache/$NODE/mpl
  export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONHASHSEED=0
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export TRITON_CACHE_DIR=$VI/cache/$NODE/triton TORCHINDUCTOR_CACHE_DIR=$VI/cache/$NODE/inductor XDG_CACHE_HOME=$VI/cache/$NODE/xdg
  T=(); D=(--data-root "$VI/b200/data"); SW=8; SPL=all
  CK=$VI/ckpt/hpcgpu06/0056/e3b/qwen3.5-9b/seed0/ckpt/final/model
  export EV_PAUSE=$LOGS/train.pause.$NODE EV_STATUS_GLOB="nonexistent_queue.status" EV_NO_GIT=1
  export EV_DF_PATH=$VI
  CLAIM_MAX_S=${CLAIM_MAX_S:-3600}
  EV_CAP_S=$(awk -v h="${EV_CAP_H:?任务书的上限（小时）}" 'BEGIN{print int(h * 3600)}')
fi
OUT=$REPO/experiments/W2/${RUN_ID}_u_hands_$NODE   # eval_lib 的 $OUT：steps.log、logs/、同步循环都认它
OUT_E1=$OUT/${RUN_ID}_u_e1poly                      # 两个系统各一个 --out，名字也带运行编号
OUT_E3=$OUT/${RUN_ID}_u_e3bgrid
for o in "$OUT" "$OUT_E1" "$OUT_E3"; do case "$(basename "$o")" in [0-9][0-9][0-9][0-9]_*) ;; *) echo "目录名没有运行编号：$o" >&2; exit 2;; esac; done
# shellcheck disable=SC1091
source "$WT/scripts/server/eval_lib.sh"
cd "$WT" || exit 1
if [ "$TINY" != 1 ]; then
  [ "${CUDA_VISIBLE_DEVICES:-}" = 0 ] || { progress "blocked: 不在 srun 的资源范围里"; exit 1; }
  "$PY" -c "import json,sys; sys.exit(0 if json.load(open('configs/w2.json'))['frozen'] is True else 1)" || { progress "blocked: configs/w2.json 没有冻结"; exit 1; }
  got=$(sha "$CK/manifest.json" | cut -d' ' -f1)
  [ "$got" = "$E3B_MANIFEST_SHA" ] || { progress "blocked: E3b 检查点 manifest 不符（$got）"; exit 1; }
  t0=$(date +%s)
  while :; do
    st=$(cut -d' ' -f2- "$LOGS/eval_0097.status" 2>/dev/null || true)
    case "$st" in "done 0097"*) break;; blocked*) progress "blocked: RUN-0097 是 blocked：$st"; exit 1;; esac
    progress "waiting for eval_0097.status to start with 'done 0097' (now: ${st:-missing})"
    [ $(( $(date +%s) - t0 )) -ge 172800 ] && { progress "blocked: 等 RUN-0097 超时"; exit 1; }
    sleep 60
  done
fi
NOK=0
claim_gpu "0100-hpc-06-u-hands.md" || { progress "blocked: gpu not free after ${CLAIM_MAX_S}s"; exit 1; }
echo "$(now) start commit $(git -C "$WT" rev-parse HEAD 2>/dev/null || echo unknown) node $NODE model $M" >> "$OUT/steps.log"
res_of() { [ -d "$1/$M" ] && echo --resume; }  # 只看该系统自己的 --out  # 输出目录里已有该模型的内容：续跑
step u_e1poly "$PY" scripts/run_u_hands.py --system e1_poly_structured --model $M ${T[@]+"${T[@]}"} --split "$SPL" --test-approval D154 \
  ${D[@]+"${D[@]}"} --out "$OUT_E1" $(res_of "$OUT_E1") || NOK=$((NOK + 1))
echo "$(now) e3b checkpoint $CK manifest $(sha "$CK/manifest.json" | cut -d' ' -f1)" >> "$OUT/steps.log"
step u_e3bgrid "$PY" scripts/run_u_hands.py --system e3b_grid --model $M ${T[@]+"${T[@]}"} --split "$SPL" --test-approval D154 \
  --checkpoint "$CK" --seed 0 --score-workers "$SW" ${D[@]+"${D[@]}"} --out "$OUT_E3" $(res_of "$OUT_E3") || NOK=$((NOK + 1))
echo "$(now) done; failed steps: $NOK" >> "$OUT/steps.log"
progress "done $RUN_ID (failed steps: $NOK)"
push_out "U hands ${RUN_ID} done (failed steps: $NOK)"
