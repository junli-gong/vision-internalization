#!/usr/bin/env bash
# RUN-0092（ITSC HPC hpcgpu06）：通过 G-EX 的候选在 W2 的 test 上运行（预注册 X-6、X-6a；D147、D148）。
#   每个候选：run_x2 --split <5 个 test 划分> --settings x2_spline_trimap x2_ellipse_trimap --test-gate <它的 gate_explore.json> --test-approval D147。
#   run_x2 自己核对 gate 的结论（通过、非彩排、判定点全真、检查点与设置一致），不合格就拒跑。
#   与 RUN-0072 在 06 上的 W2-c 逐句配对（同一份 test 清单、同一个取样函数）。排在 RUN-0091 之后（等 eval_0091.status 以 "done 0091" 开头）。
# 铁律：任何情况下都不能取消或停止 Slurm 任务 157296、157297。只用本节点自己的暂停文件。
# 用法：srun --jobid=157296 --overlap -n1 -c 8 env PATH=… TINY=0 CANDS="名|检查点|gate_explore.json …" EV_CAP_H=… bash …/x_test.sh
# 彩排（开发机）：TINY=1 WT=… REPO=… LOGS=… PY=… CANDS=… bash x_test.sh（只核对目录名与命令行）
set -uo pipefail
TINY=${TINY:-0}
RUN_ID=0092
: "${CANDS:?通过的候选：\"名|检查点|gate_explore.json\"，空格分隔}"
M=qwen3.5-9b
SPLITS=(test_refcoco_unc_testA test_refcoco_unc_testB "test_refcoco+_unc_testA" "test_refcoco+_unc_testB" test_refcocog_umd_test)
SETTINGS=(x2_spline_trimap x2_ellipse_trimap)
if [ "$TINY" = 1 ]; then
  : "${WT:?}" "${REPO:?}" "${LOGS:?}" "${PY:?}"
  NODE=tiny; D=(); CW=1; TW=1
  export EV_NO_GIT=1
else
  NODE=hpcgpu06
  VI=/home/mail/chenyuma/visint
  WT=$VI/wt/wt0092
  REPO=$VI/repo
  LOGS=$VI/logs
  PY=$VI/env/bin/python
  export HF_HOME=$VI/hf HF_HUB_CACHE=$VI/hf/hub HF_HUB_OFFLINE=1
  export TMPDIR=$VI/tmp PIP_NO_CACHE_DIR=1 NLTK_DATA=$VI/b200/nltk_data
  export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONHASHSEED=0
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export TRITON_CACHE_DIR=$VI/cache/$NODE/triton TORCHINDUCTOR_CACHE_DIR=$VI/cache/$NODE/inductor XDG_CACHE_HOME=$VI/cache/$NODE/xdg
  D=(--data-root "$VI/b200/data"); CW=8; TW=8
  export EV_PAUSE=$LOGS/train.pause.$NODE EV_STATUS_GLOB="nonexistent_queue.status" EV_NO_GIT=1
  export EV_DF_PATH=$VI
  CLAIM_MAX_S=${CLAIM_MAX_S:-3600}
  EV_CAP_S=$(awk -v h="${EV_CAP_H:?任务书的上限（小时）}" 'BEGIN{print int(h * 3600)}')
fi
OUT=$REPO/experiments/X/0092_x_test_$NODE
out_of() { echo "$REPO/experiments/X/0092_$1_test_$NODE"; }  # 每个候选一个 --out，目录名带运行编号
for c in $CANDS; do
  o=$(out_of "${c%%|*}")
  case "$(basename "$o")" in [0-9][0-9][0-9][0-9]_*) ;; *) echo "目录名没有运行编号：$o" >&2; exit 2;; esac
done
if [ "$TINY" = 1 ]; then
  mkdir -p "$OUT"; : > "$OUT/cmd.txt"
  for c in $CANDS; do IFS='|' read -r name ck gate <<< "$c"; for SP in "${SPLITS[@]}"; do
    echo "$PY scripts/run_x2.py --model $M --checkpoint $ck --split $SP --out $(out_of "$name") --settings ${SETTINGS[*]} --test-gate $gate --test-approval D147" >> "$OUT/cmd.txt"
  done; done
  echo "tiny：只核对目录名与命令行 → $OUT/cmd.txt"; exit 0
fi
# shellcheck disable=SC1091
source "$WT/scripts/server/eval_lib.sh"
cd "$WT" || exit 1
[ "${CUDA_VISIBLE_DEVICES:-}" = 0 ] || { progress "blocked: 不在 srun 的资源范围里"; exit 1; }
"$PY" -c "import json,sys; sys.exit(0 if json.load(open('configs/w2.json'))['frozen'] is True else 1)" || { progress "blocked: configs/w2.json 没有冻结"; exit 1; }
t0=$(date +%s)
while :; do
  st=$(cut -d' ' -f2- "$LOGS/eval_0091.status" 2>/dev/null || true)
  case "$st" in "done 0091"*) break;; blocked*) progress "blocked: RUN-0091 是 blocked：$st"; exit 1;; esac
  progress "waiting for eval_0091.status to start with 'done 0091' (now: ${st:-missing})"
  [ $(( $(date +%s) - t0 )) -ge 172800 ] && { progress "blocked: 等 RUN-0091 超时"; exit 1; }
  sleep 60
done
NOK=0
claim_gpu "0092-hpc-06-x-test.md" || { progress "blocked: gpu not free after ${CLAIM_MAX_S}s"; exit 1; }
mkdir -p "$OUT"
echo "$(now) start commit $(git -C "$WT" rev-parse HEAD 2>/dev/null || echo unknown) node $NODE cands [$CANDS]" >> "$OUT/steps.log"
for c in $CANDS; do
  IFS='|' read -r name ck gate <<< "$c"
  echo "$(now) candidate $name checkpoint $(sha "$ck/manifest.json" | cut -d' ' -f1) gate $(sha "$gate" | cut -d' ' -f1)" >> "$OUT/steps.log"
  for SP in "${SPLITS[@]}"; do
    RES=(); [ -d "$(out_of "$name")/$M/refseg/x2_spline_trimap/$SP" ] && RES=(--resume)
    step "${name}_$SP" "$PY" scripts/run_x2.py --model $M ${D[@]+"${D[@]}"} --checkpoint "$ck" --split "$SP" --out "$(out_of "$name")" \
      --settings "${SETTINGS[@]}" --test-gate "$gate" --test-approval D147 --candidate-workers "$CW" --tool-workers "$TW" \
      ${RES[@]+"${RES[@]}"} || NOK=$((NOK + 1))
  done
done
echo "$(now) done; failed steps: $NOK" >> "$OUT/steps.log"
progress "done $RUN_ID (failed steps: $NOK)"
push_out "X test ${RUN_ID} done (failed steps: $NOK)"
