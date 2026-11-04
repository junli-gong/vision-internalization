#!/usr/bin/env bash
# RUN-0111（ITSC HPC，节点可选：06 或 07，补充 1）：G-XM 通过后的两项（D165 第四、五节；D177、D178、D179）。
#   A. Rex-Omni 外部参照，只做检测（scripts/run_rexomni.py；两个设置：rexomni_det_joint = 它自己的协议、标题读数；rexomni_det = 逐类诊断）：
#      1. xm_tune 冒烟：joint 的 f1_rexomni_exact < 0.60 时 run_rexomni 退出码 3，本驱动跳过 2、3（D178）；
#      2. xm_final（--xm-gate = RUN-0108 的 gate_xm.json）；
#      3. lvis_sub（同一 gate；每图最多 3 类，XM-2b，DEV-0082；与 RUN-0112 的 X-M 同一个 commit）。
#   B. X-M 在 W2 test 的 5 个划分上（run_x2 --xm-gate … --test-approval D165；设置同 RUN-0092），与 RUN-0092 的 X-2 同在 06 上，逐句配对。
#   命令行照抄 RUN-0092（run_x2 的 D、CW、TW、续跑）与 DEV-0083、DEV-0082 的总结。权重先由服务器在 hpcmgt 上下载并核对（任务书第 2 步）。
# 铁律：任何情况下都不能取消或停止 Slurm 任务 157296、157297。只用本节点自己的暂停文件。
# 补充 1（D180）：06 又被别的项目的 vLLM 占满，节点改为可选：NODE=hpcgpu06（与 RUN-0092 同节点）或 hpcgpu07（等 WAIT_DONE，缺省 eval_0112 跑完；
#   X-M 的 test 与 RUN-0092 不同节点，表里注明，D177）。
# 用法：srun --jobid=<157296（06）或 157297（07）> --overlap -n1 -c 8 env PATH=… TINY=0 NODE=<hpcgpu06|hpcgpu07> EV_CAP_H=… [STAGES=…] bash …/rexomni_xmtest.sh < /dev/null
# 彩排（开发机）：TINY=1 WT=… REPO=… LOGS=… PY=… PAIRS_JSON=… bash rexomni_xmtest.sh（Rex-Omni 只跑 tiny 的 xm_tune；其余只写命令行）
set -uo pipefail
TINY=${TINY:-0}
RUN_ID=0111
STAGES=${STAGES:-"rexomni xmtest"}
M=qwen3.5-9b
SPLITS=(test_refcoco_unc_testA test_refcoco_unc_testB "test_refcoco+_unc_testA" "test_refcoco+_unc_testB" test_refcocog_umd_test)
SETTINGS=(x2_spline_trimap x2_ellipse_trimap)
if [ "$TINY" = 1 ]; then
  : "${WT:?}" "${REPO:?}" "${LOGS:?}" "${PY:?}" "${PAIRS_JSON:?}"
  NODE=tiny; D=(); CW=1; TW=1; RXD=(--pairs-json "$PAIRS_JSON"); RXT=(--tiny)
  CK=/nonexistent/xm_ck; GATE_JSON=/nonexistent/gate_xm.json
  export EV_NO_GIT=1
else
  : "${NODE:?hpcgpu06 或 hpcgpu07}"
  case "$NODE" in
    hpcgpu06) WAIT_DONE=${WAIT_DONE:-} ;;
    hpcgpu07) WAIT_DONE=${WAIT_DONE:-"eval_0112.status:done 0112"} ;;
    *) echo "NODE 只能是 hpcgpu06 或 hpcgpu07：$NODE" >&2; exit 2;;
  esac
  VI=/home/mail/chenyuma/visint
  WT=$VI/wt/wt$RUN_ID
  REPO=$VI/repo
  LOGS=$VI/logs
  PY=$VI/env/bin/python
  export HF_HOME=$VI/hf HF_HUB_CACHE=$VI/hf/hub HF_HUB_OFFLINE=1
  export TMPDIR=$VI/tmp PIP_NO_CACHE_DIR=1 NLTK_DATA=$VI/b200/nltk_data MPLCONFIGDIR=$VI/cache/$NODE/mpl
  export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONHASHSEED=0
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export TRITON_CACHE_DIR=$VI/cache/$NODE/triton TORCHINDUCTOR_CACHE_DIR=$VI/cache/$NODE/inductor XDG_CACHE_HOME=$VI/cache/$NODE/xdg
  D=(--data-root "$VI/b200/data"); CW=8; TW=8   # run_x2：同 RUN-0092（合并核对也读这个根）
  RXD=(--data-root "$VI/data"); RXT=()          # run_rexomni：X-M 的数据根（COCO、LVIS）
  CK=$REPO/experiments/XM/0104_xm_hpcgpu07/0104_xm_train/w2c/qwen3.5-9b/seed0/ckpt/final/model
  GATE_JSON=$REPO/experiments/XM/0108_gate_xm_hpcgpu07/0108_gate/gate_xm.json
  export EV_PAUSE=$LOGS/train.pause.$NODE EV_STATUS_GLOB="nonexistent_queue.status" EV_NO_GIT=1
  export EV_DF_PATH=$VI
  CLAIM_MAX_S=${CLAIM_MAX_S:-3600}
  EV_CAP_S=$(awk -v h="${EV_CAP_H:?任务书的上限（小时）}" 'BEGIN{print int(h * 3600)}')
fi
OUT=$REPO/experiments/XM/${RUN_ID}_rexomni_xmtest_$NODE
O_RT=$OUT/${RUN_ID}_rexomni_tune
O_RF=$OUT/${RUN_ID}_rexomni_final
O_RL=$OUT/${RUN_ID}_rexomni_lvis
O_T=$OUT/${RUN_ID}_xm_test
for o in "$OUT" "$O_RT" "$O_RF" "$O_RL" "$O_T"; do case "$(basename "$o")" in [0-9][0-9][0-9][0-9]_*) ;; *) echo "目录名没有运行编号：$o" >&2; exit 2;; esac; done
for o in "$O_RT" "$O_RF" "$O_RL"; do case "$(basename "$o")" in *_rexomni*) ;; *) echo "Rex-Omni 的目录名要含 _rexomni：$o" >&2; exit 2;; esac; done
# shellcheck disable=SC1091
source "$WT/scripts/server/eval_lib.sh"
cd "$WT" || exit 1
if [ "$TINY" != 1 ]; then
  [ "${CUDA_VISIBLE_DEVICES:-}" = 0 ] || { progress "blocked: 不在 srun 的资源范围里"; exit 1; }
  "$PY" -c "import json,sys; sys.exit(0 if json.load(open('configs/w2.json'))['frozen'] is True else 1)" || { progress "blocked: configs/w2.json 没有冻结"; exit 1; }
  [ -f "$CK/manifest.json" ] || { progress "blocked: 没有 X-M 的检查点 $CK"; exit 1; }
  [ -f "$GATE_JSON" ] || { progress "blocked: 没有 G-XM 的判定文件 $GATE_JSON"; exit 1; }
  if [ -n "$WAIT_DONE" ]; then
    wf=${WAIT_DONE%%:*}; wp=${WAIT_DONE#*:}; t0=$(date +%s)
    while :; do
      st=$(cut -d' ' -f2- "$LOGS/$wf" 2>/dev/null || true)
      case "$st" in "$wp"*) break;; blocked*) progress "blocked: $wf 是 blocked：$st"; exit 1;; esac
      progress "waiting for $wf to start with '$wp' (now: ${st:-missing})"
      [ $(( $(date +%s) - t0 )) -ge 172800 ] && { progress "blocked: 等 $wf 超时"; exit 1; }
      sleep 60
    done
  fi
fi
NOK=0
claim_gpu "0111-hpc-rexomni-xmtest.md ($NODE)" || { progress "blocked: gpu not free after ${CLAIM_MAX_S}s"; exit 1; }
has() { case " $STAGES " in *" $1 "*) return 0;; esac; return 1; }
echo "$(now) stages [$STAGES]" >> "$OUT/steps.log"
echo "$(now) start commit $(git -C "$WT" rev-parse HEAD 2>/dev/null || echo unknown) node $NODE x-m checkpoint manifest $( [ -f "$CK/manifest.json" ] && sha "$CK/manifest.json" | cut -d' ' -f1 || echo none) gate $( [ -f "$GATE_JSON" ] && sha "$GATE_JSON" | cut -d' ' -f1 || echo none)" >> "$OUT/steps.log"
rres() { [ -d "$1/rex-omni" ] && echo --resume; }
if has rexomni; then
  step rexomni_tune "$PY" scripts/run_rexomni.py run ${RXT[@]+"${RXT[@]}"} --split xm_tune ${RXD[@]+"${RXD[@]}"} --out "$O_RT" $(rres "$O_RT")
  rc=$?
  if [ "$rc" = 3 ]; then
    echo "$(now) rexomni smoke line not met (joint f1_rexomni_exact < 0.60): skip rexomni_final, rexomni_lvis (D178)" >> "$OUT/steps.log"; NOK=$((NOK + 1))
  elif [ "$rc" != 0 ]; then
    NOK=$((NOK + 1))
  elif [ "$TINY" = 1 ]; then
    { echo "$PY scripts/run_rexomni.py run --split xm_final --xm-gate \$GATE_JSON --data-root \$VI/data --out $O_RF"
      echo "$PY scripts/run_rexomni.py run --split lvis_sub --xm-gate \$GATE_JSON --data-root \$VI/data --out $O_RL"; } > "$OUT/cmd_rexomni.txt"
  else
    step rexomni_final "$PY" scripts/run_rexomni.py run --split xm_final --xm-gate "$GATE_JSON" ${RXD[@]+"${RXD[@]}"} --out "$O_RF" $(rres "$O_RF") || NOK=$((NOK + 1))
    step rexomni_lvis "$PY" scripts/run_rexomni.py run --split lvis_sub --xm-gate "$GATE_JSON" ${RXD[@]+"${RXD[@]}"} --out "$O_RL" $(rres "$O_RL") || NOK=$((NOK + 1))
  fi
fi
if has xmtest; then
  if [ "$TINY" = 1 ]; then
    : > "$OUT/cmd_xmtest.txt"
    for SP in "${SPLITS[@]}"; do
      echo "$PY scripts/run_x2.py --model $M --data-root \$VI/b200/data --checkpoint \$CK --split $SP --out $O_T --settings ${SETTINGS[*]} --xm-gate \$GATE_JSON --test-approval D165 --candidate-workers 8 --tool-workers 8" >> "$OUT/cmd_xmtest.txt"
    done
  else
    for SP in "${SPLITS[@]}"; do
      RES=(); [ -d "$O_T/$M/refseg/x2_spline_trimap/$SP" ] && RES=(--resume)
      step "xm_$SP" "$PY" scripts/run_x2.py --model $M ${D[@]+"${D[@]}"} --checkpoint "$CK" --split "$SP" --out "$O_T" \
        --settings "${SETTINGS[@]}" --xm-gate "$GATE_JSON" --test-approval D165 --candidate-workers "$CW" --tool-workers "$TW" \
        ${RES[@]+"${RES[@]}"} || NOK=$((NOK + 1))
    done
  fi
fi
echo "$(now) done; failed steps: $NOK" >> "$OUT/steps.log"
progress "done $RUN_ID (failed steps: $NOK)"
push_out "Rex-Omni + X-M test ${RUN_ID} done (failed steps: $NOK)"
