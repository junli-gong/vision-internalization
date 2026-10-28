#!/usr/bin/env bash
# RUN-0090（ITSC HPC）：探索的判定 G-EX（预注册 X §3、§4 与候选的预注册；D126-4、D130、D131）。**唯一一次在密封的 devcheck3 上跑系统。**
#   1. W2-c（RUN-0069 的检查点，D122 冻结的 W2 配置）在 devcheck3 三层上：run_w2 --setting w2c --unseal-devcheck3 G-EX。
#   2. 每个候选（CANDS，至多 2 个）：run_x2 --checkpoint <候选> --settings <冻结的设置> --unseal-devcheck3 G-EX。
#   3. 每个候选：scripts/gate_explore.py（500 句合并 cIoU、配对 bootstrap 97.5%、10,000 次；规则 (i)/(ii)），
#      健康检查用候选在同一节点上已有的读数（健康检查是确定性的固定协议，同节点重测与已有读数逐题相同，RUN-0074 核对过）。
#   同一节点上跑完全部（gate_explore 要求两边同机）。不调任何参数；不看结果再改。
# 铁律：任何情况下都不能取消或停止 Slurm 任务 157296、157297。只用本节点自己的暂停文件。
# 用法：srun --jobid=… --overlap -n1 -c 8 env PATH=… TINY=0 NODE=… CANDS="名|检查点|设置|候选健康检查 …" HEALTH_BASE=… EV_CAP_H=… bash …/gate_gex.sh
# 彩排（开发机）：TINY=1 WT=… REPO=… LOGS=… PY=… W2C_CK=… CANDS=… HEALTH_BASE=… GATE_SPLITS=<tiny 清单 json> bash gate_gex.sh
set -uo pipefail
TINY=${TINY:-0}
: "${CANDS:?候选：\"名|检查点|设置|候选健康检查\"，空格分隔，至多 2 个}" "${HEALTH_BASE:?同节点基座的健康检查 JSON}"
GATE=G-EX
D3=(devcheck3_refcoco_unc_val "devcheck3_refcoco+_unc_val" devcheck3_refcocog_umd_val)
if [ "$TINY" = 1 ]; then
  : "${WT:?}" "${REPO:?}" "${LOGS:?}" "${PY:?}" "${W2C_CK:?}" "${GATE_SPLITS:?}"
  NODE=tiny; RUN_ID=0090-tiny; M=qwen3.5-0.8b; T=(--tiny); D=(); CW=1; TW=1
  export EV_NO_GIT=1
else
  : "${NODE:?hpcgpu06 或 hpcgpu07}"
  RUN_ID=0090-$NODE
  VI=/home/mail/chenyuma/visint
  WT=$VI/wt/wt0090
  REPO=$VI/repo
  LOGS=$VI/logs
  PY=$VI/env/bin/python
  M=qwen3.5-9b
  W2C_CK=$REPO/experiments/W2/0069_w2c_hpcgpu06/0069_train/w2c/qwen3.5-9b/seed0/ckpt/final/model
  GATE_SPLITS=configs/explore_splits.json
  export HF_HOME=$VI/hf HF_HUB_CACHE=$VI/hf/hub HF_HUB_OFFLINE=1
  export TMPDIR=$VI/tmp PIP_NO_CACHE_DIR=1 NLTK_DATA=$VI/b200/nltk_data
  export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONHASHSEED=0
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export TRITON_CACHE_DIR=$VI/cache/$NODE/triton TORCHINDUCTOR_CACHE_DIR=$VI/cache/$NODE/inductor XDG_CACHE_HOME=$VI/cache/$NODE/xdg
  T=(); D=(--data-root "$VI/b200/data"); CW=8; TW=8
  export EV_PAUSE=$LOGS/train.pause.$NODE EV_STATUS_GLOB="nonexistent_queue.status" EV_NO_GIT=1
  export EV_DF_PATH=$VI
  CLAIM_MAX_S=${CLAIM_MAX_S:-3600}
  EV_CAP_S=$(awk -v h="${EV_CAP_H:?任务书的上限（小时）}" 'BEGIN{print int(h * 3600)}')
fi
OUT=$REPO/experiments/X/0090_gex_$NODE
REF=$OUT/0090_w2c
n_c=0; for c in $CANDS; do n_c=$((n_c + 1)); done
[ "$n_c" -ge 1 ] && [ "$n_c" -le 2 ] || { echo "候选必须是 1 或 2 个（预注册 X §3），收到 $n_c" >&2; exit 2; }
for o in "$REF" $(for c in $CANDS; do echo "$OUT/0090_${c%%|*}"; done); do
  case "$(basename "$o")" in [0-9][0-9][0-9][0-9]_*) ;; *) echo "目录名没有运行编号：$o" >&2; exit 2;; esac
done
# shellcheck disable=SC1091
source "$WT/scripts/server/eval_lib.sh"
cd "$WT" || exit 1
[ "$TINY" = 1 ] || [ "${CUDA_VISIBLE_DEVICES:-}" = 0 ] || { progress "blocked: 不在 srun 的资源范围里"; exit 1; }
if [ "$TINY" != 1 ] && ! "$PY" -c "import json,sys; sys.exit(0 if json.load(open('configs/w2.json'))['frozen'] is True else 1)"; then
  progress "blocked: configs/w2.json 没有冻结"; exit 1
fi
NOK=0
claim_gpu "0090-hpc-gex.md" || { progress "blocked: gpu not free after ${CLAIM_MAX_S}s"; exit 1; }
mkdir -p "$OUT"
echo "$(now) start commit $(git -C "$WT" rev-parse HEAD 2>/dev/null || echo unknown) node $NODE gate $GATE cands [$CANDS] splits $(sha "$WT/configs/explore_splits.json" | cut -d' ' -f1)" >> "$OUT/steps.log"
# 1. W2-c 参照
for SP in "${D3[@]}"; do
  step "w2c_$SP" "$PY" scripts/run_w2.py --model $M ${T[@]+"${T[@]}"} ${D[@]+"${D[@]}"} --splits-file configs/explore_splits.json --split "$SP" \
    --unseal-devcheck3 $GATE --w2-config configs/w2.json --out "$REF" --candidate-workers "$CW" --task refseg --setting w2c --checkpoint "$W2C_CK" \
    || NOK=$((NOK + 1))
done
# 2–3. 候选
for c in $CANDS; do
  IFS='|' read -r name ck setting hc <<< "$c"
  CO=$OUT/0090_$name
  echo "$(now) candidate $name checkpoint $(sha "$ck/manifest.json" | cut -d' ' -f1) setting $setting health $(sha "$hc" | cut -d' ' -f1)" >> "$OUT/steps.log"
  for SP in "${D3[@]}"; do
    step "${name}_$SP" "$PY" scripts/run_x2.py --model $M ${T[@]+"${T[@]}"} ${D[@]+"${D[@]}"} --checkpoint "$ck" --splits-file configs/explore_splits.json \
      --split "$SP" --unseal-devcheck3 $GATE --settings "$setting" --out "$CO" --candidate-workers "$CW" --tool-workers "$TW" || NOK=$((NOK + 1))
  done
  REFS=(); CANDD=()
  for SP in "${D3[@]}"; do REFS+=("$REF/$M/refseg/w2c/$SP"); CANDD+=("$CO/$M/refseg/$setting/$SP"); done
  step "gate_$name" "$PY" scripts/gate_explore.py --candidate-name "$name" --reference "${REFS[@]}" --candidate "${CANDD[@]}" \
    --health-base "$HEALTH_BASE" --health-candidate "$hc" --splits-file "$GATE_SPLITS" --unseal-devcheck3 $GATE --out-dir "$OUT/gate_$name" \
    || NOK=$((NOK + 1))
done
echo "$(now) done; failed steps: $NOK" >> "$OUT/steps.log"
progress "done $RUN_ID (failed steps: $NOK)"
push_out "G-EX ${RUN_ID} done (failed steps: $NOK)"
