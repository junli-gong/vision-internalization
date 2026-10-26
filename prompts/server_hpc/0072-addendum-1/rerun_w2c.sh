#!/usr/bin/env bash
# RUN-0072 补充 1（ITSC HPC hpcgpu06）：W2 test 里 W2-c 一项（--setting w2c --checkpoint）的重跑，5 个 test 划分。
#   RUN-0072 的 w2c 步骤全部 rc=2：W2-c 的模型目录里多了恢复时保留的 adapter_model.safetensors.gz 与 LARGE_FILES.sha256，
#   加载器要求目录与 manifest 恰好一致（D127 的恢复指令所致）。两份文件已移到模型目录之外后重跑。
#   代码用 RUN-0072 自己的工作树 wt0072（与这次 test 的其余设置同一个 commit）；结果写回 RUN-0072 的 OUT（同一批划分、同一节点，
#   与 w1f、w2a 等配对）。心跳 eval_0072r.status；排在 RUN-0074 之后（等 eval_0074.status 不再是 running、GPU 上没有进程）。
# 铁律：任何情况下都不能取消或停止 Slurm 任务 157296、157297。只用本节点自己的暂停文件。
# 用法：srun --jobid=157296 --overlap -n1 -c 8 env PATH=… EV_CAP_H=… bash …/rerun_w2c.sh < /dev/null
# 彩排（开发机）：TINY=1 CODE=… REPO=… LOGS=… PY=… W2C_CK=<tiny 检查点> bash rerun_w2c.sh
set -uo pipefail
TINY=${TINY:-0}
RUN_ID=0072r
M=qwen3.5-9b
SPLITS=(test_refcoco_unc_testA test_refcoco_unc_testB "test_refcoco+_unc_testA" "test_refcoco+_unc_testB" test_refcocog_umd_test)
if [ "$TINY" = 1 ]; then
  : "${CODE:?}" "${REPO:?}" "${LOGS:?}" "${PY:?}" "${W2C_CK:?}"
  NODE=tiny; M=qwen3.5-0.8b; SPLITS=(devtune800)
  T=(--tiny); D=(); CW=1
  export EV_NO_GIT=1
  OUT=$REPO/experiments/W2/0072_w2_test_tiny
else
  NODE=hpcgpu06
  VI=/home/mail/chenyuma/visint
  CODE=$VI/wt/wt0072
  REPO=$VI/repo
  LOGS=$VI/logs
  PY=$VI/env/bin/python
  W2C_CK=$REPO/experiments/W2/0069_w2c_hpcgpu06/0069_train/w2c/qwen3.5-9b/seed0/ckpt/final/model
  export HF_HOME=$VI/hf HF_HUB_CACHE=$VI/hf/hub HF_HUB_OFFLINE=1
  export TMPDIR=$VI/tmp PIP_NO_CACHE_DIR=1 NLTK_DATA=$VI/b200/nltk_data
  export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONHASHSEED=0
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export TRITON_CACHE_DIR=$VI/cache/$NODE/triton TORCHINDUCTOR_CACHE_DIR=$VI/cache/$NODE/inductor XDG_CACHE_HOME=$VI/cache/$NODE/xdg
  T=(); D=(--data-root "$VI/b200/data"); CW=8
  export EV_PAUSE=$LOGS/train.pause.$NODE EV_STATUS_GLOB="eval_0074.status" EV_NO_GIT=1
  export EV_DF_PATH=$VI
  CLAIM_MAX_S=${CLAIM_MAX_S:-86400}
  EV_CAP_S=$(awk -v h="${EV_CAP_H:?任务书的上限（小时）}" 'BEGIN{print int(h * 3600)}')
  OUT=$REPO/experiments/W2/0072_w2_test_hpcgpu06
fi
# shellcheck disable=SC1091
source "$CODE/scripts/server/eval_lib.sh"
cd "$CODE" || exit 1
[ "$TINY" = 1 ] || [ "${CUDA_VISIBLE_DEVICES:-}" = 0 ] || { progress "blocked: 不在 srun 的资源范围里"; exit 1; }
if [ "$TINY" != 1 ] && ! "$PY" -c "import json,sys; sys.exit(0 if json.load(open('configs/w2.json'))['frozen'] is True else 1)"; then
  progress "blocked: configs/w2.json 没有冻结"; exit 1
fi
# 模型目录必须恰好是 manifest 列的文件（加载器的规则；这次失败的原因）
if ! "$PY" - "$W2C_CK" <<'PYEOF'
import json, sys
from pathlib import Path
ck = Path(sys.argv[1])
man = json.loads((ck / "manifest.json").read_text(encoding="utf-8"))
files = man.get("files") or {}
listed = set(files) if isinstance(files, dict) else {f["path"] if isinstance(f, dict) else str(f) for f in files}
have = {str(p.relative_to(ck)) for p in ck.rglob("*") if p.is_file() and p.name != "manifest.json"
        and not any(part.startswith(".") for part in p.relative_to(ck).parts)}  # 同加载器：以 "." 开头的不算
extra, missing = sorted(have - listed), sorted(listed - have)
print(f"manifest files {len(listed)}; extra {extra}; missing {missing}")
sys.exit(1 if extra or missing else 0)
PYEOF
then progress "blocked: $W2C_CK 里的文件与 manifest 不符"; exit 1; fi
NOK=0
claim_gpu "0072-addendum-1.md" || { progress "blocked: gpu not free after ${CLAIM_MAX_S}s"; exit 1; }
echo "$(now) rerun_w2c start code $(git -C "$CODE" rev-parse HEAD 2>/dev/null || echo unknown) node $NODE checkpoint manifest $(sha "$W2C_CK/manifest.json" | cut -d' ' -f1)" >> "$OUT/steps.log"
for SP in "${SPLITS[@]}"; do
  # 模型目录修好之后 RUN-0072 自己的 w2c 步骤可能已经跑完了这个划分：metrics.json 的 complete 为 true 就跳过
  if "$PY" -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('complete') is True else 1)" \
      "$OUT/$M/refseg/w2c/$SP/metrics.json" 2>/dev/null; then
    echo "$(now) w2c_rerun_${M}_$SP skip（RUN-0072 已跑完这个划分）" | tee -a "$OUT/steps.log"; continue
  fi
  step "w2c_rerun_${M}_$SP" "$PY" scripts/run_w2.py --model "$M" ${T[@]+"${T[@]}"} ${D[@]+"${D[@]}"} --split "$SP" --w2-config configs/w2.json \
    --out "$OUT" --candidate-workers "$CW" --task refseg --setting w2c --checkpoint "$W2C_CK" || NOK=$((NOK + 1))
done
echo "$(now) rerun_w2c done; failed steps: $NOK" >> "$OUT/steps.log"
progress "done $RUN_ID (failed steps: $NOK)"
push_out "W2 test 0072 w2c rerun done (failed steps: $NOK)"
