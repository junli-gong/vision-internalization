#!/usr/bin/env bash
# RUN-0022 第 3 步：在 GPU 上跑全部检查。不需要联网（权重与数据已由 prefetch.sh 下好）。
# 用法：bash run_all.sh <工作目录> [结果目录，默认 <工作目录>/results_0022]
# 某一步失败不会中断后面的步骤；每一步的退出码记在 <结果目录>/steps.log。预计 2.5–3.5 小时，大头是全量测试（CPU）。
set -uo pipefail
WORK=${1:?用法: bash run_all.sh <工作目录> [结果目录]}
OUT=${2:-$WORK/results_0022}
# shellcheck disable=SC1091
source "$WORK/venv/bin/activate"
export HF_HOME=${HF_HOME:-$WORK/hf}
export HF_HUB_OFFLINE=1
export TMPDIR=${TMPDIR:-$WORK/tmp}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 实例默认的 OMP_NUM_THREADS=0 会触发 libgomp 警告并拖慢 CPU 测试（RUN-0014）；按核数设，最多 16
NCPU=$(nproc 2>/dev/null || echo 8); [ "$NCPU" -gt 16 ] && NCPU=16
export OMP_NUM_THREADS=${OMP_NUM_THREADS_OVERRIDE:-$NCPU}
mkdir -p "$OUT" "$TMPDIR"
cd "$WORK/vision-internalization"
step() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $1 rc=$2" | tee -a "$OUT/steps.log"; }
IMG="$TMPDIR/000000039769.jpg"
B200="experiments/P0/0017_health_v3_base"   # B200 上的健康检查 v3（RUN-0017），随代码快照一起

# 1. 机器与环境
{
  date -u; hostname; cat PACKAGE_VERSION.txt
  nvidia-smi; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
  df -h "$WORK"; free -g 2>/dev/null || true; echo "nproc $(nproc)  OMP_NUM_THREADS=$OMP_NUM_THREADS"
  python -V
  python -c "import torch,torchvision;print('torch',torch.__version__,'torchvision',torchvision.__version__,torch.version.cuda,torch.cuda.get_device_name(0),torch.cuda.get_device_capability(0))"
} > "$OUT/facts.txt" 2>&1
step facts $?
cp "$WORK/pip_freeze.txt" "$WORK/pip_check.txt" "$WORK/pip_vs_b200.txt" "$OUT/" 2>/dev/null
cp -r "$WORK/prefetch_logs" "$OUT/" 2>/dev/null

# 2. 全量单元测试（CPU 为主）。-rf 列出失败的测试名
python -m pytest tests/ -q -rfs -p no:cacheprovider > "$OUT/pytest.txt" 2>&1
step pytest $?

# 3. E5 生成器的跨机器摘要（300 个，a、b 为版本 4）
python scripts/synth_pin_digests.py check > "$OUT/synth_digests.txt" 2>&1
step synth_digests $?

# 4. 小型随机模型的冒烟（CPU，不用权重）
for m in qwen3.5-0.8b gemma-4-e2b-it internvl3.5-2b-hf; do
  python scripts/smoke_adapter.py --tiny --model "$m" --out "$OUT/smoke_tiny_$m.json" > "$OUT/smoke_tiny_$m.log" 2>&1
  step "smoke_tiny_$m" $?
done

# 5. 真实权重冒烟（GPU，bf16）
for m in qwen3.5-0.8b qwen3.5-9b internvl3.5-2b-hf; do
  extra=""; [ "$m" = internvl3.5-2b-hf ] && extra="--think-max-new-tokens 2048"
  python scripts/smoke_adapter.py --model "$m" --device cuda --dtype bf16 --image "$IMG" $extra \
    --out "$OUT/smoke_$m.json" > "$OUT/smoke_$m.log" 2>&1
  step "smoke_$m" $?
done
if [ "${SKIP_GEMMA:-0}" != "1" ]; then
  python scripts/smoke_adapter.py --model gemma-4-e2b-it --device cuda --dtype bf16 --image "$IMG" \
    --out "$OUT/smoke_gemma-4-e2b-it.json" > "$OUT/smoke_gemma-4-e2b-it.log" 2>&1
  step smoke_gemma-4-e2b-it $?
fi

# 6. 健康检查 v3（只生成，不评 IFEval），与 B200 的同一套题逐题比较，量出跨机器差异
for m in qwen3.5-0.8b qwen3.5-9b internvl3.5-2b-hf; do
  python scripts/health_check.py --model "$m" --device cuda --dtype bf16 --skip-ifeval-scoring \
    --out "$OUT/health_$m.json" > "$OUT/health_$m.log" 2>&1
  step "health_$m" $?
done
python - "$OUT" "$B200" > "$OUT/health_vs_b200.txt" 2>&1 <<'PY'
import json, sys
from pathlib import Path
out, ref = Path(sys.argv[1]), Path(sys.argv[2])
for p in sorted(out.glob("health_*.json")):
    m = p.stem[len("health_"):]
    r = ref / f"health_{m}.json"
    if not r.exists():
        print(m, "no B200 reference"); continue
    a, b = json.loads(p.read_text()), json.loads(r.read_text())
    for bench, recs in a["records"].items():
        if bench not in b["records"]:
            continue
        rb = {x["id"]: x for x in b["records"][bench]}
        key = "choice" if bench == "mmlu_pro_letter" else "raw_text"
        same = sum(1 for x in recs if x["id"] in rb and x.get(key) == rb[x["id"]].get(key))
        print(f"{m:20s} {bench:18s} identical to B200 ({key}): {same}/{len(recs)}")
    sa, sb = a.get("scores", {}), b.get("scores", {})
    for k in ("mmlu_pro_letter", "mmlu_pro_free256", "gsm8k"):
        print(f"{m:20s} {k:18s} score here={json.dumps(sa.get(k))}  b200={json.dumps(sb.get(k))}")
PY
step health_vs_b200 $?

# 7. 训练接口核对（受限 logits 与 LoRA 合并的一致率；要 RefCOCO 开发数据，prefetch 没落地就跳过）
if [ -f "$WORK/data/refcoco/MANIFEST.json" ]; then
  for m in qwen3.5-0.8b qwen3.5-9b; do
    python scripts/check_training_interfaces.py --model "$m" --checks restricted merge --n 200 \
      --data-root "$WORK/data" --out "$OUT/train_iface_$m.json" > "$OUT/train_iface_$m.log" 2>&1
    step "train_iface_$m" $?
  done
else
  step train_iface_skipped_no_refcoco 0
fi

df -h "$WORK" >> "$OUT/facts.txt"
echo "[run_all] 完成。下一步：bash collect_results.sh $WORK"
