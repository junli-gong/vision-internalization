#!/usr/bin/env bash
# RUN-0014 第 3 步：在 GPU 节点上跑全部检查。不需要联网（权重已由 prefetch.sh 下好）。
# 用法：bash run_all.sh <工作目录> [结果目录，默认 <工作目录>/results_0014]
# 某一步失败不会中断后面的步骤；每一步的退出码记在 <结果目录>/steps.log。
set -uo pipefail
WORK=${1:?用法: bash run_all.sh <工作目录> [结果目录]}
OUT=${2:-$WORK/results_0014}
# shellcheck disable=SC1091
source "$WORK/venv/bin/activate"
export HF_HOME=${HF_HOME:-$WORK/hf}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TMPDIR=${TMPDIR:-$WORK/tmp}
mkdir -p "$OUT" "$TMPDIR"
cd "$WORK/vision-internalization"
step() { echo "$(date -u +%H:%M:%S) $1 rc=$2" | tee -a "$OUT/steps.log"; }

# 1. 机器与环境
{
  date -u; hostname; cat PACKAGE_VERSION.txt
  nvidia-smi
  df -h "$WORK"; free -g 2>/dev/null || true
  python -V
  python -c "import torch;print('torch',torch.__version__,torch.version.cuda,torch.cuda.get_device_name(0),torch.cuda.get_device_capability(0))"
  pip freeze
} > "$OUT/facts.txt" 2>&1
step facts $?

# 2. 全量单元测试（CPU 为主）
python -m pytest tests/ -q -rs -p no:cacheprovider > "$OUT/pytest.txt" 2>&1
step pytest $?

# 3. 小型随机模型的冒烟（CPU，不用权重）
for m in qwen3.5-0.8b gemma-4-e2b-it internvl3.5-2b-hf; do
  python scripts/smoke_adapter.py --tiny --model "$m" --out "$OUT/smoke_tiny_$m.json" > "$OUT/smoke_tiny_$m.log" 2>&1
  step "smoke_tiny_$m" $?
done

# 4. 真实权重冒烟（GPU，bf16）
IMG="$WORK/tmp/000000039769.jpg"
python scripts/smoke_adapter.py --model qwen3.5-0.8b --device cuda --dtype bf16 --image "$IMG" \
  --out "$OUT/smoke_qwen3.5-0.8b.json" > "$OUT/smoke_qwen3.5-0.8b.log" 2>&1
step smoke_qwen3.5-0.8b $?
python scripts/smoke_adapter.py --model internvl3.5-2b-hf --device cuda --dtype bf16 --image "$IMG" \
  --think-max-new-tokens 2048 --out "$OUT/smoke_internvl3.5-2b-hf.json" > "$OUT/smoke_internvl3.5-2b-hf.log" 2>&1
step smoke_internvl3.5-2b-hf $?
if [ "${SKIP_GEMMA:-0}" != "1" ]; then
  python scripts/smoke_adapter.py --model gemma-4-e2b-it --device cuda --dtype bf16 --image "$IMG" \
    --out "$OUT/smoke_gemma-4-e2b-it.json" > "$OUT/smoke_gemma-4-e2b-it.log" 2>&1
  step smoke_gemma-4-e2b-it $?
fi

# 5. 生成吞吐与批处理一致性（同一张测试图，8 条不同指令，三种批大小）
python - "$IMG" "$OUT/vb_rows.jsonl" <<'PY'
import json, sys
img, out = sys.argv[1], sys.argv[2]
prompts = [
    "Describe this image in one sentence.",
    "How many cats are in this image? Answer with a single number.",
    "What color is the blanket?",
    "Locate the cat on the left in this image and output its bounding box.",
    "Which cat is closer to the camera, the left one or the right one?",
    "List every object you can see in the image.",
    "Where are the remote controls in the image?",
    "Describe the spatial layout of the objects in this image.",
]
with open(out, "w") as f:
    for i, p in enumerate(prompts):
        f.write(json.dumps({"id": f"r{i}", "image": img, "instruction": p,
                            "enable_thinking": False, "max_new_tokens": 128}) + "\n")
PY
for m in qwen3.5-0.8b internvl3.5-2b-hf; do
  python scripts/validate_batching.py --model "$m" --jsonl "$OUT/vb_rows.jsonl" --batch-sizes 4 8 \
    --out "$OUT/vb_$m.json" > "$OUT/vb_$m.log" 2>&1
  step "vb_$m" $?
done
if [ "${SKIP_GEMMA:-0}" != "1" ]; then
  python scripts/validate_batching.py --model gemma-4-e2b-it --jsonl "$OUT/vb_rows.jsonl" --batch-sizes 4 8 \
    --out "$OUT/vb_gemma-4-e2b-it.json" > "$OUT/vb_gemma-4-e2b-it.log" 2>&1
  step vb_gemma-4-e2b-it $?
fi

df -h "$WORK" >> "$OUT/facts.txt"
echo "[run_all] 完成。下一步：bash collect_results.sh $WORK"
