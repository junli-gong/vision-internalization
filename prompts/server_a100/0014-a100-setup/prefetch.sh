#!/usr/bin/env bash
# RUN-0014 第 2 步：在登录节点上（需要联网）按固定 revision 下载 3 个模型的权重和 1 张测试图。
# 用法：bash prefetch.sh <工作目录>
# Gemma 4 是 gated 模型：先用接受过 Gemma 条款的 HF 账号 `hf auth login`（或 export HF_TOKEN=...）。
# 拿不到 Gemma 的话，设 SKIP_GEMMA=1，其余照跑，并在回传时说明。
set -euo pipefail
WORK=${1:?用法: bash prefetch.sh <工作目录>}
# shellcheck disable=SC1091
source "$WORK/venv/bin/activate"
export HF_HOME=${HF_HOME:-$WORK/hf}
cd "$WORK/vision-internalization"
KEYS="qwen3.5-0.8b internvl3.5-2b-hf"
if [ "${SKIP_GEMMA:-0}" != "1" ]; then KEYS="$KEYS gemma-4-e2b-it"; fi
for key in $KEYS; do
  python - "$key" <<'PY'
import sys
from huggingface_hub import snapshot_download
from src.adapters.registry import get_model
e = get_model(sys.argv[1])
p = snapshot_download(e.repo, revision=e.revision)
print(sys.argv[1], e.repo, e.revision, p)
PY
done
# 单元测试会读其余 5 个模型仓库的小文件（config、tokenizer、processor），也按固定 revision 先下好，不下权重。
python - <<'PY'
from huggingface_hub import snapshot_download
from src.adapters.registry import MODELS
pats = ["*.json", "*.txt", "*.model", "*.jinja", "*.tiktoken", "*.py"]
for key, e in MODELS.items():
    try:
        snapshot_download(e.repo, revision=e.revision, allow_patterns=pats)
        print("small files", key)
    except Exception as exc:  # Gemma 没有权限时照常往下走，回传时说明
        print("SKIP small files", key, type(exc).__name__, exc)
PY
mkdir -p "$WORK/tmp"
curl -fL -o "$WORK/tmp/000000039769.jpg" http://images.cocodataset.org/val2017/000000039769.jpg
echo "dea9e7ef97386345f7cff32f9055da4982da5471c48d575146c796ab4563b04e  $WORK/tmp/000000039769.jpg" | sha256sum -c -
du -sh "$HF_HOME"
echo "[prefetch] 完成。下一步：在 GPU 节点上 bash run_all.sh $WORK（集群用 sbatch，见 slurm_example.sbatch）"
