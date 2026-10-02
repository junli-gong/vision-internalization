#!/usr/bin/env bash
# RUN-0022 第 2 步：按固定 revision 下载模型权重、逐文件核对哈希，落地开发数据，下 1 张测试图。需要联网。
# 用法：bash prefetch.sh <工作目录>
# - Gemma 4 是 gated 模型：先用接受过 Gemma 条款的 HF 账号 `hf auth login`（或 export HF_TOKEN=...）；拿不到就设 SKIP_GEMMA=1。
# - huggingface.co 慢或连不上时，可以 export HF_ENDPOINT=https://hf-mirror.com：每个文件都按清单核对哈希，镜像给错的文件会被查出来。
set -euo pipefail
WORK=${1:?用法: bash prefetch.sh <工作目录>}
PKG=$(cd "$(dirname "$0")" && pwd)
# shellcheck disable=SC1091
source "$WORK/venv/bin/activate"
export HF_HOME=${HF_HOME:-$WORK/hf}
export TMPDIR=${TMPDIR:-$WORK/tmp}
export PIP_NO_CACHE_DIR=1
mkdir -p "$TMPDIR" "$WORK/prefetch_logs"
cd "$WORK/vision-internalization"
KEYS="qwen3.5-0.8b qwen3.5-2b qwen3.5-4b qwen3.5-9b internvl3.5-2b-hf internvl3.5-4b-hf"
if [ "${SKIP_GEMMA:-0}" != "1" ]; then KEYS="$KEYS gemma-4-e2b-it gemma-4-e4b-it"; fi
echo "[prefetch] 权重：$KEYS（约 50 GB；含 Gemma 约 75 GB）"
for key in $KEYS; do
  python - "$key" <<'PY'
import sys
from huggingface_hub import snapshot_download
from src.adapters.registry import get_model
e = get_model(sys.argv[1])
p = snapshot_download(e.repo, revision=e.revision)
print(sys.argv[1], e.repo, e.revision, p, flush=True)
PY
done
# 单元测试会读其余模型仓库的小文件（config、tokenizer、processor、README），按固定 revision 先下好，不下权重
python - <<'PY'
from huggingface_hub import snapshot_download
from src.adapters.registry import MODELS
pats = ["*.json", "*.txt", "*.model", "*.jinja", "*.tiktoken", "*.py", "*.md"]
for key, e in MODELS.items():
    try:
        snapshot_download(e.repo, revision=e.revision, allow_patterns=pats)
        print("small files", key)
    except Exception as exc:  # Gemma 没有权限时照常往下走
        print("SKIP small files", key, type(exc).__name__, exc)
PY
# 逐文件核对哈希（清单是 B200 上 RUN-0009 生成的：safetensors 用 sha256，其余用 git blob sha1）
python "$PKG/verify_weights.py" experiments/P0/0009_ifeval_rescore/weights_manifest.json $KEYS | tee "$WORK/prefetch_logs/verify_weights.txt"
# 开发数据：RefCOCO 三个数据集（约 1.1 GB），训练接口核对要用。失败不影响其余步骤，回传时说明。
mkdir -p "$WORK/data"
python scripts/data/land_e1.py refcoco --root "$WORK/data" --tmpdir "$TMPDIR" --budget-gb 5 \
  > "$WORK/prefetch_logs/land_refcoco.log" 2>&1 && echo "[prefetch] refcoco 落地 rc=0" \
  || echo "[prefetch] refcoco 落地失败（rc=$?），日志见 prefetch_logs/land_refcoco.log；照常往下走"
curl -fL -o "$TMPDIR/000000039769.jpg" http://images.cocodataset.org/val2017/000000039769.jpg
echo "dea9e7ef97386345f7cff32f9055da4982da5471c48d575146c796ab4563b04e  $TMPDIR/000000039769.jpg" | sha256sum -c -
du -sh "$HF_HOME" "$WORK/data" 2>/dev/null || true
echo "[prefetch] 完成。下一步：bash run_all.sh $WORK"
