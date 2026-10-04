#!/usr/bin/env bash
# RUN-0029 第 1 步：换代码快照（数据、venv、权重都不动），在 GPU 上跑上采样的单元测试（含 CUDA 用例）。约 5 分钟。
PKG=$(cd "$(dirname "$0")" && pwd)
# shellcheck disable=SC1091
source "$PKG/common.sh" "$@"
if [ "$TINY" != 1 ]; then
  if [ -d "$CODE" ]; then mv "$CODE" "$CODE.before0029.$(date +%s)"; fi
  tar -xzf "$PKG"/vision-internalization-*.tar.gz -C "$WORK" && cp "$PKG/VERSION.txt" "$CODE/PACKAGE_VERSION.txt"
  step update_code $?
  cd "$CODE" || exit 1
  pip install --dry-run -r requirements.txt 2>&1 | grep -i 'would install' > "$OUT/pip_dryrun.txt"; echo "(空 = 没有要新装的包)" >> "$OUT/pip_dryrun.txt"
  { date -u; hostname; cat PACKAGE_VERSION.txt; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv; df -h "$WORK"; } > "$OUT/facts.txt" 2>&1
fi
cd "$CODE" || exit 1
# 探针卷积与矩阵乘的精度设置（D64-4：只记录，不改）
python - > "$OUT/torch_backends.json" 2>&1 <<'PY'
import json, torch
b = torch.backends
def g(o, k):
    try:
        return getattr(o, k)
    except Exception as e:  # noqa: BLE001
        return f"<{type(e).__name__}>"
conv = getattr(b.cudnn, "conv", None)
print(json.dumps({"torch": torch.__version__, "cudnn.allow_tf32": g(b.cudnn, "allow_tf32"),
                  "cudnn.deterministic": g(b.cudnn, "deterministic"), "cudnn.benchmark": g(b.cudnn, "benchmark"),
                  "cudnn.conv.fp32_precision": g(conv, "fp32_precision") if conv is not None else None,
                  "cuda.matmul.fp32_precision": g(b.cuda.matmul, "fp32_precision"),
                  "cuda_available": torch.cuda.is_available()}, indent=1))
PY
step torch_backends $?
python -m pytest tests/test_e2_bilinear.py -q -rs -p no:cacheprovider > "$OUT/test_e2_bilinear.log" 2>&1
step test_e2_bilinear $?
tail -3 "$OUT/test_e2_bilinear.log"
echo "[step1] 完成。下一步：bash step2_estimate.sh $WORK"
