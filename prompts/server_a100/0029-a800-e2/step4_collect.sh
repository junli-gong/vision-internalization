#!/usr/bin/env bash
# RUN-0029 第 4 步：打包结果（不含检查点），传回 Google Drive。第 3 步跑到一半也可以先运行它，回传已完成的部分。
set -euo pipefail
WORK=${1:?用法: bash step4_collect.sh <工作目录>}
NAME="a800_results_0029_$(date -u +%Y%m%dT%H%MZ).tar.gz"
tar -czf "$WORK/$NAME" -C "$WORK" --exclude='*/ckpt/*' --exclude='*.pt' --exclude='*.safetensors' results_0029
ls -la "$WORK/$NAME"; sha256sum "$WORK/$NAME" 2>/dev/null || shasum -a 256 "$WORK/$NAME"
echo "把 $WORK/$NAME 和 notes.txt 上传到 Google Drive 的 vision-internalization-a100/ 文件夹。"
