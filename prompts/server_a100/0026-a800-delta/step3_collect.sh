#!/usr/bin/env bash
# RUN-0026 第 3 步：打包结果（不含检查点与数据），传回 Google Drive 的 vision-internalization-a100/ 文件夹。
set -euo pipefail
WORK=${1:?用法: bash step3_collect.sh <工作目录>}
NAME="a800_results_0026_$(date -u +%Y%m%dT%H%MZ).tar.gz"
tar -czf "$WORK/$NAME" -C "$WORK" --exclude='*/ckpt/*' --exclude='*.pt' --exclude='*.safetensors' results_0026
ls -la "$WORK/$NAME"
echo "把 $WORK/$NAME 和 notes.txt 上传到 Google Drive 的 vision-internalization-a100/ 文件夹。"
