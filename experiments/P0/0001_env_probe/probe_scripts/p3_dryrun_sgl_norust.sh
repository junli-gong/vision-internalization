#!/bin/bash
# RUN-0001 §3.4 retry: SGLang main dry-run without Rust extensions (no cargo on host). TMPDIR on /dev/shm.
source /home/ubuntu/workspace/wzg/gjl/vision-internalization/scripts/env.sh
L=$PROJ/logs/0001
export TMPDIR=/dev/shm/visint/tmp-pip-sgl SGLANG_BUILD_RUST_EXTS=none
rm -rf $TMPDIR; mkdir -p $TMPDIR
cd $TMPDIR && (time pip install --dry-run --no-cache-dir --report $L/pip_dryrun_sglang_main_norust.json 'sglang[all] @ git+https://github.com/sgl-project/sglang.git#subdirectory=python') > $L/pip_dryrun_sglang_main_norust.log 2>&1
echo "exit=$?" > $L/pip_dryrun_sgl_norust.done
