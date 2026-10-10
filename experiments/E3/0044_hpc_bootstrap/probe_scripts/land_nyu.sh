#!/bin/bash
# RUN-0044 §4: native landing of nyu_depth_train from hf-mirror (HF Range on nyu_depth_v2.zip; central-directory sha256 + member CRC32 checked by the lander).
export VI=/home/mail/chenyuma/visint
export HF_HOME=$VI/hf
export HF_HUB_CACHE=$VI/hf/hub
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export TMPDIR=$VI/tmp
export XDG_CACHE_HOME=$VI/cache/hpcgpu06/xdg
export PIP_NO_CACHE_DIR=1
export OMP_NUM_THREADS=16
export PATH=$VI/env/bin:$PATH
cd $VI/wt/wt0044 || exit 1
echo "== $(date -u +%FT%TZ) host=$(hostname -s) wt=$(git -C $VI/wt/wt0044 rev-parse HEAD 2>/dev/null || cat $VI/wt/wt0044.commit)"
python scripts/data/land_train.py nyu_depth_train --dry-run --root $VI/data/train_data --tmpdir $VI/tmp 2>&1 | tail -5
t0=$(date +%s)
for a in 1 2 3; do python scripts/data/land_train.py nyu_depth_train --root $VI/data/train_data --tmpdir $VI/tmp --hf-base https://hf-mirror.com --workers 8 2>&1 | grep -vE "it/s\]" | tail -40; rc=${PIPESTATUS[0]}
  echo "=== $(date -u +%FT%TZ) land attempt $a rc=$rc $(( $(date +%s)-t0 ))s"; [ $rc -eq 0 ] && break; done
echo "== done $(date -u +%FT%TZ)"
