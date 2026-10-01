#!/bin/bash
# RUN-0010 driver: land E1 sources one by one in land_e1.py's `all` order, with a disk-ledger entry before/after each source.
# nyuv2 runs without --nyu-local-dir: the default dir ($ROOT/datasets/nyuv2, other project, read-only) has no labeled .mat.
source /home/ubuntu/workspace/wzg/gjl/vision-internalization/scripts/env.sh
export TMPDIR=/dev/shm/visint/tmp; mkdir -p $TMPDIR
L=$PROJ/logs/0010; DC=$PROJ/scripts/diskcheck10.sh
cd $REPO
for S in refcoco coco_val2017 odinw13 countbench nyuv2 blink_reldepth cvbench embspatial; do
  $DC "before land $S" | tee -a $L/land_driver.log || { echo "DISK RED LINE" | tee -a $L/land_driver.log; exit 3; }
  T0=$(date +%s)
  python scripts/data/land_e1.py $S --root $PROJ/data --tmpdir /dev/shm/visint/tmp > $L/land_$S.log 2>&1
  RC=$?; T1=$(date +%s)
  echo "=== $S rc=$RC wall=$((T1-T0))s $(date -Is)" | tee -a $L/land_driver.log
  $DC "after land $S" | tee -a $L/land_driver.log
  if [ $RC -eq 3 ] || [ $RC -eq 4 ]; then echo "=== STOP: budget/red line on $S (rc=$RC)" | tee -a $L/land_driver.log; exit $RC; fi
done
echo "=== land driver done $(date -Is)" | tee -a $L/land_driver.log
