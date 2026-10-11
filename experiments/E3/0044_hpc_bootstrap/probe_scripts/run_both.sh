#!/bin/bash
# RUN-0044 §7 then §8 on this node (after MiniMax parked on both nodes, 2026-10-10 18:29Z / 18:36Z)
D=/home/mail/chenyuma/visint/probe/0044/driver_0044.sh
bash $D consistency; echo "consistency rc=$?"
bash $D throughput; echo "throughput rc=$?"
