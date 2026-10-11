set -u
B=/home/mail/chenyuma/minimax-h3-fp8
SQ() { echo "## squeue $(date -u +%FT%TZ) [$1]"; squeue -h -j 157296,157297 -o "%i %T %M %N"; [ $(squeue -h -j 157296,157297 -t RUNNING | wc -l) -eq 2 ] || { echo "NOT BOTH RUNNING - STOP"; exit 9; }; }
TREE() { ssh -n -o BatchMode=yes hpcgpu06 "ps -u chenyuma -o pid,ppid,pgid,etimes,args --forest | egrep '[s]lurm_script|[s]tart_minimax|[m]ain.py --listen|[u]vicorn api|[s]leep infinity' | cut -c1-400"; }
SQ "step5-06 before"
echo "--- tree before"; T=$(TREE); echo "$T"
chk() { echo "$T" | awk -v pid=$1 -v pp=$2 -v pat="$3" '$1==pid && $2==pp && index($0,pat)>0 {f=1} END {exit !f}' || { echo "PID check failed: $1 (ppid $2, $3) - STOP"; exit 8; }; }
chk 3065231 3065227 "slurm_script"; chk 3065235 3065231 "start_minimax_h3_a100_1gpu_fp16.sh"; chk 3065243 3065235 "main.py --listen 127.0.0.1 --port 18606"; chk 3066512 3065235 "uvicorn"
echo "PIDs as recorded"
touch $B/run/park-hpcgpu06; ls -la $B/run/park-hpcgpu06
srun --jobid=157296 --overlap -n1 bash -c 'kill 3066512; echo "kill uvicorn rc=$?"; sleep 2; kill 3065243; echo "kill comfy rc=$?"' < /dev/null
sleep 10
echo "--- tree after 10s"; TREE
ssh -n hpcgpu06 'for p in 3066512 3065243 3065235 3065231; do if kill -0 $p 2>/dev/null; then echo "$p alive"; else echo "$p gone"; fi; done'
SQ "step5-06 after"
