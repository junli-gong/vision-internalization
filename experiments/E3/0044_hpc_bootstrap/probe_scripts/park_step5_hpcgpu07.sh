set -u
B=/home/mail/chenyuma/minimax-h3-fp8
SQ() { echo "## squeue $(date -u +%FT%TZ) [$1]"; squeue -h -j 157296,157297 -o "%i %T %M %N"; [ $(squeue -h -j 157296,157297 -t RUNNING | wc -l) -eq 2 ] || { echo "NOT BOTH RUNNING - STOP"; exit 9; }; }
TREE() { ssh -n -o BatchMode=yes hpcgpu07 "ps -u chenyuma -o pid,ppid,pgid,etimes,args --forest | egrep '[s]lurm_script|[s]tart_minimax|[m]ain.py --listen|[u]vicorn api|[s]leep (300|infinity)' | cut -c1-400"; }
SQ "step5-07 before"
[ "$(sha256sum < $B/start_minimax_h3_a100_1gpu_fp16.sh | cut -d' ' -f1)" = 6a9acbf76c594448a49038522b2207f0ffb0730a5f74825631373b46c77df906 ] || { echo "new start script not in place - STOP"; exit 7; }
echo "--- tree before"; T=$(TREE); echo "$T" | cut -c1-200
chk() { echo "$T" | awk -v pid=$1 -v pp=$2 -v pat="$3" '$1==pid && $2==pp && index($0,pat)>0 {f=1} END {exit !f}' || { echo "PID check failed: $1 (ppid $2, $3) - STOP"; exit 8; }; }
chk 4010090 4010086 "slurm_script"; chk 4010094 4010090 "start_minimax_h3_a100_1gpu_fp16.sh"; chk 4012165 4010094 "main.py --listen 127.0.0.1 --port 18607"; chk 4013622 4010094 "uvicorn"
echo "PIDs as recorded"
touch $B/run/park-hpcgpu07; ls -la $B/run/park-hpcgpu07
srun --jobid=157297 --overlap -n1 bash -c '
st() { for p in 4013622 4012165 4010094 4010090; do s=$(awk "/^State/{print \$2}" /proc/$p/status 2>/dev/null); echo "  $p ${s:-gone}"; done; }
echo "states before:"; st
kill 4013622; echo "SIGTERM uvicorn rc=$?"; sleep 2; kill 4012165; echo "SIGTERM comfy rc=$?"
kill -CONT 4013622; echo "SIGCONT uvicorn rc=$?"; kill -CONT 4012165; echo "SIGCONT comfy rc=$?"
for i in $(seq 1 15); do sleep 2; a=0; for p in 4013622 4012165; do kill -0 $p 2>/dev/null && a=1; done; [ $a = 0 ] && { echo "both exited after $((i*2))s"; break; }; done
for p in 4013622 4012165; do if kill -0 $p 2>/dev/null; then echo "$p still alive after 30s -> kill -9"; kill -9 $p; echo "kill -9 $p rc=$?"; fi; done
sleep 5; echo "states after:"; st' < /dev/null
echo "--- tree after"; TREE | cut -c1-200
tail -2 $B/logs/slurm-h3-fp16-a100-07-157297.log; tail -2 $B/logs/api-a100-1gpu-hpcgpu07.log
SQ "step5-07 after"
