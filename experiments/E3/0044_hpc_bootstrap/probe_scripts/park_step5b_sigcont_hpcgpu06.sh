set -u
SQ() { echo "## squeue $(date -u +%FT%TZ) [$1]"; squeue -h -j 157296,157297 -o "%i %T %M %N"; [ $(squeue -h -j 157296,157297 -t RUNNING | wc -l) -eq 2 ] || { echo "NOT BOTH RUNNING - STOP"; exit 9; }; }
SQ "step5b-06 before"
srun --jobid=157296 --overlap -n1 bash -c '
st() { for p in 3066512 3065243 3065235 3065231; do s=$(awk "/^State/{print \$2}" /proc/$p/status 2>/dev/null); echo "  $p ${s:-gone}"; done; }
echo "states before:"; st
kill -CONT 3066512; echo "SIGCONT uvicorn rc=$?"; kill -CONT 3065243; echo "SIGCONT comfy rc=$?"
for i in $(seq 1 15); do sleep 2; a=0; for p in 3066512 3065243; do kill -0 $p 2>/dev/null && a=1; done; [ $a = 0 ] && { echo "both exited after $((i*2))s"; break; }; done
for p in 3066512 3065243; do if kill -0 $p 2>/dev/null; then echo "$p still alive after 30s -> kill -9"; kill -9 $p; echo "kill -9 $p rc=$?"; fi; done
sleep 5; echo "states after:"; st' < /dev/null
ssh -n hpcgpu06 "ps -u chenyuma -o pid,ppid,etimes,args --forest | egrep '[s]lurm_script|[s]tart_minimax|[m]ain.py --listen|[u]vicorn api|[s]leep (300|infinity)' | cut -c1-200; nvidia-smi --query-compute-apps=gpu_bus_id,pid,used_memory --format=csv,noheader | grep 65:00"
tail -3 /home/mail/chenyuma/minimax-h3-fp8/logs/slurm-h3-fp16-a100-06-157296.log
tail -4 /home/mail/chenyuma/minimax-h3-fp8/logs/api-a100-1gpu-hpcgpu06.log
SQ "step5b-06 after"
