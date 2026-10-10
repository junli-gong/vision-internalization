B=/home/mail/chenyuma/minimax-h3-fp8
SQ() { echo "## squeue $(date -u +%FT%TZ) [$1]"; squeue -h -j 157296,157297 -o "%i %T %M %N"; }
SQ "park step1 before"
stat -c "%s %y %a %i %n" $B/start_minimax_h3_a100_1gpu_fp16.sh; sha256sum $B/start_minimax_h3_a100_1gpu_fp16.sh; ls -la $B/run/ | grep -E "park|startup-failed" || echo "no park/startup-failed files"
squeue -s -j 157296,157297 -o "%i %j %M %N"
for n in hpcgpu06 hpcgpu07; do echo "--- $n"; ssh -n -o BatchMode=yes $n "ps -u chenyuma -o pid,ppid,pgid,etimes,args --forest | egrep '[s]lurm_script|[s]tart_minimax|[m]ain.py --listen|[u]vicorn api' | cut -c1-170; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | head"; done
SQ "park step1 after"
