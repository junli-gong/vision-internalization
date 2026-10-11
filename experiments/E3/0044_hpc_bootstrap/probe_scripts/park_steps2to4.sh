set -u
B=/home/mail/chenyuma/minimax-h3-fp8; F=$B/start_minimax_h3_a100_1gpu_fp16.sh; NEW=$B/.start_minimax_h3_a100_1gpu_fp16.sh.new
SQ() { echo "## squeue $(date -u +%FT%TZ) [$1]"; squeue -h -j 157296,157297 -o "%i %T %M %N"; [ $(squeue -h -j 157296,157297 -t RUNNING | wc -l) -eq 2 ] || { echo "NOT BOTH RUNNING - STOP"; exit 9; }; }
EXP=e42bfde15b8d62f0b32f5251338ecab520252d75098c6b5b4d3bcf321f1afae9
SQ "step2 before"
[ "$(sha256sum < $F | cut -d' ' -f1)" = $EXP ] || { echo "start script changed - STOP"; exit 8; }
[ -e $F.orig ] && { echo ".orig already exists - STOP"; exit 7; }
cp -p $F $F.orig; [ "$(sha256sum < $F.orig | cut -d' ' -f1)" = $EXP ] || { echo ".orig sha mismatch - STOP"; exit 6; }
stat -c "%s %y %a %i %n" $F $F.orig; SQ "step2 after"
SQ "step3 before"
cp /home/mail/chenyuma/visint/probe/0044/park_start.sh $NEW; chmod 755 $NEW; bash -n $NEW || { echo "syntax error - STOP"; rm -f $NEW; exit 5; }
sha256sum $NEW; stat -c "%a %n" $NEW; SQ "step3 after"
SQ "step4 before"
I0=$(stat -c %i $F); mv -f $NEW $F; I1=$(stat -c %i $F)
stat -c "%s %y %a %i %n" $F; sha256sum $F; echo "inode before=$I0 after=$I1 $( [ $I0 != $I1 ] && echo changed || echo SAME)"
[ "$(sha256sum < $F.orig | cut -d' ' -f1)" = $EXP ] && echo ".orig intact"
ls -la $B | grep start_minimax_h3_a100_1gpu_fp16
SQ "step4 after"
