set -u
VI=/home/mail/chenyuma/visint; B=$VI/b200
echo "## squeue $(date -u +%FT%TZ) [0047 verify]"; squeue -h -j 157296,157297 -o "%i %T %M %N"
echo "dotfiles removed: $(find $B \( -name '._*' -o -name '.DS_Store' \) -print -delete | wc -l)"
cd $B && grep -v '  logs/' SHA256SUMS.b200 > $VI/tmp/sha256_b200_nologs.txt; wc -l < $VI/tmp/sha256_b200_nologs.txt
t0=$(date +%s); nice -n 10 sha256sum -c --quiet $VI/tmp/sha256_b200_nologs.txt > $VI/logs/0047_sha256c.txt 2>&1; rc=$?
echo "sha256sum -c rc=$rc $(( $(date +%s)-t0 ))s; FAILED lines: $(grep -c 'FAILED' $VI/logs/0047_sha256c.txt); missing: $(grep -c 'No such file' $VI/logs/0047_sha256c.txt)"; head -5 $VI/logs/0047_sha256c.txt
echo "regular files under b200: $(find $B -type f | wc -l) (expected 16163 + 2 = 16165)"
find $B -type f | grep -v -e '^'$B'/SHA256SUMS.b200$' -e '^'$B'/README.md$' | sed "s|^$B/||" | sort > $VI/tmp/b200_files.txt; awk '{print $2}' $VI/tmp/sha256_b200_nologs.txt | sort | diff - $VI/tmp/b200_files.txt | head -5; echo "list diff lines: $(awk '{print $2}' $VI/tmp/sha256_b200_nologs.txt | sort | diff - $VI/tmp/b200_files.txt | wc -l)"
du -s --apparent-size -B1 $B; timeout 60 lfs quota -u chenyuma /home | sed -n 3p
echo "## squeue $(date -u +%FT%TZ) [0047 after]"; squeue -h -j 157296,157297 -o "%i %T %M %N"
