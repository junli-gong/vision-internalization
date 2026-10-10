#!/bin/bash
VI=/home/mail/chenyuma/visint; cd $VI/repo || exit 1
echo "== $(date -u +%FT%TZ)"; squeue -h -j 157296,157297 -o "%i %T %M %N"
git bundle verify $VI/tmp/repo.bundle 2>&1 | tail -2
git bundle list-heads $VI/tmp/repo.bundle
t0=$(date +%s); git fetch -q $VI/tmp/repo.bundle "refs/remotes/origin/main:refs/remotes/origin/main"; echo "fetch-bundle rc=$? $(( $(date +%s)-t0 ))s"
git config --local -l | egrep "sshcommand|user\.|remote.origin.url"
for a in 1 2 3 4 5; do t0=$(date +%s); timeout 1200 git fetch -q origin; rc=$?; echo "fetch-origin attempt $a rc=$rc $(( $(date +%s)-t0 ))s"; [ $rc -eq 0 ] && break; sleep 30; done
git checkout -q -B main origin/main; echo "checkout rc=$?"; git log --oneline -3; git status -sb | head -1
echo "== done $(date -u +%FT%TZ)"
