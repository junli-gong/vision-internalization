#!/bin/bash
# RUN-0044 push helper on hpcmgt: flock, pull --rebase, add given paths, commit, push. usage: push.sh "<message>" <path>...
VI=/home/mail/chenyuma/visint; MSG=$1; shift
cd $VI/repo || exit 1
exec 9>$VI/logs/git.lock; flock 9
git pull -q --rebase origin main || { echo "pull failed"; exit 2; }
git add -- "$@" && git commit -q -m "$MSG" && for a in 1 2 3; do git push -q origin HEAD:main && { echo "pushed $(git rev-parse --short HEAD)"; exit 0; }; sleep 20; git pull -q --rebase origin main; done
echo "push failed"; exit 3
