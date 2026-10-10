#!/bin/bash
VI=/home/mail/chenyuma/visint
SSHC="ssh -i $VI/.ssh/id_ed25519_github -o IdentitiesOnly=yes -o UserKnownHostsFile=$VI/.ssh/known_hosts -o StrictHostKeyChecking=yes -o ServerAliveInterval=30"
for a in 1 2 3; do
  t0=$(date +%s); rm -rf $VI/repo
  GIT_SSH_COMMAND="$SSHC" git clone -q git@github.com:junli-gong/vision-internalization.git $VI/repo; rc=$?
  echo "$(date -u +%FT%TZ) clone attempt $a rc=$rc $(( $(date +%s)-t0 ))s"; [ $rc -eq 0 ] && break
done
cd $VI/repo && git config core.sshCommand "$SSHC" && git config user.name hpc-a100-server-cc && git config user.email hpc-a100-server-cc@users.noreply.github.com && git config --local -l | egrep "sshcommand|user\.|remote.origin.url" && git log --oneline -1
echo "$(date -u +%FT%TZ) clone done"
