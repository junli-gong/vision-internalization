#!/bin/bash
# RUN-0044 §2 env fix: compute nodes have no git (tests/test_runlog and runlog.git_info provenance). Install git into OUR env only; nothing global.
export VI=/home/mail/chenyuma/visint
export CONDA_PKGS_DIRS=$VI/conda-pkgs
export TMPDIR=$VI/tmp
C=/home/mail/chenyuma/soft/miniconda3/bin/conda
echo "== $(date -u +%FT%TZ) host=$(hostname -s)"
$VI/env/bin/python -m pip freeze > $VI/probe/0044/hpc_pip_freeze_before_git.txt
$C install -y -p $VI/env --solver classic --freeze-installed git 2>&1 | grep -v libmamba | grep -vE "^\s*$|\|" | tail -30; echo "conda install git rc=${PIPESTATUS[0]}"
$VI/env/bin/git --version; $VI/env/bin/python -V
$VI/env/bin/python -m pip freeze > $VI/probe/0044/hpc_pip_freeze.txt
diff $VI/probe/0044/hpc_pip_freeze_before_git.txt $VI/probe/0044/hpc_pip_freeze.txt && echo "pip freeze unchanged"
echo "== done $(date -u +%FT%TZ)"
