#!/usr/bin/env python
"""RUN-0025 §3（完整开发子集，S3）：在调用 scripts/run_e1.py 的 main 之前打开 torch.use_deterministic_algorithms(True, warn_only=True)，
其余与 run_e1.py 完全相同（同一个入口、同样的参数）；记录全程的 warnings（去重）到 --warn-out。只调用现有接口，不改 run_e1。
usage (在仓库根下): python run_e1_det.py --warn-out <json> -- <run_e1.py 的参数>
"""
import json, sys, warnings
from pathlib import Path
REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
i = sys.argv.index("--"); warn_out = sys.argv[sys.argv.index("--warn-out") + 1]; rest = sys.argv[i + 1:]
warnings.simplefilter("always"); caught = []
_o = warnings.showwarning
def _show(message, category, filename, lineno, file=None, line=None):
    caught.append((category.__name__, str(message)[:500], filename.split("site-packages/")[-1], lineno)); _o(message, category, filename, lineno, file, line)
warnings.showwarning = _show
import torch  # noqa: E402
torch.use_deterministic_algorithms(True, warn_only=True)
import run_e1  # noqa: E402
rc = 1
try:
    rc = run_e1.main(rest)
finally:
    Path(warn_out).write_text(json.dumps({"deterministic_algorithms": torch.are_deterministic_algorithms_enabled(), "rc": rc,
                                          "warnings_unique": sorted(set(caught))}, indent=1, ensure_ascii=False) + "\n")
sys.exit(rc)
