"""DEV-0066：引入的 LLMSeg 官方代码（third_party/llmseg @ f2191e66）。

- 出处：PROVENANCE.md 的表与现算的逐文件 sha256 相同、文件集合相同；LICENSE 是 MIT；commit 是任务书给的那个。
- 官方自己的离线测试（tests/test_offline.py：路由、坐标、指标、一次脚本化的完整管线）在引入之后照常通过（子进程，没有任何 API key）。
- 加载：官方模块确实来自 third_party/llmseg。
- 主配置：我们用的参数与官方默认相同（evals/common.DEFAULT_ITERATIONS = 8；pipeline.py 的命令行：--iterations 默认 8，--topology、
  --refine-box、--no-postprocess 都是 store_true 的开关，默认关）。
"""

from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

from src.llmseg import runner as R
from src.llmseg import vendor as V

REPO = Path(__file__).resolve().parents[1]
KEYS = ("GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY")


def test_provenance_table_matches_files():
    code = V.check_vendor()
    assert code["commit"] == "f2191e6605e40a5be5d932187fc055708ae098cf"
    assert code["license"] == "MIT" and code["modified_files"] == []
    assert V.provenance_table() == V.file_digests() == code["files_sha256"]
    assert set(V.provenance_table()) == set(V.VENDORED_FILES)
    assert (V.VENDOR_DIR / "LICENSE").read_text().startswith("MIT License")
    assert V.UPSTREAM_COMMIT in V.PROVENANCE_PATH.read_text(encoding="utf-8")
    # 只放了运行管线必需的文件（加 PROVENANCE.md），没有 toy 图与评测数据
    files = sorted(str(p.relative_to(V.VENDOR_DIR)) for p in V.VENDOR_DIR.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    assert files == sorted([*V.VENDORED_FILES, "PROVENANCE.md"])


def test_official_offline_tests_pass_unmodified(tmp_path):
    env = {k: v for k, v in os.environ.items() if k not in KEYS}
    env["TMPDIR"] = str(tmp_path)
    p = subprocess.run([sys.executable, "-m", "pytest", str(V.VENDOR_DIR / "tests" / "test_offline.py"), "-q", "-p", "no:cacheprovider"],
                       capture_output=True, text=True, cwd=REPO, env=env, timeout=600)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-2000:]
    assert re.search(r"\b10 passed\b", p.stdout), p.stdout[-1000:]


def test_modules_load_from_vendor_dir():
    mods = V.load()
    for name in V.MODULES:
        assert Path(getattr(mods, name).__file__).resolve() == (V.VENDOR_DIR / f"{name}.py").resolve()
    assert V.load().llm is mods.llm  # 重复调用是同一批模块


def test_main_config_equals_official_defaults():
    mods = V.load()
    spec = importlib.util.spec_from_file_location("llmseg_evals_common_test", V.VENDOR_DIR / "evals" / "common.py")
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)
    assert common.DEFAULT_ITERATIONS == R.MAIN_ITERATIONS == 8
    src = (V.VENDOR_DIR / "pipeline.py").read_text()
    assert re.search(r'add_argument\("--iterations", type=int, default=8,', src)
    for flag in ("--topology", "--refine-box", "--no-postprocess"):
        assert re.search(rf'add_argument\("{flag}", action="store_true"', src), flag
    assert R.MAIN_SWITCHES == {"topology": False, "refine_box": False, "postprocess": True}
    ident = R.PipelineConfig().identity(mods.pipeline)
    assert ident["main_config"] is True and ident["max_instances"] == mods.pipeline.MAX_INSTANCES == 200
    assert ident["calls_bound"] == 18
    assert R.PipelineConfig(iterations=0).identity(mods.pipeline)["main_config"] is False
