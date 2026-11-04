"""DEV-0083：Rex-Omni 外部参照的隔离（硬约束第 1 条；D165 第四节、D177-c；同 tests/test_lisa_isolation.py）。

Rex-Omni 是学习型检测器，只作单独运行的对照系统；它的输出绝不进入我们的训练数据、标签、奖励或任何挑选。静态核对：
- src/ 下除 src/rexomni/ 以外的全部代码，以及生成训练数据、标签与训练的脚本（scripts/train_*.py、scripts/build_*.py、scripts/data/、
  scripts/xm/），都不 import src/rexomni、不引用 scripts/run_rexomni.py、权重清单与仓库名，也不引用 Rex-Omni 的结果目录（运行目录
  NNNN_…rexomni…：src/rexomni/runner.check_run_dir 强制末级目录名含 "_rexomni"；叶目录 rex-omni/xm/rexomni_det/ 与
  rex-omni/xm/rexomni_det_joint/）。
  模式不认 X-M 自己的指标名 f1_rexomni、rexomni_exact（那是"与 Rex-Omni 一致的 F1 口径"，不是 Rex-Omni 的输出）。
- 正对照：模式在 Rex-Omni 一侧的文件里确实能找到；结果路径、import 写法的样例都能认出。
- 只有 Rex-Omni 一侧 import 本包。
- Rex-Omni 一侧只 import 它需要的东西：标准库、numpy、PIL、torch、transformers、tokenizers、huggingface_hub、yaml，以及 src.rexomni、
  src.xm.data（读对）、src.xm.metrics（计分）、src.xm.gate（gate_xm.json 的基座那条路）、src.runlog（meta.json）、src.adapters.registry
  （只在 --tiny 时取分词器的来源）。不直接 import 训练代码、X-M 的提示（src.xm.prompts）与后处理（src.xm.post）、X-M 的运行器。
  注意：src.xm.gate 自己在模块顶层 import 了 src.xm.prompts（为了 FROZEN）；Rex-Omni 一侧不调用它（test_gate_transitive_import_is_known）。
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

from src.rexomni import runner as RR

REPO = Path(__file__).resolve().parents[1]
FORBIDDEN = re.compile(r"src[./]rexomni\b|from\s+src\s+import\s+[^\n]*\brexomni\b|\brun_rexomni\b|\brex-omni\b|\brexomni_det(?:_joint)?\b"
                       r"|[0-9]{4}_[A-Za-z0-9_.-]*rexomni|IDEA-Research/Rex-Omni|\brexomni_weights\b")
REX_SRC = REPO / "src" / "rexomni"
REX_FILES = ("scripts/run_rexomni.py",)
ALLOWED_THIRD_PARTY = {"numpy", "PIL", "torch", "transformers", "tokenizers", "huggingface_hub", "yaml"}
ALLOWED_SRC = {"src.rexomni", "src.rexomni.protocol", "src.rexomni.model", "src.rexomni.runner", "src.xm.data", "src.xm.metrics",
               "src.xm.gate", "src.runlog", "src.adapters.registry"}


def scanned_files() -> list[Path]:
    out = [p for p in sorted((REPO / "src").rglob("*.py")) if p.relative_to(REPO / "src").parts[0] != "rexomni"]
    for pat in ("train_*.py", "build_*.py"):
        out += sorted((REPO / "scripts").glob(pat))
    for sub in ("data", "xm"):
        out += sorted((REPO / "scripts" / sub).rglob("*.py"))
    return out


def test_training_data_label_reward_code_never_touches_rexomni():
    files = scanned_files()
    rels = {p.relative_to(REPO).as_posix() for p in files}
    for must in ("src/w2/w2c_data.py", "src/w2/w2c_replay.py", "src/w2/w2c_train.py", "src/data/train_loaders.py", "src/explore/x2_data.py",
                 "src/xm/train_data.py", "src/xm/targets.py", "src/xm/exclusions.py", "src/xm/data.py", "src/xm/runner.py",
                 "scripts/build_w2c_data.py", "scripts/build_x2_data.py", "scripts/build_xm_data.py", "scripts/train_w2c.py",
                 "scripts/train_e3b.py", "scripts/xm/build_targets.py", "scripts/xm/build_configs.py"):
        assert must in rels, must
    assert not any(r.startswith("src/rexomni/") for r in rels)
    offenders = {p.relative_to(REPO).as_posix(): FORBIDDEN.findall(p.read_text(encoding="utf-8")) for p in files}
    offenders = {k: v for k, v in offenders.items() if v}
    assert not offenders, f"这些文件引用了 Rex-Omni 或它的结果（硬约束第 1 条）：{offenders}"


def test_pattern_finds_the_rexomni_side_itself():
    for f in (*REX_FILES, "src/rexomni/runner.py", "src/rexomni/model.py", "src/rexomni/protocol.py", "configs/rexomni_weights.json"):
        assert FORBIDDEN.search((REPO / f).read_text(encoding="utf-8")), f
    for s in ("experiments/XM/0110_rexomni_hpcgpu07/0110_rexomni_final/rex-omni/xm/rexomni_det/xm_final/predictions.jsonl",
              "experiments/XM/0110_xm_rexomni/x.json", "from src.rexomni import runner", "import src.rexomni.model as RM",
              "from src import rexomni", "IDEA-Research/Rex-Omni", "python scripts/run_rexomni.py run", "configs/rexomni_weights.json",
              "xm/rexomni_det_joint/xm_final/predictions.jsonl", "setting = 'rexomni_det_joint'"):
        assert FORBIDDEN.search(s), s
    for s in ("f1_rexomni", "XMM.rexomni_exact(rows)", "f1_rexomni_exact_miou", "Rex-Omni 的 evaluation/metrics/coco_lvis_metric.py"):
        assert not FORBIDDEN.search(s), s  # X-M 自己的指标名与文献引用不算
    # 运行目录必须带 "_rexomni"，所以任何写死的结果路径都会被上面的模式认出
    RR.check_run_dir("x/0110_rexomni_tune")
    for bad in ("x/0110_tune", "x/rexomni_tune"):
        try:
            RR.check_run_dir(bad)
        except RR.RexOmniRunError:
            continue
        raise AssertionError(bad)
    assert (RR.SYSTEM, RR.TASK, tuple(RR.SETTINGS)) == ("rex-omni", "xm", ("rexomni_det_joint", "rexomni_det"))
    assert all(FORBIDDEN.search(f"x/{s}/y") for s in RR.SETTINGS)


def test_only_rexomni_files_import_the_package():
    importers = []
    for base in ("src", "scripts"):
        for p in sorted((REPO / base).rglob("*.py")):
            rel = p.relative_to(REPO).as_posix()
            if rel.startswith("src/rexomni/") or rel in REX_FILES:
                continue
            if re.search(r"^\s*(from\s+src\.rexomni\b|from\s+src\s+import\s+[^\n]*\brexomni\b|import\s+src\.rexomni\b)",
                         p.read_text(encoding="utf-8"), re.M):
                importers.append(rel)
    assert importers == [], importers


def _imports(path: Path) -> set[str]:
    """文件里所有的 import（顶层与函数里的），绝对模块名；from src.xm import data → src.xm.data。"""
    out: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            out |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                raise AssertionError(f"{path}：不用相对 import")
            mod = node.module or ""
            if mod in ("src", "src.xm", "src.rexomni", "src.adapters"):
                out |= {f"{mod}.{a.name}" for a in node.names}
            else:
                out.add(mod)
    return out


def test_rexomni_side_imports_only_what_it_needs():
    files = sorted(REX_SRC.glob("*.py")) + [REPO / f for f in REX_FILES]
    bad = {}
    for p in files:
        for mod in _imports(p):
            top = mod.split(".")[0]
            if top == "__future__" or top in sys.stdlib_module_names or top in ALLOWED_THIRD_PARTY:
                continue
            if mod in ALLOWED_SRC:
                continue
            bad.setdefault(p.relative_to(REPO).as_posix(), []).append(mod)
    assert not bad, f"Rex-Omni 一侧 import 了不该用的模块：{bad}"
    # 训练代码、X-M 的提示 / 后处理 / 运行器一个都不直接 import
    every = set().union(*(_imports(p) for p in files))
    for m in ("src.xm.prompts", "src.xm.post", "src.xm.runner", "src.xm.targets", "src.xm.train_data", "src.w2", "src.e3", "src.explore",
              "src.adapters.qwen35", "src.adapters.training"):
        assert not any(x == m or x.startswith(m + ".") for x in every), m
    # src.adapters.registry 只在 tiny 的分词器里用（函数内的 import）
    tree = ast.parse((REX_SRC / "model.py").read_text(encoding="utf-8"))
    top_level = {f"{n.module}.{a.name}" for n in tree.body if isinstance(n, ast.ImportFrom) for a in n.names}
    assert "src.adapters.registry" not in top_level


def test_gate_transitive_import_is_known():
    """src.xm.gate（任务书指定"走 src/xm/gate 的基座那条路"）在顶层 import src.xm.prompts，后者再 import src.xm.targets 与
    src.explore.x2_prompts。Rex-Omni 一侧只调 gate.check_xm_gate / append_log / GATED / UNLOCK_LOG，不调 require_frozen_prompts，
    也不碰任何提示。已上报待裁（DEV-0083 完成总结）；这里钉住现状，src.xm.gate 不再依赖 prompts 时这条测试提醒改掉说明。"""
    gate_imports = _imports(REPO / "src" / "xm" / "gate.py")
    assert "src.xm.prompts" in gate_imports
    used = set()
    for p in [*REX_SRC.glob("*.py"), *(REPO / f for f in REX_FILES)]:
        used |= set(re.findall(r"\bXG\.([A-Za-z_]+)", p.read_text(encoding="utf-8")))
    assert used <= {"check_xm_gate", "append_log", "GATED", "UNLOCK_LOG", "XMGateError"}, used
