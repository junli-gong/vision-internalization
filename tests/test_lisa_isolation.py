"""DEV-0075：LISA-7B 外部对照的隔离（硬约束第 1 条；D154；简报 §8.8）。

LISA 把 SAM 接在 LLM 旁边（外接），只作对照单独运行；它的输出绝不进入我们的训练数据、标签、奖励或任何挑选。静态核对：
- src/ 下除 src/lisa/ 以外的全部代码，以及生成训练数据、训练与建数据的脚本（scripts/train_*.py、scripts/build_*.py、scripts/data/），
  都不 import src/lisa、third_party/lisa、LISA 的官方模块（model.LISA），不引用 LISA 的脚本（run_lisa、lisa_w2），也不引用 LISA 的
  结果目录（设置名 lisa_7b，运行目录 NNNN_lisa…——src/lisa/common.RUN_DIR_NAME 强制运行目录名含 "_lisa"）。
- 正对照：同一个模式在 src/lisa/ 与两个 LISA 脚本里确实能找到（模式没写错）；运行目录规则确实要求 "_lisa"。
- LISA 那一侧（scripts/run_lisa.py）只 import 标准库、numpy 与 src.lisa.common（tests/test_lisa_vendor.py），读不到我们的训练代码。
"""

from __future__ import annotations

import re
from pathlib import Path

from src.lisa import common as C

REPO = Path(__file__).resolve().parents[1]
FORBIDDEN = re.compile(r"src[./]lisa|from\s+src\s+import\s+[^\n]*\blisa\b|third_party[./]lisa|\blisa_7b\b|[0-9]{4}_lisa|model\.LISA"
                       r"|\brun_lisa\b|\blisa_w2\b")
LISA_FILES = ("scripts/run_lisa.py", "scripts/lisa_w2.py")


def scanned_files() -> list[Path]:
    out = [p for p in sorted((REPO / "src").rglob("*.py")) if "lisa" not in p.relative_to(REPO / "src").parts[:1]]
    for pat in ("train_*.py", "build_*.py"):
        out += sorted((REPO / "scripts").glob(pat))
    out += sorted((REPO / "scripts" / "data").rglob("*.py"))
    return out


def test_training_data_label_reward_code_never_touches_lisa():
    files = scanned_files()
    rels = {p.relative_to(REPO).as_posix() for p in files}
    # 扫描范围确实覆盖生成训练数据、标签、奖励与训练的代码
    for must in ("src/w2/w2c_data.py", "src/w2/w2c_replay.py", "src/w2/w2c_train.py", "src/data/train_loaders.py", "src/explore/x2_data.py",
                 "scripts/build_w2c_data.py", "scripts/build_x2_data.py", "scripts/train_w2c.py", "scripts/train_e3b.py"):
        assert must in rels, must
    assert not any(r.startswith("src/lisa/") for r in rels)
    offenders = {p.relative_to(REPO).as_posix(): FORBIDDEN.findall(p.read_text(encoding="utf-8")) for p in files}
    offenders = {k: v for k, v in offenders.items() if v}
    assert not offenders, f"这些文件引用了 LISA 或它的结果（硬约束第 1 条）：{offenders}"


def test_pattern_finds_the_lisa_side_itself():
    for f in (*LISA_FILES, "src/lisa/scoring.py", "src/lisa/data.py"):
        assert FORBIDDEN.search((REPO / f).read_text(encoding="utf-8")), f
    assert FORBIDDEN.search("experiments/W2/0099_lisa_hpcgpu06/lisa-7b-v1/refseg/lisa_7b/test_refcoco_unc_testA")
    assert FORBIDDEN.search("from src.lisa import scoring") and FORBIDDEN.search("from src import lisa")
    assert FORBIDDEN.search("sys.path.insert(0, 'third_party/lisa')") and FORBIDDEN.search("import model.LISA")
    # 运行目录必须带 "_lisa"，所以任何写死的结果路径都会被上面的模式认出
    assert C.RUN_DIR_NAME.fullmatch("0099_lisa_hpcgpu06") and not C.RUN_DIR_NAME.fullmatch("0099_hpcgpu06")
    assert C.SETTING == "lisa_7b"


def test_only_lisa_files_import_the_lisa_package():
    importers = []
    for base in ("src", "scripts"):
        for p in sorted((REPO / base).rglob("*.py")):
            rel = p.relative_to(REPO).as_posix()
            if rel.startswith("src/lisa/") or rel in LISA_FILES:
                continue
            if re.search(r"^\s*(from\s+src\.lisa\b|from\s+src\s+import\s+[^\n]*\blisa\b|import\s+src\.lisa\b)", p.read_text(encoding="utf-8"), re.M):
                importers.append(rel)
    assert importers == [], importers
