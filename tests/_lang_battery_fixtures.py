"""DEV-0064：scripts/lang_battery.py 的 --tiny 彩排用的小型原始数据目录（布局同真数据的 --raw-dir）。

- MMLU-Pro：tests/fixtures/health/mmlu_pro_validation.parquet（MMLU-Pro 的 validation 划分原样，70 题，14 个学科各 5 题）放在
  test 文件的位置上，当总体用（--tiny 不核对固定的 sha256 与行数）；
- IFEval：健康检查的 50 条（configs/health_check_items.json，还原成 HF 文件的行）加 2 条不在健康检查里的自造提示。

    python -m tests._lang_battery_fixtures <目录>
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "lang_battery.py"
MMLU_PRO_VALIDATION = REPO / "tests" / "fixtures" / "health" / "mmlu_pro_validation.parquet"
# 自造的 IFEval 行（key 不与 IFEval 的任何 key 相同；指令取官方 registry 里有的）
EXTRA_IFEVAL = (
    {"key": 990001, "prompt": "Write two sentences about the sea. Do not use any commas in your response.",
     "instruction_id_list": ["punctuation:no_comma"], "kwargs": [{}]},
    {"key": 990002, "prompt": "Describe a cat in all lowercase letters.",
     "instruction_id_list": ["change_case:english_lowercase"], "kwargs": [{}]},
)


def load_script():
    spec = importlib.util.spec_from_file_location("_lang_battery_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def health_ifeval_rows() -> list[dict]:
    items = json.loads((REPO / "configs" / "health_check_items.json").read_text(encoding="utf-8"))
    return [{"key": it["id"], "prompt": it["prompt"], "instruction_id_list": it["instruction_id_list"], "kwargs": it["kwargs"]}
            for it in items["items"]["ifeval"]]


def write_tiny_raw_dir(dest: str | Path) -> Path:
    """写出小型原始数据目录，返回它。"""
    lb = load_script()
    dest = Path(dest)
    mmlu = lb.raw_path(dest, "mmlu_pro")
    mmlu.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(MMLU_PRO_VALIDATION, mmlu)
    ife = lb.raw_path(dest, "ifeval")
    ife.parent.mkdir(parents=True, exist_ok=True)
    rows = health_ifeval_rows() + [dict(r) for r in EXTRA_IFEVAL]
    ife.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return dest


if __name__ == "__main__":
    print(write_tiny_raw_dir(sys.argv[1]))
