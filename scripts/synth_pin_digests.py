#!/usr/bin/env python
"""生成 / 核对 tests/data/synth_digests_v2.json：E5 生成器钉住的样本摘要（DEV-0009，D24）。

清单：三个任务 × 三个划分 × 序号 0..INDICES_PER_SPLIT-1，档位按默认轮转（bin = 序号 % 档数）。
- c（occlusion）、e（imagery）：共 120 个，任何平台上都必须逐位相同。
- d（physics）：60 个，只在规范平台（src.synth.constants.CANONICAL_PLATFORM，x86_64 Linux）上比对。
核对规则与失败时的处理见 tests/test_synth_repro.py 的模块说明。

用法（在仓库根目录；服务器用 $PROJ/venv 的 python）：
  python scripts/synth_pin_digests.py check                  # 逐个比对，打印不同的 case，不改文件；有不同时退出码 1
  python scripts/synth_pin_digests.py write                  # 重写整个文件（生成器输出变了、版本加一之后，在开发机上）
  python scripts/synth_pin_digests.py write --only physics   # 只重写 d 的清单（在规范平台上填入 d 的摘要）
  python scripts/synth_pin_digests.py write --only physics --out experiments/P0/NNNN_x/synth_digests_v2.json
      # 服务器 CC 不改 tests/：以仓库里的文件为底、只换 d 的清单，写到运行目录，由管理员复制回 tests/data/

写 d 的清单时记下平台与 pymunk 版本：在规范平台上写，状态为 canonical；在其他平台上写，状态为 pending_canonical（待服务器填入）。
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from src.synth import generate  # noqa: E402
from src.synth.constants import CANONICAL_PLATFORM, SPLITS, SYNTH_GENERATOR_VERSION, TASKS  # noqa: E402

DIGEST_FILE = REPO / "tests" / "data" / "synth_digests_v2.json"
INDICES_PER_SPLIT = 20  # 任务书 DEV-0009 §5：每个划分 20 个序号
ANY_PLATFORM_TASKS = ("occlusion", "imagery")
PHYSICS = "physics"


def case_keys() -> list[str]:
    return [f"{split}/{i}" for split in SPLITS for i in range(INDICES_PER_SPLIT)]


def digests_for(task: str, keys) -> dict[str, str]:
    out = {}
    for key in keys:
        split, index = key.split("/")
        out[key] = generate(task, split, int(index)).digest()
    return out


def on_canonical_platform() -> bool:
    return (platform.machine(), platform.system()) == CANONICAL_PLATFORM


def platform_info(task: str) -> dict:
    info = {"machine": platform.machine(), "system": platform.system(), "python": platform.python_version(), "numpy": np.__version__}
    if task == PHYSICS:
        import pymunk

        # Chipmunk 调用系统 libm（cos/sin 等），libc 版本也是"规范平台"的一部分
        info.update({"pymunk": pymunk.version, "chipmunk": pymunk.chipmunk_version, "libc": list(platform.libc_ver())})
    return info


def task_block(task: str) -> dict:
    block = {"generated_on": platform_info(task), "digests": digests_for(task, case_keys())}
    if task == PHYSICS:
        canonical = on_canonical_platform()
        block.update(
            {
                "compare_on": "canonical_platform",
                "canonical_platform": {"machine": CANONICAL_PLATFORM[0], "system": CANONICAL_PLATFORM[1]},
                "status": "canonical" if canonical else "pending_canonical",
                "status_note": (
                    "规范平台上生成的摘要。"
                    if canonical
                    else "待服务器填入：当前是非规范平台（见 generated_on）上的值，只作占位。在服务器上运行 "
                    "`python scripts/synth_pin_digests.py write --only physics` 替换（D24）。"
                ),
            }
        )
    else:
        block["compare_on"] = "any_platform"
    return block


def load() -> dict:
    return json.loads(DIGEST_FILE.read_text(encoding="utf-8"))


def save(data: dict, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=1, ensure_ascii=False, sort_keys=False) + "\n", encoding="utf-8")


def cmd_write(only: str | None, out: Path) -> None:
    if only is None:
        data = {
            "generator_version": SYNTH_GENERATOR_VERSION,
            "about": "E5 生成器钉住的样本摘要（DEV-0009，D24）。由 scripts/synth_pin_digests.py 生成；核对规则见 tests/test_synth_repro.py。",
            "indices_per_split": INDICES_PER_SPLIT,
            "tasks": {task: task_block(task) for task in TASKS},
        }
    else:
        data = load()
        if data["generator_version"] != SYNTH_GENERATOR_VERSION:
            raise SystemExit(f"文件是生成器版本 {data['generator_version']}，代码是 {SYNTH_GENERATOR_VERSION}：先在开发机上重写整个文件")
        data["tasks"][only] = task_block(only)
    save(data, out)
    print(f"wrote {out}", {t: len(b["digests"]) for t, b in data["tasks"].items()})


def cmd_check() -> int:
    data = load()
    bad_total = 0
    print(f"platform {platform.machine()} {platform.system()}; canonical: {on_canonical_platform()}; "
          f"file version {data['generator_version']}, code version {SYNTH_GENERATOR_VERSION}")
    for task, block in data["tasks"].items():
        got = digests_for(task, block["digests"])
        bad = [k for k in block["digests"] if got[k] != block["digests"][k]]
        applies = task in ANY_PLATFORM_TASKS or on_canonical_platform()
        status = block.get("status", "")
        print(f"{task}: {len(block['digests']) - len(bad)}/{len(block['digests'])} equal {status}"
              + ("" if applies else "（非规范平台，不计）"))
        for k in bad:
            print(f"  DIFF {task}/{k} file={block['digests'][k]} here={got[k]}")
        if applies:
            bad_total += len(bad)
    return 1 if bad_total else 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["check", "write"])
    ap.add_argument("--only", choices=list(TASKS), default=None, help="write 时只重写这个任务的清单（其余照抄仓库里的文件）")
    ap.add_argument("--out", type=Path, default=DIGEST_FILE, help="write 的输出路径，默认 tests/data/synth_digests_v2.json")
    a = ap.parse_args()
    if a.mode == "write":
        cmd_write(a.only, a.out.resolve())
    else:
        raise SystemExit(cmd_check())


if __name__ == "__main__":
    main()
