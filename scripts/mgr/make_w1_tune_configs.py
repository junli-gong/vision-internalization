#!/usr/bin/env python
"""管理员用：由 configs/w1.json 生成 W1 调参的四个臂 configs/w1_tune/T1.json … T4.json（W1 预注册 §10b，写于任何调参运行之前）。

逐级累加，与 configs/w1.json 只差 §10b 表里的项；按任务的提示、上限、批参数都不动。
  T1：最后一轮只给提交工具；调用满额之后也只给提交工具；最后一轮的提示改写。
  T2：T1 + 每轮结束报剩余预算（prompts.messages.status）。
  T3：T2 + 系统提示加两句（一轮可以发几个调用；坐标尽量精确）。
  T4：T3 + 视图坐标（workbench.view_frames）+ 系统提示的坐标段改写成带视图编号的说法。
用法：python scripts/mgr/make_w1_tune_configs.py [--check]   （--check：只核对现有文件与生成结果逐字节相同）
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "configs" / "w1.json"
OUT_DIR = ROOT / "configs" / "w1_tune"

LAST_ROUND = "This is your last turn. Only the submit tools are available now: submit your answer."
FINAL_ROUND_REFUSED = "Only the submit tools can be used in the last turn; this call was not executed."
STATUS = "[Turn {round} of {max_rounds} done; tool calls used: {calls_used} of {max_tool_calls}; turns left: {rounds_left}.]"
STATUS_FIELDS_USED = ["round", "max_rounds", "calls_used", "max_tool_calls", "rounds_left"]
ANCHOR_TURN = "their results (text and images) come back in the next turn."
ADD_MULTI = " You can make several tool calls in one turn; they run in order and all their results come back together in the next turn."
ANCHOR_COORD = "and the tool reports which region a view covers."
ADD_PRECISE = " Coordinates can take any value from 0 to 1000; give them as precisely as you can."
# T4：坐标段整段改写（原文"every tool uses coordinates relative to the full original image … Zoomed views use the same
# original-image coordinates"与 view 参数矛盾，DEV-0054 的完成总结第 5 条）；T3 加的"尽量精确"一句保留在段末。
COORD_OLD_START = "Coordinates: every tool uses coordinates relative to the full original image"
COORD_VIEWS = ("Coordinates: tools use coordinates from 0 to 1000, with x first (x = 0 is the left edge and x = 1000 the right edge;"
               " y = 0 is the top and y = 1000 the bottom), relative to the full original image unless you give a view number."
               " Every image you are shown has a view number (the original image is view 0); tools that take coordinates accept an"
               " optional view number, and the coordinates are then 0 to 1000 relative to that view." + ADD_PRECISE)


def _insert_after(text: str, anchor: str, add: str) -> str:
    if text.count(anchor) != 1:
        raise SystemExit(f"系统提示里找不到唯一的锚点：{anchor!r}")
    return text.replace(anchor, anchor + add)


def arms(base: dict) -> dict[str, dict]:
    out: dict[str, dict] = {}
    c = copy.deepcopy(base)
    c["protocol"]["final_round_tools"] = "submit_only"
    c["protocol"]["capped_tools"] = "submit_only"
    c["prompts"]["messages"]["last_round"] = LAST_ROUND
    c["prompts"]["messages"]["final_round_refused"] = FINAL_ROUND_REFUSED
    out["T1"] = copy.deepcopy(c)
    c["prompts"]["messages"]["status"] = STATUS
    c["prompts"]["placeholders"]["status"] = STATUS_FIELDS_USED  # DEV-0053：模板用了哪些占位符就如数声明（D108-3）
    out["T2"] = copy.deepcopy(c)
    s = _insert_after(c["prompts"]["system"], ANCHOR_TURN, ADD_MULTI)
    c["prompts"]["system"] = _insert_after(s, ANCHOR_COORD, ADD_PRECISE)
    out["T3"] = copy.deepcopy(c)
    paras = c["prompts"]["system"].split("\n\n")
    hit = [i for i, para in enumerate(paras) if para.startswith(COORD_OLD_START)]
    if len(hit) != 1 or not paras[hit[0]].endswith(ANCHOR_COORD + ADD_PRECISE):
        raise SystemExit("系统提示里找不到唯一的坐标段")
    paras[hit[0]] = COORD_VIEWS
    c["prompts"]["system"] = "\n\n".join(paras)
    c["workbench"]["view_frames"] = True
    out["T4"] = copy.deepcopy(c)
    base_sha = hashlib.sha256(BASE.read_bytes()).hexdigest()
    for k, v in out.items():
        v["name"] = f"W1 调参臂 {k}（PREREGISTRATION.md W1 §10b；由 scripts/mgr/make_w1_tune_configs.py 从 configs/w1.json 生成）"
        v["status"] = f"调参用，只能跑开发-调参集。生成自 configs/w1.json（sha256 {base_sha}）。"
        v["frozen"] = False
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", action="store_true")
    a = p.parse_args()
    base = json.loads(BASE.read_text(encoding="utf-8"))
    for key in ("final_round_tools", "capped_tools"):
        if key not in base["protocol"]:
            raise SystemExit(f"configs/w1.json 的 protocol 没有 {key}（DEV-0053 合入了吗？）")
    if "workbench" not in base:
        raise SystemExit("configs/w1.json 没有 workbench 一节（DEV-0053 合入了吗？）")
    OUT_DIR.mkdir(exist_ok=True)
    bad = 0
    for k, cfg in arms(base).items():
        text = json.dumps(cfg, ensure_ascii=False, indent=1) + "\n"
        path = OUT_DIR / f"{k}.json"
        if a.check:
            same = path.is_file() and path.read_text(encoding="utf-8") == text
            bad += not same
            print(f"{k}: {'相同' if same else '不同'}")
        else:
            path.write_text(text, encoding="utf-8")
            print(f"{k}: {path.relative_to(ROOT)} sha256 {hashlib.sha256(text.encode()).hexdigest()}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
