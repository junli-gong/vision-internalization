#!/usr/bin/env python
"""管理员用：由 configs/w1.json 生成 W1 调参的四个臂 configs/w1_tune/T1.json … T4.json（W1 预注册 §10b，写于任何调参运行之前）。

逐级累加，与 configs/w1.json 只差 §10b 表里的项；按任务的提示、上限、批参数都不动。
  T1：最后一轮只给提交工具；调用满额之后也只给提交工具；最后一轮的提示改写。
  T2：T1 + 每轮结束报剩余预算（prompts.messages.status）。
  T3：T2 + 系统提示加两句（一轮可以发几个调用；坐标尽量精确）。
  T4：T3 + 视图坐标（workbench.view_frames）+ 系统提示的坐标段改写成带视图编号的说法。
第 2 轮（§10b 第 4 条：第 1 轮没有臂的提交率 ≥ 90%；DEV-0055、D110）：以第 1 轮的某个臂 B 为底，只改提示怎样送到模型：
  U1：B，状态提示与"最后一轮"提示的正文前加一个空行（仍追加在 tool 消息里；D110-2：原来与工具输出之间没有任何分隔）；
  U2：U1，"最后一轮"提示改为单独的 user 消息（last_round_role = user，只在第 6 轮丢掉前几轮的思考段）；
  U3：U2，状态提示也改为单独的 user 消息（status_role = user，从第 2 轮起看不到之前的思考段）；B 没有状态提示时不生成 U3。
第 1 轮的 T1–T4 保留 RUN-0061 用过的原文件（D110-4），不再重新生成。
用法：python scripts/mgr/make_w1_tune_configs.py [--check]   （第 1 轮；--check：只核对现有文件与生成结果逐字节相同；D110 之后会报"不同"）
      python scripts/mgr/make_w1_tune_configs.py --round2 T<k> [--check]
      python scripts/mgr/make_w1_tune_configs.py --freeze <臂> --decision D<nnn>
        冻结：按 §10b 选中的臂，从 configs/w1_tune/<臂>.json 写出新的 configs/w1.json：内容与该臂相同（当前 configs/w1.json 有、
        臂文件里没有的键按当前文件补上，例如 D110 的 status_role、last_round_role），只改 name、status 与 frozen = true。
        之后再生成或冻结会报错。
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


SEP = "\n\n"


def round2(base_cfg: dict, base_name: str, current: dict) -> dict[str, dict]:
    """第 2 轮的臂（见模块说明）。base_cfg = configs/w1_tune/<B>.json；current = 当前 configs/w1.json（取 D110 新加的键名与默认值）。"""
    for key in ("status_role", "last_round_role"):
        if key not in current["prompts"]:
            raise SystemExit(f"configs/w1.json 的 prompts 没有 {key}（DEV-0055 合入了吗？）")
    msgs = base_cfg["prompts"]["messages"]
    status, last = msgs.get("status"), msgs.get("last_round")
    if not last:
        raise SystemExit(f"{base_name} 没有 last_round 提示")
    out: dict[str, dict] = {}
    c = copy.deepcopy(base_cfg)
    c["prompts"]["status_role"] = "tool"
    c["prompts"]["last_round_role"] = "tool"
    c["prompts"]["messages"]["last_round"] = SEP + last
    if status:
        c["prompts"]["messages"]["status"] = SEP + status
    out["U1"] = copy.deepcopy(c)
    c["prompts"]["last_round_role"] = "user"
    c["prompts"]["messages"]["last_round"] = last
    out["U2"] = copy.deepcopy(c)
    if status:
        c["prompts"]["status_role"] = "user"
        c["prompts"]["messages"]["status"] = status
        out["U3"] = copy.deepcopy(c)
    base_sha = hashlib.sha256((OUT_DIR / f"{base_name}.json").read_bytes()).hexdigest()
    for k, v in out.items():
        v["name"] = f"W1 调参臂 {k}（第 2 轮，底为 {base_name}；PREREGISTRATION.md W1 §10b；scripts/mgr/make_w1_tune_configs.py --round2 {base_name}）"
        v["status"] = f"调参用，只能跑开发-调参集。生成自 configs/w1_tune/{base_name}.json（sha256 {base_sha}）。"
        v["frozen"] = False
    return out


def _fill_missing(dst: dict, src: dict) -> None:
    """src 有、dst 没有的键（递归）补进 dst。"""
    for k, v in src.items():
        if k not in dst:
            dst[k] = copy.deepcopy(v)
        elif isinstance(v, dict) and isinstance(dst[k], dict):
            _fill_missing(dst[k], v)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", action="store_true")
    p.add_argument("--round2", choices=["T1", "T2", "T3", "T4"], help="第 2 轮：以这个臂为底生成 U1–U3")
    p.add_argument("--freeze", help="冻结：configs/w1_tune/ 里的臂名（T1–T4、U1–U3）")
    p.add_argument("--decision", help="冻结的 D 编号（--freeze 时必填）")
    a = p.parse_args()
    base = json.loads(BASE.read_text(encoding="utf-8"))
    if base.get("frozen") is True:
        raise SystemExit("configs/w1.json 已冻结，不再生成或冻结调参的臂")
    if a.freeze:
        if not a.decision:
            raise SystemExit("--freeze 要给 --decision D<nnn>")
        path = OUT_DIR / f"{a.freeze}.json"
        arm = json.loads(path.read_text(encoding="utf-8"))
        _fill_missing(arm, base)
        arm_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        arm["name"] = base["name"]
        arm["status"] = (f"已冻结（{a.decision}）：W1 预注册 §10b 选中臂 {a.freeze}（configs/w1_tune/{a.freeze}.json，sha256 {arm_sha}），"
                         "内容与该臂相同，只改 name、status、frozen。之后不再改动；开发-检查集与 test 只用这一份。")
        arm["frozen"] = True
        text = json.dumps(arm, ensure_ascii=False, indent=1) + "\n"
        BASE.write_text(text, encoding="utf-8")
        print(f"冻结：configs/w1.json ← {a.freeze}，sha256 {hashlib.sha256(text.encode()).hexdigest()}")
        return 0
    if a.round2:
        gen = round2(json.loads((OUT_DIR / f"{a.round2}.json").read_text(encoding="utf-8")), a.round2, base)
    else:
        gen = arms(base)
    for key in ("final_round_tools", "capped_tools"):
        if key not in base["protocol"]:
            raise SystemExit(f"configs/w1.json 的 protocol 没有 {key}（DEV-0053 合入了吗？）")
    if "workbench" not in base:
        raise SystemExit("configs/w1.json 没有 workbench 一节（DEV-0053 合入了吗？）")
    OUT_DIR.mkdir(exist_ok=True)
    bad = 0
    for k, cfg in gen.items():
        text = json.dumps(cfg, ensure_ascii=False, indent=1) + "\n"
        path = OUT_DIR / f"{k}.json"
        if a.check:
            same = path.is_file() and path.read_text(encoding="utf-8") == text
            bad += not same
            print(f"{k}: {'相同' if same else '不同'}")
        else:
            if not a.round2:
                raise SystemExit("第 1 轮的 T1–T4 保留 RUN-0061 用过的原文件（D110-4），不再重新生成；只能 --check")
            path.write_text(text, encoding="utf-8")
            print(f"{k}: {path.relative_to(ROOT)} sha256 {hashlib.sha256(text.encode()).hexdigest()}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
