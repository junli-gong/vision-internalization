"""拒绝作答的诊断规则：唯一来源（DEV-0019；原先 E1 的 DEV-0015 与 E5 的 DEV-0018 各写一份，逐字相同，合并到这里）。

规则（只作诊断，不改失败原因）：记录解析失败、回答（text）里一个阿拉伯数字都没有（DIGIT_RE 搜不到）、且 REFUSAL_RE 能匹配
（不分大小写，整词）。E1（src/e1/scoring.py 的 failure_detail）与 E5（src/e5/scoring.py 的 is_refusal）都 import 这两个正则。
改规则要同时把 E1 的 DIAG_RULE_VERSION 加一，并改 tests/test_e1_scoring.py 里钉住的正则原文。
"""

from __future__ import annotations

import re

REFUSAL_RE = re.compile(r"\b(?:I['’]m sorry|I am sorry|I apologi[sz]e|unable to|not able to|can['’]?t|cannot|can not)\b",
                        re.IGNORECASE)
DIGIT_RE = re.compile(r"[0-9]")

__all__ = ["DIGIT_RE", "REFUSAL_RE"]
