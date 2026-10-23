"""W2、W2-c 测试用的"未冻结草稿"夹具（DEV-0062；同 W1 的 D112）。

configs/w2.json 在 W2-a 调参之后由 scripts/mgr/w2_tune.py freeze 冻结成选中的臂（布局 grid / separate、挑选时思考开 / 关、K = 4 / 6 / 8、
提出 box / box_points 都可能），同时把 configs/w2c_train.json 的 w2_config.sha256 填上。测"默认、未冻结"行为、断言草稿取值（思考开、box、
grid、K = 8、frozen = false、w2_config.sha256 = null）的测试读这里的两份夹具，不读仓库配置：

- W2_DRAFT_PATH：tests/fixtures/w2/w2_draft.json，冻结前的 configs/w2.json 原样（commit 4576fc76，DEV-0059 的初稿加 DEV-0058 的默认值）；
- W2C_TRAIN_DRAFT_PATH：tests/fixtures/w2/w2c_train_draft.json，同一时刻的 configs/w2c_train.json 原样（w2_config.sha256 是占位 null）。

两份的 sha256 钉在下面（tests/test_w2_config.py 核对）。仓库配置只测与冻结无关的不变量（tests/test_w2_config.py 末尾）：未冻结时与这两份
逐字节相同；冻结后满足冻结的不变量。草稿的取值只在这里与夹具里，别处 import。
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from src.w2.config import load_w2_config

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "w2"
W2_DRAFT_PATH = FIXTURES / "w2_draft.json"
W2_DRAFT_SHA256 = "a4c013c344c83f1885c82db345be11f252e706d1a833316396302aeb4abbc6fd"
W2C_TRAIN_DRAFT_PATH = FIXTURES / "w2c_train_draft.json"
W2C_TRAIN_DRAFT_SHA256 = "ac450a95dfc34fd0cc7aac787a7d35b9b8ec7ffc477034cd81b27c0a0dbe8d9a"

W2_DRAFT = load_w2_config(W2_DRAFT_PATH)
_DOC = json.loads(W2_DRAFT_PATH.read_text(encoding="utf-8"))


def w2_draft_doc() -> dict[str, Any]:
    """草稿 W2 配置的一份深拷贝（测试在上面改几项再写到临时文件）。"""
    return copy.deepcopy(_DOC)
