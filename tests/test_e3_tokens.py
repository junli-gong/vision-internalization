"""DEV-0025 §1：E3b 的网格词表（src/e3/tokens.py）与配置的对应。

- 词表 2 + 64 + 1 = 67 个，名字过 check_grid_token_names，个数与常量逐数相同；计划追加（ADE 150 类）一起累计不超过 300；
- 加进三个家族的小模型：id 互不相同、不与停止 token 撞，允许集合 = 本任务的值 token + 行分隔 + 停止（升序）；token_table 记下 67 个；
- 适配器的累计预算：再加到超过 300 个就报错。
"""

from __future__ import annotations

import copy

import pytest
import torch

from src.adapters.factory import make_adapter
from src.adapters.image_policy import parse_policy
from src.constants import ADE_NUM_CLASSES, DEPTH_NUM_BINS, GRID_SIZE_E3B, MAX_NEW_GRID_TOKENS
from src.e3.config import ConfigError, load_config
from src.e3.tokens import GridTokenError, GridVocab, add_vocab, family_stop_id, vocab_from_config

FAMILIES = ("qwen3.5-0.8b", "gemma-4-e2b-it", "internvl3.5-2b-hf")


def _tiny(key):
    cfg = load_config()
    from src.adapters import registry

    fam = registry.get_model(key).family
    return make_adapter(key, device="cpu", dtype=torch.float32, image_policy=parse_policy(cfg.doc["tiny"]["image_policies"][fam]),
                        tiny=True, init_std=float(cfg.doc["tiny"]["init_std"]))


def test_vocab_from_config_counts_names_and_budget():
    cfg = load_config()
    v = cfg.vocab
    assert len(v.mask) == 2 and len(v.depth) == DEPTH_NUM_BINS and len(v.names) == 2 + DEPTH_NUM_BINS + 1 == 67
    assert v.names[:2] == tuple(cfg.doc["vocab"]["mask"]) and v.names[-1] == cfg.doc["vocab"]["row"]
    assert v.depth[0] == "<|e3_d00|>" and v.depth[-1] == f"<|e3_d{DEPTH_NUM_BINS - 1:02d}|>"
    b = v.budget()
    assert b == {"n_vocab": 67, "n_planned_extra": 150, "n_total_planned": 217, "limit": MAX_NEW_GRID_TOKENS}
    assert b["n_total_planned"] <= MAX_NEW_GRID_TOKENS
    assert cfg.doc["vocab"]["planned_extra"]["ade20k_classes"] == ADE_NUM_CLASSES  # 配置里的数字由常量钉住
    assert cfg.grid_n == GRID_SIZE_E3B == cfg.doc["grid"]["size"]  # 配置里的数字与常量逐数相同


def test_vocab_rejects_bad_configs():
    doc = copy.deepcopy(load_config().doc)
    doc["vocab"]["planned_extra"] = {"ade20k_classes": MAX_NEW_GRID_TOKENS - 67 + 1}
    with pytest.raises(GridTokenError, match="超过上限"):
        vocab_from_config(doc)
    doc = copy.deepcopy(load_config().doc)
    doc["vocab"]["mask"] = ["<|e3_m0|>"]
    with pytest.raises(GridTokenError, match="mask 词表"):
        vocab_from_config(doc)
    doc = copy.deepcopy(load_config().doc)
    doc["vocab"]["row"] = "<|e3_d05|>"  # 与深度档重名
    with pytest.raises(GridTokenError, match="重复"):
        vocab_from_config(doc)
    with pytest.raises(GridTokenError, match="depth 词表"):
        GridVocab(mask=("a", "b"), depth=("c",), row="r", planned_extra={})


def test_config_rejects_token_row_weight_decay_and_grid_size(tmp_path):
    import json

    doc = copy.deepcopy(load_config().doc)
    doc["optim"]["token_rows"]["weight_decay"] = 0.01
    p = tmp_path / "c.json"
    p.write_text(json.dumps(doc))
    with pytest.raises(ConfigError, match="weight_decay = 0"):
        load_config(p)
    doc = copy.deepcopy(load_config().doc)
    doc["grid"]["size"] = 16
    p.write_text(json.dumps(doc))
    with pytest.raises(GridTokenError, match="GRID_SIZE_E3B"):
        load_config(p)


@pytest.mark.parametrize("key", FAMILIES)
def test_add_vocab_layout_and_allowed_sets(key):
    ad = _tiny(key)
    v = load_config().vocab
    tok = add_vocab(ad, v)
    ids = tok.grid_ids
    assert len(set(ids)) == 67 and tok.stop_id not in ids and tok.stop_id == family_stop_id(ad) == ad._stop_ids()[0]
    assert tok.value_ids("refseg") == (tok.id_of(v.mask[0]), tok.id_of(v.mask[1]))
    assert len(tok.value_ids("depth")) == DEPTH_NUM_BINS
    seg = tok.allowed_ids("refseg", "task")
    assert seg == sorted([*tok.value_ids("refseg"), tok.row_id, tok.stop_id])
    dep = tok.allowed_ids("depth", "task")
    assert dep == sorted([*tok.value_ids("depth"), tok.row_id, tok.stop_id]) and len(dep) == DEPTH_NUM_BINS + 2
    assert tok.allowed_ids("refseg", "all") == tok.allowed_ids("depth", "all") == sorted([*ids, tok.stop_id])
    with pytest.raises(GridTokenError):
        tok.allowed_ids("refseg", "nope")
    table = ad.token_table()
    assert table["n_grid_tokens"] == 67 and sorted(g["id"] for g in table["grid_tokens"]) == sorted(ids)
    lay = tok.layout()
    assert lay["n_grid_tokens"] == 67 and lay["stop"] == tok.stop_id and lay["row"] == tok.row_id
    # 每个新 token 都是单个 token（适配器在 add_grid_tokens 里核对；这里从 tokenizer 再看一遍 surface）
    surf = {g["name"]: g["surface"] for g in table["grid_tokens"]}
    for n in (v.mask[0], v.depth[17], v.row):
        assert ad.tokenizer(surf[n], add_special_tokens=False)["input_ids"] == [tok.id_of(n)]
    # 适配器的累计预算（D10）：已有 67 个，再加 234 个就超过 300
    with pytest.raises(ValueError, match="超过上限"):
        ad.add_grid_tokens([f"<|x{i}|>" for i in range(MAX_NEW_GRID_TOKENS - 67 + 1)])
