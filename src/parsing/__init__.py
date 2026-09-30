"""输出解析层（简报 §8.1"解析器：对各模型的输出格式鲁棒，并记录解析失败率"；DEV-0006）。

与模型无关的部分在这里；各家族原生框格式的解析在 src/adapters/<family>.py（适配器的 parse_boxes / parse_one_box）。
典型用法：

    res = adapter.generate(inputs, enable_thinking=False, max_new_tokens=256)
    p = parse_generation(res, adapter.parse_one_box, width=W, height=H)   # 或 parse_count、parse_choice(letters=…) 等
    tally.add(p)                                                          # ParseTally：解析失败率
"""

from src.parsing.answers import normalize_phrase, parse_choice, parse_count, parse_named_choice
from src.parsing.core import (
    BoxParse,
    OneBoxParse,
    ParseFailure,
    ParseResult,
    ParseTally,
    one_box,
    parse_generation,
)
from src.parsing.json_extract import extract_json
from src.parsing.spatial import parse_json_boxes, parse_polygons

__all__ = [
    "BoxParse",
    "OneBoxParse",
    "ParseFailure",
    "ParseResult",
    "ParseTally",
    "extract_json",
    "normalize_phrase",
    "one_box",
    "parse_choice",
    "parse_count",
    "parse_generation",
    "parse_json_boxes",
    "parse_named_choice",
    "parse_polygons",
]
