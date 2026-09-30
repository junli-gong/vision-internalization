"""最小 parquet 读取器（src/health/parquet_lite.py，DEV-0008）的测试。

fixture：tests/fixtures/health/mmlu_pro_validation.parquet 是 TIGER-Lab/MMLU-Pro 固定 revision 的 validation 文件
（MIT，70 行，42,857 字节）：与 test 文件同一写出器（parquet-cpp-arrow 20.0.0）、同一 schema，含 LIST 列、字典页、
RLE 定义级/重复级、snappy。期望的规范化哈希由 HF datasets-server 的 /rows JSON 独立算出（不经本模块）。
开发时另对两个 test 文件（MMLU-Pro 12,032 行、GSM8K 1,319 行）逐行逐字段与 /rows JSON 比对，0 处不一致（完成总结）。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.health.parquet_lite import ParquetError, read_rows, rle_hybrid, snappy_decompress

FIX = Path(__file__).resolve().parent / "fixtures/health"
VALIDATION = FIX / "mmlu_pro_validation.parquet"
VALIDATION_FILE_SHA256 = "139423c23722e480c807ac4a191409a710cfce4eba744c1d641cf88e730e2078"
# sha256(json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")))，rows 取自 datasets-server
VALIDATION_ROWS_SHA256 = "c06631dfc6dc2efe5ad713fcc0673edb2b93a34796fab0b02f33fe4b24dd70e3"


def _canon(rows) -> str:
    return hashlib.sha256(json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def test_reads_mmlu_pro_validation_like_hf():
    assert hashlib.sha256(VALIDATION.read_bytes()).hexdigest() == VALIDATION_FILE_SHA256
    rows = read_rows(VALIDATION)
    assert len(rows) == 70
    assert list(rows[0]) == ["question_id", "question", "options", "answer", "answer_index", "cot_content", "category", "src"]
    assert _canon(rows) == VALIDATION_ROWS_SHA256
    # 与内容无关的自洽检查：解码错一位多半会破坏这些
    assert [r["question_id"] for r in rows] == list(range(70))
    assert all(r["answer"] == "ABCDEFGHIJ"[r["answer_index"]] for r in rows)
    assert len({r["category"] for r in rows}) == 14
    assert all(isinstance(o, str) for r in rows for o in r["options"])
    assert rows[0]["options"] == ["0", "30", "3", "10", "12", "50", "2", "100", "20", "5"]


def test_snappy_elements():
    # literal
    assert snappy_decompress(b"\x05\x10hello") == b"hello"
    # 1 字节偏移的重叠拷贝："abc" + copy(len=6, offset=3)
    assert snappy_decompress(b"\x09\x08abc\x09\x03") == b"abcabcabc"
    # 2 字节偏移的拷贝："abcd" + copy(len=4, offset=4)
    assert snappy_decompress(b"\x08\x0cabcd\x0e\x04\x00") == b"abcdabcd"
    # 4 字节偏移的拷贝
    assert snappy_decompress(b"\x08\x0cabcd\x0f\x04\x00\x00\x00") == b"abcdabcd"
    # 长 literal：长度写在后面的 1 个字节里
    payload = bytes(range(100))
    assert snappy_decompress(b"\x64\xf0\x63" + payload) == payload


def test_snappy_rejects_bad_streams():
    with pytest.raises(ParquetError, match="偏移"):
        snappy_decompress(b"\x04\x09\x00")  # 输出为空时拷贝
    with pytest.raises(ParquetError, match="长度"):
        snappy_decompress(b"\x06\x10hello")  # 声明 6，实际 5
    with pytest.raises(ParquetError, match="越界"):
        snappy_decompress(b"\x05\x10hel")


def test_rle_hybrid_rle_and_bitpacked():
    # RLE 段：5 个 6（bit width 3，值占 1 字节）
    assert rle_hybrid(b"\x0a\x06", 0, 2, 3, 5) == [6] * 5
    # bit-packed 段：parquet 规范的例子，0..7 以 bit width 3 编码为 10001000 11000110 11111010
    assert rle_hybrid(b"\x03\x88\xc6\xfa", 0, 4, 3, 8) == list(range(8))
    # 只要前 5 个：末尾填充丢弃；两段连起来
    assert rle_hybrid(b"\x03\x88\xc6\xfa\x04\x01", 0, 6, 3, 10) == [0, 1, 2, 3, 4, 5, 6, 7, 1, 1]
    assert rle_hybrid(b"\x03\x88\xc6\xfa", 0, 4, 3, 5) == [0, 1, 2, 3, 4]
    # bit width 0：全是 0
    assert rle_hybrid(b"\x08", 0, 1, 0, 4) == [0, 0, 0, 0]
    with pytest.raises(ParquetError, match="不足"):
        rle_hybrid(b"\x0a\x06", 0, 2, 3, 6)


def test_rejects_non_parquet(tmp_path):
    p = tmp_path / "x.parquet"
    p.write_bytes(b"not a parquet file at all")
    with pytest.raises(ParquetError, match="PAR1"):
        read_rows(p)
