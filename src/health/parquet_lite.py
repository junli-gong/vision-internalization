"""最小 parquet 读取器：只供 scripts/build_health_items.py 读 MMLU-Pro 与 GSM8K 的原始 parquet（DEV-0008）。

为什么自己写：开发机与服务器都没有 pyarrow，任务书只允许新增 IFEval checker 的依赖。健康检查运行时不读 parquet
（题目文件 configs/health_check_items.json 已含全部评分字段），只有重建题目文件时用到本模块。

只支持这两个文件实际用到的子集（固定 revision 的文件由 parquet-cpp-arrow 12.0.0 / 20.0.0 写出），其余一律抛
ParquetError，不猜：
- 压缩：UNCOMPRESSED、SNAPPY（原始 snappy 块格式，本模块自带解压）；
- 页：DICTIONARY_PAGE（PLAIN 编码）与 DATA_PAGE（v1）；DATA_PAGE_V2 报错；
- 值编码：PLAIN、PLAIN_DICTIONARY / RLE_DICTIONARY；定义级与重复级只接受 RLE（RLE/bit-packing 混合）；
- 物理类型：INT32、INT64、BYTE_ARRAY（带 UTF8/STRING 注解时解码成 str，否则 bytes）；
- 列结构：根下直接的叶子（REQUIRED / OPTIONAL），或标准三层 LIST（OPTIONAL|REQUIRED group(LIST) → REPEATED group →
  OPTIONAL|REQUIRED 叶子）。

用法：rows = read_rows(path) → list[dict]，按文件里的行序；列名为顶层字段名，LIST 列的值是 list（整列为 null 时 None）。
正确性的证据：开发时对两个固定 revision 的 test 文件逐行逐字段与 HF datasets-server 的 /rows JSON 比对（完成总结里记数），
测试里用 MMLU-Pro 的 validation 文件（同一写出器、同一 schema，含 LIST 列）做一致性检查。
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

MAGIC = b"PAR1"

# parquet.thrift 的枚举取值（只列用到的）
_TYPE_INT32, _TYPE_INT64, _TYPE_BYTE_ARRAY = 1, 2, 6
_REP_REQUIRED, _REP_OPTIONAL, _REP_REPEATED = 0, 1, 2
_CONVERTED_UTF8, _CONVERTED_LIST = 0, 3
_CODEC_UNCOMPRESSED, _CODEC_SNAPPY = 0, 1
_PAGE_DATA, _PAGE_DICTIONARY, _PAGE_DATA_V2 = 0, 2, 3
_ENC_PLAIN, _ENC_PLAIN_DICTIONARY, _ENC_RLE, _ENC_RLE_DICTIONARY = 0, 2, 3, 8


class ParquetError(ValueError):
    """文件超出本模块支持的子集，或内容自相矛盾。"""


# ---------------------------------------------------------------------------
# Thrift compact protocol：通用解码成 {字段号: 值}
# ---------------------------------------------------------------------------


class _Thrift:
    def __init__(self, buf: bytes, pos: int = 0) -> None:
        self.buf, self.pos = buf, pos

    def _byte(self) -> int:
        if self.pos >= len(self.buf):
            raise ParquetError("thrift 数据越界")
        b = self.buf[self.pos]
        self.pos += 1
        return b

    def _uvarint(self) -> int:
        out = shift = 0
        while True:
            b = self._byte()
            out |= (b & 0x7F) << shift
            if b < 0x80:
                return out
            shift += 7

    def _zigzag(self) -> int:
        n = self._uvarint()
        return (n >> 1) ^ -(n & 1)

    def _value(self, t: int) -> Any:
        if t == 1:
            return True
        if t == 2:
            return False
        if t == 3:
            b = self._byte()
            return b - 256 if b > 127 else b
        if t in (4, 5, 6):
            return self._zigzag()
        if t == 7:
            if self.pos + 8 > len(self.buf):
                raise ParquetError("thrift 数据越界")
            v = struct.unpack_from("<d", self.buf, self.pos)[0]
            self.pos += 8
            return v
        if t == 8:
            n = self._uvarint()
            if self.pos + n > len(self.buf):
                raise ParquetError("thrift 数据越界")
            s = bytes(self.buf[self.pos : self.pos + n])
            self.pos += n
            return s
        if t in (9, 10):
            h = self._byte()
            n, et = h >> 4, h & 0x0F
            if n == 15:
                n = self._uvarint()
            if et in (1, 2):  # 集合里的 bool 每个占一个字节
                return [self._byte() == 1 for _ in range(n)]
            return [self._value(et) for _ in range(n)]
        if t == 11:
            n = self._uvarint()
            if n == 0:
                return {}
            kv = self._byte()
            return {self._value(kv >> 4): self._value(kv & 0x0F) for _ in range(n)}
        if t == 12:
            return self.struct()
        raise ParquetError(f"未知的 thrift 类型 {t}")

    def struct(self) -> dict[int, Any]:
        out: dict[int, Any] = {}
        last = 0
        while True:
            h = self._byte()
            if h == 0:
                return out
            delta, t = h >> 4, h & 0x0F
            fid = last + delta if delta else self._zigzag()
            out[fid] = self._value(t)
            last = fid


# ---------------------------------------------------------------------------
# snappy（原始块格式）、RLE/bit-packing 混合编码、PLAIN 编码
# ---------------------------------------------------------------------------


def _uvarint_at(buf: bytes, pos: int) -> tuple[int, int]:
    out = shift = 0
    while True:
        if pos >= len(buf):
            raise ParquetError("varint 越界")
        b = buf[pos]
        pos += 1
        out |= (b & 0x7F) << shift
        if b < 0x80:
            return out, pos
        shift += 7


def snappy_decompress(src: bytes) -> bytes:
    """原始 snappy 块（不是 framing 格式）：开头是解压后长度的 varint，然后是 literal / copy 元素。"""
    n, pos = _uvarint_at(src, 0)
    out = bytearray()
    end = len(src)
    while pos < end:
        tag = src[pos]
        pos += 1
        kind = tag & 3
        if kind == 0:  # literal
            ln = tag >> 2
            if ln >= 60:
                nb = ln - 59
                ln = int.from_bytes(src[pos : pos + nb], "little")
                pos += nb
            ln += 1
            if pos + ln > end:
                raise ParquetError("snappy literal 越界")
            out += src[pos : pos + ln]
            pos += ln
            continue
        if kind == 1:
            ln = ((tag >> 2) & 7) + 4
            off = ((tag >> 5) << 8) | src[pos]
            pos += 1
        elif kind == 2:
            ln = (tag >> 2) + 1
            off = int.from_bytes(src[pos : pos + 2], "little")
            pos += 2
        else:
            ln = (tag >> 2) + 1
            off = int.from_bytes(src[pos : pos + 4], "little")
            pos += 4
        if off == 0 or off > len(out):
            raise ParquetError(f"snappy copy 偏移 {off} 非法（已输出 {len(out)} 字节）")
        start = len(out) - off
        if off >= ln:
            out += out[start : start + ln]
        else:  # 重叠拷贝：以最后 off 个字节为周期重复
            pattern = bytes(out[start:])
            out += (pattern * (ln // off + 1))[:ln]
    if len(out) != n:
        raise ParquetError(f"snappy 解压长度 {len(out)} 与声明的 {n} 不一致")
    return bytes(out)


def rle_hybrid(buf: bytes, pos: int, end: int, bit_width: int, count: int) -> list[int]:
    """RLE/bit-packing 混合编码，从 buf[pos:end] 读 count 个值（bit-packed 段末尾的填充丢弃）。"""
    if not 0 <= bit_width <= 32:
        raise ParquetError(f"bit width {bit_width} 超出范围")
    out: list[int] = []
    byte_width = (bit_width + 7) // 8
    mask = (1 << bit_width) - 1
    while len(out) < count:
        if pos >= end:
            raise ParquetError(f"RLE 数据不足：要 {count} 个值，只读到 {len(out)} 个")
        header, pos = _uvarint_at(buf, pos)
        if header & 1:  # bit-packed：header>>1 组，每组 8 个值占 bit_width 个字节，低位在前
            groups = header >> 1
            if pos + groups * bit_width > end:
                raise ParquetError("bit-packed 段越界")
            for _ in range(groups):
                chunk = int.from_bytes(buf[pos : pos + bit_width], "little")
                pos += bit_width
                out.extend((chunk >> (i * bit_width)) & mask for i in range(8))
        else:  # RLE：header>>1 次重复，值占 byte_width 个字节
            run = header >> 1
            if pos + byte_width > end:
                raise ParquetError("RLE 段越界")
            v = int.from_bytes(buf[pos : pos + byte_width], "little")
            pos += byte_width
            out.extend([v] * run)
    return out[:count]


def _plain(buf: bytes, pos: int, end: int, physical: int, count: int, utf8: bool) -> list[Any]:
    out: list[Any] = []
    if physical in (_TYPE_INT32, _TYPE_INT64):
        width, fmt = (4, "<i") if physical == _TYPE_INT32 else (8, "<q")
        if pos + width * count > end:
            raise ParquetError("PLAIN 整数越界")
        for k in range(count):
            out.append(struct.unpack_from(fmt, buf, pos + width * k)[0])
        return out
    if physical == _TYPE_BYTE_ARRAY:
        for _ in range(count):
            if pos + 4 > end:
                raise ParquetError("PLAIN BYTE_ARRAY 越界")
            ln = struct.unpack_from("<I", buf, pos)[0]
            pos += 4
            if pos + ln > end:
                raise ParquetError("PLAIN BYTE_ARRAY 越界")
            raw = bytes(buf[pos : pos + ln])
            pos += ln
            out.append(raw.decode("utf-8") if utf8 else raw)
        return out
    raise ParquetError(f"不支持的物理类型 {physical}")


# ---------------------------------------------------------------------------
# schema 与列
# ---------------------------------------------------------------------------


class _Leaf:
    """一个叶子列：顶层字段名、物理类型、是否 UTF8、最大定义级/重复级，以及 LIST 结构里各层的定义级。"""

    def __init__(self, name: str, physical: int, utf8: bool, max_def: int, max_rep: int, is_list: bool,
                 list_present_def: int | None) -> None:
        self.name, self.physical, self.utf8 = name, physical, utf8
        self.max_def, self.max_rep, self.is_list = max_def, max_rep, is_list
        self.list_present_def = list_present_def  # LIST 列：定义级 ≥ 它表示列表本身非 null


def _is_utf8(el: dict[int, Any]) -> bool:
    logical = el.get(10) or {}
    return el.get(6) == _CONVERTED_UTF8 or 1 in logical  # LogicalType 的第 1 个字段是 STRING


def _leaves(schema: list[dict[int, Any]]) -> list[_Leaf]:
    root = schema[0]
    n_top = root.get(5, 0)
    leaves: list[_Leaf] = []
    i = 1
    for _ in range(n_top):
        el = schema[i]
        name = el[4].decode("utf-8")
        rep = el.get(3, _REP_REQUIRED)
        if rep == _REP_REPEATED:
            raise ParquetError(f"{name}：不支持顶层 REPEATED 字段")
        top_def = 1 if rep == _REP_OPTIONAL else 0
        if not el.get(5):  # 叶子
            leaves.append(_Leaf(name, el.get(1), _is_utf8(el), top_def, 0, False, None))
            i += 1
            continue
        # 标准三层 LIST
        if el.get(6) != _CONVERTED_LIST and 3 not in (el.get(10) or {}):
            raise ParquetError(f"{name}：只支持 LIST 注解的 group")
        if el[5] != 1 or i + 2 >= len(schema):
            raise ParquetError(f"{name}：LIST group 结构不符")
        mid, leaf = schema[i + 1], schema[i + 2]
        if mid.get(3) != _REP_REPEATED or mid.get(5) != 1 or leaf.get(5):
            raise ParquetError(f"{name}：LIST 的中间层必须是只有一个叶子的 REPEATED group")
        leaf_rep = leaf.get(3, _REP_REQUIRED)
        if leaf_rep == _REP_REPEATED:
            raise ParquetError(f"{name}：LIST 元素不能是 REPEATED")
        max_def = top_def + 1 + (1 if leaf_rep == _REP_OPTIONAL else 0)
        leaves.append(_Leaf(name, leaf.get(1), _is_utf8(leaf), max_def, 1, True, top_def))
        i += 3
    if i != len(schema):
        raise ParquetError(f"schema 有 {len(schema)} 个元素，只解释了 {i} 个")
    return leaves


def _decompress(codec: int, payload: bytes, expect: int) -> bytes:
    if codec == _CODEC_UNCOMPRESSED:
        data = payload
    elif codec == _CODEC_SNAPPY:
        data = snappy_decompress(payload)
    else:
        raise ParquetError(f"不支持的压缩编码 {codec}")
    if len(data) != expect:
        raise ParquetError(f"页解压后 {len(data)} 字节，页头声明 {expect}")
    return data


def _read_column_chunk(data: bytes, meta: dict[int, Any], leaf: _Leaf) -> tuple[list[int], list[int], list[Any]]:
    """读一个列块的全部页，返回 (重复级, 定义级, 非 null 值)。"""
    if meta.get(1) != leaf.physical:
        raise ParquetError(f"{leaf.name}：列块的物理类型 {meta.get(1)} 与 schema 的 {leaf.physical} 不一致")
    codec = meta[4]
    start = meta.get(11) or meta[9]
    end = start + meta[7]
    if end > len(data):
        raise ParquetError(f"{leaf.name}：列块越界")
    dictionary: list[Any] | None = None
    reps: list[int] = []
    defs: list[int] = []
    values: list[Any] = []
    pos = start
    while pos < end:
        th = _Thrift(data, pos)
        ph = th.struct()
        body = th.pos
        page_type, raw_size, comp_size = ph[1], ph[2], ph[3]
        if body + comp_size > end:
            raise ParquetError(f"{leaf.name}：页越界")
        page = _decompress(codec, data[body : body + comp_size], raw_size)
        pos = body + comp_size
        if page_type == _PAGE_DICTIONARY:
            dh = ph[7]
            if dictionary is not None:
                raise ParquetError(f"{leaf.name}：一个列块里有两个字典页")
            if dh.get(2) not in (_ENC_PLAIN, _ENC_PLAIN_DICTIONARY):
                raise ParquetError(f"{leaf.name}：字典页编码 {dh.get(2)} 不受支持")
            dictionary = _plain(page, 0, len(page), leaf.physical, dh[1], leaf.utf8)
            continue
        if page_type == _PAGE_DATA_V2:
            raise ParquetError(f"{leaf.name}：不支持 DATA_PAGE_V2")
        if page_type != _PAGE_DATA:
            raise ParquetError(f"{leaf.name}：不支持的页类型 {page_type}")
        dh = ph[5]
        n = dh[1]
        p = 0
        page_reps = [0] * n
        page_defs = [leaf.max_def] * n
        for max_level, enc_field, target in ((leaf.max_rep, 4, "rep"), (leaf.max_def, 3, "def")):
            if max_level == 0:
                continue
            if dh.get(enc_field) != _ENC_RLE:
                raise ParquetError(f"{leaf.name}：{target} 级编码 {dh.get(enc_field)} 不受支持（只支持 RLE）")
            ln = struct.unpack_from("<I", page, p)[0]
            p += 4
            levels = rle_hybrid(page, p, p + ln, max_level.bit_length(), n)
            p += ln
            if target == "rep":
                page_reps = levels
            else:
                page_defs = levels
        n_values = sum(1 for d in page_defs if d == leaf.max_def)
        enc = dh[2]
        if enc == _ENC_PLAIN:
            page_values = _plain(page, p, len(page), leaf.physical, n_values, leaf.utf8)
        elif enc in (_ENC_PLAIN_DICTIONARY, _ENC_RLE_DICTIONARY):
            if dictionary is None:
                raise ParquetError(f"{leaf.name}：字典编码的数据页前面没有字典页")
            bit_width = page[p]
            idx = rle_hybrid(page, p + 1, len(page), bit_width, n_values)
            if any(k >= len(dictionary) for k in idx):
                raise ParquetError(f"{leaf.name}：字典下标越界")
            page_values = [dictionary[k] for k in idx]
        else:
            raise ParquetError(f"{leaf.name}：不支持的值编码 {enc}")
        reps += page_reps
        defs += page_defs
        values += page_values
    if pos != end:
        raise ParquetError(f"{leaf.name}：页没有恰好铺满列块")
    if len(defs) != meta[5]:
        raise ParquetError(f"{leaf.name}：读到 {len(defs)} 个级别，列块声明 {meta[5]} 个值")
    return reps, defs, values


def _assemble(leaf: _Leaf, reps: list[int], defs: list[int], values: list[Any]) -> list[Any]:
    """把一个列块的级别与值拼回每行的值。"""
    it = iter(values)
    if not leaf.is_list:
        return [next(it) if d == leaf.max_def else None for d in defs]
    rows: list[Any] = []
    for r, d in zip(reps, defs):
        if r == 0:
            rows.append(None if d < leaf.list_present_def else [])
        elif not rows or rows[-1] is None:
            raise ParquetError(f"{leaf.name}：重复级在行首之前出现，或出现在 null 列表里")
        if d > leaf.list_present_def:  # 有元素（值或 null 元素）
            rows[-1].append(next(it) if d == leaf.max_def else None)
    return rows


def read_rows(path: str | Path) -> list[dict[str, Any]]:
    """读整个文件，按行序返回 {列名: 值} 的列表。"""
    data = Path(path).read_bytes()
    if len(data) < 12 or data[:4] != MAGIC or data[-4:] != MAGIC:
        raise ParquetError(f"{path} 不是 parquet 文件（首尾没有 PAR1）")
    n_meta = struct.unpack_from("<I", data, len(data) - 8)[0]
    meta = _Thrift(data, len(data) - 8 - n_meta).struct()
    leaves = _leaves(meta[2])
    rows: list[dict[str, Any]] = []
    for rg in meta[4]:
        n_rows = rg[3]
        chunks = rg[1]
        if len(chunks) != len(leaves):
            raise ParquetError(f"行组有 {len(chunks)} 个列块，schema 有 {len(leaves)} 个叶子")
        cols: dict[str, list[Any]] = {}
        for leaf, cc in zip(leaves, chunks):
            cm = cc[3]
            path_in_schema = [p.decode("utf-8") for p in cm[3]]
            if path_in_schema[0] != leaf.name:
                raise ParquetError(f"列块路径 {path_in_schema} 与叶子 {leaf.name} 不对应")
            col = _assemble(leaf, *_read_column_chunk(data, cm, leaf))
            if len(col) != n_rows:
                raise ParquetError(f"{leaf.name}：拼出 {len(col)} 行，行组声明 {n_rows} 行")
            cols[leaf.name] = col
        rows.extend({name: cols[name][k] for name in cols} for k in range(n_rows))
    if len(rows) != meta[3]:
        raise ParquetError(f"读到 {len(rows)} 行，文件声明 {meta[3]} 行")
    return rows
