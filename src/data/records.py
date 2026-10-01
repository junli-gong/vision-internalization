"""统一的样本记录（DEV-0012 任务书 §4）。

每个加载器都返回 LoadResult：records（Sample 列表，按 id 排序）与 stats（各类剔除的计数）。
Sample 只描述数据，不含任何模型相关的东西；提示的拼法属于 E1 评测框架。

字段约定：
- id：全局唯一，形如 "<source>/<split>/<本源的 id>"。
- images：图像绝对路径的元组（BLINK 等多图任务也放这里；当前 E1 的任务都是单图）。
- image_keys：每张图的规范身份键，用于按 image id 去重（D11-7），与 images 一一对应。约定：
  coco:<image_id>（COCO 2014/2017 共用 id）、nyuv2:<1-based 下标>、ade20k:<ADE_val_xxxxxxxx>、
  odinw:<数据集>:<file_name>、countbench:<行号>、blink:<idx>、embspatial:<question_id>、
  sunrgbd:<路径>、hypersim:<scene/cam/frame>、nuscenes:<文件名>。
- phash：每张图的感知哈希（落地时算好，存在清单里），与 images 一一对应；拿不到时为 None。
- task：refexp（指代定位与分割共用）、count、ovd（开放词汇检测）、rel_depth、spatial_qa。
- fields：任务输入（指代表达、类别列表、题干与选项、点坐标等）。
- gt：真值。框一律是绝对像素、浮点 xyxy（src/constants.BOX_FORMAT）；mask 为
  {"polygons": [[x0, y0, x1, y1, ...], ...]} 或 {"rle": {...}}，都附 height、width；计数为 int；
  选择题为选项字母（"A"、"B"…）；相对深度为更近的那个点（"A" 或 "B"）。
- source、split：数据源名与官方划分名。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass(frozen=True)
class Sample:
    id: str
    task: str
    images: tuple[str, ...]
    image_keys: tuple[str, ...]
    fields: dict[str, Any]
    gt: dict[str, Any]
    source: str
    split: str
    phash: tuple[str | None, ...] = ()

    def __post_init__(self) -> None:
        if len(self.images) != len(self.image_keys):
            raise ValueError(f"{self.id}: images 与 image_keys 数目不同")
        if self.phash and len(self.phash) != len(self.images):
            raise ValueError(f"{self.id}: phash 与 images 数目不同")


@dataclass
class LoadResult:
    records: list[Sample]
    stats: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.records)


def xywh_to_xyxy(b: Sequence[float]) -> list[float]:
    """COCO 的 [x, y, w, h] → 绝对像素 xyxy（浮点）。"""
    x, y, w, h = (float(v) for v in b)
    return [x, y, x + w, y + h]


def letter(i: int) -> str:
    """选项下标 → 字母（0 → "A"）。"""
    if not 0 <= i < 26:
        raise ValueError(f"选项下标越界：{i}")
    return chr(ord("A") + i)


def parse_paren_letter(ans: str) -> str:
    """"(C)" → "C"。格式不对就报错，不猜。"""
    s = ans.strip()
    if len(s) == 3 and s[0] == "(" and s[2] == ")" and "A" <= s[1] <= "Z":
        return s[1]
    raise ValueError(f"答案不是 (X) 形式：{ans!r}")
