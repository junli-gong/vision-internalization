"""LISA-7B 外部对照的清单（DEV-0075；D154）：在我们的环境里按 W2 的取样导出，LISA 的单独环境只读这份清单。

- 句子：configs/w2_llmseg_subsets.json（W2 的 5 个 test 划分 × 300 句，hash_rank，salt w2_llmseg_v1；预注册 U-1），与 LLMSeg
  （RUN-0078）同一份。核对（check_subsets）、读取（load_subsets）与取样（subset_samples）全部直接用 src/llmseg/runner.py 的函数，
  样本由 W2 的加载方式（src/w2/data.w2_items）给出，逐 id、逐顺序与 LLMSeg 相同。
- 清单的一行（src/lisa/common.MANIFEST_KEYS）：W2 的样本 id、划分、sent_id、COCO 的 image_id 与 ann_id、图像的绝对路径与 sha256、
  原图尺寸（标注的 height、width）、句子（REFER 的 sent = 样本的 fields.expression；官方评测再 strip、lower）。不含 GT。
- --tiny（开发机彩排）：tests/fixtures 的 W2 划分，子集按同一规则现算（同 scripts/run_llmseg.py --tiny）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.data.records import Sample
from src.data.subsets import dump_config
from src.e1.runner import ids_digest
from src.e1.tasks import DataPaths, _sent_id
from src.lisa import common as C
from src.llmseg import runner as LR
from src.w2.data import W2_SPLITS_PATH, load_splits, tiny_paths, tiny_splits, w2_items

MANIFEST_VERSION = 1


def load_selection(*, tiny: bool) -> tuple[dict[str, Any], dict[str, Any], list[Path]]:
    """(W2 的划分清单, 子集清单, 用到的配置文件)。真运行先 check_subsets（与按规则重算的逐字节相同）。"""
    if tiny:
        splits_doc = tiny_splits()
        return splits_doc, LR.build_subsets(splits_doc, splits=LR.test_splits(), source={"path": "tiny（src/w2/data.tiny_splits）"}), []
    problems = LR.check_subsets()
    if problems:
        raise C.LisaError("；".join(problems))
    return load_splits(W2_SPLITS_PATH), LR.load_subsets(), [W2_SPLITS_PATH, LR.SUBSETS_PATH]


def data_paths(data_root: str | Path | None, *, tiny: bool) -> DataPaths:
    if tiny:
        return tiny_paths()
    if data_root is None:
        raise C.LisaError("真数据要给 --data-root（或设 VISINT_DATA_ROOT）")
    return DataPaths(root=Path(data_root))


def split_samples(split: str, paths: DataPaths, splits_doc: Mapping[str, Any], subsets_doc: Mapping[str, Any],
                  limit: int | None = None) -> list[Sample]:
    """一个划分的子集样本（W2 的加载顺序；同 scripts/run_llmseg.py）。"""
    s = LR.subset_samples(w2_items(C.TASK, split, paths, splits=splits_doc), subsets_doc["splits"][split]["ids"])
    return s if limit is None else s[:limit]


def manifest_row(sample: Sample, split: str) -> dict[str, Any]:
    img = Path(sample.images[0]).resolve()
    w, h = (int(v) for v in sample.fields["image_size"])
    m = sample.gt["mask"]
    if (int(m["height"]), int(m["width"])) != (h, w):
        raise C.LisaError(f"{sample.id}：图像尺寸 {(h, w)} 与 GT 的 {(m['height'], m['width'])} 不同")
    row = {"id": sample.id, "w2_split": split, "sent_id": _sent_id(sample), "image_id": int(sample.fields["image_id"]),
           "ann_id": int(sample.fields["ann_id"]), "image": str(img), "image_sha256": C.sha256_file(img), "height": h, "width": w,
           "sent": str(sample.fields["expression"])}
    C.validate_manifest_row(row)
    return row


def build_manifest(samples_by_split: Mapping[str, Sequence[Sample]], *, subsets_doc: Mapping[str, Any], tiny: bool,
                   limit: int | None) -> tuple[dict[str, Any], str]:
    """(meta, manifest.jsonl 的文本)。meta 不含时间与路径以外的易变信息：同一份数据重导出逐字节相同。"""
    rows = [manifest_row(s, split) for split, ss in samples_by_split.items() for s in ss]
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    splits = {}
    for split, ss in samples_by_split.items():
        sub = subsets_doc["splits"][split]
        splits[split] = {"n": len(ss), "ids_sha256": ids_digest([s.id for s in ss]), "subset_n": sub["n"], "subset_sha256": sub["sha256"]}
    meta = {"version": MANIFEST_VERSION, "experiment": C.EXPERIMENT, "system": C.SYSTEM,
            "name": "LISA-7B 外部对照的输入清单（DEV-0075；D154）：W2 test 子集，句子同 LLMSeg（预注册 U-1）",
            "subsets": {"config": "tiny（同一规则现算）" if tiny else "configs/w2_llmseg_subsets.json",
                        "sha256": C.sha256_bytes(dump_config(subsets_doc).encode("utf-8")) if tiny else C.sha256_file(LR.SUBSETS_PATH),
                        "rule": subsets_doc["rule"],
                        "salt": subsets_doc["salt"], "n_per_split": subsets_doc["n_per_split"]},
            "keys": list(C.MANIFEST_KEYS), "splits": splits, "tiny": tiny, "limit": limit,
            "rows_sha256": C.sha256_bytes(text.encode("utf-8")),
            "note": "不含 GT；GT 只在我们的环境计分时读（scripts/lisa_w2.py score）"}
    return meta, text


def export(out: str | Path, *, paths: DataPaths, splits: Sequence[str], limit: int | None, tiny: bool) -> dict[str, Any]:
    """写 <out>/lisa-7b-v1/inputs/{manifest.json, manifest.jsonl}。已有清单时：内容逐字节相同就不动，不同就报错（不覆盖）。"""
    splits_doc, subsets_doc, _ = load_selection(tiny=tiny)
    by = {s: split_samples(s, paths, splits_doc, subsets_doc, limit) for s in splits}
    meta, text = build_manifest(by, subsets_doc=subsets_doc, tiny=tiny, limit=limit)
    meta_text = json.dumps(meta, indent=1, ensure_ascii=False) + "\n"
    d = C.inputs_dir(out)
    mp, rp = d / C.MANIFEST_META, d / C.MANIFEST_ROWS
    if mp.exists() or rp.exists():
        same = mp.is_file() and rp.is_file() and mp.read_text(encoding="utf-8") == meta_text and rp.read_text(encoding="utf-8") == text
        if not same:
            raise C.LisaError(f"{d} 里已有不同的清单，拒绝覆盖（换一个运行目录）")
        return meta
    d.mkdir(parents=True, exist_ok=True)
    rp.write_text(text, encoding="utf-8")
    mp.write_text(meta_text, encoding="utf-8")
    return meta


__all__ = ["build_manifest", "data_paths", "export", "load_selection", "manifest_row", "split_samples"]
