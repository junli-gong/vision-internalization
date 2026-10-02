"""RefCOCO / RefCOCO+ / RefCOCOg 的落地（DEV-0012 任务书 §1.1；data_sources A §1）。

来源选择：官方主机 bvisionweb1.cs.unc.edu 已不可解析，refer README 指向的 issue #14 给出 2022-04-13 的 Wayback 快照
（`if_` 原始字节形式，支持 Range）。快照里的 zip 就是官方主机当年提供的文件（原站 Last-Modified 2016）。用 Range 只取
需要的两个成员：refs(unc|umd).p 与 instances.json，不下整包。

与官方的核对：
1. 逐字节：zipfile 读完成员时核对 zip 中央目录里的 CRC32（2016 年打包时写入）；refs pickle 的 sha256 与 P0 记录的值
   （data_sources A §1.2，refs_stats.json）比对。
2. 逐条：与独立的 HF 镜像 jxu124/refcoco*（固定 revision）在评测划分上逐 ref 比对 ref_id、ann_id、image_id、category_id、
   split、sent_ids、句子文本，并比对 raw_anns 的 bbox 与 instances.json 的 bbox。
3. 计数：各评测划分的 refs / sentences / images 与 P0 的统计（data_sources A §1.3）一致。

本地副本（DEV-0026，国内网络连不上 web.archive.org 时）：--refer-zip-dir 指向放着 refcoco.zip、refcoco+.zip、refcocog.zip
的目录，不访问 web.archive.org。核对不少于远端：中央目录（尾部）的 sha256 = SPEC refer_zip_pins、成员 CRC32、refs 的
sha256 = P0、instances.json 的字节数与 CRC32 = SPEC，整个文件的 sha256 记进清单（SPEC refer_zip_sha256 钉了就比对）；
之后的计数与镜像比对同远端。读本地 zip 的代码与训练源 refcoco_train 共用（fetch_refer_members）。不给时行为不变。

mask 来源：refer zip 自带的 instances.json（REFER API 读的就是它；data_sources A §1.6 来源 A）。
评测划分：RefCOCO、RefCOCO+ 取 unc 的 val/testA/testB，RefCOCOg 取 umd 的 val/test（D11-1）。训练划分不落地。
图像：只下评测划分用到的 COCO train2014 图像，逐张按单图 URL（A §1.5）。

落地后的布局（<root>/refcoco/）：
- <dataset>_<splitBy>.json：{dataset, splitBy, eval_splits, refs: [...], anns: {ann_id: {...}}, images: {image_id: {...}},
  categories: {id: name}}。refs 只含评测划分；每个 ref 带 sentences [{sent_id, sent, raw}]。
- images/COCO_train2014_<id:012d>.jpg：三个数据集共用。
- MANIFEST.json。
"""

from __future__ import annotations

import json
import pickle
import zipfile
from pathlib import Path
from typing import Any

from src.data.land import net
from src.data.land.common import (
    LandContext,
    LocalZip,
    MemberJob,
    budget_gate,
    check_zip_pins,
    download_images,
    extract_member,
    fetch_hf_file,
    fetch_zip_members,
    hf_tree,
    open_remote_zip,
    read_zip_index,
    write_json,
    zip_source,
)

SOURCE = "refcoco"
# --refer-zip-dir 时清单 upstream 里的核对说明（不给时沿用 DEV-0012 的原说明，见 _fetch_wayback_members）
LOCAL_OFFICIAL_CHECK = (
    "--refer-zip-dir 的本地副本（Wayback 2022-04-13 快照的拷贝，不访问 web.archive.org）；中央目录 sha256 = SPEC；成员 CRC32；"
    "refs sha256 = P0；instances.json 字节数与 CRC32 = SPEC；整个文件的 sha256 记进清单（钉住时比对）；评测划分逐条比对 HF 镜像 jxu124"
)
_WB = "https://web.archive.org/web/{ts}if_/https://bvisionweb1.cs.unc.edu/licheng/referit/data/{zip}"

SPEC: dict[str, Any] = {
    "datasets": {
        "refcoco": {
            "zip_url": _WB.format(ts="20220413011718", zip="refcoco.zip"),
            "zip_bytes": 53_993_993,
            "split_by": "unc",
            "refs_member": "refcoco/refs(unc).p",
            "refs_sha256": "d4a8dd3152f130127924f9d0f0ea30f4dc43ec19e1b47bc55af972f90614f8ba",
            "instances_member": "refcoco/instances.json",
            "eval_splits": ["val", "testA", "testB"],
            # (refs, sentences, images)，P0 refs_stats.json
            "expect": {"val": (3811, 10834, 1500), "testA": (1975, 5657, 750), "testB": (1810, 5095, 750)},
            "mirror": {
                "repo": "jxu124/refcoco",
                "rev": "9fba8200c5326e996f789191f095bd464ef1d09e",
                # 镜像的文件名 → 官方 split（jxu124 把 testA 叫 test，见 data_sources A §1.7）
                "files": {"validation": "val", "test": "testA", "testB": "testB"},
            },
        },
        "refcoco+": {
            "zip_url": _WB.format(ts="20220413011656", zip="refcoco+.zip"),
            "zip_bytes": 45_613_210,
            "split_by": "unc",
            "refs_member": "refcoco+/refs(unc).p",
            "refs_sha256": "7de24c182449758b9d9ae87d968b1f25831236a54d38d45408e2e0fdd318e802",
            "instances_member": "refcoco+/instances.json",
            "eval_splits": ["val", "testA", "testB"],
            "expect": {"val": (3805, 10758, 1500), "testA": (1975, 5726, 750), "testB": (1798, 4889, 750)},
            "mirror": {
                "repo": "jxu124/refcocoplus",
                "rev": "12e20b0b6039fbf656e89e2f26597e84c1037847",
                "files": {"validation": "val", "test": "testA", "testB": "testB"},
            },
        },
        "refcocog": {
            "zip_url": _WB.format(ts="20220413012904", zip="refcocog.zip"),
            "zip_bytes": 56_712_951,
            "split_by": "umd",
            "refs_member": "refcocog/refs(umd).p",
            "refs_sha256": "0331c7533537b67c2f7ac8c8bab0da2d379d1754c6f5c110fd79f70e17e7bddb",
            "instances_member": "refcocog/instances.json",
            "eval_splits": ["val", "test"],
            "expect": {"val": (2573, 4896, 1300), "test": (5023, 9602, 2600)},
            "mirror": {
                "repo": "jxu124/refcocog",
                "rev": "55319436ad54b9480cefda6b9d64397de92456dd",
                "files": {"validation": "val", "test": "test"},
            },
        },
    },
    # refer zip 的中央目录（尾部）与 instances.json 的钉住值：开发机 2026-10-02 用 Range 读中央目录得到（DEV-0022，原在
    # refcoco_train.SPEC，DEV-0026 搬到这里供 E1 的 --refer-zip-dir 共用）。size 就是上面的 zip_bytes。
    "refer_zip_pins": {
        "refcoco": {"size": 53_993_993, "n_entries": 3, "start_dir": 53_993_698, "tail_bytes": 295, "tail_sha256": "df4f6a89577f53ffef1cac2eace0a95e46d6ab7760b7fcd73f9172c4e2a9b56c"},
        "refcoco+": {"size": 45_613_210, "n_entries": 2, "start_dir": 45_613_005, "tail_bytes": 205, "tail_sha256": "caa09ba1ed952b64715a94d6fbe1058099b9ab4d91542d97ed8e79d3b9f5a7d2"},
        "refcocog": {"size": 56_712_951, "n_entries": 4, "start_dir": 56_712_574, "tail_bytes": 377, "tail_sha256": "d122663e698b8570c0994978e57c54eea04279e1e3724933ccda15f98683f499"},
    },
    "instances_pins": {  # (解压后字节数, CRC32)
        "refcoco": (120_213_916, "cecd6518"),
        "refcoco+": (120_197_185, "30ba2d2b"),
        "refcocog": (124_416_571, "78cbaae4"),
    },
    # 整个 zip 文件的 sha256（--refer-zip-dir 时核对）。中央目录推不出它（要读全部字节，含没取的 refs(google).p），
    # P0 与 B200 的落地清单也没记；None = 还没钉住：只记进清单、不比对。管理员下载后用 scripts/data/refer_zip_sha256.py 补钉。
    "refer_zip_sha256": {"refcoco": None, "refcoco+": None, "refcocog": None},
    "image_url": "http://images.cocodataset.org/train2014/COCO_train2014_{id:012d}.jpg",
    # 评测划分三者并集 6,549 张、1,105,415,725 B（data_sources A §1.5，逐张 HEAD）；--limit 时按均值估。
    "eval_union_images": 6549,
    "eval_union_bytes": 1_105_415_725,
    "mean_image_bytes": 169_660,
    "hf_base": "https://huggingface.co",
    "license": {
        "annotations": "未声明（refer README 与 zip 内都没有数据许可；仓库级 Apache-2.0 指 API 代码，是否覆盖数据未核实）。data_sources A §1.1",
        "images": "COCO train2014。COCO Terms of Use 原文：\"The annotations in this dataset along with this website belong to the COCO Consortium and are licensed under a Creative Commons Attribution 4.0 License.\" / \"The COCO Consortium does not own the copyright of the images. Use of the images must abide by the Flickr Terms of Use. The users of the images accept full responsibility for the use of the dataset…\" https://cocodataset.org/#termsofuse",
        "project_terms": "D28-9：限非商业研究，不再分发任何数据或由数据生成的图像",
    },
}


class _NoGlobals(pickle.Unpickler):
    """refs pickle 只含 list/dict/str/int；拒绝一切 global，防止反序列化执行代码。"""

    def find_class(self, module: str, name: str):  # noqa: D401
        raise pickle.UnpicklingError(f"refs pickle 里出现 global {module}.{name}，拒绝")


def load_refs_pickle(path: Path) -> list[dict[str, Any]]:
    with open(path, "rb") as f:
        return _NoGlobals(f).load()


def image_file_name(image_id: int) -> str:
    return f"COCO_train2014_{image_id:012d}.jpg"


# ---------------------------------------------------------------------------
# refer zip 的成员：E1 的 --refer-zip-dir 与训练源 refcoco_train 共用这一份（DEV-0026）
# ---------------------------------------------------------------------------


def refer_zip_path(local_dir: str | Path, name: str) -> Path:
    """--refer-zip-dir 里的本地副本，用官方文件名：refcoco.zip、refcoco+.zip、refcocog.zip。"""
    return Path(local_dir) / f"{name}.zip"


def fetch_refer_members(
    ctx: LandContext,
    m,
    name: str,
    ds: dict[str, Any],
    scratch: Path,
    *,
    zip_pins: dict[str, Any],
    instances_pin: tuple[int, str],
    zip_sha256: str | None,
    local: str | Path | None,
    official_check: str,
) -> tuple[Path, Path]:
    """取一个 refer zip 里的 refs pickle 与 instances.json 到 scratch（TMPDIR），返回两者的路径。

    来源：local 给了就读本地 zip（--refer-zip-dir 的副本），否则远端 ds["zip_url"]（Range）。核对，任何一项不符都抛错、
    不往下落地：中央目录（尾部）的大小、条目数与 sha256 = zip_pins；读成员时核对 CRC32 与大小；refs 的 sha256 = P0
    （ds["refs_sha256"]）；instances.json 的字节数与 CRC32 = instances_pin。本地 zip 另算整个文件的 sha256 记进清单
    （copy_of 记它是哪个 URL 的副本），zip_sha256 钉了就比对。成员已在 scratch 且大小与 CRC32 相符时跳过（幂等）。"""
    if local is not None and not Path(local).is_file():
        raise FileNotFoundError(f"{local} 不存在：--refer-zip-dir 里要放 {name}.zip（官方文件名）")
    src = zip_source(ctx, ds["zip_url"], local)
    index = read_zip_index(src)
    check_zip_pins(m, f"refer {name}.zip", index, zip_pins)
    extra: dict[str, Any] = {}
    if isinstance(src, LocalZip):
        full = net.sha256_file(src.path)
        extra = {"copy_of": ds["zip_url"], "zip_sha256": full}
        if zip_sha256 is None:
            m.note(f"refer {name}.zip：本地副本整个文件的 sha256 为 {full}，SPEC 还没钉住（refer_zip_sha256），本次只核对了中央目录与成员")
        else:
            m.verify(f"refer {name}.zip：整个文件的 sha256 = SPEC 钉住的值", full == zip_sha256, got=full, expected=zip_sha256)
            if full != zip_sha256:
                raise zipfile.BadZipFile(f"refer {name}.zip：整个文件的 sha256 {full} ≠ 钉住的 {zip_sha256}")
    refs_tmp, inst_tmp = scratch / f"{name}_refs.p", scratch / f"{name}_instances.json"
    jobs = [MemberJob(ds["refs_member"], refs_tmp), MemberJob(ds["instances_member"], inst_tmp)]
    ok, failures = fetch_zip_members(src, index, jobs, workers=min(ctx.workers, 2), log=ctx.log)
    if failures:
        raise net.FetchError(src.where, f"refer 成员取回失败：{failures}")
    refs_sha = net.sha256_file(refs_tmp)
    m.verify(f"{name}：refs pickle 的 sha256 与 P0 记录一致", refs_sha == ds["refs_sha256"], got=refs_sha, expected=ds["refs_sha256"])
    if refs_sha != ds["refs_sha256"]:
        raise ValueError(f"{name} refs pickle sha256 不符")
    want_bytes, want_crc = instances_pin
    inst_mm = index.members[ds["instances_member"]]
    got_inst = (inst_mm.file_size, f"{inst_mm.crc:08x}")
    m.verify(f"{name}：instances.json 的字节数与 CRC32 = 钉住的值", got_inst == (want_bytes, want_crc), got=got_inst, expected=(want_bytes, want_crc))
    if got_inst != (want_bytes, want_crc):
        raise zipfile.BadZipFile(f"{name}：instances.json 的 (字节数, CRC32) {got_inst} ≠ 钉住的 {(want_bytes, want_crc)}")
    m.data["upstream"].append(
        {
            "dataset": name,
            **src.record(),
            **extra,
            "zip": index.pins(),
            "members": {job.member: {"sha256": rec["sha256"], "bytes": rec["bytes"], "crc32": rec["crc32"]} for job, rec in ok},
            "official_check": official_check,
        }
    )
    return refs_tmp, inst_tmp


def _fetch_wayback_members(ctx: LandContext, m, name: str, ds: dict[str, Any], scratch: Path) -> tuple[Path, Path]:
    """不给 --refer-zip-dir 时的原路径（DEV-0012，行为与清单格式不变）：zipfile 经 HTTPRangeFile 读 Wayback 上的 zip。"""
    zf, rf = open_remote_zip(ds["zip_url"], ctx)
    if rf.size != ds["zip_bytes"]:
        raise net.FetchError(ds["zip_url"], f"zip 大小 {rf.size} 与 P0 记录 {ds['zip_bytes']} 不符")
    refs_tmp = scratch / f"{name}_refs.p"
    inst_tmp = scratch / f"{name}_instances.json"
    if not (refs_tmp.exists() and net.sha256_file(refs_tmp) == ds["refs_sha256"]):
        extract_member(zf, ds["refs_member"], refs_tmp)
    refs_sha = net.sha256_file(refs_tmp)
    m.verify(f"{name}：refs pickle 的 sha256 与 P0 记录一致", refs_sha == ds["refs_sha256"], got=refs_sha, expected=ds["refs_sha256"])
    if refs_sha != ds["refs_sha256"]:
        raise ValueError(f"{name} refs pickle sha256 不符")
    info_inst = zf.getinfo(ds["instances_member"])
    if not (inst_tmp.exists() and inst_tmp.stat().st_size == info_inst.file_size):
        extract_member(zf, ds["instances_member"], inst_tmp)
    inst_sha = net.sha256_file(inst_tmp)
    info_refs = zf.getinfo(ds["refs_member"])
    m.data["upstream"].append(
        {
            "dataset": name,
            "url": ds["zip_url"],
            "zip_bytes": rf.size,
            "members": {
                ds["refs_member"]: {"sha256": refs_sha, "bytes": info_refs.file_size, "crc32": f"{info_refs.CRC:08x}"},
                ds["instances_member"]: {"sha256": inst_sha, "bytes": info_inst.file_size, "crc32": f"{info_inst.CRC:08x}"},
            },
            "range_bytes_fetched": rf.bytes_fetched,
            "official_check": "zip CRC32（读成员时核对）；refs sha256 = P0 记录；评测划分逐条比对 HF 镜像 jxu124",
        }
    )
    return refs_tmp, inst_tmp


def build_eval_json(name: str, ds: dict[str, Any], refs: list[dict[str, Any]], inst: dict[str, Any]) -> dict[str, Any]:
    """只保留评测划分的 refs 与它们用到的 anns、images。"""
    eval_refs = [r for r in refs if r["split"] in ds["eval_splits"]]
    anns = {a["id"]: a for a in inst["annotations"]}
    imgs = {im["id"]: im for im in inst["images"]}
    out_refs, out_anns, out_imgs = [], {}, {}
    for r in sorted(eval_refs, key=lambda r: r["ref_id"]):
        a = anns[r["ann_id"]]
        if a["image_id"] != r["image_id"]:
            raise ValueError(f"{name} ref {r['ref_id']}：ann {a['id']} 的 image_id {a['image_id']} ≠ ref 的 {r['image_id']}")
        out_refs.append(
            {
                "ref_id": r["ref_id"],
                "ann_id": r["ann_id"],
                "image_id": r["image_id"],
                "category_id": r["category_id"],
                "split": r["split"],
                "sentences": [{"sent_id": s["sent_id"], "sent": s["sent"], "raw": s["raw"]} for s in r["sentences"]],
            }
        )
        out_anns[str(a["id"])] = {k: a[k] for k in ("id", "image_id", "category_id", "bbox", "area", "iscrowd", "segmentation")}
        im = imgs[r["image_id"]]
        out_imgs[str(im["id"])] = {"id": im["id"], "file_name": image_file_name(im["id"]), "height": im["height"], "width": im["width"]}
    cats = {str(c["id"]): c["name"] for c in inst["categories"]}
    return {
        "dataset": name,
        "splitBy": ds["split_by"],
        "eval_splits": ds["eval_splits"],
        "refs": out_refs,
        "anns": out_anns,
        "images": out_imgs,
        "categories": cats,
    }


def split_counts(doc: dict[str, Any]) -> dict[str, tuple[int, int, int]]:
    out = {}
    for sp in doc["eval_splits"]:
        rs = [r for r in doc["refs"] if r["split"] == sp]
        out[sp] = (len(rs), sum(len(r["sentences"]) for r in rs), len({r["image_id"] for r in rs}))
    return out


def crosscheck_mirror(ctx: LandContext, m, name: str, ds: dict[str, Any], doc: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    """与 jxu124 镜像在评测划分上逐 ref 比对。镜像 parquet 下到 TMPDIR，比完即删。"""
    import pyarrow.parquet as pq

    mir = ds["mirror"]
    tree = hf_tree(mir["repo"], mir["rev"], "data", base=spec["hf_base"])
    ours = {r["ref_id"]: r for r in doc["refs"]}
    seen: set[int] = set()
    mismatches: list[dict[str, Any]] = []
    n_rows = 0
    scratch = ctx.scratch(SOURCE) / "mirror"
    for prefix, split in mir["files"].items():
        paths = [p for p in tree if p.split("/")[-1].startswith(prefix + "-")]
        if len(paths) != 1:
            raise ValueError(f"{mir['repo']}：{prefix} 对应的文件 {paths}")
        p = paths[0]
        dest = scratch / mir["repo"].replace("/", "__") / Path(p).name
        fetch_hf_file(ctx, m, mir["repo"], mir["rev"], p, dest, tree[p], base=spec["hf_base"])
        t = pq.read_table(dest, columns=["ref_id", "ann_id", "image_id", "category_id", "split", "sent_ids", "sentences", "raw_anns"])
        for row in t.to_pylist():
            n_rows += 1
            rid = row["ref_id"]
            seen.add(rid)
            o = ours.get(rid)
            if o is None:
                mismatches.append({"ref_id": rid, "why": "镜像有、官方评测划分没有"})
                continue
            bbox_m = json.loads(row["raw_anns"])["bbox"]
            bbox_o = doc["anns"][str(o["ann_id"])]["bbox"]
            checks = {
                "ann_id": row["ann_id"] == o["ann_id"],
                "image_id": row["image_id"] == o["image_id"],
                "category_id": row["category_id"] == o["category_id"],
                "split": row["split"] == o["split"] == split,
                "sent_ids": list(row["sent_ids"]) == [s["sent_id"] for s in o["sentences"]],
                "sentences": [s["sent"] for s in row["sentences"]] == [s["sent"] for s in o["sentences"]],
                "bbox": [float(v) for v in bbox_m] == [float(v) for v in bbox_o],
            }
            bad = [k for k, v in checks.items() if not v]
            if bad:
                mismatches.append({"ref_id": rid, "fields": bad})
        dest.unlink()
    missing = sorted(set(ours) - seen)
    rec = {
        "mirror": f"{mir['repo']}@{mir['rev']}",
        "n_mirror_rows": n_rows,
        "n_official_eval_refs": len(ours),
        "n_mismatch": len(mismatches),
        "mismatches_head": mismatches[:20],
        "n_official_missing_in_mirror": len(missing),
        "fields_compared": ["ref_id", "ann_id", "image_id", "category_id", "split", "sent_ids", "sentences[].sent", "raw_anns.bbox"],
    }
    m.verify(f"{name}：与镜像逐条比对", not mismatches and not missing, **rec)
    return rec


def land(ctx: LandContext, spec: dict[str, Any] | None = None) -> None:
    spec = {**SPEC, **(spec or {})}
    which = ctx.options.get("refcoco_datasets") or list(spec["datasets"])
    # --refer-zip-dir（DEV-0026）：refer 的三个 zip 读本地副本，不访问 web.archive.org；核对同远端，另记整个文件的 sha256。
    # 不给时走原路径，行为不变。--mirror-hf-base：与 jxu124 镜像比对时用的 HF 根地址（不给时用 SPEC 的 huggingface.co）。
    local_dir = ctx.options.get("refer_zip_dir")
    if ctx.options.get("mirror_hf_base"):
        spec["hf_base"] = ctx.options["mirror_hf_base"]
    if local_dir:
        missing = [str(refer_zip_path(local_dir, n)) for n in which if not refer_zip_path(local_dir, n).is_file()]
        if missing:
            raise FileNotFoundError(f"--refer-zip-dir {local_dir} 里缺 {missing}（官方文件名 refcoco.zip、refcoco+.zip、refcocog.zip）")
    src_dir = ctx.source_dir(SOURCE)
    m = ctx.manifest(SOURCE)
    m.data["license"] = spec["license"]
    scratch = ctx.scratch(SOURCE)
    if local_dir:
        m.note(f"refer zip 读 --refer-zip-dir 的本地副本（{local_dir}），没有访问 web.archive.org；每个 zip 的完整 sha256 见 upstream[].zip_sha256")

    # 预算：标注很小；图像按 P0 的评测并集字节估（--limit 时按均值 × 张数），留 10% 余量。
    if ctx.limit is None:
        est_images = spec["eval_union_bytes"]
    else:
        est_images = spec["mean_image_bytes"] * ctx.limit * len(which)
    est = int(est_images * 1.1) + 30_000_000
    budget_gate(ctx, m, est, {"images": est_images, "metadata": 30_000_000, "margin": "10%"})

    docs: dict[str, dict[str, Any]] = {}
    for name in which:
        ds = spec["datasets"][name]
        if local_dir:
            refs_tmp, inst_tmp = fetch_refer_members(
                ctx,
                m,
                name,
                ds,
                scratch,
                zip_pins=spec["refer_zip_pins"][name],
                instances_pin=spec["instances_pins"][name],
                zip_sha256=spec["refer_zip_sha256"][name],
                local=refer_zip_path(local_dir, name),
                official_check=LOCAL_OFFICIAL_CHECK,
            )
        else:
            refs_tmp, inst_tmp = _fetch_wayback_members(ctx, m, name, ds, scratch)
        m.df(f"{name}：成员已取到 TMPDIR")
        refs = load_refs_pickle(refs_tmp)
        inst = json.loads(inst_tmp.read_text())
        doc = build_eval_json(name, ds, refs, inst)
        del refs, inst
        counts = split_counts(doc)
        expect = {k: tuple(v) for k, v in ds["expect"].items()}
        m.verify(f"{name}：评测划分计数（refs, sentences, images）= P0", counts == expect, got=counts, expected=expect)
        if counts != expect:
            raise ValueError(f"{name} 计数不符：{counts} vs {expect}")
        if not ctx.options.get("skip_mirror_check"):
            crosscheck_mirror(ctx, m, name, ds, doc, spec)
        write_json(m, src_dir, src_dir / f"{name}_{ds['split_by']}.json", doc)
        docs[name] = doc
        refs_tmp.unlink()
        inst_tmp.unlink()
        m.df(f"{name}：评测标注已写出")

    # 图像：评测划分用到的全部 train2014 图像（--limit 时每个数据集只取前 limit 句所在的图）
    want: dict[int, None] = {}
    for name, doc in docs.items():
        if ctx.limit is None:
            ids = [r["image_id"] for r in doc["refs"]]
        else:
            ids = []
            for r in doc["refs"]:
                if r["split"] == "val":
                    ids.append(r["image_id"])
                if len(ids) >= ctx.limit:
                    break
        for i in ids:
            want.setdefault(i, None)
    image_ids = sorted(want)
    jobs = [(spec["image_url"].format(id=i), f"images/{image_file_name(i)}", {"image_key": f"coco:{i}"}) for i in image_ids]
    download_images(ctx, m, src_dir, jobs, "COCO train2014 图像")

    # 核对图像尺寸与 instances 里记的一致
    dims = {}
    for doc in docs.values():
        for k, im in doc["images"].items():
            dims[im["file_name"]] = (im["width"], im["height"])
    bad = []
    for rel, rec in m.data["files"].items():
        if rel.startswith("images/"):
            fn = rel.split("/", 1)[1]
            if dims.get(fn) != (rec["width"], rec["height"]):
                bad.append({"file": fn, "file_wh": (rec["width"], rec["height"]), "instances_wh": dims.get(fn)})
    m.verify("图像宽高与 instances.json 一致", not bad, n_bad=len(bad), bad_head=bad[:10])

    m.data["selection"] = {
        "eval_splits": {n: spec["datasets"][n]["eval_splits"] for n in which},
        "images": "评测划分用到的全部 COCO train2014 图像（三个数据集的并集）" if ctx.limit is None else f"--limit {ctx.limit}：每个数据集 val 前 {ctx.limit} 个 ref 的图",
        "n_images_requested": len(image_ids),
    }
    m.data["stats"] = {n: {"split_counts": split_counts(d), "n_images": len(d["images"]), "n_anns": len(d["anns"])} for n, d in docs.items()}
    m.data["dedupe_facts"] = {
        "image_key_scheme": "coco:<image_id>（COCO 2014/2017 共用 id）",
        "known": "与 COCO val2017 的图像交集为 0（data_sources §3）；umd train ∩ unc 评测 = 1,860 张，unc train ∩ umd 评测 = 1,891 张（D11-1，训练集构造时处理）",
    }
    m.write()
