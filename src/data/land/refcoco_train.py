"""RefCOCO 系列训练图与 train 划分标注的落地（DEV-0022 任务书 §1；D11-1；data_sources A §1）。

D11-1：训练图 = RefCOCO(unc)、RefCOCO+(unc)、RefCOCOg(umd) 三者 train 划分的图像并集，减去三者任一评测划分
（unc 的 val/testA/testB、umd 的 val/test）里出现过的全部图像（按 COCO image_id）。减法必须读三份完整的 refs pickle，
所以标注部分不受 --limit 影响；--limit 只限制下载的图像张数。

来源：
- refs 与 mask：复用 E1 钉住的 refer 存档（src/data/land/refcoco.SPEC：2022-04-13 的 Wayback 快照），用 Range 只取
  refs(unc|umd).p 与 instances.json；核对 zip 中央目录 sha256（本模块 SPEC 钉住）、成员 CRC32、refs 的 sha256（P0）。
- 图像：COCO 官方 train2014.zip（13.5 GB），用 Range 只取需要的成员（stored，不压缩），逐个核对 CRC32；中央目录
  （尾部 9.9 MB）的 sha256 钉在 SPEC 里。所有 24,407 张训练图都在 zip 里（开发机 2026-10-02 用 P0 的 image id 清单核对）。
- 国内网络连不上 web.archive.org 或 images.cocodataset.org 时：--refer-zip-dir 指向放着 refcoco.zip、refcoco+.zip、
  refcocog.zip 的目录，--local-zip 指向本地的 train2014.zip；同样核对中央目录 sha256 与成员 CRC32。

落地后的布局（<root>/refcoco_train/）：
- <dataset>_<splitBy>_train.json：与 E1 的评测文件同一格式（src/data/land/refcoco.build_eval_json），splits=["train"]，
  只含 D11-1 剔除之后的图上的 train refs；
- d11_1.json：D11-1 的并集、交叉表、保留与剔除的 image id（剔除的逐张写明出现在哪个评测划分）；
- images/COCO_train2014_<id:012d>.jpg（清单里每张记 image_key=coco:<id>、split=train、宽高与感知哈希）；
- MANIFEST.json。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.data.land import net, refcoco
from src.data.land.common import (
    LandContext,
    MemberJob,
    budget_gate,
    check_zip_pins,
    fetch_zip_members,
    image_meta,
    read_zip_index,
    write_json,
    zip_source,
)
from src.data.subsets import population_digest

SOURCE = "refcoco_train"
SCRIPT = "scripts/data/land_train.py"
DATASETS = ("refcoco", "refcoco+", "refcocog")
TRAIN_SPLIT = "train"

SPEC: dict[str, Any] = {
    # refer 存档：URL、refs 的 sha256、评测划分沿用 E1（refcoco.SPEC），这里只补中央目录与 instances.json 的钉住值
    # （开发机 2026-10-02 用 Range 读中央目录，每个 zip 几百字节）。
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
    # train 划分（refs, sentences, images），P0 refs_stats.json（data_sources A §1.3）
    "expect_train": {"refcoco": (42_404, 120_624, 16_994), "refcoco+": (42_278, 120_191, 16_992), "refcocog": (42_226, 80_512, 21_899)},
    # D11-1 的结果：开发机用 P0 的 refs_image_ids.json（官方 pickle 逐条统计）算出，落地时用官方 pickle 重算并比对
    "expect_d11_1": {
        "n_train_union": 28_158,
        "n_eval_union": 6_549,
        "n_removed": 3_751,
        "n_kept": 24_407,
        "kept_digest": "9c0c94335097ba432c3c6813822733d55aa7e478e35f2403195082000a668f15",
        "removed_digest": "7c4c87019f6fd64874cd76f75cd520aa2fe688b4df5db3141b25c7e9ea8d632c",
        "cross": {"refcocog train ∩ refcoco eval": 1_860, "refcoco train ∩ refcocog eval": 1_891, "refcoco+ train ∩ refcocog eval": 1_891, "refcocog train ∩ refcoco+ eval": 1_860},
    },
    "coco_zip_url": "http://images.cocodataset.org/zips/train2014.zip",
    # 开发机 2026-10-02：HEAD Last-Modified 2018-07-11、ETag "289af3433b84b5b8622626b10146c1bc-1611"；82,784 条目
    # （82,783 张图 + 1 个目录），全部 stored；尾部用 Range 读取（9,866,071 B）。
    "coco_zip_pins": {"size": 13_510_573_713, "n_entries": 82_784, "start_dir": 13_500_707_642, "tail_bytes": 9_866_071, "tail_sha256": "7b2e21314b40712477e485052bd885142507f7929c658228e4abe20a5894549e"},
    "coco_member": "train2014/COCO_train2014_{id:012d}.jpg",
    # 保留的 24,407 张图按 HEAD 与 zip 中央目录都是 4,146,591,678 B（data_sources A §1.5 的"去掉评测图像后"）
    "kept_bytes": 4_146_591_678,
    "mean_image_bytes": 169_660,
    "annotations_estimate_bytes": 250_000_000,
    "zip_gap": 0,  # 训练图在 train2014.zip 里是乱序的（不按文件名），只合并紧挨着的成员
    "license": refcoco.SPEC["license"],
}


def train_eval_images(refs: Iterable[Mapping[str, Any]], eval_splits: Iterable[str]) -> tuple[set[int], set[int]]:
    """一个数据集的 (train 划分图像, 评测划分图像)，按 image_id。"""
    ev = set(eval_splits)
    train, evi = set(), set()
    for r in refs:
        if r["split"] == TRAIN_SPLIT:
            train.add(int(r["image_id"]))
        elif r["split"] in ev:
            evi.add(int(r["image_id"]))
    return train, evi


def d11_1_split(train_by_ds: Mapping[str, set[int]], eval_by_ds: Mapping[str, Mapping[str, set[int]]]) -> dict[str, Any]:
    """D11-1 的减法。train_by_ds：{数据集: train 图像}；eval_by_ds：{数据集: {评测划分: 图像}}。

    返回保留的图（升序）、剔除的图与它们出现的评测划分（"数据集/划分"）、并集大小与两两交叉表（A train ∩ B eval）。"""
    train_union = set().union(*train_by_ds.values()) if train_by_ds else set()
    where: dict[int, list[str]] = {}
    for ds, splits in eval_by_ds.items():
        for sp, ids in splits.items():
            for i in ids:
                where.setdefault(i, []).append(f"{ds}/{sp}")
    eval_union = set(where)
    kept = sorted(train_union - eval_union)
    removed = sorted(train_union & eval_union)
    cross = {}
    for a, tr in train_by_ds.items():
        for b, splits in eval_by_ds.items():
            cross[f"{a} train ∩ {b} eval"] = len(tr & set().union(*splits.values()))
    return {
        "n_train_union": len(train_union),
        "n_eval_union": len(eval_union),
        "n_removed": len(removed),
        "n_kept": len(kept),
        "kept": kept,
        "removed": [{"image_id": i, "in_eval": sorted(where[i])} for i in removed],
        "cross": cross,
        "kept_digest": population_digest(str(i) for i in kept),
        "removed_digest": population_digest(str(i) for i in removed),
    }


def build_train_doc(name: str, ds: dict[str, Any], refs: list[dict[str, Any]], inst: dict[str, Any], kept: set[int]) -> dict[str, Any]:
    """train 划分里落在 D11-1 保留图上的 refs，及它们用到的 anns、images（格式同 E1 的评测文件）。"""
    keep_refs = [r for r in refs if r["split"] == TRAIN_SPLIT and int(r["image_id"]) in kept]
    doc = refcoco.build_eval_json(name, {**ds, "eval_splits": [TRAIN_SPLIT]}, keep_refs, inst)
    doc["splits"] = doc.pop("eval_splits")
    doc["d11_1"] = "只含 D11-1 剔除之后的图（三者 train 图像并集减去三者任一评测划分的图像，按 COCO image_id）"
    return doc


def _counts(refs: Iterable[Mapping[str, Any]]) -> tuple[int, int, int]:
    rs = list(refs)
    return len(rs), sum(len(r["sentences"]) for r in rs), len({r["image_id"] for r in rs})


def _fetch_refer_members(ctx: LandContext, m, name: str, ds: dict[str, Any], spec: dict[str, Any], scratch: Path) -> tuple[Path, Path]:
    """取一个 refer zip 里的 refs pickle 与 instances.json 到 TMPDIR，核对中央目录、CRC32、refs 的 sha256。"""
    local_dir = ctx.options.get("refer_zip_dir")
    local = Path(local_dir) / f"{name}.zip" if local_dir else None  # 官方文件名：refcoco.zip、refcoco+.zip、refcocog.zip
    src = zip_source(ctx, ds["zip_url"], local)
    index = read_zip_index(src)
    check_zip_pins(m, f"refer {name}.zip", index, spec["refer_zip_pins"][name])
    refs_tmp, inst_tmp = scratch / f"{name}_refs.p", scratch / f"{name}_instances.json"
    jobs = [MemberJob(ds["refs_member"], refs_tmp), MemberJob(ds["instances_member"], inst_tmp)]
    ok, failures = fetch_zip_members(src, index, jobs, workers=min(ctx.workers, 2), log=ctx.log)
    if failures:
        raise net.FetchError(src.where, f"refer 成员取回失败：{failures}")
    refs_sha = net.sha256_file(refs_tmp)
    m.verify(f"{name}：refs pickle 的 sha256 与 P0 记录一致", refs_sha == ds["refs_sha256"], got=refs_sha, expected=ds["refs_sha256"])
    if refs_sha != ds["refs_sha256"]:
        raise ValueError(f"{name} refs pickle sha256 不符")
    want_bytes, want_crc = spec["instances_pins"][name]
    inst_mm = index.members[ds["instances_member"]]
    m.verify(f"{name}：instances.json 的字节数与 CRC32 = 钉住的值", (inst_mm.file_size, f"{inst_mm.crc:08x}") == (want_bytes, want_crc), got=(inst_mm.file_size, f"{inst_mm.crc:08x}"), expected=(want_bytes, want_crc))
    m.data["upstream"].append(
        {
            "dataset": name,
            **src.record(),
            "zip": index.pins(),
            "members": {job.member: {"sha256": rec["sha256"], "bytes": rec["bytes"], "crc32": rec["crc32"]} for job, rec in ok},
            "official_check": "E1 钉住的 refer 存档（Wayback 2022-04-13 快照，或 --refer-zip-dir 的本地副本）；中央目录 sha256 = SPEC；成员 CRC32；refs sha256 = P0",
        }
    )
    return refs_tmp, inst_tmp


def land(ctx: LandContext, spec: dict[str, Any] | None = None) -> None:
    spec = {**SPEC, **(spec or {})}
    rspec = {**refcoco.SPEC, **spec.get("refcoco_spec", {})}
    src_dir = ctx.source_dir(SOURCE)
    m = ctx.manifest(SOURCE)
    m.data["script"]["path"] = SCRIPT
    m.data["license"] = spec["license"]
    scratch = ctx.scratch(SOURCE)

    n_img = spec["expect_d11_1"]["n_kept"] if ctx.limit is None else ctx.limit
    est_images = spec["kept_bytes"] if ctx.limit is None else spec["mean_image_bytes"] * ctx.limit
    est = int(est_images * 1.05) + spec["annotations_estimate_bytes"]
    budget_gate(ctx, m, est, {"n_images": n_img, "images": est_images, "annotations": spec["annotations_estimate_bytes"], "margin": "5%", "tmpdir_peak_bytes": 160_000_000})

    # 1) 三份 refs 与 instances：先算 D11-1（要全量 pickle），再写 train 文件
    refs_by: dict[str, list[dict[str, Any]]] = {}
    train_by: dict[str, set[int]] = {}
    eval_by: dict[str, dict[str, set[int]]] = {}
    inst_paths: dict[str, Path] = {}
    for name in DATASETS:
        ds = rspec["datasets"][name]
        refs_tmp, inst_tmp = _fetch_refer_members(ctx, m, name, ds, spec, scratch)
        refs = refcoco.load_refs_pickle(refs_tmp)
        refs_tmp.unlink()
        refs_by[name] = refs
        train_by[name], _ = train_eval_images(refs, ds["eval_splits"])
        eval_by[name] = {sp: {int(r["image_id"]) for r in refs if r["split"] == sp} for sp in ds["eval_splits"]}
        got = _counts(r for r in refs if r["split"] == TRAIN_SPLIT)
        want = tuple(spec["expect_train"][name])
        m.verify(f"{name}：train 划分计数（refs, sentences, images）= P0", got == want, got=got, expected=want)
        if got != want:
            raise ValueError(f"{name} train 计数 {got} ≠ {want}")
        inst_paths[name] = inst_tmp
        m.df(f"{name}：refer 成员已取到 TMPDIR")

    d = d11_1_split(train_by, eval_by)
    exp = spec["expect_d11_1"]
    got = {k: d[k] for k in ("n_train_union", "n_eval_union", "n_removed", "n_kept", "kept_digest", "removed_digest")}
    got_cross = {k: d["cross"][k] for k in exp["cross"]}
    ok = got == {k: exp[k] for k in got} and got_cross == exp["cross"]
    m.verify("D11-1：并集、剔除数、保留数、id 指纹与交叉表 = 开发机用 P0 清单算出的值", ok, got={**got, "cross": got_cross}, expected=exp)
    if not ok:
        raise ValueError(f"D11-1 结果与钉住的不符：{got} {got_cross}")
    kept = set(d["kept"])
    write_json(m, src_dir, src_dir / "d11_1.json", {"rule": "D11-1：RefCOCO(unc)、RefCOCO+(unc)、RefCOCOg(umd) 三者 train 划分图像的并集，减去三者任一评测划分（unc val/testA/testB、umd val/test）里出现过的图像，按 COCO image_id", **d})

    stats: dict[str, Any] = {}
    dims: dict[int, tuple[int, int]] = {}
    for name in DATASETS:
        ds = rspec["datasets"][name]
        inst = json.loads(inst_paths[name].read_text())
        doc = build_train_doc(name, ds, refs_by[name], inst, kept)
        del inst
        inst_paths[name].unlink()
        before = _counts(r for r in refs_by[name] if r["split"] == TRAIN_SPLIT)
        after = _counts(doc["refs"])
        stats[name] = {"train_before_d11_1": dict(zip(("refs", "sentences", "images"), before)), "train_after_d11_1": dict(zip(("refs", "sentences", "images"), after)), "n_anns": len(doc["anns"])}
        for k, im in doc["images"].items():
            dims[int(k)] = (im["width"], im["height"])
        write_json(m, src_dir, src_dir / f"{name}_{ds['split_by']}_train.json", doc)
        refs_by[name] = []
        m.df(f"{name}：train 标注已写出")
    missing_dims = sorted(kept - set(dims))
    m.verify("每张保留的图都在某个 train 文件里有宽高", not missing_dims, n_missing=len(missing_dims), head=missing_dims[:10])

    # 2) 图像：train2014.zip 的成员
    ids = d["kept"] if ctx.limit is None else d["kept"][: ctx.limit]
    src = zip_source(ctx, spec["coco_zip_url"], ctx.options.get("local_zip"))
    index = read_zip_index(src)
    check_zip_pins(m, "COCO train2014.zip", index, spec["coco_zip_pins"])
    jobs = [MemberJob(spec["coco_member"].format(id=i), src_dir / "images" / refcoco.image_file_name(i), {"image_id": i}) for i in ids]
    missing = [j.member for j in jobs if j.member not in index.members]
    m.verify("每张要取的训练图都在 train2014.zip 里", not missing, n_missing=len(missing), head=missing[:10])
    _land_images(ctx, m, src_dir, src, index, jobs, dims, gap=spec["zip_gap"])
    m.data["upstream"].append({"role": "COCO train2014 图像", **src.record(), "zip": index.pins(), "official_check": "官方 images.cocodataset.org/zips/train2014.zip（或 --local-zip 的本地副本）；中央目录 sha256 = SPEC；逐个成员核对 CRC32 与大小；宽高与 instances.json 一致"})

    m.data["selection"] = {
        "rule": "D11-1（见 d11_1.json）",
        "images": "全部保留的图" if ctx.limit is None else f"--limit {ctx.limit}：保留的图按 image id 升序的前 {ctx.limit} 张（标注照常全量写出）",
        "n_images_requested": len(ids),
    }
    m.data["stats"] = {
        "d11_1": {k: d[k] for k in ("n_train_union", "n_eval_union", "n_removed", "n_kept")} | {"cross": d["cross"]},
        "per_dataset": stats,
        "n_images_landed": sum(1 for r in m.data["files"].values() if r.get("kind") == "image"),
    }
    m.data["dedupe_facts"] = {
        "image_key_scheme": "coco:<image_id>（COCO 2014/2017 共用 id）",
        "split_field": "清单里每张图记 split=train（去重报告据此把本源归为训练集 refcoco_train/train）",
        "known": "D11-1 已剔除与 RefCOCO 三者评测划分重合的图；COCO train2014 与 val2017 的 image_id 不相交（data_sources §3），去重报告按 id 与感知哈希再核对一遍",
    }
    m.df("完成")
    m.write()


def _land_images(ctx: LandContext, m, src_dir: Path, src, index, jobs: list[MemberJob], dims: dict[int, tuple[int, int]], *, gap: int) -> None:
    done = [0]

    def process(job: MemberJob, path: Path, rec: dict[str, Any]) -> dict[str, Any]:
        rel = path.relative_to(src_dir).as_posix()
        prev = m.known(rel)
        if rec["skipped"] and prev and prev.get("sha256") == rec["sha256"] and "phash" in prev:
            meta = {k: prev[k] for k in ("kind", "width", "height", "format", "phash")}
        else:
            meta = image_meta(path)
        want = dims.get(job.meta["image_id"])
        if want is not None and (meta["width"], meta["height"]) != want:
            raise ValueError(f"{rel} 的宽高 {(meta['width'], meta['height'])} ≠ instances.json 的 {want}")
        return meta

    def on_done(job: MemberJob, rec: dict[str, Any]) -> None:
        rel = job.dest.relative_to(src_dir).as_posix()
        extra = {k: v for k, v in rec.items() if k not in ("sha256", "bytes", "skipped")}
        m.add_file(rel, sha256=rec["sha256"], nbytes=rec["bytes"], image_key=f"coco:{job.meta['image_id']}", split=TRAIN_SPLIT, **extra)
        done[0] += 1
        if done[0] % 2000 == 0:
            m.df(f"train2014 图像（已完成 {done[0]}）")

    ok, failures = fetch_zip_members(src, index, jobs, workers=ctx.workers, gap=gap, process=process, on_done=on_done, log=ctx.log)
    for f in failures:
        m.fail(step="train2014 图像", **f)
        ctx.log(f"[失败] {f['member']}: {f['error']}")
    ctx.log(f"[train2014 图像] 成功 {len(ok)}（其中跳过 {sum(r['skipped'] for _, r in ok)}），失败 {len(failures)}")
    m.df("train2014 图像")

