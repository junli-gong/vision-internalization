"""从 HF 落地的四个基准：CountBench、BLINK Relative_Depth（val）、CV-Bench（test）、EmbSpatial-Bench（test）。
DEV-0012 任务书 §1.4、§1.6、§1.7；data_sources A §3、B §2、C §2、C §3。

共同做法：
- revision 一律钉 P0 记录的 commit sha；文件按 HF tree API 给的 LFS sha256（非 LFS 文件按 git blob sha1）核对。
- 原始文件（parquet / json）下到 TMPDIR，图像字节**原样**落成文件（不转码，扩展名按文件头），其余字段原样写成
  jsonl；原始文件的 sha256 记进清单后删除。
- 每张图记宽高与感知哈希（加载时去重用）。

落地后的布局（<root>/<源>/）：<split>.jsonl（每行一题，含 image 的相对路径与 sha256）、images/…、MANIFEST.json。
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from src.data.land import net
from src.data.land.common import (
    LandContext,
    budget_gate,
    fetch_hf_file,
    hf_tree,
    save_image_bytes,
    write_jsonl,
)
from src.data.records import parse_paren_letter

HF_BASE = "https://huggingface.co"
PROJECT_TERMS = "D28-9：限非商业研究，不再分发任何数据或由数据生成的图像"


def _parquet_rows(path: Path, columns: list[str] | None = None) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    return pq.read_table(path, columns=columns).to_pylist()


def _cleanup(*paths: Path) -> None:
    for p in paths:
        p.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# CountBench（D11-4：用 nielsr/countbench 的打包版，491 张有图；不抓原始 URL）
# ---------------------------------------------------------------------------

COUNTBENCH = {
    "source": "countbench",
    "repo": "nielsr/countbench",
    "rev": "e32ce2541299d12755894e1d487f6dd75bb0176c",
    "dir": "data",
    # 官方发布（只有 URL）：用来逐条核对 (image_url, text, number) 与顺序
    "official_json_url": "https://raw.githubusercontent.com/teaching-clip-to-count/teaching-clip-to-count.github.io/main/CountBench.json",
    "official_json_sha256": "2b1586d1b945243429d54afd9f719c787ec56bdb0e66dfa8a1f151ad58c164b7",
    "estimate_bytes": 23_353_242 + 2_000_000,
    "license": {
        "text": "未声明。官方仓库 teaching-clip-to-count/teaching-clip-to-count.github.io 的 GitHub license 为 null，项目页没有许可声明；HF nielsr/countbench 卡片也没有 license。图像取自 LAION-400M（其许可未核实）。data_sources A §3",
        "project_terms": PROJECT_TERMS,
    },
}


def land_countbench(ctx: LandContext, spec: dict[str, Any] | None = None) -> None:
    spec = {**COUNTBENCH, **(spec or {}), "hf_base": (spec or {}).get("hf_base", HF_BASE)}
    name = spec["source"]
    src_dir = ctx.source_dir(name)
    m = ctx.manifest(name)
    m.data["license"] = spec["license"]
    budget_gate(ctx, m, spec["estimate_bytes"])
    tree = hf_tree(spec["repo"], spec["rev"], spec["dir"], base=spec["hf_base"])
    parquets = [p for p in tree if p.endswith(".parquet")]
    if len(parquets) != 1:
        raise ValueError(f"{spec['repo']}: parquet 文件 {parquets}")
    pth = parquets[0]
    tmp = ctx.scratch(name) / Path(pth).name
    rec = fetch_hf_file(ctx, m, spec["repo"], spec["rev"], pth, tmp, tree[pth], base=spec["hf_base"])
    m.data["upstream"].append({**rec, "official_check": "LFS sha256；(image_url, text, number) 与官方 CountBench.json 逐条同序比对"})
    rows = _parquet_rows(tmp)

    official = None
    if spec.get("official_json_url"):
        raw = net.get_bytes(spec["official_json_url"])
        sha = hashlib.sha256(raw).hexdigest()
        m.verify("官方 CountBench.json 的 sha256 = P0 记录", sha == spec["official_json_sha256"], got=sha, expected=spec["official_json_sha256"])
        official = json.loads(raw)
        m.data["upstream"].append({"url": spec["official_json_url"], "sha256": sha, "bytes": len(raw), "role": "官方发布（URL 清单），只用于核对"})
    if official is not None:
        same = len(official) == len(rows) and all(
            (o["image_url"], o["text"], o["number"]) == (r["image_url"], r["text"], r["number"]) for o, r in zip(official, rows)
        )
        m.verify("HF 打包版与官方 CountBench.json 逐条同序一致（image_url, text, number）", same, n_official=len(official), n_hf=len(rows))

    out = []
    n_img = 0
    with_img = [i for i, r in enumerate(rows) if r["image"] is not None]
    take = set(with_img if ctx.limit is None else with_img[: ctx.limit])
    for i, r in enumerate(rows):
        img = r["image"]
        rec_row = {"row": i, "image_url": r["image_url"], "text": r["text"], "number": int(r["number"]), "has_image": img is not None}
        if img is not None:
            n_img += 1
            if i in take:
                frec = save_image_bytes(m, src_dir, f"images/{i:03d}", img["bytes"], image_key=f"countbench:{i}")
                rec_row["image"] = frec["path"]
                rec_row["image_sha256"] = frec["sha256"]
        out.append(rec_row)
    write_jsonl(m, src_dir, src_dir / "countbench.jsonl", out)
    _cleanup(tmp)
    hist = {}
    for r in out:
        if r["has_image"]:
            hist[r["number"]] = hist.get(r["number"], 0) + 1
    m.data["stats"] = {"n_rows": len(out), "n_with_image": n_img, "count_hist_with_image": dict(sorted(hist.items())), "n_landed_images": len(take)}
    m.data["selection"] = {"rule": "D11-4：HF nielsr/countbench 打包版中 image 非空的行（报告注明 491/540）；id = 官方 CountBench.json 的 0 起行号"}
    m.data["dedupe_facts"] = {"image_key_scheme": "countbench:<row>", "known": "图像来自 LAION-400M，无 COCO/NYU/ADE 的 id 可对；跨集只能靠感知哈希"}
    m.df("完成")
    m.write()


# ---------------------------------------------------------------------------
# BLINK Relative_Depth（D11-3：只用 val 124）
# ---------------------------------------------------------------------------

BLINK = {
    "source": "blink_reldepth",
    "repo": "BLINK-Benchmark/BLINK",
    "rev": "a3666eb249237ba3d5eca8db21176cc47967e040",
    "path": "Relative_Depth/val-00000-of-00001.parquet",
    "estimate_bytes": 4_604_789 + 1_000_000,
    "license": {
        "text": "Apache-2.0（HF 卡片与 GitHub LICENSE）。卡片 Disclaimers 原文：\"Blink makes use of data from existing image datasets … if you are the copyright owner of any image … please contact us\"。图像来自 Depth in the Wild，DIW 的许可未核实。data_sources B §2",
        "project_terms": PROJECT_TERMS,
    },
}


def land_blink(ctx: LandContext, spec: dict[str, Any] | None = None) -> None:
    spec = {**BLINK, **(spec or {}), "hf_base": (spec or {}).get("hf_base", HF_BASE)}
    name = spec["source"]
    src_dir = ctx.source_dir(name)
    m = ctx.manifest(name)
    m.data["license"] = spec["license"]
    budget_gate(ctx, m, spec["estimate_bytes"])
    tree = hf_tree(spec["repo"], spec["rev"], spec["path"].rsplit("/", 1)[0], base=spec["hf_base"])
    tmp = ctx.scratch(name) / Path(spec["path"]).name
    rec = fetch_hf_file(ctx, m, spec["repo"], spec["rev"], spec["path"], tmp, tree[spec["path"]], base=spec["hf_base"])
    m.data["upstream"].append({**rec, "official_check": "官方 HF 仓库，LFS sha256"})
    rows = _parquet_rows(tmp)
    rows = sorted(rows, key=lambda r: int(r["idx"].rsplit("_", 1)[1]))
    out = []
    for k, r in enumerate(rows):
        for extra in ("image_2", "image_3", "image_4"):
            if r.get(extra) is not None:
                raise ValueError(f"{r['idx']}: {extra} 非空，Relative_Depth 应只有一张图")
        row = {key: r[key] for key in ("idx", "question", "sub_task", "choices", "answer", "prompt", "explanation")}
        row["answer_letter"] = parse_paren_letter(r["answer"])
        if ctx.limit is None or k < ctx.limit:
            frec = save_image_bytes(m, src_dir, f"images/{r['idx']}", r["image_1"]["bytes"], image_key=f"blink:{r['idx']}")
            row["image"] = frec["path"]
            row["image_sha256"] = frec["sha256"]
        out.append(row)
    write_jsonl(m, src_dir, src_dir / "val.jsonl", out)
    _cleanup(tmp)
    hist = {}
    for r in out:
        hist[r["answer_letter"]] = hist.get(r["answer_letter"], 0) + 1
    m.data["stats"] = {"n_rows": len(out), "answer_hist": hist}
    m.data["selection"] = {"rule": "D11-3：Relative_Depth 的 val 全部 124 题（test 答案隐藏，官方评测渠道已于 2026-01-31 关闭）"}
    m.data["dedupe_facts"] = {"image_key_scheme": "blink:<idx>", "known": "图像来自 Depth in the Wild，无 COCO/NYU/ADE 的 id 可对"}
    m.df("完成")
    m.write()


# ---------------------------------------------------------------------------
# CV-Bench（test 2,638）
# ---------------------------------------------------------------------------

CVBENCH = {
    "source": "cvbench",
    "repo": "nyu-visionx/CV-Bench",
    "rev": "bc284db50d036958861cb60cdd7b77612052ce0d",
    "parquets": ["test_2d.parquet", "test_3d.parquet"],
    "jsonls": ["test_2d.jsonl", "test_3d.jsonl"],
    "estimate_bytes": 404_808_364 + 5_000_000,
    "license": {
        "text": "apache-2.0（HF 卡片）。卡片未另外声明上游图像许可（ADE20K、COCO、Omni3D）；Omni3D 标注为 CC BY-NC 4.0（Omni3D_json.zip 的 LICENSE.md）。data_sources C §2.1",
        "project_terms": PROJECT_TERMS,
    },
}

_NYU_RE = re.compile(r"NYU(\d{4})")
META_COLS = ["idx", "type", "task", "question", "choices", "answer", "prompt", "filename", "source", "source_dataset", "source_filename", "target_class", "target_size", "bbox"]


def cvbench_image_key(row: dict[str, Any]) -> str:
    """CV-Bench 一行的源图像规范键（D11-7 去重用）。"""
    sd, fn = row["source_dataset"], row["source_filename"]
    if sd == "COCO 2017 Validation Set":
        return f"coco:{int(Path(fn).stem)}"
    if sd == "ADE20K Validation Set":
        return f"ade20k:{Path(fn).stem}"
    if sd == "Omni3D_SUNRGBD":
        mm = _NYU_RE.search(fn)
        if "/NYUdata/" in fn and mm:
            return f"nyuv2:{int(mm.group(1))}"
        return f"sunrgbd:{fn}"
    if sd == "Omni3D_Hypersim":
        return f"hypersim:{fn}"
    if sd == "Omni3D_nuScenes":
        return f"nuscenes:{Path(fn).name}"
    raise ValueError(f"未知的 source_dataset：{sd!r}")


def land_cvbench(ctx: LandContext, spec: dict[str, Any] | None = None) -> None:
    spec = {**CVBENCH, **(spec or {}), "hf_base": (spec or {}).get("hf_base", HF_BASE)}
    name = spec["source"]
    src_dir = ctx.source_dir(name)
    m = ctx.manifest(name)
    m.data["license"] = spec["license"]
    est = spec["estimate_bytes"] if ctx.limit is None else 5_000_000
    budget_gate(ctx, m, est, {"tmpdir_peak_bytes": 404_808_364})
    tree = hf_tree(spec["repo"], spec["rev"], "", base=spec["hf_base"])
    scratch = ctx.scratch(name)

    # jsonl（官方元数据，小）：逐条核对 parquet 的非图像列
    jsonl_rows: dict[str, dict[str, Any]] = {}
    for p in spec["jsonls"]:
        tmp = scratch / p
        rec = fetch_hf_file(ctx, m, spec["repo"], spec["rev"], p, tmp, tree[p], base=spec["hf_base"])
        m.data["upstream"].append({**rec, "role": "元数据，用于逐条核对"})
        for line in tmp.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                jsonl_rows[r["filename"]] = r
        tmp.unlink()

    # 每行的元数据以 jsonl 为准；idx 与图像取自 parquet，其余非图像列与 jsonl 逐条比对。
    # --limit（开发机小样本）：不下 405 MB 的 parquet，用 Range 只读每个 parquet 的第一个 row group，取前 limit 行的图像。
    import pyarrow.parquet as pq

    from src.data.land.common import hf_resolve_url

    by_fn: dict[str, dict[str, Any]] = {}
    for fn, j in jsonl_rows.items():
        meta = {k: j.get(k) for k in META_COLS}
        meta["answer_letter"] = parse_paren_letter(meta["answer"])
        meta["image_key"] = cvbench_image_key(meta)
        by_fn[fn] = meta
    stems = [str(Path(fn).with_suffix("")) for fn in by_fn]
    if len(set(stems)) != len(stems):  # 图像按 filename 去掉扩展名落盘（扩展名按文件头），去掉后必须仍唯一
        raise ValueError("CV-Bench 的 filename 去掉扩展名后有重复")
    mism = []
    n_parquet_rows = 0
    for p in spec["parquets"]:
        if ctx.limit is None:
            tmp = scratch / p
            rec = fetch_hf_file(ctx, m, spec["repo"], spec["rev"], p, tmp, tree[p], base=spec["hf_base"])
            m.data["upstream"].append({**rec, "official_check": "官方 HF 仓库，LFS sha256；非图像列与仓库内 jsonl 逐条比对"})
            m.df(f"{p} 已下载到 TMPDIR")
            pf = pq.ParquetFile(tmp)
            batches = (b.to_pylist() for b in pf.iter_batches(batch_size=64))
        else:
            tmp = None
            url = hf_resolve_url(spec["repo"], spec["rev"], p, base=spec["hf_base"])
            rf = net.HTTPRangeFile(url, size=tree[p]["size"], block_size=64 << 10, retries=ctx.retries, backoff=ctx.backoff)
            pf = pq.ParquetFile(rf)
            # idx 只在 parquet 里有（jsonl 没有这一列）：先读全部行的 idx 与 filename 两列（很小），再读第一个 row group 取图
            for r in pf.read(columns=["idx", "filename"]).to_pylist():
                if r["filename"] in by_fn:
                    by_fn[r["filename"]]["idx"] = r["idx"]
            batches = iter([pf.read_row_group(0).to_pylist()[: ctx.limit]])
            m.data["upstream"].append({"url": url, "path": p, "revision": spec["rev"], "lfs_sha256": tree[p]["sha256"], "range_bytes_fetched": rf.bytes_fetched, "official_check": f"--limit {ctx.limit}：Range 读 idx/filename 两列与第一个 row group（未下整个文件，未核对整文件 sha256）"})
        for rows in batches:
            for r in rows:
                n_parquet_rows += 1
                meta = by_fn.get(r["filename"])
                if meta is None or any(meta[k] != r.get(k) for k in ("type", "task", "question", "choices", "answer", "source", "source_dataset", "source_filename")):
                    mism.append(r["filename"])
                    continue
                meta["idx"] = r["idx"]  # 规范 id：parquet 的 idx（jsonl 没有）
                stem = str(Path(r["filename"]).with_suffix(""))
                frec = save_image_bytes(m, src_dir, f"images/{stem}", r["image"]["bytes"], image_key=meta["image_key"], idx=meta["idx"])
                meta["image"] = frec["path"]
                meta["image_sha256"] = frec["sha256"]
        if tmp is not None:
            tmp.unlink()
            m.df(f"{p} 已解包并删除")
    out = list(by_fn.values())
    if ctx.limit is None:
        missing_img = [r["filename"] for r in out if "image" not in r]
        m.verify("每一行都从 parquet 取到了图像", not missing_img, n_missing=len(missing_img), head=missing_img[:10])
    idxs = [r["idx"] for r in out]
    out.sort(key=lambda r: (r["idx"] is None, r["idx"] if r["idx"] is not None else 0, r["filename"]))
    m.verify("parquet 的非图像列与 jsonl 逐条一致", not mism, n_mismatch=len(mism), head=mism[:10], n_jsonl=len(jsonl_rows), n_parquet_rows_read=n_parquet_rows)
    m.verify("idx 全局唯一且每行都有", None not in idxs and len(set(idxs)) == len(idxs), n=len(idxs))
    write_jsonl(m, src_dir, src_dir / "test.jsonl", out)

    # 去重相关的事实（只陈述）
    by_src: dict[str, set[str]] = {}
    for r in out:
        by_src.setdefault(r["source_dataset"], set()).add(r["image_key"])
    coco_keys = sorted(by_src.get("COCO 2017 Validation Set", set()))
    nyu_keys = sorted(k for k in by_src.get("Omni3D_SUNRGBD", set()) if k.startswith("nyuv2:"))
    facts: dict[str, Any] = {
        "rows_per_source_dataset": {k: sum(1 for r in out if r["source_dataset"] == k) for k in sorted(by_src)},
        "unique_images_per_source_dataset": {k: len(v) for k, v in sorted(by_src.items())},
        "coco_val2017_image_keys": coco_keys,
        "ade20k_val_image_keys_n": len(by_src.get("ADE20K Validation Set", set())),
        "nyuv2_image_keys": nyu_keys,
        "nyuv2_rows": sum(1 for r in out if r["image_key"].startswith("nyuv2:")),
    }
    coco_sub = ctx.root / "coco_val2017" / "MANIFEST.json"
    if coco_sub.exists():
        sel = set(json.loads(coco_sub.read_text())["selection"]["image_ids"])
        facts["overlap_with_coco_val2017_e1sub500"] = sorted(k for k in coco_keys if int(k.split(":")[1]) in sel)
    nyu_idx = ctx.root / "nyuv2" / "index.json"
    if nyu_idx.exists():
        test = {f"nyuv2:{r['idx']}" for r in json.loads(nyu_idx.read_text())["images"] if r["split"] == "test"}
        facts["nyuv2_keys_in_official_test"] = sum(1 for k in nyu_keys if k in test)
    m.data["dedupe_facts"] = {
        "image_key_scheme": "COCO → coco:<id>；ADE20K → ade20k:<stem>；SUN RGB-D 的 NYUdata/NYU#### → nyuv2:<#>（1-based，与 splits.mat 同基）；其余 sunrgbd:/hypersim:/nuscenes:",
        "known": "data_sources C §2.3：ADE20K val 633 张、COCO val2017 805 张、NYUv2 官方 test 36 张（89 行）；Hypersim 全在官方 test；nuScenes 划分未核实",
        **facts,
    }
    m.data["stats"] = {"n_rows": len(out), "by_type_task": {f"{t}/{k}": sum(1 for r in out if (r["type"], r["task"]) == (t, k)) for t, k in sorted({(r["type"], r["task"]) for r in out})}}
    m.data["selection"] = {"rule": "test 全部 2,638 题（只有 test 划分）"}
    m.write()


# ---------------------------------------------------------------------------
# EmbSpatial-Bench（test 3,640；官方数据仓库 Phineas476/EmbSpatial-Bench 的 embspatial_bench.json）
# ---------------------------------------------------------------------------

EMBSPATIAL = {
    "source": "embspatial",
    "repo": "Phineas476/EmbSpatial-Bench",
    "rev": "e2733cacf0f20371518c687176e881d79644e61e",
    "path": "embspatial_bench.json",
    "estimate_bytes": 200_000_000,  # base64 解码后约为 json 的 3/4（253.5 MB × 0.75 ≈ 190 MB）
    "license": {
        "text": "cc-by-4.0（官方数据仓库 Phineas476/EmbSpatial-Bench 的卡片）；官方 GitHub mengfeidu/EmbSpatial-Bench 没有 LICENSE 文件；FlagEval 转换版没有 license。图像来自 MP3D、AI2-THOR、ScanNet 渲染/采集，上游许可未核实。data_sources C §3.1",
        "project_terms": PROJECT_TERMS,
    },
}


def iter_json_array(f, limit: int | None = None, chunk: int = 1 << 20):
    """从文件对象流式解析顶层 JSON 数组的元素（不把整个文件读进内存），最多 limit 个。"""
    import codecs

    dec = json.JSONDecoder()
    utf8 = codecs.getincrementaldecoder("utf-8")()
    buf, pos, started, n = "", 0, False, 0
    eof = False
    while limit is None or n < limit:
        # 跳过空白、逗号与开头的 [
        while True:
            while pos < len(buf) and buf[pos] in " \t\r\n,":
                pos += 1
            if not started and pos < len(buf):
                if buf[pos] != "[":
                    raise ValueError("不是 JSON 数组")
                started, pos = True, pos + 1
                continue
            break
        if pos < len(buf) and buf[pos] == "]":
            return
        try:
            obj, end = dec.raw_decode(buf, pos)
        except json.JSONDecodeError:
            if eof:
                raise
            data = f.read(chunk)
            if not data:
                eof = True
                buf += utf8.decode(b"", final=True)
            else:
                buf = buf[pos:] + utf8.decode(data)
                pos = 0
            continue
        yield obj
        n += 1
        pos = end


def land_embspatial(ctx: LandContext, spec: dict[str, Any] | None = None) -> None:
    spec = {**EMBSPATIAL, **(spec or {}), "hf_base": (spec or {}).get("hf_base", HF_BASE)}
    name = spec["source"]
    src_dir = ctx.source_dir(name)
    m = ctx.manifest(name)
    m.data["license"] = spec["license"]
    est = spec["estimate_bytes"] if ctx.limit is None else 5_000_000
    budget_gate(ctx, m, est, {"tmpdir_peak_bytes": 253_512_493})
    tree = hf_tree(spec["repo"], spec["rev"], "", base=spec["hf_base"])
    tmp = ctx.scratch(name) / spec["path"]
    if ctx.limit is None:
        rec = fetch_hf_file(ctx, m, spec["repo"], spec["rev"], spec["path"], tmp, tree[spec["path"]], base=spec["hf_base"])
        m.data["upstream"].append({**rec, "official_check": "官方数据仓库（作者 README 指向），LFS sha256"})
        m.df("json 已下载到 TMPDIR")
        rows = json.loads(tmp.read_text())
    else:
        # 开发机小样本：不下 253 MB 的 json，用 Range 流式解析开头的 limit 条
        from src.data.land.common import hf_resolve_url

        url = hf_resolve_url(spec["repo"], spec["rev"], spec["path"], base=spec["hf_base"])
        rf = net.HTTPRangeFile(url, size=tree[spec["path"]]["size"], retries=ctx.retries, backoff=ctx.backoff)
        rows = list(iter_json_array(rf, ctx.limit))
        m.data["upstream"].append({"url": url, "revision": spec["rev"], "lfs_sha256": tree[spec["path"]]["sha256"], "range_bytes_fetched": rf.bytes_fetched, "official_check": f"--limit {ctx.limit}：Range 流式读开头 {ctx.limit} 条（未下整个文件，未核对整文件 sha256）"})
    out = []
    for k, r in enumerate(rows):
        row = {key: v for key, v in r.items() if key != "image"}
        ans = int(r["answer"])
        row["answer_letter"] = chr(ord("A") + ans)
        if ctx.limit is None or k < ctx.limit:
            frec = save_image_bytes(m, src_dir, f"images/{r['question_id']}", base64.b64decode(r["image"]), image_key=f"embspatial:{r['question_id']}")
            row["image"] = frec["path"]
            row["image_sha256"] = frec["sha256"]
        out.append(row)
    qids = [r["question_id"] for r in out]
    m.verify("question_id 唯一", len(set(qids)) == len(qids), n=len(qids))
    write_jsonl(m, src_dir, src_dir / "test.jsonl", out)
    tmp.unlink(missing_ok=True)
    hist: dict[str, int] = {}
    for r in out:
        hist[r["data_source"]] = hist.get(r["data_source"], 0) + 1
    m.data["stats"] = {"n_rows": len(out), "by_data_source": dict(sorted(hist.items())), "keys": sorted(rows[0].keys()) if rows else []}
    m.data["selection"] = {"rule": "test 全部 3,640 题（benchmark 只有 test 划分）"}
    m.data["dedupe_facts"] = {
        "image_key_scheme": "embspatial:<question_id>",
        "known": "图像来自 MP3D（26 个场景：17 个官方 test、9 个官方 val）、AI2-THOR、ScanNet（175 个 scan 全在官方 val）；与 COCO val2017、ADE20K、NYUv2 无来源重合（data_sources C §3.3）",
        "bbox_note": "objects[].bbox 的格式按来源不同（mp3d、scannet 行只能按 xywh 解释，ai2thor 行只能按 xyxy 解释，data_sources C §3.5）；原样保存，E1 的问答不用它",
    }
    m.df("完成")
    m.write()
