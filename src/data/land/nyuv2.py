"""NYUv2 的落地（DEV-0012 任务书 §1.5；D11-2；data_sources B §1）。

来源：官方 labeled 集 nyu_depth_v2_labeled.mat（MATLAB v7.3 = HDF5，2,972,037,809 B）与官方 splits.mat。
- 本地已有：--nyu-local-dir（默认 $ROOT/datasets/nyuv2，别的项目的目录）里找大小与官方相同的 .mat。官方没有发布校验和，
  "sha256 与官方一致"的核对方式（--nyu-verify）：
  - stream（默认）：把官方 URL 流式读一遍只算 sha256（约 3 GB 网络、零落盘），与本地文件的 sha256 比；相等就只读使用本地
    文件（不复制、不修改），否则下载官方文件到 TMPDIR。
  - none：只比大小与 HDF5 schema，清单注明"未与官方逐字节核对"（需管理员批准才用）。
  本地没有就直接下载到 TMPDIR（下载时算出的 sha256 就是官方文件的 sha256）。
- 抽取：Eigen 654 张测试图（splits.mat testNdxs，1-based）与开发子集的官方 train 图（configs/e1_dev_subsets.json 的
  nyuv2_train），每张存 RGB（无损 PNG）与深度（米，float32）：depth_raw = rawDepths（未补洞，与 BTS 的 Eigen 测试 GT 同源，
  data_sources B §1.5），depth_filled = depths（官方 Levin 着色补洞）。深度用确定性的 npz（zlib 压缩，zip 时间戳固定），
  重跑逐字节相同。
- 开发机小样本（--nyu-remote）：不下 .mat，用 Range 直接以 h5py 打开远端文件，读 schema 与 --limit 张图。

Eigen crop 的两种写法（data_sources §3、B §1.5）：官方 toolbox 与 BTS 写的是同一组数字，但官方是 MATLAB 的 1-based
闭区间（427×561），BTS 是 Python 的 0-based 半开区间（426×560）。我们用 BTS 的写法（D11-2），即 src/constants.NYU_EIGEN_CROP；
index.json 里两种写法都由这个常量生成。

落地后的布局（<root>/nyuv2/）：splits.mat、rgb/<idx:04d>.png、depth/<idx:04d>.npz、index.json、MANIFEST.json。
下标 idx 一律是 labeled 集的 1-based 下标（与 splits.mat 同基）。
"""

from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.constants import NYU_DEPTH_MAX, NYU_DEPTH_MIN, NYU_EIGEN_CROP, NYU_EVAL_HW
from src.data.land import net
from src.data.land.common import LandContext, budget_gate, image_meta, register, write_json

SOURCE = "nyuv2"
N_LABELED = 1449

SPEC: dict[str, Any] = {
    "mat_url": "https://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat",
    "mat_bytes": 2_972_037_809,
    "splits_url": "https://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat",
    "splits_sha256": "6e081404491a8bfba2f066beaa9713ef6046f9007aa5f799f2a350506a580cee",  # P0：本地与官方相同
    "local_dir": "/home/ubuntu/workspace/wzg/gjl/datasets/nyuv2",
    # 每张：PNG RGB 约 0.45–0.6 MB，npz 两张深度约 0.8–1.6 MB（未实测，偏保守取 2.2 MB）
    "bytes_per_image": 2_200_000,
    "license": {
        "text": "官方主页 https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html 没有任何 license、terms 或 copyright 文字，只要求引用 Silberman et al., ECCV 2012。data_sources B §1.1",
        "project_terms": "D28-9：限非商业研究，不再分发任何数据或由数据生成的图像",
    },
}

# h5py 视角的形状（MATLAB 维度的转置），data_sources B §1.2
H5_SCHEMA = {
    "images": ((N_LABELED, 3, 640, 480), "uint8"),
    "depths": ((N_LABELED, 640, 480), "float32"),
    "rawDepths": ((N_LABELED, 640, 480), "float32"),
}


def read_splits(path: Path) -> tuple[list[int], list[int]]:
    from scipy.io import loadmat

    d = loadmat(path)
    train = [int(v) for v in np.asarray(d["trainNdxs"]).ravel()]
    test = [int(v) for v in np.asarray(d["testNdxs"]).ravel()]
    if sorted(train + test) != list(range(1, N_LABELED + 1)) or set(train) & set(test):
        raise ValueError("splits.mat 不是 {1..1449} 的不相交划分")
    return train, test


def check_schema(f) -> dict[str, Any]:
    out = {}
    for k, (shape, dtype) in H5_SCHEMA.items():
        ds = f[k]
        ok = tuple(ds.shape) == shape and str(ds.dtype) == dtype
        out[k] = {"shape": list(ds.shape), "dtype": str(ds.dtype), "ok": ok}
        if not ok:
            raise ValueError(f".mat 里 {k} 的 shape/dtype {ds.shape}/{ds.dtype} 与预期 {shape}/{dtype} 不符")
    return out


def _cell_str(f, name: str, i0: int) -> str:
    ref = f[name][0, i0]
    arr = np.asarray(f[ref]).ravel()
    return "".join(chr(int(c)) for c in arr)


def read_labeled(f, idx: int) -> dict[str, Any]:
    """读 labeled 集第 idx 张（1-based）：rgb (480,640,3) uint8、depth_raw / depth_filled (480,640) float32、scene、scene_type。"""
    i0 = idx - 1
    rgb = np.ascontiguousarray(np.asarray(f["images"][i0]).transpose(2, 1, 0))
    raw = np.ascontiguousarray(np.asarray(f["rawDepths"][i0]).T.astype(np.float32, copy=False))
    filled = np.ascontiguousarray(np.asarray(f["depths"][i0]).T.astype(np.float32, copy=False))
    if rgb.shape != (*NYU_EVAL_HW, 3) or raw.shape != NYU_EVAL_HW:
        raise ValueError(f"第 {idx} 张形状不对：rgb {rgb.shape}，depth {raw.shape}")
    return {"rgb": rgb, "depth_raw": raw, "depth_filled": filled, "scene": _cell_str(f, "scenes", i0), "scene_type": _cell_str(f, "sceneTypes", i0)}


def npz_bytes(**arrays: np.ndarray) -> bytes:
    """确定性的压缩 npz：zip 成员时间戳固定为 1980-01-01、按名字排序，np.load 可直接读。"""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for name in sorted(arrays):
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            arr_buf = io.BytesIO()
            np.lib.format.write_array(arr_buf, np.ascontiguousarray(arrays[name]), allow_pickle=False)
            zf.writestr(info, arr_buf.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)
    return buf.getvalue()


def png_bytes(rgb: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buf, format="PNG", compress_level=6)
    return buf.getvalue()


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() == data:
        return
    tmp = path.with_name(path.name + ".part")
    tmp.write_bytes(data)
    tmp.replace(path)


def stream_sha256(url: str, ctx: LandContext) -> tuple[str, int]:
    """把 URL 整个读一遍只算 sha256（不落盘）。中途断开时重试从头算。"""
    import urllib.request

    def go():
        h = hashlib.sha256()
        n = 0
        req = urllib.request.Request(url, headers={"User-Agent": net.USER_AGENT})
        with urllib.request.urlopen(req, timeout=net.DEFAULT_TIMEOUT) as r:
            for chunk in iter(lambda: r.read(8 << 20), b""):
                h.update(chunk)
                n += len(chunk)
        return h.hexdigest(), n

    last = None
    for attempt in range(ctx.retries + 1):
        try:
            return go()
        except Exception as e:  # noqa: BLE001 — 网络错误统一重试
            last = e
            ctx.log(f"流式校验第 {attempt + 1} 次失败：{e}")
    raise net.FetchError(url, f"流式校验失败：{last}")


def find_local_mat(local_dir: Path, size: int) -> Path | None:
    if not local_dir.is_dir():
        return None
    for p in sorted(local_dir.glob("*.mat")):
        if p.stat().st_size == size:
            return p
    return None


def dev_train_ids(train: list[int], dev_config: str | Path | None = None) -> list[int]:
    """开发子集的 NYU train 图：优先读入库的 configs/e1_dev_subsets.json，并核对它与规则重算的一致。"""
    from src.data.subsets import DEV_SUBSETS_PATH, dev_subset_k, load_dev_subsets, seeded_sample

    rule = [int(s) for s in seeded_sample([str(i) for i in train], dev_subset_k("nyuv2_train", len(train)), "dev:nyuv2_train")]
    path = Path(dev_config) if dev_config else DEV_SUBSETS_PATH
    if path.exists():
        cfg = load_dev_subsets(path)
        ids = [int(s) for s in cfg["subsets"]["nyuv2_train"]["ids"]]
        if ids != rule:
            raise ValueError(f"{path} 的 nyuv2_train 与按规则重算的不一致")
        return ids
    return rule


def land(ctx: LandContext, spec: dict[str, Any] | None = None) -> None:
    import h5py

    spec = {**SPEC, **(spec or {})}
    src_dir = ctx.source_dir(SOURCE)
    m = ctx.manifest(SOURCE)
    m.data["license"] = spec["license"]
    remote = bool(ctx.options.get("nyu_remote"))
    verify_mode = ctx.options.get("nyu_verify", "stream")
    local_dir = Path(ctx.options.get("nyu_local_dir") or spec["local_dir"])

    # splits.mat（2.6 KB）：官方 URL，sha256 = P0 记录
    splits_path = src_dir / "splits.mat"
    net.download(spec["splits_url"], splits_path, expected_sha256=spec["splits_sha256"], retries=ctx.retries, backoff=ctx.backoff, log=ctx.log)
    register(m, src_dir, splits_path, kind="metadata", url=spec["splits_url"])
    m.verify("splits.mat 的 sha256 = P0 记录（P0 已核对本地与官方相同）", True, sha256=spec["splits_sha256"])
    train, test = read_splits(splits_path)
    dev = dev_train_ids(train, ctx.options.get("dev_config"))
    if ctx.limit is None:
        want = [(i, "test") for i in test] + [(i, "train") for i in dev]
    else:  # 小样本：前 limit−1 张 test 加 1 张开发子集的 train 图，两条路径都走到
        want = [(i, "test") for i in test[: max(ctx.limit - 1, 1)]] + ([(dev[0], "train")] if ctx.limit >= 2 else [])
    est = int(len(want) * spec["bytes_per_image"])
    budget_gate(ctx, m, est, {"n_images": len(want), "assumed_bytes_per_image": spec["bytes_per_image"], "tmpdir_peak_bytes": 0 if remote else spec["mat_bytes"]})

    # 打开 .mat
    upstream: dict[str, Any] = {"url": spec["mat_url"], "bytes": spec["mat_bytes"]}
    mat_tmp = ctx.scratch(SOURCE) / "nyu_depth_v2_labeled.mat"
    rf = None
    if remote:
        rf = net.HTTPRangeFile(spec["mat_url"], retries=ctx.retries, backoff=ctx.backoff)
        if rf.size != spec["mat_bytes"]:
            raise net.FetchError(spec["mat_url"], f"大小 {rf.size} ≠ {spec['mat_bytes']}")
        fobj: Any = rf
        upstream["official_check"] = "--nyu-remote：Range 直接读官方文件（开发机小样本，未算整文件 sha256）"
    else:
        local = find_local_mat(local_dir, spec["mat_bytes"])
        if local is not None and verify_mode == "stream":
            ctx.log(f"本地候选 {local}，计算 sha256 …")
            local_sha = net.sha256_file(local)
            ctx.log("流式读取官方文件计算 sha256（不落盘）…")
            off_sha, off_n = stream_sha256(spec["mat_url"], ctx)
            same = local_sha == off_sha and off_n == spec["mat_bytes"]
            m.verify("本地 .mat 与官方文件 sha256 相同（官方流式计算）", same, local=str(local), local_sha256=local_sha, official_sha256=off_sha)
            upstream.update(sha256=off_sha, local_path=str(local), local_sha256=local_sha)
            if same:
                fobj = str(local)
                upstream["official_check"] = "官方 URL 流式 sha256 与本地文件相同；只读使用本地文件，未复制、未修改"
            else:
                local = None
        elif local is not None and verify_mode == "none":
            fobj = str(local)
            upstream.update(local_path=str(local), local_sha256=net.sha256_file(local), official_check="未与官方逐字节核对（--nyu-verify none）：只比大小与 HDF5 schema")
            m.note("NYUv2 .mat 未与官方逐字节核对（--nyu-verify none）")
        if local is None:
            res = net.download(spec["mat_url"], mat_tmp, expected_size=spec["mat_bytes"], retries=ctx.retries, backoff=ctx.backoff, log=ctx.log)
            fobj = str(mat_tmp)
            upstream.update(sha256=res.sha256, tmp_path=str(mat_tmp), official_check="官方 URL 下载到 TMPDIR，sha256 即官方文件的 sha256")
    m.data["upstream"].append(upstream)
    m.df(".mat 就绪")

    index = []
    with h5py.File(fobj, "r") as f:
        upstream["h5_schema"] = check_schema(f)
        for idx, split in want:
            d = read_labeled(f, idx)
            rgb_rel, dep_rel = f"rgb/{idx:04d}.png", f"depth/{idx:04d}.npz"
            _write(src_dir / rgb_rel, png_bytes(d["rgb"]))
            _write(src_dir / dep_rel, npz_bytes(depth_raw=d["depth_raw"], depth_filled=d["depth_filled"]))
            rrec = register(m, src_dir, src_dir / rgb_rel, **image_meta(src_dir / rgb_rel), image_key=f"nyuv2:{idx}")
            drec = register(m, src_dir, src_dir / dep_rel, kind="depth")
            raw = d["depth_raw"]
            valid = (raw > NYU_DEPTH_MIN) & (raw < NYU_DEPTH_MAX)
            index.append(
                {
                    "idx": idx,
                    "split": split,
                    "dev": split == "train",
                    "scene": d["scene"],
                    "scene_type": d["scene_type"],
                    "rgb": rgb_rel,
                    "depth": dep_rel,
                    "rgb_sha256": rrec["sha256"],
                    "depth_sha256": drec["sha256"],
                    "depth_raw_valid_frac": round(float(valid.mean()), 6),
                }
            )
    if rf is not None:
        upstream["range_bytes_fetched"] = rf.bytes_fetched
    index.sort(key=lambda r: r["idx"])
    r0, r1, c0, c1 = NYU_EIGEN_CROP
    write_json(
        m,
        src_dir,
        src_dir / "index.json",
        {
            "images": index,
            "depth_units": "meters (float32)",
            "depth_keys": {"depth_raw": "rawDepths（未补洞；评测 GT，BTS 同源）", "depth_filled": "depths（官方补洞）"},
            "eigen_crop": {
                "ours": f"rows {r0}:{r1}, cols {c0}:{c1}（0-based 半开，{r1 - r0}×{c1 - c0}；BTS 写法，D11-2；src/constants.NYU_EIGEN_CROP）",
                "matlab_toolbox": f"mask({r0}:{r1}, {c0}:{c1})（1-based 闭区间，{r1 - r0 + 1}×{c1 - c0 + 1}；未采用）",
            },
            "index_base": "1-based（与 splits.mat 同基；BTS/Marigold 的文件名是 0-based）",
        },
    )
    if not remote and mat_tmp.exists():
        mat_tmp.unlink()
    m.data["selection"] = {
        "test": "Eigen 654（splits.mat testNdxs）",
        "dev_train": f"configs/e1_dev_subsets.json 的 nyuv2_train（{len(dev)} 张官方 train 图）",
        "landed": [i for i, _ in want] if ctx.limit is not None else "全部",
    }
    m.data["stats"] = {"n_test": sum(1 for _, s in want if s == "test"), "n_dev_train": sum(1 for _, s in want if s == "train")}
    m.data["dedupe_facts"] = {
        "image_key_scheme": "nyuv2:<1-based idx>",
        "known": "CV-Bench 含 36 张 NYUv2 图，全在官方 test（data_sources C §2.3）；train 249 个场景与 test 215 个场景不相交（B §1.3）；BTS 训练清单里 bookstore_0001d–j 的 1,383 帧属于 test 场景（D11-2，训练集构造时处理）",
    }
    m.df("完成")
    m.write()
