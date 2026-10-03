"""NYUv2 表面法向 GT 的落地（DEV-0022 任务书 §1；D11-2 第 3 点；data_sources B §1.6）。

来源的选择（D11-2：训练与测试同源、覆盖 654 测试集、能核实）：Bae, Budvytis, Cipolla, ICCV 2021
（baegwangbin/surface_normal_uncertainty）README 链接的 nyu_dataset.zip（Google Drive，1,008,950,467 B）。
- 开发机 2026-10-02 用 Range 只读了中央目录（尾部 296,384 B）：nyu/train/{img,norm} 各 795 张、nyu/test/{img,norm}
  各 654 张，文件名是 labeled 集的 0 起下标；加 1 后 train 恰为 splits.mat 的 trainNdxs、test 恰为 testNdxs
  （对照 P0 存档的 GeoNet-test 镜像 test_split.txt 与 splits.mat 的统计）。训练与测试来自同一个包、同一种编码。
- README 原文 "We used the ground truth and data split provided by GeoNet"（Qi et al., CVPR 2018）：GT 是 GeoNet 的。
- 其他候选为什么不选：Eigen & Fergus 的 normals_gt.tgz 只有 Wayback，是 tar.gz，不能随机访问，成员数核实不了
  （要整包下载 496 MB）；Ladický 2014 的 zip 已失效且没有存档；GeoNet-test 的 HF 镜像与 Marigold 的评测包只有 654 张
  测试图，没有同源的 795 张训练图；tanganke/nyuv2 是 288×384 的缩小版，来源链不清。

编码（Bae 的 data/dataloader_nyu.py@fe2b9f1）：norm PNG 是 8 位 RGB，n = rgb / 255 × 2 − 1；三个通道都为 0 的像素
无效（GT 只在图像中心区域有定义）；水平翻转时取反 x 分量（x 指向图像右侧）。加载器（src/data/train_loaders.py）
解码后再归一化成单位向量。

国内网络连不上 Google Drive 时：先用任何能用的办法把 nyu_dataset.zip 放到本地，--local-zip 指过去；
中央目录 sha256 与每个成员的 CRC32 照样核对，与远端取的逐字节相同。

落地后的布局（<root>/nyu_normals/）：train/img/<idx>.png、train/norm/<idx>.png、test/…（idx 为 labeled 集的
1 起下标、4 位补零，与 E1 的 nyuv2 同基）、index.json、MANIFEST.json。
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from src.constants import NYU_EVAL_HW
from src.data.land import net, nyuv2
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
from src.data.land.nyu_depth_train import read_png_header

SOURCE = "nyu_normals"
SCRIPT = "scripts/data/land_train.py"
SPLITS = ("train", "test")
ROLE = {"train": "train", "test": "eval"}
# 开发机 --limit 实测：img 与 norm 都是 8 位 RGBA（颜色类型 6）；Bae 的加载器 .convert("RGB") 丢掉 α。也接受 RGB（2）。
PNG_COLOR_TYPES = (2, 6)

SPEC: dict[str, Any] = {
    "zip_url": "https://drive.usercontent.google.com/download?id=1Rl-x6GwIXWdRW6LbnSayjIAfqCqygvYQ&export=download&confirm=t",
    "drive_file_id": "1Rl-x6GwIXWdRW6LbnSayjIAfqCqygvYQ",
    "drive_folder": "https://drive.google.com/drive/folders/1Ku25Am69h_HrbtcCptXn4aetjo7sB33F（Bae 2021 README 的 NYUv2 (official) 链接）",
    "last_modified": "Mon, 02 May 2022 16:55:27 GMT",
    "zip_pins": {"size": 1_008_950_467, "n_entries": 2_905, "start_dir": 1_008_654_083, "tail_bytes": 296_384, "tail_sha256": "0202a05c4842962739fafbd5b233f3aac3f9d766cfd85904a54dda89dccd3291"},
    "member": "nyu/{split}/{part}/{idx0:06d}.png",
    # sha256(",".join(1 起下标升序))：test = splits.mat 的 testNdxs，train = trainNdxs（开发机核对）
    "expect": {"train": (795, "8db3dc4c0cf4f885dfa708df185bd06757cbe04733b78f4d9744ce3bc67b1a73"), "test": (654, "6e8da89f5e5d8d0972916b91530bd5ac245258e502a069be28388dce8d8f0a15")},
    "members_bytes": 1_008_804_374,
    "mean_pair_bytes": 696_200,
    "hw": NYU_EVAL_HW,  # (H, W)，src/constants.py
    "code_ref": "baegwangbin/surface_normal_uncertainty@fe2b9f1e8a4cac1c73475b023f6454bd23827a48 data/dataloader_nyu.py",
    "zip_gap": 1 << 20,
    # 官方 splits.mat（2.6 KB）：URL 与 sha256 沿用 E1 的 nyuv2.SPEC（单一来源）
    "splits_url": nyuv2.SPEC["splits_url"],
    "splits_sha256": nyuv2.SPEC["splits_sha256"],
    "license": {
        "text": "未声明。Bae 2021 仓库为 MIT（指代码，GitHub API）；README 写 GT 与划分来自 GeoNet，GeoNet 仓库没有 LICENSE（GitHub API license 为 null）；NYU 官方主页没有许可文字（data_sources B §1.1、§1.6）",
        "project_terms": "D28-9：限非商业研究，不再分发任何数据或由数据生成的图像",
    },
}


def index_digest(ids: list[int]) -> str:
    return hashlib.sha256(",".join(str(i) for i in sorted(ids)).encode()).hexdigest()


def zip_split_ids(index, split: str) -> list[int]:
    """zip 里某个划分 img 与 norm 都有的 labeled 下标（1 起）。两边不一致就报错。"""
    pre_i, pre_n = f"nyu/{split}/img/", f"nyu/{split}/norm/"
    img = sorted(int(Path(k).stem) + 1 for k in index.members if k.startswith(pre_i) and k.endswith(".png"))
    nrm = sorted(int(Path(k).stem) + 1 for k in index.members if k.startswith(pre_n) and k.endswith(".png"))
    if img != nrm:
        raise ValueError(f"{split}：img 与 norm 的下标不一致")
    return img


def land(ctx: LandContext, spec: dict[str, Any] | None = None) -> None:
    spec = {**SPEC, **(spec or {})}
    src_dir = ctx.source_dir(SOURCE)
    m = ctx.manifest(SOURCE, script=SCRIPT)
    m.data["license"] = spec["license"]
    est = int((spec["members_bytes"] if ctx.limit is None else spec["mean_pair_bytes"] * ctx.limit) * 1.05) + 2_000_000
    budget_gate(ctx, m, est, {"n_pairs": 795 + 654 if ctx.limit is None else ctx.limit, "margin": "5%", "tmpdir_peak_bytes": 0})

    src = zip_source(ctx, spec["zip_url"], ctx.options.get("local_zip"))
    index = read_zip_index(src)
    check_zip_pins(m, "Bae 2021 nyu_dataset.zip", index, spec["zip_pins"])
    ids = {sp: zip_split_ids(index, sp) for sp in SPLITS}
    for sp in SPLITS:
        n, dig = spec["expect"][sp]
        m.verify(f"{sp}：zip 里的下标数与指纹 = 官方 splits.mat（开发机核对）", (len(ids[sp]), index_digest(ids[sp])) == (n, dig), got=(len(ids[sp]), index_digest(ids[sp])), expected=(n, dig))
    # 能拿到官方 splits.mat（2.6 KB，sha256 钉在 E1 的 nyuv2.SPEC）就再逐个比一次；拿不到不算失败（上面的指纹已钉住）
    try:
        sm = ctx.scratch(SOURCE) / "splits.mat"
        net.download(spec["splits_url"], sm, expected_sha256=spec["splits_sha256"], retries=min(ctx.retries, 1), backoff=ctx.backoff, log=ctx.log)
        tr, te = nyuv2.read_splits(sm)
        sm.unlink()
        m.verify("zip 的 train/test 下标 = 官方 splits.mat 的 trainNdxs/testNdxs（逐个）", ids["train"] == sorted(tr) and ids["test"] == sorted(te))
    except (net.FetchError, OSError) as e:
        m.note(f"官方 splits.mat 取不到（{e}）；划分只按钉住的下标指纹核对")

    if ctx.limit is None:
        want = [(i, "test") for i in ids["test"]] + [(i, "train") for i in ids["train"]]
    else:  # 小样本：前 limit−1 张 test 加 1 张 train，两条路径都走到
        want = [(i, "test") for i in ids["test"][: max(ctx.limit - 1, 1)]] + ([(ids["train"][0], "train")] if ctx.limit >= 2 else [])
    jobs = []
    for idx, sp in want:
        for part in ("img", "norm"):
            jobs.append(MemberJob(spec["member"].format(split=sp, part=part, idx0=idx - 1), src_dir / sp / part / f"{idx:04d}.png", {"idx": idx, "split": sp, "part": part}))
    H, W = spec["hw"]

    def process(job: MemberJob, path: Path, rec: dict[str, Any]) -> dict[str, Any]:
        hdr = read_png_header(path)
        if (hdr["width"], hdr["height"], hdr["bit_depth"]) != (W, H, 8) or hdr["color_type"] not in PNG_COLOR_TYPES:
            raise ValueError(f"{path.name}：应为 8 位 RGB/RGBA {W}×{H} 的 PNG，实际 {hdr}")
        if job.meta["part"] == "norm":
            return {"kind": "normal", "width": W, "height": H, "png_color_type": hdr["color_type"], "encoding": "n = rgb/255*2-1（只用 RGB 三个通道）；(0,0,0) 无效"}
        return {**image_meta(path), "png_color_type": hdr["color_type"]}

    def on_done(job: MemberJob, rec: dict[str, Any]) -> None:
        rel = job.dest.relative_to(src_dir).as_posix()
        extra = {k: v for k, v in rec.items() if k not in ("sha256", "bytes", "skipped")}
        if job.meta["part"] == "img":
            extra["image_key"] = f"nyuv2:{job.meta['idx']}"
        m.add_file(rel, sha256=rec["sha256"], nbytes=rec["bytes"], split=job.meta["split"], idx=job.meta["idx"], **extra)

    ok, failures = fetch_zip_members(src, index, jobs, workers=ctx.workers, gap=spec["zip_gap"], process=process, on_done=on_done, log=ctx.log)
    for f in failures:
        m.fail(step="法向", **f)
        ctx.log(f"[失败] {f['member']}: {f['error']}")
    m.data["upstream"].append({"role": "法向 GT 与对应 RGB", "drive_folder": spec["drive_folder"], "last_modified": spec["last_modified"], **src.record(), "zip": index.pins(), "official_check": "作者在 README 里给的 Drive 文件（或 --local-zip 的本地副本）；中央目录 sha256 = SPEC；逐个成员 CRC32；下标集合 = 官方 795/654；PNG 均为 8 位 RGB 640×480"})

    files = m.data["files"]
    rows = []
    for idx, sp in sorted(want, key=lambda t: (t[1] != "test", t[0])):
        img, nrm = f"{sp}/img/{idx:04d}.png", f"{sp}/norm/{idx:04d}.png"
        if img in files and nrm in files:
            rows.append({"idx": idx, "split": sp, "img": img, "norm": nrm, "img_sha256": files[img]["sha256"], "norm_sha256": files[nrm]["sha256"]})
    write_json(
        m,
        src_dir,
        src_dir / "index.json",
        {
            "images": rows,
            "index_base": "1-based（与 splits.mat、E1 的 nyuv2 同基；zip 里的文件名是 0-based）",
            "decode": "n = rgb/255*2-1，再归一化为单位向量；rgb == (0,0,0) 的像素无效（Bae 2021 data/dataloader_nyu.py）",
            "axes": "x 指向图像右侧（水平翻转时取反 x，Bae 2021 dataloader_nyu.py）；y、z 的朝向沿用 GeoNet/Bae 的约定，未另行核实",
            "gt_origin": "GeoNet（Qi et al., CVPR 2018），经 Bae 2021 打包（README 原文 \"We used the ground truth and data split provided by GeoNet\"）",
        },
    )
    m.data["selection"] = {"rule": "官方 795/654 全部", "landed": "全部" if ctx.limit is None else [f"{sp}:{i}" for i, sp in want]}
    m.data["stats"] = {"n_train": sum(1 for _, s in want if s == "train"), "n_test": sum(1 for _, s in want if s == "test"), "n_pairs_landed": len(rows)}
    m.data["dedupe_facts"] = {
        "image_key_scheme": "nyuv2:<1-based idx>（与 E1 的 nyuv2 同键）",
        "split_field": "清单里每个文件记 split=train|test；train 是训练集，test 是评测集（与 E1 的 NYU 654 是同一批图）",
        "known": "train 795 张里含 E1 的 NYU 开发子集 50 张（configs/e1_dev_subsets.json 的 nyuv2_train，同键），去重报告会把它们列入训练剔除清单；test 与 E1 的 nyuv2 test 同键，是评测集之间的已知重合，用来标定感知哈希阈值",
    }
    m.df("完成")
    m.write()

