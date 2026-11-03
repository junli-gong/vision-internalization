"""DEV-0080：X-M 的后处理（src/xm/post.py）——NMS、逐实例贴边与 X-2 的工具逐位相同、退回、进程池、不 import 学习型模块。"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.executors.geometry import box_iou
from src.executors.raster import rle_to_mask
from src.explore import guidance as G
from src.explore import x2_prompts as XP
from src.explore import x2_runner as X2R
from src.explore import x2_tools as XT
from src.xm import post as PO
from src.xm.runner import pairs_from_json
from tests import _xm_run_fakes as F

REPO = Path(__file__).resolve().parents[1]


def test_nms_rules():
    a = [10.0, 10.0, 50.0, 50.0]
    near = [10.0, 10.0, 50.0, 55.0]  # IoU = 1600 / 1800 ≈ 0.889 > 0.7
    far = [60.0, 60.0, 90.0, 90.0]
    assert PO.nms([a, near, far, list(a)]) == [0, 2]  # 输出顺序贪心，去掉后者
    assert PO.nms([near, a]) == [0]
    assert PO.nms([]) == []
    sq, exact = [0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 7.0]  # IoU 恰为 0.7（70 / 100）：保留（只去掉 > 阈值的）
    assert PO.box_iou_matrix([sq], [exact])[0, 0] == 0.7
    assert PO.nms([sq, exact], iou=0.7) == [0, 1] and PO.nms([sq, exact], iou=0.69) == [0]
    zero = [5.0, 5.0, 5.0, 9.0]
    assert PO.nms([zero, zero]) == [0, 1]  # 两个面积为 0 的框 IoU 记 0
    rng = np.random.default_rng(0)
    bs = [sorted(rng.uniform(0, 100, 2)) for _ in range(20)]
    boxes = [[x[0], y[0], x[1], y[1]] for x, y in zip(bs[:10], bs[10:])]
    m = PO.box_iou_matrix(boxes, boxes)
    for i in range(10):
        for j in range(10):
            assert abs(m[i, j] - box_iou(boxes[i], boxes[j])) < 1e-12
    with pytest.raises(ValueError):
        PO.nms([a], iou=1.5)


def test_parameters_are_x2s():
    rc = X2R.load_run_config()
    assert PO.TRIMAP_MODE == rc.doc["spline"]["trimap"]["mode"] == "trimap"
    ident = PO.identity()
    assert ident["band_pct"] == G.BAND_PCT_DEFAULT and ident["context_pct"] == G.CONTEXT_PCT_DEFAULT and ident["nms_iou"] == 0.7
    assert PO.SPLINE_FAILURE_FALLBACK in PO.FALLBACKS and PO.BOX_GRABCUT in G.LIBRARY_V2


@pytest.fixture(scope="module")
def fixture_pairs(tmp_path_factory):
    d = tmp_path_factory.mktemp("xmfx")
    pj = F.write_fixture(d)
    return pairs_from_json(pj), pj.parent


def test_pair_tool_job_matches_x2_tools(fixture_pairs, monkeypatch):
    pairs, root = fixture_pairs
    p = pairs[0]
    path = str(root / p.image)
    boxes = [list(i.box_xyxy) for i in p.instances]
    q = [XT.ellipse_ctrl_q(b, p.width, p.height, 16) for b in boxes]
    splines = [[[min(1000, v[0] + 7), v[1]] for v in q[0]], None]
    out = PO.pair_tool_job((path, "xm", boxes, splines, 16))
    assert len(out["instances"]) == 2 and out["wall_s"] >= 0
    img = XT.load_rgb(path)
    for k, (b, inst) in enumerate(zip(boxes, out["instances"])):
        ell = XT.ellipse_ctrl_q(b, p.width, p.height, 16)
        ref = XT.spline_tool_job((path, b, splines[k], ell, PO.TRIMAP_MODE))  # X-2 的同一个工具一步
        assert inst["ellipse_q"] == ell and inst["ellipse"] == ref["ellipse"]
        assert inst["spline"] == ref["spline"]
        gc = G.generate_v2(img, b, ["gc_default"])[0]
        np.testing.assert_array_equal(rle_to_mask(inst["box_gc"]["rle"]), gc.mask)
        assert inst["box_gc"]["ok"] is gc.ok
    assert out["instances"][1]["spline"] is None
    base = PO.pair_tool_job((path, "base", boxes, None, 16))
    assert [set(i) for i in base["instances"]] == [{"box_gc"}] * 2 and base["instances"][0]["box_gc"] == out["instances"][0]["box_gc"]
    assert PO.pair_tool_job((path, "xm", [], None, 16))["instances"] == []
    with pytest.raises(ValueError):
        PO.pair_tool_job((path, "x2", boxes, None, 16))
    # 退回：样条失败 → 框 → 缺省 GrabCut（D168 的缺省，同 X-2 的 1 号候选），或框内椭圆（可选）
    inst = out["instances"][1]
    assert PO.SPLINE_FAILURE_FALLBACK == "box_grabcut"
    assert PO.instance_mask(inst, "spline_trimap") == (inst["box_gc"]["rle"], "box_grabcut")
    assert PO.instance_mask(inst, "spline_raster") == (inst["box_gc"]["rle"], "box_grabcut")
    assert PO.instance_mask(inst, "spline_trimap", "ellipse") == (inst["ellipse"]["trimap_rle"], "ellipse")
    assert PO.instance_mask(inst, "spline_raster", "ellipse") == (inst["ellipse"]["raster_rle"], "ellipse")
    monkeypatch.setattr(PO, "SPLINE_FAILURE_FALLBACK", "ellipse")  # 缺省值在调用时读
    assert PO.instance_mask(inst, "spline_trimap") == (inst["ellipse"]["trimap_rle"], "ellipse")
    assert PO.identity()["spline_failure_fallback"] == "ellipse"
    with pytest.raises(ValueError):
        PO.instance_mask(inst, "spline_trimap", "magic")
    ok = out["instances"][0]
    assert PO.instance_mask(ok, "spline_trimap") == (ok["spline"]["trimap_rle"], None)
    assert PO.instance_mask(ok, "box_grabcut") == (ok["box_gc"]["rle"], None)
    assert PO.instance_mask(ok, "ellipse_trimap") == (ok["ellipse"]["trimap_rle"], None)
    # 光栅化 = x2_prompts.spline_mask（采样点夹在框内）
    np.testing.assert_array_equal(rle_to_mask(ok["spline"]["raster_rle"]), XP.spline_mask(splines[0], boxes[0], img.shape[:2]))


def test_tool_pool_is_bitwise_identical(fixture_pairs):
    pairs, root = fixture_pairs
    jobs = [(str(root / p.image), "xm", [list(i.box_xyxy) for i in p.instances], None, 16) for p in pairs]
    with PO.ToolPool(1) as pool:
        a = [r["instances"] for r in pool.map(PO.pair_tool_job, jobs)]
    with PO.ToolPool(2) as pool:
        b = [r["instances"] for r in pool.map(PO.pair_tool_job, jobs)]
    assert a == b


def test_tool_modules_import_no_learned_modules():
    code = ("import json, sys; import src.xm.post, src.xm.prompts, src.xm.metrics; import src.explore.guidance as G; "
            "print(json.dumps(G.loaded_learned_modules()))")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO, check=True, env={**os.environ, "HF_HUB_OFFLINE": "1"})
    assert json.loads(r.stdout.strip().splitlines()[-1]) == []  # 进程池的子进程只 import 这些：连 torch 都不带进来
    import re

    for f in ("post.py", "prompts.py", "metrics.py", "gate.py", "runner.py", "data.py"):
        src = (REPO / "src" / "xm" / f).read_text(encoding="utf-8")
        assert not re.search(r"^\s*(?:from|import)\s+(?:" + "|".join(G.BANNED_MODULE_ROOTS) + r")\b", src, re.M), f
