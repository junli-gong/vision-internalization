"""e 文本到意象：文字完全决定答案（从指令反解出位姿再光栅化，与目标逐位相同）、难度档、旋转限制、速度。"""

import math
import re
import time

import numpy as np

from src.synth import generate
from src.synth import shapes as S
from src.synth.constants import IMAGERY_BINS
from src.synth.shapes import pose_polygon, qcoord, rasterize

_APP = {a.phrase: a.index for a in S.APPEARANCES}
_SHAPE = {v.phrase: (v.family, v.index) for f in S.VARIANTS for v in S.VARIANTS[f]}
_OBJ_RE = re.compile(
    r"a (?P<app>"
    + "|".join(sorted(map(re.escape, _APP), key=len, reverse=True))
    + r") (?P<shape>"
    + "|".join(sorted(map(re.escape, _SHAPE), key=len, reverse=True))
    + r") centered at \((?P<x>\d\.\d\d), (?P<y>\d\.\d\d)\) with radius (?P<r>\d\.\d\d)"
    + r"(?:, rotated (?P<a>\d+) degrees clockwise)?"
)


def _h(text: str) -> int:
    return int(round(float(text) * 100))


def _parse(instruction: str):
    objs = [
        (_APP[m["app"]], _SHAPE[m["shape"]], _h(m["x"]), _h(m["y"]), _h(m["r"]), int(m["a"] or 0))
        for m in _OBJ_RE.finditer(instruction)
    ]
    tm = re.search(r"The (?P<ref>[a-zA-Z\- ]+?) is (?:first )?(?:scaled|turned|moved)", instruction)
    ref = tm["ref"]
    matches = [o for o in objs if f"{S.appearance(o[0]).phrase} {S.variant(*o[1]).noun}" == ref]
    assert len(matches) == 1, (ref, objs)
    s = re.search(r"scaled by (\d\.\d\d)", instruction)
    rot = re.search(r"turned (\d+) degrees (clockwise|counterclockwise)", instruction)
    mv = re.search(r"moved (.+?)\.(?:\s|$)", instruction)
    tx = ty = 0
    if mv:
        for amount, word in re.findall(r"(\d\.\d\d) (?:to the )?(right|left|down|up)", mv[1]):
            v = _h(amount)
            if word in ("right", "left"):
                tx = v if word == "right" else -v
            else:
                ty = v if word == "down" else -v
    theta = 0 if rot is None else int(rot[1]) * (1 if rot[2] == "clockwise" else -1)
    s_h = 100 if s is None else _h(s[1])
    return objs, matches[0], s_h, theta, (tx, ty)


def _mask(obj, x_h, y_h, r_h, a, s_h, h, w):
    short = min(h, w)
    (fam, var) = obj[1]
    p = pose_polygon(fam, var, qcoord(x_h * w / 100.0), qcoord(y_h * h / 100.0), qcoord(r_h * s_h * short / 10000.0), a)
    return rasterize(p, h, w)


def test_text_alone_determines_the_target():
    for canvas in ((512, 512), (384, 512)):
        h, w = canvas
        for i in range(24):
            s = generate("imagery", "train", 50 + i, canvas_hw=canvas)
            assert s.image is None
            objs, t, s_h, theta, (tx, ty) = _parse(s.instruction)
            assert len(objs) == len(s.meta["scene"]["objects"])
            _, _, x_h, y_h, r_h, a = t
            before = _mask(t, x_h, y_h, r_h, a, 100, h, w)
            after = _mask(t, x_h + tx, y_h + ty, r_h, a + theta, s_h, h, w)
            assert np.array_equal(after, s.target_mask), (canvas, i, s.instruction)
            assert np.array_equal(before, s.baselines["no_transform"].mask)
            tr = s.meta["transform"]
            assert (tr["scale"] is None) == (s_h == 100) and (tr["rotate_deg"] or 0) == theta


def test_difficulty_bins_and_translation_displacement():
    for b, (lo, hi) in enumerate(IMAGERY_BINS):
        for i in range(10):
            s = generate("imagery", "val", 400 + i, bin=b)
            d = s.meta["difficulty"]
            assert d["bin"] == b and lo <= d["value"] < hi
            tr = s.meta["transform"]
            if tr["type"] == "translate":  # 纯平移：每一点位移相同，平均位移 = |t| / min(H, W)
                o0 = s.meta["scene"]["objects"][s.meta["target_obj_id"]]
                o1 = s.meta["scene_after"]["objects"][s.meta["target_obj_id"]]
                disp = math.sqrt((o1["cx"] - o0["cx"]) ** 2 + (o1["cy"] - o0["cy"]) ** 2) / 512
                assert abs(d["value"] - disp) < 1e-9


def test_rotations_respect_symmetry_and_transform_is_never_identity():
    for i in range(80):
        s = generate("imagery", "test", 600 + i)
        o = s.meta["scene"]["objects"][s.meta["target_obj_id"]]
        period = S.rotation_period(o["family"], o["variant"])
        th = s.meta["transform"]["rotate_deg"]
        if th is not None:
            assert period != 0.0 and 1 <= abs(th) <= min(180.0, period / 2.0)
        assert s.meta["baseline_iou"]["no_transform"]["full"] < 1.0
        # 其余物体不动
        for k, (a, b) in enumerate(zip(s.meta["scene"]["objects"], s.meta["scene_after"]["objects"])):
            if k != s.meta["target_obj_id"]:
                assert a == b


def test_speed_at_least_50_per_second_single_core():
    generate("imagery", "train", 0)
    t = time.process_time()
    n = 80
    for i in range(n):
        generate("imagery", "train", 1000 + i)
    rate = n / (time.process_time() - t)
    print(f"imagery: {rate:.1f} samples/s (CPU time, single process)")
    assert rate >= 50
