"""去重与剔除（src/data/dedupe.py；D11-7、D20）：感知哈希、规范键、剔除顺序与逐条记录。"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

from src.data import constants as C
from src.data.dedupe import (
    REASON_DEV,
    REASON_PROBE,
    Exclusion,
    apply_exclusions,
    check_sample,
    cross_overlap,
    hamming,
    phash_bytes,
    phash_image,
    probe_exclusion,
)
from src.data.records import Sample


def natural(w: int, h: int, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    base = rng.uniform(0, 255, size=(8, 8, 3))
    big = np.kron(base, np.ones((h // 8 + 1, w // 8 + 1, 1)))[:h, :w]
    return Image.fromarray(big.astype(np.uint8))


def test_phash_pinned_and_format():
    a = np.zeros((60, 90), np.uint8)
    a[:, 45:] = 255
    assert phash_image(Image.fromarray(a)) == "9100000000000000"
    assert (C.PHASH_SIZE, C.PHASH_HIGHFREQ, C.PHASH_MAX_HAMMING) == (8, 4, 8)
    for k in C.PROBE_IMAGE_KEYS:
        h = C.PROBE_IMAGE_PHASH[k]
        assert len(h) == 16 and int(h, 16) >= 0


def test_phash_robust_to_reencode_and_resize_but_separates_images():
    im = natural(200, 150, 0)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=60)
    h0 = phash_image(im)
    assert hamming(h0, phash_bytes(buf.getvalue())) <= C.PHASH_MAX_HAMMING
    assert hamming(h0, phash_image(im.resize((133, 100)))) <= C.PHASH_MAX_HAMMING
    assert hamming(h0, phash_image(natural(200, 150, 1))) > C.PHASH_MAX_HAMMING


def sample(i: str, key: str, ph: str | None) -> Sample:
    return Sample(id=i, task="t", images=("/x",), image_keys=(key,), fields={}, gt={}, source="s", split="v", phash=(ph,))


def test_exclusion_order_and_records():
    ex = probe_exclusion()
    assert set(ex.keys) == set(C.PROBE_IMAGE_KEYS) and len(ex.phashes) == 2
    ex.add_ids(["s/dev"], REASON_DEV)
    probe_h = C.PROBE_IMAGE_PHASH["coco:139"]
    near = f"{int(probe_h, 16) ^ 0b111:016x}"  # 3 位之差
    far = f"{int(probe_h, 16) ^ 0xFFFF:016x}"  # 16 位之差
    ss = [
        sample("s/dev", "coco:139", None),  # 样本 id 优先于图像键
        sample("s/key", "coco:39769", None),
        sample("s/near", "countbench:1", near),
        sample("s/far", "countbench:2", far),
        sample("s/none", "blink:x", None),
    ]
    kept, st = apply_exclusions(ss, ex)
    assert [s.id for s in kept] == ["s/far", "s/none"]
    by = {x["id"]: x for x in st["excluded"]}
    assert by["s/dev"] == {"id": "s/dev", "reason": REASON_DEV, "by": "sample_id"}
    assert by["s/key"]["by"] == "image_key" and by["s/key"]["reason"] == REASON_PROBE
    assert by["s/near"]["by"] == "phash" and by["s/near"]["hamming"] == 3 and by["s/near"]["matched"] == "coco:139"
    assert st["excluded_by_reason"] == {REASON_DEV: 1, REASON_PROBE: 2} and st["n_kept"] == 2 and st["n_without_phash"] == 3
    assert check_sample(sample("ok", "x:1", far), Exclusion()) is None


def test_cross_overlap_reports_same_key_and_near_duplicates():
    a = {"coco:1": "0" * 16, "coco:2": "f" * 16}
    b = {"coco:1": "0" * 16, "ade20k:x": f"{0b11:016x}", "nuscenes:y": None}
    r = cross_overlap(a, b)
    assert r["same_key"] == ["coco:1"]
    assert {(x["a"], x["b"], x["hamming"]) for x in r["phash_near"]} == {("coco:1", "coco:1", 0), ("coco:1", "ade20k:x", 2)}
