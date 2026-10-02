import numpy as np


def cells(h, w):
    return [(0, 0), (h - 1, w - 1), (1, w - 2), (h - 2, 1), (h // 2, w // 2)]


def perm(name, h, w):
    """sigma: 序列下标 k -> 空间格 (r, c)（假想的"真实"顺序）。"""
    n = h * w
    if name == "col_major":
        return [(k % h, k // h) for k in range(n)]
    if name == "hflip":
        return [(k // w, w - 1 - k % w) for k in range(n)]
    if name == "vflip":
        return [(h - 1 - k // w, k % w) for k in range(n)]
    if name == "serpentine":
        return [(k // w, (k % w) if (k // w) % 2 == 0 else w - 1 - k % w) for k in range(n)]
    if name == "in_block_transpose":
        res = []
        for k in range(n):
            r, c = divmod(k, w)
            if r < h - h % 2 and c < w - w % 2 and (r % 2) != (c % 2):
                r, c = r - r % 2 + c % 2, c - c % 2 + r % 2
            res.append((r, c))
        return res
    if name == "block_major_2x2":
        hb, wb = h // 2, w // 2
        res = []
        for br in range(hb):
            for bc in range(wb):
                for i in range(2):
                    for j in range(2):
                        res.append((2 * br + i, 2 * bc + j))
        rest = [(r, c) for r in range(h) for c in range(w) if not (r < 2 * hb and c < 2 * wb)]
        return res + rest
    if name == "swap_middle_pair":
        res = [divmod(k, w) for k in range(n)]
        k = (h // 3) * w + w // 3
        res[k], res[k + 1] = res[k + 1], res[k]
        return res
    if name == "shift_by_one":
        return [divmod((k + 1) % n, w) for k in range(n)]
    raise ValueError(name)


NAMES = ["col_major", "hflip", "vflip", "serpentine", "in_block_transpose", "block_major_2x2", "swap_middle_pair", "shift_by_one"]
for (h, w) in [(15, 20), (20, 15), (14, 19), (19, 14), (16, 16)]:
    print((h, w))
    for name in NAMES:
        s = perm(name, h, w)
        assert sorted(s) == [(r, c) for r in range(h) for c in range(w)], name
        inv = {rc: k for k, rc in enumerate(s)}
        caught = [rc for rc in cells(h, w) if inv[rc] != rc[0] * w + rc[1]]
        nfix = sum(1 for k, rc in enumerate(s) if k == rc[0] * w + rc[1])
        print(f"   {name}: {'CAUGHT' if caught else 'MISSED'} (by {len(caught)}/5 test cells; fixed points {nfix}/{h * w})")
