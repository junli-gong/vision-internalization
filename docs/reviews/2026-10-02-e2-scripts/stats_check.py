import sys

import torch

sys.path.insert(0, ".")
from src.e2.features import ChannelStats, Normalizer  # noqa: E402

g = torch.Generator().manual_seed(0)
D = 64
batches = []
for i in range(50):
    n = int(torch.randint(1, 400, (1,), generator=g))
    x = torch.randn(n, D, generator=g) * torch.linspace(0.01, 50, D) + torch.linspace(-3e3, 1e4, D)
    batches.append(x.to(torch.bfloat16))
s = ChannelStats()
for b in batches:
    s.update(b)
allx = torch.cat([b.double() for b in batches])
ref_mean, ref_var = allx.mean(0), allx.var(0, unbiased=False)
print("n", s.n, allx.shape[0])
print("max |mean-ref| / std:", float(((s.mean - ref_mean).abs() / ref_var.sqrt()).max()))
print("max rel err var:", float(((s.var() - ref_var).abs() / ref_var).max()))
st = ChannelStats.from_state(s.state())
nm = Normalizer.from_stats({0: st}, eps=1e-6, device="cpu")
grid = batches[0][: (batches[0].shape[0] // 4) * 4].reshape(-1, 4, D)
y = nm.apply(0, grid)
print("normalized shape", tuple(y.shape), "dtype", y.dtype)
ref = ((grid.double() - ref_mean) / (ref_var + 1e-6).sqrt()).permute(2, 0, 1)
print("max |norm - ref|:", float((y.double() - ref).abs().max()))
