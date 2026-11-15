"""Where does one evaluator beat another? Held-out loss by how decided the position is (its label), for trainer checkpoints.

A search compares positions that are close to each other, mostly balanced ones. A network can lower its average loss by
calibrating positions that are already decided (cheap to gain, worthless for choosing a move). This splits the loss.

  python3 zero/slice_loss.py <held-out shard tag> <checkpoint .pt> [<checkpoint .pt> ...]
"""
import sys, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, 'zero'); from compact import decode_torch
DEV = 'cuda'; tag = sys.argv[1]
_i = np.arange(768); LUT = torch.from_numpy(np.concatenate([((_i // 64 + 6) % 12) * 64 + ((_i % 64) ^ 56), [768]]).astype(np.int64)).to(DEV)
PT_W = torch.cat([torch.tensor([0, 1, 1, 2, 4, 0] * 2, dtype=torch.float32).repeat_interleave(64), torch.zeros(1)]).to(DEV)
X = torch.from_numpy(np.load(f'data/ev_{tag}_c.npy', mmap_mode='r')[:2_000_000].copy()).to(DEV); Y = torch.from_numpy(np.load(f'data/ev_{tag}_y.npy', mmap_mode='r')[:2_000_000].copy()).to(DEV).float() / 255


def mh(idx):
    return torch.zeros(idx.shape[0], 769, device=DEV).scatter_add_(1, idx, torch.ones(idx.shape, device=DEV))[:, :768]


@torch.no_grad()
def out(w, idx):
    Mo, Mp = mh(idx), mh(LUT[idx]); nb = w['wo'].shape[0]; B = idx.shape[0]; bk = ((Mo.sum(1) - 2) // 4).clamp(0, nb - 1).long()
    xo = (w['b'] + Mo @ w['E'][:768]).clamp(0, 1) ** 2; xp = (w['b'] + Mp @ w['E'][:768]).clamp(0, 1) ** 2
    lin = w['c'] + Mo @ w['S'][:768]; ph = ((Mo @ PT_W[:768]) / 24).clamp(max=1.0)
    o = ph * lin[:, 0] + (1 - ph) * lin[:, 1] + w['cb'][bk] + (xo * w['wo'][bk]).sum(1) + (xp * w['wp'][bk]).sum(1)
    if 'L1' in w:
        h1 = w['L1'].shape[0] // nb; ar = torch.arange(B, device=DEV)
        u = (torch.cat([xo, xp], 1) @ w['L1'].T + w['b1']).view(B, nb, h1)[ar, bk].clamp(0, 1) ** 2
        o = o + ((torch.bmm(w['L2'][bk], u[:, :, None]).squeeze(2) + w['b2'][bk]).clamp(0, 1) * w['L3'][bk]).sum(1)
    return o


edges = [0.0, 0.05, 0.2, 0.35, 0.65, 0.8, 0.95, 1.0001]; names = ['lost (<5%)', '5-20%', '20-35%', 'balanced 35-65%', '65-80%', '80-95%', 'won (>95%)']
res = {}
for p in sys.argv[2:]:
    w = {k: v.to(DEV).float() for k, v in torch.load(p, map_location='cpu').items()}
    o = torch.cat([out(w, decode_torch(X[j:j + 16384]).long()) for j in range(0, len(X), 16384)]); res[p] = (F.binary_cross_entropy_with_logits(o, Y, reduction='none'), (torch.sigmoid(o) - Y) ** 2)
print(f'{len(X)/1e6:.1f}M held-out positions from shard {tag}; log-loss (and squared error x1000) by how decided the label is')
print(f'{"":18s}{"share":>7s}  ' + '  '.join(f'{p.split("/")[-1][:22]:>22s}' for p in sys.argv[2:]))
for k, nm in enumerate(names):
    m = (Y >= edges[k]) & (Y < edges[k + 1])
    print(f'{nm:18s}{m.float().mean().item():7.1%}  ' + '  '.join(f'{res[p][0][m].mean().item():12.5f} ({res[p][1][m].mean().item() * 1000:6.2f})' for p in sys.argv[2:]))
print(f'{"all":18s}{1:7.0%}  ' + '  '.join(f'{res[p][0].mean().item():12.5f} ({res[p][1].mean().item() * 1000:6.2f})' for p in sys.argv[2:]))
