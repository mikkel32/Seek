"""Can the fast evaluator tell how far a search will move its own score?

For every training position there is the evaluator's static score and a label that is (mostly) a 2,500-position search
score. Their distance, in score units (logits), is how wrong the static score turned out to be. If that distance can be
predicted from the evaluator's own hidden layer, the search can use the prediction: prune with narrow margins where the
evaluator is reliable, wide margins where it is not (today every position gets the same margins).

Fits one linear read-out of the hidden layer per piece-count group (exact least squares, on the GPU), on one shard, and
tests it on another: explained variance, and the real distance in each tenth of the predicted distance.

  python3 zero/vol_probe.py <network .ema.pt> <train shard tag> <test shard tag> [positions each=8000000]
"""
import math, os, sys
import numpy as np, torch
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
DEV = torch.device('cuda'); torch.backends.cuda.matmul.allow_tf32 = True
w = {k: v.to(DEV).float() for k, v in torch.load(sys.argv[1], map_location='cpu').items()}; H = w['b'].shape[0]; NB = w['wo'].shape[0]
NMAX = int(sys.argv[4]) if len(sys.argv) > 4 else 8_000_000
i = np.arange(768); LUT = torch.from_numpy(np.concatenate([((i // 64 + 6) % 12) * 64 + ((i % 64) ^ 56), [768]]).astype(np.int64)).to(DEV)
PT_W = torch.cat([torch.tensor([0, 1, 1, 2, 4, 0] * 2, dtype=torch.float32).repeat_interleave(64), torch.zeros(1)]).to(DEV)


@torch.no_grad()
def feats(idx):
    """-> hidden activations (B, 2H), static output (B) in logits, piece-count group (B)"""
    B = idx.shape[0]; one = torch.ones(idx.shape, device=DEV)
    Mo = torch.zeros(B, 769, device=DEV).scatter_add_(1, idx, one)[:, :768]; Mp = torch.zeros(B, 769, device=DEV).scatter_add_(1, LUT[idx], one)[:, :768]
    bk = ((Mo.sum(1) - 2) // 4).clamp(0, NB - 1).long()
    xo = (w['b'] + Mo @ w['E'][:768]).clamp(0, 1) ** 2; xp = (w['b'] + Mp @ w['E'][:768]).clamp(0, 1) ** 2
    lin = w['c'] + Mo @ w['S'][:768]; ph = ((Mo @ PT_W[:768]) / 24).clamp(max=1.0)
    out = ph * lin[:, 0] + (1 - ph) * lin[:, 1] + w['cb'][bk] + (xo * w['wo'][bk]).sum(1) + (xp * w['wp'][bk]).sum(1)
    return torch.cat([xo, xp, torch.ones(B, 1, device=DEV)], 1), out, bk


def passes(tag, fn):
    x = np.load(f'data/ev_{tag}_x.npy', mmap_mode='r'); y = np.load(f'data/ev_{tag}_y.npy', mmap_mode='r'); n = min(NMAX, len(x))
    for s in range(0, n, 1_000_000):
        X = torch.from_numpy(np.ascontiguousarray(x[s:s + 1_000_000])).to(DEV); Y = torch.from_numpy(np.ascontiguousarray(y[s:s + 1_000_000])).to(DEV)
        for j in range(0, len(X), 32768):
            f, out, bk = feats(X[j:j + 32768].long()); yy = (Y[j:j + 32768].float() / 255).clamp(1 / 510, 1 - 1 / 510)
            fn(f, (torch.logit(yy) - out).abs().clamp(max=3.0), bk, out)


D = 2 * H + 1; A = torch.zeros(NB, D, D, device=DEV, dtype=torch.float64); Bv = torch.zeros(NB, D, device=DEV, dtype=torch.float64)
def acc(f, t, bk, out):
    for g in range(NB):
        m = bk == g
        if m.any():
            fg = f[m].double(); A[g] += fg.T @ fg; Bv[g] += fg.T @ t[m].double()
passes(sys.argv[2], acc)
W = torch.stack([torch.linalg.solve(A[g] + 1e-3 * torch.eye(D, device=DEV, dtype=torch.float64) * (A[g].diagonal().mean() + 1e-9), Bv[g]) for g in range(NB)]).float()
P, T, O = [], [], []
def test(f, t, bk, out):
    P.append((f * W[bk]).sum(1)); T.append(t); O.append(out)
passes(sys.argv[3], test)
p = torch.cat(P); t = torch.cat(T); o = torch.cat(O)
r2 = 1 - float(((t - p) ** 2).mean() / t.var()); cp = 271.58
print(f'{len(t)/1e6:.1f}M held-out positions. Distance between static score and label: mean {float(t.mean()) * cp:.0f} centipawns, median {float(t.median()) * cp:.0f}')
print(f'linear read-out of the hidden layer: explains {r2:.1%} of its variance (correlation {math.sqrt(max(r2, 0)):.2f})')
base = (o.abs() * 0.0); b2 = 1 - float(((t - (o.abs() * (t * o.abs()).mean() / (o.abs() ** 2).mean())) ** 2).mean() / t.var())
print(f'for comparison, predicting it from the size of the static score alone explains {b2:.1%}')
q = torch.quantile(p[:2_000_000], torch.linspace(0, 1, 11, device=DEV)); print('real distance (centipawns) in each tenth of the predicted distance, lowest to highest:')
print('  ' + '  '.join(f'{float(t[(p >= q[k]) & (p <= q[k + 1])].mean()) * cp:.0f}' for k in range(10)))
near = o.abs() < 1.0; q = torch.quantile(p[near][:2_000_000], torch.linspace(0, 1, 11, device=DEV)); tn, pn = t[near], p[near]
print(f'the same for positions the evaluator scores within +-1 logit (about +-2.7 pawns; {float(near.float().mean()):.0%} of all):')
print('  ' + '  '.join(f'{float(tn[(pn >= q[k]) & (pn <= q[k + 1])].mean()) * cp:.0f}' for k in range(10)))
torch.save({'W': W.cpu()}, 'zero/vol_head.pt')
