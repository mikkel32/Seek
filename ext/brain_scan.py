"""Look inside a trained network: what its hidden units do, how much of the evaluation each carries, what could be cut.

  python3 ext/brain_scan.py <network .ema.pt> [validation tags, default ival,gval] [positions per tag, default 400000]

For king-bucketed networks with the shared table (the BIGM/BIGS/BIGX family). Printed per data set:
  - how the evaluation splits between the piece-square shortcut and the hidden units
  - how many hidden units are never active, always saturated, or active in a small share of positions
  - how concentrated the evaluation is: the share of the hidden part's variance carried by the top units, and what
    happens to the held-out loss when the least important units are replaced by their average
  - the size of the numbers: weights and sums against what the engine's 16-bit arithmetic allows
"""
import math, sys
import numpy as np, torch
import torch.nn.functional as F

path = sys.argv[1]; tags = (sys.argv[2] if len(sys.argv) > 2 else 'ival,gval').split(','); NPOS = int(sys.argv[3]) if len(sys.argv) > 3 else 400000
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
w = {k: v.to(dev).float() for k, v in torch.load(path, map_location='cpu').items()}
H = w['b'].shape[0]; NKB = (w['E'].shape[0] - 1) // 768; NEX = w['wo'].shape[0]
assert NKB == 16 and 'Esh' in w, 'written for 16 mirrored king buckets with the shared table'
KBT = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13] + [14, 14, 15, 15] * 3, device=dev)
pc = torch.arange(768, device=dev) // 64; sq = torch.arange(768, device=dev) % 64
LUT = torch.cat([((pc + 6) % 12) * 64 + (sq ^ 56), torch.tensor([768], device=dev)])
PT_W = torch.cat([torch.tensor([0, 1, 1, 2, 4, 0] * 2, dtype=torch.float32, device=dev).repeat_interleave(64), torch.zeros(1, device=dev)])
Efull = torch.cat([w['E'][:NKB * 768] + w['Esh'][:768].repeat(NKB, 1), torch.zeros(1, H, device=dev)])


def kfeat(idx):
    isk = (idx >= 320) & (idx < 384); ks = ((idx * isk).sum(1) - 320).clamp(0, 63)
    mir = ((ks & 7) >= 4).long() * 7; kq = ks ^ mir; kb = KBT[(kq >> 3) * 4 + (kq & 7)]
    f = kb[:, None] * 768 + (idx // 64) * 64 + ((idx % 64) ^ mir[:, None])
    return torch.where(idx == 768, torch.full_like(idx, NKB * 768), f)


def hidden(idx):
    """-> pre-activations of both views, bucket, shortcut value"""
    ao = w['b'] + F.embedding_bag(kfeat(idx), Efull, mode='sum', padding_idx=NKB * 768)
    ap = w['b'] + F.embedding_bag(kfeat(LUT[idx]), Efull, mode='sum', padding_idx=NKB * 768)
    bk = (((idx < 768).sum(1) - 2) // 4).clamp(0, NEX - 1)
    lin = w['c'] + F.embedding_bag(idx, w['S'], mode='sum', padding_idx=768); ph = (PT_W[idx].sum(1) / 24).clamp(max=1.0)
    return ao, ap, bk, ph * lin[:, 0] + (1 - ph) * lin[:, 1]


def loss(z, y): return float(F.binary_cross_entropy_with_logits(z, y))


print(f'{path}: {H} hidden units a view, {NKB} king buckets, {NEX} read-out buckets')
# --- the numbers themselves
esh = w['Esh'][:768]; ecor = w['E'][:NKB * 768].view(NKB, 768, H)
print(f'input weights: shared table |w| mean {float(esh.abs().mean()):.4f} max {float(esh.abs().max()):.3f}; bucket corrections |w| mean {float(ecor.abs().mean()):.4f} max {float(ecor.abs().max()):.3f}; '
      f'corrections carry {100 * float((ecor ** 2).mean() / ((ecor ** 2).mean() + (esh ** 2).mean())):.0f}% of the input weights\' squared size')
print(f'read-out weights: |w| mean {float(w["wo"].abs().mean()):.3f} (own view) {float(w["wp"].abs().mean()):.3f} (other view), largest {float(max(w["wo"].abs().max(), w["wp"].abs().max())):.2f} (the engine clips at 8)')
# the engine's integer scale: bias + the 32 largest per-square maxima of |weight|, per unit and bucket
m = Efull[:NKB * 768].abs().view(NKB, 12, 64, H).max(1).values                    # (bucket, square, unit)
worst = (w['b'].abs() + m.topk(32, dim=1).values.sum(1)).max()                    # over buckets and units
qb = 10
while qb > 4 and float(worst) * (1 << qb) > 32000: qb -= 1
print(f'largest sum a unit could reach by the engine\'s bound: {float(worst):.1f} -> integer scale 2^{qb} (a unit\'s working range 0..1 has {1 << qb} steps)')

for tag in tags:
    X = torch.from_numpy(np.load(f'data/ev_{tag}_x.npy', mmap_mode='r')[:NPOS].astype(np.int64)); Y = torch.from_numpy(np.asarray(np.load(f'data/ev_{tag}_y.npy', mmap_mode='r')[:NPOS], dtype=np.float32))
    n = len(X); s1 = torch.zeros(2, H, device=dev); dead = torch.zeros(2, H, device=dev); sat = torch.zeros(2, H, device=dev); amax = torch.zeros(H, device=dev)
    csum = torch.zeros(2, H, device=dev); csq = torch.zeros(2, H, device=dev); outs = []; bases = []; nact = 0.0; nsat = 0.0
    with torch.no_grad():
        for i in range(0, n, 32768):
            idx = X[i:i + 32768].to(dev); ao, ap, bk, base = hidden(idx)
            amax = torch.maximum(amax, torch.maximum(ao.abs().max(0).values, ap.abs().max(0).values))
            for v, a_, wt in ((0, ao, w['wo']), (1, ap, w['wp'])):
                x = a_.clamp(0, 1) ** 2; c = x * wt[bk]
                s1[v] += x.sum(0); dead[v] += (a_ <= 0).float().sum(0); sat[v] += (a_ >= 1).float().sum(0); csum[v] += c.sum(0); csq[v] += (c ** 2).sum(0)
                nact += float((a_ > 0).float().mean()) * len(idx) / 2; nsat += float((a_ >= 1).float().mean()) * len(idx) / 2
            outs.append(base + ((ao.clamp(0, 1) ** 2) * w['wo'][bk]).sum(1) + ((ap.clamp(0, 1) ** 2) * w['wp'][bk]).sum(1) + w['cb'][bk]); bases.append(base)
    out = torch.cat(outs); base = torch.cat(bases); Yd = Y.to(dev)
    cmean = csum / n; cvar = csq / n - cmean ** 2; imp = cvar.sum(0)                 # a unit's importance: the variance of what it adds, both views
    order = torch.argsort(imp, descending=True); cum = torch.cumsum(imp[order], 0) / imp.sum()
    print(f'\n== {tag}: {n} positions; held-out loss {loss(out, Yd):.5f}')
    print(f'evaluation (logit units; 1 = 272 centipawns): spread of the whole {float(out.std()):.3f}, of the piece-square shortcut {float(base.std()):.3f}, of the hidden part {float((out - base).std()):.3f}')
    print(f'a position has on average {100 * nact / n:.1f}% of the units active (above zero) and {100 * nsat / n:.1f}% saturated (at one)')
    dfrac = dead.sum(0) / (2 * n); sfrac = sat.sum(0) / (2 * n)
    print(f'units never active: {int((dfrac >= 0.9999).sum())}; active in under 1% of positions: {int((dfrac >= 0.99).sum())}; under 5%: {int((dfrac >= 0.95).sum())}; '
          f'saturated in over 95%: {int((sfrac >= 0.95).sum())}; active in over 95%: {int((dfrac <= 0.05).sum())}')
    print('share of the hidden part\'s variance carried by the most important units: ' + ', '.join(f'top {k} {100 * float(cum[k - 1]):.1f}%' for k in (16, 64, 128, 256, 384, 512, 640) if k <= H))
    print(f'largest sum actually reached by any unit on these positions: {float(amax.max()):.2f} (median unit {float(amax.median()):.2f})')
    # replace the least important units by their average contribution and re-measure
    with torch.no_grad():
        line = []
        for cut in (0.1, 0.2, 0.3, 0.4, 0.5):
            k = int(cut * H); drop = order[H - k:]; keep = torch.ones(H, device=dev); keep[drop] = 0; z = []
            for i in range(0, n, 32768):
                idx = X[i:i + 32768].to(dev); ao, ap, bk, b0 = hidden(idx)
                co = (ao.clamp(0, 1) ** 2) * w['wo'][bk]; cp = (ap.clamp(0, 1) ** 2) * w['wp'][bk]
                z.append(b0 + (co * keep + cmean[0] * (1 - keep)).sum(1) + (cp * keep + cmean[1] * (1 - keep)).sum(1) + w['cb'][bk])
            line.append(f'{int(100 * cut)}% of the units averaged out: loss {loss(torch.cat(z), Yd):.5f}')
        print('; '.join(line))
# the piece-square shortcut in pawns: average over squares, middlegame / endgame
S = w['S'][:768].view(12, 64, 2) * 271.58
names = ['pawn', 'knight', 'bishop', 'rook', 'queen', 'king']
print('\nshortcut table, mean over squares in centipawns (own piece middlegame/endgame; opponent\'s the same with the sign turned): ' +
      ', '.join(f'{names[p]} {float(S[p, :, 0].mean()):.0f}/{float(S[p, :, 1].mean()):.0f} ({float(-S[p + 6, :, 0].mean()):.0f}/{float(-S[p + 6, :, 1].mean()):.0f})' for p in range(5)))
