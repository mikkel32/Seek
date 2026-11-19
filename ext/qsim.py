"""How far is the engine's integer network from the trained one on REAL positions, and what would finer integers buy?
   python3 ext/qsim.py <.ema.pt> [positions=20000]"""
import sys, numpy as np, torch
w = {k: v.numpy().astype(np.float64) for k, v in torch.load(sys.argv[1], map_location='cpu').items()}; n = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
E = w['E'][:-1].copy(); NKB = E.shape[0] // 768; H = E.shape[1]
if 'Esh' in w:
    E += np.tile(w['Esh'][:768], (NKB, 1))
b = w['b']; wo, wp, cb = w['wo'], w['wp'], w['cb']; nb = wo.shape[0]
X = np.load('data/ev_ival_x.npy', mmap_mode='r')[:n].astype(np.int64)
KBT = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13] + [14, 14, 15, 15] * 3)
i = np.arange(768); LUT = np.concatenate([((i // 64 + 6) % 12) * 64 + ((i % 64) ^ 56), [768]])
def rows(ix):
    """(n,32) plain view indices -> rows of the input table (padding -> -1)"""
    if NKB == 1:
        return np.where(ix == 768, -1, ix)
    isk = (ix >= 320) & (ix < 384); ks = np.clip((ix * isk).sum(1) - 320, 0, 63); mir = ((ks & 7) >= 4) * 7; kq = ks ^ mir
    kb = KBT[(kq >> 3) * 4 + (kq & 7)]
    return np.where(ix == 768, -1, kb[:, None] * 768 + (ix // 64) * 64 + ((ix % 64) ^ mir[:, None]))
ro, rp = rows(X), rows(LUT[X]); bk = np.clip(((X < 768).sum(1) - 2) // 4, 0, nb - 1)
def acc(T, r):                                  # sum of table rows (padding row = zeros)
    Tz = np.vstack([T, np.zeros((1, T.shape[1]), T.dtype)]); return Tz[r].sum(1)
ao, ap_ = b + acc(E, ro), b + acc(E, rp)
ref = (np.clip(ao, 0, 1) ** 2 * wo[bk]).sum(1) + (np.clip(ap_, 0, 1) ** 2 * wp[bk]).sum(1)
bal = np.abs(ref + cb[bk]) * 271.58 < 300
wq_o, wq_p = np.round(np.clip(wo, -8, 8) * 2048) / 2048, np.round(np.clip(wp, -8, 8) * 2048) / 2048
def report(name, qa_o, qa_p, base):
    """accumulators given in real units (already rounded); the activation is resolved in steps of 1/base"""
    xo, xp = np.clip(np.round(qa_o * base) / base, 0, 1), np.clip(np.round(qa_p * base) / base, 0, 1)
    out = (xo ** 2 * wq_o[bk]).sum(1) + (xp ** 2 * wq_p[bk]).sum(1); d = (out - ref) * 271.58
    print(f'  {name:62s} error: mean |.| {np.abs(d).mean():5.2f} cp, rms {np.sqrt((d ** 2).mean()):5.2f}; balanced positions (within 300 cp, {bal.mean():.0%}): mean |.| {np.abs(d[bal]).mean():5.2f}, rms {np.sqrt((d[bal] ** 2).mean()):5.2f}', flush=True)
print(f'{sys.argv[1]}: {n} held-out Lichess positions, {NKB} king buckets, width {H}')
report('output weights in steps of 1/2048 only (exact hidden layer)', ao, ap_, 1 << 20)
for q in (7, 8, 9, 10):
    Eq = np.round(E * (1 << q)) / (1 << q); bq = np.round(b * (1 << q)) / (1 << q)
    report(f'everything in steps of 1/{1 << q}' + (' (the engine now)' if q == 7 else ''), bq + acc(Eq, ro), bq + acc(Eq, rp), 1 << q)
# per-unit scale: the finest of 1/128 .. 1/1024 at which the unit's worst case fits 16 bits; activation resolved at 1/128
m = np.abs(E.reshape(NKB, 12, 64, H)).max(1); worst = (np.abs(b)[None] + np.sort(m, axis=1)[:, -32:, :].sum(1)).max(0)
qi = np.full(H, 7)
for q in (8, 9, 10):
    qi[worst * (1 << q) <= 32000] = q
sc = (1 << qi).astype(np.float64); Eq = np.round(E * sc) / sc; bq = np.round(b * sc) / sc
print('  units at 1/128, 1/256, 1/512, 1/1024:', [(qi == q).sum() for q in (7, 8, 9, 10)])
report('per-unit steps, activation resolved at 1/128', bq + acc(Eq, ro), bq + acc(Eq, rp), 128)
report('per-unit steps, activation resolved at 1/256', bq + acc(Eq, ro), bq + acc(Eq, rp), 256)
