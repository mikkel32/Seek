"""Which is the better static judge of a position on self-play data neither has seen: the calculator's network or OURO?

Positions: played positions from fresh games (python3 zero/judge_cmp.py <network .ema.pt> <ouro .pt> <glob of raw files> [max positions]).
Reported: log-loss against the game result, against the search score, and against the production target (half and half).
"""
import glob, sys, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, '.')
from model import Ouro

REC = np.dtype([('idx', '<i2', 32), ('score', '<i2'), ('result', 'i1'), ('ply', 'i1')])
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sd = torch.load(sys.argv[1], map_location=DEV); ck = torch.load(sys.argv[2], map_location=DEV)
NMAX = int(sys.argv[4]) if len(sys.argv) > 4 else 200_000
files = sorted(glob.glob(sys.argv[3])); rs = np.random.RandomState(0); rs.shuffle(files)
r = np.concatenate([np.fromfile(f, dtype=REC) for f in files[:400]]); r = r[r['ply'] >= 0]
r = r[rs.permutation(len(r))[:NMAX]]
idx = torch.from_numpy(r['idx'].astype(np.int64)).to(DEV)
wp = torch.sigmoid(torch.from_numpy(r['score'].astype(np.float32)).to(DEV) / 271.58); z = (torch.from_numpy(r['result'].astype(np.float32)).to(DEV) + 1) / 2
i = np.arange(768); LUT = torch.from_numpy(np.concatenate([((i // 64 + 6) % 12) * 64 + ((i % 64) ^ 56), [768]]).astype(np.int64)).to(DEV)
PT_W = torch.cat([torch.tensor([0, 1, 1, 2, 4, 0] * 2, dtype=torch.float32).repeat_interleave(64), torch.zeros(1)]).to(DEV)


@torch.no_grad()
def nnue(idx):
    ao = sd['b'] + F.embedding_bag(idx, sd['E'], mode='sum', padding_idx=768); ap = sd['b'] + F.embedding_bag(LUT[idx], sd['E'], mode='sum', padding_idx=768)
    lin = sd['c'] + F.embedding_bag(idx, sd['S'], mode='sum', padding_idx=768)
    ph = (PT_W[idx].sum(1) / 24).clamp(max=1.0); bk = (((idx < 768).sum(1) - 2) // 4).clamp(0, sd['wo'].shape[0] - 1)
    return ph * lin[:, 0] + (1 - ph) * lin[:, 1] + sd['cb'][bk] + (ao.clamp(0, 1) ** 2 * sd['wo'][bk]).sum(1) + (ap.clamp(0, 1) ** 2 * sd['wp'][bk]).sum(1)


@torch.no_grad()
def ouro(idx, loops):
    board = torch.zeros(len(idx), 65, dtype=torch.long, device=DEV)
    board.scatter_(1, torch.where(idx < 768, idx % 64, torch.full_like(idx, 64)), torch.where(idx < 768, idx // 64 + 1, torch.zeros_like(idx)))
    board = board[:, :64]
    pol, v, ql, mask = model(board, torch.zeros(len(idx), dtype=torch.long, device=DEV), torch.full((len(idx),), 64, dtype=torch.long, device=DEV), loops=loops)
    return v.float()


model = Ouro(**ck['cfg']).to(DEV); model.load_state_dict(ck['ema']); model.eval()
a = torch.cat([nnue(idx[j:j + 20000]) for j in range(0, len(idx), 20000)])
print(f'{len(idx)} fresh played positions; OURO at step {ck.get("step")}; network hidden {sd["E"].shape[1]}')
ll = lambda logit, t: float(F.binary_cross_entropy_with_logits(logit, t))
row = lambda name, lg: print(f'  {name:34s} vs result {ll(lg, z):.4f}   vs search score {ll(lg, wp):.4f}   vs production target {ll(lg, 0.5 * wp + 0.5 * z):.4f}')
row('calculator network (static)', a)
row('search score itself (5000 nodes)', torch.logit(wp.clamp(1e-4, 1 - 1e-4)))
for loops in (1, 2, 4):
    o = torch.cat([ouro(idx[j:j + 500], loops) for j in range(0, len(idx), 500)])
    row(f'OURO, {loops} loop(s)', o)
    if loops == 2:
        row('average of network and OURO', (a + o) / 2)
        c = float(torch.corrcoef(torch.stack([a - torch.logit((0.5 * wp + 0.5 * z).clamp(0.02, 0.98)), o - torch.logit((0.5 * wp + 0.5 * z).clamp(0.02, 0.98))]))[0, 1])
        print(f'  correlation of their errors: {c:.3f}')

# ---- the boring checks: (1) recalibrate every judge to the result (scale and offset fitted on one half, tested on the
# other), so nobody loses for being over- or under-confident; (2) does OURO add anything once the search score is known?
o = torch.cat([ouro(idx[j:j + 500], 1) for j in range(0, len(idx), 500)])
s = torch.logit(wp.clamp(1e-4, 1 - 1e-4))
half = len(idx) // 2


def fit(cols):
    X = torch.stack(cols + [torch.ones_like(z)], 1).double(); w = torch.zeros(X.shape[1], dtype=torch.float64, device=DEV)
    for _ in range(30):                                  # Newton steps on a convex problem
        p = torch.sigmoid(X[:half] @ w); g = X[:half].T @ (p - z[:half].double())
        Hm = (X[:half] * (p * (1 - p))[:, None]).T @ X[:half] + 1e-6 * torch.eye(X.shape[1], dtype=torch.float64, device=DEV)
        w = w - torch.linalg.solve(Hm, g)
    return float(F.binary_cross_entropy_with_logits(X[half:] @ w, z[half:].double())), [round(float(x), 3) for x in w]


print('recalibrated to the result (fitted on half, tested on the other half):')
for name, cols in (('calculator network', [a]), ('search score', [s]), ('OURO', [o]), ('network + search', [a, s]), ('search + OURO', [s, o]), ('network + OURO', [a, o]), ('all three', [a, s, o])):
    l, w = fit(cols); print(f'  {name:20s} log-loss vs result {l:.4f}   weights {w}')
# by phase of the game: where does OURO know more than the search?
npc = (idx < 768).sum(1)
for lo, hi in ((2, 10), (11, 20), (21, 28), (29, 32)):
    m = (npc >= lo) & (npc <= hi)
    print(f'  pieces {lo:2d}-{hi:2d}: {int(m.sum()):6d} positions   network {ll(a[m], z[m]):.4f}   search {ll(s[m], z[m]):.4f}   OURO {ll(o[m], z[m]):.4f}')
