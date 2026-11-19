"""Two-brain teacher: better training targets for the calculator's network from OURO + the search.

On fresh self-play positions the game result is predicted best by
    OURO's static value  +  what the search found beyond the static network        (see zero/judge_cmp.py)
This script fits that combination on the played positions (whose results are known), then writes it as the training
target for every position, played or from inside the trees. A control target uses the same recipe WITHOUT OURO, so
a match between the two networks isolates what OURO contributes.

  python3 zero/teacher.py <tag> <network .ema.pt> <ouro .pt> [rows=20000000]
  reads  data/ev_<tag>_{x,y,s,pl}.npy (zero/foresight.py)      writes data/ev_<tag>T_{x,y,yN,yO}.npy
"""
import sys, time, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, '.')
from model import Ouro

tag = sys.argv[1]; ROWS = int(sys.argv[4]) if len(sys.argv) > 4 else 20_000_000
DEV = torch.device('cuda'); sd = torch.load(sys.argv[2], map_location=DEV); ck = torch.load(sys.argv[3], map_location=DEV)
x = np.load(f'data/ev_{tag}_x.npy', mmap_mode='r')[:ROWS]; n = len(x)
y = np.load(f'data/ev_{tag}_y.npy', mmap_mode='r')[:ROWS].astype(np.float32) / 255
s8 = np.load(f'data/ev_{tag}_s.npy', mmap_mode='r')[:ROWS].astype(np.float32) / 255; pl = np.load(f'data/ev_{tag}_pl.npy', mmap_mode='r')[:ROWS] > 0
i = np.arange(768); LUT = torch.from_numpy(np.concatenate([((i // 64 + 6) % 12) * 64 + ((i % 64) ^ 56), [768]]).astype(np.int64)).to(DEV)
PT_W = torch.cat([torch.tensor([0, 1, 1, 2, 4, 0] * 2, dtype=torch.float32).repeat_interleave(64), torch.zeros(1)]).to(DEV)
model = Ouro(**ck['cfg']).to(DEV); model.load_state_dict(ck['ema']); model.eval()


@torch.no_grad()
def nnue(idx):
    ao = sd['b'] + F.embedding_bag(idx, sd['E'], mode='sum', padding_idx=768); ap = sd['b'] + F.embedding_bag(LUT[idx], sd['E'], mode='sum', padding_idx=768)
    lin = sd['c'] + F.embedding_bag(idx, sd['S'], mode='sum', padding_idx=768)
    ph = (PT_W[idx].sum(1) / 24).clamp(max=1.0); bk = (((idx < 768).sum(1) - 2) // 4).clamp(0, sd['wo'].shape[0] - 1)
    return ph * lin[:, 0] + (1 - ph) * lin[:, 1] + sd['cb'][bk] + (ao.clamp(0, 1) ** 2 * sd['wo'][bk]).sum(1) + (ap.clamp(0, 1) ** 2 * sd['wp'][bk]).sum(1)


@torch.no_grad()
def ouro(idx):
    board = torch.zeros(len(idx), 65, dtype=torch.long, device=DEV)
    board.scatter_(1, torch.where(idx < 768, idx % 64, torch.full_like(idx, 64)), torch.where(idx < 768, idx // 64 + 1, torch.zeros_like(idx)))
    return model(board[:, :64], torch.zeros(len(idx), dtype=torch.long, device=DEV), torch.full((len(idx),), 64, dtype=torch.long, device=DEV), loops=1)[1].float()


a = np.zeros(n, dtype=np.float32); o = np.zeros(n, dtype=np.float32); t0 = time.time(); B = 1024
for j in range(0, n, 65536):
    idx = torch.from_numpy(np.ascontiguousarray(x[j:j + 65536]).astype(np.int64)).to(DEV)
    a[j:j + 65536] = nnue(idx).cpu().numpy()
    o[j:j + 65536] = torch.cat([ouro(idx[k:k + B]) for k in range(0, len(idx), B)]).cpu().numpy()
    if (j // 65536) % 20 == 0:
        print(f'  {j/1e6:.1f}M of {n/1e6:.1f}M positions judged, {(j + 65536)/(time.time() - t0):.0f} per second', flush=True)
s = np.log(np.clip(s8, 0.004, 0.996) / (1 - np.clip(s8, 0.004, 0.996))).astype(np.float32)
z = np.clip(np.round(2 * (2 * y - s8)) / 2, 0, 1).astype(np.float32)          # played rows: the result, recovered from target = (score + result) / 2
P = np.nonzero(pl)[0]; half = len(P) // 2
T = lambda v: torch.from_numpy(v).double()


def fit(cols):
    X = torch.stack([T(c[P]) for c in cols] + [torch.ones(len(P), dtype=torch.float64)], 1); zz = T(z[P]); w = torch.zeros(X.shape[1], dtype=torch.float64)
    for _ in range(30):                                  # Newton steps on a convex problem, first half of the played rows
        p = torch.sigmoid(X[:half] @ w); g = X[:half].T @ (p - zz[:half])
        w = w - torch.linalg.solve((X[:half] * (p * (1 - p))[:, None]).T @ X[:half] + 1e-6 * torch.eye(X.shape[1], dtype=torch.float64), g)
    return w.numpy(), float(F.binary_cross_entropy_with_logits(X[half:] @ w, zz[half:]))


wN, lN = fit([a, s]); wO, lO = fit([a, s, o]); _, lS = fit([s]); _, lA = fit([a]); _, lOo = fit([o])
print(f'{n} positions, {len(P)} played. Held-out log-loss against the result: network {lA:.4f}  search {lS:.4f}  OURO {lOo:.4f}  network+search {lN:.4f}  network+search+OURO {lO:.4f}')
print(f'weights without OURO (network, search, offset) {np.round(wN, 3)}   with OURO (network, search, OURO, offset) {np.round(wO, 3)}')
sig = lambda v: 1 / (1 + np.exp(-v))
tN = sig(wN[0] * a + wN[1] * s + wN[2]); tO = sig(wO[0] * a + wO[1] * s + wO[2] * o + wO[3])
tN[pl] = 0.75 * tN[pl] + 0.25 * z[pl]; tO[pl] = 0.75 * tO[pl] + 0.25 * z[pl]     # played positions keep a quarter of the actual result
b8 = lambda v: np.round(255 * np.clip(v, 0, 1)).astype(np.uint8)
np.save(f'data/ev_{tag}T_x.npy', np.ascontiguousarray(x)); np.save(f'data/ev_{tag}T_y.npy', b8(y)); np.save(f'data/ev_{tag}T_yN.npy', b8(tN)); np.save(f'data/ev_{tag}T_yO.npy', b8(tO))
print(f'mean |target - production target|: without OURO {np.abs(tN - y).mean():.4f}, with OURO {np.abs(tO - y).mean():.4f}; with vs without OURO {np.abs(tO - tN).mean():.4f}')
