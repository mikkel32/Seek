"""Evaluation weights by direct solve — no training loop, no gradient descent.

Model: logit(win probability) = phase * (c_mg + sum S_mg[piece, square]) + (1 - phase) * (c_eg + sum S_eg[piece, square])
1,538 numbers. The log-loss of this model is convex, so its exact optimum is found with a handful of
Newton steps, each one a linear solve of a 1538 x 1538 system built from N labelled positions.

  python3 calc/solve_pst.py [N=2000000] -> calc/pst.bin (engine format), calc/pst.npz
"""
import sys, time, struct, numpy as np, torch

N = int(sys.argv[1]) if len(sys.argv) > 1 else 2_000_000
TAGS = sys.argv[2].split(',') if len(sys.argv) > 2 else ['b']          # data files to solve on
OUT = sys.argv[3] if len(sys.argv) > 3 else 'calc/pst'
torch.set_num_threads(4)
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')      # the dense linear algebra is far faster on a GPU
X = np.concatenate([np.load(f'data/ev_{t}_x.npy', mmap_mode='r')[:N] for t in TAGS])[:N]
Y = np.concatenate([np.load(f'data/ev_{t}_y.npy', mmap_mode='r')[:N] for t in TAGS])[:N]
N = len(X)
PT_W = torch.tensor([0, 1, 1, 2, 4, 0] * 2, dtype=torch.float32).repeat_interleave(64)       # phase weight of each feature
PT_W = torch.cat([PT_W, torch.zeros(1)]).to(DEV)
D = 2 * 769                                              # [mg table (768) + const, eg table (768) + const]


def design(idx):
    """idx (n,32) long -> dense design matrix (n, D) float32"""
    n = idx.shape[0]
    cnt = torch.zeros(n, 769, device=DEV).scatter_add_(1, idx, torch.ones(n, 32, device=DEV))
    ph = (PT_W[idx].sum(1) / 24).clamp(max=1.0)[:, None]
    cnt[:, 768] = 1.0                                    # the padding column becomes the constant
    return torch.cat([cnt * ph, cnt * (1 - ph)], 1)


theta = torch.zeros(D, dtype=torch.float64, device=DEV)
lam = 1e-3 * N / 1e6
for it in range(7):
    t = time.time()
    H = torch.zeros(D, D, dtype=torch.float64, device=DEV); g = torch.zeros(D, dtype=torch.float64, device=DEV); loss = 0.0
    for i in range(0, N, 100_000):
        idx = torch.from_numpy(X[i:i + 100_000].astype(np.int64)).to(DEV); y = torch.from_numpy(Y[i:i + 100_000].astype(np.float32)).to(DEV) / 255
        Z = design(idx)
        p = torch.sigmoid(Z @ theta.float())
        loss += float(-(y * torch.log(p + 1e-9) + (1 - y) * torch.log(1 - p + 1e-9)).sum())
        g += (Z.T @ (p - y)).double()
        H += ((Z * (p * (1 - p))[:, None]).T @ Z).double()
    g += lam * theta; H += lam * torch.eye(D, dtype=torch.float64, device=DEV)
    step = torch.linalg.solve(H, g)
    theta -= step
    print(f'newton step {it}: log-loss before step {loss / N:.5f}  |step| {float(step.norm()):.4f}  ({time.time() - t:.0f}s)', flush=True)
    if float(step.norm()) < 1e-3:
        break

th = theta.float().cpu().numpy()
S_mg, c_mg, S_eg, c_eg = th[:768], th[768], th[769:1537], th[1537]
# piece values as this data sees them (own minus opponent, average of both tables), handed to the search for exchanges and move ordering
def pval(S, i):
    lo, hi = (8, 56) if i == 0 else (0, 64)
    return (S[i * 64 + lo:i * 64 + hi].mean() - S[(i + 6) * 64 + lo:(i + 6) * 64 + hi].mean()) / 2 * 271.58
PV = [int(min(2500, max(30, round((pval(S_mg, i) + pval(S_eg, i)) / 2)))) for i in range(5)]
np.savez(OUT + '.npz', S_mg=S_mg, c_mg=c_mg, S_eg=S_eg, c_eg=c_eg, pval=np.array(PV))
with open(OUT + '.bin', 'wb') as f:
    f.write(struct.pack('<ii', 0x4F55524F, 0))
    f.write(struct.pack('<f', c_mg)); f.write(S_mg.astype('<f4').tobytes())
    f.write(struct.pack('<f', c_eg)); f.write(S_eg.astype('<f4').tobytes())
    f.write(struct.pack('<i5i', 0x4C415650, *PV))
print('piece values handed to the search (centipawns, P N B R Q):', PV)
# what the solve says a piece is worth (own pieces, average over squares, in pawns of the middlegame table)
names = 'PNBRQK'
pawn = S_mg[0 * 64 + 8:0 * 64 + 56].mean()
print('solved piece values (middlegame, own side, pawn = 1):', {names[i]: round(float(S_mg[i * 64:(i + 1) * 64].mean() / pawn), 2) for i in range(5)})
print('tempo (side to move) in centipawns: mg %.0f eg %.0f' % (c_mg * 271.58, c_eg * 271.58))
