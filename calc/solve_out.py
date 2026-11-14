"""Direct-weight experiment 2: re-solve the network's output layer exactly.

With the first layer frozen, the output weights of each piece-count bucket are a logistic regression on
513 features (2 x 256 hidden activations + bias). That problem is convex, so Newton steps find its exact
optimum; no gradient-descent training is involved. We then check on validation whether the solved layer
beats the trained one.

  python3 calc/solve_out.py calc/net256b.bin.ema.pt calc/net256b_solved.bin [N=4000000]
"""
import sys, math, struct, numpy as np, torch, torch.nn.functional as F

src, dst = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 4_000_000
torch.set_num_threads(4)
sd = torch.load(src)
H = sd['E'].shape[1]; NBK = sd['wo'].shape[0]
i = np.arange(768); pc, sq = i // 64, i % 64
LUT = torch.from_numpy(np.concatenate([((pc + 6) % 12) * 64 + (sq ^ 56), [768]]).astype(np.int64))
PT_W = torch.cat([torch.tensor([0, 1, 1, 2, 4, 0] * 2, dtype=torch.float32).repeat_interleave(64), torch.zeros(1)])


@torch.no_grad()
def feats(idx):
    ao = sd['b'] + F.embedding_bag(idx, sd['E'], mode='sum', padding_idx=768)
    ap = sd['b'] + F.embedding_bag(LUT[idx], sd['E'], mode='sum', padding_idx=768)
    lin = sd['c'] + F.embedding_bag(idx, sd['S'], mode='sum', padding_idx=768)
    ph = (PT_W[idx].sum(1) / 24).clamp(max=1.0)
    off = ph * lin[:, 0] + (1 - ph) * lin[:, 1]
    bk = (((idx < 768).sum(1) - 2) // 4).clamp(0, NBK - 1)
    Z = torch.cat([ao.clamp(0, 1) ** 2, ap.clamp(0, 1) ** 2, torch.ones(len(idx), 1)], 1)
    return Z, off, bk


def logloss(theta, X, Y):
    tot = 0.0; n = 0
    for j in range(0, len(X), 50000):
        idx = torch.from_numpy(X[j:j + 50000].astype(np.int64)); y = torch.from_numpy(np.asarray(Y[j:j + 50000], dtype=np.float32))
        Z, off, bk = feats(idx)
        out = off + (Z * theta[bk]).sum(1)
        tot += float(F.binary_cross_entropy_with_logits(out, y, reduction='sum')); n += len(idx)
    return tot / n


theta = torch.cat([sd['wo'], sd['wp'], sd['cb'][:, None]], 1).clone()           # (NBK, 2H+1), the trained output layer
VX = np.load('data/ev_val_x.npy'); VY = np.load('data/ev_val_y.npy')
X = np.load('data/ev_d_x.npy', mmap_mode='r')[:N]; Y = np.load('data/ev_d_y.npy', mmap_mode='r')[:N]
print(f'validation log-loss, trained output layer: {logloss(theta, VX, VY):.6f}', flush=True)
D = 2 * H + 1
for it in range(4):
    Hs = torch.zeros(NBK, D, D, dtype=torch.float64); gs = torch.zeros(NBK, D, dtype=torch.float64)
    for j in range(0, N, 100_000):
        idx = torch.from_numpy(X[j:j + 100_000].astype(np.int64)); y = torch.from_numpy(Y[j:j + 100_000].astype(np.float32)) / 255
        Z, off, bk = feats(idx)
        p = torch.sigmoid(off + (Z * theta[bk]).sum(1))
        for k in range(NBK):
            m = bk == k
            if m.any():
                Zk = Z[m]; pk = p[m]
                gs[k] += (Zk.T @ (pk - y[m])).double(); Hs[k] += ((Zk * (pk * (1 - pk))[:, None]).T @ Zk).double()
    for k in range(NBK):
        lam = 1e-2
        step = torch.linalg.solve(Hs[k] + lam * torch.eye(D, dtype=torch.float64), gs[k] + lam * theta[k].double())
        theta[k] -= step.float()
    print(f'newton step {it}: validation log-loss {logloss(theta, VX, VY):.6f}', flush=True)

with open(dst, 'wb') as f:
    f.write(struct.pack('<iii', 0x4F555232, H, NBK))
    f.write(sd['E'][:768].numpy().astype('<f4').tobytes()); f.write(sd['b'].numpy().astype('<f4').tobytes())
    for k in range(NBK):
        f.write(theta[k, :H].numpy().astype('<f4').tobytes()); f.write(theta[k, H:2 * H].numpy().astype('<f4').tobytes())
    f.write(struct.pack('<f', float(sd['c'][0]))); f.write(sd['S'][:768, 0].numpy().astype('<f4').tobytes())
    f.write(struct.pack('<f', float(sd['c'][1]))); f.write(sd['S'][:768, 1].numpy().astype('<f4').tobytes())
    f.write(theta[:, 2 * H].numpy().astype('<f4').tobytes())
print('written', dst)
