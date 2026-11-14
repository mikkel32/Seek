"""How much can expert routing add on top of a given hidden layer? Measured with exact solves, no gradient training.

Hidden layer frozen. Baseline: the single output head per piece-count group, re-solved exactly on the training
positions (logistic regression is convex, Newton steps reach the optimum). Experts: M heads per group and a
linear router, fitted by hard EM:
    1. solve each expert on the positions currently routed to it
    2. find, for every position, the expert that predicts it best
    3. fit the router (a convex problem) to predict that expert from the hidden activations
    4. route by the router, repeat
Reported: validation log-loss with router-chosen experts against the single-head baseline, on positions from a
generation the network never trained on.

  python3 zero/moe_em.py <net.ema.pt> <train tag> <val tag> [M=2] [N=1500000]
"""
import sys, numpy as np, torch, torch.nn.functional as F

torch.set_num_threads(3); torch.manual_seed(0)
sd = torch.load(sys.argv[1]); TR, VA = sys.argv[2], sys.argv[3]
M = int(sys.argv[4]) if len(sys.argv) > 4 else 2
FULL = len(sys.argv) > 6 and sys.argv[6] == "full"              # router reads every hidden unit instead of 32 per side
N = int(sys.argv[5]) if len(sys.argv) > 5 else 1_500_000
H = sd['E'].shape[1]; NB = 8; G = min(32, H)
r0 = sd['wo'].shape[0] // NB                                     # experts per group in the given network (1 for a plain one)
i = np.arange(768); pc, sq = i // 64, i % 64
LUT = torch.from_numpy(np.concatenate([((pc + 6) % 12) * 64 + (sq ^ 56), [768]]).astype(np.int64))
PT_W = torch.cat([torch.tensor([0, 1, 1, 2, 4, 0] * 2, dtype=torch.float32).repeat_interleave(64), torch.zeros(1)])


@torch.no_grad()
def feats(x):
    Z, O, B = [], [], []
    for j in range(0, len(x), 100_000):
        idx = torch.from_numpy(np.asarray(x[j:j + 100_000]).astype(np.int64))
        ao = sd['b'] + F.embedding_bag(idx, sd['E'], mode='sum', padding_idx=768); ap = sd['b'] + F.embedding_bag(LUT[idx], sd['E'], mode='sum', padding_idx=768)
        lin = sd['c'] + F.embedding_bag(idx, sd['S'], mode='sum', padding_idx=768)
        ph = (PT_W[idx].sum(1) / 24).clamp(max=1.0)
        Z.append(torch.cat([ao.clamp(0, 1) ** 2, ap.clamp(0, 1) ** 2, torch.ones(len(idx), 1)], 1))
        O.append(ph * lin[:, 0] + (1 - ph) * lin[:, 1]); B.append((((idx < 768).sum(1) - 2) // 4).clamp(0, NB - 1))
    return torch.cat(Z), torch.cat(O), torch.cat(B)


def solve(Z, off, y, theta, iters=6, lam=1e-2, w=None):
    """exact logistic regression with an offset: Newton steps from `theta`"""
    D = Z.shape[1]
    for _ in range(iters):
        p = torch.sigmoid(off + Z @ theta)
        g = Z.T @ ((p - y) if w is None else (p - y) * w)
        hw = p * (1 - p) if w is None else p * (1 - p) * w
        Hm = (Z * hw[:, None]).T @ Z
        theta = theta - torch.linalg.solve(Hm.double() + lam * torch.eye(D, dtype=torch.float64), (g + lam * theta).double()).float()
    return theta


def bce(logit, y):
    return F.binary_cross_entropy_with_logits(logit, y, reduction='none')


xt = np.load(f'data/ev_{TR}_x.npy', mmap_mode='r')[:N]; yt = torch.from_numpy(np.load(f'data/ev_{TR}_y.npy', mmap_mode='r')[:N].astype(np.float32)) / 255
xv = np.load(f'data/ev_{VA}_x.npy'); yv = torch.from_numpy(np.load(f'data/ev_{VA}_y.npy').astype(np.float32))
Zt, Ot, Bt = feats(xt); Zv, Ov, Bv = feats(xv)
gate_cols = list(range(2 * H + 1)) if FULL else list(range(G)) + list(range(H, H + G)) + [2 * H]
head0 = torch.cat([sd['wo'], sd['wp'], sd['cb'][:, None]], 1)[::r0].clone()       # (8, 2H+1) the network's own heads

# the network as trained (no re-solve), for reference
lv_net = bce(Ov + (Zv * head0[Bv]).sum(1), yv)
# baseline: one head per group, solved exactly on the training positions
base = head0.clone()
for b in range(NB):
    m = Bt == b
    if m.sum() > 2000:
        base[b] = solve(Zt[m], Ot[m], yt[m], head0[b].clone())
lv_base = bce(Ov + (Zv * base[Bv]).sum(1), yv)

# experts by hard EM
exp = base[:, None, :].repeat(1, M, 1).clone()                  # (8, M, D)
gate = torch.zeros(NB, M, len(gate_cols))
route_t = torch.zeros(len(yt), dtype=torch.long)
for b in range(NB):                                             # initial split: along the main direction of variation of the activations
    m = (Bt == b).nonzero().squeeze(1)
    if len(m) < 2000 * M:
        continue
    Zc = Zt[m][:, :2 * H]; Zc = Zc - Zc.mean(0)
    u = torch.pca_lowrank(Zc[:50000], q=2)[2]
    proj = Zc @ u[:, 0]
    route_t[m] = torch.bucketize(proj, torch.quantile(proj[:200000], torch.linspace(0, 1, M + 1)[1:-1]))
for rnd in range(8):
    for b in range(NB):
        mb = (Bt == b).nonzero().squeeze(1)
        if len(mb) < 2000 * M:
            continue
        Z, O, y = Zt[mb], Ot[mb], yt[mb]
        for k in range(M):                                      # 1. each expert on its own positions
            mk = route_t[mb] == k
            if mk.sum() > 1000:
                exp[b, k] = solve(Z[mk], O[mk], y[mk], exp[b, k].clone(), iters=3)
        losses = torch.stack([bce(O + Z @ exp[b, k], y) for k in range(M)], 1)      # 2. who predicts each position best
        best = losses.argmin(1); gain = losses.max(1).values - losses.min(1).values
        Zg = Z[:, gate_cols]
        if M == 2:                                              # 3. router: logistic regression towards the better expert, weighted by how much it matters
            th = solve(Zg, torch.zeros(len(y)), (best == 1).float(), gate[b, 1] - gate[b, 0], iters=4, lam=1e-1, w=gain / (gain.mean() + 1e-9))
            gate[b, 0] = 0; gate[b, 1] = th
        else:                                                   # one-vs-rest routers
            for k in range(M):
                gate[b, k] = solve(Zg, torch.zeros(len(y)), (best == k).float(), gate[b, k].clone(), iters=4, lam=1e-1, w=gain / (gain.mean() + 1e-9))
        route_t[mb] = (Zg @ gate[b].T).argmax(1)                # 4. route by the router
    rv = torch.zeros(len(yv), dtype=torch.long)
    for b in range(NB):
        mv = Bv == b
        rv[mv] = (Zv[mv][:, gate_cols] @ gate[b].T).argmax(1)
    lv = bce(Ov + (Zv * exp[Bv, rv]).sum(1), yv)
    d = lv - lv_base
    use = [round(float((rv[Bv == b] == 0).float().mean()), 2) if (Bv == b).any() else None for b in range(NB)]
    print(f'round {rnd}: validation log-loss experts {lv.mean():.5f}  single head {lv_base.mean():.5f}  difference {d.mean():+.5f} +- {2 * d.std() / len(d) ** 0.5:.5f}  share routed to expert 0 per group {use}', flush=True)
print(f'network as trained {lv_net.mean():.5f} | single head re-solved {lv_base.mean():.5f} | {M} experts per group {lv.mean():.5f}')
torch.save({'exp': exp, 'gate': gate, 'base': base}, f'zero/moe_em_M{M}.pt')
