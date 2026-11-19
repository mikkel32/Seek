"""Two-brain teacher, in the production score scale.

The fast evaluator learns from labels made by its own search. OURO (the looped transformer) judges a position without
search about as well as that search does, from a different angle. Here the label for EVERY position, played or from inside
the search trees, becomes the best combination of three opinions: the fast evaluator itself, the search score, and OURO.
The combination is fitted on played positions against the game result (first half), then brought back to the scale of the
search score, so that the search's margins (which are in that scale) stay right and the networks can be compared directly.

  yN   the same recipe without OURO (the control: isolates what OURO adds)
  yO   with OURO
  y95  production's label (lambda-return, lambda 0.95) for the same positions

  python3 zero/teacher2.py <tag> <folder of raw files> <network .ema.pt> <ouro .pt> [processes=8]
  writes data/ev_<tag>_x.npy, _y95.npy, _yN.npy, _yO.npy  (one shuffle for all)
"""
import glob, os, re, sys, time
import numpy as np, torch, torch.nn.functional as F
from multiprocessing import Pool

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')); sys.path.insert(0, '.')
REC = np.dtype([('idx', '<i2', 32), ('score', '<i2'), ('result', 'i1'), ('ply', 'i1')]); LAM = 0.5
src = open('zero/cont.py').read(); exec(re.search(r'(def target\(q\):.*?)(?=\ndef make_shard)', src, re.S).group(1))


def one(path):
    r = np.fromfile(path, dtype=REC)
    return r['idx'], lam_targets(r, 0.95).astype(np.float32), (r['score'].astype(np.float32) / 271.58), ((r['result'].astype(np.float32) + 1) / 2), r['ply'] >= 0


if __name__ == '__main__':
    tag = sys.argv[1]; files = sorted(glob.glob(sys.argv[2] + '/*.bin')); procs = int(sys.argv[5]) if len(sys.argv) > 5 else 8
    with Pool(procs) as pool:
        res = pool.map(one, files, chunksize=4)
    x = np.concatenate([r[0] for r in res]); y95 = np.concatenate([r[1] for r in res]); s = np.clip(np.concatenate([r[2] for r in res]), -12, 12)
    z = np.concatenate([r[3] for r in res]); pl = np.concatenate([r[4] for r in res]); n = len(x); del res
    DEV = torch.device('cuda'); sd = torch.load(sys.argv[3], map_location=DEV); ck = torch.load(sys.argv[4], map_location=DEV)
    from model import Ouro
    model = Ouro(**ck['cfg']).to(DEV); model.load_state_dict(ck['ema']); model.eval()
    i = np.arange(768); LUT = torch.from_numpy(np.concatenate([((i // 64 + 6) % 12) * 64 + ((i % 64) ^ 56), [768]]).astype(np.int64)).to(DEV)
    PT_W = torch.cat([torch.tensor([0, 1, 1, 2, 4, 0] * 2, dtype=torch.float32).repeat_interleave(64), torch.zeros(1)]).to(DEV)

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
        with torch.autocast('cuda', dtype=torch.bfloat16):
            return model(board[:, :64], torch.zeros(len(idx), dtype=torch.long, device=DEV), torch.full((len(idx),), 64, dtype=torch.long, device=DEV), loops=1)[1].float()

    torch.backends.cuda.matmul.allow_tf32 = True
    a = np.zeros(n, dtype=np.float32); o = np.zeros(n, dtype=np.float32); t0 = time.time(); B = 4096
    for j in range(0, n, 262144):
        idx = torch.from_numpy(np.ascontiguousarray(x[j:j + 262144]).astype(np.int64)).to(DEV)
        a[j:j + 262144] = nnue(idx).cpu().numpy()
        o[j:j + 262144] = torch.cat([ouro(idx[k:k + B]) for k in range(0, len(idx), B)]).cpu().numpy()
        if (j // 262144) % 20 == 0:
            print(f'  {j/1e6:.1f}M of {n/1e6:.1f}M positions judged, {(j + 262144)/(time.time() - t0):.0f} per second', flush=True)
    a = np.clip(a, -12, 12); o = np.clip(o, -12, 12)
    P = np.nonzero(pl)[0]; half = len(P) // 2; T = lambda v: torch.from_numpy(v).double()

    def fit(cols):
        X = torch.stack([T(c[P]) for c in cols] + [torch.ones(len(P), dtype=torch.float64)], 1); zz = T(z[P]); w = torch.zeros(X.shape[1], dtype=torch.float64)
        for _ in range(30):
            p = torch.sigmoid(X[:half] @ w); g = X[:half].T @ (p - zz[:half])
            w = w - torch.linalg.solve((X[:half] * (p * (1 - p))[:, None]).T @ X[:half] + 1e-6 * torch.eye(X.shape[1], dtype=torch.float64), g)
        return w.numpy(), float(F.binary_cross_entropy_with_logits(X[half:] @ w, zz[half:]))

    wN, lN = fit([a, s]); wO, lO = fit([a, s, o]); _, lS = fit([s]); _, lA = fit([a]); _, lOo = fit([o])
    print(f'{n} positions, {len(P)} played. Held-out log-loss against our own game result: evaluator {lA:.4f}  search {lS:.4f}  OURO {lOo:.4f}  evaluator+search {lN:.4f}  evaluator+search+OURO {lO:.4f}')
    print(f'weights without OURO (evaluator, search, offset) {np.round(wN, 3)}   with OURO (evaluator, search, OURO, offset) {np.round(wO, 3)}')
    lgN = wN[0] * a + wN[1] * s + wN[2]; lgO = wO[0] * a + wO[1] * s + wO[2] * o + wO[3]
    slope = lambda lg: float(np.cov(lg[::7], s[::7])[0, 1] / s[::7].var())            # how many result-logits one search-score logit is worth
    kN, kO = slope(lgN), slope(lgO)
    sig = lambda v: 1 / (1 + np.exp(-v)); tN = sig(lgN / kN); tO = sig(lgO / kO)       # back in the scale of the search score
    print(f'scale brought back to the search score: divided by {kN:.3f} (without OURO) and {kO:.3f} (with)')
    b8 = lambda v: np.round(255 * np.clip(v, 0, 1)).astype(np.uint8)
    perm = np.random.RandomState(0).permutation(n)
    np.save(f'data/ev_{tag}_x.npy', x[perm]); np.save(f'data/ev_{tag}_y95.npy', b8(y95)[perm]); np.save(f'data/ev_{tag}_yN.npy', b8(tN)[perm]); np.save(f'data/ev_{tag}_yO.npy', b8(tO)[perm])
    ss = sig(s)
    print(f'mean |label - search score|: production {np.abs(y95 - ss).mean():.4f}, without OURO {np.abs(tN - ss).mean():.4f}, with OURO {np.abs(tO - ss).mean():.4f}; with against without {np.abs(tO - tN).mean():.4f}')
