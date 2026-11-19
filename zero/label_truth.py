"""Which training label is closest to what a position is really worth?

The evaluator is trained on labels made from our own self-play: the search score, the game result, and blends of them.
eval_gap.py showed those games end differently from strong play in more than a quarter of the positions. Here every
candidate label is compared with the same ground truth: the result when Stockfish 19 plays the position out against
itself (20,000 positions per move). Stockfish is the measuring instrument only; no label uses it.

Candidates: the 2,500-position search score, the game result, their half-and-half blend, the lambda-return used in
production at several values of lambda, a deeper search of our own (20,000 positions), OURO's value (the transformer, no
search), our evaluator alone, and blends with OURO.

Two numbers per label, on held-out positions; lower is better in both:
  calibrated  log-loss after one slope and offset fitted on the other half (how much the label KNOWS)
  raw         mean squared distance to the truth as the label stands (what a network that copies it would inherit)

  python3 zero/label_truth.py [positions=8000] [processes=44]
"""
import glob, json, math, os, re, sys
import numpy as np, chess, torch
from multiprocessing import Pool

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')); sys.path.insert(0, '.'); sys.path.insert(0, 'zero')
import eval_gap as eg
N = int(sys.argv[1]) if len(sys.argv) > 1 else 8000; PROCS = int(sys.argv[2]) if len(sys.argv) > 2 else 44
eg.PROCS = PROCS; REC = eg.REC; LAM = 0.5
src = open('zero/cont.py').read(); exec(re.search(r'(def target\(q\):.*?)(?=\ndef make_shard)', src, re.S).group(1))      # target(), lam_targets(): the loop's own code
LAMS = (0.7, 0.85, 0.95, 0.98)


def one_file(a):
    path, per, seed = a
    r = np.fromfile(path, dtype=REC); rs = np.random.RandomState(seed)
    lab = {f'lambda-return, lambda {l}': lam_targets(r, l) for l in LAMS}
    wp = 1 / (1 + np.exp(-r['score'].astype(np.float64) / 271.58)); z = (r['result'].astype(np.float64) + 1) / 2
    ok = (r['ply'] >= 0) & (r['ply'] < 127) & ~((r['idx'] == 324) | (r['idx'] == 764)).any(1)
    out = []
    for k in rs.permutation(np.nonzero(ok)[0]):
        b = eg.fen_of(r['idx'][k])
        if b.is_valid() and not b.is_check():
            out.append((b.fen(), r['idx'][k].copy(), {'search score, 2,500 positions': wp[k], 'game result': z[k], 'half score, half result': 0.5 * wp[k] + 0.5 * z[k],
                                                      **{n: v[k] for n, v in lab.items()}}))
            if len(out) >= per:
                break
    return out


def scores(L, zp, half):
    p = np.clip(L, 0.004, 0.996); s = 100 * np.log(p / (1 - p))                     # as a score in centipawn-like units for the shared fitting code
    return eg.fit_and_test(s, zp, half).mean(), float(((L[half:] - zp[half:]) ** 2).mean()), eg.fit_and_test(s, zp, half)


if __name__ == '__main__':
    files = sorted(glob.glob('zero/pool/*.bin')); rs = np.random.RandomState(3); rs.shuffle(files); files = files[:max(20, min(len(files), 120))]
    per = N // len(files) + 1
    with Pool(min(PROCS, len(files))) as pool:
        rows = [x for part in pool.map(one_file, [(f, per, i) for i, f in enumerate(files)]) for x in part]
    rs.shuffle(rows); rows = rows[:N]; n = len(rows); half = n // 2
    fens = [x[0] for x in rows]; idx = np.stack([x[1] for x in rows]); lab = {k: np.array([x[2][k] for x in rows]) for k in rows[0][2]}
    G = json.load(open('zero/cont_state.json'))['gen']
    print(f'{n} played positions from {len(files)} files of generation-{G} self-play', flush=True)
    # OURO (no search) and our evaluator (no search), both as win probabilities
    from model import Ouro
    DEV = torch.device('cuda'); ck = torch.load('zero/ouro.pt', map_location=DEV); model = Ouro(**ck['cfg']).to(DEV); model.load_state_dict(ck['ema']); model.eval()
    with torch.no_grad():
        ix = torch.from_numpy(idx.astype(np.int64)).to(DEV); board = torch.zeros(n, 65, dtype=torch.long, device=DEV)
        board.scatter_(1, torch.where(ix < 768, ix % 64, torch.full_like(ix, 64)), torch.where(ix < 768, ix // 64 + 1, torch.zeros_like(ix)))
        ov = torch.cat([model(board[k:k + 1024, :64], torch.zeros(len(board[k:k + 1024]), dtype=torch.long, device=DEV),
                              torch.full((len(board[k:k + 1024]),), 64, dtype=torch.long, device=DEV), loops=1)[1].float() for k in range(0, n, 1024)]).cpu().numpy().astype(np.float64)
    print(f'OURO step {ck.get("step")}; its value output ranges {ov.min():.2f} to {ov.max():.2f}', flush=True)
    ev = eg.run(eg.OURS, 'static', 0, fens); deep = eg.run(eg.OURS, 'search', 20000, fens)
    sig = lambda v: 1 / (1 + np.exp(-v))
    with Pool(PROCS) as pool:
        parts = pool.map(eg.playout_job, [fens[i::PROCS] for i in range(PROCS)])
    zp = np.zeros(n)
    for i, r in enumerate(parts):
        zp[i::PROCS] = r
    print(f'Stockfish 19 playouts: wins {np.mean(zp == 1):.3f}, draws {np.mean(zp == 0.5):.3f}, losses {np.mean(zp == 0):.3f}; our own game ended the same way in {np.mean(zp == lab["game result"]):.3f}', flush=True)
    # OURO's output is fitted to a probability on the first half against OUR game results (what it could be calibrated on without Stockfish)
    def to_prob(v, y):
        X = np.stack([v, np.ones_like(v)], 1); w = np.zeros(2)
        for _ in range(40):
            q = sig(X[:half] @ w); w = w - np.linalg.solve((X[:half] * (q * (1 - q))[:, None]).T @ X[:half] + 1e-6 * np.eye(2), X[:half].T @ (q - y[:half]))
        return sig(X @ w)
    lab['our evaluator, no search'] = to_prob(np.clip(ev, -3000, 3000) / 100, lab['game result'])
    lab['our search, 20,000 positions'] = sig(np.clip(deep, -3000, 3000) / 271.58)
    lab['OURO value, no search'] = to_prob(ov, lab['game result'])
    lr = lab['lambda-return, lambda 0.95']
    lab['half lambda-return 0.95, half OURO'] = 0.5 * lr + 0.5 * lab['OURO value, no search']
    lab['half 20,000-position search, half OURO'] = 0.5 * lab['our search, 20,000 positions'] + 0.5 * lab['OURO value, no search']
    lab['half lambda-return 0.95, half 20,000-position search'] = 0.5 * lr + 0.5 * lab['our search, 20,000 positions']
    lab['third each: lambda-return, 20,000 search, OURO'] = (lr + lab['our search, 20,000 positions'] + lab['OURO value, no search']) / 3
    base = scores(lr, zp, half)
    print(f'\ntruth = Stockfish 19 playing the position out; fitted on {half}, tested on {n - half}. Lower is better. Difference is to production\'s label (lambda 0.95).')
    print(f'  {"label":54s} calibrated   difference          raw')
    for k in sorted(lab, key=lambda k: scores(lab[k], zp, half)[0]):
        c, raw, per_pos = scores(lab[k], zp, half); d = per_pos - base[2]
        print(f'  {k:54s} {c:.4f}   {d.mean():+.4f} +- {1.96 * d.std() / math.sqrt(len(d)):.4f}   {raw:.4f}', flush=True)
    np.savez('zero/label_truth.npz', zp=zp, idx=idx, **{re.sub(r'[^a-z0-9]+', '_', k.lower()): v for k, v in lab.items()})
