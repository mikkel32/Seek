"""Is the evaluator comparison of zero/search_gap.py fair? Three checks on the same 8,000 positions.

  1. Quiet positions only. A static evaluator is meant for positions where no capture is pending (the search resolves
     those). The positions are already: not in check, and the move our search chose there was not a capture. Stricter,
     and independent of any engine: no capture or promotion is available that wins material by static exchange
     (worked out here with python-chess). The comparison is repeated on that subset.
  2. Each evaluator in its own best scale. Three ways of turning a score into a win probability, each fitted on one half
     of the positions and tested on the other: one scale constant K (p = 1/(1+exp(-score/K))); a scale and an offset
     (what search_gap.py uses); and a free monotone curve (isotonic regression), which also removes any penalty for a
     score that is not linear in the logit.
  3. How sure is the difference: paired over positions, 95% range from 2,000 bootstrap resamples of the test half.

The positions are from OUR self-play games, which favours our evaluator (it was trained on this kind of position);
the truth is Stockfish 19 playing each position out, which favours evaluators that think like Stockfish.

  python3 zero/hygiene.py [processes=4]
"""
import math, sys
import chess, numpy as np
import search_gap as sg

sg.PROCS = int(sys.argv[1]) if len(sys.argv) > 1 else 4
VAL = {1: 100, 2: 320, 3: 330, 4: 500, 5: 950, 6: 20000}


def see(b, m):
    """static exchange value of a capture or promotion for the side to move (pins ignored, as engines do)"""
    to, frm = m.to_square, m.from_square; occ = b.occupied ^ chess.BB_SQUARES[frm]
    if b.is_en_passant(m):
        gain = [100]; occ ^= chess.BB_SQUARES[to + (-8 if b.turn else 8)]
    else:
        gain = [VAL[b.piece_type_at(to)] if b.piece_type_at(to) else 0]
    on_sq = VAL[b.piece_type_at(frm)]
    if m.promotion:
        gain[0] += VAL[m.promotion] - 100; on_sq = VAL[m.promotion]
    side = not b.turn
    while True:
        att = b.attackers_mask(side, to, occ) & occ
        if not att:
            break
        for pt in range(1, 7):
            bb = att & b.pieces_mask(pt, side)
            if bb:
                break
        if pt == 6 and (b.attackers_mask(not side, to, occ) & occ):       # the king may not take a defended piece
            break
        gain.append(on_sq - gain[-1]); on_sq = VAL[pt]; occ ^= bb & -bb; side = not side
    for i in range(len(gain) - 1, 0, -1):
        gain[i - 1] = -max(-gain[i - 1], gain[i])
    return gain[0]


def strictly_quiet(fen):
    b = chess.Board(fen)
    return not b.is_check() and not any((b.is_capture(m) or m.promotion) and see(b, m) > 0 for m in b.legal_moves)


def ll(q, z):
    q = np.clip(q, 1e-7, 1 - 1e-7)
    return -(z * np.log(q) + (1 - z) * np.log(1 - q))


def fit_k(x, z):
    """one scale constant: golden-section search on the log-loss"""
    lo, hi = 0.05, 20.0; g = (math.sqrt(5) - 1) / 2
    f = lambda k: ll(1 / (1 + np.exp(-x / k)), z).mean()
    for _ in range(60):
        a, c = hi - g * (hi - lo), lo + g * (hi - lo)
        if f(a) < f(c):
            hi = c
        else:
            lo = a
    return (lo + hi) / 2


def fit_affine(x, z):
    X = np.stack([x, np.ones_like(x)], 1); w = np.zeros(2)
    for _ in range(40):
        q = 1 / (1 + np.exp(-X @ w)); H = (X * (q * (1 - q))[:, None]).T @ X + 1e-6 * np.eye(2); w = w - np.linalg.solve(H, X.T @ (q - z))
    return w


def fit_isotonic(x, z):
    """pool adjacent violators -> (block centres, block values), lightly smoothed so no block claims certainty"""
    o = np.argsort(x, kind='stable'); xs, zs = x[o], z[o]; blocks = []                # [sum of z, count, sum of x]
    for xv, zv in zip(xs, zs):
        blocks.append([zv, 1.0, xv])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] >= blocks[-1][0] / blocks[-1][1]:
            s, n, sx = blocks.pop(); blocks[-1][0] += s; blocks[-1][1] += n; blocks[-1][2] += sx
    return np.array([bk[2] / bk[1] for bk in blocks]), np.array([(bk[0] + 0.5) / (bk[1] + 1.0) for bk in blocks])


def calibrated(x, z, half):
    """per-position test log-loss under the three calibrations (fitted on the first half)"""
    a, b = x[:half], z[:half]; t = x[half:]
    k = fit_k(a, b); w = fit_affine(a, b); cx, cv = fit_isotonic(a, b)
    return {'K': ll(1 / (1 + np.exp(-t / k)), z[half:]), 'scale+offset': ll(1 / (1 + np.exp(-(w[0] * t + w[1]))), z[half:]),
            'monotone': ll(np.interp(t, cx, cv), z[half:])}, k


if __name__ == '__main__':
    d = np.load('zero/label_truth.npz'); fens = [sg.fen_of(r) for r in d['idx']]; zp = d['zp']; N = len(fens)
    quiet = np.array([strictly_quiet(f) for f in fens])
    b0 = [chess.Board(f) for f in fens[:2000]]
    print(f'{N} positions; in check: {sum(b.is_check() for b in b0)} of the first 2000; a material-winning capture or promotion is available in {int((~quiet).sum())} '
          f'({(~quiet).mean():.1%}); strictly quiet: {int(quiet.sum())}', flush=True)
    judges = {}
    for name, spec in (('our evaluator', sg.OURS), ('Stockfish 14.1 hand-written', sg.SF14C), ('Stockfish 14.1 network', sg.SF14N), ('Stockfish 19 network', sg.SF19)):
        v, _ = sg.run(spec, 0, fens); judges[name] = np.clip(v / 100, -30, 30)
        print(f'  {name}: read for {int((~np.isnan(v)).sum())} positions', flush=True)
    rs = np.random.RandomState(5)
    for label, mask in (('all positions', np.ones(N, bool)), ('strictly quiet positions only', quiet)):
        ok = mask & ~np.isnan(np.stack(list(judges.values()))).any(0); idx = np.nonzero(ok)[0]; half = len(idx) // 2; z = zp[idx]
        print(f'\n{label}: {len(idx)} (fitted on {half}, tested on {len(idx) - half}). Log-loss against the truth, lower is better.')
        print(f'  {"":30s} {"own K":>8s} {"K only":>9s} {"scale+offset":>13s} {"monotone":>9s}')
        res = {}
        for name, v in judges.items():
            res[name], k = calibrated(v[idx], z, half)
            print(f'  {name:30s} {k:8.2f} {res[name]["K"].mean():9.4f} {res[name]["scale+offset"].mean():13.4f} {res[name]["monotone"].mean():9.4f}')
        for other in ('Stockfish 14.1 hand-written', 'Stockfish 14.1 network'):
            for cal in ('scale+offset', 'monotone'):
                diff = res[other][cal] - res['our evaluator'][cal]; boot = [diff[rs.randint(0, len(diff), len(diff))].mean() for _ in range(2000)]
                print(f'  {other} minus ours, {cal}: {diff.mean():+.4f}  (95% range {np.percentile(boot, 2.5):+.4f} to {np.percentile(boot, 97.5):+.4f}; positive = ours is better)')
