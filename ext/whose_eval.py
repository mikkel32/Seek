"""Whose static evaluation is closer to a deep search? Ours against Stockfish's, on the same held-out game positions.

The held-out set's labels are deep Stockfish searches (Lichess server analysis) turned into a win chance. Every evaluator
gets its own best scale (one number: how many of its units make the same win chance), so different units do not matter.

  python3 ext/whose_eval.py <positions> <tag of the position file> "<name>|<stockfish binary>|<option=value,...>" ... "<name>|<our engine> <network>" ...
"""
import re, subprocess, sys
import numpy as np

n = int(sys.argv[1]); tag = sys.argv[2]
x = np.asarray(np.load(f'data/ev_{tag}_x.npy', mmap_mode='r')[:n]); y = np.asarray(np.load(f'data/ev_{tag}_y.npy', mmap_mode='r')[:n]).astype(np.float64)
PC = 'PNBRQKpnbrqk'


def fen(r):
    """32 feature numbers in the mover's view (piece code x 64 + square; 768 = no piece) -> a FEN with the mover as White"""
    board = [''] * 64; rows = []
    for i in r:
        if i < 768:
            board[i % 64] = PC[i // 64]
    for rank in range(7, -1, -1):
        row = ''; gap = 0
        for f in range(8):
            c = board[rank * 8 + f]
            if c:
                row += (str(gap) if gap else '') + c; gap = 0
            else:
                gap += 1
        rows.append(row + (str(gap) if gap else ''))
    return '/'.join(rows) + ' w - - 0 1'


fens = [fen(r) for r in x]
pieces = np.array([sum(c.isalpha() for c in f.split()[0]) for f in fens])


def stockfish(binary, opts):
    text = ''.join(f'setoption name {o.split("=")[0]} value {o.split("=")[1]}\n' for o in opts.split(',') if o) + ''.join(f'position fen {f}\neval\n' for f in fens) + 'quit\n'
    out = subprocess.run([binary], input=text, capture_output=True, text=True).stdout
    v = [float(m) if m != 'none' else np.nan for m in re.findall(r'Final evaluation:?\s+([-+]?[0-9.]+|none)', out)]
    assert len(v) == len(fens), f'{len(v)} answers for {len(fens)} positions'
    return np.array(v) * 100


def ours(cmd):
    text = ''.join(f'position fen {f}\neval\n' for f in fens) + 'quit\n'
    out = subprocess.run(cmd.split(), input=text, capture_output=True, text=True).stdout
    v = [int(l.split()[1]) for l in out.split('\n') if l.startswith('eval ')]
    assert len(v) == len(fens), f'{len(v)} answers for {len(fens)} positions'
    return np.array(v, float)


def score(e, s, m):
    p = 1 / (1 + np.exp(-np.clip(e[m] / s, -40, 40))); p = np.clip(p, 1e-7, 1 - 1e-7); t = y[m]
    return float(-(t * np.log(p) + (1 - t) * np.log(1 - p)).mean()), float(np.abs(p - t).mean())


res = {}
for spec in sys.argv[3:]:
    name, rest = spec.split('|', 1)
    res[name] = stockfish(*rest.split('|')) if '|' in rest else ours(rest)
ok = np.ones(len(fens), bool)
for e in res.values():
    ok &= ~np.isnan(e)
t = y[ok]; floor = float(-(t * np.log(np.clip(t, 1e-7, 1)) + (1 - t) * np.log(np.clip(1 - t, 1e-7, 1))).mean())
print(f'{ok.sum()} positions; a perfect evaluator would score a loss of {floor:.5f}')
bal = ok & (np.abs(y - 0.5) < 0.2)                 # the label says neither side is clearly winning: the positions games are decided in
scales = np.geomspace(60, 1200, 160)
print(f'balanced = the deep search gives the mover between 30% and 70%: {bal.sum()} positions')
for name, e in res.items():
    best = min(scales, key=lambda s: score(e, s, ok)[0]); l, a = score(e, best, ok)
    bb = min(scales, key=lambda s: score(e, s, bal)[1]); ab = score(e, bb, bal)[1]
    side = float((np.sign(e[bal]) == np.sign(y[bal] - 0.5)).mean())
    line = f'{name}: all positions: loss above perfect {l - floor:.5f}, mean error {a:.4f} (scale {best:.0f}); balanced: mean error {ab:.4f} (scale {bb:.0f}), names the better side in {100 * side:.1f}%'
    for lo, hi in ((2, 12), (13, 22), (23, 32)):
        m = bal & (pieces >= lo) & (pieces <= hi); line += f'; balanced with {lo}-{hi} pieces: {score(e, bb, m)[1]:.4f}'
    print(line)

import os
if os.environ.get('DUMP'):                          # the balanced endgame positions one evaluator gets most wrong, with everyone's verdict in win chance
    name = os.environ['DUMP']; e = res[name]; names = list(res)
    sc = {k: min(scales, key=lambda s: score(res[k], s, bal)[1]) for k in names}
    pr = {k: 1 / (1 + np.exp(-np.clip(res[k] / sc[k], -40, 40))) for k in names}
    m = np.where(bal & (pieces <= 12))[0]; err = np.abs(pr[name][m] - y[m]); order = m[np.argsort(-err)]
    print(f'{len(m)} balanced positions with 12 pieces or fewer; share of {name}\'s error in them from its worst 10%: {np.sort(err)[-len(m) // 10:].sum() / err.sum():.2f}')
    for q in (0.5, 0.75, 0.9, 0.97):
        print(f'  error at the {int(q * 100)}th percentile: ' + ', '.join(f'{k} {np.quantile(np.abs(pr[k][m] - y[m]), q):.3f}' for k in names))
    for i in order[:int(os.environ.get('DUMPN', 25))]:
        print(f'  {fens[i]}  deep search {y[i]:.2f}  ' + '  '.join(f'{k} {pr[k][i]:.2f} ({res[k][i]:+.0f})' for k in names))
