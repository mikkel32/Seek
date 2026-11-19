"""Are the read-out buckets consistent with each other?  (positions and deep analysis of ext/quality.py)

Every test position with a two-line deep analysis gives three positions with a deep score: itself and the two that arise
after the best and second-best move. Per read-out bucket (piece count) the evaluator's mean error against the deep win
chance is printed, and how often the static scores pick the better move when both moves stay in the bucket, and when one
of them is a capture (the two positions then sit in different buckets, so any offset between buckets tilts the choice).

  python3 ext/brain_buckets.py "<name>|<engine command with network>[|<options>][|<environment>]" [scale]
"""
import math, os, sys
import chess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quality as Q
from brain_metric import evals

name, cmd, opts, env = (sys.argv[1].split('|') + ['', '', ''])[:4]; S = float(sys.argv[2]) if len(sys.argv) > 2 else 340.0
tr = [t for t in Q.usable() if t['m2'] is not None]; par, c1, c2, cap1, cap2 = [], [], [], [], []
for t in tr:
    b = chess.Board(t['fen']); par.append(t['fen'])
    for m, dst, cp in ((t['m1'], c1, cap1), (t['m2'], c2, cap2)):
        mv = chess.Move.from_uci(m); cp.append(b.is_capture(mv)); b.push(mv); dst.append(b.fen()); b.pop()
n = len(tr); v = evals(cmd, opts, env, par + c1 + c2); vp, v1, v2 = v[:n], v[n:2 * n], v[2 * n:]
pcs = lambda f: sum(ch.isalpha() for ch in f.split()[0]); bk = lambda f: min(7, (pcs(f) - 2) // 4)
sg = lambda x: 1 / (1 + math.exp(-x / S))
rows = [(par[i], vp[i], Q.E(tr[i]['cp1'])) for i in range(n)] + [(c1[i], v1[i], 1 - Q.E(tr[i]['cp1'])) for i in range(n)] + [(c2[i], v2[i], 1 - Q.E(tr[i]['cp2'])) for i in range(n)]
print(f'{name}: {len(rows)} positions with a deep score; win chance from the static score at scale {S:g}')
print('bucket (pieces)   positions   mean error (static - deep)   mean |error|   mean static win chance')
for k in range(8):
    r = [(sg(x) - y, sg(x)) for f, x, y in rows if bk(f) == k]
    if r: print(f'  {k} ({4 * k + 2:2d}-{min(32, 4 * k + 5):2d})      {len(r):7d}        {sum(e for e, _ in r) / len(r):+.4f}                 {sum(abs(e) for e, _ in r) / len(r):.4f}          {sum(p for _, p in r) / len(r):.3f}')
g = [Q.E(t['cp1']) - Q.E(t['cp2']) for t in tr]
for label, sel in (('both moves quiet', lambda i: not cap1[i] and not cap2[i]), ('best move a capture, the other quiet', lambda i: cap1[i] and not cap2[i]),
                   ('best move quiet, the other a capture', lambda i: not cap1[i] and cap2[i]), ('both captures', lambda i: cap1[i] and cap2[i])):
    idx = [i for i in range(n) if sel(i) and g[i] >= 0.04]
    if idx: print(f'{label:40s} {len(idx):5d} clear cases: static choice right {100 * sum(v1[i] < v2[i] for i in idx) / len(idx):.1f}%; mean static margin {sum(v2[i] - v1[i] for i in idx) / len(idx):+.1f} centipawns')
