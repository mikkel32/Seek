"""The transformer on the questions of ext/brain_metric.py, without any search: which of Stockfish's two best moves is the
better one, and how well does it predict the deep score?  (positions and deep analysis of ext/quality.py)

  OURO_GPU=1 python3 ext/brain_lm.py <checkpoint> [loops, default 2]

The transformer can prefer one of two moves in three ways, printed separately:
  policy       the move probabilities at the position itself (one evaluation)
  move values  its value for every move, at the position itself (the same evaluation)
  value        its value of the two positions after the moves (what a static evaluator is asked; two more evaluations)
and, as in brain_metric.py: choice = share right where the deep analysis sees a clear difference (0.04 of a point or more),
regret = expected score given away by following the preference (thousandths of a point), fit = squared error of the
position's own value against the deep score (as it comes, and at its best scale).
"""
import math, os, sys
import chess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import quality as Q
from engine import Net

net = Net(sys.argv[1]); loops = int(sys.argv[2]) if len(sys.argv) > 2 else 2
tr = [t for t in Q.usable() if t['m2'] is not None]; n = len(tr)
g = [Q.E(t['cp1']) - Q.E(t['cp2']) for t in tr]; crit = [x >= 0.04 for x in g]; tw = [Q.E(t['cp1']) for t in tr]
print(f'{n} positions with two analysed moves, {sum(crit)} with a clear difference', flush=True)


def run(boards):                                             # a finished game has no moves to evaluate: mate = 0 for the side to move, else a draw
    live = [b for b in boards if any(b.legal_moves)]; res = []
    for i in range(0, len(live), 256): res += net.evaluate(live[i:i + 256], loops=loops)
    it = iter(res)
    return [next(it) if any(b.legal_moves) else ([], [], 0.0 if b.is_check() else 0.5, []) for b in boards]


par = [chess.Board(t['fen']) for t in tr]; kids = [[], []]
for b, t in zip(par, tr):
    for k, m in enumerate((t['m1'], t['m2'])):
        c = b.copy(stack=False); c.push(chess.Move.from_uci(m)); kids[k].append(c)
rp = run(par); r1 = run(kids[0]); r2 = run(kids[1])


def report(name, pref):                                       # pref: +1 the better move is preferred, 0 a tie, -1 the worse one
    choice = sum((0.5 + 0.5 * p) for p, c in zip(pref, crit) if c) / sum(crit)
    regret = 1000 * sum(x * (0.5 - 0.5 * p) for x, p in zip(g, pref)) / n
    print(f'{name:22s} choice {100 * choice:5.2f}%   regret {regret:6.2f}', flush=True)


sgn = lambda x: int(x > 0) - int(x < 0)
pol, mv, top = [], [], 0
for t, (moves, p, v, q) in zip(tr, rp):
    u = [m.uci() for m in moves]; i1, i2 = u.index(t['m1']), u.index(t['m2'])
    pol.append(sgn(p[i1] - p[i2])); mv.append(sgn(q[i1] - q[i2])); top += u[max(range(len(u)), key=lambda i: p[i])] == t['m1']
report('policy', pol); report('move values', mv)
report('value of the children', [sgn(b[2] - a[2]) for a, b in zip(r1, r2)])      # a child's value is the opponent's: lower is better for the mover
print(f"the policy's first choice is Stockfish's best move in {100 * top / n:.1f}% of the positions")
vp = [min(1 - 1e-6, max(1e-6, r[2])) for r in rp]; lg = [math.log(x / (1 - x)) for x in vp]
raw = sum((x - y) ** 2 for x, y in zip(vp, tw)) / n
best = min((sum((1 / (1 + math.exp(-a * x)) - y) ** 2 for x, y in zip(lg, tw)) / n, a) for a in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.4, 1.6, 1.8, 2.0, 2.3))
print(f'fit {raw:.5f} as it comes, {best[0]:.5f} at its best scale ({best[1]} x its own logit)')
if os.environ.get('QUIET_BY'):                               # QUIET_BY="<engine command with network>": split as ext/brain_quiet.py does
    import brain_metric as BM
    f1 = [b.fen() for b in kids[0]]; f2 = [b.fen() for b in kids[1]]; ev = []
    for qv in ('', '1'):
        os.environ['QEVAL'] = qv; v = BM.evals(os.environ['QUIET_BY'], '', '', f1 + f2); ev.append((v[:n], v[n:]))
    quiet = [i for i in range(n) if ev[0][0][i] == ev[1][0][i] and ev[0][1][i] == ev[1][1][i]]; rest = sorted(set(range(n)) - set(quiet))
    for nm, pref in (('policy', pol), ('move values', mv), ('value of the children', [sgn(b[2] - a[2]) for a, b in zip(r1, r2)])):
        line = f'{nm:22s}'
        for lab, idx in (('quiet pairs', quiet), ('the rest', rest)):
            cl = [i for i in idx if crit[i]]; line += f'  {lab} ({len(cl)} clear): {100 * sum(0.5 + 0.5 * pref[i] for i in cl) / len(cl):5.2f}%'
        print(line, flush=True)
