"""The static move-choice test of ext/brain_metric.py split by whether the positions are quiet for the search.

A search leaf is scored after its captures are played out. Where that changes nothing (the static score and the played-out
score agree for both of the positions compared, under the first evaluator named), the leaf value IS the network's judgment;
elsewhere the capture search has a say. Printed for every evaluator: choice among the clear cases on the quiet pairs and on
the rest, static and played out.

  python3 ext/brain_quiet.py "<name>|<engine command with network>" ...        (the first one defines which pairs are quiet)
"""
import os, sys
import chess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quality as Q
import brain_metric as BM

tr = [t for t in Q.usable() if t['m2'] is not None]; c1, c2 = [], []
for t in tr:
    b = chess.Board(t['fen'])
    for m, dst in ((t['m1'], c1), (t['m2'], c2)):
        b.push(chess.Move.from_uci(m)); dst.append(b.fen()); b.pop()
g = [Q.E(t['cp1']) - Q.E(t['cp2']) for t in tr]; n = len(tr); quiet = None


def both(cmd):
    out = []
    for q in ('', '1'):
        os.environ['QEVAL'] = q; v = BM.evals(cmd, '', '', c1 + c2); out.append((v[:n], v[n:]))
    return out


def choice(v1, v2, idx):
    idx = [i for i in idx if g[i] >= 0.04]
    return 100 * sum(1.0 if v1[i] < v2[i] else 0.5 if v1[i] == v2[i] else 0.0 for i in idx) / len(idx), len(idx)


for spec in sys.argv[1:]:
    name, cmd = spec.split('|')[:2]; (s1, s2), (q1, q2) = both(cmd)
    if quiet is None:
        quiet = [i for i in range(n) if s1[i] == q1[i] and s2[i] == q2[i]]; rest = sorted(set(range(n)) - set(quiet))
        print(f'{n} pairs; quiet for the search (by {name}): {len(quiet)}; the rest: {len(rest)}', flush=True)
    a, na = choice(s1, s2, quiet); b_, _ = choice(q1, q2, quiet); c, nc = choice(s1, s2, rest); d, _ = choice(q1, q2, rest)
    print(f'{name:16s} quiet pairs ({na} clear): static {a:5.2f}%  played out {b_:5.2f}%   |   the rest ({nc} clear): static {c:5.2f}%  played out {d:5.2f}%', flush=True)
