"""How well does an evaluator tell the better of two moves, without any search?  (positions and deep analysis of ext/quality.py)

For every test position with a deep two-line Stockfish analysis, the evaluator scores the two positions that arise after
Stockfish's best and second-best move (the engine's own `eval` command, so the numbers are exactly what the search sees).
Printed:
  - choice: in the positions where the deep analysis says one move is clearly better (at least 0.04 of a point), how often
    the evaluator's static scores prefer that move
  - regret: the expected score given away by following the static preference between the two moves, over all positions
    (thousandths of a point)
  - fit: how well the static score of the position itself predicts the deep score (squared error of the win chance at the
    evaluator's best scale)
A network is a function of single positions; this asks the question the search asks of it: which of two positions is better.

  python3 ext/brain_metric.py "<name>|<engine command with network>[|<options Name=value,...>][|<environment NAME=value ...>]" ...
"""
import json, math, os, subprocess, sys
import chess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quality as Q


def evals(cmd, opts, env, fens):
    e = dict(os.environ); e.update(dict(kv.split('=', 1) for kv in env.split() if kv))
    word = 'qeval' if os.environ.get('QEVAL') else 'eval'           # QEVAL=1: the value after captures are played out (what a search leaf returns)
    text = ''.join(f'setoption name {k} value {v}\n' for k, v in Q.opts_of(opts)) + ''.join(f'position fen {f}\n{word}\n' for f in fens) + 'quit\n'
    out = subprocess.run(cmd.split(), input=text, capture_output=True, text=True, env=e).stdout
    v = [int(l.split()[1]) for l in out.splitlines() if l.startswith('eval ')]
    assert len(v) == len(fens), (len(v), len(fens))
    return v


if __name__ == '__main__':
    tr = [t for t in Q.usable() if t['m2'] is not None]
    par, c1, c2 = [], [], []
    for t in tr:
        b = chess.Board(t['fen']); par.append(t['fen'])
        for m, dst in ((t['m1'], c1), (t['m2'], c2)):
            b.push(chess.Move.from_uci(m)); dst.append(b.fen()); b.pop()
    g = [Q.E(t['cp1']) - Q.E(t['cp2']) for t in tr]; crit = [x >= 0.04 for x in g]; tw = [Q.E(t['cp1']) for t in tr]
    print(f'{len(tr)} positions with two analysed moves, {sum(crit)} with a clear difference', flush=True)
    for spec in sys.argv[1:]:
        name, cmd, opts, env = (spec.split('|') + ['', '', ''])[:4]
        v = evals(cmd, opts, env, par + c1 + c2); n = len(tr); vp, v1, v2 = v[:n], v[n:2 * n], v[2 * n:]
        # the children are scored from the opponent's side: the mover prefers the child with the LOWER score
        right = [a < b_ for a, b_ in zip(v1, v2)]; tie = [a == b_ for a, b_ in zip(v1, v2)]
        choice = sum((r + 0.5 * t_) for r, t_, c in zip(right, tie, crit) if c) / sum(crit)
        regret = 1000 * sum(x * (0.0 if r else (0.5 if t_ else 1.0)) for x, r, t_ in zip(g, right, tie)) / n
        best = min((sum((1 / (1 + math.exp(-x / s)) - y) ** 2 for x, y in zip(vp, tw)) / n, s) for s in (120, 150, 180, 210, 240, 270, 300, 340, 380, 430, 500))
        print(f'{name:22s} choice {100 * choice:5.2f}%   regret {regret:6.2f}   fit {best[0]:.5f} (scale {best[1]})', flush=True)
