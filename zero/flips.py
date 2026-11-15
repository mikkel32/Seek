"""How many attack-map inputs change per move? (12 maps: squares attacked by each kind of piece of each colour.) Games played by the engine itself."""
import subprocess, sys, chess, random, numpy as np
sys.path.insert(0, '.')
from match import openings
eng, net = sys.argv[1], sys.argv[2]
p = subprocess.Popen([eng, net], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
def send(s): p.stdin.write(s + '\n'); p.stdin.flush()
def best():
    while True:
        l = p.stdout.readline()
        if l.startswith('bestmove'): return l.split()[1]
send('uci'); send('setoption name Hash value 32'); send('isready')
def maps(b):
    m = {}
    for c in (True, False):
        for pt in range(1, 7):
            a = 0
            for s in b.pieces(pt, c): a |= int(b.attacks(s))
            m[(c, pt)] = a
    return m
groups = {'all 12 kinds': [[(c, pt)] for c in (True, False) for pt in range(1, 7)], 'pawn, knight, king (non-sliders)': [[(c, pt)] for c in (True, False) for pt in (1, 2, 6)],
          'bishop, rook, queen (sliders)': [[(c, pt)] for c in (True, False) for pt in (3, 4, 5)], 'one map per colour (any piece)': [[(c, pt) for pt in range(1, 7)] for c in (True, False)],
          'pawns only': [[(c, 1)] for c in (True, False)], 'by value class (P, NB, R, Q, K)': [[(c, pt) for pt in g] for c in (True, False) for g in ((1,), (2, 3), (4,), (5,), (6,))]}
def gm(m, g):
    out = []
    for grp in g:
        a = 0
        for k in grp: a |= m[k]
        out.append(a)
    return out
flips = {k: [] for k in groups}; on = {k: [] for k in groups}; ps = []
for fen in openings(24, 7):
    b = chess.Board(fen); mv = []; send('ucinewgame')
    while not b.is_game_over(claim_draw=True) and b.ply() < 160:
        send(f'position fen {fen}' + (' moves ' + ' '.join(mv) if mv else '')); send('go nodes 3000'); u = best()
        m0 = maps(b); cap = b.is_capture(chess.Move.from_uci(u)); b.push_uci(u); mv.append(u); m1 = maps(b); ps.append(3 if cap else 2)
        for k, g in groups.items():
            a0, a1 = gm(m0, g), gm(m1, g); flips[k].append(sum(bin(x ^ y).count('1') for x, y in zip(a0, a1))); on[k].append(sum(bin(x).count('1') for x in a1))
send('quit')
print(f'{len(ps)} moves from {24} engine games (3,000 positions per move); a move changes {np.mean(ps):.1f} piece-on-square inputs on average')
for k in groups:
    f = np.array(flips[k]); print(f'  {k:36s} inputs on: {np.mean(on[k]):5.1f}   changed by a move: mean {f.mean():5.1f}, median {np.median(f):4.0f}, 90% of moves <= {np.percentile(f, 90):4.0f}')
