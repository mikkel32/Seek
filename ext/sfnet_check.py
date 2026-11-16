"""The diagnostic build's Stockfish-network arithmetic against the installed Stockfish's own `eval` output."""
import re, subprocess, sys, os
import numpy as np
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1000; PCN = 'PNBRQKpnbrqk'
def mkfen(r):
    board = [''] * 64; rows = []
    for i in r:
        if i < 768: board[i % 64] = PCN[i // 64]
    for rank in range(7, -1, -1):
        row = ''; gap = 0
        for f in range(8):
            c = board[rank * 8 + f]
            if c: row += (str(gap) if gap else '') + c; gap = 0
            else: gap += 1
        rows.append(row + (str(gap) if gap else ''))
    return '/'.join(rows) + ' w - - 0 1'
X = np.load('/root/loopchess/data/ev_ival_x.npy', mmap_mode='r')[:N]; fens = []
for i, r in enumerate(X):
    f = mkfen(r)
    if i % 2: f = '/'.join(row.swapcase() for row in f.split()[0].split('/')[::-1]) + ' b - - 0 1'
    fens.append(f)
out = subprocess.run(['/usr/games/stockfish'], input=''.join(f'position fen {f}\neval\n' for f in fens) + 'quit\n', capture_output=True, text=True).stdout
sf = [float(m) * 100 if m != 'none' else None for m in re.findall(r'NNUE evaluation:?\s+([-+]?[0-9.]+|none)', out)]
env = dict(os.environ, SFNET_FILE='/root/ext/sf141_embedded.bin')
out = subprocess.run(['/root/loopchess/calc/calc_s19sf', '/root/loopchess/zero/w_BIGM.bin'], input=''.join(f'position fen {f}\nsfraw\n' for f in fens) + 'quit\n', capture_output=True, text=True, env=env).stdout
ours = [int(l.split()[1]) for l in out.split('\n') if l.startswith('sfraw ')]
assert len(ours) == len(fens) == len(sf), (len(ours), len(sf))
d = [abs((o if ' w ' in f else -o) * 100 / 208 - s) for o, s, f in zip(ours, sf, fens) if s is not None]
print(f'{len(d)} positions (half with Black to move): largest difference {max(d):.2f} hundredths of a pawn, mean {sum(d) / len(d):.3f}; the engine prints two decimals')
