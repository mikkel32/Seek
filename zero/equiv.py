"""How many positions must the fast engine search to choose the move its own 2,500-position search chooses as often as the
transformer does (48.5%)? Positions from the engine's own games."""
import subprocess, sys, chess, random
sys.path.insert(0, '.')
from match import openings
eng, net = sys.argv[1], sys.argv[2]
p = subprocess.Popen([eng, net], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
def send(s): p.stdin.write(s + '\n'); p.stdin.flush()
def best(fen, nodes, depth=0):
    send('ucinewgame'); send(f'position fen {fen}'); send(f'go depth {depth}' if depth else f'go nodes {nodes}'); n = 0
    while True:
        l = p.stdout.readline()
        if l.startswith('info depth') and ' nodes ' in l: n = int(l.split()[l.split().index('nodes') + 1])
        if l.startswith('bestmove'): return (l.split()[1], n) if depth else l.split()[1]
send('uci'); send('setoption name Hash value 16'); send('isready')
random.seed(3); pos = []
for fen in openings(40, 11):
    b = chess.Board(fen)
    while not b.is_game_over(claim_draw=True) and b.ply() < 140:
        m = best(b.fen(), 2500)
        if random.random() < 0.12 and len(list(b.legal_moves)) > 1: pos.append((b.fen(), m))
        b.push_uci(m)
print(f'{len(pos)} positions from 40 engine games; the move of a fresh 2,500-position search is the reference')
for d in (1, 2, 3, 4, 5, 6, 7, 8):
    r = [best(f, 0, d) for f, m in pos]; hit = sum(x[0] == m for x, (f, m) in zip(r, pos)); nn = sorted(x[1] for x in r)
    print(f'  depth {d}: same move in {hit / len(pos):.1%}; positions searched: mean {sum(nn) / len(nn):.0f}, median {nn[len(nn) // 2]}')
send('quit')
