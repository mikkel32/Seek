"""Fixed-depth search over a set of positions: total positions searched, time, speed and a signature of the results.

Two builds that are meant to search the same tree (a pure speed change) must print the same signature; a change to the
search itself shows up as a different node count for the same depth.

  python3 zero/bench.py <engine binary> <network> [depth=13] [option=value,...]
"""
import hashlib, subprocess, sys, time

FENS = ['rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
        'r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P1B2/2PBPN2/PP1N1PPP/R2Q1RK1 b - - 0 8',
        'r2q1rk1/1b1nbppp/p2ppn2/1p6/3NPP2/1BN1B3/PPP1Q1PP/R4RK1 w - - 0 12',
        '2rq1rk1/pb1n1ppp/1p2pn2/2pp4/2PP4/1PN1PN2/PB3PPP/2RQ1RK1 w - - 0 12',
        'r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10',
        '3r1rk1/pp3ppp/2n1b3/2p1P3/5P2/2P1BN2/P1P3PP/R4RK1 b - - 0 16',
        '8/5pk1/6p1/1p2P2p/1P1r3P/5PK1/2R3P1/8 w - - 0 40',
        '6k1/5pp1/7p/3n4/8/1P3N1P/5PP1/6K1 w - - 0 35',
        'r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1',
        '2r3k1/1q1nbppp/r3p3/3pP3/pPpP4/P1Q2N2/2RN1PPP/2R4K b - - 0 23',
        '4rrk1/pp1n3p/3q2pQ/2p1pb2/2PP4/2P3N1/P2B2PP/4RRK1 b - - 7 19',
        '8/8/4kpp1/3p1b2/p6P/2B5/6P1/6K1 b - - 2 47']
binary, net = sys.argv[1], sys.argv[2]; depth = int(sys.argv[3]) if len(sys.argv) > 3 else 13
opts = [kv.split('=', 1) for kv in sys.argv[4].split(',')] if len(sys.argv) > 4 and sys.argv[4] else []
text = 'uci\n' + ''.join(f'setoption name {k} value {v}\n' for k, v in opts) + 'setoption name Hash value 64\nisready\n' + ''.join(f'ucinewgame\nposition fen {f}\ngo depth {depth}\n' for f in FENS) + 'quit\n'
t0 = time.time(); out = subprocess.run([binary, net], input=text, capture_output=True, text=True).stdout; dt = time.time() - t0
nodes = 0; sig = hashlib.md5(); last = None
for l in out.split('\n'):
    if l.startswith('info depth'):
        last = l.split()
    elif l.startswith('bestmove') and last:
        n = int(last[last.index('nodes') + 1]); nodes += n; sig.update(f'{n} {last[last.index("score") + 2]} {l.split()[1]};'.encode())
print(f'depth {depth}: {nodes:,} positions in {dt:.2f}s = {nodes / dt / 1e3:.0f}k per second; signature {sig.hexdigest()[:10]}')
