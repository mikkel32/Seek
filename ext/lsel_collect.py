"""Collection for the learned search judgment (engine build -DLSEL).

  positions  self-play games of the engine from openings the test sets do not use; a few positions per game -> a FEN list
  collect    the -DLSEL build searches every position; its records (what was known about a late move, and whether a
             full-depth look at it beat the bound) are gathered in one file

  python3 ext/lsel_collect.py positions --cmd "<engine> <net>" --opts Hash=16,... --first 3300 --games 200 --nodes 20000 --per 12 --out ls/pos.txt
  python3 ext/lsel_collect.py collect --cmd "<LSEL engine> <net>" --opts ... --fens ls/pos.txt --nodes 300000 --rate 64 --out ls/rec.bin
"""
import argparse, os, random, subprocess, sys, time
from multiprocessing import Pool
import chess


class Eng:
    def __init__(self, cmd, opts, env=None):
        self.p = subprocess.Popen(cmd.split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1, env=env)
        for o in filter(None, opts.split(',')):
            k, v = o.split('='); self.send(f'setoption name {k} value {v}')
        self.send('isready'); self.wait('readyok')

    def send(self, s): self.p.stdin.write(s + '\n'); self.p.stdin.flush()

    def wait(self, w):
        while True:
            l = self.p.stdout.readline()
            if not l: raise RuntimeError('engine died')
            if l.startswith(w): return l

    def move(self, fen, moves, nodes):
        self.send('position fen ' + fen + (' moves ' + ' '.join(moves) if moves else '')); self.send(f'go nodes {nodes}')
        return self.wait('bestmove').split()[1]

    def quit(self):
        self.send('quit')
        try: self.p.wait(timeout=20)
        except Exception: self.p.kill()


def w_games(a):
    cmd, opts, nodes, per, fens, seed = a; e = Eng(cmd, opts); rnd = random.Random(seed); out = []
    for fen in fens:
        e.send('ucinewgame'); b = chess.Board(fen); moves = []; seen = []
        while not (b.is_game_over(claim_draw=True) or b.ply() > 300):
            if len(moves) >= 4 and b.legal_moves.count() >= 3: seen.append(b.fen())
            mv = e.move(fen, moves, nodes); m = chess.Move.from_uci(mv)
            if m not in b.legal_moves: break
            b.push(m); moves.append(mv)
        out += rnd.sample(seen, min(per, len(seen)))
    e.quit()
    return out


def w_collect(a):
    cmd, opts, nodes, rate, fens, part = a
    env = dict(os.environ, LSEL_FILE=part, LSEL_RATE=str(rate)); e = Eng(cmd, opts, env)
    for fen in fens:
        e.send('ucinewgame'); e.move(fen, None, nodes)
    e.quit()
    return part


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('what'); ap.add_argument('--cmd'); ap.add_argument('--opts', default='Hash=16')
    ap.add_argument('--first', type=int, default=3300); ap.add_argument('--games', type=int, default=200); ap.add_argument('--nodes', type=int, default=20000)
    ap.add_argument('--per', type=int, default=12); ap.add_argument('--out'); ap.add_argument('--fens'); ap.add_argument('--rate', type=int, default=64)
    ap.add_argument('--workers', type=int, default=4); ap.add_argument('--openings', default='data/openings.txt')
    a = ap.parse_args(); t0 = time.time()
    if a.what == 'positions':
        op = [f for f in open(a.openings).read().split('\n') if f]; random.Random(0).shuffle(op); op = op[a.first:a.first + a.games]      # the order match.py and the test sets use
        with Pool(a.workers) as pool:
            res = pool.map(w_games, [(a.cmd, a.opts, a.nodes, a.per, op[w::a.workers], w) for w in range(a.workers)])
        fens = sorted({f for r in res for f in r}); random.Random(1).shuffle(fens); open(a.out, 'w').write('\n'.join(fens) + '\n')
        print(f'{len(op)} games, {len(fens)} positions -> {a.out} ({time.time() - t0:.0f}s)')
    else:
        fens = [f for f in open(a.fens).read().split('\n') if f]; parts = [f'{a.out}.part{w}' for w in range(a.workers)]
        for p in parts:
            if os.path.exists(p): os.remove(p)
        with Pool(a.workers) as pool:
            pool.map(w_collect, [(a.cmd, a.opts, a.nodes, a.rate, fens[w::a.workers], parts[w]) for w in range(a.workers)])
        with open(a.out, 'wb') as f:
            for p in parts:
                f.write(open(p, 'rb').read()); os.remove(p)
        print(f'{len(fens)} positions searched at {a.nodes} nodes -> {a.out}, {os.path.getsize(a.out) // 40} records ({time.time() - t0:.0f}s)')
