"""Is the gap to Stockfish in the evaluator or in the search? A measurement that can tell.

Stockfish 14.1 can play with its network switched off, on its older hand-written evaluator ("classical"). That gives
an opponent whose SEARCH is Stockfish's but whose EVALUATOR is not a strong network:

  1. how good is each evaluator alone (no search), against what the positions are really worth
     (the 8,000 positions of zero/label_truth.npz; truth = Stockfish 19 playing each one out against itself);
  2. how good is each engine when both search the SAME number of positions.

If the hand-written evaluator judges single positions no better than ours, yet that engine gains more from the same
number of searched positions (and wins the games, see zero/search_gap.sh), the difference is search, and search can be
improved without any training data. Stockfish is a measuring instrument here; nothing is learned from it.

  python3 zero/search_gap.py [searched positions=2000] [processes=4]
"""
import json, math, os, re, subprocess, sys
import numpy as np, chess
from multiprocessing import Pool

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
NS = int(sys.argv[1]) if len(sys.argv) > 1 else 2000; PROCS = int(sys.argv[2]) if len(sys.argv) > 2 else 4
G = json.load(open('zero/cont_state.json'))['gen']
OURS = (f'calc/calc_w zero/g{G}.bin', ()); SF14N = ('/usr/games/stockfish', ()); SF14C = ('/usr/games/stockfish', (('Use NNUE', 'false'),)); SF19 = ('/root/sf19/stockfish19', ())
SYM = 'PNBRQKpnbrqk'


def fen_of(idx):
    b = chess.Board(None)
    for v in idx:
        if v < 768:
            b.set_piece_at(int(v) % 64, chess.Piece.from_symbol(SYM[int(v) // 64]))
    b.turn = chess.WHITE
    return b.fen()


class Eng:
    def __init__(self, spec):
        cmd, opts = spec
        self.p = subprocess.Popen(cmd.split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self.send('uci'); self.until('uciok'); self.send('setoption name Hash value 16'); self.send('setoption name Threads value 1')
        for k, v in opts:
            self.send(f'setoption name {k} value {v}')
        self.send('isready'); self.until('readyok'); self.ours = cmd.startswith('calc/')

    def send(self, s):
        self.p.stdin.write(s + '\n'); self.p.stdin.flush()

    def until(self, word):
        out = []
        while True:
            l = self.p.stdout.readline()
            if not l:
                raise RuntimeError('engine died')
            out.append(l)
            if l.startswith(word):
                return out

    def static(self, fen):
        self.send('position fen ' + fen); self.send('eval'); self.send('isready'); out = ''.join(self.until('readyok'))
        m = re.search(r'^eval (-?\d+)', out, re.M) if self.ours else re.search(r'Final evaluation:?\s+([+-]?\d+\.\d+)', out)
        return None if m is None else (float(m.group(1)) if self.ours else 100 * float(m.group(1)))

    def search(self, fen, nodes):
        self.send('ucinewgame'); self.send('position fen ' + fen); self.send(f'go nodes {nodes}'); sc = None; depth = 0
        for l in self.until('bestmove'):
            m = re.search(r'score (cp|mate) (-?\d+)', l)
            if m and 'bound' not in l:
                sc = float(m.group(2)) if m.group(1) == 'cp' else math.copysign(30000, float(m.group(2)))
                d = re.search(r'depth (\d+)', l); depth = int(d.group(1)) if d else depth
        return sc, depth


def job(a):
    spec, nodes, fens = a
    e = Eng(spec); out = [(e.static(f), 0) if nodes == 0 else e.search(f, nodes) for f in fens]
    e.send('quit')
    return out


def run(spec, nodes, fens):
    parts = [fens[i::PROCS] for i in range(PROCS)]
    with Pool(PROCS) as pool:
        res = pool.map(job, [(spec, nodes, p) for p in parts])
    out = [None] * len(fens)
    for i, r in enumerate(res):
        out[i::PROCS] = r
    return np.array([np.nan if v[0] is None else v[0] for v in out], dtype=np.float64), float(np.mean([v[1] for v in out]))


def fit_and_test(s, z, half):
    """one slope and offset for the judge's number (in pawns, clipped), fitted on the first half; per-position log-loss on the second"""
    x = np.clip(s / 100, -30, 30); X = np.stack([x, np.ones_like(x)], 1); w = np.zeros(2)
    for _ in range(40):
        q = 1 / (1 + np.exp(-X[:half] @ w)); H = (X[:half] * (q * (1 - q))[:, None]).T @ X[:half] + 1e-6 * np.eye(2)
        w = w - np.linalg.solve(H, X[:half].T @ (q - z[:half]))
    q = np.clip(1 / (1 + np.exp(-X[half:] @ w)), 1e-7, 1 - 1e-7)
    return -(z[half:] * np.log(q) + (1 - z[half:]) * np.log(1 - q))


if __name__ == '__main__':
    d = np.load('zero/label_truth.npz'); fens = [fen_of(r) for r in d['idx']]; zp = d['zp']; N = len(fens)
    print(f'{N} positions with the truth from Stockfish 19 playouts; our evaluator is generation {G}', flush=True)
    rows = []
    for name, spec in (('our evaluator, no search', OURS), ('Stockfish 14.1 hand-written evaluator, no search', SF14C),
                       ('Stockfish 14.1 network, no search', SF14N), ('Stockfish 19 network, no search', SF19)):
        v, _ = run(spec, 0, fens); ok = ~np.isnan(v)
        rows.append((name, v, ok, N, 0.0))
    for nodes in (2500, 20000):
        for name, spec in (('ours', OURS), ('Stockfish 14.1 hand-written', SF14C), ('Stockfish 14.1 network', SF14N)):
            v = np.full(N, np.nan); v[:NS], depth = run(spec, nodes, fens[:NS]); rows.append((f'{name}, searching {nodes:,} positions', v, ~np.isnan(v), NS, depth))
    base = None
    for scope in (N, NS):
        half = scope // 2
        print(f'\nfirst {scope} positions (fitted on {half}, tested on {scope - half}); log-loss against the truth, lower is better; difference to our evaluator; depth reached')
        for name, v, ok, n, depth in rows:
            if n < scope or not ok[:scope].all():
                if n >= scope:
                    print(f'  {name:52s} unreadable for {int((~ok[:scope]).sum())} positions')
                continue
            ll = fit_and_test(v[:scope], zp[:scope], half)
            if name.startswith('our evaluator'):
                base = ll
            dd = ll - base
            print(f'  {name:52s} {ll.mean():.4f}   {dd.mean():+.4f} +- {1.96 * dd.std() / math.sqrt(len(dd)):.4f}' + (f'   depth {depth:.1f}' if depth else ''), flush=True)
