"""Matches: OURO checkpoints vs each other (searchless or with search) and vs Stockfish levels.

player spec strings:
  net:<ckpt>:<mode p|q|mix>:<loops>
  mcts:<ckpt>:<sims>:<leaf_loops>:<deepen e.g. 8-2,48-4 | none>[:<q_prior>]
  sf:<elo or 0 for full strength>:<movetime seconds>
"""
import argparse, json, math, os, random, sys, time
import numpy as np, chess, chess.engine
from multiprocessing import Pool

SF = '/usr/games/stockfish'


def openings(n, seed=0):
    if os.path.exists('data/openings.txt'):
        out = open('data/openings.txt').read().split('\n')
        out = [f for f in out if f]
        random.Random(seed).shuffle(out)
        return out[:n]
    T = np.load('data/cb_test.npz')
    idx = np.arange(1, len(T['boards']), 2)          # dev half only
    fens = T['fens'][idx]
    qmax = np.where(T['moves'][idx] >= 0, T['qs'][idx], -1).max(1)
    out = []
    for f, q in zip(fens, qmax):
        parts = f.split(' ')
        if 5 <= int(parts[5]) <= 10 and abs(q - 0.5) < 0.08:
            out.append(f)
    random.Random(seed).shuffle(out)
    return out[:n]


def elo_from(w, d, l):
    n = w + d + l
    s = (w + 0.5 * d) / n
    var = (w * (1 - s) ** 2 + d * (0.5 - s) ** 2 + l * (0 - s) ** 2) / n
    se = math.sqrt(var / n)
    f = lambda x: -400 * math.log10(1 / min(max(x, 1e-4), 1 - 1e-4) - 1)
    return dict(n=n, w=w, d=d, l=l, score=round(s, 4), elo=round(f(s), 1), lo=round(f(s - 1.96 * se), 1), hi=round(f(s + 1.96 * se), 1))


def result_of(board, max_plies, sf=None):
    if board.is_checkmate():
        return 0.0 if board.turn == chess.WHITE else 1.0      # white's score
    if board.is_stalemate() or board.is_insufficient_material() or board.halfmove_clock >= 100 or board.is_repetition(3):
        return 0.5
    if board.ply() >= max_plies:
        if sf is None:
            return 0.5
        info = sf.analyse(board, chess.engine.Limit(depth=12))
        cp = info['score'].white().score(mate_score=10000)
        return 1.0 if cp > 300 else (0.0 if cp < -300 else 0.5)
    return None


# ---------------------------------------------------------------- batched searchless
def play_batched(netA, a, netB, b, fens, max_plies=300):
    """every opening is played twice (colours swapped). returns list of A's scores."""
    games = []
    for f in fens:
        games.append([chess.Board(f), True])     # A is white
        games.append([chess.Board(f), False])
    scores = [None] * len(games)
    sf = chess.engine.SimpleEngine.popen_uci(SF)
    active = list(range(len(games)))
    try:
        while active:
            for is_a in (True, False):
                # A moves when (white to move and A is white) or (black to move and A is black)
                sel = [i for i in active if (games[i][0].turn == chess.WHITE) == (games[i][1] == is_a)]
                if sel:
                    net, cfg = (netA, a) if is_a else (netB, b)
                    mv = net.pick([games[i][0] for i in sel], mode=cfg[0], loops=cfg[1])
                    for i, m in zip(sel, mv):
                        games[i][0].push(m)
                nxt = []
                for i in active:                      # a game can end after either side's move
                    r = result_of(games[i][0], max_plies, sf)
                    if r is None:
                        nxt.append(i)
                    else:
                        scores[i] = r if games[i][1] else 1.0 - r
                active = nxt
    finally:
        sf.quit()
    return scores


# ---------------------------------------------------------------- generic players
class NetPlayer:
    def __init__(self, ckpt, mode, loops):
        from engine import Net
        self.net = Net(ckpt, threads=1); self.mode = mode; self.loops = int(loops)
        self.moves = 0; self.secs = 0.0

    def move(self, board, history):
        t = time.time()
        m = self.net.pick([board], self.mode, self.loops)[0]
        self.moves += 1; self.secs += time.time() - t
        return m

    def stats(self):
        return dict(moves=self.moves, secs=self.secs, evals=self.moves, block_evals=0)


class MCTSPlayer:
    def __init__(self, ckpt, sims, leaf, deepen, q_prior=1.0):
        from engine import Net, MCTS
        self.net = Net(ckpt, threads=1)
        dp = () if deepen == 'none' else tuple(tuple(int(x) for x in t.split('-')) for t in deepen.split(','))
        self.m = MCTS(self.net, sims=int(sims), leaf_loops=int(leaf), deepen=dp, q_prior=float(q_prior))
        self.moves = 0; self.secs = 0.0

    def move(self, board, history):
        t = time.time()
        m = self.m.search(board, history)[0]
        self.moves += 1; self.secs += time.time() - t
        return m

    def stats(self):
        return dict(moves=self.moves, secs=self.secs, evals=self.m.evals, block_evals=self.m.block_evals)


class SFPlayer:
    def __init__(self, elo, t):
        self.e = chess.engine.SimpleEngine.popen_uci(SF)
        self.e.configure({'Threads': 1, 'Hash': 16})
        if int(elo) > 0:
            self.e.configure({'UCI_LimitStrength': True, 'UCI_Elo': int(elo)})
        self.t = float(t)

    def move(self, board, history):
        lim = chess.engine.Limit(nodes=int(self.t)) if self.t >= 100 else chess.engine.Limit(time=self.t)   # >= 100 means a node budget
        return self.e.play(board, lim).move

    def close(self):
        self.e.quit()


class CalcPlayer:
    """the C calculator: calc:<weights file | none>:<seconds per move, or nodes if >= 100>[:<engine binary>[:<Option=value,...>]]"""
    def __init__(self, net, t, binary='calc/calc', opts=''):
        self.e = chess.engine.SimpleEngine.popen_uci([binary] + ([net] if net != 'none' else []))
        if opts:                                   # engine options, e.g. ProbCut=180,CutLmr=1
            self.e.configure({k: int(v) for k, v in (kv.split('=') for kv in opts.split(','))})
        self.t = float(t); self.moves = 0; self.secs = 0.0; self.nodes = 0; self.depth = 0

    def go(self, board, t):
        lim = chess.engine.Limit(nodes=int(t)) if t >= 100 else chess.engine.Limit(time=t)     # >= 100 means a node budget
        r = self.e.play(board, lim, info=chess.engine.INFO_BASIC)
        self.nodes += r.info.get('nodes', 0); self.depth += r.info.get('depth', 0)
        return r.move

    def move(self, board, history):
        t = time.time()
        m = self.go(board, self.t)
        self.moves += 1; self.secs += time.time() - t
        return m

    def stats(self):                      # evals = nodes searched, block_evals = search depth reached
        return dict(moves=self.moves, secs=self.secs, evals=self.nodes, block_evals=self.depth)

    def close(self):
        self.e.quit()


class DuoPlayer(CalcPlayer):
    """two systems: the calculator searches, OURO adds a prior to the root moves.
    duo:<weights>:<ouro ckpt>:<seconds per move>:<centipawns per nat of policy>[:<cap>[:<kind p|q>]]"""
    def __init__(self, net, ckpt, t, lam, cap=40, kind='p'):
        from engine import Net
        CalcPlayer.__init__(self, net, t)
        self.ouro = Net(ckpt, threads=1); self.lam = float(lam); self.cap = float(cap); self.kind = kind

    def move(self, board, history):
        t = time.time()
        moves, p, v, q = self.ouro.evaluate([board], loops=2)[0]
        if self.kind == 'q':              # value-based prior: win-probability gap to the best move, in centipawn-like units
            g = self.lam * 100.0 * (q - q.max())
        else:
            lp = np.log(p + 1e-9); g = self.lam * (lp - lp.max())
        g = np.maximum(g, -self.cap)
        self.e.configure({'Bonus': ' '.join(f'{m.uci()[:4]} {int(round(x))}' for m, x in zip(moves, g))})
        m = self.go(board, max(0.01, self.t - (time.time() - t)))
        self.moves += 1; self.secs += time.time() - t
        return m


class UCIPlayer:
    """any other UCI engine: uci:<command line>:<seconds per move>"""
    def __init__(self, cmd, t):
        self.e = chess.engine.SimpleEngine.popen_uci(cmd.split()); self.t = float(t)
        try:
            self.e.configure({'Hash': 32})
        except Exception:
            pass

    def move(self, board, history):
        return self.e.play(board, chess.engine.Limit(time=self.t)).move

    def close(self):
        self.e.quit()


def make(spec):
    p = spec.split(':')
    if p[0] == 'uci':
        return UCIPlayer(p[1], p[2])
    if p[0] == 'calc':
        return CalcPlayer(*p[1:])
    if p[0] == 'duo':
        return DuoPlayer(*p[1:])
    if p[0] == 'net':
        return NetPlayer(p[1], p[2], p[3])
    if p[0] == 'mcts':
        return MCTSPlayer(*p[1:])
    if p[0] == 'sf':
        return SFPlayer(p[1], p[2])
    raise ValueError(spec)


def _worker(args):
    import torch
    torch.set_num_threads(1)
    spec_a, spec_b, fens, max_plies = args
    A, B = make(spec_a), make(spec_b)
    sf = chess.engine.SimpleEngine.popen_uci(SF)
    out = []
    for f in fens:
        for a_white in (True, False):
            board = chess.Board(f)
            hist = [board._transposition_key()]
            while True:
                r = result_of(board, max_plies, sf)
                if r is not None:
                    break
                pl = A if (board.turn == chess.WHITE) == a_white else B
                board.push(pl.move(board, hist[:-1]))
                hist.append(board._transposition_key())
            out.append((r if a_white else 1.0 - r, board.ply()))
    st = [p.stats() if hasattr(p, 'stats') else None for p in (A, B)]
    for p in (A, B):
        if hasattr(p, 'close'):
            p.close()
    sf.quit()
    return out, st


def play(spec_a, spec_b, n_openings, workers=4, seed=0, max_plies=300):
    fens = openings(n_openings, seed)
    chunks = [fens[i::workers] for i in range(workers)]
    with Pool(workers) as pool:
        res = pool.map(_worker, [(spec_a, spec_b, c, max_plies) for c in chunks if c])
    sc = [s for r, _ in res for s, _p in r]
    plies = [p for r, _ in res for _s, p in r]
    out = elo_from(sum(s == 1.0 for s in sc), sum(s == 0.5 for s in sc), sum(s == 0.0 for s in sc))
    out['avg_plies'] = round(float(np.mean(plies)), 1)
    for side, k in (('a', 0), ('b', 1)):            # search cost per move (network block applications)
        sts = [st[k] for _, st in res if st[k]]
        if sts:
            mv = max(1, sum(x['moves'] for x in sts))
            out[f'{side}_evals_per_move'] = round(sum(x['evals'] for x in sts) / mv, 1)
            out[f'{side}_blocks_per_move'] = round(sum(x['block_evals'] for x in sts) / mv, 1)
            out[f'{side}_secs_per_move'] = round(sum(x['secs'] for x in sts) / mv, 3)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', required=True); ap.add_argument('--b', required=True)
    ap.add_argument('--openings', type=int, default=50); ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--batched', action='store_true')
    ap.add_argument('--max_plies', type=int, default=300)
    a = ap.parse_args()
    t = time.time()
    if a.batched:
        import torch
        from engine import Net
        torch.set_num_threads(a.workers)
        pa, pb = a.a.split(':'), a.b.split(':')
        na = Net(pa[1]); nb = na if pb[1] == pa[1] else Net(pb[1])
        sc = play_batched(na, (pa[2], int(pa[3])), nb, (pb[2], int(pb[3])), openings(a.openings, a.seed), a.max_plies)
        out = elo_from(sum(s == 1.0 for s in sc), sum(s == 0.5 for s in sc), sum(s == 0.0 for s in sc))
    else:
        out = play(a.a, a.b, a.openings, a.workers, a.seed, a.max_plies)
    out['a'] = a.a; out['b'] = a.b; out['secs'] = round(time.time() - t)
    print(json.dumps(out))
