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


class RawUCI:
    """a UCI engine driven over plain pipes: one write and a few reads per move. The chess library's engine layer
    (a background event loop that parses every info line) cost about as much per move as a 4,000-node search."""
    def __init__(self, cmd):
        import subprocess
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self.send('uci'); self.wait('uciok'); self.send('isready'); self.wait('readyok')

    def send(self, line):
        self.p.stdin.write(line + '\n'); self.p.stdin.flush()

    def wait(self, word):
        last = ''
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError('engine process died')
            if line.startswith(word):
                return line, last
            if line.startswith('info') and ' depth ' in line:
                last = line

    def configure(self, opts):
        for k, v in opts.items():
            self.send(f'setoption name {k} value {v}')

    def play(self, board, nodes=None, seconds=None):
        """-> (move, nodes searched, depth reached)"""
        fen = getattr(board, 'start_fen', None) or board.root().fen()        # the worker notes the opening on the board: rebuilding it costs a full replay
        self.send('position fen ' + fen + (' moves ' + ' '.join(m.uci() for m in board.move_stack) if board.move_stack else ''))
        self.send(f'go nodes {int(nodes)}' if nodes else f'go movetime {max(1, int(1000 * seconds))}')
        best, info = self.wait('bestmove')
        t = info.split(); g = lambda k: int(t[t.index(k) + 1]) if k in t else 0
        return chess.Move.from_uci(best.split()[1]), g('nodes'), g('depth')

    def quit(self):
        try:
            self.send('quit'); self.p.wait(timeout=2)
        except Exception:
            self.p.kill()


class CalcPlayer:
    """the C calculator: calc:<weights file | none>:<seconds per move, or nodes if >= 100>[:<engine binary>[:<Option=value,...>]]"""
    def __init__(self, net, t, binary='calc/calc', opts=''):
        self.e = RawUCI([binary] + ([net] if net != 'none' else []))
        if opts:                                   # engine options, e.g. ProbCut=180,CutLmr=1
            self.e.configure({k: int(v) for k, v in (kv.split('=') for kv in opts.split(','))})
        self.t = float(t); self.moves = 0; self.secs = 0.0; self.nodes = 0; self.depth = 0

    def go(self, board, t):
        m, n, d = self.e.play(board, nodes=t) if t >= 100 else self.e.play(board, seconds=t)     # >= 100 means a node budget
        self.nodes += n; self.depth += d
        return m

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
    def __init__(self, net, ckpt, t, lam, cap=40, kind='p', binary='calc/calc'):
        from engine import Net
        CalcPlayer.__init__(self, net, t, binary)
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


class DuoVPlayer(CalcPlayer):
    """two brains: the calculator searches; each root move's score is corrected by OURO's judgment of the position
    after it, measured against the calculator network's own static judgment of that position.
    On fresh self-play positions the game result is predicted best by  search + 0.88 * OURO - 0.87 * static network
    (zero/judge_cmp.py), i.e. OURO's opinion plus what the search found beyond the static network.
    duov:<weights .bin, with its .ema.pt beside it>:<ouro ckpt>:<nodes or seconds>:<scale>[:<cap>[:<engine binary>]]"""
    def __init__(self, net, ckpt, t, scale=1.0, cap=100, binary='calc/calc'):
        import torch
        from engine import Net
        CalcPlayer.__init__(self, net, t, binary)
        self.ouro = Net(ckpt, threads=1); self.scale = float(scale); self.cap = float(cap)
        self.sd = torch.load(net + '.ema.pt', map_location='cpu'); self.torch = torch
        i = np.arange(768); self.lut = torch.from_numpy(np.concatenate([((i // 64 + 6) % 12) * 64 + ((i % 64) ^ 56), [768]]).astype(np.int64))
        self.ptw = torch.cat([torch.tensor([0, 1, 1, 2, 4, 0] * 2, dtype=torch.float32).repeat_interleave(64), torch.zeros(1)])

    def static(self, boards):
        """the calculator network's static judgment (logit of the win probability of the side to move)"""
        torch, sd = self.torch, self.sd; F = torch.nn.functional
        idx = torch.full((len(boards), 32), 768, dtype=torch.long)
        for k, b in enumerate(boards):
            for j, (sq, pc) in enumerate(list(b.piece_map().items())[:32]):
                idx[k, j] = ((0 if pc.color == b.turn else 6) + pc.piece_type - 1) * 64 + (sq if b.turn == chess.WHITE else sq ^ 56)
        with torch.no_grad():
            ao = sd['b'] + F.embedding_bag(idx, sd['E'], mode='sum', padding_idx=768); ap = sd['b'] + F.embedding_bag(self.lut[idx], sd['E'], mode='sum', padding_idx=768)
            lin = sd['c'] + F.embedding_bag(idx, sd['S'], mode='sum', padding_idx=768)
            ph = (self.ptw[idx].sum(1) / 24).clamp(max=1.0); bk = (((idx < 768).sum(1) - 2) // 4).clamp(0, sd['wo'].shape[0] - 1)
            return (ph * lin[:, 0] + (1 - ph) * lin[:, 1] + sd['cb'][bk] + (ao.clamp(0, 1) ** 2 * sd['wo'][bk]).sum(1) + (ap.clamp(0, 1) ** 2 * sd['wp'][bk]).sum(1)).numpy()

    def move(self, board, history):
        t = time.time(); moves = list(board.legal_moves); kids = []
        for m in moves:
            board.push(m); kids.append(board.copy(stack=False)); board.pop()
        live = [k for k, b in enumerate(kids) if any(True for _ in b.legal_moves)]       # positions that are not mate or stalemate
        if len(moves) > 1 and live:
            ev = self.ouro.evaluate([kids[k] for k in live], loops=1)
            o = np.array([math.log(max(1e-4, min(1 - 1e-4, e[2])) / (1 - max(1e-4, min(1 - 1e-4, e[2])))) for e in ev])   # the opponent's view
            a = self.static([kids[k] for k in live])
            g = np.zeros(len(moves)); g[live] = -271.58 * self.scale * (0.88 * o - 0.87 * a)                               # for the side to move at the root
            g[live] -= g[live].max(); g = np.maximum(g, -self.cap)
            self.e.configure({'Bonus': ' '.join(f'{m.uci()[:4]} {int(round(x))}' for m, x in zip(moves, g))})
        m = self.go(board, self.t if self.t >= 100 else max(0.01, self.t - (time.time() - t)))
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
    if p[0] == 'duov':
        return DuoVPlayer(*p[1:])
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
            board = chess.Board(f); board.start_fen = f
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
