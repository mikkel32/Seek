"""How much of the gap to Stockfish is judgement of a single position, and how much is search?

Positions from our own fresh self-play games, each with the result of its game. Every judge gives its number for the
position; one slope and offset are fitted on half of the positions (so every judge is read in its own best scale), and
the log-loss against the game result is measured on the other half. Lower is better.

Judges: our evaluator without search, Stockfish's evaluators without search, and both engines searching the same number
of positions. The positions and results are from OUR games, which favours our evaluator: it was trained on exactly this.
Stockfish is only a measuring instrument here; nothing is learned from it.

A second, cleaner truth for part of the positions: the result when Stockfish 19 plays the position out against itself
(20,000 positions per move). That is as close to "what the position is really worth" as this machine can get quickly,
and it does not depend on how our own engine happened to play on.

  python3 zero/eval_gap.py [positions=20000] [searched and played out=4000] [processes=4]
"""
import glob, json, math, os, re, subprocess, sys
import numpy as np, chess
from multiprocessing import Pool

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
N = int(sys.argv[1]) if len(sys.argv) > 1 else 20000; NS = int(sys.argv[2]) if len(sys.argv) > 2 else 4000; PROCS = int(sys.argv[3]) if len(sys.argv) > 3 else 4
REC = np.dtype([('idx', '<i2', 32), ('score', '<i2'), ('result', 'i1'), ('ply', 'i1')])
G = json.load(open('zero/cont_state.json'))['gen']
OURS = f'calc/calc_untuned zero/g{G}.bin'; SF14 = '/usr/games/stockfish'; SF19 = '/root/sf19/stockfish19'
SYM = 'PNBRQKpnbrqk'


def fen_of(idx):
    """the mover's-view feature list as a position with White to move (no castling rights, no en passant)"""
    b = chess.Board(None)
    for v in idx:
        if v < 768:
            b.set_piece_at(int(v) % 64, chess.Piece.from_symbol(SYM[int(v) // 64]))
    b.turn = chess.WHITE
    return b


class Eng:
    def __init__(self, cmd):
        self.p = subprocess.Popen(cmd.split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self.send('uci'); self.until('uciok'); self.send('setoption name Hash value 16'); self.send('setoption name Threads value 1'); self.send('isready'); self.until('readyok')

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

    def static(self, fen, ours):
        self.send('position fen ' + fen); self.send('eval'); self.send('isready'); out = ''.join(self.until('readyok'))
        m = re.search(r'^eval (-?\d+)', out, re.M) if ours else re.search(r'Final evaluation:?\s+([+-]?\d+\.\d+)', out)
        return None if m is None else (float(m.group(1)) if ours else 100 * float(m.group(1)))

    def search(self, fen, nodes):
        self.send('ucinewgame'); self.send('position fen ' + fen); self.send(f'go nodes {nodes}'); sc = None
        for l in self.until('bestmove'):
            m = re.search(r'score (cp|mate) (-?\d+)', l)
            if m and 'bound' not in l:
                sc = float(m.group(2)) if m.group(1) == 'cp' else math.copysign(30000, float(m.group(2)))
        return sc


def job(a):
    cmd, kind, nodes, fens = a
    e = Eng(cmd); out = [e.static(f, cmd == OURS) if kind == 'static' else e.search(f, nodes) for f in fens]
    e.send('quit')
    return out


def playout_job(fens):
    """Stockfish 19 against itself from each position, 20,000 positions per move -> result for the side to move at the start"""
    e = Eng(SF19); out = []
    for f in fens:
        b = chess.Board(f); moves = []; hi = lo = 0; res = None; e.send('ucinewgame')
        while res is None:
            if b.is_checkmate():
                res = 0.0 if b.turn == chess.WHITE else 1.0
            elif b.is_stalemate() or b.is_insufficient_material() or b.halfmove_clock >= 100 or b.is_repetition(3) or len(moves) >= 400:
                res = 0.5
            else:
                e.send('position fen ' + f + (' moves ' + ' '.join(moves) if moves else '')); e.send('go nodes 20000'); sc = 0.0; bm = None
                for l in e.until('bestmove'):
                    m = re.search(r'score (cp|mate) (-?\d+)', l)
                    if m:
                        sc = float(m.group(2)) if m.group(1) == 'cp' else math.copysign(30000, float(m.group(2)))
                    if l.startswith('bestmove'):
                        bm = l.split()[1]
                w = sc if b.turn == chess.WHITE else -sc                       # decided games are stopped once both sides agree for six plies
                hi = hi + 1 if w >= 1000 else 0; lo = lo + 1 if w <= -1000 else 0
                if hi >= 6:
                    res = 1.0
                elif lo >= 6:
                    res = 0.0
                else:
                    b.push_uci(bm); moves.append(bm)
        out.append(res)
    e.send('quit')
    return out


def run(cmd, kind, nodes, fens):
    parts = [fens[i::PROCS] for i in range(PROCS)]
    with Pool(PROCS) as pool:
        res = pool.map(job, [(cmd, kind, nodes, p) for p in parts])
    out = [None] * len(fens)
    for i, r in enumerate(res):
        out[i::PROCS] = r
    return np.array([np.nan if v is None else v for v in out], dtype=np.float64)


def fit_and_test(s, z, half):
    """one slope and offset for the judge's number (in pawns, clipped), fitted on the first half; per-position log-loss on the second"""
    x = np.clip(s / 100, -30, 30); X = np.stack([x, np.ones_like(x)], 1); w = np.zeros(2)
    for _ in range(40):
        q = 1 / (1 + np.exp(-X[:half] @ w)); H = (X[:half] * (q * (1 - q))[:, None]).T @ X[:half] + 1e-6 * np.eye(2)
        w = w - np.linalg.solve(H, X[:half].T @ (q - z[:half]))
    q = np.clip(1 / (1 + np.exp(-X[half:] @ w)), 1e-7, 1 - 1e-7)
    return -(z[half:] * np.log(q) + (1 - z[half:]) * np.log(1 - q))


if __name__ == '__main__':
    files = sorted(f for f in glob.glob('zero/pool/*.bin'))
    rs = np.random.RandomState(11); rs.shuffle(files); pos = []; per = max(8, 3 * N // max(1, len(files)))
    for f in files:
        r = np.fromfile(f, dtype=REC); r = r[r['ply'] >= 0]
        idx = r['idx']; ok = ~((idx == 324) | (idx == 764)).any(1)                 # both kings have left home: castling is certainly gone
        r = r[ok]
        for k in rs.choice(len(r), min(per, len(r)), replace=False):
            b = fen_of(r['idx'][k])
            if b.is_valid() and not b.is_check():
                pos.append((b.fen(), float(r['score'][k]), (float(r['result'][k]) + 1) / 2))
        if len(pos) >= N:
            break
    pos = pos[:N]; rs.shuffle(pos); fens = [p[0] for p in pos]; rec = np.array([p[1] for p in pos]); z = np.array([p[2] for p in pos])
    print(f'{len(pos)} positions from {len(files)} files of generation-{G} self-play; results: wins {np.mean(z == 1):.3f}, draws {np.mean(z == 0.5):.3f}, losses {np.mean(z == 0):.3f} for the side to move', flush=True)
    judges = {'our evaluator, no search': run(OURS, 'static', 0, fens),
              'Stockfish 14.1 evaluator, no search': run(SF14, 'static', 0, fens),
              'Stockfish 19 evaluator, no search': run(SF19, 'static', 0, fens),
              'our search as played (2,500 positions)': rec}
    sub = fens[:NS]
    for nodes in (2500, 50000):
        for name, cmd in (('ours', OURS), ('Stockfish 19', SF19)):
            v = np.full(len(fens), np.nan); v[:NS] = run(cmd, 'search', nodes, sub); judges[f'{name} searching {nodes:,} positions'] = v
    with Pool(PROCS) as pool:
        parts = pool.map(playout_job, [sub[i::PROCS] for i in range(PROCS)])
    zp = np.zeros(NS)
    for i, r in enumerate(parts):
        zp[i::PROCS] = r
    print(f'Stockfish 19 played out {NS} positions against itself: wins {np.mean(zp == 1):.3f}, draws {np.mean(zp == 0.5):.3f}, losses {np.mean(zp == 0):.3f} for the side to move; '
          f'agrees with the result of our own game in {np.mean(zp == z[:NS]):.3f} of them', flush=True)
    half = NS // 2; base = fit_and_test(judges['our evaluator, no search'][:NS], zp, half)
    print(f'\nTRUTH = Stockfish 19 playing the position out ({NS} positions; fitted on {half}, tested on {NS - half}):  log-loss, and the difference to our evaluator (negative = better)')
    for k, v in judges.items():
        if not np.isnan(v[:NS]).any():
            ll = fit_and_test(v[:NS], zp, half); d = ll - base
            print(f'  {k:42s} {ll.mean():.4f}   {d.mean():+.4f} +- {1.96 * d.std() / math.sqrt(len(d)):.4f}', flush=True)
    for scope, n in (('TRUTH = the result of our own game, all positions', len(fens)), (f'TRUTH = the result of our own game, the {NS} searched positions', NS)):
        names = [k for k, v in judges.items() if not np.isnan(v[:n]).any()]; half = n // 2
        if not names:
            continue
        base = fit_and_test(judges['our evaluator, no search'][:n], z[:n], half)
        print(f'\n{scope} ({n}; fitted on {half}, tested on {n - half}):  log-loss against the game result, and the difference to our evaluator (negative = better)')
        for k in names:
            ll = fit_and_test(judges[k][:n], z[:n], half); d = ll - base
            print(f'  {k:42s} {ll.mean():.4f}   {d.mean():+.4f} +- {1.96 * d.std() / math.sqrt(len(d)):.4f}', flush=True)
    c = lambda a, b: float(np.corrcoef(np.clip(a, -1500, 1500), np.clip(b, -1500, 1500))[0, 1])
    print(f"\nagreement between evaluators without search (correlation of scores): ours and Stockfish 19 {c(judges['our evaluator, no search'], judges['Stockfish 19 evaluator, no search']):.3f}, "
          f"Stockfish 14.1 and Stockfish 19 {c(judges['Stockfish 14.1 evaluator, no search'], judges['Stockfish 19 evaluator, no search']):.3f}")
