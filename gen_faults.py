"""Fault mining: positions the network actually reaches in its own games, labelled by
Stockfish with the value of EVERY legal move (MultiPV = all). Output is a dense-Q dataset
in the training format (moves sorted best-first, wps = mover's win probability)."""
import argparse, math, random, sys, time
import numpy as np, chess, chess.engine, torch
from multiprocessing import Pool
from chessenc import encode_fen, encode_move

SF = '/usr/games/stockfish'


def selfplay_positions(ckpt, n_openings, eps, seed, loops=2, max_plies=160, threads=4):
    from engine import Net
    from match import openings
    torch.set_num_threads(threads)
    net = Net(ckpt)
    rnd = random.Random(seed)
    boards = [chess.Board(f) for f in openings(n_openings, seed)]
    seen = {}
    active = list(range(len(boards)))
    while active:
        res = net.evaluate([boards[i] for i in active], loops)
        nxt = []
        for i, (moves, p, v, q) in zip(active, res):
            b = boards[i]
            key = ' '.join(b.fen().split(' ')[:4])
            if key not in seen:
                seen[key] = b.fen()
            if rnd.random() < eps:                       # exploration: sample from the policy
                mv = rnd.choices(moves, weights=np.maximum(p, 1e-4))[0]
            else:
                mv = moves[int(np.argmax(np.log(p + 1e-12) + net.beta * q))]
            b.push(mv)
            if not (b.is_game_over() or b.ply() >= max_plies or b.halfmove_clock >= 60 or b.is_repetition(2)):
                nxt.append(i)
        active = nxt
    return list(seen.values())


def label(args):
    fens, depth = args
    e = chess.engine.SimpleEngine.popen_uci(SF)
    e.configure({'Threads': 1, 'Hash': 32})
    out = []
    for f in fens:
        b = chess.Board(f)
        n = b.legal_moves.count()
        if n == 0:
            continue
        try:
            infos = e.analyse(b, chess.engine.Limit(depth=depth), multipv=n)
        except Exception:
            continue
        item = []
        for inf in infos:
            if 'pv' not in inf or not inf['pv']:
                continue
            sc = inf['score'].pov(b.turn)
            if sc.is_mate():
                w = 1.0 if sc.mate() > 0 else 0.0
            else:
                w = 1.0 / (1.0 + math.exp(-0.00368208 * sc.score()))
            item.append((w, inf['pv'][0].uci()))
        if item:
            item.sort(key=lambda x: -x[0])
            out.append((f, item))
    e.quit()
    return out


def to_arrays(labelled, out):
    K = max(len(it) for _, it in labelled)
    n = len(labelled)
    boards = np.zeros((n, 64), dtype=np.uint8); meta = np.zeros((n, 2), dtype=np.uint8)
    moves = np.full((n, K), -1, dtype=np.int16); wps = np.zeros((n, K), dtype=np.float16)
    fens = []
    for i, (f, it) in enumerate(labelled):
        board, castle, ep, black = encode_fen(f)
        boards[i] = np.frombuffer(board, dtype=np.uint8); meta[i] = (castle, ep)
        for j, (w, mv) in enumerate(it):
            moves[i, j] = encode_move(mv, black, board); wps[i, j] = w
        fens.append(f)
    np.savez(out, boards=boards, meta=meta, moves=moves, wps=wps, fens=np.array(fens))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt'); ap.add_argument('out')
    ap.add_argument('--openings', type=int, default=400); ap.add_argument('--eps', type=float, default=0.12)
    ap.add_argument('--max_pos', type=int, default=20000); ap.add_argument('--depth', type=int, default=10)
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    t = time.time()
    fens = selfplay_positions(a.ckpt, a.openings, a.eps, a.seed)
    random.Random(a.seed).shuffle(fens)
    fens = fens[:a.max_pos]
    print('positions', len(fens), 'selfplay secs', round(time.time() - t), flush=True)
    t = time.time()
    chunks = [fens[i::a.workers * 8] for i in range(a.workers * 8)]
    with Pool(a.workers) as pool:
        res = pool.map(label, [(c, a.depth) for c in chunks], chunksize=1)
    labelled = [x for r in res for x in r]
    to_arrays(labelled, a.out)
    print('labelled', len(labelled), 'secs', round(time.time() - t), flush=True)
