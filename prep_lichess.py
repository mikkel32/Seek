"""Lichess position-evaluation shard -> compact training arrays.

Per unique position: up to 5 candidate moves (first move of each PV, union across
the stored evals, deepest eval wins on conflicts) with the mover's win probability.
moves[:,0] is the best move, wps[:,0] the position value.
"""
import sys, math, numpy as np, pyarrow.parquet as pq
from multiprocessing import Pool
from chessenc import encode_fen, encode_move

PATH = sys.argv[1]
OUT = sys.argv[2]
G0, G1 = int(sys.argv[3]), int(sys.argv[4])
K = 5


def wp_white(cp, mate):
    if mate == mate and mate is not None:  # not NaN
        return 1.0 if mate > 0 else 0.0
    return 1.0 / (1.0 + math.exp(-0.00368208 * cp))


def work(g):
    f = pq.ParquetFile(PATH)
    t = f.read_row_group(g)
    fens = t['fen'].to_pylist()
    lines = t['line'].to_pylist()
    depth = t['depth'].to_pylist()
    cps = t['cp'].to_pylist()
    mates = t['mate'].to_pylist()
    n = len(fens)
    B, C, E, M, W, P, Q = [], [], [], [], [], [], []
    i = 0
    while i < n:
        fen = fens[i]
        j = i
        cand = {}
        while j < n and fens[j] == fen:
            ln = lines[j]
            if ln:
                mv = ln[:5] if (len(ln) > 4 and ln[4] != ' ') else ln[:4]
                d = depth[j]
                if mates[j] is not None:
                    w = 1.0 if mates[j] > 0 else 0.0
                elif cps[j] is not None:
                    w = 1.0 / (1.0 + math.exp(-0.00368208 * cps[j]))
                else:
                    j += 1
                    continue
                old = cand.get(mv)
                if old is None or d > old[0]:
                    cand[mv] = (d, w, ln)
            j += 1
        i = j
        if not cand:
            continue
        try:
            board, castle, ep, black = encode_fen(fen)
            if len(board) != 64 or board.count(6) != 1 or board.count(12) != 1:
                continue
            items = []
            for mv, (d, w, ln) in cand.items():
                if black:
                    w = 1.0 - w
                items.append((w, d, encode_move(mv, black, board), ln))
        except Exception:
            continue
        items.sort(key=lambda x: (-x[0], -x[1]))
        items = items[:K]
        ok = True
        for w, d, m, _ in items:
            fr = (m // 64) if m < 4096 else 48 + (m - 4096) // 24
            if not (1 <= board[fr] <= 6):
                ok = False
        if not ok:
            continue
        mv = [m for _, _, m, _ in items] + [-1] * (K - len(items))
        ws = [w for w, _, _, _ in items] + [0.0] * (K - len(items))
        # look-ahead targets: opponent's reply and own follow-up along the best line (root frame)
        pv = [-1, -1]
        toks = items[0][3].split(' ')
        m1 = mv[0]
        f1 = (m1 // 64) if m1 < 4096 else 48 + (m1 - 4096) // 24
        for j in (1, 2):
            if len(toks) > j and len(toks[j]) >= 4:
                u = toks[j]
                f = (ord(u[0]) - 97) + 8 * (ord(u[1]) - 49)
                t = (ord(u[2]) - 97) + 8 * (ord(u[3]) - 49)
                if black:
                    f ^= 56; t ^= 56
                if j == 1 and f == 60 and board[60] == 12 and board[t] == 10 and t in (63, 56):
                    t = 62 if t == 63 else 58
                if j == 2 and f == 4 and board[4] == 6 and board[t] == 4 and t in (7, 0) and f1 not in (4, t):
                    t = 6 if t == 7 else 2
                pv[j - 1] = f * 64 + t
        P.append(pv)
        # is the reply to each candidate a capture? 1 yes, 0 no, 2 unknown  (used to keep only quiet positions for evaluation training)
        rq = [2] * K
        for jj, (w_, d_, m_, ln_) in enumerate(items):
            tk = ln_.split(' ')
            if len(tk) > 1 and len(tk[1]) >= 4:
                fa = (m_ // 64) if m_ < 4096 else 48 + (m_ - 4096) // 24
                ta = (m_ % 64) if m_ < 4096 else 56 + ((m_ - 4096) // 3) % 8
                t2 = (ord(tk[1][2]) - 97) + 8 * (ord(tk[1][3]) - 49)
                if black:
                    t2 ^= 56
                rq[jj] = 1 if (t2 == ta or (1 <= board[t2] <= 6 and t2 != fa)) else 0
        Q.append(rq)
        B.append(board); C.append(castle); E.append(ep); M.append(mv); W.append(ws)
    boards = np.frombuffer(b''.join(B), dtype=np.uint8).reshape(-1, 64)
    meta = np.stack([np.array(C, dtype=np.uint8), np.array(E, dtype=np.uint8)], 1)
    return boards, meta, np.array(M, dtype=np.int16), np.array(W, dtype=np.float16), np.array(P, dtype=np.int16), np.array(Q, dtype=np.uint8)


if __name__ == '__main__':
    with Pool(4) as p:
        res = p.map(work, range(G0, G1), chunksize=1)
    boards = np.concatenate([r[0] for r in res]); meta = np.concatenate([r[1] for r in res])
    moves = np.concatenate([r[2] for r in res]); wps = np.concatenate([r[3] for r in res])
    pv = np.concatenate([r[4] for r in res]); rq = np.concatenate([r[5] for r in res])
    np.savez(OUT, boards=boards, meta=meta, moves=moves, wps=wps, pv=pv, rq=rq)
    print('positions', len(boards), 'avg candidates', float((moves >= 0).sum(1).mean()),
          'multi-move frac', float(((moves >= 0).sum(1) > 1).mean()))
