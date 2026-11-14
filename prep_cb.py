"""ChessBench (DeepMind searchless_chess) -> arrays.
 train shard : one (position, random move, Q) record per position  -> Q-only samples
 test file   : all legal moves with Q for ~62k positions          -> benchmark (split A/B)
"""
import sys, struct, os, numpy as np, collections
from chessenc import encode_fen, encode_move


def records(path, limit=None):
    sz = os.path.getsize(path)
    with open(path, 'rb') as f:
        f.seek(sz - 8)
        idx_start = struct.unpack('<Q', f.read(8))[0]
        n = (sz - idx_start) // 8
        if limit:
            n = min(n, limit)
        f.seek(idx_start)
        offs = np.frombuffer(f.read(8 * n), dtype='<u8')
        f.seek(0)
        buf = f.read(int(offs[-1]))
    prev = 0
    for i in range(n):
        e = int(offs[i])
        rec = buf[prev:e]
        prev = e
        l = rec[0]
        ml = rec[1 + l]
        yield rec[1:1 + l].decode(), rec[2 + l:2 + l + ml].decode(), struct.unpack('>d', rec[2 + l + ml:2 + l + ml + 8])[0]


def prep_train(path, out, limit):
    B, C, E, M, W = [], [], [], [], []
    for fen, mv, q in records(path, limit):
        board, castle, ep, black = encode_fen(fen)
        B.append(board); C.append(castle); E.append(ep)
        M.append(encode_move(mv, black, board)); W.append(q)
    n = len(B)
    moves = np.full((n, 5), -1, dtype=np.int16); moves[:, 0] = M
    wps = np.zeros((n, 5), dtype=np.float16); wps[:, 0] = W
    np.savez(out, boards=np.frombuffer(b''.join(B), dtype=np.uint8).reshape(-1, 64),
             meta=np.stack([np.array(C, dtype=np.uint8), np.array(E, dtype=np.uint8)], 1), moves=moves, wps=wps)
    print('cb train', n)


def prep_test(path, out):
    groups = collections.OrderedDict()
    for fen, mv, q in records(path):
        groups.setdefault(fen, []).append((mv, q))
    fens = sorted(groups.keys())
    rng = np.random.RandomState(0)
    rng.shuffle(fens)
    mx = max(len(v) for v in groups.values())
    n = len(fens)
    boards = np.zeros((n, 64), dtype=np.uint8); meta = np.zeros((n, 2), dtype=np.uint8)
    moves = np.full((n, mx), -1, dtype=np.int16); qs = np.zeros((n, mx), dtype=np.float32)
    for i, fen in enumerate(fens):
        board, castle, ep, black = encode_fen(fen)
        boards[i] = np.frombuffer(board, dtype=np.uint8); meta[i] = (castle, ep)
        for j, (mv, q) in enumerate(groups[fen]):
            moves[i, j] = encode_move(mv, black, board); qs[i, j] = q
    np.savez(out, boards=boards, meta=meta, moves=moves, qs=qs, fens=np.array(fens))
    print('cb test', n, 'max moves', mx)


if __name__ == '__main__':
    if sys.argv[1] == 'train':
        prep_train(sys.argv[2], sys.argv[3], int(sys.argv[4]))
    else:
        prep_test(sys.argv[2], sys.argv[3])
