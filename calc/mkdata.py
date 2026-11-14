"""Evaluation training data for the calculator: (position, win probability for the side to move).

Sources: Lichess root positions (deep Stockfish value) and the positions after each evaluated candidate
move (value = 1 - that move's win probability); ChessBench positions after a random legal move.
Positions are stored as up to 32 feature indices (piece*64 + square, mover's view), padded with 768.

  python3 calc/mkdata.py b c d e cb
"""
import sys, numpy as np

SWAP = np.array([0, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6], np.uint8)
K_PAD = 768


def apply_moves(b, m):
    """b (n,64) canonical boards, m (n,) move indices -> boards after the move, seen by the new side to move"""
    b = b.copy(); m = m.astype(np.int64); r = np.arange(len(b))
    under = m >= 4096; k = m - 4096
    frm = np.where(under, 48 + (k // 3) // 8, m // 64); to = np.where(under, 56 + (k // 3) % 8, m % 64)
    piece = b[r, frm]
    ep = (piece == 1) & ((frm % 8) != (to % 8)) & (b[r, to] == 0)
    b[r[ep], to[ep] - 8] = 0
    ck = (piece == 6) & (frm == 4) & (to == 6); b[r[ck], 7] = 0; b[r[ck], 5] = 4
    cq = (piece == 6) & (frm == 4) & (to == 2); b[r[cq], 0] = 0; b[r[cq], 3] = 4
    newp = piece.copy()
    newp[(piece == 1) & (to >= 56) & ~under] = 5
    newp[under] = (2 + k[under] % 3).astype(np.uint8)
    b[r, to] = newp; b[r, frm] = 0
    return SWAP[b.reshape(-1, 8, 8)[:, ::-1, :].reshape(-1, 64)]


def to_idx(b):
    """boards (n,64) -> (n,32) int16 feature indices, padded with 768"""
    n = len(b)
    r, s = np.nonzero(b)
    cnt = np.bincount(r, minlength=n)
    slot = np.arange(len(r)) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    out = np.full((n, 32), K_PAD, np.int16)
    keep = slot < 32
    out[r[keep], slot[keep]] = (b[r, s][keep].astype(np.int16) - 1) * 64 + s[keep]
    return out


def build(tag, chunk=1_000_000):
    z = np.load('data/cb_train.npz' if tag == 'cb' else f'data/lichess_{tag}.npz')
    boards, moves, wps = z['boards'], z['moves'], z['wps']
    z_rq = z['rq'] if 'rq' in z.files else None
    n = len(boards)
    if tag == 'e':
        n -= 200_000                                    # the last 200k roots of the last file are the validation set
    X, Y = [], []
    for i in range(0, n, chunk):
        b = boards[i:i + chunk][:n - i] if i + chunk > n else boards[i:i + chunk]
        mv = moves[i:i + len(b)]; w = wps[i:i + len(b)].astype(np.float32)
        rq = z_rq[i:i + len(b)] if z_rq is not None else None
        if tag != 'cb':
            m0 = mv[:, 0].astype(np.int64); rr = np.arange(len(b))
            to0 = np.where(m0 >= 4096, 56, m0 % 64); fr0 = np.where(m0 >= 4096, 48, m0 // 64)
            piece = b[rr, fr0]
            noisy = (b[rr, to0] >= 7) | (m0 >= 4096) | ((piece == 1) & (to0 >= 56)) | ((piece == 1) & ((fr0 % 8) != (to0 % 8)))
            X.append(to_idx(b[~noisy])); Y.append(w[~noisy, 0])          # quiet roots only: best move is not a capture / promotion
        for j in range(mv.shape[1] if tag != 'cb' else 1):
            ok = (mv[:, j] >= 0) & ((rq[:, j] == 0) if rq is not None else True)   # children whose known reply is quiet
            if ok.any():
                X.append(to_idx(apply_moves(b[ok], mv[ok, j]))); Y.append(1.0 - w[ok, j])
    X = np.concatenate(X); Y = np.concatenate(Y)
    p = np.random.RandomState(hash(tag) % 1000 + 7).permutation(len(X))
    np.save(f'data/ev_{tag}_x.npy', X[p]); np.save(f'data/ev_{tag}_y.npy', (Y[p] * 255).round().astype(np.uint8))
    print(tag, 'positions', len(X), 'mean target', float(Y.mean()), flush=True)
    if tag == 'e':
        b = boards[n:]; w = wps[n:].astype(np.float32)
        m0 = moves[n:, 0].astype(np.int64); rr = np.arange(len(b)); to0 = np.where(m0 >= 4096, 56, m0 % 64); fr0 = np.where(m0 >= 4096, 48, m0 // 64)
        noisy = (b[rr, to0] >= 7) | (m0 >= 4096) | ((b[rr, fr0] == 1) & ((to0 >= 56) | ((fr0 % 8) != (to0 % 8))))
        b = b[~noisy]; w = w[~noisy]
        np.save('data/ev_val_x.npy', to_idx(b)); np.save('data/ev_val_y.npy', w[:, 0])
        print('validation roots', len(b), flush=True)


if __name__ == '__main__':
    for t in sys.argv[1:]:
        build(t)
