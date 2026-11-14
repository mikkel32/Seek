"""Raw self-play files (game-ordered) -> training arrays with foresight targets and alternative value targets.

Self-play writes, per game, the positions its searches settled inside the tree (pad = -depth) as it goes, then the
game's played positions in ply order (pad = ply, result = outcome for the side to move). Because the played
positions of one game sit together and in order, each one can be told what happened later in its own game:

  value targets (one byte each, 0..255 = win probability of the side to move)
    y    0.5 * search score + 0.5 * game result                      (what production trains on)
    ys   0.5 * my NEXT turn's search score + 0.5 * game result      (a two-ply deeper look at the same line, for free)
    yl   lambda-return, lambda 0.95 per ply: later search scores, the game result at the end        (TD(lambda))
  foresight targets (aux, 28 bytes; only for played positions, column 0 says whether they are valid)
    1-3    my search score 2 / 4 / 8 plies later (the result once the game is over)
    4      plies until the game ends (capped at 100)
    5-14   change in the count of each piece type (mine P N B R Q, theirs P N B R Q) 8 plies later, +16
    15-24  the same at the end of the game, +16
    25,26  from and to square of the move played here (255 = unknown)
    27     how far the search score moves over the next two plies

Positions from inside the trees have no future in the game: their three value targets are all the search score and
their foresight targets are switched off.

  python3 zero/foresight.py <out tag> <glob of raw files> [processes=4]
"""
import glob, os, sys
import numpy as np
from multiprocessing import Pool

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
REC = np.dtype([('idx', '<i2', 32), ('score', '<i2'), ('result', 'i1'), ('ply', 'i1')])
K = 28
_i = np.arange(768); LUT = np.concatenate([((_i // 64 + 6) % 12) * 64 + ((_i % 64) ^ 56), [768]]).astype(np.int64)
L_T = 0.95


def counts(idx):
    """(m,32) feature lists -> (m,10) piece counts: mine P N B R Q, theirs P N B R Q"""
    code = np.where(idx < 768, idx // 64, 12)
    c = (code[:, :, None] == np.arange(12)[None, None, :]).sum(1)
    return c[:, [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]]


def one_game(r):
    """r: the played records of one game, in ply order -> (y, ys, yl, aux)"""
    m = len(r)
    wp = 1 / (1 + np.exp(-r['score'].astype(np.float64) / 271.58)); z = (r['result'].astype(np.float64) + 1) / 2
    ply = r['ply'].astype(np.int64); known = ply < 127
    # plies past 127 are stored as 127: keep the order, count one ply per record from there on
    eff = ply.copy()
    over = np.nonzero(~known)[0]
    if len(over):
        eff[over] = 127 + np.arange(len(over))
    par = eff & 1                                              # parity is only trustworthy where known
    end = eff[-1] + 1                                          # the game ended (at the earliest) one ply after its last recorded position
    y = 0.5 * wp + 0.5 * z
    ys = y.copy(); yl = y.copy()
    aux = np.zeros((m, K), dtype=np.uint8); aux[:, 25:27] = 255
    cnt = counts(r['idx']); swap = cnt[:, [5, 6, 7, 8, 9, 0, 1, 2, 3, 4]]
    occ = np.zeros((m, 769), dtype=bool); np.put_along_axis(occ, r['idx'].astype(np.int64), True, 1)
    occ_other = np.zeros((m, 769), dtype=bool); np.put_along_axis(occ_other, LUT[r['idx'].astype(np.int64)], True, 1)
    last_known = np.nonzero(known)[0]
    for i in range(m):
        if not known[i]:                                       # whose move it is cannot be told past ply 127: production target, no foresight
            continue
        d = eff[i:] - eff[i]; kn = known[i:]                   # only later positions whose side to move is known take part
        v = np.where(par[i:] == par[i], wp[i:], 1 - wp[i:])    # later scores seen from this position's side to move
        w = (L_T ** d) * kn; tail = L_T ** (end - eff[i]) / (1 - L_T)
        yl[i] = ((w * v).sum() + tail * z[i]) / (w.sum() + tail)
        nxt = i + 1 + np.nonzero(known[i + 1:] & (par[i + 1:] == par[i]))[0]      # my later turns
        f = [wp[nxt[k]] if len(nxt) > k else z[i] for k in (0, 1, 3)]
        ys[i] = 0.5 * f[0] + 0.5 * z[i]                        # my next turn's search has seen two plies further down the line that was played
        aux[i, 0] = 1
        aux[i, 1:4] = np.round(255 * np.array(f))
        aux[i, 4] = min(100, end - eff[i])
        j8 = nxt[3] if len(nxt) > 3 else (nxt[-1] if len(nxt) else i)
        aux[i, 5:15] = np.clip(cnt[j8] - cnt[i], -16, 16) + 16
        je = last_known[-1]
        ce = cnt[je] if par[je] == par[i] else swap[je]
        aux[i, 15:25] = np.clip(ce - cnt[i], -16, 16) + 16
        aux[i, 27] = np.round(255 * abs(f[0] - wp[i]))
        if i + 1 < m and known[i + 1] and ply[i + 1] == ply[i] + 1:               # the move played: compare with the next position
            a = occ[i, :384]; b = occ_other[i + 1, :384]       # my pieces now / my pieces one ply later, both in my view
            fr = np.nonzero(a & ~b)[0]; to = np.nonzero(b & ~a)[0]
            if len(fr) == 2 and len(to) == 2:                  # castling: the king's move stands for it
                fr = fr[fr // 64 == 5]; to = to[to // 64 == 5]
            if len(fr) == 1 and len(to) == 1:
                aux[i, 25] = fr[0] % 64; aux[i, 26] = to[0] % 64
    b8 = lambda t: np.round(255 * np.clip(t, 0, 1)).astype(np.uint8)
    return b8(y), b8(ys), b8(yl), aux


def one_file(path):
    r = np.fromfile(path, dtype=REC)
    n = len(r)
    wp = 1 / (1 + np.exp(-r['score'].astype(np.float64) / 271.58))
    y = np.round(255 * wp).astype(np.uint8); ys = y.copy(); yl = y.copy()
    aux = np.zeros((n, K), dtype=np.uint8); aux[:, 25:27] = 255
    played = r['ply'] >= 0
    # one game's played positions form a run; a ply that does not increase inside a run starts the next game
    idx = np.nonzero(played)[0]
    if len(idx):
        brk = np.nonzero((np.diff(idx) > 1) | ((np.diff(r['ply'][idx].astype(np.int64)) <= 0) & (r['ply'][idx][1:] < 127)))[0] + 1
        for g in np.split(idx, brk):
            y[g], ys[g], yl[g], aux[g] = one_game(r[g])
    return r['idx'], y, ys, yl, aux, played, len(brk) + 1 if len(idx) else 0, np.round(255 * wp).astype(np.uint8)


if __name__ == '__main__':
    tag = sys.argv[1]; files = sorted(glob.glob(sys.argv[2])); procs = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    val_files = files[::50]; vs = set(val_files); tr_files = [f for f in files if f not in vs]
    with Pool(procs) as pool:
        out = pool.map(one_file, tr_files, chunksize=4); vout = pool.map(one_file, val_files, chunksize=4)
    cat = lambda k, o=out: np.concatenate([t[k] for t in o])
    x, y, ys, yl, aux, played = cat(0), cat(1), cat(2), cat(3), cat(4), cat(5)
    games = sum(t[6] for t in out)
    perm = np.random.RandomState(0).permutation(len(x))
    np.save(f'data/ev_{tag}_x.npy', x[perm]); np.save(f'data/ev_{tag}_y.npy', y[perm]); np.save(f'data/ev_{tag}_ys.npy', ys[perm])
    np.save(f'data/ev_{tag}_yl.npy', yl[perm]); np.save(f'data/ev_{tag}_aux.npy', aux[perm])
    np.save(f'data/ev_{tag}_s.npy', cat(7)[perm]); np.save(f'data/ev_{tag}_pl.npy', played[perm].astype(np.uint8))      # the search score alone, and which rows were played
    pi = perm[played[perm]][:1_500_000]                        # played positions for the exact table solve
    np.save(f'data/ev_{tag}p_x.npy', x[pi]); np.save(f'data/ev_{tag}p_y.npy', y[pi])
    vx, vy, vp = cat(0, vout), cat(1, vout), cat(5, vout)      # held-out files, played positions, the production target
    np.save(f'data/ev_{tag}val_x.npy', vx[vp]); np.save(f'data/ev_{tag}val_y.npy', vy[vp].astype(np.float32) / 255)
    a = aux[played]
    print(f'{tag}: {len(files)} files, {games} games, {len(x)/1e6:.2f}M positions ({played.sum()/1e6:.2f}M played), validation {vp.sum()} played positions')
    print(f'  foresight valid {a[:, 0].mean():.3f} of played; move known {(a[:, 25] < 64).mean():.3f}; mean plies to end {a[a[:, 0] > 0, 4].mean():.1f}; '
          f'mean |score change over 2 plies| {a[a[:, 0] > 0, 27].mean() / 255:.4f}')
    d = lambda u, v: float(np.abs(u[played].astype(np.float64) - v[played]).mean() / 255)
    print(f'  mean |target difference| on played positions: next-turn score vs production {d(ys, y):.4f}, lambda-return vs production {d(yl, y):.4f}')
