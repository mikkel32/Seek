"""Self-play records -> training arrays.  python3 zero/data.py <generation> <lambda>
Played positions: target = lambda * sigmoid(search score) + (1 - lambda) * game result, from the mover's side.
Tree positions (resolved exactly inside a search, never played): target = sigmoid(search score).
Writes ev_z<g> (played only), ev_z<g>t (played + tree) and ev_z<g>v (validation: 4% of the played positions)."""
import sys, glob, numpy as np
g = int(sys.argv[1]); lam = float(sys.argv[2])
rec = np.dtype([('idx', '<i2', 32), ('score', '<i2'), ('result', 'i1'), ('ply', 'i1')])
parts = [np.fromfile(f, dtype=rec) for f in sorted(glob.glob(f'zero/g{g}_p*.bin') + glob.glob(f'zero/g{g}_c*_p*.bin'))]
# Validation must not share games with training: positions from one game (and its search trees) are near-duplicates.
# Each file is written game after game, so its last 4% is a block of whole games: that block is the validation set.
val = np.concatenate([p[int(len(p) * 0.96):] for p in parts]); r = np.concatenate([p[:int(len(p) * 0.96)] for p in parts])
r = r[np.random.RandomState(g).permutation(len(r))]
def target(q):
    sc = 1 / (1 + np.exp(-q['score'].astype(np.float32) / 271.58))
    return np.where(q['ply'] < 0, sc, lam * sc + (1 - lam) * (q['result'].astype(np.float32) + 1) / 2)
tree = r['ply'] < 0; y = target(r)
p = r[~tree]; yp = y[~tree]
vp = val[val['ply'] >= 0]
np.save(f'data/ev_z{g}v_x.npy', vp['idx']); np.save(f'data/ev_z{g}v_y.npy', target(vp).astype(np.float32))
np.save(f'data/ev_z{g}_x.npy', p['idx']); np.save(f'data/ev_z{g}_y.npy', (yp * 255).round().astype(np.uint8))
if tree.any():
    xa = np.concatenate([p['idx'], r['idx'][tree]]); ya = np.concatenate([yp, y[tree]])
    q = np.random.RandomState(g + 99).permutation(len(xa))
    np.save(f'data/ev_z{g}t_x.npy', xa[q]); np.save(f'data/ev_z{g}t_y.npy', (ya[q] * 255).round().astype(np.uint8))
    first4 = tree & (r['result'] == 0)
    if (tree & (r['result'] == 1)).any():               # this generation harvested more than four per move: also write the four-per-move subset
        xh = np.concatenate([p['idx'], r['idx'][first4]]); yh = np.concatenate([yp, y[first4]])
        q = np.random.RandomState(g + 199).permutation(len(xh))
        np.save(f'data/ev_z{g}h_x.npy', xh[q]); np.save(f'data/ev_z{g}h_y.npy', (yh[q] * 255).round().astype(np.uint8))
print(f'generation {g}: played positions {len(p)}  tree positions {int(tree.sum())}  results W/D/L from the mover {np.mean(p["result"] == 1):.3f}/{np.mean(p["result"] == 0):.3f}/{np.mean(p["result"] == -1):.3f}  '
      f'mean |score| {np.abs(p["score"]).mean():.0f}  corr(score, result) {np.corrcoef(p["score"], p["result"])[0, 1]:.3f}')
