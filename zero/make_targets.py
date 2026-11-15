"""The same self-play games, labelled several ways: one shuffled training set with one value target per lambda.

  python3 zero/make_targets.py <tag> <folder of raw files> [processes=8]
  writes data/ev_<tag>_x.npy and data/ev_<tag>_y95.npy, _y70.npy, _y50.npy, _y00.npy
  (lambda-return with lambda 0.95 / 0.7 / 0.5, and y00 = the search score alone; positions from inside the search trees
  carry the search score in every variant, as in production)
"""
import glob, os, re, sys
import numpy as np
from multiprocessing import Pool

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
REC = np.dtype([('idx', '<i2', 32), ('score', '<i2'), ('result', 'i1'), ('ply', 'i1')]); LAM = 0.5
src = open('zero/cont.py').read(); exec(re.search(r'(def target\(q\):.*?)(?=\ndef make_shard)', src, re.S).group(1))      # the loop's own target code
VARS = (('y95', 0.95), ('y70', 0.7), ('y50', 0.5))


def one(path):
    r = np.fromfile(path, dtype=REC)
    out = [np.round(255 * np.clip(lam_targets(r, l), 0, 1)).astype(np.uint8) for _, l in VARS]
    out.append(np.round(255 / (1 + np.exp(-r['score'].astype(np.float64) / 271.58))).astype(np.uint8))
    return r['idx'], out, int((r['ply'] >= 0).sum())


if __name__ == '__main__':
    tag = sys.argv[1]; files = sorted(glob.glob(sys.argv[2] + '/*.bin')); procs = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    with Pool(procs) as pool:
        res = pool.map(one, files, chunksize=4)
    x = np.concatenate([r[0] for r in res]); perm = np.random.RandomState(0).permutation(len(x))
    np.save(f'data/ev_{tag}_x.npy', x[perm])
    names = [v[0] for v in VARS] + ['y00']; ys = {}
    for j, nme in enumerate(names):
        ys[nme] = np.concatenate([r[1][j] for r in res]); np.save(f'data/ev_{tag}_{nme}.npy', ys[nme][perm])
    played = sum(r[2] for r in res)
    d = lambda a, b: float(np.abs(ys[a].astype(np.float64) - ys[b]).mean() / 255)
    print(f'{tag}: {len(files)} files, {len(x)/1e6:.1f}M positions, {played/1e6:.1f}M of them played ({played/len(x):.3f}); '
          f'mean |difference| to the lambda-0.95 target over all positions: y70 {d("y70", "y95"):.4f}, y50 {d("y50", "y95"):.4f}, search score alone {d("y00", "y95"):.4f}')
