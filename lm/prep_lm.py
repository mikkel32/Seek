"""Training data for the large transformer: one parquet file of the Lichess evaluation database -> a shard of arrays.

The converter is prep_lichess.py's (per unique position up to 5 candidate moves, the first move of each stored line, with
the mover's win probability; the reply and the follow-up along the best line), run over all row groups of the file with
every processor, and written as separate arrays so that training can map them without loading:
  <out>_b.npy boards (n,64) uint8   <out>_m.npy castle, ep (n,2) uint8   <out>_v.npy moves (n,5) int16
  <out>_w.npy win probabilities (n,5) float16   <out>_p.npy reply and follow-up (n,2) int16
Labels are Stockfish's (the outside-data line).

  python3 lm/prep_lm.py <file.parquet> <out prefix> [processes]
"""
import os, sys
import numpy as np, pyarrow.parquet as pq
from multiprocessing import Pool
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
PATH, OUT = sys.argv[1], sys.argv[2]; NP = int(sys.argv[3]) if len(sys.argv) > 3 else 22
sys.argv = [sys.argv[0], PATH, OUT, '0', '0']                      # prep_lichess reads its file from the command line
import prep_lichess as PL

if __name__ == '__main__':
    ng = pq.ParquetFile(PATH).num_row_groups
    with Pool(NP) as p:
        res = p.map(PL.work, range(ng), chunksize=1)
    res = [r for r in res if len(r[0])]
    rs = np.random.RandomState(abs(hash(os.path.basename(OUT))) % (2 ** 31)); n = sum(len(r[0]) for r in res); perm = rs.permutation(n)
    for k, suf in ((0, 'b'), (1, 'm'), (2, 'v'), (3, 'w'), (4, 'p')):
        np.save(f'{OUT}_{suf}.npy', np.concatenate([r[k] for r in res])[perm])
    mv = np.concatenate([r[2] for r in res])
    print(f'{os.path.basename(PATH)}: {n} positions, {float((mv >= 0).sum(1).mean()):.2f} labelled moves each, {100 * float(((mv >= 0).sum(1) > 1).mean()):.0f}% with more than one', flush=True)
