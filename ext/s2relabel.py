"""Second-stage shards written with score scale 600 -> scale 950, the scale of the Lichess labels (Stockfish's internal
units to its normalised centipawns; confirmed by the evaluation-scale test of the first probe). Keeps the old labels as
_y6.npy and writes the game result (0 loss, 1 draw, 2 win for the mover) as _yres.npy. Skips shards already done.
  python3 ext/s2relabel.py <first shard> <last shard>"""
import os, sys, numpy as np
t = np.clip(np.arange(256) / 255, 0.5 / 255, 254.5 / 255); LUT = np.rint(255 / (1 + np.exp(-(600 / 950) * np.log(t / (1 - t))))).astype(np.uint8)
n = 0
for k in range(int(sys.argv[1]), int(sys.argv[2]) + 1):
    b = f'data/ev_s{k}'
    if os.path.exists(b + '_y6.npy') or not os.path.exists(b + '_ym.npy'):
        continue
    y6 = np.load(b + '_y.npy'); ym = np.load(b + '_ym.npy'); r = (ym / 255.0 - 0.75 * y6 / 255.0) * 8
    res = np.clip(np.rint(r), 0, 2).astype(np.uint8); worst = float(np.abs(r - res).max())
    assert worst < 0.2, (k, worst)
    np.save(b + '_y6.npy', y6); np.save(b + '_yres.npy', res); np.save(b + '_y.tmp.npy', LUT[y6]); os.replace(b + '_y.tmp.npy', b + '_y.npy'); n += 1
    if n == 1:
        print(f'shard s{k}: results loss/draw/win {np.bincount(res, minlength=3) / len(res)}, worst rounding distance {worst:.3f}; labels within 30-70%: {(np.abs(LUT[y6] / 255 - .5) < .2).mean():.3f} (was {(np.abs(y6 / 255 - .5) < .2).mean():.3f})', flush=True)
print(f'{n} shards relabelled')
