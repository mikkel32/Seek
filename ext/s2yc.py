"""Second-stage shards: a label file _yc.npy on the Lichess labels' scale = 0.75 x win chance from the score at scale 950
(instead of the 600 the shards were written with) + 0.25 x game result. Derived from the two label files already there
(_y.npy score only at 600, _ym.npy its 0.75 / 0.25 mix with the result). Skips shards already done.
  python3 ext/s2yc.py <first shard> <last shard>"""
import os, sys, numpy as np
t = np.clip(np.arange(256) / 255, 0.5 / 255, 254.5 / 255); P950 = 1 / (1 + np.exp(-(600 / 950) * np.log(t / (1 - t)))); n = 0
for k in range(int(sys.argv[1]), int(sys.argv[2]) + 1):
    b = f'data/ev_s{k}'
    if os.path.exists(b + '_yc.npy') or not os.path.exists(b + '_ym.npy'):
        continue
    y6 = np.load(b + '_y.npy'); ym = np.load(b + '_ym.npy'); r = (ym / 255.0 - 0.75 * y6 / 255.0) * 8; res = np.clip(np.rint(r), 0, 2)
    assert float(np.abs(r - res).max()) < 0.2, k
    np.save(b + '_yc.tmp.npy', np.rint(255 * (0.75 * P950[y6] + 0.25 * res / 2)).astype(np.uint8)); os.replace(b + '_yc.tmp.npy', b + '_yc.npy'); n += 1
    if n == 1:
        yc = np.load(b + '_yc.npy') / 255; print(f'shard s{k}: results loss/draw/win {np.bincount(res.astype(int), minlength=3) / len(res)}; labels within 30-70%: {(np.abs(yc - .5) < .2).mean():.3f} (score-only at 600: {(np.abs(y6 / 255 - .5) < .2).mean():.3f}, mixed at 600: {(np.abs(ym / 255 - .5) < .2).mean():.3f})', flush=True)
print(f'{n} shards')
