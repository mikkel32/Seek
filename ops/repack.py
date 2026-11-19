import sys, os, numpy as np
for i in sys.argv[1:]:
    f = f'/root/loopchess/data/lm_{i}_b.npy'; x = np.load(f, mmap_mode='r')
    if x.shape[1] == 32: print(i, 'already packed'); continue
    out = np.lib.format.open_memmap(f + '.tmp', mode='w+', dtype=np.uint8, shape=(len(x), 32))
    for s in range(0, len(x), 2_000_000):
        c = np.asarray(x[s:s + 2_000_000]); out[s:s + 2_000_000] = c[:, 0::2] | (c[:, 1::2] << 4)
    out.flush(); del out, x; os.replace(f + '.tmp', f); print(i, 'packed')
