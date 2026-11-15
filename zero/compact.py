"""Positions in 24 bytes instead of 64, and an archiver that keeps every training shard the loop would delete.

The evaluator is short of data for its size: wider networks only paid once the memory grew. The loop's memory is limited
by disk (a position is 32 two-byte feature numbers = 64 bytes), and shards that leave the window are deleted. Here a
position is stored as which squares are occupied (8 bytes) and a 4-bit piece code for each occupied square in square
order (16 bytes): 2.7 times as many positions on the same disk. The trainer decodes on the GPU (calc/train_nnue.py reads
data/ev_<tag>_c.npy when it exists).

  python3 zero/compact.py check <tag>            encode data/ev_<tag>_x.npy in memory, decode, compare (as sets of features)
  python3 zero/compact.py archive                forever: every shard data/ev_sh<N>_x.npy that is not yet archived becomes
                                                 data/ev_a<N>_c.npy (+ a hard link to its targets as data/ev_a<N>_y.npy)
"""
import glob, os, re, sys, time
import numpy as np

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


def encode(idx):
    """(n,32) int16 feature lists (piece code * 64 + square, 768 = empty slot) -> (n,24) uint8"""
    idx = np.asarray(idx); key = np.where(idx < 768, ((idx & 63).astype(np.uint16) << 4) | (idx >> 6).astype(np.uint16), np.uint16(0xFFFF))
    key.sort(axis=1)                                                    # by square; empty slots last
    ok = key != 0xFFFF; sq = (key >> 4).astype(np.uint64); code = np.where(ok, key & 15, 0).astype(np.uint8)
    occ = np.bitwise_or.reduce(np.where(ok, np.uint64(1) << np.where(ok, sq, 0), np.uint64(0)), axis=1)
    out = np.empty((len(idx), 24), np.uint8)
    out[:, :8] = occ.astype('<u8').view(np.uint8).reshape(-1, 8)
    out[:, 8:] = code[:, 0::2] | (code[:, 1::2] << 4)
    return out


def decode(c):
    """(n,24) uint8 -> (n,32) int16 feature lists (in square order, empty slots = 768); numpy, for checking"""
    occ = np.ascontiguousarray(c[:, :8]).view('<u8').reshape(-1)
    bits = ((occ[:, None] >> np.arange(64, dtype=np.uint64)) & np.uint64(1)).astype(bool)
    sq = np.sort(np.where(bits, np.arange(64, dtype=np.int16), np.int16(64)), axis=1)[:, :32]
    code = np.empty((len(c), 32), np.int16); code[:, 0::2] = c[:, 8:] & 15; code[:, 1::2] = c[:, 8:] >> 4
    return np.where(sq < 64, code * 64 + sq, 768).astype(np.int16)


def decode_torch(c):
    """the same on the GPU: (n,24) uint8 tensor -> (n,32) int16 tensor"""
    import torch
    b = c[:, :8].long(); occ = sum(b[:, k] << (8 * k) for k in range(8))                                   # 64 bits in a signed int64: bit tests still work
    bits = ((occ[:, None] >> torch.arange(64, device=c.device)) & 1).bool()
    sq = torch.where(bits, torch.arange(64, device=c.device, dtype=torch.int16), torch.tensor(64, device=c.device, dtype=torch.int16)).sort(1).values[:, :32]
    code = torch.stack([c[:, 8:] & 15, c[:, 8:] >> 4], 2).reshape(len(c), 32).to(torch.int16)
    return torch.where(sq < 64, code * 64 + sq, torch.tensor(768, device=c.device, dtype=torch.int16))


def encode_file(src, dst, block=2_000_000):
    x = np.load(src, mmap_mode='r'); out = np.lib.format.open_memmap(dst + '.tmp', mode='w+', dtype=np.uint8, shape=(len(x), 24))
    for s in range(0, len(x), block):
        out[s:s + block] = encode(x[s:s + block])
    out.flush(); del out; os.replace(dst + '.tmp', dst)
    return len(x)


if __name__ == '__main__':
    if sys.argv[1] == 'check':
        x = np.load(f'data/ev_{sys.argv[2]}_x.npy', mmap_mode='r')[:1_000_000]; c = encode(x); d = decode(c)
        same = (np.sort(np.asarray(x), axis=1) == np.sort(d, axis=1)).all()
        msg = f'{len(x)} positions: numpy round trip {"exact" if same else "DIFFERENT"}'
        try:
            import torch
            dev = 'cuda' if torch.cuda.is_available() else 'cpu'
            t = decode_torch(torch.from_numpy(c).to(dev)).cpu().numpy(); msg += f', torch ({dev}) decode {"exact" if (t == d).all() else "DIFFERENT"}'
        except ImportError:
            pass
        print(msg); sys.exit(0 if same else 1)
    if sys.argv[1] == 'archive':
        while not os.path.exists('zero/ARCHIVE_STOP'):
            for src in sorted(glob.glob('data/ev_sh*_x.npy'), key=lambda f: int(re.search(r'ev_sh(\d+)_x', f).group(1)) if re.search(r'ev_sh(\d+)_x', f) else -1):
                m = re.search(r'ev_sh(\d+)_x\.npy$', src)
                if not m:
                    continue
                n = m.group(1); dst = f'data/ev_a{n}_c.npy'; ysrc = f'data/ev_sh{n}_y.npy'
                if os.path.exists(dst) or not os.path.exists(ysrc) or time.time() - os.path.getmtime(ysrc) < 30:      # not while the loop is still writing it
                    continue
                try:
                    if not os.path.exists(f'data/ev_a{n}_y.npy'):
                        os.link(ysrc, f'data/ev_a{n}_y.npy')
                    t0 = time.time(); k = encode_file(src, dst)
                    print(f'[{time.strftime("%H:%M:%S", time.gmtime())}] shard {n}: {k/1e6:.1f}M positions archived in {time.time() - t0:.0f}s', flush=True)
                except FileNotFoundError:                                # the loop dropped it meanwhile
                    for f in (dst + '.tmp', f'data/ev_a{n}_y.npy'):
                        if os.path.exists(f) and not os.path.exists(dst):
                            os.remove(f)
            time.sleep(30)
