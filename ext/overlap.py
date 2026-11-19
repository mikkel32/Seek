"""How many positions do two sets of shards share? Both are thinned to the same 1-in-1024 of all possible positions by a hash."""
import sys, numpy as np
def sample(tags):
    out = []
    for t in tags:
        c = np.load(f'data/ev_{t}_c.npy', mmap_mode='r'); v = np.ascontiguousarray(c).view('<u8').reshape(-1, 3)
        h = (v[:, 0] * np.uint64(0x9E3779B97F4A7C15)) ^ (v[:, 1] * np.uint64(0xC2B2AE3D27D4EB4F)) ^ (v[:, 2] * np.uint64(0x165667B19E3779F9)); h ^= h >> np.uint64(29)
        out.append(h[(h & np.uint64(1023)) == 0])
    return np.concatenate(out)
def rng(s):
    pre, r = s.split(':'); a, b = r.split('-'); return [f'{pre}{k}' for k in range(int(a), int(b) + 1)]
A = sample(rng(sys.argv[1])); B = sample(rng(sys.argv[2])); ua = np.unique(A); ub = np.unique(B)
print(f'{sys.argv[1]}: {len(A)} sampled, {len(ua)} distinct ({100 * (1 - len(ua) / len(A)):.1f}% repeats inside); {sys.argv[2]}: {len(B)} sampled, {len(ub)} distinct ({100 * (1 - len(ub) / len(B)):.1f}% repeats inside); shared {len(np.intersect1d(ua, ub))} = {100 * len(np.intersect1d(ua, ub)) / len(ub):.1f}% of the second')
