"""The transformer's value of the fast network's training positions: a third teacher.

  OURO_GPU=1 python3 lm/label_nnue.py <checkpoint> <rows per shard> <shard tag> [more tags ...]

For the first <rows per shard> positions of data/ev_<tag>_x.npy (32 feature numbers a position, the mover's view: piece
code x 64 + square, 0-5 the mover's P N B R Q K, 6-11 the opponent's -- the transformer's own frame with codes 1-12)
the transformer's win chance for the mover is written to data/tl_<tag>.npy, one byte a position (x255). The shards carry
no castling rights or en-passant square; the transformer is told there are none.
"""
import os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from engine import Net

net = Net(sys.argv[1]); m = net.model; dev = net.dev; N = int(sys.argv[2]); BS = 4096


def board_of(x):
    """a batch of a shard -> (B, 64) piece codes 0-12. Shards hold either 32 feature numbers a position or the compact
    form of zero/compact.py: 64 occupancy bits, then a 4-bit piece code for each occupied square in square order."""
    B = x.shape[0]; bd = torch.zeros(B, 65, dtype=torch.long, device=dev)
    if x.shape[1] == 24:
        b = x[:, :8].long(); occ = sum(b[:, k] << (8 * k) for k in range(8))
        bits = ((occ[:, None] >> torch.arange(64, device=dev)) & 1).bool()
        sq = torch.where(bits, torch.arange(64, device=dev), torch.tensor(64, device=dev)).sort(1).values[:, :32]
        code = torch.stack([x[:, 8:] & 15, x[:, 8:] >> 4], 2).reshape(B, 32).long()
        bd.scatter_(1, sq, torch.where(sq < 64, code + 1, torch.zeros_like(code)))
    else:
        idx = x.long(); sq = torch.where(idx < 768, idx % 64, torch.full_like(idx, 64)); bd.scatter_(1, sq, torch.where(idx < 768, idx // 64 + 1, torch.zeros_like(idx)))
    return bd[:, :64]


@torch.no_grad()
def value(x):
    bd = board_of(x); B = bd.shape[0]
    ctx = m.prepare(bd, torch.zeros(B, dtype=torch.long, device=dev), torch.full((B,), 64, dtype=torch.long, device=dev)); st = None
    for _ in range(2): st = m.loop(ctx, st)
    return torch.sigmoid(m.readout(ctx, st)[1].float())


if __name__ == '__main__':
    for tag in sys.argv[3:]:
        f = f'data/ev_{tag}_x.npy' if os.path.exists(f'data/ev_{tag}_x.npy') else f'data/ev_{tag}_c.npy'
        X = np.load(f, mmap_mode='r'); n = min(N, len(X)); out = np.empty(n, np.uint8); t0 = time.time()
        for i in range(0, n, BS):
            out[i:i + BS] = (value(torch.from_numpy(np.ascontiguousarray(X[i:min(i + BS, n)])).to(dev)) * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
        np.save(f'data/tl_{tag}.npy', out)
        y = next((np.load(f'data/ev_{tag}_{k}.npy', mmap_mode='r')[:n] for k in ('yc', 'y') if os.path.exists(f'data/ev_{tag}_{k}.npy')), None)
        chk = f'; against the shard\'s own labels: correlation {np.corrcoef(out[:200000], y[:200000])[0, 1]:.3f}, mean difference {np.abs(out[:200000].astype(float) - y[:200000]).mean() / 255:.3f}' if y is not None else ''
        print(f'{tag}: {n} positions in {time.time() - t0:.0f}s ({n / (time.time() - t0):.0f} a second); mean win chance {out.mean() / 255:.3f}{chk}', flush=True)
