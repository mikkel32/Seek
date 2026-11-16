"""Stockfish training records (bullet format, 32 bytes each) on standard input -> our shards data/ev_<prefix><k>_c.npy / _y.npy
of 15M positions. Label = 0.75 x winning chance from the stored score (scale 280, the best fit to the results) + 0.25 x game result.
THESE LABELS ARE STOCKFISH'S (and Leela's games): a network trained on them is not self-taught.
  zstd -dc file.bullet.bin.zst | python3 ext/bullet2shards.py <prefix> [first shard number=0] [max shards=10000] [score scale]
With a score scale (how many stored units make one logit of win chance; 600 puts the Leela-derived second-stage files on the
scale of the Lichess labels, measured by running our own evaluator over them: ext/s2cal.py), _y.npy holds the score alone and
_ym.npy the 0.75 / 0.25 mix with the game result, so the two can be compared (--ytag ym)."""
import sys, numpy as np
LUTN = np.array([0, 1, 2, 3, 4, 5, 0, 0, 6, 7, 8, 9, 10, 11, 0, 0], np.uint8)      # their piece code (colour x 8 + kind) -> ours (kind, + 6 for the opponent)
P = sys.argv[1]; k = int(sys.argv[2]) if len(sys.argv) > 2 else 0; MAXS = int(sys.argv[3]) if len(sys.argv) > 3 else 10000
KS = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
K, LAM, N = KS or 280.0, 0.75, 15_000_000; tot = 0; buf = sys.stdin.buffer; k0 = k
while k - k0 < MAXS:
    raw = buf.read(N * 32)
    if len(raw) < 32:
        break
    n = len(raw) // 32; r = np.frombuffer(raw, np.uint8, n * 32).reshape(n, 32)
    score = r[:, 24:26].copy().view('<i2').reshape(-1).astype(np.float32); res = r[:, 26].astype(np.float32) / 2
    ps = 1 / (1 + np.exp(-score / K)); y = LAM * ps + (1 - LAM) * res; p = np.random.RandomState(k).permutation(n)
    if KS:
        np.save(f'data/ev_{P}{k}_ym.npy', np.round(y[p] * 255).astype(np.uint8)); y = ps
    c = np.empty((n, 24), np.uint8); c[:, :8] = r[:, :8]; pc = r[:, 8:24]; c[:, 8:] = LUTN[pc & 15] | (LUTN[pc >> 4] << 4)
    np.save(f'data/ev_{P}{k}_c.npy', c[p]); np.save(f'data/ev_{P}{k}_y.npy', np.round(y[p] * 255).astype(np.uint8)); k += 1; tot += n
    print(f'shard {P}{k - 1}: {n/1e6:.1f}M positions, {tot/1e9:.2f}B so far', file=sys.stderr, flush=True)
print(tot)
