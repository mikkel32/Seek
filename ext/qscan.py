"""Why is the integer network coarse? The engine's worst-case bound per hidden unit and bucket, and where it comes from."""
import sys, numpy as np, torch
w = {k: v.numpy() for k, v in torch.load(sys.argv[1], map_location='cpu').items()}
E = w['E'][:-1].copy(); NKB = E.shape[0] // 768; H = E.shape[1]
if 'Esh' in w:
    E += np.tile(w['Esh'][:768], (NKB, 1))
b = w['b']; E = np.abs(E.reshape(NKB, 12, 64, H))
m = E.max(1)                                   # (bucket, square, unit): largest piece weight on the square
top = np.sort(m, axis=1)[:, -32:, :].sum(1)   # (bucket, unit): 32 largest squares
worst = np.abs(b)[None] + top                 # (bucket, unit)
q = 10
while q > 4 and worst.max() * (1 << q) > 32000: q -= 1
print(f'{sys.argv[1]}: buckets {NKB}, width {H}; worst-case sum {worst.max():.1f} -> q_bits {q} (steps of 1/{1 << q})')
print('  worst case by bucket:', ' '.join(f'{x:.0f}' for x in worst.max(1)))
print('  |weight| percentiles 50 / 99 / 99.9 / 99.99 / max:', ' '.join(f'{np.percentile(E, p):.3f}' for p in (50, 99, 99.9, 99.99)), f'{E.max():.2f}')
for qq in (6, 7, 8, 9):
    print(f'  units whose worst case would not fit at q_bits {qq}: {(worst.max(0) * (1 << qq) > 32000).sum()} of {H}')
# which piece rows carry the largest weights
pm = E.max((2, 3))                             # (bucket, piece)
print('  largest |weight| by piece row (own P N B R Q K, their P N B R Q K), over all buckets:', ' '.join(f'{x:.1f}' for x in pm.max(0)))
