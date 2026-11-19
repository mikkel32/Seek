"""Label training positions with the transformer instead of a search.

The transformer judges a position about as well as the calculator's 2,500-position search, and on the graphics card it
judges tens of thousands of positions a second, several times what the self-play workers label by searching. If an
evaluator trained on its labels plays as well as one trained on search labels, the card can multiply the amount of
labelled data, which is what the evaluator is short of. This writes the labels for an existing set of positions:

  python3 zero/ouro_label.py <tag> <transformer checkpoint>     data/ev_<tag>_x.npy, _s.npy (search scores) -> data/ev_<tag>_yT.npy

The transformer's value is brought to the search-score scale with one slope and offset (least squares on the scores).
"""
import os, sys, time
import numpy as np, torch
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')); sys.path.insert(0, '.')
from model import Ouro

tag, ckpt = sys.argv[1], sys.argv[2]; DEV = torch.device('cuda'); B = 8192
torch.backends.cuda.matmul.allow_tf32 = True
ck = torch.load(ckpt, map_location='cpu', weights_only=False); model = Ouro(**ck['cfg']).to(DEV); model.load_state_dict(ck['ema']); model.eval()
x = np.load(f'data/ev_{tag}_x.npy', mmap_mode='r'); s8 = np.load(f'data/ev_{tag}_s.npy', mmap_mode='r'); n = len(x); o = np.zeros(n, np.float32); t0 = time.time()
with torch.no_grad():
    for j in range(0, n, 262144):
        idx = torch.from_numpy(np.ascontiguousarray(x[j:j + 262144]).astype(np.int64)).to(DEV); out = []
        for k in range(0, len(idx), B):
            q = idx[k:k + B]; board = torch.zeros(len(q), 65, dtype=torch.long, device=DEV)
            board.scatter_(1, torch.where(q < 768, q % 64, torch.full_like(q, 64)), torch.where(q < 768, q // 64 + 1, torch.zeros_like(q)))
            with torch.autocast('cuda', dtype=torch.bfloat16):
                out.append(model(board[:, :64], torch.zeros(len(q), dtype=torch.long, device=DEV), torch.full((len(q),), 64, dtype=torch.long, device=DEV), loops=1)[1].float())
        o[j:j + 262144] = torch.cat(out).cpu().numpy()
dt = time.time() - t0
sub = np.random.RandomState(0).choice(n, min(n, 4_000_000), replace=False); sub.sort()
s = np.asarray(s8[sub]).astype(np.float64) / 255; ok = (s > 0.02) & (s < 0.98); sl = np.log(s[ok] / (1 - s[ok])); oo = o[sub][ok].astype(np.float64)
a, b = np.polyfit(oo, sl, 1)
yT = 1 / (1 + np.exp(-(a * o.astype(np.float64) + b)))
np.save(f'data/ev_{tag}_yT.npy', np.round(255 * yT).astype(np.uint8))
print(f'{n/1e6:.1f}M positions labelled by the transformer in {dt:.0f}s = {n/dt:,.0f} positions per second; scale {a:.3f}, offset {b:+.3f}; '
      f'correlation with the search score (logits) {np.corrcoef(oo, sl)[0, 1]:.3f}; mean |label - search score| {np.abs(yT[sub] - np.asarray(s8[sub]) / 255).mean():.4f}')
