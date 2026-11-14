"""The C evaluation must equal the trained network (king-bucketed format) on real positions."""
import sys, subprocess, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, '.'); sys.path.insert(0, 'calc')
from chessenc import encode_fen
from mkdata import to_idx
sd = torch.load(sys.argv[1]); weights = sys.argv[2]
NKB = (sd['E'].shape[0] - 1) // 768; NBK = sd['wo'].shape[0]
i = np.arange(768); pc, sq = i // 64, i % 64
LUT = torch.from_numpy(np.concatenate([((pc + 6) % 12) * 64 + (sq ^ 56), [768]]).astype(np.int64))
PT_W = torch.cat([torch.tensor([0, 1, 1, 2, 4, 0] * 2, dtype=torch.float32).repeat_interleave(64), torch.zeros(1)])
KB11 = torch.tensor([0, 0, 1, 2, 3, 4, 5, 5, 6, 6, 6, 7, 7, 8, 8, 8] + [9] * 16 + [10] * 32)
def kfeat(idx):
    if NKB == 1: return idx
    if NKB == 8:                                       # phase specialists: bucket by piece count (same for both views)
        return torch.where(idx == 768, torch.full_like(idx, NKB * 768), PB + idx)
    ks = ((idx * ((idx >= 320) & (idx < 384))).sum(1) - 320).clamp(0, 63)
    return torch.where(idx == 768, torch.full_like(idx, NKB * 768), KB11[ks][:, None] * 768 + idx)
T = np.load('data/cb_test.npz'); fens = [str(f) for f in T['fens'][1:1200:2]]
idx = torch.from_numpy(to_idx(np.stack([np.frombuffer(encode_fen(f)[0], np.uint8) for f in fens])).astype(np.int64))
PB = ((((idx < 768).sum(1) - 2) // 4).clamp(0, 7) * 768)[:, None]
with torch.no_grad():
    ao = sd['b'] + F.embedding_bag(kfeat(idx), sd['E'], mode='sum', padding_idx=NKB * 768)
    ap = sd['b'] + F.embedding_bag(kfeat(LUT[idx]), sd['E'], mode='sum', padding_idx=NKB * 768)
    lin = sd['c'] + F.embedding_bag(idx, sd['S'], mode='sum', padding_idx=768)
    ph = (PT_W[idx].sum(1) / 24).clamp(max=1.0); bk = (((idx < 768).sum(1) - 2) // 4).clamp(0, NBK - 1)
    xo = ao.clamp(0, 1) ** 2; xp = ap.clamp(0, 1) ** 2
    if 'gw' in sd:                                   # learned router: run the highest-scoring expert of the piece-count group
        M = sd['gw'].shape[0] // 8; G = sd['gw'].shape[1] // 2
        bk = (((idx < 768).sum(1) - 2) // 4).clamp(0, 7)
        e = bk[:, None] * M + torch.arange(M)
        logits = (torch.cat([xo[:, :G], xp[:, :G]], 1)[:, None, :] * sd['gw'][e]).sum(2) + sd['gb'][e]
        bk = bk * M + logits.argmax(1)
    py = ((ph * lin[:, 0] + (1 - ph) * lin[:, 1] + sd['cb'][bk] + (xo * sd['wo'][bk]).sum(1) + (xp * sd['wp'][bk]).sum(1)) * 271.58).numpy()
cmds = []
for f in fens: cmds += [f'position fen {f}', 'eval']
out = subprocess.run([sys.argv[3] if len(sys.argv) > 3 else 'calc/calc', weights], input='\n'.join(cmds) + '\nquit\n', capture_output=True, text=True).stdout
c = np.array([int(l.split()[1]) for l in out.split('\n') if l.startswith('eval')])
d = float(np.abs(c - np.clip(py, -20000, 20000)).max())
print('positions', len(c), 'max |C - network| in centipawns', d)
sys.exit(0 if len(c) == len(fens) and d < 2.0 else 1)
