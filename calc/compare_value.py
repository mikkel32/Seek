"""Same positions, three evaluators: who predicts Stockfish's verdict best without any search?

  solved table   1,538 weights from the direct solve
  network        the calculator's trained evaluation (averaged weights)
  OURO           the 2M-parameter transformer's value head (2 loops)

Positions: the quiet validation roots (last 200k positions of the last Lichess file; never trained on by the
calculator's evaluation; OURO never trained on this file at all).
"""
import sys, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, '.'); sys.path.insert(0, 'calc')
from mkdata import to_idx
from engine import Net

torch.set_num_threads(4)
z = np.load('data/lichess_e.npz')
n = len(z['boards']) - 200_000
b = z['boards'][n:]; meta = z['meta'][n:]; w = z['wps'][n:].astype(np.float32); m0 = z['moves'][n:, 0].astype(np.int64)
rr = np.arange(len(b)); to0 = np.where(m0 >= 4096, 56, m0 % 64); fr0 = np.where(m0 >= 4096, 48, m0 // 64)
noisy = (b[rr, to0] >= 7) | (m0 >= 4096) | ((b[rr, fr0] == 1) & ((to0 >= 56) | ((fr0 % 8) != (to0 % 8))))
b = b[~noisy]; meta = meta[~noisy]; y = torch.from_numpy(w[~noisy, 0])
N = int(sys.argv[1]) if len(sys.argv) > 1 else 60000
b = b[:N]; meta = meta[:N]; y = y[:N]
idx = torch.from_numpy(to_idx(b).astype(np.int64))
i = np.arange(768); pc, sq = i // 64, i % 64
LUT = torch.from_numpy(np.concatenate([((pc + 6) % 12) * 64 + (sq ^ 56), [768]]).astype(np.int64))
PT_W = torch.cat([torch.tensor([0, 1, 1, 2, 4, 0] * 2, dtype=torch.float32).repeat_interleave(64), torch.zeros(1)])
ph = (PT_W[idx].sum(1) / 24).clamp(max=1.0)


def report(name, out):
    print(f'{name:<14} log-loss {float(F.binary_cross_entropy_with_logits(out, y)):.4f}   mean error in win probability {float((torch.sigmoid(out) - y).abs().mean()):.4f}', flush=True)


with torch.no_grad():
    p = np.load('calc/pst.npz')
    S = torch.zeros(769, 2); S[:768, 0] = torch.from_numpy(p['S_mg']); S[:768, 1] = torch.from_numpy(p['S_eg'])
    lin = torch.tensor([float(p['c_mg']), float(p['c_eg'])]) + F.embedding_bag(idx, S, mode='sum', padding_idx=768)
    report('solved table', ph * lin[:, 0] + (1 - ph) * lin[:, 1])
    sd = torch.load(sys.argv[2] if len(sys.argv) > 2 else 'calc/net256b.bin.ema.pt')
    ao = sd['b'] + F.embedding_bag(idx, sd['E'], mode='sum', padding_idx=768); ap = sd['b'] + F.embedding_bag(LUT[idx], sd['E'], mode='sum', padding_idx=768)
    lin = sd['c'] + F.embedding_bag(idx, sd['S'], mode='sum', padding_idx=768)
    bk = (((idx < 768).sum(1) - 2) // 4).clamp(0, sd['wo'].shape[0] - 1)
    report('network', ph * lin[:, 0] + (1 - ph) * lin[:, 1] + sd['cb'][bk] + ((ao.clamp(0, 1) ** 2) * sd['wo'][bk]).sum(1) + ((ap.clamp(0, 1) ** 2) * sd['wp'][bk]).sum(1))
    net = Net('ckpt/B.pt', threads=4)
    vs = []
    for j in range(0, N, 500):
        board = torch.from_numpy(b[j:j + 500].astype(np.int64))
        pol, v, ql, _ = net.model(board, torch.from_numpy(meta[j:j + 500, 0].astype(np.int64)), torch.from_numpy(meta[j:j + 500, 1].astype(np.int64)), loops=2, tbptl=False)
        vs.append(v.float())
    report('OURO', torch.cat(vs))
