"""A "where to search" expert: which move will a search choose here, read off the fast evaluator's hidden layer.

In the engine's search the first move tried ends the node in 85% of cut-offs, and when the refuting move is a quiet one
the right quiet move is tried first only 76% of the time. Every miss costs a subtree. The ordering of quiet moves comes
from history tables that know nothing about the position in front of them. This trains a small read-out of the
evaluator's hidden layer (which the search has already computed for the node) on our own self-play games: for every
position, the move that a 2,500-position search chose.

  hidden layer (2 x H activations) -> HID units (ReLU) -> a score for each from-square (64) and for each piece type
  and to-square (6 x 64); a move's score is from-score + (piece type, to-square)-score.

The evaluator itself is not changed. The read-out is appended to the network file as a trailer ('POL1') that the engine
(calc_s9) reads; it is evaluated only at nodes with enough depth left for move order to matter.

  python3 zero/pol_train.py <evaluator .ema.pt> <network .bin (may already carry trailers)> <out .bin> [HID=32] [positions=12000000]
"""
import glob, os, struct, sys, time
import numpy as np, torch, torch.nn.functional as F
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')); sys.path.insert(0, '.')
from model import RuleGraph, moves_to_mask

net_pt, net_bin, out_bin = sys.argv[1:4]; HID = int(sys.argv[4]) if len(sys.argv) > 4 else 32; NPOS = int(sys.argv[5]) if len(sys.argv) > 5 else 12_000_000
DEV = torch.device('cuda'); torch.backends.cuda.matmul.allow_tf32 = True; torch.manual_seed(0); BS = 8192
OREC = np.dtype([('board', 'u1', 64), ('castle', 'u1'), ('ep', 'u1'), ('move', '<u2'), ('score', '<i2'), ('result', 'i1'), ('ply', 'u1')])
w = {k: v.to(DEV).float() for k, v in torch.load(net_pt, map_location='cpu').items()}; H = w['b'].shape[0]
_i = np.arange(768); LUT = torch.from_numpy(np.concatenate([((_i // 64 + 6) % 12) * 64 + ((_i % 64) ^ 56), [768]]).astype(np.int64)).to(DEV)
SQ = torch.arange(64, device=DEV); graph = RuleGraph().to(DEV)


@torch.no_grad()
def inputs(board, castle, ep):
    """-> hidden activations (B, 2H) in the engine's order (mover's, then opponent's), legal-move mask (B,64,64), piece type on each square"""
    B = board.shape[0]; idx = torch.where(board > 0, (board - 1) * 64 + SQ, torch.full_like(board, 768)); one = torch.ones(idx.shape, device=DEV)
    Mo = torch.zeros(B, 769, device=DEV).scatter_add_(1, idx, one)[:, :768]; Mp = torch.zeros(B, 769, device=DEV).scatter_add_(1, LUT[idx], one)[:, :768]
    h = torch.cat([(w['b'] + Mo @ w['E'][:768]).clamp(0, 1) ** 2, (w['b'] + Mp @ w['E'][:768]).clamp(0, 1) ** 2], 1)
    _, _, M = graph(board, castle, ep)
    return h, moves_to_mask(M, board)[:, :4096].view(B, 64, 64), (board.clamp(1, 6) - 1)


class Head(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.l1 = torch.nn.Linear(2 * H, HID); self.l2 = torch.nn.Linear(HID, 64 + 384)

    def forward(self, h, ptype):
        o = self.l2(F.relu(self.l1(h))); B = h.shape[0]
        return o[:, :64, None] + o[:, 64:].view(B, 6, 64).gather(1, ptype[:, :, None].expand(B, 64, 64))        # (B, from, to)


def load(files, cap):
    r = np.concatenate([np.fromfile(f, dtype=OREC) for f in files]); r = r[r['move'] < 4096][:cap]
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a))                  # kept in main memory; batches go to the card (it is shared with other jobs)
    return t(r['board']), t(r['castle']), t(r['ep']), t(r['move'].astype(np.int64))


files = sorted(glob.glob('zero/opool/*.o'), key=os.path.getmtime); files = files[-min(len(files), NPOS // 20000 + 30):]
val = files[::25]; vs = set(val); D = load([f for f in files if f not in vs], NPOS); V = load(val, 400_000)
head = Head().to(DEV); opt = torch.optim.Adam(head.parameters(), lr=2e-3); n = len(D[0]); steps = int(3 * n / BS); t0 = time.time()
print(f'{n/1e6:.2f}M positions from {len(files) - len(val)} files, {len(V[0])/1e6:.2f}M held out; evaluator hidden {H}, read-out {2*H} -> {HID} -> {64 + 384}; {steps} steps', flush=True)


def loss_of(b, c, e, mv):
    h, mask, ptype = inputs(b.long(), c.long(), e.long()); L = head(h, ptype)
    mask = mask.reshape(-1, 4096).scatter(1, mv[:, None], True); L = L.reshape(-1, 4096).masked_fill(~mask, -1e9)
    return F.cross_entropy(L, mv), L, mask


for step in range(steps):
    k = torch.randint(0, n, (BS,))
    for g in opt.param_groups:
        g['lr'] = 2e-3 * (0.03 + 0.97 * 0.5 * (1 + np.cos(np.pi * step / steps)))
    loss, _, _ = loss_of(*(x[k].to(DEV) for x in D)); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    if (step + 1) % 1000 == 0:
        print(f'step {step + 1}/{steps} loss {float(loss):.3f}', flush=True)
with torch.no_grad():
    top1 = top3 = qn = q1 = qrank = nmoves = 0.0; tot = 0; ce = 0.0
    for j in range(0, len(V[0]), 8192):
        b, c, e, mv = (x[j:j + 8192].to(DEV) for x in V); l, L, mask = loss_of(b, c, e, mv); ce += float(l) * len(mv); tot += len(mv)
        rank = (L > L.gather(1, mv[:, None])).sum(1); top1 += float((rank == 0).sum()); top3 += float((rank < 3).sum()); nmoves += float(mask.sum())
        to_empty = b.long().gather(1, (mv % 64)[:, None]).squeeze(1) == 0                                       # the chosen move is quiet: rank it among the quiet moves only
        qmask = mask & (b.long()[:, None, :] == 0).expand(-1, 64, -1).reshape(-1, 4096); Lq = L.masked_fill(~qmask, -1e9)
        r2 = (Lq > Lq.gather(1, mv[:, None])).sum(1)[to_empty]; qn += float(to_empty.sum()); q1 += float((r2 == 0).sum()); qrank += float(r2.float().sum())
    print(f'held-out positions: the search\'s move is ranked first in {top1/tot:.1%}, in the first three in {top3/tot:.1%} (a position has {nmoves/tot:.0f} legal moves on average; loss {ce/tot:.3f})')
    print(f'when the search\'s move is a quiet one ({qn/tot:.0%} of positions): ranked first among the quiet moves in {q1/qn:.1%}; on average {qrank/qn:.1f} quiet moves are ranked above it')
sd = {k: v.detach().cpu().numpy().astype('<f4') for k, v in head.state_dict().items()}
open(out_bin, 'wb').write(open(net_bin, 'rb').read() + struct.pack('<ii', 0x314C4F50, HID) + sd['l1.weight'].tobytes() + sd['l1.bias'].tobytes() + sd['l2.weight'].tobytes() + sd['l2.bias'].tobytes())
torch.save({k: torch.from_numpy(v) for k, v in sd.items()}, out_bin + '.pol.pt')
print(f'{time.time() - t0:.0f}s -> {out_bin}')
