"""Train the calculator's evaluation: directly-solved tapered piece-square table + a small trained network on top.

  out = taper(S_mg, S_eg)(position) + wo . screlu(b + sum E[own-view features]) + wp . screlu(b + sum E[opp-view features])
  target: sigmoid(out) = win probability of the side to move (deep Stockfish)

  python3 calc/train_nnue.py --h 128 --files b,cb --epochs 2 --out calc/net.bin
"""
import argparse, math, struct, time, numpy as np, torch, torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument('--h', type=int, default=128); ap.add_argument('--files', default='b,cb')
ap.add_argument('--epochs', type=float, default=2.0); ap.add_argument('--bs', type=int, default=16384)
ap.add_argument('--lr', type=float, default=1e-3); ap.add_argument('--out', default='calc/net.bin')
ap.add_argument('--pst', default='calc/pst.npz'); ap.add_argument('--minutes', type=float, default=1e9)
ap.add_argument('--threads', type=int, default=4); ap.add_argument('--chunk', type=int, default=4_000_000)
ap.add_argument('--resume', default=''); ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--buckets', type=int, default=1)       # output weights per piece-count bucket
ap.add_argument('--ema', type=float, default=0.0)        # weight averaging decay (0 = off); the averaged weights are exported
ap.add_argument('--kb', type=int, default=1)             # 7 = king-bucketed, mirrored hidden-layer inputs; 1 = plain
ap.add_argument('--warm', default='')                    # start from a plain (kb=1) network's weights
ap.add_argument('--val', default='val')                  # validation data tag
ap.add_argument('--device', default='cpu')               # cpu | cuda | auto
ap.add_argument('--experts', type=int, default=1)        # specialists per piece-count group, chosen by a learned router
ap.add_argument('--phase', action='store_true')          # eight phase specialists: separate input weights per piece-count group (--kb 8)
ap.add_argument('--overlap', type=float, default=0.0)    # share of boundary positions that train the neighbouring specialist instead (keeps neighbours consistent)
ap.add_argument('--opt', default='adam')                 # optimiser for the big input matrix: adam | muon | muon2 | normuon (the rest stays on Adam)
ap.add_argument('--mlr', type=float, default=0.03)       # learning rate of the matrix optimiser
ap.add_argument('--max_rows', type=int, default=0)       # use only this many positions per file (quick experiments)
ap.add_argument('--gate', type=int, default=32)          # the router reads this many hidden units per side
ap.add_argument('--widen', default='')                   # start from a narrower network: copy its units, new units start with zero output weights
ap.add_argument('--dense', type=int, default=-1)         # 1 = matrix-product formulation (fast on a GPU), 0 = lookup-and-sum (fast on a CPU), -1 = choose by device
ap.add_argument('--ytag', default='y')                   # which value target to train on: y (production), ys (smoothed), yl (lambda-return) -- see zero/foresight.py
ap.add_argument('--aux_set', default='all')              # all = every foresight target; inv = only those that do not depend on move order (square map 8 plies on, the reply, volatility)
ap.add_argument('--aux', type=float, default=0.0)        # weight of the foresight heads (training only; the exported network is unchanged). Needs data/ev_<tag>_aux.npy
ap.add_argument('--mirror', type=float, default=0.0)     # share of training rows shown mirrored left-right (the game is symmetric; doubles what each game teaches)
ap.add_argument('--mirror_all', type=int, default=0)     # 0 = only rows where both kings have left their home squares (castling is certainly gone), 1 = every row
ap.add_argument('--prod', type=int, default=0)           # 1 = product terms on top of the squares: unit i x unit i+H/2 within each view, and my unit i x the opponent's unit i
                                                         # (piece-pair interactions the squares cannot express; a few thousand extra output weights, the big input table unchanged)
ap.add_argument('--extrap', type=float, default=0.0)     # move every label this share further away from what the STARTING network says (in score units): the search
                                                         # corrects the evaluator only part of the way, and game results sit further along the same line (zero/teacher2.py)
a = ap.parse_args()
torch.set_num_threads(a.threads); torch.manual_seed(a.seed); rs = np.random.RandomState(a.seed)

# feature index seen from the other side: swap own/opp, mirror the board vertically
i = np.arange(768); pc, sq = i // 64, i % 64
LUT = torch.from_numpy(np.concatenate([((pc + 6) % 12) * 64 + (sq ^ 56), [768]]).astype(np.int64))
PT_W = torch.cat([torch.tensor([0, 1, 1, 2, 4, 0] * 2, dtype=torch.float32).repeat_interleave(64), torch.zeros(1)])
H = a.h
DEV = torch.device('cuda' if a.device == 'cuda' or (a.device == 'auto' and torch.cuda.is_available()) else 'cpu')
M = a.experts; G = min(a.gate, a.h); NEX = a.buckets * M       # NEX output experts in total
ROUTE = {'tau': 1.0, 'hard': False, 'train': True}                       # router temperature / hard top-1 routing (set by the training loop)
NKB = a.kb
KBT = torch.tensor([0, 0, 1, 2, 3, 3, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5] + [6] * 16)
KB11 = torch.tensor([0, 0, 1, 2, 3, 4, 5, 5, 6, 6, 6, 7, 7, 8, 8, 8] + [9] * 16 + [10] * 32)


def kfeat(idx):
    """plain view-relative indices (B,32) -> king-bucketed, mirrored indices; the view's own king is piece code 5"""
    isk = (idx >= 320) & (idx < 384)
    ks = ((idx * isk).sum(1) - 320).clamp(0, 63)    # a handful of source positions are malformed (king missing): send them to bucket of a1
    if NKB == 11:                                    # un-mirrored buckets: every bucket can start as a copy of a plain network
        return torch.where(idx == 768, torch.full_like(idx, NKB * 768), KB11[ks][:, None] * 768 + idx)
    mir = ((ks & 7) >= 4).long() * 7
    kq = ks ^ mir
    kb = KBT[(kq >> 3) * 4 + (kq & 7)]
    f = kb[:, None] * 768 + (idx // 64) * 64 + ((idx % 64) ^ mir[:, None])
    return torch.where(idx == 768, torch.full_like(idx, NKB * 768), f)


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.E = torch.nn.Parameter(torch.randn(NKB * 768 + 1, H) * 0.05)
        self.b = torch.nn.Parameter(torch.full((H,), 0.1))
        self.wo = torch.nn.Parameter(torch.randn(NEX, H) * (1.0 / math.sqrt(H)))
        self.wp = torch.nn.Parameter(torch.randn(NEX, H) * (1.0 / math.sqrt(H)))
        self.cb = torch.nn.Parameter(torch.zeros(NEX))
        if M > 1:                                               # router: one score per expert from 2G hidden activations
            self.gw = torch.nn.Parameter(torch.randn(NEX, 2 * G) * 0.3)
            self.gb = torch.nn.Parameter(torch.zeros(NEX))
        if a.prod:                                              # start at zero: a network grown from one without products computes the same function
            self.qo = torch.nn.Parameter(torch.zeros(NEX, H // 2)); self.qp = torch.nn.Parameter(torch.zeros(NEX, H // 2)); self.qx = torch.nn.Parameter(torch.zeros(NEX, H))
        self.S = torch.nn.Parameter(torch.zeros(769, 2))            # [:,0] middlegame, [:,1] endgame
        self.c = torch.nn.Parameter(torch.zeros(2))
        if a.aux > 0:                                               # foresight heads: 3 later scores, plies left, 10 + 10 material changes, from, to, volatility
            self.A = torch.nn.Linear(2 * H, 153 + 192 + 128)    # ... + who holds each square 8 plies on (64 x 3) + the opponent's reply (from, to)

    def forward(self, idx, feats=False):
        if DENSE:
            # The same network written as matrix products with a (batch, 768) piece-count matrix. On a GPU the lookup-and-sum
            # form spends over 80% of a step scattering gradients into a 769-row table; this form is about six times faster.
            B = idx.shape[0]; one = torch.ones(idx.shape, device=idx.device)
            Mo = torch.zeros(B, 769, device=idx.device).scatter_add_(1, idx, one)[:, :768]
            Mp = torch.zeros(B, 769, device=idx.device).scatter_add_(1, LUT[idx], one)[:, :768]
            bk = ((Mo.sum(1) - 2) // 4).clamp(0, a.buckets - 1).long() if a.buckets > 1 else torch.zeros(B, dtype=torch.long, device=idx.device)
            co = (self.b + Mo @ self.E[:768]).clamp(0, 1); cp = (self.b + Mp @ self.E[:768]).clamp(0, 1); xo = co ** 2; xp = cp ** 2
            lin = self.c + Mo @ self.S[:768]; ph = ((Mo @ PT_W[:768]) / 24).clamp(max=1.0)
            heads = xo @ self.wo.T + xp @ self.wp.T + self.cb                       # every piece-count head at once; the position's own is picked below
            if a.prod:
                h = H // 2; heads = heads + (co[:, :h] * co[:, h:]) @ self.qo.T + (cp[:, :h] * cp[:, h:]) @ self.qp.T + (co * cp) @ self.qx.T
            out = ph * lin[:, 0] + (1 - ph) * lin[:, 1] + (heads * F.one_hot(bk, NEX).to(heads.dtype)).sum(1)
            return (out, torch.cat([xo, xp], 1)) if feats else out
        bk = (((idx < 768).sum(1) - 2) // 4).clamp(0, a.buckets - 1) if a.buckets > 1 else torch.zeros(idx.shape[0], dtype=torch.long, device=idx.device)
        if a.phase and a.overlap > 0 and ROUTE['train']:          # a position one capture away from the next phase sometimes trains that phase's specialist
            o = ((idx < 768).sum(1) - 2) % 4; pick = torch.rand(idx.shape[0], device=idx.device) < a.overlap
            bk = torch.where(pick & (o == 0) & (bk > 0), bk - 1, torch.where(pick & (o == 3) & (bk < 7), bk + 1, bk))
        if a.phase:                                             # each position only touches the weights of its own phase specialist
            pb = (bk * 768)[:, None]
            ip0 = LUT[idx]
            io = torch.where(idx == 768, torch.full_like(idx, NKB * 768), pb + idx); ip = torch.where(ip0 == 768, torch.full_like(idx, NKB * 768), pb + ip0)
        else:
            io, ip = (idx, LUT[idx]) if NKB == 1 else (kfeat(idx), kfeat(LUT[idx]))
        ao = self.b + F.embedding_bag(io, self.E, mode='sum', padding_idx=NKB * 768)
        ap_ = self.b + F.embedding_bag(ip, self.E, mode='sum', padding_idx=NKB * 768)
        lin = self.c + F.embedding_bag(idx, self.S, mode='sum', padding_idx=768)
        ph = (PT_W[idx].sum(1) / 24).clamp(max=1.0)
        xo = ao.clamp(0, 1) ** 2; xp = ap_.clamp(0, 1) ** 2
        base = ph * lin[:, 0] + (1 - ph) * lin[:, 1]
        if M == 1:
            out = base + self.cb[bk] + (xo * self.wo[bk]).sum(1) + (xp * self.wp[bk]).sum(1)
            if a.prod:
                h = H // 2; co = ao.clamp(0, 1); cp = ap_.clamp(0, 1)
                out = out + (co[:, :h] * co[:, h:] * self.qo[bk]).sum(1) + (cp[:, :h] * cp[:, h:] * self.qp[bk]).sum(1) + (co * cp * self.qx[bk]).sum(1)
            return (out, torch.cat([xo, xp], 1)) if feats else out
        e = bk[:, None] * M + torch.arange(M, device=idx.device)                                        # the piece-count group's experts
        outs = (xo[:, None, :] * self.wo[e]).sum(2) + (xp[:, None, :] * self.wp[e]).sum(2) + self.cb[e]
        logits = (torch.cat([xo[:, :G], xp[:, :G]], 1)[:, None, :] * self.gw[e]).sum(2) + self.gb[e]
        p = torch.softmax(logits / ROUTE['tau'], 1)
        if ROUTE['hard']:                                                            # the engine runs exactly one expert: train that way at the end
            p = F.one_hot(logits.argmax(1), M).float() + p - p.detach()
        return base + (p * outs).sum(1)

    def export(self, path):
        with open(path, 'wb') as f:
            if M > 1:
                f.write(struct.pack('<iiiiii', 0x4F555235, H, NEX, NKB, M, G))
            else:
                f.write(struct.pack('<iiii', 0x4F555236 if a.prod else 0x4F555233, H, a.buckets, NKB) if NKB > 1 or a.prod else struct.pack('<iii', 0x4F555232, H, a.buckets))
            for t in (self.E[:NKB * 768], self.b):
                f.write(t.detach().cpu().numpy().astype('<f4').tobytes())
            for k in range(NEX):
                f.write(self.wo[k].detach().cpu().numpy().astype('<f4').tobytes()); f.write(self.wp[k].detach().cpu().numpy().astype('<f4').tobytes())
            f.write(struct.pack('<f', float(self.c[0]))); f.write(self.S[:768, 0].detach().cpu().numpy().astype('<f4').tobytes())
            f.write(struct.pack('<f', float(self.c[1]))); f.write(self.S[:768, 1].detach().cpu().numpy().astype('<f4').tobytes())
            f.write(self.cb.detach().cpu().numpy().astype('<f4').tobytes())
            if M > 1:
                f.write(self.gw.detach().cpu().numpy().astype('<f4').tobytes()); f.write(self.gb.detach().cpu().numpy().astype('<f4').tobytes())
            if a.prod:
                for k in range(NEX):
                    for t in (self.qo[k], self.qp[k], self.qx[k]):
                        f.write(t.detach().cpu().numpy().astype('<f4').tobytes())
            if PVAL is not None:
                f.write(struct.pack('<i5i', 0x4C415650, *PVAL))


PVAL = [int(x) for x in np.load(a.pst)['pval']] if a.pst and 'pval' in np.load(a.pst).files else None
net = Net()
ema = Net() if a.ema > 0 else None
if a.resume:
    net.load_state_dict(torch.load(a.resume, map_location='cpu'))
elif a.widen:
    w = torch.load(a.widen, map_location='cpu'); h0 = w['E'].shape[1]
    with torch.no_grad():                               # function-preserving growth: the new network computes what the old one did
        net.E[:, :h0] = w['E']; net.b[:h0] = w['b']
        r = NEX // w['wo'].shape[0]                     # experts per old output head
        net.wo.zero_(); net.wp.zero_()
        net.wo[:, :h0] = w['wo'].repeat_interleave(r, 0); net.wp[:, :h0] = w['wp'].repeat_interleave(r, 0)
        net.cb.copy_(w['cb'].repeat_interleave(r, 0))
        if r > 1:                                       # identical twins would stay identical: nudge them apart
            net.wo[:, :h0] += torch.randn(NEX, h0) * 0.02 * w['wo'].std(); net.wp[:, :h0] += torch.randn(NEX, h0) * 0.02 * w['wp'].std()
        if M > 1 and 'gw' in w and w['gw'].shape == net.gw.shape:
            net.gw.copy_(w['gw']); net.gb.copy_(w['gb'])
        for k in ('S', 'c'):
            getattr(net, k).copy_(w[k])
elif a.warm:
    w = torch.load(a.warm, map_location='cpu')
    with torch.no_grad():
        for k in ('b', 'wo', 'wp', 'cb', 'S', 'c'):
            getattr(net, k).copy_(w[k])
        E0 = w['E'][:768].view(12, 64, H)
        Em = E0.view(12, 8, 8, H).flip(2).reshape(12, 64, H)            # the same table mirrored left-right
        sym = ((E0 + Em) / 2).reshape(768, H) if NKB == 7 else w['E'][:768]   # mirrored buckets share a table across both wings; un-mirrored ones copy exactly
        net.E[:NKB * 768] = sym.repeat(NKB, 1); net.E[NKB * 768] = 0
elif a.pst:
    z = np.load(a.pst)
    with torch.no_grad():
        net.S[:768, 0] = torch.from_numpy(z['S_mg']); net.S[:768, 1] = torch.from_numpy(z['S_eg'])
        net.c[0] = float(z['c_mg']); net.c[1] = float(z['c_eg'])
import os
def _positions(t):                                       # a shard in the usual form (32 feature numbers a position) or the compact one (24 bytes, zero/compact.py)
    return np.load(f'data/ev_{t}_x.npy', mmap_mode='r') if os.path.exists(f'data/ev_{t}_x.npy') else np.load(f'data/ev_{t}_c.npy', mmap_mode='r')
files = [(_positions(t), np.load(f'data/ev_{t}_{a.ytag}.npy', mmap_mode='r'),
          np.load(f'data/ev_{t}_aux.npy', mmap_mode='r') if a.aux > 0 else None) for t in a.files.split(',')]
if a.max_rows:
    files = [(x[:a.max_rows], y[:a.max_rows], None if ax is None else ax[:a.max_rows]) for x, y, ax in files]
total = sum(len(f_[0]) for f_ in files)
steps_total = int(a.epochs * total / a.bs)

net.to(DEV)
if ema is not None:
    ema.to(DEV)
DENSE = (a.dense == 1 or (a.dense < 0 and DEV.type == 'cuda')) and NKB == 1 and not a.phase and M == 1
if DEV.type == 'cuda':
    torch.backends.cuda.matmul.allow_tf32 = True
LUT, PT_W, KBT, KB11 = LUT.to(DEV), PT_W.to(DEV), KBT.to(DEV), KB11.to(DEV)


def newton_schulz(G, steps=5):
    """approximate orthogonalisation (polar factor) with the quintic iteration used by Muon"""
    a_, b_, c_ = 3.4445, -4.7750, 2.0315
    X = G / (G.norm() + 1e-7)
    tall = X.shape[0] > X.shape[1]
    if tall:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a_ * X + (b_ * A + c_ * A @ A) @ X
    return X.T if tall else X


class MatOpt:
    """Muon-family optimisers for one matrix parameter.
    muon    : Nesterov momentum -> orthogonalise                              (Jordan et al. 2024)
    muon2   : momentum / (sqrt(second moment) + eps) -> orthogonalise         (Liu et al. 2026, Algorithm 1)
    normuon : momentum -> orthogonalise -> per-neuron second-moment normalise (Li et al. 2026, reference code)"""
    def __init__(self, p, kind, b1=0.95, b2=0.95, eps=1e-8):
        self.p, self.kind, self.b1, self.b2, self.eps = p, kind, b1, b2, eps
        self.m = torch.zeros_like(p); self.v = torch.zeros_like(p) if kind == 'muon2' else None
        self.v2 = torch.zeros(p.shape[1], 1, device=p.device) if kind == 'normuon' else None        # one value per hidden unit (neuron)

    @torch.no_grad()
    def step(self, lr):
        g = self.p.grad
        if self.kind == 'muon2':
            self.m.mul_(self.b1).add_(g, alpha=1 - self.b1); self.v.mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
            O = newton_schulz(self.m / (self.v.sqrt() + self.eps))
        else:
            self.m.lerp_(g, 1 - self.b1)
            O = newton_schulz(g.lerp(self.m, self.b1))                             # Nesterov
            if self.kind == 'normuon':
                O = O.T                                                            # rows = hidden units
                vnorm = O.norm()
                self.v2.lerp_((O * O).mean(1, keepdim=True), 1 - self.b2)
                O = O / (self.v2.sqrt() + 1e-10)
                O = (O * (vnorm / (O.norm() + 1e-10))).T
        self.p.add_(O, alpha=-lr)
        self.p[-1].zero_()                                                         # the padding row stays zero


matopt = MatOpt(net.E, a.opt) if a.opt != 'adam' else None
opt = torch.optim.Adam([p for n_, p in net.named_parameters() if matopt is None or n_ != 'E'], lr=a.lr, betas=(0.9, 0.999), eps=1e-8, fused=DEV.type == 'cuda')
fwd = torch.compile(net) if DENSE and DEV.type == 'cuda' else net        # the compiled model is the same network in fewer GPU kernels
REF = None
if a.extrap != 0:                                                        # a frozen copy of the network as it starts
    REF = Net().to(DEV); REF.load_state_dict(net.state_dict())
    for p_ in REF.parameters():
        p_.requires_grad_(False)
if ema is not None:
    ema.load_state_dict(net.state_dict())
VX = torch.from_numpy(np.load(f'data/ev_{a.val}_x.npy').astype(np.int64)).to(DEV); VY = torch.from_numpy(np.load(f'data/ev_{a.val}_y.npy').astype(np.float32)).to(DEV)


@torch.no_grad()
def validate(m=None):
    m = m or net
    keep = dict(ROUTE); ROUTE.update(tau=1.0, hard=True, train=False)   # validate the way the engine runs
    out = torch.cat([m(VX[i:i + 50000]) for i in range(0, len(VX), 50000)])
    ROUTE.update(keep)
    return float(F.binary_cross_entropy_with_logits(out, VY)), float((torch.sigmoid(out) - VY).abs().mean())


print(f'positions {total/1e6:.1f}M, hidden {H}, steps {steps_total}, start val (log-loss, MAE) {validate()}', flush=True)
step = 0; t0 = time.time(); seen = 0; run = 0.0; nrun = 0
chunks = [(fi, s) for fi, f_ in enumerate(files) for s in range(0, len(f_[0]), a.chunk)]


def aux_loss(feat, ax):
    """foresight: what the hidden layer must be able to tell about the rest of the game (played positions only)"""
    m = ax[:, 0] > 0
    if int(m.sum()) < 8:
        return feat.sum() * 0.0, {}
    o = net.A(feat[m]); t = ax[m].float(); fr = ax[m][:, 25].long(); to = ax[m][:, 26].long(); pm = fr < 64
    parts = {'later': F.binary_cross_entropy_with_logits(o[:, 0:3], t[:, 1:4] / 255),
             'left': F.smooth_l1_loss(o[:, 3], t[:, 4] / 50),
             'mat8': F.mse_loss(o[:, 4:14], t[:, 5:15] - 16),
             'matend': F.mse_loss(o[:, 14:24], (t[:, 15:25] - 16) / 4),
             'move': (F.cross_entropy(o[pm][:, 24:88], fr[pm]) + F.cross_entropy(o[pm][:, 88:152], to[pm])) if bool(pm.any()) else o.sum() * 0.0,
             'vol': F.binary_cross_entropy_with_logits(o[:, 152], t[:, 27] / 255)}
    if ax.shape[1] >= 47:
        tb = ax[m][:, 28:44].long(); own = torch.stack([(tb >> k) & 3 for k in (0, 2, 4, 6)], 2).reshape(-1, 64); ov = ax[m][:, 44] > 0
        parts['squares'] = F.cross_entropy(o[ov][:, 153:345].reshape(-1, 3), own[ov].reshape(-1)) if bool(ov.any()) else o.sum() * 0.0
        rf = ax[m][:, 45].long(); rt = ax[m][:, 46].long(); rm = rf < 64
        parts['reply'] = (F.cross_entropy(o[rm][:, 345:409], rf[rm]) + F.cross_entropy(o[rm][:, 409:473], rt[rm])) if bool(rm.any()) else o.sum() * 0.0
    if a.aux_set == 'inv':
        total_ = parts['squares'] + 0.1 * parts['reply'] + 0.5 * parts['vol']
        return total_, {k_: float(v_) for k_, v_ in parts.items() if k_ in ('squares', 'reply', 'vol')}
    total_ = parts['later'] + 0.2 * parts['left'] + 0.5 * parts['mat8'] + 0.5 * parts['matend'] + 0.1 * parts['move'] + 0.5 * parts['vol']
    if 'squares' in parts:
        total_ = total_ + parts['squares'] + 0.1 * parts['reply']
    return total_, {k_: float(v_) for k_, v_ in parts.items()}


def mirrored(idx):
    """a random share of the rows mirrored left-right. The network never sees castling rights; a position whose kings have
    both left their home squares is exactly as legal and exactly as good mirrored. In the mover's view the own king's home
    is e1 (feature 5*64+4) and the opponent's is e8 (feature 11*64+60)."""
    m = torch.rand(idx.shape[0], device=idx.device) < a.mirror
    if not a.mirror_all:
        m &= ~((idx == 324) | (idx == 764)).any(1)
    return torch.where(m[:, None] & (idx < 768), (idx // 64) * 64 + ((idx % 64) ^ 7), idx)


def unpack(c):
    """(n,24) bytes -> (n,32) feature numbers: 64 occupancy bits, then a 4-bit piece code for each occupied square in square order"""
    b = c[:, :8].long(); occ = sum(b[:, k] << (8 * k) for k in range(8))
    bits = ((occ[:, None] >> torch.arange(64, device=c.device)) & 1).bool()
    sq = torch.where(bits, torch.arange(64, device=c.device, dtype=torch.int16), torch.tensor(64, device=c.device, dtype=torch.int16)).sort(1).values[:, :32]
    code = torch.stack([c[:, 8:] & 15, c[:, 8:] >> 4], 2).reshape(len(c), 32).to(torch.int16)
    return torch.where(sq < 64, code * 64 + sq, torch.tensor(768, device=c.device, dtype=torch.int16))


AUXLOG = {}
done = False
assert not (a.mirror > 0 and a.aux > 0), 'the foresight targets are not mirrored'
while not done:
    for ci in rs.permutation(len(chunks)):
        fi, s = chunks[ci]
        X = torch.from_numpy(np.ascontiguousarray(files[fi][0][s:s + a.chunk])).to(DEV); Y = torch.from_numpy(np.ascontiguousarray(files[fi][1][s:s + a.chunk])).to(DEV)
        if X.shape[1] == 24:                                     # compact shard: unpack on the device, half a million positions at a time
            X = torch.cat([unpack(X[j:j + 500_000]) for j in range(0, len(X), 500_000)])
        AX = torch.from_numpy(np.ascontiguousarray(files[fi][2][s:s + a.chunk])).to(DEV) if a.aux > 0 else None
        perm = torch.randperm(len(X), device=DEV)
        for j in range(0, len(X) - a.bs + 1, a.bs):
            k = perm[j:j + a.bs]
            lr = a.lr * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * min(1.0, step / steps_total)))) * min(1.0, (step + 1) / 200)
            for g in opt.param_groups:
                g['lr'] = lr
            frac = step / max(1, steps_total)
            ROUTE['tau'] = max(0.3, 1.0 - frac); ROUTE['hard'] = frac >= 0.7      # soft mixture first, exactly-one-expert for the last 30%
            if a.aux > 0:
                out_, feat_ = fwd(X[k].long(), feats=True)
                loss = F.binary_cross_entropy_with_logits(out_, Y[k].float() / 255)
                la, AUXLOG = aux_loss(feat_, AX[k])
                run_main = loss.detach(); loss = loss + a.aux * la
            else:
                xb = mirrored(X[k].long()) if a.mirror > 0 else X[k].long()
                yb = Y[k].float() / 255
                if REF is not None:
                    with torch.no_grad():
                        ly = torch.logit(yb.clamp(1 / 510, 1 - 1 / 510)); yb = torch.sigmoid(ly + a.extrap * (ly - REF(xb)))
                loss = F.binary_cross_entropy_with_logits(fwd(xb), yb); run_main = loss.detach()
            opt.zero_grad(set_to_none=True)
            if matopt is not None:
                net.E.grad = None
            loss.backward(); opt.step()
            if matopt is not None:
                matopt.step(a.mlr * lr / a.lr)                                 # same warm-up and cosine schedule as the rest
            if ema is not None:
                with torch.no_grad():
                    d = min(a.ema, (1 + step) / (10 + step))
                    for pe, pm in zip(ema.parameters(), net.parameters()):
                        pe.lerp_(pm, 1 - d)
            step += 1; seen += a.bs; run = run + run_main; nrun += 1          # the loss stays on the GPU until it is printed: no wait every step
            if step % 500 == 0:
                print(f'step {step}/{steps_total} seen {seen/1e6:.1f}M loss {float(run)/nrun:.5f} {seen/(time.time()-t0)/1e3:.0f}k pos/s'
                      + ('  foresight ' + ' '.join(f'{k_} {v_:.3f}' for k_, v_ in AUXLOG.items()) if AUXLOG else ''), flush=True)
                run = 0.0; nrun = 0
            if step >= steps_total or time.time() - t0 > a.minutes * 60:
                done = True; break
        if done:
            break
        if (ci % 4) == 0:
            (ema or net).export(a.out); torch.save({k_: v_.cpu() for k_, v_ in net.state_dict().items()}, a.out + '.pt')
            print(f'  checkpoint: val (log-loss, MAE) {validate(ema or net)}', flush=True)
(ema or net).export(a.out); torch.save({k_: v_.cpu() for k_, v_ in net.state_dict().items()}, a.out + '.pt')
if ema is not None:
    torch.save({k_: v_.cpu() for k_, v_ in ema.state_dict().items()}, a.out + '.ema.pt')
print(f'done: steps {step}, val (log-loss, MAE) raw {validate()} averaged {validate(ema) if ema is not None else None}, {time.time()-t0:.0f}s -> {a.out}', flush=True)
