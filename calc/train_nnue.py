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
ap.add_argument('--kb', type=int, default=1)             # 7, 16 or 32 = king-bucketed, mirrored hidden-layer inputs; 11 = king-bucketed, not mirrored; 1 = plain
ap.add_argument('--kf', type=int, default=0)             # with --kb: 1 = every bucket's table is a shared table plus the bucket's own correction (summed in the exported file):
                                                         # what all king positions have in common is learned from every position, rare buckets inherit it
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
ap.add_argument('--lr_end', type=float, default=0.0)      # learning rate at the last step (0 = the usual 2% of --lr). Long runs of other self-taught engines decay to 1e-6
ap.add_argument('--hk', type=int, default=0)             # with --head: the layers read only the first hk units of each side (0 = all); the engine pays per unit read
ap.add_argument('--hm', type=int, default=0)             # 1 = mirrored inputs: a view's inputs are flipped left-right when its own king is on files a-d (one set of weights for both wings)
ap.add_argument('--head', default='')                    # '16,32' = a small layered read-out per piece-count bucket on top of the direct one (2H -> 16 -> 32 -> 1); starts at zero
ap.add_argument('--prod', type=int, default=0)           # 1 = product terms on top of the squares: unit i x unit i+H/2 within each view, and my unit i x the opponent's unit i
                                                         # (piece-pair interactions the squares cannot express; a few thousand extra output weights, the big input table unchanged)
ap.add_argument('--extrap', type=float, default=0.0)     # move every label this share further away from what the STARTING network says (in score units): the search
                                                         # corrects the evaluator only part of the way, and game results sit further along the same line (zero/teacher2.py)
ap.add_argument('--sharp', type=float, default=1.0)      # train on labels made steeper: win chance -> sigmoid(sharp x its logit). 2 = the loss sees a 50-centipawn error near
                                                         # equality twice as large (Stockfish's own trainer uses a scale about 2.5 times steeper than these labels'). The
                                                         # exported network's values are sharp times larger: play it with EvalScale = 100 / sharp.
ap.add_argument('--distill', default='')                 # a teacher network's weights (.ema.pt; king-bucketed with the shared table folded or not, any width,
                                                         # the same bucket scheme as the student): its win chance for each training position joins the label
                                                         # Several teachers joined by '+' (e.g. zero/w_BIGX.bin.ema.pt+sf:<file>): the mean of their win chances
ap.add_argument('--dw', type=float, default=0.7)         # the teacher's share of the training target (1 = the teacher alone)
                                                         # --distill sf:<file>: the teacher is Stockfish 14.1's own network (ext/sfteach.py, the file from
                                                         # ext/sfnet_extract.py); its value is brought to the labels' scale with --tscale
ap.add_argument('--pow', type=float, default=0.0)        # > 0: the loss is |win chance - target| to this power (2 = squared error; 0 = the usual log-loss)
ap.add_argument('--wclip', type=float, default=0.0)      # > 0: input weights (shared table and bucket corrections each) are kept within +- this after every step
ap.add_argument('--obk', type=int, default=0)            # how the read-out bucket is chosen: 0 piece count (--buckets 8); 1 piece count x any queen on the board (--buckets 16);
                                                         # 2 piece count x my queen x their queen (--buckets 32)
ap.add_argument('--am', default='')                      # attack-map inputs computed from the rules, "a piece of this kind attacks this square" (sliders stop at the
                                                         # first piece): all = 12 kinds, class = pawn / knight+bishop / rook / queen / king per side, colour = one map
                                                         # per side, slide, nonslide, pawn. Not king-bucketed; added into the same hidden layer.
ap.add_argument('--amk', type=int, default=0)            # the attack maps feed only the first amk hidden units of each side (0 = all; the engine pays per unit fed)
ap.add_argument('--tl', type=float, default=0.0)         # > 0: a stored teacher, data/tl_<tag>.npy (one byte a position, win chance x255: the transformer's value, lm/label_nnue.py),
                                                         # takes this share of the teachers' part of the target (1 = it alone); needs --max_rows no larger than what was labelled
ap.add_argument('--sib', default='')                     # sibling families: shards of the transformer's data (lm/prep_lm.py prefixes, comma separated): positions with several
                                                         # candidate moves, each with Stockfish's deep win chance. The network scores the positions after the quiet candidates and
                                                         # is trained on which of two siblings is better and by how much -- the question the search asks of it
ap.add_argument('--sibw', type=float, default=1.0)       # weight of the pair loss: log-loss of sigmoid(z_a - z_b) against sigmoid(t_a - t_b), logits of the two siblings
ap.add_argument('--sibp', type=float, default=0.0)       # weight of the plain loss on the same sibling positions (each against its own deep win chance)
ap.add_argument('--sibn', type=int, default=4096)        # families a step
ap.add_argument('--tscale', type=float, default=0.277)   # our units per Stockfish internal unit (361 of its units = 100 of ours)
a = ap.parse_args()
torch.set_num_threads(a.threads); torch.manual_seed(a.seed); rs = np.random.RandomState(a.seed)

# feature index seen from the other side: swap own/opp, mirror the board vertically
i = np.arange(768); pc, sq = i // 64, i % 64
LUT = torch.from_numpy(np.concatenate([((pc + 6) % 12) * 64 + (sq ^ 56), [768]]).astype(np.int64))
PT_W = torch.cat([torch.tensor([0, 1, 1, 2, 4, 0] * 2, dtype=torch.float32).repeat_interleave(64), torch.zeros(1)])
# who attacks whom, from the rules, in the mover's view (codes 0-5 the mover's P N B R Q K, its pawns move up the board; 6-11 the opponent's)
_ATT = np.zeros((13, 64, 64), bool); _BETW = np.zeros((64, 64, 64), bool)
for _a in range(64):
    _ar, _af = divmod(_a, 8)
    for _b in range(64):
        if _a == _b:
            continue
        _br, _bf = divmod(_b, 8); _dr, _df = _br - _ar, _bf - _af; _diag = abs(_dr) == abs(_df); _line = _dr == 0 or _df == 0
        _ATT[0, _a, _b] = _dr == 1 and abs(_df) == 1; _ATT[6, _a, _b] = _dr == -1 and abs(_df) == 1
        for _c in (1, 7): _ATT[_c, _a, _b] = (abs(_dr), abs(_df)) in ((1, 2), (2, 1))
        for _c in (2, 8): _ATT[_c, _a, _b] = _diag
        for _c in (3, 9): _ATT[_c, _a, _b] = _line
        for _c in (4, 10): _ATT[_c, _a, _b] = _diag or _line
        for _c in (5, 11): _ATT[_c, _a, _b] = max(abs(_dr), abs(_df)) == 1
        if _diag or _line:
            _sr, _sf = (_dr > 0) - (_dr < 0), (_df > 0) - (_df < 0); _r, _f = _ar + _sr, _af + _sf
            while (_r, _f) != (_br, _bf):
                _BETW[_a, _b, _r * 8 + _f] = True; _r += _sr; _f += _sf
ATT_T = torch.from_numpy(_ATT.reshape(-1))
BETW64 = torch.from_numpy((_BETW.reshape(4096, 64).astype(np.uint64) << np.arange(64, dtype=np.uint64)).sum(1).astype(np.uint64).view(np.int64))   # squares between two squares, as 64 bits
AM_G = {'all': [[k] for k in range(6)], 'pawn': [[0]], 'nonslide': [[0], [1], [5]], 'slide': [[2], [3], [4]], 'colour': [[0, 1, 2, 3, 4, 5]], 'class': [[0], [1, 2], [3], [4], [5]]}


@torch.no_grad()
def attack_map(idx):
    """(B,768): input code*64+s is on when some piece of that kind attacks square s (sliders stop at the first piece in the way)"""
    valid = idx < 768; code = torch.where(valid, idx // 64, torch.full_like(idx, 12)); sq = torch.where(valid, idx % 64, torch.zeros_like(idx))
    occ = torch.where(valid, torch.ones_like(sq) << sq, torch.zeros_like(sq)).sum(1)
    S64 = torch.arange(64, device=idx.device); base = sq[:, :, None] * 64 + S64
    hit = ATT_T[code[:, :, None] * 4096 + base] & ((BETW64[base] & occ[:, None, None]) == 0) & valid[:, :, None]
    f = torch.where(hit, code[:, :, None] * 64 + S64, torch.full_like(base, 768)).view(idx.shape[0], -1)
    return torch.zeros(idx.shape[0], 769, device=idx.device).scatter_(1, f, 1.0)[:, :768]


def am_groups(Am, spec):
    """merge the 12 attack maps into groups of kinds (the same grouping for the mover and for the opponent): (B,768) -> (B, groups*2*64)"""
    if spec == 'all':
        return Am
    A = Am.view(-1, 12, 64)
    return torch.cat([A[:, [k + o for k in g]].amax(1) for o in (0, 6) for g in AM_G[spec]], 1)


H = a.h
DEV = torch.device('cuda' if a.device == 'cuda' or (a.device == 'auto' and torch.cuda.is_available()) else 'cpu')
M = a.experts; G = min(a.gate, a.h); NEX = a.buckets * M       # NEX output experts in total
ROUTE = {'tau': 1.0, 'hard': False, 'train': True}                       # router temperature / hard top-1 routing (set by the training loop)
NKB = a.kb
HEAD = [int(x) for x in a.head.split(',')] if a.head else None; HK = a.hk or a.h
KBT = torch.tensor([0, 0, 1, 2, 3, 3, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5] + [6] * 16)
KB11 = torch.tensor([0, 0, 1, 2, 3, 4, 5, 5, 6, 6, 6, 7, 7, 8, 8, 8] + [9] * 16 + [10] * 32)
if NKB == 16:                                            # 16 mirrored buckets: every king square of the first two ranks its own, then half-ranks
    KBT = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13] + [14, 14, 15, 15] * 3)
if NKB == 32:                                            # every mirrored king square its own bucket
    KBT = torch.arange(32)


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


def hmir(idx):
    """flip the files of a view's inputs when that view's own king (piece code 5) stands on files a-d"""
    isk = (idx >= 320) & (idx < 384); ks = ((idx * isk).sum(1) - 320).clamp(0, 63)
    return torch.where((((ks & 7) < 4)[:, None]) & (idx < 768), idx ^ 7, idx)


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.E = torch.nn.Parameter(torch.randn(NKB * 768 + 1, H) * 0.05)
        if a.kf:                                                # shared table (its last row is padding and stays zero); the buckets' corrections start at zero
            self.E = torch.nn.Parameter(torch.zeros(NKB * 768 + 1, H))
            self.Esh = torch.nn.Parameter(torch.cat([torch.randn(768, H) * 0.05, torch.zeros(1, H)]))
        self.b = torch.nn.Parameter(torch.full((H,), 0.1))
        self.wo = torch.nn.Parameter(torch.randn(NEX, H) * (1.0 / math.sqrt(H)))
        self.wp = torch.nn.Parameter(torch.randn(NEX, H) * (1.0 / math.sqrt(H)))
        self.cb = torch.nn.Parameter(torch.zeros(NEX))
        if M > 1:                                               # router: one score per expert from 2G hidden activations
            self.gw = torch.nn.Parameter(torch.randn(NEX, 2 * G) * 0.3)
            self.gb = torch.nn.Parameter(torch.zeros(NEX))
        if a.prod:                                              # start at zero: a network grown from one without products computes the same function
            self.qo = torch.nn.Parameter(torch.zeros(NEX, H // 2)); self.qp = torch.nn.Parameter(torch.zeros(NEX, H // 2)); self.qx = torch.nn.Parameter(torch.zeros(NEX, H))
        if HEAD:                                                # the last layer starts at zero: a network grown from one without the layers computes the same function
            h1, h2 = HEAD; P_ = torch.nn.Parameter
            self.L1 = P_(torch.randn(NEX * h1, 2 * HK) / math.sqrt(2 * HK)); self.b1 = P_(torch.zeros(NEX * h1))
            self.L2 = P_(torch.randn(NEX, h2, h1) / math.sqrt(h1)); self.b2 = P_(torch.zeros(NEX, h2)); self.L3 = P_(torch.zeros(NEX, h2))
        if a.am:                                                    # attack-map inputs: start at zero
            self.Eam = torch.nn.Parameter(torch.zeros(len(AM_G[a.am]) * 128, a.amk or H))
        self.S = torch.nn.Parameter(torch.zeros(769, 2))            # [:,0] middlegame, [:,1] endgame
        self.c = torch.nn.Parameter(torch.zeros(2))
        if a.aux > 0:                                               # foresight heads: 3 later scores, plies left, 10 + 10 material changes, from, to, volatility
            self.A = torch.nn.Linear(2 * H, 153 + 192 + 128)    # ... + who holds each square 8 plies on (64 x 3) + the opponent's reply (from, to)

    def layered(self, xo, xp, bk):
        """the layered read-out of the position's own bucket: squared clipped units, then clipped units, then a sum"""
        h1, h2 = HEAD; B = xo.shape[0]; ar = torch.arange(B, device=xo.device)
        u = (torch.cat([xo[:, :HK], xp[:, :HK]], 1) @ self.L1.T + self.b1).view(B, NEX, h1)[ar, bk].clamp(0, 1) ** 2
        v = (torch.bmm(self.L2[bk], u[:, :, None]).squeeze(2) + self.b2[bk]).clamp(0, 1)
        return (v * self.L3[bk]).sum(1)

    def bucketed(self, ix, w=None):
        """one view's hidden-layer sums with king-bucketed inputs, as matrix products: the batch is sorted by the view's king
        bucket and each bucket's rows meet that bucket's table. The same arithmetic as the plain form (every row meets one
        768-row table), where lookup-and-sum would scatter gradients into the table one piece at a time.
        With w (output weights, (NEX, H)): the view's share of every piece-count read-out instead, (B, NEX) -- the squared
        clipped units are read in sorted order and only those few numbers are put back in batch order."""
        B = ix.shape[0]; isk = (ix >= 320) & (ix < 384); ks = ((ix * isk).sum(1) - 320).clamp(0, 63)
        if NKB == 11:
            kb = KB11[ks]; f = ix
        else:
            mir = ((ks & 7) >= 4).long() * 7; kq = ks ^ mir; kb = KBT[(kq >> 3) * 4 + (kq & 7)]
            f = torch.where(ix == 768, ix, (ix // 64) * 64 + ((ix % 64) ^ mir[:, None]))
        order = torch.argsort(kb); cnt = torch.bincount(kb, minlength=NKB).tolist()
        Ms = torch.zeros(B, 769, device=ix.device).scatter_add_(1, f[order], torch.ones(ix.shape, device=ix.device))[:, :768]
        W = self.E[:NKB * 768].view(NKB, 768, H)
        W = (W + self.Esh[:768] if a.kf else W).unbind(0)
        inv = torch.empty_like(order); inv[order] = torch.arange(B, device=ix.device)
        if w is not None and B >= 24576:                        # large batches: the whole read-out bucket by bucket, nothing of hidden-layer size is joined
            return torch.cat([((self.b + sg @ W[k]).clamp(0, 1) ** 2) @ w.T for k, sg in enumerate(Ms.split(cnt)) if cnt[k]])[inv]
        h = self.b + torch.cat([sg @ W[k] for k, sg in enumerate(Ms.split(cnt)) if cnt[k]])
        return h[inv] if w is None else ((h.clamp(0, 1) ** 2) @ w.T)[inv]

    def forward(self, idx, feats=False):
        if DENSE:
            # The same network written as matrix products with a (batch, 768) piece-count matrix. On a GPU the lookup-and-sum
            # form spends over 80% of a step scattering gradients into a 769-row table; this form is about six times faster.
            B = idx.shape[0]; one = torch.ones(idx.shape, device=idx.device)
            Mo = torch.zeros(B, 769, device=idx.device).scatter_add_(1, idx, one)[:, :768]
            Mp = torch.zeros(B, 769, device=idx.device).scatter_add_(1, LUT[idx], one)[:, :768]
            bk = ((Mo.sum(1) - 2) // 4).clamp(0, a.buckets - 1).long() if a.buckets > 1 else torch.zeros(B, dtype=torch.long, device=idx.device)
            if a.obk:                                                       # the queens join the key: own queen is piece code 4, the opponent's 10
                pcb = ((Mo.sum(1) - 2) // 4).clamp(0, 7).long(); mq = (Mo[:, 256:320].sum(1) > 0).long(); oq = (Mo[:, 640:704].sum(1) > 0).long()
                bk = pcb * 2 + ((mq + oq) > 0).long() if a.obk == 1 else pcb * 4 + mq * 2 + oq
            lin = self.c + Mo @ self.S[:768]; ph = ((Mo @ PT_W[:768]) / 24).clamp(max=1.0)
            if NKB > 1 and not (a.prod or HEAD or feats or a.am):           # king buckets, direct read-out only: nothing of hidden-layer size leaves sorted order
                heads = self.bucketed(idx, self.wo) + self.bucketed(LUT[idx], self.wp) + self.cb
                return ph * lin[:, 0] + (1 - ph) * lin[:, 1] + (heads * F.one_hot(bk, NEX).to(heads.dtype)).sum(1)
            if NKB > 1:                                                     # king buckets: each view's rows grouped by its bucket, one matrix product per bucket
                ho = self.bucketed(idx); hp = self.bucketed(LUT[idx])
                if a.am:                                                    # the same attack maps seen from each side
                    Am = attack_map(idx); pad = (0, H - (a.amk or H))
                    ho = ho + F.pad(am_groups(Am, a.am) @ self.Eam, pad); hp = hp + F.pad(am_groups(Am[:, LUT[:768]], a.am) @ self.Eam, pad)
                co = ho.clamp(0, 1); cp = hp.clamp(0, 1)
            else:
                if a.hm:                                                    # the hidden layer sees the mirrored inputs; the linear shortcut and the bucket do not
                    Eo = torch.zeros(B, 769, device=idx.device).scatter_add_(1, hmir(idx), one)[:, :768]; Ep = torch.zeros(B, 769, device=idx.device).scatter_add_(1, hmir(LUT[idx]), one)[:, :768]
                else:
                    Eo, Ep = Mo, Mp
                co = (self.b + Eo @ self.E[:768]).clamp(0, 1); cp = (self.b + Ep @ self.E[:768]).clamp(0, 1)
            xo = co ** 2; xp = cp ** 2
            heads = xo @ self.wo.T + xp @ self.wp.T + self.cb                       # every piece-count head at once; the position's own is picked below
            if a.prod:
                h = H // 2; heads = heads + (co[:, :h] * co[:, h:]) @ self.qo.T + (cp[:, :h] * cp[:, h:]) @ self.qp.T + (co * cp) @ self.qx.T
            out = ph * lin[:, 0] + (1 - ph) * lin[:, 1] + (heads * F.one_hot(bk, NEX).to(heads.dtype)).sum(1)
            if HEAD:
                out = out + self.layered(xo, xp, bk)
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
            if a.hm:
                io, ip = hmir(io), hmir(ip)
        ao = self.b + F.embedding_bag(io, self.E, mode='sum', padding_idx=NKB * 768)
        ap_ = self.b + F.embedding_bag(ip, self.E, mode='sum', padding_idx=NKB * 768)
        if a.kf:
            sh = lambda t: torch.where(t == NKB * 768, torch.full_like(t, 768), t % 768)
            ao = ao + F.embedding_bag(sh(io), self.Esh, mode='sum', padding_idx=768); ap_ = ap_ + F.embedding_bag(sh(ip), self.Esh, mode='sum', padding_idx=768)
        lin = self.c + F.embedding_bag(idx, self.S, mode='sum', padding_idx=768)
        ph = (PT_W[idx].sum(1) / 24).clamp(max=1.0)
        xo = ao.clamp(0, 1) ** 2; xp = ap_.clamp(0, 1) ** 2
        base = ph * lin[:, 0] + (1 - ph) * lin[:, 1]
        if M == 1:
            out = base + self.cb[bk] + (xo * self.wo[bk]).sum(1) + (xp * self.wp[bk]).sum(1)
            if a.prod:
                h = H // 2; co = ao.clamp(0, 1); cp = ap_.clamp(0, 1)
                out = out + (co[:, :h] * co[:, h:] * self.qo[bk]).sum(1) + (cp[:, :h] * cp[:, h:] * self.qp[bk]).sum(1) + (co * cp * self.qx[bk]).sum(1)
            if HEAD:
                out = out + self.layered(xo, xp, bk)
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
            for t in (self.E[:NKB * 768] + self.Esh[:768].repeat(NKB, 1) if a.kf else self.E[:NKB * 768], self.b):
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
            if a.hm:
                f.write(struct.pack('<i', 0x3152494D))          # trailer 'MIR1': mirrored inputs (the engine reads it from calc_s14 on)
            if HEAD:                                            # trailer 'MLP1': the layered read-out (the engine reads it from calc_s13 on)
                f.write(struct.pack('<iii', 0x31504C4D, *HEAD) if HK == H else struct.pack('<iiii', 0x32504C4D, *HEAD, HK))     # 'MLP2' (calc_s16 on): the layers read the first HK units a side
                for t in (self.L1, self.b1, self.L2, self.b2, self.L3):
                    f.write(t.detach().cpu().numpy().astype('<f4').tobytes())


PVAL = [int(x) for x in np.load(a.pst)['pval']] if a.pst and 'pval' in np.load(a.pst).files else None
net = Net()
ema = Net() if a.ema > 0 else None
if a.resume:
    _sd = torch.load(a.resume, map_location='cpu'); _miss = [k for k in net.state_dict() if k not in _sd]
    assert all(k in ('Eam', 'qo', 'qp', 'qx', 'L1', 'b1', 'L2', 'b2', 'L3') for k in _miss), _miss      # parts the earlier network lacks start at zero: the same function to begin with
    net.load_state_dict(_sd, strict=False)
    if _miss: print('new parts added to the resumed network:', _miss, flush=True)
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
          np.load(f'data/ev_{t}_aux.npy', mmap_mode='r') if a.aux > 0 else None,
          np.load(f'data/tl_{t}.npy', mmap_mode='r') if a.tl > 0 else None) for t in a.files.split(',')]
if a.max_rows:
    files = [(x[:a.max_rows], y[:a.max_rows], None if ax is None else ax[:a.max_rows], None if tl is None else tl[:a.max_rows]) for x, y, ax, tl in files]
assert a.tl == 0 or all(len(f_[3]) >= len(f_[0]) for f_ in files), 'the stored teacher covers fewer positions than are trained on: set --max_rows'
total = sum(len(f_[0]) for f_ in files)
steps_total = int(a.epochs * total / a.bs)

net.to(DEV)
if ema is not None:
    ema.to(DEV)
DENSE = (a.dense == 1 or (a.dense < 0 and DEV.type == 'cuda')) and not a.phase and M == 1
if DEV.type == 'cuda':
    torch.backends.cuda.matmul.allow_tf32 = True
LUT, PT_W, KBT, KB11 = LUT.to(DEV), PT_W.to(DEV), KBT.to(DEV), KB11.to(DEV); ATT_T, BETW64 = ATT_T.to(DEV), BETW64.to(DEV)


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
fwd = torch.compile(net) if DENSE and DEV.type == 'cuda' and NKB == 1 else net        # the compiled model is the same network in fewer GPU kernels (bucketed inputs: batch-dependent shapes, not compiled)
TEACH = None


def _teacher(spec):
    """one teacher: 'sf:<file>' = Stockfish 14.1's network (ext/sfteach.py), otherwise the weights of one of our own networks. Returns positions -> logit"""
    if spec.startswith('sf:'):
        import sys; sys.path.insert(0, 'ext'); import sfteach
        _sf = sfteach.load(spec[3:], DEV); print(f'teacher: Stockfish 14.1 network from {spec[3:]}, scale {a.tscale}', flush=True)
        return lambda idx: _sf(idx)[1] * (a.tscale / 271.58)
    _t = {k: v.to(DEV) for k, v in torch.load(spec, map_location='cpu').items()}
    _TE = _t['E'][:-1]; _TN = _TE.shape[0] // 768; _TH = _TE.shape[1]
    assert _TN == NKB and NKB in (7, 16, 32), 'the teacher must use the student\'s king buckets'
    _TW = _TE.view(_TN, 768, _TH); _TW = (_TW + _t['Esh'][:768] if 'Esh' in _t else _TW).contiguous()
    print(f'teacher {spec}: {_TH} wide, {_TN} king buckets', flush=True)

    @torch.no_grad()
    def f_(idx):
        """the teacher's value (a logit) for a batch of positions: the same arithmetic as Net.forward's king-bucketed dense form"""
        B = idx.shape[0]; one = torch.ones(idx.shape, device=idx.device)
        Mo = torch.zeros(B, 769, device=idx.device).scatter_add_(1, idx, one)[:, :768]
        nb = _t['wo'].shape[0]; bk = ((Mo.sum(1) - 2) // 4).clamp(0, nb - 1).long()
        lin = _t['c'] + Mo @ _t['S'][:768]; ph = ((Mo @ PT_W[:768]) / 24).clamp(max=1.0)
        heads = _t['cb'].expand(B, nb).clone()
        for ix, w in ((idx, _t['wo']), (LUT[idx], _t['wp'])):
            isk = (ix >= 320) & (ix < 384); ks = ((ix * isk).sum(1) - 320).clamp(0, 63)
            mir = ((ks & 7) >= 4).long() * 7; kq = ks ^ mir; kb = KBT[(kq >> 3) * 4 + (kq & 7)]
            f = torch.where(ix == 768, ix, (ix // 64) * 64 + ((ix % 64) ^ mir[:, None]))
            order = torch.argsort(kb); cnt = torch.bincount(kb, minlength=_TN).tolist()
            Ms = torch.zeros(B, 769, device=ix.device).scatter_add_(1, f[order], torch.ones(ix.shape, device=ix.device))[:, :768]
            inv = torch.empty_like(order); inv[order] = torch.arange(B, device=ix.device)
            heads = heads + torch.cat([((_t['b'] + sg @ _TW[k]).clamp(0, 1) ** 2) @ w.T for k, sg in enumerate(Ms.split(cnt)) if cnt[k]])[inv]
        return ph * lin[:, 0] + (1 - ph) * lin[:, 1] + heads.gather(1, bk[:, None])[:, 0]
    return f_


if a.distill:                                                            # several teachers, joined by '+': the student is shown the mean of their win chances
    _TS = [_teacher(t_) for t_ in a.distill.split('+')]
    print(f'{len(_TS)} teacher(s), share of the target {a.dw}', flush=True)

    @torch.no_grad()
    def TEACH(idx):
        if len(_TS) == 1:
            return _TS[0](idx)
        return torch.logit(torch.stack([torch.sigmoid(t_(idx)) for t_ in _TS]).mean(0).clamp(1e-6, 1 - 1e-6))
REF = None
if a.extrap != 0:                                                        # a frozen copy of the network as it starts
    REF = Net().to(DEV); REF.load_state_dict(net.state_dict())
    for p_ in REF.parameters():
        p_.requires_grad_(False)
if ema is not None:
    ema.load_state_dict(net.state_dict())
VX = torch.from_numpy(np.load(f'data/ev_{a.val}_x.npy').astype(np.int64)).to(DEV); VY = torch.from_numpy(np.load(f'data/ev_{a.val}_y.npy').astype(np.float32)).to(DEV)


if TEACH is not None and a.resume == a.distill:                # self-check: the teacher function must equal the student's own forward for the same weights
    with torch.no_grad():
        print(f'teacher function against the network itself on 2,000 positions: largest difference {float((TEACH(VX[:2000]) - net(VX[:2000])).abs().max()):.6f} (logit)', flush=True)


@torch.no_grad()
def validate(m=None):
    m = m or net
    keep = dict(ROUTE); ROUTE.update(tau=1.0, hard=True, train=False)   # validate the way the engine runs
    out = torch.cat([m(VX[i:i + 50000]) for i in range(0, len(VX), 50000)])
    ROUTE.update(keep)
    out = out / a.sharp                                                 # back on the labels' own scale, so the figures stay comparable
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


SIBLOG = [0.0, 0.0, 0.0, 0]
if a.sib:
    _sb = [tuple(np.load(f'{p_}_{k_}.npy', mmap_mode='r') for k_ in 'bvw') for p_ in a.sib.split(',')]
    _SWAP = torch.tensor([0, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6], device=DEV); _S64 = torch.arange(64, device=DEV); _TRI = torch.triu(torch.ones(5, 5, dtype=torch.bool, device=DEV), 1)
    SIB_HOLD = 200_000                                       # the last rows of the last shard are never trained on

    def sib_rows(fi, s, n):
        """rows s..s+n of a shard -> the positions after the quiet candidate moves as feature numbers (n, 5, 32), seen by the
        side then to move; the first mover's win chance after each move (n, 5); which candidates are used (n, 5).
        Boards: square 0 = the mover's a1; 0 empty, 1-6 the mover's P N B R Q K, 7-12 the opponent's. Moves: from x 64 + to
        (castling = the king's two-square move; 4096 and up are under-promotions, not used here)."""
        B_, V_, W_ = _sb[fi]
        pk = torch.from_numpy(np.ascontiguousarray(B_[s:s + n])).to(DEV).long(); mv = torch.from_numpy(np.ascontiguousarray(V_[s:s + n])).to(DEV).long()
        w = torch.from_numpy(np.ascontiguousarray(W_[s:s + n]).astype(np.float32)).to(DEV)
        bd = torch.stack([pk & 15, pk >> 4], 2).reshape(n, 64); m = mv.clamp(0, 4095); fr = m // 64; to = m % 64
        pc = bd.gather(1, fr); tg = bd.gather(1, to)
        ok = (mv >= 0) & (mv < 4096) & (tg == 0) & ~((pc == 1) & (((fr % 8) != (to % 8)) | (to >= 56)))     # no captures (en passant included), no promotions
        cb = bd[:, None, :].repeat(1, 5, 1); cb.scatter_(2, fr[:, :, None], 0); cb.scatter_(2, to[:, :, None], pc[:, :, None])
        for kto, rf, rt in ((6, 7, 5), (2, 0, 3)):                                                          # castling: the rook moves too
            c_ = (pc == 6) & (fr == 4) & (to == kto)
            cb[:, :, rf] = torch.where(c_, torch.zeros_like(pc), cb[:, :, rf]); cb[:, :, rt] = torch.where(c_, torch.full_like(pc, 4), cb[:, :, rt])
        f = torch.where(cb > 0, (_SWAP[cb] - 1) * 64 + (_S64 ^ 56), torch.full_like(cb, 768)).sort(2).values[:, :, :32]
        return f, w, ok

    def sib_terms(m_, f, w, ok):
        """-> pair loss, plain loss, share of clearly different pairs ranked right, number of pairs"""
        z = torch.zeros(ok.shape, device=DEV); z[ok] = -m_(f[ok]).float()                                   # a child is scored from the other side
        t = torch.logit(w.clamp(0.02, 0.98)); pm = ok[:, :, None] & ok[:, None, :] & _TRI
        dz = (z[:, :, None] - z[:, None, :])[pm]; dt = (t[:, :, None] - t[:, None, :])[pm]; dw = (w[:, :, None] - w[:, None, :])[pm]
        clear = dw.abs() >= 0.04
        return (F.binary_cross_entropy_with_logits(dz, torch.sigmoid(dt)), F.binary_cross_entropy_with_logits(z[ok], w[ok]),
                ((dz[clear] > 0) == (dw[clear] > 0)).float().mean(), int(pm.sum()))

    def sib_step():
        fi = rs.randint(len(_sb)); last = len(_sb[fi][0]) - a.sibn - (SIB_HOLD if fi == len(_sb) - 1 else 0)
        return sib_terms(fwd, *sib_rows(fi, rs.randint(0, last), a.sibn))

    @torch.no_grad()
    def sib_validate(m_=None):
        fi = len(_sb) - 1; n0 = len(_sb[fi][0]) - SIB_HOLD; acc = [0.0, 0.0, 0.0, 0]
        for s_ in range(n0, n0 + SIB_HOLD, 20_000):
            pl, pt, ch, npair = sib_terms(m_ or net, *sib_rows(fi, s_, 20_000))
            acc = [acc[0] + float(pl) * npair, acc[1] + float(pt) * npair, acc[2] + float(ch) * npair, acc[3] + npair]
        return f'held-out sibling pairs {acc[3]}: pair loss {acc[0] / acc[3]:.5f}, plain loss {acc[1] / acc[3]:.5f}, clearly different pairs ranked right {100 * acc[2] / acc[3]:.2f}%'
    print('start ' + sib_validate(), flush=True)

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
        TL = torch.from_numpy(np.ascontiguousarray(files[fi][3][s:s + a.chunk])).to(DEV) if a.tl > 0 else None
        perm = torch.randperm(len(X), device=DEV)
        for j in range(0, len(X) - a.bs + 1, a.bs):
            k = perm[j:j + a.bs]
            end_ = a.lr_end / a.lr if a.lr_end > 0 else 0.02
            lr = a.lr * (end_ + (1 - end_) * 0.5 * (1 + math.cos(math.pi * min(1.0, step / steps_total)))) * min(1.0, (step + 1) / 200)
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
                if TEACH is not None or TL is not None:
                    tp = torch.sigmoid(TEACH(xb)) if TEACH is not None else None
                    if TL is not None: tp = TL[k].float() / 255 if tp is None or a.tl >= 1 else (1 - a.tl) * tp + a.tl * TL[k].float() / 255
                    yb = (1 - a.dw) * yb + a.dw * tp
                if a.sharp != 1.0:
                    yb = torch.sigmoid(a.sharp * torch.logit(yb.clamp(1 / 510, 1 - 1 / 510)))
                if REF is not None:
                    with torch.no_grad():
                        ly = torch.logit(yb.clamp(1 / 510, 1 - 1 / 510)); yb = torch.sigmoid(ly + a.extrap * (ly - REF(xb)))
                loss = (torch.sigmoid(fwd(xb)) - yb).abs().pow(a.pow).mean() if a.pow > 0 else F.binary_cross_entropy_with_logits(fwd(xb), yb); run_main = loss.detach()
            if a.sib:
                sl_pair, sl_point, sl_right, _ = sib_step()
                loss = loss + a.sibw * sl_pair + a.sibp * sl_point
                SIBLOG = [SIBLOG[0] + sl_pair.detach(), SIBLOG[1] + sl_point.detach(), SIBLOG[2] + sl_right, SIBLOG[3] + 1]
            opt.zero_grad(set_to_none=True)
            if matopt is not None:
                net.E.grad = None
            loss.backward(); opt.step()
            if a.wclip > 0:
                with torch.no_grad():
                    net.E.clamp_(-a.wclip, a.wclip)
                    if a.kf:
                        net.Esh.clamp_(-a.wclip, a.wclip)
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
                      + ('  foresight ' + ' '.join(f'{k_} {v_:.3f}' for k_, v_ in AUXLOG.items()) if AUXLOG else '')
                      + (f'  siblings: pair {float(SIBLOG[0]) / SIBLOG[3]:.5f} plain {float(SIBLOG[1]) / SIBLOG[3]:.5f} right {100 * float(SIBLOG[2]) / SIBLOG[3]:.1f}%' if a.sib else ''), flush=True)
                run = 0.0; nrun = 0; SIBLOG = [0.0, 0.0, 0.0, 0]
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
if a.sib:
    print('end ' + sib_validate(ema or net), flush=True)
