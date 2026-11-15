"""OURO — a looped, rule-grounded chess transformer.

Three ideas, one network:
 1. Rule-graph attention. The attention graph is biased by relations computed
    exactly from the rules (who attacks whom, x-rays through one blocker,
    pseudo-legal moves). Zero parameters, zero learning needed for geometry.
 2. Dory-style loop. Input block -> shared Recurrent block applied k times with
    input re-injection and state normalisation -> Output block. Trained with a
    random loop count and truncated backprop through the last loop only.
    The state after k loops can be cached and *resumed* to k+1.
 3. Dueling move heads. One pass yields policy pi(a), value V and Q(a)=V+A(a)
    for every move, so blunders can be vetoed without search.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

N_MOVES = 4096 + 192
N_REL = 9

# ----------------------------------------------------------------------------
# geometry tables
# ----------------------------------------------------------------------------
_DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]  # (drank, dfile)


def _build_tables():
    ray = torch.full((64, 8, 7), 64, dtype=torch.long)
    for s in range(64):
        r, f = divmod(s, 8)
        for d, (dr, df) in enumerate(_DIRS):
            for k in range(7):
                rr, ff = r + dr * (k + 1), f + df * (k + 1)
                if 0 <= rr < 8 and 0 <= ff < 8:
                    ray[s, d, k] = rr * 8 + ff
    kn = torch.zeros(64, 64, dtype=torch.bool)
    kg = torch.zeros(64, 64, dtype=torch.bool)
    pu = torch.zeros(64, 64, dtype=torch.bool)   # pawn attacks, moving up
    pd = torch.zeros(64, 64, dtype=torch.bool)   # pawn attacks, moving down
    for s in range(64):
        r, f = divmod(s, 8)
        for dr, df in [(1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1)]:
            rr, ff = r + dr, f + df
            if 0 <= rr < 8 and 0 <= ff < 8:
                kn[s, rr * 8 + ff] = True
        for dr in (-1, 0, 1):
            for df in (-1, 0, 1):
                rr, ff = r + dr, f + df
                if (dr or df) and 0 <= rr < 8 and 0 <= ff < 8:
                    kg[s, rr * 8 + ff] = True
        for df in (-1, 1):
            if 0 <= f + df < 8:
                if r + 1 < 8:
                    pu[s, (r + 1) * 8 + f + df] = True
                if r - 1 >= 0:
                    pd[s, (r - 1) * 8 + f + df] = True
    return ray, kn, kg, pu, pd


class RuleGraph(nn.Module):
    """Exact, parameter-free relational structure of a position (batched)."""

    def __init__(self):
        super().__init__()
        ray, kn, kg, pu, pd = _build_tables()
        self.register_buffer('ray', ray, persistent=False)
        self.register_buffer('ray_flat', ray.view(1, 64, 56), persistent=False)
        self.register_buffer('onboard', (ray < 64).view(1, 64, 8, 7), persistent=False)
        self.register_buffer('kn', kn, persistent=False)
        self.register_buffer('kg', kg, persistent=False)
        self.register_buffer('pu', pu, persistent=False)
        self.register_buffer('pd', pd, persistent=False)
        ar = torch.arange(64)
        self.register_buffer('rank', ar // 8, persistent=False)

    @torch.no_grad()
    def forward(self, board, castle, ep):
        """board (B,64) long, castle (B,) long, ep (B,) long (64 = none).
        returns rel (B,N_REL,64,64) bool [query i, key j], cnt (B,64,12) float,
                moves (B,64,64) bool pseudo-legal own moves from->to."""
        B = board.shape[0]
        dev = board.device
        occ = board > 0
        own = (board >= 1) & (board <= 6)
        opp = board >= 7
        occ_pad = torch.cat([occ, torch.ones(B, 1, dtype=torch.bool, device=dev)], 1)
        occ_ray = occ_pad[:, self.ray]                       # B,64,8,7
        csum = occ_ray.to(torch.int8).cumsum(-1)
        before = csum - occ_ray.to(torch.int8)
        vis = (before == 0) & self.onboard
        xr = (before == 1) & self.onboard
        orth = (board == 4) | (board == 5) | (board == 10) | (board == 11)
        diag = (board == 3) | (board == 5) | (board == 9) | (board == 11)
        dir_ok = torch.cat([orth[:, :, None].expand(B, 64, 4), diag[:, :, None].expand(B, 64, 4)], 2)[..., None]
        A = torch.zeros(B, 64, 65, dtype=torch.bool, device=dev)
        A.scatter_(2, self.ray_flat.expand(B, 64, 56), (vis & dir_ok).reshape(B, 64, 56))
        X = torch.zeros(B, 64, 65, dtype=torch.bool, device=dev)
        X.scatter_(2, self.ray_flat.expand(B, 64, 56), (xr & dir_ok).reshape(B, 64, 56))
        A = A[:, :, :64]
        X = X[:, :, :64]
        knight = (board == 2) | (board == 8)
        king = (board == 6) | (board == 12)
        A = A | (knight[:, :, None] & self.kn) | (king[:, :, None] & self.kg) \
              | ((board == 1)[:, :, None] & self.pu) | ((board == 7)[:, :, None] & self.pd)
        # attack counts per target square by attacker piece type (12 channels)
        onehot = F.one_hot(board, 13)[:, :, 1:].float()       # B,64,12
        cnt = torch.einsum('bst,bsp->btp', A.float(), onehot)
        # pseudo-legal moves for the side to move
        empty = ~occ
        own_s = own[:, :, None]
        pawn = (board == 1)[:, :, None]
        M = A & own_s & ~pawn & ~own[:, None, :]
        ep_hot = F.one_hot(ep, 65)[:, :64].bool()
        M = M | (pawn & self.pu & (opp | ep_hot)[:, None, :])
        # pushes
        e1 = torch.roll(empty, -8, 1)            # e1[s] = empty[s+8]
        e2 = torch.roll(empty, -16, 1)
        p = board == 1
        single = p & e1 & (self.rank < 7)
        double = p & e1 & e2 & (self.rank == 1)
        push = torch.zeros(B, 64, 64, dtype=torch.bool, device=dev)
        idx = torch.arange(56, device=dev)
        push[:, idx, idx + 8] = single[:, :56]
        idx2 = torch.arange(8, 16, device=dev)
        push[:, idx2, idx2 + 16] = double[:, 8:16]
        M = M | push
        # castling (attacked-square conditions are left to the true move generator)
        ks = ((castle & 1) > 0) & (board[:, 4] == 6) & (board[:, 7] == 4) & empty[:, 5] & empty[:, 6]
        qs = ((castle & 2) > 0) & (board[:, 4] == 6) & (board[:, 0] == 4) & empty[:, 1] & empty[:, 2] & empty[:, 3]
        M[:, 4, 6] |= ks
        M[:, 4, 2] |= qs
        # relation planes, indexed [query i, key j]
        At = A.transpose(1, 2)                               # At[i,j] = j attacks i
        same = (own[:, :, None] & own[:, None, :]) | (opp[:, :, None] & opp[:, None, :])
        diff = (own[:, :, None] & opp[:, None, :]) | (opp[:, :, None] & own[:, None, :])
        rel = torch.stack([
            At & own[:, None, :],            # 0 an own piece on j attacks/defends square i
            At & opp[:, None, :],            # 1 an opp piece on j attacks square i
            A & empty[:, None, :],           # 2 i controls empty square j
            A & same,                        # 3 i defends j
            A & diff,                        # 4 i attacks enemy on j
            X.transpose(1, 2),               # 5 j x-rays i
            X,                               # 6 i x-rays j
            M,                               # 7 i can move to j
            M.transpose(1, 2),               # 8 j can move to i
        ], 1)
        return rel, cnt, M


def moves_to_mask(M, board):
    """(B,64,64) pseudo-legal from->to  ->  (B, N_MOVES) bool incl. under-promotions"""
    B = M.shape[0]
    base = M.reshape(B, 4096)
    pm = M[:, 48:56, 56:64] & (board[:, 48:56] == 1)[:, :, None]
    under = pm.reshape(B, 64, 1).expand(B, 64, 3).reshape(B, 192)
    return torch.cat([base, under], 1)


# ----------------------------------------------------------------------------
# network
# ----------------------------------------------------------------------------
def norm(x):
    return F.layer_norm(x, x.shape[-1:])


class Block(nn.Module):
    """Pre-norm transformer block. Attention logits = scaled q.k + a bias that is handed in
    (the rule-graph bias of this block's group). Linear layers run in bf16 (AMX), maps in fp32."""

    def __init__(self, d, heads, mlp):
        super().__init__()
        self.h = heads
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.fc = nn.Linear(d, mlp, bias=False)
        self.pr = nn.Linear(mlp, d, bias=False)
        nn.init.zeros_(self.o.weight)
        nn.init.zeros_(self.pr.weight)
        self.qk_scale = nn.Parameter(torch.full((heads, 1, 1), 1.0 / math.sqrt(d // heads)))

    def forward(self, x, bias, amp=True):
        """x (B,T,d) fp32; bias (B*H,T,T) fp32, contiguous."""
        B, T, d = x.shape
        H = self.h
        with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=amp):
            qkv = F.linear(norm(x), torch.cat([self.q.weight, self.k.weight, self.v.weight], 0))
        q, k, v = qkv.float().view(B, T, 3, H, d // H).permute(2, 0, 3, 1, 4).unbind(0)   # each B,H,T,dh
        q = (q * self.qk_scale).reshape(B * H, T, -1)
        att = torch.baddbmm(bias, q, k.reshape(B * H, T, -1).transpose(1, 2)).softmax(-1)
        a = torch.bmm(att, v.reshape(B * H, T, -1)).view(B, H, T, -1).transpose(1, 2).reshape(B, T, d)
        with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=amp):
            x = x + self.o(a)
            x = x + self.pr(F.gelu(self.fc(norm(x))))
        return x.float()


class Ouro(nn.Module):
    amp = True

    def __init__(self, d=96, heads=4, mlp=288, n_in=2, n_rec=3, n_out=2, use_rel=True, use_cnt=True,
                 max_loops=4, smolgen=0, fut=0):
        super().__init__()
        self.cfg = dict(d=d, heads=heads, mlp=mlp, n_in=n_in, n_rec=n_rec, n_out=n_out,
                        use_rel=use_rel, use_cnt=use_cnt, max_loops=max_loops)
        if smolgen:
            self.cfg['smolgen'] = smolgen
        if fut:
            self.cfg['fut'] = fut
        self.smolgen = smolgen
        if smolgen:
            # Smolgen (Leela Chess Zero's transformers): attention biases GENERATED from the whole position. The rule graph
            # already supplies exact who-attacks-whom biases; this adds a learned, position-dependent bias on top. Each
            # token is squeezed to 8 numbers, the 512 numbers of the board go through a bottleneck of `smolgen` units to
            # 64 numbers per head, and one shared map turns those into a 64x64 bias per head. One generator for the
            # recurrent block (run at every loop, from that loop's state) and one for the output block. The shared map
            # starts at zero: the network begins as the same function without it.
            self.sg_comp = nn.ModuleList([nn.Linear(d, 8, bias=False) for _ in range(2)])
            self.sg_1 = nn.ModuleList([nn.Linear(512, smolgen) for _ in range(2)])
            self.sg_2 = nn.ModuleList([nn.Linear(smolgen, heads * 64) for _ in range(2)])
            self.sg_out = nn.Linear(64, 4096, bias=False)
            nn.init.zeros_(self.sg_out.weight)
        # training-only head: who will hold each square 8 plies from now (empty / mine / theirs), read off each square's token
        self.fut = nn.Linear(d, 3) if fut else None
        self.d = d
        self.heads = heads
        self.max_loops = max_loops
        self.use_rel, self.use_cnt = use_rel, use_cnt
        self.graph = RuleGraph()
        self.piece_emb = nn.Embedding(13, d)
        self.sq_emb = nn.Parameter(torch.randn(64, d) * 0.02)
        self.flag_emb = nn.Embedding(4, d)           # 0 none, 1 ep target, 2 rook with castling right
        self.cnt_in = nn.Linear(12, d, bias=False) if use_cnt else None
        mk = lambda n: nn.ModuleList([Block(d, heads, mlp) for _ in range(n)])
        self.inp, self.rec, self.out = mk(n_in), mk(n_rec), mk(n_out)
        # rule-graph attention bias: one set per group (input / recurrent / output), shared by its blocks
        self.rel_w = nn.Parameter(torch.zeros(3, heads, N_REL)) if use_rel else None
        self.static = nn.Parameter(torch.zeros(3, heads, 64, 64))
        # heads
        self.pf = nn.Linear(d, d, bias=False)
        self.pt = nn.Linear(d, d, bias=False)
        self.pu = nn.Linear(d, 3, bias=False)
        self.p_static = nn.Parameter(torch.zeros(64, 64))
        self.af = nn.Linear(d, d, bias=False)
        self.at = nn.Linear(d, d, bias=False)
        self.a_static = nn.Parameter(torch.zeros(64, 64))
        self.v1 = nn.Linear(d, d)
        self.v2 = nn.Linear(d, 1)
        # look-ahead heads: opponent's reply and own follow-up along the principal variation (training signal)
        self.la = nn.Linear(d, 4 * 64, bias=False)
        self.la_static = nn.Parameter(torch.zeros(2, 64, 64))
        nn.init.normal_(self.piece_emb.weight, std=0.02)
        nn.init.zeros_(self.flag_emb.weight)
        nn.init.zeros_(self.at.weight)
        nn.init.zeros_(self.pu.weight)
        nn.init.zeros_(self.v2.weight)
        nn.init.zeros_(self.v2.bias)

    # ---- stages -----------------------------------------------------------
    def _smol(self, g, u):
        B = u.shape[0]
        h = norm(F.gelu(self.sg_1[g](self.sg_comp[g](u).reshape(B, 512))))
        h = norm(F.gelu(self.sg_2[g](h)).view(B, self.heads, 64))
        return self.sg_out(h).reshape(B * self.heads, 64, 64)

    def _biases(self, rel, B):
        """attention-logit bias for each group: 1x1 conv over the exact relation planes + a learned static map."""
        out = []
        relf = rel.float() if self.use_rel else None
        for g in range(3):
            if self.use_rel:
                b = F.conv2d(relf, self.rel_w[g][:, :, None, None]) + self.static[g]
            else:
                b = self.static[g].unsqueeze(0).expand(B, -1, -1, -1)
            out.append(b.reshape(B * self.heads, 64, 64))
        return out

    def prepare(self, board, castle, ep):
        """rule graph + input block. returns ctx dict (can be cached per position)."""
        rel, cnt, M = self.graph(board, castle, ep)
        flag = torch.zeros_like(board)
        flag.scatter_(1, ep.clamp(max=63)[:, None], (ep < 64).long()[:, None])
        rookflag = torch.zeros_like(board)
        rookflag[:, 7] = ((castle & 1) > 0).long() * 2
        rookflag[:, 0] = ((castle & 2) > 0).long() * 2
        rookflag[:, 63] = ((castle & 4) > 0).long() * 2
        rookflag[:, 56] = ((castle & 8) > 0).long() * 2
        flag = torch.maximum(flag, rookflag)
        x = self.piece_emb(board) + self.sq_emb + self.flag_emb(flag)
        if self.use_cnt:
            x = x + self.cnt_in(cnt.clamp(max=3.0))
        b_in, b_rec, b_out = self._biases(rel, board.shape[0])
        for blk in self.inp:
            x = blk(x, b_in, self.amp)
        return dict(x=norm(x), rec_bias=b_rec, out_bias=b_out, mask=moves_to_mask(M, board))

    def loop(self, ctx, s=None):
        """one application of the shared recurrent block (Dory recurrence):
           u_t = N(s_{t-1} + x),  s_t = N(F(u_t)),  s_0 = x"""
        x = ctx['x']
        u = x if s is None else norm(s + x)
        bias = ctx['rec_bias'] + self._smol(0, u) if self.smolgen else ctx['rec_bias']
        for blk in self.rec:
            u = blk(u, bias, self.amp)
        return norm(u)

    def readout(self, ctx, s, aux=False):
        z = s
        ob = ctx['out_bias'] + self._smol(1, s) if self.smolgen else ctx['out_bias']
        for blk in self.out:
            z = blk(z, ob, self.amp)
        z = norm(z)
        B = z.shape[0]
        with torch.autocast(z.device.type, dtype=torch.bfloat16, enabled=self.amp):
            proj = F.linear(z, torch.cat([self.pf.weight, self.pt.weight, self.af.weight, self.at.weight], 0))
        pf, pt, af, at = proj.float().split(self.d, dim=-1)
        sc = 1.0 / math.sqrt(self.d)
        pol = torch.baddbmm(self.p_static.expand(B, 64, 64), pf, pt.transpose(1, 2), alpha=sc)   # B,64,64
        under = pol[:, 48:56, 56:64].reshape(B, 64, 1) + self.pu(z[:, 56:64]).repeat(1, 8, 1).reshape(B, 64, 3)
        pol = torch.cat([pol.reshape(B, 4096), under.reshape(B, 192)], 1)
        v = self.v2(F.gelu(self.v1(z.mean(1)))).squeeze(-1)                                # B (logit)
        adv = torch.baddbmm(self.a_static.expand(B, 64, 64), af, at.transpose(1, 2), alpha=sc)
        ql = v[:, None] + torch.cat([adv.reshape(B, 4096),
                                     adv[:, 48:56, 56:64].reshape(B, 64, 1).expand(B, 64, 3).reshape(B, 192)], 1)
        if not aux:
            return pol, v, ql
        with torch.autocast(z.device.type, dtype=torch.bfloat16, enabled=self.amp):
            la = self.la(z)
        f2, t2, f3, t3 = la.float().split(64, dim=-1)
        pv2 = torch.baddbmm(self.la_static[0].expand(B, 64, 64), f2, t2.transpose(1, 2), alpha=0.125)
        pv3 = torch.baddbmm(self.la_static[1].expand(B, 64, 64), f3, t3.transpose(1, 2), alpha=0.125)
        pvl = torch.stack([pv2.reshape(B, 4096), pv3.reshape(B, 4096)], 1)
        if self.fut is not None:
            return pol, v, ql, (pvl, self.fut(z))
        return pol, v, ql, pvl

    def forward(self, board, castle, ep, loops=1, tbptl=True, aux=False):
        ctx = self.prepare(board, castle, ep)
        s = None
        if tbptl and loops > 1:
            with torch.no_grad():
                for _ in range(loops - 1):
                    s = self.loop(ctx, s)
            s = self.loop(ctx, s)
        else:
            for _ in range(loops):
                s = self.loop(ctx, s)
        if aux:
            pol, v, ql, pvl = self.readout(ctx, s, aux=True)
            return pol, v, ql, ctx['mask'], pvl
        pol, v, ql = self.readout(ctx, s)
        return pol, v, ql, ctx['mask']

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


# ----------------------------------------------------------------------------
# Muon (orthogonalised momentum) for hidden matrices
# ----------------------------------------------------------------------------
def _ns5(G, steps=5):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G / (G.norm() + 1e-7)
    t = G.size(0) > G.size(1)
    if t:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        Bm = b * A + c * A @ A
        X = a * X + Bm @ X
    return X.T if t else X


def _ns5_batched(G, steps=5):
    """the same iteration as _ns5 for a stack of equally shaped matrices (K, m, n): one batched product instead of K small ones"""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G / (G.flatten(1).norm(dim=1)[:, None, None] + 1e-7)
    t = G.size(1) > G.size(2)
    if t:
        X = X.transpose(1, 2)
    for _ in range(steps):
        A = X @ X.transpose(1, 2)
        Bm = b * A + c * A @ A
        X = a * X + Bm @ X
    return X.transpose(1, 2) if t else X


class Muon(torch.optim.Optimizer):
    """Muon (momentum, then orthogonalise the update). Matrices of the same shape are orthogonalised together: the model has
    47 of them in three shapes, and doing them one by one cost about 750 tiny matrix products per step, more time than the
    forward and backward pass together. The arithmetic is unchanged."""
    def __init__(self, params, lr=0.02, momentum=0.95, wd=0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, wd=wd))
        self._stacks = {}

    @torch.no_grad()
    def step(self):
        for gi, g in enumerate(self.param_groups):
            by_shape = {}
            for p in g['params']:
                if p.grad is not None:
                    by_shape.setdefault(tuple(p.shape), []).append(p)
            for shape, ps in by_shape.items():
                key = (gi, shape, len(ps))
                S = self._stacks.get(key)
                if S is None or any(self.state[p].get('buf') is None or self.state[p]['buf'].data_ptr() != S[i].data_ptr() for i, p in enumerate(ps)):
                    # (re)build the stacked momentum buffer; each parameter's 'buf' is a view into it, so checkpoints keep working
                    S = torch.stack([self.state[p]['buf'] if 'buf' in self.state[p] else torch.zeros_like(p.grad) for p in ps])
                    for i, p in enumerate(ps):
                        self.state[p]['buf'] = S[i]
                    self._stacks[key] = S
                G = torch.stack([p.grad for p in ps])
                S.lerp_(G, 1 - g['momentum'])
                U = _ns5_batched(G.lerp(S, g['momentum']))
                scale = -g['lr'] * max(1.0, shape[0] / shape[1]) ** 0.5
                for i, p in enumerate(ps):
                    if g['wd']:
                        p.mul_(1 - g['lr'] * g['wd'])
                    p.add_(U[i], alpha=scale)


def make_optimizers(model, muon_lr=0.02, adam_lr=3e-3, wd=0.0, use_muon=True):
    hidden, other = [], []
    for n, p in model.named_parameters():
        if use_muon and p.ndim == 2 and any(n.startswith(s) for s in ('inp.', 'rec.', 'out.')):
            hidden.append(p)
        else:
            other.append(p)
    opts = [torch.optim.AdamW(other, lr=adam_lr, betas=(0.9, 0.95), weight_decay=0.0, fused=all(p.is_cuda for p in other))]
    if hidden:
        opts.append(Muon(hidden, lr=muon_lr, momentum=0.95, wd=wd))
    return opts
