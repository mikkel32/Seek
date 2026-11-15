"""More judgement per parameter and per position: four ideas against plain width, on the same fresh positions.

Every variant is trained from scratch on the same positions in the same order for the same number of steps, and judged on
the same held-out positions (loss against the search-based label; lower is better). The yardstick is what doubling the
width buys: an idea is worth building into the engine if it beats that at fewer parameters.

  plain 384 / plain 768      the evaluator as it is, and twice as wide
  mirrored                   inputs mirrored left-right so that the own king is always on files a-d: one set of weights
                             serves both wings (what other engines call horizontal mirroring)
  king base+delta            the input weights depend on the own king's region, as a shared base table plus a small
                             correction per region (the earlier king-region test gave every region its own table, so
                             each saw a fraction of the data; here they share everything they have in common)
  rule graph                 a second, small accumulator fed by RELATIONS instead of pieces: for every attack or defence
                             (piece on i -> piece on j, from the exact rules) the product of two learned vectors, one for
                             the attacker and one for the target, is added up. The transformer's attention is biased by
                             the same relations; this is its idea in a form a fast evaluator can update.
  attack maps                768 more inputs computed from the rules: "a piece of this kind attacks this square" (sliders
                             stop at the first piece in the way). The engine can keep them as 12 bitboards per position.
  threats                    one input per (kind of attacker, attacked piece on its square), 12 x 768 of them, added into
                             the same hidden layer: who attacks whom, without the attacker's own square
  layered head               the hidden layer is read by a small network (16 -> 32 units per piece-count bucket) on top of
                             the direct read-out, instead of by the direct read-out alone
  looped head                the hidden layer is summarised in 32 numbers, which one small block with shared weights
                             refines three times (the transformer's loop, on the fast evaluator)
  hard positions             plain 384, but each position weighs in the loss by how wrong the evaluator currently is
                             about it: the data is fed where it teaches most
  other seed                 plain 384 again with another start and another order: how much of a difference is chance

  python3 zero/arch_probe.py [train shards=b0,b1,b2,b3] [held-out shard=b35] [steps=8000] [variants=all]
"""
import math, os, sys, time
import numpy as np, torch, torch.nn.functional as F
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')); sys.path.insert(0, '.'); sys.path.insert(0, 'zero')
from compact import decode_torch

TR = (sys.argv[1] if len(sys.argv) > 1 else 'b0,b1,b2,b3').split(','); HO = sys.argv[2] if len(sys.argv) > 2 else 'b35'
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 8000; WHICH = sys.argv[4].split(',') if len(sys.argv) > 4 else None
DEV = torch.device('cuda'); BS = 8192; torch.backends.cuda.matmul.allow_tf32 = True
_i = np.arange(768); LUT = torch.from_numpy(np.concatenate([((_i // 64 + 6) % 12) * 64 + ((_i % 64) ^ 56), [768]]).astype(np.int64)).to(DEV)
PT_W = torch.cat([torch.tensor([0, 1, 1, 2, 4, 0] * 2, dtype=torch.float32).repeat_interleave(64), torch.zeros(1)]).to(DEV)
KB11 = torch.tensor([0, 0, 1, 2, 3, 4, 5, 5, 6, 6, 6, 7, 7, 8, 8, 8] + [9] * 16 + [10] * 32, device=DEV)
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
ATT_T = torch.from_numpy(_ATT.reshape(-1)).to(DEV)
BETW64 = torch.from_numpy((_BETW.reshape(4096, 64).astype(np.uint64) << np.arange(64, dtype=np.uint64)).sum(1).astype(np.uint64).view(np.int64)).to(DEV)   # squares between two squares, as 64 bits

X = torch.cat([torch.from_numpy(np.load(f'data/ev_{t}_c.npy')) for t in TR]).to(DEV); Y = torch.cat([torch.from_numpy(np.load(f'data/ev_{t}_y.npy')) for t in TR]).to(DEV)
HX = torch.from_numpy(np.load(f'data/ev_{HO}_c.npy')[:1_000_000]).to(DEV); HY = torch.from_numpy(np.load(f'data/ev_{HO}_y.npy')[:1_000_000]).to(DEV)
VX = torch.from_numpy(np.load('data/ev_bval_x.npy').astype(np.int64)).to(DEV); VY = torch.from_numpy(np.load('data/ev_bval_y.npy')).to(DEV)
N = len(X)


def king_sq(idx):
    return ((idx * ((idx >= 320) & (idx < 384))).sum(1) - 320).clamp(0, 63)


def mirror(idx):
    m = ((king_sq(idx) & 7) >= 4)[:, None] & (idx < 768)
    return torch.where(m, (idx // 64) * 64 + ((idx % 64) ^ 7), idx)


S64 = torch.arange(64, device=DEV); LUT768 = LUT[:768]


def pieces(idx):
    valid = idx < 768; code = torch.where(valid, idx // 64, torch.full_like(idx, 12)); sq = torch.where(valid, idx % 64, torch.zeros_like(idx))
    occ = torch.where(valid, torch.ones_like(sq) << sq, torch.zeros_like(sq)).sum(1)                          # occupied squares as 64 bits
    return valid, code, sq, occ


@torch.no_grad()
def relations(idx):
    """the exact attack and defence relations between the pieces, from the rules: piece i -> piece j, as four (B,32,32) matrices
    (mine defends mine, theirs defends theirs, mine attacks theirs, theirs attacks mine)"""
    valid, code, sq, occ = pieces(idx); pair = sq[:, :, None] * 64 + sq[:, None, :]
    a = ATT_T[code[:, :, None] * 4096 + pair] & ((BETW64[pair] & occ[:, None, None]) == 0) & valid[:, :, None] & valid[:, None, :]
    own = code < 6; oi, oj = own[:, :, None], own[:, None, :]
    return [(a & oi & oj).float(), (a & ~oi & ~oj).float(), (a & oi & ~oj).float(), (a & ~oi & oj).float()]


@torch.no_grad()
def attack_map(idx):
    """(B,768): input code*64+s is on when some piece of that kind attacks square s (sliders stop at the first piece in the way)"""
    valid, code, sq, occ = pieces(idx); base = sq[:, :, None] * 64 + S64
    hit = ATT_T[code[:, :, None] * 4096 + base] & ((BETW64[base] & occ[:, None, None]) == 0) & valid[:, :, None]
    f = torch.where(hit, code[:, :, None] * 64 + S64, torch.full_like(base, 768)).view(idx.shape[0], -1)
    return torch.zeros(idx.shape[0], 769, device=DEV).scatter_(1, f, 1.0)[:, :768]


@torch.no_grad()
def threat_lists(idx):
    """every attack or defence piece i -> piece j as one input number per side's view: attacker kind * 768 + the target's piece-on-square
    -> (inputs in the mover's view, in the opponent's view, start of each position's list)"""
    valid, code, sq, occ = pieces(idx); pair = sq[:, :, None] * 64 + sq[:, None, :]
    a = ATT_T[code[:, :, None] * 4096 + pair] & ((BETW64[pair] & occ[:, None, None]) == 0) & valid[:, :, None] & valid[:, None, :]
    b, i, j = a.nonzero(as_tuple=True); ci = code[b, i]; fj = idx[b, j]
    off = torch.zeros(idx.shape[0] + 1, dtype=torch.long, device=DEV); off[1:] = torch.bincount(b, minlength=idx.shape[0]).cumsum(0)
    return ci * 768 + fj, ((ci + 6) % 12) * 768 + LUT[fj], off[:-1]


def multi_hot(idx):
    return torch.zeros(idx.shape[0], 769, device=DEV).scatter_add_(1, idx, torch.ones(idx.shape, device=DEV))[:, :768]


class Net(torch.nn.Module):
    def __init__(self, H, hm=False, kb=False, rel=0, am=False, head=None, thr=False):
        super().__init__(); self.H, self.hm, self.kb, self.rel, self.am, self.head, self.thr = H, hm, kb, rel, am, head, thr
        P = torch.nn.Parameter
        self.E = P(torch.randn(768, H) * 0.05); self.b = P(torch.full((H,), 0.1))
        self.wo = P(torch.randn(8, H) / math.sqrt(H)); self.wp = P(torch.randn(8, H) / math.sqrt(H)); self.cb = P(torch.zeros(8))
        self.S = P(torch.zeros(768, 2)); self.c = P(torch.zeros(2))
        if kb:
            self.D = P(torch.zeros(11 * 768 + 1, H))                           # correction per king region; starts at zero
        if rel:                                                                # one row per piece-on-square, 4 relation kinds x rel numbers
            self.EA = P(torch.randn(769, 4 * rel) * 0.2); self.EB = P(torch.randn(769, 4 * rel) * 0.2)
            self.br = P(torch.full((rel,), 0.1)); self.wr = P(torch.randn(8, rel) / math.sqrt(rel))
        if am:
            self.E2 = P(torch.zeros(768, H))
        if thr:
            self.E3 = P(torch.zeros(12 * 768, H))
        if head == 'mlp':
            self.L1 = P(torch.randn(2 * H, 8 * 16) / math.sqrt(2 * H)); self.b1 = P(torch.zeros(8 * 16))
            self.L2 = P(torch.randn(8, 16, 32) / 4); self.b2 = P(torch.zeros(8, 32)); self.L3 = P(torch.zeros(8, 32))
        if head == 'loop':
            self.Li = P(torch.randn(2 * H, 32) / math.sqrt(2 * H)); self.be = P(torch.zeros(8, 32))
            self.A1 = P(torch.randn(32, 64) / math.sqrt(32)); self.a1 = P(torch.zeros(64)); self.A2 = P(torch.randn(64, 32) * 0.02); self.Lo = P(torch.zeros(8, 32))

    def acc(self, idx):
        a = self.b + multi_hot(idx) @ self.E
        if self.kb:
            d = torch.where(idx == 768, torch.full_like(idx, 11 * 768), KB11[king_sq(idx)][:, None] * 768 + idx)
            a = a + F.embedding_bag(d, self.D, mode='sum', padding_idx=11 * 768)
        return a

    def forward(self, idx):
        io, ip = idx, LUT[idx]; B = idx.shape[0]
        Mo = multi_hot(idx); bk = ((Mo.sum(1) - 2) // 4).clamp(0, 7).long()
        lin = self.c + Mo @ self.S; ph = ((Mo @ PT_W[:768]) / 24).clamp(max=1.0)
        if self.hm:
            io, ip = mirror(io), mirror(ip)
        ao, ap = self.acc(io), self.acc(ip)
        if self.am:                                                            # the same attack map seen from each side
            Am = attack_map(idx); ao = ao + Am @ self.E2; ap = ap + Am[:, LUT768] @ self.E2
        if self.thr:
            fo, fp, off = threat_lists(idx); ao = ao + F.embedding_bag(fo, self.E3, off, mode='sum'); ap = ap + F.embedding_bag(fp, self.E3, off, mode='sum')
        xo = ao.clamp(0, 1) ** 2; xp = ap.clamp(0, 1) ** 2
        out = ph * lin[:, 0] + (1 - ph) * lin[:, 1] + self.cb[bk] + (xo * self.wo[bk]).sum(1) + (xp * self.wp[bk]).sum(1)
        if self.rel:
            att = relations(idx); ea = F.embedding(idx, self.EA).view(B, 32, 4, self.rel); eb = F.embedding(idx, self.EB).view(B, 32, 4, self.rel)
            r = self.br + sum((ea[:, :, t] * torch.bmm(att[t], eb[:, :, t])).sum(1) for t in range(4))
            out = out + ((r.clamp(0, 1) ** 2) * self.wr[bk]).sum(1)
        if self.head == 'mlp':
            x = torch.cat([xo, xp], 1); ar = torch.arange(B, device=DEV)
            h1 = (x @ self.L1 + self.b1).view(B, 8, 16)[ar, bk].clamp(0, 1) ** 2
            h2 = (torch.bmm(h1[:, None, :], self.L2[bk]).squeeze(1) + self.b2[bk]).clamp(0, 1)
            out = out + (h2 * self.L3[bk]).sum(1)
        if self.head == 'loop':
            u = torch.cat([xo, xp], 1) @ self.Li + self.be[bk]
            for _ in range(3):
                u = u + F.relu(u @ self.A1 + self.a1) @ self.A2
            out = out + (u * self.Lo[bk]).sum(1)
        return out


def run(name, hard=False, seed=0, **kw):
    torch.manual_seed(seed); net = Net(**kw).to(DEV); ema = Net(**kw).to(DEV); ema.load_state_dict(net.state_dict())
    groups = [{'params': [p for n_, p in net.named_parameters() if n_ != 'D']}] + ([{'params': [net.D], 'weight_decay': 0.0}] if kw.get('kb') else [])
    opt = torch.optim.Adam(groups, lr=1e-3); gen = torch.Generator(device='cpu'); gen.manual_seed(1 + seed); t0 = time.time()
    for step in range(STEPS):
        lr = 1e-3 * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * step / STEPS))) * min(1.0, (step + 1) / 200)
        for g in opt.param_groups:
            g['lr'] = lr
        k = torch.randint(0, N, (BS,), generator=gen).to(DEV); idx = decode_torch(X[k]).long(); y = Y[k].float() / 255
        o = net(idx); l = F.binary_cross_entropy_with_logits(o, y, reduction='none')
        if hard:                                                               # weight by the current error (mean weight 1)
            w = (torch.sigmoid(o).detach() - y).abs() + 0.03; l = l * (w / w.mean())
        loss = l.mean() + (1e-6 * (net.D ** 2).sum() if kw.get('kb') else 0.0)     # the corrections pay for their size: what regions share stays in the base table
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        with torch.no_grad():
            d = min(0.999, (1 + step) / (10 + step))
            for pe, pm in zip(ema.parameters(), net.parameters()):
                pe.lerp_(pm, 1 - d)
    dt = time.time() - t0
    with torch.no_grad():
        ho = torch.cat([ema(decode_torch(HX[j:j + 8192]).long()) for j in range(0, len(HX), 8192)]); hl = float(F.binary_cross_entropy_with_logits(ho, HY.float() / 255))
        vo = torch.cat([ema(VX[j:j + 8192]) for j in range(0, len(VX), 8192)]); vl = float(F.binary_cross_entropy_with_logits(vo, VY))
    npar = sum(p.numel() for p in net.parameters())
    print(f'{name:34s} parameters {npar/1e3:7.0f}k   held-out label loss {hl:.5f}   played positions (half score, half result) {vl:.5f}   {STEPS * BS / dt / 1e3:.0f}k positions/s', flush=True)


print(f'{N/1e6:.0f}M training positions from {len(TR)} shards, 1.0M held out from another shard, {len(VX)} played positions; {STEPS} steps of {BS}; every variant from scratch, same order', flush=True)
VAR = [('plain 384', dict(H=384)), ('plain 768', dict(H=768)), ('attack maps 384', dict(H=384, am=True)), ('threats 384', dict(H=384, thr=True)),
       ('rule graph 384+64', dict(H=384, rel=64)), ('layered head 384', dict(H=384, head='mlp')), ('looped head 384', dict(H=384, head='loop')),
       ('mirrored 384', dict(H=384, hm=True)), ('king base+delta 384', dict(H=384, kb=True)), ('hard positions 384', dict(H=384, hard=True)),
       ('other seed 384', dict(H=384, seed=1)), ('threats + attack maps 384', dict(H=384, thr=True, am=True)), ('attack maps + layered head 384', dict(H=384, am=True, head='mlp')),
       ('attack maps 256', dict(H=256, am=True)), ('rule graph 384+128', dict(H=384, rel=128)), ('attack maps + rule graph 384+64', dict(H=384, am=True, rel=64)),
       ('threats 256', dict(H=256, thr=True)), ('plain 1536', dict(H=1536))]
for name, kw in VAR:
    if WHICH is None or any(wh in name for wh in WHICH):
        run(name, **kw)
