"""Would the calculator's network judge better if it could see which pieces are attacked and defended?

Today its inputs say only where each piece stands. Here every piece also carries a status computed exactly from the rules:
    attacked by an enemy piece / defended by an own piece / attacked by an enemy piece of lower value
and the input "piece P on square S" becomes "piece P on square S with status T". Same network otherwise, same positions,
same targets, same number of training steps. Reported: log-loss on held-out positions against the training target, and
on played positions against the game result and against the search score. No engine support is needed for this
measurement; it says whether building that support is worth it.

  python3 zero/status_exp.py <tag> [hidden=256] [steps=6000]        reads data/ev_<tag>_{x,y,s,pl}.npy
"""
import math, sys, time, numpy as np, torch, torch.nn.functional as F

tag = sys.argv[1]; H = int(sys.argv[2]) if len(sys.argv) > 2 else 256; STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 6000
DEV = torch.device('cuda'); BS = 16384; torch.backends.cuda.matmul.allow_tf32 = True; torch.manual_seed(0)

# ---- rule tables in the mover's view: codes 0-5 the mover's P N B R Q K (its pawns move up the board), 6-11 the opponent's
ATT = np.zeros((13, 64, 64), bool); BETW = np.zeros((64, 64, 64), bool)
for a in range(64):
    ar, af = divmod(a, 8)
    for b in range(64):
        if a == b:
            continue
        br, bf = divmod(b, 8); dr, df = br - ar, bf - af
        diag = abs(dr) == abs(df); line = dr == 0 or df == 0
        ATT[0, a, b] = dr == 1 and abs(df) == 1; ATT[6, a, b] = dr == -1 and abs(df) == 1
        kn = (abs(dr), abs(df)) in ((1, 2), (2, 1)); kg = max(abs(dr), abs(df)) == 1
        for c in (1, 7): ATT[c, a, b] = kn
        for c in (2, 8): ATT[c, a, b] = diag
        for c in (3, 9): ATT[c, a, b] = line
        for c in (4, 10): ATT[c, a, b] = diag or line
        for c in (5, 11): ATT[c, a, b] = kg
        if diag or line:
            sr, sf = (dr > 0) - (dr < 0), (df > 0) - (df < 0); r, f = ar + sr, af + sf
            while (r, f) != (br, bf):
                BETW[a, b, r * 8 + f] = True; r += sr; f += sf
ATT_T = torch.from_numpy(ATT.reshape(-1)).to(DEV); BETW_T = torch.from_numpy(BETW.reshape(4096, 64)).to(DEV)
VAL_T = torch.tensor([1, 3, 3, 5, 9, 100] * 2 + [0], device=DEV)


@torch.no_grad()
def status(idx):
    """(B,32) feature lists -> (B,32) status per piece: bit 0 attacked, bit 1 defended, bit 2 attacked by a cheaper piece"""
    valid = idx < 768; code = torch.where(valid, idx // 64, torch.full_like(idx, 12)); sq = torch.where(valid, idx % 64, torch.zeros_like(idx))
    occ = torch.zeros(idx.shape[0], 65, dtype=torch.bool, device=idx.device).scatter_(1, torch.where(valid, sq, torch.full_like(sq, 64)), True)[:, :64]
    geo = ATT_T[code[:, :, None] * 4096 + sq[:, :, None] * 64 + sq[:, None, :]]                               # attacker i -> victim j on an empty board
    blocked = (BETW_T[sq[:, :, None] * 64 + sq[:, None, :]] & occ[:, None, None, :]).any(-1)
    att = geo & ~blocked & valid[:, :, None] & valid[:, None, :]
    own = code < 6; enemy = own[:, :, None] != own[:, None, :]
    attacked = (att & enemy).any(1); defended = (att & ~enemy).any(1)
    cheaper = (att & enemy & (VAL_T[code][:, :, None] < VAL_T[code][:, None, :])).any(1)
    return (attacked.long() + 2 * defended.long() + 4 * cheaper.long()) * valid


def check(x, n=1500):
    """the status tensor against python-chess on real positions"""
    import chess
    rs = np.random.RandomState(1); sel = rs.randint(0, len(x), n); idx = torch.from_numpy(x[np.sort(sel)].astype(np.int64)).to(DEV)
    st = status(idx).cpu().numpy(); idx = idx.cpu().numpy(); bad = tot = 0
    for k in range(n):
        b = chess.Board(None)
        for v in idx[k]:
            if v < 768:
                c, s = divmod(int(v), 64); b.set_piece_at(s, chess.Piece(c % 6 + 1, c < 6))
        vals = {1: 1, 2: 3, 3: 3, 4: 5, 5: 9, 6: 100}
        for j, v in enumerate(idx[k]):
            if v >= 768:
                continue
            c, s = divmod(int(v), 64); col = c < 6
            atk = b.attackers(not col, s); a = bool(atk); d = bool(b.attackers(col, s))
            ch = any(vals[b.piece_type_at(q)] < vals[c % 6 + 1] for q in atk)
            tot += 1; bad += int(st[k, j]) != a + 2 * d + 4 * ch
    return bad, tot


x = np.load(f'data/ev_{tag}_x.npy', mmap_mode='r'); N = len(x)
y = torch.from_numpy(np.load(f'data/ev_{tag}_y.npy')).to(DEV); s8 = torch.from_numpy(np.load(f'data/ev_{tag}_s.npy')).to(DEV); pl = torch.from_numpy(np.load(f'data/ev_{tag}_pl.npy')).to(DEV) > 0
bad, tot = check(x); print(f'status check against python-chess: {bad} mismatches in {tot} pieces', flush=True)
assert bad == 0
t0 = time.time(); X = torch.from_numpy(np.ascontiguousarray(x)).to(DEV); ST = torch.zeros(N, 32, dtype=torch.uint8, device=DEV)
for j in range(0, N, 8192):
    ST[j:j + 8192] = status(X[j:j + 8192].long()).to(torch.uint8)
print(f'status computed for {N/1e6:.1f}M positions in {time.time() - t0:.0f}s; share of pieces attacked {float((ST & 1).float().sum() / (X < 768).sum()):.3f}, '
      f'defended {float(((ST >> 1) & 1).float().sum() / (X < 768).sum()):.3f}, attacked by a cheaper piece {float(((ST >> 2) & 1).float().sum() / (X < 768).sum()):.3f}', flush=True)
NV = 500_000; NT = N - NV
i = np.arange(768); LUT = torch.from_numpy(np.concatenate([((i // 64 + 6) % 12) * 64 + ((i % 64) ^ 56), [768]]).astype(np.int64)).to(DEV)
PT_W = torch.cat([torch.tensor([0, 1, 1, 2, 4, 0] * 2, dtype=torch.float32).repeat_interleave(64), torch.zeros(1)]).to(DEV)
pst = np.load('zero/fss.npz')


class Net(torch.nn.Module):
    def __init__(self, mask):
        super().__init__()
        self.mask = mask; self.S_ = mask + 1 if mask else 1                      # number of statuses a piece can show
        self.E = torch.nn.Parameter(torch.randn(self.S_ * 768, H) * 0.05); self.b = torch.nn.Parameter(torch.full((H,), 0.1))
        self.wo = torch.nn.Parameter(torch.randn(8, H) / math.sqrt(H)); self.wp = torch.nn.Parameter(torch.randn(8, H) / math.sqrt(H)); self.cb = torch.nn.Parameter(torch.zeros(8))
        self.S = torch.nn.Parameter(torch.stack([torch.from_numpy(pst['S_mg']), torch.from_numpy(pst['S_eg'])], 1).float())
        self.c = torch.nn.Parameter(torch.tensor([float(pst['c_mg']), float(pst['c_eg'])]))

    def forward(self, idx, st):
        B = idx.shape[0]; D = self.S_ * 768; one = torch.ones(idx.shape, device=idx.device); pad = idx == 768
        t = (st & self.mask) * 768
        fo = torch.where(pad, torch.full_like(idx, D), t + idx); fp = torch.where(pad, torch.full_like(idx, D), t + LUT[idx])
        Mo = torch.zeros(B, D + 1, device=idx.device).scatter_add_(1, fo, one)[:, :D]; Mp = torch.zeros(B, D + 1, device=idx.device).scatter_add_(1, fp, one)[:, :D]
        P = torch.zeros(B, 769, device=idx.device).scatter_add_(1, idx, one)[:, :768]
        bk = ((P.sum(1) - 2) // 4).clamp(0, 7).long()
        xo = (self.b + Mo @ self.E).clamp(0, 1) ** 2; xp = (self.b + Mp @ self.E).clamp(0, 1) ** 2
        lin = self.c + P @ self.S; ph = ((P @ PT_W[:768]) / 24).clamp(max=1.0)
        heads = xo @ self.wo.T + xp @ self.wp.T + self.cb
        return ph * lin[:, 0] + (1 - ph) * lin[:, 1] + (heads * F.one_hot(bk, 8).float()).sum(1)


def run(name, mask):
    torch.manual_seed(0); net = Net(mask).to(DEV); ema = Net(mask).to(DEV); ema.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, fused=True); fwd = torch.compile(net); t0 = time.time()
    for step in range(STEPS):
        lr = 1e-3 * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * step / STEPS))) * min(1.0, (step + 1) / 200)
        for g in opt.param_groups:
            g['lr'] = lr
        k = torch.randint(0, NT, (BS,), device=DEV)
        loss = F.binary_cross_entropy_with_logits(fwd(X[k].long(), ST[k].long()), y[k].float() / 255)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        with torch.no_grad():
            d = min(0.999, (1 + step) / (10 + step))
            for pe, pm in zip(ema.parameters(), net.parameters()):
                pe.lerp_(pm, 1 - d)
    with torch.no_grad():
        out = torch.cat([ema(X[j:j + 20000].long(), ST[j:j + 20000].long()) for j in range(NT, N, 20000)])
        yt = y[NT:].float() / 255; wp = s8[NT:].float() / 255; p = pl[NT:]; z = (2 * yt - wp).clamp(0, 1)
        bce = lambda o, t: float(F.binary_cross_entropy_with_logits(o, t))
        # the result needs a rescaling to be judged fairly: one slope and offset fitted on half of the played positions, tested on the other half
        o = out[p].double(); zz = z[p].double(); h = len(o) // 2; w = torch.zeros(2, dtype=torch.float64, device=DEV); Xm = torch.stack([o, torch.ones_like(o)], 1)
        for _ in range(25):
            q = torch.sigmoid(Xm[:h] @ w); w = w - torch.linalg.solve((Xm[:h] * (q * (1 - q))[:, None]).T @ Xm[:h] + 1e-6 * torch.eye(2, dtype=torch.float64, device=DEV), Xm[:h].T @ (q - zz[:h]))
        res = float(F.binary_cross_entropy_with_logits(Xm[h:] @ w, zz[h:]))
    print(f'{name:44s} inputs {net.S_ * 768:5d}   target {bce(out, yt):.5f}   played: target {bce(out[p], yt[p]):.5f}  search score {bce(out[p], wp[p]):.5f}  result (rescaled) {res:.5f}   '
          f'{STEPS * BS / (time.time() - t0) / 1e6:.1f}M positions/s', flush=True)


print(f'{NT/1e6:.1f}M training positions, {NV} held out ({int(pl[NT:].sum())} of them played), hidden {H}, {STEPS} steps; lower is better in every column')
run('piece positions only (today)', 0)
run('+ attacked', 1)
run('+ attacked, defended', 3)
run('+ attacked, defended, attacked by cheaper', 7)
