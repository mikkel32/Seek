"""Where does a step of the calculator-network trainer go? (python3 zero/nnue_prof.py <shard tag> [hidden=384])"""
import sys, time, math, numpy as np, torch, torch.nn.functional as F
tag = sys.argv[1]; H = int(sys.argv[2]) if len(sys.argv) > 2 else 384; DEV = torch.device('cuda'); BS = 16384
x = np.load(f'data/ev_{tag}_x.npy', mmap_mode='r')[:4_000_000]; y = np.load(f'data/ev_{tag}_y.npy', mmap_mode='r')[:4_000_000]
X = torch.from_numpy(np.ascontiguousarray(x)).to(DEV); Y = torch.from_numpy(np.ascontiguousarray(y)).to(DEV)
i = np.arange(768); LUT = torch.from_numpy(np.concatenate([((i // 64 + 6) % 12) * 64 + ((i % 64) ^ 56), [768]]).astype(np.int64)).to(DEV)
PT_W = torch.cat([torch.tensor([0, 1, 1, 2, 4, 0] * 2, dtype=torch.float32).repeat_interleave(64), torch.zeros(1)]).to(DEV)


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.E = torch.nn.Parameter(torch.randn(769, H) * 0.05); self.b = torch.nn.Parameter(torch.full((H,), 0.1))
        self.wo = torch.nn.Parameter(torch.randn(8, H) / math.sqrt(H)); self.wp = torch.nn.Parameter(torch.randn(8, H) / math.sqrt(H)); self.cb = torch.nn.Parameter(torch.zeros(8))
        self.S = torch.nn.Parameter(torch.zeros(769, 2)); self.c = torch.nn.Parameter(torch.zeros(2))

    def forward(self, idx):
        if DENSE:                                                     # the same sums as a matrix product with a (batch, 768) piece-count matrix
            bk = (((idx < 768).sum(1) - 2) // 4).clamp(0, 7); one = torch.ones(idx.shape, device=idx.device)
            Mo = torch.zeros(idx.shape[0], 769, device=idx.device).scatter_add_(1, idx, one)[:, :768]
            Mp = torch.zeros(idx.shape[0], 769, device=idx.device).scatter_add_(1, LUT[idx], one)[:, :768]
            ao = self.b + Mo @ self.E[:768]; ap = self.b + Mp @ self.E[:768]; lin = self.c + Mo @ self.S[:768]; ph = ((Mo @ PT_W[:768]) / 24).clamp(max=1.0)
            return ph * lin[:, 0] + (1 - ph) * lin[:, 1] + self.cb[bk] + (ao.clamp(0, 1) ** 2 * self.wo[bk]).sum(1) + (ap.clamp(0, 1) ** 2 * self.wp[bk]).sum(1)
        bk = (((idx < 768).sum(1) - 2) // 4).clamp(0, 7)
        ao = self.b + F.embedding_bag(idx, self.E, mode='sum', padding_idx=768); ap = self.b + F.embedding_bag(LUT[idx], self.E, mode='sum', padding_idx=768)
        lin = self.c + F.embedding_bag(idx, self.S, mode='sum', padding_idx=768); ph = (PT_W[idx].sum(1) / 24).clamp(max=1.0)
        return ph * lin[:, 0] + (1 - ph) * lin[:, 1] + self.cb[bk] + (ao.clamp(0, 1) ** 2 * self.wo[bk]).sum(1) + (ap.clamp(0, 1) ** 2 * self.wp[bk]).sum(1)


sync = torch.cuda.synchronize; DENSE = False
torch.backends.cuda.matmul.allow_tf32 = True


def run(name, bs=BS, ema_every=1, fused=False, compiled=False, sync_loss=True, steps=120):
    net = Net().to(DEV); ema = Net().to(DEV); ema.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, fused=fused); fwd = torch.compile(net) if compiled else net
    perm = torch.randperm(len(X), device=DEV); T = dict(index=0.0, forward=0.0, backward=0.0, optimizer=0.0, ema=0.0, loss_read=0.0)
    for s in range(steps + 20):
        k = perm[(s * bs) % (len(X) - bs):][:bs]
        sync(); t0 = time.time(); xb = X[k].long(); yb = Y[k].float() / 255; sync(); t1 = time.time()
        loss = F.binary_cross_entropy_with_logits(fwd(xb), yb); sync(); t2 = time.time()
        opt.zero_grad(set_to_none=True); loss.backward(); sync(); t3 = time.time()
        opt.step(); sync(); t4 = time.time()
        if s % ema_every == 0:
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), net.parameters()):
                    pe.lerp_(pm, 0.001)
        sync(); t5 = time.time()
        if sync_loss:
            float(loss)
        t6 = time.time()
        if s >= 20:
            for kk, v in zip(T, (t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4, t6 - t5)):
                T[kk] += v
    tot = sum(T.values())
    print(f'{name:34s} {bs * steps / tot / 1e6:5.2f}M positions/s  {1000 * tot / steps:5.1f} ms/step   ' + '  '.join(f'{kk} {100 * v / tot:.0f}%' for kk, v in T.items()), flush=True)


# the two formulations must compute the same function
torch.manual_seed(0); n0 = Net().to(DEV); xb = X[:4096].long()
DENSE = False; a = n0(xb); DENSE = True; b = n0(xb); print('largest difference between the two formulations:', float((a - b).abs().max()))
DENSE = False; run('as now: scatter (batch 16384)')
DENSE = True; run('dense product'); run('dense product + fused Adam', fused=True); run('dense + fused + compiled', fused=True, compiled=True)
