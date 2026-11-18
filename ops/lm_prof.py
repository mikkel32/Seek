import sys, time, torch, numpy as np
sys.path.insert(0, '/root/loopchess')
from model import Ouro, make_optimizers
dev = torch.device('cuda'); torch.backends.cuda.matmul.allow_tf32 = True
B = 512
m = Ouro(d=256, heads=8, mlp=768, n_in=2, n_rec=4, n_out=2, max_loops=2).to(dev); opts = make_optimizers(m)
b = np.load('/root/loopchess/data/lm_0_b.npy', mmap_mode='r')[:B]; meta = np.load('/root/loopchess/data/lm_0_m.npy', mmap_mode='r')[:B]
board = torch.from_numpy(np.ascontiguousarray(b)).to(dev).long(); castle = torch.from_numpy(np.ascontiguousarray(meta[:, 0])).to(dev).long(); ep = torch.from_numpy(np.ascontiguousarray(meta[:, 1])).to(dev).long()
def T(f, n=10):
    torch.cuda.synchronize(); t = time.time()
    for _ in range(n): r = f()
    torch.cuda.synchronize(); return 1000 * (time.time() - t) / n, r
t_graph, g = T(lambda: m.graph(board, castle, ep))
t_prep, ctx = T(lambda: m.prepare(board, castle, ep))
t_loop, s = T(lambda: m.loop(ctx))
t_read, _ = T(lambda: m.readout(ctx, s, aux=True))
def full():
    pol, v, ql, mask, pvl = m(board, castle, ep, loops=1, aux=True); loss = pol.logsumexp(1).mean() + v.mean() + ql.mean() + pvl.logsumexp(-1).mean()
    for o in opts: o.zero_grad(set_to_none=True)
    loss.backward(); return loss
t_fb, _ = T(full, 5)
def step():
    for o in opts: o.step()
t_opt, _ = T(step, 5)
print(f'batch {B}: rule graph {t_graph:.1f} ms; prepare (graph + {len(m.inp)} blocks) {t_prep:.1f}; one loop of {len(m.rec)} blocks {t_loop:.1f}; read-out {t_read:.1f}; forward + backward {t_fb:.1f}; optimiser step {t_opt:.1f}  -> {1000 * B / (t_fb + t_opt):.0f} positions/s alone')
