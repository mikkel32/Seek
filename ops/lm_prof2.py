import sys, time, torch, numpy as np
sys.path.insert(0, '/root/lmtest')
from model import Ouro, make_optimizers
dev = torch.device('cuda'); torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.benchmark = True
B = int(sys.argv[1]); mode = sys.argv[2]
m = Ouro(d=256, heads=8, mlp=768, n_in=2, n_rec=4, n_out=2, max_loops=2).to(dev); opts = make_optimizers(m)
b = np.load('/root/loopchess/data/lm_0_b.npy', mmap_mode='r')[:B]; meta = np.load('/root/loopchess/data/lm_0_m.npy', mmap_mode='r')[:B]
board = torch.from_numpy(np.ascontiguousarray(b)).to(dev).long(); castle = torch.from_numpy(np.ascontiguousarray(meta[:, 0])).to(dev).long(); ep = torch.from_numpy(np.ascontiguousarray(meta[:, 1])).to(dev).long()
if mode == 'graph': m.graph = torch.compile(m.graph)
if mode == 'all': fm = torch.compile(m)
else: fm = m
def full():
    pol, v, ql, mask, pvl = fm(board, castle, ep, loops=1, aux=True); loss = pol.logsumexp(1).mean() + v.mean() + ql.mean() + pvl.logsumexp(-1).mean()
    for o in opts: o.zero_grad(set_to_none=True)
    loss.backward()
    for o in opts: o.step()
    return loss
t0 = time.time(); full(); full(); torch.cuda.synchronize(); tc = time.time() - t0
torch.cuda.reset_peak_memory_stats(); t0 = time.time()
for _ in range(8): full()
torch.cuda.synchronize(); dt = (time.time() - t0) / 8
print(f'{mode}, batch {B}: {1000 * dt:.0f} ms a step = {B / dt:.0f} positions/s; peak memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GB; first two steps {tc:.0f}s')
