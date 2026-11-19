"""Which operations dominate an OURO training step? (python3 zero/ouro_prof2.py <records .o>)"""
import sys, time, numpy as np, torch
sys.path.insert(0, '.')
from model import Ouro, make_optimizers
import importlib.util
spec = importlib.util.spec_from_file_location('og', 'zero/ouro_gpu.py'); og = importlib.util.module_from_spec(spec); spec.loader.exec_module(og)
from torch.profiler import profile, ProfilerActivity
DEV = torch.device('cuda'); r = np.fromfile(sys.argv[1], dtype=og.OREC); buf = og.Buffer(len(r)); buf.add(r)
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
model = Ouro(**og.CFG).to(DEV); opts = make_optimizers(model, 0.02, 3e-3, use_muon=True); sync = torch.cuda.synchronize
def step(loops, bs=1024):
    b = buf.batch(np.random.randint(0, buf.n, bs)); l = og.losses(model, b, loops); loss = l[0] + l[1] + l[2]
    for o in opts: o.zero_grad(set_to_none=True)
    loss.backward()
    for o in opts: o.step()
for i in range(4): step(1 + i % 4)
# stage timing inside the forward pass
board, castle, ep = buf.batch(np.random.randint(0, buf.n, 1024))[:3]
with torch.no_grad():
    sync(); t0 = time.time()
    for _ in range(5): ctx = model.prepare(board, castle, ep)
    sync(); t1 = time.time(); s = None
    for _ in range(5): s = model.loop(ctx, None)
    sync(); t2 = time.time()
    for _ in range(5): rel = model.graph(board, castle, ep)
    sync(); t3 = time.time()
print(f'forward stages at batch 1024, no gradients: prepare (rule graph + input blocks) {200*(t1-t0):.1f} ms, of which the rule graph alone {200*(t3-t2):.1f} ms; one pass of the looped blocks {200*(t2-t1):.1f} ms')
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for i in range(4): step(1 + i % 4)
    sync()
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=14, max_name_column_width=44)[:5200])
