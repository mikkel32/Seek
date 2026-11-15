"""Does compiling the model speed up an OURO training step? Plain and compiled, measured alternately so other GPU jobs
disturb both alike. (python3 zero/ouro_prof3.py <records .o>)"""
import sys, time, numpy as np, torch
sys.path.insert(0, '.')
from model import Ouro, make_optimizers
import importlib.util
spec = importlib.util.spec_from_file_location('og', 'zero/ouro_gpu.py'); og = importlib.util.module_from_spec(spec); spec.loader.exec_module(og)
DEV = torch.device('cuda'); r = np.fromfile(sys.argv[1], dtype=og.OREC); buf = og.Buffer(len(r)); buf.add(r); sync = torch.cuda.synchronize
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True


def make(compiled, mode=None):
    model = Ouro(**og.CFG).to(DEV); opts = make_optimizers(model, 0.02, 3e-3, use_muon=True)
    fwd = torch.compile(model, mode=mode) if compiled else model
    def step(i, bs=1024):
        b = buf.batch(np.random.randint(0, buf.n, bs)); l = og.losses(fwd, b, 1 + i % 4); loss = l[0] + l[1] + l[2]
        for o in opts: o.zero_grad(set_to_none=True)
        loss.backward()
        for o in opts: o.step()
        return float(loss)
    return step


variants = {'plain': make(False), 'compiled': make(True)}
for name, st in variants.items():
    t0 = time.time()
    for i in range(8): st(i)                       # warm-up: every loop count seen twice (the compiled model builds its four versions here)
    sync(); print(f'{name}: warm-up {time.time() - t0:.0f}s', flush=True)
res = {k: [] for k in variants}
for rnd in range(4):
    for name, st in variants.items():
        sync(); t0 = time.time()
        for i in range(16): st(i)
        sync(); res[name].append(1024 * 16 / (time.time() - t0))
for name, v in res.items():
    print(f'{name:9s}: {np.median(v):7.0f} positions/s (rounds: {" ".join(f"{x:.0f}" for x in v)})')
