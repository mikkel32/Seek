"""Where does an OURO training step spend its time? (python3 zero/ouro_prof.py <file of self-play records .o>)"""
import sys, time, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, '.')
from model import Ouro, make_optimizers
import importlib.util
spec = importlib.util.spec_from_file_location('og', 'zero/ouro_gpu.py'); og = importlib.util.module_from_spec(spec); spec.loader.exec_module(og)
DEV = torch.device('cuda'); r = np.fromfile(sys.argv[1], dtype=og.OREC); print(len(r), 'positions')
buf = og.Buffer(len(r)); buf.add(r)
sync = torch.cuda.synchronize


def run(bs, steps=24, tf32=False, compiled=False):
    torch.backends.cuda.matmul.allow_tf32 = tf32; torch.backends.cudnn.allow_tf32 = tf32
    model = Ouro(**og.CFG).to(DEV); ema = Ouro(**og.CFG).to(DEV); ema.load_state_dict(model.state_dict())
    opts = make_optimizers(model, 0.02, 3e-3, use_muon=True)
    T = dict(data=0.0, forward=0.0, backward=0.0, optimizer=0.0, ema=0.0); torch.cuda.reset_peak_memory_stats()
    for step in range(steps + 4):
        loops = 1 + (step * 7919) % 4
        sync(); t0 = time.time()
        b = buf.batch(np.random.randint(0, buf.n, bs)); sync(); t1 = time.time()
        l_pol, l_val, l_q, top1 = og.losses(model, b, loops); loss = l_pol + l_val + l_q; sync(); t2 = time.time()
        for o in opts:
            o.zero_grad(set_to_none=True)
        loss.backward(); sync(); t3 = time.time()
        for o in opts:
            o.step()
        sync(); t4 = time.time()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 0.001)
        sync(); t5 = time.time()
        if step >= 4:                                         # the first steps include one-off set-up
            for k, v in zip(T, (t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4)):
                T[k] += v
    tot = sum(T.values())
    print(f'batch {bs:5d}{" tf32" if tf32 else ""}: {bs * steps / tot:7.0f} positions/s   {1000 * tot / steps:6.1f} ms per step   ' +
          '  '.join(f'{k} {100 * v / tot:.0f}%' for k, v in T.items()) + f'   peak memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GB', flush=True)


for bs in (1024, 2048, 3072):
    run(bs)
run(2048, tf32=True)
print('parameter tensors:', sum(1 for _ in Ouro(**og.CFG).parameters()))
