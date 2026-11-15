"""Three facts about the transformer that decide which speed-ups are worth building.

  1. What one evaluation costs on one CPU thread (1, 2 and 4 loops), next to the calculator's cost per position.
  2. How much each extra loop changes the answer, on fresh self-play positions: value loss and move-match after 1-4 loops.
  3. Loop-drift halting: stop after loop 2 when the value moved less than eps between loops 1 and 2, otherwise run all 4.
     How many loops that averages, and what it costs in accuracy.

  python3 zero/ouro_probe.py [checkpoint=zero/ouro_eval.pt] [positions=20000]
"""
import glob, os, sys, time
import numpy as np, torch, torch.nn.functional as F
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')); sys.path.insert(0, '.')
from model import Ouro

CK = sys.argv[1] if len(sys.argv) > 1 else 'zero/ouro_eval.pt'; N = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
OREC = np.dtype([('board', 'u1', 64), ('castle', 'u1'), ('ep', 'u1'), ('move', '<u2'), ('score', '<i2'), ('result', 'i1'), ('ply', 'u1')])
ck = torch.load(CK, map_location='cpu', weights_only=False); model = Ouro(**ck['cfg']); model.load_state_dict(ck['ema']); model.eval()
files = sorted(glob.glob('zero/opool/*.o'))[-12:]; r = np.concatenate([np.fromfile(f, dtype=OREC) for f in files])
r = r[np.random.RandomState(0).choice(len(r), min(N, len(r)), replace=False)]
T = lambda k, dev: torch.from_numpy(r[k].astype(np.int64)).to(dev)
print(f'{CK}: step {ck.get("step")}, {sum(p.numel() for p in model.parameters()):,} parameters; {len(r)} positions from {len(files)} fresh self-play files')

torch.set_num_threads(1)
with torch.no_grad():
    b, c, e = T('board', 'cpu'), T('castle', 'cpu'), T('ep', 'cpu')
    for amp in (False,):
        model.amp = amp
        for loops in (1, 2, 4):
            for bs in (1, 16):
                n = 0; t = time.time()
                while time.time() - t < 3.0:
                    i = (n * 7) % (len(r) - bs); ctx = model.prepare(b[i:i + bs], c[i:i + bs], e[i:i + bs]); s = None
                    for _ in range(loops):
                        s = model.loop(ctx, s)
                    model.readout(ctx, s); n += bs
                dt = time.time() - t
                print(f'one CPU thread, {loops} loop(s), {bs} position(s) at a time: {1000 * dt / n:.2f} ms per position = {n / dt:,.0f} positions per second')

dev = 'cuda' if torch.cuda.is_available() else 'cpu'; model.to(dev); model.amp = True
b, c, e, mv = T('board', dev), T('castle', dev), T('ep', dev), T('move', dev)
wp = torch.sigmoid(torch.from_numpy(r['score'].astype(np.float32)).to(dev) / 271.58); res = (torch.from_numpy(r['result'].astype(np.float32)).to(dev) + 1) / 2; vt = 0.5 * wp + 0.5 * res
V = []; top = []
with torch.no_grad():
    for j in range(0, len(r), 2048):
        ctx = model.prepare(b[j:j + 2048], c[j:j + 2048], e[j:j + 2048]); s = None; vv = []; tt = []
        for _ in range(4):
            s = model.loop(ctx, s); pol, v, ql = model.readout(ctx, s)
            mask = ctx['mask'].scatter(1, mv[j:j + 2048, None], True)
            vv.append(v.float()); tt.append(pol.float().masked_fill(~mask, -1e9).argmax(1) == mv[j:j + 2048])
        V.append(torch.stack(vv, 1)); top.append(torch.stack(tt, 1))
V = torch.cat(V); top = torch.cat(top).float(); P = torch.sigmoid(V)
bce = lambda v: float(F.binary_cross_entropy_with_logits(v, vt))
print('\nafter k loops:   value loss (against half search score, half result)   move-match with the search')
for k in range(4):
    print(f'  {k + 1} loop(s):      {bce(V[:, k]):.4f}                                              {float(top[:, k].mean()):.3f}')
d12 = (P[:, 1] - P[:, 0]).abs()
print(f'\nhow far the value (win probability) moves between loop 1 and loop 2: median {float(d12.median()):.4f}, 90th percentile {float(d12.quantile(0.9)):.4f}, 99th {float(d12.quantile(0.99)):.4f}')
print('halting at loop 2 when the value moved less than eps between loops 1 and 2 (otherwise 4 loops):')
for eps in (0.005, 0.01, 0.02, 0.05):
    halt = d12 < eps; v = torch.where(halt, V[:, 1], V[:, 3]); t = torch.where(halt, top[:, 1], top[:, 3])
    print(f'  eps {eps}: halts on {float(halt.float().mean()):.1%} of positions, {float(2 * halt.float().mean() + 4 * (1 - halt.float().mean())):.2f} loops on average, value loss {bce(v):.4f}, move-match {float(t.mean()):.3f}')
print(f'for reference, always 2 loops: {bce(V[:, 1]):.4f} / {float(top[:, 1].mean()):.3f};  always 4 loops: {bce(V[:, 3]):.4f} / {float(top[:, 3].mean()):.3f}')
