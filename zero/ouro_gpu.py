"""OURO on the GPU: the transformer learns continuously from the calculator's own self-play.

Every position the calculator plays is written with the move its search chose, the search's score and, at the end of
the game, the result. OURO is trained to predict that move (policy), the outcome (value) and the value of the chosen
move (Q). This is the AlphaZero recipe with the fast calculator as the search; nothing from any other engine is used.

  python3 zero/ouro_gpu.py            (runs until zero/OURO_STOP exists; safe to stop and start again)
"""
import glob, json, math, os, sys, time
import numpy as np, torch, torch.nn.functional as F

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, '.')
from model import Ouro, make_optimizers

E = os.environ.get
DEV = torch.device(E('DEVICE') or ('cuda' if torch.cuda.is_available() else 'cpu'))
BS = int(E('OURO_BS', 1024)); CAP = int(E('OURO_CAP', 20_000_000)); LAM = float(E('LAM', 0.5))
LR = float(E('OURO_LR', 0.5)); MIN_FILL = int(E('OURO_MIN', 300_000)); MAX_REUSE = float(E('OURO_REUSE', 4.0))
CFG = dict(d=160, heads=5, mlp=480, n_in=2, n_rec=3, n_out=2, max_loops=4)
OREC = np.dtype([('board', 'u1', 64), ('castle', 'u1'), ('ep', 'u1'), ('move', '<u2'), ('score', '<i2'), ('result', 'i1'), ('ply', 'u1')])
torch.set_num_threads(int(E('OURO_THREADS', 2)))
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True     # the GPU's faster 32-bit matrix mode: +50% measured, negligible precision loss
LOG = open('zero/ouro.log', 'a')


def log(msg):
    line = f'[{time.strftime("%H:%M:%S", time.gmtime())}] {msg}'
    print(line, flush=True); LOG.write(line + '\n'); LOG.flush()


class Buffer:
    """ring buffer of positions in memory"""
    def __init__(self, cap):
        self.cap, self.n, self.w = cap, 0, 0
        self.d = np.zeros(cap, dtype=OREC)

    def add(self, r):
        r = r[-self.cap:]
        k = len(r); end = self.w + k
        if end <= self.cap:
            self.d[self.w:end] = r
        else:
            self.d[self.w:] = r[:self.cap - self.w]; self.d[:end - self.cap] = r[self.cap - self.w:]
        self.w = end % self.cap; self.n = min(self.cap, self.n + k)

    def batch(self, idx):
        q = self.d[idx]
        board = torch.from_numpy(q['board'].astype(np.int64)).to(DEV)
        castle = torch.from_numpy(q['castle'].astype(np.int64)).to(DEV); ep = torch.from_numpy(q['ep'].astype(np.int64)).to(DEV)
        mv = torch.from_numpy(q['move'].astype(np.int64)).to(DEV)
        wp = torch.sigmoid(torch.from_numpy(q['score'].astype(np.float32)).to(DEV) / 271.58)
        res = (torch.from_numpy(q['result'].astype(np.float32)).to(DEV) + 1) / 2
        return board, castle, ep, mv, LAM * wp + (1 - LAM) * res, wp


def losses(model, b, loops):
    board, castle, ep, mv, vt, qt = b
    pol, v, ql, mask = model(board, castle, ep, loops=loops)
    mask = mask.scatter(1, mv[:, None], True)                       # the searched move is legal by construction
    lp = F.log_softmax(pol.float().masked_fill(~mask, -1e9), 1)
    l_pol = -lp.gather(1, mv[:, None]).mean()
    l_val = F.binary_cross_entropy_with_logits(v.float(), vt)
    l_q = F.binary_cross_entropy_with_logits(ql.float().gather(1, mv[:, None]).squeeze(1), qt)
    return l_pol, l_val, l_q, (lp.argmax(1) == mv).float().mean()


if __name__ == '__main__':
    os.makedirs('zero/opool', exist_ok=True)
    model = Ouro(**CFG).to(DEV); ema = Ouro(**CFG).to(DEV); ema.load_state_dict(model.state_dict())
    opts = make_optimizers(model, 0.02, 3e-3, use_muon=True)
    for o in opts:
        for g in o.param_groups:
            g['base_lr'] = g['lr']
    step, seen, consumed = 0, 0, 0
    if os.path.exists('zero/ouro_train.pt'):
        ck = torch.load('zero/ouro_train.pt', map_location=DEV)
        model.load_state_dict(ck['model']); ema.load_state_dict(ck['ema'])
        for o, s_ in zip(opts, ck['opts']):
            o.load_state_dict(s_)
        step, seen, consumed = ck['step'], ck['seen'], ck.get('consumed', 0)
    buf, val = Buffer(CAP), Buffer(200_000)
    log(f'OURO trainer on {DEV}: {sum(p.numel() for p in model.parameters()):,} parameters, batch {BS}, resumed at step {step}')
    t0 = time.time(); acc = np.zeros(4); nacc = 0; nfile = 0; seen0 = seen
    while not os.path.exists('zero/OURO_STOP'):
        files = sorted(glob.glob('zero/opool/*.o'))
        for f in files[:400]:                                       # take in what self-play has produced since last time
            r = np.fromfile(f, dtype=OREC); os.remove(f); nfile += 1
            if len(r) == 0:
                continue
            (val if nfile % 50 == 0 else buf).add(r)                # one file in fifty is kept out of training, whole, for validation
            if nfile % 50:
                consumed += len(r)
        if buf.n < MIN_FILL or seen > MAX_REUSE * max(consumed, 1):   # wait for data: never chew the same positions too many times
            time.sleep(10); continue
        loops = 1 + (step * 7919) % CFG['max_loops']
        b = buf.batch(np.random.randint(0, buf.n, BS))
        l_pol, l_val, l_q, top1 = losses(model, b, loops)
        loss = l_pol + l_val + l_q
        for o in opts:
            o.zero_grad(set_to_none=True)
        loss.backward()
        m = LR * min(1.0, (step + 1) / 300)
        for o in opts:
            for g in o.param_groups:
                g['lr'] = g['base_lr'] * m
            o.step()
        step += 1; seen += BS
        with torch.no_grad():
            d = min(0.999, (1 + step) / (10 + step))
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - d)
        acc += [l_pol.item(), l_val.item(), l_q.item(), top1.item()]; nacc += 1
        if step % 500 == 0:
            msg = (f'step {step} seen {seen/1e6:.2f}M (pool {buf.n/1e6:.2f}M, reuse {seen/max(consumed,1):.1f}x) policy {acc[0]/nacc:.3f} value {acc[1]/nacc:.3f} '
                   f'q {acc[2]/nacc:.3f} move-match {acc[3]/nacc:.3f} | {(seen - seen0)/(time.time()-t0):.0f} positions/s')
            acc[:] = 0; nacc = 0
            if val.n >= 20_000 and step % 2000 == 0:
                ema.eval()
                with torch.no_grad():
                    rs = np.random.RandomState(0); vi = rs.randint(0, val.n, 8192); tot = np.zeros(4)
                    for j in range(0, len(vi), 1024):
                        tot += [x.item() for x in losses(ema, val.batch(vi[j:j + 1024]), 2)]
                    tot /= math.ceil(len(vi) / 1024)
                msg += f' || held-out games: policy {tot[0]:.3f} value {tot[1]:.3f} move-match {tot[3]:.3f}'
                json.dump({'step': step, 'seen': seen, 'val_policy': tot[0], 'val_value': tot[1], 'val_top1': tot[3]}, open('zero/ouro_val.json', 'w'))
            log(msg)
        if step % 1000 == 0:
            torch.save(dict(cfg=CFG, ema={k: v.cpu() for k, v in ema.state_dict().items()}, beta=4.0, step=step, seen=seen), 'zero/ouro.pt.tmp')
            os.replace('zero/ouro.pt.tmp', 'zero/ouro.pt')
        if step % 5000 == 0:
            torch.save(dict(model=model.state_dict(), ema=ema.state_dict(), opts=[o.state_dict() for o in opts], step=step, seen=seen, consumed=consumed), 'zero/ouro_train.pt.tmp')
            os.replace('zero/ouro_train.pt.tmp', 'zero/ouro_train.pt')
    log('OURO trainer stopped')
