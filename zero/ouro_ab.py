"""A/B for changes to the transformer: every arm trains from scratch on the SAME positions, in the same order, for the
same number of steps, and is judged on the same held-out games.

  python3 zero/ouro_ab.py <arm name> <steps> [key=value ...]
     smolgen=128     Smolgen: attention biases generated from the whole position (model.py)
     aux=1           two training-only targets that do not depend on move order: who holds each square 8 plies from
                     now (read off each square's token), and the opponent's reply on the next ply
     d=192 heads=6 mlp=576   a larger plain network, to compare "more parameters" against "better parameters"

Data: the self-play files in zero/opool (read, not removed); one file in 25 is held out whole. Reports, every 2,000
steps on the held-out games: policy loss, value loss, move-match with the search (2 loops, averaged weights), and the
training speed in positions per second (what the change costs).
"""
import glob, json, math, os, sys, time
import numpy as np, torch, torch.nn.functional as F
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')); sys.path.insert(0, '.')
from model import Ouro, make_optimizers

ARM = sys.argv[1]; STEPS = int(sys.argv[2]); kv = dict(a.split('=') for a in sys.argv[3:])
AUX = int(kv.pop('aux', 0)); BS = 1024; DEV = torch.device('cuda'); MAXPOS = int(os.environ.get('AB_MAXPOS', 12_000_000))
CFG = dict(d=160, heads=5, mlp=480, n_in=2, n_rec=3, n_out=2, max_loops=4); CFG.update({k: int(v) for k, v in kv.items()})
if AUX:
    CFG['fut'] = 1
OREC = np.dtype([('board', 'u1', 64), ('castle', 'u1'), ('ep', 'u1'), ('move', '<u2'), ('score', '<i2'), ('result', 'i1'), ('ply', 'u1')])
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
LOG = open(f'zero/ouro_ab_{ARM}.log', 'a')


def log(msg):
    line = f'[{time.strftime("%H:%M:%S", time.gmtime())}] {msg}'; print(line, flush=True); LOG.write(line + '\n'); LOG.flush()


def load(files):
    r = np.concatenate([np.fromfile(f, dtype=OREC) for f in files])[:MAXPOS]; n = len(r); ply = r['ply'].astype(np.int64); mv = r['move'].astype(np.int64)
    # the opponent's reply: the next record's move (same game = the ply goes up by one), turned into this position's view
    nxt = np.full(n, -1, np.int64); ok = np.zeros(n, bool); ok[:-1] = (ply[1:] == ply[:-1] + 1) & (ply[:-1] < 254) & (mv[1:] < 4096)
    m1 = np.roll(mv, -1); nxt[ok] = ((m1[ok] // 64) ^ 56) * 64 + ((m1[ok] % 64) ^ 56)
    # who holds each square 8 plies on: record i+8 of the same game has the same side to move, so the same view
    fut = np.full(n, -1, np.int64); ok8 = np.zeros(n, bool); ok8[:-8] = (ply[8:] == ply[:-8] + 8) & (ply[:-8] < 240); fut[ok8] = np.nonzero(ok8)[0] + 8
    t = lambda a, dt: torch.from_numpy(np.ascontiguousarray(a)).to(DEV, dt)
    return dict(n=n, board=t(r['board'], torch.uint8), castle=t(r['castle'], torch.uint8), ep=t(r['ep'], torch.uint8), move=t(mv, torch.int64),
                wp=torch.sigmoid(t(r['score'], torch.float32) / 271.58), res=(t(r['result'], torch.float32) + 1) / 2, nxt=t(nxt, torch.int64), fut=t(fut, torch.int64))


def losses(model, D, idx, loops, aux):
    board = D['board'][idx].long(); mv = D['move'][idx]; vt = 0.5 * D['wp'][idx] + 0.5 * D['res'][idx]
    out = model(board, D['castle'][idx].long(), D['ep'][idx].long(), loops=loops, aux=aux)
    pol, v, ql, mask = out[:4]
    mask = mask.scatter(1, mv[:, None], True)
    lp = F.log_softmax(pol.float().masked_fill(~mask, -1e9), 1)
    l_pol = -lp.gather(1, mv[:, None]).mean(); l_val = F.binary_cross_entropy_with_logits(v.float(), vt)
    l_q = F.binary_cross_entropy_with_logits(ql.float().gather(1, mv[:, None]).squeeze(1), D['wp'][idx])
    l_aux = torch.zeros((), device=DEV)
    if aux:
        pvl, futl = out[4]; nx = D['nxt'][idx]; a = nx >= 0; fi = D['fut'][idx]; b = fi >= 0
        if bool(a.any()):
            l_aux = l_aux + 0.05 * F.cross_entropy(pvl[a][:, 0].float(), nx[a])
        if bool(b.any()):
            later = D['board'][fi[b]].long(); own = (later > 0).long() + (later > 6).long()             # 0 empty, 1 mine, 2 theirs
            l_aux = l_aux + 0.1 * F.cross_entropy(futl[b].float().reshape(-1, 3), own.reshape(-1))
    return l_pol, l_val, l_q, (lp.argmax(1) == mv).float().mean(), l_aux


if __name__ == '__main__':
    files = sorted(glob.glob('zero/opool/*.o'), key=os.path.getmtime); keep = json.load(open('zero/ouro_ab_files.json')) if os.path.exists('zero/ouro_ab_files.json') else None
    if keep is None:                                                 # the first arm fixes the file list; later arms use exactly the same files
        keep = files; json.dump(keep, open('zero/ouro_ab_files.json', 'w'))
    val_files = keep[::25]; vs = set(val_files); D = load([f for f in keep if f not in vs]); V = load(val_files)
    torch.manual_seed(0); model = Ouro(**CFG).to(DEV); ema = Ouro(**CFG).to(DEV); ema.load_state_dict(model.state_dict())
    opts = make_optimizers(model, 0.02, 3e-3, use_muon=True)
    for o in opts:
        for g in o.param_groups:
            g['base_lr'] = g['lr']
    fwd = torch.compile(model); gen = torch.Generator(device='cpu'); gen.manual_seed(1)
    log(f'arm {ARM}: {CFG}{" + auxiliary targets" if AUX else ""}; {sum(p.numel() for p in model.parameters()):,} parameters; {D["n"]/1e6:.2f}M training positions from {len(keep) - len(val_files)} files, '
        f'{V["n"]/1e6:.2f}M held out; {STEPS} steps of {BS}')
    t0 = time.time(); acc = np.zeros(5); nacc = 0
    for step in range(STEPS):
        idx = torch.randint(0, D['n'], (BS,), generator=gen).to(DEV); loops = 1 + (step * 7919) % CFG['max_loops']
        l_pol, l_val, l_q, top1, l_aux = losses(fwd, D, idx, loops, AUX)
        for o in opts:
            o.zero_grad(set_to_none=True)
        (l_pol + l_val + l_q + l_aux).backward()
        m = 0.5 * min(1.0, (step + 1) / 300)
        for o in opts:
            for g in o.param_groups:
                g['lr'] = g['base_lr'] * m
            o.step()
        with torch.no_grad():
            d = min(0.999, (2 + step) / (11 + step))
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - d)
        acc += [l_pol.item(), l_val.item(), l_q.item(), top1.item(), float(l_aux)]; nacc += 1
        if (step + 1) % 2000 == 0 or step + 1 == STEPS:
            ema.eval(); tot = np.zeros(4)
            with torch.no_grad():
                vi = torch.randint(0, V['n'], (16384,), generator=torch.Generator().manual_seed(7)).to(DEV)
                for j in range(0, len(vi), 2048):
                    tot += [x.item() for x in losses(ema, V, vi[j:j + 2048], 2, 0)[:4]]
            tot /= 8
            log(f'step {step + 1}: train policy {acc[0]/nacc:.3f} value {acc[1]/nacc:.4f} move-match {acc[3]/nacc:.3f}' + (f' aux {acc[4]/nacc:.3f}' if AUX else '')
                + f' || held-out: policy {tot[0]:.4f} value {tot[1]:.4f} move-match {tot[3]:.4f} | {(step + 1) * BS / (time.time() - t0):,.0f} positions/s')
            acc[:] = 0; nacc = 0
    torch.save(dict(cfg=CFG, ema={k: v.cpu() for k, v in ema.state_dict().items()}, beta=4.0, step=STEPS), f'zero/ouro_ab_{ARM}.pt')
    log(f'arm {ARM} finished')
