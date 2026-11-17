"""The large transformer on the graphics card: OURO's model and losses (model.py, train.py) on shards of the Lichess
evaluation database (lm/prep_lm.py). Same targets as before: a soft policy over the labelled moves, the position's value,
a value for every labelled move, the reply and the follow-up along the best line. Muon + AdamW, averaged weights.

  python3 lm/train_lm.py --name L1 --shards 0-5 --d 256 --heads 8 --mlp 768 --n_in 2 --n_rec 4 --n_out 2 --bs 2048 --epochs 1

The checkpoint (ckpt/<name>.pt) has the old format, so engine.Net and duo_uci.py read it. Held out: the last 20,000
positions of the first shard (never trained on) and the ChessBench test positions (data/cb_test.npz, odd half).
"""
import argparse, copy, json, math, os, sys, time
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from model import Ouro, make_optimizers

p = argparse.ArgumentParser()
p.add_argument('--name', required=True); p.add_argument('--shards', default='0-0')
p.add_argument('--d', type=int, default=256); p.add_argument('--heads', type=int, default=8); p.add_argument('--mlp', type=int, default=768)
p.add_argument('--n_in', type=int, default=2); p.add_argument('--n_rec', type=int, default=4); p.add_argument('--n_out', type=int, default=2)
p.add_argument('--max_loops', type=int, default=2); p.add_argument('--smolgen', type=int, default=0)
p.add_argument('--bs', type=int, default=2048); p.add_argument('--epochs', type=float, default=1.0); p.add_argument('--minutes', type=float, default=1e9)
p.add_argument('--muon_lr', type=float, default=0.02); p.add_argument('--adam_lr', type=float, default=3e-3)
p.add_argument('--tau', type=float, default=0.015); p.add_argument('--pv_w', type=float, default=0.3); p.add_argument('--warmup', type=int, default=500)
p.add_argument('--eval_every', type=int, default=2000); p.add_argument('--eval_n', type=int, default=10000); p.add_argument('--seed', type=int, default=0)
p.add_argument('--init', default='')                     # start from a checkpoint's averaged weights (fresh optimisers)
p.add_argument('--chunk', type=int, default=1_000_000)   # positions read from a shard at a time
a = p.parse_args()
torch.manual_seed(a.seed); rng = np.random.RandomState(a.seed); dev = torch.device('cuda'); torch.backends.cuda.matmul.allow_tf32 = True
os.makedirs('ckpt', exist_ok=True); os.makedirs('logs', exist_ok=True); ck_path = f'ckpt/{a.name}.pt'
s0, s1 = (int(x) for x in a.shards.split('-')); SH = []
for s in range(s0, s1 + 1):
    SH.append([np.load(f'data/lm_{s}_{k}.npy', mmap_mode='r') for k in 'bmvwp'])
HOLD = 20000; N = sum(len(x[0]) for x in SH) - HOLD; steps_total = int(a.epochs * N / a.bs)
T = np.load('data/cb_test.npz'); dev_idx = np.arange(1, len(T['boards']), 2)[:a.eval_n]


def to_batch(arrs):
    b, m, v, w, pv = arrs
    return (torch.from_numpy(np.ascontiguousarray(b)).to(dev).long(), torch.from_numpy(np.ascontiguousarray(m[:, 0])).to(dev).long(), torch.from_numpy(np.ascontiguousarray(m[:, 1])).to(dev).long(),
            torch.from_numpy(np.ascontiguousarray(v)).to(dev).long(), torch.from_numpy(np.ascontiguousarray(w).astype(np.float32)).to(dev), torch.from_numpy(np.ascontiguousarray(pv)).to(dev).long())


def chunks():
    """one pass: every shard cut into chunks, chunks in random order; a chunk's positions are already shuffled (lm/prep_lm.py)"""
    plan = [(si, o) for si, sh in enumerate(SH) for o in range(0, len(sh[0]) - (HOLD if si == 0 else 0) - a.bs, a.chunk)]
    for k in rng.permutation(len(plan)):
        si, o = plan[k]; e = min(o + a.chunk, len(SH[si][0]) - (HOLD if si == 0 else 0))
        yield to_batch([x[o:e] for x in SH[si]])


def loss_fn(model, batch, loops):
    board, castle, ep, moves, wps, pv = batch
    pol, v, ql, mask, pvl = model(board, castle, ep, loops=loops, aux=True)
    valid = moves >= 0; mv = moves.clamp(min=0)
    mask = mask.scatter(1, mv, mask.gather(1, mv) | valid)
    logp = F.log_softmax(pol.masked_fill(~mask, -1e9), dim=1).gather(1, mv)
    tgt = torch.exp((wps - wps[:, :1]) / a.tau) * valid; tgt = tgt / tgt.sum(1, keepdim=True)
    l_pol = -(tgt * logp).sum(1).mean()
    l_val = F.binary_cross_entropy_with_logits(v, wps[:, 0])
    l_q = (F.binary_cross_entropy_with_logits(ql.gather(1, mv), wps, reduction='none') * valid).sum() / valid.sum()
    has = pv >= 0
    l_pv = (F.cross_entropy(pvl.reshape(-1, 4096), pv.clamp(min=0).reshape(-1), reduction='none') * has.reshape(-1)).sum() / has.sum().clamp(min=1)
    return l_pol + l_val + l_q + a.pv_w * 2 * l_pv, torch.stack([l_pol.detach(), l_val.detach(), l_q.detach(), l_pv.detach()])


@torch.no_grad()
def evaluate(model):
    """held-out Lichess positions: policy and value loss, best-move match; ChessBench test: best-move match, blunders, value error"""
    model.eval(); out = {}
    hb = to_batch([x[len(SH[0][0]) - HOLD:] for x in SH[0]]); lp = lv = hit = 0.0
    for i in range(0, HOLD, 2000):
        b = tuple(t[i:i + 2000] for t in hb); pol, v, ql, mask = model(b[0], b[1], b[2], loops=1, tbptl=False)
        mv = b[3].clamp(min=0); mask = mask.scatter(1, mv, mask.gather(1, mv) | (b[3] >= 0)); z = pol.masked_fill(~mask, -1e9)
        lp += float(F.cross_entropy(z, mv[:, 0], reduction='sum')); lv += float(F.binary_cross_entropy_with_logits(v, b[4][:, 0], reduction='sum')); hit += float((z.argmax(1) == mv[:, 0]).sum())
    out.update(pol_loss=lp / HOLD, val_loss=lv / HOLD, best_move=hit / HOLD)
    hp = bl = n = 0; vm = 0.0
    for i in range(0, len(dev_idx), 1000):
        j = dev_idx[i:i + 1000]
        board = torch.from_numpy(T['boards'][j].astype(np.int64)).to(dev); castle = torch.from_numpy(T['meta'][j, 0].astype(np.int64)).to(dev); ep = torch.from_numpy(T['meta'][j, 1].astype(np.int64)).to(dev)
        tm = torch.from_numpy(T['moves'][j].astype(np.int64)).to(dev); tq = torch.from_numpy(T['qs'][j]).to(dev)
        pol, v, ql, _ = model(board, castle, ep, loops=1, tbptl=False)
        valid = tm >= 0; lpm = pol.gather(1, tm.clamp(min=0)).masked_fill(~valid, -1e9); qstar = tq.masked_fill(~valid, -1).max(1).values
        r = qstar - tq.gather(1, lpm.argmax(1, keepdim=True)).squeeze(1)
        hp += int((r < 1e-6).sum()); bl += int((r > 0.1).sum()); vm += float((torch.sigmoid(v) - qstar).abs().sum()); n += len(j)
    out.update(cb_best_move=hp / n, cb_blunders=bl / n, cb_value_err=vm / n)
    model.train()
    return out


def lr_mult(step):
    if step < a.warmup: return (step + 1) / a.warmup
    return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, step / max(1, steps_total))))


step = seen = 0
if os.path.exists(ck_path):
    ck = torch.load(ck_path, weights_only=False); model = Ouro(**ck['cfg']).to(dev); model.load_state_dict(ck['model'])
    ema = Ouro(**ck['cfg']).to(dev); ema.load_state_dict(ck['ema']); opts = make_optimizers(model, a.muon_lr, a.adam_lr)
    for o, s in zip(opts, ck['opts']): o.load_state_dict(s)
    step, seen = ck['step'], ck['seen']; print('resumed at step', step, flush=True)
else:
    if a.init:
        ck = torch.load(a.init, weights_only=False); model = Ouro(**ck['cfg']).to(dev); model.load_state_dict(ck['ema'])
    else:
        model = Ouro(d=a.d, heads=a.heads, mlp=a.mlp, n_in=a.n_in, n_rec=a.n_rec, n_out=a.n_out, max_loops=a.max_loops, smolgen=a.smolgen).to(dev)
    ema = copy.deepcopy(model); opts = make_optimizers(model, a.muon_lr, a.adam_lr)
for o in opts:
    for g in o.param_groups: g['base_lr'] = g.get('base_lr', g['lr'])
print(f'{model.n_params()} parameters, {model.cfg}; {N} positions in {len(SH)} shards, {steps_total} steps of {a.bs}', flush=True)
t0 = time.time(); acc = torch.zeros(4, device=dev); nacc = 0; logf = open(f'logs/{a.name}.jsonl', 'a'); done = False; tl = t0
while not done:
    for chunk in chunks():
        n = len(chunk[0]); perm = torch.randperm(n, device=dev)
        for i in range(0, n - a.bs + 1, a.bs):
            if step >= steps_total or time.time() - t0 > a.minutes * 60: done = True; break
            idx = perm[i:i + a.bs]; batch = tuple(t[idx] for t in chunk)
            loops = 1 + (step * 7919 + a.seed) % a.max_loops if a.max_loops > 1 else 1
            loss, parts = loss_fn(model, batch, loops)
            for o in opts: o.zero_grad(set_to_none=True)
            loss.backward(); m = lr_mult(step)
            for o in opts:
                for g in o.param_groups: g['lr'] = g['base_lr'] * m
                o.step()
            step += 1; seen += a.bs
            with torch.no_grad():
                d = min(0.999, (1 + step) / (10 + step))
                for pe, pm in zip(ema.parameters(), model.parameters()): pe.lerp_(pm, 1 - d)
            acc += parts; nacc += 1
            if step % 200 == 0:
                l = (acc / nacc).tolist(); now = time.time(); rec = dict(step=step, seen=seen, l_pol=l[0], l_val=l[1], l_q=l[2], l_pv=l[3], pos_per_s=nacc * a.bs / (now - tl), lr=m)
                acc.zero_(); nacc = 0
                if step % a.eval_every == 0:
                    rec['eval'] = e = evaluate(ema)
                    print(f"step {step}/{steps_total} seen {seen / 1e6:.1f}M pol {l[0]:.3f} val {l[1]:.4f} q {l[2]:.4f} pv {l[3]:.3f} {rec['pos_per_s']:.0f} pos/s | held-out: policy loss {e['pol_loss']:.3f} value loss {e['val_loss']:.4f} "
                          f"best move {100 * e['best_move']:.1f}% | ChessBench: best move {100 * e['cb_best_move']:.1f}% blunders {100 * e['cb_blunders']:.1f}% value error {e['cb_value_err']:.4f}", flush=True)
                elif step % 1000 == 0:
                    print(f"step {step}/{steps_total} seen {seen / 1e6:.1f}M pol {l[0]:.3f} val {l[1]:.4f} q {l[2]:.4f} pv {l[3]:.3f} {rec['pos_per_s']:.0f} pos/s", flush=True)
                tl = time.time(); logf.write(json.dumps(rec) + '\n'); logf.flush()
            if step % 5000 == 0:
                torch.save(dict(cfg=model.cfg, model=model.state_dict(), ema=ema.state_dict(), opts=[o.state_dict() for o in opts], step=step, seen=seen), ck_path + '.tmp'); os.replace(ck_path + '.tmp', ck_path)
        if done: break
    if step >= steps_total: done = True
torch.save(dict(cfg=model.cfg, model=model.state_dict(), ema=ema.state_dict(), opts=[o.state_dict() for o in opts], step=step, seen=seen), ck_path + '.tmp'); os.replace(ck_path + '.tmp', ck_path)
e = evaluate(ema)
print(f"done: {step} steps, {seen / 1e6:.1f}M positions, {(time.time() - t0) / 60:.0f} minutes | held-out: policy loss {e['pol_loss']:.3f} value loss {e['val_loss']:.4f} best move {100 * e['best_move']:.1f}% | "
      f"ChessBench: best move {100 * e['cb_best_move']:.1f}% blunders {100 * e['cb_blunders']:.1f}% value error {e['cb_value_err']:.4f}", flush=True)
