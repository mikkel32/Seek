"""Train OURO. Resumable, time-budgeted, logs JSON lines."""
import argparse, json, os, time, math, copy
import numpy as np, torch, torch.nn.functional as F
from model import Ouro, make_optimizers, N_MOVES

p = argparse.ArgumentParser()
p.add_argument('--name', required=True)
p.add_argument('--minutes', type=float, default=10)
p.add_argument('--total_steps', type=int, default=20000)
p.add_argument('--sched_steps', type=int, default=0)   # LR schedule horizon (0 = total_steps)
p.add_argument('--bs', type=int, default=256)
p.add_argument('--d', type=int, default=96)
p.add_argument('--heads', type=int, default=6)
p.add_argument('--mlp', type=int, default=288)
p.add_argument('--n_in', type=int, default=2)
p.add_argument('--n_rec', type=int, default=3)
p.add_argument('--n_out', type=int, default=2)
p.add_argument('--max_loops', type=int, default=4)
p.add_argument('--no_rel', action='store_true')
p.add_argument('--no_cnt', action='store_true')
p.add_argument('--no_muon', action='store_true')
p.add_argument('--muon_lr', type=float, default=0.02)
p.add_argument('--adam_lr', type=float, default=3e-3)
p.add_argument('--tau', type=float, default=0.015)
p.add_argument('--pv_w', type=float, default=0.3)
p.add_argument('--warmup', type=int, default=100)
p.add_argument('--threads', type=int, default=4)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--eval_every', type=int, default=500)
p.add_argument('--eval_n', type=int, default=3000)
p.add_argument('--cb_frac', type=float, default=0.25)
p.add_argument('--data', default='data/lichess_b.npz')
p.add_argument('--cb', default='data/cb_train.npz')
p.add_argument('--extra', default='')            # extra dataset (e.g. fault positions) mixed at --extra_frac
p.add_argument('--extra_frac', type=float, default=0.0)
p.add_argument('--init', default='')             # init weights from checkpoint (fresh optimiser)
p.add_argument('--lr_scale', type=float, default=1.0)
p.add_argument('--flat_lr', action='store_true')
p.add_argument('--save_every', type=int, default=1000)
p.add_argument('--snap_every', type=int, default=0)   # keep numbered EMA snapshots (for soups)
args = p.parse_args()

torch.set_num_threads(args.threads)
torch.manual_seed(args.seed)
rng = np.random.RandomState(args.seed)
os.makedirs('ckpt', exist_ok=True)
ck_path = f'ckpt/{args.name}.pt'
log_path = f'logs/{args.name}.jsonl'


def load(path):
    z = np.load(path)
    pv = z['pv'] if 'pv' in z.files else np.full((len(z['boards']), 2), -1, dtype=np.int16)
    return [z['boards'], z['meta'], z['moves'], z['wps'], pv]


L = load(args.data)
C = load(args.cb) if args.cb_frac > 0 else None
X = load(args.extra) if args.extra else None
HOLD = 20000                                      # last HOLD lichess positions are never trained on
nL = len(L[0]) - HOLD
permL = np.random.RandomState(1234).permutation(nL)
T = np.load('data/cb_test.npz')
# dev half (B) = odd indices; held-out half (A) = even indices
dev_idx = np.arange(1, len(T['boards']), 2)[:args.eval_n]


def batch_from(src, idx):
    b = torch.from_numpy(src[0][idx].astype(np.int64))
    m = src[1][idx]
    return b, torch.from_numpy(m[:, 0].astype(np.int64)), torch.from_numpy(m[:, 1].astype(np.int64)), \
        torch.from_numpy(src[2][idx].astype(np.int64)), torch.from_numpy(src[3][idx].astype(np.float32)), \
        torch.from_numpy(src[4][idx].astype(np.int64))


def get_batch(step):
    nx = int(round(args.bs * args.extra_frac)) if X is not None else 0
    nc = int(round(args.bs * args.cb_frac)) if C is not None else 0
    nl = args.bs - nc - nx
    start = (step * nl) % (nL - nl)
    parts = [batch_from(L, np.sort(permL[start:start + nl]))]
    kinds = [torch.zeros(nl)]
    if nc:
        parts.append(batch_from(C, np.sort(rng.randint(0, len(C[0]), nc))))
        kinds.append(torch.ones(nc))
    if nx:
        parts.append(batch_from(X, np.sort(rng.randint(0, len(X[0]), nx))))
        kinds.append(torch.zeros(nx))
    K = max(q[3].shape[1] for q in parts)            # sources may carry different numbers of labelled moves
    parts = [(q[0], q[1], q[2], F.pad(q[3], (0, K - q[3].shape[1]), value=-1), F.pad(q[4], (0, K - q[4].shape[1])), q[5])
             for q in parts]
    out = [torch.cat([q[i] for q in parts]) for i in range(6)]
    return out + [torch.cat(kinds)]


def loss_fn(model, batch, loops):
    board, castle, ep, moves, wps, pv, kind = batch
    pol, v, ql, mask, pvl = model(board, castle, ep, loops=loops, aux=True)
    valid = moves >= 0
    mv = moves.clamp(min=0)
    mask = mask.scatter(1, mv, mask.gather(1, mv) | valid)
    logp = F.log_softmax(pol.masked_fill(~mask, -1e9), dim=1).gather(1, mv)
    tgt = torch.exp((wps - wps[:, :1]) / args.tau) * valid
    tgt = tgt / tgt.sum(1, keepdim=True)
    is_l = (kind == 0).float()
    l_pol = (-(tgt * logp).sum(1) * is_l).sum() / is_l.sum().clamp(min=1)
    l_val = (F.binary_cross_entropy_with_logits(v, wps[:, 0], reduction='none') * is_l).sum() / is_l.sum().clamp(min=1)
    l_q = (F.binary_cross_entropy_with_logits(ql.gather(1, mv), wps, reduction='none') * valid).sum() / valid.sum()
    loss = l_pol + l_val + l_q
    l_pv = torch.zeros(())
    if args.pv_w > 0:
        has = pv >= 0
        l_pv = (F.cross_entropy(pvl.reshape(-1, 4096), pv.clamp(min=0).reshape(-1), reduction='none') * has.reshape(-1)).sum() \
            / has.sum().clamp(min=1)
        loss = loss + args.pv_w * 2 * l_pv
    return loss, (l_pol.item(), l_val.item(), l_q.item(), l_pv.item())


@torch.no_grad()
def evaluate(model, idx, loops_list=(1, 2, 4), bs=500):
    model.eval()
    res = {}
    for loops in loops_list:
        hit_p = hit_q = hit_m = 0; reg_p = reg_q = reg_m = 0.0; bl_p = bl_q = bl_m = 0; n = 0; vmae = 0.0
        for i in range(0, len(idx), bs):
            j = idx[i:i + bs]
            board = torch.from_numpy(T['boards'][j].astype(np.int64))
            castle = torch.from_numpy(T['meta'][j, 0].astype(np.int64)); ep = torch.from_numpy(T['meta'][j, 1].astype(np.int64))
            tm = torch.from_numpy(T['moves'][j].astype(np.int64)); tq = torch.from_numpy(T['qs'][j])
            pol, v, ql, _ = model(board, castle, ep, loops=loops, tbptl=False)
            valid = tm >= 0
            mv = tm.clamp(min=0)
            lp = pol.gather(1, mv).masked_fill(~valid, -1e9)
            lq = ql.gather(1, mv).masked_fill(~valid, -1e9)
            qstar = tq.masked_fill(~valid, -1).max(1).values
            lm = F.log_softmax(lp, 1) + 4.0 * torch.sigmoid(lq)        # simple pi x Q mix
            for name, sc in (('p', lp), ('q', lq), ('m', lm)):
                r = qstar - tq.gather(1, sc.argmax(1, keepdim=True)).squeeze(1)
                if name == 'p': hit_p += (r < 1e-6).sum().item(); reg_p += r.sum().item(); bl_p += (r > 0.1).sum().item()
                if name == 'q': hit_q += (r < 1e-6).sum().item(); reg_q += r.sum().item(); bl_q += (r > 0.1).sum().item()
                if name == 'm': hit_m += (r < 1e-6).sum().item(); reg_m += r.sum().item(); bl_m += (r > 0.1).sum().item()
            vmae += (torch.sigmoid(v) - qstar).abs().sum().item()
            n += len(j)
        res[f'k{loops}'] = dict(acc_p=hit_p / n, acc_q=hit_q / n, acc_m=hit_m / n, reg_p=reg_p / n, reg_q=reg_q / n,
                                reg_m=reg_m / n, bl_p=bl_p / n, bl_q=bl_q / n, bl_m=bl_m / n, vmae=vmae / n)
    model.train()
    return res


def lr_mult(step):
    if step < args.warmup:
        return (step + 1) / args.warmup
    if args.flat_lr:
        return 1.0
    f = min(1.0, step / (args.sched_steps or args.total_steps))
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * f))


if __name__ == '__main__':
    step = 0
    seen = 0
    if os.path.exists(ck_path):
        ck = torch.load(ck_path, weights_only=False)
        model = Ouro(**ck['cfg'])
        model.load_state_dict(ck['model'])
        ema = Ouro(**ck['cfg']); ema.load_state_dict(ck['ema'])
        opts = make_optimizers(model, args.muon_lr, args.adam_lr, use_muon=not args.no_muon)
        for o, s in zip(opts, ck['opts']):
            o.load_state_dict(s)
        step, seen = ck['step'], ck['seen']
        rng.set_state(ck['rng'])
        print('resumed at step', step, flush=True)
    else:
        if args.init:
            ck = torch.load(args.init, weights_only=False)
            model = Ouro(**ck['cfg']); model.load_state_dict(ck['ema'])
        else:
            model = Ouro(d=args.d, heads=args.heads, mlp=args.mlp, n_in=args.n_in, n_rec=args.n_rec, n_out=args.n_out,
                         use_rel=not args.no_rel, use_cnt=not args.no_cnt, max_loops=args.max_loops)
        ema = copy.deepcopy(model)
        opts = make_optimizers(model, args.muon_lr, args.adam_lr, use_muon=not args.no_muon)
    for o in opts:
        for g in o.param_groups:
            g['base_lr'] = g.get('base_lr', g['lr'])
    print('params', model.n_params(), 'cfg', model.cfg, flush=True)
    ema_decay = 0.999
    t0 = time.time()
    acc = np.zeros(4); nacc = 0
    logf = open(log_path, 'a')
    while time.time() - t0 < args.minutes * 60 and step < args.total_steps:
        loops = 1 + (step * 7919 + args.seed) % args.max_loops if args.max_loops > 1 else 1
        batch = get_batch(step)
        loss, parts = loss_fn(model, batch, loops)
        for o in opts:
            o.zero_grad(set_to_none=True)
        loss.backward()
        m = lr_mult(step) * args.lr_scale
        for o in opts:
            for g in o.param_groups:
                g['lr'] = g['base_lr'] * m
            o.step()
        step += 1; seen += args.bs
        with torch.no_grad():
            d = min(ema_decay, (1 + step) / (10 + step))
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - d)
        acc += parts; nacc += 1
        if step % 50 == 0:
            el = time.time() - t0
            rec = dict(step=step, seen=seen, l_pol=acc[0] / nacc, l_val=acc[1] / nacc, l_q=acc[2] / nacc, l_pv=acc[3] / nacc,
                       pos_per_s=nacc * args.bs / max(1e-9, el - getattr(evaluate, 'last', 0)), lr=m)
            evaluate.last = el
            acc[:] = 0; nacc = 0
            if step % args.eval_every == 0:
                te = time.time()
                rec['eval'] = evaluate(ema, dev_idx)
                evaluate.last += time.time() - te
                t0 += 0  # eval time counts toward wall budget
                e = rec['eval']
                print(f"step {step} seen {seen/1e6:.2f}M pol {rec['l_pol']:.3f} val {rec['l_val']:.3f} q {rec['l_q']:.3f} pv {rec['l_pv']:.3f} "
                      f"{rec['pos_per_s']:.0f} pos/s | acc_p k1 {e['k1']['acc_p']:.3f} k2 {e['k2']['acc_p']:.3f} k4 {e['k4']['acc_p']:.3f} "
                      f"| acc_m k2 {e['k2']['acc_m']:.3f} bl_p k2 {e['k2']['bl_p']:.3f} bl_m k2 {e['k2']['bl_m']:.3f}", flush=True)
            logf.write(json.dumps(rec) + '\n'); logf.flush()
        if step % args.save_every == 0 or time.time() - t0 >= args.minutes * 60 or step >= args.total_steps:
            torch.save(dict(cfg=model.cfg, model=model.state_dict(), ema=ema.state_dict(), opts=[o.state_dict() for o in opts],
                            step=step, seen=seen, rng=rng.get_state()), ck_path + '.tmp')
            os.replace(ck_path + '.tmp', ck_path)
        if args.snap_every and step % args.snap_every == 0:
            torch.save(dict(cfg=model.cfg, ema=ema.state_dict(), step=step, seen=seen), f'ckpt/{args.name}_s{step}.pt')
    print('done', step, seen, flush=True)
