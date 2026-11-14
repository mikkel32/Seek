"""Weight-space programming on OURO — the fixed recipe, steps 0-11, no training of any kind.

Adaptation of the recipe (Experiment 8, "Sparks of Weight-Space Programming") from a looping LM to a
chess network. Everything below was fixed BEFORE any model answer was looked at:

  goal        fewer blunders. A position is FAILED when the engine's searchless move
              (argmax of log pi + 4*Q, 2 loops) gives up more than 0.10 win probability against the
              best move in the reference (Stockfish value of every legal move).
  splits      practice = ChessBench test positions with odd index, held-out = even index.
  contrast    matched pairs (failed practice position, passed practice position with the same piece
              count, same best-value bucket, same legal-move bucket). ONE direction u = mean residual
              state after the last block on failed minus on passed (all 64 square tokens), normalised.
  layers      the two blocks whose MLPs write the most along u (failed minus passed).
  neurons     score_j = (mean activation on failed - on passed) * (u . W_out[:, j]); top k, score > 0.
  edit        W_out[:, j] <- W_out[:, j] - alpha * u * (u . W_out[:, j]) for the chosen columns only.
  candidates  C1 k=64 a=6 | C2 k=128 a=6 | C3 k=256 a=6 | C4 k=64 a=1 | control: random direction, k=64 a=6.
  screen      edit in memory; the blunder move must get less likely on fresh failed positions while the
              played move on matched passed positions and on ordinary positions stays put (>= -0.05 nats).
  gate        fixed >= max(3, 20% of failures checked), broke <= fixed/4, blunder share down >= 25%.
  escalation  next candidate (at most two more), then one second round on the edited copy, then restore.
  verdict     held-out looked at once; "clear gain" only if fixed > broke and sign test p < 0.05.
Never touched: embeddings, output heads, final norm. No alpha tuning. Held-out never chooses anything.
"""
import hashlib, json, math, os, platform, shutil, sys, time
import numpy as np, torch, torch.nn.functional as F
from model import Ouro

BETA, LOOPS, FAULT = 4.0, 2, 0.10
CANDS = [('C1', 64, 6.0), ('C2', 128, 6.0), ('C3', 256, 6.0), ('C4', 64, 1.0)]
N_DIR, N_SCR, N_NORMAL, STAY = 1024, 512, 512, -0.05
LOG = None


def log(*a):
    msg = ' '.join(str(x) for x in a)
    print(msg, flush=True)
    if LOG:
        LOG.write(msg + '\n'); LOG.flush()


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()[:16]


def load_model(path):
    ck = torch.load(path, weights_only=False)
    m = Ouro(**ck['cfg']); m.load_state_dict(ck['ema']); m.eval()
    return m, ck


def tens(z, idx):
    b = torch.from_numpy(z['boards'][idx].astype(np.int64))
    return b, torch.from_numpy(z['meta'][idx, 0].astype(np.int64)), torch.from_numpy(z['meta'][idx, 1].astype(np.int64))


@torch.no_grad()
def answers(model, T, idx, bs=500):
    """the engine's searchless answer on labelled positions -> chosen column, regret, log-lik of every move"""
    choice = np.zeros(len(idx), dtype=np.int64); regret = np.zeros(len(idx), dtype=np.float32)
    lls = []
    for i in range(0, len(idx), bs):
        j = idx[i:i + bs]
        board, castle, ep = tens(T, j)
        tm = torch.from_numpy(T['moves'][j].astype(np.int64)); tq = torch.from_numpy(T['qs'][j].astype(np.float32))
        pol, v, ql, _ = model(board, castle, ep, loops=LOOPS, tbptl=False)
        valid = tm >= 0; mv = tm.clamp(min=0)
        lp = pol.gather(1, mv).masked_fill(~valid, -1e9)
        lq = ql.gather(1, mv)
        sc = (F.log_softmax(lp, 1) + BETA * torch.sigmoid(lq)).masked_fill(~valid, -1e9)
        ll = F.log_softmax(sc, 1)
        c = sc.argmax(1)
        qstar = tq.masked_fill(~valid, -1).max(1).values
        choice[i:i + len(j)] = c.numpy()
        regret[i:i + len(j)] = (qstar - tq.gather(1, c[:, None]).squeeze(1)).numpy()
        lls.append(ll.numpy())
    return choice, regret, np.concatenate(lls)


@torch.no_grad()
def normal_ll(model, Z, idx, ref_choice=None, bs=256):
    """ordinary positions (no labels): log-lik of a reference move under the engine's choice distribution"""
    out = []; ch = []
    for i in range(0, len(idx), bs):
        j = idx[i:i + bs]
        board, castle, ep = tens(Z, j)
        pol, v, ql, mask = model(board, castle, ep, loops=LOOPS, tbptl=False)
        sc = (F.log_softmax(pol.masked_fill(~mask, -1e9), 1) + BETA * torch.sigmoid(ql)).masked_fill(~mask, -1e9)
        ll = F.log_softmax(sc, 1)
        c = sc.argmax(1) if ref_choice is None else torch.from_numpy(ref_choice[i:i + len(j)])
        ch.append(c.numpy()); out.append(ll.gather(1, c[:, None]).squeeze(1).numpy())
    return np.concatenate(out), np.concatenate(ch)


def blocks_of(model):
    return [('inp.%d' % i, b) for i, b in enumerate(model.inp)] + [('rec.%d' % i, b) for i, b in enumerate(model.rec)] + \
           [('out.%d' % i, b) for i, b in enumerate(model.out)]


@torch.no_grad()
def read_states(model, T, idx, bs=256):
    """mean residual state after each block and mean MLP neuron activation, over all tokens / applications"""
    st = {}; ac = {}; n = {}
    hooks = []
    for name, blk in blocks_of(model):
        st[name] = torch.zeros(model.d, dtype=torch.float64); ac[name] = torch.zeros(blk.pr.in_features, dtype=torch.float64); n[name] = 0
        def h_out(mod, inp, out, name=name):
            st[name] += out.detach().double().sum((0, 1)); n[name] += out.shape[0] * out.shape[1]
        def h_act(mod, inp, name=name):
            ac[name] += inp[0].detach().double().sum((0, 1))
        hooks += [blk.register_forward_hook(h_out), blk.pr.register_forward_pre_hook(h_act)]
    for i in range(0, len(idx), bs):
        board, castle, ep = tens(T, idx[i:i + bs])
        model(board, castle, ep, loops=LOOPS, tbptl=False)
    for h in hooks:
        h.remove()
    return {k: (st[k] / n[k]).float() for k in st}, {k: (ac[k] / n[k]).float() for k in ac}


def matched_pairs(T, idx, failed, limit):
    """(failed, passed) practice pairs matched on piece count, best-value bucket and legal-move bucket"""
    valid = T['moves'][idx] >= 0
    key = np.stack([(T['boards'][idx] > 0).sum(1), np.round(np.where(valid, T['qs'][idx], -1).max(1) * 10).astype(int),
                    valid.sum(1) // 8], 1)
    pool = {}
    for i in np.nonzero(~failed)[0]:
        pool.setdefault(tuple(key[i]), []).append(i)
    pairs = []
    for i in np.nonzero(failed)[0]:
        lst = pool.get(tuple(key[i]))
        if lst:
            pairs.append((i, lst.pop(0)))
            if len(pairs) >= limit:
                break
    return pairs


def plan_edit(model, sU, sW, aU, aW, k, alpha, layers=None, rng=None):
    """ONE direction, read from the residual state after the last block (what the heads see).
    Each layer's MLP is then asked how much it writes along it (failed minus passed); the two
    biggest writers are edited. -> {block: (columns, u, alpha)}"""
    last = blocks_of(model)[-1][0]
    d = sU[last] - sW[last]
    u = d / d.norm()
    info = {}
    for name, blk in blocks_of(model):
        W = blk.pr.weight.detach()                          # (d, mlp): column j = neuron j's output
        score = (aU[name] - aW[name]) * (u @ W)
        info[name] = dict(score=score, order=torch.argsort(score, descending=True), share=float(score.sum()), dnorm=float(d.norm()))
    if layers is None:
        layers = sorted(info, key=lambda n: -info[n]['share'])[:2]
    plan = {}
    for name in layers:
        order = info[name]['order'][:k]
        cols = order[info[name]['score'][order] > 0]
        uu = u
        if rng is not None:                                 # control: random direction, same columns, same size
            uu = torch.from_numpy(rng.standard_normal(u.shape[0]).astype(np.float32)); uu = uu / uu.norm()
        plan[name] = (cols, uu, alpha)
    return plan, info, layers


@torch.no_grad()
def apply_edit(model, plan):
    """W[:, j] -= alpha * u (u . W[:, j]) on the chosen columns only. returns number of weights touched."""
    blocks = dict(blocks_of(model)); touched = 0
    for name, (cols, u, alpha) in plan.items():
        W = blocks[name].pr.weight
        Wc = W[:, cols]
        W[:, cols] = Wc - alpha * torch.outer(u, u @ Wc)
        touched += Wc.numel()
    return touched


def clone_model(model):
    m = Ouro(**model.cfg); m.load_state_dict(model.state_dict()); m.eval()
    return m


def sign_test(a, b):
    n = a + b
    if n == 0:
        return 1.0
    k = min(a, b)
    lg = lambda x: math.lgamma(x + 1)
    tail = sum(math.exp(lg(n) - lg(i) - lg(n - i) - n * math.log(2)) for i in range(0, k + 1))
    return min(1.0, 2 * tail)


def round_of_editing(model, T, P_idx, base_failed_mask, tag, rep):
    """steps 5-6 on `model`: contrast, direction, four candidates + control, screen. returns ranked plans."""
    choice, regret, ll = answers(model, T, P_idx)
    failed = regret > FAULT
    pairs = matched_pairs(T, P_idx, failed, N_DIR + N_SCR)
    n_dir = min(N_DIR, (2 * len(pairs)) // 3)
    dir_pairs, scr_pairs = pairs[:n_dir], pairs[n_dir:n_dir + N_SCR]
    log(f'[{tag}] practice: {failed.sum()} failed of {len(failed)} ({failed.mean():.1%}); matched pairs {len(pairs)} '
        f'-> {len(dir_pairs)} for the direction, {len(scr_pairs)} for the screen')
    if len(dir_pairs) < 12:
        log(f'[{tag}] fewer than 12 matched pairs: cannot read a direction'); return [], None
    U = P_idx[[p[0] for p in dir_pairs]]; W = P_idx[[p[1] for p in dir_pairs]]
    sU, aU = read_states(model, T, U); sW, aW = read_states(model, T, W)
    # screening sets
    su = np.array([p[0] for p in scr_pairs]); sw = np.array([p[1] for p in scr_pairs])
    Z = np.load('data/lichess_b.npz')
    zidx = np.arange(len(Z['boards']) - 20000, len(Z['boards']) - 20000 + N_NORMAL)       # never trained on
    base_u = ll[su, choice[su]].mean(); base_w = ll[sw, choice[sw]].mean()
    base_n, zc = normal_ll(model, Z, zidx)
    base_n = base_n.mean()

    def screen(plan):
        m = clone_model(model); touched = apply_edit(m, plan)
        c2, r2, ll2 = answers(m, T, P_idx[np.concatenate([su, sw])])
        du = ll2[:len(su)][np.arange(len(su)), choice[su]].mean() - base_u
        dw = ll2[len(su):][np.arange(len(sw)), choice[sw]].mean() - base_w
        dn = normal_ll(m, Z, zidx, zc)[0].mean() - base_n
        return dict(unwanted=float(du), wanted=float(dw), normal=float(dn), touched=touched,
                    scr_fixed=int((r2[:len(su)] <= FAULT).sum()), scr_broke=int((r2[len(su):] > FAULT).sum()), n_scr=len(su))

    plan0, info, layers = plan_edit(model, sU, sW, aU, aW, 64, 6.0)
    log(f'[{tag}] one direction read after the last block (|mean diff| = {info[layers[0]]["dnorm"]:.3f}); what each layer\'s MLP writes '
        f'along it, failed minus passed: ' + ', '.join(f"{n}={info[n]['share']:+.3f}" for n in info) + f' -> writer layers {layers}')
    rows = []
    for name, k, alpha in CANDS:
        plan, _, _ = plan_edit(model, sU, sW, aU, aW, k, alpha, layers)
        s = screen(plan); s.update(name=name, k=k, alpha=alpha, cols={n: int(len(p[0])) for n, p in plan.items()})
        s['eligible'] = bool(s['unwanted'] < 0 and s['wanted'] >= STAY and s['normal'] >= STAY)
        rows.append((s, plan))
    ctrl, _, _ = plan_edit(model, sU, sW, aU, aW, 64, 6.0, layers, rng=np.random.default_rng(0))
    c = screen(ctrl); c.update(name='control(random)', k=64, alpha=6.0)
    for s in [r[0] for r in rows] + [c]:
        log(f"[{tag}] screen {s['name']:<15} k={s['k']:<3} a={s['alpha']:<3} | blunder-move loglik {s['unwanted']:+.3f} | played-move on "
            f"matched passed {s['wanted']:+.3f} | on ordinary {s['normal']:+.3f} | screen set: fixed {s['scr_fixed']}/{s['n_scr']} "
            f"broke {s['scr_broke']}/{s['n_scr']} | {'eligible' if s.get('eligible') else ('-' if 'eligible' not in s else 'rejected')}")
    rep[tag] = dict(failed=int(failed.sum()), n=int(len(failed)), pairs=len(pairs), layers=layers,
                    share={n: info[n]['share'] for n in info}, candidates=[r[0] for r in rows], control=c)
    ranked = sorted([r for r in rows if r[0]['eligible']], key=lambda r: r[0]['unwanted'])
    return ranked, layers


def gate(model, T, P_idx, base_failed, six, tag):
    chk = np.concatenate([np.nonzero(base_failed)[0], six])
    c, r, _ = answers(model, T, P_idx[chk])
    nf = int(base_failed.sum())
    fixed = int((r[:nf] <= FAULT).sum()); broke = int((r[nf:] > FAULT).sum())
    before = nf / len(chk); after = (nf - fixed + broke) / len(chk)
    need = max(3, math.ceil(0.2 * nf))
    ok = fixed >= need and broke <= fixed / 4 and after <= 0.75 * before
    log(f'[{tag}] GATE on {nf} base failures + 6 base passes: fixed {fixed} (need >= {need}), broke {broke} (allowed <= {fixed/4:.1f}), '
        f'blunder share {before:.3f} -> {after:.3f} (need <= {0.75*before:.3f})  => {"PASS" if ok else "FAIL"}')
    return ok, dict(fixed=fixed, need=need, broke=broke, share_before=before, share_after=after, passed=bool(ok))


def main(base_path, copy_path):
    global LOG
    LOG = open('logs/wsp.log', 'w')
    rep = dict(goal=f'fewer blunders (searchless move gives up > {FAULT} win prob)', candidates_def=CANDS)
    # 1. machine
    log(f'step 1 machine: {platform.processor() or platform.machine()}, {os.cpu_count()} cores, torch {torch.__version__}, '
        f'cuda {torch.cuda.is_available()}, threads {torch.get_num_threads()}')
    # 2. copy
    shutil.copyfile(base_path, copy_path)
    log(f'step 2 copy: {base_path} ({sha(base_path)}) -> {copy_path}; every edit goes into the copy')
    # 3. splits (fixed by index parity before any answer is seen)
    Tz = np.load('data/cb_test.npz'); T = {k: Tz[k] for k in ('boards', 'meta', 'moves', 'qs')}
    P_idx = np.arange(1, len(T['boards']), 2); H_idx = np.arange(0, len(T['boards']), 2)
    log(f'step 3 splits: practice {len(P_idx)} positions (odd index), held-out {len(H_idx)} (even index)')
    # 4. base on both splits
    base, ck = load_model(base_path)
    cP, rP, _ = answers(base, T, P_idx); cH, rH, _ = answers(base, T, H_idx)
    np.savez('logs/wsp_base_answers.npz', cP=cP, rP=rP, cH=cH, rH=rH)
    fP = rP > FAULT
    log(f'step 4 base run: practice blunders {fP.sum()}/{len(fP)} = {fP.mean():.2%} (held-out baseline stored, not shown)')
    # 5. room for gain
    rep['practice_fail_rate'] = float(fP.mean())
    if fP.mean() < 0.10:
        log(f'step 5 WARNING: room for gain is {fP.mean():.1%}, under ~10%. The recipe says to warn before going on: '
            f'even a perfect edit can move the score by at most this much.')
    six = np.nonzero(~fP)[0][:6]
    # 6. direction, candidates, screen
    ranked, layers = round_of_editing(base, T, P_idx, fP, 'round1', rep)
    kept = None; attempts = []
    # 7-9. write, gate, escalate
    for s, plan in ranked[:3]:
        m = clone_model(base); apply_edit(m, plan)
        ok, g = gate(m, T, P_idx, fP, six, f"round1 {s['name']}")
        attempts.append(dict(candidate=s['name'], round=1, **g))
        if ok:
            kept = (m, [plan], s['name']); break
    if kept is None and ranked:
        log('escalation: second round on the copy edited with the top-ranked candidate')
        m1 = clone_model(base); apply_edit(m1, ranked[0][1])
        ranked2, _ = round_of_editing(m1, T, P_idx, fP, 'round2', rep)
        if ranked2:
            m2 = clone_model(m1); apply_edit(m2, ranked2[0][1])
            ok, g = gate(m2, T, P_idx, fP, six, f"round2 {ranked2[0][0]['name']}")
            attempts.append(dict(candidate=ranked[0][0]['name'] + '+' + ranked2[0][0]['name'], round=2, **g))
            if ok:
                kept = (m2, [ranked[0][1], ranked2[0][1]], attempts[-1]['candidate'])
    if not ranked:
        log('no candidate survived the screen (none made the blunder move less likely while leaving passed/ordinary positions alone)')
    rep['gate_attempts'] = attempts
    if kept is None:
        shutil.copyfile(base_path, copy_path)
        log(f'steps 7-9: gate never passed -> copy restored to the base weights ({sha(copy_path)} == {sha(base_path)}). '
            f'Held-out is not consulted for a model that was not changed.')
        rep['verdict'] = 'no edit kept (gate failed; copy restored)'
        rep['edited'] = False
    else:
        model, plans, name = kept
        ck2 = torch.load(copy_path, weights_only=False)
        new_sd = model.state_dict()
        changed = {}
        for k_, v_ in ck2['ema'].items():
            diff = (new_sd[k_] != v_)
            if diff.any():
                changed[k_] = (int(diff.any(0).sum()) if diff.ndim == 2 else int(diff.sum()), int(diff.sum()))
        intended = set(n + '.pr.weight' for p in plans for n in p)
        assert set(changed) == intended, (set(changed), intended)
        ck2['ema'] = {k_: v_.clone() for k_, v_ in new_sd.items()}
        ck2['wsp'] = dict(candidate=name, changed=changed)
        torch.save(ck2, copy_path)
        total = sum(v_.numel() for v_ in new_sd.values())
        nw = sum(c[1] for c in changed.values())
        log(f'step 7 written: candidate {name}; changed tensors {changed} (columns, weights); {nw} of {total} weights = {nw/total:.3%}; '
            f'all other tensors identical')
        rep['weights_changed'] = dict(tensors=changed, n=nw, total=total)
        # 10. held-out, once
        c2, r2, _ = answers(model, T, H_idx)
        fb, fe = rH > FAULT, r2 > FAULT
        fixed = int((fb & ~fe).sum()); broke = int((~fb & fe).sum()); p = sign_test(fixed, broke)
        verdict = 'clear gain' if (fixed > broke and p < 0.05) else ('consistent with a gain, not proven' if fixed > broke else 'no gain')
        log(f'step 10 HELD-OUT (looked at once): blunders {fb.sum()} -> {fe.sum()} of {len(fb)} ({fb.mean():.2%} -> {fe.mean():.2%}); '
            f'fixed {fixed}, broke {broke}, sign test p = {p:.4g}; mean regret {rH.mean():.4f} -> {r2.mean():.4f}; '
            f'best-move rate {(rH < 1e-6).mean():.4f} -> {(r2 < 1e-6).mean():.4f}  => {verdict}')
        rep['heldout'] = dict(base_blunders=int(fb.sum()), edited_blunders=int(fe.sum()), n=int(len(fb)), fixed=fixed, broke=broke, p=p,
                              regret_base=float(rH.mean()), regret_edited=float(r2.mean()),
                              acc_base=float((rH < 1e-6).mean()), acc_edited=float((r2 < 1e-6).mean()))
        rep['verdict'] = verdict; rep['edited'] = True; rep['candidate'] = name
    json.dump(rep, open('logs/wsp_report.json', 'w'), indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
    log('step 11 report written to logs/wsp_report.json')


if __name__ == '__main__':
    torch.set_num_threads(int(os.environ.get('THREADS', '4')))
    main(sys.argv[1], sys.argv[2])
