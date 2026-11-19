"""Tune search options at a deep node budget on the quality test (ext/quality.py) by simultaneous perturbation.

Each round moves every option one step up or down at random, lets both mirrored settings choose a move in the same batch
of test positions, and shifts every option a little towards the side that gave away less. The engine is deterministic at a
node budget, so the two settings differ only where they really choose differently.

  python3 ext/q_tune.py --name t1 --cmd "calc/calc_s19f zero/w_BIGM.bin" --base "Hash=64,PvPrune=1,ChkExt=0,SeDouble=20,CutIir=1" \
      --nodes 300000 --batch 3000 --rounds 200 --params "LmrBase:68:12:20:220,LmrDiv:240:20:120:420,..."

--params  Name:start:step:min:max, comma separated (integers; step is the size of the up/down probe)
--first/--limit  the slice of the usable positions to tune on (keep the rest for confirmation)
State (data/q/tune_<name>.json) is rewritten every round; the same command line resumes. Every round gives way to a test
that announces itself with zero/Q_BUSY (ext/q_screen.sh), like ext/q_fill.sh.
"""
import argparse, json, os, random, sys, time
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quality as Q


def w_run(a):
    cmd, opts, nodes, items, tag = a; e = Q.Eng(cmd, Q.opts_of(opts)); out = []
    for i, fen in items:
        e.new(); out.append((tag, i, e.move(fen, nodes)))
    e.quit()
    return out


def lock():
    while True:
        open('zero/FILL_BUSY', 'w').close()
        if not os.path.exists('zero/Q_BUSY'):
            return
        os.remove('zero/FILL_BUSY'); time.sleep(5)


def unlock():
    if os.path.exists('zero/FILL_BUSY'):
        os.remove('zero/FILL_BUSY')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True); ap.add_argument('--cmd', required=True); ap.add_argument('--base', default='Hash=64')
    ap.add_argument('--nodes', type=int, default=300000); ap.add_argument('--batch', type=int, default=3000); ap.add_argument('--rounds', type=int, default=200)
    ap.add_argument('--params', required=True); ap.add_argument('--workers', type=int, default=22)
    ap.add_argument('--first', type=int, default=0); ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--gain', type=float, default=0.5)        # a step of this many probe-steps per unit of loss difference, at the start
    ap.add_argument('--seed', type=int, default=1)
    a = ap.parse_args()
    P = [(n, float(v), float(c), float(lo), float(hi)) for n, v, c, lo, hi in (p.split(':') for p in a.params.split(','))]
    tr = Q.usable()[a.first:a.limit or None]; path = f'{Q.D}/tune_{a.name}.json'
    st = json.load(open(path)) if os.path.exists(path) else {'k': 0, 'theta': [p[1] for p in P], 'hist': []}
    theta = st['theta']; dn = max(200000, max(t['nodes'] for t in tr) // 2)
    print(f'{len(tr)} positions, batches of {a.batch}, {a.nodes} nodes; starting at round {st["k"]}', flush=True)
    while st['k'] < a.rounds:
        k = st['k']; rnd = random.Random(a.seed * 100003 + k); t0 = time.time()
        idx = rnd.sample(range(len(tr)), min(a.batch, len(tr))); delta = [rnd.choice((-1, 1)) for _ in P]
        sets = []
        for sgn in (1, -1):
            vals = [int(round(min(p[4], max(p[3], th + sgn * d * p[2])))) for p, th, d in zip(P, theta, delta)]
            sets.append(a.base + ',' + ','.join(f'{p[0]}={v}' for p, v in zip(P, vals)))
        items = [(i, tr[i]['fen']) for i in idx]; jobs = []
        for tag, o in enumerate(sets):
            jobs += [(a.cmd, o, a.nodes, items[j:j + 6], tag) for j in range(0, len(items), 6)]
        rnd.shuffle(jobs); mv = [{}, {}]
        lock()
        with Pool(a.workers) as pool:
            for r in pool.imap_unordered(w_run, jobs):
                for tag, i, m in r:
                    mv[tag][i] = m
        dem = json.load(open(f'{Q.D}/demand.json')) if os.path.exists(f'{Q.D}/demand.json') else {}
        need = sorted({(tr[i]['fen'], m[i]) for m in mv for i in idx if m[i] not in (tr[i]['m1'], tr[i]['m2']) and (tr[i]['fen'] + ' ' + m[i]) not in dem})
        if need:
            with Pool(a.workers) as pool:
                for r in pool.imap_unordered(Q.w_demand, [(dn, need[j:j + 3]) for j in range(0, len(need), 3)]):
                    for fen, m, cp in r:
                        dem[fen + ' ' + m] = cp
            json.dump(dem, open(f'{Q.D}/demand.json', 'w'))
        unlock()
        sub = [tr[i] for i in idx]
        L = [1000 * sum(Q.scores(sub, [m[i] for i in idx], dem)[0]) / len(idx) for m in mv]
        ak = a.gain / (1 + k / 25.0) ** 0.6
        for j, (p, d) in enumerate(zip(P, delta)):
            theta[j] = min(p[4], max(p[3], theta[j] - ak * (L[0] - L[1]) / 2 * d * p[2]))
        st['k'] = k + 1; st['hist'].append({'k': k, 'plus': L[0], 'minus': L[1], 'theta': [round(x, 2) for x in theta]})
        json.dump(st, open(path, 'w'))
        tail = st['hist'][-40:]; avg = [sum(h['theta'][j] for h in tail) / len(tail) for j in range(len(P))]
        print(f'round {k}: {L[0]:.2f} / {L[1]:.2f} ({len(need)} moves to Stockfish, {time.time() - t0:.0f}s)  ' + ' '.join(f'{p[0]}={x:.0f}' for p, x in zip(P, theta)) +
              '  | mean of the last 40: ' + ','.join(f'{p[0]}={x:.0f}' for p, x in zip(P, avg)), flush=True)
