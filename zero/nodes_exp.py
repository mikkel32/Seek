"""How many nodes per move should self-play spend? Measured at equal CPU time.

Four arms play at the same time with the same network and the same number of processes for the same wall time:
1,250 / 2,500 / 5,000 / 10,000 nodes per move. Fewer nodes means more games but weaker moves and noisier scores.
Each arm's games train one network from scratch (same width, same number of training steps), and the four
networks play each other at an equal node budget. The arm whose network wins is the better use of a CPU-second.

Production self-play is halted while the arms play (so the timing is clean) and restarted as soon as they finish;
the 5,000-node arm's games are handed to production afterwards, so they are not wasted.

  python3 zero/nodes_exp.py [minutes=45] [processes per arm=10]        (restartable; results in zero/nodes_exp.json)
"""
import glob, itertools, json, os, signal, subprocess, sys, time
import numpy as np

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
MIN = float(sys.argv[1]) if len(sys.argv) > 1 else 45.0
PP = int(sys.argv[2]) if len(sys.argv) > 2 else 10
ARMS = [1250, 2500, 5000, 10000]
ENGINE = os.environ.get('ENGINE', 'calc/calc_z5'); H = int(os.environ.get('NX_H', 128)); STEPS = int(os.environ.get('NX_STEPS', 6000))
TEST_NODES = int(os.environ.get('NX_TEST', 10000)); LAM = 0.5
REC = np.dtype([('idx', '<i2', 32), ('score', '<i2'), ('result', 'i1'), ('ply', 'i1')])
LOG = open('zero/nodes_exp.log', 'a')
RES = json.load(open('zero/nodes_exp.json')) if os.path.exists('zero/nodes_exp.json') else {}


def log(msg):
    line = f'[{time.strftime("%H:%M:%S", time.gmtime())}] {msg}'
    print(line, flush=True); LOG.write(line + '\n'); LOG.flush()


def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return '\n'.join(l for l in (r.stdout + r.stderr).split('\n') if l and 'Warning' not in l and 'from_numpy' not in l and 'Consider using' not in l)


def save():
    json.dump(RES, open('zero/nodes_exp.json.tmp', 'w'), indent=1); os.replace('zero/nodes_exp.json.tmp', 'zero/nodes_exp.json')


def engines():
    return int(subprocess.run(['pgrep', '-xc', os.path.basename(ENGINE)], capture_output=True, text=True).stdout.strip() or 0)


def target(q):
    sc = 1 / (1 + np.exp(-q['score'].astype(np.float32) / 271.58))
    return np.where(q['ply'] < 0, sc, LAM * sc + (1 - LAM) * (q['result'].astype(np.float32) + 1) / 2)


WORKER = r'''
i=$1; N=$2; G=$3; D=$4; NET=$5
while [ $(date +%%s) -lt $D ]; do
  f=zero/nx/n${N}_w${i}_$(date +%%s%%N)
  %s $NET selfplay $f.tmp $G $N $(( ($(date +%%s%%N) / 1000 + i * 7919 + N) %% 2000000000 )) 1 4 > /dev/null 2>&1 && mv $f.tmp $f.bin
done
'''

if __name__ == '__main__':
    os.makedirs('zero/nx', exist_ok=True)
    # ---- 1. the four arms play, production halted
    if 'play' not in RES:
        log(f'node-budget experiment: {len(ARMS)} arms x {PP} processes x {MIN} minutes; halting production (a cycle in progress finishes first)')
        open('zero/HALT', 'w').write('nodes_exp')
        for _ in range(900):                                      # a training cycle in progress finishes first (its match uses the same engine)
            if engines() == 0:
                break
            time.sleep(2)
        if engines():
            os.remove('zero/HALT'); log('production did not stop within 30 minutes; nothing done'); sys.exit(1)
        time.sleep(3)
        gen, net = open('zero/current').read().split()            # the newest network, read once production has stopped
        log(f'production stopped; the arms play with {net}')
        for f in glob.glob('zero/nx/*'):
            os.remove(f)
        sh('cp data/ev_shval_x.npy data/ev_nxval_x.npy; cp data/ev_shval_y.npy data/ev_nxval_y.npy')
        open('zero/nx_worker.sh', 'w').write(WORKER % ENGINE)
        t0 = time.time(); deadline = int(t0 + MIN * 60)
        procs = [subprocess.Popen(['bash', 'zero/nx_worker.sh', str(i), str(n), str(max(2, round(20 * 5000 / n))), str(deadline), net], start_new_session=True)
                 for n in ARMS for i in range(PP)]
        while time.time() < deadline:
            time.sleep(5)
        for p in procs:                                           # equal time for every arm: unfinished files are dropped
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except Exception:
                pass
        time.sleep(2)
        for f in glob.glob('zero/nx/*.tmp'):
            os.remove(f)
        RES['play'] = {'minutes': MIN, 'procs_per_arm': PP, 'net': net, 'gen': int(gen), 'cpu_seconds_per_arm': round(PP * (time.time() - t0))}
        save()
    # ---- 2. data sets; hand the 5,000-node games to production and restart it
    gen = RES['play']['gen']
    if 'data' not in RES:
        RES['data'] = {}
        for n in ARMS:
            files = sorted(glob.glob(f'zero/nx/n{n}_w*.bin'))
            r = np.concatenate([np.fromfile(f, dtype=REC) for f in files])
            games = int(sum(max(2, round(20 * 5000 / n)) for _ in files)); played = r[r['ply'] >= 0]
            r = r[np.random.RandomState(n).permutation(len(r))]
            np.save(f'data/ev_nx{n}_x.npy', r['idx']); np.save(f'data/ev_nx{n}_y.npy', (target(r) * 255).round().astype(np.uint8))
            p = r[r['ply'] >= 0][:1_500_000]
            np.save(f'data/ev_nxp{n}_x.npy', p['idx']); np.save(f'data/ev_nxp{n}_y.npy', (target(p) * 255).round().astype(np.uint8))
            RES['data'][str(n)] = {'games': games, 'positions': int(len(r)), 'played': int(len(played)), 'tree': int(len(r) - len(played)),
                                   'decisive': round(float((played['result'] != 0).mean()), 3), 'plies_per_game': round(len(played) / games, 1)}
            log(f'arm {n} nodes: {games} games, {len(r)/1e6:.1f}M positions ({len(played)/1e6:.1f}M played + {(len(r)-len(played))/1e6:.1f}M from the trees), '
                f'{len(played)/games:.0f} plies per game, {100*float((played["result"] != 0).mean()):.0f}% of positions from decisive games')
            del r, played, p
        for f in glob.glob('zero/nx/n5000_w*.bin'):               # same network, same settings as production: use them
            os.replace(f, f'zero/pool/g{gen}_x{os.path.basename(f)}')
        for f in glob.glob('zero/nx/*.bin'):
            os.remove(f)
        save()
    if os.path.exists('zero/HALT') and open('zero/HALT').read() == 'nodes_exp':
        os.remove('zero/HALT')
        time.sleep(25)                                            # let the halted loop finish stopping before the new one starts
        subprocess.run('tmux new-window -d -t cont -n prod "cd /root/loopchess && NPROC=42 python3 zero/cont.py 2>&1 | tee -a zero/cont_stdout.log"', shell=True)
        log('production restarted')
    # ---- 3. one network per arm, from scratch, same width and the same number of training steps
    for n in ARMS:
        if os.path.exists(f'zero/nx{n}.bin.ema.pt'):
            continue
        out = sh(f'CUDA_VISIBLE_DEVICES= python3 calc/solve_pst.py 400000 nxp{n} zero/nx{n}s')
        if not os.path.exists(f'zero/nx{n}s.npz'):
            log(f'arm {n}: table solve failed: {out[-200:]}'); sys.exit(1)
        ep = STEPS * 16384 / RES['data'][str(n)]['positions']
        t0 = time.time()
        out = sh(f'python3 calc/train_nnue.py --h {H} --buckets 8 --ema 0.999 --files nx{n} --val nxval --epochs {ep:.4f} --lr 1e-3 --threads 4 --device cuda '
                 f'--pst zero/nx{n}s.npz --out zero/nx{n}.bin')
        done = [l for l in out.split('\n') if l.startswith('done')]
        if not os.path.exists(f'zero/nx{n}.bin.ema.pt'):
            log(f'arm {n}: training failed: {out[-300:]}'); sys.exit(1)
        log(f'arm {n}: trained {STEPS} steps = {ep:.2f} epochs in {time.time()-t0:.0f}s; {done[0][:120] if done else out[-120:]}')
    # ---- 4. the four networks play each other at an equal node budget
    RES.setdefault('matches', {})
    for a, b in itertools.combinations(ARMS, 2):
        key = f'{a}v{b}'
        if key in RES['matches']:
            continue
        r = sh(f'python3 match.py --a calc:zero/nx{a}.bin:{TEST_NODES}:{ENGINE} --b calc:zero/nx{b}.bin:{TEST_NODES}:{ENGINE} --openings 500 --workers 40')
        try:
            d = json.loads(r.strip().split('\n')[-1])
            RES['matches'][key] = {k: d[k] for k in ('n', 'w', 'd', 'l', 'score', 'elo', 'lo', 'hi')}; save()
            log(f'{a} nodes vs {b} nodes: {d["w"]}-{d["d"]}-{d["l"]}  score {d["score"]}  Elo {d["elo"]} ({d["lo"]} to {d["hi"]})')
        except Exception:
            log(f'match {key} output: ' + r[-300:])
    pts = {n: 0.0 for n in ARMS}; cnt = {n: 0 for n in ARMS}
    for key, d in RES['matches'].items():
        a, b = (int(x) for x in key.split('v'))
        pts[a] += d['score'] * d['n']; pts[b] += (1 - d['score']) * d['n']; cnt[a] += d['n']; cnt[b] += d['n']
    RES['table'] = {str(n): round(pts[n] / max(1, cnt[n]), 4) for n in ARMS}; save()
    log('round robin score by arm (nodes per move in self-play): ' + '  '.join(f'{n}: {100*pts[n]/max(1,cnt[n]):.1f}%' for n in ARMS))
