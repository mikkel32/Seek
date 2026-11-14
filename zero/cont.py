"""Continuous self-taught loop: self-play never stops; a new network is trained on a sliding window of recent games.

  workers   NPLAY processes play games with the newest network, file after file, into zero/pool/
  trainer   every cycle: turn the finished files into a shard of training positions (whole files of games are set
            aside for validation), solve the piece-square table for the search's piece values, train the next network
            on the last WINDOW shards (GPU if there is one), play it against the previous one at an equal node budget,
            publish it to the workers. Every few generations self-play is paused for a clean rating against
            strength-limited Stockfish (measurement only).
No position, evaluation or game from any other engine is ever learned from.

  NPROC=42 python3 zero/cont.py           (safe to stop and start again)
"""
import glob, json, os, signal, subprocess, sys, time
import numpy as np

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
E = os.environ.get
NPROC = int(E('NPROC', os.cpu_count() or 4))
NPLAY = int(E('NPLAY', max(1, NPROC - max(2, NPROC // 7))))      # self-play processes; the rest serve matches, conversion and the trainer
SIDE = max(1, NPROC - NPLAY)
ENGINE = E('ENGINE', 'calc/calc_z5')
NODES = int(E('NODES', 5000)); TREE = int(E('TREE', 4)); FILE_GAMES = int(E('FILE_GAMES', 100))
MIN_NEW = int(E('MIN_NEW', 24000))                               # games a new shard must hold before the next network is trained
WINDOW = int(E('WINDOW', 6)); EPOCHS = float(E('EPOCHS', 2.0)); LAM = float(E('LAM', 0.5))
RATE_EVERY = int(E('RATE_EVERY', 3))
DEVICE = E('DEVICE') or ('cuda' if subprocess.run('python3 -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"', shell=True).returncode == 0 else 'cpu')
HIDDEN = lambda g: 128 if g <= 5 else (256 if g <= 11 else 384)
REC = np.dtype([('idx', '<i2', 32), ('score', '<i2'), ('result', 'i1'), ('ply', 'i1')])
LOG = open('zero/loop.log', 'a')


def log(msg):
    line = f'[{time.strftime("%H:%M:%S", time.gmtime())}] {msg}'
    print(line, flush=True); LOG.write(line + '\n'); LOG.flush()


def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return '\n'.join(l for l in (r.stdout + r.stderr).split('\n') if l and 'Warning' not in l and 'from_numpy' not in l and 'Consider using' not in l)


def state():
    return json.load(open('zero/cont_state.json'))


def save_state(st):
    json.dump(st, open('zero/cont_state.json.tmp', 'w')); os.replace('zero/cont_state.json.tmp', 'zero/cont_state.json')


def publish(gen):
    open('zero/current.tmp', 'w').write(f'{gen} zero/g{gen}.bin'); os.replace('zero/current.tmp', 'zero/current')


WORKER = r'''
i=$1
while [ ! -f zero/STOP ]; do
  read G NET < zero/current
  f=zero/pool/g${G}_w${i}_$(date +%%s%%N)
  %s $NET selfplay $f.tmp %d %d $(( ($(date +%%s%%N) / 1000 + i * 7919) %% 2000000000 )) 1 %d 1 > /dev/null 2>&1 && mv $f.tmp.o zero/opool/$(basename $f).o && mv $f.tmp $f.bin
  rm -f $f.tmp $f.tmp.o
done
'''


def start_workers():
    os.makedirs('zero/pool', exist_ok=True); os.makedirs('zero/opool', exist_ok=True)
    for f in glob.glob('zero/pool/*.tmp') + glob.glob('zero/pool/*.tmp.o'):
        os.remove(f)
    if os.path.exists('zero/STOP'):
        os.remove('zero/STOP')
    open('zero/worker.sh', 'w').write(WORKER % (ENGINE, FILE_GAMES, NODES, TREE))
    return [subprocess.Popen(['bash', 'zero/worker.sh', str(i)], start_new_session=True) for i in range(NPLAY)]


def stop_workers(procs):
    open('zero/STOP', 'w').write('stop')
    for p in procs:
        try:
            os.killpg(p.pid, signal.SIGTERM)
        except Exception:
            pass


def pause(flag):                                               # freeze / thaw the self-play engines (for clean timed ratings)
    subprocess.run(['pkill', '-STOP' if flag else '-CONT', '-x', os.path.basename(ENGINE)])


def target(q):
    sc = 1 / (1 + np.exp(-q['score'].astype(np.float32) / 271.58))
    return np.where(q['ply'] < 0, sc, LAM * sc + (1 - LAM) * (q['result'].astype(np.float32) + 1) / 2)


def make_shard(k, files):
    """finished game files -> one shuffled training shard; one file in twenty-five is held out, whole, for validation"""
    val_files = files[::25]; vs = set(val_files); tr_files = [f for f in files if f not in vs]
    r = np.concatenate([np.fromfile(f, dtype=REC) for f in tr_files])
    r = r[np.random.RandomState(k).permutation(len(r))]
    np.save(f'data/ev_sh{k}_x.npy', r['idx']); np.save(f'data/ev_sh{k}_y.npy', (target(r) * 255).round().astype(np.uint8))
    played = r[r['ply'] >= 0][:1_500_000]                      # played positions only, for the exact table solve
    np.save(f'data/ev_sp{k}_x.npy', played['idx']); np.save(f'data/ev_sp{k}_y.npy', (target(played) * 255).round().astype(np.uint8))
    v = np.concatenate([np.fromfile(f, dtype=REC) for f in val_files]); v = v[v['ply'] >= 0]
    np.save('data/ev_shval_x.npy', v['idx']); np.save('data/ev_shval_y.npy', target(v).astype(np.float32))
    for f in files:
        os.remove(f)
    return len(r), len(v)


if __name__ == '__main__':
    if not os.path.exists('zero/cont_state.json'):             # first start: continue from the newest rated generation of the stepwise loop
        g0 = max(int(k) for k in json.load(open('zero/rating.json')))
        save_state({'gen': g0, 'shards': [], 'next_shard': 0})
    st = state(); publish(st['gen'])
    log(f'continuous loop: generation {st["gen"]}, {NPLAY} self-play processes, {SIDE} side threads, training on {DEVICE}, {NODES} nodes per move')
    procs = start_workers()
    try:
        while not os.path.exists('zero/HALT'):
            files = sorted(glob.glob('zero/pool/*.bin'))
            if len(files) * FILE_GAMES < MIN_NEW:
                time.sleep(20); continue
            t0 = time.time(); g = st['gen'] + 1; k = st['next_shard']; h = HIDDEN(g)
            n, nv = make_shard(k, files)
            st['shards'] = (st['shards'] + [k])[-WINDOW:]; st['next_shard'] = k + 1
            save_state(st)                                           # a shard that exists is never made twice, whatever happens next
            for old in glob.glob('data/ev_sh*_x.npy') + glob.glob('data/ev_sp*_x.npy'):       # drop shards that left the window
                kk = old.split('ev_s')[1][1:].split('_')[0]
                if kk.isdigit() and int(kk) not in st['shards']:
                    os.remove(old); os.remove(old.replace('_x.npy', '_y.npy'))
            log(f'=== generation {g}: shard {k} with {n/1e6:.1f}M positions from {len(files) * FILE_GAMES} games ({nv} validation positions), hidden {h}')
            out = sh(f'CUDA_VISIBLE_DEVICES= python3 calc/solve_pst.py 400000 sp{k} zero/g{g}s')
            if not os.path.exists(f'zero/g{g}s.npz'):                # only refreshes the search's piece values: fall back to the previous ones
                log('table solve failed, keeping the previous piece values: ' + out[-160:])
                sh(f'cp zero/g{g-1}s.npz zero/g{g}s.npz; cp zero/g{g-1}s.bin zero/g{g}s.bin')
            prev = f'zero/g{g-1}.bin.ema.pt'
            tags = ','.join(f'sh{i}' for i in st['shards'])
            train_cmd = (f'python3 calc/train_nnue.py --h {h} --buckets 8 --ema 0.999 --files {tags} --val shval --epochs {EPOCHS} --lr 1e-3 '
                         f'--threads {SIDE} --device {DEVICE} --pst zero/g{g}s.npz {"--widen " + prev if os.path.exists(prev) else ""} --out zero/g{g}.bin')
            out = sh(train_cmd)
            if not os.path.exists(f'zero/g{g}.bin.ema.pt'):
                log('training failed, retrying once: ' + out[-200:]); time.sleep(30); out = sh(train_cmd)
            if not os.path.exists(f'zero/g{g}.bin.ema.pt'):
                log('training failed again; self-play continues and the next cycle retries: ' + out[-200:]); save_state(st); continue
            done = [l for l in out.split('\n') if l.startswith('done')]
            r = sh(f'python3 match.py --a calc:zero/g{g}.bin:4000:{ENGINE} --b calc:zero/g{g-1}.bin:4000:{ENGINE} --openings 100 --workers {SIDE}')
            try:
                d = json.loads(r.strip().split('\n')[-1])
                log(f'generation {g} vs {g-1}: {d["w"]}-{d["d"]}-{d["l"]}  score {d["score"]}  Elo {d["elo"]} ({d["lo"]} to {d["hi"]})')
            except Exception:
                log('match output: ' + r[-300:])
            st['gen'] = g; save_state(st); publish(g)
            if g % RATE_EVERY == 0:
                est = json.load(open('zero/rating.json')) if os.path.exists('zero/rating.json') else {}
                last = est[max(est, key=int)] if est else 2000
                pause(True)
                # rated by node count, not clock time, so the number means the same on any machine:
                # ours at 100,000 nodes per move against Stockfish levels at 100,000 nodes per move
                r = sh(f'python3 elo_ladder.py --player calc:zero/g{g}.bin:100000:{ENGINE} --sf_time 100000 --start {max(1400, min(3100, int(round((last + 100) / 100.0)) * 100))} '
                       f'--openings 20 --levels 3 --workers {min(NPROC, 40)} --out zero/elo_g{g}.json')
                pause(False)
                try:
                    d = json.loads(r.strip().split('\n')[-1]); est[str(g)] = d['rating']; json.dump(est, open('zero/rating.json', 'w'))
                    log(f'generation {g} rating {d["rating"]} +- {d["ci95"]} ({d["games"]} games, node-based)')
                except Exception:
                    log('ladder output: ' + r[-300:])
            log(f'generation {g} published after {time.time() - t0:.0f}s of training-side work; {done[0][:110] if done else ""}')
    finally:
        stop_workers(procs); pause(False)
        log('continuous loop stopped')
