"""Self-play that never stops, on one machine: the data side of the production loop (zero/prod2.sh trains and publishes).

Like zero/farm.py, with what a rolling loop needs: shards are numbered on from the highest one present (old ones may be
deleted by the loop); the network file is read afresh by every worker run (200 games), so a newly published network is
picked up within minutes; `touch zero/FARM_PAUSE` freezes all workers (for matches that need the processors) until the
file is removed; and the farm waits by itself while less than 2.5 GB of disk is free.

Original description of the farm:
Fresh training positions for a wider evaluator, made on one machine without the production loop.

Self-play workers (the engine's own self-play mode, with the reliability read-out steering its pruning) write raw game
files; whenever enough have finished they become one shuffled shard in the compact 24-byte format with the production
labels (lambda-return 0.95 for played positions, the search score for positions from inside the trees), and the raw files
are deleted. One file in 25 of the first shard is held out, whole, as validation data.

  python3 zero/farm2.py <engine binary> <network> <workers> <shards to make> [files per shard=240] [positions searched per move=2500]
  -> data/ev_b<N>_c.npy, data/ev_b<N>_y.npy, data/ev_bval_x.npy, data/ev_bval_y.npy          stop early: touch zero/FARM_STOP
"""
import glob, os, re, shutil, signal, subprocess, sys, threading, time
import numpy as np
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')); sys.path.insert(0, 'zero')
from compact import encode

ENGINE, NET, W, NSH = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
PER = int(sys.argv[5]) if len(sys.argv) > 5 else 240; NODES = int(sys.argv[6]) if len(sys.argv) > 6 else 2500
REC = np.dtype([('idx', '<i2', 32), ('score', '<i2'), ('result', 'i1'), ('ply', 'i1')]); LAM = 0.5; LAMBDA = 0.95
exec(re.search(r'(def target\(q\):.*?)(?=\ndef make_shard)', open('zero/cont.py').read(), re.S).group(1))      # target(), lam_targets(): the loop's own label code
os.makedirs('/tmp/sp', exist_ok=True); os.makedirs('data', exist_ok=True)
worker = f'i=$1; while [ ! -f zero/FARM_STOP ]; do f=/tmp/sp/w${{i}}_$(date +%s%N); {ENGINE} {NET} selfplay $f.tmp 200 {NODES} $(( ($(date +%s%N) / 1000 + i * 7919) % 2000000000 )) 1 4 0 > /dev/null 2>&1 && mv $f.tmp $f.bin; rm -f $f.tmp; done'
procs = [subprocess.Popen(['bash', '-c', worker, 'w', str(i)], start_new_session=True) for i in range(W)]
num = lambda f: int(re.search(r'ev_b(\d+)_c', f).group(1))
done = max([num(f) for f in glob.glob('data/ev_b[0-9]*_c.npy')] + [-1]) + 1; NSH += done; t0 = time.time(); total = 0
def watcher():
    """freeze or thaw every worker (its shell and its engine) within a second of zero/FARM_PAUSE appearing or going; the main
    loop may be busy for minutes writing a shard, and a match that needs the processors cannot wait for it"""
    paused = False
    while True:
        want = os.path.exists('zero/FARM_PAUSE') or shutil.disk_usage('.').free < 2.5e9
        if want != paused:
            for p in procs:
                try:
                    os.killpg(p.pid, signal.SIGSTOP if want else signal.SIGCONT)
                except Exception:
                    pass
            paused = want; print(f'[{time.strftime("%H:%M:%S", time.gmtime())}] workers {"paused" if paused else "running again"}', flush=True)
        time.sleep(1)


threading.Thread(target=watcher, daemon=True).start()
print(f'{W} self-play workers on {NET}, {NODES} positions per move; {PER} files a shard; shards {done} -> {NSH}', flush=True)
try:
    while done < NSH and not os.path.exists('zero/FARM_STOP'):
        files = sorted(glob.glob('/tmp/sp/*.bin'), key=os.path.getmtime)
        if len(files) < PER:
            time.sleep(5); continue
        files = files[:PER]; val = files[::25] if not os.path.exists('data/ev_bval_x.npy') else []; vs = set(val)
        parts = [np.fromfile(f, dtype=REC) for f in files if f not in vs]
        tg = np.concatenate([lam_targets(p, LAMBDA) for p in parts]); r = np.concatenate(parts); perm = np.random.RandomState(done).permutation(len(r))
        out = np.lib.format.open_memmap(f'data/ev_b{done}_c.npy.tmp', mode='w+', dtype=np.uint8, shape=(len(r), 24))
        for s in range(0, len(r), 2_000_000):
            out[s:s + 2_000_000] = encode(r['idx'][perm[s:s + 2_000_000]])
        out.flush(); del out; np.save(f'data/ev_b{done}_y.npy', (tg[perm] * 255).round().astype(np.uint8)); os.replace(f'data/ev_b{done}_c.npy.tmp', f'data/ev_b{done}_c.npy')
        if val:
            v = np.concatenate([np.fromfile(f, dtype=REC) for f in val]); v = v[v['ply'] >= 0]
            np.save('data/ev_bval_x.npy', v['idx']); np.save('data/ev_bval_y.npy', target(v).astype(np.float32))
        for f in files:
            os.remove(f)
        total += len(r); done += 1
        print(f'[{time.strftime("%H:%M:%S", time.gmtime())}] shard {done}/{NSH}: {len(r)/1e6:.1f}M positions ({np.mean(r["ply"] >= 0):.0%} played); {total/(time.time()-t0)/1e3:.0f}k positions a second so far', flush=True)
finally:
    open('zero/FARM_STOP', 'w').write('stop')
    for p in procs:
        try:
            os.killpg(p.pid, signal.SIGCONT); os.killpg(p.pid, 15)
        except Exception:
            pass
    time.sleep(1); os.remove('zero/FARM_STOP')
    for f in glob.glob('/tmp/sp/*'):
        os.remove(f)
    print('farm stopped', flush=True)
