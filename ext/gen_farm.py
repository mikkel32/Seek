"""Self-play with a finished network on idle processors: the generation test for the outside-data line.

The engine's own self-play mode (zero/farm2.py's workers) with the network given, written to shards of their own:
positions that were played go to data/ev_g<N>, positions harvested from inside the search trees to data/ev_u<N>, both in
the compact 24-byte format with the self-taught loop's labels (played: lambda-return 0.95 over the later search scores of
the game, ending in its result; tree positions: their search score). One raw file in 25 of the first shard is held out
(data/ev_gval_*). `touch zero/GEN_PAUSE` freezes the workers, `touch zero/GEN_STOP` ends the run.

  python3 ext/gen_farm.py <engine binary> <network> <workers> <shards to make> [files per shard=240] [positions searched per move=5000]
"""
import glob, os, re, shutil, signal, subprocess, sys, threading, time
import numpy as np
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')); sys.path.insert(0, 'zero')
from compact import encode

ENGINE, NET, W, NSH = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
PER = int(sys.argv[5]) if len(sys.argv) > 5 else 240; NODES = int(sys.argv[6]) if len(sys.argv) > 6 else 5000
REC = np.dtype([('idx', '<i2', 32), ('score', '<i2'), ('result', 'i1'), ('ply', 'i1')]); LAM = 0.5; LAMBDA = 0.95
exec(re.search(r'(def target\(q\):.*?)(?=\ndef make_shard)', open('zero/cont.py').read(), re.S).group(1))      # target(), lam_targets(): the loop's own label code
TMP = '/tmp/spg'; os.makedirs(TMP, exist_ok=True); os.makedirs('data', exist_ok=True)
worker = (f'i=$1; while [ ! -f zero/GEN_STOP ]; do f={TMP}/w${{i}}_$(date +%s%N); {ENGINE} {NET} selfplay $f.tmp 200 {NODES} '
          f'$(( ($(date +%s%N) / 1000 + i * 7919) % 2000000000 )) 1 4 0 > /dev/null 2>&1 && mv $f.tmp $f.bin; done')
procs = [subprocess.Popen(['bash', '-c', worker, 'w', str(i)], start_new_session=True) for i in range(W)]
num = lambda f: int(re.search(r'ev_g(\d+)_c', f).group(1))
done = max([num(f) for f in glob.glob('data/ev_g[0-9]*_c.npy')] + [-1]) + 1; NSH += done; t0 = time.time(); total = 0


def watcher():
    paused = False
    while True:
        want = os.path.exists('zero/GEN_PAUSE') or shutil.disk_usage('.').free < 4e9
        if want != paused:
            for p in procs:
                try:
                    os.killpg(p.pid, signal.SIGSTOP if want else signal.SIGCONT)
                except Exception:
                    pass
            paused = want; print(f'[{time.strftime("%H:%M:%S", time.gmtime())}] workers {"paused" if paused else "running again"}', flush=True)
        time.sleep(1)


def write(tag, n, idx, y):
    out = np.lib.format.open_memmap(f'data/ev_{tag}{n}_c.npy.tmp', mode='w+', dtype=np.uint8, shape=(len(idx), 24))
    for s in range(0, len(idx), 2_000_000):
        out[s:s + 2_000_000] = encode(idx[s:s + 2_000_000])
    out.flush(); del out; np.save(f'data/ev_{tag}{n}_y.npy', (y * 255).round().astype(np.uint8)); os.replace(f'data/ev_{tag}{n}_c.npy.tmp', f'data/ev_{tag}{n}_c.npy')


threading.Thread(target=watcher, daemon=True).start()
print(f'{W} self-play workers on {NET}, {NODES} positions per move; {PER} files a shard; shards {done} -> {NSH}', flush=True)
try:
    while done < NSH and not os.path.exists('zero/GEN_STOP'):
        files = sorted(glob.glob(f'{TMP}/*.bin'), key=os.path.getmtime)
        if len(files) < PER:
            time.sleep(5); continue
        files = files[:PER]; val = files[::25] if not os.path.exists('data/ev_gval_x.npy') else []; vs = set(val)
        parts = [np.fromfile(f, dtype=REC) for f in files if f not in vs]
        tg = np.concatenate([lam_targets(p, LAMBDA) for p in parts]); r = np.concatenate(parts); perm = np.random.RandomState(done).permutation(len(r))
        r = r[perm]; tg = tg[perm]; pl = r['ply'] >= 0
        write('g', done, r['idx'][pl], tg[pl]); write('u', done, r['idx'][~pl], tg[~pl])
        if val:
            v = np.concatenate([np.fromfile(f, dtype=REC) for f in val]); v = v[v['ply'] >= 0]
            np.save('data/ev_gval_x.npy', v['idx']); np.save('data/ev_gval_y.npy', target(v).astype(np.float32))
        for f in files:
            os.remove(f)
        total += len(r); done += 1
        print(f'[{time.strftime("%H:%M:%S", time.gmtime())}] shard {done}/{NSH}: {pl.sum()/1e6:.2f}M played and {(~pl).sum()/1e6:.2f}M tree positions; '
              f'{total/(time.time()-t0)/1e3:.1f}k positions a second so far', flush=True)
finally:
    open('zero/GEN_STOP', 'w').write('stop')
    for p in procs:
        try:
            os.killpg(p.pid, signal.SIGCONT); os.killpg(p.pid, 15)
        except Exception:
            pass
    time.sleep(1); os.remove('zero/GEN_STOP')
    for f in glob.glob(f'{TMP}/*'):
        os.remove(f)
    print('generation stopped', flush=True)
