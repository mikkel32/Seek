"""Production with a long memory, on the machine with the big disk.

Its own self-play farm (zero/farm2.py, shards data/ev_b<N>) plus the shards the lab machine sends (data/ev_l<N>,
zero/ship.sh). Nothing is deleted until the data pass a disk budget (oldest first). Every STEP new shards the next
generation is trained from the previous one on the FRESH newest shards plus as many drawn at random from everything
older: new play is always in, and the history is revisited without the training time growing with it.

  python3 zero/vprod.py <starting network .bin with its .pt>      W=22 STEP=4 FRESH=24 EP=0.5 BUDGET_GB=190     stop: touch zero/PROD_STOP
"""
import glob, json, os, random, shutil, subprocess, sys, time
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
E = os.environ.get; W = int(E('W', 22)); STEP = int(E('STEP', 4)); FRESH = int(E('FRESH', 24)); EP = float(E('EP', 0.5)); BUDGET = float(E('BUDGET_GB', 190)) * 1e9


def say(msg):
    open('vprod.log', 'a').write(f'[{time.strftime("%H:%M:%S", time.gmtime())}] {msg}\n')


def shards():
    """finished shards (positions and labels both present), oldest first"""
    fs = [f for f in glob.glob('data/ev_[bl][0-9]*_c.npy') if os.path.exists(f.replace('_c.npy', '_y.npy'))]
    return sorted(fs, key=os.path.getmtime)


tag = lambda f: os.path.basename(f)[3:-6]
if not os.path.exists('zero/cur.bin'):
    shutil.copy(sys.argv[1], 'zero/cur.bin'); shutil.copy(sys.argv[1] + '.pt', 'zero/cur.pt')
st = json.load(open('zero/vprod.json')) if os.path.exists('zero/vprod.json') else {'gen': 0, 'seen': []}
seen = set(st['seen']); g = st['gen']; rs = random.Random(g + 1)
farm = subprocess.Popen(['python3', 'zero/farm2.py', 'calc/calc_sp', 'zero/cur.bin', str(W), '1000000'], stdout=open('farm2.log', 'a'), stderr=subprocess.STDOUT)
say(f'production started at generation {g}: {W} workers here, a generation every {STEP} new shards, trained on the {FRESH} newest plus {FRESH} drawn from the older ones, {EP} passes, data kept up to {BUDGET/1e9:.0f} GB')
try:
    while not os.path.exists('zero/PROD_STOP'):
        time.sleep(20); S = shards()
        size = lambda f: os.path.getsize(f) + os.path.getsize(f.replace('_c.npy', '_y.npy'))
        while S and sum(size(f) for f in S) > BUDGET:                       # the disk budget: the oldest shard goes
            os.remove(S[0]); os.remove(S[0].replace('_c.npy', '_y.npy')); S = S[1:]
        if len([f for f in S if f not in seen]) < STEP or len(S) < FRESH:        # not before a full set of fresh shards is here: a few shards would only spoil the network
            continue
        fresh = S[-FRESH:]; older = S[:-FRESH]; pick = rs.sample(older, min(FRESH, len(older))); g += 1; t0 = time.time()
        cmd = (f'python3 calc/train_nnue.py --h 384 --buckets 8 --ema 0.999 --files {",".join(tag(f) for f in pick + fresh)} --val bval --epochs {EP} --lr 1e-3 --threads 4 '
               f'--device cuda --pst zero/g101s.npz --widen zero/cur.pt --out zero/v{g}.bin > zero/v{g}.train.log 2>&1')
        subprocess.run(cmd, shell=True); done = [l for l in open(f'zero/v{g}.train.log') if l.startswith('done')]
        if done:
            shutil.copy(f'zero/v{g}.bin', 'zero/cur.bin.tmp'); os.replace('zero/cur.bin.tmp', 'zero/cur.bin'); shutil.copy(f'zero/v{g}.bin.pt', 'zero/cur.pt')
            seen.update(S); json.dump({'gen': g, 'seen': sorted(seen & set(S))}, open('zero/vprod.json', 'w'))
            val = done[-1].split('averaged (')[1].split(',')[0] if 'averaged (' in done[-1] else '?'
            say(f'generation {g}: {len(S)} shards kept ({sum(size(f) for f in S)/1e9:.0f} GB, {sum(os.path.getsize(f) for f in S)/24e9:.2f}B positions), trained on {len(fresh)} newest + {len(pick)} older; played held-out {val[:8]}; {time.time() - t0:.0f}s')
            for old in range(1, g - 6):                                       # keep the last few and every tenth generation
                if old % 10:
                    for f in glob.glob(f'zero/v{old}.*'):
                        os.remove(f)
        else:
            say(f'generation {g}: training failed'); g -= 1; time.sleep(60)
finally:
    open('zero/FARM_STOP', 'w').write('stop'); farm.wait(); say('production stopped')
