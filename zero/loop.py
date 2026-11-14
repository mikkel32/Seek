"""The self-taught loop. Each generation:
   1. solve a piece-square table on the newest self-play data (exact, no training) -> piece values for the search
   2. train the evaluation network on the last three generations of data, starting from the previous network
      (wider networks start as exact copies of the narrower one)
   3. play the new generation against the previous one at an equal node budget
   4. self-play with the new generation to produce the next data set
No position, evaluation or game from any other engine enters this loop.

  python3 zero/loop.py <first generation> <last generation>
"""
import json, os, subprocess, sys, time

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
#        hidden units, self-play games per process, nodes per move, weight of the search score in the target
SCHED = {2: (0, 6000, 4000, 0.4), 3: (64, 6000, 4000, 0.5), 4: (128, 6000, 5000, 0.5), 5: (128, 6000, 5000, 0.5),
         6: (256, 6000, 6000, 0.5), 7: (256, 6000, 6000, 0.5), 8: (256, 6000, 6000, 0.5), 9: (256, 6000, 6000, 0.5),
         10: (384, 6000, 8000, 0.5), 11: (384, 6000, 8000, 0.5), 12: (384, 6000, 8000, 0.5), 13: (512, 6000, 8000, 0.5), 14: (512, 6000, 8000, 0.5)}
ENGINE = os.environ.get('ENGINE', 'calc/calc_z3')        # the binary used for self-play, matches and ratings
CHUNKS = 6                                               # self-play is cut into chunks so an interrupted run loses little
NPROC = int(os.environ.get('NPROC', os.cpu_count() or 4))   # self-play processes and match workers: set to the number of cores
DEVICE = os.environ.get('DEVICE') or ('cuda' if subprocess.run('python3 -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"', shell=True).returncode == 0 else 'cpu')   # where the network trains
LOG = open('zero/loop.log', 'a')


def sh(cmd, quiet=False):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    out = '\n'.join(l for l in (r.stdout + r.stderr).split('\n') if l and 'Warning' not in l and 'from_numpy' not in l and 'Consider using' not in l)
    if not quiet:
        log(out[-900:])
    return out


def log(msg):
    line = f'[{time.strftime("%H:%M:%S", time.gmtime())}] {msg}'
    print(line, flush=True); LOG.write(line + '\n'); LOG.flush()


for g in range(int(sys.argv[1]), int(sys.argv[2]) + 1):
    h, games, nodes, lam = SCHED[g]
    games = int(games * float(os.environ.get('GAMES_MULT', 1)))      # more games per generation on a bigger machine
    if os.path.exists(f'zero/g{g}.done'):
        continue
    log(f'=== generation {g}: hidden {h}')
    # 1. exact solve on the newest data
    if not os.path.exists(f'zero/g{g}s.bin'):
        tag = f'z{g-1}' if os.path.exists(f'data/ev_z{g-1}_x.npy') else f'z{g-1}t'      # played positions if kept, else played + tree
        sh(f'python3 calc/solve_pst.py 1500000 {tag} zero/g{g}s')
        if not os.path.exists(f'zero/g{g}s.npz'):
            log('the table solve failed; stopping'); sys.exit(1)
    if h == 0:
        sh(f'cp zero/g{g}s.bin zero/g{g}.bin')
    elif not os.path.exists(f'zero/g{g}.bin'):
        prev = f'zero/g{g-1}.bin.ema.pt'
        init = f'--widen {prev}' if os.path.exists(prev) else ''
        # a network whose averaged weights file exists has finished training: an interrupted run does not train it again
        tr = lambda tags, out: None if os.path.exists(out + '.ema.pt') else sh(f'python3 calc/train_nnue.py --h {h} --buckets 8 --ema 0.999 --files {tags} --val z{g-1}v --epochs 5 --lr 1e-3 --threads {min(NPROC, 32)} --device {DEVICE} '
                                  f'--pst zero/g{g}s.npz {init} --out {out}')
        gens = [i for i in range(max(1, g - 3), g) if os.path.exists(f'data/ev_z{i}t_x.npy') or os.path.exists(f'data/ev_z{i}_x.npy')]
        has_tree = lambda i: os.path.exists(f'data/ev_z{i}t_x.npy')
        plain = ','.join(f'z{i}' for i in gens); withtree = ','.join(f'z{i}t' if has_tree(i) else f'z{i}' for i in gens)
        if not os.path.exists('zero/tree.json') and any(has_tree(i) for i in gens):
            # one-off experiment: do the extra positions harvested from inside the searches make a stronger engine?
            tr(plain, f'zero/g{g}a.bin'); tr(withtree, f'zero/g{g}b.bin')
            r = sh(f'python3 match.py --a calc:zero/g{g}b.bin:4000:{ENGINE} --b calc:zero/g{g}a.bin:4000:{ENGINE} --openings 200 --workers {NPROC}', quiet=True)
            d = json.loads(r.strip().split('\n')[-1])
            log(f'tree-harvest experiment: with tree positions vs without: {d["w"]}-{d["d"]}-{d["l"]}  score {d["score"]}  Elo {d["elo"]} ({d["lo"]} to {d["hi"]})')
            use = d['score'] > 0.5
            json.dump({'use_tree': bool(use), 'result': d}, open('zero/tree.json', 'w'))
            w = 'b' if use else 'a'
            sh(f'cp zero/g{g}{w}.bin zero/g{g}.bin; cp zero/g{g}{w}.bin.ema.pt zero/g{g}.bin.ema.pt', quiet=True)
        else:
            cfg = json.load(open('zero/tree.json')) if os.path.exists('zero/tree.json') else {'use_tree': False}
            if cfg['use_tree'] and 'exp8' not in cfg and os.path.exists(f'data/ev_z{g-1}h_x.npy'):
                # one-off experiment: eight harvested positions per move against four, same games
                four = ','.join(f'z{i}h' if os.path.exists(f'data/ev_z{i}h_x.npy') else (f'z{i}t' if has_tree(i) else f'z{i}') for i in gens)
                tr(four, f'zero/g{g}a.bin'); tr(withtree, f'zero/g{g}b.bin')
                r = sh(f'python3 match.py --a calc:zero/g{g}b.bin:4000:{ENGINE} --b calc:zero/g{g}a.bin:4000:{ENGINE} --openings 200 --workers {NPROC}', quiet=True)
                d = json.loads(r.strip().split('\n')[-1])
                log(f'harvest experiment: eight per move vs four per move: {d["w"]}-{d["d"]}-{d["l"]}  score {d["score"]}  Elo {d["elo"]} ({d["lo"]} to {d["hi"]})')
                cfg['exp8'] = d; cfg['tree_n'] = 8 if d['score'] > 0.5 else 4
                json.dump(cfg, open('zero/tree.json', 'w'))
                w = 'b' if d['score'] > 0.5 else 'a'
                sh(f'cp zero/g{g}{w}.bin zero/g{g}.bin; cp zero/g{g}{w}.bin.ema.pt zero/g{g}.bin.ema.pt', quiet=True)
            else:
                if cfg.get('tree_n', 4) == 4:               # settled on four per move: use the four-per-move subset where a generation harvested more
                    withtree = ','.join(f'z{i}h' if os.path.exists(f'data/ev_z{i}h_x.npy') else (f'z{i}t' if has_tree(i) else f'z{i}') for i in gens)
                tr(withtree if cfg['use_tree'] else plain, f'zero/g{g}.bin')
    if not os.path.exists(f'zero/g{g}.bin') or os.path.getsize(f'zero/g{g}.bin') < 1000:
        log(f'no weights for generation {g}; stopping'); sys.exit(1)
    if os.path.exists(f'zero/elo_g{g}.json'):
        log(f'generation {g} already matched and rated')
    else:
      # 3. new generation against the previous one
      r = sh(f'python3 match.py --a calc:zero/g{g}.bin:4000:{ENGINE} --b calc:zero/g{g-1}.bin:4000:{ENGINE} --openings 100 --workers {NPROC}', quiet=True)
      try:
          d = json.loads(r.strip().split('\n')[-1])
          log(f'generation {g} vs {g-1}: {d["w"]}-{d["d"]}-{d["l"]}  score {d["score"]}  Elo {d["elo"]} ({d["lo"]} to {d["hi"]})')
      except Exception:
          log('match output: ' + r[-300:])
      # absolute rating against strength-limited Stockfish (measurement only; nothing from these games is learned from)
      est = json.load(open('zero/rating.json')) if os.path.exists('zero/rating.json') else {}
      start = int(round((est.get(str(g - 1), 2000) + 100) / 100.0)) * 100
      r = sh(f'python3 elo_ladder.py --player calc:zero/g{g}.bin:0.1:{ENGINE} --start {max(1400, min(3100, start))} --openings 20 --levels 3 --workers {NPROC} --out zero/elo_g{g}.json', quiet=True)
      try:
          d = json.loads(r.strip().split('\n')[-1]); est[str(g)] = d['rating']; json.dump(est, open('zero/rating.json', 'w'))
          log(f'generation {g} rating {d["rating"]} +- {d["ci95"]} ({d["games"]} games)')
      except Exception:
          log('ladder output: ' + r[-300:])
    # 4. self-play
    if not os.path.exists(f'data/ev_z{g}v_x.npy'):
        for c in range(CHUNKS):
            if not os.path.exists(f'zero/g{g}_c{c}.ok'):
                cfg = json.load(open('zero/tree.json')) if os.path.exists('zero/tree.json') else {}
                tree_n = cfg.get('tree_n', 8 if 'exp8' not in cfg else 4)      # harvest eight per move until the experiment has decided
                per_proc = max(1, games * 4 // CHUNKS // NPROC)             # the schedule's game count is per process on 4 processes
                sh(f'ENGINE={ENGINE} NPROC={NPROC} zero/gen.sh {g} zero/g{g}.bin {per_proc} {nodes} 1 {tree_n} {c}')
                sizes = [os.path.getsize(f'zero/g{g}_c{c}_p{i}.bin') if os.path.exists(f'zero/g{g}_c{c}_p{i}.bin') else 0 for i in range(NPROC)]
                if min(sizes) < 200_000:                       # a step that produced no games must never be marked done
                    log(f'self-play chunk {c} produced no data (sizes {sizes}); stopping'); sys.exit(1)
                open(f'zero/g{g}_c{c}.ok', 'w').write('ok')
        sh(f'python3 zero/data.py {g} {lam}')
        if not os.path.exists(f'data/ev_z{g}t_x.npy'):
            log('data conversion failed; stopping'); sys.exit(1)
        sh(f'rm -f zero/g{g}_c*_p*.bin', quiet=True)                                          # raw records are no longer needed once converted
    open(f'zero/g{g}.done', 'w').write('ok')
log('loop finished')
