"""Compare two Seek builds or networks against Stockfish on identical openings."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile

from scipy.stats import t
from readout_arena import OPTIONS, ROOT, eligible_book, opening_key, paired_estimate


def pair_means(path):
    scores = defaultdict(list)
    for line in path.read_text().splitlines():
        g = json.loads(line)
        scores[opening_key(g['fen'])].append(g['score'])
    return {fen: statistics.mean(pair) for fen, pair in scores.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ('baseline', 'trained', 'weights', 'stockfish', 'output'):
        p.add_argument(arg, type=Path)
    p.add_argument('--trained-weights', type=Path, help='use a different network for the trained arm')
    p.add_argument('--eval-scale', type=int, default=100, help='same evaluation scale for both arms')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--pairs', type=int, default=100)
    control = p.add_mutually_exclusive_group()
    control.add_argument('--tc', default='10+0.1')
    control.add_argument('--nodes', type=int, help='Fixed search nodes per move for each engine')
    p.add_argument('--offset', type=int, default=2400)
    p.add_argument('--pool-size', type=int, default=570)
    p.add_argument('--seed', type=int, default=403287)
    a = p.parse_args()
    if a.eval_scale <= 0:
        p.error('Evaluation scale must be positive')
    if a.workers < 2 or not 2 <= a.pairs <= a.pool_size or a.offset < 0:
        p.error('Use two or more workers and a nonnegative book offset with sufficient opening pairs')
    if a.nodes is not None and a.nodes <= 0:
        p.error('Node budget must be positive')
    out = a.output.resolve()
    archive = ROOT / 'zero/results/games/games_bigs.jsonl.sc.jsonl.gz'
    book = eligible_book(archive)[a.offset:a.offset + a.pool_size]
    if len(book) != a.pool_size:
        raise ValueError('Unexpected opening book')
    out.mkdir(exist_ok=False, parents=True)
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        (work/'data').mkdir()
        (work/'data/openings.txt').write_text('\n'.join(book)+'\n')
        for label in ('baseline','trained','stockfish','weights'):
            (work/label).symlink_to(getattr(a,label).resolve(strict=True))
        (work/'trained-weights').symlink_to((a.trained_weights or a.weights).resolve(strict=True))
        def run(label):
            log = out/(label+'.jsonl')
            net = 'trained-weights' if label == 'trained' else 'weights'
            cmd = [sys.executable, str(ROOT/'gauntlet.py'), '--us', f'./{label} {net}',
                   '--us_opts', f'{OPTIONS},EvalScale={a.eval_scale}',
                   '--opp', 'Stockfish 14.1|./stockfish|Threads=1,Hash=64,Use NNUE=true',
                   '--openings', str(a.pairs), '--seed', str(a.seed), '--workers', str(a.workers//2)]
            cmd += ['--nodes', str(a.nodes)] if a.nodes is not None else ['--tc', a.tc]
            r = subprocess.run(cmd, cwd=work, env={**os.environ,'GAMES_LOG':str(log)}, capture_output=True, text=True, check=True)
            raw = [json.loads(line) for line in r.stdout.splitlines() if line.startswith('{')][-1]
            return [raw, paired_estimate(log,a.pairs)]
        with ThreadPoolExecutor(max_workers=2) as pool:
            result = dict(zip(('baseline','trained'),pool.map(run,('baseline','trained'))))
    old,new = [pair_means(out/(name+'.jsonl')) for name in ('baseline','trained')]
    if old.keys()!=new.keys():
        raise ValueError('Arms played different openings')
    diff = [new[k]-v for k,v in old.items()]
    mean = statistics.mean(diff)
    half = t.ppf(.975,len(diff)-1)*statistics.stdev(diff)/math.sqrt(len(diff))
    result['trained_minus_baseline_points_per_game'] = [mean,mean-half,mean+half]
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
