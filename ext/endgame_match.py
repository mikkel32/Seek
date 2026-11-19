"""Play a clock match for Syzygy-enabled Seek, with paired openings and full result logs.

Pass four absolute paths for the baseline engine, the candidate, their shared weights,
and Stockfish 14.1. Stockfish has the same tables in both external match arms.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import tempfile

from scipy.stats import t
from readout_arena import OPTIONS, ROOT, eligible_book, opening_key, paired_estimate


def play(label, challenger, opponent, openings, workers, args):
    games = args.output / (label + '.jsonl')
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        (work/'data').mkdir()
        (work/'data/openings.txt').write_text('\n'.join(openings)+'\n')
        for name in ('parent', 'candidate', 'weights', 'fish', 'tables'):
            (work/name).symlink_to(getattr(args, name))
        seek_options = OPTIONS + (',SyzygyPath=tables' if challenger == 'candidate' else '')
        opponent_spec = ('Stockfish 14.1|./fish|Threads=1,Hash=64,Use NNUE=true,SyzygyPath=tables'
                         if opponent == 'fish' else 'parent|./parent weights|' + OPTIONS)
        cmd = [sys.executable, str(ROOT/'gauntlet.py'), '--us', './'+challenger+' weights',
               '--us_opts', seek_options, '--opp', opponent_spec, '--tc', args.tc,
               '--openings', str(len(openings)), '--workers', str(workers), '--seed', '29481']
        r = subprocess.run(cmd, cwd=work, check=True, capture_output=True, text=True,
                           env={**os.environ, 'GAMES_LOG': str(games)})
    raw = next(json.loads(l) for l in reversed(r.stdout.splitlines()) if l.startswith('{'))
    estimate = paired_estimate(games, len(openings))
    scores = {opening_key(fen): [None, None] for fen in openings}
    for line in games.read_text().splitlines():
        g = json.loads(line)
        scores[opening_key(g['fen'])][0 if g['us_white'] else 1] = g['score']
    return {'summary': raw, 'paired': estimate, 'scores': list(scores.values())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('parent','candidate','weights','fish','tables','output'):
        p.add_argument(key, type=Path)
    p.add_argument('--tc', default='10+0.1')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--self-pairs', type=int, default=160)
    p.add_argument('--fish-pairs', type=int, default=100)
    args = p.parse_args()
    for name in ('parent','candidate','weights','fish','tables'):
        setattr(args, name, getattr(args,name).resolve(strict=True))
    args.output = args.output.resolve()
    archive = ROOT/'zero/results/games/games_bigs.jsonl.sc.jsonl.gz'
    book = eligible_book(archive)
    if args.workers < 2 or min(args.self_pairs,args.fish_pairs) < 2 or args.self_pairs+args.fish_pairs > len(book):
        p.error('Need at least two workers and enough distinct openings for both matches')
    random.Random(163817).shuffle(book)
    args.output.mkdir(parents=True,exist_ok=False)
    own = book[:args.self_pairs]
    against_fish = book[args.self_pairs:args.self_pairs+args.fish_pairs]
    (args.output/'openings.json').write_text(json.dumps({'self':own,'fish':against_fish})+'\n')
    result = {'self': play('self','candidate','parent',own,args.workers,args)}
    print(json.dumps({'self':result['self']}),flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = {name:pool.submit(play,name,name,'fish',against_fish,args.workers//2,args)
                   for name in ('parent','candidate')}
        for name, job in pending.items():
            result[name] = job.result()
            print(json.dumps({name:result[name]}),flush=True)
    diff = [(sum(c) - sum(b)) / 2 for c, b in zip(result['candidate']['scores'],result['parent']['scores'])]
    mean = statistics.mean(diff)
    half = t.ppf(.975,len(diff)-1)*statistics.stdev(diff)/math.sqrt(len(diff))
    result['score_difference'] = [mean,mean-half,mean+half]
    print(json.dumps({'score_difference':result['score_difference']}),flush=True)
    (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__':
    main()
