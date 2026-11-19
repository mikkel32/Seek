"""Collect complete iterative-deepening traces, splitting entire openings before training."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
from pathlib import Path
import random
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ext"))
from readout_arena import OPTIONS
from weight_edit_cases import roots


def collect(job):
    engine, network, positions, nodes, scale = job
    commands = ['uci']
    for option in (OPTIONS + f',EvalScale={scale}').split(','):
        key, value = option.split('=')
        commands.append(f'setoption name {key} value {value}')
    commands.append('isready')
    for _, _, fen, _ in positions:
        commands.extend(['ucinewgame', 'isready', 'position fen '+fen, f'go nodes {nodes}'])
    commands.append('quit')
    result = subprocess.run([engine, network], input='\n'.join(commands)+'\n', capture_output=True, text=True, check=True)
    records = []
    moves = None
    for line in result.stdout.splitlines():
        if line.startswith('trstart '):
            parts = line.split()
            moves = parts[2:]
            if len(moves) != int(parts[1]) or len(set(moves)) != len(moves):
                raise ValueError('Invalid legal root vocabulary')
            history, leaders, orders, logits = [], [], [], {}
        elif line.startswith('trstep '):
            _, depth, leader, payload = line.split()
            if int(depth) != len(history)+1:
                raise ValueError('Nonconsecutive completed depths')
            t = np.frombuffer(bytes.fromhex(payload), dtype=np.uint8).reshape(len(moves),7)
            if t[int(leader),0] != 2 or np.count_nonzero(t[:,0]==2)!=1:
                raise ValueError('Leader token does not match pointer')
            history.append(t)
            leaders.append(int(leader))
        elif line.startswith('trbase '):
            orders.append(tuple(map(int,line.split()[2:])))
        elif line.startswith('trlog '):
            parts=line.split()
            logits[int(parts[1])]=np.array([tuple(map(int,item.split(','))) for item in parts[2:]],np.int32)
        elif line.startswith('bestmove '):
            if moves is None or not history:
                raise ValueError('Missing search trace')
            records.append((np.stack(history), np.asarray(leaders, np.uint8), ' '.join(moves), orders, logits))
            moves = None
    if len(records)!=len(positions):
        raise ValueError('Incomplete trace collection')
    return list(zip(positions,records))


def select_positions(archive, counts, seed):
    pools = roots(archive)
    rng = random.Random(seed)
    positions = []
    for name, limit in zip(('train','tune','test'), counts):
        part = [p for p in pools if p[0]==name]
        positions.extend(rng.sample(sorted(part), min(limit,len(part))))
    return positions


def main(args):
    positions = select_positions(args.archive, args.counts, args.seed)
    # Limit each child process's stdout buffer without restarting the engine for every board.
    jobs = [(args.engine.resolve(), args.network.resolve(), positions[i:i+24], args.nodes, args.scale)
            for i in range(0,len(positions),24)]
    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for batch in pool.map(collect,jobs): records.extend(batch)
    samples, leaders, moves, fens, groups, splits, dims = [], [], [], [], [], [], []
    for (split, group, fen, _),(x,y,vocabulary,_,_) in records:
        if len(y)<4 or x.shape[1]<2:continue
        samples.append(x.reshape(-1,7));leaders.append(y);moves.append(vocabulary)
        fens.append(fen);groups.append(group);splits.append(split);dims.append(x.shape[:2])
    with args.output.open('xb') as f:
        np.savez_compressed(f, tokens=np.concatenate(samples), leaders=np.concatenate(leaders), moves=moves,
                            fens=fens, groups=groups, splits=splits, dims=dims, nodes=args.nodes,
                            scale=args.scale, network_sha=hashlib.sha256(args.network.read_bytes()).hexdigest())
    print({k:splits.count(k) for k in set(splits)}, 'root move observations',sum(len(x) for x in samples),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('engine','network','archive','output'):p.add_argument(name,type=Path)
    p.add_argument('--nodes',type=int,default=140000)
    p.add_argument('--scale',type=int,default=72)
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--counts',type=int,nargs=3,default=(3200,750,850))
    p.add_argument('--seed',type=int,default=44261)
    a=p.parse_args()
    if min(a.nodes,a.workers,*a.counts)<1 or a.output.exists():p.error('Use positive budgets and a new path')
    main(a)
