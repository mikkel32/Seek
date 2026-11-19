"""Generate training positions by rolling the unmodified Seek engine forward from opening FENs."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
import random

import chess
import chess.engine
from trace_data import OPTIONS


def position_key(fen):
    return ' '.join(fen.split()[:4])


def play_batch(job):
    binary, network, openings, nodes, scale = job
    options = dict(item.split('=') for item in OPTIONS.split(','))
    options['EvalScale'] = scale
    positions = []
    with chess.engine.SimpleEngine.popen_uci([str(binary), str(network)]) as engine:
        engine.configure({name: int(value) for name, value in options.items()})
        for fen in openings:
            board = chess.Board(fen)
            origin = position_key(fen)
            fold = hashlib.sha256(origin.encode()).digest()[0] % 10
            split = 'train' if fold < 6 else ('tune' if fold < 8 else 'test')
            game = object()
            for ply in range(61):
                if board.is_game_over(claim_draw=True):
                    break
                if ply >= 8 and (ply - 8) % 3 == 0 and board.legal_moves.count() >= 2:
                    positions.append((split, origin, board.fen(), 0))
                if ply < 60:
                    move = engine.play(board, chess.engine.Limit(nodes=nodes), game=game).move
                    board.push(move)
    return positions


def main(args):
    # Only opening FENs are read. All subsequent moves and all search labels come from Seek.
    with gzip.open(args.openings, 'rt') as stream:
        openings = sorted({json.loads(line)['fen'] for line in stream})
    jobs = [(args.engine.resolve(), args.network.resolve(), openings[i:i + 8],
             args.nodes, args.scale) for i in range(0, len(openings), 8)]
    unique = {}
    conflicts = set()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for batch in pool.map(play_batch, jobs):
            for row in batch:
                key = position_key(row[2])
                if key in unique and unique[key][0] != row[0]:
                    conflicts.add(key)
                unique.setdefault(key, row)
    rng = random.Random(args.seed)
    samples = []
    for split, count in zip(('train', 'tune', 'test'), args.counts):
        pool = sorted(row for key, row in unique.items() if key not in conflicts and row[0] == split)
        if len(pool) < count:
            raise ValueError(f'Only {len(pool)} eligible positions for {split}, requested {count}')
        samples.extend(rng.sample(pool, count))
    with args.output.open('x') as stream:
        for row in samples:
            print(json.dumps(row), file=stream)
    print(f'{len(samples)} positions from {len(openings)} opening FENs', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('engine', 'network', 'openings', 'output'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--nodes', type=int, default=1000)
    parser.add_argument('--scale', type=int, default=72)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--counts', nargs=3, type=int, default=(3200, 750, 850))
    parser.add_argument('--seed', type=int, default=84563)
    args = parser.parse_args()
    if min(args.nodes, args.workers, *args.counts) < 1 or args.output.exists():
        parser.error('Use positive budgets and a new output path')
    main(args)
