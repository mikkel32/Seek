"""Generate counterfactual move pairs for direct input-weight editing of a Seek network."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
import random

import chess
from quality import E, Eng, opts_of
from readout_arena import OPTIONS, opening_key


def roots(archive):
    samples = []
    rng = random.Random(413885)
    with gzip.open(archive, 'rt') as stream:
        for line in stream:
            game = json.loads(line)
            origin = opening_key(game['fen'])
            fold = hashlib.sha256(origin.encode()).digest()[0] % 10
            split = 'train' if fold < 6 else ('tune' if fold < 8 else 'test')
            board = chess.Board(game['fen'])
            pool = []
            for i, uci in enumerate(game['moves']):
                if (board.turn == game['us_white'] and 12 <= board.fullmove_number <= 55
                        and board.occupied.bit_count() > 7 and not board.is_check()
                        and abs(game['sc'][i]) < 400):
                    drop = E(game['sc'][i]) + E(game['sc'][i + 1]) - 1
                    pool.append((drop, board.fen()))
                board.push_uci(uci)
            rng.shuffle(pool)
            costly = sorted(pool, key=lambda row: -row[0])[:3]
            used = {fen for _, fen in costly}
            typical = [row for row in pool if row[1] not in used][:3]
            samples.extend((split, origin, fen, drop) for drop, fen in costly + typical)
    # Keep all occurrences of the same position in a single opening split.
    splits = {}
    for split, _, fen, _ in samples:
        splits.setdefault(opening_key(fen), set()).add(split)
    unique = {}
    for row in samples:
        key = opening_key(row[2])
        if len(splits[key]) == 1:
            unique.setdefault(key, row)
    return list(unique.values())


def collect(args):
    jobs, engine, network, fish, nodes, teacher_nodes, scale = args
    seek = Eng(f'{engine} {network}', opts_of(OPTIONS) + [('EvalScale', str(scale))])
    judge = Eng(str(fish), [('Threads', '1'), ('Hash', '32'), ('MultiPV', '2'), ('Use NNUE', 'true')])
    out = []
    try:
        for split, origin, fen, drop in jobs:
            seek.new()
            played = seek.move(fen, nodes)
            judge.new()
            top = judge.lines(fen, teacher_nodes)
            if len(top) != 2:
                continue
            alternative = top[1][0] if top[1][0] != played else top[2][0]
            judge.new()
            checked = judge.lines(fen, teacher_nodes * 2, ' searchmoves ' + played + ' ' + alternative)
            scores = {m: cp for m, cp in checked.values()}
            if set(scores) != {played, alternative} or max(map(abs, scores.values())) >= 1500:
                continue
            out.append(dict(split=split, opening=origin, fen=fen, played=played, alternative=alternative,
                            played_cp=scores[played], alternative_cp=scores[alternative], archive_drop=drop))
    finally:
        seek.quit()
        judge.quit()
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('archive', 'engine', 'network', 'stockfish', 'output'):
        p.add_argument(name, type=Path)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--nodes', type=int, default=10000)
    p.add_argument('--teacher-nodes', type=int, default=40000)
    p.add_argument('--scale', type=int, default=72)
    a = p.parse_args()
    if min(a.workers, a.nodes, a.teacher_nodes, a.scale) <= 0:
        p.error('Worker count, budgets and scale must be positive')
    rows = roots(a.archive)
    jobs = [(rows[i::a.workers], a.engine.resolve(), a.network.resolve(), a.stockfish.resolve(),
             a.nodes, a.teacher_nodes, a.scale) for i in range(a.workers)]
    with a.output.open('x') as f, ProcessPoolExecutor(max_workers=a.workers) as pool:
        for part in pool.map(collect, jobs):
            for row in part:
                print(json.dumps(row), file=f)
            f.flush()
    print('Sampled', len(rows), 'roots', flush=True)
