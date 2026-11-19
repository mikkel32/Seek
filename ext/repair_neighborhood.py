"""Test whether an edited decision transfers to nearby legal two-ply continuations."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import random

import chess
import chess.engine

from quality import Eng, opts_of
from readout_arena import OPTIONS, opening_key


def quiet_moves(board):
    moves = (m for m in board.legal_moves if not board.is_capture(m) and not board.gives_check(m))
    return sorted(moves, key=chess.Move.uci)


def nearby_positions(fen, target, seed, limit):
    board = chess.Board(fen)
    positions = {}
    for first in quiet_moves(board):
        board.push(first)
        for second in quiet_moves(board):
            board.push(second)
            if target in board.legal_moves and board.is_capture(target):
                positions.setdefault(opening_key(board.fen()), board.fen())
            board.pop()
        board.pop()
    all_positions = sorted(positions.values())
    random.Random(seed).shuffle(all_positions)
    return len(all_positions), all_positions[:limit]


def examine(job):
    args, positions = job
    options = opts_of(OPTIONS) + [('EvalScale', str(args.scale))]
    networks = [Eng([args.engine, net], options) for net in (args.parent, args.candidate)]
    rows = []
    try:
        with chess.engine.SimpleEngine.popen_uci(args.stockfish) as judge:
            judge.configure({'Hash': 64, 'Threads': 1, 'Use NNUE': True})
            for fen in positions:
                board = chess.Board(fen)
                limit = chess.engine.Limit(nodes=args.judge_nodes)
                analysis = judge.analyse(board, limit, multipv=2, game=object())
                if len(analysis) != 2:
                    continue
                best, runner = analysis
                best_move = best['pv'][0].uci()
                gap = (best['score'].pov(board.turn).score(mate_score=30000)
                       - runner['score'].pov(board.turn).score(mate_score=30000))
                moves = []
                for engine in networks:
                    engine.new()
                    moves.append(engine.move(fen, args.nodes))
                rows.append(dict(fen=fen, top=best_move, gap=gap, parent=moves[0], edited=moves[1]))
    finally:
        for engine in networks:
            engine.quit()
    return rows


def summarize(rows, target, gap):
    # Score choosing the referee's top move, including the positions where a different
    # move is better than the original intended capture.
    def arm(group):
        return dict(positions=len(group),
                    parent=sum(r['parent'] == r['top'] for r in group),
                    edited=sum(r['edited'] == r['top'] for r in group),
                    fixed=sum(r['parent'] != r['top'] and r['edited'] == r['top'] for r in group),
                    regressed=sum(r['parent'] == r['top'] and r['edited'] != r['top'] for r in group),
                    parent_captures=sum(r['parent'] == target for r in group),
                    edited_captures=sum(r['edited'] == target for r in group))

    clear = [r for r in rows if r['gap'] >= gap]
    return dict(all=arm(rows), clear_capture_best=arm([r for r in clear if r['top'] == target]),
                clear_other_best=arm([r for r in clear if r['top'] != target]))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ('engine', 'parent', 'candidate', 'stockfish', 'output'):
        p.add_argument(arg, type=Path)
    p.add_argument('--fen', required=True)
    p.add_argument('--target', required=True)
    p.add_argument('--nodes', type=int, default=10000)
    p.add_argument('--judge-nodes', type=int, default=400000)
    p.add_argument('--gap', type=int, default=50)
    p.add_argument('--sample', type=int, default=120)
    p.add_argument('--seed', type=int, default=119733)
    p.add_argument('--scale', type=int, default=72)
    p.add_argument('--workers', type=int, default=3)
    a = p.parse_args()
    if min(a.nodes, a.judge_nodes, a.sample, a.workers) < 1:
        p.error('Search budgets, sample and workers must be positive')
    target = chess.Move.from_uci(a.target)
    initial = chess.Board(a.fen)
    if target not in initial.legal_moves or not initial.is_capture(target):
        p.error('The target must be a legal capture in the initial position')
    count, positions = nearby_positions(a.fen, target, a.seed, a.sample)
    if not positions:
        p.error('No eligible continuations')
    for arg in ('engine', 'parent', 'candidate', 'stockfish'):
        setattr(a, arg, getattr(a, arg).resolve(strict=True))
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        rows = [row for group in pool.map(examine, [(a, positions[i::a.workers]) for i in range(a.workers)]) for row in group]
    summary = dict(eligible=count, sampled=len(positions), reviewed=len(rows),
                   **summarize(rows, a.target, a.gap))
    with a.output.open('x') as out:
        json.dump(dict(summary=summary, positions=rows), out, indent=2)
        out.write('\n')
    print(json.dumps(summary))
