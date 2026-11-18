"""Play two Seek builds against Stockfish from long, theoretically won endgames."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import random

import chess
import chess.engine
import chess.syzygy


MATERIAL = ('BN/', 'Q/R', 'RB/R', 'NN/P')
SEEK = dict(Hash=64, EvalScale=72, PvPrune=1, ChkExt=0, SeDouble=20, CutIir=1)


def positions(tables, count):
    rng = random.Random(731884)
    selected = []
    for material in MATERIAL:
        attackers, defenders = material.split('/')
        pieces = 'K' + attackers + 'k' + defenders.lower()
        group = []
        for _ in range(200_000):
            board = chess.Board.empty()
            for piece, square in zip(pieces, rng.sample(range(64), len(pieces))):
                board.set_piece_at(square, chess.Piece.from_symbol(piece))
            if not board.is_valid() or board.is_check():
                continue
            dtz = tables.probe_dtz(board)
            if 35 <= dtz <= 85:
                if len(group) % 2:
                    board = board.mirror()
                group.append((material, board.fen(), dtz))
            if len(group) == count:
                break
        if len(group) != count or len({fen for _, fen, _ in group}) != count:
            raise ValueError('Unable to find distinct seed positions for ' + material)
        selected.extend(group)
    return selected


def duel(task):
    (material, fen, dtz), args = task
    row = {'material': material, 'fen': fen, 'initial_dtz': dtz}
    for name in ('parent', 'candidate'):
        seek_command = [str(getattr(args, name)), str(args.weights)]
        with (chess.engine.SimpleEngine.popen_uci(seek_command) as us,
              chess.engine.SimpleEngine.popen_uci([str(args.fish)]) as fish):
            options = dict(SEEK)
            if name == 'candidate':
                options['SyzygyPath'] = str(args.tables)
            us.configure(options)
            fish.configure({'Threads': 1, 'Hash': 64, 'Use NNUE': True, 'SyzygyPath': str(args.tables)})
            board = chess.Board(fen)
            attacker = board.turn
            moves = []
            while len(moves) < 400 and not board.is_game_over(claim_draw=True):
                engine = us if board.turn == attacker else fish
                move = engine.play(board, chess.engine.Limit(nodes=args.nodes)).move
                if move not in board.legal_moves:
                    raise ValueError(f'Engine returned illegal move {move} for {board.fen()}')
                board.push(move)
                moves.append(move.uci())
            outcome = board.outcome(claim_draw=True)
            result = 'capped'
            if outcome:
                result = 'draw' if outcome.winner is None else ('win' if outcome.winner == attacker else 'loss')
            row[name] = {'result': result, 'end': outcome.termination.name if outcome else None, 'moves': moves}
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = ('parent', 'candidate', 'weights', 'fish', 'tables')
    for key in (*inputs, 'output'):
        parser.add_argument(key, type=Path)
    parser.add_argument('--each', type=int, default=8)
    parser.add_argument('--nodes', type=int, default=8000)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    if min(args.each, args.nodes, args.workers) < 1:
        parser.error('Positions per group, node budget and workers must be positive')
    for name in inputs:
        setattr(args, name, getattr(args, name).resolve(strict=True))
    with chess.syzygy.open_tablebase(str(args.tables)) as tables:
        cases = positions(tables, args.each)
    print('Seed positions', len(cases), flush=True)
    with args.output.open('x') as stream, ProcessPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(duel, ((case, args) for case in cases)):
            print(json.dumps(row), file=stream, flush=True)


if __name__ == '__main__':
    main()
