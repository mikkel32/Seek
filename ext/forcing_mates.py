"""Measure bounded checking-mate proofs on held-out, game-disjoint Lichess positions."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
import subprocess
from pathlib import Path

import chess
from quality import Eng, opts_of
from readout_arena import OPTIONS


def select_rows(path, development, held_out):
    groups = defaultdict(list)
    with path.open() as stream:
        for row in csv.DictReader(stream):
            if 'mate' not in row['Themes'].split():
                continue
            moves = row['Moves'].split()
            distance = len(moves) // 2
            if distance < 3 or len(moves) % 2:
                continue
            game = row['GameUrl'].split('#')[0].split('/black')[0].rsplit('/', 1)[-1]
            order = hashlib.sha256(('seek-mate-20261119/' + row['PuzzleId']).encode()).hexdigest()
            groups[min(5, distance)].append((order, game, row['PuzzleId'], row['FEN'], moves, distance))
    selected = []
    used_games = set()
    used_positions = set()
    for distance in (5, 4, 3):
        count = 0
        for _, game, puzzle_id, fen, moves, length in sorted(groups[distance]):
            if game in used_games:
                continue
            board = chess.Board(fen)
            board.push_uci(moves[0])
            key = ' '.join(board.fen().split()[:4])
            if key in used_positions:
                continue
            used_games.add(game)
            used_positions.add(key)
            selected.append(dict(puzzle=puzzle_id, game=game, fen=board.fen(), target=moves[1],
                                 distance=length, split='dev' if count < development else 'test'))
            count += 1
            if count == development + held_out:
                break
        if count != development + held_out:
            raise ValueError(f'Only {count} independent puzzles in distance group {distance}')
    return selected


def verify(board, tokens):
    """Check a proof against python-chess, with no use of the engine's legality code."""
    cursor = iter(tokens)
    counts = [0, 0]

    def attack(ply):
        if ply >= 31 or board.is_fivefold_repetition() or board.is_seventyfive_moves() or board.is_insufficient_material():
            raise ValueError('Invalid attack beyond the playing horizon or automatic draw')
        move = chess.Move.from_uci(next(cursor))
        if move not in board.legal_moves or not board.gives_check(move):
            raise ValueError('Illegal or nonchecking attacker move')
        counts[0] += 1
        board.push(move)
        distance = defense(ply + 1)
        board.pop()
        return distance

    def defense(ply):
        legal = set(board.legal_moves)
        seen = set()
        bound = ply
        if legal and (board.can_claim_fifty_moves() or board.can_claim_threefold_repetition() or board.is_insufficient_material()):
            raise ValueError('Defense can claim a draw')
        for token in cursor:
            if token == ')':
                if seen != legal or (not legal and not board.is_checkmate()):
                    raise ValueError('Omitted defense or stalemate leaf')
                return bound
            move = chess.Move.from_uci(token)
            if move not in legal or move in seen:
                raise ValueError('Invalid or duplicate defense')
            seen.add(move)
            counts[1] += 1
            board.push(move)
            bound = max(bound, attack(ply + 1))
            board.pop()
        raise ValueError('Unterminated defense list')

    distance = attack(0)
    if next(cursor, None) is not None:
        raise ValueError('Trailing proof tokens')
    return distance, counts


def run_batch(job):
    engine, net, positions, budget, divisor = job
    result = []
    options = opts_of(OPTIONS) + [('EvalScale', '72')]
    if divisor is not None:
        options.append(('MateProofDivisor', str(divisor)))
    eng = Eng([str(engine), str(net)], options)
    try:
        for row in positions:
            eng.new()
            eng.send('position fen ' + row['fen'])
            eng.send(f'go nodes {budget}')
            lines = []
            move = eng.wait('bestmove', lines).split()[1]
            info = lines[-1].split() if lines else []
            score = info[info.index('score') + 1:info.index('score') + 3] if info else []
            result.append({**row, 'move': move, 'agrees': move == row['target'], 'score': score})
    finally:
        eng.quit()
    return result


def audit_batch(job):
    engine, net, rows, budget = job
    commands = []
    for row in rows:
        commands.extend(['position fen ' + row['fen'], f'mateproof 11 {budget}'])
    output = subprocess.check_output([str(engine), str(net)], text=True, input='\n'.join([*commands, 'quit']) + '\n')
    results = []
    tree = []
    for line in output.splitlines():
        if line.startswith('mcert '):
            tree = line.split()[1:]
        elif line.startswith('mateproof '):
            _, move, length, spent = line.split()
            row = rows[len(results)]
            distance, counts = (0, [0, 0])
            if move != '0000':
                distance, counts = verify(chess.Board(row['fen']), tree)
                if distance != int(length) or tree[0] != move:
                    raise ValueError('Returned move or distance differs from certificate')
            elif tree:
                raise ValueError('Certificate without a proof')
            results.append({**row, 'proved': move != '0000', 'move': move, 'plies': distance,
                            'attacks': counts[0], 'defenses': counts[1], 'spent': int(spent),
                            'certificate': tree})
            tree = []
    if len(results) != len(rows):
        raise ValueError('Truncated certificate output')
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    select = sub.add_parser('select')
    for name in ('csv', 'out'):
        select.add_argument(name, type=Path)
    select.add_argument('--dev', type=int, default=100)
    select.add_argument('--test', type=int, default=300)
    run = sub.add_parser('play')
    for name in ('engine', 'net', 'positions', 'out'):
        run.add_argument(name, type=Path)
    run.add_argument('--split', choices=['dev', 'test'], required=True)
    run.add_argument('--nodes', type=int, default=18500)
    run.add_argument('--divisor', type=int)
    run.add_argument('--workers', type=int, default=4)
    audit = sub.add_parser('audit')
    for name in ('engine', 'net', 'positions', 'out'):
        audit.add_argument(name, type=Path)
    audit.add_argument('--split', choices=['dev', 'test'], default='test')
    audit.add_argument('--nodes', type=int, required=True, help='node budget for the proof search')
    audit.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.mode == 'select':
        rows = select_rows(args.csv, args.dev, args.test)
        with args.out.open('x') as stream:
            for row in rows:
                print(json.dumps(row), file=stream)
        return
    with args.positions.open() as stream:
        positions = [row for row in map(json.loads, stream) if row['split'] == args.split]
    if args.workers < 1 or args.nodes < 1:
        parser.error('Use a positive worker count and node limit')
    worker = run_batch if args.mode == 'play' else audit_batch
    options = (args.divisor,) if args.mode == 'play' else ()
    jobs = [(args.engine.resolve(), args.net.resolve(), positions[i:i + 20], args.nodes, *options)
            for i in range(0, len(positions), 20)]
    with args.out.open('x') as stream, ProcessPoolExecutor(args.workers) as pool:
        for batch in pool.map(worker, jobs):
            for row in batch:
                print(json.dumps(row), file=stream)
            stream.flush()


if __name__ == '__main__':
    main()
