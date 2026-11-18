"""Find lost tablebase points in archived Seek games and replay the mistakes."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

import chess
import chess.engine
import chess.syzygy


def outcome(board, tables):
    """Three-way result from the mover's side, including the recorded 50-move clock."""
    if board.is_checkmate():
        return -1
    if board.halfmove_clock >= 100 or board.is_stalemate() or board.is_insufficient_material() or board.is_repetition(3):
        return 0
    dtz = tables.probe_dtz(board)
    if abs(dtz) + board.halfmove_clock > 100:
        return 0
    return (dtz > 0) - (dtz < 0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('archive','tables','engine','weights','output'):
        p.add_argument(name,type=Path)
    p.add_argument('--nodes',type=int,default=12000)
    a = p.parse_args()
    counts = Counter(); losses = Counter(); repairs = Counter(); games = set(); examples = []
    with chess.syzygy.open_tablebase(str(a.tables)) as tables, gzip.open(a.archive,'rt') as src, \
         chess.engine.SimpleEngine.popen_uci([str(a.engine.resolve()),str(a.weights.resolve())]) as engine:
        engine.configure({'Hash':64,'SyzygyPath':str(a.tables.resolve()),'PvPrune':1,'ChkExt':0,'SeDouble':20,'CutIir':1})
        for index, line in enumerate(src):
            game = json.loads(line); board = chess.Board(game['fen']); reached = False
            counts['games'] += 1
            for uci in game['moves']:
                if board.occupied.bit_count() > 5 or board.castling_rights:
                    board.push_uci(uci)
                    continue
                reached = True
                if board.turn != game['us_white']:
                    board.push_uci(uci)
                    continue
                before = outcome(board,tables)
                board.push_uci(uci)
                after = -outcome(board,tables)
                counts['seek_moves_checked'] += 1
                if before <= after:
                    continue
                games.add(index); board.pop()
                best = engine.play(board,chess.engine.Limit(nodes=a.nodes),game=index).move
                board.push(best); recovered = -outcome(board,tables); board.pop()
                key = f'{before} to {after}'
                losses[key] += 1
                repairs[key] += int(recovered == before)
                examples.append({'game':index,'fen':board.fen(),'played':uci,'old_outcome':after,
                                 'position_outcome':before,'candidate':best.uci(),'candidate_outcome':recovered})
                board.push_uci(uci)
            counts['games_reached'] += reached or board.occupied.bit_count() <= 5
    result = {'counts':dict(counts),'games_with_loss_of_outcome':len(games),'mistakes':dict(losses),
              'corrected_with_tables':dict(repairs),'examples':examples}
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='examples'}))


if __name__ == '__main__':
    main()
