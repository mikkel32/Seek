"""Label trace roots with three independent Stockfish continuations for challenger training."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import chess
import chess.engine
from trace_data import select_positions


def label(job):
    binary, positions, nodes = job
    records=[]
    with chess.engine.SimpleEngine.popen_uci(binary) as engine:
        engine.configure({'Threads':1,'Hash':64,'Use NNUE':True})
        for split,group,fen,_ in positions:
            board=chess.Board(fen)
            lines=engine.analyse(board,chess.engine.Limit(nodes=nodes),multipv=3,game=object())
            moves={r['pv'][0].uci():r['score'].pov(board.turn).score(mate_score=20000)
                   for r in lines if r.get('pv') and 'score' in r}
            records.append(dict(fen=fen,moves=moves))
    return records


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('stockfish','archive','output'):p.add_argument(arg,type=Path)
    p.add_argument('--workers',type=int,default=6)
    p.add_argument('--nodes',type=int,default=180000)
    p.add_argument('--counts',nargs=3,type=int,default=(3200,750,850))
    p.add_argument('--seed',type=int,default=44261)
    a=p.parse_args()
    if min(a.workers,a.nodes,*a.counts)<1:p.error('Use positive budgets')
    positions=select_positions(a.archive,a.counts,a.seed)
    jobs=[(a.stockfish.resolve(),positions[i:i+35],a.nodes) for i in range(0,len(positions),35)]
    with a.output.open('x') as out, ProcessPoolExecutor(max_workers=a.workers) as pool:
        for batch in pool.map(label,jobs):
            for row in batch:out.write(json.dumps(row)+'\n')
            out.flush()
