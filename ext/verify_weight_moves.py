"""Re-search unseen positions with the original and edited network, and judge changed moves."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path
import statistics

from scipy.stats import t
from quality import E, Eng, opts_of
from readout_arena import OPTIONS


def review(job):
    rows, a = job
    opts = opts_of(OPTIONS) + [('EvalScale',str(a.scale))]
    engines = [Eng(f'{a.engine} {net}', opts) for net in (a.parent, a.candidate)]
    judge = Eng(str(a.stockfish), [('Threads','1'),('Hash','32'),('MultiPV','2'),('Use NNUE','true')])
    out = []
    try:
        for case in rows:
            fen = case['fen']
            choices = []
            for eng in engines:
                eng.new()
                choices.append(eng.move(fen, a.nodes))
            if choices[0] != case['played']:
                raise ValueError('Original engine failed to reproduce the recorded choice')
            old, new = choices
            if old == new:
                out.append(dict(opening=case['opening'], change=0.0, cp=0, moved=False))
                continue
            judge.new()
            options = judge.lines(fen, a.judge_nodes, ' searchmoves ' + old + ' ' + new)
            values = {move:cp for move,cp in options.values()}
            if set(values) != {old,new}:
                raise ValueError('Incomplete referee analysis')
            out.append(dict(opening=case['opening'], fen=fen, parent=old, candidate=new,
                            parent_cp=values[old], candidate_cp=values[new],
                            change=E(values[new])-E(values[old]), cp=values[new]-values[old], moved=True))
    finally:
        for eng in engines:
            eng.quit()
        judge.quit()
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('engine','parent','candidate','stockfish','cases','output'):
        p.add_argument(name,type=Path)
    p.add_argument('--scale',type=int,default=72)
    p.add_argument('--nodes',type=int,default=10000)
    p.add_argument('--judge-nodes',type=int,default=240000)
    p.add_argument('--workers',type=int,default=4)
    a = p.parse_args()
    cases = [r for r in map(json.loads,a.cases.read_text().splitlines()) if r['split']=='test']
    if not cases or min(a.workers,a.nodes,a.judge_nodes) <= 0:
        p.error('Need test cases and positive search budgets')
    with a.output.open('x') as f, ProcessPoolExecutor(max_workers=a.workers) as pool:
        rows = []
        for group in pool.map(review, [(cases[i::a.workers],a) for i in range(a.workers)]):
            rows.extend(group)
            for r in group:
                print(json.dumps(r),file=f)
    openings = defaultdict(list)
    for row in rows:
        openings[row['opening']].append(row['change'])
    means = [statistics.mean(group) for group in openings.values()]
    avg = statistics.mean(means)
    half = t.ppf(.975,len(means)-1)*statistics.stdev(means)/math.sqrt(len(means))
    print(json.dumps(dict(positions=len(rows), openings=len(means), moves_changed=sum(r['moved'] for r in rows),
                          better=sum(r['cp']>0 for r in rows), worse=sum(r['cp']<0 for r in rows),
                          equal_changed=sum(r['cp']==0 and r['moved'] for r in rows),
                          opening_mean_score_change=avg, lower=avg-half, upper=avg+half)))
