"""Verify integer inference and benchmark the new predictions against native ordering on identical traces."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
from trace_data import collect


def logits_for(tokens, depth, model):
    hidden=np.broadcast_to(model['bias'],(tokens.shape[1],len(model['bias']))).astype(np.int32).copy()
    for lag in range(min(3,depth)):
        ids=tokens[depth-1-lag]
        hidden+=model['embedding'][lag][ids].sum(axis=1,dtype=np.int32)
    return np.clip(hidden,0,384)@model['heads'].astype(np.int32)


def run(a):
    source=np.load(a.data)
    model=dict(np.load(a.model))
    positions=[('test',opening,fen,0.) for split,opening,fen in zip(source['splits'],source['groups'],source['fens']) if split=='test']
    teacher={r['fen']:r['moves'] for r in map(json.loads,a.teacher.read_text().splitlines())}
    nodes=int(source['nodes']); scale=int(source['scale'])
    # The production code may reorder later depths. Compare on exactly the trace that C emitted.
    special=['1r5k/P7/8/8/8/8/8/5K2 w - - 0 1','4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 25',
             'r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1']
    checks=positions[:20]+[('test','fixture',f,0.) for f in special]
    checked=0
    for _,(tokens,_,_,_,integer) in collect((a.instrumented.resolve(),a.net.resolve(),checks,30000,scale)):
        for depth,native in integer.items():
            calculated=logits_for(tokens,depth,model)
            np.testing.assert_array_equal(native,calculated)
            checked+=native.size
    if not checked: raise RuntimeError('No integer predictions were tested')
    jobs=[(a.baseline.resolve(),a.net.resolve(),positions[i:i+12],nodes,scale) for i in range(0,len(positions),12)]
    grouped={}
    one=np.load(a.snapshot) if a.snapshot else None
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for batch in pool.map(collect,jobs):
            for (_,group,fen,_),(tokens,leaders,vocabulary,order,_) in batch:
                moves=vocabulary.split()
                counts=grouped.setdefault(group,np.zeros((2,5),np.int64))
                vals=np.array([teacher[fen].get(m,-np.inf) for m in moves])
                for d in range(1,len(leaders)-1):
                    current=leaders[d]; actual=leaders[d+1]
                    scores=logits_for(tokens,d+1,model)[:,1];scores[current]=-2**30
                    short=scores.copy() if one is None else logits_for(tokens,d+1,one)[:,1]
                    short[current]=-2**30
                    picks=np.array([np.argmax(scores),np.argmax(short),order[d][0],order[d][1]])
                    if actual!=current:
                        counts[0,0]+=1; counts[0,1:]+=picks==actual
                    ranked=vals.copy();ranked[current]=-np.inf
                    if np.isfinite(ranked).any():
                        counts[1,0]+=1;counts[1,1:]+=picks==np.argmax(ranked)
    all_counts=np.stack(list(grouped.values()))
    summed=all_counts.sum(axis=0)
    values=summed[:,1:]/summed[:,:1]
    rng=np.random.default_rng(154293)
    sampled=all_counts[rng.integers(len(all_counts),size=(4000,len(all_counts)))].sum(axis=1)
    differences=(sampled[:,:,1]-sampled[:,:,4])/sampled[:,:,0]
    ci=np.quantile(differences,[.025,.975],axis=0)
    result=dict(integer_logits_verified=checked,positions=len(positions),openings=len(grouped))
    for i,name in enumerate(('leader_switch','teacher_challenger')):
        result[name]=dict(events=int(summed[i,0]),model=values[i,0],snapshot=values[i,1],
                          native_order=values[i,2],cumulative_effort=values[i,3],
                          difference_to_effort=values[i,0]-values[i,3],paired_bootstrap_95=ci[:,i].tolist())
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('baseline','instrumented','net','model','data','teacher','output'):p.add_argument(k,type=Path)
    p.add_argument('--snapshot',type=Path);p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();run(a)
