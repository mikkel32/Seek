"""Find a small quantized input-weight edit that consistently repairs a specified training move."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import struct
import tempfile

import numpy as np
from edit_input_weights import child_fens, evaluate, export
from quality import Eng, opts_of
from readout_arena import OPTIONS


BUDGETS = (7500, 10000, 15000, 22000)


def load_and_play(engine, path, fen, nodes):
    engine.send('setoption name EvalFile value ' + str(path))
    engine.send('isready')
    engine.wait('readyok')
    return play(engine, fen, nodes)


def play(engine, fen, nodes):
    engine.new()
    return engine.move(fen, nodes)


def repair(job):
    a, tasks = job
    original = a.parent.read_bytes()
    offset = 16 if struct.unpack_from('<I',original)[0] == 0x4F555233 else 12
    result = []
    eng = Eng([a.engine, a.parent], opts_of(OPTIONS) + [('EvalScale',str(a.scale))])
    try:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'candidate.bin'
            for case in tasks:
                fen = case['fen']
                target = case['alternative']
                load_and_play(eng, a.parent, fen, 10000)
                before = [play(eng,fen,b) for b in BUDGETS]
                if before[1] != case['played'] or before.count(case['played']) < 3 or target in before:
                    continue
                shape, children = evaluate(a.extractor, a.parent, child_fens([case]))
                width, features, _ = shape
                diff = np.zeros((features,width), dtype=np.float32)
                for i, child in enumerate(children):
                    for view in range(2):
                        ids = child['rows'][view]
                        ids = ids[ids >= 0]
                        diff[ids] += (1 if i == 0 else -1) * child['gradient'][view]
                direction = diff.ravel()
                pool = np.flatnonzero(direction)
                order = pool[np.argsort(-abs(direction[pool]),kind='stable')][:20]
                found = False
                for count in (1,4,16):
                    groups = [order[k:k+count] for k in range(0,min(len(order),8 if count==1 else count*2),count)]
                    for group in groups:
                        if len(group)!=count:
                            continue
                        vector = direction[group]/max(abs(direction[group]))
                        for magnitude in (.08,.20,.40):
                            ticks = export(original,offset,shape,group,vector*magnitude,path)
                            if load_and_play(eng,path,fen,10000) != target:
                                continue
                            after = [play(eng,fen,b) for b in BUDGETS]
                            if after.count(target) < 3:
                                continue
                            export(original,offset,shape,group,-vector*magnitude,path)
                            load_and_play(eng,path,fen,10000)
                            reverse = [play(eng,fen,b) for b in BUDGETS]
                            if reverse.count(target) > 1:
                                continue
                            result.append(dict(fen=fen, source_opening=case['opening'], target=target,
                                               mistake=case['played'], parent=before, candidate=after,
                                               reverse=reverse, budgets=BUDGETS, scale=a.scale,
                                               played_cp=case['played_cp'], alternative_cp=case['alternative_cp'],
                                               shape=shape, coordinates=group.tolist(), ticks=ticks.tolist()))
                            found = True
                            break
                        if found:
                            break
                    if found:
                        break
    finally:
        eng.quit()
    return result


def main(a):
    rows = [r for r in map(json.loads,a.cases.read_text().splitlines())
            if r['split']=='train' and 80 <= r['alternative_cp']-r['played_cp'] <= 550
            and max(abs(r['alternative_cp']), abs(r['played_cp'])) < 600]
    random.Random(94155).shuffle(rows)
    tasks = rows[:a.positions]
    a.output.mkdir(exist_ok=False,parents=True)
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        results = [r for group in pool.map(repair, [(a,tasks[i::a.workers]) for i in range(a.workers)]) for r in group]
    (a.output/'search-summary.json').write_text(json.dumps(dict(attempted=len(tasks),passed=len(results)))+'\n')
    if not results:
        print('No repair passed the four-budget and reverse-edit criteria')
        return
    (a.output/'successful.json').write_text(json.dumps(results)+'\n')
    best = min(results,key=lambda r:(len(r['coordinates']),-r['candidate'].count(r['target']),
                                    max(map(abs,r['ticks'])),r['fen']))
    original = a.parent.read_bytes()
    offset = 16 if struct.unpack_from('<I',original)[0] == 0x4F555233 else 12
    shape=tuple(best['shape'])
    selected=np.array(best['coordinates'])
    step=2.**-shape[2]
    path=a.output/'candidate.bin'
    export(original,offset,shape,selected,step*np.array(best['ticks']),path)
    old=np.frombuffer(original,'<f4',offset=offset,count=shape[0]*shape[1])
    new=np.fromfile(path,'<f4',offset=offset,count=shape[0]*shape[1])
    best['edits']=[dict(row=int(k//shape[0]),unit=int(k%shape[0]),before=float(old[k]),after=float(new[k])) for k in selected]
    best['parent_sha256']=hashlib.sha256(original).hexdigest()
    best['candidate_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    restored = bytearray(path.read_bytes())
    target = np.frombuffer(restored, '<f4', offset=offset, count=len(old))
    target[selected] = old[selected]
    assert bytes(restored) == original
    (a.output/'repair.json').write_text(json.dumps(best,indent=2)+'\n')
    print(json.dumps(best))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('engine','extractor','parent','cases','output'):
        p.add_argument(arg,type=Path)
    p.add_argument('--positions',type=int,default=36)
    p.add_argument('--workers',type=int,default=3)
    p.add_argument('--scale',type=int,default=72)
    args=p.parse_args()
    if min(args.workers, args.positions, args.scale) <= 0:
        p.error('Positive budgets required')
    for k in ('engine', 'extractor', 'parent', 'cases'):
        setattr(args, k, getattr(args, k).resolve(strict=True))
    main(args)
