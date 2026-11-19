"""Combine sparse input-weight corrections while replaying all accepted positions."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import struct
import tempfile

import numpy as np

from edit_input_weights import export
from program_weight_repair import BUDGETS, play
from quality import Eng, opts_of
from readout_arena import OPTIONS


def combine(args):
    original = args.parent.read_bytes()
    offset = 16 if struct.unpack_from('<I', original)[0] == 0x4F555233 else 12
    cases = json.loads(args.bank.read_text())
    if not cases:
        raise ValueError('The bank contains no successful repairs')
    cases.sort(key=lambda r: (len(r['coordinates']), -r['candidate'].count(r['target']),
                              max(map(abs, r['ticks'])), r['fen']))
    width, rows, bits = shape = tuple(cases[0]['shape'])
    if any(tuple(r['shape']) != shape or tuple(r['budgets']) != BUDGETS or
           r['scale'] != args.scale for r in cases):
        raise ValueError('All repairs must use the same archive, architecture, scale and budgets')
    args.output.mkdir(parents=True, exist_ok=False)
    path = args.output / 'candidate.bin'
    engine = Eng([args.engine, args.parent], opts_of(OPTIONS) + [('EvalScale', str(args.scale))])
    accepted = []
    shifts = Counter()

    def replay(selected, output):
        engine.send('setoption name EvalFile value ' + str(output))
        engine.send('isready')
        engine.wait('readyok')
        moves = []
        for row in selected:
            actual = [play(engine, row['fen'], n) for n in BUDGETS]
            if actual[1] != row['target'] or actual.count(row['target']) < 3:
                return None
            moves.append(actual)
        return moves

    try:
        for row in cases:
            if [play(engine, row['fen'], n) for n in BUDGETS] != row['parent']:
                raise ValueError('The source network did not reproduce a recorded parent move')
        with tempfile.TemporaryDirectory() as tmp:
            trial = Path(tmp) / 'trial.bin'
            for row in cases:
                trial_shifts = shifts.copy()
                trial_shifts.update(dict(zip(row['coordinates'], row['ticks'])))
                coordinates = np.array(sorted(k for k,v in trial_shifts.items() if v), dtype=int)
                if len(coordinates) > args.max_weights:
                    continue
                delta = np.array([trial_shifts[k] for k in coordinates]) * 2.**-bits
                export(original, offset, shape, coordinates, delta, trial)
                if replay([row, *accepted], trial) is not None:
                    shifts = trial_shifts
                    accepted.append(row)
                    path.write_bytes(trial.read_bytes())
        if not accepted:
            raise ValueError('No repair survived full search')
        final = replay(accepted, path)
        if final is None:
            raise ValueError('The exported network failed its replay')
    finally:
        engine.quit()
    data = path.read_bytes()
    old = np.frombuffer(original, '<f4', offset=offset, count=width*rows)
    new = np.frombuffer(data, '<f4', offset=offset, count=width*rows)
    coords = np.flatnonzero(old != new)
    edits = [dict(row=int(k//width), unit=int(k%width), before=float(old[k]), after=float(new[k])) for k in coords]
    end = offset + 4*width*rows
    assert data[:offset] == original[:offset] and data[end:] == original[end:]
    manifest = dict(parent_sha256=hashlib.sha256(original).hexdigest(),
                    candidate_sha256=hashlib.sha256(data).hexdigest(), shape=shape, edits=edits)
    (args.output/'edits.json').write_text(json.dumps(manifest, indent=2)+'\n')
    replays = [dict(fen=r['fen'], target=r['target'], parent=r['parent'], combined=m,
                    budgets=BUDGETS) for r, m in zip(accepted, final)]
    (args.output/'repairs.json').write_text(json.dumps(replays, indent=2)+'\n')
    summary = dict(original_repairs=len(cases), combined_repairs=len(accepted), changed=len(edits),
                   all_four=sum(m.count(r['target']) == 4 for r,m in zip(accepted, final)))
    (args.output/'summary.json').write_text(json.dumps(summary)+'\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ('engine', 'parent', 'bank', 'output'):
        parser.add_argument(argument, type=Path)
    parser.add_argument('--max-weights', type=int, default=32)
    parser.add_argument('--scale', type=int, default=72)
    args = parser.parse_args()
    for argument in ('engine', 'parent', 'bank'):
        setattr(args, argument, getattr(args, argument).resolve(strict=True))
    if args.max_weights < 1:
        parser.error('Need at least one weight')
    combine(args)
