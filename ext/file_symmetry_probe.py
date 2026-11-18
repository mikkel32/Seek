"""Measure a mirrored-input evaluator against the mean of its two file orientations."""
import argparse
import gzip
import json
from pathlib import Path
import random
import subprocess

import chess
import numpy as np


def evaluate(engine, weights, fens):
    commands = ''.join('position fen '+fen+'\neval\n' for fen in fens)+'quit\n'
    run = subprocess.run([engine,weights],input=commands,text=True,capture_output=True,check=True)
    scores = [int(line.split()[1]) for line in run.stdout.splitlines() if line.startswith('eval ')]
    if len(scores) != len(fens):
        raise ValueError('Missing static evaluations')
    return np.array(scores)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('parent','candidate','weights','archive'):
        p.add_argument(name,type=Path)
    a = p.parse_args()
    rng = random.Random(117323)
    examples = {}; with_rights = {}
    with gzip.open(a.archive,'rt') as f:
        for line in f:
            g = json.loads(line); b = chess.Board(g['fen'])
            for move,score in zip(g['moves'],g['sc']):
                m = chess.Move.from_uci(move)
                if not b.is_check() and not b.is_capture(m) and not m.promotion and b.occupied.bit_count()>7 and abs(score)<900:
                    target = with_rights if b.castling_rights else examples
                    target.setdefault(b.fen(en_passant='legal'),score)
                b.push(m)
    sample = rng.sample(sorted(examples),min(len(examples),4000))
    other = rng.sample(sorted(with_rights),min(len(with_rights),256))
    reflected = [chess.Board(fen).transform(chess.flip_horizontal).fen() for fen in sample]
    source = sample+reflected+other
    base = evaluate(a.parent,a.weights,source)
    candidate = evaluate(a.candidate,a.weights,source)
    n = len(sample)
    expected = (base[:n]+base[n:2*n])/2
    labels = np.array([examples[f] for f in sample])
    absolute = np.abs(base[:n]-base[n:2*n])
    result = {'rights_free':n,'with_rights':len(other),'parent_orientation_mae_cp':float(absolute.mean()),
              'parent_orientation_p95_cp':float(np.percentile(absolute,95)),
              'candidate_orientation_max_cp':int(np.max(np.abs(candidate[:n]-candidate[n:2*n]))),
              'candidate_mean_max_error_cp':float(np.max(np.abs(candidate[:n]-expected))),
              'with_rights_max_change_cp':int(np.max(np.abs(candidate[2*n:]-base[2*n:]))),
              'parent_label_mae_cp':float(np.mean(np.abs(base[:n]-labels))),
              'candidate_label_mae_cp':float(np.mean(np.abs(candidate[:n]-labels)))}
    print(json.dumps(result))
    if result['with_rights_max_change_cp'] or result['candidate_orientation_max_cp']>1 or result['candidate_mean_max_error_cp']>1.5:
        raise ValueError('Projection failed the numerical check')


if __name__ == '__main__':
    main()
