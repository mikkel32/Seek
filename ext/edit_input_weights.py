"""Edit selected first-layer NNUE weights to separate a better move from a weaker one.

The existing engine extracts the active piece-square rows for both viewpoints and
the derivative at the end of its capture search. The saved network is quantized
and scored again for each tuning candidate before one is selected.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile

import chess
import numpy as np
from scipy import sparse
from scipy.linalg import solve


def child_fens(cases):
    children = []
    for row in cases:
        board = chess.Board(row['fen'])
        for key in ('played', 'alternative'):
            board.push_uci(row[key])
            children.append(board.fen())
            board.pop()
    return children


def evaluate(extractor, net, boards, scores_only=False):
    command = [extractor, net] + (['--scores'] if scores_only else [])
    raw = subprocess.run(command, input=('\n'.join(boards)+'\n').encode(),
                         capture_output=True, check=True).stdout
    width, features, bits = struct.unpack_from('<iii', raw)
    fmt = '<i4' if scores_only else np.dtype([('score','<i4'), ('rows','<i4',(2,32)),
                                              ('gradient','<f4',(2,width))])
    result = np.frombuffer(raw, dtype=fmt, offset=12)
    if result.size != len(boards):
        raise ValueError('Extractor returned an unexpected number of positions')
    return (width, features, bits), result


def indicators(rows, count):
    keep = rows >= 0
    endpoints = np.r_[0, np.cumsum(keep.sum(axis=1))]
    columns = rows[keep]
    return sparse.csr_matrix((np.ones(len(columns), dtype=np.float32), columns, endpoints),
                             shape=(len(rows), count))


def rank_coordinates(observations, indicators_by_view, scale, cases, train, shape):
    width, features, _ = shape
    scores = observations['score'].reshape(-1, 2)
    margin = (scores[:, 0] - scores[:, 1]) * scale
    teacher = np.array([r['alternative_cp'] - r['played_cp'] for r in cases])
    direction = np.sign(teacher)
    desired = np.clip(abs(teacher), 0, 160) * .7
    repair = direction * np.clip(desired - direction * margin, 0, 175)
    repair[abs(teacher) < 20] = 0
    folds = np.array([hashlib.sha256(r['opening'].encode()).digest()[1] % 2 for r in cases])
    estimates = []
    for fold in range(2):
        weight = np.where(train & (folds == fold), repair, 0)
        signed = np.column_stack([weight, -weight]).ravel().astype(np.float32)
        grad = np.zeros((features, width), np.float32)
        support = np.zeros(features)
        for view, x in enumerate(indicators_by_view):
            grad += x.T @ (signed[:, None] * observations['gradient'][:, view])
            support += x.T @ (signed != 0).astype(np.float32)
        grad /= np.sqrt(np.maximum(support, 1))[:, None]
        grad[support < 12] = 0
        estimates.append(grad)
    a, b = estimates
    # Keep coordinates for which independent sets of opening pairs agree on sign.
    strength = np.minimum(abs(a), abs(b)) * (np.sign(a) == np.sign(b))
    order = np.argsort(-strength.ravel(), kind='stable')
    return order[strength.ravel()[order] > 0], repair


def design(selected, observations, indicators_by_view, width, scale):
    matrix = np.zeros((len(observations), len(selected)), dtype=np.float32)
    for view, source in enumerate(indicators_by_view):
        x = source.tocsc()
        for j, coord in enumerate(selected):
            feature, unit = divmod(int(coord), width)
            ids = x.indices[x.indptr[feature]:x.indptr[feature + 1]]
            matrix[ids, j] += observations['gradient'][ids, view, unit] * scale
    return matrix


def metrics(margin, teacher):
    meaningful = abs(teacher) >= 25
    if not meaningful.any():
        raise ValueError('Split contains no clear teacher preferences')
    signed = margin[meaningful] * np.sign(teacher[meaningful])
    gap = np.minimum(abs(teacher[meaningful]), 300)
    wrong = (signed < 0).astype(float) + .5 * (signed == 0)
    return dict(pairs=int(len(margin)), clear=int(meaningful.sum()),
                preferred_pct=float(100 * np.mean(1 - wrong)), regret_cp=float(np.mean(gap * wrong)),
                margin_shortfall_cp=float(np.mean(np.maximum(.7 * np.minimum(gap, 160) - signed, 0))))


def export(original, offset, shape, selected, delta, output):
    width, features, bits = shape
    data = bytearray(original)
    weights = np.frombuffer(data, dtype='<f4', count=width * features, offset=offset)
    step = 2.0 ** -bits
    # Integral grid increments preserve each original floating-point rounding residual.
    ticks = np.rint(delta / step).astype(np.int32)
    weights[selected] += (step * ticks).astype(np.float32)
    output.write_bytes(data)
    return ticks


def main(args):
    out = args.output
    out.mkdir(exist_ok=False, parents=True)
    original = args.parent.read_bytes()
    magic, width, buckets = struct.unpack_from('<iii', original)
    if magic not in (0x4F555232, 0x4F555233) or buckets not in (1, 8):
        raise ValueError('Expected OUR2/OUR3 with direct output buckets')
    offset = 16 if magic == 0x4F555233 else 12
    bank_count = struct.unpack_from('<i', original, 12)[0] if offset == 16 else 1
    cases = [json.loads(l) for l in args.cases.read_text().splitlines()]
    boards = child_fens(cases)
    shape, obs = evaluate(args.extractor, args.parent, boards)
    if shape[:2] != (width, 768 * bank_count):
        raise ValueError('Extractor and file architecture disagree')
    # Exclude identical child positions across splits, as well as terminal/mate lines.
    owners = {}
    keys = [' '.join(f.split()[:4]) for f in boards]
    for i, key in enumerate(keys):
        owners.setdefault(key, set()).add(cases[i // 2]['split'])
    use = np.array([len(owners[keys[2*i]]) == len(owners[keys[2*i+1]]) == 1 for i in range(len(cases))])
    use &= np.max(abs(obs['score'].reshape(-1,2)), axis=1) < 1800
    cases = [row for row, valid in zip(cases, use) if valid]
    boards = [fen for fen, valid in zip(boards, np.repeat(use, 2)) if valid]
    obs = obs[np.repeat(use, 2)]
    splits = np.array([r['split'] for r in cases])
    train, tune, test = (splits == label for label in ('train','tune','test'))
    if not all(x.any() for x in (train, tune, test)):
        raise ValueError('Expected three nonempty opening splits')
    scale = args.scale / 100
    teacher = np.array([r['alternative_cp'] - r['played_cp'] for r in cases])
    base_scores = obs['score'].astype(float)
    base_margin = (base_scores[::2] - base_scores[1::2]) * scale
    x = [indicators(obs['rows'][:, i], shape[1]) for i in range(2)]
    ranked, repair = rank_coordinates(obs, x, scale, cases, train, shape)
    if not len(ranked):
        raise ValueError('No stable coordinate directions')
    results = []
    candidates = []
    base_weights = np.frombuffer(original, dtype='<f4', count=shape[0]*shape[1], offset=offset)
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        for budget in (256, 2048):
            selected = ranked[:budget]
            features = design(selected, obs, x, width, scale)
            sampled = features.reshape(len(cases), 2, -1)[train]
            difference = sampled[:, 0] - sampled[:, 1]
            # Fit the desired increase in move separation and penalize changes to
            # each original evaluation. Ridge regularization couples no coordinates.
            gram = difference.T @ difference
            flat = sampled.reshape(-1, len(selected))
            gram += .22 * (flat.T @ flat)
            gram.flat[::len(selected)+1] += len(difference) * 500
            delta = solve(gram, difference.T @ repair[train].astype(np.float32), assume_a='pos')
            for bound, factor in ((.06,.60),(.12,1.0)):
                edit = factor * np.clip(delta, -bound, bound)
                path = Path(tmp) / f'w{budget}-{factor}.bin'
                ticks = export(original, offset, shape, selected, edit, path)
                validation_boards = [board for board, keep in zip(boards, np.repeat(tune,2)) if keep]
                loaded, scores = evaluate(args.extractor, path, validation_boards, scores_only=True)
                if loaded != shape:
                    raise ValueError('Edit altered the deployed quantizer scale')
                margin = (scores[::2] - scores[1::2]) * scale
                m = metrics(margin, teacher[tune])
                drift = float(np.mean(abs(scores-base_scores[np.repeat(tune,2)]))*scale)
                results.append(dict(budget=budget, factor=factor, changed=int(np.count_nonzero(ticks)),
                                    mean_abs_change_cp=drift, **m))
                candidates.append((m['margin_shortfall_cp'] + .15 * drift, path, selected, ticks))
        _, chosen, selected, ticks = min(candidates, key=lambda row: row[0])
        winner = out/'candidate.bin'
        os.link(chosen, winner)
    loaded, scores = evaluate(args.extractor, winner, boards, scores_only=True)
    assert loaded == shape
    candidate_margin = (scores[::2] - scores[1::2]) * scale
    changed = selected[ticks != 0]
    weights = np.fromfile(winner, dtype='<f4', count=shape[0]*shape[1], offset=offset)
    assert np.array_equal(np.sort(changed), np.flatnonzero(weights != base_weights))
    blob = winner.read_bytes()
    assert original[:offset] == blob[:offset] and original[offset+4*len(weights):] == blob[offset+4*len(weights):]
    # Exact old/new coefficients make the binary edits independently auditable.
    edits = [dict(row=int(k//width),unit=int(k%width),before=float(base_weights[k]),after=float(weights[k]))
             for k in changed]
    (out/'edits.json').write_text(json.dumps(edits)+'\n')
    summary = dict(shape=shape, changed=len(changed), first_layer_size=len(weights), training=int(train.sum()),
                   tuning=int(tune.sum()), testing=int(test.sum()), grid=2.0**-shape[2],
                   parent_sha256=hashlib.sha256(original).hexdigest(), candidate_sha256=hashlib.sha256(blob).hexdigest(),
                   tuning_candidates=results, test_parent=metrics(base_margin[test],teacher[test]),
                   test_candidate=metrics(candidate_margin[test],teacher[test]),
                   test_mean_abs_change_cp=float(np.mean(abs(scores-base_scores)[np.repeat(test,2)])*scale))
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ('extractor','parent','cases','output'):
        p.add_argument(arg, type=Path)
    p.add_argument('--scale',type=int,default=72)
    a = p.parse_args()
    main(a)
