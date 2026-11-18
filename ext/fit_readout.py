r"""Fit only the additive NNUE read-out on archived Stockfish-scored games.

  gcc -O3 -march=native -DHMAX=768 ext/readout_features.c -lm -o /tmp/readout_features
  OPENBLAS_NUM_THREADS=2 python ext/fit_readout.py /tmp/readout_features zero/nets/g101_vol.bin \
      zero/results/games/games_bigs.jsonl.sc.jsonl.gz /tmp/student.bin
"""
import argparse
from contextlib import ExitStack
import gzip
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile

import chess
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from readout_arena import opening_key


def holdout(fen):
    return hashlib.sha256(opening_key(fen).encode()).digest()[0] < 51


def examples(archive):
    # Split whole opening pairs before collecting or deduplicating any positions.
    positions = [{}, {}]
    with gzip.open(archive, "rt") as stream:
        for line in stream:
            game = json.loads(line)
            board = chess.Board(game["fen"])
            split = positions[holdout(game["fen"])]
            if len(game["sc"]) != len(game["moves"]) + 1:
                raise ValueError("Expected a side-to-move label for every position")
            for move, score in zip(game["moves"], game["sc"]):
                move = chess.Move.from_uci(move)
                if not board.is_check() and not move.promotion and not board.is_capture(move):
                    fen = opening_key(board.fen())
                    if abs(score) < 2000:
                        split.setdefault(fen, score)
                board.push(move)
    for fen in positions[0]:
        positions[1].pop(fen, None)
    return positions


def feature_rows(extractor, parent, positions, width, stack, settle):
    temp = stack.enter_context(tempfile.TemporaryFile())
    fens = "".join(f"{fen} 0 1\n" for group in positions for fen in group)
    cmd = [extractor.resolve(strict=True), parent] + (["--settle"] if settle else [])
    subprocess.run(cmd, input=fens.encode(), stdout=temp, check=True)
    shape = (sum(map(len, positions)), 2 + 2 * width)
    if temp.tell() != np.prod(shape) * 4:
        raise ValueError("Unexpected feature stream length")
    return np.memmap(temp, dtype=np.float32, mode="r", shape=shape)


def fit(x, base, target, penalty):
    # Bernoulli cross entropy with soft teacher labels and an L2 penalty on the change.
    def objective(delta):
        z = base + x @ delta
        loss = np.logaddexp(0.0, z).mean() - target @ z / len(z)
        grad = x.T @ (expit(z) - target) / len(z)
        return loss + penalty * (delta @ delta) / 2, grad + penalty * delta
    solution = minimize(objective, np.zeros(x.shape[1]), jac=True, method="L-BFGS-B",
                        options={"ftol": 1e-12, "gtol": 1e-7, "maxiter": 250})
    if not solution.success:
        raise RuntimeError(solution.message)
    return solution.x


def run(args):
    if args.output.exists():
        raise ValueError("Output already exists. Choose a new path for the fitted network")
    data = bytearray(args.parent.read_bytes())
    magic, width, buckets = struct.unpack_from("<iii", data)
    if magic not in (0x4F555232, 0x4F555233) or buckets not in (1, 8) or width < 1:
        raise ValueError("Expected an OUR2 or OUR3 NNUE with one or eight output buckets")
    king_buckets = struct.unpack_from("<i", data, 12)[0] if magic == 0x4F555233 else 1
    header = 16 if magic == 0x4F555233 else 12
    offset = header + 4 * width * (768 * king_buckets + 1)
    out = np.frombuffer(data, dtype="<f4", count=2 * width * buckets,
                        offset=offset).reshape(buckets, 2 * width)
    positions = examples(args.archive)
    if not all(positions):
        raise ValueError("Both opening splits must contain quiet labelled positions")
    num_train = len(positions[0])
    targets = expit(np.array([cp for group in positions for cp in group.values()]) / 271.58)
    with ExitStack() as stack:
        rows = feature_rows(args.extractor, args.parent, positions, width, stack, args.settle)
        groups = rows[:, 0]
        base = rows[:, 1].astype(float) / 271.58
        for bucket in range(buckets):
            train = np.flatnonzero(groups[:num_train] == bucket)
            if len(train) < 2 * width:
                continue
            x = rows[train, 2:].astype(float)
            delta = fit(x, base[train], targets[train], args.penalty)
            out[bucket] += delta
        # Reload the exported weights. Quantization and a different chosen capture line
        # can both change the deployed score relative to the floating-point training fit.
        with tempfile.NamedTemporaryFile(dir=args.output.parent) as candidate:
            candidate.write(data)
            candidate.flush()
            scored = feature_rows(args.extractor, candidate.name, [positions[1]], width, stack, args.settle)
            student = scored[:, 1].astype(float) / 271.58
            os.link(candidate.name, args.output)  # publish the validated file without replacing an existing one
        test_labels = targets[num_train:]
        for name, scores in [("parent", base[num_train:]), ("student", student)]:
            loss = np.mean(np.logaddexp(0, scores) - test_labels * scores)
            print(f"{name} held-out cross entropy {loss:.6f}")
    print(f"unique quiet positions: {num_train} train, {len(positions[1])} held out")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("extractor", type=Path)
    parser.add_argument("parent", type=Path)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--settle", action="store_true", help="fit at the leaf selected by a capture search")
    parser.add_argument("--penalty", type=float, default=0.005)
    args = parser.parse_args()
    if not np.isfinite(args.penalty) or args.penalty <= 0:
        parser.error("--penalty must be positive and finite")
    run(args)
