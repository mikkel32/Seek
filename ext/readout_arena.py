"""Replay a fitted-readout match on openings absent from its entire teacher corpus.

  python ext/readout_arena.py /tmp/calc zero/nets/g101_vol.bin /tmp/student.bin --games-log /tmp/run.jsonl
"""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
import math
import os
import random
import statistics
import subprocess
import sys
import tempfile

from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
OPTIONS = "Hash=64,PvPrune=1,ChkExt=0,SeDouble=20,CutIir=1"


def opening_key(fen):
    return " ".join(fen.split()[:4])


def eligible_book(archive):
    with gzip.open(archive, "rt") as stream:
        used = {opening_key(json.loads(line)["fen"]) for line in stream}
    all_fens = (ROOT / "data/openings.txt").read_text().splitlines()
    unused = {opening_key(fen): fen for fen in all_fens if fen and opening_key(fen) not in used}
    book = sorted(unused.values())
    random.Random(901181).shuffle(book)
    return book


def paired_estimate(log, expected):
    scores = defaultdict(dict)
    for line in log.read_text().splitlines():
        game = json.loads(line)
        pair = scores[opening_key(game["fen"])]
        if game["us_white"] in pair:
            raise ValueError("Opening and color appear twice")
        pair[game["us_white"]] = game["score"]
    if len(scores) != expected or any(len(pair) != 2 for pair in scores.values()):
        raise ValueError("Expected one complete color pair per opening")
    means = [sum(pair.values()) / 2 for pair in scores.values()]
    mean = statistics.mean(means)
    margin = t.ppf(0.975, expected - 1) * statistics.stdev(means) / math.sqrt(expected)

    def elo(score):
        if score <= 0 or score >= 1:
            return None
        return round(400 * math.log10(score / (1 - score)), 1)

    return {"pairs": expected, "score": mean, "elo": elo(mean),
            "paired_95_lo": elo(mean - margin), "paired_95_hi": elo(mean + margin)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engine", type=Path)
    parser.add_argument("--student-engine", type=Path, help="candidate executable for search experiments")
    parser.add_argument("parent", type=Path)
    parser.add_argument("student", type=Path)
    parser.add_argument("--archive", type=Path,
                        default=ROOT / "zero/results/games/games_bigs.jsonl.sc.jsonl.gz")
    parser.add_argument("--exclude-games", nargs="+", type=Path, default=[],
                        help="exclude openings in these earlier match logs before slicing the book")
    parser.add_argument("--offset", type=int, default=640, help="start of reserved opening pool")
    parser.add_argument("--pool-size", type=int, default=640)
    parser.add_argument("--pairs", type=int, default=100)
    parser.add_argument("--common-options", default=OPTIONS, help="comma-separated UCI options for both engines")
    parser.add_argument("--student-options", default="", help="additional UCI options just for the challenger")
    parser.add_argument("--parent-options", default="", help="additional UCI options just for the incumbent")
    parser.add_argument("--student-eval-scale", type=int, default=100, help="UCI EvalScale for the candidate")
    parser.add_argument("--parent-eval-scale", type=int, default=100, help="UCI EvalScale for a calibration control")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=181126)
    control = parser.add_mutually_exclusive_group()
    control.add_argument("--nodes", type=int)
    control.add_argument("--tc", default="10+0.1")
    parser.add_argument("--games-log", required=True, type=Path, help="new file for per-game results")
    args = parser.parse_args()
    book = eligible_book(args.archive)
    if args.exclude_games:
        excluded = set()
        for prior in args.exclude_games:
            with prior.open() as games:
                excluded.update(opening_key(json.loads(line)["fen"]) for line in games if line.strip())
        book = [fen for fen in book if opening_key(fen) not in excluded]
    if args.offset < 0 or args.offset + args.pool_size > len(book) or not 2 <= args.pairs <= args.pool_size:
        parser.error("Choose at least two pairs from a pool inside the unused opening book")
    if args.workers < 1 or (args.nodes is not None and args.nodes <= 0):
        parser.error("Workers and fixed node budgets must be positive")
    engine, parent, student = [p.resolve(strict=True) for p in (args.engine, args.parent, args.student)]
    student_engine = args.student_engine.resolve(strict=True) if args.student_engine else engine
    log = args.games_log.resolve()
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        data = work / "data"
        data.mkdir()
        (data / "openings.txt").write_text("\n".join(book[args.offset:args.offset + args.pool_size]) + "\n")
        # gauntlet uses space-delimited UCI commands. Short symlinks keep paths with spaces usable.
        for name, target in [("engine", engine), ("challenger", student_engine), ("parent.bin", parent), ("student.bin", student)]:
            (work / name).symlink_to(target)
        parent_options = ",".join(filter(None, [args.common_options, args.parent_options, f"EvalScale={args.parent_eval_scale}"]))
        student_options = ",".join(filter(None, [args.common_options, args.student_options, f"EvalScale={args.student_eval_scale}"]))
        cmd = [sys.executable, str(ROOT / "gauntlet.py"), "--us", "./challenger student.bin",
               "--us_opts", student_options, "--opp", f"parent|./engine parent.bin|{parent_options}",
               "--workers", str(args.workers), "--openings", str(args.pairs), "--seed", str(args.seed)]
        cmd += ["--nodes", str(args.nodes)] if args.nodes else ["--tc", args.tc]
        with log.open("x"):
            pass
        subprocess.run(cmd, cwd=work, env={**os.environ, "GAMES_LOG": str(log)}, check=True)
    print(json.dumps(paired_estimate(log, args.pairs)))


if __name__ == "__main__":
    main()
