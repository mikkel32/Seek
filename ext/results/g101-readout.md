# G101 evaluation scale and output-layer fitting

On s19, the readout fitted to archived Stockfish labels scored +42.6 Elo over stock G101 across 1,400 games at 30,000 nodes per move (95% opening-paired interval +26.7 to +58.8). A simpler G101 with its existing UCI EvalScale set to 72 then beat the fitted readout by 48.4 Elo in a direct 600-game control. This experiment uses the public 384-wide G101 checkpoint. A comparison to the private BIGS champion or to Stockfish needs a separate match.

| Candidate | Opponent | Control | W–D–L | Elo | Paired 95% interval |
|---|---|---|---|---:|---:|
| Fitted G101 | G101 | 30k nodes, first book | 151–116–133 | +15.6 | −13.9 to +45.5 |
| Fitted G101 | G101 | 30k nodes, independent replication | 441–271–288 | +53.6 | +34.6 to +72.9 |
| Fitted G101 | G101 | 10 s + 0.1 s per move | 58–98–44 | +24.4 | −10.0 to +59.2 |
| Fitted G101 | G101, EvalScale 72 | 30k nodes, calibration control | 166–185–249 | −48.4 | −70.6 to −26.5 |
| Exploratory capture-leaf fit | G101 | 10k nodes | 82–50–68 | +24.4 | −14.3 to +63.6 |

Every pair reverses colors. The engine, hash, threads and search options were held fixed, with the explicit EvalScale change shown above. Across these games there were zero time losses or illegal moves. Intervals use the mean and standard error of the two-game opening scores and a Student-t critical value, transformed to the Elo scale. All match openings come from the repository book after excluding every starting position in the teacher archive. The first and replication books for the same control are disjoint. The compressed TSV contains one row per opening per match.

## What the fit learned

The teacher archive contains 1,000 rescored games from BIGS against Stockfish. Splitting the 500 opening pairs before extracting quiet positions gave 386 training pairs and 114 validation pairs. After deduplication, training used 73,578 positions, and the held-out set contained 21,918. The fitter freezes the input features, biases and auxiliary heads. It fits a penalized additive adjustment to each adequately represented output bucket and reloads the quantized evaluator for scoring. Playing the fitted model adds no new inference operations.

Held-out teacher cross entropy fell from 0.672743 to 0.662270. Fitting one scalar per checkpoint on its training positions gave 0.659019 for G101 and 0.658658 for the fitted readout. The slope estimates were 0.5673 and 0.7829. Their ratio motivated the G101 EvalScale 72 control before that match was played. The match tests a relative scale match. It does not establish that 72 is optimal for G101 or another network. The source labels come from Stockfish, so the fitted checkpoint belongs to the project's external-data track.

## Reproduction

Run from repository root with GCC, Python, python-chess, NumPy and SciPy. All arena examples below use the same engine executable on both sides. Build and fit with:

```sh
gcc -O3 -march=native -DHMAX=768 calc/calc_s19.c -lm -o /tmp/seek-s19
gcc -O3 -march=native -DHMAX=768 ext/readout_features.c -lm -o /tmp/seek-features
OPENBLAS_NUM_THREADS=2 python ext/fit_readout.py /tmp/seek-features zero/nets/g101_vol.bin \
  zero/results/games/games_bigs.jsonl.sc.jsonl.gz /tmp/g101-fit.bin
```

The published G101 candidate is `ext/nets/g101_archived_readout.bin` with SHA-256 `ebb7d278cbece1d615b7489f05e553c587c2bfa45fa8cf73fd3b2adeb242d43c`. The original `zero/nets/g101_vol.bin` has SHA-256 `dbd1f8cdfd6fc6352e27a5ae29554089c0ccdfe2f0cded7fbb9b4c9615ef4345`. Floating-point libraries can differ in the least significant bit of a fitted weight. Use the published bytes to replay the tournament candidate.

```sh
parent=zero/nets/g101_vol.bin
student=ext/nets/g101_archived_readout.bin
OPENBLAS_NUM_THREADS=1 python ext/readout_arena.py /tmp/seek-s19 "$parent" "$student" \
  --offset 1920 --pool-size 640 --pairs 500 --seed 731 --nodes 30000 --games-log /tmp/replication.jsonl
OPENBLAS_NUM_THREADS=1 python ext/readout_arena.py /tmp/seek-s19 "$parent" "$student" \
  --offset 2400 --pool-size 580 --pairs 300 --seed 93017 --nodes 30000 --parent-eval-scale 72 \
  --games-log /tmp/calibration-control.jsonl
```

The first node screen used offset 0 and 200 pairs. The first clock screen used offset 640 and 100 pairs at `--tc 10+0.1`. Both used pool size 640 and seed 181126. The exploratory model used the same fitter with `--settle`, then offset 1280, pool size 640 and seed 181126 for 100 pairs. Each invocation requires a new game-log path.

To test the private 768-wide BIGS, pass its file as the parent in the fitting command and use that same file as the arena opponent. The arena accepts independent EvalScale values for the parent and student, so the simpler scale-only hypothesis can be tested with the original BIGS on both sides.
