# Training a quiet-move history prior

These models predict which quiet moves the engine should try first. They change the initial direction of the search without changing the NNUE evaluation. At the start of each root search, the experimental engine moves its existing history table halfway toward the trained values for the current material phase. The normal history updates still run during search. A default build decays toward zero as before.

**Stockfish move choices.** We used the Stockfish side of the archived BIGS–Stockfish games. In each eligible position, the model assigned a probability to the played move among all legal non-capturing, non-promotion moves. It learned a from–to score shared across material phases and a regularized adjustment for each of four phases. Black moves were reflected vertically. Whole opening pairs were assigned to training or validation before duplicate positions were removed. The 42,838 training and 12,566 held-out choices yielded held-out log loss of 2.93895 versus 3.05989 under a uniform distribution over the available quiet moves. The loss numbers use the clipped and quantized table used in play.

**Search cutoffs.** A separate collection build searched 1,024 training roots and 256 held-out roots at 24,000 nodes per root. We recorded a pair when a quiet move at search depth three or greater caused a beta cutoff after an earlier quiet move had failed. We excluded nodes where the side to move was in check. A regularized pairwise logistic model learned to put the successful move first. Roots come from separate opening groups. Deeper transpositions can still occur in different trees. The labels reflect moves the incumbent actually searched. They do not establish the value of unsearched alternatives.

## Reproduction

Run from the repository root. Use a new directory for each generated model. The training archive is `zero/results/games/games_bigs.jsonl.sc.jsonl.gz`.

```sh
mkdir -p /tmp/seek-teacher /tmp/seek-cutoff
archive=zero/results/games/games_bigs.jsonl.sc.jsonl.gz
gcc -O3 -march=native -DHMAX=768 calc/calc_s19.c -lm -o /tmp/seek-baseline
gcc -O3 -march=native -DHMAX=768 -DSEEK_COLLECT_PRIOR calc/calc_s19.c -lm -o /tmp/seek-collector
OPENBLAS_NUM_THREADS=1 python ext/train_history_prior.py "$archive" /tmp/seek-teacher/history_prior.h
OPENBLAS_NUM_THREADS=1 python ext/train_cutoff_prior.py /tmp/seek-collector "$parent" "$archive" \
  /tmp/seek-cutoff/history_prior.h --workers 8 --train 1024 --valid 256 --nodes 24000
gcc -O3 -march=native -DHMAX=768 -DSEEK_HISTORY_PRIOR=256 -I/tmp/seek-cutoff calc/calc_s19.c -lm -o /tmp/seek-cutoff-256
```

Set `parent` to the path to the desired NNUE before running these commands. Use `--student-engine /tmp/seek-cutoff-256` in `ext/readout_arena.py` and specify that same NNUE for both sides. In the reported matches, both sides of every G101 test used EvalScale 72 and both sides of every BIGS test used 100. The Stockfish-trained table in `ext/nets/history_prior.h` has SHA-256 `f8bb3095e49763a9fca7ab3982fef7db4ec9bdb98728f412606dc5199e8bb895`.

## G101 match results

The G101 collection build used s19’s standard scale 100 to generate 481,094 training pairs and 119,320 held-out pairs. Its quantized cutoff predictor scored 0.67365 held-out log loss versus 0.69315 for an equal-probability prediction. Both sides in the playing tests used EvalScale 72. Each screening match used the same 200 color-reversed openings at 15,000 nodes per move (offset 1920, pool size 640, seed 111978). Prior strength is the compile-time `SEEK_HISTORY_PRIOR` coefficient.

| Candidate | Games | Wins–draws–losses | Elo | Opening-paired 95% interval |
|---|---:|---:|---:|---:|
| Stockfish choices, 256 | 400 | 121–152–127 | −5.2 | −32.8 to +22.3 |
| Stockfish choices, 1024 | 400 | 130–150–120 | +8.7 | −18.1 to +35.6 |
| G101 cutoffs, 256 | 400 | 128–144–128 | 0.0 | −26.3 to +26.3 |
| G101 cutoffs, 1024 | 400 | 142–147–111 | +27.0 | −1.6 to +55.9 |
| G101 cutoffs, 1024 at 10+0.1 | 240 | 35–163–42 | −10.1 | −32.3 to +11.9 |

The strongest screen advanced to 120 fresh pairs at the clock (offset 1280, seed 55489). Its clock match provided no evidence of an additional Elo gain over calibrated G101. All games were legal and neither side lost on time. The published G101 cutoff table is `ext/nets/g101-cutoff/history_prior.h` with SHA-256 `19745aa33159ad57a46c6594040dc4787e1de72ff48bb5c0edffe2ed434811b9`. Opening-paired game scores are in `ext/results/history-g101-pairs.tsv.gz`.
