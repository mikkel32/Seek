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

## Production BIGS

The collector used the 768-wide BIGS network. It produced 487,824 training comparisons and 118,917 held-out comparisons. With the quantized table, held-out pairwise log loss fell from 0.69315 to 0.68164. The NNUE SHA-256 was `9b204a84cbb78acab433ce79fbefdc848e02c71df2a2966e73affb3a2f2fe2fa`. The trained history table SHA-256 was `38657505c19eb44e41b352ac69afea636ed6d645e6bb14613c2d7bbd0e0d77fd`. The trained table and game logs are stored privately in `/root/loopchess/zero/results/seek_policy_61311764/` on the project vault.

Each candidate played unchanged BIGS on the same 250 color-reversed openings at 20,000 nodes per move (pool offset 0, seed 971116). The candidate with the highest score then played 200 additional color-reversed openings at 10 seconds plus 0.1 seconds per move (pool offset 640, seed 77436). Both sides used EvalScale 100.

| Seek candidate | Wins–draws–losses | Points scored | Match Elo | Opening-paired 95% Elo interval |
|---|---:|---:|---:|---:|
| Stockfish choices, 512 | 120–231–149 | 47.10% | −20.2 | −42.6 to +2.1 |
| BIGS cutoffs, 256 | 125–224–151 | 47.40% | −18.1 | −40.6 to +4.3 |
| BIGS cutoffs, 1024 | 140–230–130 | 51.00% | +6.9 | −14.9 to +28.9 |
| BIGS cutoffs, 1024 at 10+0.1 | 42–304–54 | 48.50% | −10.4 | −26.0 to +5.1 |

## Direct match against Stockfish 14.1

Unchanged BIGS and BIGS with cutoff prior 1024 played the same 100 additional openings in both colors. Each side had one thread, 64 MiB of hash and 10 seconds plus 0.1 seconds per move. The arms ran concurrently on the same host, with five boards per build. The opening pool excluded the entire source game archive and the 200 openings from the BIGS clock match. Its offset was 2400 and the opening seed was 403287.

The opponent was compiled from the official `sf_14.1` tag with `make build ARCH=x86-64-bmi2` on an Intel Xeon E5-2680 v4. Its default NNUE was enabled. Both Seek executables used `gcc -O3 -march=native -DHMAX=768`. An independent fit on the match machine reproduced the private history table byte for byte. Network SHA-256 was `13406b1dcbe0a36dfc5e07da51d8db38b7a45d7da43078ec25a7624c2b83236f`. The complete match can be repeated with `ext/history_stockfish_match.py`. The full match summaries, including both paired estimates, are archived in [history-bigs-summary.json](history-bigs-summary.json). The separate transfer of the Stockfish per-game files to the vault timed out before receipt was confirmed.

| Seek candidate | Wins–draws–losses | Points scored | Match Elo | Opening-paired 95% Elo interval |
|---|---:|---:|---:|---:|
| Unchanged BIGS | 5–95–100 | 26.25% | −179.5 | −212.9 to −148.8 |
| Trained BIGS | 3–97–100 | 25.75% | −184.0 | −221.3 to −150.2 |

The trained model changed the score against Stockfish by −0.50 percentage points (paired 95% interval −5.23 to +4.23). This interval uses the difference between the two Seek builds on each opening, averaged across the two colors.

All 1,900 BIGS head-to-head games and all 400 Stockfish games completed with zero time losses and zero illegal moves. The 400-game BIGS clock match and the matched Stockfish comparison both leave the existing champion in place.

For historical context, the archived 1,000-game run gave BIGS 39 wins, 616 draws and 345 losses against Stockfish 14.1, or 34.7% of the points. That run used a different match setup. The 26.25% baseline above is the appropriate comparison for the new model. [Archived run](../../zero/results/vault_clock.log)
