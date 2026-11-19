# Direct changes to Seek’s input weights

One input weight made G101 find a rook capture at 6,000 nodes. The original first found it at 70,000 among the twelve budgets tested. Stockfish 14.1 ranked the capture first at four million nodes, 230 centipawns above the original move. White still had the worse position. Restoring the original coefficient restored the original file byte for byte. A change of the same size in the opposite direction also restored the original move at the four development budgets.

## Programming individual moves

We sampled 96 training positions where the referee preferred an alternative by 80 to 550 centipawns. In 19 of them, a small group of input-weight edits made G101 play that alternative at three or more of the four search budgets. Combining edits changed some of the earlier answers. A greedy merge that replayed every accepted position retained seven corrections in one net by changing 22 of 294,912 input coefficients. It found the intended move at all four budgets in five positions, and at three budgets in the other two. Deeper, two-million-node forced comparisons favored the edits over the original moves in all seven positions by 8 to 370 centipawns. Three of the seven were also the first choice in a separate unrestricted two-million-node search. In one rook ending, the original rook move evaluated to −4.18 pawns. Moving the king instead evaluated to −0.48. The unrestricted referee preferred a third move.

To test transfer near the rook-capture position, we sampled 120 of 706 unique legal continuations consisting of two quiet non-checking moves with the original capture still available. Neither G101 model participated in choosing the sample. Across all 120 variants the original matched the referee’s top choice 89 times and the edited model matched 94 times. At 400,000 nodes the Stockfish top move led by at least half a pawn in 47 cases where the capture was best and 50 cases where another move was best. In the first group the original found the capture 44 times, the one-weight model 42 times. In the second group the original chose the capture 7 times and the edited model 6 times. These positions share a tactical position by construction.

In a separate group of 21 unseen positions from 20 openings, the referee preferred capturing a more valuable piece by at least 80 centipawns over the compared alternative. The original, one-weight and combined models all found the preferred move 20 times. All three found the four pawn-takes-rook examples.

## Fitting weights across openings

The case generator sampled up to three high-cost decisions and three other decisions from each of 1,000 archived games. It replayed the candidate network from each root to identify its actual choice. Stockfish 14.1 proposed an alternative at 40,000 nodes and compared the two at 80,000. Opening hashes determined the training, tuning and test splits. The deterministic fitter ranked first-layer coefficients by agreement on two training subsets, estimated changes against the preference margins, then rounded to the engine’s deployed grid. All later network tensors were frozen.

G101 changed 255 first-layer weights. On the 747 held-out pairs with a clear preference its static agreement moved from 61.98% to 61.85%, with a mean absolute shift of 2.84 cp over held-out child evaluations. Production BIGS changed 227 of 9,437,184 coefficients on a 1/128 grid. Static agreement moved from 64.53% to 64.60% on 750 clear held-out pairs. Ties receive half credit.

## Unseen positions with full search

G101 played each root at 10,000 nodes. The referee compared changed choices at 240,000 nodes. Intervals group positions by their source opening. The expected-score scale uses the repository’s logistic conversion.

- **G101, 22 programmed edits.** 358 of 1,170 moves changed. The referee preferred the new move in 189 positions and the old move in 159, with 10 ties. The estimated change in expected score was +0.152 percentage points (95% interval -0.139 to +0.444).
- **G101, 255 fitted edits.** 412 of 1,170 moves changed. The referee preferred the new move in 211 positions and the old move in 184, with 17 ties. The estimated change in expected score was +0.039 percentage points (95% interval -0.304 to +0.382).
- **BIGS, 227 fitted edits.** 244 of 1,168 moves changed. The referee preferred the new move in 106 positions and the old move in 129, with 9 ties. The estimated change in expected score was -0.007 percentage points (95% interval -0.321 to +0.306).

## Programming the production network

A separate run programmed the production BIGS network directly on its own training positions. 7 of 48 positions admitted a sparse repair under the four-budget and opposite-direction controls. The selected patch changed 1 input coefficient. On its selected position BIGS chose the intended move 0/4 times before editing, 4/4 afterward and 0/4 with the opposite edit. A fresh two-million-node comparison evaluated the selected move at +112 cp and the original move at +52 cp, a difference of +60 cp.


## Games against the original network

Both sides used the same executable, one thread, 64 MiB of hash and identical search settings. G101 used EvalScale 72 and BIGS used 100. Each opening was played with both colors. Intervals use opening-pair scores. After the 500-game fitted G101 screen and its first clock match, the same frozen checkpoint played another 1,500 fixed-node games on 750 fresh opening pairs. Those confirmation games are reported separately. One additional 440-game test used the previously successful interrupted-root search fix on both sides to isolate the effect of the weights with that build. For the production move audit we reconstructed the original BIGS file from the private fitted checkpoint and its reversible edits and verified its SHA-256 before play. Only fixed-node results measure search at equal node budgets. A clock such as 5+0.05 gives each player five seconds initially and 0.05 seconds per move.

| Candidate | Limit per side | Wins–draws–losses | Score | Match Elo | Paired 95% Elo interval |
|---|---|---:|---:|---:|---:|
| One edit, G101 | 18500 nodes | 106–134–120 | 48.06% | -13.5 | -41.2 to +14.0 |
| 22 programmed edits, G101 | 18500 nodes | 125–142–133 | 49.00% | -6.9 | -35.4 to +21.4 |
| 22 programmed edits, G101 | 5+0.05 | 47–139–54 | 48.54% | -10.1 | -39.7 to +19.3 |
| 255 fitted edits, G101 | 18500 nodes | 168–180–152 | 51.60% | +11.1 | -12.8 to +35.1 |
| 255 fitted edits, G101 | 5+0.05 | 70–168–62 | 51.33% | +9.3 | -12.7 to +31.3 |
| 255 fitted edits, G101, fresh confirmation | 18500 nodes | 474–558–468 | 50.20% | +1.4 | -12.5 to +15.2 |
| 255 fitted edits, G101, improved root search on both sides | 18500 nodes | 134–174–132 | 50.23% | +1.6 | -22.5 to +25.7 |
| 227 fitted edits, BIGS | 18500 nodes | 130–220–130 | 50.00% | 0.0 | -21.5 to +21.5 |
| 227 fitted edits, BIGS | 10+0.1 | 31–141–28 | 50.75% | +5.2 | -18.0 to +28.5 |
| Programmed production BIGS edit | 18500 nodes | 58–115–67 | 48.12% | -13.0 | -32.7 to +6.6 |

## Matched games against Stockfish

The original G101 and the 22-edit checkpoint each played the same 100 reserved openings twice against Stockfish 14.1 NNUE. Each engine received 18,500 nodes per move and 64 MiB of hash. Changing the weights moved G101’s score by +1.00 percentage points (opening-matched 95% interval -2.30 to +4.30). This measures the G101 network in the standard search configuration. The previously tested BIGS interrupted-root change is a separate search experiment.

| G101 vs Stockfish 14.1 | Wins–draws–losses | Score | Match Elo | Paired 95% Elo interval |
|---|---:|---:|---:|---:|
| Original | 3–13–184 | 4.75% | -520.9 | -648.3 to -445.1 |
| 22 programmed edits | 2–19–179 | 5.75% | -485.8 | -585.5 to -420.1 |

Across these 5,060 games there were 0 time forfeits and 0 illegal moves. [Machine-readable results](input-weight-summary.json)

## Reproduction

From the repository root, use Python with NumPy, SciPy, and python-chess. Supply an official Stockfish 14.1 executable with its default NNUE. Build both Seek binaries for the same CPU.

```sh
cc -O3 -march=native -DHMAX=768 calc/calc_s19.c -lm -o seek
cc -O3 -march=native -DHMAX=768 ext/input_weight_features.c -lm -o features
python ext/weight_edit_cases.py zero/results/games/games_bigs.jsonl.sc.jsonl.gz ./seek zero/nets/g101_vol.bin ./stockfish pairs.jsonl
python ext/program_weight_repair.py ./seek ./features zero/nets/g101_vol.bin pairs.jsonl one --positions 36
python ext/program_weight_repair.py ./seek ./features zero/nets/g101_vol.bin pairs.jsonl bank --positions 96
python ext/stack_weight_repairs.py ./seek zero/nets/g101_vol.bin bank/successful.json combined
python ext/edit_input_weights.py ./features zero/nets/g101_vol.bin pairs.jsonl fitted --scale 72
```

Use `ext/repair_neighborhood.py` to test a supplied repair on quiet two-ply continuations. Use `ext/apply_input_edits.py` to apply or reverse an exact patch with source and destination SHA-256 checks. Use `ext/verify_weight_moves.py` for the search review, `ext/readout_arena.py` for paired matches, and `ext/history_stockfish_match.py --trained-weights` to compare the two networks against the same Stockfish opponent.
