# Learning the sequence of Seek’s search decisions

This model reads evidence produced by iterative deepening and forecasts which legal root move will become the principal move one depth later. A second output nominates an alternative. Both outputs are distributions over the legal moves actually present in the current position. The selected integration gives four alternatives earlier search slots. If a particular move’s forecast odds exceed those of the previous leader by a factor of exp(1/3), it may receive the first slot from depth five onward. The chess engine verifies moves with its ordinary principal variation search. Its NNUE coefficients and centipawn evaluations are unchanged.

The primary 1,200-game G101 confirmation measured +12.7 match Elo (95% interval -2.5 to +28.1). The independent 800-game G102 confirmation measured -8.7 match Elo (95% interval -28.1 to +10.7). The 95% intervals for gains over the original search included zero or fell below it in every confirmation and transfer condition. The Stockfish-assisted version scored lower against Stockfish than original G101. The fully self-taught version measured -0.9 match Elo (95% interval -16.1 to +14.4) against original G101, and its direct Stockfish score was 7.71%. Both research builds remain opt-in.

## Model and training

Each legal move has a seven-token observation after each completed depth. The tokens encode leadership, the gap from the best search score, nodes spent on the move in that depth, search order, piece and capture or promotion class, attacked origin and destination squares, and the direction of change in the root evaluation. A failed scout’s reported score can be an upper bound. A move retains its own history when the root list is reordered. Promotions to different pieces occupy distinct output slots.

The encoder sums learned embeddings over three consecutive observations, applies 16 clipped hidden activations, then scores two pointers with linear heads. The next-leader head maximizes the probability of the actual next leader. The challenger head excludes the current leader. Its target mixes independent Stockfish continuations with the first leader switch in the following three completed depths when one occurs. Only finished depths are visible at inference. The model has 3,312 parameters and uses 6,624 bytes of signed 16-bit parameter data. The host search needs no GPU or model server.

We sampled 4,800 unique archived positions from moves 12 through 55, combining ordinary positions with moves that had lost expected score in archived games. Entire opening positions were allocated to the training, tuning or test split. These contained 312, 87 and 101 opening groups, respectively. Seek searched 140,000 nodes per position. Stockfish 14.1 independently searched 180,000 nodes with three principal variations. Training used 3,200 roots and 34,652 next-leader tokens. Epoch selection used the quantized model on 750 tuning roots. The remaining 850 roots were reserved for evaluation. A complete independent rerun from the same data and seed reproduced every quantized parameter and held-out test metric. The single-depth ablation was separately trained with the same targets, widths, splits and selection rule.

| Held-out prediction | Three depths | One depth | Incumbent search order | Cumulative work |
|---|---:|---:|---:|---:|
| Incoming leader on 2,156 changes | 36.32% | 36.22% | 29.50% | 31.35% |
| Stockfish’s best alternative on 9,193 depth transitions | 29.03% | 27.67% | 21.18% | 24.95% |

On leader changes the model gained 4.96 percentage points over cumulative work (opening-bootstrap 95% interval +3.13 to +6.74). Across all transitions, simply retaining the current leader scored 76.55% versus 76.41% for the model’s highest-probability move. Its next-token log loss improved from 1.047 nats with one depth to 1.015 with three.

## Entirely self-taught variant

The original project learns from its own play. For this version we read only the 500 starting FENs from the archived games. The unmodified G101 engine supplied every subsequent move, playing both sides at 1,000 nodes per move. We sampled positions at fixed ply intervals and discarded cross-split duplicates. Each root received a fresh 140,000-node Seek search. We retained 3,200 training, 750 tuning and 850 test positions. Neither the archived game moves nor external evaluations enter these positions or labels.

The next-leader target is the observed next move. To teach the challenger, up to six later completed depths vote for their principal moves. Later depths receive higher weight. When any future leader differs from the current leader, these votes contribute 60% of the target and the remaining 40% follows the future work spent on alternative moves. When the leader holds, all the target mass follows future search work. Only past observations are available during play. The model uses the same 3,312-parameter architecture and root ordering as the assisted version. The two training conditions differ in both positions and targets.

On its own held-out traces the challenger identified 39.13% of 1,965 next-depth leadership changes. On the original shared test set, it identified 36.92% of 2,156 actual changes. Both models therefore have a direct comparison with the same native ordering (29.50%) and cumulative-work rule (31.35%). This self-taught gain over the work rule was 5.57 percentage points (opening-bootstrap 95% interval +4.02 to +7.15). The separately compiled self-taught release agreed with the experiment executable on all 850 of its own reserved roots. We evaluated the frozen self-taught version against unchanged G101 on the same 600 opening pairs as the assisted version, yielding -0.9 match Elo (95% interval -16.1 to +14.4). The assisted arm’s prior result was known before this follow-up. Against Stockfish the self-taught model scored 7.71%. Its paired score difference from original G101 against Stockfish was -2.92 percentage points (95% interval -7.03 to +1.19). Stockfish was used only after training for the shared diagnostic evaluation and match.

## Playing tests

Both arms used the interrupted-root search fix. G101 used EvalScale 72, and the separate G102 transfer match used the default scale 100 for both players. Each opening was played twice with colors reversed. The first four screens used identical openings. We froze the highest-scoring learned integration and tested it on 600 new pairs. The two G101 clock arms used the same 90 additional opening pairs. Fresh book samples also tested the frozen model at 4,000 and 60,000 nodes per move. The adaptive clock uses next-leader uncertainty to change the standard soft time allowance by a bounded factor. The hard stop still applies. The G102 follow-up used a prespecified 52% screen threshold to trigger 400 more fixed-node opening pairs and 80 timed pairs. Those follow-up sets were disjoint from the G102 screen and from each other. The current production BIGS network was outside this study.

| G101 unless indicated | Games | Limit | Wins–draws–losses | Score | Match Elo | Opening-paired 95% interval |
|---|---:|---|---:|---:|---:|---:|
| Nominate two challengers | 400 | 18500 nodes | 126–150–124 | 50.25% | +1.7 | -25.1 to +28.6 |
| Nominate four and allow a forecast switch | 400 | 18500 nodes | 136–148–116 | 52.50% | +17.4 | -7.5 to +42.5 |
| Nominate two and selectively search one ply deeper | 400 | 18500 nodes | 129–147–124 | 50.62% | +4.3 | -22.1 to +30.9 |
| Cumulative search work only | 400 | 18500 nodes | 128–147–125 | 50.38% | +2.6 | -26.2 to +31.5 |
| Selected model vs G101, fresh confirmation | 1200 | 18500 nodes | 401–442–357 | 51.83% | +12.7 | -2.5 to +28.1 |
| Selected model vs cumulative-work ordering | 440 | 18500 nodes | 132–158–150 | 47.95% | -14.2 | -39.9 to +11.4 |
| Three-depth model vs single-depth model | 440 | 18500 nodes | 149–148–143 | 50.68% | +4.7 | -20.8 to +30.3 |
| Selected model vs G101 on the clock | 180 | 5+0.05 | 39–101–40 | 49.72% | -1.9 | -32.6 to +28.7 |
| Selected model with adaptive clock vs G101 | 180 | 5+0.05 | 34–105–41 | 48.06% | -13.5 | -42.2 to +15.0 |
| G102 transfer, initial screen | 500 | 18500 nodes | 180–163–157 | 52.30% | +16.0 | -7.4 to +39.5 |
| G101 transfer to a tighter node budget | 600 | 4000 nodes | 244–139–217 | 52.25% | +15.6 | -7.6 to +39.0 |
| G101 transfer to a deeper node budget | 320 | 60000 nodes | 86–137–97 | 48.28% | -11.9 | -37.6 to +13.5 |
| Self-taught model vs G101, same confirmation openings | 1200 | 18500 nodes | 377–443–380 | 49.88% | -0.9 | -16.1 to +14.4 |
| G102 transfer, independent confirmation | 800 | 18500 nodes | 254–272–274 | 48.75% | -8.7 | -28.1 to +10.7 |
| G102 transfer on the clock | 160 | 5+0.05 | 31–92–37 | 48.12% | -13.0 | -49.1 to +22.8 |

## Independent Stockfish opponent

The original, assisted and self-taught G101 engines each played official Stockfish 14.1 NNUE on the same 120 fresh openings with both colors. All engines used one thread and 64 MiB of hash at 18,500 nodes per move. The assisted model’s score difference from original G101 was -2.08 percentage points (opening-paired 95% interval -6.08 to +1.92).

| G101 vs Stockfish 14.1 | Wins–draws–losses | Score | Match Elo |
|---|---:|---:|---:|
| Original search | 9–33–198 | 10.62% | -370.0 |
| Trace search, Stockfish-assisted | 8–25–207 | 8.54% | -411.9 |
| Trace search, Seek-only training | 6–25–209 | 7.71% | -431.3 |

A separate audit played the original and selected G101 searches at 18,500 nodes on the 850 reserved positions. They disagreed on 258 moves. Stockfish compared those choices at 450,000 nodes. It favored the new engine 130 times, the old engine 118 times and scored 10 equally. The opening-weighted difference on the repository’s expected-score scale was +0.084 percentage points (95% interval -0.341 to +0.510). The replay examples were selected from both extremes and reanalyzed at four million nodes to illustrate individual choices. In one, the trained engine played 25...Qf7, evaluated as level, while 25...Rxc2 left Black 5.78 pawns worse. In another it played 26...Qxf2, evaluated at −6.06 for Black, while the original 26...Ng6 held the balance.

Across all 8,340 match games there were 0 time forfeits and 0 illegal moves. The native and reference implementations matched on 11,918 integer logits, including promotions, castling and en passant positions. The offline browser replay matched C on another 3,918 integer logits. On this machine, scoring one completed depth for an average of 33.9 legal moves took 0.622 microseconds of thread CPU time in a warmed standalone benchmark (127,500 calls). This covers history copying and inference. Complete-game clock tests include search and ordering overhead. [Machine-readable measurements](trace-language-summary.json)

## Reproduction

From the repository root use a C compiler, NumPy, SciPy and python-chess. Supply a Stockfish 14.1 executable with its default NNUE. Choose fresh output paths for generated data and models.

```sh
cc -O3 -march=native -DHMAX=768 -DSEEK_PROVEN_ROOT -DSEEK_TRACE_COLLECT calc/calc_s19.c -lm -o trace-collector
python lm/trace_data.py ./trace-collector zero/nets/g101_vol.bin zero/results/games/games_bigs.jsonl.sc.jsonl.gz traces.npz
python lm/trace_teacher.py ./stockfish zero/results/games/games_bigs.jsonl.sc.jsonl.gz teacher.jsonl
OPENBLAS_NUM_THREADS=1 python lm/train_trace_language.py traces.npz teacher.jsonl model
cc -O3 -march=native -DHMAX=768 -DSEEK_PROVEN_ROOT -DSEEK_TRACE_MODEL=4 -DSEEK_TRACE_CHALLENGE -Imodel calc/calc_s19.c -lm -o seek-trace
./seek-trace zero/nets/g101_vol.bin
```

For the primary fixed-budget match, build the reference from the same revision and replay the frozen opening selection.

```sh
cc -O3 -march=native -DHMAX=768 -DSEEK_PROVEN_ROOT calc/calc_s19.c -lm -o seek-reference
python ext/readout_arena.py ./seek-reference zero/nets/g101_vol.bin zero/nets/g101_vol.bin --student-engine ./seek-trace --parent-eval-scale 72 --student-eval-scale 72 --offset 760 --pool-size 1100 --pairs 600 --seed 495187 --nodes 18500 --games-log confirm.jsonl
```

For Seek-only training, use the original reference for the rollouts and use the same instrumented collector as above.

```sh
python lm/trace_selfplay.py ./seek-reference zero/nets/g101_vol.bin zero/results/games/games_bigs.jsonl.sc.jsonl.gz self-positions.jsonl
python lm/trace_data.py ./trace-collector zero/nets/g101_vol.bin self-positions.jsonl self-traces.npz --positions
OPENBLAS_NUM_THREADS=1 python lm/train_trace_language.py self-traces.npz self-model --self-taught
cc -O3 -march=native -DHMAX=768 -DSEEK_PROVEN_ROOT -DSEEK_TRACE_MODEL=4 -DSEEK_TRACE_CHALLENGE -Iself-model calc/calc_s19.c -lm -o seek-self-taught
```

Use `--taps 1` for the single-depth ablation. Add `-DSEEK_TRACE_CLOCK` to the selected build for the adaptive clock. Compile without the trace flags for the matched reference. A model-free build continues to work without a learned header.

Pointer distributions over variable input sets and modeling internal search traces have research precedents. Seek’s specific experiment is forecasting the next iterative-deepening principal move and nominating challengers within its existing alpha-beta search. [Pointer Networks](https://arxiv.org/abs/1506.03134) · [Beyond A*](https://arxiv.org/abs/2402.14083)
