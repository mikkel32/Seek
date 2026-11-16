# The outside-data line: networks taught by Stockfish's evaluations of Lichess positions

These networks are **not self-taught**: their labels are Stockfish's. The self-taught line (`zero/`) is kept separate;
nothing is mixed. Same engine, same search, same hardware for every comparison; every opening is played with both colours.
Elo figures are followed by their 95% interval.

## Data

| Source | Positions kept | Shards | Converter |
|---|---|---|---|
| Lichess evaluation database (deep Stockfish analysis of positions) | 257M | `e0–e11` | `ext/lichess_hf.sh`, `ext/lconv.c` |
| Lichess game positions with the server's Stockfish analysis, 18 months (2023-12 to 2025-03) | 4,117M | `f0–f39`, `h0–h209` | `ext/fishnet.sh`, `ext/pq2csv.py`, `ext/lconv.c` |
| the same, 10 more months (2023-02 to 2023-11) | 2,683M (`i139` is held out and never trained on) | `i0–i139` | `ext/fishnet.sh`, `ext/more_months.sh` |
| Stockfish's own first-stage training file | 1,230M | deleted | `ext/bullet2shards.py` |

Kept positions: from move 9 on, side to move not in check, best or played move not a capture or promotion, no forced
mate (47% of the game positions). A position is 24 bytes and a 1-byte label (win chance of the side to move).

## What each step was worth

| Step | 30,000 positions per move (2,000 games) | 10+0.1 | 60+0.6 |
|---|---|---|---|
| Lichess deep evaluations (LE, 257M) against the self-taught network | +151 (137 to 165) | +137 (109 to 167) | +153 (117 to 194) |
| Stockfish's own first-stage training file against the self-taught network | −16 (−29 to −3) | | |
| + two months of game positions (LFf, 787M) against LE | +43 (31 to 55) | +58 (34 to 82) | |
| cheap layered read-out on the same 787M | +8 (−3 to 19) | | |
| 768 wide instead of 384 on the same 787M (LFW against LFf) | +55 (44 to 67) | +25 (4 to 47) | +37 (13 to 61) |
| all 4.37 billion positions, 384 wide (BIG against LFf) | +3 (−8 to 15) | | |
| all 4.37 billion positions, 768 wide (BIGW against LFW) | +8 (−3 to 20) | +17 (−4 to 39) | −14 (−40 to 12) |
| 768 wide instead of 384 on the 4.37 billion (BIGW against BIG) | +61 (50 to 73) | | |

The plain design is full: 5.5 times the data adds 3 Elo at 384 wide and 8 at 768 wide (at the clock +17 and −14, both inside the noise). Capacity is the limit, and width
buys it at a price in speed (1,462k positions per second at 384 wide, 1,180k at 768).

## Standing against Stockfish 14.1 (one thread each, 64 MB hash)

Our options: `PvPrune=1 ChkExt=0 SeDouble=20 CutIir=1`.

| Network | Stockfish 14.1, 10+0.1 | Stockfish 14.1, 60+0.6 | its hand-written evaluator, 10+0.1 | its hand-written evaluator, 60+0.6 |
|---|---|---|---|---|
| self-taught (N384) | −402 | | | |
| LE (257M, 384 wide) | −246 (−295 to −205), 200 games | | 0 (−38 to 38) | −21 (−62 to 20) |
| LFf (787M, 384 wide) | −215 (−257 to −178), 200 games | −191 (−233 to −153), 150 games | +56 (19 to 94) | +56 (20 to 94) |
| BIGW (4.37 billion, 768 wide) | −153 (−179 to −129), 400 games | −168 (−196 to −143), 300 games | +100 (74 to 127), 400 games | +51 (20 to 82), 200 games |
| BIGK (4.37 billion, 768 wide, 16 king buckets; engine `calc_s18`) | −143 (−169 to −119), 400 games | −163 (−190 to −137), 300 games | | |

## King-bucketed inputs

`calc/train_nnue.py --kb 16 --kf 1`, engine `calc/calc_s18.c`. Each view's input table is chosen by where its own king
stands (16 buckets, mirrored left-right), and every bucket's table is a shared table plus its own correction: sixteen
times the first-layer weights at nearly the same cost per position (2,503k positions per second against 2,567k on the
training machine, 1,059k against 1,170k on the match machine). `ext/bigk.sh`, `zero/results/vault_bigk.log`.

| Test | Games | Result |
|---|---|---|
| 16 king buckets (BIGK) against plain (BIGW), both 768 wide on the 4.37 billion, 30,000 positions per move | 2,000 | 654-879-467, +33 (21 to 44) |
| the same pair at the clock, 10+0.1 | 400 | 86-254-60, +23 (2 to 43) |
| the same pair at the clock, 60+0.6 | 200 | 34-144-22, +21 (−5 to 46) |
| the same design on all 7.06 billion positions, same number of training steps (BIGK2) against BIGK, 30,000 positions per move | 2,000 | 497-993-510, −2 (−13 to 9) |

Held-out loss on one million Lichess game positions no network was trained on (`ival`):

| Network | Loss | Mean error in win chance |
|---|---|---|
| LE (384 wide, 257M) | 0.54949 | 0.0691 |
| LFf (384 wide, 787M) | 0.54509 | 0.0614 |
| BIG (384 wide, 4.37 billion) | 0.54455 | 0.0605 |
| LFW (768 wide, 787M) | 0.54348 | 0.0587 |
| BIGW (768 wide, 4.37 billion) | 0.54280 | 0.0578 |
| BIGK (768 wide, 16 king buckets, 4.37 billion) | 0.54110 | 0.0552 |
| BIGK2 (768 wide, 16 king buckets, 7.06 billion) | 0.54049 | 0.0541 |

More positions do not help the bucketed design either: 60% more data lowers the held-out loss by 0.0006 and changes
nothing in games. Three data steps have now given +3, +8 and −2 Elo; the network's size is the limit, not the data.

The engine's integer copy of these networks (hidden layer in steps of 1/128) differs from the trained network by 1.9
centipawns on average on real balanced positions (`ext/qsim.py`); finer integers would bring that to 1.0 and are not
worth the change.

## Where the gap to Stockfish 14.1 is (16 November)

**The reference opponent is a slow build.** `/usr/games/stockfish` is the distribution's Stockfish 14.1, compiled for SSE2
only. On the same positions and the same node budget it needs 1.77 times as long as our engine on the match machine
(194k against 351k nodes per second of wall time under match load, `ext/nodetime.py`, `zero/results/lab_nodetime.log`).
Every equal-time result above therefore gives us almost twice Stockfish's nodes. A properly compiled Stockfish 14.1 would
be stronger than the opponent measured here.

**Equal nodes** (`ext/node_ladder.sh`, engine `calc/calc_s19.c`, which keeps a small node budget exactly; `calc_s18`
overshot to the next 2,048 and the first ladder run was discarded for that, `vault_ladder_unequal.log`). BIGK, standard options:

| Nodes per move, each side | Games | Against Stockfish 14.1 | Against its hand-written evaluator |
|---|---|---|---|
| 300 | 2,000 | +5 (−9 to +20) | +129 (114 to 144) |
| 3,000 | 2,000 | −185 (−200 to −170) | +165 (150 to 181) |
| 10,000 | 1,000 | −223 (−245 to −203) | |
| 30,000 | 1,000 | −247 (−268 to −228) | +189 |

With almost no search the two evaluators are level; the gap opens as the search deepens.

**It is not depth.** Geometric-mean nodes to finish a depth on the twelve bench positions (`ext/ebf.py`): ours 9,119 /
33,971 / 92,176 / 239,265 for depths 8 / 10 / 12 / 14, Stockfish 14.1 6,574 / 28,590 / 78,522 / 185,116.

**It is not a search setting.** Each against Stockfish 14.1 at 10,000 nodes, 1,000 games (`ext/sweep.sh`,
`vault_sweep.log`); `LmrPrune`, `LmpDepth`, `NmpMargin` are new options of `calc_s19`, off by default:

| Setting | Result |
|---|---|
| as it is | −223 |
| pruning rules count the reduced depth (`LmrPrune=1`) | −269 |
| late-move pruning to depth 12 (`LmpDepth=12`) | −232 |
| null move only well above beta at low depth (`NmpMargin=100`) | −231 |
| deeper reductions (`LmrDiv=180`) | −230 |
| deeper still (`LmrDiv=150,LmrBase=100`) | −237 |
| one ply more at expected cut nodes (`CutLmr=1`) | −213 |
| two plies more (`CutLmr=2`) | −237 |
| `LmrPrune=1,LmrDiv=180,CutLmr=1` | −258 |

Search statistics of our engine at 300,000 nodes per position (`-DSSTAT`): quiescence 28% of nodes; 60% of main-search
nodes end by reverse futility; 82% of move-loop cutoffs come on the first move; reduced searches are re-searched 1.5% of
the time; a null move succeeds in 41% of its tries.

**Whose evaluator is closer to a deep search** (`ext/whose_eval.py`, `ext/whose_eval2.py`; every evaluator gets its own best scale):

- Human-game positions (the held-out Lichess set, 60,000 positions, scale fitted per game phase): squared error of BIGK
  against Stockfish 14.1's network 0.0107 / 0.0140 with 2–6 pieces, 0.0102 / 0.0120 with 7–12, 0.0087 / 0.0106 with
  13–22, 0.0055 / 0.0075 with 23–32. Ours is the more accurate in every phase. (A first reading that showed a large
  deficit in balanced endgames was an artefact of one scale for all phases: Stockfish compresses its endgame scores.)
- Engine-game positions (18,554 quiet positions from our own engine's games at 30,000 nodes, each labelled by a 200,000-node
  Stockfish 16 search, `ext/engine_positions.py`). On the 74% that are balanced, always answering 50% has a mean error of
  0.0383. BIGK brings it to 0.0343 and names the better side in 62.9%; Stockfish 14.1's network 0.0315 and 65.6%;
  Stockfish 16's 0.0264 and 68.3%. The positions were chosen by our own search, which biases this test against our
  networks by an unknown amount.

**Engine-game training data.** Stockfish's second-stage training files (Hugging Face `linrock/bullet-training-data`, `S2`:
positions from Leela's self-play games with a search score and the game result, already quiet) are read by
`ext/s2data.sh` and `ext/bullet2shards.py`: 547 million positions per file in about four minutes, twelve files, two files
share 13% of their positions (`ext/overlap.py`). Their score unit was measured against our own evaluator (`ext/s2cal.py`):
600 units make one logit of win chance on the Lichess labels' scale. These labels are neither self-taught nor Stockfish's
play; this stays the outside-data line.

Probes: the same 768-wide, 16-bucket network from scratch on 1.02 billion positions, one pass (`ext/probe_v.sh`,
`vault_probe_v.log`), 2,000 games at 30,000 nodes against the Lichess-data reference and 1,000 games at 10,000 nodes against
Stockfish 14.1 (`ext/vs_sf.sh`):

| Probe | Against the reference | Against Stockfish 14.1 at 10,000 nodes | Engine positions: names the better side |
|---|---|---|---|
| Lichess game positions (reference) | | −281 (−306 to −259) | 63.2% |
| engine-game positions, score only | +5 (−7 to +17), 604-820-576 | −254 (−277 to −232) | 65.0% |
| 1536 wide, Lichess | +38 (28 to 49); search 31% slower | | |
| 1024 wide, Lichess | +12 (1 to 23) | | |

Held-out loss of further one-pass variants on the match machine's card (`lab_probe.log`): plain 768 0.54447; 16 buckets
0.54250; 16 buckets without the shared table 0.54335; 32 buckets 0.54253; 7 buckets 0.54263; half the learning rate 0.54290.

Logs: `zero/results/vault_*.log` (training machine), `zero/results/lab_*.log` (match machine).
