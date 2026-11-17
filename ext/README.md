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
| BIGK again with engine `calc_s19` | −137 (−161 to −115), 500 games | | | |
| **BIGM** (4.37 billion Lichess + 3.83 billion engine-game positions, same network; engine `calc_s19`) | **−116 (−136 to −97), 500 games** | **−115 (−139 to −93), 300 games** | | |

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

| Nodes per move, each side | BIGK against Stockfish 14.1 | BIGK against its hand-written evaluator | Stockfish 14.1 against its own hand-written evaluator |
|---|---|---|---|
| 300 | +5 (−9 to +20) | +129 (114 to 144) | +106 |
| 3,000 | −185 (−200 to −170) | +165 (150 to 181) | +343 |
| 10,000 | −223 (−245 to −203) | | |
| 30,000 | −247 (−268 to −228) | +189 | +469 |
| 300,000 | −218 (−242 to −195) | | |

(2,000 games per cell in the first two rows of our columns, 1,000 at 10,000 and 30,000, 500 at 300,000 and in the last column.)

With almost no search the two engines are level; the gap opens as the search deepens. Stockfish's network gains 363 Elo
on its own hand-written evaluator between 300 and 30,000 nodes with the same search; ours gains 60 on it.

The level score at 300 nodes is not evaluator parity. The same ladder for our other networks (`vault_vs_sf.log`; 2,000
games at 300 and 3,000 nodes, 1,000 or 500 above):

| Nodes per move, each side | self-taught N384 | LFf (Lichess, 384 wide) | Lichess probe | mixed-data probe | BIGK |
|---|---|---|---|---|---|
| 300 | −126 | −92 | −20 | −26 | +5 |
| 3,000 | −342 | −288 | −224 | −190 | −185 |
| 10,000 | −453 | −343 | −281 | −218 | −223 |
| 30,000 | | | −297 | −231 | −247 |
| 100,000 / 300,000 | | | | −194 | −218 |

What the level score at 300 nodes means was settled by letting each engine play itself with ten times fewer nodes
(`ext/scaling.sh`, `gauntlet_odds.py`, 500 games each, `vault_scaling.log`):

| Ten times the nodes, against itself | 300 → 3,000 | 3,000 → 30,000 |
|---|---|---|
| BIGM | +561 | +727 |
| self-taught N384 | +616 | +651 |
| Stockfish 14.1 | +1009 (497 wins, 3 draws, no loss) | +685 |
| Stockfish 14.1, hand-written evaluator | +659 | +861 |

Stockfish's network is unusually weak with almost no search; from 3,000 nodes on it scales no better than we do. So the
gap is not a scaling defect of ours and the 300-node result is the outlier: from 3,000 nodes up we are a roughly constant
180 to 250 behind with BIGK and 175 to 190 behind with BIGM.

**Search or evaluator? The evaluator held equal.** The installed Stockfish's network (found inside its binary: 22,528
inputs = 32 mirrored king buckets × 704, 1024 units a side, eight material buckets each with layers 2048 → 8 → 32 → 1;
larger than ours, which has 16 buckets × 768 inputs and 768 units a side) was re-implemented in a diagnostic build of our
engine (`-DSFNET`, never shipped, no weights in this repository; `ext/sfnet_extract.py`, `ext/sfnet_ref.py`,
`ext/sfnet_check.py`: our arithmetic equals the engine's own `eval` output on 2,000 positions to the printed precision).
Our search with Stockfish's network against Stockfish 14.1, 10,000 nodes each (`ext/sfnet_matches.sh`, `vault_sfnet.log`):

| Match | Games | Result |
|---|---|---|
| our search + Stockfish's network against Stockfish 14.1 | 1,000 | 157-319-524, **−134 (−153 to −116)** |
| our search + BIGM against Stockfish 14.1 (for comparison) | 4,000 | −173 (−183 to −164) |

The result depends on how Stockfish's units are converted to ours (`EvalScale`, 100 = 361 of its units to 100 of ours):
−163 at 70, −134 at 100, **−105 (−123 to −88) at 140**, −141 at 200, −254 at 280. Stockfish's network tolerates a larger
scale than ours do (ours lose badly above 100), i.e. our pruning margins can be tighter against its values. At the best
setting we are about 105 behind with the same evaluator on both sides, and in our own search Stockfish's network beats
BIGM by 43 (26 to 61) at scale 100: of the equal-node gap, roughly 105 is the search and 45 to 70 the network.

**Search changes measured against the standing search** (same network BIGM, 10,000 nodes a move, 4,000 games each,
`lab_hist_test.log`; options of `calc_s19`, all off by default):

| Change | Result |
|---|---|
| `HistV2=1`: continuation histories four and six plies back, the one-ply one counted twice in move ordering | +3 (−6 to +11) |
| `HistV2=2`: a counter-move slot after the killers | +3 (−6 to +11) |
| `HistV2=4`: a best move that raised alpha without cutting off is rewarded like a cutoff move | +7 (−1 to +16) |
| `HistV2=8`: when every reply fails low, the move that led there is rewarded | 0 (−8 to +9) |
| `HistV2=16`: when a node cuts off, the opponent's first-choice quiet move that led there is penalised | −14 (−22 to −5) |
| **`HistV2=7`** (the first three together) | **+19 (10 to 27)** |
| `HistV2=15` | +19 (10 to 27) |
| on top of `HistV2=15`, `PruneV2=1`: skip late quiet moves with two negative continuation histories at reduced depth under 5 | −36 (−45 to −27) |
| on top of `HistV2=15`, `PruneV2=2`: futility on the reduced depth with a history term | 0 (−9 to +8) |

The history batch does not survive deeper searches: `HistV2=7` against the standing search is +19 at 10,000 nodes,
−10 (−21 to +1, 2,000 games) at 30,000, −9 (−22 to +5, 1,000 games) at 100,000 and 144-661-195, −18 (−30 to −5), at 10+0.1;
it has no cost in speed (487k against 486k nodes a second under match load). It is not adopted. At 100,000 nodes the parts
separately: `HistV2=1` +1 (−12 to +15), `HistV2=4` −15 (−29 to −2): rewarding best moves that did not cut off is what turns
against a deep search. The lesson is about the
test: 10,000 nodes a move rewards what suits a shallow search, so search changes have to be judged at 100,000 nodes or at the clock.
The re-run of the earlier options at 100,000 nodes was stopped when the work turned to training.


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
`vault_probe_v.log`), 2,000 games at 30,000 nodes against the Lichess-data reference and 1,000 games against Stockfish 14.1
at 10,000 nodes each (`ext/vs_sf.sh`, `vault_vs_sf.log`). The new files' labels were written on a scale about 1.6 times
ours, so those networks are also played with the engine's new `EvalScale` option (the network's value in percent as the
search sees it); the best setting tried is given.

| Probe | Against the reference, 30,000 nodes | Against Stockfish 14.1, 10,000 nodes | at 300 nodes | at 3,000 nodes |
|---|---|---|---|---|
| Lichess game positions (reference) | | −281 (−306 to −259) | −20 | −224 |
| engine-game positions, score only | +5 (−7 to +17), 604-820-576 | −254; −243 with `EvalScale=60` | −56 | −218 |
| engine-game positions, 0.75 score + 0.25 game result | −39 (−51 to −27) | −264; **−218 with `EvalScale=70`**; −231 with 50 | | |
| half Lichess, half engine-game (score only), 1.09 billion | +57 (45 to 68) | −238; −216 with `EvalScale=75` | | |
| the same mix, engine-game labels with the game result | +39 (27 to 50) | −241; −209 with `EvalScale=65` | | |
| the same mix, engine-game labels rescaled to ours (score scale 950) with the game result (`ext/s2yc.py`, `--ytag yc`) | **+60 (48 to 71)** | **−218; −216 with `EvalScale=80`** | | |
| 1536 wide, Lichess | +38 (28 to 49); search 31% slower | | | |
| 1024 wide, Lichess | +12 (1 to 23) | −240 | | |

For scale: BIGK, the same design after four passes over 4.37 billion Lichess positions, scores −223 against Stockfish at
10,000 nodes; with `EvalScale` 60 / 80 / 130 it scores −225 / −224 / −250 (its scale is right; a value that is too large
costs far more than one that is too small: the engine-game probe at 130 scores −341).

Every variant with engine-game positions lands between −209 and −218 against Stockfish once its scale is right, about 65
better than the same network on Lichess positions alone and level with BIGK, which had sixteen times the training. The
label details do not separate at 1,000 games; the rescaled labels need no scale correction and are the ones used at full
scale: **BIGM** = the same network, two passes over the 4.37 billion Lichess positions and 3.83 billion engine-game
positions (seven of the twelve files; the disk holds no more next to the Lichess shards), `ext/big_run.sh`.

**BIGM, the full-scale mixed-data network** (`vault_big_run.log`, `lab_bigm_clock.log`; held-out Lichess loss 0.54174, mean error 0.0552):

| Test | Games | Result |
|---|---|---|
| BIGM against BIGK, 30,000 positions per move | 2,000 | 709-853-438, +47 (36 to 59) |
| BIGM against BIGK, 10+0.1 | 500 | 131-309-60, +50 (31 to 69) |
| BIGM against Stockfish 14.1, 10,000 nodes each | 1,000 | −188 (−209 to −169); BIGK −223, the one-pass mixed probe −218 |
| BIGM against Stockfish 14.1, 30,000 nodes each | 1,000 | −185 (−204 to −167); BIGK −247, the one-pass mixed probe −231 |
| standing against Stockfish 14.1, 10+0.1 | 500 | −116 (−136 to −97); BIGK −137 and −143 |
| standing against Stockfish 14.1, 60+0.6 | 300 | −115 (−139 to −93); BIGK −163 |
| BIGM against BIGK, 60+0.6 | 200 | 38-150-12, +45 (22 to 69) |
| BIGM against Stockfish 14.1, 10,000 nodes each, again on the match machine | 4,000 | −173 (−183 to −164) |

Bigger networks on the mixed data, one pass over 1.09 billion positions, against the mixed reference probe at 30,000
nodes (2,000 games, `vault_probe_v.log`): 32 king buckets −3 (−14 to +9); 1024 wide +14 (3 to 25), held-out loss 0.54287
against 0.54329.

**Width at the clock.** The 1536-wide probe (one pass over the same 1.09 billion mixed positions, `lab_probe_mixed.log`):

| Test | Games | Result |
|---|---|---|
| against the 768-wide reference probe, 30,000 nodes | 2,000 | 677-885-438, +42 (30 to 53) |
| against Stockfish 14.1, 10,000 nodes each | 2,000 | −182 (−196 to −169); reference −218 |
| against the reference probe, 10+0.1 | 1,000 | 172-576-252, **−28 (−42 to −14)** |

Search speed on equal node budgets (`ext/nodetime.py`): on the quiet training machine 768 wide 2.92 million nodes a second,
1024 wide 2.57 million (−12%), 1536 wide 2.17 million (−26%); on the match machine under match load (76 games on 88
threads) 768 wide 443k to 536k, 1536 wide 239k to 299k (about −45%). A wider network gains at equal nodes and gives it back
in speed; under our match conditions it gives back more than it gains. The installed Stockfish needs 1.9 times our time per
node on the quiet machine as well (1.55 against 2.96 million nodes a second), so that handicap is its build and its larger
network, not the load.

At 60+0.6 the 1536-wide probe against the reference probe: 32-230-38, −7 (−26 to +12), 300 games.

**Teachers.** A 768-wide student whose training target is part label, part another network's judgment of the same
position (`--distill <weights> --dw <teacher's share>`; one pass over the same 1.09 billion mixed positions; `lab_probe_mixed.log`;
2,000 games per cell):

| training target | against the label-trained reference probe, 30,000 nodes | against Stockfish 14.1, 10,000 nodes each |
|---|---|---|
| labels only (the reference) | 0 | −218 |
| BIGM's judgment only (`--dw 1.0`) | +12 (1 to 23) | −204 (−219 to −190) |
| Stockfish 14.1's network only (`--distill sf:<file> --dw 1.0`) | +21 (9 to 32) | −208 (−222 to −194) |
| half labels, half BIGM | +27 (16 to 38) | −195 (−209 to −181) |
| half labels, half Stockfish 14.1's network | **+38 (27 to 50)** | **−193 (−208 to −180)** |

A teacher's judgment added to the label is the better target at equal training, and the mix beats either alone.
`ext/sfteach.py` is the installed Stockfish's network in matrix form for the graphics card (equal to the verified `-DSFNET`
engine build on 2,000 positions, to the unit); `--distill sf:<file>` takes its value as Stockfish's search sees it,
brought to the labels' scale. Several teachers joined by `+` are averaged (mean of their win chances). Networks taught
this way are Stockfish's judgments once more (this line always was Stockfish-taught).

A teacher added to a finished network gives little: BIGM continued for one pass over the same 1.09 billion positions
with the half-Stockfish target at a quarter of the learning rate (`ext/lab_probe3.sh`) scores +7 (−4 to +18) against BIGM
at 30,000 nodes and −173 (−187 to −159) against Stockfish 14.1 at 10,000 (BIGM: −173). **BIGS** is the from-scratch test:
768 wide, two passes over the full mix, half labels and half the mean judgment of BIGX and Stockfish's network
(`ext/after_bigx.sh`).

32 king buckets together with 1024 wide: +12 (1 to 23), no better than 1024 wide alone.

**BIGX, the full-scale 1536-wide network** (16 buckets, 18.9 million weights, two passes over 1.8 billion Lichess, 6.02
billion second-stage and 0.35 billion first-stage positions; `vault_big_run.log`, `lab_bigx_clock.log`):

| Match | Games | BIGX | BIGM for comparison |
|---|---|---|---|
| against BIGM, 30,000 nodes a move | 2,000 | **+44 (33 to 56)** | |
| against Stockfish 14.1, 10,000 nodes each | 1,000 | −145 (−164 to −127) | −173 (4,000 games) |
| against Stockfish 14.1, 30,000 nodes each | 1,000 | −159 (−177 to −142) | −185 |
| against BIGM, 10+0.1, 76 boards at once | 500 | −14 (−33 to +5) | |
| against BIGM, 10+0.1, 44 boards at once (one per physical core) | 500 | −1 (−20 to +17) | |
| against BIGM, 60+0.6, 76 boards | 200 | +10 (−10 to +31) | |
| against Stockfish 14.1, 10+0.1, 76 boards | 500 | −134 (−157 to −113) | −116 (−136 to −97) |
| against Stockfish 14.1, 10+0.1, 44 boards | 500 | −134 (−156 to −113) | −115 (−135 to −96) |
| against Stockfish 14.1, 60+0.6, 76 boards | 300 | −127 (−150 to −105) | −115 (−139 to −93) |

In our own search BIGX is worth what Stockfish 14.1's network is (that network: +43 over BIGM; BIGX: +44), and at the clock
the gain is spent on speed: level with BIGM head-to-head, no better against Stockfish. The standing does not depend on the
load of the match machine (BIGM −116 with 76 boards at once, −115 with 44).

**Where the evaluator's time goes** (`ext/evalcost.py`, a `-DTSTAT` build that counts processor cycles; match machine,
nothing else running, bench positions at depth 14; `lab_evalcost.log`):

| | cycles per position searched | bringing the sums up to date | read-out | evaluator's share |
|---|---|---|---|---|
| BIGM (768 wide) | 1,811 | 511 per evaluation, 26% | 367, 19% | 45% |
| BIGX (1536 wide) | 2,537 | 909, 33% | 700, 25% | 58% |
| layered read-out probe (768 wide, 2×64 → 16 → 32 → 1 on top) | 2,546 | 587, 21% | 863, 31% | 52% |

BIGX costs 40% more cycles per position (29% fewer positions a second alone; under 76 copies at once 292k against 537k,
45% fewer). Both parts are arithmetic on data already in the near caches, not waiting for memory: asking for the input
matrix rows of a move as soon as it is made (4 to 48 cache lines per row) made both networks slower, alone and under
load (`lab_pf_bench.log`: BIGX 22.2 million a second in total over 76 copies without, 22.1 to 19.0 with; BIGM 40.8 against
39.5 to 36.5), and 512-bit instead of 256-bit vectors for the compiler's loops changed nothing on the training machine.
The option was removed again.

Tried on top of the mixed recipe and not adopted:

- Steeper training labels (`--sharp`, the loss sees errors near equality larger; played with `EvalScale` = 100 / sharp):
  sharp 2 against the reference probe +5 (−6 to +16), against Stockfish 14.1 at 10,000 nodes −210 and −209 (reference −218);
  sharp 1.5 against the reference +2 (−9 to +13).
- History tables indexed by whether the squares are attacked (`ThrHist=1`): −202 against a −218 baseline at equal nodes
  (4,000 games each), but with BIGM at 10+0.1 against the standing options 141-693-166, −9 (−21 to +3): its cost in time
  outweighs it.
- 85% engine-game positions instead of half: −9 against the reference, −227 against Stockfish (reference −218).

**This network size is full again.** BIGM2 = BIGM continued for one pass over 3.56 billion positions it had not seen
(four more second-stage files, the first-stage sample, and 1.02 billion Lichess positions) at a quarter of the learning
rate: against BIGM −6 (−16 to +5, 2,000 games at 30,000 nodes); against Stockfish 14.1 −192 at 10,000 nodes and −183 at
30,000 (BIGM −188 / −173 and −185). More data and more training no longer help at 768 wide with 16 buckets.

Composition of the mix hardly matters at probe scale: 85% engine-game positions against the half-and-half reference
−9 (−20 to +3); a third each of Lichess, second-stage and first-stage (Stockfish's own self-play, `ext/s1data.sh`)
positions +5 (−7 to +16).

At the clock the result-mixed probe with `EvalScale=65` stands at −168 (−191 to −145, 500 games) against Stockfish 14.1 at
10+0.1 (`lab_probe_clock.log`).

The search self-tuning on BIGK (15,000 iterations at 20+0.2) ended with every value within a few percent of where it
started; tuned against standing values at 10+0.1: 160-660-180, −7 (−20 to +6), 1,000 games. The standing values stay.

Architecture on the mixed data, same recipe with one change, on the match machine's card (`ext/lab_probe.sh`,
`lab_probe_mixed.log`; 2,000 games against the mixed reference probe at 30,000 nodes, 2,000 against Stockfish 14.1 at
10,000 nodes; speed on equal node budgets, `ext/nodetime.py`):

| Change | Against the reference | Against Stockfish 14.1 (reference: −218, −229 to −208 over 4,000 games) | Search speed |
|---|---|---|---|
| product terms (`--prod 1`) | +6 (−5 to +18) | −213 (−228 to −199) | −18% |
| layered read-out on 64 units a side (`--head 16,32 --hk 64`) | +12 (1 to 23) | −207 (−222 to −193) | −15% |

Neither pays for its speed. Search options under the new network family, the mixed probe against Stockfish 14.1 at 10,000
nodes, 4,000 games each (`ext/lab_sweep.sh`, `lab_sweep.log`): as it is −218; `CutLmr=1` −211; `ProbCut=150` −217;
`NegExt=1` −217; `TtCapLmr=1` −214; `TtBucket=1` −217; `NmpBase=4` −217; `SeDepth=6` −212; `SeDouble=0` −220. Each
interval is ±10: nothing moves.

Two readings. The engine-game network is worse than the Lichess one with almost no search (−56 against −20 at 300 nodes)
and better with more (−243 against −281 at 10,000): it gives up shallow tactical knowledge the search supplies anyway and
gains what the search cannot supply. And a match between two of our own networks does not predict the match against
Stockfish (the result-mixed probe loses to the reference by 39 and is 63 better against Stockfish): candidates are now
judged against Stockfish at 10,000 nodes or more.

Held-out loss of further one-pass variants on the match machine's card (`lab_probe_loss.log`): plain 768 0.54447; 16 buckets
0.54250; 16 buckets without the shared table 0.54335; 32 buckets 0.54253; 7 buckets 0.54263; half the learning rate 0.54290;
twice the learning rate 0.54289.

Logs: `zero/results/vault_*.log` (training machine), `zero/results/lab_*.log` (match machine).

## Search quality at a fixed node budget (16 November, evening)

From here on the work runs on one rented machine (the training machine: 24 threads, one card). The match machine was
stopped; its scripts are in `ext/lab/`, its last logs in `zero/results/lab_*.log`.

**Depth at equal nodes** (`ext/depth_at_nodes.py`, 12 positions, mean depth of the last finished iteration):

| Nodes per move | Our depth | Stockfish 14.1's depth |
|---|---|---|
| 3,000 | 6.5 | 7.7 |
| 30,000 | 9.7 | 10.8 |
| 300,000 | 14.5 | 16.1 |
| 3,000,000 | 20.8 | 24.1 |

Every search setting we have was tuned and judged at 2,500 to 30,000 nodes a move. The options meant to deepen the search
(`CutLmr`, `ProbCut`, `NmpDiv`, `TtCapLmr`, `LmrPrune`) change the depth at 3,000,000 nodes by at most 0.7 plies together.
Games cannot judge a deep search at a useful rate: at 1,000,000 nodes a move the machine plays about 660 games an hour.

**The test** (`ext/quality.py`). Positions from games between BIGM and Stockfish 14.1 at 30,000 nodes (8 per game, 1,600
games, 12,775 distinct positions); Stockfish 14.1 analyses each with 4,000,000 nodes and two lines (positions already
decided by more than 6 pawns after a quick look are set aside: 8,964 remain). A configuration then chooses a move in every
position with a fixed node budget and a fresh hash table; the move is scored against Stockfish's analysis (a move outside
its two lines by a Stockfish search of that move alone at half the budget, cached). Loss of a move = expected score of
Stockfish's best move minus that of the move chosen, from 1 / (1 + exp(−centipawns / 200)). The unit below is a thousandth
of a point per move. Our engine is deterministic at a node budget, so two configurations differ only where they choose
differently and the comparison is paired.

First measurements (`zero/results/vault_quality.log`; BIGM and BIGX with the standing options, Stockfish 14.1 as installed):

| Nodes per move | BIGM: loss, one-good-move positions solved | Stockfish 14.1 | BIGX |
|---|---|---|---|
| 30,000 | 19.76, 76.3% | 14.99, 79.7% | 18.64, 76.4% |
| 100,000 | 13.96, 81.8% | 9.57, 86.4% | |
| 300,000 | 10.04, 86.1% | 6.31, 91.3% | 9.69, 86.2% |
| 1,000,000 | 7.78, 88.9% | 3.79, 95.4% | |

Stockfish is judged by its own deeper search here, which flatters it more the closer its budget comes to the judge's
(its loss must reach zero at 4,000,000 nodes; ours cannot, because our evaluator disagrees with the judge's in some
positions at any depth). Comparisons between our own configurations do not have that problem.

Paired differences (negative = the first gives away less; standard error in brackets):

| Comparison | Difference | What games say |
|---|---|---|
| BIGM 100,000 against 30,000 nodes | −5.80 (0.49) | |
| BIGM 300,000 against 100,000 | −3.92 (0.35) | |
| BIGM 1,000,000 against 300,000 | −2.26 (0.25) | |
| Stockfish against BIGM, 30,000 each | −4.77 (0.56) | −185 for us (−204 to −167), 1,000 games |
| Stockfish against BIGM, 100,000 each | −4.39 (0.40) | −194 for us (−211 to −178), 1,000 games |
| Stockfish 30,000 against BIGM 100,000 | +1.03 (0.46) | +21 for us (6 to 37), 1,000 games |
| Stockfish 100,000 against BIGM 300,000 | −0.47 (0.34) | −33 for us (−47 to −18), 800 games |
| BIGX against BIGM, 30,000 each | −1.12 (0.56) | +44 (33 to 56), 2,000 games |
| BIGX against BIGM, 300,000 each | −0.36 (0.29) | |
| `HistV2=7` against the standing search, 30,000 | −0.15 (0.58) | −10 (−21 to +1), 2,000 games |
| `HistV2=7` against the standing search, 300,000 | +0.49 (0.31) | −9 (−22 to +5) at 100,000 nodes, −18 (−30 to −5) at 10+0.1 |
| `EvalScale=140` against 100, 100,000 | +0.73 (0.40) | worse in every match played |

The signs agree with the games everywhere (`zero/results/vault_calib.log`; `gauntlet.py --opp_nodes` gives the opponent its
own budget). Over the four Stockfish rows one unit is worth about 40 Elo, and the node-tripling row gives the same (206 Elo
for 5.80 units), so a comparison on these 8,964 positions has a standard error of
about 20 Elo at 30,000 nodes (half a minute of searching plus about a minute of Stockfish checks), 15 at 100,000, 11 at
300,000 (a minute and a half plus under a minute) and 9 at 1,000,000 (four and a half minutes plus under a minute). Games
between two of our own configurations reach the same standard errors on this machine in under a minute, three minutes,
about twenty minutes and well over an hour and a half: the test buys nothing at 30,000 nodes, little at 100,000, a
factor of about nine at 300,000 and about twenty at 1,000,000. It is the instrument for deep searches, where games had
stopped being one. Its precision grows with the number of positions; a second collection is being analysed (`ext/q_fill.sh`).

The two unequal matches say something by themselves: with three times Stockfish's nodes we are 21 ahead at 100,000 against
30,000 and 33 behind at 300,000 against 100,000. Stockfish's search is worth about 2.7 times our nodes at 30,000 and more
than three times at 100,000: its lead, counted in nodes, grows with depth.

**The search alone** (`zero/results/vault_sfnet_quality.log`). The diagnostic build that runs our search on Stockfish 14.1's
own network (`-DSFNET`, `EvalScale=140`) takes the evaluator out of the comparison: it is the judge's on both sides.

| Nodes per move | Our search, Stockfish's network | Stockfish 14.1 | Difference | Stockfish's search is worth |
|---|---|---|---|---|
| 30,000 | 18.57 | 14.99 | +3.58 (0.54) | 2.0 times our nodes |
| 100,000 | 12.35 | 9.57 | +2.77 (0.38) | 2.8 times |
| 300,000 | 9.33 | 6.31 | +3.01 (0.30) | 3.1 times |
| 1,000,000 | 6.07 | 3.79 | +2.27 (0.21) | |

(Last column: where Stockfish's loss falls on our own curve between two budgets. Directly: Stockfish at 100,000 nodes
against our search at 300,000 is +0.24 (0.34), at 300,000 against our 1,000,000 +0.25 (0.24): level.) With the same
evaluator our search needs about three times Stockfish's nodes from 100,000 nodes up, twice at 30,000. That factor of
three, about 3 units or 120 Elo at equal nodes, is what the search work is after; this table is its yardstick.

Which best moves we miss (`ext/q_classes.py`, the 3,093 one-good-move positions, share solved at 300,000 nodes, BIGM
against Stockfish): quiet moves 81.6% against 87.1% (1,454 positions), safe captures 93.1 / 96.9 (999), king moves
83.2 / 90.6 (256), quiet checks 88.3 / 93.8 (128), captures that give up material at first sight 82.7 / 91.8 (98), quiet
moves that offer a piece 73.7 / 84.2 (38). The deficit has the same proportion in every class: no single pruning rule is
throwing a kind of move away.

**Reducing or pruning more does not help at depth** (`zero/results/vault_screen300k.log`; each option against the standing
options at 300,000 nodes, same unit, positive = worse, standard error 0.27 to 0.31): `CutLmr=1` +0.31, `CutLmr=2` +0.34,
`LmrBase=108` +0.45, `LmrBase=148` +0.19, `LmrDiv=200` +0.14, `LmrDiv=170` +0.43, `NmpBase=4` +0.21, `NmpDiv=3` +0.16, both
+0.05, `CutLmr=1,LmrBase=108` +0.72, `CutLmr=2,LmrBase=108,NmpBase=4` +0.15, `SearchV3=48` +0.13, `SearchV3=16` −0.10,
`SearchV3=64` −0.17, `SearchV3=1` +0.51. Twelve ways of looking less at late moves, all on the worse side of zero: the
standing amounts are not too timid for a deep search. With `LmrBase=148` the search reaches 19.0 instead of 17.9 plies at
1,000,000 nodes (21.2 with every such option together, Stockfish's depth) and chooses no better. Stockfish's factor of
three is not in how much is reduced but in which moves are. (The scan was stopped there; the processor went to the
generation loop.)

## BIGS: the 768-wide student of two teachers (16 November)

`ext/after_bigx.sh`: the BIGM recipe from scratch on the same 8.2 billion positions, the target half the label and half
the mean of two teachers' win chances (BIGX and Stockfish 14.1's network, `--distill ...+sf:... --dw 0.5`). Same size and
speed as BIGM (1,270k against 1,299k positions a second on the bench).

| Match (`zero/results/vault_big_run.log`) | Result |
|---|---|
| BIGS against BIGM, 30,000 nodes, 2,000 games | 602-886-512, **+16 (4 to 27)** |
| BIGS against Stockfish 14.1, 10,000 nodes each, 1,000 games | −179 (−199 to −160); BIGM −173 |
| BIGS against Stockfish 14.1, 30,000 nodes each, 1,000 games | −162 (−179 to −145); BIGM −185 on the same machine the same evening |

At the clock on the one machine now in use (22 boards, 1,000 games each, `zero/results/vault_clock.log`):

| 10+0.1 against Stockfish 14.1 | Result |
|---|---|
| BIGM | 26-590-384, −130 (−143 to −117) |
| **BIGS** | 34-624-342, **−111 (−123 to −98)** |

BIGS is the standing network: +20 at the clock, +16 at equal nodes, the same speed. (BIGM's −116 in the table further up
was played on the stopped match machine; on this machine it is −130, so numbers from the two machines are not compared.)

## Generations on this line (from 16 November, evening)

The self-taught line improved for nine generations of "play yourself, learn from it, beat your predecessor" and then
stopped (main README). The same loop now runs on the outside-data line. The player is the strongest network at equal
nodes (BIGX, 1536 wide), playing itself at 5,000 nodes a move (`ext/gen_farm.py`: positions that were played with the
loop's lambda-return labels, and positions from inside the search trees with their search score). A child is its parent's
weights trained further with the parent's own recipe on that data mixed with a slice of all the earlier data
(`ext/gen_round.sh`), and is then played against its parent at 30,000 nodes (2,000 games) and against Stockfish 14.1 at
10,000. Every round has a control: the same further training without the self-play data.

Round 1a, 53 million self-play positions (9.5 million played, 44 million from the trees), each child against its own
parent over 2,000 games at 30,000 nodes (`zero/results/vault_gen_round.log`):

| Child | Against its parent | Against Stockfish 14.1 at 10,000 nodes (BIGS −179, BIGX −145) |
|---|---|---|
| G1a: BIGS, self-play positions 22% of the mix, learning rate 3e-4 | −3 (−14 to +8) | −165 |
| C1a: BIGS, the same further training without self-play (control) | +6 (−4 to +17) | −150 |
| X1a: BIGX on its own self-play, same mix | −7 (−18 to +4) | −137 |
| G1b: BIGS, self-play 37% of the mix, learning rate 1e-3 | **−18 (−29 to −7)** | −179 |

No child beats its parent, the control is the best of them, and the stronger dose is clearly worse. These networks already
hold labels from far deeper analysis than a 5,000-node search of ours; learning from their own shallow search pulls them
down. The self-taught line climbed by generations because it had no better teacher. Self-play is paused.

## A learned judgment of late moves (16 November, night)

`calc_s19.c -DLSEL` searches a sample of the moves after the first at full depth whatever the rules say and writes what
was known about the move beforehand (34 features: move number, histories, exchange value, static evaluation minus the
bound, what the hash table says, node type, checks, attacked squares, ...) with whether the full-depth look beat the bound
(`ext/lsel_collect.py`; 2,399 positions from self-play on openings the test sets do not use, 300,000 nodes each, 16.5
million records). `ext/lsel_fit.py` fits what the standing rule knows (its reduction, the depth, the pruning rule that
would have fired), an additive model over buckets of every feature, and a small network, and judges them on a fifth of
the searches held out (1.2 million records):

| Predictor | Log-loss removed | Area under the curve | Depth 8+: share of the moves that matter inside the half judged safest |
|---|---|---|---|
| standing rule | 15.3% | 0.822 | 10.4% |
| additive model | 31.6% | 0.907 | 2.0% |
| small network | 33.2% | 0.914 | 1.8% |

The additive model went into the engine (options `Sel`, `SelA`, `SelB`, `SelFile`; reduction = SelA/100 − SelB/100 × (log-odds
+ 5) plies; 5% slower). Against the standing rule at 300,000 nodes on the quality test (`zero/results/vault_sel_test.log`,
standard error 0.26 to 0.30): 210/50 +0.11, 175/75 −0.33, 135/100 +0.16, 260/50 +0.42, 185/100 +0.06, 160/50 −0.02. Level.
The model tells far better which late moves matter, and that buys nothing: a move that beats the bound at full depth
nearly always beats it in the reduced look too and is then searched again in full, so the standing rule was not losing
those moves. Not adopted.

Where the nodes go (`-DSSTAT`, 1,000,000 nodes a position): quiescence 32%, main nodes with one ply left 33%, two plies
14%, three 7.5%, four 5%, everything deeper 9%. By what the search was doing: inside a reduced look 67%, inside a null-move
try 17 to 20% (a try succeeds 45 to 49% of the time), re-searches after a reduced look liked the move 5.5%, singular tests
4%, full-window re-searches 2 to 6%. No single bucket explains a factor of three.

## Stockfish 14.1's known settings in our search, and which rule hides the moves we miss (16 November, night)

Groups of our options set to Stockfish 14.1's values as far as they are known from memory, on the search-only yardstick
(`ext/sf_bundle.sh`: our search on Stockfish's network, 100,000 nodes; the standing options are +2.77 behind Stockfish
itself there; `zero/results/vault_sf_bundle.log`; standard error 0.33 to 0.40):

| Group | Against the standing options |
|---|---|
| reductions: `LmrDiv=214,LmrBase=52,CutLmr=2,TtCapLmr=1,SearchV3=2,PruneV2=8` | −0.50 |
| history bonuses: `HbQuad=6,HbLin=229,HbMax=4167` | −0.37 |
| null move: `NmpBase=4,NmpDiv=3,NmpMargin=62` | −0.14 |
| singular extensions: `SeMul=1,SeDouble=29,NegExt=1` | +0.08 |
| late-move pruning at every depth: `LmpBase=3,LmpDepth=63` | +0.14 |
| static margins: `Rfp=83,ProbCut=81` | +0.57 |
| pruning on the reduced depth: `LmrPrune=1` | +0.64 |
| all of them | +0.66 |

Nothing is established and the whole set is worse: Stockfish's headline numbers in our frame do not make Stockfish's search.

`ext/q_fail.py` with the diagnostic option `Off` (single rules switched off): the one-good-move positions Stockfish solves at
100,000 nodes and we still miss at 1,000,000 are searched again with one rule off at a time (`vault_fail1.log`, BIGM, 162
positions; `vault_fail2.log`, our search on Stockfish's network, 98 positions). With BIGM the standing search solves 45% of
them at 3,000,000 nodes and 63% at 10,000,000; at the same 1,000,000 nodes, without null move 43%, without reverse futility
46%, without reductions 48%, without late-move pruning 41%, without quiet futility 49%, without losing-exchange pruning
48%, without bad-history pruning 27%, without singular tests 36%, with the check extension 52%. Any change, including
clearly harmful ones, solves about half. The set was chosen by one configuration's failures, and a failure there is close
to a coin toss between two moves: no rule is throwing these moves away. The same holds with Stockfish's network in our
search (46 to 60% for every variant).

## Inside the network, and the shelved architecture study taken up again (17 November)

`ext/brain_scan.py` on BIGS, 400,000 held-out positions (`zero/results/vault_brain_scan_BIGS.log`): a position has 10.5% of
the hidden units active and 0.8% saturated; no unit is dead, 303 of 768 are active in under 5% of positions; the top 256
units carry 69% of the hidden part's variance, and averaging out the weakest 20% raises the held-out loss from 0.5438 to
0.5452 (nothing to prune for free). The engine's integer scale is set by a worst-case bound of 244 per unit (bias plus the
32 largest per-square maxima), which gives each unit's working range 128 steps; the largest sum any unit reaches on real
positions is 53. The engine differs from the trainer by 3.1 centipawns on average (18 at most) for BIGS.

`ext/brain_err.py`: on 15,553 positions with Stockfish's 4,000,000-node analysis as the reference, squared error in expected
score (thousandths): BIGM 24.0, BIGS 22.7, BIGX 22.2, Stockfish 14.1's own network 18.6 (it is judged by its own search).
Stockfish's network is ahead by a similar ratio (0.73 to 0.85) in every category tried: quiet and tactical positions,
attacked kings, passed pawns, material imbalances, piece counts (`zero/results/vault_brain_err.log`). No blind spot.

A meaningless change shows how the hash table differs from Stockfish's, not the search (`zero/results/vault_stab.log`,
300,000 nodes): another table size changes our move in 19% of the positions and Stockfish's in 1.1%; with two entries a
slot ours changes in 1.3%, and plays +0.72 (0.28) worse on the quality test; a 512 MB table plays the same as 64 MB. A 1%
change of the node budget changes 0.1% of our moves and 0.4% of Stockfish's.

What is and is not up to date. The 40-variant study of 15 November (`zero/results/architecture_probe.md`: attack-map
inputs, threat inputs, layered read-outs, the two-speed design) ran on 59 million self-play positions with 384-wide plain
networks and was shelved "for the point where the data is several times larger". The data is now 8.2 billion positions
and the plain design is measured as full. Re-tested on the current family so far: product terms (+6) and the layered
read-out on 64 units a side (+12). Never run: the queen-aware read-out heads (`--obk`), the loss exponent (`--pow`), weight
clipping (`--wclip`), the expert router and attack maps at this scale. `ext/vault_probe.sh` runs them on the current recipe
(768 wide, 16 king buckets, half labels and half the two teachers, 1.09 billion positions); `calc/train_nnue.py --am`
adds attack-map inputs to the main trainer (not yet in the engine).

Probes on the current recipe, each one pass over 1.09 billion positions (`zero/results/vault_vault_probe.log`; reference
`r0`: held-out loss 0.54415, −195 against Stockfish 14.1 at 10,000 nodes each):

| Probe | Held-out loss | Against `r0`, 30,000 nodes, 2,000 games | Against Stockfish 14.1, 10,000 nodes |
|---|---|---|---|
| 16 read-out heads: piece count × any queen on the board (`--buckets 16 --obk 1`) | 0.54402 | +8 (−4 to +19) | −185 |
| 32 heads: piece count × my queen × the opponent's queen (`--buckets 32 --obk 2`) | 0.54397 | **+11 (0 to +22)** | −184 |
| input weights clipped to ±2 (`--wclip 2`) | 0.54415 | match lost in the log | −205 |

The 32 queen-aware heads cost nothing at run time and go into the next full-scale network. Not yet run at this scale:
the loss exponent, attack maps, the full layered read-out, product terms, the expert router (the card went to the
transformer, below).

The engine's integer scale (`zero/results/vault_scale_match.log`, `vault_brain.log`): the engine-against-trainer difference
falls from 3.1 to 1.1 centipawns with a finer scale, and nothing changes in play: a single scale of 2^9 or 2^8 against
2^7 is +4 (−3 to +11) each over 4,000 games at 30,000 nodes, a per-unit scale −3 (−10 to +5) for BIGS and −7 (−16 to +1)
for BIGX. The default stays the single scale (`QUNIT=1` selects per-unit, `QBITS=n` forces one).

## The transformer again: what it knows, and where that knowledge reaches (17 November)

The looped transformer of the first weeks (`model.py`: 64 square tokens, attention biased by the rules' relations between
squares, one block applied repeatedly, heads for the move, the value and a value for every move) was last trained at 2
million weights on 15 November. This section scales it up on the Lichess evaluation database and measures where it helps.

**Pipeline.** `lm/prep_lm.sh` / `lm/prep_lm.py`: the database's files -> shards of positions with up to five candidate
moves and Stockfish's deep win chance for each (171 million positions in nine shards; 45 to 62% of the positions of a
shard have more than one labelled move). `lm/train_lm.py`: the trainer on the graphics card. Profiling showed the rule
relations and the 64×64 attention maps to dominate, not the blocks; half-precision attention through PyTorch's own
attention kernel and a compiled model took training from 6,100 to 14,200 positions per second for the pilot (same best
move as the plain path in 98.8% of positions). `lm/serve.py`: one transformer on the card for every board of a match
(requests within a millisecond share a batch). `duo_uci.py`: the two-brain player, a fast search whose root moves get the
transformer's move probabilities as an additive prior; the transformer's time is charged to the mover's clock.

| Transformer | Weights | Trained on | Held-out best move | ChessBench test positions: best move / blunders / value error |
|---|---|---|---|---|
| old (`ckpt/B.pt`) | 2.0M | as of 15 November | | 48.9% / 6.6% |
| pilot `L1p` | 5.8M | 15 million positions, 18 minutes | 51.6% | 55.3% / 4.0% / 0.044 |
| `L2` (training) | 14.3M | 171 million positions, about 7 hours on the shared card | 50.4% after 6% of the run | 55.0% / 4.0% after 6% |

**As a prior on the root moves** (`lm/lm_test.sh`, quality test of `ext/quality.py`, BIGM search with and without the
transformer at the same node budget; thousandths of a point per move, negative = better, about 40 Elo a unit;
`zero/results/vault_lm_duo.log`):

| Nodes per move | old 2M, prior 8 (cap 40) | pilot, prior 8 (cap 40) | pilot, prior 16 (cap 80) |
|---|---|---|---|
| 30,000 | −1.44 (0.50) | −2.05 (0.53) | **−2.53 (0.50)** |
| 100,000 | | −0.87 (0.36) | −0.70 (0.40) |
| 300,000 | −0.31 (0.28) | −0.30 (0.28) | +0.29 (0.30) |

At the clock (`lm/duo_clock.sh`, `zero/results/vault_clock.log`): BIGS with the pilot as root prior, 1,000 games at 10+0.1
against Stockfish 14.1: 30-602-368, **−122 (−135 to −109)**; BIGS alone, the same evening: −111 (−123 to −98). The
transformer took a median 20 ms a move on a card it shared with a training run. A prior on the root is worth about 100
Elo to a 30,000-node search and nothing at playing depth, and a better transformer did not change the deep figure: a
search of 300,000 nodes already knows what the transformer knows about the root's moves.

**What it knows, without search** (`ext/brain_lm.py`, the same questions as `ext/brain_metric.py`: on 29,848 positions with
Stockfish's two-line 4,000,000-node analysis, which of the two moves is better; 10,334 of them with a clear difference):

| Evaluator | Clear cases chosen right | Score given away (thousandths) | Fit of the position's own value |
|---|---|---|---|
| BIGS, static | 70.90% | 15.72 | 0.01785 |
| Stockfish 14.1's network, static | 71.57% | 15.29 | 0.01511 |
| BIGS, captures played out | 74.70% | 11.96 | 0.01186 |
| Stockfish 14.1's network, captures played out | 74.78% | 11.90 | 0.01004 |
| pilot transformer, move probabilities | **76.55%** | 11.42 | |
| pilot transformer, its value for every move | 76.23% | 11.49 | |
| pilot transformer, value of the two positions | 76.08% | **11.11** | 0.01303 |

An 18-minute pilot already tells the better of two moves more often than Stockfish 14.1's network does after its
captures are played out. Its judgment is above anything the fast evaluator has been taught from (its teachers were
BIGX and Stockfish's network), and it is only ever asked at the root. The work from here is to get that judgment into
the leaves.
