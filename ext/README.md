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

32 king buckets together with 1024 wide: +12 (1 to 23), no better than 1024 wide alone. **BIGX**, the full-scale 1536-wide
network (16 buckets, two passes over 1.8 billion Lichess, 6.02 billion second-stage and 0.35 billion first-stage positions),
is the run that tests whether width pays where data no longer does. The trainer can also teach a network from another
network's judgments (`--distill <weights> --dw <share>`; the teacher function reproduces the network's own forward pass
exactly), so that a wide network can later teach a fast 768-wide one.

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
