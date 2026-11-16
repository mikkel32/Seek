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

Logs: `zero/results/vault_*.log` (training machine), `zero/results/lab_*.log` (match machine).
