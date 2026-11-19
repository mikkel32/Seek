# The strongest known configuration (15 November)

Engine `calc/calc_s9.c`, network `zero/nets/g101_vol.bin` (generation 101 with the reliability read-out), UCI options:

    VolMix=150 VolRef=70 SeDouble=20 CutIir=1 VolLmr=1 ContCorr=1 ChkExt=0 PvPrune=1 VolTm=1

Measured (same network on both sides, every opening with both colours):

| Against | Time control | Games | Result |
|---|---|---|---|
| `calc_s4`-equivalent (the same binary with default options) | 10+0.1 | 180 | 51-101-28, +45 Elo (11 to 79) |
| old engine (`calc_s0.c`) vs `calc_s4.c` | 10+0.1 | 132 | 49-60-23, +69 (26 to 115) |

Found afterwards and not yet in the combined test: `NpCorr=0` (+17, −4 to 38, at equal positions per move).
The clock-based tuning of the numeric parameters (`zero/spsa_clock.py`) starts from this configuration.

`zero/nets/g102.bin` is generation 102 without read-outs; `zero/nets/g101_vol_pol.bin` adds the where-to-search read-out
(`PolW=1000` was +13, −6 to 32, at equal positions; heavier weights lose).

## At the long time control (60+0.6)

| Test | Games | Result |
|---|---|---|
| `calc_s4.c` against the old engine | 80 | 23-52-5, +80 Elo (37 to 125) |
| the option set above against default options | 200 + 300 | 92-329-79 together, about +9 (−11 to 29) |
| two-entry hash buckets (`calc_s10.c`, `TtBucket=1`) against one entry per slot, 64 MB | 194 | 25-139-30, −9 (−35 to 17) |

The engine change holds at both time controls; the option set is a fast-time-control gain.

## Standing with the option set (10+0.1, one thread, 64 MB, every opening with both colours)

| Opponent | Wins-draws-losses | Elo difference |
|---|---|---|
| Stockfish 14.1 | 2-32-166 | −402 (−473 to −349) |
| Stockfish 14.1 on its hand-written evaluator | 32-71-97 | −117 (−158 to −79) |
| Toga II 3.0 | 94-6-0 | +604 |
| Glaurung 2.2 | 95-3-2 | +576 |
| Fruit 2.1 | 94-4-2 | +552 |

This morning's engine scored 5.3% against Stockfish 14.1 (about −500) and −175 against its hand-written evaluator.

## What holds at the long time control, and what does not (later the same day)

| Test at 60+0.6 | Games | Result |
|---|---|---|
| Pruning and extension half alone (`PvPrune=1 ChkExt=0 SeDouble=20 CutIir=1`) against defaults | 122 | 26-85-11, +43 (10 to 77) |
| Standing against Stockfish 14.1 on its hand-written evaluator (full option set) | 100 | 14-51-35, −74 (−123 to −28) |

The where-to-search read-out (`PolW=1000`) was +13 and +25 at 30,000 positions per move and −53 (−95 to −13) at
10+0.1 on top of the pruning half: not adopted.

Recommended options for now: `PvPrune=1 ChkExt=0 SeDouble=20 CutIir=1` (confirmed at 10+0.1 within the full set and
at 60+0.6 alone); the uncertainty options are a fast-time-control gain whose long-time-control test is still running.

## The uncertainty half at 60+0.6, and why the read-out lost

| Test | Games | Result |
|---|---|---|
| Uncertainty half alone (`VolMix=150 VolRef=70 VolLmr=1 VolTm=1 ContCorr=1 NpCorr=0`) against defaults, 60+0.6 | 300 | 50-201-49, +1 (−22 to 24) |
| Where-to-search read-out on top of the pruning half, 30,000 positions per move | 536 | 150-208-178, −18 (−41 to 5) |
| the same read-out without the pruning half, 30,000 positions per move (three runs) | 448 / 1,428 / 614 | +25, +8, +18 |

So: the pruning half is the part of the day's settings that holds at the long time control (+43); the uncertainty options
are worth about +30 at 50,000 positions per move, +8 at 5+0.05 and nothing at 60+0.6. The read-out helps only without the
pruning half and clashes with it, and the pruning half is worth more.

**Settings to use:** `PvPrune=1 ChkExt=0 SeDouble=20 CutIir=1`.
