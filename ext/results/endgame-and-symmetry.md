# Seek search, time allocation and endgame results

At 18,500 nodes per move, the new root-search rule raised BIGS's score against Stockfish 14.1 from 29.53% to 33.91%. The gain was 4.38 percentage points (opening-paired 95% interval +1.80 to +6.95) on 160 openings played with both colors. The corresponding match Elo gap narrowed from -151.1 to -116.0. In a separate 10+0.1 clock match against original BIGS the estimate was -15.5 Elo (-41.4 to +10.2). Stockfish still earned more points than either Seek build.

The trained clock model on G101 scored +19.1 Elo across 528 games at 5+0.05 (+2.1 to +36.2). The second set of 264 games scored +6.6 (-17.5 to +30.7). Separately, Syzygy lookup converted all 32 predefined, theoretically winning endgames to checkmate against Stockfish, up from six for G101 without tables. Each option can be enabled independently. The existing default UCI configuration is preserved.

## Exact endgames

The optional `SEEK_SYZYGY` build adds a `SyzygyPath` UCI option to CALC. It uses Fathom to read local Syzygy win/draw/loss and distance-to-zeroing tables. A winning root uses DTZ to select a move that makes progress under the fifty-move rule. At a drawn root CALC searches the draw-preserving moves. In a lost position it uses the ordinary search. Interior WDL probes supply search bounds only when the fifty-move clock is zero. Castling rights, unavailable files and positions beyond the loaded tables fall back to normal search.

The benchmark used 145 `.rtbw` and 145 `.rtbz` files for three through five pieces, approximately 939 MiB. Sorted by filename, the SHA-256 of the concatenated `filename space sha256 newline` records is `b75fef07589d190d1c2ff6bf4ce2d092e2df80ec985622be48f4554071dfdd99`. The dependency is Fathom commit `c9c6fef0dddc05d2e242c183acf5833149ab676d`. The matching source archive SHA-256 is `ba4553045d796f0521a8554ef21bad44b20fd4a7657eb08ff85d17c9f27ff8ac`. [Fathom](https://github.com/jdart1/Fathom/tree/c9c6fef0dddc05d2e242c183acf5833149ab676d) · [WDL files](https://tablebase.lichess.ovh/tables/standard/3-4-5-wdl/) · [DTZ files](https://tablebase.lichess.ovh/tables/standard/3-4-5-dtz/)

Build from the repository root, pointing `fathom` to the pinned checkout and `tables` to a directory containing the downloaded files.

```sh
cc -O3 -march=native -DHMAX=768 calc/calc_s19.c -lm -o /tmp/seek-parent
cc -O3 -march=native -DHMAX=768 -DSEEK_SYZYGY -I"$fathom/src" \
    calc/calc_s19.c "$fathom/src/tbprobe.c" -pthread -lm -o /tmp/seek-endgame
```

For AMD Zen 1/2 add `-DNO_PEXT` to both builds. Set `SyzygyPath` in the UCI client before the first search. For the full opening match, the referee sets that option on both the candidate and Stockfish. It saves every move and each opening's scores with colors reversed.

```sh
python ext/endgame_match.py /tmp/seek-parent /tmp/seek-endgame "$bigs" "$stockfish" "$tables" /tmp/seek-endgame-match --workers 9
```

## Conversion test against Stockfish 14.1

The suite contains 32 generated legal, theoretically won endgames. Each had 35–85 half-moves to the next zeroing event under optimal play. Material classes, the sampling rule and the random seed were set before playing. The attacking side alternates White and Black within each class. Each version of G101 used EvalScale 72 and 8,000 nodes per move. Stockfish 14.1 defended every position at 8,000 nodes per move with its official NNUE and the same endgame tables. The tablebase build checkmated in all 32 games. The longest of those games lasted 143 half-moves. All 64 games were replayed to verify legality and the reported result.

| Attacker and defender material | Games per build | G101 wins | G101 with tables wins |
|---|---:|---:|---:|
| Bishop and knight against king | 8 | 4 | 8 |
| Queen against rook | 8 | 2 | 8 |
| Rook and bishop against rook | 8 | 0 | 8 |
| Two knights against pawn | 8 | 0 | 8 |
| Total | 32 | 6 | 32 |

All 26 other baseline games were draws. This test measures conversion of known wins and does not estimate Elo from ordinary openings. The reproduction command writes the full games in JSONL.

```sh
python ext/endgame_conversion.py /tmp/seek-parent /tmp/seek-endgame zero/nets/g101_vol.bin "$stockfish" "$tables" /tmp/seek-conversions.jsonl
```

## Archived BIGS games

Of the 1,000 archived games against Stockfish 14.1, 384 reached five pieces or fewer. Across 6,906 Seek moves in eligible endgames, the independent oracle found one move that worsened the theoretical result. In game 820, playing Black on move 140, `Qb5+?` turned a draw into a loss. On a replay the tablebase build played `Qh7`, retaining the draw. The archive counts a game on reaching a covered position even if it ended before Seek moved again.

```sh
python ext/endgame_audit.py zero/results/games/games_bigs.jsonl.sc.jsonl.gz "$tables" /tmp/seek-endgame zero/nets/g101_vol.bin /tmp/seek-archive-audit.json
```

## Symmetry experiment

After castling rights disappear, reflecting the board across the files preserves the rules of chess. BIGS already mirrors the hidden features around each side's king. Its two linear piece-square tables can still give different values to the two orientations. The optional `SEEK_FILE_SYMMETRY` build averages each pair of reflected weights once at load time. It switches to these tables when a search starts with no castling rights, refreshes the current accumulator and clears transposition scores from the previous evaluator. Searches beginning with castling rights use the original tables. This method requires one network evaluation per node.

On 4,000 rights-free archived middlegame positions, unmodified BIGS assigned the two orientations values differing by an average 2.29 centipawns. The 95th percentile was 7. The projected output matched the arithmetic mean of the two original evaluations to within 1 centipawn. The maximum change on 256 positions with castling rights was zero. Mean absolute error against the archive's search labels was 83.78 centipawns for the parent and 83.79 for the projection. These positions check the transformation numerically. The game matches below measure playing strength.

```sh
cc -O3 -march=native -DHMAX=768 -DSEEK_FILE_SYMMETRY calc/calc_s19.c -lm -o /tmp/seek-symmetric
python ext/file_symmetry_probe.py /tmp/seek-parent /tmp/seek-symmetric "$bigs" zero/results/games/games_bigs.jsonl.sc.jsonl.gz
```

## Production clock and fixed-node matches

The tested checkpoint is the 768-wide BIGS champion. The endgame run used a Ryzen 5 3600 with identical `-DNO_PEXT` builds for the parent and the candidate. Every opening was played with colors reversed. The 160 openings of the direct match were disjoint from the 100 shared by the two Stockfish matches. Stockfish had its own official evaluation network and the full five-piece tables in both comparisons. All clock games used 10 seconds plus 0.1 seconds per move, one thread and 64 MiB of hash per player. The separate file-symmetry experiment began with 20,000 nodes per move and used a different, disjoint pool for its clock match.

| Match | Games | Wins–draws–losses | Score | Elo | Opening-paired 95% Elo interval |
|---|---:|---:|---:|---:|---:|
| BIGS with tables vs current BIGS | 320 | 58–213–49 | 51.41% | +9.8 | -9.3 to +28.9 |
| Current BIGS vs Stockfish 14.1 with tables | 200 | 2–104–94 | 27.00% | -172.8 | -204.4 to -143.6 |
| BIGS with tables vs Stockfish 14.1 with tables | 200 | 7–102–91 | 29.00% | -155.5 | -189.4 to -124.3 |
| File-symmetric BIGS vs current BIGS, fixed nodes | 400 | 118–167–115 | 50.38% | +2.6 | -20.6 to +25.9 |
| File-symmetric BIGS vs current BIGS, clock | 240 | 28–175–37 | 48.12% | -13.0 | -32.7 to +6.6 |

The endgame tables changed BIGS’s score against the common Stockfish opponent by +2.00 percentage points (paired 95% interval -2.66 to +6.66). The interval uses the difference in color-averaged scores on each opening. [Match summaries](endgame-study-summary.json)

## Additional G101 search tests

Both engines used EvalScale 72. An online quiet-move history updated the previous mover’s score from the change in static evaluation. A separate search added safe quiet checks at the first quiescence horizon. Each initial screen used 18,000 nodes per move. The strongest screen by point estimate advanced to 160 different color-reversed openings at 10+0.1. The root-interruption check used a separate 18,500-node match. These are G101 experiments.

| Candidate | Games | Wins–draws–losses | Elo | Opening-paired 95% Elo interval |
|---|---:|---:|---:|---:|
| Quiet evaluation history, strength 3 | 300 | 98–114–88 | +11.6 | -18.8 to +42.2 |
| Quiet evaluation history, strength 12 | 300 | 90–117–93 | -3.5 | -34.8 to +27.8 |
| Safe quiet checks on the principal variation | 320 | 90–105–125 | -38.2 | -70.1 to -6.9 |
| Safe quiet checks at every first horizon | 320 | 82–123–115 | -36.0 | -64.5 to -7.9 |
| Quiet evaluation history, strength 3, clock | 320 | 58–204–58 | +0.0 | -21.3 to +21.3 |
| Retain a proven root move on interruption | 440 | 166–146–128 | +30.1 | +10.1 to +50.3 |

## Root search and learned time allocation

The optional `SEEK_PROVEN_ROOT` build retains the last supported root move when a search ends before its current aspiration window has been resolved. In 2,400 archived middlegame positions searched with G101 at 42,000 nodes, this changed 64 decisions. An independent Stockfish 14.1 comparison of the two moves at 240,000 nodes preferred the new move 39 times and the old move 24 times, and rated them equal 1 time. The median difference across the changed positions was +21.5 centipawns. The 440-game row above measures the effect in ordinary play.

A separate clock model learned how Stockfish distributed its thinking time using 54,347 recorded moves. It predicts variation within six game phases from the number of legal moves and captures, check status, and remaining material. Whole opening pairs were held out. On 16,006 validation moves, root mean square error in log time was 0.7069, compared with 0.7115 using game phase alone. During play the model adjusts the soft time budget. The original hard clock limit still applies.

In a match from 132 new openings at five seconds plus 0.05 seconds per move, the trained clock on G101 scored 66 wins, 156 draws and 42 losses against the original clock. The estimate was +31.7 Elo (opening-paired 95% interval +7.4 to +56.3). Both sides used EvalScale 72. This screen used the public G101 network.

```sh
OPENBLAS_NUM_THREADS=1 python ext/train_clock.py zero/results/games/games_bigs.jsonl.sc.jsonl.gz /tmp/clock_weights.h
cc -O3 -march=native -DHMAX=768 -DSEEK_TRAINED_CLOCK -I/tmp calc/calc_s19.c -lm -o /tmp/seek-trained-clock
```

## Match interpretation

From ordinary openings the production tablebase build scored 29.00% against Stockfish 14.1, and the incumbent scored 27.00%. The paired comparison above estimates the effect of the change against that opponent. The 32-position conversion suite instead measures how reliably the search finishes positions already won by force.

All 1,360 production games in this study finished without an illegal move or a time forfeit.

The existing 1,000-game [archive analysis](../../zero/results/vault_autopsy.log) places the last move with at least 35% estimated expected score between moves 21 and 40 in 211 of the 345 losses. That is the largest group for targeted middlegame training.

## Independent tests of the interrupted-search fix

The fixed-node G101 screen used 220 opening pairs. The G101 clock match excluded all those openings. Subsequently, unchanged production BIGS and the same NNUE with the opt-in fix played a separate 180-pair screen (book offset 1360, pool 600, seed 482711). A score of at least 47.5% qualified for a 112-pair clock match from a disjoint book (offset 1980, pool 600, seed 105774). Each clock game gave each engine 10 seconds plus 0.1 seconds per move. The match used a larger search than the 8,000-node endgame conversion suite.

| NNUE | Limit | Games | Wins–draws–losses | Elo | Opening-paired 95% Elo interval |
|---|---|---:|---:|---:|---:|
| G101 | 18,500 nodes | 440 | 166–146–128 | +30.1 | +10.1 to +50.3 |
| G101 | 10+0.1 | 256 | 44–174–38 | +8.1 | -14.1 to +30.4 |
| BIGS | 18,500 nodes | 360 | 99–185–76 | +22.2 | +5.8 to +38.8 |
| BIGS | 10+0.1 | 224 | 28–158–38 | -15.5 | -41.4 to +10.2 |

The fixed-node and wall-clock tests measure different effects. Running out of nodes or time can interrupt different iterations and change which root move is returned. This study tested this fix against the corresponding Seek baseline. The following section reports the separate direct comparison against Stockfish at equal nodes.

To reproduce the root candidate, add `-DSEEK_PROVEN_ROOT` when compiling `calc/calc_s19.c` and use identical UCI options and weights for the two players.

## Direct root-search comparison with Stockfish 14.1

The parent and proven-root production builds played the same 160 openings with both colors. Every engine used one thread, 64 MiB of hash and 18,500 nodes per move. Stockfish used the official 14.1 release with its default NNUE. These games used standard search without tablebases.

| Seek build | Wins–draws–losses | Score | Match Elo | Opening-paired 95% Elo interval |
|---|---:|---:|---:|---:|
| Original BIGS | 37–115–168 | 29.53% | -151.1 | -182.9 to -121.5 |
| BIGS with proven-root search | 48–121–151 | 33.91% | -116.0 | -144.4 to -88.9 |

Changing the root search changed BIGS’s average score against Stockfish by +4.38 percentage points (opening-matched 95% interval +1.80 to +6.95). The benchmark uses equal node budgets. Equal time is a different comparison.

```sh
python ext/history_stockfish_match.py "$parent_engine" "$root_engine" "$bigs" "$stockfish" /tmp/seek-root-direct --workers 10 --nodes 18500 --pairs 160 --offset 65 --pool-size 600 --seed 905197
```

## Replication of the trained G101 clock

A second 5+0.05 match excluded all openings from the first clock-model test. Across 132 new pairs, the fitted model scored 57 wins, 155 draws and 52 losses, or 50.95% and +6.6 Elo (opening-paired 95% interval -17.5 to +30.7). There were 0 time losses for the candidate, 0 for the parent and 0 illegal moves.

Across both disjoint sets (528 games), the score was 52.75% for the trained clock, or +19.1 Elo (opening-paired 95% interval +2.1 to +36.2). Both arms used the same G101 weights and EvalScale 72. The direct production Stockfish test used the root fix and the original clock. The tablebase, root-search and trained-clock estimates come from different comparisons and cannot be added.
