# Evaluator architecture: what the loss says and what games say (15 November)

## 1. The probe: 40 variants, same data, from scratch

`zero/arch_probe.py`. Every variant is trained from scratch on the same 59M fresh self-play positions in the same order
(8,000 steps of 8,192) and judged on 1.0M held-out positions from another shard (loss against the search-based label).
Plain 384 is the evaluator as it is. A second seed of plain 384 differs by 0.0003: that is the noise.

| Variant | Parameters | Held-out loss | Against plain 384 |
|---|---|---|---|
| plain 384 | 303k | 0.50323 | |
| plain 384, other seed | 303k | 0.50349 | +0.0003 (noise) |
| plain 768 | 604k | 0.50255 | −0.0007 |
| plain 1536 | 1,207k | 0.50194 | −0.0013 |
| loss weighted by current error ("hard positions") | 303k | 0.50647 | +0.0032 (worse) |
| rule-graph accumulator 384+64 (factorised attack/defence relations) | 697k | 0.50267 | −0.0006 |
| mirrored inputs | 303k | 0.50232 | −0.0009 |
| king regions as shared table + correction | 3,547k | 0.50200 | −0.0012 |
| threat inputs (kind of attacker × attacked piece on its square) | 3,842k | 0.50195 | −0.0013 |
| attack maps: pawns only | 352k | 0.50280 | −0.0004 |
| attack maps: pawn, knight, king | 450k | 0.50159 | −0.0016 |
| attack maps: one per side | 352k | 0.50168 | −0.0016 |
| attack maps: bishop, rook, queen | 450k | 0.50092 | −0.0023 |
| attack maps, width 256 | 399k | 0.50060 | −0.0026 |
| attack maps into 128 of the 384 units | 401k | 0.50055 | −0.0027 |
| attack maps: value classes (P, N+B, R, Q, K) | 549k | 0.50009 | −0.0031 |
| **attack maps (12 kinds)** | 598k | 0.50006 | **−0.0032** |
| layers read the first 32 / 64 / 128 units a side | 316k / 324k / 340k | 0.50089 / 0.50050 / 0.50003 | −0.0023 / −0.0027 / −0.0032 |
| layers on a narrow path of their own, 32 / 64 units | 353k / 394k | 0.50060 / 0.50031 | −0.0026 / −0.0029 |
| looped read-out (32 numbers refined three times by one block) | 332k | 0.49963 | −0.0036 |
| **layered read-out 16-32** | 406k | 0.49952 | **−0.0037** |
| mirrored + layered read-out | 406k | 0.49909 | −0.0041 |
| wide 256 + narrow 64 with attack maps and layers | 343k | 0.49990 | −0.0033 |
| **wide 384 + narrow 64 with attack maps and layers ("two-speed")** | 443k | 0.49943 | **−0.0038** |
| the same, narrow 128 | 574k | 0.49938 | −0.0039 |
| attack maps + layered read-out | 701k | 0.49877 | −0.0045 |

Attack maps: 768 inputs computed from the rules, "a piece of this kind attacks this square" (sliders stop at the first
piece). The rule tables are checked against python-chess. In the engine's own games a move changes 12.2 of them on
average, against 2.2 piece inputs (`zero/flips.py`).

On this measure the layered read-out, the attack maps and the two-speed design are worth about five doublings of width.

## 2. The same question asked of games

The layered read-out was built into the trainer (`--head 16,32`) and the engine (`calc/calc_s13.c`; first layer in
integers over the non-zero input pairs, 79% of hidden units being exactly zero on real positions) and trained by the
production recipe: generation 101 + two passes over 538M fresh positions, with (N384H) and without (N384).

| Test | Games | Result |
|---|---|---|
| N384H against N384, 5,000 positions per move | 2,000 | 724-609-667, +9.9 Elo (−2.8 to +22.6) |
| N384H against N384, 30,000 positions per move | 2,000 | 636-780-584, +9.0 (−2.9 to +20.9) |
| N384H against N384, 150,000 positions per move | 1,000 | 265-511-224, +14.3 (−0.8 to +29.3) |
| search speed with the layers | | 787k positions/s against 1,468k (−46%) |

About +10 Elo at equal positions, for half the speed. Its held-out gain in this setting was 0.0006 (0.518579 → 0.517967 on
played positions), and `zero/slice_loss.py` shows the gain sits in balanced positions, where it should. The loss was
right about the direction and wrong by a factor of several about what a game pays for it.

For scale, the same evening, same data:

| Change | 30,000 positions per move | 10+0.1 | 60+0.6 |
|---|---|---|---|
| 538M fresh positions (N384 against generation 101) | +38 (5 to 71) | +69 (23 to 117) | +10 (−12 to 33) |
| width 768 instead of 384 (same data) | +20 (3 to 37) | −51 (−99 to −5) | 0 against generation 101 |
| layered read-out | +9 (−3 to 21) | not played: −46% speed | |

## 3. Reading

At this stage the evaluator is limited by data and the engine by time per position, not by the evaluator's form. A richer
evaluator has to pay for every nanosecond it adds, and an improvement that looks large in loss buys about ten Elo.
The designs are kept for the point where the data is several times larger (every other self-taught engine's wider and
deeper networks paid only then): the two-speed design is the one that puts the expensive inputs and layers where they
are cheap, and mirrored inputs cost the engine nothing (`calc/calc_s14.c`, `--hm 1`; game test in `lab_mir.log`).

Also tried the same day, not adopted: a read-out that keeps learning during the search (`calc/calc_s12.c`: −14 and −9
at 30,000 positions per move), a profile-guided build (no measurable speed), singular extensions from depth 6 with a
deeper null-move reduction (+14 and +8 separately at 200,000 positions per move, −4 together at 10+0.1).
