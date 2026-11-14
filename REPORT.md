# OURO + CALC — a two-system chess engine built in one session on 4 CPU cores

Two systems, measured separately and together:

- **OURO** — judgment. A 2.05M-parameter looped transformer with rule-graph attention. Looks at a position once.
- **CALC** — calculation. A C alpha-beta searcher with a small evaluation network, about a million positions per second.

All ratings are on Stockfish 16's `UCI_Elo` scale (Stockfish limited to a stated strength, 0.1 s per move,
each opening played with both colours). It is an approximate anchor, not a human rating.

## 1. Results

| Engine | Rating | Games | Time per move |
|---|---|---|---|
| OURO alone, no search | 1498 ± 50 | 180 | 0.011 s |
| OURO + 64-simulation tree search | 1748 ± 62 | 120 | 0.17 s |
| CALC, evaluation weights solved directly (no training), first search version | 2172 ± 59 | 160 | 0.07 s |
| CALC, trained evaluation, search version 5 | 2706 ± 53 | 160 | 0.09 s |
| CALC, king-aware evaluation, search version 7 | **2716 ± 55** | 120 | 0.09 s |
| CALC + OURO prior at the root (two systems) | 2781 ± 54 | 120 | 0.09 s in total |

The last two rows are not significantly different from each other or from the row above them; head-to-head results
(sections 4 and 5) are sharper than ladder ratings at this number of games.

Puzzles (ChessBench's 10,000-puzzle set, whole solution must match): OURO alone solves 61.0%; CALC at 0.1 s per move
solves 95.9% of the first 2,000. DeepMind's searchless models solve 88.9% (9M parameters) and 95.4% (270M).

Independent check against other engines, 0.1 s per move each, 120 games per opponent:

| Opponent | Its CCRL 40/15 rating | Score | Elo difference |
|---|---|---|---|
| Fruit 2.1 | about 2695 | 84 wins, 11 draws, 25 losses (74.6%) | +187 (+125 to +262) |
| Toga II 3.0 | about 2849 | 65 wins, 8 draws, 47 losses (57.5%) | +53 (−7 to +116) |

(Glaurung 2.2 stopped responding mid-match, so there is no result for it.) Against these two the calculator performs
around 2880–2900 on their scale, higher than its Stockfish-ladder rating. The scales differ, and 0.1 s per move is far
faster than the time control CCRL ratings are measured at; "2700–2900 depending on the yardstick" is the honest range.
Rating sources: [Fruit 2.1](https://talkchess.com/viewtopic.php?p=858790), [Toga II 3.0](https://talkchess.com/viewtopic.php?p=909186).

## 2. OURO — the transformer

Five ideas, in order of evidence:

| # | Idea | What it is | Evidence |
|---|------|-----------|----------|
| 1 | Rule-graph attention | 9 exact relation planes between squares, computed from the rules with no parameters (attacks, defences, x-rays, legal moves), added to the attention scores | Ablated: clear win |
| 2 | Looped core (Nvidia Dory) | a shared block group applied 1–4 times with the input re-injected; gradient through the last loop only | Small real effect (below) |
| 3 | Dueling heads | policy, value and a value for every move from one pass | Not ablated |
| 4 | Look-ahead heads | training-only prediction of the reply and the follow-up | Not ablated |
| 5 | Loop-deepening tree search | visited nodes get more loops, resumed from a cached state | Not significant in 60 games |

Size: width 160, 5 heads, 2 input + 3 looped + 2 output blocks, 2,054,635 parameters. Muon + AdamW, EMA weights.
Training: 3.58M positions (about a third of one pass over 8.9M Lichess positions) — all that 4 CPU cores do in 3.5 hours.

Ablations on a 0.79M proxy:

| Run | Positions seen | Best-move match | Blunders |
|---|---|---|---|
| Full model | 115k | 33.0% | 15.6% |
| No rule graph | 115k | 29.2% | 25.4% |
| No rule graph, same wall-clock time | 192k | 32.0% | 22% |
| Full model vs no loop vs batch 128 | 205k | 37.9 / 37.9 / 38.0% | 11.8 / 11.9 / 11.7% |

Held-out benchmark (31,281 ChessBench test positions that never influenced a choice):

| | Best-move match, 1 / 2 / 4 loops | Blunders, 1 / 2 / 4 loops | Puzzles solved, 1 / 2 / 4 loops (10,000) |
|---|---|---|---|
| A (step 12,500) | 48.7 / 49.1 / 49.2% | 6.67 / 6.48 / 6.42% | 59.3 / 60.2 / 60.3% |
| B (step 14,000) | 48.9 / 49.3 / 49.4% | 6.60 / 6.34 / 6.31% | 60.4 / 61.0 / 61.2% |

(An earlier puzzle figure of about 25% was wrong: the first move listed for each puzzle is the opponent's set-up move, and
my code asked the engine to find it. The table above uses the correct protocol.)

The loop in play (same weights, 1,000 searchless games each): 2 loops beat 1 loop by +20 Elo (95% range +5 to +36);
4 loops vs 1 loop was +7 (−9 to +22), not significant. For scale: DeepMind's 9M-parameter model solves 88.9% of
the same puzzles after roughly 10,000 times more training samples.

## 3. Weight-space programming on OURO (A → A*)

The recipe was followed as given: goal, splits, four candidates and a random control fixed before any answer
was seen; no training; no tuning. The skill's own files were not on this machine, so `wsp.py` implements the
description of the recipe.

- Goal: fewer blunders. Base run: 2,092 blunders in 31,280 practice positions (6.69%, under the 10% room-for-gain mark).
- The three strong edits (α = 6) lowered the blunder move and the good moves alike (−0.6 to −1.2 log-likelihood on
  all three sets): not specific, rejected at the screen.
- The gentle edit (α = 1) changed nothing; at the gate it fixed 180 of 2,092 failures (419 needed). The second
  round fixed 311 (419 needed).
- **Gate failed, copy restored: A\* is byte-identical to A.** Held-out was never consulted.
- A\* vs B (the new checkpoint, 1,500 more steps): 48.0% over 1,000 searchless games, −14 Elo (−30 to +2).

Why: the recipe removes one behaviour a model writes along one direction. A blunder is a missing calculation,
different in every position. The direction separating failed from passed positions reads as "this position is
hard"; pushing against it lowers confidence everywhere.

## 4. CALC — the calculator

The conclusion from section 3 set the direction: edits cannot add knowledge, but exact computation can be wired in.
The rule graph did that inside the network; CALC does it outside.

**Search** (`calc/calc.c`, single thread): bitboard move generation (verified against standard perft counts and
python-chess), principal-variation alpha-beta, transposition table, null move, late-move reductions, quiescence
with static exchange evaluation, singular extensions, history tables. Each version was tested against the
previous one at an equal node budget:

| Version | Change | Result |
|---|---|---|
| 2 | incrementally updated evaluation | same play, 2× speed |
| 3 | history, razoring, pruning, quiescence table | +72 Elo (+41 to +105), 400 games |
| 4 | singular extensions | +36 Elo (+6 to +66), 400 games |
| 7 | more history tables, evaluation correction | +43 Elo (+14 to +72), 400 games |

**Evaluation** — three stages:

1. *Solved, not trained.* 1,538 weights (a piece-square table for the middlegame and one for the endgame). The
   log-loss of this model is convex, so Newton steps reach its exact optimum: seven linear solves on 2M positions,
   three minutes, no gradient descent. This alone gave 2172.
2. *Trained network on top.* 256 hidden units per side, 8 output heads chosen by piece count, weight averaging,
   3 passes over 89M quiet positions (54 minutes). Validation log-loss 0.603 → 0.575. Rating 2706.
3. *King-aware inputs.* 11 king-position buckets, every bucket initialised as an exact copy of the trained
   weights so training starts at the same loss (0.57499 before and after the copy). 2.5 more passes (67 minutes)
   took validation log-loss to 0.5718. In play the gain is not significant: +14 Elo (−25 to +53) over 200 games at
   equal time against the plain network.

Two findings on data and direct solves:

- The first solve valued a queen at almost nothing in the middlegame. Cause: positions in the middle of an
  exchange, where material on the board contradicts the true evaluation. Keeping only quiet positions (best move
  and known reply are not captures) fixed it: 122M → 89M positions.
- Solving the trained network's output layer exactly (a convex problem, Newton steps on 3M positions) did not
  help: validation log-loss 0.57525 against 0.57499 for the trained layer. Gradient training had already found it.

## 5. Two systems together

CALC searches as usual. OURO evaluates the root position once and adds a bonus of at most 40 centipawns to the
root moves it prefers (8 centipawns per unit of log-probability under its policy).

| Test, 400 games each | Score | Elo | 95% range |
|---|---|---|---|
| CALC + OURO policy prior vs CALC, equal node budget | 56.9% | +48 | +20 to +76 |
| CALC + OURO move-value prior vs CALC, equal node budget | 55.1% | +36 | +8 to +64 |
| CALC + stronger OURO prior (16 per unit, cap 80) vs CALC, equal node budget | 58.6% | +60 | +33 to +89 |
| The same stronger prior vs CALC, **equal total time** (0.1 s per move) | 45.5% | −31 | −56 to −7 |

When OURO's thinking is free, a network that plays at 1500 on its own improves an engine at 2700: it breaks near-ties
between tactically safe moves better than the fast evaluation does. I predicted no gain; that prediction was wrong.
But OURO costs 12–15 ms per move, and at 0.1 s per move that cost outweighs the benefit: head-to-head at equal total
time the two-system engine loses by 31 Elo.

Does the prior still help when the calculator searches deeper? Yes, less: at 80,000 nodes per move (four times the
budget above) the gentle prior (8 per unit, cap 40) scored 54.6% over 400 games, +32 Elo (+7 to +58), again with
OURO's time free. So the two-system engine should win wherever OURO's cost is small next to the move time: longer
time controls, or OURO on a GPU. That case was not measured here.

Why OURO has something to add — the same 60,000 quiet validation positions, no search, three evaluators:

| Evaluator | Log-loss | Mean error in win probability | Speed |
|---|---|---|---|
| Solved table (1,538 weights) | 0.5995 | 0.125 | millions per second |
| CALC's trained network (plain inputs) | 0.5739 | 0.093 | about a million per second |
| OURO value head, 2 loops | **0.5636** | **0.081** | a few hundred per second |

OURO judges a position better than the fast network does, after seeing 25 times fewer positions; it is also about
3,000 times slower. Which system should decide depends on how much time there is.

## 6. What is new and what is not

- Not new: the calculator's design (alpha-beta + a small efficiently-updated network) is how strong CPU engines
  are built. The loop comes from Nvidia's Dory (arXiv 2611.03998); benchmark data and protocol from DeepMind's
  ChessBench (arXiv 2402.04494); labels from the Lichess evaluation database.
- New as far as I could find, each small: the parameter-free rule graph as an attention bias; evaluation weights
  from a closed-form solve as the starting point of the network; function-preserving weight copies to grow the
  network; a 2M-parameter transformer's policy as an additive prior on a classical search's root moves.

## 7. What would move it further

- **Time and hardware.** All ratings are at 0.1 s per move on one CPU core.
- **A GPU for OURO.** OURO saw 3.58M positions; DeepMind's small model saw about 10,000 times more samples. More
  training makes its prior better, and a GPU makes its cost per move negligible — both favour the two-system engine.
- **Evaluation capacity.** The fast network's training and validation losses are equal, so it is limited by size, not data.
- **Search tuning.** The search parameters were set by hand and never tuned; each tested change was worth 35–70 Elo
  at an equal node budget.

## 8. Files

- `model.py`, `train.py`, `engine.py`, `bench.py`, `wsp.py` — OURO, its training, tree search, benchmarks, the recipe.
- `calc/calc.c` — the calculator (UCI: `calc/calc <weights>`); `calc/solve_pst.py`, `calc/train_nnue.py`, `calc/mkdata.py`.
- `match.py`, `elo_ladder.py` — matches and the Stockfish ladder. `uci.py` — UCI front end for OURO.
- `duo_uci.py` — UCI front end for the two-system engine (`python3 duo_uci.py calc/net256k.bin ckpt/B.pt`).
- `ckpt/A.pt`, `ckpt/Astar.pt`, `ckpt/B.pt` — OURO checkpoints. `calc/*.bin` — calculator weights.
