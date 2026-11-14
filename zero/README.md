# The self-taught line

An engine that starts from the rules of chess and learns only from its own games. No position, evaluation or game
from any other engine enters the learning loop. Stockfish appears only as a yardstick (`elo_ladder.py`).

## How it learns

| Generation | Evaluation | Where it came from |
|---|---|---|
| 0 | number of my moves minus number of yours | the rules |
| 1, 2 | 1,538 piece-square weights | solved exactly (Newton steps on a convex problem) from the previous generation's game results |
| 3+ | small network on top of the solved table | trained on the last three generations of self-play; wider networks start as exact copies of the narrower one |

Each generation: solve a table on the newest data (this also gives the search its piece values) → train the
network → play the previous generation → rate against Stockfish levels → play itself to make the next data set.
Self-play records the positions played **and** positions its searches resolved exactly along the way
("tree harvesting"); the second kind is free and was worth +87 Elo in a 400-game test.

## Run it

```bash
gcc -O3 -march=native -funroll-loops -o calc/calc_z3 calc/calc.c -lm     # the engine (needs BMI2 for the lookup tables)
zero/gen.sh 0 mob 3000 4000 0 && python3 zero/data.py 0 0.0             # generation 0: rules only
python3 calc/solve_pst.py 2000000 z0 zero/g1                            # generation 1: solved from those games
zero/gen.sh 1 zero/g1.bin 6000 4000 1 && python3 zero/data.py 1 0.3
python3 zero/loop.py 2 14                                               # everything after that; safe to interrupt and rerun
```

On a bigger machine set `NPROC` to the number of cores: `NPROC=64 python3 zero/loop.py 2 14`. Self-play, matches
and ratings scale with cores; network training is the step a GPU would speed up.

`zero/spsa.py` lets the engine tune its own search parameters by self-play.
