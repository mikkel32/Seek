#!/bin/bash
# Stage 2a: held-out benchmarks + the searchless matches. Sequential; nothing else on the CPU.
cd /home/user/loopchess
M=logs/matches.jsonl; : > $M
python3 bench.py ckpt/A.pt --half A --loops 1,2,4 --puzzles 10000 > logs/bench_A.json 2> logs/bench_A.err
python3 bench.py ckpt/B.pt --half A --loops 1,2,4 --puzzles 10000 > logs/bench_B.json 2> logs/bench_B.err
# the match the task asks for: the weight-space-programmed checkpoint (A*, identical to A) vs the new checkpoint B
python3 match.py --a net:ckpt/Astar.pt:mix:2 --b net:ckpt/B.pt:mix:2 --openings 500 --batched >> $M
# loop elasticity in play: same weights, different number of loops
python3 match.py --a net:ckpt/B.pt:mix:4 --b net:ckpt/B.pt:mix:1 --openings 500 --batched >> $M
python3 match.py --a net:ckpt/B.pt:mix:2 --b net:ckpt/B.pt:mix:1 --openings 500 --batched >> $M
# loop-deepening search vs plain one-loop search, same weights, same simulations
python3 match.py --a mcts:ckpt/B.pt:64:1:8-2,32-4 --b mcts:ckpt/B.pt:64:1:none --openings 30 >> $M
echo stage2a done
