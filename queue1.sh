#!/bin/bash
# Resumable job queue: every job runs once and leaves a marker; after an interruption just run this script again.
cd /home/user/loopchess; mkdir -p logs/done
run() { key=$1; shift; [ -f logs/done/$key ] && return; echo "[$(date -u +%H:%M:%S)] start $key"; bash -c "$*" && touch logs/done/$key; echo "[$(date -u +%H:%M:%S)] end $key"; }
M=logs/matches.jsonl
# rating on Stockfish 16's UCI_Elo scale (Stockfish gets 0.1 s per move)
run elo_B_net  "python3 elo_ladder.py --player net:ckpt/B.pt:mix:2 --start 1500 --openings 30 --levels 4 --out logs/elo_B_net.json > logs/elo_B_net.log 2>&1"
run elo_B_m64  "python3 elo_ladder.py --player mcts:ckpt/B.pt:64:1:none --start 1700 --openings 20 --levels 4 --out logs/elo_B_m64.json > logs/elo_B_m64.log 2>&1"
# loop elasticity in play: same weights, different number of loops
run loops_4v1  "python3 match.py --a net:ckpt/B.pt:mix:4 --b net:ckpt/B.pt:mix:1 --openings 500 --batched >> $M"
run loops_2v1  "python3 match.py --a net:ckpt/B.pt:mix:2 --b net:ckpt/B.pt:mix:1 --openings 500 --batched >> $M"
# loop-deepening search vs plain one-loop search, same weights, same simulations
run deepen     "python3 match.py --a mcts:ckpt/B.pt:64:1:8-2,32-4 --b mcts:ckpt/B.pt:64:1:none --openings 30 >> $M"
# the weight-space-programmed checkpoint vs the new checkpoint, with search on both sides
run AstarB_m64 "python3 match.py --a mcts:ckpt/Astar.pt:64:1:none --b mcts:ckpt/B.pt:64:1:none --openings 30 >> $M"
echo queue1 done
