#!/bin/bash
# After the king-aware network: verify it, rate the calculator and the two-system engine, puzzles, old-vs-new evaluation.
cd /home/user/loopchess; mkdir -p logs/done
run() { key=$1; shift; [ -f logs/done/$key ] && return; echo "[$(date -u +%H:%M:%S)] start $key"; bash -c "$*" && touch logs/done/$key; echo "[$(date -u +%H:%M:%S)] end $key"; }
N=calc/net256k.bin; M=logs/matches3.jsonl
python3 calc/check_net.py calc/net256k.bin.ema.pt $N || { echo "NETWORK CHECK FAILED"; exit 1; }
run elo_calc_k   "python3 elo_ladder.py --player calc:$N:0.1 --start 2700 --openings 20 --levels 5 --out logs/elo_calc_k.json > logs/elo_calc_k.log 2>&1"
run duo_time     "python3 match.py --a duo:$N:ckpt/B.pt:0.1:16:80:p --b calc:$N:0.1 --openings 200 --workers 4 >> $M"
run elo_duo_k    "python3 elo_ladder.py --player duo:$N:ckpt/B.pt:0.1:16:80:p --start 2800 --openings 20 --levels 5 --out logs/elo_duo_k.json > logs/elo_duo_k.log 2>&1"
run puzzles      "python3 calc/puzzles.py 'calc/calc $N' 0.1 2000 > logs/puzzles_calc.log 2>&1"
run k_vs_b       "python3 match.py --a calc:$N:0.1 --b calc:calc/net256b.bin:0.1 --openings 100 --workers 4 >> $M"
echo queue4 done
