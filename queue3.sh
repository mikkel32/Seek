#!/bin/bash
# After the evaluation network is trained: rating ladder for the calculator, then the two-system tests.
cd /home/user/loopchess; mkdir -p logs/done
run() { key=$1; shift; [ -f logs/done/$key ] && return; echo "[$(date -u +%H:%M:%S)] start $key"; bash -c "$*" && touch logs/done/$key; echo "[$(date -u +%H:%M:%S)] end $key"; }
N=calc/net256b.bin; M=logs/matches2.jsonl
run elo_calc_nn   "python3 elo_ladder.py --player calc:$N:0.1 --start 2500 --openings 20 --levels 5 --out logs/elo_calc_nn.json > logs/elo_calc_nn.log 2>&1"
# does OURO's judgment add anything on top of the calculator? equal node budget, so OURO's own thinking time is free here
run duo_p8        "python3 match.py --a duo:$N:ckpt/B.pt:20000:8:40:p --b calc:$N:20000 --openings 200 --workers 4 >> $M"
run duo_q2        "python3 match.py --a duo:$N:ckpt/B.pt:20000:2:40:q --b calc:$N:20000 --openings 200 --workers 4 >> $M"
# network vs directly solved weights, same engine, same time
run nn_vs_pst     "python3 match.py --a calc:$N:0.1 --b calc:calc/pst.bin:0.1 --openings 50 --workers 4 >> $M"
echo queue3 done
