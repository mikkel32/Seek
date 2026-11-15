#!/bin/bash
# On outside data (deep evaluations + two months of game positions, 787M): the cheap layered read-out (layers reading 64
# units a side) and a 768-wide plain network, from scratch like LFf, each against LFf at equal positions per move.
cd /root/loopchess; EP=6
say() { echo "[$(date -u +%H:%M:%S)] $*" >> arch2.log; }
tags=$(seq -s, -f "e%g" 0 11),$(seq -s, -f "f%g" 0 39)
arm() { n=$1; shift; t0=$(date +%s); python3 calc/train_nnue.py --buckets 8 --ema 0.999 --files $tags --val bval --epochs $EP --lr 1e-3 --threads 4 --device cuda --pst zero/g101s.npz "$@" --out zero/w_$n.bin > zero/w_$n.train.log 2>&1; say "network $n ($*) trained in $(( $(date +%s) - t0 ))s"; }
arm LFK --h 384 --head 16,32 --hk 64
bash gt.sh "layers reading 64 units a side (LFK) against plain (LFf), same outside data, 30,000 positions per move" arch2.log 22 --us "calc/calc_s17 zero/w_LFK.bin" --opp "LFf|calc/calc_s17 zero/w_LFf.bin|Hash=64" --openings 1000 --nodes 30000
arm LFW --h 768
bash gt.sh "768 wide (LFW) against 384 wide (LFf), same outside data, 30,000 positions per move" arch2.log 22 --us "calc/calc_s17 zero/w_LFW.bin" --opp "LFf|calc/calc_s17 zero/w_LFf.bin|Hash=64" --openings 1000 --nodes 30000
say finished
