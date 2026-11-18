#!/bin/bash
# BIGS + L2 as root prior at the clock (the fast network's training is paused meanwhile: it shares the card), then L2 steering at fixed nodes
cd /root/loopchess
TR=$(pgrep -f "^python3 calc/train_nnue_v6.py" | tr "\n" " "); for p in $TR; do kill -STOP $p; done
bash duo_clock.sh L2 8 40 500
for p in $TR; do kill -CONT $p; done
bash lm/lmtree_test.sh L2 lm_tree.log "300000 8 2000 1 1 22"
