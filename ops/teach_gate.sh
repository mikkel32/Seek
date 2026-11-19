#!/bin/bash
# L2 as a teacher for the fast network, the cheap gate: label 5M positions of 12 shards, continue BIGS on them with and
# without the stored teacher, score the quiet pairs.
cd /root/loopchess; say() { echo "[$(date -u +%H:%M:%S)] $*" >> teach_gate.log; }
TR=$(pgrep -f "^python3 calc/train_nnue_v6.py" | tr "\n" " "); for p in $TR; do kill -STOP $p; done
say "labelling with L2 (the network training is paused)"
OURO_GPU=1 python3 lm/label_nnue.py ckpt/L2.pt 5000000 h0 h1 h2 h3 h4 h5 s0 s1 s2 s3 s4 s5 2>&1 | grep -v "Warning\|out\[i" >> teach_gate.log
for p in $TR; do kill -CONT $p; done
say "labelled; probes"
TRAINER=calc/train_nnue_v7.py NOPLAY=1 G="h:0-5,s:0-5" bash ext/sib_probe.sh "c5|--max_rows 5000000" "t34|--max_rows 5000000 --tl 0.34" "t100|--max_rows 5000000 --tl 1"
grep -A4 "^\[.*\] \(c5\|t34\|t100\) " sib_probe.log | cut -c1-200 >> teach_gate.log
E=calc/calc_s19h; python3 ext/brain_quiet.py "BIGS|$E zero/w_BIGS.bin" "c5|$E zero/q_c5.bin" "t34|$E zero/q_t34.bin" "t100|$E zero/q_t100.bin" 2>&1 | tail -5 >> teach_gate.log
say "finished"
