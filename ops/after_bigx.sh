#!/bin/bash
# The moment BIGX is trained (its matches then run on the processors), the card starts the fast student:
# 768 wide, 16 king buckets, two passes over the same mix, half of every target from the labels and half from the
# mean judgment of two teachers, BIGX and Stockfish 14.1's network.
cd /root/loopchess
while ! grep -q "network BIGX trained" big_run.log; do sleep 15; done
sleep 20
TRAINER=calc/train_nnue_distill3.py THREADS=${1:-8} bash ext/big_run2.sh BIGS 2 16 768 "e:0-11,f:0-39,h:0-59,s:0-406,t:0-22" BIGM calc/calc_s19c --ytag yc --distill "zero/w_BIGX.bin.ema.pt+sf:/root/ext/sf141_embedded.bin" --dw 0.5
