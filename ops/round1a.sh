#!/bin/bash
# round 1a: the pipeline on the first 53 million self-play positions, and its control (the same further training without them)
cd /root/loopchess
T="--distill zero/w_BIGX.bin.ema.pt+sf:/root/ext/sf141_embedded.bin --dw 0.5"
OLD="e:0-11/5,f:0-39/7,h:0-59/9,s:0-406/50,t:0-22/11"
bash ext/gen_round.sh G1a BIGS 768 calc/calc_s19c "g:0-3*4,u:0-3*2,$OLD" $T
bash ext/gen_round.sh C1a BIGS 768 calc/calc_s19c "$OLD" $T
