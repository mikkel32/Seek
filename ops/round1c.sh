#!/bin/bash
# round 1a, a stronger dose: twice the weight on the self-play positions and the old loop's learning rate
cd /root/loopchess
while ! grep -q "X1a finished\|X1a: no network" gen_round.log 2>/dev/null; do sleep 15; done
T="--distill zero/w_BIGX.bin.ema.pt+sf:/root/ext/sf141_embedded.bin --dw 0.5"
OLD="e:1-11/5,f:1-39/7,h:1-59/9,s:1-406/50,t:1-22/11"
bash ext/gen_round.sh G1b BIGS 768 calc/calc_s19c "g:0-3*8,u:0-3*4,$OLD" $T --lr 1e-3
