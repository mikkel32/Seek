#!/bin/bash
# round 1a for the wide line: BIGX further trained on its own self-play (its recipe has no teacher)
cd /root/loopchess
while ! grep -q "C1a finished\|C1a: no network" gen_round.log 2>/dev/null; do sleep 15; done
OLD="e:0-11/5,f:0-39/7,h:0-59/9,s:0-406/50,t:0-22/11"
bash ext/gen_round.sh X1a BIGX 1536 calc/calc_s19x "g:0-3*4,u:0-3*2,$OLD"
