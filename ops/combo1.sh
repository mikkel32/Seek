#!/bin/bash
# the three groups that leaned the right way on the yardstick, together: there, and with our own network at 300,000
cd /root/loopchess
while ! grep -q finished fail2.log 2>/dev/null; do sleep 10; done
C="LmrDiv=214,LmrBase=52,CutLmr=2,TtCapLmr=1,SearchV3=2,PruneV2=8,HbQuad=6,HbLin=229,HbMax=4167"
bash ext/sf_bundle.sh 100000 sf_bundle.log "lmrhist|$C" "lmrhistnmp|$C,NmpBase=4,NmpDiv=3,NmpMargin=62"
bash ext/q_screen.sh calc/calc_s19h 300000 screen300k.log "$C" "$C,NmpBase=4,NmpDiv=3,NmpMargin=44" "HbQuad=6,HbLin=229,HbMax=4167" "LmrDiv=214,LmrBase=52,CutLmr=2,TtCapLmr=1,SearchV3=2,PruneV2=8"
