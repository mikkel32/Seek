#!/bin/bash
# After the search self-tuning ends: the tuned values against the standing ones, same network, at the clock.
cd /root/loopchess
while pgrep -f "[s]psa_clock" > /dev/null; do sleep 10; done; sleep 5
T=$(grep "^options:" spsa_lfw.log | tail -1 | sed 's/^options: //'); B="Hash=64,PvPrune=1,ChkExt=0,SeDouble=20,CutIir=1"
echo "[$(date -u +%H:%M:%S)] tuned: $T" >> spsa_test.log
[ -n "$T" ] || exit 1
bash gt.sh "BIGK with the tuned search values against the standing ones, 10+0.1" spsa_test.log 76 --us "calc/calc_s18 zero/w_BIGK.bin" --us_opts "$B,$T" --opp "standing values|calc/calc_s18 zero/w_BIGK.bin|$B" --tc 10+0.1 --openings 500
bash gt.sh "BIGK with the tuned search values against the standing ones, 60+0.6" spsa_test.log 76 --us "calc/calc_s18 zero/w_BIGK.bin" --us_opts "$B,$T" --opp "standing values|calc/calc_s18 zero/w_BIGK.bin|$B" --tc 60+0.6 --openings 150
echo "[$(date -u +%H:%M:%S)] finished" >> spsa_test.log
